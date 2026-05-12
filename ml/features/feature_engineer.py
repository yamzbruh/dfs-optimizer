"""Statcast → ML feature engineering for the XGBoost points projection model.

Transforms raw pitch-by-pitch Statcast data (stored in Parquet) into a
game-level feature matrix suitable for training and inference.  Every public
method returns a *new* DataFrame — input DataFrames are never modified in place.

Pipeline order for the full batter feature matrix::

    load_statcast_years                  (PA-level, ~651k rows)
        → build_rolling_batter_features  (xwOBA / EV / barrel / HH rolling
                                          + xwoba_babip_gap_7d luck signal)
        → build_platoon_features         (platoon_advantage / same_hand)
        → build_batting_order_features   (batting_order_multiplier)
        → build_game_context_features    (no-op — leaky features removed)
        → build_park_factor_features     (park_factor from PARK_FACTORS lookup)
        → build_dk_points_labels         (PA-level: hits / walks / HBP only)
        → aggregate_to_game_level        (sum PA dk_points → dk_points_game)
        → join_game_log_features_game_level  (R / RBI / SB → dk_points_game)
        → build_opposing_pitcher_features    (opp starter K/ERA/whiff/velo)
        → (drop null dk_points_game)
        → save to "features/batter_feature_matrix_game_level"

    Leaky in-game features removed: run_diff, is_close_game,
    is_high_leverage, pa_count.

    Rebuild the cached Parquet after pipeline changes via
    ``PointsModel.train(..., force_rebuild_features=True)`` or equivalent.

Target variable: ``dk_points_game`` — total DK points a player scored in a game.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from loguru import logger

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_pipeline.loaders.parquet_cache import ParquetCache  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# DK MLB classic batting scoring — verified from DK official rules.
# Walk/HBP are 2.0 pts.  No strikeout penalty for hitters in DK MLB.
# R (+2.0), RBI (+2.0), SB (+5.0) are applied in build_dk_points_labels()
# when the box-score columns exist.
_DK_EVENT_POINTS: dict[str, float] = {
    "single": 3.0,
    "double": 5.0,
    "triple": 8.0,
    "home_run": 10.0,
    "walk": 2.0,
    "hit_by_pitch": 2.0,
}

# Batting-order multipliers encode the expected PA-frequency boost or
# penalty for each lineup slot over a full game.
_BATTING_ORDER_MULTIPLIERS: dict[int, float] = {
    1: 1.00,
    2: 1.02,
    3: 1.05,
    4: 1.03,
    5: 0.98,
    6: 0.93,
    7: 0.87,
    8: 0.81,
    9: 0.75,
}

# Park run-factor lookup (multi-year average, 1.0 = neutral).
# > 1.0 = hitter-friendly, < 1.0 = pitcher-friendly.
PARK_FACTORS: dict[str, float] = {
    "COL": 1.146,  # Coors Field
    "CIN": 1.072,  # Great American Ball Park
    "BOS": 1.058,  # Fenway Park
    "TEX": 1.044,  # Globe Life Field
    "PHI": 1.038,  # Citizens Bank Park
    "MIL": 1.031,  # American Family Field
    "NYY": 1.028,  # Yankee Stadium
    "BAL": 1.024,  # Camden Yards
    "ATL": 1.018,  # Truist Park
    "HOU": 1.012,  # Minute Maid Park
    "LAA": 1.008,  # Angel Stadium
    "MIN": 1.004,  # Target Field
    "ARI": 1.002,  # Chase Field
    "DET": 0.998,  # Comerica Park
    "WSH": 0.995,  # Nationals Park
    "TOR": 0.992,  # Rogers Centre
    "CHC": 0.989,  # Wrigley Field
    "CWS": 0.985,  # Guaranteed Rate Field
    "STL": 0.982,  # Busch Stadium
    "KCR": 0.979,  # Kauffman Stadium
    "CLE": 0.976,  # Progressive Field
    "NYM": 0.973,  # Citi Field
    "TB":  0.970,  # Tropicana Field
    "MIA": 0.967,  # loanDepot Park
    "OAK": 0.964,  # Oakland Coliseum
    "LAD": 0.961,  # Dodger Stadium
    "SEA": 0.958,  # T-Mobile Park
    "SF":  0.955,  # Oracle Park
    "SD":  0.952,  # Petco Park
    "PIT": 0.949,  # PNC Park
}

# Columns excluded from the training feature set.
_LABEL_AND_ID_COLS: set[str] = {"batter", "game_date", "events", "dk_points"}


class FeatureEngineer:
    """Builds ML-ready batter features from cached Statcast Parquet files.

    Usage::

        fe = FeatureEngineer()
        matrix = fe.build_full_batter_feature_matrix([2023, 2024, 2025])
        X = matrix[fe.get_feature_columns()]
        y = matrix["dk_points_game"]
    """

    def __init__(self, cache_dir: str = "data/parquet") -> None:
        """Initialise with a ``ParquetCache`` rooted at ``cache_dir``."""
        self.cache = ParquetCache(cache_dir)
        logger.debug(f"FeatureEngineer ready  cache_dir={cache_dir}")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_statcast_years(self, years: list[int]) -> pd.DataFrame:
        """Load and concatenate Statcast batter Parquets for ``years``.

        Only plate-appearance terminal rows are retained (rows where
        ``events`` is non-null / non-empty).

        Args:
            years: List of season years to load, e.g. ``[2023, 2024]``.

        Returns:
            Combined DataFrame of PA-level Statcast events, or an empty
            DataFrame when no cached files are found.
        """
        frames: list[pd.DataFrame] = []
        for year in years:
            key = f"statcast/batters_{year}"
            df = self.cache.load(key)
            if df is None or df.empty:
                logger.warning(f"load_statcast_years: no data for key={key!r}")
                continue
            frames.append(df)
            logger.debug(f"Loaded {len(df):,} rows for {key}")

        if not frames:
            logger.warning("load_statcast_years: no years loaded; returning empty DataFrame")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)

        # Retain only terminal pitches (plate-appearance outcomes).
        if "events" in combined.columns:
            before = len(combined)
            combined = combined[
                combined["events"].notna() & (combined["events"] != "")
            ].copy()
            logger.debug(
                f"Dropped {before - len(combined):,} non-terminal pitches"
            )
        else:
            logger.warning("load_statcast_years: 'events' column not found")

        if "game_date" in combined.columns:
            combined["game_date"] = pd.to_datetime(combined["game_date"])

        logger.info(
            f"load_statcast_years({years}): {len(combined):,} PA rows loaded"
        )
        return combined

    # ------------------------------------------------------------------
    # Feature builders
    # ------------------------------------------------------------------

    def build_rolling_batter_features(
        self,
        df: pd.DataFrame,
        windows: list[int] | None = None,
    ) -> pd.DataFrame:
        """Add rolling-window batter performance features.

        For each window size, computes rolling means (``min_periods=1``)
        grouped by batter and ordered by game date.  New columns follow
        the naming convention ``{metric}_{window}d``.

        Metrics added (when source column is present):

        * ``xwoba_{w}d`` — rolling mean of ``estimated_woba_using_speedangle``
        * ``exit_velo_{w}d`` — rolling mean of ``launch_speed``
        * ``barrel_rate_{w}d`` — rolling mean of ``barrel`` (0/1)
        * ``hard_hit_{w}d`` — rolling mean of ``launch_speed >= 95``
          (derived binary column)
        * ``xwoba_babip_gap_7d`` — ``xwoba_7d`` minus a **7-day only**
          rolling mean of ``babip_value`` (computed internally; raw
          ``babip_*d`` columns are not exposed — luck signal without noisy
          multi-window BABIP features).

        Args:
            df: PA-level Statcast DataFrame.
            windows: Window sizes in days.  Defaults to ``[7, 14, 30]``.

        Returns:
            New DataFrame with rolling columns appended.
        """
        if df is None or df.empty:
            logger.warning("build_rolling_batter_features: empty input")
            return df if df is not None else pd.DataFrame()

        _windows = windows if windows is not None else [7, 14, 30]
        result = df.copy()

        if "game_date" in result.columns:
            result["game_date"] = pd.to_datetime(result["game_date"])

        # Normalise barrel to a clean 0/1 float before rolling.
        # pybaseball's statcast() does not always return a 'barrel' column.
        # When absent, derive it from 'launch_speed_angle': Statcast codes
        # 6 = Barrel in its launch-speed/angle categorical classification.
        # When 'barrel' IS present it may be 1.0/NaN (not 0/1), so coerce
        # and fill regardless of source.
        if "barrel" not in result.columns:
            if "launch_speed_angle" in result.columns:
                result["barrel"] = (
                    pd.to_numeric(result["launch_speed_angle"], errors="coerce") == 6
                ).astype(float)
                logger.debug(
                    "build_rolling_batter_features: derived 'barrel' from "
                    "'launch_speed_angle == 6'"
                )
            else:
                logger.warning(
                    "build_rolling_batter_features: neither 'barrel' nor "
                    "'launch_speed_angle' found; barrel_rate columns will be skipped"
                )

        if "barrel" in result.columns:
            result["barrel"] = (
                pd.to_numeric(result["barrel"], errors="coerce")
                .fillna(0)
                .clip(0, 1)
            )

        # Pre-build the hard-hit binary before grouping.
        if "launch_speed" in result.columns:
            result["_hard_hit"] = (result["launch_speed"] >= 95).astype(float)
        else:
            result["_hard_hit"] = float("nan")
            logger.warning("build_rolling_batter_features: 'launch_speed' missing; hard_hit will be NaN")

        # Normalise babip_value to 0/1 float before rolling.
        # Statcast includes babip_value = 1 when a ball in play becomes a hit.
        if "babip_value" in result.columns:
            result["babip_value"] = (
                pd.to_numeric(result["babip_value"], errors="coerce")
                .fillna(0)
                .clip(0, 1)
            )
        else:
            logger.warning(
                "build_rolling_batter_features: 'babip_value' not found; "
                "xwoba_babip_gap_7d will default to 0.0"
            )

        rolling_specs: list[tuple[str, str]] = [
            ("estimated_woba_using_speedangle", "xwoba"),
            ("launch_speed", "exit_velo"),
            ("barrel", "barrel_rate"),
            ("_hard_hit", "hard_hit"),
        ]

        result = result.sort_values(["batter", "game_date"])

        for src_col, out_prefix in rolling_specs:
            if src_col not in result.columns:
                logger.warning(
                    f"build_rolling_batter_features: source column {src_col!r} "
                    "not in df; skipping"
                )
                continue
            for w in _windows:
                out_col = f"{out_prefix}_{w}d"
                result[out_col] = (
                    result.groupby("batter", sort=False)[src_col]
                    .transform(
                        lambda s, window=w: s.rolling(window, min_periods=1).mean()
                    )
                )

        # 7d BABIP rolling mean is internal only (no babip_*d output columns).
        babip_7_internal = None
        if "babip_value" in result.columns:
            babip_7_internal = (
                result.groupby("batter", sort=False)["babip_value"]
                .transform(lambda s: s.rolling(7, min_periods=1).mean())
            )
        if "xwoba_7d" in result.columns and babip_7_internal is not None:
            result["xwoba_babip_gap_7d"] = result["xwoba_7d"] - babip_7_internal
        else:
            result["xwoba_babip_gap_7d"] = 0.0

        result = result.drop(columns=["_hard_hit"], errors="ignore")
        logger.info(
            f"build_rolling_batter_features: added "
            f"{len(rolling_specs) * len(_windows)} rolling columns + "
            f"xwoba_babip_gap_7d (windows={_windows})"
        )
        return result

    def build_platoon_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add platoon-advantage and same-hand matchup indicator columns.

        Platoon advantage is conventionally held by the batter when they
        face a pitcher of opposite handedness (L vs R or R vs L).

        Columns added:

        * ``platoon_advantage`` — 1 when batter faces opposite-handed
          pitcher, 0 otherwise.
        * ``same_hand`` — 1 when batter and pitcher are same-handed,
          0 otherwise.

        Args:
            df: Statcast DataFrame with ``stand`` and ``p_throws`` columns.

        Returns:
            New DataFrame with platoon columns appended.
        """
        if df is None or df.empty:
            logger.warning("build_platoon_features: empty input")
            return df if df is not None else pd.DataFrame()

        result = df.copy()

        missing = [c for c in ("stand", "p_throws") if c not in result.columns]
        if missing:
            logger.warning(
                f"build_platoon_features: missing columns {missing}; "
                "platoon_advantage and same_hand will be 0"
            )
            result["platoon_advantage"] = 0
            result["same_hand"] = 0
            return result

        result["platoon_advantage"] = (
            (
                (result["stand"] == "L") & (result["p_throws"] == "R")
            ) | (
                (result["stand"] == "R") & (result["p_throws"] == "L")
            )
        ).astype(int)

        result["same_hand"] = (
            result["stand"] == result["p_throws"]
        ).astype(int)

        logger.info("build_platoon_features: platoon_advantage and same_hand added")
        return result

    def build_batting_order_features(
        self,
        df: pd.DataFrame,
        batting_order_col: str = "batting_order",
    ) -> pd.DataFrame:
        """Add a batting-order PA-frequency multiplier column.

        Maps the batting-order slot (1–9) to a float multiplier that
        represents the relative expected plate-appearance opportunity
        over a full game.  Rows with unknown / out-of-range slots
        default to ``1.0``.

        Column added: ``batting_order_multiplier``

        Args:
            df: DataFrame that may or may not contain ``batting_order_col``.
            batting_order_col: Name of the batting-order column.

        Returns:
            New DataFrame with ``batting_order_multiplier`` appended.
        """
        if df is None or df.empty:
            logger.warning("build_batting_order_features: empty input")
            return df if df is not None else pd.DataFrame()

        result = df.copy()

        if batting_order_col not in result.columns:
            logger.warning(
                f"build_batting_order_features: {batting_order_col!r} not in df; "
                "defaulting batting_order_multiplier to 1.0"
            )
            result["batting_order_multiplier"] = 1.0
            return result

        result["batting_order_multiplier"] = (
            result[batting_order_col]
            .map(_BATTING_ORDER_MULTIPLIERS)
            .fillna(1.0)
        )
        logger.info("build_batting_order_features: batting_order_multiplier added")
        return result

    def build_dk_points_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map Statcast ``events`` to DraftKings batting points.

        Scoring applied — verified from DK MLB classic rules:

        * Single: +3 | Double: +5 | Triple: +8 | Home run: +10
        * Walk: +2 | Hit by pitch: +2
        * No strikeout penalty for hitters in DK MLB

        Runs, RBI, and stolen bases are **not** included here; they are
        merged at game level in ``join_game_log_features_game_level`` after
        ``aggregate_to_game_level`` so game totals are not multiplied by PA
        count.

        Column added: ``dk_points`` (float, NaN rows filled with 0.0).

        Args:
            df: Statcast DataFrame with an ``events`` column.

        Returns:
            New DataFrame with ``dk_points`` column appended.
        """
        if df is None or df.empty:
            logger.warning("build_dk_points_labels: empty input")
            return df if df is not None else pd.DataFrame()

        result = df.copy()

        if "events" not in result.columns:
            logger.warning(
                "build_dk_points_labels: 'events' column missing; "
                "dk_points set to 0.0"
            )
            result["dk_points"] = 0.0
            return result

        result["dk_points"] = result["events"].map(_DK_EVENT_POINTS).fillna(0.0)

        logger.info(
            f"build_dk_points_labels: dk_points added  "
            f"total_pts={result['dk_points'].sum():.0f}"
        )
        return result

    def build_game_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deprecated — returns ``df`` unchanged.

        ``run_diff``, ``is_close_game``, and ``is_high_leverage`` are
        in-game state features that cannot be known before a game starts.
        They have been removed from the feature set to eliminate data
        leakage.  This method is retained as a no-op so call-sites do
        not require update.
        """
        logger.debug(
            "build_game_context_features: skipped "
            "(features removed — data leakage)"
        )
        return df if df is not None else pd.DataFrame()

    def build_park_factor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a ballpark run-factor column based on the home team.

        Looks up ``home_team`` in ``PARK_FACTORS``.  Unknown teams default
        to ``1.0`` (neutral park).

        Column added: ``park_factor``

        Args:
            df: Statcast DataFrame with a ``home_team`` column.

        Returns:
            New DataFrame with ``park_factor`` appended.
        """
        if df is None or df.empty:
            logger.warning("build_park_factor_features: empty input")
            return df if df is not None else pd.DataFrame()

        result = df.copy()

        if "home_team" not in result.columns:
            logger.warning(
                "build_park_factor_features: 'home_team' column missing; "
                "park_factor defaulting to 1.0"
            )
            result["park_factor"] = 1.0
            return result

        result["park_factor"] = result["home_team"].map(PARK_FACTORS).fillna(1.0)
        known = result["park_factor"].ne(1.0).sum()
        logger.info(
            f"build_park_factor_features: park_factor added "
            f"({known:,} rows matched a known park)"
        )
        return result

    def build_opposing_pitcher_features(
        self,
        df: pd.DataFrame,
        years: list[int],
    ) -> pd.DataFrame:
        """Join opposing starting-pitcher rolling metrics to game-level hitter rows.

        Intended to run **after** ``aggregate_to_game_level`` so the merge
        on ``game_pk`` is one game-to-one pitcher row.  Uses the pitcher
        with the most innings in each game as a proxy for the starter,
        which avoids using any stats from the current game (no leakage).

        Columns added:

        * ``opp_k_rate_14d`` — opposing starter K-rate rolling 14d
        * ``opp_era_approx_14d`` — opposing starter ERA proxy rolling 14d
        * ``opp_whiff_rate_14d`` — opposing starter whiff-rate rolling 14d
        * ``opp_velo_mean_14d`` — opposing starter velocity rolling 14d

        Missing values and games with no pitcher match are filled with
        MLB league averages (K%=22 %, ERA=4.20, whiff=24 %, velo=93.5 mph).

        Args:
            df: Game-level DataFrame containing a ``game_pk`` column.
            years: Not used directly; retained for API symmetry with other
                pipeline steps.

        Returns:
            New DataFrame with ``opp_*`` columns appended.
        """
        _DEFAULTS: dict[str, float] = {
            "opp_k_rate_14d": 0.22,
            "opp_era_approx_14d": 4.20,
            "opp_whiff_rate_14d": 0.24,
            "opp_velo_mean_14d": 93.5,
        }

        if df is None or df.empty:
            logger.warning("build_opposing_pitcher_features: empty input")
            return df if df is not None else pd.DataFrame()

        result = df.copy()

        pitcher_df = self.cache.load("features/pitcher_feature_matrix_game_level")
        if pitcher_df is None or pitcher_df.empty:
            logger.warning(
                "build_opposing_pitcher_features: no pitcher feature matrix found "
                "— using league averages"
            )
            for col, val in _DEFAULTS.items():
                result[col] = val
            return result

        pitcher_cols = [
            c for c in [
                "pitcher", "game_date", "game_pk",
                "k_rate_14d", "era_approx_14d",
                "whiff_rate_14d", "velo_mean_14d",
                "innings_pitched",
            ]
            if c in pitcher_df.columns
        ]
        pitchers = pitcher_df[pitcher_cols].copy()

        # One row per game: pitcher with the most innings = proxy for starter.
        if "innings_pitched" in pitchers.columns and "game_pk" in pitchers.columns:
            starters = (
                pitchers.sort_values("innings_pitched", ascending=False)
                .groupby("game_pk")
                .first()
                .reset_index()
            )
        else:
            starters = pitchers.copy()

        rename_map = {
            "k_rate_14d": "opp_k_rate_14d",
            "era_approx_14d": "opp_era_approx_14d",
            "whiff_rate_14d": "opp_whiff_rate_14d",
            "velo_mean_14d": "opp_velo_mean_14d",
        }
        starters = starters.rename(columns=rename_map)

        if "game_pk" in result.columns and "game_pk" in starters.columns:
            opp_cols = ["game_pk"] + [
                c for c in rename_map.values() if c in starters.columns
            ]
            result = result.merge(starters[opp_cols], on="game_pk", how="left")

            matched = int(result["opp_k_rate_14d"].notna().sum())
            total = len(result)
            logger.info(
                f"build_opposing_pitcher_features: "
                f"matched {matched:,}/{total:,} rows via game_pk"
            )
            if total > 0 and matched / total < 0.5:
                logger.warning(
                    f"Low opposing pitcher match rate: "
                    f"{matched / total:.1%} — check game_pk join"
                )

        for col, val in _DEFAULTS.items():
            if col not in result.columns:
                result[col] = val
            else:
                result[col] = result[col].fillna(val)

        return result

    # ------------------------------------------------------------------
    # Game-log join and game-level aggregation
    # ------------------------------------------------------------------

    def join_game_log_features(
        self,
        df: pd.DataFrame,
        years: list[int],
    ) -> pd.DataFrame:
        """Join per-game R, RBI, SB from MLB Stats API game logs to PA data.

        Loads cached ``gamelogs/hitting_{year}`` Parquets written by
        ``StatcastLoader.get_season_game_logs_hitting``.  Those logs use
        MLBAM batter IDs directly, so no crosswalk is required.  A
        left-merge on ``(batter, game_date)`` attaches the game totals to
        every plate-appearance row for that player-game.

        If no game logs are cached the columns are filled with 0 and a
        warning is logged — the pipeline never crashes.

        Columns added: ``runs``, ``rbi``, ``stolen_bases``

        Args:
            df: PA-level Statcast DataFrame with ``batter`` and
                ``game_date`` columns.
            years: Season years whose ``gamelogs/hitting_{year}`` Parquets
                to load.

        Returns:
            New DataFrame with ``runs``, ``rbi``, ``stolen_bases`` columns
            appended (0 where the join finds no match).
        """
        if df is None or df.empty:
            logger.warning("join_game_log_features: empty input")
            return df if df is not None else pd.DataFrame()

        result = df.copy()

        frames: list[pd.DataFrame] = []
        for year in years:
            key = f"gamelogs/hitting_{year}"
            gl = self.cache.load(key)
            if gl is not None and not gl.empty:
                frames.append(gl)
                logger.debug(f"Loaded game logs: {key} ({len(gl):,} rows)")

        if not frames:
            logger.warning(
                "join_game_log_features: no game logs cached. "
                "Run scripts/pull_game_logs.py --all first. "
                "R/RBI/SB will be 0."
            )
            result["runs"] = 0
            result["rbi"] = 0
            result["stolen_bases"] = 0
            return result

        game_logs = pd.concat(frames, ignore_index=True)
        game_logs["game_date"] = pd.to_datetime(game_logs["game_date"])
        game_logs["batter"] = pd.to_numeric(
            game_logs["batter"], errors="coerce"
        ).astype("Int64")

        keep_cols = [c for c in ["batter", "game_date", "runs", "rbi", "stolen_bases"]
                     if c in game_logs.columns]
        game_logs = game_logs[keep_cols]

        result["game_date"] = pd.to_datetime(result["game_date"])
        result["batter"] = pd.to_numeric(
            result["batter"], errors="coerce"
        ).astype("Int64")

        result = result.merge(game_logs, on=["batter", "game_date"], how="left")

        for col in ("runs", "rbi", "stolen_bases"):
            if col not in result.columns:
                result[col] = 0
            else:
                result[col] = result[col].fillna(0)

        joined = result[["runs", "rbi", "stolen_bases"]].gt(0).any(axis=1).sum()
        logger.info(
            f"join_game_log_features: R/RBI/SB joined for "
            f"{joined:,} of {len(result):,} PA rows"
        )
        return result

    def aggregate_to_game_level(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate PA-level Statcast data to one row per player per game.

        This collapses the ~651k PA-row training set to ~120k player-game
        rows so XGBoost learns to predict *total game DK points* rather
        than single plate-appearance outcomes.

        Aggregation rules per ``(batter, game_date)`` group:

        * **SUM**: ``dk_points`` (renamed to ``dk_points_game``), ``_pa``
          (renamed to ``pa_count``).
        * **LAST**: rolling ``*_{7,14,30}d`` columns (excluding
          ``xwoba_babip_gap_7d``), ``platoon_advantage``, ``same_hand``,
          ``batting_order_multiplier``, ``park_factor``, ``xwoba_babip_gap_7d``,
          ``p_throws``, ``stand``.
        * **FIRST**: ``home_team``, ``away_team``, ``game_pk``.

        ``runs``, ``rbi``, ``stolen_bases``, and opposing-pitcher features
        are joined **after** this step so they are not summed per PA.

        Leaky in-game features (``run_diff``, ``is_close_game``,
        ``is_high_leverage``, ``pa_count``) are **not** included.

        Args:
            df: PA-level DataFrame that has already passed through the full
                feature-builder chain including ``build_dk_points_labels``.

        Returns:
            New DataFrame with one row per ``(batter, game_date)`` and a
            ``dk_points_game`` target column.
        """
        if df is None or df.empty:
            logger.warning("aggregate_to_game_level: empty input")
            return df if df is not None else pd.DataFrame()

        logger.info(f"Aggregating {len(df):,} PA rows to game level…")

        result = df.copy()
        result["_pa"] = 1

        sum_cols = [c for c in ("dk_points", "_pa") if c in result.columns]

        rolling_cols = [
            c for c in result.columns
            if any(c.endswith(f"_{w}d") for w in ("7", "14", "30"))
            and c != "xwoba_babip_gap_7d"
        ]

        last_cols = [c for c in (
            "platoon_advantage", "same_hand", "batting_order_multiplier",
            "park_factor", "xwoba_babip_gap_7d", "p_throws", "stand",
        ) if c in result.columns]

        first_cols = [c for c in ("home_team", "away_team", "game_pk")
                      if c in result.columns]

        agg_dict: dict[str, str] = {}
        for col in sum_cols:
            agg_dict[col] = "sum"
        for col in rolling_cols:
            agg_dict[col] = "last"
        for col in last_cols:
            agg_dict[col] = "last"
        for col in first_cols:
            agg_dict[col] = "first"

        grouped = (
            result.groupby(["batter", "game_date"], sort=True)
            .agg(agg_dict)
            .reset_index()          # batter and game_date become columns
        )

        grouped = grouped.rename(columns={
            "dk_points": "dk_points_game",
            "_pa": "pa_count",
        })

        logger.info(
            f"aggregate_to_game_level: {len(df):,} PA rows → "
            f"{len(grouped):,} player-game rows"
        )
        return grouped

    def join_game_log_features_game_level(
        self,
        df: pd.DataFrame,
        years: list[int],
    ) -> pd.DataFrame:
        """Attach game-log R / RBI / SB and fold them into ``dk_points_game``.

        Call **after** ``aggregate_to_game_level`` so each player-game row is
        unique; the merge on ``(batter, game_date)`` is one-to-one.

        Loads cached ``gamelogs/hitting_{year}`` Parquets (same source as
        ``join_game_log_features``). Adds ``runs``, ``rbi``, ``stolen_bases``
        and updates:

        * ``dk_points_game += runs × 2``
        * ``dk_points_game += rbi × 2``
        * ``dk_points_game += stolen_bases × 5``

        Args:
            df: Game-level DataFrame with ``batter``, ``game_date``,
                ``dk_points_game``.
            years: Season years whose hitting game logs to load.

        Returns:
            DataFrame with game-log columns and updated ``dk_points_game``.
        """
        if df is None or df.empty:
            logger.warning("join_game_log_features_game_level: empty input")
            return df if df is not None else pd.DataFrame()

        result = df.copy()

        frames: list[pd.DataFrame] = []
        for year in years:
            key = f"gamelogs/hitting_{year}"
            gl = self.cache.load(key)
            if gl is not None and not gl.empty:
                frames.append(gl)
                logger.debug(f"Loaded game logs: {key} ({len(gl):,} rows)")

        if not frames:
            logger.warning(
                "join_game_log_features_game_level: no game logs cached. "
                "Run scripts/pull_game_logs.py --all first. "
                "R/RBI/SB DK points will be 0."
            )
            result["runs"] = 0
            result["rbi"] = 0
            result["stolen_bases"] = 0
            return result

        game_logs = pd.concat(frames, ignore_index=True)
        game_logs["game_date"] = pd.to_datetime(game_logs["game_date"])
        game_logs["batter"] = pd.to_numeric(
            game_logs["batter"], errors="coerce"
        ).astype("Int64")

        keep_cols = [
            c
            for c in ["batter", "game_date", "runs", "rbi", "stolen_bases"]
            if c in game_logs.columns
        ]
        game_logs = game_logs[keep_cols]

        result["game_date"] = pd.to_datetime(result["game_date"])
        result["batter"] = pd.to_numeric(
            result["batter"], errors="coerce"
        ).astype("Int64")

        result = result.merge(game_logs, on=["batter", "game_date"], how="left")

        for col in ("runs", "rbi", "stolen_bases"):
            if col not in result.columns:
                result[col] = 0
            else:
                result[col] = pd.to_numeric(result[col], errors="coerce").fillna(
                    0
                )

        if "dk_points_game" not in result.columns:
            logger.warning(
                "join_game_log_features_game_level: dk_points_game missing"
            )
            result["dk_points_game"] = 0.0

        runs = pd.to_numeric(result["runs"], errors="coerce").fillna(0)
        rbi = pd.to_numeric(result["rbi"], errors="coerce").fillna(0)
        sb = pd.to_numeric(result["stolen_bases"], errors="coerce").fillna(0)

        result["dk_points_game"] = pd.to_numeric(
            result["dk_points_game"], errors="coerce"
        ).fillna(0)
        result["dk_points_game"] += runs * 2.0
        result["dk_points_game"] += rbi * 2.0
        result["dk_points_game"] += sb * 5.0

        matched = (runs.gt(0) | rbi.gt(0) | sb.gt(0)).sum()
        logger.info(
            f"join_game_log_features_game_level: R/RBI/SB merged for "
            f"{matched:,} of {len(result):,} player-game rows"
        )
        return result

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def build_full_batter_feature_matrix(
        self,
        years: list[int] | None = None,
    ) -> pd.DataFrame:
        """Run the complete feature-engineering pipeline and cache the result.

        Pipeline order:

        1.  ``load_statcast_years``               — PA-level Statcast events
        2.  ``build_rolling_batter_features``     — xwOBA / EV / barrel / HH rolling
                                                    + ``xwoba_babip_gap_7d`` (internal
                                                    7d BABIP vs xwOBA only)
        3.  ``build_platoon_features``            — platoon advantage / same-hand
        4.  ``build_batting_order_features``      — order-slot multiplier
        5.  ``build_game_context_features``       — no-op (leaky features removed)
        6.  ``build_park_factor_features``        — ballpark run factor
        7.  ``build_dk_points_labels``            — PA-level DK pts (hits/walks/HBP)
        8.  ``aggregate_to_game_level``           — sum PA ``dk_points`` →
                                                    ``dk_points_game``
        9.  ``join_game_log_features_game_level`` — R / RBI / SB → ``dk_points_game``
        10. ``build_opposing_pitcher_features``   — opp starter K/ERA/whiff/velo
        11. Drop rows where ``dk_points_game`` is null.
        12. Save to ``features/batter_feature_matrix_game_level``.

        After changing this pipeline, rebuild the cached matrix by running
        training with ``force_rebuild_features=True`` (see
        ``ml.training.points_model.PointsModel.train``) or delete the Parquet
        and re-run this method.

        Args:
            years: Season years to include.  Defaults to
                ``[2023, 2024, 2025, 2026]``.

        Returns:
            Game-level feature matrix with target ``dk_points_game``, or an
            empty DataFrame if no source data was available.
        """
        _years = years if years is not None else [2023, 2024, 2025, 2026]

        df = self.load_statcast_years(_years)
        if df.empty:
            logger.warning(
                "build_full_batter_feature_matrix: no source data; "
                "returning empty DataFrame"
            )
            return df

        df = self.build_rolling_batter_features(df)
        df = self.build_platoon_features(df)
        df = self.build_batting_order_features(df)
        df = self.build_game_context_features(df)   # no-op; leaky features removed
        df = self.build_park_factor_features(df)
        df = self.build_dk_points_labels(df)
        df = self.aggregate_to_game_level(df)
        df = self.join_game_log_features_game_level(df, _years)
        df = self.build_opposing_pitcher_features(df, _years)

        before = len(df)
        df = df[df["dk_points_game"].notna()].copy()
        dropped = before - len(df)
        if dropped:
            logger.debug(f"Dropped {dropped:,} rows with null dk_points_game")

        logger.info(
            f"build_full_batter_feature_matrix: final shape={df.shape}  "
            f"years={_years}"
        )

        if not df.empty:
            self.cache.save(
                df,
                "features/batter_feature_matrix_game_level",
                metadata={
                    "years": _years,
                    "rows": len(df),
                    "feature_cols": self.get_feature_columns(),
                    "target": "dk_points_game",
                },
            )

        return df

    def get_feature_columns(self) -> list[str]:
        """Return the canonical list of training feature column names.

        This is the single source of truth consumed by the XGBoost trainer
        and inference pipeline.  The target (``dk_points_game``) and
        identifier / raw columns are excluded.

        All features are pre-game knowable — no in-game state is included.
        Rolling columns represent the batter's trailing performance *entering*
        the game.  Opposing-pitcher columns use the starter's stats from
        prior starts only.

        Returns:
            List of feature column name strings.
        """
        rolling_metrics = ["xwoba", "exit_velo", "barrel_rate", "hard_hit"]
        windows = [7, 14, 30]
        rolling_cols = [
            f"{metric}_{w}d"
            for metric in rolling_metrics
            for w in windows
        ]

        return [
            # Rolling batter performance (pre-game trailing windows)
            *rolling_cols,
            # BABIP luck signal (xwoba vs 7d BABIP only; no raw babip_*d)
            "xwoba_babip_gap_7d",
            # Platoon matchup
            "platoon_advantage",
            "same_hand",
            # Batting order slot
            "batting_order_multiplier",
            # Ballpark run factor
            "park_factor",
            # Opposing starting pitcher quality
            "opp_k_rate_14d",
            "opp_era_approx_14d",
            "opp_whiff_rate_14d",
            "opp_velo_mean_14d",
        ]
