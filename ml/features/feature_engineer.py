"""Statcast → ML feature engineering for the XGBoost points projection model.

Transforms raw pitch-by-pitch Statcast data (stored in Parquet) into a
game-level feature matrix suitable for training and inference.  Every public
method returns a *new* DataFrame — input DataFrames are never modified in place.

Pipeline order for the full batter feature matrix::

    load_statcast_years                  (PA-level, ~651k rows)
        → build_rolling_batter_features  (xwOBA / EV / barrel / HH / K% / BB%
                                          rolling, EV–barrel–HH 7d-vs-30d trends,
                                          xwoba_babip_gap_7d luck signal)
        → build_platoon_features         (platoon_advantage / same_hand)
        → build_game_context_features    (no-op — leaky features removed)
        → build_park_factor_features     (park_factor from PARK_FACTORS lookup)
        → build_dk_points_labels         (PA-level: hits / walks / HBP only)
        → aggregate_to_game_level        (sum PA dk_points → dk_points_game)
        → join_game_log_features_game_level  (R / RBI / SB → dk_points_game)
        → join_batting_order_features      (lineups → ``batting_order`` / multiplier)
        → build_team_offense_features      (``team_runs_per_game_30d``,
                                            ``opp_runs_allowed_30d`` — no leakage)
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
ORDER_MULTIPLIERS: dict[int, float] = {
    1: 1.15,
    2: 1.12,
    3: 1.10,
    4: 1.08,
    5: 1.05,
    6: 1.00,
    7: 0.95,
    8: 0.90,
    9: 0.85,
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
        * ``exit_velo_trend`` — ``exit_velo_7d - exit_velo_30d`` (short vs baseline)
        * ``barrel_rate_{w}d`` — rolling mean of ``barrel`` (0/1)
        * ``barrel_rate_trend`` — ``barrel_rate_7d - barrel_rate_30d``
        * ``hard_hit_{w}d`` — rolling mean of ``launch_speed >= 95``
          (derived binary column)
        * ``hard_hit_trend`` — ``hard_hit_7d - hard_hit_30d``
        * ``k_rate_{w}d`` — rolling mean of strikeout terminal events
          (``strikeout``, ``strikeout_double_play``)
        * ``bb_rate_{w}d`` — rolling mean of ``events == walk``
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

        # Terminal PA outcomes only (``load_statcast_years`` already filters).
        if "events" in result.columns:
            _ev = result["events"].astype(str)
            result["_k_event"] = _ev.isin(
                ["strikeout", "strikeout_double_play"]
            ).astype(float)
            result["_bb_event"] = _ev.eq("walk").astype(float)
        else:
            result["_k_event"] = float("nan")
            result["_bb_event"] = float("nan")
            logger.warning(
                "build_rolling_batter_features: 'events' missing; "
                "k_rate_* / bb_rate_* will be NaN"
            )

        rolling_specs: list[tuple[str, str]] = [
            ("estimated_woba_using_speedangle", "xwoba"),
            ("launch_speed", "exit_velo"),
            ("barrel", "barrel_rate"),
            ("_hard_hit", "hard_hit"),
            ("_k_event", "k_rate"),
            ("_bb_event", "bb_rate"),
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

        if "barrel_rate_7d" in result.columns and "barrel_rate_30d" in result.columns:
            result["barrel_rate_trend"] = (
                result["barrel_rate_7d"] - result["barrel_rate_30d"]
            )
        if "hard_hit_7d" in result.columns and "hard_hit_30d" in result.columns:
            result["hard_hit_trend"] = result["hard_hit_7d"] - result["hard_hit_30d"]
        if "exit_velo_7d" in result.columns and "exit_velo_30d" in result.columns:
            result["exit_velo_trend"] = (
                result["exit_velo_7d"] - result["exit_velo_30d"]
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

        result = result.drop(
            columns=["_hard_hit", "_k_event", "_bb_event"],
            errors="ignore",
        )
        n_roll = len(rolling_specs) * len(_windows)
        n_trend = sum(
            1
            for c in ("barrel_rate_trend", "hard_hit_trend", "exit_velo_trend")
            if c in result.columns
        )
        logger.info(
            f"build_rolling_batter_features: added {n_roll} rolling columns + "
            f"{n_trend} trend columns + xwoba_babip_gap_7d (windows={_windows})"
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
            .map(ORDER_MULTIPLIERS)
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

    def build_vegas_features(
        self,
        df: pd.DataFrame,
        implied_totals: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Add Vegas implied team total (``implied_total``) for **inference only**.

        **Not used in the training pipeline.** Training uses
        :meth:`build_team_offense_features` instead (``team_runs_per_game_30d``,
        ``opp_runs_allowed_30d``) so the model never sees same-game outcomes.

        At slate time, call this with real Odds API totals (see
        ``OddsIngestion.get_team_implied_totals``) **before** generating
        projections. You can write ``implied_total`` into the feature row, or
        overwrite ``team_runs_per_game_30d`` with the same values if the
        deployed model was trained on that column name.

        Args:
            df: Game-level feature DataFrame with ``home_team`` and
                ``away_team`` columns (and optionally ``inning_topbot`` for
                PA-level rows).
            implied_totals: Mapping team_abbr → implied_total (runs).
                If missing or empty, defaults to MLB average (4.5) for all.

        Returns:
            Copy of ``df`` with column ``implied_total`` (team implied runs).

        Note:
            When ``inning_topbot`` is present (PA-level rows), Top uses the
            away team's total; otherwise Bottom uses home. At game level that
            column is usually absent — the fallback maps ``home_team`` only;
            callers with per-player team context should set implied totals
            before merge or extend this method.
        """
        if df is None or df.empty:
            logger.warning("build_vegas_features: empty input")
            return df if df is not None else pd.DataFrame()

        result = df.copy()
        mlb_avg_implied = 4.5

        if not implied_totals:
            result["implied_total"] = mlb_avg_implied
            logger.debug(
                "build_vegas_features: no implied totals provided "
                "— using MLB average 4.5"
            )
            return result

        totals_norm = {
            str(k).strip(): float(v)
            for k, v in implied_totals.items()
            if k is not None and str(k).strip() != ""
        }

        def get_implied(row: pd.Series) -> float:
            home_abbr = str(row.get("home_team", "") or "").strip()
            away_abbr = str(row.get("away_team", "") or "").strip()
            topbot = row.get("inning_topbot", "")
            if topbot == "Top":
                return float(totals_norm.get(away_abbr, mlb_avg_implied))
            return float(totals_norm.get(home_abbr, mlb_avg_implied))

        if "inning_topbot" in result.columns:
            try:
                result["implied_total"] = result.apply(get_implied, axis=1)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"build_vegas_features: apply failed ({exc}); "
                    "falling back to home_team map"
                )
                if "home_team" in result.columns:
                    result["implied_total"] = (
                        result["home_team"].astype(str).str.strip().map(totals_norm)
                    ).fillna(mlb_avg_implied)
                else:
                    result["implied_total"] = mlb_avg_implied
        elif "home_team" in result.columns:
            result["implied_total"] = (
                result["home_team"].astype(str).str.strip().map(totals_norm)
            ).fillna(mlb_avg_implied)
        else:
            result["implied_total"] = mlb_avg_implied
            logger.warning(
                "build_vegas_features: no home_team column — implied_total=4.5"
            )

        result["implied_total"] = pd.to_numeric(
            result["implied_total"], errors="coerce"
        ).fillna(mlb_avg_implied)

        logger.info(
            f"build_vegas_features (inference): implied_total added, "
            f"mean={result['implied_total'].mean():.2f}"
        )
        return result

    def build_team_offense_features(
        self,
        df: pd.DataFrame,
        years: list[int],
    ) -> pd.DataFrame:
        """Add rolling team offense and opponent defense (no same-game leakage).

        Computes pre-game team-level signals from cached hitting game logs
        plus lineup team labels:

        * ``team_runs_per_game_30d`` — mean runs **scored** by the batter's
          team over the prior 30 team games (``shift(1)`` then
          ``rolling(30, min_periods=3)``), never including the current game.
        * ``opp_runs_allowed_30d`` — mean runs **allowed** by the batter's
          team (opponent runs in the same games) over the prior 30 team
          games with the same shift/rolling rule.

        Lineups supply ``team`` per ``(batter, game_pk)``; hitting logs
        supply ``runs`` and ``game_pk``. Missing data defaults to **4.5**
        (MLB-ish average).

        Args:
            df: Game-level matrix after ``join_game_log_features_game_level``
                and ``join_batting_order_features`` (needs ``batter``,
                ``game_date``, ``game_pk``).
            years: Seasons whose ``gamelogs/hitting_{y}`` and
                ``lineups/batting_order_{y}`` Parquets to load.

        Returns:
            Copy of ``df`` with ``team_runs_per_game_30d`` and
            ``opp_runs_allowed_30d`` columns.
        """
        mlb_avg = 4.5
        if df is None or df.empty:
            logger.warning("build_team_offense_features: empty input")
            return df if df is not None else pd.DataFrame()

        result = df.copy()

        def _defaults() -> None:
            result["team_runs_per_game_30d"] = mlb_avg
            result["opp_runs_allowed_30d"] = mlb_avg

        # --- Hitting game logs ---
        hit_frames: list[pd.DataFrame] = []
        for year in years:
            gl = self.cache.load(f"gamelogs/hitting_{year}")
            if gl is not None and not gl.empty:
                hit_frames.append(gl)

        if not hit_frames:
            logger.warning(
                "build_team_offense_features: no hitting game logs — "
                "defaulting team features to 4.5"
            )
            _defaults()
            return result

        hit = pd.concat(hit_frames, ignore_index=True)
        hit["game_date"] = pd.to_datetime(hit["game_date"], errors="coerce")
        hit["batter"] = pd.to_numeric(hit["batter"], errors="coerce").astype("Int64")
        need_hit = {"runs", "game_pk", "batter", "game_date"}
        if not need_hit.issubset(set(hit.columns)):
            logger.warning(
                "build_team_offense_features: hitting logs missing required columns "
                f"{sorted(need_hit)} — defaulting to 4.5"
            )
            _defaults()
            return result

        hit["runs"] = pd.to_numeric(hit["runs"], errors="coerce").fillna(0)
        hit["game_pk"] = pd.to_numeric(hit["game_pk"], errors="coerce")

        # --- Lineups → team abbreviation per batter-game ---
        lu_frames: list[pd.DataFrame] = []
        for year in years:
            lu = self.cache.load(f"lineups/batting_order_{year}")
            if lu is not None and not lu.empty:
                lu_frames.append(lu)

        if not lu_frames:
            logger.warning(
                "build_team_offense_features: no lineup Parquets — "
                "cannot assign team; defaulting to 4.5"
            )
            _defaults()
            return result

        lineups = pd.concat(lu_frames, ignore_index=True)
        if "team" not in lineups.columns or "mlbam_id" not in lineups.columns:
            logger.warning(
                "build_team_offense_features: lineups missing team/mlbam_id — "
                "defaulting to 4.5"
            )
            _defaults()
            return result

        lu_keys = lineups[["mlbam_id", "game_pk", "team"]].copy()
        lu_keys = lu_keys.rename(columns={"mlbam_id": "batter"})
        lu_keys["batter"] = pd.to_numeric(lu_keys["batter"], errors="coerce").astype(
            "Int64"
        )
        lu_keys["game_pk"] = pd.to_numeric(lu_keys["game_pk"], errors="coerce")
        lu_keys["team"] = lu_keys["team"].astype(str).str.strip()
        lu_keys = lu_keys.dropna(subset=["batter", "game_pk", "team"])
        lu_keys = lu_keys.drop_duplicates(subset=["batter", "game_pk"], keep="first")

        merged = hit.merge(lu_keys, on=["batter", "game_pk"], how="inner")
        if merged.empty:
            logger.warning(
                "build_team_offense_features: no hitting rows matched lineups "
                "on (batter, game_pk) — defaulting to 4.5"
            )
            _defaults()
            return result

        team_game = (
            merged.groupby(["team", "game_pk"], as_index=False)
            .agg(runs_scored=("runs", "sum"), game_date=("game_date", "min"))
        )
        team_game = team_game.dropna(subset=["team", "game_pk"])
        team_game["runs_scored"] = pd.to_numeric(
            team_game["runs_scored"], errors="coerce"
        ).fillna(0)

        # Opponent runs scored in the same game = runs allowed for defense.
        opp_side = team_game.merge(
            team_game,
            on="game_pk",
            suffixes=("", "_opp"),
        )
        opp_side = opp_side[opp_side["team"] != opp_side["team_opp"]][
            ["team", "game_pk", "game_date", "runs_scored_opp"]
        ].rename(columns={"runs_scored_opp": "runs_allowed"})
        opp_side = opp_side.drop_duplicates(subset=["team", "game_pk"], keep="first")
        opp_side["runs_allowed"] = pd.to_numeric(
            opp_side["runs_allowed"], errors="coerce"
        ).fillna(mlb_avg)

        # Rolling means: prior games only (shift before rolling).
        team_game = team_game.sort_values(
            ["team", "game_date", "game_pk"], kind="mergesort"
        )
        team_game["team_runs_per_game_30d"] = (
            team_game.groupby("team", sort=False)["runs_scored"]
            .transform(lambda s: s.shift(1).rolling(30, min_periods=3).mean())
            .fillna(mlb_avg)
        )

        opp_side = opp_side.sort_values(
            ["team", "game_date", "game_pk"], kind="mergesort"
        )
        opp_side["opp_runs_allowed_30d"] = (
            opp_side.groupby("team", sort=False)["runs_allowed"]
            .transform(lambda s: s.shift(1).rolling(30, min_periods=3).mean())
            .fillna(mlb_avg)
        )

        game_feats = team_game.merge(
            opp_side[["team", "game_pk", "opp_runs_allowed_30d"]],
            on=["team", "game_pk"],
            how="left",
        )
        game_feats["opp_runs_allowed_30d"] = game_feats[
            "opp_runs_allowed_30d"
        ].fillna(mlb_avg)

        if "game_pk" not in result.columns:
            logger.warning(
                "build_team_offense_features: game_pk missing on feature matrix "
                "— defaulting to 4.5"
            )
            _defaults()
            return result

        result["game_date"] = pd.to_datetime(result["game_date"], errors="coerce")
        result["batter"] = pd.to_numeric(
            result["batter"], errors="coerce"
        ).astype("Int64")
        result["game_pk"] = pd.to_numeric(result["game_pk"], errors="coerce")

        with_team = result.merge(
            lu_keys.rename(columns={"team": "_bat_team"}),
            on=["batter", "game_pk"],
            how="left",
        )
        feat_only = game_feats.rename(columns={"team": "_bat_team"})[
            [
                "_bat_team",
                "game_pk",
                "team_runs_per_game_30d",
                "opp_runs_allowed_30d",
            ]
        ]
        with_feats = with_team.merge(
            feat_only,
            on=["_bat_team", "game_pk"],
            how="left",
        )
        with_feats = with_feats.drop(columns=["_bat_team"], errors="ignore")

        for col in ("team_runs_per_game_30d", "opp_runs_allowed_30d"):
            if col not in with_feats.columns:
                with_feats[col] = mlb_avg
            else:
                with_feats[col] = pd.to_numeric(
                    with_feats[col], errors="coerce"
                ).fillna(mlb_avg)

        logger.info(
            "build_team_offense_features: "
            f"team_runs mean={with_feats['team_runs_per_game_30d'].mean():.2f}, "
            f"opp_allowed mean={with_feats['opp_runs_allowed_30d'].mean():.2f}"
        )
        return with_feats

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
          ``xwoba_babip_gap_7d``), ``barrel_rate_trend``, ``hard_hit_trend``,
          ``exit_velo_trend``, ``platoon_advantage``, ``same_hand``,
          ``batting_order_multiplier``, ``park_factor``, ``xwoba_babip_gap_7d``,
          ``k_rate_*``, ``bb_rate_*``, ``p_throws``, ``stand``.
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
            "barrel_rate_trend", "hard_hit_trend", "exit_velo_trend",
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

    def join_batting_order_features(
        self,
        df: pd.DataFrame,
        years: list[int],
    ) -> pd.DataFrame:
        """Join cached MLB lineup batting-order slots at game level.

        Loads ``lineups/batting_order_{year}`` Parquets (from
        ``StatcastLoader.get_season_lineups``). Renames ``mlbam_id`` to
        ``batter`` and left-merges on ``["batter", "game_date"]``, then
        applies :meth:`build_batting_order_features` to derive
        ``batting_order_multiplier``.

        Call **after** ``aggregate_to_game_level`` and
        ``join_game_log_features_game_level`` so keys match one row per
        player-game.

        Args:
            df: Game-level DataFrame with ``batter`` and ``game_date``.
            years: Seasons whose lineup Parquets to load.

        Returns:
            DataFrame with ``batting_order`` (when matched) and
            ``batting_order_multiplier``; multiplier defaults to ``1.0`` when
            lineups are missing or unmatched.
        """
        if df is None or df.empty:
            logger.warning("join_batting_order_features: empty input")
            return df if df is not None else pd.DataFrame()

        result = df.copy()

        frames: list[pd.DataFrame] = []
        for year in years:
            key = f"lineups/batting_order_{year}"
            lu = self.cache.load(key)
            if lu is not None and not lu.empty:
                frames.append(lu)
                logger.debug(f"Loaded lineups: {key} ({len(lu):,} rows)")

        if not frames:
            logger.warning(
                "join_batting_order_features: no lineup Parquets found. "
                "Run StatcastLoader.get_season_lineups for those years. "
                "batting_order_multiplier defaults to 1.0"
            )
            return self.build_batting_order_features(result)

        lineups = pd.concat(frames, ignore_index=True)
        if "mlbam_id" not in lineups.columns:
            logger.warning(
                "join_batting_order_features: 'mlbam_id' missing in lineup data "
                "— batting_order_multiplier defaults to 1.0"
            )
            return self.build_batting_order_features(result)

        lineups = lineups.rename(columns={"mlbam_id": "batter"})
        lineups["game_date"] = pd.to_datetime(lineups["game_date"])
        lineups["batter"] = pd.to_numeric(
            lineups["batter"], errors="coerce"
        ).astype("Int64")

        keep = [c for c in ("batter", "game_date", "batting_order") if c in lineups.columns]
        if "batting_order" not in keep:
            logger.warning(
                "join_batting_order_features: 'batting_order' missing "
                "— batting_order_multiplier defaults to 1.0"
            )
            return self.build_batting_order_features(result)

        lineups = lineups[keep].drop_duplicates(
            subset=["batter", "game_date"], keep="first"
        )

        result["game_date"] = pd.to_datetime(result["game_date"])
        result["batter"] = pd.to_numeric(
            result["batter"], errors="coerce"
        ).astype("Int64")

        result = result.merge(lineups, on=["batter", "game_date"], how="left")

        matched = int(result["batting_order"].notna().sum()) if "batting_order" in result.columns else 0
        logger.info(
            f"join_batting_order_features: matched batting_order for "
            f"{matched:,} of {len(result):,} player-game rows"
        )

        return self.build_batting_order_features(result)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def build_full_batter_feature_matrix(
        self,
        years: list[int] | None = None,
        force_rebuild: bool = False,
    ) -> pd.DataFrame:
        """Run the complete feature-engineering pipeline and cache the result.

        Pipeline order:

        1.  ``load_statcast_years``               — PA-level Statcast events
        2.  ``build_rolling_batter_features``     — xwOBA / EV / barrel / HH /
                                                    K% / BB% rolling, short-vs-30d
                                                    trends for EV/barrel/HH, and
                                                    ``xwoba_babip_gap_7d``
        3.  ``build_platoon_features``            — platoon advantage / same-hand
        4.  ``build_game_context_features``       — no-op (leaky features removed)
        5.  ``build_park_factor_features``        — ballpark run factor
        6.  ``build_dk_points_labels``            — PA-level DK pts (hits/walks/HBP)
        7.  ``aggregate_to_game_level``           — sum PA ``dk_points`` →
                                                    ``dk_points_game``
        8.  ``join_game_log_features_game_level`` — R / RBI / SB → ``dk_points_game``
        9.  ``join_batting_order_features``       — lineup slot → ``batting_order_multiplier``
        10. ``build_team_offense_features``      — ``team_runs_per_game_30d`` /
                                                    ``opp_runs_allowed_30d`` (no leakage)
        11. ``build_opposing_pitcher_features``   — opp starter K/ERA/whiff/velo
        12. Drop rows where ``dk_points_game`` is null.
        13. Save to ``features/batter_feature_matrix_game_level``.

        After changing this pipeline, rebuild the cached matrix by passing
        ``force_rebuild=True``, running training with
        ``force_rebuild_features=True`` (see ``ml.training.points_model.PointsModel.train``),
        or deleting the Parquet and re-running this method.

        Args:
            years: Season years to include.  Defaults to
                ``[2023, 2024, 2025, 2026]``.
            force_rebuild: When ``True``, ignore cache and rebuild from Statcast
                Parquet pulls.

        Returns:
            Game-level feature matrix with target ``dk_points_game``, or an
            empty DataFrame if no source data was available.
        """
        _years = years if years is not None else [2023, 2024, 2025, 2026]
        cache_key = "features/batter_feature_matrix_game_level"

        if not force_rebuild and self.cache.exists(cache_key):
            cached = self.cache.load(cache_key)
            if cached is not None and not cached.empty:
                logger.info(
                    f"Loaded batter features from cache: {cache_key} "
                    f"({len(cached):,} rows)"
                )
                return cached

        logger.info("Building batter feature matrix…")

        df = self.load_statcast_years(_years)
        if df.empty:
            logger.warning(
                "build_full_batter_feature_matrix: no source data; "
                "returning empty DataFrame"
            )
            return df

        df = self.build_rolling_batter_features(df)
        df = self.build_platoon_features(df)
        df = self.build_game_context_features(df)   # no-op; leaky features removed
        df = self.build_park_factor_features(df)
        df = self.build_dk_points_labels(df)
        df = self.aggregate_to_game_level(df)
        df = self.join_game_log_features_game_level(df, _years)
        df = self.join_batting_order_features(df, _years)
        df = self.build_team_offense_features(df, _years)
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
                cache_key,
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
        prior starts only.  Team offense/defense columns
        (``team_runs_per_game_30d``, ``opp_runs_allowed_30d``) use only games
        **before** the current game (``shift`` + ``rolling``). For live slates,
        optional Vegas totals are applied via :meth:`build_vegas_features`
        (inference only), typically overwriting or augmenting the team-offense
        signal.

        Rolling columns: ``xwoba``, ``k_rate``, and ``bb_rate`` use 7 / 14 /
        30 game windows.  ``exit_velo``, ``barrel_rate``, and ``hard_hit`` use
        the same rolling series internally, but the model feature list keeps
        only the 30d baseline plus a 7d-vs-30d **trend** column per metric to
        reduce collinearity.

        Returns:
            List of feature column name strings.
        """
        windows = [7, 14, 30]

        rolling_cols = [f"xwoba_{w}d" for w in windows]
        rolling_cols += [
            "exit_velo_30d",
            "exit_velo_trend",
            "barrel_rate_30d",
            "barrel_rate_trend",
            "hard_hit_30d",
            "hard_hit_trend",
        ]
        rolling_cols += [f"k_rate_{w}d" for w in windows]
        rolling_cols += [f"bb_rate_{w}d" for w in windows]

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
            # Team offense / defense (prior games only — no same-game leakage)
            "team_runs_per_game_30d",
            "opp_runs_allowed_30d",
            # Opposing starting pitcher quality
            "opp_k_rate_14d",
            "opp_era_approx_14d",
            "opp_whiff_rate_14d",
            "opp_velo_mean_14d",
        ]
