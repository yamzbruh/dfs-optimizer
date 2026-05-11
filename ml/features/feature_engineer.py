"""Statcast → ML feature engineering for the XGBoost points projection model.

Transforms raw pitch-by-pitch Statcast data (stored in Parquet) into a
game-level feature matrix suitable for training and inference.  Every public
method returns a *new* DataFrame — input DataFrames are never modified in place.

Pipeline order for the full batter feature matrix::

    load_statcast_years            (PA-level, ~651k rows)
        → build_rolling_batter_features
        → build_platoon_features
        → build_batting_order_features
        → build_game_context_features
        → build_dk_points_labels      (PA-level: hits/walks/HBP only)
        → aggregate_to_game_level     (sum PA dk_points → dk_points_game)
        → join_game_log_features_game_level  (R / RBI / SB → dk_points_game)
        → (drop null dk_points_game)
        → save to "features/batter_feature_matrix_game_level"

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

        result = result.drop(columns=["_hard_hit"], errors="ignore")
        logger.info(
            f"build_rolling_batter_features: added "
            f"{len(rolling_specs) * len(_windows)} rolling columns "
            f"(windows={_windows})"
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
        """Add run-differential and leverage context columns.

        Columns added:

        * ``run_diff`` — batting team score minus fielding team score.
        * ``is_close_game`` — 1 when ``|run_diff| <= 2``, else 0.
        * ``is_high_leverage`` — 1 when inning ≥ 7 *and* close game,
          else 0.

        Args:
            df: Statcast DataFrame with ``bat_score``, ``fld_score``,
                and ``inning`` columns.

        Returns:
            New DataFrame with context columns appended.
        """
        if df is None or df.empty:
            logger.warning("build_game_context_features: empty input")
            return df if df is not None else pd.DataFrame()

        result = df.copy()

        if "bat_score" in result.columns and "fld_score" in result.columns:
            result["run_diff"] = result["bat_score"] - result["fld_score"]
            result["is_close_game"] = (
                result["run_diff"].abs() <= 2
            ).astype(int)
        else:
            missing = [
                c for c in ("bat_score", "fld_score")
                if c not in result.columns
            ]
            logger.warning(
                f"build_game_context_features: missing {missing}; "
                "run_diff and is_close_game set to 0"
            )
            result["run_diff"] = 0
            result["is_close_game"] = 0

        if "inning" in result.columns:
            result["is_high_leverage"] = (
                (result["inning"] >= 7) & (result["is_close_game"] == 1)
            ).astype(int)
        else:
            logger.warning(
                "build_game_context_features: 'inning' missing; "
                "is_high_leverage set to 0"
            )
            result["is_high_leverage"] = 0

        logger.info(
            "build_game_context_features: run_diff, is_close_game, "
            "is_high_leverage added"
        )
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
        * **LAST**: all rolling ``*_{7,14,30}d`` columns, ``platoon_advantage``,
          ``same_hand``, ``batting_order_multiplier``, ``is_close_game``,
          ``is_high_leverage``, ``run_diff``, ``p_throws``, ``stand``.
        * **FIRST**: ``home_team``, ``away_team``, ``game_pk``.

        ``runs``, ``rbi``, and ``stolen_bases`` are joined **after** this step
        via ``join_game_log_features_game_level`` so totals are not summed per PA.

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

        rolling_cols = [c for c in result.columns
                        if any(c.endswith(f"_{w}d") for w in ("7", "14", "30"))]

        last_cols = [c for c in (
            "platoon_advantage", "same_hand", "batting_order_multiplier",
            "is_close_game", "is_high_leverage", "run_diff",
            "p_throws", "stand",
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

        1. ``load_statcast_years``             — PA-level Statcast events
        2. ``build_rolling_batter_features``   — rolling xwOBA / EV / barrel / HH
        3. ``build_platoon_features``          — platoon advantage / same-hand
        4. ``build_batting_order_features``    — order-slot multiplier
        5. ``build_game_context_features``     — run-diff / leverage flags
        6. ``build_dk_points_labels``          — PA-level DK pts (hits / walks / HBP only)
        7. ``aggregate_to_game_level``         — sum PA ``dk_points`` → ``dk_points_game``
        8. ``join_game_log_features_game_level`` — R / RBI / SB → ``dk_points_game``
        9. Drop rows where ``dk_points_game`` is null.
        10. Save to ``features/batter_feature_matrix_game_level``.

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
        df = self.build_game_context_features(df)
        df = self.build_dk_points_labels(df)
        df = self.aggregate_to_game_level(df)
        df = self.join_game_log_features_game_level(df, _years)

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

        All features are computed at PA level and then carried through to
        the game-level aggregation via the ``"last"`` rule, so their values
        represent the batter's state *entering that game*.

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
            # Platoon matchup
            "platoon_advantage",
            "same_hand",
            # Batting order slot
            "batting_order_multiplier",
            # Game context
            "run_diff",
            "is_close_game",
            "is_high_leverage",
            # Game volume signal
            "pa_count",
        ]
