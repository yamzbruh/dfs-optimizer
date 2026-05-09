"""Statcast → ML feature engineering for the XGBoost points projection model.

Transforms raw pitch-by-pitch Statcast data (stored in Parquet) into a
feature matrix suitable for training and inference.  Every public method
returns a *new* DataFrame — input DataFrames are never modified in place.

Pipeline order for the full batter feature matrix::

    load_statcast_years
        → build_rolling_batter_features
        → build_platoon_features
        → build_game_context_features
        → build_dk_points_labels
        → (drop null dk_points)
        → save to cache
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

# DK MLB classic batting scoring (per plate-appearance event).
_DK_EVENT_POINTS: dict[str, float] = {
    "single": 3.0,
    "double": 5.0,
    "triple": 8.0,
    "home_run": 10.0,
    "walk": 3.0,
    "hit_by_pitch": 3.0,
    # Strikeout penalty — not part of standard DK scoring, but a
    # useful negative signal for projection.  Included as an optional
    # feature; downstream you can zero it out if you prefer strict DK
    # scoring.
    "strikeout": -0.5,
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
        y = matrix["dk_points"]
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

        Scoring applied:

        * Single: +3 | Double: +5 | Triple: +8 | Home run: +10
        * Walk: +3 | Hit by pitch: +3
        * Strikeout: −0.5 (optional negative signal; not standard DK
          scoring — set to 0 downstream if strict DK scoring is needed)

        Components intentionally omitted (require box-score join):

        * Runs scored (+2 pts) — not in Statcast pitch-by-pitch.
        * RBIs (+2 pts) — same reason.
        * Stolen bases (+5 pts) — see note in ``StatcastLoader``.

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

        result["dk_points"] = (
            result["events"].map(_DK_EVENT_POINTS).fillna(0.0)
        )
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
    # Full pipeline
    # ------------------------------------------------------------------

    def build_full_batter_feature_matrix(
        self,
        years: list[int] | None = None,
    ) -> pd.DataFrame:
        """Run the complete feature-engineering pipeline and cache the result.

        Pipeline order:

        1. ``load_statcast_years``
        2. ``build_rolling_batter_features``
        3. ``build_platoon_features``
        4. ``build_game_context_features``
        5. ``build_dk_points_labels``
        6. Drop rows where ``dk_points`` is null.
        7. Save to ``features/batter_feature_matrix``.

        Args:
            years: Season years to include.  Defaults to
                ``[2023, 2024, 2025, 2026]``.

        Returns:
            Final feature matrix, or an empty DataFrame if no source
            data was available.
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

        before = len(df)
        df = df[df["dk_points"].notna()].copy()
        dropped = before - len(df)
        if dropped:
            logger.debug(f"Dropped {dropped:,} rows with null dk_points")

        logger.info(
            f"build_full_batter_feature_matrix: final shape={df.shape}  "
            f"years={_years}"
        )

        if not df.empty:
            self.cache.save(
                df,
                "features/batter_feature_matrix",
                metadata={
                    "years": _years,
                    "rows": len(df),
                    "feature_cols": self.get_feature_columns(),
                },
            )

        return df

    def get_feature_columns(self) -> list[str]:
        """Return the canonical list of training feature column names.

        This is the single source of truth consumed by the XGBoost
        trainer and inference pipeline.  The target (``dk_points``) and
        identifier / raw columns are excluded.

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
            # Rolling batter performance
            *rolling_cols,
            # Platoon
            "platoon_advantage",
            "same_hand",
            # Batting order
            "batting_order_multiplier",
            # Game context
            "run_diff",
            "is_close_game",
            "is_high_leverage",
        ]
