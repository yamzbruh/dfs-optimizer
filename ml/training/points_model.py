"""XGBoost quantile regression points projection model.

Trains three separate XGBRegressor models — one for each quantile
(q15 / q50 / q85) — on game-level batter features.  Each training row
is one player-game; the target is ``dk_points_game`` — the total DK
points that player scored across the entire game (hits + R + RBI + SB
+ walks + HBP).

The three models give a floor, median, and ceiling projection for every
player on a slate.  Quantile monotonicity (q15 ≤ q50 ≤ q85) is enforced
explicitly after prediction.

High ``interval_width`` (q85 − q15) signals high-variance players that
are natural GPP targets for the optimizer.

Feature matrix cache key: ``features/batter_feature_matrix_game_level``
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_pipeline.loaders.parquet_cache import ParquetCache  # noqa: E402
from ml.features.feature_engineer import FeatureEngineer  # noqa: E402
from ml.features.pitcher_feature_engineer import PitcherFeatureEngineer  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUANTILES: list[float] = [0.15, 0.50, 0.85]
QUANTILE_NAMES: list[str] = ["q15", "q50", "q85"]
MODEL_DIR = Path("data/models")

XGBOOST_PARAMS: dict = {
    "objective": "reg:quantileerror",
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
    "enable_categorical": True,
}

# Columns that are never features: identifiers, raw inputs, and targets.
_NON_FEATURE_COLS: frozenset[str] = frozenset({
    "batter", "game_date", "events",
    "dk_points", "dk_points_game",          # both target variants excluded
    "runs", "rbi", "stolen_bases",           # game-log raw totals (in target)
    "pitcher", "player_name",
})


class PointsModel:
    """Three XGBoost quantile regressors (q15 / q50 / q85) for DK points.

    Trained on game-level data — one row per player per game — with
    target ``dk_points_game`` (total DK score for that game).

    Usage::

        model = PointsModel()
        metrics = model.train(years=[2023, 2024, 2025])
        run_path = model.save_models()

        # Later, at inference time:
        model2 = PointsModel()
        model2.load_models("20260508_183000")
        preds = model2.predict(slate_features_df)
        # preds has columns: q15, q50, q85
    """

    def __init__(self, model_dir: str | Path = "data/models") -> None:
        """Initialise the model container and create the model directory."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.models: dict[str, XGBRegressor] = {}
        self.feature_columns: list[str] = []
        self.shap_values: dict[str, np.ndarray] = {}

        self._cache = ParquetCache()
        logger.debug(f"PointsModel ready  model_dir={self.model_dir.resolve()}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        years: list[int] | None = None,
        holdout_year: int = 2025,
        holdout_month: int = 5,
        force_rebuild_features: bool = False,
    ) -> dict:
        """Train q15 / q50 / q85 quantile regressors on game-level data.

        The feature matrix used is the game-level cache written by
        ``FeatureEngineer.build_full_batter_feature_matrix()``.  Each row
        is one player-game; the target is ``dk_points_game``.

        Args:
            years: All seasons to include in the feature matrix.
                Defaults to ``[2023, 2024, 2025, 2026]``.
            holdout_year: Year component of the holdout window.
            holdout_month: Month component of the holdout window.
                Training uses every row *outside* this year+month;
                evaluation uses every row *inside* it.
                Default: May 2025 — recent enough to be representative,
                old enough that 2026 data is untouched.
            force_rebuild_features: When ``True``, ignore the cached
                feature matrix and rebuild from Statcast Parquet files.

        Returns:
            Metrics dict with RMSE, MAE, coverage, and interval width.
        """
        _years = years if years is not None else [2023, 2024, 2025, 2026]
        t_total = time.time()

        # -- a. Load or build feature matrix ---------------------------------
        df = self._load_feature_matrix(
            train_years=_years,
            force_rebuild=force_rebuild_features,
        )
        if df is None or df.empty:
            raise RuntimeError(
                "No feature matrix available — run "
                "FeatureEngineer().build_full_batter_feature_matrix() first "
                "or pass force_rebuild_features=True."
            )

        # -- b. Train / test split by year+month holdout ---------------------
        if "game_date" not in df.columns:
            raise RuntimeError("Feature matrix missing 'game_date' column.")

        df["game_date"] = pd.to_datetime(df["game_date"])
        test_mask = (
            (df["game_date"].dt.year == holdout_year)
            & (df["game_date"].dt.month == holdout_month)
        )
        train_mask = ~test_mask

        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()

        import calendar
        month_name = calendar.month_name[holdout_month]
        logger.info(
            f"Train: {len(train_df):,} rows "
            f"(all data except {month_name} {holdout_year})"
        )
        logger.info(
            f"Test holdout: {len(test_df):,} rows "
            f"({month_name} {holdout_year})"
        )

        if train_df.empty:
            raise RuntimeError("No training data found for the given years.")
        if test_df.empty:
            logger.warning(
                f"No holdout data found for {month_name} {holdout_year}; "
                "evaluation metrics will be skipped."
            )

        # -- c. Resolve feature columns --------------------------------------
        all_features = FeatureEngineer().get_feature_columns()
        present = [c for c in all_features if c in df.columns]
        missing = [c for c in all_features if c not in df.columns]

        if missing:
            logger.warning(f"Features missing from matrix (will be skipped): {missing}")
        logger.info(f"Training on {len(present)} features: {present}")

        self.feature_columns = present
        X_train = train_df[present].fillna(0.0)
        y_train = train_df["dk_points_game"].fillna(0.0)
        X_test = test_df[present].fillna(0.0) if not test_df.empty else pd.DataFrame()
        y_test = test_df["dk_points_game"].fillna(0.0) if not test_df.empty else pd.Series(dtype=float)

        # -- d. Fit one model per quantile -----------------------------------
        raw_preds: dict[str, np.ndarray] = {}
        per_q_metrics: dict[str, dict[str, float]] = {}

        for quantile, name in zip(QUANTILES, QUANTILE_NAMES):
            logger.info(f"Training {name} (alpha={quantile})…")
            t0 = time.time()

            model = XGBRegressor(**{**XGBOOST_PARAMS, "quantile_alpha": quantile})
            model.fit(X_train, y_train)
            self.models[name] = model

            elapsed = time.time() - t0
            logger.info(f"  {name} trained in {elapsed:.1f}s")

            if not X_test.empty:
                preds = model.predict(X_test)
                raw_preds[name] = preds
                rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
                mae = float(mean_absolute_error(y_test, preds))
                per_q_metrics[name] = {"rmse": rmse, "mae": mae}
                logger.info(f"  {name}  RMSE={rmse:.4f}  MAE={mae:.4f}")

        # -- e. Enforce quantile monotonicity --------------------------------
        if not X_test.empty and all(n in raw_preds for n in QUANTILE_NAMES):
            q15 = raw_preds["q15"]
            q50 = raw_preds["q50"]
            q85 = raw_preds["q85"]

            q15_violations = int(np.sum(q15 > q50))
            q85_violations = int(np.sum(q85 < q50))

            q15_fixed = np.minimum(q15, q50)
            q85_fixed = np.maximum(q85, q50)

            raw_preds["q15"] = q15_fixed
            raw_preds["q85"] = q85_fixed

            if q15_violations:
                logger.info(
                    f"Monotonicity correction (q15): {q15_violations:,} rows "
                    f"where q15 > q50 — clipped down"
                )
            if q85_violations:
                logger.info(
                    f"Monotonicity correction (q85): {q85_violations:,} rows "
                    f"where q85 < q50 — clipped up"
                )

        # -- f. Calibration coverage -----------------------------------------
        coverage: dict[str, float] = {}
        interval_width: float = 0.0

        if not X_test.empty and all(n in raw_preds for n in QUANTILE_NAMES):
            y_arr = y_test.to_numpy()
            q15_arr = raw_preds["q15"]
            q85_arr = raw_preds["q85"]

            q15_coverage = float(np.mean(y_arr < q15_arr))
            q85_coverage = float(np.mean(y_arr < q85_arr))
            interval_width = float(np.mean(q85_arr - q15_arr))

            coverage = {
                "q15_coverage": q15_coverage,
                "q85_coverage": q85_coverage,
                "interval_width": interval_width,
            }

            logger.info(
                f"Calibration:  "
                f"q15_coverage={q15_coverage:.1%} (target ~15%)  "
                f"q85_coverage={q85_coverage:.1%} (target ~85%)  "
                f"interval_width={interval_width:.3f}"
            )

            # MLB DK scoring is zero-heavy; q15 coverage > 25% is expected
            # because many players score exactly 0 (below any positive floor
            # prediction), not a model defect.
            if q15_coverage > 0.25:
                logger.info(
                    f"  Note: q15_coverage={q15_coverage:.1%} > 25% — "
                    "expected for zero-heavy DK MLB scoring distribution."
                )

        total_elapsed = time.time() - t_total
        logger.info(f"Training complete in {total_elapsed:.1f}s")

        # -- g. Return metrics dict ------------------------------------------
        metrics: dict = {
            "q15_rmse": per_q_metrics.get("q15", {}).get("rmse", None),
            "q15_mae": per_q_metrics.get("q15", {}).get("mae", None),
            "q50_rmse": per_q_metrics.get("q50", {}).get("rmse", None),
            "q50_mae": per_q_metrics.get("q50", {}).get("mae", None),
            "q85_rmse": per_q_metrics.get("q85", {}).get("rmse", None),
            "q85_mae": per_q_metrics.get("q85", {}).get("mae", None),
            "q15_coverage": coverage.get("q15_coverage"),
            "q85_coverage": coverage.get("q85_coverage"),
            "interval_width": coverage.get("interval_width"),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "feature_count": len(present),
            "train_years": _years,
            "holdout_year": holdout_year,
            "holdout_month": holdout_month,
        }
        return metrics

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------

    def compute_shap(
        self,
        X_sample: pd.DataFrame | None = None,
        quantile_name: str = "q50",
        n_samples: int = 5000,
    ) -> None:
        """Compute and store SHAP values for ``quantile_name``.

        Uses ``shap.TreeExplainer`` (not ``shap.Explainer``) for
        compatibility with XGBoost quantile-error models.

        Args:
            X_sample: Feature DataFrame to explain.  When ``None``, the
                cached feature matrix is loaded and ``n_samples`` rows
                are sampled randomly.
            quantile_name: One of ``"q15"``, ``"q50"``, ``"q85"``.
            n_samples: Number of rows to sample when ``X_sample`` is
                ``None``.  5000 gives a representative picture without
                excessive compute.
        """
        if quantile_name not in self.models:
            raise ValueError(
                f"Model {quantile_name!r} not loaded. "
                "Call train() or load_models() first."
            )

        if X_sample is None:
            df = self._cache.load("features/batter_feature_matrix_game_level")
            if df is None or df.empty:
                raise RuntimeError(
                    "Feature matrix not in cache — run train() first."
                )
            df = df[self.feature_columns].fillna(0.0)
            if len(df) > n_samples:
                df = df.sample(n_samples, random_state=42)
            X_sample = df

        X_aligned = self._align_features(X_sample)
        model = self.models[quantile_name]

        logger.info(
            f"Computing SHAP for {quantile_name} on {len(X_aligned):,} rows…"
        )
        t0 = time.time()
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X_aligned)
        self.shap_values[quantile_name] = values

        mean_abs = np.abs(values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:10]
        top = [(self.feature_columns[i], float(mean_abs[i])) for i in top_idx]

        logger.info(f"SHAP computed in {time.time() - t0:.1f}s")
        logger.info("Top 10 features by mean |SHAP|:")
        for rank, (feat, val) in enumerate(top, 1):
            logger.info(f"  {rank:>2}. {feat:<35} {val:.4f}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_models(self, run_id: str | None = None) -> Path:
        """Save all three models and the feature column list to disk.

        Args:
            run_id: Optional identifier appended to filenames.  When
                ``None``, the current timestamp (``YYYYMMDD_HHMMSS``) is
                used.

        Returns:
            ``Path`` to ``MODEL_DIR``.
        """
        if not self.models:
            raise RuntimeError("No models to save — call train() first.")

        _run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        for name, model in self.models.items():
            path = self.model_dir / f"points_{name}_{_run_id}.joblib"
            joblib.dump(model, path)
            logger.info(f"Saved {name} model → {path}")

        feat_path = self.model_dir / f"feature_columns_{_run_id}.json"
        with feat_path.open("w") as fh:
            json.dump(self.feature_columns, fh, indent=2)
        logger.info(f"Saved feature columns → {feat_path}")

        return self.model_dir

    def load_models(self, run_id: str) -> None:
        """Load all three models and the feature column list for ``run_id``.

        Args:
            run_id: The timestamp or identifier used when ``save_models``
                was called.
        """
        for name in QUANTILE_NAMES:
            path = self.model_dir / f"points_{name}_{run_id}.joblib"
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")
            self.models[name] = joblib.load(path)
            logger.info(f"Loaded {name} model ← {path}")

        feat_path = self.model_dir / f"feature_columns_{run_id}.json"
        if not feat_path.exists():
            raise FileNotFoundError(f"Feature columns file not found: {feat_path}")
        with feat_path.open() as fh:
            self.feature_columns = json.load(fh)
        logger.info(
            f"Loaded {len(self.feature_columns)} feature columns ← {feat_path}"
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        df: pd.DataFrame,
        enforce_monotonicity: bool = True,
    ) -> pd.DataFrame:
        """Predict q15 / q50 / q85 DK points for each row in ``df``.

        Column order is aligned to ``self.feature_columns`` before
        prediction — XGBoost relies on positional column ordering and
        will silently produce wrong results if columns are shuffled.
        Any feature in ``self.feature_columns`` that is absent from
        ``df`` is filled with ``0.0`` (with a warning logged).

        Args:
            df: Feature DataFrame.  Extra columns are ignored.
            enforce_monotonicity: When ``True``, clips q15 ≤ q50 ≤ q85.

        Returns:
            DataFrame with columns ``["q15", "q50", "q85"]``, indexed
            identically to ``df``.

        Raises:
            ValueError: If no models are loaded.
        """
        if not self.models:
            raise ValueError(
                "No models loaded. Call train() or load_models() first."
            )
        if not self.feature_columns:
            raise ValueError("feature_columns is empty — cannot predict.")

        X = self._align_features(df)

        result_cols: dict[str, np.ndarray] = {}
        for name in QUANTILE_NAMES:
            if name not in self.models:
                raise ValueError(f"Model {name!r} not loaded.")
            result_cols[name] = self.models[name].predict(X)

        if enforce_monotonicity:
            result_cols["q15"] = np.minimum(result_cols["q15"], result_cols["q50"])
            result_cols["q85"] = np.maximum(result_cols["q85"], result_cols["q50"])

        return pd.DataFrame(result_cols, index=df.index)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(
        self, quantile_name: str = "q50"
    ) -> pd.DataFrame:
        """Return feature importances for ``quantile_name``, sorted descending.

        Args:
            quantile_name: One of ``"q15"``, ``"q50"``, ``"q85"``.

        Returns:
            DataFrame with columns ``["feature", "importance"]``.
        """
        if quantile_name not in self.models:
            raise ValueError(f"Model {quantile_name!r} not loaded.")

        model = self.models[quantile_name]
        importances = model.feature_importances_
        fi = pd.DataFrame(
            {"feature": self.feature_columns, "importance": importances}
        ).sort_values("importance", ascending=False).reset_index(drop=True)

        logger.info(f"Top 10 feature importances ({quantile_name}):")
        for _, row in fi.head(10).iterrows():
            logger.info(f"  {row['feature']:<35} {row['importance']:.4f}")

        return fi

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with exactly ``self.feature_columns`` in order.

        Missing features are filled with ``0.0``; extra columns are dropped.
        """
        aligned = pd.DataFrame(index=df.index)
        for col in self.feature_columns:
            if col in df.columns:
                aligned[col] = df[col].fillna(0.0)
            else:
                logger.warning(
                    f"Feature {col!r} missing from input; filling with 0.0"
                )
                aligned[col] = 0.0
        return aligned

    def _load_feature_matrix(
        self,
        train_years: list[int],
        force_rebuild: bool,
    ) -> pd.DataFrame | None:
        """Load the cached game-level feature matrix or rebuild it if needed."""
        if not force_rebuild:
            df = self._cache.load("features/batter_feature_matrix_game_level")
            if df is not None and not df.empty:
                logger.info(
                    f"Loaded game-level feature matrix from cache: {len(df):,} rows"
                )
                return df

        logger.info("Building feature matrix from Statcast Parquet files…")
        all_years = sorted(set(train_years))
        df = FeatureEngineer().build_full_batter_feature_matrix(years=all_years)
        return df if (df is not None and not df.empty) else None


class PitcherPointsModel:
    """Three XGBoost quantile regressors (q15 / q50 / q85) for pitcher DK points.

    Trained on ``PitcherFeatureEngineer`` game-level rows; target is
    ``dk_points_game``.  Models and metadata are saved with a
    ``pitcher_`` filename prefix so they never collide with hitter
    ``PointsModel`` artifacts.

    Usage::

        pm = PitcherPointsModel()
        metrics = pm.train(years=[2023, 2024, 2025])
        pm.save_models()
    """

    def __init__(self, model_dir: str | Path = "data/models") -> None:
        """Initialise the pitcher model container."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.models: dict[str, XGBRegressor] = {}
        self.feature_columns: list[str] = []
        self.shap_values: dict[str, np.ndarray] = {}
        self._cache = ParquetCache()
        logger.debug(
            f"PitcherPointsModel ready  model_dir={self.model_dir.resolve()}"
        )

    def train(
        self,
        years: list[int] | None = None,
        holdout_year: int = 2025,
        holdout_month: int = 5,
        force_rebuild_features: bool = False,
    ) -> dict:
        """Train q15 / q50 / q85 quantile regressors on pitcher game-level data.

        Loads ``features/pitcher_feature_matrix_game_level`` (or rebuilds via
        :meth:`PitcherFeatureEngineer.build_full_pitcher_feature_matrix`).
        Holdout defaults to **May 2025**, matching :class:`PointsModel`.

        Args:
            years: Seasons for the feature matrix.
            holdout_year: Holdout evaluation year.
            holdout_month: Holdout month (1–12).
            force_rebuild_features: Rebuild pitcher matrix from Statcast / logs.

        Returns:
            Metrics dict (same structure as :meth:`PointsModel.train`).
        """
        _years = years if years is not None else [2023, 2024, 2025, 2026]
        t_total = time.time()

        df = self._load_pitcher_feature_matrix(_years, force_rebuild_features)
        if df is None or df.empty:
            raise RuntimeError(
                "No pitcher feature matrix — run "
                "PitcherFeatureEngineer().build_full_pitcher_feature_matrix() "
                "first or pass force_rebuild_features=True."
            )

        if "game_date" not in df.columns:
            raise RuntimeError("Pitcher feature matrix missing 'game_date'.")

        df["game_date"] = pd.to_datetime(df["game_date"])
        test_mask = (
            (df["game_date"].dt.year == holdout_year)
            & (df["game_date"].dt.month == holdout_month)
        )
        train_df = df.loc[~test_mask].copy()
        test_df = df.loc[test_mask].copy()

        import calendar
        month_name = calendar.month_name[holdout_month]
        logger.info(
            f"[Pitcher] Train: {len(train_df):,} rows "
            f"(all except {month_name} {holdout_year})"
        )
        logger.info(
            f"[Pitcher] Test holdout: {len(test_df):,} rows "
            f"({month_name} {holdout_year})"
        )

        if train_df.empty:
            raise RuntimeError("No pitcher training rows for the given years.")
        if test_df.empty:
            logger.warning(
                f"[Pitcher] No holdout rows for {month_name} {holdout_year}."
            )

        all_features = PitcherFeatureEngineer().get_feature_columns()
        present = [c for c in all_features if c in df.columns]
        missing = [c for c in all_features if c not in df.columns]
        if missing:
            logger.warning(
                f"[Pitcher] Features missing from matrix (skipped): {missing}"
            )
        logger.info(f"[Pitcher] Training on {len(present)} features: {present}")

        self.feature_columns = present
        X_train = train_df[present].fillna(0.0)
        y_train = train_df["dk_points_game"].fillna(0.0)
        X_test = test_df[present].fillna(0.0) if not test_df.empty else pd.DataFrame()
        y_test = (
            test_df["dk_points_game"].fillna(0.0)
            if not test_df.empty
            else pd.Series(dtype=float)
        )

        raw_preds: dict[str, np.ndarray] = {}
        per_q_metrics: dict[str, dict[str, float]] = {}

        for quantile, name in zip(QUANTILES, QUANTILE_NAMES):
            logger.info(f"[Pitcher] Training {name} (alpha={quantile})…")
            t0 = time.time()
            model = XGBRegressor(**{**XGBOOST_PARAMS, "quantile_alpha": quantile})
            model.fit(X_train, y_train)
            self.models[name] = model
            logger.info(f"  [Pitcher] {name} trained in {time.time() - t0:.1f}s")

            if not X_test.empty:
                preds = model.predict(X_test)
                raw_preds[name] = preds
                rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
                mae = float(mean_absolute_error(y_test, preds))
                per_q_metrics[name] = {"rmse": rmse, "mae": mae}
                logger.info(f"  [Pitcher] {name}  RMSE={rmse:.4f}  MAE={mae:.4f}")

        if not X_test.empty and all(n in raw_preds for n in QUANTILE_NAMES):
            q15 = raw_preds["q15"]
            q50 = raw_preds["q50"]
            q85 = raw_preds["q85"]
            raw_preds["q15"] = np.minimum(q15, q50)
            raw_preds["q85"] = np.maximum(q85, q50)

        coverage: dict[str, float] = {}
        if not X_test.empty and all(n in raw_preds for n in QUANTILE_NAMES):
            y_arr = y_test.to_numpy()
            q15_arr = raw_preds["q15"]
            q85_arr = raw_preds["q85"]
            coverage = {
                "q15_coverage": float(np.mean(y_arr < q15_arr)),
                "q85_coverage": float(np.mean(y_arr < q85_arr)),
                "interval_width": float(np.mean(q85_arr - q15_arr)),
            }

        logger.info(f"[Pitcher] Training complete in {time.time() - t_total:.1f}s")

        return {
            "q15_rmse": per_q_metrics.get("q15", {}).get("rmse"),
            "q15_mae": per_q_metrics.get("q15", {}).get("mae"),
            "q50_rmse": per_q_metrics.get("q50", {}).get("rmse"),
            "q50_mae": per_q_metrics.get("q50", {}).get("mae"),
            "q85_rmse": per_q_metrics.get("q85", {}).get("rmse"),
            "q85_mae": per_q_metrics.get("q85", {}).get("mae"),
            "q15_coverage": coverage.get("q15_coverage"),
            "q85_coverage": coverage.get("q85_coverage"),
            "interval_width": coverage.get("interval_width"),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "feature_count": len(present),
            "train_years": _years,
            "holdout_year": holdout_year,
            "holdout_month": holdout_month,
        }

    def save_models(self, run_id: str | None = None) -> Path:
        """Save pitcher quantile models and feature column JSON."""
        if not self.models:
            raise RuntimeError("No pitcher models to save — call train() first.")
        _run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        for name, model in self.models.items():
            path = self.model_dir / f"pitcher_{name}_{_run_id}.joblib"
            joblib.dump(model, path)
            logger.info(f"[Pitcher] Saved {name} → {path}")
        feat_path = self.model_dir / f"pitcher_feature_columns_{_run_id}.json"
        with feat_path.open("w") as fh:
            json.dump(self.feature_columns, fh, indent=2)
        logger.info(f"[Pitcher] Saved feature columns → {feat_path}")
        return self.model_dir

    def load_models(self, run_id: str) -> None:
        """Load pitcher models and ``pitcher_feature_columns_{run_id}.json``."""
        for name in QUANTILE_NAMES:
            path = self.model_dir / f"pitcher_{name}_{run_id}.joblib"
            if not path.exists():
                raise FileNotFoundError(f"Pitcher model not found: {path}")
            self.models[name] = joblib.load(path)
            logger.info(f"[Pitcher] Loaded {name} ← {path}")
        feat_path = self.model_dir / f"pitcher_feature_columns_{run_id}.json"
        if not feat_path.exists():
            raise FileNotFoundError(f"Pitcher feature columns not found: {feat_path}")
        with feat_path.open() as fh:
            self.feature_columns = json.load(fh)
        logger.info(
            f"[Pitcher] Loaded {len(self.feature_columns)} feature columns"
        )

    def predict(
        self,
        df: pd.DataFrame,
        enforce_monotonicity: bool = True,
    ) -> pd.DataFrame:
        """Predict q15 / q50 / q85 pitcher DK points."""
        if not self.models or not self.feature_columns:
            raise ValueError("Pitcher models not loaded — train() or load_models().")
        X = self._align_features(df)
        out: dict[str, np.ndarray] = {}
        for name in QUANTILE_NAMES:
            out[name] = self.models[name].predict(X)
        if enforce_monotonicity:
            out["q15"] = np.minimum(out["q15"], out["q50"])
            out["q85"] = np.maximum(out["q85"], out["q50"])
        return pd.DataFrame(out, index=df.index)

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        aligned = pd.DataFrame(index=df.index)
        for col in self.feature_columns:
            if col in df.columns:
                aligned[col] = df[col].fillna(0.0)
            else:
                logger.warning(
                    f"[Pitcher] Feature {col!r} missing; filling with 0.0"
                )
                aligned[col] = 0.0
        return aligned

    def _load_pitcher_feature_matrix(
        self,
        train_years: list[int],
        force_rebuild: bool,
    ) -> pd.DataFrame | None:
        key = "features/pitcher_feature_matrix_game_level"
        if not force_rebuild:
            df = self._cache.load(key)
            if df is not None and not df.empty:
                logger.info(
                    f"[Pitcher] Loaded game-level matrix: {len(df):,} rows"
                )
                return df
        logger.info("[Pitcher] Building pitcher feature matrix from Parquet…")
        years = sorted(set(train_years))
        df = PitcherFeatureEngineer().build_full_pitcher_feature_matrix(
            years=years,
            force_rebuild=force_rebuild,
        )
        return df if df is not None and not df.empty else None
