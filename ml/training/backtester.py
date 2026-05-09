"""Backtester for the XGBoost quantile points projection model.

Evaluates model predictions against actual DK scoring results.  The
MLB DK scoring distribution is heavily zero-inflated — many players
appear in a lineup slot, score 0 points, and dominate the lower tail.
All metrics and calibration reports therefore report breakdowns for
the full distribution *and* for the non-zero subset separately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class Backtester:
    """Evaluates quantile model predictions against hold-out actuals.

    Usage::

        bt = Backtester()
        metrics = bt.backtest(model, test_df)
        report = bt.coverage_report(predictions_df, actuals_series)
    """

    def __init__(self) -> None:
        """Initialise the backtester."""
        logger.debug("Backtester ready")

    def backtest(
        self,
        model: "PointsModel",  # noqa: F821 — forward ref to avoid circular import
        test_df: pd.DataFrame,
        actual_col: str = "dk_points",
    ) -> dict:
        """Run the full model on ``test_df`` and compare to actuals.

        Handles zero-scoring skew:

        * Uses ``median_absolute_error`` as a supplemental metric since
          RMSE is sensitive to the zero-heavy tail.
        * Reports ``pct_zero_actual`` so callers know the zero fraction.
        * Never divides by zero or assumes a normal distribution.

        Args:
            model: A trained or loaded ``PointsModel`` instance.
            test_df: Feature DataFrame that must contain ``actual_col``.
            actual_col: Name of the ground-truth DK points column.

        Returns:
            Metrics dict including RMSE, MAE, median AE, coverage, and
            distribution diagnostics.
        """
        if test_df is None or test_df.empty:
            logger.warning("backtest: empty test_df; returning empty metrics")
            return {}

        if actual_col not in test_df.columns:
            raise ValueError(
                f"actual_col={actual_col!r} not found in test_df. "
                f"Available columns: {list(test_df.columns)[:20]}"
            )

        actuals = test_df[actual_col].fillna(0.0)
        predictions = model.predict(test_df)

        q50 = predictions["q50"].to_numpy()
        q15 = predictions["q15"].to_numpy()
        q85 = predictions["q85"].to_numpy()
        y = actuals.to_numpy()

        rmse = float(np.sqrt(mean_squared_error(y, q50)))
        mae = float(mean_absolute_error(y, q50))
        median_ae = float(np.median(np.abs(y - q50)))

        q15_coverage = float(np.mean(y < q15))
        q85_coverage = float(np.mean(y < q85))
        interval_width = float(np.mean(q85 - q15))

        mean_actual = float(np.mean(y))
        mean_pred_q50 = float(np.mean(q50))
        pct_zero_actual = float(np.mean(y == 0.0))
        n_samples = len(y)

        logger.info(
            f"Backtest results  n={n_samples:,}  "
            f"RMSE={rmse:.4f}  MAE={mae:.4f}  MedianAE={median_ae:.4f}"
        )
        logger.info(
            f"  mean_actual={mean_actual:.3f}  "
            f"mean_pred_q50={mean_pred_q50:.3f}  "
            f"pct_zero={pct_zero_actual:.1%}"
        )
        logger.info(
            f"  q15_coverage={q15_coverage:.1%}  "
            f"q85_coverage={q85_coverage:.1%}  "
            f"interval_width={interval_width:.3f}"
        )

        return {
            "rmse": rmse,
            "mae": mae,
            "median_ae": median_ae,
            "q15_coverage": q15_coverage,
            "q85_coverage": q85_coverage,
            "interval_width": interval_width,
            "mean_actual": mean_actual,
            "mean_predicted_q50": mean_pred_q50,
            "pct_zero_actual": pct_zero_actual,
            "n_samples": n_samples,
        }

    def coverage_report(
        self,
        predictions: pd.DataFrame,
        actuals: pd.Series,
    ) -> dict:
        """Calibration report split by zero-scoring and non-zero rows.

        Quantile calibration is typically misread for MLB DK scoring
        because the zero-scoring mass is large enough to drag q15
        coverage well above 15%.  Reporting the non-zero subset
        separately gives a cleaner read on whether the model is
        properly calibrated for players who *do* score.

        Args:
            predictions: DataFrame with columns ``q15``, ``q50``,
                ``q85`` (as returned by ``model.predict()``).
            actuals: Ground-truth DK points Series, aligned to
                ``predictions`` by index.

        Returns:
            Dict with overall and non-zero calibration breakdowns.
        """
        required_cols = {"q15", "q50", "q85"}
        missing = required_cols - set(predictions.columns)
        if missing:
            raise ValueError(
                f"predictions DataFrame missing columns: {missing}"
            )

        y = actuals.reindex(predictions.index).fillna(0.0).to_numpy()
        q15 = predictions["q15"].to_numpy()
        q85 = predictions["q85"].to_numpy()

        n_total = len(y)
        nonzero_mask = y != 0.0
        n_nonzero = int(nonzero_mask.sum())
        pct_zero = float(1.0 - n_nonzero / n_total) if n_total > 0 else 0.0

        q15_overall = float(np.mean(y < q15))
        q85_overall = float(np.mean(y < q85))

        if n_nonzero > 0:
            q15_nonzero = float(np.mean(y[nonzero_mask] < q15[nonzero_mask]))
            q85_nonzero = float(np.mean(y[nonzero_mask] < q85[nonzero_mask]))
        else:
            q15_nonzero = float("nan")
            q85_nonzero = float("nan")

        logger.info(
            f"Coverage report  n_total={n_total:,}  n_nonzero={n_nonzero:,}  "
            f"pct_zero={pct_zero:.1%}"
        )
        logger.info(
            f"  Overall:   q15={q15_overall:.1%}  q85={q85_overall:.1%}"
        )
        logger.info(
            f"  Non-zero:  q15={q15_nonzero:.1%}  q85={q85_nonzero:.1%}"
        )

        if q85_overall < 0.80:
            logger.warning(
                f"q85_coverage_overall={q85_overall:.1%} is below 80% — "
                "model may be under-confident on the upside. "
                "Check for data leakage or feature drift."
            )

        return {
            "q15_coverage_overall": q15_overall,
            "q85_coverage_overall": q85_overall,
            "q15_coverage_nonzero": q15_nonzero,
            "q85_coverage_nonzero": q85_nonzero,
            "pct_zero": pct_zero,
            "n_total": n_total,
            "n_nonzero": n_nonzero,
        }
