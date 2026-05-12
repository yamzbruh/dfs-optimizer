"""Show feature importance for hitter and pitcher models.

Usage:
    python scripts/feature_importance.py
"""

import sys
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="ERROR")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.training.points_model import PitcherPointsModel, PointsModel


def main() -> None:
    models_dir = _ROOT / "data/models"

    # Hitter model — all three quantiles
    hitter_files = sorted(models_dir.glob("points_q50_*.joblib"))
    if hitter_files:
        run_id = hitter_files[-1].stem.replace("points_q50_", "")
        model = PointsModel()
        model.load_models(run_id)

        for quantile in ["q15", "q50", "q85"]:
            fi = model.get_feature_importance(quantile)
            print(f"\n=== HITTER {quantile.upper()} IMPORTANCE ===")
            print(fi.to_string(index=False))
    else:
        print("No hitter models found")

    # Pitcher model
    pitcher_files = sorted(models_dir.glob("pitcher_q50_*.joblib"))
    if pitcher_files:
        run_id = pitcher_files[-1].stem.replace("pitcher_q50_", "")
        pmodel = PitcherPointsModel()
        pmodel.load_models(run_id)
        fi = pmodel.get_feature_importance("q50")
        print("\n=== PITCHER Q50 IMPORTANCE ===")
        print(fi.to_string(index=False))
    else:
        print("No pitcher models found")


if __name__ == "__main__":
    main()
