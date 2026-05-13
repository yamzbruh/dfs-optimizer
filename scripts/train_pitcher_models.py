"""Train StarterPointsModel and RelieverPointsModel.

Usage:
    python scripts/train_pitcher_models.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loguru import logger

from ml.training.points_model import RelieverPointsModel, StarterPointsModel


def main() -> None:
    years = [2023, 2024, 2025, 2026]
    holdout_year, holdout_month = 2025, 5

    starter = StarterPointsModel()
    starter_metrics = starter.train(
        years=years,
        holdout_year=holdout_year,
        holdout_month=holdout_month,
    )
    print("[Starter] metrics:", json.dumps(starter_metrics, indent=2, default=str))
    starter_path = starter.save_models()
    print(f"[Starter] saved under {starter_path}")

    reliever = RelieverPointsModel()
    reliever_metrics = reliever.train(
        years=years,
        holdout_year=holdout_year,
        holdout_month=holdout_month,
    )
    print("[Reliever] metrics:", json.dumps(reliever_metrics, indent=2, default=str))
    reliever_path = reliever.save_models()
    print(f"[Reliever] saved under {reliever_path}")

    logger.info("train_pitcher_models: complete")


if __name__ == "__main__":
    main()
