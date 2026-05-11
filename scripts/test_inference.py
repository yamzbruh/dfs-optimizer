"""Test real model inference on the May 8 slate.

Usage:
    python scripts/test_inference.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_pipeline.ingestion.dk_csv_parser import DKCSVParser
from ml.inference.slate_inference import SlateInference


def main() -> None:
    parser = DKCSVParser()
    parser.parse("data/uploads/DKSalaries_MAY8_26.csv")
    result = parser.last_result
    print(f"Loaded {len(result.players)} players")

    inference = SlateInference()
    inference.load_models()
    inference.load_feature_matrices()

    print("Building projections...")
    projections = inference.build_projections(result.players, use_models=True)

    print()
    print("=== MODEL PROJECTIONS ===")
    print(
        f"{'Name':<25} {'Pos':<5} {'Salary':>8} "
        f"{'Q15':>6} {'Q50':>6} {'Q85':>6} {'Source'}"
    )
    print("-" * 70)

    # Sort by q50 descending
    sorted_projs = sorted(projections, key=lambda x: -x.pts_q50)

    for proj in sorted_projs[:30]:
        p = proj.player
        # Detect if model was used (q50 != DK avg)
        is_model = abs(proj.pts_q50 - p.avg_points_per_game) > 0.1
        source = "MODEL" if is_model else "DK AVG"

        print(
            f"{p.name:<25} {p.dk_position:<5} "
            f"${p.salary:>7,} "
            f"{proj.pts_q15:>6.1f} "
            f"{proj.pts_q50:>6.1f} "
            f"{proj.pts_q85:>6.1f} "
            f"{source}"
        )

    print()
    total = len(projections)
    model_total = sum(
        1
        for p in projections
        if abs(p.pts_q50 - p.player.avg_points_per_game) > 0.1
    )
    print(f"Model used: {model_total}/{total} players")
    print(f"Fallback:   {total - model_total}/{total} players")


if __name__ == "__main__":
    main()
