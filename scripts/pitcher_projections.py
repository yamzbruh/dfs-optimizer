"""Show model vs DK projections for all pitchers on a slate.

Usage:
    python scripts/pitcher_projections.py
    python scripts/pitcher_projections.py data/uploads/MySlate.csv
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="ERROR")

from data_pipeline.ingestion.dk_csv_parser import DKCSVParser
from ml.inference.slate_inference import SlateInference


def main() -> None:
    arg = (
        sys.argv[1]
        if len(sys.argv) > 1
        else str(_ROOT / "data/uploads/DKSalaries_MAY8_26.csv")
    )
    csv_path = Path(arg)
    if not csv_path.is_absolute():
        csv_path = _ROOT / csv_path
    if not csv_path.is_file():
        print(f"File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    parser = DKCSVParser()
    parser.parse(csv_path)
    result = parser.last_result

    inference = SlateInference()
    inference.load_models()
    inference.load_feature_matrices()
    projections = inference.build_projections(
        result.players, use_models=True
    )
    proj_by_id = {p.player.dk_id: p for p in projections}

    pitchers = sorted(
        [p for p in result.players if p.is_pitcher],
        key=lambda x: -proj_by_id[x.dk_id].pts_q50,
    )

    print(
        f"{'Name':<25} {'Pos':<6} {'Team':<5} {'Salary':>8} "
        f"{'DK Avg':>7} {'Q15':>6} {'Q50':>6} {'Q85':>6} "
        f"{'Diff':>6} {'Src':<6}"
    )
    print("-" * 88)

    for p in pitchers:
        proj = proj_by_id.get(p.dk_id)
        if not proj:
            continue
        pos = p.dk_position or "?"
        diff = proj.pts_q50 - p.avg_points_per_game
        src = (
            "MODEL"
            if abs(proj.pts_q50 - p.avg_points_per_game) > 0.1
            else "DK"
        )
        print(
            f"{p.name:<25} {pos:<6} {p.team:<5} "
            f"${p.salary:>6,} {p.avg_points_per_game:>7.1f} "
            f"{proj.pts_q15:>6.1f} {proj.pts_q50:>6.1f} "
            f"{proj.pts_q85:>6.1f} {diff:>+6.1f} {src:<6}"
        )


if __name__ == "__main__":
    main()
