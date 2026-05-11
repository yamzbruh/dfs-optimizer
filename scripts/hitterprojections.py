"""Print hitter projections from SlateInference for a DK salary CSV.

Usage:
    python scripts/hitterprojections.py
    python scripts/hitterprojections.py path/to/DKSalaries.csv

Logs from Loguru are suppressed so only the table prints (equivalent to
piping through grep -v DEBUG|INFO|WARNING).
"""

from __future__ import annotations

import argparse
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

DEFAULT_CSV = _ROOT / "data/uploads/DKSalaries_MAY8_26.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description="SlateInference hitter projection table")
    ap.add_argument(
        "csv_path",
        nargs="?",
        default=str(DEFAULT_CSV),
        type=str,
        help=f"DraftKings salary CSV (default: {DEFAULT_CSV})",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.is_file():
        print(f"File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    parser = DKCSVParser()
    parser.parse(csv_path)
    result = parser.last_result

    inference = SlateInference()
    inference.load_models()
    inference.load_feature_matrices()
    projections = inference.build_projections(result.players, use_models=True)
    proj_by_id = {p.player.dk_id: p for p in projections}

    print(
        f"{'Name':<25} {'Pos':<8} {'Team':<5} {'Salary':>8} "
        f"{'DK Avg':>7} {'Q15':>6} {'Q50':>6} {'Q85':>6} {'Src':<6}"
    )
    print("-" * 85)

    hitters = sorted(
        [p for p in result.players if not p.is_pitcher],
        key=lambda x: -proj_by_id[x.dk_id].pts_q50,
    )
    for p in hitters:
        proj = proj_by_id.get(p.dk_id)
        if proj:
            pos = p.dk_position or "?"
            src = (
                "MODEL"
                if abs(proj.pts_q50 - p.avg_points_per_game) > 0.1
                else "DK"
            )
            print(
                f"{p.name:<25} {pos:<8} {p.team:<5} "
                f"${p.salary:>6,} {p.avg_points_per_game:>7.1f} "
                f"{proj.pts_q15:>6.1f} {proj.pts_q50:>6.1f} "
                f"{proj.pts_q85:>6.1f} {src:<6}"
            )


if __name__ == "__main__":
    main()
