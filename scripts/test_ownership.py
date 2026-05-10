"""Test ownership projection on a local DK salary CSV.

Usage:
    python scripts/test_ownership.py

Set ``CSV_PATH`` below to a parsed ``DKSalaries*.csv`` on disk, or pass
``python scripts/test_ownership.py /path/to/DKSalaries.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_pipeline.ingestion.dk_csv_parser import DKCSVParser
from ml.features.ownership_projector import OwnershipProjector
from optimizer.constraints.lineup_optimizer import PlayerProjection


def main() -> None:
    csv_path = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else _ROOT / "data/uploads/DKSalaries_MAY8_26.csv"
    )
    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}")
        print("Pass a path: python scripts/test_ownership.py /path/to/DKSalaries.csv")
        sys.exit(1)

    parser = DKCSVParser()
    parser.parse(csv_path)
    result = parser.last_result
    print(f"Loaded {len(result.players)} players from {csv_path.name}")

    projections: list[PlayerProjection] = []
    for p in result.players:
        avg = float(p.avg_points_per_game)
        projections.append(
            PlayerProjection(
                player=p,
                pts_q15=max(0.0, avg * 0.5),
                pts_q50=avg,
                pts_q85=avg * 1.8 if not p.is_pitcher else avg * 2.0,
                ownership_proj=10.0 if not p.is_pitcher else 15.0,
                leverage_score=avg / 10.0,
            )
        )

    projector = OwnershipProjector()
    print("Running ownership projection (10k sims)...")
    ownership = projector.project(
        players=result.players,
        base_projections=projections,
        n_sims=10000,
        n_jobs=4,
    )

    updated = projector.update_projections_with_ownership(projections, ownership)

    print("\n=== OWNERSHIP PROJECTIONS ===")
    print(
        f"{'Name':<25} {'Pos':<8} {'Salary':>8} "
        f"{'PtsQ50':>8} {'Own%':>7} {'Lev':>7}"
    )
    print("-" * 68)

    sorted_projs = sorted(updated, key=lambda x: -x.ownership_proj)
    for proj in sorted_projs[:20]:
        pl = proj.player
        print(
            f"{pl.name:<25} {pl.dk_position:<8} "
            f"${pl.salary:>7,} {proj.pts_q50:>8.1f} "
            f"{proj.ownership_proj:>6.1f}% "
            f"{proj.leverage_score:>7.3f}"
        )

    print()
    print("=== TOP LEVERAGE PLAYS (GPP EDGE) ===")
    print(f"{'Name':<25} {'Pos':<8} {'Own%':>7} {'Q85':>7} {'Lev':>7}")
    print("-" * 58)

    leverage_sorted = sorted(
        [p for p in updated if not p.player.is_pitcher],
        key=lambda x: -x.leverage_score,
    )[:15]

    for proj in leverage_sorted:
        pl = proj.player
        print(
            f"{pl.name:<25} {pl.dk_position:<8} "
            f"{proj.ownership_proj:>6.1f}% "
            f"{proj.pts_q85:>7.1f} "
            f"{proj.leverage_score:>7.3f}"
        )


if __name__ == "__main__":
    main()
