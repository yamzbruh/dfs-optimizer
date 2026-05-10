"""Quick test script for Odds API ingestion.

Usage:
    python scripts/test_odds_api.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_pipeline.ingestion.odds_ingestion import OddsIngestion


def main() -> None:
    odds = OddsIngestion()

    print("Fetching MLB implied totals...")
    df = odds.get_mlb_implied_totals(force_refresh=True)

    if df.empty:
        print("No odds data returned")
        return

    print(f"\nReturned {len(df)} team rows")
    print(f"Columns: {df.columns.tolist()}")
    print()
    print("Team implied totals:")
    print("-" * 40)

    totals = odds.get_team_implied_totals()
    for team, total in sorted(totals.items(), key=lambda x: -x[1]):
        bar = "█" * max(0, int(total))
        print(f"  {team:<6} {total:.1f}  {bar}")

    print()
    print("Full data:")
    cols = [
        c
        for c in (
            "team",
            "implied_total",
            "opposing_implied",
            "game_total",
            "is_home",
        )
        if c in df.columns
    ]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
