"""Pull historical batting order data from MLB Stats API.

Usage:
    python scripts/pull_lineups.py --season 2024
    python scripts/pull_lineups.py --all
    python scripts/pull_lineups.py --today
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loguru import logger

from data_pipeline.ingestion.statcast_loader import StatcastLoader
from data_pipeline.loaders.parquet_cache import ParquetCache

logger.remove()
logger.add(sys.stderr, level="INFO")

SEASONS = [2023, 2024, 2025, 2026]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull MLB batting order data"
    )
    parser.add_argument("--season", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--today", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not any([args.season, args.all, args.today]):
        parser.print_help()
        sys.exit(1)

    cache = ParquetCache()
    loader = StatcastLoader(cache=cache)

    if args.today:
        df = loader.get_todays_lineups()
        if not df.empty:
            print(f"Today's lineups: {len(df)} players")
            print(
                df[["team", "player_name", "batting_order"]]
                .sort_values(["team", "batting_order"])
                .to_string(index=False)
            )
        return

    seasons = SEASONS if args.all else [args.season]

    for season in seasons:
        print(f"Pulling batting orders for {season}...")
        t0 = time.time()
        df = loader.get_season_lineups(season, force_refresh=args.force)
        elapsed = time.time() - t0
        print(f"  {season}: {len(df):,} rows in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
