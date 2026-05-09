"""Pull per-game batting and pitching stats from MLB Stats API.

Run after pull_baseline_data.py. Pulls game logs for every
unique player in the Statcast cache.

Usage::

    # Pull specific season:
    python scripts/pull_game_logs.py --season 2024

    # Pull all seasons:
    python scripts/pull_game_logs.py --all

    # Force re-pull even if cached:
    python scripts/pull_game_logs.py --all --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_pipeline.ingestion.statcast_loader import StatcastLoader  # noqa: E402
from data_pipeline.loaders.parquet_cache import ParquetCache  # noqa: E402

SEASONS = [2023, 2024, 2025, 2026]


def get_unique_batter_ids(cache: ParquetCache, season: int) -> list[int]:
    """Get MLBAM batter IDs for hitting game logs: 50+ PAs, not in pitcher pool.

    Cross-references batter Statcast with pitcher Statcast and requires at
    least 50 plate appearances in the season cache (filters call-ups and
    fringe players).
    """
    batter_df = cache.load(f"statcast/batters_{season}")
    if batter_df is None or batter_df.empty:
        return []

    pitcher_df = cache.load(f"statcast/pitchers_{season}")
    pitcher_ids: set[int] = set()
    if pitcher_df is not None and not pitcher_df.empty:
        if "pitcher" in pitcher_df.columns:
            p = pd.to_numeric(pitcher_df["pitcher"], errors="coerce").dropna()
            pitcher_ids = set(p.astype(int).unique().tolist())

    if "batter" not in batter_df.columns:
        return []

    b = pd.to_numeric(batter_df["batter"], errors="coerce").dropna()
    batter_ids = set(b.astype(int).unique().tolist())

    pa_counts = (
        batter_df["batter"]
        .dropna()
        .apply(lambda x: pd.to_numeric(x, errors="coerce"))
        .dropna()
        .astype(int)
        .value_counts()
    )
    qualified_ids = set(
        pa_counts[pa_counts >= 50].index.astype(int)
    )

    hitter_ids = qualified_ids - pitcher_ids
    logger.info(
        f"Season {season}: {len(batter_ids)} total batters, "
        f"{len(pitcher_ids)} pitchers excluded, "
        f"{len(qualified_ids)} with 50+ PAs, "
        f"{len(hitter_ids)} qualifying hitters"
    )
    return sorted(hitter_ids)


def get_unique_pitcher_ids(cache: ParquetCache, season: int) -> list[int]:
    """Get unique MLBAM pitcher IDs from cached Statcast data."""
    df = cache.load(f"statcast/pitchers_{season}")
    if df is None or df.empty:
        return []
    if "pitcher" not in df.columns:
        return []
    s = pd.to_numeric(df["pitcher"], errors="coerce").dropna().astype(int)
    ids = sorted(s.unique().tolist())
    logger.info(f"Found {len(ids)} unique pitchers for {season}")
    return ids


def pull_season(
    loader: StatcastLoader,
    cache: ParquetCache,
    season: int,
    force: bool = False,
) -> dict:
    """Pull hitting and pitching game logs for one season."""
    results: dict = {}

    hit_key = f"gamelogs/hitting_{season}"
    if not force and cache.exists(hit_key):
        df = cache.load(hit_key)
        rows = len(df) if df is not None else 0
        logger.info(f"SKIP: {hit_key} already cached ({rows:,} rows)")
        results["hitting"] = {"status": "cached", "rows": rows}
    else:
        batter_ids = get_unique_batter_ids(cache, season)
        if batter_ids:
            t0 = time.time()
            df = loader.get_season_game_logs_hitting(
                season, batter_ids, force_refresh=force
            )
            elapsed = time.time() - t0
            rows = len(df) if df is not None else 0
            results["hitting"] = {
                "status": "pulled",
                "rows": rows,
                "elapsed": elapsed,
            }
        else:
            results["hitting"] = {"status": "no_ids", "rows": 0}

    pitch_key = f"gamelogs/pitching_{season}"
    if not force and cache.exists(pitch_key):
        df = cache.load(pitch_key)
        rows = len(df) if df is not None else 0
        logger.info(f"SKIP: {pitch_key} already cached ({rows:,} rows)")
        results["pitching"] = {"status": "cached", "rows": rows}
    else:
        pitcher_ids = get_unique_pitcher_ids(cache, season)
        if pitcher_ids:
            t0 = time.time()
            df = loader.get_season_game_logs_pitching(
                season, pitcher_ids, force_refresh=force
            )
            elapsed = time.time() - t0
            rows = len(df) if df is not None else 0
            results["pitching"] = {
                "status": "pulled",
                "rows": rows,
                "elapsed": elapsed,
            }
        else:
            results["pitching"] = {"status": "no_ids", "rows": 0}

    return results


def print_summary(all_results: dict) -> None:
    """Print formatted summary table."""
    print("\n" + "=" * 65)
    print(f"{'Dataset':<35} {'Status':<10} {'Rows':>10} {'Time':>8}")
    print("=" * 65)
    for season, results in all_results.items():
        for stat_type, r in results.items():
            label = f"gamelogs/{stat_type}_{season}"
            status = r.get("status", "unknown")
            rows = r.get("rows", 0)
            elapsed = r.get("elapsed", 0)
            time_str = f"{elapsed:.1f}s" if elapsed else "-"
            print(
                f"{label:<35} {status:<10} {rows:>10,} {time_str:>8}"
            )
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull MLB Stats API game logs"
    )
    parser.add_argument("--season", type=int, help="Pull one season")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Pull all seasons",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-pull even if cached",
    )
    args = parser.parse_args()

    if not args.season and not args.all:
        parser.print_help()
        sys.exit(1)

    seasons = SEASONS if args.all else [args.season]

    cache = ParquetCache()
    loader = StatcastLoader(cache=cache)
    all_results: dict = {}

    for season in seasons:
        logger.info(f"Processing season {season}...")
        all_results[season] = pull_season(
            loader, cache, season, force=args.force
        )

    print_summary(all_results)


if __name__ == "__main__":
    main()
