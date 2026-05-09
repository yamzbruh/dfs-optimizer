"""Baseline data pull script — run once to seed the Parquet cache.

Pulls FanGraphs and Statcast data for 2023 and 2024, caching each
result to ``data/parquet/``.  FanGraphs is pulled first because it is
significantly faster and confirms the pipeline works before committing
to the hour-plus Statcast pulls.

Usage::

    # Pull only what's not already cached:
    python scripts/pull_baseline_data.py

    # Force re-pull everything (ignores cache):
    python scripts/pull_baseline_data.py --force

Run from the project root with the venv active:
    source venv/bin/activate
    python scripts/pull_baseline_data.py
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

# Ensure project root is on the path when the script is invoked
# directly (python scripts/pull_baseline_data.py).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_pipeline.ingestion.statcast_loader import StatcastLoader  # noqa: E402
from data_pipeline.loaders.parquet_cache import ParquetCache  # noqa: E402


# ---------------------------------------------------------------------------
# Pull job definitions
# ---------------------------------------------------------------------------


@dataclass
class PullJob:
    """Metadata for one data-pull task."""

    label: str
    key: str       # cache key, for existence check
    method: str    # StatcastLoader method name
    kwargs: dict = field(default_factory=dict)


JOBS: list[PullJob] = [
    # FanGraphs first — they are fast and confirm the pipeline is healthy.
    PullJob(
        label="FanGraphs batting 2023",
        key="fangraphs/batting_2023",
        method="get_fangraphs_batting",
        kwargs={"year": 2023},
    ),
    PullJob(
        label="FanGraphs batting 2024",
        key="fangraphs/batting_2024",
        method="get_fangraphs_batting",
        kwargs={"year": 2024},
    ),
    PullJob(
        label="FanGraphs pitching 2023",
        key="fangraphs/pitching_2023",
        method="get_fangraphs_pitching",
        kwargs={"year": 2023},
    ),
    PullJob(
        label="FanGraphs pitching 2024",
        key="fangraphs/pitching_2024",
        method="get_fangraphs_pitching",
        kwargs={"year": 2024},
    ),
    # Statcast pulls are large — expect 5-30 min each depending on internet.
    PullJob(
        label="Statcast batters 2023",
        key="statcast/batters_2023",
        method="get_statcast_batters",
        kwargs={"year": 2023},
    ),
    PullJob(
        label="Statcast batters 2024",
        key="statcast/batters_2024",
        method="get_statcast_batters",
        kwargs={"year": 2024},
    ),
    PullJob(
        label="Statcast pitchers 2023",
        key="statcast/pitchers_2023",
        method="get_statcast_pitchers",
        kwargs={"year": 2023},
    ),
    PullJob(
        label="Statcast pitchers 2024",
        key="statcast/pitchers_2024",
        method="get_statcast_pitchers",
        kwargs={"year": 2024},
    ),
    PullJob(
        label="FanGraphs batting 2025",
        key="fangraphs/batting_2025",
        method="get_fangraphs_batting",
        kwargs={"year": 2025},
    ),
    PullJob(
        label="FanGraphs batting 2026",
        key="fangraphs/batting_2026",
        method="get_fangraphs_batting",
        kwargs={"year": 2026},
    ),
    PullJob(
        label="FanGraphs pitching 2025",
        key="fangraphs/pitching_2025",
        method="get_fangraphs_pitching",
        kwargs={"year": 2025},
    ),
    PullJob(
        label="FanGraphs pitching 2026",
        key="fangraphs/pitching_2026",
        method="get_fangraphs_pitching",
        kwargs={"year": 2026},
    ),
    PullJob(
        label="Statcast batters 2025",
        key="statcast/batters_2025",
        method="get_statcast_batters",
        kwargs={"year": 2025},
    ),
    PullJob(
        label="Statcast batters 2026",
        key="statcast/batters_2026",
        method="get_statcast_batters",
        kwargs={"year": 2026},
    ),
    PullJob(
        label="Statcast pitchers 2025",
        key="statcast/pitchers_2025",
        method="get_statcast_pitchers",
        kwargs={"year": 2025},
    ),
    PullJob(
        label="Statcast pitchers 2026",
        key="statcast/pitchers_2026",
        method="get_statcast_pitchers",
        kwargs={"year": 2026},
    ),
]


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------


@dataclass
class JobResult:
    label: str
    key: str
    status: str    # "cached" | "pulled" | "error"
    rows: int
    elapsed_s: float
    error_msg: str = ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull and cache baseline MLB training data."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-pull all datasets even if already cached.",
    )
    return parser.parse_args()


def _print_summary(results: list[JobResult]) -> None:
    """Print a formatted summary table to stdout."""
    col_label = max(len(r.label) for r in results) + 2
    header = (
        f"{'Dataset':<{col_label}} | {'Status':<8} | {'Rows':>8} | {'Time':>7}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results:
        status_display = r.status
        rows_display = f"{r.rows:,}" if r.rows >= 0 else "—"
        time_display = f"{r.elapsed_s:.1f}s"
        err_suffix = f"  ← {r.error_msg}" if r.error_msg else ""
        print(
            f"{r.label:<{col_label}} | {status_display:<8} | "
            f"{rows_display:>8} | {time_display:>7}{err_suffix}"
        )
    print(sep)
    pulled = sum(1 for r in results if r.status == "pulled")
    cached = sum(1 for r in results if r.status == "cached")
    errors = sum(1 for r in results if r.status == "error")
    print(
        f"Summary: {pulled} pulled, {cached} skipped (cached), "
        f"{errors} error(s)\n"
    )


def main() -> None:
    """Entry point — run all pull jobs, print summary table."""
    args = _parse_args()

    cache = ParquetCache()
    loader = StatcastLoader(cache=cache)
    results: list[JobResult] = []

    for job in JOBS:
        # Check cache before pulling (unless --force).
        if not args.force and cache.exists(job.key):
            meta = cache.get_metadata(job.key)
            rows = (meta or {}).get("rows", -1)
            logger.info(f"SKIP: {job.key!r} already cached ({rows:,} rows)")
            results.append(
                JobResult(
                    label=job.label,
                    key=job.key,
                    status="cached",
                    rows=rows,
                    elapsed_s=0.0,
                )
            )
            continue

        logger.info(f"PULLING: {job.key!r} …")
        t0 = time.time()
        try:
            method_fn = getattr(loader, job.method)
            kwargs = {**job.kwargs}
            if args.force:
                kwargs["force_refresh"] = True
            df = method_fn(**kwargs)
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - t0
            logger.error(f"ERROR pulling {job.key!r}: {exc}")
            results.append(
                JobResult(
                    label=job.label,
                    key=job.key,
                    status="error",
                    rows=0,
                    elapsed_s=elapsed,
                    error_msg=str(exc)[:80],
                )
            )
            continue

        elapsed = time.time() - t0
        rows = 0 if df is None or df.empty else len(df)
        status = "pulled" if rows > 0 else "error"
        error_msg = "returned empty DataFrame" if rows == 0 else ""

        logger.info(
            f"{'OK' if rows > 0 else 'EMPTY'}: {job.key!r} "
            f"— {rows:,} rows in {elapsed:.1f}s"
        )
        results.append(
            JobResult(
                label=job.label,
                key=job.key,
                status=status,
                rows=rows,
                elapsed_s=elapsed,
                error_msg=error_msg,
            )
        )

    _print_summary(results)


if __name__ == "__main__":
    main()
