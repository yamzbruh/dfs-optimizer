"""Pre-slate intelligence report — sanity-check data before lineup generation.

Usage:
    python3 scripts/check_slate.py
    python3 scripts/check_slate.py --csv data/uploads/DKSalaries_2026-05-18_dg147547.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="ERROR")

import pandas as pd

from automation.scheduler import (  # noqa: E402
    _check_sp_confirmation,
    get_mlb_probable_pitchers,
)
from data_pipeline.ingestion.dk_csv_parser import DKCSVParser, DKPlayer
from data_pipeline.ingestion.lineup_status import (
    LineupStatusChecker,
    team_ids_from_dk_players,
)
from data_pipeline.ingestion.odds_ingestion import OddsIngestion
from data_pipeline.loaders.parquet_cache import ParquetCache
from ml.features.ownership_projector import OwnershipProjector
from ml.inference.slate_inference import SlateInference
from optimizer.constraints.lineup_optimizer import PlayerProjection

_UPLOADS = _ROOT / "data" / "uploads"
_HOT_IMPLIED = 5.0
_COLD_IMPLIED = 3.5
_INFLATION_Q50 = 25.0
_OWNERSHIP_SIMS = 1000


def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _subsection(title: str) -> None:
    print()
    print(f"--- {title} ---")


def _find_latest_salary_csv() -> Path | None:
    candidates = sorted(
        _UPLOADS.glob("DKSalaries*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _resolve_csv(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = _ROOT / path
        if not path.is_file():
            print(f"CSV not found: {path}", file=sys.stderr)
            sys.exit(1)
        return path
    latest = _find_latest_salary_csv()
    if latest is None:
        print(f"No DKSalaries*.csv in {_UPLOADS}", file=sys.stderr)
        sys.exit(1)
    return latest


def _format_ts(dt: datetime | None) -> str:
    if dt is None:
        return "not available"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().strftime("%Y-%m-%d %H:%M %Z")


def _statcast_freshness() -> tuple[str, str, str, str]:
    """Return (batters_cache, pitchers_cache, batters_max_game, pitchers_max_game)."""
    cache = ParquetCache(_ROOT / "data" / "parquet")
    year = date.today().year
    key_bat = f"statcast/batters_{year}"
    key_pit = f"statcast/pitchers_{year}"

    bat_mtime = _format_ts(cache.get_last_updated(key_bat))
    pit_mtime = _format_ts(cache.get_last_updated(key_pit))

    def _max_game_date(key: str) -> str:
        df = cache.load(key)
        if df is None or df.empty or "game_date" not in df.columns:
            return "no data"
        mx = pd.to_datetime(df["game_date"], errors="coerce").max()
        if pd.isna(mx):
            return "no game_date"
        return pd.Timestamp(mx).strftime("%Y-%m-%d")

    return bat_mtime, pit_mtime, _max_game_date(key_bat), _max_game_date(key_pit)


def _vegas_fetch_time(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "no odds loaded"
    if "fetched_at" in df.columns:
        raw = df["fetched_at"].dropna()
        if not raw.empty:
            return str(raw.iloc[0])
    return "cached session (no fetched_at column)"


def _chadwick_last_updated() -> str:
    candidates: list[Path] = []
    for root in (Path.home() / "pybaseball", Path.home() / ".pybaseball"):
        if not root.is_dir():
            continue
        candidates.extend(p for p in root.rglob("*chadwick*") if p.is_file())

    crosswalk = _ROOT / "data" / "parquet" / "crosswalk" / "player_ids.parquet"
    if crosswalk.is_file():
        candidates.append(crosswalk)

    if not candidates:
        return "unknown — run chadwick_register(save=True) or morning setup"

    best = max(
        datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc) for p in candidates
    )
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return f"{_format_ts(best)}  ({newest.name})"


def _report_vegas() -> pd.DataFrame:
    _section("1. VEGAS IMPLIED TOTALS")
    odds = OddsIngestion()
    df = odds.get_mlb_implied_totals()
    if df is None or df.empty:
        print("No Vegas implied totals (missing API key or fetch failed).")
        return df

    work = df.copy()
    work["implied_total"] = pd.to_numeric(
        work.get("implied_total", 0), errors="coerce"
    ).fillna(0.0)
    by_team = (
        work.sort_values("implied_total", ascending=False)
        .drop_duplicates(subset=["team"], keep="first")
    )

    print(
        f"{'Team':<6} {'Implied':>8}  {'Flag':<12}  "
        f"{'Opp':<6} {'Book':<12}"
    )
    print("-" * 52)
    for _, row in by_team.iterrows():
        team = str(row.get("team", "")).strip()
        implied = float(row["implied_total"])
        flag = ""
        if implied > _HOT_IMPLIED:
            flag = "HOT (>5.0)"
        elif implied < _COLD_IMPLIED:
            flag = "COLD (<3.5)"
        opp = str(row.get("opposing_implied", "") or "")
        book = str(row.get("bookmaker", "") or "")
        print(f"{team:<6} {implied:>8.2f}  {flag:<12}  {opp:>6}  {book:<12}")

    hot = by_team[by_team["implied_total"] > _HOT_IMPLIED]["team"].tolist()
    cold = by_team[by_team["implied_total"] < _COLD_IMPLIED]["team"].tolist()
    if hot:
        print(f"\nHot offenses: {', '.join(str(t) for t in hot)}")
    if cold:
        print(f"Cold offenses: {', '.join(str(t) for t in cold)}")

    return df


def _report_auto_bans(
    players: list[DKPlayer], inference: SlateInference
) -> LineupStatusChecker:
    _section("2. AUTO-BAN LIST")
    checker = LineupStatusChecker()
    tids = team_ids_from_dk_players(players)
    checker.load_statuses(tids)
    match = inference.match_dk_player_to_mlbam

    banned_ids = checker.get_unavailable_dk_ids(players, match)
    rows = [
        r
        for r in checker.get_status_report(players, match)
        if r.get("status") == "unavailable"
    ]
    rows.sort(key=lambda r: (r.get("team", ""), r.get("name", "")))

    print(f"Teams checked: {len(tids)}  |  Auto-banned: {len(banned_ids)}")
    if not rows:
        print("No auto-banned players (IL/OUT/SUSP/RM).")
        return checker

    print(f"{'Name':<26} {'Team':<5} {'Reason'}")
    print("-" * 72)
    for row in rows:
        print(
            f"{row['name']:<26} {row['team']:<5} "
            f"{row.get('reason', 'Unavailable')}"
        )
    return checker


def _report_sp_confirmation(players: list[DKPlayer]) -> None:
    _section("3. SP CONFIRMATION STATUS (MLB Stats API)")
    probable_map = get_mlb_probable_pitchers()
    ok_lines, bad_lines = _check_sp_confirmation(players, probable_map)

    dk_sps = [
        p
        for p in players
        if p.is_pitcher and (p.dk_position or "").upper() == "SP"
    ]
    print(f"MLB probable pitchers listed: {len(probable_map)}")
    print(f"DK SP pool: {len(dk_sps)}")

    _subsection(f"Confirmed ({len(ok_lines)})")
    if ok_lines:
        for line in ok_lines:
            print(f"  ✓ {line}")
    else:
        print("  (none)")

    _subsection(f"Unconfirmed / mismatch ({len(bad_lines)})")
    if bad_lines:
        for line in bad_lines:
            print(f"  ⚠ {line}")
    else:
        print("  (none)")


def _is_rp(player: DKPlayer) -> bool:
    pos = (player.dk_position or "").upper()
    return player.is_pitcher and pos == "RP"


def _is_sp(player: DKPlayer) -> bool:
    return player.is_pitcher and (player.dk_position or "").upper().startswith("SP")


def _report_projections(
    players: list[DKPlayer], projections: list[PlayerProjection]
) -> list[PlayerProjection]:
    _section("4. TOP PROJECTIONS PREVIEW")
    by_id = {p.player.dk_id: p for p in projections}

    def _rows(pool: list[DKPlayer], n: int) -> list[tuple[DKPlayer, PlayerProjection]]:
        out: list[tuple[DKPlayer, PlayerProjection]] = []
        for pl in pool:
            proj = by_id.get(pl.dk_id)
            if proj:
                out.append((pl, proj))
        out.sort(key=lambda x: -x[1].pts_q50)
        return out[:n]

    hitters = [p for p in players if not p.is_pitcher]
    sps = [p for p in players if _is_sp(p)]
    rps = [p for p in players if _is_rp(p)]

    def _print_table(title: str, ranked: list[tuple[DKPlayer, PlayerProjection]]) -> None:
        _subsection(title)
        if not ranked:
            print("  (none)")
            return
        print(f"  {'Name':<24} {'Pos':<5} {'Team':<5} {'Sal':>8} {'Q50':>6}")
        print("  " + "-" * 56)
        for pl, proj in ranked:
            print(
                f"  {pl.name:<24} {(pl.dk_position or '?'):<5} {pl.team:<5} "
                f"${pl.salary:>7,} {proj.pts_q50:>6.1f}"
            )

    _print_table("Top 10 hitters (q50)", _rows(hitters, 10))
    _print_table("Top 10 SPs (q50)", _rows(sps, 10))
    _print_table("Top 5 RPs (q50)", _rows(rps, 5))

    inflated = [
        (p.name, by_id[p.dk_id].pts_q50, p.dk_position or "?")
        for p in players
        if p.dk_id in by_id and by_id[p.dk_id].pts_q50 > _INFLATION_Q50
    ]
    inflated.sort(key=lambda x: -x[1])
    _subsection(f"Inflation flags (q50 > {_INFLATION_Q50:.0f})")
    if inflated:
        for name, q50, pos in inflated:
            print(f"  ⚠ {name} ({pos}) — {q50:.1f} pts q50")
    else:
        print("  (none)")

    return projections


def _report_ownership(projections: list[PlayerProjection]) -> list[PlayerProjection]:
    _section("5. OWNERSHIP PREVIEW")
    projector = OwnershipProjector()
    players = [p.player for p in projections]
    print(f"Running ownership simulation ({_OWNERSHIP_SIMS:,} sims)...")
    ownership = projector.project(
        players=players,
        base_projections=projections,
        n_sims=_OWNERSHIP_SIMS,
        n_jobs=4,
    )
    updated = projector.update_projections_with_ownership(projections, ownership)

    top_own = sorted(updated, key=lambda p: -p.ownership_proj)[:10]
    _subsection("Top 10 projected ownership")
    print(f"  {'Name':<24} {'Pos':<5} {'Own%':>7} {'Q50':>6}")
    print("  " + "-" * 48)
    for proj in top_own:
        pl = proj.player
        print(
            f"  {pl.name:<24} {(pl.dk_position or '?'):<5} "
            f"{proj.ownership_proj:>6.1f}% {proj.pts_q50:>6.1f}"
        )

    def _leverage_score(proj: PlayerProjection) -> float:
        return proj.pts_q50 / max(proj.ownership_proj, 0.1)

    top_lev = sorted(updated, key=_leverage_score, reverse=True)[:10]
    _subsection("Top 10 leverage (q50 / ownership)")
    print(f"  {'Name':<24} {'Pos':<5} {'Own%':>7} {'Q50':>6} {'Lev':>7}")
    print("  " + "-" * 56)
    for proj in top_lev:
        pl = proj.player
        lev = _leverage_score(proj)
        print(
            f"  {pl.name:<24} {(pl.dk_position or '?'):<5} "
            f"{proj.ownership_proj:>6.1f}% {proj.pts_q50:>6.1f} {lev:>7.2f}"
        )

    return updated


def _report_freshness(vegas_df: pd.DataFrame) -> None:
    _section("6. DATA FRESHNESS")
    bat_cache, pit_cache, bat_game, pit_game = _statcast_freshness()
    print(f"Statcast batters cache file:  {bat_cache}")
    print(f"Statcast batters max game:    {bat_game}")
    print(f"Statcast pitchers cache file: {pit_cache}")
    print(f"Statcast pitchers max game:   {pit_game}")
    print(f"Vegas odds fetch time:        {_vegas_fetch_time(vegas_df)}")
    print(f"Chadwick register cache:      {_chadwick_last_updated()}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pre-slate intelligence report for DFS lineup prep"
    )
    ap.add_argument(
        "--csv",
        type=str,
        default=None,
        help="DraftKings salary CSV (default: latest DKSalaries*.csv in data/uploads/)",
    )
    args = ap.parse_args()

    csv_path = _resolve_csv(args.csv)
    print(f"Slate CSV: {csv_path}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    parser = DKCSVParser()
    parser.parse(csv_path)
    result = parser.last_result
    players = result.players
    info = result.slate_info
    print(
        f"Pool: {len(players)} players | "
        f"{info.get('pitcher_count', 0)} pitchers | "
        f"{info.get('hitter_count', 0)} hitters | "
        f"{len(info.get('games', []))} games"
    )

    vegas_df = _report_vegas()
    inference = SlateInference()
    inference.load_models()
    inference.load_feature_matrices()
    inference.load_vegas()

    _report_auto_bans(players, inference)
    _report_sp_confirmation(players)

    print("\nBuilding projections (models + Vegas + probable SP filter)...")
    probable_pitchers = get_mlb_probable_pitchers()
    projections = inference.build_projections(
        players,
        use_models=True,
        probable_pitchers=probable_pitchers,
    )
    _report_projections(players, projections)
    _report_ownership(projections)
    _report_freshness(vegas_df)

    print()
    print("=" * 72)
    print("Report complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
