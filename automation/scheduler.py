"""DFS automation — APScheduler jobs for MLB slates (ET).

Morning job pulls all today's DK salary CSVs (one per draft group) and Vegas
odds. The War Room picks a slate via the API; T-3hr automation targets the
largest contest-count slate.

Run::

    python -m automation.scheduler

Requires: ``apscheduler``, ``beautifulsoup4``, ``pybaseball``, ``pytz``,
``DISCORD_WEBHOOK_URL`` (optional).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytz
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from loguru import logger
from zoneinfo import ZoneInfo

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_pipeline.ingestion.dk_csv_exporter import DKLineupExporter
from data_pipeline.ingestion.dk_csv_parser import DKCSVParser, DKPlayer
from data_pipeline.ingestion.lineup_status import (
    LineupStatusChecker,
    team_ids_from_dk_players,
)
from data_pipeline.ingestion.odds_ingestion import OddsIngestion
from data_pipeline.loaders.parquet_cache import ParquetCache
from ml.features.ownership_projector import OwnershipProjector
from ml.inference.slate_inference import SlateInference
from optimizer.constraints.lineup_optimizer import (
    LineupOptimizer,
    LineupResult,
    PlayerProjection,
)

# ---------------------------------------------------------------------------
# Globals (module-level)
# ---------------------------------------------------------------------------

_slate_inference: SlateInference | None = None
_current_players: list[DKPlayer] = []
_current_projections: list[PlayerProjection] = []
_current_lineups: list[LineupResult] = []
_discord_url: str = ""
_models_loaded: bool = False

_status_checker: LineupStatusChecker | None = None
_ownership_projector: OwnershipProjector | None = None
_lineup_optimizer: LineupOptimizer | None = None
_dk_exporter: DKLineupExporter | None = None
_dk_parser: DKCSVParser | None = None

_todays_slates: list[dict[str, Any]] = []

_t2_banned_snapshot: frozenset[str] = frozenset()
_t2_bp_sp_signature: str = ""

_ET = ZoneInfo("America/New_York")
_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0"}


def send_discord(message: str, webhook_url: str) -> None:
    """Post a message to Discord. Never raises — logs on failure."""
    if not webhook_url:
        return
    try:
        requests.post(webhook_url, json={"content": message}, timeout=10)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Discord webhook failed: {exc}")


def _et_today() -> date:
    return datetime.now(_ET).date()


def _et_date_slug() -> str:
    return _et_today().isoformat()


def _job_wrapper(name: str, fn: Callable[[], None]) -> Callable[[], None]:
    """Return a callable that logs duration and swallows exceptions."""

    def _run() -> None:
        t0 = time.perf_counter()
        logger.info(f"JOB START {name}")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"JOB FAIL {name}: {exc}")
            send_discord(f"❌ **{name}** failed: `{exc}`", _discord_url)
        finally:
            dt_s = time.perf_counter() - t0
            logger.info(f"JOB END {name}  duration={dt_s:.1f}s")

    return _run


def initialize_pipeline() -> None:
    """Load models, feature matrices, Vegas odds. Called once at startup."""
    global _slate_inference, _models_loaded, _discord_url
    global _status_checker, _ownership_projector, _lineup_optimizer, _dk_exporter, _dk_parser

    load_dotenv(_ROOT / ".env")
    _discord_url = os.getenv("DISCORD_WEBHOOK_URL", "")

    _slate_inference = SlateInference()
    _slate_inference.load_models()
    _slate_inference.load_feature_matrices()
    _slate_inference.load_vegas()

    _status_checker = LineupStatusChecker()
    _ownership_projector = OwnershipProjector()
    _lineup_optimizer = LineupOptimizer()
    _dk_exporter = DKLineupExporter()
    _dk_parser = DKCSVParser()

    _models_loaded = True
    logger.info("Pipeline initialized")
    send_discord("🤖 DFS automation pipeline online", _discord_url)


def _statcast_incremental_append() -> None:
    """Append new Statcast rows from last cached ``game_date`` through today."""
    from pybaseball import statcast  # noqa: PLC0415

    cache = ParquetCache(_ROOT / "data" / "parquet")
    today = _et_today()
    year = today.year
    key_bat = f"statcast/batters_{year}"
    key_pit = f"statcast/pitchers_{year}"

    existing = cache.load(key_bat)
    if existing is None or existing.empty:
        start_dt = f"{year}-03-01"
    else:
        last = pd.to_datetime(existing["game_date"], errors="coerce").max()
        if pd.isna(last):
            start_dt = f"{year}-03-01"
        else:
            start_dt = (last.date() + timedelta(days=1)).isoformat()

    end_dt = today.isoformat()
    if start_dt > end_dt:
        logger.info("Statcast incremental: cache already includes today")
        return

    logger.info(f"Statcast incremental pull {start_dt} → {end_dt}")
    try:
        df_new = statcast(start_dt=start_dt, end_dt=end_dt)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"statcast incremental failed: {exc}")
        return

    if df_new is None or df_new.empty:
        logger.warning("Statcast incremental: no new rows returned")
        return

    df_new = df_new.copy()
    df_new["game_date"] = pd.to_datetime(df_new["game_date"], errors="coerce")

    base = existing if existing is not None else pd.DataFrame()
    combined = pd.concat([base, df_new], ignore_index=True)
    dedupe_subset = [
        c
        for c in ("game_pk", "at_bat_number", "pitch_number")
        if c in combined.columns
    ]
    if dedupe_subset:
        combined = combined.drop_duplicates(subset=dedupe_subset, keep="last")
    else:
        combined = combined.drop_duplicates()

    meta = {"year": year, "rows": len(combined), "incremental": True}
    cache.save(combined, key_bat, metadata=meta)
    cache.save(combined, key_pit, metadata=meta)
    logger.info(f"Statcast cache updated: {len(combined):,} rows for {year}")


def _parse_sd_ms(sd: str) -> datetime | None:
    m = re.search(r"\d+", sd or "")
    if not m:
        return None
    return datetime.fromtimestamp(int(m.group()) / 1000, tz=timezone.utc)


def _prize_value(name: str) -> float:
    m = re.search(r"\$(\d+(?:\.\d+)?)(M|K)?", name)
    if not m:
        return 0.0
    val = float(m.group(1))
    suf = m.group(2)
    if suf == "M":
        val *= 1_000_000
    elif suf == "K":
        val *= 1_000
    return val


def _download_salary_csv(dg: int, lock_time: datetime) -> Path | None:
    """Download DK salary CSV for a draft group. Returns path or None on failure."""
    try:
        date_str = lock_time.astimezone(_ET).strftime("%Y-%m-%d")
        out_path = _ROOT / "data" / "uploads" / f"DKSalaries_{date_str}_dg{dg}.csv"
        if out_path.exists():
            logger.info(f"Salary CSV already exists: {out_path}")
            return out_path
        url = (
            "https://www.draftkings.com/lineup/getavailableplayerscsv"
            f"?contestTypeId=21&draftGroupId={dg}"
        )
        r = requests.get(url, headers=_HTTP_HEADERS, timeout=30)
        r.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(r.content)
        logger.info(f"Downloaded salary CSV: {out_path} ({len(r.content):,} bytes)")
        return out_path
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to download salary CSV for dg={dg}: {exc}")
        return None


def _get_todays_slates() -> list[dict[str, Any]]:
    """Fetch today's MLB Classic slates from DK (one row per unique ``dg``).

    Each slate dict: ``dg``, ``name``, ``lock_time`` (UTC-aware datetime),
    ``lock_time_et``, ``contest_count``, ``csv_path`` (``Path`` or ``None``).
    """
    et = pytz.timezone("America/New_York")
    today_et = datetime.now(et).date()

    r = requests.get(
        "https://www.draftkings.com/lobby/getcontests?sport=MLB",
        headers=_HTTP_HEADERS,
        timeout=20,
    )
    r.raise_for_status()
    contests = r.json().get("Contests", [])

    by_dg: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for c in contests:
        if not isinstance(c, dict):
            continue
        if c.get("gameType") != "Classic":
            continue
        name = str(c.get("n", "") or "")
        if "MLB" not in name:
            continue
        sd = _parse_sd_ms(str(c.get("sd", "") or ""))
        if sd is None:
            continue
        if sd.astimezone(et).date() != today_et:
            continue
        dg = c.get("dg")
        if dg is None:
            continue
        by_dg[int(dg)].append(c)

    slates: list[dict[str, Any]] = []
    for dg, group in sorted(by_dg.items()):
        group.sort(key=lambda x: _prize_value(str(x.get("n", ""))), reverse=True)
        label = str(group[0].get("n", "") or f"MLB Classic dg={dg}")
        times: list[datetime] = []
        for row in group:
            t = _parse_sd_ms(str(row.get("sd", "") or ""))
            if t is not None:
                times.append(t)
        if not times:
            continue
        lock_time = min(times)
        lock_time_et = lock_time.astimezone(et).strftime("%I:%M %p ET")
        slate: dict[str, Any] = {
            "dg": dg,
            "name": label,
            "lock_time": lock_time,
            "lock_time_et": lock_time_et,
            "contest_count": len(group),
            "csv_path": None,
        }
        slate["csv_path"] = _download_salary_csv(dg, lock_time)
        slates.append(slate)

    return slates


def job_morning_setup() -> None:
    """Morning data refresh (8:00 AM ET)."""
    global _todays_slates

    try:
        from pybaseball import chadwick_register  # noqa: PLC0415

        chadwick_register(save=True)
        logger.info("Chadwick register refreshed (save=True)")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Chadwick refresh failed: {exc}")

    _statcast_incremental_append()

    slates = _get_todays_slates()
    _todays_slates = slates

    try:
        vegas_dict = OddsIngestion().get_team_implied_totals()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Vegas pull in morning job: {exc}")
        vegas_dict = {}

    lines = ["☀️ **Morning Setup Complete**\n"]
    lines.append("📊 Statcast updated | Chadwick refreshed\n")
    lines.append("⚾ **Today's Slates:**")
    for s in slates:
        lock_str = s["lock_time"].astimezone(_ET).strftime("%I:%M %p ET")
        csv_ok = "✅" if s.get("csv_path") else "❌"
        lines.append(
            f"  {csv_ok} dg={s['dg']} | Lock: {lock_str} | {s['contest_count']} contests"
        )

    top_implied = sorted(vegas_dict.items(), key=lambda x: -x[1])[:5]
    lines.append("\n💰 **Top Vegas Implied:**")
    for team, implied in top_implied:
        lines.append(f"  {team}: {implied:.1f}")

    send_discord("\n".join(lines), _discord_url)


def _top_implied_totals_text(n: int = 3) -> tuple[str, str]:
    """Return (top_teams_line, vegas_bar_line) for Discord."""
    try:
        odds = OddsIngestion()
        df = odds.get_mlb_implied_totals(force_refresh=True)
        if df is None or df.empty or "team" not in df.columns:
            return "n/a", "n/a"
        work = df.copy()
        work["implied_total"] = pd.to_numeric(
            work.get("implied_total", 0), errors="coerce"
        ).fillna(0.0)
        top = (
            work.sort_values("implied_total", ascending=False)
            .drop_duplicates(subset=["team"])
            .head(n)
        )
        parts = [
            f"{str(row['team']).strip()}: {float(row['implied_total']):.2f}"
            for _, row in top.iterrows()
        ]
        bar = " | ".join(parts) if parts else "n/a"
        tops = ", ".join(parts) if parts else "n/a"
        return tops, bar
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"implied totals formatting failed: {exc}")
        return "n/a", "n/a"


def _apply_bans(
    projections: list[PlayerProjection], banned: set[str]
) -> list[PlayerProjection]:
    return [
        replace(p, is_banned=(p.player.dk_id in banned)) for p in projections
    ]


def job_t3hr() -> None:
    """T-3hr: main slate (most contests), injury check, projections, ownership."""
    global _current_players, _current_projections, _current_lineups

    if (
        _slate_inference is None
        or _dk_parser is None
        or _status_checker is None
        or _ownership_projector is None
    ):
        logger.error("Pipeline not initialized — skipping T-3hr")
        return

    slates = _get_todays_slates()
    for s in slates:
        if not s.get("csv_path") or not Path(s["csv_path"]).exists():
            s["csv_path"] = _download_salary_csv(s["dg"], s["lock_time"])

    if not slates:
        send_discord("⚠️ T-3hr: no MLB Classic slates found for today", _discord_url)
        return

    main = max(slates, key=lambda s: int(s.get("contest_count") or 0))
    csv_path = main.get("csv_path")
    if not csv_path or not Path(csv_path).exists():
        send_discord(
            f"⚠️ T-3hr: could not download salary CSV for dg={main['dg']}",
            _discord_url,
        )
        return

    players = _dk_parser.parse(Path(csv_path))
    _current_players = players
    _current_lineups = []

    tids = team_ids_from_dk_players(players)
    _status_checker.reset_cache()
    _status_checker.load_statuses(tids)
    banned = _status_checker.get_unavailable_dk_ids(
        players, _slate_inference.match_dk_player_to_mlbam
    )

    _slate_inference.load_feature_matrices()
    _slate_inference.load_vegas()
    raw_projs = _slate_inference.build_projections(players, use_models=True)
    base = _apply_bans(raw_projs, banned)

    ownership = _ownership_projector.project(
        players=players,
        base_projections=base,
        n_sims=1000,
        n_jobs=4,
    )
    updated = _ownership_projector.update_projections_with_ownership(base, ownership)
    _current_projections = updated

    tops, bar = _top_implied_totals_text(3)
    lock_str = main["lock_time"].astimezone(_ET).strftime("%I:%M %p ET")
    msg = (
        f"⚾ **T-3hr — auto slate dg={main['dg']}** ({main['contest_count']} contests, "
        f"lock {lock_str})\n"
        f"{len(players)} players, {len(banned)} auto-banned (IL/OUT)\n"
        f"Ownership sims applied (1k). Top implied: {tops}\n"
        f"Vegas: {bar}"
    )
    send_discord(msg, _discord_url)


def _normalize_name_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def scrape_baseball_press_confirmed_sps() -> dict[str, str]:
    """Scrape Baseball Press for confirmed SP per team (abbr → name)."""
    url = "https://www.baseballpress.com/lineups"
    try:
        r = requests.get(url, headers=_HTTP_HEADERS, timeout=25)
        r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Baseball Press fetch failed: {exc}")
        return {}

    soup = BeautifulSoup(r.text, "html.parser")
    out: dict[str, str] = {}

    for div in soup.select("div.lineup"):
        team_el = div.select_one("[data-team]")
        abbr = ""
        if team_el and team_el.has_attr("data-team"):
            abbr = str(team_el["data-team"]).strip().upper()
        if not abbr:
            hdr = div.find(["h2", "h3", "h4"])
            if hdr:
                m = re.search(r"\b([A-Z]{2,4})\b", hdr.get_text(" ", strip=True))
                if m:
                    abbr = m.group(1)
        if not abbr:
            continue
        pitch_link = None
        for a in div.select('a[href*="/players/"]'):
            t = a.get_text(" ", strip=True)
            if t and "pitcher" not in t.lower():
                pitch_link = t
                break
        if pitch_link:
            out[abbr] = pitch_link

    if out:
        return out

    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue
            team_txt = tds[0].get_text(" ", strip=True)
            m = re.search(r"\b([A-Z]{2,4})\b", team_txt)
            if not m:
                continue
            abbr = m.group(1)
            link = tds[1].find("a")
            name = (
                link.get_text(" ", strip=True) if link else tds[1].get_text(" ", strip=True)
            )
            if name:
                out[abbr] = name
    return out


def _sp_mismatch_flags(
    players: list[DKPlayer], confirmed: dict[str, str]
) -> tuple[list[str], list[str]]:
    """Return (confirmed_lines, unconfirmed_lines) for DK SPs."""
    confirmed_lines: list[str] = []
    unconfirmed_lines: list[str] = []

    dk_sps = [
        p
        for p in players
        if p.is_pitcher and (p.dk_position or "").upper().startswith("SP")
    ]
    for p in dk_sps:
        team = (p.team or "").strip().upper()
        bp_name = confirmed.get(team)
        if not bp_name:
            unconfirmed_lines.append(f"{p.name} ({team}) — no Baseball Press row")
            continue
        if _normalize_name_key(p.name) in _normalize_name_key(bp_name):
            confirmed_lines.append(f"{p.name} ({team}) ↔ BP {bp_name}")
        elif _normalize_name_key(bp_name) in _normalize_name_key(p.name):
            confirmed_lines.append(f"{p.name} ({team}) ↔ BP {bp_name}")
        else:
            unconfirmed_lines.append(
                f"{p.name} ({team}) ⚠ BP lists **{bp_name}**"
            )
    return confirmed_lines, unconfirmed_lines


def job_t2hr() -> None:
    """T-2hr: Baseball Press SP check, ownership sims, preliminary lineups."""
    global _current_projections, _current_lineups, _t2_banned_snapshot, _t2_bp_sp_signature

    if (
        _slate_inference is None
        or _ownership_projector is None
        or _lineup_optimizer is None
        or _status_checker is None
    ):
        logger.error("Pipeline not initialized — skipping T-2hr")
        return

    if not _current_players:
        send_discord("⚠️ T-2hr: no players in memory — run T-3hr first", _discord_url)
        return

    bp = scrape_baseball_press_confirmed_sps()
    _t2_bp_sp_signature = json.dumps(sorted(bp.items()), sort_keys=True)
    ok_sp, bad_sp = _sp_mismatch_flags(_current_players, bp)

    tids = team_ids_from_dk_players(_current_players)
    _status_checker.reset_cache()
    _status_checker.load_statuses(tids)
    banned = _status_checker.get_unavailable_dk_ids(
        _current_players, _slate_inference.match_dk_player_to_mlbam
    )
    _t2_banned_snapshot = frozenset(banned)

    _slate_inference.load_vegas()
    raw = _slate_inference.build_projections(_current_players, use_models=True)
    base = _apply_bans(raw, banned)

    ownership = _ownership_projector.project(
        players=_current_players,
        base_projections=base,
        n_sims=1000,
        n_jobs=4,
    )
    updated = _ownership_projector.update_projections_with_ownership(base, ownership)
    _current_projections = updated

    lineups = _lineup_optimizer.generate_lineups(
        updated,
        n_lineups=20,
        banned_ids=banned,
        use_monte_carlo=True,
        n_simulations=500,
    )
    _current_lineups = lineups

    top_lev = sorted(updated, key=lambda x: x.leverage_score, reverse=True)[:5]
    lev_lines = "\n".join(
        f"• {p.player.name} ({p.player.team}) — {p.pts_q50:.1f} pts / "
        f"{p.ownership_proj:.1f}% own"
        for p in top_lev
    )
    avg_pts = (
        sum(l.projected_pts for l in lineups) / len(lineups) if lineups else 0.0
    )

    ok_block = "\n".join(f"• {x}" for x in ok_sp) or "• (none)"
    bad_block = "\n".join(f"• {x}" for x in bad_sp) or "• (none)"
    msg = (
        "📊 **T-2hr update**\n"
        "✅ Confirmed SPs:\n"
        f"{ok_block}\n"
        "⚠️ Unconfirmed:\n"
        f"{bad_block}\n"
        f"🚫 Auto-banned: {', '.join(sorted(banned)) or '(none)'}\n\n"
        "**Top 5 leverage plays:**\n"
        f"{lev_lines or '(n/a)'}\n\n"
        f"**{len(lineups)} lineups generated** — avg proj: {avg_pts:.1f} pts"
    )
    send_discord(msg, _discord_url)


def job_t1hr() -> None:
    """T-1hr: Re-check rosters + Baseball Press; regenerate if needed."""
    global _current_projections, _current_lineups

    if _slate_inference is None or _status_checker is None or _lineup_optimizer is None:
        logger.error("Pipeline not initialized — skipping T-1hr")
        return

    if not _current_players:
        send_discord("⚠️ T-1hr: no slate in memory", _discord_url)
        return

    _status_checker.reset_cache()
    tids = team_ids_from_dk_players(_current_players)
    _status_checker.load_statuses(tids)
    banned = _status_checker.get_unavailable_dk_ids(
        _current_players, _slate_inference.match_dk_player_to_mlbam
    )
    banned_f = frozenset(banned)

    bp = scrape_baseball_press_confirmed_sps()
    bp_sig = json.dumps(sorted(bp.items()), sort_keys=True)

    changed = banned_f != _t2_banned_snapshot or bp_sig != _t2_bp_sp_signature

    if changed:
        new_bans = sorted(banned_f - _t2_banned_snapshot)
        name_hint = ""
        if new_bans:
            id_to_name = {p.dk_id: p.name for p in _current_players}
            names = [id_to_name.get(i, i) for i in new_bans]
            name_hint = " — " + ", ".join(names)

        _slate_inference.load_vegas()
        raw = _slate_inference.build_projections(_current_players, use_models=True)
        base = _apply_bans(raw, banned)
        _current_projections = base
        lineups = _lineup_optimizer.generate_lineups(
            base,
            n_lineups=20,
            banned_ids=banned,
            use_monte_carlo=True,
            n_simulations=500,
        )
        _current_lineups = lineups
        send_discord(
            "🚨 **LINEUP CHANGE**"
            f"{name_hint} — bans or Baseball Press differ from T-2hr snapshot; "
            "**projections + lineups regenerated**",
            _discord_url,
        )
    else:
        send_discord("✅ T-1hr check — no changes, lineups stable", _discord_url)


def _pitcher_usage_lines(lineups: list[LineupResult], top: int = 5) -> str:
    ctr: Counter[str] = Counter()
    for lu in lineups:
        for pl, slot in lu.players:
            if slot == "P" and pl.is_pitcher:
                ctr[pl.name] += 1
    lines = []
    n = len(lineups) or 1
    for name, cnt in ctr.most_common(top):
        lines.append(f"• {name}: {cnt} lineups ({100.0 * cnt / n:.0f}%)")
    return "\n".join(lines) if lines else "(n/a)"


def _hitter_exposure_lines(lineups: list[LineupResult], top: int = 5) -> str:
    ctr: Counter[str] = Counter()
    for lu in lineups:
        for pl, slot in lu.players:
            if not pl.is_pitcher:
                ctr[pl.name] += 1
    lines = []
    n = len(lineups) or 1
    for name, cnt in ctr.most_common(top):
        lines.append(f"• {name}: {cnt} lineups ({100.0 * cnt / n:.0f}%)")
    return "\n".join(lines) if lines else "(n/a)"


def job_t30min() -> None:
    """T-30min: Final roster check and export upload CSV."""
    if _status_checker is None or _dk_exporter is None:
        logger.error("Pipeline not initialized — skipping T-30min")
        return

    if not _current_lineups:
        send_discord("⚠️ T-30min: no lineups to export", _discord_url)
        return

    if _current_players:
        _status_checker.reset_cache()
        _status_checker.load_statuses(team_ids_from_dk_players(_current_players))

    slug = _et_date_slug()
    export_path = _ROOT / "data" / "exports" / f"DK_Upload_{slug}.csv"
    lineup_data = [lu.players for lu in _current_lineups]
    _dk_exporter.export(lineup_data, export_path)

    n = len(_current_lineups)
    avg = sum(l.projected_pts for l in _current_lineups) / n if n else 0.0
    avg_salary = (
        sum(l.total_salary for l in _current_lineups) / n if n else 0.0
    )

    msg = (
        f"🔒 **FINAL LINEUPS — {n} lineups exported**\n\n"
        f"Avg projected pts: {avg:.1f}\n"
        f"Salary used: avg ${avg_salary:,.0f}\n\n"
        "**Pitcher breakdown:**\n"
        f"{_pitcher_usage_lines(_current_lineups)}\n\n"
        "**Top hitter exposure:**\n"
        f"{_hitter_exposure_lines(_current_lineups)}\n\n"
        f"📁 Lineups saved to `{export_path}`\n"
        "⚠️ Upload to DraftKings before **7:05pm ET**"
    )
    send_discord(msg, _discord_url)


if __name__ == "__main__":
    initialize_pipeline()

    scheduler = BackgroundScheduler(timezone="America/New_York")
    scheduler.add_job(_job_wrapper("morning_setup", job_morning_setup), "cron", hour=8, minute=0)
    scheduler.add_job(_job_wrapper("t3hr", job_t3hr), "cron", hour=16, minute=5)
    scheduler.add_job(_job_wrapper("t2hr", job_t2hr), "cron", hour=17, minute=5)
    scheduler.add_job(_job_wrapper("t1hr", job_t1hr), "cron", hour=18, minute=5)
    scheduler.add_job(_job_wrapper("t30min", job_t30min), "cron", hour=18, minute=35)

    scheduler.start()
    logger.info("Scheduler started — waiting for jobs")

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Scheduler stopped")
