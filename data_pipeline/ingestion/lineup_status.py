"""MLB player availability checker for DFS slate filtering.

Pulls injury/roster status from MLB Stats API (40-man roster per team and
optional injuries feed). Bans players on injured list (``D7``/``D10``/etc.),
``OUT``, ``SUSP``, ``RM`` (reassigned to minors), and other ``D``+digit
variants the API may add.

Day-to-day (DTD) is flagged but not auto-banned.

Usage:
    checker = LineupStatusChecker()
    checker.load_statuses(team_ids=[121, 120])
    banned_ids = checker.get_unavailable_dk_ids(
        dk_players, match_player=slate.match_dk_player_to_mlbam
    )
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from typing import Any

import requests
from loguru import logger

from data_pipeline.ingestion.dk_csv_parser import DKPlayer
from data_pipeline.ingestion.statcast_loader import MLB_TEAM_ID_TO_ABBR

ABBR_TO_MLB_TEAM_ID: dict[str, int] = {v: k for k, v in MLB_TEAM_ID_TO_ABBR.items()}

_ROSTER_TYPE = "40Man"

_BAN_CODES = frozenset({"D7", "D10", "D15", "D60", "OUT", "SUSP", "RM"})
_DTD_CODES = frozenset({"DTD"})

_STATSAPI = "https://statsapi.mlb.com/api/v1"
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "dfs-optimizer-lineup-status/1.0"})


def _season_year() -> int:
    return dt.date.today().year


def _status_code(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, dict):
        c = raw.get("code")
        return str(c).strip().upper() if c is not None else ""
    return str(raw).strip().upper()


def _is_ban_code(code: str) -> bool:
    if not code:
        return False
    if code in _BAN_CODES:
        return True
    # Catch any future Dx-day variants (D7, D10, D15, D60, D180...)
    if code.startswith("D") and code[1:].isdigit():
        return True
    return False


def _is_dtd_code(code: str) -> bool:
    return code in _DTD_CODES


class LineupStatusChecker:
    def __init__(self) -> None:
        self._unavailable_mlbam: set[int] = set()
        self._dtd_mlbam: set[int] = set()
        self._status_loaded = False
        self._status_reasons: dict[int, str] = {}
        self._cache_key: str | None = None
        self._roster_name_to_mlbam: dict[str, int] = {}

    def reset_cache(self) -> None:
        """Clear session cache (e.g. new slate upload)."""
        self._unavailable_mlbam.clear()
        self._dtd_mlbam.clear()
        self._status_reasons.clear()
        self._status_loaded = False
        self._cache_key = None
        self._roster_name_to_mlbam = {}

    def _fetch_roster(self, team_id: int, season: int) -> list[dict[str, Any]]:
        url = f"{_STATSAPI}/teams/{team_id}/roster"
        params = {"rosterType": _ROSTER_TYPE, "season": season}
        r = _SESSION.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        return list(data.get("roster") or [])

    def _fetch_injuries(self, season: int) -> None:
        url = f"{_STATSAPI}/injuries"
        params = {"sportId": 1, "season": season}
        r = _SESSION.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        injuries = data.get("injuries") or data.get("teams") or []
        if isinstance(injuries, dict):
            injuries = injuries.get("injuries", [])
        if not isinstance(injuries, list):
            return
        for row in injuries:
            if not isinstance(row, dict):
                continue
            pid = row.get("playerId") or row.get("id")
            if pid is None:
                continue
            mlbam = int(pid)
            desc = str(row.get("description") or row.get("note") or "")
            st = row.get("status") or row.get("injuryStatus") or ""
            code = _status_code(st) if isinstance(st, dict) else str(st).strip().upper()
            if _is_ban_code(code):
                self._unavailable_mlbam.add(mlbam)
                self._status_reasons.setdefault(
                    mlbam, f"Injuries feed: {desc or code}"
                )
            elif _is_dtd_code(code):
                self._dtd_mlbam.add(mlbam)
                self._status_reasons.setdefault(mlbam, f"DTD ({desc or code})")

    def _ingest_roster_row(self, entry: dict[str, Any]) -> None:
        person = entry.get("person") or {}
        pid = person.get("id")
        if pid is None:
            return
        mlbam = int(pid)
        full_name = str(person.get("fullName", "") or "").strip()
        if full_name:
            self._roster_name_to_mlbam[full_name.lower()] = mlbam
        st = entry.get("status") or {}
        code = _status_code(st)
        desc = ""
        if isinstance(st, dict):
            desc = str(st.get("description") or "").strip()
        reason = desc or code or "roster status"

        if _is_ban_code(code):
            self._unavailable_mlbam.add(mlbam)
            self._status_reasons[mlbam] = reason
        elif _is_dtd_code(code):
            self._dtd_mlbam.add(mlbam)
            self._status_reasons.setdefault(mlbam, f"DTD — {reason}")

    def load_statuses(self, team_ids: list[int] | None = None) -> None:
        """Fetch 40-man roster status for ``team_ids``; merge injuries feed.

        Cached for the session when the cache key
        ``(local date, sorted team_ids)`` is unchanged.
        """
        season = _season_year()
        ids = sorted(set(team_ids or []))
        key = f"{dt.date.today().isoformat()}|{','.join(str(i) for i in ids)}"
        if self._status_loaded and self._cache_key == key:
            return

        self._unavailable_mlbam.clear()
        self._dtd_mlbam.clear()
        self._status_reasons.clear()
        self._roster_name_to_mlbam.clear()
        self._cache_key = key
        self._status_loaded = True

        if not ids:
            logger.warning("LineupStatusChecker.load_statuses: no team_ids; skipping")
            return

        try:
            for tid in ids:
                try:
                    roster = self._fetch_roster(tid, season)
                    for entry in roster:
                        if isinstance(entry, dict):
                            self._ingest_roster_row(entry)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        f"LineupStatusChecker: roster team={tid} failed: {exc}"
                    )

            try:
                self._fetch_injuries(season)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"LineupStatusChecker: injuries endpoint failed: {exc}"
                )

        except Exception as exc:  # noqa: BLE001
            logger.warning(f"LineupStatusChecker.load_statuses failed: {exc}")
            self._unavailable_mlbam.clear()
            self._dtd_mlbam.clear()
            self._status_reasons.clear()
            self._roster_name_to_mlbam.clear()

    def _match_dk_player_by_roster_name(self, player: DKPlayer) -> int | None:
        """Match DK player name to MLBAM ID using roster API name lookup.

        Exact full-name match only. More reliable than Statcast/Chadwick for
        injured players who have no recent game data.
        """
        dk_name = player.name.strip().lower()
        return self._roster_name_to_mlbam.get(dk_name)

    def get_unavailable_mlbam_ids(self) -> set[int]:
        return set(self._unavailable_mlbam)

    def get_dtd_mlbam_ids(self) -> set[int]:
        return set(self._dtd_mlbam)

    def get_unavailable_dk_ids(
        self,
        dk_players: list[DKPlayer],
        match_player: Callable[[DKPlayer], int | None],
    ) -> set[str]:
        """Map unavailable MLBAM ids to DK ``dk_id`` strings."""
        out: set[str] = set()
        for p in dk_players:
            mid = self._match_dk_player_by_roster_name(p)
            if mid is None:
                mid = match_player(p)
            if mid is None:
                continue
            if mid in self._unavailable_mlbam:
                out.add(p.dk_id)
        return out

    def get_scratched_dk_ids(
        self,
        dk_players: list[DKPlayer],
        match_player: Callable[[DKPlayer], int | None],
    ) -> set[str]:
        """DK ids for hitters not in Rotowire's confirmed lineup (per team).

        Uses :func:`rotowire_lineups.get_confirmed_lineups` with fuzzy name
        matching (``token_sort_ratio`` >= 80). Only teams with at least 8
        confirmed batters are evaluated. Pitchers are excluded.
        """
        del match_player  # name-based; signature kept for API parity

        try:
            from data_pipeline.ingestion.rotowire_lineups import (
                get_confirmed_lineups,
            )
            from rapidfuzz import fuzz
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"get_scratched_dk_ids import failed: {exc}")
            return set()

        confirmed_by_team = get_confirmed_lineups()
        if not confirmed_by_team:
            return set()

        min_lineup = 8
        scratched: set[str] = set()

        for p in dk_players:
            if p.is_pitcher:
                continue
            team = (p.team or "").strip().upper()
            lineup_names = confirmed_by_team.get(team)
            if not lineup_names or len(lineup_names) < min_lineup:
                continue

            dk_name = p.name.strip().lower()
            best = max(
                fuzz.token_sort_ratio(dk_name, n.lower()) for n in lineup_names
            )
            if best < 80:
                scratched.add(p.dk_id)

        if scratched:
            logger.info(
                f"Rotowire scratched: {len(scratched)} DK hitters "
                f"not in confirmed lineups"
            )
        return scratched

    def get_status_report(
        self,
        dk_players: list[DKPlayer],
        match_player: Callable[[DKPlayer], int | None],
    ) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for p in dk_players:
            mid = self._match_dk_player_by_roster_name(p)
            if mid is None:
                mid = match_player(p)
            if mid is None:
                continue
            reason = self._status_reasons.get(mid, "")
            if mid in self._unavailable_mlbam:
                rows.append(
                    {
                        "name": p.name,
                        "team": p.team,
                        "dk_id": p.dk_id,
                        "status": "unavailable",
                        "reason": reason or "Unavailable",
                    }
                )
            elif mid in self._dtd_mlbam:
                rows.append(
                    {
                        "name": p.name,
                        "team": p.team,
                        "dk_id": p.dk_id,
                        "status": "dtd",
                        "reason": reason or "Day-to-day",
                    }
                )
        return rows


def team_ids_from_dk_players(players: list[DKPlayer]) -> list[int]:
    """Resolve unique MLB Stats API team ids from DK abbreviations."""
    ids: list[int] = []
    seen: set[int] = set()
    for p in players:
        ab = (p.team or "").strip().upper()
        tid = ABBR_TO_MLB_TEAM_ID.get(ab)
        if tid is not None and tid not in seen:
            seen.add(tid)
            ids.append(tid)
    return ids
