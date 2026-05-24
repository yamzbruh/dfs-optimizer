"""Rotowire daily MLB lineups — confirmed batting orders and starting pitchers.

Scrapes https://www.rotowire.com/baseball/daily-lineups.php
Results are cached in-process for 10 minutes.
"""

from __future__ import annotations

import time
from typing import Any

import requests
from bs4 import BeautifulSoup
from loguru import logger

_ROTOWIRE_URL = "https://www.rotowire.com/baseball/daily-lineups.php"
_CACHE_TTL_SEC = 600
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; dfs-optimizer/1.0; +https://github.com/)"
    ),
}

ROTOWIRE_TO_DK: dict[str, str] = {
    "OAK": "ATH",
    "WSH": "WSH",
    "CWS": "CWS",
}

_cache_ts: float = 0.0
_cache_lineups: dict[str, list[str]] | None = None
_cache_starters: dict[str, str] | None = None


def _class_has(classes: list[str] | str | None, token: str) -> bool:
    if not classes:
        return False
    if isinstance(classes, str):
        return token in classes.split()
    return token in classes


def _sp_name_from_link(sp_link: Any) -> str:
    """Full SP name from Rotowire link; expand abbreviated text via URL slug."""
    href = sp_link.get("href", "") or ""
    if href and "/baseball/player/" in href:
        slug = href.split("/baseball/player/")[-1].split("?")[0].strip("/")
        slug = slug.rsplit("-", 1)[0]
        if slug:
            return " ".join(word.capitalize() for word in slug.split("-"))
    return sp_link.get_text(strip=True)


def _scrape_rotowire() -> tuple[dict[str, list[str]], dict[str, str]]:
    """Fetch and parse Rotowire daily lineups. Never raises."""
    confirmed_lineups: dict[str, list[str]] = {}
    confirmed_starters: dict[str, str] = {}

    try:
        r = requests.get(_ROTOWIRE_URL, headers=_HEADERS, timeout=20)
        r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Rotowire fetch failed: {exc}")
        return confirmed_lineups, confirmed_starters

    try:
        soup = BeautifulSoup(r.text, "html.parser")
        games = soup.select("div.lineup.is-mlb")
        if not games:
            games = [
                d
                for d in soup.find_all("div", class_=True)
                if d.get("class")
                and "lineup" in d.get("class", [])
                and "is-mlb" in d.get("class", [])
            ]

        confirmed_games = 0
        for div in games:
            abbr_divs = div.find_all("div", class_="lineup__abbr")
            if len(abbr_divs) < 2:
                continue

            visit_abbr = abbr_divs[0].get_text(strip=True).upper()
            home_abbr = abbr_divs[1].get_text(strip=True).upper()
            visit_abbr = ROTOWIRE_TO_DK.get(visit_abbr, visit_abbr)
            home_abbr = ROTOWIRE_TO_DK.get(home_abbr, home_abbr)

            game_had_confirmed = False
            for side, abbr in [("is-visit", visit_abbr), ("is-home", home_abbr)]:
                if not abbr:
                    continue

                ul = div.find(
                    "ul",
                    class_=lambda x: x
                    and _class_has(x, "lineup__list")
                    and _class_has(x, side),
                )
                if not ul:
                    continue

                confirmed = ul.find(
                    "li",
                    class_=lambda x: x and _class_has(x, "is-confirmed"),
                ) is not None

                sp_li = ul.find(
                    "li",
                    class_=lambda x: x and _class_has(x, "lineup__player-highlight"),
                )
                sp_name = ""
                if sp_li:
                    sp_link = sp_li.find("a")
                    if sp_link:
                        sp_name = _sp_name_from_link(sp_link)

                players: list[str] = []
                for li in ul.find_all(
                    "li",
                    class_=lambda x: x
                    and _class_has(x, "lineup__player")
                    and not _class_has(x, "highlight"),
                ):
                    a = li.find("a", title=True)
                    if a:
                        title = (a.get("title") or "").strip()
                        if title:
                            players.append(title)
                            continue
                    fallback = li.find("a")
                    if fallback:
                        text = fallback.get_text(strip=True)
                        if text:
                            players.append(text)

                if confirmed and players:
                    confirmed_lineups[abbr] = players
                    game_had_confirmed = True
                if confirmed and sp_name:
                    confirmed_starters[abbr] = sp_name

            if game_had_confirmed:
                confirmed_games += 1

        n_confirmed_teams = len(confirmed_lineups)
        logger.info(
            f"Rotowire: {confirmed_games} confirmed games, "
            f"{n_confirmed_teams} teams with lineups, "
            f"{len(confirmed_starters)} confirmed SPs"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Rotowire parse failed: {exc}")
        return {}, {}

    return confirmed_lineups, confirmed_starters


def _refresh_cache_if_needed() -> None:
    global _cache_ts, _cache_lineups, _cache_starters
    now = time.time()
    if (
        _cache_lineups is not None
        and _cache_starters is not None
        and (now - _cache_ts) < _CACHE_TTL_SEC
    ):
        return
    lineups, starters = _scrape_rotowire()
    _cache_lineups = lineups
    _cache_starters = starters
    _cache_ts = now


def get_confirmed_lineups() -> dict[str, list[str]]:
    """Map team abbreviation → confirmed starter full names (batting order).

    Empty list when that team's lineup is not confirmed or not yet posted.
    """
    try:
        _refresh_cache_if_needed()
        return dict(_cache_lineups or {})
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"get_confirmed_lineups failed: {exc}")
        return {}


def get_confirmed_starters() -> dict[str, str]:
    """Map team abbreviation → starting pitcher name (confirmed lineups only)."""
    try:
        _refresh_cache_if_needed()
        lineups = _cache_lineups or {}
        starters = _cache_starters or {}
        return {
            team: name
            for team, name in starters.items()
            if lineups.get(team)
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"get_confirmed_starters failed: {exc}")
        return {}
