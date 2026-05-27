"""SP confirmation from two sources:
  1. MLB Stats API probable pitchers (authoritative, free)
  2. Rotowire confirmed starters (cross-reference)

Combined into a single get_confirmed_sps() function that returns
dict[str, str]: team_abbr -> pitcher_full_name
"""

from __future__ import annotations

from datetime import date

import requests
from loguru import logger

from data_pipeline.ingestion.rotowire_lineups import get_confirmed_starters

MLB_API_TO_DK: dict[str, str] = {
    "ARI": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BOS": "BOS",
    "CHC": "CHC",
    "CWS": "CWS",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "DET": "DET",
    "HOU": "HOU",
    "KC": "KC",
    "LAA": "LAA",
    "LAD": "LAD",
    "MIA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NYM": "NYM",
    "NYY": "NYY",
    "ATH": "ATH",
    "OAK": "ATH",
    "PHI": "PHI",
    "PIT": "PIT",
    "SD": "SD",
    "SF": "SF",
    "SEA": "SEA",
    "STL": "STL",
    "TB": "TB",
    "TEX": "TEX",
    "TOR": "TOR",
    "WSH": "WSH",
    "WSN": "WSH",
}


def get_mlb_probable_pitchers() -> dict[str, str]:
    """Fetch probable pitchers from MLB Stats API schedule endpoint.

    Returns dict: team_abbr -> pitcher_full_name
    Returns {} on failure — never raises.
    """
    try:
        today = date.today().strftime("%Y-%m-%d")
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={
                "sportId": 1,
                "date": today,
                "hydrate": "probablePitcher,team",
            },
            timeout=15,
        )
        r.raise_for_status()
        dates = r.json().get("dates", [])
        games = dates[0].get("games", []) if dates else []
        result: dict[str, str] = {}
        for g in games:
            for side in ["home", "away"]:
                team_data = g.get("teams", {}).get(side, {})
                abbr = team_data.get("team", {}).get("abbreviation", "")
                dk_abbr = MLB_API_TO_DK.get(abbr, abbr)
                name = team_data.get("probablePitcher", {}).get("fullName", "")
                if dk_abbr and name:
                    result[dk_abbr] = name
        logger.info(f"MLB Stats API: {len(result)} probable pitchers confirmed")
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"get_mlb_probable_pitchers failed: {exc}")
        return {}


def get_confirmed_sps() -> dict[str, str]:
    """Merge MLB Stats API probables with Rotowire confirmed starters.

    MLB Stats API is primary source — populated first.
    Rotowire fills gaps where MLB API has TBD.
    Where both have a name and they conflict, log a warning but keep MLB API.

    Returns dict: team_abbr -> pitcher_full_name
    """
    mlb_probables = get_mlb_probable_pitchers()
    rotowire_starters = get_confirmed_starters()  # team_abbr -> name

    merged: dict[str, str] = dict(mlb_probables)

    rw_fills = 0
    rw_conflicts = 0
    for team, rw_name in rotowire_starters.items():
        if team not in merged:
            merged[team] = rw_name
            rw_fills += 1
        else:
            mlb_name = merged[team]
            mlb_last = mlb_name.split()[-1].lower() if mlb_name else ""
            rw_last = rw_name.split()[-1].lower() if rw_name else ""
            if mlb_last and rw_last and mlb_last != rw_last:
                logger.warning(
                    f"SP conflict {team}: MLB API='{mlb_name}' vs Rotowire='{rw_name}' "
                    "— keeping MLB API"
                )
                rw_conflicts += 1

    logger.info(
        f"get_confirmed_sps: {len(merged)} total | "
        f"MLB API={len(mlb_probables)} | "
        f"Rotowire fills={rw_fills} | "
        f"conflicts={rw_conflicts}"
    )
    return merged

