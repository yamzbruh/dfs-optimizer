"""Vegas odds ingestion for DFS ownership modeling.

Pulls implied team totals from The Odds API for today's
MLB slate. Used as a key signal in the ownership proxy model.

Free tier: 500 requests/month. Each call costs 1 request.
Run once per slate upload, cache for the session.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "baseball_mlb"

# Session cache: avoid repeat calls during one process lifetime.
_ODDS_SESSION_DF: pd.DataFrame | None = None


class OddsIngestion:
    """Pull and parse MLB game odds from The Odds API."""

    def __init__(self) -> None:
        if not ODDS_API_KEY:
            logger.warning(
                "ODDS_API_KEY not set in .env — "
                "Vegas features will be unavailable"
            )
        logger.debug("OddsIngestion ready")

    def get_mlb_implied_totals(
        self,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Pull implied team totals for today's MLB games.

        Calls ``/sports/baseball_mlb/odds`` with ``markets=totals,h2h,spreads``,
        ``bookmakers=draftkings,fanduel,betmgm``, and a UTC window
        ``commenceTimeFrom`` / ``commenceTimeTo`` (now through +24h) so only
        pre-game / upcoming events are returned.

        Returns a DataFrame with one row per team per game and columns:
            game_id, home_team, away_team, team, is_home, commence_time,
            game_total, home_implied, away_implied, implied_total,
            opposing_implied, moneyline, bookmaker, fetched_at

        Returns an empty DataFrame if the API key is missing, the request
        fails, or parsing yields no rows. Caches a successful non-empty
        result for the session unless ``force_refresh`` is ``True``.

        Args:
            force_refresh: When ``True``, bypass session cache and refetch.
        """
        global _ODDS_SESSION_DF

        if not ODDS_API_KEY:
            logger.warning(
                "get_mlb_implied_totals: no API key — returning empty"
            )
            return pd.DataFrame()

        if force_refresh:
            _ODDS_SESSION_DF = None

        if (
            not force_refresh
            and _ODDS_SESSION_DF is not None
            and not _ODDS_SESSION_DF.empty
        ):
            logger.debug("Returning cached MLB implied totals (session)")
            return _ODDS_SESSION_DF.copy()

        url = f"{ODDS_BASE_URL}/sports/{SPORT}/odds"
        now_utc = datetime.now(timezone.utc)
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": "totals,h2h,spreads",
            "oddsFormat": "american",
            "bookmakers": "draftkings,fanduel,betmgm",
            "commenceTimeFrom": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "commenceTimeTo": (now_utc + timedelta(hours=24)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        }

        try:
            logger.info("Fetching MLB odds from The Odds API...")
            resp = requests.get(url, params=params, timeout=15)

            remaining = resp.headers.get("x-requests-remaining", "?")
            used = resp.headers.get("x-requests-used", "?")
            logger.info(
                f"Odds API quota: {remaining} remaining, {used} used"
            )

            resp.raise_for_status()
            games = resp.json()

            if not games:
                logger.warning("Odds API returned no games")
                return pd.DataFrame()

            rows = self._parse_games(games)

            if not rows:
                logger.warning("No parseable odds data found")
                return pd.DataFrame()

            df = pd.DataFrame(rows)
            # Deduplicate: keep first bookmaker entry per team
            if not df.empty and "team" in df.columns:
                df = df.drop_duplicates(subset=["team"], keep="first")
            logger.info(f"Parsed odds for {len(df)} team-game rows")
            if not df.empty:
                _ODDS_SESSION_DF = df
            return df.copy()

        except requests.exceptions.RequestException as exc:
            logger.error(f"Odds API request failed: {exc}")
            return pd.DataFrame()
        except (TypeError, ValueError, KeyError) as exc:
            logger.error(f"Odds ingestion parse failed: {exc}")
            return pd.DataFrame()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Odds ingestion failed: {exc}")
            return pd.DataFrame()

    def _parse_games(self, games: list[dict]) -> list[dict]:
        """Parse raw Odds API response into team-level rows.

        For each game extracts:

        - ``game_total`` from the totals market (Over point).
        - Home / away moneyline from the h2h market.
        - Home team spread from the spreads market (negative = home favored).

        Implied runs:

        - With home spread ``s`` (home point from API): ``home_implied``
          = ``(game_total - s) / 2``, ``away_implied`` = ``(game_total + s) / 2``.
          Example: total 8.5, home ``-1.5`` → home 5.0, away 3.5.
        - If no total: skip the game (no fallback guess).
        - If no spread: ``s = 0`` → each team implied = ``game_total / 2``.

        Returns one dict per team per game.
        """
        rows: list[dict] = []
        fetched_at = datetime.now(timezone.utc).isoformat()

        for game in games:
            try:
                game_id = game.get("id", "")
                home_team = game.get("home_team", "")
                away_team = game.get("away_team", "")
                commence_time = game.get("commence_time", "")

                bookmakers = game.get("bookmakers") or []
                if not bookmakers:
                    continue

                bookmaker = bookmakers[0]
                bookmaker_key = bookmaker.get("key", "")
                markets = {
                    m.get("key", ""): m
                    for m in (bookmaker.get("markets") or [])
                    if isinstance(m, dict)
                }

                game_total: float | None = None
                totals_market = markets.get("totals") or {}
                for outcome in totals_market.get("outcomes") or []:
                    if not isinstance(outcome, dict):
                        continue
                    if outcome.get("name") == "Over":
                        pt = outcome.get("point")
                        if pt is not None:
                            game_total = float(pt)
                        break

                if game_total is None:
                    logger.debug(
                        f"No total found for {away_team} @ {home_team} — skipping"
                    )
                    continue

                home_spread = 0.0
                spreads_market = markets.get("spreads") or {}
                for outcome in spreads_market.get("outcomes") or []:
                    if not isinstance(outcome, dict):
                        continue
                    if outcome.get("name") == home_team:
                        pt = outcome.get("point")
                        if pt is not None:
                            home_spread = float(pt)
                        break

                if abs(home_spread) > 3.0:
                    logger.warning(
                        f"Extreme spread {home_spread} detected for "
                        f"{home_team} @ {away_team} — "
                        f"likely live odds, defaulting to even split"
                    )
                    home_spread = 0.0

                home_implied = (game_total - home_spread) / 2.0
                away_implied = (game_total + home_spread) / 2.0

                home_implied = max(2.0, min(home_implied, 9.0))
                away_implied = max(2.0, min(away_implied, 9.0))

                home_ml: int | float | None = None
                away_ml: int | float | None = None
                h2h_market = markets.get("h2h") or {}
                for outcome in h2h_market.get("outcomes") or []:
                    if not isinstance(outcome, dict):
                        continue
                    name = outcome.get("name")
                    price = outcome.get("price")
                    if name == home_team:
                        home_ml = price
                    elif name == away_team:
                        away_ml = price

                home_abbr = self._normalize_team(home_team)
                away_abbr = self._normalize_team(away_team)

                rows.append(
                    {
                        "game_id": game_id,
                        "home_team": home_abbr,
                        "away_team": away_abbr,
                        "team": home_abbr,
                        "is_home": True,
                        "commence_time": commence_time,
                        "game_total": game_total,
                        "home_implied": home_implied,
                        "away_implied": away_implied,
                        "implied_total": home_implied,
                        "opposing_implied": away_implied,
                        "home_ml": home_ml,
                        "away_ml": away_ml,
                        "moneyline": home_ml,
                        "bookmaker": bookmaker_key,
                        "fetched_at": fetched_at,
                    }
                )
                rows.append(
                    {
                        "game_id": game_id,
                        "home_team": home_abbr,
                        "away_team": away_abbr,
                        "team": away_abbr,
                        "is_home": False,
                        "commence_time": commence_time,
                        "game_total": game_total,
                        "home_implied": home_implied,
                        "away_implied": away_implied,
                        "implied_total": away_implied,
                        "opposing_implied": home_implied,
                        "home_ml": home_ml,
                        "away_ml": away_ml,
                        "moneyline": away_ml,
                        "bookmaker": bookmaker_key,
                        "fetched_at": fetched_at,
                    }
                )
            except (TypeError, ValueError, KeyError) as exc:
                logger.warning(f"Skipping game parse error: {exc}")
                continue

        return rows

    def _normalize_team(self, full_name: str) -> str:
        """Convert full MLB team name to DK-style abbreviation.

        The Odds API returns full team names like ``Los Angeles Dodgers``.
        DraftKings uses ``LAD``.
        """
        if not full_name:
            return ""

        mapping = {
            "Arizona Diamondbacks": "ARI",
            "Atlanta Braves": "ATL",
            "Baltimore Orioles": "BAL",
            "Boston Red Sox": "BOS",
            "Chicago Cubs": "CHC",
            "Chicago White Sox": "CWS",
            "Cincinnati Reds": "CIN",
            "Cleveland Guardians": "CLE",
            "Colorado Rockies": "COL",
            "Detroit Tigers": "DET",
            "Houston Astros": "HOU",
            "Kansas City Royals": "KC",
            "Los Angeles Angels": "LAA",
            "Los Angeles Dodgers": "LAD",
            "Miami Marlins": "MIA",
            "Milwaukee Brewers": "MIL",
            "Minnesota Twins": "MIN",
            "New York Mets": "NYM",
            "New York Yankees": "NYY",
            "Oakland Athletics": "OAK",
            "Philadelphia Phillies": "PHI",
            "Pittsburgh Pirates": "PIT",
            "San Diego Padres": "SD",
            "San Francisco Giants": "SF",
            "Seattle Mariners": "SEA",
            "St. Louis Cardinals": "STL",
            "Tampa Bay Rays": "TB",
            "Texas Rangers": "TEX",
            "Toronto Blue Jays": "TOR",
            "Washington Nationals": "WSH",
            "Athletics": "OAK",
        }
        return mapping.get(full_name, full_name[:3].upper())

    def get_team_implied_totals(self) -> dict[str, float]:
        """Return mapping of team abbreviation → implied total.

        Convenience helper for the ownership model. Returns ``{}`` if
        odds are unavailable.
        """
        try:
            df = self.get_mlb_implied_totals()
            if df.empty or "team" not in df.columns:
                return {}
            out: dict[str, float] = {}
            # Average across bookmakers per team
            averaged = (
                df.groupby("team")["implied_total"]
                .mean()
                .reset_index()
            )
            for _, row in averaged.iterrows():
                if row["team"] and pd.notna(row["implied_total"]):
                    out[str(row["team"])] = float(row["implied_total"])
            return out
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"get_team_implied_totals: {exc}")
            return {}
