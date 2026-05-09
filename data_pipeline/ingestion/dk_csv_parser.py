"""DraftKings salary CSV parser.

Parses the raw DraftKings salary export (9 columns) into structured
``DKPlayer`` dataclasses. Designed to be resilient: bad rows are
collected as validation errors rather than raising, so a single
malformed row never blocks an entire slate ingestion.

Expected DK salary CSV columns (in order):
    Position, Name + ID, Name, ID, Roster Position, Salary,
    Game Info, TeamAbbrev, AvgPointsPerGame
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


# DK salary CSV is fixed at these 9 columns. Any deviation is a
# critical validation error — we still attempt to parse, but the
# caller is informed.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "Position",
    "Name + ID",
    "Name",
    "ID",
    "Roster Position",
    "Salary",
    "Game Info",
    "TeamAbbrev",
    "AvgPointsPerGame",
)

# Game Info format: "ATL@LAD 05/08/2026 10:10PM ET"
# Captures: away, home, MM/DD/YYYY date, "10:10PM ET" time string.
_GAME_INFO_RE = re.compile(
    r"^\s*([A-Z]{2,4})\s*@\s*([A-Z]{2,4})\s+"
    r"(\d{1,2}/\d{1,2}/\d{4})\s+"
    r"(\d{1,2}:\d{2}\s*(?:AM|PM)\s*[A-Z]{2,3})\s*$"
)


@dataclass
class DKPlayer:
    """A single player parsed from the DK salary CSV."""

    dk_id: str
    name: str
    dk_position: str
    position_eligibility: list[str]
    salary: int
    game_info_raw: str
    away_team: str
    home_team: str
    game_date: date
    game_time_et: str
    team: str
    avg_points_per_game: float
    is_pitcher: bool
    roster_position: str | None = None  # Filled in by the optimizer.


@dataclass
class ParseResult:
    """Container for the output of a DK salary CSV parse."""

    players: list[DKPlayer]
    file_hash: str
    slate_info: dict[str, Any]
    validation_errors: list[dict[str, Any]] = field(default_factory=list)
    raw_row_count: int = 0
    parsed_row_count: int = 0


class DKCSVParser:
    """Parses a DraftKings salary CSV into ``DKPlayer`` records.

    Usage:
        parser = DKCSVParser()
        players = parser.parse("DKSalaries.csv")
        info = parser.get_slate_info()
        result = parser.last_result  # ParseResult with errors + hash
    """

    def __init__(self) -> None:
        self._players: list[DKPlayer] = []
        self._validation_errors: list[dict[str, Any]] = []
        self._file_hash: str | None = None
        self._raw_row_count: int = 0
        self._parsed_row_count: int = 0

    @property
    def last_result(self) -> ParseResult:
        """Return a ``ParseResult`` snapshot of the most recent parse."""
        return ParseResult(
            players=list(self._players),
            file_hash=self._file_hash or "",
            slate_info=self.get_slate_info(),
            validation_errors=list(self._validation_errors),
            raw_row_count=self._raw_row_count,
            parsed_row_count=self._parsed_row_count,
        )

    @staticmethod
    def get_file_hash(file_path: str | Path) -> str:
        """Return the SHA256 hex digest of the raw bytes of ``file_path``."""
        path = Path(file_path)
        hasher = hashlib.sha256()
        # Stream in chunks so very large CSVs don't blow up memory.
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def parse(self, file_path: str | Path) -> list[DKPlayer]:
        """Parse a DK salary CSV file into a list of ``DKPlayer``.

        The parser collects validation errors rather than raising on
        bad rows; inspect ``last_result.validation_errors`` to review.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"DK salary CSV not found: {path}")

        self._players = []
        self._validation_errors = []
        self._raw_row_count = 0
        self._parsed_row_count = 0
        self._file_hash = self.get_file_hash(path)
        logger.info(f"Parsing DK CSV {path} (sha256={self._file_hash[:12]}…)")

        df = pd.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)
        self._raw_row_count = len(df)

        self._validate_columns(df)

        for idx, row in df.iterrows():
            row_num = int(idx) + 2  # +1 for 0-index, +1 for header row
            try:
                player = self._parse_row(row, row_num)
            except Exception as exc:  # noqa: BLE001 - intentionally broad
                logger.warning(f"Row {row_num} failed to parse: {exc}")
                self._validation_errors.append(
                    {
                        "row": row_num,
                        "field": "row",
                        "message": f"Unhandled error parsing row: {exc}",
                        "severity": "error",
                    }
                )
                continue

            if player is None:
                continue

            self._players.append(player)
            self._parsed_row_count += 1

        logger.info(
            f"Parsed {self._parsed_row_count}/{self._raw_row_count} rows; "
            f"{len(self._validation_errors)} validation errors"
        )
        return list(self._players)

    def get_slate_info(self) -> dict[str, Any]:
        """Return summary metadata about the most recent parse."""
        games = sorted({p.game_info_raw for p in self._players})
        teams = sorted({p.team for p in self._players})
        slate_dates = sorted({p.game_date for p in self._players})
        pitcher_count = sum(1 for p in self._players if p.is_pitcher)
        hitter_count = len(self._players) - pitcher_count

        return {
            "slate_date": slate_dates[0] if slate_dates else None,
            "games": games,
            "teams": teams,
            "player_count": len(self._players),
            "pitcher_count": pitcher_count,
            "hitter_count": hitter_count,
        }

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Record a critical error if the CSV header doesn't match DK's format."""
        actual = tuple(df.columns)
        if actual != EXPECTED_COLUMNS:
            self._validation_errors.append(
                {
                    "row": 1,
                    "field": "header",
                    "message": (
                        f"Unexpected DK CSV columns. "
                        f"Expected {EXPECTED_COLUMNS}, got {actual}"
                    ),
                    "severity": "critical",
                }
            )

    def _parse_row(self, row: pd.Series, row_num: int) -> DKPlayer | None:
        """Parse a single DataFrame row into a ``DKPlayer``.

        Returns ``None`` if the row has unrecoverable errors.
        """
        dk_id = self._clean_str(row.get("ID", ""))
        name = self._clean_str(row.get("Name", ""))
        dk_position = self._clean_str(row.get("Position", ""))
        roster_position_raw = self._clean_str(row.get("Roster Position", ""))
        salary_raw = self._clean_str(row.get("Salary", ""))
        game_info_raw = self._clean_str(row.get("Game Info", ""))
        team = self._clean_str(row.get("TeamAbbrev", ""))
        avg_pts_raw = self._clean_str(row.get("AvgPointsPerGame", ""))

        if not dk_id:
            self._record_error(row_num, "ID", "missing dk_id", "error")
            return None
        if not name:
            self._record_error(row_num, "Name", "missing name", "error")
            return None
        if not roster_position_raw:
            self._record_error(
                row_num, "Roster Position", "missing roster position", "error"
            )
            return None

        salary = self._parse_salary(salary_raw, row_num)
        if salary is None:
            return None

        position_eligibility = self._parse_position_eligibility(roster_position_raw)
        if not position_eligibility:
            self._record_error(
                row_num,
                "Roster Position",
                f"could not parse position eligibility: {roster_position_raw!r}",
                "error",
            )
            return None

        is_pitcher = "P" in position_eligibility

        away_team, home_team, game_date, game_time_et = self._parse_game_info(
            game_info_raw, row_num
        )
        if game_date is None:
            return None

        avg_pts = self._parse_avg_points(avg_pts_raw, row_num)

        return DKPlayer(
            dk_id=dk_id,
            name=name,
            dk_position=dk_position,
            position_eligibility=position_eligibility,
            salary=salary,
            game_info_raw=game_info_raw,
            away_team=away_team,
            home_team=home_team,
            game_date=game_date,
            game_time_et=game_time_et,
            team=team,
            avg_points_per_game=avg_pts,
            is_pitcher=is_pitcher,
        )

    def _parse_salary(self, raw: str, row_num: int) -> int | None:
        """Parse the Salary column to int; tolerant of commas."""
        if not raw:
            self._record_error(row_num, "Salary", "missing salary", "error")
            return None
        try:
            return int(raw.replace(",", "").strip())
        except ValueError:
            self._record_error(
                row_num, "Salary", f"non-integer salary: {raw!r}", "error"
            )
            return None

    @staticmethod
    def _parse_position_eligibility(raw: str) -> list[str]:
        """Split DK Roster Position string ("1B/OF") into a list."""
        return [p.strip() for p in raw.split("/") if p.strip()]

    def _parse_game_info(
        self, raw: str, row_num: int
    ) -> tuple[str, str, date | None, str]:
        """Parse "ATL@LAD 05/08/2026 10:10PM ET" → (away, home, date, time_str)."""
        if not raw:
            self._record_error(row_num, "Game Info", "missing game info", "error")
            return ("", "", None, "")

        match = _GAME_INFO_RE.match(raw)
        if not match:
            self._record_error(
                row_num,
                "Game Info",
                f"unrecognized game info format: {raw!r}",
                "error",
            )
            return ("", "", None, "")

        away, home, date_str, time_str = match.groups()
        try:
            game_date = datetime.strptime(date_str, "%m/%d/%Y").date()
        except ValueError:
            self._record_error(
                row_num, "Game Info", f"invalid date {date_str!r}", "error"
            )
            return (away, home, None, time_str)

        # Normalize whitespace inside the time string (e.g. "10:10PM  ET").
        normalized_time = re.sub(r"\s+", " ", time_str.strip())
        return (away, home, game_date, normalized_time)

    def _parse_avg_points(self, raw: str, row_num: int) -> float:
        """Parse AvgPointsPerGame to float; missing/blank → 0.0."""
        if not raw:
            return 0.0
        try:
            return float(raw)
        except ValueError:
            self._record_error(
                row_num,
                "AvgPointsPerGame",
                f"non-numeric avg points: {raw!r} (defaulting to 0.0)",
                "warning",
            )
            return 0.0

    @staticmethod
    def _clean_str(value: Any) -> str:
        """Coerce a CSV cell to a stripped string."""
        if value is None:
            return ""
        return str(value).strip()

    def _record_error(
        self, row_num: int, field_name: str, message: str, severity: str
    ) -> None:
        """Append a structured validation error and log it."""
        logger.warning(f"Row {row_num} [{field_name}] {severity}: {message}")
        self._validation_errors.append(
            {
                "row": row_num,
                "field": field_name,
                "message": message,
                "severity": severity,
            }
        )
