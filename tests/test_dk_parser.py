"""Tests for the DK salary CSV parser and DK lineup exporter."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from data_pipeline.ingestion.dk_csv_parser import (
    DKCSVParser,
    DKPlayer,
    ParseResult,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_dk_salaries.csv"


# ----------------------------------------------------------------------
# Shared fixtures
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def parser() -> DKCSVParser:
    """A parser with the fixture CSV already parsed."""
    p = DKCSVParser()
    p.parse(FIXTURE_PATH)
    return p


@pytest.fixture(scope="module")
def players(parser: DKCSVParser) -> list[DKPlayer]:
    return parser.last_result.players


@pytest.fixture(scope="module")
def by_name(players: list[DKPlayer]) -> dict[str, DKPlayer]:
    return {p.name: p for p in players}


# ----------------------------------------------------------------------
# Basic parsing
# ----------------------------------------------------------------------


def test_parse_returns_correct_player_count(parser: DKCSVParser) -> None:
    result = parser.last_result
    assert result.raw_row_count == 10
    assert result.parsed_row_count == 10
    assert len(result.players) == 10


def test_parse_returns_dkplayer_instances(players: list[DKPlayer]) -> None:
    assert all(isinstance(p, DKPlayer) for p in players)


def test_parse_result_shape(parser: DKCSVParser) -> None:
    result = parser.last_result
    assert isinstance(result, ParseResult)
    assert result.file_hash
    assert "slate_date" in result.slate_info
    assert isinstance(result.validation_errors, list)


# ----------------------------------------------------------------------
# SHA256 file hash
# ----------------------------------------------------------------------


def test_get_file_hash_is_consistent() -> None:
    h1 = DKCSVParser.get_file_hash(FIXTURE_PATH)
    h2 = DKCSVParser.get_file_hash(FIXTURE_PATH)
    assert h1 == h2
    assert len(h1) == 64  # SHA256 hex digest length
    assert all(c in "0123456789abcdef" for c in h1)


def test_parse_populates_file_hash(parser: DKCSVParser) -> None:
    result = parser.last_result
    assert result.file_hash == DKCSVParser.get_file_hash(FIXTURE_PATH)


def test_different_content_gives_different_hash(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_bytes(b"hello")
    b.write_bytes(b"world")
    assert DKCSVParser.get_file_hash(a) != DKCSVParser.get_file_hash(b)


# ----------------------------------------------------------------------
# Game info parsing
# ----------------------------------------------------------------------


def test_game_info_extracts_atl_lad(by_name: dict[str, DKPlayer]) -> None:
    glasnow = by_name["Tyler Glasnow"]
    assert glasnow.away_team == "ATL"
    assert glasnow.home_team == "LAD"
    assert glasnow.game_date == date(2026, 5, 8)
    assert glasnow.game_time_et == "10:10PM ET"
    assert glasnow.game_info_raw == "ATL@LAD 05/08/2026 10:10PM ET"


def test_game_info_extracts_nym_ari(by_name: dict[str, DKPlayer]) -> None:
    soto = by_name["Juan Soto"]
    assert soto.away_team == "NYM"
    assert soto.home_team == "ARI"
    assert soto.game_date == date(2026, 5, 8)
    assert soto.game_time_et == "09:40PM ET"


def test_game_info_extracts_pit_sf(by_name: dict[str, DKPlayer]) -> None:
    cruz = by_name["Oneil Cruz"]
    assert cruz.away_team == "PIT"
    assert cruz.home_team == "SF"


# ----------------------------------------------------------------------
# Position eligibility parsing
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("P", ["P"]),
        ("1B/OF", ["1B", "OF"]),
        ("SS/OF", ["SS", "OF"]),
        ("OF", ["OF"]),
        ("C", ["C"]),
    ],
)
def test_position_eligibility_parses(raw: str, expected: list[str]) -> None:
    parsed = DKCSVParser._parse_position_eligibility(raw)
    assert parsed == expected


def test_position_eligibility_glasnow_pitcher(
    by_name: dict[str, DKPlayer],
) -> None:
    assert by_name["Tyler Glasnow"].position_eligibility == ["P"]


def test_position_eligibility_ohtani_dual(
    by_name: dict[str, DKPlayer],
) -> None:
    ohtani = by_name["Shohei Ohtani"]
    assert ohtani.position_eligibility == ["1B", "OF"]
    assert ohtani.dk_position == "1B/OF"


def test_position_eligibility_dubon_dual(
    by_name: dict[str, DKPlayer],
) -> None:
    dubon = by_name["Mauricio Dubon"]
    assert dubon.position_eligibility == ["SS", "OF"]
    # Note: DK lists Dubon's display Position as "OF/SS" but his
    # Roster Position (eligibility) is "SS/OF".
    assert dubon.dk_position == "OF/SS"


# ----------------------------------------------------------------------
# is_pitcher flag
# ----------------------------------------------------------------------


def test_is_pitcher_for_starting_pitcher(by_name: dict[str, DKPlayer]) -> None:
    assert by_name["Tyler Glasnow"].is_pitcher is True


def test_is_pitcher_for_relief_pitcher(by_name: dict[str, DKPlayer]) -> None:
    assert by_name["Landen Roupp"].is_pitcher is True


def test_is_pitcher_false_for_hitters(by_name: dict[str, DKPlayer]) -> None:
    assert by_name["Juan Soto"].is_pitcher is False
    assert by_name["Daniel Susac"].is_pitcher is False
    assert by_name["Shohei Ohtani"].is_pitcher is False


def test_pitcher_count_matches(parser: DKCSVParser) -> None:
    info = parser.get_slate_info()
    assert info["pitcher_count"] == 2
    assert info["hitter_count"] == 8
    assert info["player_count"] == 10


# ----------------------------------------------------------------------
# Salary parsing
# ----------------------------------------------------------------------


def test_salary_parsed_as_int(by_name: dict[str, DKPlayer]) -> None:
    glasnow = by_name["Tyler Glasnow"]
    assert isinstance(glasnow.salary, int)
    assert glasnow.salary == 10200


def test_salary_for_each_player(by_name: dict[str, DKPlayer]) -> None:
    expected = {
        "Tyler Glasnow": 10200,
        "Landen Roupp": 8800,
        "Shohei Ohtani": 6400,
        "Mauricio Dubon": 3000,
        "Juan Soto": 6300,
        "Oneil Cruz": 6200,
        "Daniel Susac": 2800,
        "Gavin Sheets": 3000,
        "Jake Cronenworth": 2200,
        "Ramon Urias": 3100,
    }
    for name, salary in expected.items():
        assert by_name[name].salary == salary, name


# ----------------------------------------------------------------------
# avg_points_per_game
# ----------------------------------------------------------------------


def test_avg_points_per_game_parsed_as_float(
    by_name: dict[str, DKPlayer],
) -> None:
    assert by_name["Tyler Glasnow"].avg_points_per_game == pytest.approx(22.21)
    assert by_name["Ramon Urias"].avg_points_per_game == pytest.approx(3.8)


def test_avg_points_handles_zero(tmp_path: Path) -> None:
    """A row with AvgPointsPerGame of 0 must parse as 0.0 without error."""
    csv_path = tmp_path / "zero_pts.csv"
    csv_path.write_text(
        "Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,"
        "TeamAbbrev,AvgPointsPerGame\n"
        "OF,Rookie Player (99999999),Rookie Player,99999999,OF,2000,"
        "ATL@LAD 05/08/2026 10:10PM ET,LAD,0\n",
        encoding="utf-8",
    )
    p = DKCSVParser()
    players = p.parse(csv_path)
    assert len(players) == 1
    assert players[0].avg_points_per_game == 0.0
    assert all(
        e["severity"] != "error" for e in p.last_result.validation_errors
    )


def test_avg_points_handles_blank(tmp_path: Path) -> None:
    """A row with a blank AvgPointsPerGame must default to 0.0."""
    csv_path = tmp_path / "blank_pts.csv"
    csv_path.write_text(
        "Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,"
        "TeamAbbrev,AvgPointsPerGame\n"
        "OF,Blank Player (99999998),Blank Player,99999998,OF,2000,"
        "ATL@LAD 05/08/2026 10:10PM ET,LAD,\n",
        encoding="utf-8",
    )
    p = DKCSVParser()
    players = p.parse(csv_path)
    assert len(players) == 1
    assert players[0].avg_points_per_game == 0.0


# ----------------------------------------------------------------------
# Validation errors — must not crash
# ----------------------------------------------------------------------


def test_bad_row_does_not_crash(tmp_path: Path) -> None:
    """Malformed rows should be collected as validation errors, not raised."""
    csv_path = tmp_path / "bad_rows.csv"
    csv_path.write_text(
        "Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,"
        "TeamAbbrev,AvgPointsPerGame\n"
        # Good row.
        "OF,Good Player (1),Good Player,1,OF,3000,"
        "ATL@LAD 05/08/2026 10:10PM ET,LAD,5.0\n"
        # Bad salary.
        "OF,Bad Salary (2),Bad Salary,2,OF,not_a_number,"
        "ATL@LAD 05/08/2026 10:10PM ET,LAD,5.0\n"
        # Bad game info.
        "OF,Bad Game (3),Bad Game,3,OF,3000,total garbage,LAD,5.0\n"
        # Missing roster position.
        "OF,No Pos (4),No Pos,4,,3000,"
        "ATL@LAD 05/08/2026 10:10PM ET,LAD,5.0\n",
        encoding="utf-8",
    )
    p = DKCSVParser()
    players = p.parse(csv_path)
    result = p.last_result

    assert result.raw_row_count == 4
    assert result.parsed_row_count == 1
    assert len(players) == 1
    assert players[0].name == "Good Player"
    assert len(result.validation_errors) >= 3

    fields_with_errors = {e["field"] for e in result.validation_errors}
    assert "Salary" in fields_with_errors
    assert "Game Info" in fields_with_errors
    assert "Roster Position" in fields_with_errors


def test_clean_fixture_has_no_errors(parser: DKCSVParser) -> None:
    """The well-formed fixture must produce zero validation errors."""
    assert parser.last_result.validation_errors == []


def test_missing_file_raises(tmp_path: Path) -> None:
    p = DKCSVParser()
    with pytest.raises(FileNotFoundError):
        p.parse(tmp_path / "does_not_exist.csv")


# ----------------------------------------------------------------------
# Slate info
# ----------------------------------------------------------------------


def test_slate_info_summary(parser: DKCSVParser) -> None:
    info = parser.get_slate_info()
    assert info["slate_date"] == date(2026, 5, 8)
    assert info["player_count"] == 10
    assert info["pitcher_count"] == 2
    assert info["hitter_count"] == 8

    assert "ATL@LAD 05/08/2026 10:10PM ET" in info["games"]
    assert "PIT@SF 05/08/2026 10:15PM ET" in info["games"]
    assert "NYM@ARI 05/08/2026 09:40PM ET" in info["games"]
    assert "STL@SD 05/08/2026 09:45PM ET" in info["games"]
    assert len(info["games"]) == 4

    teams = set(info["teams"])
    assert {"LAD", "SF", "NYM", "PIT", "SD", "STL", "ATL"}.issubset(teams)
