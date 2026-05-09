"""Tests for the lineup validator and bankroll safety checker."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pytest

from data_pipeline.ingestion.dk_csv_parser import DKPlayer
from optimizer.constraints.bankroll_safety import (
    BankrollSafetyChecker,
    BankrollSafetyResult,
)
from optimizer.constraints.lineup_validator import (
    SALARY_CAP,
    SALARY_FLOOR,
    LineupValidator,
    RosterSlot,
    ValidatedLineup,
    ValidationError,
)


# ----------------------------------------------------------------------
# Helpers / fixtures
# ----------------------------------------------------------------------


def _game_info(away: str, home: str) -> str:
    return f"{away}@{home} 05/08/2026 10:10PM ET"


def make_player(
    name: str,
    dk_id: str,
    salary: int,
    position_eligibility: list[str],
    team: str,
    away_team: str,
    home_team: str,
    lineup_status: str = "confirmed_starting",
    avg_points_per_game: float = 5.0,
) -> DKPlayer:
    """Build a DKPlayer with sensible defaults for tests.

    ``lineup_status`` is attached as a runtime attribute since the
    parser dataclass doesn't carry it as a formal field yet — the
    validator reads it via ``getattr``.
    """
    is_pitcher = "P" in position_eligibility and len(position_eligibility) == 1
    primary = position_eligibility[0]
    player = DKPlayer(
        dk_id=dk_id,
        name=name,
        dk_position=primary,
        position_eligibility=list(position_eligibility),
        salary=salary,
        game_info_raw=_game_info(away_team, home_team),
        away_team=away_team,
        home_team=home_team,
        game_date=date(2026, 5, 8),
        game_time_et="10:10PM ET",
        team=team,
        avg_points_per_game=avg_points_per_game,
        is_pitcher=is_pitcher,
    )
    # lineup_status is not a formal DKPlayer field; attach it here so
    # _get_status can pick it up via getattr.
    player.lineup_status = lineup_status  # type: ignore[attr-defined]
    return player


def make_valid_lineup() -> list[tuple[DKPlayer, str]]:
    """Build a fully valid 10-player lineup spanning two games.

    Games used:
        * ATL @ LAD
        * NYM @ ARI

    Salary ladder is tuned so the total sits exactly at the cap
    ($50,000) and well above the floor ($47,000):

        P    9500  (pitcher 1)
        P    4000  (pitcher 2)
        C    4500
        1B   5000
        2B   4500
        3B   4500
        SS   5000
        OF   4500
        OF   4500
        OF   4000
        ----------
             50000

    Index map (used by several tests):
        0=P1  1=P2  2=C  3=1B  4=2B  5=3B  6=SS  7=OF1  8=OF2  9=OF3
    """
    pitcher1 = make_player(
        "Pitcher One", "p1", 9500, ["P"], "LAD", "ATL", "LAD"
    )
    pitcher2 = make_player(
        "Pitcher Two", "p2", 4000, ["P"], "ATL", "ATL", "LAD"
    )
    catcher = make_player(
        "Catcher One", "c1", 4500, ["C"], "ATL", "ATL", "LAD"
    )
    first_base = make_player(
        "First One", "1b1", 5000, ["1B"], "NYM", "NYM", "ARI"
    )
    second_base = make_player(
        "Second One", "2b1", 4500, ["2B"], "ARI", "NYM", "ARI"
    )
    third_base = make_player(
        "Third One", "3b1", 4500, ["3B"], "LAD", "ATL", "LAD"
    )
    shortstop = make_player(
        "SS One", "ss1", 5000, ["SS"], "ATL", "ATL", "LAD"
    )
    outfield_1 = make_player(
        "OF One", "of1", 4500, ["OF"], "NYM", "NYM", "ARI"
    )
    outfield_2 = make_player(
        "OF Two", "of2", 4500, ["OF"], "ARI", "NYM", "ARI"
    )
    outfield_3 = make_player(
        "OF Three", "of3", 4000, ["OF"], "LAD", "ATL", "LAD"
    )

    return [
        (pitcher1,    RosterSlot.P.value),           # index 0
        (pitcher2,    RosterSlot.P.value),           # index 1
        (catcher,     RosterSlot.C.value),           # index 2
        (first_base,  RosterSlot.FIRST_BASE.value),  # index 3
        (second_base, RosterSlot.SECOND_BASE.value), # index 4
        (third_base,  RosterSlot.THIRD_BASE.value),  # index 5
        (shortstop,   RosterSlot.SS.value),          # index 6
        (outfield_1,  RosterSlot.OF.value),          # index 7
        (outfield_2,  RosterSlot.OF.value),          # index 8
        (outfield_3,  RosterSlot.OF.value),          # index 9
    ]


def _has_rule(errors: list[ValidationError], rule: str) -> bool:
    return any(e.rule == rule for e in errors)


# ----------------------------------------------------------------------
# Sanity check on the helpers themselves
# ----------------------------------------------------------------------


def test_make_valid_lineup_is_actually_valid() -> None:
    lineup = make_valid_lineup()
    assert len(lineup) == 10
    total = sum(p.salary for p, _ in lineup)
    assert SALARY_FLOOR <= total <= SALARY_CAP
    assert total == 50000

    games = {(p.away_team, p.home_team) for p, _ in lineup}
    assert len(games) >= 2

    dk_ids = [p.dk_id for p, _ in lineup]
    assert len(dk_ids) == len(set(dk_ids))


def test_valid_lineup_passes_validator() -> None:
    validator = LineupValidator()
    result = validator.validate(make_valid_lineup())
    assert result.is_valid is True
    assert result.errors == []
    assert result.total_salary == 50000


def test_validatedlineup_is_dataclass_instance() -> None:
    validator = LineupValidator()
    result = validator.validate(make_valid_lineup())
    assert isinstance(result, ValidatedLineup)


# ----------------------------------------------------------------------
# Player count
# ----------------------------------------------------------------------


def test_nine_players_fails_player_count() -> None:
    lineup = make_valid_lineup()[:-1]
    result = LineupValidator().validate(lineup)
    assert result.is_valid is False
    assert _has_rule(result.errors, "player_count")


def test_eleven_players_fails_player_count() -> None:
    lineup = make_valid_lineup()
    extra = make_player(
        "Extra Player", "extra1", 3500, ["OF"], "NYM", "NYM", "ARI"
    )
    lineup.append((extra, RosterSlot.OF.value))
    result = LineupValidator().validate(lineup)
    assert result.is_valid is False
    assert _has_rule(result.errors, "player_count")


# ----------------------------------------------------------------------
# Duplicates
# ----------------------------------------------------------------------


def test_duplicate_player_fails() -> None:
    lineup = make_valid_lineup()
    # Replace the last OF with the same player as the first OF.
    of_player = lineup[7][0]  # OF One (index 7)
    lineup[-1] = (of_player, RosterSlot.OF.value)
    result = LineupValidator().validate(lineup)
    assert result.is_valid is False
    assert _has_rule(result.errors, "no_duplicates")


# ----------------------------------------------------------------------
# Salary cap / floor
# ----------------------------------------------------------------------


def test_salary_cap_exceeded_fails() -> None:
    lineup = make_valid_lineup()
    # Bump the pitcher's salary so total > 50000.
    expensive = make_player(
        "Expensive Pitcher", "p_expensive", 12000, ["P"], "LAD", "ATL", "LAD"
    )
    lineup[0] = (expensive, RosterSlot.P.value)
    result = LineupValidator().validate(lineup)
    assert result.is_valid is False
    assert _has_rule(result.errors, "salary_cap")
    assert result.total_salary > SALARY_CAP


def test_salary_below_floor_fails() -> None:
    lineup = make_valid_lineup()
    # Replace pitcher1 (9500→4000) and OF One (4500→2500) to drop below 47k.
    # Total drops from 50000 to 42500 (< 47000 floor).
    cheap_pitcher = make_player(
        "Cheap Pitcher", "p_cheap", 4000, ["P"], "LAD", "ATL", "LAD"
    )
    cheap_of = make_player(
        "Cheap OF", "of_cheap", 2500, ["OF"], "NYM", "NYM", "ARI"
    )
    lineup[0] = (cheap_pitcher, RosterSlot.P.value)
    lineup[7] = (cheap_of, RosterSlot.OF.value)  # index 7 = OF One
    result = LineupValidator().validate(lineup)
    assert result.is_valid is False
    assert _has_rule(result.errors, "salary_floor")


# ----------------------------------------------------------------------
# Position slots / eligibility
# ----------------------------------------------------------------------


def test_wrong_position_in_slot_fails() -> None:
    """An OF-only player assigned to 1B is ineligible."""
    lineup = make_valid_lineup()
    of_only = make_player(
        "OF Only", "of_only", 5000, ["OF"], "NYM", "NYM", "ARI"
    )
    lineup[2] = (of_only, RosterSlot.FIRST_BASE.value)
    result = LineupValidator().validate(lineup)
    assert result.is_valid is False
    assert _has_rule(result.errors, "position_eligibility")


def test_dual_eligible_player_fills_either_slot() -> None:
    """A 1B/OF player should be valid in both 1B and OF slot assignments."""
    base = make_valid_lineup()

    # Variant A: dual (1B/OF, same salary as original 1B) fills the 1B slot.
    # Swapping a 5000 1B for a 5000 dual — salary unchanged → total stays 50000.
    dual_a = make_player(
        "Ohtani-like", "dual1", 5000, ["1B", "OF"], "LAD", "ATL", "LAD"
    )
    variant_a = list(base)
    variant_a[3] = (dual_a, RosterSlot.FIRST_BASE.value)  # index 3 = 1B
    result_a = LineupValidator().validate(variant_a)
    assert result_a.is_valid is True, result_a.errors

    # Variant B: dual (1B/OF, same salary as original OF1) fills an OF slot.
    # Swapping a 4500 OF for a 4500 dual — salary unchanged → total stays 50000.
    dual_b = make_player(
        "Ohtani-like B", "dual1b", 4500, ["1B", "OF"], "LAD", "ATL", "LAD"
    )
    variant_b = list(base)
    variant_b[7] = (dual_b, RosterSlot.OF.value)  # index 7 = OF One
    result_b = LineupValidator().validate(variant_b)
    assert result_b.is_valid is True, result_b.errors


def test_missing_required_slot_fails() -> None:
    """Replacing 3B with an extra OF should fire a position_slots error."""
    lineup = make_valid_lineup()
    # Replace 3B (index 5) with an OF-only player at the OF slot.
    # This leaves the 3B slot unfilled and adds a 4th OF, both wrong.
    extra_of = make_player(
        "Extra OF", "of_extra", 5000, ["OF"], "ATL", "ATL", "LAD"
    )
    lineup[5] = (extra_of, RosterSlot.OF.value)
    result = LineupValidator().validate(lineup)
    assert result.is_valid is False
    assert _has_rule(result.errors, "position_slots")


# ----------------------------------------------------------------------
# Min games
# ----------------------------------------------------------------------


def test_all_players_same_game_fails() -> None:
    """Stack all 10 players in the ATL@LAD game → min_games error."""
    players_one_game = [
        make_player("P1",   "p1",  9500, ["P"],  "LAD", "ATL", "LAD"),
        make_player("P2",   "p2",  4000, ["P"],  "ATL", "ATL", "LAD"),
        make_player("C1",   "c1",  4500, ["C"],  "ATL", "ATL", "LAD"),
        make_player("F1",   "1b1", 5000, ["1B"], "LAD", "ATL", "LAD"),
        make_player("Sec1", "2b1", 4500, ["2B"], "ATL", "ATL", "LAD"),
        make_player("T1",   "3b1", 4500, ["3B"], "LAD", "ATL", "LAD"),
        make_player("S1",   "ss1", 5000, ["SS"], "ATL", "ATL", "LAD"),
        make_player("O1",   "of1", 4500, ["OF"], "LAD", "ATL", "LAD"),
        make_player("O2",   "of2", 4500, ["OF"], "ATL", "ATL", "LAD"),
        make_player("O3",   "of3", 4000, ["OF"], "LAD", "ATL", "LAD"),
    ]
    slots = [
        RosterSlot.P.value, RosterSlot.P.value,
        RosterSlot.C.value,
        RosterSlot.FIRST_BASE.value, RosterSlot.SECOND_BASE.value,
        RosterSlot.THIRD_BASE.value, RosterSlot.SS.value,
        RosterSlot.OF.value, RosterSlot.OF.value, RosterSlot.OF.value,
    ]
    lineup = list(zip(players_one_game, slots))
    result = LineupValidator().validate(lineup)
    assert result.is_valid is False
    assert _has_rule(result.errors, "min_games")


# ----------------------------------------------------------------------
# Lineup status (scratched / unknown / projected_starting)
# ----------------------------------------------------------------------


def test_scratched_player_fails() -> None:
    lineup = make_valid_lineup()
    scratched_of = make_player(
        "Scratched OF", "of_scratch", 4500, ["OF"], "NYM",
        "NYM", "ARI", lineup_status="scratched",
    )
    lineup[7] = (scratched_of, RosterSlot.OF.value)
    result = LineupValidator().validate(lineup)
    assert result.is_valid is False
    assert _has_rule(result.errors, "no_scratched")


def test_unknown_status_emits_warning_not_error() -> None:
    lineup = make_valid_lineup()
    unknown = make_player(
        "Unknown Player", "of_unknown", 4500, ["OF"], "NYM",
        "NYM", "ARI", lineup_status="unknown",
    )
    lineup[7] = (unknown, RosterSlot.OF.value)
    result = LineupValidator().validate(lineup)
    assert result.is_valid is True
    assert any(
        w.rule == "lineup_status_unconfirmed" for w in result.warnings
    )


def test_projected_starting_emits_warning() -> None:
    lineup = make_valid_lineup()
    projected = make_player(
        "Projected Player", "of_projected", 4500, ["OF"], "NYM",
        "NYM", "ARI", lineup_status="projected_starting",
    )
    lineup[7] = (projected, RosterSlot.OF.value)
    result = LineupValidator().validate(lineup)
    assert result.is_valid is True
    assert any(
        w.rule == "lineup_status_unconfirmed" for w in result.warnings
    )


# ----------------------------------------------------------------------
# Banned / locked
# ----------------------------------------------------------------------


@dataclass
class _Constraint:
    """Minimal stand-in for a player_constraints row."""
    is_banned: bool = False
    is_locked: bool = False
    name: str | None = None


def test_banned_player_fails() -> None:
    lineup = make_valid_lineup()
    banned_dk_id = lineup[0][0].dk_id
    constraints = {banned_dk_id: _Constraint(is_banned=True)}
    result = LineupValidator(constraints).validate(lineup)
    assert result.is_valid is False
    assert _has_rule(result.errors, "no_banned")


def test_locked_player_missing_fails() -> None:
    lineup = make_valid_lineup()
    constraints = {
        "must_be_in_lineup": _Constraint(
            is_locked=True, name="Must Lock Player"
        )
    }
    result = LineupValidator(constraints).validate(lineup)
    assert result.is_valid is False
    assert _has_rule(result.errors, "locked_player_missing")


def test_locked_player_present_passes() -> None:
    lineup = make_valid_lineup()
    locked_dk_id = lineup[0][0].dk_id
    constraints = {locked_dk_id: _Constraint(is_locked=True, name="Pitcher One")}
    result = LineupValidator(constraints).validate(lineup)
    assert result.is_valid is True
    assert not _has_rule(result.errors, "locked_player_missing")


# ----------------------------------------------------------------------
# Batch validation
# ----------------------------------------------------------------------


def test_validate_batch_counts_correctly() -> None:
    valid_a = make_valid_lineup()
    valid_b = make_valid_lineup()  # Same shape; still valid.

    invalid = make_valid_lineup()[:-1]  # 9 players → invalid.

    results = LineupValidator().validate_batch([valid_a, valid_b, invalid])
    assert len(results) == 3
    assert sum(1 for r in results if r.is_valid) == 2
    assert sum(1 for r in results if not r.is_valid) == 1


def test_validate_batch_returns_validatedlineup_instances() -> None:
    results = LineupValidator().validate_batch(
        [make_valid_lineup(), make_valid_lineup()]
    )
    assert all(isinstance(r, ValidatedLineup) for r in results)


# ----------------------------------------------------------------------
# Total salary calculation
# ----------------------------------------------------------------------


def test_total_salary_calculated_correctly() -> None:
    lineup = make_valid_lineup()
    expected = sum(p.salary for p, _ in lineup)
    result = LineupValidator().validate(lineup)
    assert result.total_salary == expected
    assert result.total_salary == 50000


# ----------------------------------------------------------------------
# Bankroll safety checker
# ----------------------------------------------------------------------


def _validated(*lineups: list[tuple[DKPlayer, str]]) -> list[ValidatedLineup]:
    return LineupValidator().validate_batch(list(lineups))


def test_bankroll_blocks_with_fewer_than_20_lineups() -> None:
    validated = _validated(*[make_valid_lineup() for _ in range(5)])
    result = BankrollSafetyChecker().pre_export_check(validated)
    assert isinstance(result, BankrollSafetyResult)
    assert result.safe_to_export is False
    assert any("valid lineups" in msg for msg in result.blocking_issues)


def test_bankroll_passes_with_20_valid_lineups() -> None:
    validated = _validated(*[make_valid_lineup() for _ in range(20)])
    result = BankrollSafetyChecker().pre_export_check(validated)
    assert result.safe_to_export is True
    assert result.blocking_issues == []
    assert result.lineup_count == 20
    assert result.valid_lineup_count == 20


def test_bankroll_blocks_with_scratched_player() -> None:
    """A single scratched player anywhere in the batch must block export."""
    lineups: list[list[tuple[DKPlayer, str]]] = []
    for i in range(20):
        lineup = make_valid_lineup()
        if i == 0:
            scratched = make_player(
                f"Scratched OF {i}", f"scratched_{i}", 4500, ["OF"],
                "NYM", "NYM", "ARI", lineup_status="scratched",
            )
            lineup[7] = (scratched, RosterSlot.OF.value)
        lineups.append(lineup)

    validated = _validated(*lineups)
    result = BankrollSafetyChecker().pre_export_check(validated)
    assert result.safe_to_export is False
    assert any("scratched" in msg for msg in result.blocking_issues)


def test_bankroll_warns_on_projected_starting_player() -> None:
    """20 valid lineups, but one contains a projected_starting player."""
    lineups: list[list[tuple[DKPlayer, str]]] = []
    for i in range(20):
        lineup = make_valid_lineup()
        if i == 0:
            projected = make_player(
                f"Projected OF {i}", f"proj_{i}", 4500, ["OF"],
                "NYM", "NYM", "ARI", lineup_status="projected_starting",
            )
            lineup[7] = (projected, RosterSlot.OF.value)
        lineups.append(lineup)

    validated = _validated(*lineups)
    result = BankrollSafetyChecker().pre_export_check(validated)
    assert result.safe_to_export is True
    assert any("projected_starting" in w for w in result.warnings)


def test_bankroll_blocks_unknown_player_within_30min_of_lock() -> None:
    """Unknown-status player + lock < 30min → block."""
    lineups: list[list[tuple[DKPlayer, str]]] = []
    for i in range(20):
        lineup = make_valid_lineup()
        if i == 0:
            unknown = make_player(
                f"Unknown OF {i}", f"unk_{i}", 4500, ["OF"],
                "NYM", "NYM", "ARI", lineup_status="unknown",
            )
            lineup[7] = (unknown, RosterSlot.OF.value)
        lineups.append(lineup)

    validated = _validated(*lineups)
    result = BankrollSafetyChecker().pre_export_check(
        validated, minutes_to_lock=15
    )
    assert result.safe_to_export is False
    assert any("unknown" in msg for msg in result.blocking_issues)


def test_bankroll_does_not_block_unknown_when_lock_is_far_away() -> None:
    """Unknown-status player + lock > 30min → no block on that ground."""
    lineups: list[list[tuple[DKPlayer, str]]] = []
    for i in range(20):
        lineup = make_valid_lineup()
        if i == 0:
            unknown = make_player(
                f"Unknown OF {i}", f"unk_{i}", 4500, ["OF"],
                "NYM", "NYM", "ARI", lineup_status="unknown",
            )
            lineup[7] = (unknown, RosterSlot.OF.value)
        lineups.append(lineup)

    validated = _validated(*lineups)
    result = BankrollSafetyChecker().pre_export_check(
        validated, minutes_to_lock=120
    )
    assert result.safe_to_export is True
    assert not any("unknown" in msg for msg in result.blocking_issues)


def test_bankroll_resolves_minutes_from_lock_time() -> None:
    """When only lock_time is given, minutes_to_lock should be derived."""
    lineups: list[list[tuple[DKPlayer, str]]] = []
    for i in range(20):
        lineup = make_valid_lineup()
        if i == 0:
            unknown = make_player(
                f"Unknown OF {i}", f"unk_{i}", 4500, ["OF"],
                "NYM", "NYM", "ARI", lineup_status="unknown",
            )
            lineup[7] = (unknown, RosterSlot.OF.value)
        lineups.append(lineup)

    validated = _validated(*lineups)
    soon_lock = datetime.now(timezone.utc) + timedelta(minutes=10)
    result = BankrollSafetyChecker().pre_export_check(
        validated, lock_time=soon_lock
    )
    assert result.safe_to_export is False
