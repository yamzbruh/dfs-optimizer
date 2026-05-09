"""Lineup validity engine for DraftKings MLB classic GPP slates.

Every lineup the optimizer produces must pass through ``LineupValidator``
before it can be exported. The validator runs a fixed set of rules,
collects every violation it finds (it never short-circuits), and
reports back a ``ValidatedLineup`` describing exactly which rules
failed and why.

The slate rules implemented here mirror the actual DraftKings MLB
classic contest rules:

* exactly 10 players
* salary cap $50,000 (hard error if exceeded)
* salary floor $47,000 (hard error; manual override exists upstream)
* one each of P / C / 1B / 2B / 3B / SS / UTIL, three OF
* a slot must be in the player's ``position_eligibility`` (UTIL is
  open to any non-pitcher)
* players must come from at least two different games
* no duplicate players
* no scratched players, banned players, or missing locks
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from loguru import logger

from data_pipeline.ingestion.dk_csv_parser import DKPlayer


# DK MLB classic slate constants.
LINEUP_SIZE: int = 10
SALARY_CAP: int = 50_000
SALARY_FLOOR: int = 47_000

# Required slot counts in a complete classic lineup.
SLOT_REQUIREMENTS: dict[str, int] = {
    "P": 1,
    "C": 1,
    "1B": 1,
    "2B": 1,
    "3B": 1,
    "SS": 1,
    "OF": 3,
    "UTIL": 1,
}

# DK's lineup_status values. Any value outside this set is treated as
# "unknown" by the safety rules.
VALID_LINEUP_STATUSES: frozenset[str] = frozenset(
    {
        "confirmed_starting",
        "projected_starting",
        "bench",
        "scratched",
        "unknown",
    }
)


class RosterSlot(str, Enum):
    """The eight DK MLB classic roster slots, as displayed in the upload CSV."""

    P = "P"
    C = "C"
    FIRST_BASE = "1B"
    SECOND_BASE = "2B"
    THIRD_BASE = "3B"
    SS = "SS"
    OF = "OF"
    UTIL = "UTIL"


# Set of valid slot strings, for quick membership checks.
_VALID_SLOTS: frozenset[str] = frozenset(s.value for s in RosterSlot)


@dataclass
class ValidationError:
    """A single rule violation produced by the validator."""

    rule: str
    message: str
    severity: str  # "error" or "warning"
    player_name: str | None = None


@dataclass
class ValidatedLineup:
    """The result of running ``LineupValidator.validate`` on one lineup."""

    players: list[DKPlayer]
    total_salary: int
    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)


# Type alias: a lineup is a list of (player, assigned slot string) pairs.
LineupAssignment = list[tuple[DKPlayer, str]]


def _get_status(player: DKPlayer) -> str:
    """Return the player's ``lineup_status``, defaulting to confirmed.

    ``DKPlayer`` does not yet carry ``lineup_status`` as a formal field
    (it's set on the ``salaries`` row, not in the CSV), so we read it
    defensively. Once the upstream pipeline annotates the player, this
    will pick it up automatically.
    """
    status = getattr(player, "lineup_status", "confirmed_starting")
    return status if status in VALID_LINEUP_STATUSES else "unknown"


def _game_key(player: DKPlayer) -> str:
    """Stable identifier for the game a player appears in."""
    if player.away_team and player.home_team:
        return f"{player.away_team}@{player.home_team}"
    return player.game_info_raw or "unknown_game"


class LineupValidator:
    """Runs the full DraftKings MLB classic rule set against a lineup."""

    def __init__(
        self,
        constraints_by_dk_id: Mapping[str, Any] | None = None,
    ) -> None:
        """Build a validator.

        Args:
            constraints_by_dk_id: optional map of ``dk_id`` →
                ``player_constraint``. The constraint object is duck-typed:
                we look for ``is_banned``, ``is_locked``, and (optionally)
                ``max_exposure``. Either a dataclass or a dict works.
        """
        self._constraints: dict[str, Any] = dict(constraints_by_dk_id or {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, lineup: LineupAssignment) -> ValidatedLineup:
        """Validate one lineup; return a ``ValidatedLineup`` with every issue.

        Never raises. Every rule runs even if earlier rules failed, so
        the caller gets the complete picture of what's wrong.
        """
        players = [p for p, _ in lineup]
        slots = [s for _, s in lineup]
        total_salary = sum(p.salary for p in players)

        all_errors: list[ValidationError] = []
        all_warnings: list[ValidationError] = []

        for issues in (
            self._check_player_count(lineup),
            self._check_no_duplicates(players),
            self._check_salary_cap(total_salary),
            self._check_salary_floor(total_salary),
            self._check_position_slots(slots),
            self._check_position_eligibility(lineup),
            self._check_util_not_pitcher(lineup),
            self._check_min_games(players),
            self._check_no_scratched(players),
            self._check_lineup_status_warning(players),
            self._check_no_banned(players),
            self._check_locked_present(players),
        ):
            for issue in issues:
                if issue.severity == "warning":
                    all_warnings.append(issue)
                else:
                    all_errors.append(issue)

        return ValidatedLineup(
            players=players,
            total_salary=total_salary,
            is_valid=not all_errors,
            errors=all_errors,
            warnings=all_warnings,
        )

    def validate_batch(
        self, lineups: list[LineupAssignment]
    ) -> list[ValidatedLineup]:
        """Validate a list of lineups and log a one-line summary."""
        results = [self.validate(lu) for lu in lineups]
        valid = sum(1 for r in results if r.is_valid)
        invalid = len(results) - valid
        logger.info(
            f"Validated {len(results)} lineups: {valid} valid, {invalid} invalid"
        )
        return results

    # ------------------------------------------------------------------
    # Rule implementations — each returns list[ValidationError]
    # ------------------------------------------------------------------

    def _check_player_count(
        self, lineup: LineupAssignment
    ) -> list[ValidationError]:
        """Lineup must contain exactly 10 players."""
        if len(lineup) != LINEUP_SIZE:
            return [
                ValidationError(
                    rule="player_count",
                    message=(
                        f"lineup has {len(lineup)} players; "
                        f"expected exactly {LINEUP_SIZE}"
                    ),
                    severity="error",
                )
            ]
        return []

    def _check_no_duplicates(
        self, players: list[DKPlayer]
    ) -> list[ValidationError]:
        """No player may appear twice (matched on dk_id)."""
        counts = Counter(p.dk_id for p in players)
        errors: list[ValidationError] = []
        for dk_id, count in counts.items():
            if count > 1:
                name = next(
                    (p.name for p in players if p.dk_id == dk_id),
                    dk_id,
                )
                errors.append(
                    ValidationError(
                        rule="no_duplicates",
                        message=(
                            f"player {name} ({dk_id}) appears {count} times; "
                            "duplicates are not allowed"
                        ),
                        severity="error",
                        player_name=name,
                    )
                )
        return errors

    def _check_salary_cap(self, total_salary: int) -> list[ValidationError]:
        """Total salary must not exceed the DK salary cap."""
        if total_salary > SALARY_CAP:
            return [
                ValidationError(
                    rule="salary_cap",
                    message=(
                        f"total salary ${total_salary:,} exceeds cap "
                        f"${SALARY_CAP:,}"
                    ),
                    severity="error",
                )
            ]
        return []

    def _check_salary_floor(self, total_salary: int) -> list[ValidationError]:
        """Total salary must be at or above the soft floor.

        DK does not enforce a floor itself; we treat it as a hard error
        because under-spending on GPP is bankroll-negative. A manual
        override exists in the export pipeline if the user knows better.
        """
        if total_salary < SALARY_FLOOR:
            return [
                ValidationError(
                    rule="salary_floor",
                    message=(
                        f"total salary ${total_salary:,} is below floor "
                        f"${SALARY_FLOOR:,} (manual override available)"
                    ),
                    severity="error",
                )
            ]
        return []

    def _check_position_slots(
        self, slots: list[str]
    ) -> list[ValidationError]:
        """Each required slot must appear the correct number of times."""
        counts = Counter(slots)
        errors: list[ValidationError] = []

        for slot, required in SLOT_REQUIREMENTS.items():
            actual = counts.get(slot, 0)
            if actual != required:
                errors.append(
                    ValidationError(
                        rule="position_slots",
                        message=(
                            f"slot {slot}: have {actual}, need {required}"
                        ),
                        severity="error",
                    )
                )

        unknown = set(counts) - _VALID_SLOTS
        for slot in sorted(unknown):
            errors.append(
                ValidationError(
                    rule="position_slots",
                    message=f"unknown roster slot {slot!r}",
                    severity="error",
                )
            )

        return errors

    def _check_position_eligibility(
        self, lineup: LineupAssignment
    ) -> list[ValidationError]:
        """A player's assigned slot must match their eligibility.

        UTIL is open to any non-pitcher; everything else requires the
        slot to be present in ``position_eligibility``.
        """
        errors: list[ValidationError] = []
        for player, slot in lineup:
            if slot == RosterSlot.UTIL.value:
                # UTIL eligibility is checked separately (no pitchers).
                continue
            if slot not in player.position_eligibility:
                errors.append(
                    ValidationError(
                        rule="position_eligibility",
                        message=(
                            f"{player.name} assigned to {slot} but is "
                            f"only eligible for "
                            f"{'/'.join(player.position_eligibility) or 'nothing'}"
                        ),
                        severity="error",
                        player_name=player.name,
                    )
                )
        return errors

    def _check_util_not_pitcher(
        self, lineup: LineupAssignment
    ) -> list[ValidationError]:
        """The UTIL slot must not be filled by a pitcher."""
        errors: list[ValidationError] = []
        for player, slot in lineup:
            if slot != RosterSlot.UTIL.value:
                continue
            # A pitcher is anyone whose only eligibility is "P", or
            # whose is_pitcher flag is set. In practice DK only allows
            # P-only players to be pitchers, but we check both for safety.
            is_p_only = player.position_eligibility == ["P"]
            if is_p_only or getattr(player, "is_pitcher", False):
                errors.append(
                    ValidationError(
                        rule="util_not_pitcher",
                        message=(
                            f"{player.name} is a pitcher and cannot fill "
                            "the UTIL slot"
                        ),
                        severity="error",
                        player_name=player.name,
                    )
                )
        return errors

    def _check_min_games(
        self, players: list[DKPlayer]
    ) -> list[ValidationError]:
        """Players must come from at least 2 different games."""
        if not players:
            return []
        games = {_game_key(p) for p in players}
        if len(games) < 2:
            game = next(iter(games))
            return [
                ValidationError(
                    rule="min_games",
                    message=(
                        f"all 10 players are from the same game ({game}); "
                        "lineups must span at least 2 games"
                    ),
                    severity="error",
                )
            ]
        return []

    def _check_no_scratched(
        self, players: list[DKPlayer]
    ) -> list[ValidationError]:
        """Scratched players block the lineup outright."""
        errors: list[ValidationError] = []
        for player in players:
            if _get_status(player) == "scratched":
                errors.append(
                    ValidationError(
                        rule="no_scratched",
                        message=(
                            f"{player.name} is scratched and cannot be "
                            "in a lineup"
                        ),
                        severity="error",
                        player_name=player.name,
                    )
                )
        return errors

    def _check_lineup_status_warning(
        self, players: list[DKPlayer]
    ) -> list[ValidationError]:
        """Soft-warn on unknown / projected-starting players."""
        warnings_out: list[ValidationError] = []
        for player in players:
            status = _get_status(player)
            if status in ("unknown", "projected_starting"):
                warnings_out.append(
                    ValidationError(
                        rule="lineup_status_unconfirmed",
                        message=(
                            f"{player.name} has lineup_status={status!r}; "
                            "confirm before lock"
                        ),
                        severity="warning",
                        player_name=player.name,
                    )
                )
        return warnings_out

    def _check_no_banned(
        self, players: list[DKPlayer]
    ) -> list[ValidationError]:
        """Players flagged as banned in the constraints map are blocked."""
        errors: list[ValidationError] = []
        for player in players:
            constraint = self._constraints.get(player.dk_id)
            if constraint is None:
                continue
            if self._constraint_flag(constraint, "is_banned"):
                errors.append(
                    ValidationError(
                        rule="no_banned",
                        message=f"{player.name} is banned from this slate",
                        severity="error",
                        player_name=player.name,
                    )
                )
        return errors

    def _check_locked_present(
        self, players: list[DKPlayer]
    ) -> list[ValidationError]:
        """Every locked player in the constraints map must appear here."""
        errors: list[ValidationError] = []
        present_ids = {p.dk_id for p in players}
        for dk_id, constraint in self._constraints.items():
            if not self._constraint_flag(constraint, "is_locked"):
                continue
            if dk_id in present_ids:
                continue
            name = self._constraint_attr(constraint, "name") or dk_id
            errors.append(
                ValidationError(
                    rule="locked_player_missing",
                    message=(
                        f"locked player {name} ({dk_id}) is missing from "
                        "this lineup"
                    ),
                    severity="error",
                    player_name=str(name),
                )
            )
        return errors

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _constraint_flag(constraint: Any, attr: str) -> bool:
        """Read a boolean flag from either an object or a dict."""
        if isinstance(constraint, Mapping):
            return bool(constraint.get(attr, False))
        return bool(getattr(constraint, attr, False))

    @staticmethod
    def _constraint_attr(constraint: Any, attr: str) -> Any:
        """Read an arbitrary attribute from either an object or a dict."""
        if isinstance(constraint, Mapping):
            return constraint.get(attr)
        return getattr(constraint, attr, None)
