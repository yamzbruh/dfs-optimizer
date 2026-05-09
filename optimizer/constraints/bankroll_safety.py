"""Final pre-export bankroll safety gate.

``LineupValidator`` says "is this lineup *legal*". This checker says
"is this *batch* of lineups safe to push live to DraftKings". The two
are intentionally separate: a lineup may be perfectly legal yet still
be unsafe to submit (e.g. one of its players is on a rain-delay
pre-game scratch list 25 minutes before lock).

A failing ``BankrollSafetyResult`` is a hard block on export — never
auto-overridden — to protect bankroll.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

from loguru import logger

from data_pipeline.ingestion.dk_csv_parser import DKPlayer
from optimizer.constraints.lineup_validator import (
    SALARY_FLOOR,
    ValidatedLineup,
    _get_status,
)


# Below this many lineups, we refuse to export at all — the user
# asked for 20 and we should be honest about not having them.
TARGET_LINEUP_COUNT: int = 20

# Within this many minutes of lock, "unknown" lineup status is
# unacceptable; a lineup with such a player must be blocked.
UNKNOWN_STATUS_BLOCK_MINUTES: int = 30

# A salary in this range is technically above the floor but uses
# significantly less of the cap than expected — we warn.
LOW_SALARY_WARNING_BAND: tuple[int, int] = (SALARY_FLOOR, 48_000)


@dataclass
class BankrollSafetyResult:
    """Outcome of the bankroll safety check on a batch of lineups."""

    safe_to_export: bool
    blocking_issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    lineup_count: int = 0
    valid_lineup_count: int = 0


class BankrollSafetyChecker:
    """Aggregate-level checks that run once before export."""

    def pre_export_check(
        self,
        lineups: list[ValidatedLineup],
        lock_time: datetime | None = None,
        minutes_to_lock: int | None = None,
    ) -> BankrollSafetyResult:
        """Run every pre-export check and return a single verdict.

        Args:
            lineups: the validated lineups about to be exported.
            lock_time: contest lock time. Used to derive
                ``minutes_to_lock`` when it isn't supplied directly.
            minutes_to_lock: explicit override for time-to-lock.
                When both are provided, this takes precedence.
        """
        result = BankrollSafetyResult(
            safe_to_export=True,
            lineup_count=len(lineups),
            valid_lineup_count=sum(1 for lu in lineups if lu.is_valid),
        )

        effective_minutes = self._resolve_minutes_to_lock(
            lock_time, minutes_to_lock
        )

        # ---- blocking checks ----
        if result.valid_lineup_count < TARGET_LINEUP_COUNT:
            result.blocking_issues.append(
                f"only {result.valid_lineup_count} valid lineups; "
                f"need {TARGET_LINEUP_COUNT} before export"
            )

        scratched_count = self._count_lineups_with_player_status(
            lineups, {"scratched"}
        )
        if scratched_count:
            result.blocking_issues.append(
                f"{scratched_count} lineup(s) contain a scratched player"
            )

        below_floor = self._count_lineups_below_salary(lineups, SALARY_FLOOR)
        if below_floor:
            result.blocking_issues.append(
                f"{below_floor} lineup(s) have total salary below "
                f"${SALARY_FLOOR:,}"
            )

        if (
            effective_minutes is not None
            and effective_minutes <= UNKNOWN_STATUS_BLOCK_MINUTES
        ):
            unknown_count = self._count_lineups_with_player_status(
                lineups, {"unknown"}
            )
            if unknown_count:
                result.blocking_issues.append(
                    f"{unknown_count} lineup(s) have an unknown-status "
                    f"player and lock is in "
                    f"{effective_minutes} minute(s)"
                )

        # ---- non-blocking warnings ----
        projected_count = self._count_lineups_with_player_status(
            lineups, {"projected_starting"}
        )
        if projected_count:
            result.warnings.append(
                f"{projected_count} lineup(s) include a "
                "projected_starting (unconfirmed) player"
            )

        low_salary_count = self._count_lineups_in_salary_band(
            lineups, *LOW_SALARY_WARNING_BAND
        )
        if low_salary_count:
            lo, hi = LOW_SALARY_WARNING_BAND
            result.warnings.append(
                f"{low_salary_count} lineup(s) have total salary in "
                f"${lo:,}–${hi:,} (low utilization)"
            )

        if result.lineup_count < TARGET_LINEUP_COUNT:
            result.warnings.append(
                f"only {result.lineup_count} lineups generated; "
                f"target is {TARGET_LINEUP_COUNT}"
            )

        result.safe_to_export = not result.blocking_issues

        if result.safe_to_export:
            logger.info(
                f"Bankroll safety: OK — {result.valid_lineup_count}/"
                f"{result.lineup_count} valid; "
                f"{len(result.warnings)} warning(s)"
            )
        else:
            logger.warning(
                "Bankroll safety: BLOCKED — "
                + "; ".join(result.blocking_issues)
            )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_minutes_to_lock(
        lock_time: datetime | None,
        minutes_to_lock: int | None,
    ) -> int | None:
        """Combine lock_time and minutes_to_lock into a single number."""
        if minutes_to_lock is not None:
            return int(minutes_to_lock)
        if lock_time is None:
            return None
        # Use a UTC-aware "now" if the lock_time is timezone-aware,
        # naive otherwise — comparing the two would raise.
        now = (
            datetime.now(timezone.utc)
            if lock_time.tzinfo is not None
            else datetime.now()
        )
        delta = lock_time - now
        return max(int(delta.total_seconds() // 60), 0)

    @staticmethod
    def _count_lineups_with_player_status(
        lineups: list[ValidatedLineup],
        target_statuses: Iterable[str],
    ) -> int:
        """Count lineups containing any player whose status is in the set."""
        target_set = set(target_statuses)
        count = 0
        for lu in lineups:
            if any(_get_status(p) in target_set for p in lu.players):
                count += 1
        return count

    @staticmethod
    def _count_lineups_below_salary(
        lineups: list[ValidatedLineup], threshold: int
    ) -> int:
        return sum(1 for lu in lineups if lu.total_salary < threshold)

    @staticmethod
    def _count_lineups_in_salary_band(
        lineups: list[ValidatedLineup], lo: int, hi: int
    ) -> int:
        return sum(1 for lu in lineups if lo <= lu.total_salary <= hi)


# Re-export DKPlayer in case downstream callers want it from the
# safety module rather than reaching back into the parser.
__all__ = [
    "BankrollSafetyChecker",
    "BankrollSafetyResult",
    "DKPlayer",
]
