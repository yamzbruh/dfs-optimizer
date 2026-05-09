"""DraftKings lineup upload CSV writer.

Writes optimizer-produced lineups in the exact format that the
DraftKings MLB classic contest upload UI accepts:

    P,P,C,1B,2B,3B,SS,OF,OF,OF

Each cell contains the original "Name (dk_id)" string from the salary
CSV (DK matches uploads on this exact token).
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from loguru import logger

from data_pipeline.ingestion.dk_csv_parser import DKPlayer


# Header columns and their order, exactly as DK MLB classic expects.
DK_LINEUP_COLUMNS: tuple[str, ...] = (
    "P",
    "P",
    "C",
    "1B",
    "2B",
    "3B",
    "SS",
    "OF",
    "OF",
    "OF",
)

# DK MLB classic salary cap.
DK_SALARY_CAP: int = 50000

# Required position counts in a valid classic MLB lineup (2 P, no UTIL).
REQUIRED_POSITIONS: dict[str, int] = {
    "P": 2,
    "C": 1,
    "1B": 1,
    "2B": 1,
    "3B": 1,
    "SS": 1,
    "OF": 3,
}


class DKLineupExporter:
    """Writes a list of optimizer lineups to a DK-upload-compatible CSV."""

    def export(
        self,
        lineups: list[list[tuple[DKPlayer, str]]],
        output_path: str | Path,
    ) -> str:
        """Write ``lineups`` to a DK upload CSV at ``output_path``.

        Each inner list is one lineup: 10 ``(DKPlayer, roster_slot)`` pairs
        where ``roster_slot`` is ``"P"``, ``"C"``, ``"1B"``, etc. The export
        orders cells to match ``DK_LINEUP_COLUMNS`` (first P column = first
        pitcher in that slot group, and likewise for the three OF columns).

        Returns the absolute output path as a string.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(DK_LINEUP_COLUMNS)

            for idx, lineup in enumerate(lineups, start=1):
                errors = self.validate_export_format(lineup)
                if errors:
                    # We still emit a row so DK can flag it during upload,
                    # but the caller is logged the precise issues.
                    logger.error(
                        f"Lineup {idx} has export-format errors: {errors}"
                    )

                writer.writerow(self._row_for_lineup(lineup))

        logger.info(f"Wrote {len(lineups)} lineups to {path}")
        return str(path.resolve())

    def validate_export_format(
        self, lineup: list[tuple[DKPlayer, str]]
    ) -> list[str]:
        """Return a list of human-readable errors for ``lineup``.

        Empty list ⇒ valid. Checks: exactly 10 players, salary cap,
        and that every required roster slot is filled the correct
        number of times.
        """
        errors: list[str] = []

        if len(lineup) != 10:
            errors.append(
                f"lineup has {len(lineup)} players; expected exactly 10"
            )

        total_salary = sum(p.salary for p, _ in lineup)
        if total_salary > DK_SALARY_CAP:
            errors.append(
                f"total salary {total_salary} exceeds cap {DK_SALARY_CAP}"
            )

        missing_slot = [
            (p, slot) for p, slot in lineup if not (slot or "").strip()
        ]
        if missing_slot:
            errors.append(
                f"{len(missing_slot)} players missing assigned roster_slot"
            )
            return errors  # Position counts are meaningless without slots.

        slot_counts = Counter(slot.strip().upper() for _, slot in lineup)
        for slot, required in REQUIRED_POSITIONS.items():
            actual = slot_counts.get(slot, 0)
            if actual != required:
                errors.append(
                    f"position {slot}: have {actual}, need {required}"
                )

        unknown_slots = set(slot_counts) - set(REQUIRED_POSITIONS)
        if unknown_slots:
            errors.append(
                f"unknown roster_slot values: {sorted(unknown_slots)}"
            )

        return errors

    @staticmethod
    def _row_for_lineup(
        lineup: list[tuple[DKPlayer, str]],
    ) -> list[str]:
        """Order ``lineup`` into DK column order and emit "Name (id)" cells."""
        by_slot: dict[str, list[DKPlayer]] = {}
        for player, slot_raw in lineup:
            slot = (slot_raw or "").strip().upper()
            by_slot.setdefault(slot, []).append(player)

        row: list[str] = []
        used: dict[str, int] = {}
        for column in DK_LINEUP_COLUMNS:
            slot_players = by_slot.get(column, [])
            taken = used.get(column, 0)
            if taken < len(slot_players):
                player = slot_players[taken]
                row.append(f"{player.name} ({player.dk_id})")
                used[column] = taken + 1
            else:
                # Missing player for this slot — emit empty cell so DK
                # surfaces the gap rather than silently shifting columns.
                row.append("")
                used[column] = taken + 1
        return row
