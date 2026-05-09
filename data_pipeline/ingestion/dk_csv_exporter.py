"""DraftKings lineup upload CSV writer.

Writes optimizer-produced lineups in the exact format that the
DraftKings contest upload UI accepts:

    P,C,1B,2B,3B,SS,OF,OF,OF,UTIL

Each cell contains the original "Name (dk_id)" string from the salary
CSV (DK matches uploads on this exact token).
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from loguru import logger

from data_pipeline.ingestion.dk_csv_parser import DKPlayer


# Header columns and their order, exactly as DK expects.
DK_LINEUP_COLUMNS: tuple[str, ...] = (
    "P",
    "C",
    "1B",
    "2B",
    "3B",
    "SS",
    "OF",
    "OF",
    "OF",
    "UTIL",
)

# DK MLB classic salary cap.
DK_SALARY_CAP: int = 50000

# Required position counts in a valid classic MLB lineup.
REQUIRED_POSITIONS: dict[str, int] = {
    "P": 1,
    "C": 1,
    "1B": 1,
    "2B": 1,
    "3B": 1,
    "SS": 1,
    "OF": 3,
    "UTIL": 1,
}


class DKLineupExporter:
    """Writes a list of optimizer lineups to a DK-upload-compatible CSV."""

    def export(
        self,
        lineups: list[list[DKPlayer]],
        output_path: str | Path,
    ) -> str:
        """Write ``lineups`` to a DK upload CSV at ``output_path``.

        Each inner list must contain 10 ``DKPlayer`` instances with
        ``roster_position`` already assigned by the optimizer.
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

    def validate_export_format(self, lineup: list[DKPlayer]) -> list[str]:
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

        total_salary = sum(p.salary for p in lineup)
        if total_salary > DK_SALARY_CAP:
            errors.append(
                f"total salary {total_salary} exceeds cap {DK_SALARY_CAP}"
            )

        missing_position = [
            p for p in lineup if not getattr(p, "roster_position", None)
        ]
        if missing_position:
            errors.append(
                f"{len(missing_position)} players missing roster_position"
            )
            return errors  # Position counts are meaningless without slots.

        slot_counts = Counter(
            (p.roster_position or "").upper() for p in lineup
        )
        for slot, required in REQUIRED_POSITIONS.items():
            actual = slot_counts.get(slot, 0)
            if actual != required:
                errors.append(
                    f"position {slot}: have {actual}, need {required}"
                )

        unknown_slots = set(slot_counts) - set(REQUIRED_POSITIONS)
        if unknown_slots:
            errors.append(
                f"unknown roster_position values: {sorted(unknown_slots)}"
            )

        return errors

    @staticmethod
    def _row_for_lineup(lineup: list[DKPlayer]) -> list[str]:
        """Order ``lineup`` into DK column order and emit "Name (id)" cells."""
        # Group players by their assigned slot.
        by_slot: dict[str, list[DKPlayer]] = {}
        for player in lineup:
            slot = (player.roster_position or "").upper()
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
