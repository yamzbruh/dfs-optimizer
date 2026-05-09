"""PuLP / HiGHS linear-programming lineup optimizer.

Generates up to 20 diverse DraftKings MLB classic lineups from a list
of ``PlayerProjection`` objects.  Each lineup is a feasible solution
to an integer linear program (ILP) that respects the DK salary cap,
position-slot requirements, dual-eligibility rules, minimum-game
diversity, and a stacking overlap penalty that prevents the solver from
regenerating the same lineup on successive calls.

Solver hierarchy
----------------
1. HiGHS (via ``highspy`` / PuLP's built-in HiGHS binding) — in-process,
   no disk I/O, fast.
2. CBC (PuLP's bundled solver) — fallback when HiGHS is unavailable.

Thread model
------------
Each worker gets ``threads=1`` so a pool of workers doesn't fight for
CPU cores inside the solver.  ``msg=0`` silences all solver stdout.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pulp
from loguru import logger

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_pipeline.ingestion.dk_csv_parser import DKPlayer  # noqa: E402
from optimizer.constraints.lineup_validator import (  # noqa: E402
    LineupValidator,
    SALARY_CAP,
    SALARY_FLOOR,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# DK MLB classic required slot counts.
_REQUIRED_SLOTS: dict[str, int] = {
    "P":    1,
    "C":    1,
    "1B":   1,
    "2B":   1,
    "3B":   1,
    "SS":   1,
    "OF":   3,
    "UTIL": 1,
}

# Positions eligible for the UTIL slot (all non-pitchers).
_UTIL_ELIGIBLE: frozenset[str] = frozenset({"C", "1B", "2B", "3B", "SS", "OF"})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PlayerProjection:
    """A single player's projection for one slate."""

    player: DKPlayer
    pts_q15: float
    pts_q50: float
    pts_q85: float
    ownership_proj: float     # Projected ownership percentage (0–100).
    leverage_score: float     # pts_q50 / ownership_proj (higher = better GPP).
    is_locked: bool = False
    is_banned: bool = False
    max_exposure: float = 0.70


@dataclass
class LineupResult:
    """One fully-assigned DraftKings lineup."""

    players: list[tuple[DKPlayer, str]]   # (player, roster_slot)
    total_salary: int
    projected_pts: float
    leverage_score: float
    portfolio_score: float
    is_valid: bool
    player_dk_ids: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.player_dk_ids:
            self.player_dk_ids = {p.dk_id for p, _ in self.players}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _game_key(player: DKPlayer) -> str:
    """Return a stable game identifier from the player's game fields."""
    if player.away_team and player.home_team:
        return f"{player.away_team}@{player.home_team}"
    return player.game_info_raw or "unknown"


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class LineupOptimizer:
    """Generates diverse DK MLB classic lineups via integer linear programming.

    Usage::

        opt = LineupOptimizer()
        lineups = opt.generate_lineups(projections, n_lineups=20)
    """

    def __init__(
        self,
        solver: str = "HiGHS",
        time_limit_seconds: int = 30,
    ) -> None:
        """Set up the ILP solver.

        Args:
            solver: Preferred solver name.  ``"HiGHS"`` (default) uses
                the in-process HiGHS binding; falls back to CBC if
                HiGHS is unavailable.
            time_limit_seconds: Per-lineup solver time limit.  30 s is
                generous; typical solve time is < 1 s per lineup.
        """
        self.solver_obj = self._build_solver(solver, time_limit_seconds)
        self._validator = LineupValidator()
        logger.debug(f"LineupOptimizer ready  solver={type(self.solver_obj).__name__}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize_single(
        self,
        projections: list[PlayerProjection],
        locked_ids: set[str] | None = None,
        banned_ids: set[str] | None = None,
        previous_lineups: list[LineupResult] | None = None,
        min_overlap_penalty: int = 8,
    ) -> LineupResult | None:
        """Solve one ILP lineup problem and return the best feasible lineup.

        Args:
            projections: Full pool of player projections for this slate.
            locked_ids: Set of ``dk_id`` values that must appear.
            banned_ids: Set of ``dk_id`` values that must not appear.
            previous_lineups: Already-generated lineups; used to add
                diversity constraints so the solver can't repeat them.
            min_overlap_penalty: Maximum number of players two lineups
                may share.  8 of 10 allows meaningful diversity.

        Returns:
            A ``LineupResult`` if an optimal solution was found, or
            ``None`` if the problem was infeasible or timed out.
        """
        if not projections:
            logger.warning("optimize_single: empty projections list")
            return None

        _locked = locked_ids or set()
        _banned = banned_ids or set()
        _prev = previous_lineups or []

        n = len(projections)
        prob = pulp.LpProblem("dk_lineup", pulp.LpMaximize)

        # -- Decision variables ----------------------------------------------
        # x[i] = 1 if player i is selected in this lineup.
        x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

        # y[game_key] = 1 if at least one player from that game is selected.
        games = list({_game_key(p.player) for p in projections})
        y = {g: pulp.LpVariable(f"y_{gi}", cat="Binary") for gi, g in enumerate(games)}

        # -- Objective: maximise median projected points ---------------------
        prob += pulp.lpSum(projections[i].pts_q50 * x[i] for i in range(n))

        # -- Hard constraints ------------------------------------------------

        # a. Salary cap.
        prob += (
            pulp.lpSum(projections[i].player.salary * x[i] for i in range(n))
            <= SALARY_CAP,
            "salary_cap",
        )

        # Salary floor — soft in the validator but enforce here to avoid
        # trivially cheap lineups that the validator would reject anyway.
        prob += (
            pulp.lpSum(projections[i].player.salary * x[i] for i in range(n))
            >= SALARY_FLOOR,
            "salary_floor",
        )

        # b. Exactly 10 players.
        prob += pulp.lpSum(x) == 10, "ten_players"

        # c. Position eligibility sets.
        eligible = self._build_eligibility(projections)

        # Non-UTIL slot counts: exactly the required number, with the
        # exception of OF which is exactly 3.
        # We use >= for regular slots and keep total == 10 to implicitly
        # fill the UTIL slot with whatever's left.
        prob += (
            pulp.lpSum(x[i] for i in eligible["P"]) == 1,
            "slot_P",
        )
        for pos in ("C", "1B", "2B", "3B", "SS"):
            prob += (
                pulp.lpSum(x[i] for i in eligible[pos]) >= 1,
                f"slot_{pos}",
            )
        prob += (
            pulp.lpSum(x[i] for i in eligible["OF"]) >= 3,
            "slot_OF",
        )
        # UTIL must be fillable by a non-pitcher — guaranteed by the
        # eligibility matrix (pitchers are excluded from UTIL).
        prob += (
            pulp.lpSum(x[i] for i in eligible["UTIL"]) >= 1,
            "slot_UTIL",
        )

        # d. Minimum 2 games represented.
        #    y[g] >= x[i] for every player i in game g  →  y[g]=1 when selected.
        #    y[g] <= sum(x[i] for i in game g)           →  y[g]=0 when none selected.
        for g in games:
            game_indices = [
                i for i in range(n) if _game_key(projections[i].player) == g
            ]
            if not game_indices:
                continue
            for i in game_indices:
                prob += y[g] >= x[i], f"game_link_{g}_{i}"
            prob += (
                y[g] <= pulp.lpSum(x[i] for i in game_indices),
                f"game_upper_{g}",
            )
        prob += pulp.lpSum(y.values()) >= 2, "min_games"

        # e. Locked players.
        for i, proj in enumerate(projections):
            if proj.player.dk_id in _locked or proj.is_locked:
                prob += x[i] == 1, f"locked_{proj.player.dk_id}"

        # f. Banned players.
        for i, proj in enumerate(projections):
            if proj.player.dk_id in _banned or proj.is_banned:
                prob += x[i] == 0, f"banned_{proj.player.dk_id}"

        # g. Diversity: overlap with each previous lineup <= min_overlap_penalty.
        for li, prev in enumerate(_prev):
            prev_ids = prev.player_dk_ids
            overlap_vars = [
                x[i] for i in range(n)
                if projections[i].player.dk_id in prev_ids
            ]
            if overlap_vars:
                prob += (
                    pulp.lpSum(overlap_vars) <= min_overlap_penalty,
                    f"diversity_{li}",
                )

        # -- Solve -----------------------------------------------------------
        prob.solve(self.solver_obj)

        status = pulp.LpStatus[prob.status]
        if status != "Optimal":
            logger.warning(f"optimize_single: solver status={status!r}; returning None")
            return None

        # -- Extract solution ------------------------------------------------
        selected = [
            projections[i] for i in range(n)
            if pulp.value(x[i]) is not None and pulp.value(x[i]) > 0.5
        ]

        if len(selected) != 10:
            logger.warning(
                f"optimize_single: solver selected {len(selected)} players "
                f"(expected 10); returning None"
            )
            return None

        assigned = self.assign_positions(selected)
        total_salary = sum(p.salary for p, _ in assigned)
        projected_pts = sum(proj.pts_q50 for proj in selected)
        lev = float(np.mean([proj.leverage_score for proj in selected]))

        result = LineupResult(
            players=assigned,
            total_salary=total_salary,
            projected_pts=projected_pts,
            leverage_score=lev,
            portfolio_score=0.0,   # Filled by generate_lineups after all lineups are known.
            is_valid=False,
        )
        return result

    def assign_positions(
        self, selected: list[PlayerProjection]
    ) -> list[tuple[DKPlayer, str]]:
        """Greedily assign DK roster slots to the selected 10 players.

        Assignment order mirrors DK's slot priority:
        1. P  — must be P-only eligible (pitcher)
        2. C, 1B, 2B, 3B, SS  — first eligible unassigned player each
        3. OF  × 3             — next OF-eligible unassigned players
        4. UTIL                — the remaining player (must be non-pitcher)

        Args:
            selected: Exactly 10 ``PlayerProjection`` objects chosen
                by the ILP solver.

        Returns:
            List of ``(DKPlayer, roster_slot)`` tuples in the order the
            slots are assigned.
        """
        assigned: dict[str, str] = {}   # dk_id → slot
        remaining = list(selected)

        def _assign_slot(slot: str, eligibility_check) -> bool:
            """Pop the first eligible unassigned player into ``slot``."""
            for proj in remaining:
                if eligibility_check(proj) and proj.player.dk_id not in assigned:
                    assigned[proj.player.dk_id] = slot
                    remaining.remove(proj)
                    return True
            return False

        # 1. Pitcher.
        _assign_slot("P", lambda p: "P" in p.player.position_eligibility)

        # 2. Infield / Catcher.
        for pos in ("C", "1B", "2B", "3B", "SS"):
            _assign_slot(pos, lambda p, s=pos: s in p.player.position_eligibility)

        # 3. Three outfielders.
        for _ in range(3):
            _assign_slot("OF", lambda p: "OF" in p.player.position_eligibility)

        # 4. UTIL — whoever is left.
        for proj in list(remaining):
            if proj.player.dk_id not in assigned:
                assigned[proj.player.dk_id] = "UTIL"
                remaining.remove(proj)
                break

        # Reconstruct in input order for consistent output.
        return [
            (proj.player, assigned.get(proj.player.dk_id, "UTIL"))
            for proj in selected
        ]

    def generate_lineups(
        self,
        projections: list[PlayerProjection],
        n_lineups: int = 20,
        locked_ids: set[str] | None = None,
        banned_ids: set[str] | None = None,
    ) -> list[LineupResult]:
        """Generate ``n_lineups`` diverse and valid lineups.

        Calls ``optimize_single`` sequentially, passing all previously
        generated lineups as diversity constraints so each new lineup
        is forced to differ from every prior one.

        Args:
            projections: Full slate player pool with projections.
            n_lineups: Number of lineups to generate (default 20).
            locked_ids: Players that must appear in every lineup.
            banned_ids: Players excluded from every lineup.

        Returns:
            List of valid ``LineupResult`` objects (may be fewer than
            ``n_lineups`` if the pool is too shallow or the solver
            times out repeatedly).
        """
        if not projections:
            logger.warning("generate_lineups: empty projections list")
            return []

        results: list[LineupResult] = []

        for attempt in range(1, n_lineups + 1):
            result = self.optimize_single(
                projections=projections,
                locked_ids=locked_ids,
                banned_ids=banned_ids,
                previous_lineups=results,
            )

            if result is None:
                logger.warning(f"Lineup {attempt}/{n_lineups}: solver returned None — skipping")
                continue

            # Validate through the rule engine.
            validated = self._validator.validate(result.players)
            result.is_valid = validated.is_valid

            if validated.errors:
                logger.warning(
                    f"Lineup {attempt}/{n_lineups}: {len(validated.errors)} "
                    f"validation error(s) — {[e.rule for e in validated.errors]}"
                )

            result.portfolio_score = self.calculate_portfolio_score(
                result, projections
            )

            results.append(result)
            logger.info(
                f"Lineup {attempt:>2}/{n_lineups}: "
                f"salary=${result.total_salary:,}  "
                f"pts={result.projected_pts:.2f}  "
                f"valid={result.is_valid}"
            )

        valid_count = sum(1 for r in results if r.is_valid)
        logger.info(
            f"generate_lineups complete: {valid_count} valid of "
            f"{len(results)} generated (target={n_lineups})"
        )
        return results

    def calculate_portfolio_score(
        self,
        lineup: LineupResult,
        all_projections: list[PlayerProjection],
    ) -> float:
        """Compute a composite portfolio score for GPP ranking.

        Score = projected_pts
              + (mean_leverage_score × 1.2)
              - (mean_ownership_proj × 0.1)

        Higher leverage and lower ownership both improve the score,
        which biases selection toward contrarian, high-upside lineups
        that outperform field in GPP tournaments.

        Args:
            lineup: The lineup to score.
            all_projections: Full projection pool (used for ownership
                look-up by dk_id).

        Returns:
            Float portfolio score.
        """
        proj_by_id = {p.player.dk_id: p for p in all_projections}
        lineup_projs = [
            proj_by_id[player.dk_id]
            for player, _ in lineup.players
            if player.dk_id in proj_by_id
        ]

        if not lineup_projs:
            return lineup.projected_pts

        leverage_bonus = float(np.mean([p.leverage_score for p in lineup_projs]))
        ownership_penalty = float(np.mean([p.ownership_proj for p in lineup_projs]))

        return lineup.projected_pts + (leverage_bonus * 1.2) - (ownership_penalty * 0.1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_eligibility(
        self, projections: list[PlayerProjection]
    ) -> dict[str, list[int]]:
        """Build slot → [player index] eligibility map.

        UTIL is open to all non-pitchers (any player whose eligibility
        includes at least one non-P position).
        """
        eligible: dict[str, list[int]] = {slot: [] for slot in _REQUIRED_SLOTS}

        for i, proj in enumerate(projections):
            elig = proj.player.position_eligibility
            for pos in elig:
                if pos in eligible:
                    eligible[pos].append(i)
            # UTIL: eligible if any non-P position exists.
            if any(p in _UTIL_ELIGIBLE for p in elig):
                eligible["UTIL"].append(i)

        return eligible

    @staticmethod
    def _build_solver(name: str, time_limit: int):
        """Return a configured PuLP solver object.

        Tries HiGHS first; falls back to CBC if unavailable.
        ``msg=0`` suppresses all solver stdout (critical for parallel
        workers).  ``threads=1`` prevents in-process CPU contention.
        """
        try:
            solver = pulp.getSolver(
                name,
                msg=0,
                timeLimit=time_limit,
                threads=1,
            )
            logger.debug(f"Using HiGHS solver (timeLimit={time_limit}s)")
            return solver
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"HiGHS unavailable ({exc}); falling back to CBC"
            )
            return pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
