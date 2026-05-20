"""PuLP / HiGHS linear-programming lineup optimizer.

Generates DraftKings MLB classic lineups from ``PlayerProjection`` rows.

**Default path — Monte Carlo:** many independent ILP solves, each on a
random draw from each player's quantile band (truncated normal around
``pts_q50``).  Lineups are deduplicated, scored on *original* projections,
exposure-filtered, and the top ``N`` are returned.  No sequential diversity
constraints, so late lineups are not starved.

**Fallback — sequential:** repeated ``optimize_single`` calls with overlap
penalties versus all prior lineups (legacy behaviour).

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
from dataclasses import dataclass, field, replace
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

# DK MLB classic required slot counts (2 P, no UTIL).
_REQUIRED_SLOTS: dict[str, int] = {
    "P":  2,
    "C":  1,
    "1B": 1,
    "2B": 1,
    "3B": 1,
    "SS": 1,
    "OF": 3,
}


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


def _monte_carlo_single_simulation(
    optimizer: "LineupOptimizer",
    projections: list[PlayerProjection],
    seed: int,
    locked_ids: set[str],
    banned_ids: set[str],
) -> LineupResult | None:
    """One Monte Carlo draw + single ILP solve (picklable worker for joblib)."""
    try:
        sim_projs = optimizer.simulate_projections(projections, seed=seed)
        return optimizer.optimize_single(
            projections=sim_projs,
            locked_ids=locked_ids or None,
            banned_ids=banned_ids or None,
            previous_lineups=None,
            min_overlap_penalty=10,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Monte Carlo sim seed={seed} failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _game_key(player: DKPlayer) -> str:
    """Return a stable game identifier from the player's game fields."""
    if player.away_team and player.home_team:
        return f"{player.away_team}@{player.home_team}"
    return player.game_info_raw or "unknown"


def _opposing_team(player: DKPlayer) -> str | None:
    """Opponent abbreviation for this player's game (DK ``home_team`` / ``away_team``)."""
    team = (player.team or "").strip().upper()
    home = (player.home_team or "").strip().upper()
    away = (player.away_team or "").strip().upper()
    if not team or not home or not away:
        return None
    if team == home:
        return away
    if team == away:
        return home
    return None


def _log_anti_stack_summary(projections: list[PlayerProjection]) -> None:
    """Log anti-stack pool stats once before a batch of ILP solves."""
    n = len(projections)
    n_pitchers = 0
    n_opp_groups = 0
    for pi in range(n):
        if not projections[pi].player.is_pitcher:
            continue
        opp = _opposing_team(projections[pi].player)
        if not opp:
            continue
        n_pitchers += 1
        has_opp_hitters = any(
            not projections[hj].player.is_pitcher
            and (projections[hj].player.team or "").strip().upper() == opp
            for hj in range(n)
        )
        if has_opp_hitters:
            n_opp_groups += 1
    logger.debug(
        f"Anti-stack: {n_pitchers} pitchers, {n_opp_groups} opposing hitter groups"
    )


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class LineupOptimizer:
    """DK MLB classic lineups via ILP (Monte Carlo by default).

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

        # Slot sequence — duplicate names appear for repeated positions.
        SLOT_NAMES = ["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]
        n_slots = len(SLOT_NAMES)  # 10

        prob = pulp.LpProblem("dk_lineup", pulp.LpMaximize)

        # -- Selection variables ---------------------------------------------
        # x[i] = 1 if player i is selected in the lineup.
        x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

        # -- Slot-assignment variables ---------------------------------------
        # z[i][j] = 1 when player i fills slot j; None when ineligible.
        # Using explicit slot-assignment prevents dual-eligible players from
        # simultaneously satisfying two slot constraints (the root cause of
        # infeasible post-solve assignments in a selection-only ILP).
        z: list[list] = [[None] * n_slots for _ in range(n)]
        for i, proj in enumerate(projections):
            elig = proj.player.position_eligibility
            for j, slot_name in enumerate(SLOT_NAMES):
                if slot_name in elig:
                    z[i][j] = pulp.LpVariable(f"z_{i}_{j}", cat="Binary")

        # -- Game indicator variables ----------------------------------------
        games = list({_game_key(p.player) for p in projections})
        y = {
            g: pulp.LpVariable(f"y_{gi}", cat="Binary")
            for gi, g in enumerate(games)
        }

        # -- Objective: maximise median projected points ---------------------
        prob += pulp.lpSum(projections[i].pts_q50 * x[i] for i in range(n))

        # -- Link x[i] to slot assignments: x[i] = Σ_j z[i][j] -------------
        for i in range(n):
            slot_vars = [z[i][j] for j in range(n_slots) if z[i][j] is not None]
            if slot_vars:
                prob += (
                    x[i] == pulp.lpSum(slot_vars),
                    f"x_link_{i}",
                )
            else:
                # No eligible slot for this player — exclude them.
                prob += x[i] == 0, f"x_ineligible_{i}"

        # -- Each player fills at most one slot ------------------------------
        for i in range(n):
            slot_vars = [z[i][j] for j in range(n_slots) if z[i][j] is not None]
            if len(slot_vars) > 1:
                prob += (
                    pulp.lpSum(slot_vars) <= 1,
                    f"player_once_{i}",
                )

        # -- Each slot filled by exactly one eligible player -----------------
        for j in range(n_slots):
            eligible_vars = [
                z[i][j] for i in range(n) if z[i][j] is not None
            ]
            if eligible_vars:
                prob += (
                    pulp.lpSum(eligible_vars) == 1,
                    f"slot_{j}",
                )
            else:
                logger.warning(
                    f"optimize_single: no eligible players for slot "
                    f"{j} ({SLOT_NAMES[j]}); problem is infeasible"
                )

        # -- Salary cap ------------------------------------------------------
        prob += (
            pulp.lpSum(projections[i].player.salary * x[i] for i in range(n))
            <= SALARY_CAP,
            "salary_cap",
        )

        # -- Salary floor (enforced to avoid lineups the validator rejects) --
        prob += (
            pulp.lpSum(projections[i].player.salary * x[i] for i in range(n))
            >= SALARY_FLOOR,
            "salary_floor",
        )

        # -- Exactly 10 players (redundant but aids branch-and-bound) --------
        prob += pulp.lpSum(x) == 10, "ten_players"

        # -- Minimum 2 games represented -------------------------------------
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

        # -- DK rule: max 5 hitters from any single team ---------------------
        hitter_teams = {
            projections[i].player.team
            for i in range(n)
            if not projections[i].player.is_pitcher
        }
        for team in hitter_teams:
            hitter_idx = [
                i for i in range(n)
                if projections[i].player.team == team
                and not projections[i].player.is_pitcher
            ]
            if len(hitter_idx) > 5:
                prob += (
                    pulp.lpSum(x[i] for i in hitter_idx) <= 5,
                    f"max_hitters_{team}",
                )

        # -- Anti-stack: no hitter from a pitcher's opposing team ------------
        for pi in range(n):
            if not projections[pi].player.is_pitcher:
                continue
            opp = _opposing_team(projections[pi].player)
            if not opp:
                continue
            opp_hitter_vars = [
                x[hj]
                for hj in range(n)
                if not projections[hj].player.is_pitcher
                and (projections[hj].player.team or "").strip().upper() == opp
            ]
            if opp_hitter_vars:
                m = len(opp_hitter_vars)
                prob += (
                    pulp.lpSum(opp_hitter_vars) <= m * (1 - x[pi]),
                    f"anti_stack_{pi}",
                )

        # -- Locked players --------------------------------------------------
        for i, proj in enumerate(projections):
            if proj.player.dk_id in _locked or proj.is_locked:
                prob += x[i] == 1, f"locked_{proj.player.dk_id}"

        # -- Banned players --------------------------------------------------
        for i, proj in enumerate(projections):
            if proj.player.dk_id in _banned or proj.is_banned:
                prob += x[i] == 0, f"banned_{proj.player.dk_id}"

        # -- Diversity: overlap with prior lineups ≤ min_overlap_penalty ----
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
            logger.warning(
                f"optimize_single: solver status={status!r}; returning None"
            )
            return None

        # -- Extract slot assignments from z variables -----------------------
        # z[i][j] == 1 tells us exactly which slot player i occupies.
        assigned_triples: list[tuple[DKPlayer, str, int]] = []
        for i in range(n):
            for j in range(n_slots):
                if (
                    z[i][j] is not None
                    and pulp.value(z[i][j]) is not None
                    and pulp.value(z[i][j]) > 0.5
                ):
                    assigned_triples.append(
                        (projections[i].player, SLOT_NAMES[j], j)
                    )
                    break  # Each player is in at most one slot.

        if len(assigned_triples) != 10:
            logger.warning(
                f"optimize_single: extracted {len(assigned_triples)} slot "
                f"assignments (expected 10); returning None"
            )
            return None

        # Sort into canonical DK order: P, P, C, 1B, 2B, 3B, SS, OF, OF, OF.
        assigned_triples.sort(key=lambda t: t[2])
        assigned_pairs = [(p, s) for p, s, _ in assigned_triples]

        selected = [
            projections[i] for i in range(n)
            if pulp.value(x[i]) is not None and pulp.value(x[i]) > 0.5
        ]
        total_salary = sum(p.salary for p, _ in assigned_pairs)
        projected_pts = sum(proj.pts_q50 for proj in selected)
        lev = float(np.mean([proj.leverage_score for proj in selected]))

        return LineupResult(
            players=assigned_pairs,
            total_salary=total_salary,
            projected_pts=projected_pts,
            leverage_score=lev,
            portfolio_score=0.0,  # Filled by generate_lineups after all are built.
            is_valid=False,
        )

    def assign_positions(
        self, selected: list[PlayerProjection]
    ) -> list[tuple[DKPlayer, str]]:
        """Assign DK roster slots to the selected 10 players via backtracking.

        A pure greedy pass fails for dual-eligible players: assigning
        Ohtani (1B/OF) to 1B may later leave the OF slots with too few
        eligible players.  This method solves that by:

        1. Sorting players so the *least flexible* are tried first
           (pitchers always first, then ascending eligibility count).
           Flexible players act as gap-fillers rather than slot-takers.
        2. Recursively trying every valid assignment for each slot in the
           fixed order ``P, C, 1B, 2B, 3B, SS, OF, OF, OF, UTIL``.
        3. Backtracking immediately when a slot cannot be filled, trying
           the next candidate for the previous slot.

        Falls back to the original greedy method (with a warning) only
        if backtracking exhausts all possibilities — which should never
        happen on a legally-constructed ILP solution.

        Args:
            selected: Exactly 10 ``PlayerProjection`` objects chosen
                by the ILP solver.

        Returns:
            List of ``(DKPlayer, roster_slot)`` tuples in the order the
            slots are assigned (P first, UTIL last).
        """
        # Slot sequence for DK MLB classic (2 P, no UTIL).
        slots = ["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]

        # Sort: pitchers first, then by ascending flexibility (fewest
        # eligible positions first so dual-eligibles remain available
        # for slots that need them).
        def _sort_key(proj: PlayerProjection) -> tuple:
            elig = proj.player.position_eligibility
            is_pitcher = int("P" not in elig)   # 0 = pitcher (sort first)
            return (is_pitcher, len(elig))

        pool = sorted(selected, key=_sort_key)

        # --- Backtracking solver -------------------------------------------

        def _eligible_for(proj: PlayerProjection, slot: str) -> bool:
            return slot in proj.player.position_eligibility

        def _backtrack(
            slot_idx: int,
            assignment: dict[str, str],   # dk_id → slot
        ) -> bool:
            """Return True and populate ``assignment`` when all slots filled."""
            if slot_idx == len(slots):
                return True

            slot = slots[slot_idx]
            for proj in pool:
                dk_id = proj.player.dk_id
                if dk_id in assignment:
                    continue
                if not _eligible_for(proj, slot):
                    continue
                assignment[dk_id] = slot
                if _backtrack(slot_idx + 1, assignment):
                    return True
                del assignment[dk_id]   # backtrack

            return False   # no candidate filled this slot

        assignment: dict[str, str] = {}
        success = _backtrack(0, assignment)

        if not success:
            logger.warning(
                "assign_positions: backtracking failed — falling back to greedy. "
                "This may indicate an infeasible position combination from the ILP."
            )
            return self._greedy_assign(selected)

        # Reconstruct in DK slot order for clean output.
        slot_order = {s: i for i, s in enumerate(slots)}
        result = sorted(
            [(proj.player, assignment[proj.player.dk_id]) for proj in selected],
            key=lambda t: (slot_order.get(t[1], 99), t[0].name),
        )
        return result

    def _greedy_assign(
        self, selected: list[PlayerProjection]
    ) -> list[tuple[DKPlayer, str]]:
        """Original greedy fallback; used only when backtracking fails."""
        assigned: dict[str, str] = {}
        remaining = list(selected)

        def _take(slot: str, check) -> None:
            for proj in remaining:
                if check(proj) and proj.player.dk_id not in assigned:
                    assigned[proj.player.dk_id] = slot
                    remaining.remove(proj)
                    return

        for _ in range(2):
            _take("P", lambda p: "P" in p.player.position_eligibility)
        for pos in ("C", "1B", "2B", "3B", "SS"):
            _take(pos, lambda p, s=pos: s in p.player.position_eligibility)
        for _ in range(3):
            _take("OF", lambda p: "OF" in p.player.position_eligibility)

        return [
            (proj.player, assigned.get(proj.player.dk_id, "P"))
            for proj in selected
        ]

    def _generate_lineups_sequential(
        self,
        projections: list[PlayerProjection],
        n_lineups: int,
        locked_ids: set[str] | None = None,
        banned_ids: set[str] | None = None,
        *,
        seed_previous: list[LineupResult] | None = None,
    ) -> list[LineupResult]:
        """Sequential PuLP solves with diversity overlap vs. all prior lineups.

        Each call to :meth:`optimize_single` sees ``previous_lineups`` equal to
        every lineup accumulated so far (including ``seed_previous`` when set).

        When ``seed_previous`` is provided, those rows are used only as
        diversity anchors; the returned list contains **only** the newly
        appended lineups (not the seed rows).

        Args:
            projections: Full slate player pool with projections.
            n_lineups: Number of **new** sequential solve attempts.
            locked_ids: Players forced into every lineup.
            banned_ids: Players excluded from every lineup.
            seed_previous: Optional lineups to treat as already generated.

        Returns:
            New ``LineupResult`` rows only (empty list if ``n_lineups`` is 0).
        """
        if not projections:
            logger.warning("_generate_lineups_sequential: empty projections list")
            return []

        results: list[LineupResult] = list(seed_previous) if seed_previous else []
        prev_len = len(results)

        _log_anti_stack_summary(projections)

        for attempt in range(1, n_lineups + 1):
            result = self.optimize_single(
                projections=projections,
                locked_ids=locked_ids,
                banned_ids=banned_ids,
                previous_lineups=results,
            )

            if result is None:
                logger.warning(
                    f"Lineup {attempt}/{n_lineups}: solver returned None — skipping"
                )
                continue

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
                f"Lineup seq {len(results) - prev_len:>2}/{n_lineups}: "
                f"salary=${result.total_salary:,}  "
                f"pts={result.projected_pts:.2f}  "
                f"valid={result.is_valid}"
            )

        new_tail = results[prev_len:] if seed_previous is not None else results
        valid_count = sum(1 for r in new_tail if r.is_valid)
        logger.info(
            f"_generate_lineups_sequential: {valid_count} valid of "
            f"{len(new_tail)} new lineups (attempts={n_lineups}, seed={prev_len})"
        )
        return new_tail

    def generate_lineups(
        self,
        projections: list[PlayerProjection],
        n_lineups: int = 20,
        locked_ids: set[str] | None = None,
        banned_ids: set[str] | None = None,
        *,
        use_monte_carlo: bool = True,
        n_simulations: int = 500,
    ) -> list[LineupResult]:
        """Generate lineups using Monte Carlo (default) or sequential ILP.

        Monte Carlo is the default: many independent solves on sampled
        projections, deduplication, scoring on original quantiles, exposure
        filtering, optional sequential top-up. Sequential mode preserves the
        legacy diversity-constraint ladder.

        Args:
            projections: Full slate player pool with projections.
            n_lineups: Number of lineups to return.
            locked_ids: Players that must appear in every lineup.
            banned_ids: Players excluded from every lineup.
            use_monte_carlo: Use :meth:`run_monte_carlo` when ``True``.
            n_simulations: Monte Carlo draw count when MC is enabled.

        Returns:
            Up to ``n_lineups`` ``LineupResult`` instances (possibly fewer).
        """
        if not projections:
            logger.warning("generate_lineups: empty projections list")
            return []

        if n_lineups <= 0:
            return []

        if use_monte_carlo:
            return self.run_monte_carlo(
                projections=projections,
                n_lineups=n_lineups,
                n_simulations=n_simulations,
                locked_ids=locked_ids,
                banned_ids=banned_ids,
            )

        return self._generate_lineups_sequential(
            projections=projections,
            n_lineups=n_lineups,
            locked_ids=locked_ids,
            banned_ids=banned_ids,
            seed_previous=None,
        )

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

    def simulate_projections(
        self,
        projections: list[PlayerProjection],
        seed: int | None = None,
    ) -> list[PlayerProjection]:
        """Draw one simulated projection for each player.

        For each player, sample a random DK points value from a truncated
        normal distribution defined by their quantile predictions:

        - mean = ``pts_q50``
        - std = ``(pts_q85 - pts_q15) / (2 * 1.04)`` — the 1.04 factor maps
          the q15–q85 interval to roughly ±1 standard deviation.

        The draw is truncated at ``0`` (no negative scores) and at
        ``max(pts_q85 * 1.5, mean + std)`` to cap extreme outliers.

        Returns a new list of ``PlayerProjection`` with ``pts_q50`` replaced
        by the simulated value; ``pts_q15`` and ``pts_q85`` are preserved for
        downstream portfolio scoring on the original slate.

        Args:
            projections: Original projections with q15 / q50 / q85.
            seed: Optional RNG seed for reproducibility.

        Returns:
            New list of projections with sampled ``pts_q50`` only.
        """
        from scipy import stats

        rng = np.random.default_rng(seed)
        simulated: list[PlayerProjection] = []

        for proj in projections:
            mean = proj.pts_q50
            interval = proj.pts_q85 - proj.pts_q15
            std = max(interval / (2 * 1.04), 0.01)

            low = 0.0
            high = max(proj.pts_q85 * 1.5, mean + std)

            a = (low - mean) / std
            b = (high - mean) / std
            if b <= a:
                b = a + 1e-6

            sampled = float(
                stats.truncnorm.rvs(a, b, loc=mean, scale=std, random_state=rng)
            )

            simulated.append(
                replace(
                    proj,
                    pts_q50=sampled,
                )
            )

        return simulated

    def run_monte_carlo(
        self,
        projections: list[PlayerProjection],
        n_lineups: int = 20,
        n_simulations: int = 500,
        locked_ids: set[str] | None = None,
        banned_ids: set[str] | None = None,
        n_jobs: int = 4,
    ) -> list[LineupResult]:
        """Generate diverse lineups via Monte Carlo simulation.

        Algorithm:

        1. Run ``n_simulations`` independent ILP solves in parallel; each
           uses ``simulate_projections`` (no diversity / overlap penalties).
        2. Drop ``None`` (infeasible or failed) results.
        3. Deduplicate by ``frozenset`` of player ``dk_id`` values.
        4. Recompute ``projected_pts`` / ``leverage_score`` on **original**
           projections, then assign ``portfolio_score`` via
           ``calculate_portfolio_score``.
        5. Sort by ``portfolio_score`` descending.
        6. Greedy exposure pass: add lineups only if no player would exceed
           ``max_exposure`` times ``n_lineups`` appearances (per-player cap).
        7. If fewer than ``n_lineups`` remain, fill with sequential ILP
           (seeded with already-selected lineups).
        8. Re-validate each final lineup; return up to ``n_lineups`` rows.

        On total failure (no valid MC lineups) or unexpected errors, falls
        back to :meth:`_generate_lineups_sequential`.

        Args:
            projections: Player projections with q15 / q50 / q85.
            n_lineups: Number of final lineups to return.
            n_simulations: Number of Monte Carlo ILP draws.
            locked_ids: Players forced into every lineup.
            banned_ids: Players excluded from all lineups.
            n_jobs: Parallel workers (``joblib``; solver uses ``msg=0``,
                ``threads=1`` per worker).

        Returns:
            Up to ``n_lineups`` ``LineupResult`` instances.
        """
        if not projections:
            logger.warning("run_monte_carlo: empty projections list")
            return []

        if n_lineups <= 0:
            return []

        _locked = locked_ids or set()
        _banned = banned_ids or set()

        try:
            from joblib import Parallel, delayed

            _log_anti_stack_summary(projections)

            logger.info(
                f"Monte Carlo: {n_simulations} simulations, "
                f"target={n_lineups} lineups, n_jobs={n_jobs}"
            )

            raw_results: list[LineupResult | None] = Parallel(
                n_jobs=n_jobs,
                verbose=0,
            )(
                delayed(_monte_carlo_single_simulation)(
                    self,
                    projections,
                    seed,
                    _locked,
                    _banned,
                )
                for seed in range(n_simulations)
            )

            valid_results = [r for r in raw_results if r is not None]
            logger.info(
                f"Monte Carlo: {len(valid_results)}/{n_simulations} "
                f"simulations produced valid lineups"
            )

            if not valid_results:
                logger.warning(
                    "Monte Carlo: no valid lineups — falling back to sequential"
                )
                return self._generate_lineups_sequential(
                    projections=projections,
                    n_lineups=n_lineups,
                    locked_ids=locked_ids,
                    banned_ids=banned_ids,
                    seed_previous=None,
                )

            seen: set[frozenset[str]] = set()
            unique_results: list[LineupResult] = []
            for result in valid_results:
                key = frozenset(result.player_dk_ids)
                if key not in seen:
                    seen.add(key)
                    unique_results.append(result)

            logger.info(
                f"Monte Carlo: {len(unique_results)} unique lineups "
                f"from {len(valid_results)} valid"
            )

            proj_by_id = {p.player.dk_id: p for p in projections}
            for result in unique_results:
                lineup_projs = [
                    proj_by_id[p.dk_id]
                    for p, _ in result.players
                    if p.dk_id in proj_by_id
                ]
                if lineup_projs:
                    result.projected_pts = float(
                        sum(p.pts_q50 for p in lineup_projs)
                    )
                    result.leverage_score = float(
                        np.mean([p.leverage_score for p in lineup_projs])
                    )
                result.portfolio_score = self.calculate_portfolio_score(
                    result, projections
                )

            unique_results.sort(
                key=lambda r: r.portfolio_score,
                reverse=True,
            )

            max_appearances: dict[str, int] = {}
            selected: list[LineupResult] = []

            for result in unique_results:
                if len(selected) >= n_lineups:
                    break

                would_violate = False
                for player, _ in result.players:
                    current = max_appearances.get(player.dk_id, 0)
                    proj = proj_by_id.get(player.dk_id)
                    max_exp = proj.max_exposure if proj else 0.70
                    if (current + 1) / max(float(n_lineups), 1.0) > max_exp:
                        would_violate = True
                        break

                if not would_violate:
                    selected.append(result)
                    for player, _ in result.players:
                        max_appearances[player.dk_id] = (
                            max_appearances.get(player.dk_id, 0) + 1
                        )

            logger.info(
                f"Monte Carlo: {len(selected)} lineups after "
                f"exposure filtering (target={n_lineups})"
            )

            if len(selected) < n_lineups:
                remaining = n_lineups - len(selected)
                logger.info(
                    f"Monte Carlo: filling {remaining} lineups "
                    f"with sequential optimizer"
                )
                filler = self._generate_lineups_sequential(
                    projections=projections,
                    n_lineups=remaining,
                    locked_ids=locked_ids,
                    banned_ids=banned_ids,
                    seed_previous=list(selected),
                )
                selected.extend(filler)

            for result in selected[:n_lineups]:
                validated = self._validator.validate(result.players)
                result.is_valid = validated.is_valid

            valid_count = sum(1 for r in selected[:n_lineups] if r.is_valid)
            logger.info(
                f"Monte Carlo complete: {valid_count} valid of "
                f"{min(len(selected), n_lineups)} final lineups"
            )

            return selected[:n_lineups]

        except Exception as exc:  # noqa: BLE001
            logger.exception(
                f"Monte Carlo failed ({exc}); falling back to sequential ILP"
            )
            return self._generate_lineups_sequential(
                projections=projections,
                n_lineups=n_lineups,
                locked_ids=locked_ids,
                banned_ids=banned_ids,
                seed_previous=None,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_eligibility(
        self, projections: list[PlayerProjection]
    ) -> dict[str, list[int]]:
        """Build slot → [player index] eligibility map for the classic (2 P) format."""
        eligible: dict[str, list[int]] = {slot: [] for slot in _REQUIRED_SLOTS}

        for i, proj in enumerate(projections):
            for pos in proj.player.position_eligibility:
                if pos in eligible:
                    eligible[pos].append(i)

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
            return pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, threads=1)
