"""Ownership proxy model for DraftKings MLB GPP optimizer.

Phase 1: Heuristic model combining:

1. Simulation frequency (10k Monte Carlo sims) — backbone
2. Vegas implied team totals — human bias signal
3. Salary-tier normalized value z-score — value perception
4. Salary rank by position — public trusts DK pricing
5. Recency (rolling 7d xwOBA) — hot player bias (``pts_q50`` pool proxy until
   Statcast recency is wired into slate inference)

Pitchers and hitters use separate weight sets.

Phase 2 (after 30+ slates): Replace with XGBoost trained
on actual DK contest ownership CSVs.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_pipeline.ingestion.dk_csv_parser import DKPlayer  # noqa: E402
from data_pipeline.ingestion.odds_ingestion import OddsIngestion  # noqa: E402
from optimizer.constraints.lineup_optimizer import (  # noqa: E402
    LineupOptimizer,
    PlayerProjection,
)

# ---------------------------------------------------------------------------
# Weight sets — separate for pitchers and hitters
# ---------------------------------------------------------------------------

PITCHER_WEIGHTS = {
    "sim_frequency": 0.60,
    "value_zscore": 0.20,
    "salary_rank": 0.10,
    "recency_7d": 0.05,
    "vegas_inverse": 0.05,  # facing high-total team = lower ownership
}

HITTER_WEIGHTS = {
    "sim_frequency": 0.25,
    "vegas_modifier": 0.35,
    "value_zscore": 0.25,
    "recency_7d": 0.10,
    "salary_rank": 0.05,
}

# Slate-size-based ownership caps (per-player ceiling, fraction → %)
HITTER_CAPS = {
    "small": 0.28,
    "medium": 0.20,
    "large": 0.12,
}
PITCHER_CAPS = {
    "small": 0.40,
    "medium": 0.30,
    "large": 0.20,
}
OWNERSHIP_FLOOR = 0.01  # legacy / global floor; normalization uses 2% / 3%

_DEFAULT_HITTER_OWN = 10.0
_DEFAULT_PITCHER_OWN = 15.0


def _ownership_sim_worker(
    optimizer: LineupOptimizer,
    projections: list[PlayerProjection],
    seed: int,
) -> list[str] | None:
    """Single simulated slate + ILP solve; returns selected ``dk_id`` list."""
    try:
        sim_projs = optimizer.simulate_projections(projections, seed=seed)
        result = optimizer.optimize_single(
            projections=sim_projs,
            locked_ids=None,
            banned_ids=None,
            previous_lineups=None,
            min_overlap_penalty=10,
        )
        if result is None:
            return None
        return [p.dk_id for p, _ in result.players]
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Ownership sim seed={seed} failed: {exc}")
        return None


def _opponent_team(player: DKPlayer) -> str:
    """Opponent abbreviation for a player (same logic as salary CSV fields)."""
    if player.team and player.home_team and player.away_team:
        return player.away_team if player.team == player.home_team else player.home_team
    return ""


def _position_bucket(player: DKPlayer) -> str:
    """Primary roster bucket for salary-rank grouping (DK ``Position`` first slot)."""
    raw = (player.dk_position or "").strip()
    if not raw:
        return "UNK"
    return raw.split("/")[0].strip() or "UNK"


class OwnershipProjector:
    """Project DFS ownership percentages for a given slate.

    Usage::

        projector = OwnershipProjector()
        ownership = projector.project(
            players=dk_players,
            base_projections=player_projections,
            n_sims=10000,
        )
        # ownership maps dk_id -> percentage in [0, 100]
    """

    def __init__(self) -> None:
        """Build optimizer and odds clients used across projections."""
        self._optimizer = LineupOptimizer()
        self._odds = OddsIngestion()
        logger.debug("OwnershipProjector ready")

    def project(
        self,
        players: list[DKPlayer],
        base_projections: list[PlayerProjection],
        n_sims: int = 10000,
        n_jobs: int = 4,
    ) -> dict[str, float]:
        """Project ownership percentages for all players on the slate.

        Combines Monte Carlo lineup frequency with Vegas, value z-score,
        salary rank, and a recency proxy. On any failure, returns a flat
        ownership map (10% hitters / 15% pitchers) so callers never break.

        Args:
            players: Full DK player pool for this slate.
            base_projections: Initial ``PlayerProjection`` rows (``pts_q50``
                from DK avg or model).
            n_sims: Monte Carlo lineup draws for frequency (default 10_000).
            n_jobs: Parallel workers for simulation draws.

        Returns:
            Mapping ``dk_id`` → ownership percentage in ``[0, 100]``.
        """
        if not players or not base_projections:
            logger.warning("project: empty players or projections")
            return {}

        try:
            n_games = len(
                {
                    f"{p.away_team}@{p.home_team}"
                    for p in players
                    if p.away_team and p.home_team
                }
            )
            slate_size = self._get_slate_size(n_games)
            logger.info(
                f"OwnershipProjector: {len(players)} players, "
                f"{n_games} games, slate_size={slate_size}"
            )

            sim_freq = self._run_simulation_frequency(
                base_projections, n_sims=n_sims, n_jobs=n_jobs
            )
            if not sim_freq:
                logger.warning(
                    "Simulation frequency empty — using uniform backbone"
                )
                sim_freq = self._uniform_frequency(base_projections)

            implied_totals = self._odds.get_team_implied_totals()
            if not implied_totals:
                logger.warning(
                    "No Vegas data available — vegas-driven terms use defaults"
                )

            hitter_projs = [p for p in base_projections if not p.player.is_pitcher]
            pitcher_projs = [p for p in base_projections if p.player.is_pitcher]

            hitter_value_zscores = self._calc_value_zscores(hitter_projs)
            pitcher_value_zscores = self._calc_value_zscores(pitcher_projs)
            value_zscores = {**hitter_value_zscores, **pitcher_value_zscores}

            salary_ranks = self._calc_salary_ranks(base_projections)

            raw_scores: dict[str, float] = {}

            for proj in base_projections:
                dk_id = proj.player.dk_id
                team = proj.player.team or ""
                is_pitcher = proj.player.is_pitcher

                sim_f = float(np.clip(sim_freq.get(dk_id, 0.0), 0.0, 1.0))

                val_z = float(np.clip(value_zscores.get(dk_id, 0.0), -3.0, 3.0))
                val_score = float(1.0 / (1.0 + np.exp(-val_z)))

                sal_rank = float(np.clip(salary_ranks.get(dk_id, 0.5), 0.0, 1.0))

                recency = self._normalize_recency(
                    proj.pts_q50, base_projections, is_pitcher
                )

                if is_pitcher:
                    opp_team = _opponent_team(proj.player)
                    opp_implied = float(implied_totals.get(opp_team, 4.5))
                    vegas_inv = max(0.0, 1.0 - (opp_implied - 4.5) * 0.08)

                    raw = (
                        sim_f * PITCHER_WEIGHTS["sim_frequency"]
                        + val_score * PITCHER_WEIGHTS["value_zscore"]
                        + sal_rank * PITCHER_WEIGHTS["salary_rank"]
                        + recency * PITCHER_WEIGHTS["recency_7d"]
                        + vegas_inv * PITCHER_WEIGHTS["vegas_inverse"]
                    )
                else:
                    team_implied = float(implied_totals.get(team, 4.5))
                    vegas_mod = max(0.0, (team_implied - 4.5) * 0.03)
                    vegas_score = float(np.clip(vegas_mod / 0.15 + 0.5, 0.0, 1.0))

                    stack_mult = self._calc_stack_multiplier(
                        proj, base_projections, team_implied
                    )

                    raw = (
                        sim_f * HITTER_WEIGHTS["sim_frequency"]
                        + vegas_score * HITTER_WEIGHTS["vegas_modifier"]
                        + val_score * HITTER_WEIGHTS["value_zscore"]
                        + recency * HITTER_WEIGHTS["recency_7d"]
                        + sal_rank * HITTER_WEIGHTS["salary_rank"]
                    ) * stack_mult

                raw_scores[dk_id] = max(0.0, float(raw))

            ownership = self._apply_sigmoid_normalization(
                raw_scores=raw_scores,
                base_projections=base_projections,
                slate_size=slate_size,
            )

            sorted_own = sorted(ownership.items(), key=lambda x: -x[1])
            logger.info("Top 10 projected ownership:")
            proj_names = {p.player.dk_id: p.player.name for p in base_projections}
            for dk_id, pct in sorted_own[:10]:
                name = proj_names.get(dk_id, dk_id)
                logger.info(f"  {name:<25} {pct:.1f}%")

            return ownership

        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Ownership projection failed: {exc}")
            return self._flat_ownership(base_projections)

    def _flat_ownership(
        self, base_projections: list[PlayerProjection]
    ) -> dict[str, float]:
        """Fallback ownership map when projection pipeline fails."""
        out: dict[str, float] = {}
        for p in base_projections:
            own = (
                _DEFAULT_PITCHER_OWN if p.player.is_pitcher else _DEFAULT_HITTER_OWN
            )
            out[p.player.dk_id] = own
        return out

    def _uniform_frequency(
        self, projections: list[PlayerProjection]
    ) -> dict[str, float]:
        """Assign equal simulation backbone mass when sims yield no lineups."""
        if not projections:
            return {}
        w = 1.0 / len(projections)
        return {p.player.dk_id: w for p in projections}

    def _run_simulation_frequency(
        self,
        projections: list[PlayerProjection],
        n_sims: int,
        n_jobs: int,
    ) -> dict[str, float]:
        """Count how often each player appears across ``n_sims`` optimal lineups.

        Uses :meth:`LineupOptimizer.simulate_projections` plus a single ILP
        solve per draw (no diversity constraints). Returns normalized
        frequencies in ``[0, 1]`` keyed by ``dk_id``.

        Args:
            projections: Slate projections used for simulation draws.
            n_sims: Number of independent draws.
            n_jobs: ``joblib`` worker count.

        Returns:
            Empty dict if no successful simulations.
        """
        from joblib import Parallel, delayed

        logger.info(f"Running {n_sims} ownership sims (n_jobs={n_jobs})...")

        try:
            results: list[list[str] | None] = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(_ownership_sim_worker)(self._optimizer, projections, s)
                for s in range(n_sims)
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Parallel ownership sims failed: {exc}")
            return {}

        counts: dict[str, int] = defaultdict(int)
        valid = 0
        for r in results:
            if r is not None:
                valid += 1
                for dk_id in r:
                    counts[dk_id] += 1

        if valid == 0:
            logger.warning("No valid sims for ownership frequency")
            return {}

        freq = {dk_id: count / valid for dk_id, count in counts.items()}
        logger.info(f"Ownership sims: {valid}/{n_sims} valid")
        return freq

    def _calc_value_zscores(
        self, projections: list[PlayerProjection]
    ) -> dict[str, float]:
        """Salary-tier normalized value z-scores (``pts_q50`` per $1k salary).

        Splits the pool into salary tiers, computes raw value within each
        tier, then z-scores within tier so cheap and expensive players are
        comparable.

        Args:
            projections: Pool slice (hitters or pitchers).

        Returns:
            ``dk_id`` → z-score (0 for empty pool or degenerate tier).
        """
        if not projections:
            return {}

        df = pd.DataFrame(
            [
                {
                    "dk_id": p.player.dk_id,
                    "salary": p.player.salary,
                    "pts_q50": p.pts_q50,
                }
                for p in projections
            ]
        )

        df["raw_value"] = df["pts_q50"] / (df["salary"] / 1000.0).replace(0, np.nan)
        df["raw_value"] = df["raw_value"].fillna(0.0)

        bins = [0, 3000, 5000, 7000, 20000]
        labels = ["budget", "mid", "premium", "elite"]
        df["tier"] = pd.cut(
            df["salary"],
            bins=bins,
            labels=labels,
            right=True,
        )

        def _zscore(series: pd.Series) -> pd.Series:
            std = float(series.std())
            if std == 0.0 or np.isnan(std):
                return pd.Series(0.0, index=series.index)
            return (series - series.mean()) / std

        df["value_zscore"] = df.groupby("tier", observed=True)[
            "raw_value"
        ].transform(_zscore)
        df["value_zscore"] = df["value_zscore"].fillna(0.0)

        return dict(zip(df["dk_id"], df["value_zscore"]))

    def _calc_salary_ranks(
        self, projections: list[PlayerProjection]
    ) -> dict[str, float]:
        """Normalized salary rank (0–1) within each DK position bucket.

        ``1.0`` is the highest salary in that bucket.

        Args:
            projections: Full projection list.

        Returns:
            ``dk_id`` → rank in ``[0, 1]``.
        """
        if not projections:
            return {}

        groups: dict[str, list[PlayerProjection]] = defaultdict(list)
        for p in projections:
            pos = _position_bucket(p.player)
            groups[pos].append(p)

        ranks: dict[str, float] = {}
        for _pos, group in groups.items():
            sorted_group = sorted(group, key=lambda x: x.player.salary)
            n = len(sorted_group)
            denom = max(n - 1, 1)
            for i, proj in enumerate(sorted_group):
                ranks[proj.player.dk_id] = i / denom

        return ranks

    def _normalize_recency(
        self,
        pts_q50: float,
        all_projections: list[PlayerProjection],
        is_pitcher: bool,
    ) -> float:
        """Min–max normalize ``pts_q50`` within pitcher/hitter pool to ``[0, 1]``.

        Temporary stand-in for rolling 7d xwOBA until Statcast recency is
        joined in slate inference.

        Args:
            pts_q50: Player median projection.
            all_projections: Full slate projections.
            is_pitcher: Which pool to normalize against.

        Returns:
            ``0.5`` if the pool is empty or degenerate.
        """
        pool = [
            p.pts_q50
            for p in all_projections
            if p.player.is_pitcher == is_pitcher
        ]
        if not pool:
            return 0.5
        mn, mx = min(pool), max(pool)
        if mx == mn:
            return 0.5
        return float(np.clip((pts_q50 - mn) / (mx - mn), 0.0, 1.0))

    def _calc_stack_multiplier(
        self,
        proj: PlayerProjection,
        all_projections: list[PlayerProjection],
        team_implied: float,
    ) -> float:
        """Team stack popularity multiplier for hitters.

        Higher implied team totals raise the base multiplier; salary order
        within the team proxies batting-order correlation with stacks.

        Args:
            proj: Hitter projection row.
            all_projections: Full slate (for teammates).
            team_implied: That team's implied run total.

        Returns:
            Multiplier clipped to roughly ``[0.7, 1.5]``.
        """
        base = 1.0 + max(0.0, (team_implied - 4.5) * 0.06)

        team = proj.player.team or ""
        team_hitters = [
            p
            for p in all_projections
            if p.player.team == team and not p.player.is_pitcher
        ]
        if not team_hitters:
            return float(np.clip(base, 0.7, 1.5))

        team_hitters_sorted = sorted(
            team_hitters,
            key=lambda x: x.player.salary,
            reverse=True,
        )
        try:
            rank = next(
                i
                for i, p in enumerate(team_hitters_sorted)
                if p.player.dk_id == proj.player.dk_id
            )
        except StopIteration:
            rank = len(team_hitters_sorted) // 2

        n = len(team_hitters_sorted)
        order_factor = 1.0 - (rank / max(n, 1)) * 0.3
        return float(np.clip(base * order_factor, 0.7, 1.5))

    def _apply_sigmoid_normalization(
        self,
        raw_scores: dict[str, float],
        base_projections: list[PlayerProjection],
        slate_size: str,
    ) -> dict[str, float]:
        """Map raw scores to ownership percentages (GPP-shaped spread).

        Mean-centers raw scores, applies a signed power transform
        (exponent 0.7) to widen gaps, then a sigmoid with gain 4. Linearly
        rescales sigmoids so the pool minimum maps to a 3% floor (pitchers)
        or 2% (hitters) and the maximum to the slate-size cap. Equal-sigmoid
        degeneracy uses the mid-point between floor and cap. Not softmax.

        Args:
            raw_scores: Heuristic scores keyed by ``dk_id``.
            base_projections: Slate rows for pitcher / hitter split.
            slate_size: ``"small"``, ``"medium"``, or ``"large"``.

        Returns:
            ``dk_id`` → ownership percent in ``[pool_floor, cap*100]``.
        """
        proj_by_id = {p.player.dk_id: p for p in base_projections}

        pitcher_ids = [
            dk_id for dk_id, p in proj_by_id.items() if p.player.is_pitcher
        ]
        hitter_ids = [
            dk_id for dk_id, p in proj_by_id.items() if not p.player.is_pitcher
        ]

        result: dict[str, float] = {}

        for group_ids, is_pitcher in (
            (pitcher_ids, True),
            (hitter_ids, False),
        ):
            if not group_ids:
                continue

            cap = (
                PITCHER_CAPS[slate_size]
                if is_pitcher
                else HITTER_CAPS[slate_size]
            )

            scores = np.array(
                [raw_scores.get(i, 0.0) for i in group_ids],
                dtype=float,
            )

            # Power transform to increase spread between players
            scores_shifted = scores - scores.mean()
            scores_amplified = np.sign(scores_shifted) * (
                np.abs(scores_shifted) ** 0.7
            )
            sig = 1.0 / (
                1.0
                + np.exp(-np.clip(scores_amplified * 4.0, -50.0, 50.0))
            )

            sig_min = float(sig.min())
            sig_max = float(sig.max())

            floor_pct = 3.0 if is_pitcher else 2.0
            cap_pct = cap * 100.0

            if sig_max > sig_min:
                scaled = floor_pct + (sig - sig_min) / (sig_max - sig_min) * (
                    cap_pct - floor_pct
                )
            else:
                avg = (floor_pct + cap_pct) / 2.0
                scaled = np.full(len(group_ids), avg, dtype=float)

            for dk_id, pct in zip(group_ids, scaled):
                result[str(dk_id)] = float(
                    np.clip(pct, floor_pct, cap_pct)
                )

        return result

    @staticmethod
    def _get_slate_size(n_games: int) -> str:
        """Map game count to slate-size bucket for ownership caps."""
        if n_games <= 4:
            return "small"
        if n_games <= 9:
            return "medium"
        return "large"

    def update_projections_with_ownership(
        self,
        projections: list[PlayerProjection],
        ownership: dict[str, float],
    ) -> list[PlayerProjection]:
        """Return projections with ``ownership_proj`` and GPP leverage.

        Replaces flat ownership defaults with ``ownership`` outputs and sets
        ``leverage_score`` from ceiling projection (``pts_q85``) divided by
        projected ownership percent (ceiling-based GPP edge).

        Args:
            projections: Original projection rows.
            ownership: ``dk_id`` → percent in ``(0, 100]``.

        Returns:
            New list with updated fields (original rows untouched).
        """
        updated: list[PlayerProjection] = []
        for proj in projections:
            default = (
                _DEFAULT_PITCHER_OWN if proj.player.is_pitcher else _DEFAULT_HITTER_OWN
            )
            own_pct = float(ownership.get(proj.player.dk_id, default))
            own_pct = max(own_pct, 0.1)

            leverage = proj.pts_q85 / own_pct

            updated.append(
                replace(
                    proj,
                    ownership_proj=round(own_pct, 2),
                    leverage_score=round(leverage, 3),
                )
            )

        return updated
