"""Slate inference pipeline for DraftKings GPP optimizer.

Builds feature vectors for today's DK players by matching
them to cached Statcast data, then runs trained XGBoost
models to produce q15/q50/q85 projections.

For players with no Statcast match, falls back to DK avg.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from loguru import logger
from rapidfuzz import fuzz, process

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_pipeline.ingestion.dk_csv_parser import DKPlayer
from data_pipeline.loaders.parquet_cache import ParquetCache
from ml.features.feature_engineer import FeatureEngineer
from ml.features.pitcher_feature_engineer import PitcherFeatureEngineer
from ml.training.points_model import PitcherPointsModel, PointsModel
from optimizer.constraints.lineup_optimizer import PlayerProjection


class SlateInference:
    """Build real model projections for a DK slate.

    Matches DK players to Statcast MLBAM IDs via fuzzy name
    matching, extracts their most recent rolling feature row,
    and runs trained XGBoost quantile models.

    Falls back to DK avg_points_per_game for unmatched players.

    Usage::

        inference = SlateInference()
        inference.load_models()
        projections = inference.build_projections(dk_players)
    """

    # Minimum fuzzy match score for name matching (0-100)
    NAME_MATCH_THRESHOLD = 82

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Convert 'Last, First' to 'First Last' for matching."""
        name = name.strip()
        if "," in name:
            parts = name.split(",", 1)
            return f"{parts[1].strip()} {parts[0].strip()}"
        return name

    def __init__(self, cache_dir: str = "data/parquet") -> None:
        self.cache = ParquetCache(cache_dir)
        self._hitter_model: PointsModel | None = None
        self._pitcher_model: PitcherPointsModel | None = None
        self._hitter_features: pd.DataFrame | None = None
        self._pitcher_features: pd.DataFrame | None = None
        self._name_to_mlbam: dict[str, int] = {}
        self._normalized_name_map: dict[str, int] = {}
        self._mlbam_to_name: dict[int, str] = {}
        self._mlbam_to_team: dict[int, str] = {}
        logger.debug("SlateInference ready")

    def load_models(self) -> None:
        """Load the most recent trained hitter and pitcher models."""
        models_dir = Path("data/models")

        # Hitter model
        hitter_files = sorted(models_dir.glob("points_q50_*.joblib"))
        if hitter_files:
            run_id = hitter_files[-1].stem.replace("points_q50_", "")
            self._hitter_model = PointsModel()
            self._hitter_model.load_models(run_id)
            logger.info(f"Loaded hitter model: {run_id}")
        else:
            logger.warning("No hitter model found")

        # Pitcher model
        pitcher_files = sorted(models_dir.glob("pitcher_q50_*.joblib"))
        if pitcher_files:
            run_id = pitcher_files[-1].stem.replace("pitcher_q50_", "")
            self._pitcher_model = PitcherPointsModel()
            self._pitcher_model.load_models(run_id)
            logger.info(f"Loaded pitcher model: {run_id}")
        else:
            logger.warning("No pitcher model found")

    def load_feature_matrices(self) -> None:
        """Load cached feature matrices and build name lookup.

        Loads:
        - features/batter_feature_matrix_game_level
        - features/pitcher_feature_matrix_game_level

        Builds name -> MLBAM ID lookup from Statcast data
        for fuzzy player matching.
        """
        # Load hitter features
        hitter_df = self.cache.load("features/batter_feature_matrix_game_level")
        if hitter_df is not None and not hitter_df.empty:
            self._hitter_features = hitter_df
            logger.info(f"Loaded hitter features: {len(hitter_df):,} rows")
        else:
            logger.warning("No hitter feature matrix found")

        # Load pitcher features
        pitcher_df = self.cache.load("features/pitcher_feature_matrix_game_level")
        if pitcher_df is not None and not pitcher_df.empty:
            self._pitcher_features = pitcher_df
            logger.info(f"Loaded pitcher features: {len(pitcher_df):,} rows")
        else:
            logger.warning("No pitcher feature matrix found")

        # Build name lookup from Statcast
        self._build_name_lookup()

    def _build_name_lookup(self) -> None:
        """Build player_name -> MLBAM ID lookup.

        Uses two sources:
        1. Statcast pitcher data: player_name column has
           pitcher names in 'Last, First' format mapped to
           pitcher MLBAM IDs
        2. pybaseball playerid_lookup for batter MLBAM IDs
           mapped to full names

        Merges both into one unified lookup.
        """
        name_map: dict[str, int] = {}

        # SOURCE 1: Pitcher names from Statcast player_name column
        # (player_name in Statcast = pitcher name)
        for year in [2026, 2025]:
            key = f"statcast/pitchers_{year}"
            df = self.cache.load(key)
            if df is None or df.empty:
                continue
            if "player_name" not in df.columns:
                continue
            pairs = (
                df[["player_name", "pitcher"]]
                .dropna()
                .drop_duplicates(subset=["player_name"])
            )
            for _, row in pairs.iterrows():
                name = str(row["player_name"]).strip()
                mlbam_id = int(row["pitcher"])
                if name and mlbam_id:
                    name_map[name] = mlbam_id
            if name_map:
                logger.info(
                    f"Pitcher names from Statcast {year}: "
                    f"{len(name_map)}"
                )
                break

        # SOURCE 2: Batter names from pybaseball playerid_lookup
        # Get all unique batter MLBAM IDs from feature matrix
        batter_ids: list[int] = []
        if self._hitter_features is not None:
            id_col = "batter" if "batter" in self._hitter_features.columns else None
            if id_col:
                batter_ids = (
                    self._hitter_features[id_col]
                    .dropna()
                    .astype(int)
                    .unique()
                    .tolist()
                )

        if batter_ids:
            try:
                # playerid_lookup needs last/first name
                # Instead use chadwick bureau register which
                # has MLBAM IDs
                from pybaseball import chadwick_register

                logger.info("Loading Chadwick register for batter names...")
                chadwick = chadwick_register(save=True)

                # Filter to our batter IDs
                mlbam_col = "key_mlbam"
                name_cols_available = [
                    c for c in ["name_first", "name_last"] if c in chadwick.columns
                ]

                if mlbam_col in chadwick.columns and name_cols_available:
                    chadwick_filtered = chadwick[
                        chadwick[mlbam_col].isin(batter_ids)
                    ].dropna(subset=[mlbam_col])

                    for _, row in chadwick_filtered.iterrows():
                        mlbam_id = int(row[mlbam_col])
                        first = str(row.get("name_first", "")).strip()
                        last = str(row.get("name_last", "")).strip()
                        if first and last:
                            full_name = f"{first} {last}"
                            name_map[full_name] = mlbam_id

                    logger.info(
                        f"Batter names from Chadwick: "
                        f"{len(chadwick_filtered)} players"
                    )
            except Exception as exc:
                logger.warning(
                    f"Chadwick register failed: {exc} — "
                    f"batter names unavailable"
                )

        self._name_to_mlbam = name_map
        self._normalized_name_map = {
            self._normalize_name(k): v for k, v in name_map.items()
        }
        self._mlbam_to_name = {v: k for k, v in name_map.items()}
        self._mlbam_to_team = self._build_mlbam_to_team()
        logger.info(
            f"Name lookup built: {len(name_map)} total players "
            f"(pitchers + batters)"
        )

    def _build_mlbam_to_team(self) -> dict[int, str]:
        """Map MLBAM ID to most recent ``home_team`` from Statcast."""
        team_map: dict[int, str] = {}
        for year in [2026, 2025]:
            for key, id_col in [
                (f"statcast/batters_{year}", "batter"),
                (f"statcast/pitchers_{year}", "pitcher"),
            ]:
                df = self.cache.load(key)
                if df is None or df.empty:
                    continue
                if id_col not in df.columns or "home_team" not in df.columns:
                    continue
                if "game_date" not in df.columns:
                    continue
                pairs = (
                    df.sort_values("game_date")
                    .groupby(id_col)["home_team"]
                    .last()
                    .reset_index()
                )
                for _, row in pairs.iterrows():
                    mlbam_id = int(row[id_col])
                    team = str(row["home_team"])
                    if mlbam_id and team:
                        team_map[mlbam_id] = team
            if team_map:
                break
        return team_map

    def _match_player(self, player: DKPlayer) -> int | None:
        """Fuzzy match a DK player name to an MLBAM ID.

        Uses rapidfuzz on normalized names; when multiple candidates score
        similarly, prefers the MLBAM whose Statcast team aligns with the DK
        player's team.

        Args:
            player: DKPlayer from the parsed salary CSV.

        Returns:
            MLBAM ID if match found above threshold, else None.
        """
        if not self._name_to_mlbam:
            return None

        clean_name = player.name.strip()

        # 1. Exact match original
        if clean_name in self._name_to_mlbam:
            return self._name_to_mlbam[clean_name]

        # 2. Exact match normalized
        if clean_name in self._normalized_name_map:
            return self._normalized_name_map[clean_name]

        # 3. Fuzzy match on normalized names (top 5 + team tiebreaker)
        candidates = list(self._normalized_name_map.keys())
        results = process.extract(
            clean_name,
            candidates,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=self.NAME_MATCH_THRESHOLD,
            limit=5,
        )

        if not results:
            logger.debug(f"No match for {player.name!r}")
            return None

        if len(results) == 1:
            matched_name, score, _ = results[0]
            mlbam_id = self._normalized_name_map[matched_name]
            logger.debug(
                f"Matched {player.name!r} → {matched_name!r} "
                f"(score={score}, mlbam={mlbam_id})"
            )
            return mlbam_id

        player_team = (player.team or "").upper()

        for matched_name, score, _ in results:
            mlbam_id = self._normalized_name_map[matched_name]
            candidate_team = self._mlbam_to_team.get(mlbam_id, "")
            if player_team and candidate_team:
                cu = candidate_team.upper()
                if (
                    player_team in cu
                    or cu in player_team
                ):
                    logger.debug(
                        f"Team match: {player.name} "
                        f"{player_team} -> {matched_name} "
                        f"{candidate_team}"
                    )
                    return mlbam_id

        matched_name, score, _ = results[0]
        mlbam_id = self._normalized_name_map[matched_name]
        logger.debug(
            f"Matched {player.name!r} → {matched_name!r} "
            f"(score={score}, mlbam={mlbam_id})"
        )
        return mlbam_id

    def _get_latest_hitter_features(self, mlbam_id: int) -> pd.Series | None:
        """Get feature row for inference (pre-game rolling averages).

        Uses the **second-to-last** game row when available so rolling
        features exclude the most recent game's Statcast stats (avoids
        single-game inflation).
        """
        if self._hitter_features is None:
            return None

        df = self._hitter_features
        id_col = "batter" if "batter" in df.columns else None
        if id_col is None:
            return None

        player_rows = df[df[id_col] == mlbam_id]
        if player_rows.empty:
            return None

        if "game_date" in player_rows.columns:
            player_rows = player_rows.sort_values("game_date")

        # Use second-to-last row to get pre-game rolling averages
        # The last row's rolling features include that game's stats
        # which inflates 7d averages from single hot games
        if len(player_rows) >= 2:
            return player_rows.iloc[-2]
        return player_rows.iloc[-1]

    def _get_latest_pitcher_features(self, mlbam_id: int) -> pd.Series | None:
        """Get feature row for inference (pre-game rolling averages)."""
        if self._pitcher_features is None:
            return None

        df = self._pitcher_features
        id_col = "pitcher" if "pitcher" in df.columns else None
        if id_col is None:
            return None

        player_rows = df[df[id_col] == mlbam_id]
        if player_rows.empty:
            return None

        if "game_date" in player_rows.columns:
            player_rows = player_rows.sort_values("game_date")

        # Use second-to-last row to get pre-game rolling averages
        # The last row's rolling features include that game's stats
        # which inflates 7d averages from single hot games
        if len(player_rows) >= 2:
            return player_rows.iloc[-2]
        return player_rows.iloc[-1]

    def _build_hitter_feature_vector(
        self,
        feature_row: pd.Series,
        player: DKPlayer,
        opposing_pitcher_hand: str = "R",
    ) -> pd.DataFrame:
        """Build a single-row feature DataFrame for hitter inference.

        Uses cached rolling features from the most recent game row.
        Pre-game context features default to 0.
        Platoon features recalculated for today's matchup.

        Args:
            feature_row: Most recent game row from feature matrix.
            player: DK player for context (team, opponent, salary).
            opposing_pitcher_hand: 'R' or 'L' for platoon calc.

        Returns:
            Single-row DataFrame with all hitter features expected by the model.
        """
        fe = FeatureEngineer()
        feature_cols = fe.get_feature_columns()

        row_dict = {}
        for col in feature_cols:
            if col in feature_row.index:
                row_dict[col] = feature_row[col]
            else:
                row_dict[col] = 0.0

        # Override pre-game context features
        row_dict["run_diff"] = 0.0
        row_dict["is_close_game"] = 1.0  # assume close pre-game
        row_dict["is_high_leverage"] = 0.0
        row_dict["batting_order_multiplier"] = 1.0

        # Recalculate platoon for today's matchup
        batter_hand = (
            str(feature_row.get("stand", "R"))
            if "stand" in feature_row.index
            else "R"
        )
        platoon_advantage = 1.0 if batter_hand != opposing_pitcher_hand else 0.0
        same_hand = 1.0 if batter_hand == opposing_pitcher_hand else 0.0
        row_dict["platoon_advantage"] = platoon_advantage
        row_dict["same_hand"] = same_hand

        return pd.DataFrame([row_dict])[feature_cols]

    def _build_pitcher_feature_vector(
        self,
        feature_row: pd.Series,
        player: DKPlayer,
    ) -> pd.DataFrame:
        """Build a single-row feature DataFrame for pitcher inference.

        Args:
            feature_row: Most recent game row from pitcher matrix.
            player: DK player for context.

        Returns:
            Single-row DataFrame with all pitcher features expected by the model.
        """
        pfe = PitcherFeatureEngineer()
        feature_cols = pfe.get_feature_columns()

        row_dict = {}
        for col in feature_cols:
            if col in feature_row.index:
                row_dict[col] = feature_row[col]
            else:
                row_dict[col] = 0.0

        return pd.DataFrame([row_dict])[feature_cols]

    def _fallback_projection(self, player: DKPlayer) -> tuple[float, float, float]:
        """Fallback q15/q50/q85 from DK avg when model can't run.

        Uses DK avg_points_per_game with conservative multipliers.
        Caps reliever q50 at 15 to avoid inflated DK avgs.
        """
        avg = float(player.avg_points_per_game)

        if player.is_pitcher:
            # Check if reliever — cap at 15 pts
            dk_pos = (player.dk_position or "").upper()
            is_starter = "SP" in player.position_eligibility or "SP" in dk_pos
            if not is_starter:
                avg = min(avg, 15.0)
            q50 = avg
            q15 = max(0.0, avg * 0.4)
            q85 = avg * 2.0
        else:
            q50 = avg
            q15 = max(0.0, avg * 0.5)
            q85 = avg * 1.8

        return q15, q50, q85

    def build_projections(
        self,
        players: list[DKPlayer],
        use_models: bool = True,
    ) -> list[PlayerProjection]:
        """Build PlayerProjection list for all slate players.

        For each player:
        1. Fuzzy match name to MLBAM ID
        2. Get most recent feature row from cached matrix
        3. Build feature vector for today's matchup
        4. Run XGBoost model → q15/q50/q85
        5. Fall back to DK avg if any step fails

        Args:
            players: Full DK player pool from parsed CSV.
            use_models: If False, use DK avg fallback for all
                players (fast, for testing).

        Returns:
            List of PlayerProjection with model-based quantiles.
        """
        if not self._name_to_mlbam:
            self._build_name_lookup()

        projections = []
        matched = 0
        unmatched = 0
        model_used = 0
        fallback_used = 0

        for player in players:
            q15, q50, q85 = self._fallback_projection(player)
            used_model = False

            if use_models and self._name_to_mlbam:
                mlbam_id = self._match_player(player)

                if mlbam_id is not None:
                    matched += 1
                    try:
                        if player.is_pitcher:
                            feat_row = self._get_latest_pitcher_features(mlbam_id)
                            if feat_row is not None and self._pitcher_model is not None:
                                feat_df = self._build_pitcher_feature_vector(
                                    feat_row, player
                                )
                                preds = self._pitcher_model.predict(feat_df)
                                q15 = float(max(0, preds["q15"].iloc[0]))
                                q50 = float(max(0, preds["q50"].iloc[0]))
                                q85 = float(max(0, preds["q85"].iloc[0]))
                                used_model = True
                        else:
                            feat_row = self._get_latest_hitter_features(mlbam_id)
                            if feat_row is not None and self._hitter_model is not None:
                                feat_df = self._build_hitter_feature_vector(
                                    feat_row, player
                                )
                                preds = self._hitter_model.predict(feat_df)
                                q15 = float(max(0, preds["q15"].iloc[0]))
                                q50 = float(max(0, preds["q50"].iloc[0]))
                                q85 = float(max(0, preds["q85"].iloc[0]))
                                used_model = True
                    except Exception as exc:
                        logger.warning(
                            f"Model inference failed for {player.name}: {exc}"
                        )
                else:
                    unmatched += 1

            if used_model:
                # Sanity check: if model returns 0 for a player
                # with a non-zero DK avg, use fallback instead
                if q50 <= 0.0 and float(player.avg_points_per_game) > 0:
                    logger.warning(
                        f"{player.name}: model q50={q50:.2f} but "
                        f"DK avg={player.avg_points_per_game:.1f} "
                        f"— reverting to fallback"
                    )
                    q15, q50, q85 = self._fallback_projection(player)
                    used_model = False

            if used_model:
                model_used += 1
            else:
                fallback_used += 1

            ownership_proj = 15.0 if player.is_pitcher else 10.0
            leverage = q50 / max(ownership_proj, 0.1)

            projections.append(
                PlayerProjection(
                    player=player,
                    pts_q15=round(q15, 2),
                    pts_q50=round(q50, 2),
                    pts_q85=round(q85, 2),
                    ownership_proj=round(ownership_proj, 2),
                    leverage_score=round(leverage, 3),
                )
            )

        logger.info(
            f"build_projections: {len(players)} players — "
            f"matched={matched}, unmatched={unmatched}, "
            f"model_used={model_used}, fallback={fallback_used}"
        )
        return projections

