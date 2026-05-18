"""Pitcher feature engineering for DraftKings MLB optimizer.

Builds game-level pitcher features from Statcast pitch data and MLB Stats
API game logs.  Produces one row per pitcher per game with rolling
performance metrics and the target ``dk_points_game`` for XGBoost training.

Every public method returns a *new* DataFrame — inputs are never mutated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_pipeline.loaders.parquet_cache import ParquetCache  # noqa: E402


class PitcherFeatureEngineer:
    """Build game-level pitcher features for XGBoost training."""

    def __init__(self, cache_dir: str = "data/parquet") -> None:
        """Initialise with a ``ParquetCache`` rooted at ``cache_dir``."""
        self.cache = ParquetCache(cache_dir)
        logger.debug(f"PitcherFeatureEngineer ready  cache_dir={cache_dir}")

    def get_feature_columns(self) -> list[str]:
        """Return ordered list of pitcher feature column names.

        Target ``dk_points_game`` is not included — it is the label for
        ``PitcherPointsModel`` training.

        Returns:
            Canonical feature names matching columns produced by
            :meth:`build_full_pitcher_feature_matrix`.
        """
        return [
            # Rolling strikeout rate (K / terminal PA outcomes)
            "k_rate_7d",
            "k_rate_14d",
            "k_rate_30d",
            # Rolling walk rate
            "bb_rate_7d",
            "bb_rate_14d",
            "bb_rate_30d",
            # Rolling fastball velocity
            "velo_mean_7d",
            "velo_mean_14d",
            "velo_mean_30d",
            # Rolling whiff rate
            "whiff_rate_7d",
            "whiff_rate_14d",
            "whiff_rate_30d",
            # Rolling IP per appearance (from game logs)
            "ip_per_start_7d",
            "ip_per_start_14d",
            "ip_per_start_30d",
            "ip_per_appearance_7d",
            "ip_per_appearance_14d",
            "ip_per_appearance_30d",
            # Rolling ERA-style ratio
            "era_approx_14d",
            "era_approx_30d",
            # Game context
            "is_home",
            "pitcher_hand",  # 0=L, 1=R
            # Starter vs reliever proxy (from rolling IP/start)
            "is_starter",
        ]

    def load_statcast_years(self, years: list[int]) -> pd.DataFrame:
        """Load and concatenate Statcast pitcher Parquets for ``years``.

        Args:
            years: Season years, e.g. ``[2023, 2024]``.

        Returns:
            Combined pitch-level DataFrame, or empty if nothing cached.
        """
        frames: list[pd.DataFrame] = []
        for year in years:
            key = f"statcast/pitchers_{year}"
            df = self.cache.load(key)
            if df is not None and not df.empty:
                frames.append(df)
                logger.debug(f"Loaded {key}: {len(df):,} rows")

        if not frames:
            logger.error("load_statcast_years: no Statcast pitcher data found")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        if "game_date" in combined.columns:
            combined["game_date"] = pd.to_datetime(combined["game_date"])
        logger.info(
            f"load_statcast_years: {len(combined):,} Statcast pitcher rows "
            f"across {len(years)} season(s)"
        )
        return combined

    def build_rolling_pitcher_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate pitch-level Statcast to pitcher-game rows with rolling rates.

        First collapses to **one row per (pitcher, game_date)** with daily
        strikeouts, walks, terminal PA outcomes, swings, whiffs, and fastball
        velocity sums.  Then applies **last-7 / 14 / 30 games** rolling sums
        per pitcher (ordered by ``game_date``) and converts to rates.

        Missing ``events``, ``description``, ``pitch_type``, or
        ``release_speed`` columns are handled gracefully (rates default to
        0 or NaN-filled then 0).

        Args:
            df: Pitch-level Statcast DataFrame.

        Returns:
            Pitcher-game DataFrame with ``k_rate_*d``, ``bb_rate_*d``,
            ``whiff_rate_*d``, ``velo_mean_*d`` columns, and ``game_pk``
            when present in the source Statcast data.
        """
        if df is None or df.empty:
            logger.warning("build_rolling_pitcher_features: empty input")
            return df if df is not None else pd.DataFrame()

        if "pitcher" not in df.columns or "game_date" not in df.columns:
            logger.warning(
                "build_rolling_pitcher_features: missing pitcher or game_date"
            )
            return pd.DataFrame()

        result = df.copy()
        result["game_date"] = pd.to_datetime(result["game_date"])
        result = result.sort_values(["pitcher", "game_date"]).reset_index(drop=True)

        if "events" not in result.columns:
            result["events"] = np.nan
        ev = result["events"]
        result["is_terminal"] = (ev.notna() & (ev.astype(str) != "")).astype(float)
        result["is_strikeout"] = ev.isin(
            ["strikeout", "strikeout_double_play"]
        ).astype(float)
        result["is_walk"] = ev.eq("walk").astype(float)

        if "description" in result.columns:
            desc = result["description"].astype(str)
            swing_set = {
                "hit_into_play", "foul", "swinging_strike",
                "swinging_strike_blocked", "foul_tip", "foul_bunt",
                "missed_bunt", "hit_into_play_score", "hit_into_play_no_out",
            }
            whiff_set = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
            result["is_swing"] = desc.isin(swing_set).astype(float)
            result["is_whiff"] = desc.isin(whiff_set).astype(float)
        else:
            result["is_swing"] = 0.0
            result["is_whiff"] = 0.0
            logger.warning(
                "build_rolling_pitcher_features: no 'description'; "
                "whiff_rate features will be 0"
            )

        if "pitch_type" in result.columns and "release_speed" in result.columns:
            fb_mask = result["pitch_type"].isin(["FF", "SI", "FC"])
            result["fb_velo"] = np.where(
                fb_mask,
                pd.to_numeric(result["release_speed"], errors="coerce"),
                np.nan,
            )
            result["is_fastball"] = fb_mask.astype(float)
        else:
            result["fb_velo"] = np.nan
            result["is_fastball"] = 0.0
            logger.warning(
                "build_rolling_pitcher_features: missing pitch_type or "
                "release_speed; velo_mean features will be 0"
            )

        def _fb_velo_sum(s: pd.Series) -> float:
            return float(s.dropna().sum()) if s.notna().any() else 0.0

        agg_kwargs: dict = dict(
            terminal_events=("is_terminal", "sum"),
            strikeouts=("is_strikeout", "sum"),
            walks=("is_walk", "sum"),
            swings=("is_swing", "sum"),
            whiffs=("is_whiff", "sum"),
            fb_velo_sum=("fb_velo", _fb_velo_sum),
            fb_count=("is_fastball", "sum"),
        )
        if "game_pk" in result.columns:
            agg_kwargs["game_pk"] = ("game_pk", "first")

        daily = (
            result.groupby(["pitcher", "game_date"], sort=False)
            .agg(**agg_kwargs)
            .reset_index()
        )
        daily["pitcher"] = pd.to_numeric(daily["pitcher"], errors="coerce")
        daily = daily.dropna(subset=["pitcher"])
        daily["pitcher"] = daily["pitcher"].astype(int)
        daily = daily.sort_values(["pitcher", "game_date"]).reset_index(drop=True)

        windows = [7, 14, 30]
        g = daily.groupby("pitcher", sort=False)
        for w in windows:
            sk = g["strikeouts"].transform(
                lambda s, ww=w: s.rolling(ww, min_periods=1).sum()
            )
            tk = g["terminal_events"].transform(
                lambda s, ww=w: s.rolling(ww, min_periods=1).sum()
            )
            daily[f"k_rate_{w}d"] = (
                (sk / tk.replace(0, np.nan)).fillna(0.0).replace([np.inf, -np.inf], 0.0)
            )

            bk = g["walks"].transform(
                lambda s, ww=w: s.rolling(ww, min_periods=1).sum()
            )
            daily[f"bb_rate_{w}d"] = (
                (bk / tk.replace(0, np.nan)).fillna(0.0).replace([np.inf, -np.inf], 0.0)
            )

            wh = g["whiffs"].transform(
                lambda s, ww=w: s.rolling(ww, min_periods=1).sum()
            )
            sw = g["swings"].transform(
                lambda s, ww=w: s.rolling(ww, min_periods=1).sum()
            )
            daily[f"whiff_rate_{w}d"] = (
                (wh / sw.replace(0, np.nan)).fillna(0.0).replace([np.inf, -np.inf], 0.0)
            )

            fbs = g["fb_velo_sum"].transform(
                lambda s, ww=w: s.rolling(ww, min_periods=1).sum()
            )
            fbc = g["fb_count"].transform(
                lambda s, ww=w: s.rolling(ww, min_periods=1).sum()
            )
            daily[f"velo_mean_{w}d"] = (
                (fbs / fbc.replace(0, np.nan)).fillna(0.0).replace([np.inf, -np.inf], 0.0)
            )

        logger.info(
            f"build_rolling_pitcher_features: {len(daily):,} pitcher-game rows "
            f"with rolling features"
        )
        return daily

    def join_game_log_features(
        self,
        df: pd.DataFrame,
        years: list[int],
    ) -> pd.DataFrame:
        """Join MLB Stats API pitching game logs on ``pitcher`` + ``game_date``.

        Loads ``gamelogs/pitching_{year}`` Parquets.  Adds game-level counting
        stats needed for DK scoring.  ``strikeouts`` from the game log is
        stored as ``strikeouts_gl`` to avoid colliding with Statcast daily
        strikeout counts.

        Args:
            df: Pitcher-game DataFrame (e.g. output of
                :meth:`build_rolling_pitcher_features`).
            years: Seasons whose cached game logs to concatenate.

        Returns:
            New DataFrame with game-log columns merged (zeros when missing).
        """
        if df is None or df.empty:
            logger.warning("join_game_log_features: empty input")
            return df if df is not None else pd.DataFrame()

        result = df.copy()

        frames: list[pd.DataFrame] = []
        for year in years:
            key = f"gamelogs/pitching_{year}"
            gl = self.cache.load(key)
            if gl is not None and not gl.empty:
                frames.append(gl)

        if not frames:
            logger.warning(
                "join_game_log_features: no pitching game logs cached. "
                "Run scripts/pull_game_logs.py --all first."
            )
            return result

        game_logs = pd.concat(frames, ignore_index=True)
        game_logs["game_date"] = pd.to_datetime(game_logs["game_date"])
        game_logs["pitcher"] = pd.to_numeric(
            game_logs["pitcher"], errors="coerce"
        ).astype("Int64")

        want = [
            "pitcher",
            "game_date",
            "game_pk",
            "innings_pitched",
            "strikeouts",
            "earned_runs",
            "hits_allowed",
            "walks_allowed",
            "hit_batsmen",
            "wins",
            "complete_games",
            "shutouts",
            "no_hitters",
        ]
        cols = [c for c in want if c in game_logs.columns]
        game_logs = game_logs[cols].copy()
        if "strikeouts" in game_logs.columns:
            game_logs = game_logs.rename(columns={"strikeouts": "strikeouts_gl"})

        result["pitcher"] = pd.to_numeric(result["pitcher"], errors="coerce").astype(
            "Int64"
        )
        result["game_date"] = pd.to_datetime(result["game_date"])

        result = result.merge(game_logs, on=["pitcher", "game_date"], how="left")

        if "game_pk_x" in result.columns:
            if "game_pk_y" in result.columns:
                # Prefer MLB game-log ``game_pk`` (authoritative for crosswalks).
                result["game_pk"] = result["game_pk_y"].fillna(result["game_pk_x"])
            else:
                result["game_pk"] = result["game_pk_x"]
            result = result.drop(
                columns=["game_pk_x", "game_pk_y"],
                errors="ignore",
            )

        for col in (
            "innings_pitched", "earned_runs", "hits_allowed", "walks_allowed",
            "hit_batsmen", "wins", "complete_games", "shutouts", "strikeouts_gl",
            "no_hitters",
        ):
            if col not in result.columns:
                result[col] = 0.0
            else:
                result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)

        joined = int((result["innings_pitched"] > 0).sum()) if "innings_pitched" in result.columns else 0
        logger.info(
            f"join_game_log_features: game logs with IP>0 on "
            f"{joined:,} of {len(result):,} pitcher-game rows"
        )
        return result

    def build_ip_per_start_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling IP / appearance metrics over the last 7 / 14 / 30 games.

        Uses ``innings_pitched`` from the merged game log (decimal innings).

        * ``ip_per_start_{w}d`` — rolling mean IP per game row (legacy starter proxy).
        * ``ip_per_appearance_{w}d`` — total IP in window / total appearances
          (all games), with ``shift(1)`` so the current outing is excluded.

        Args:
            df: Pitcher-game DataFrame sorted by ``pitcher``, ``game_date``.

        Returns:
            New DataFrame with ``ip_per_start_*d``, ``ip_per_appearance_*d``,
            and ``is_starter`` columns.
        """
        if df is None or df.empty:
            logger.warning("build_ip_per_start_features: empty input")
            return df if df is not None else pd.DataFrame()

        result = df.copy()
        if "innings_pitched" not in result.columns:
            for w in (7, 14, 30):
                result[f"ip_per_start_{w}d"] = 0.0
                result[f"ip_per_appearance_{w}d"] = 0.0
            result["is_starter"] = 0.0
            return result

        result = result.sort_values(["pitcher", "game_date"]).reset_index(drop=True)
        g = result.groupby("pitcher", sort=False)["innings_pitched"]

        def _ip_per_appearance(s: pd.Series, window: int) -> pd.Series:
            prior_ip = s.fillna(0.0).shift(1)
            ip_sum = prior_ip.rolling(window, min_periods=1).sum()
            n_apps = prior_ip.rolling(window, min_periods=1).count()
            return (ip_sum / n_apps.replace(0, np.nan)).fillna(0.0)

        for w in (7, 14, 30):
            result[f"ip_per_start_{w}d"] = g.transform(
                lambda s, ww=w: s.rolling(ww, min_periods=1).mean()
            ).fillna(0.0)
            result[f"ip_per_appearance_{w}d"] = g.transform(
                lambda s, ww=w: _ip_per_appearance(s, ww)
            )

        result["is_starter"] = (result["ip_per_start_7d"] >= 3.0).astype(float)
        return result

    def build_era_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling ERA-style ratio: 9 × ER / IP over last 14 and 30 games.

        Args:
            df: Pitcher-game DataFrame with ``earned_runs`` and
                ``innings_pitched``.

        Returns:
            New DataFrame with ``era_approx_14d`` and ``era_approx_30d``.
        """
        if df is None or df.empty:
            logger.warning("build_era_features: empty input")
            return df if df is not None else pd.DataFrame()

        result = df.copy()
        if "earned_runs" not in result.columns or "innings_pitched" not in result.columns:
            result["era_approx_14d"] = 0.0
            result["era_approx_30d"] = 0.0
            return result

        result = result.sort_values(["pitcher", "game_date"]).reset_index(drop=True)
        gp = result.groupby("pitcher", sort=False)
        for w in (14, 30):
            er_sum = gp["earned_runs"].transform(
                lambda s, ww=w: s.rolling(ww, min_periods=1).sum()
            )
            ip_sum = gp["innings_pitched"].transform(
                lambda s, ww=w: s.rolling(ww, min_periods=1).sum()
            )
            result[f"era_approx_{w}d"] = np.where(
                ip_sum > 0,
                (er_sum / ip_sum) * 9.0,
                0.0,
            )
        return result

    def build_context_features(
        self,
        df: pd.DataFrame,
        statcast_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add ``is_home`` and ``pitcher_hand`` from Statcast metadata.

        ``pitcher_hand`` maps ``p_throws`` (R→1, L→0).  ``is_home`` is 1 when
        the pitcher's team's defensive half starts the game as the **top**
        of an inning with that pitcher on the mound (visitor batting first —
        home team pitching in Top), inferred from the first ``inning_topbot``
        value per ``(pitcher, game_date)``.

        Args:
            df: Pitcher-game feature DataFrame.
            statcast_df: Original pitch-level Statcast (same seasons).

        Returns:
            New DataFrame with ``is_home`` and ``pitcher_hand``.
        """
        if df is None or df.empty:
            logger.warning("build_context_features: empty input")
            return df if df is not None else pd.DataFrame()

        result = df.copy()
        sc = statcast_df.copy() if statcast_df is not None else pd.DataFrame()
        if sc.empty or "pitcher" not in sc.columns:
            result["pitcher_hand"] = 1
            result["is_home"] = 0
            logger.warning(
                "build_context_features: no Statcast context; "
                "pitcher_hand=1, is_home=0"
            )
            return result

        sc["pitcher"] = pd.to_numeric(sc["pitcher"], errors="coerce")
        sc = sc.dropna(subset=["pitcher"])
        sc["pitcher"] = sc["pitcher"].astype(int)
        if "game_date" in sc.columns:
            sc["game_date"] = pd.to_datetime(sc["game_date"])

        if "p_throws" in sc.columns:
            hand_map = (
                sc.groupby("pitcher", sort=False)["p_throws"]
                .first()
                .map({"R": 1, "L": 0})
            )
            result["pitcher"] = pd.to_numeric(result["pitcher"], errors="coerce").astype(
                int
            )
            result["pitcher_hand"] = (
                result["pitcher"].map(hand_map).fillna(1).astype(int)
            )
        else:
            result["pitcher_hand"] = 1
            logger.warning(
                "build_context_features: no p_throws; pitcher_hand=1"
            )

        if "inning_topbot" in sc.columns and "game_date" in sc.columns:
            hm = (
                sc.sort_values(["pitcher", "game_date"])
                .groupby(["pitcher", "game_date"], sort=False)["inning_topbot"]
                .first()
                .map({"Top": 1, "Bot": 0})
                .rename("is_home")
                .reset_index()
            )
            result = result.merge(hm, on=["pitcher", "game_date"], how="left")
            result["is_home"] = result["is_home"].fillna(0).astype(int)
        else:
            result["is_home"] = 0
            logger.warning(
                "build_context_features: no inning_topbot; is_home=0"
            )

        return result

    def calculate_dk_points_pitching(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute DraftKings **classic** pitcher points per game (``dk_points_game``).

        Scoring:

        * IP: ``+0.75`` per out (``+2.25`` per full inning)
        * K: ``+2.0`` each (from game log ``strikeouts_gl``)
        * Win: ``+4.0``
        * ER: ``-2.0`` each
        * H / BB / HBP allowed: ``-0.6`` each
        * CG: ``+2.5``
        * Shutout: ``+2.5`` (game-log ``shutouts`` flag)
        * No-hitter: ``+5.0`` when ``no_hitters`` > 0 from API, else when
          ``complete_games > 0`` and ``hits_allowed == 0`` (and not already
          counted via ``no_hitters``).

        Args:
            df: Pitcher-game row with merged game-log columns.

        Returns:
            New DataFrame with ``dk_points_game`` column.
        """
        if df is None or df.empty:
            logger.warning("calculate_dk_points_pitching: empty input")
            return df if df is not None else pd.DataFrame()

        result = df.copy()

        ip = pd.to_numeric(result.get("innings_pitched", 0), errors="coerce").fillna(0.0)
        full_innings = np.floor(ip.to_numpy(dtype=float))
        partial = ip.to_numpy(dtype=float) - full_innings
        extra_outs = np.round(partial * 3.0).astype(int)
        total_outs = full_innings * 3.0 + extra_outs

        k_col = "strikeouts_gl" if "strikeouts_gl" in result.columns else "strikeouts"
        ks = pd.to_numeric(result.get(k_col, 0), errors="coerce").fillna(0.0)
        wins = pd.to_numeric(result.get("wins", 0), errors="coerce").fillna(0.0)
        er = pd.to_numeric(result.get("earned_runs", 0), errors="coerce").fillna(0.0)
        h = pd.to_numeric(result.get("hits_allowed", 0), errors="coerce").fillna(0.0)
        bb = pd.to_numeric(result.get("walks_allowed", 0), errors="coerce").fillna(0.0)
        hbp = pd.to_numeric(result.get("hit_batsmen", 0), errors="coerce").fillna(0.0)
        cg = pd.to_numeric(result.get("complete_games", 0), errors="coerce").fillna(0.0)
        sho = pd.to_numeric(result.get("shutouts", 0), errors="coerce").fillna(0.0)
        if "no_hitters" in result.columns:
            nh_api = pd.to_numeric(result["no_hitters"], errors="coerce").fillna(0.0)
        else:
            nh_api = pd.Series(0.0, index=result.index)

        pts = (
            total_outs * 0.75
            + ks * 2.0
            + wins * 4.0
            - er * 2.0
            - h * 0.6
            - bb * 0.6
            - hbp * 0.6
            + cg * 2.5
            + sho * 2.5
        )
        nh_bonus = np.where(
            nh_api > 0,
            5.0,
            np.where((cg > 0) & (h == 0), 5.0, 0.0),
        )
        result["dk_points_game"] = (pts + nh_bonus).fillna(0.0)

        logger.info(
            f"calculate_dk_points_pitching: mean={result['dk_points_game'].mean():.2f}, "
            f"median={result['dk_points_game'].median():.2f}, "
            f"max={result['dk_points_game'].max():.2f}"
        )
        return result

    def build_full_pitcher_feature_matrix(
        self,
        years: list[int] | None = None,
        force_rebuild: bool = False,
    ) -> pd.DataFrame:
        """Build and cache the full game-level pitcher feature matrix.

        Pipeline:

        1. ``load_statcast_years`` — pitch-level Statcast
        2. ``build_rolling_pitcher_features`` — pitcher-game rolling rates
        3. ``join_game_log_features`` — MLB game logs (IP, K, ER, …)
        4. ``build_ip_per_start_features`` — rolling IP / game
        5. ``build_era_features`` — rolling ERA proxy
        6. ``build_context_features`` — home / handedness
        7. ``calculate_dk_points_pitching`` — target ``dk_points_game``
        8. Keep rows with ``innings_pitched > 0`` and non-null target
        9. Save to ``features/pitcher_feature_matrix_game_level``

        Args:
            years: Seasons to include.  Defaults to ``[2023, 2024, 2025, 2026]``.
            force_rebuild: When ``True``, ignore cache and rebuild.

        Returns:
            Game-level pitcher matrix, or empty DataFrame on failure.
        """
        _years = years if years is not None else [2023, 2024, 2025, 2026]
        cache_key = "features/pitcher_feature_matrix_game_level"

        if not force_rebuild and self.cache.exists(cache_key):
            cached = self.cache.load(cache_key)
            if cached is not None and not cached.empty:
                logger.info(
                    f"Loaded pitcher features from cache: {cache_key} "
                    f"({len(cached):,} rows)"
                )
                return cached

        logger.info("Building pitcher feature matrix…")

        statcast = self.load_statcast_years(_years)
        if statcast.empty:
            return pd.DataFrame()

        df = self.build_rolling_pitcher_features(statcast)
        if df.empty:
            return pd.DataFrame()

        df = self.join_game_log_features(df, _years)
        df = self.build_ip_per_start_features(df)
        df = self.build_era_features(df)
        df = self.build_context_features(df, statcast)
        df = self.calculate_dk_points_pitching(df)

        if "innings_pitched" in df.columns:
            df = df.loc[df["innings_pitched"] > 0].copy()
        df = df.dropna(subset=["dk_points_game"])

        feature_cols = self.get_feature_columns()
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            logger.warning(f"Missing feature columns (filled with 0): {missing}")
            for col in missing:
                df[col] = 0.0

        logger.info(
            f"Pitcher feature matrix: {len(df):,} rows, {len(df.columns)} cols"
        )

        self.cache.save(
            df,
            cache_key,
            metadata={
                "years": _years,
                "rows": len(df),
                "features": len(feature_cols),
                "type": "pitcher_feature_matrix_game_level",
                "target": "dk_points_game",
            },
        )
        return df
