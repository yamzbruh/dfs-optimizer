"""Pybaseball ingestion layer with Parquet caching.

Pulls Statcast and FanGraphs data via pybaseball and caches every
result to a local Parquet file.  Re-running any method with the same
arguments uses the cache unless ``force_refresh=True``.

Design rules enforced throughout:
* Every pybaseball call is wrapped in ``try/except``.
* Every result is checked for ``None`` or ``empty`` *before* any
  further method calls on it — never call ``.copy()`` or ``.groupby()``
  on an unchecked return value.
* ``game_date`` is converted to ``datetime64`` immediately after any
  statcast pull.
* End dates are capped at ``datetime.date.today()`` — future dates
  cause pybaseball to raise.
* All Parquet I/O goes through ``ParquetCache`` which uses
  ``engine='pyarrow'``.
"""

from __future__ import annotations

import datetime
import time

import pandas as pd
from loguru import logger

from data_pipeline.loaders.parquet_cache import ParquetCache

# ---------------------------------------------------------------------------
# Lazy import of pybaseball so the module can be imported in environments
# where pybaseball is unavailable (e.g. CI without network) without
# immediately exploding.  Each method imports what it needs internally.
# ---------------------------------------------------------------------------

# DraftKings MLB classic batting scoring weights (verified from DK rules).
# Walk and HBP are 2.0 pts, not 3.0.  No strikeout penalty for hitters.
_DK_BATTING_WEIGHTS: dict[str, float] = {
    "single": 3.0,
    "double": 5.0,
    "triple": 8.0,
    "home_run": 10.0,
    "walk": 2.0,
    "hit_by_pitch": 2.0,
}


class StatcastLoader:
    """Pulls Statcast / FanGraphs data and caches it to Parquet.

    Usage::

        loader = StatcastLoader()
        df = loader.get_statcast_batters(2024)
        pitching = loader.get_fangraphs_pitching(2023)
    """

    def __init__(
        self,
        cache: ParquetCache | None = None,
        cache_dir: str = "data/parquet",
    ) -> None:
        """Build a loader, creating a ``ParquetCache`` if one is not supplied."""
        self.cache: ParquetCache = cache if cache is not None else ParquetCache(cache_dir)

    # ------------------------------------------------------------------
    # Statcast
    # ------------------------------------------------------------------

    def get_statcast_batters(
        self,
        year: int,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return Statcast pitch-by-pitch data for *all* players in ``year``.

        Filters are not applied here — the full pull is cached and later
        filtered downstream.  Cache key: ``statcast/batters_{year}``.

        Args:
            year: Season year (e.g. 2024).
            force_refresh: When ``True``, ignore the cache and re-pull.

        Returns:
            DataFrame of Statcast events, or an empty DataFrame on error.
        """
        key = f"statcast/batters_{year}"
        return self._get_statcast(key, year, force_refresh)

    def get_statcast_pitchers(
        self,
        year: int,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return Statcast data for the ``year`` season (pitcher-centric view).

        The raw statcast pull contains both batter and pitcher columns.
        This method caches the same data under a separate key so callers
        can retrieve it without re-pulling.  Cache key:
        ``statcast/pitchers_{year}``.

        Args:
            year: Season year.
            force_refresh: Ignore cache and re-pull when ``True``.

        Returns:
            DataFrame of Statcast events, or an empty DataFrame on error.
        """
        key = f"statcast/pitchers_{year}"
        return self._get_statcast(key, year, force_refresh)

    # ------------------------------------------------------------------
    # FanGraphs
    # ------------------------------------------------------------------

    def get_fangraphs_batting(
        self,
        year: int,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return FanGraphs season batting stats for ``year``.

        Uses ``pybaseball.batting_stats(start_season, end_season, qual=0)``.
        Cache key: ``fangraphs/batting_{year}``.

        Args:
            year: Season year.
            force_refresh: Ignore cache when ``True``.

        Returns:
            DataFrame of batting stats, or an empty DataFrame on error.
        """
        key = f"fangraphs/batting_{year}"
        if not force_refresh:
            cached = self.cache.load(key)
            if cached is not None:
                return cached

        logger.info(f"PULLING FanGraphs batting {year}…")
        t0 = time.time()

        try:
            from pybaseball import batting_stats  # noqa: PLC0415

            df = batting_stats(year, end_season=year, qual=0)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"FanGraphs batting {year} failed: {exc}")
            return pd.DataFrame()

        if df is None or df.empty:
            logger.warning(f"FanGraphs batting {year}: no data returned")
            return pd.DataFrame()

        elapsed = time.time() - t0
        logger.info(
            f"FanGraphs batting {year}: {len(df):,} rows in {elapsed:.1f}s"
        )

        self.cache.save(
            df,
            key,
            metadata={"year": year, "type": "fangraphs_batting", "rows": len(df)},
        )
        return df

    def get_fangraphs_pitching(
        self,
        year: int,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return FanGraphs season pitching stats for ``year``.

        Uses ``pybaseball.pitching_stats(start_season, end_season, qual=0)``.
        Cache key: ``fangraphs/pitching_{year}``.

        Args:
            year: Season year.
            force_refresh: Ignore cache when ``True``.

        Returns:
            DataFrame of pitching stats, or an empty DataFrame on error.
        """
        key = f"fangraphs/pitching_{year}"
        if not force_refresh:
            cached = self.cache.load(key)
            if cached is not None:
                return cached

        logger.info(f"PULLING FanGraphs pitching {year}…")
        t0 = time.time()

        try:
            from pybaseball import pitching_stats  # noqa: PLC0415

            df = pitching_stats(year, end_season=year, qual=0)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"FanGraphs pitching {year} failed: {exc}")
            return pd.DataFrame()

        if df is None or df.empty:
            logger.warning(f"FanGraphs pitching {year}: no data returned")
            return pd.DataFrame()

        elapsed = time.time() - t0
        logger.info(
            f"FanGraphs pitching {year}: {len(df):,} rows in {elapsed:.1f}s"
        )

        self.cache.save(
            df,
            key,
            metadata={"year": year, "type": "fangraphs_pitching", "rows": len(df)},
        )
        return df

    # ------------------------------------------------------------------
    # Derived features
    # ------------------------------------------------------------------

    def get_rolling_stats(
        self,
        df: pd.DataFrame,
        player_col: str = "batter",
        date_col: str = "game_date",
        stat_cols: list[str] | None = None,
        windows: list[int] | None = None,
    ) -> pd.DataFrame:
        """Add rolling-window means to a Statcast DataFrame.

        Groups by ``player_col``, sorts by ``date_col``, then computes a
        rolling mean over each window in ``windows`` for each column in
        ``stat_cols``.  New columns are named ``{stat}_{window}d``.

        Args:
            df: Input Statcast DataFrame (modified in-place is avoided —
                a copy is returned).
            player_col: Column identifying each player.
            date_col: Column holding the game date.
            stat_cols: Statcast columns to roll up.  Defaults to a set
                of batted-ball / exit-velocity metrics.
            windows: Rolling window sizes in days.  Defaults to
                ``[7, 14, 30]``.
            min_periods: Passed to ``pd.Series.rolling``; defaults to 1
                so early-season rows still get a value.

        Returns:
            A new DataFrame with rolling-mean columns appended, or the
            original DataFrame unchanged (with a warning logged) if none
            of ``stat_cols`` are present.
        """
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            logger.warning("get_rolling_stats: received empty DataFrame")
            return df if df is not None else pd.DataFrame()

        _windows: list[int] = windows if windows is not None else [7, 14, 30]
        _stat_cols: list[str] = stat_cols if stat_cols is not None else [
            "estimated_woba_using_speedangle",
            "launch_speed",
            "launch_angle",
            "barrel",
            "hit_distance_sc",
        ]

        available = [c for c in _stat_cols if c in df.columns]
        if not available:
            logger.warning(
                f"get_rolling_stats: none of {_stat_cols} found in df "
                f"(columns: {list(df.columns)[:10]}…); returning unchanged"
            )
            return df

        result = df.copy()
        result[date_col] = pd.to_datetime(result[date_col])
        result = result.sort_values([player_col, date_col])

        for col in available:
            for window in _windows:
                out_col = f"{col}_{window}d"
                result[out_col] = (
                    result.groupby(player_col, sort=False)[col]
                    .transform(
                        lambda s, w=window: s.rolling(w, min_periods=1).mean()
                    )
                )

        logger.info(
            f"get_rolling_stats: added {len(available) * len(_windows)} "
            f"rolling columns for {result[player_col].nunique():,} players"
        )
        return result

    def get_player_crosswalk(
        self,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return the full pybaseball player-ID crosswalk.

        Pulls ``playerid_lookup('', '')`` which returns every player in
        the Chadwick Bureau register.  Cache key: ``crosswalk/player_ids``.

        Returns:
            DataFrame with columns including ``name_last``,
            ``name_first``, ``key_mlbam``, ``key_fangraphs``,
            ``key_bbref``; or an empty DataFrame on error.
        """
        key = "crosswalk/player_ids"
        if not force_refresh:
            cached = self.cache.load(key)
            if cached is not None:
                return cached

        logger.info("PULLING player ID crosswalk from pybaseball…")
        t0 = time.time()

        try:
            from pybaseball import playerid_lookup  # noqa: PLC0415

            df = playerid_lookup("", "")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"playerid_lookup failed: {exc}")
            return pd.DataFrame()

        if df is None or df.empty:
            logger.warning("playerid_lookup returned no data")
            return pd.DataFrame()

        elapsed = time.time() - t0
        logger.info(
            f"Player crosswalk: {len(df):,} players in {elapsed:.1f}s"
        )

        self.cache.save(
            df,
            key,
            metadata={"type": "player_crosswalk", "rows": len(df)},
        )
        return df

    def calculate_dk_points_batting(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate DraftKings batting points from Statcast event data.

        **Scoring applied — verified from DK MLB classic rules:**

        From pitch-by-pitch events (exact):

        - Single: +3 pts
        - Double: +5 pts
        - Triple: +8 pts
        - Home run: +10 pts
        - Walk: +2 pts
        - Hit by pitch: +2 pts

        Args:
            df: Statcast pitch-by-pitch DataFrame with an ``events``
                column.

        Returns:
            Input DataFrame with a new ``dk_points_batting`` column, or
            the original DataFrame unchanged on error.
        """
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            logger.warning("calculate_dk_points_batting: received empty DataFrame")
            return df if df is not None else pd.DataFrame()

        if "events" not in df.columns:
            logger.warning(
                "calculate_dk_points_batting: 'events' column not found; "
                "returning df unchanged"
            )
            return df

        result = df.copy()

        pts = pd.Series(0.0, index=result.index)
        for event, weight in _DK_BATTING_WEIGHTS.items():
            pts += (result["events"] == event).astype(float) * weight

        # Add box-score components when available.
        if "runs_scored" in result.columns:
            pts += pd.to_numeric(result["runs_scored"], errors="coerce").fillna(0) * 2.0
        if "rbi" in result.columns:
            pts += pd.to_numeric(result["rbi"], errors="coerce").fillna(0) * 2.0
        if "stolen_bases" in result.columns:
            pts += pd.to_numeric(result["stolen_bases"], errors="coerce").fillna(0) * 5.0

        result["dk_points_batting"] = pts
        logger.info(
            f"calculate_dk_points_batting: scored {len(result):,} rows; "
            f"total pts={pts.sum():.0f}"
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_statcast(
        self,
        key: str,
        year: int,
        force_refresh: bool,
    ) -> pd.DataFrame:
        """Shared implementation for statcast batter / pitcher pulls."""
        if not force_refresh:
            cached = self.cache.load(key)
            if cached is not None:
                return cached

        start_dt = f"{year}-03-01"
        today = datetime.date.today().isoformat()
        end_dt = min(f"{year}-11-01", today)

        logger.info(f"PULLING Statcast {key}  ({start_dt} → {end_dt})…")
        t0 = time.time()

        try:
            from pybaseball import statcast  # noqa: PLC0415

            df = statcast(start_dt=start_dt, end_dt=end_dt)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"statcast pull for {key} failed: {exc}")
            return pd.DataFrame()

        if df is None or df.empty:
            logger.warning(f"statcast pull for {key}: no data returned")
            return pd.DataFrame()

        # Coerce game_date immediately — never trust pybaseball's dtype.
        df["game_date"] = pd.to_datetime(df["game_date"])

        elapsed = time.time() - t0
        logger.info(f"Statcast {key}: {len(df):,} rows in {elapsed:.1f}s")

        self.cache.save(
            df,
            key,
            metadata={"year": year, "type": key, "rows": len(df)},
        )
        return df
