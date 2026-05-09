"""Parquet-backed cache for baseball data frames.

All reads and writes go through a single ``data/parquet/`` directory
tree. Keys map to file paths — "/" in the key becomes a directory
separator, so ``"statcast/batters_2024"`` lives at
``data/parquet/statcast/batters_2024.parquet``.

Every save is *atomic*: the data is written to a ``.tmp`` sidecar first
and only renamed to the final path on success.  A crash mid-write
therefore never leaves a partial / corrupted cache file.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger


class ParquetCache:
    """Manages persistent Parquet storage for baseball DataFrames.

    Usage::

        cache = ParquetCache("data/parquet")
        cache.save(df, "statcast/batters_2024", metadata={"year": 2024})
        df = cache.load("statcast/batters_2024")
    """

    def __init__(self, cache_dir: str | Path = "data/parquet") -> None:
        """Create the cache, making the directory if necessary."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"ParquetCache initialised at {self.cache_dir.resolve()}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        df: pd.DataFrame,
        key: str,
        metadata: dict | None = None,
    ) -> Path:
        """Atomically save ``df`` to ``{cache_dir}/{key}.parquet``.

        Args:
            df: DataFrame to persist.
            key: Cache key (may contain "/" for sub-directories).
            metadata: Optional dict saved alongside as ``{key}_meta.json``.

        Returns:
            Path to the saved ``.parquet`` file.
        """
        final_path = self._path_for(key)
        tmp_path = final_path.with_suffix(".parquet.tmp")

        final_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(tmp_path, engine="pyarrow", index=False)
        tmp_path.rename(final_path)

        size_mb = final_path.stat().st_size / (1024 * 1024)
        logger.info(
            f"Cache SAVE  key={key!r}  rows={len(df):,}  "
            f"size={size_mb:.2f} MB  path={final_path}"
        )

        if metadata is not None:
            self.save_metadata(key, metadata)

        return final_path

    def load(self, key: str) -> pd.DataFrame | None:
        """Load ``{cache_dir}/{key}.parquet`` and return a DataFrame.

        Returns ``None`` (never raises) when the file is absent.
        After loading, any column named ``game_date`` is coerced to
        ``datetime64`` automatically.
        """
        path = self._path_for(key)
        if not path.exists():
            logger.debug(f"Cache MISS  key={key!r}")
            return None

        df = pd.read_parquet(path, engine="pyarrow")

        if "game_date" in df.columns:
            df["game_date"] = pd.to_datetime(df["game_date"])

        logger.info(f"Cache LOAD  key={key!r}  rows={len(df):,}")
        return df

    def exists(self, key: str) -> bool:
        """Return ``True`` when a parquet file exists for ``key``."""
        return self._path_for(key).exists()

    def get_metadata(self, key: str) -> dict | None:
        """Return the metadata dict stored beside ``key``, or ``None``."""
        meta_path = self._meta_path_for(key)
        if not meta_path.exists():
            return None
        with meta_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def save_metadata(self, key: str, metadata: dict) -> None:
        """Persist ``metadata`` as ``{key}_meta.json``.

        Always adds/overwrites ``"saved_at"`` with the current UTC ISO
        timestamp so callers always know when the data was last written.
        """
        enriched = {**metadata, "saved_at": datetime.now(timezone.utc).isoformat()}
        meta_path = self._meta_path_for(key)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(enriched, fh, indent=2, default=str)
        logger.debug(f"Cache META  key={key!r}  path={meta_path}")

    def get_last_updated(self, key: str) -> datetime | None:
        """Return the file-system mtime of ``key``'s parquet file.

        Returns ``None`` when the file is absent.
        """
        path = self._path_for(key)
        if not path.exists():
            return None
        mtime = path.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc)

    def list_keys(self, prefix: str = "") -> list[str]:
        """Return all cached keys, optionally filtered by ``prefix``.

        Keys are relative paths without the ``.parquet`` extension,
        using ``/`` as a separator regardless of OS.
        """
        all_keys: list[str] = []
        for parquet_file in sorted(self.cache_dir.rglob("*.parquet")):
            # Exclude temp files left over from a previous crashed write.
            if parquet_file.suffix == ".tmp":
                continue
            relative = parquet_file.relative_to(self.cache_dir)
            # Drop the .parquet suffix and normalise to forward slashes.
            key = relative.with_suffix("").as_posix()
            if not prefix or key.startswith(prefix):
                all_keys.append(key)
        return all_keys

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path_for(self, key: str) -> Path:
        """Resolve a cache key to an absolute ``.parquet`` path."""
        return (self.cache_dir / key).with_suffix(".parquet")

    def _meta_path_for(self, key: str) -> Path:
        """Resolve a cache key to its ``_meta.json`` sidecar path."""
        base = self.cache_dir / key
        return base.parent / f"{base.name}_meta.json"
