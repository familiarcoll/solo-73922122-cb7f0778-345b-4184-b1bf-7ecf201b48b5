"""Runtime configuration, read from the environment at call time."""
from __future__ import annotations

import os


def db_path() -> str:
    return os.environ.get("MS_DB_PATH", "./spectra.db")


def max_spectra() -> int:
    return int(os.environ.get("MS_MAX_SPECTRA", "5000"))


def rebuild_batch_delay_ms() -> int:
    """Artificial per-batch delay for rebuilds (demo/testing of concurrency)."""
    return int(os.environ.get("MS_REBUILD_BATCH_DELAY_MS", "0"))
