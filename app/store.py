"""SQLite persistence layer.

All shared state (raw spectra, schemes, the active-scheme pointer and the
persisted per-scheme indexes) lives in one SQLite database in WAL mode, so
two API processes sharing the file — and the same process after a restart —
observe exactly the same committed state.

Invariants and how they are enforced:

* Ingest is immediately searchable: the upsert and the index-vector write
  happen in one IMMEDIATE transaction, and the active scheme is read
  *inside* that transaction, so the vector always lands in the scheme that
  is active at commit time.
* Query consistency: the active scheme row and its index vectors are read
  inside a single read transaction, so scheme + content + scores always
  come from one committed snapshot.
* Rebuild never blocks ingest/query: the new index is built under a new
  scheme id in short batch transactions; readers use WAL snapshots.
* A successful switch includes everything ingested during the rebuild:
  the final catch-up diff and the activation happen in the same IMMEDIATE
  transaction, which serializes against any concurrent ingest.
* A late rebuild cannot overwrite a newer scheme: activation is a
  compare-and-swap on the base scheme id observed at request time.
* A failed rebuild keeps the old index: the active pointer is only ever
  updated by the CAS in the final transaction; failures just mark the
  candidate scheme 'failed'.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass

import numpy as np

from . import core

SCHEMA = """
CREATE TABLE IF NOT EXISTS seq_counter (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    value INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS spectra (
    spectrum_id TEXT PRIMARY KEY,
    intensities BLOB NOT NULL,      -- 256 little-endian float64, as exported
    seq INTEGER NOT NULL            -- logical clock, bumped on every upsert
);
CREATE TABLE IF NOT EXISTS schemes (
    scheme_id INTEGER PRIMARY KEY AUTOINCREMENT,
    mask_json TEXT NOT NULL,        -- JSON array of masked channel indices
    status TEXT NOT NULL,           -- building|active|superseded|abandoned|failed
    base_scheme_id INTEGER,         -- active scheme observed when rebuild was requested
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE IF NOT EXISTS active_scheme (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    scheme_id INTEGER NOT NULL REFERENCES schemes(scheme_id)
);
CREATE TABLE IF NOT EXISTS index_vectors (
    scheme_id INTEGER NOT NULL REFERENCES schemes(scheme_id),
    spectrum_id TEXT NOT NULL,
    vector BLOB NOT NULL,           -- 256 float64, masked + L2-normalized
    built_seq INTEGER NOT NULL,     -- spectra.seq this vector reflects
    PRIMARY KEY (scheme_id, spectrum_id)
);
"""


class LibraryFullError(Exception):
    """Raised when the library already holds the maximum number of spectra."""


def connect(db_path: str) -> sqlite3.Connection:
    parent = os.path.dirname(os.path.abspath(db_path))
    os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: str) -> None:
    """Create the schema and, on first use, seed scheme 1 (empty mask)."""
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("INSERT OR IGNORE INTO seq_counter (id, value) VALUES (1, 0)")
            row = conn.execute(
                "SELECT scheme_id FROM active_scheme WHERE id = 1"
            ).fetchone()
            if row is None:
                cur = conn.execute(
                    "INSERT INTO schemes (mask_json, status, base_scheme_id) "
                    "VALUES ('[]', 'active', NULL)"
                )
                conn.execute(
                    "INSERT INTO active_scheme (id, scheme_id) VALUES (1, ?)",
                    (cur.lastrowid,),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()


def _encode(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr, dtype=np.float64).tobytes()


def _decode(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float64)


def _active_scheme_row(conn: sqlite3.Connection) -> sqlite3.Row:
    row = conn.execute(
        "SELECT s.scheme_id, s.mask_json FROM active_scheme a "
        "JOIN schemes s ON s.scheme_id = a.scheme_id WHERE a.id = 1"
    ).fetchone()
    if row is None:  # pragma: no cover - init_db guarantees this
        raise RuntimeError("no active scheme; database not initialized")
    return row


@dataclass
class IngestResult:
    spectrum_id: str
    indexed: bool           # False when all-zero after masking (skipped)
    scheme_id: int          # scheme the vector was written to
    library_size: int


def upsert_spectrum(
    db_path: str, spectrum_id: str, intensities, max_spectra: int
) -> IngestResult:
    """Insert or replace a spectrum and index it under the active scheme.

    Everything happens in one IMMEDIATE transaction: the active scheme is
    read inside the transaction, so a concurrent scheme switch either
    happens before us (we write the new scheme) or after us (the switch's
    catch-up diff copies our row into the new index).
    """
    arr = core.validate_intensities(intensities)
    blob = _encode(arr)
    conn = connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            exists = conn.execute(
                "SELECT 1 FROM spectra WHERE spectrum_id = ?", (spectrum_id,)
            ).fetchone()
            size = conn.execute("SELECT COUNT(*) FROM spectra").fetchone()[0]
            if not exists and size >= max_spectra:
                raise LibraryFullError(
                    f"library is full ({max_spectra} spectra); "
                    f"cannot add new spectrum_id {spectrum_id!r}"
                )
            conn.execute("UPDATE seq_counter SET value = value + 1 WHERE id = 1")
            seq = conn.execute("SELECT value FROM seq_counter WHERE id = 1").fetchone()[0]
            conn.execute(
                "INSERT INTO spectra (spectrum_id, intensities, seq) VALUES (?, ?, ?) "
                "ON CONFLICT(spectrum_id) DO UPDATE SET "
                "intensities = excluded.intensities, seq = excluded.seq",
                (spectrum_id, blob, seq),
            )
            scheme = _active_scheme_row(conn)
            scheme_id = scheme["scheme_id"]
            mask = json.loads(scheme["mask_json"])
            vec = core.normalize(arr, mask)
            if vec is None:
                # All-zero after masking: keep the raw spectrum but make sure
                # no stale vector (from a previous replace) remains indexed.
                conn.execute(
                    "DELETE FROM index_vectors WHERE scheme_id = ? AND spectrum_id = ?",
                    (scheme_id, spectrum_id),
                )
                indexed = False
            else:
                conn.execute(
                    "INSERT INTO index_vectors (scheme_id, spectrum_id, vector, built_seq) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(scheme_id, spectrum_id) DO UPDATE SET "
                    "vector = excluded.vector, built_seq = excluded.built_seq",
                    (scheme_id, spectrum_id, _encode(vec), seq),
                )
                indexed = True
            if not exists:
                size += 1
            conn.execute("COMMIT")
            return IngestResult(spectrum_id, indexed, scheme_id, size)
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()


@dataclass
class IndexSnapshot:
    scheme_id: int
    masked_channels: tuple[int, ...]
    ids: list[str]
    matrix: np.ndarray  # shape (n, 256), rows L2-normalized


def read_index_snapshot(db_path: str) -> IndexSnapshot:
    """Read the active scheme and its vectors in one committed snapshot."""
    conn = connect(db_path)
    try:
        conn.execute("BEGIN")  # deferred read transaction pins one WAL snapshot
        try:
            scheme = _active_scheme_row(conn)
            rows = conn.execute(
                "SELECT spectrum_id, vector FROM index_vectors "
                "WHERE scheme_id = ? ORDER BY spectrum_id",
                (scheme["scheme_id"],),
            ).fetchall()
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()
    ids = [r["spectrum_id"] for r in rows]
    matrix = (
        np.vstack([_decode(r["vector"]) for r in rows])
        if rows
        else np.zeros((0, core.N_CHANNELS), dtype=np.float64)
    )
    return IndexSnapshot(
        scheme["scheme_id"], tuple(json.loads(scheme["mask_json"])), ids, matrix
    )


def get_active_scheme(db_path: str) -> dict:
    conn = connect(db_path)
    try:
        scheme = _active_scheme_row(conn)
        indexed = conn.execute(
            "SELECT COUNT(*) FROM index_vectors WHERE scheme_id = ?",
            (scheme["scheme_id"],),
        ).fetchone()[0]
        size = conn.execute("SELECT COUNT(*) FROM spectra").fetchone()[0]
        return {
            "scheme_id": scheme["scheme_id"],
            "masked_channels": json.loads(scheme["mask_json"]),
            "indexed_count": indexed,
            "library_size": size,
        }
    finally:
        conn.close()


def list_schemes(db_path: str) -> list[dict]:
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT s.scheme_id, s.mask_json, s.status, s.base_scheme_id, s.created_at, "
            "(SELECT COUNT(*) FROM index_vectors v WHERE v.scheme_id = s.scheme_id) "
            "AS indexed_count, "
            "(SELECT scheme_id FROM active_scheme WHERE id = 1) AS active_id "
            "FROM schemes s ORDER BY s.scheme_id"
        ).fetchall()
        return [
            {
                "scheme_id": r["scheme_id"],
                "masked_channels": json.loads(r["mask_json"]),
                "status": r["status"],
                "is_active": r["scheme_id"] == r["active_id"],
                "base_scheme_id": r["base_scheme_id"],
                "indexed_count": r["indexed_count"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]
    finally:
        conn.close()
