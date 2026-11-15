"""Background index rebuild: build a candidate index, then atomically switch.

The rebuild runs in three phases, all against the shared SQLite file:

1. Incremental catch-up loop: copy spectra (by their monotonically
   increasing seq) into the candidate scheme's index in small batch
   transactions, so ingest and queries keep working meanwhile.
2. Final transaction (BEGIN IMMEDIATE): process the remaining diff and
   compare-and-swap the active-scheme pointer. Because the write lock
   serializes us against any concurrent ingest, the switched-to index
   reflects every spectrum committed before the switch.
3. If the CAS fails, a newer scheme became active after we started: this
   rebuild is late, so it is abandoned without touching the active index.
   Any unexpected error marks the candidate 'failed'; the old active index
   is never modified by a failed rebuild.
"""
from __future__ import annotations

import json
import sqlite3
import time

from . import core, store

BATCH_SIZE = 256


def request_rebuild(db_path: str, masked_channels) -> dict | None:
    """Register a candidate scheme for a new mask.

    Returns None when the requested mask already is the active scheme
    (nothing to rebuild). Otherwise returns the new scheme row; the caller
    is responsible for running run_rebuild() (usually in a thread).
    """
    mask = list(core.validate_mask(masked_channels))
    conn = store.connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            active = store._active_scheme_row(conn)
            if json.loads(active["mask_json"]) == mask:
                conn.execute("ROLLBACK")
                return None
            cur = conn.execute(
                "INSERT INTO schemes (mask_json, status, base_scheme_id) "
                "VALUES (?, 'building', ?)",
                (json.dumps(mask), active["scheme_id"]),
            )
            new_id = cur.lastrowid
            conn.execute("COMMIT")
            return {
                "scheme_id": new_id,
                "base_scheme_id": active["scheme_id"],
                "masked_channels": mask,
                "status": "building",
            }
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()


def _apply_rows(
    conn: sqlite3.Connection, scheme_id: int, mask: list[int], rows, in_txn: bool = False
) -> None:
    """(Re)build candidate-index rows for the given spectra rows."""
    if not rows:
        return
    if not in_txn:
        conn.execute("BEGIN IMMEDIATE")
    try:
        for r in rows:
            vec = core.normalize(store._decode(r["intensities"]), mask)
            if vec is None:
                # All-zero after masking: skipped from the index (and any
                # stale row for this spectrum must not survive).
                conn.execute(
                    "DELETE FROM index_vectors WHERE scheme_id = ? AND spectrum_id = ?",
                    (scheme_id, r["spectrum_id"]),
                )
            else:
                conn.execute(
                    "INSERT INTO index_vectors (scheme_id, spectrum_id, vector, built_seq) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(scheme_id, spectrum_id) DO UPDATE SET "
                    "vector = excluded.vector, built_seq = excluded.built_seq",
                    (scheme_id, r["spectrum_id"], store._encode(vec), r["seq"]),
                )
        if not in_txn:
            conn.execute("COMMIT")
    except Exception:
        if not in_txn:
            conn.execute("ROLLBACK")
        raise


def _mark_scheme(db_path: str, scheme_id: int, status: str) -> None:
    conn = store.connect(db_path)
    try:
        conn.execute(
            "UPDATE schemes SET status = ? WHERE scheme_id = ? AND status = 'building'",
            (status, scheme_id),
        )
    finally:
        conn.close()


def run_rebuild(db_path: str, scheme_id: int, batch_delay_ms: int = 0) -> str:
    """Build the candidate index and try to activate it.

    Returns the final status: 'active' | 'abandoned' | 'failed'.
    """
    conn = store.connect(db_path)
    try:
        row = conn.execute(
            "SELECT mask_json, base_scheme_id FROM schemes WHERE scheme_id = ?",
            (scheme_id,),
        ).fetchone()
        if row is None:
            return "failed"
        mask = json.loads(row["mask_json"])
        base_scheme_id = row["base_scheme_id"]

        last_seq = 0
        # Phase 1: incremental catch-up in small transactions.
        while True:
            rows = conn.execute(
                "SELECT spectrum_id, intensities, seq FROM spectra "
                "WHERE seq > ? ORDER BY seq LIMIT ?",
                (last_seq, BATCH_SIZE),
            ).fetchall()
            if not rows:
                break
            _apply_rows(conn, scheme_id, mask, rows)
            last_seq = rows[-1]["seq"]
            if batch_delay_ms:
                time.sleep(batch_delay_ms / 1000.0)

        # Phase 2: final catch-up + atomic compare-and-swap activation.
        conn.execute("BEGIN IMMEDIATE")
        try:
            rows = conn.execute(
                "SELECT spectrum_id, intensities, seq FROM spectra "
                "WHERE seq > ? ORDER BY seq",
                (last_seq,),
            ).fetchall()
            _apply_rows(conn, scheme_id, mask, rows, in_txn=True)
            cur = conn.execute(
                "UPDATE active_scheme SET scheme_id = ? "
                "WHERE id = 1 AND scheme_id = ?",
                (scheme_id, base_scheme_id),
            )
            if cur.rowcount != 1:
                # A newer scheme was activated after this rebuild started:
                # we are late and must not overwrite it.
                conn.execute("ROLLBACK")
                _mark_scheme(db_path, scheme_id, "abandoned")
                return "abandoned"
            conn.execute(
                "UPDATE schemes SET status = 'active' WHERE scheme_id = ?",
                (scheme_id,),
            )
            if base_scheme_id is not None:
                conn.execute(
                    "UPDATE schemes SET status = 'superseded' "
                    "WHERE scheme_id = ? AND status = 'active'",
                    (base_scheme_id,),
                )
            conn.execute("COMMIT")
            return "active"
        except Exception:
            conn.execute("ROLLBACK")
            raise
    except Exception:
        _mark_scheme(db_path, scheme_id, "failed")
        return "failed"
    finally:
        conn.close()
