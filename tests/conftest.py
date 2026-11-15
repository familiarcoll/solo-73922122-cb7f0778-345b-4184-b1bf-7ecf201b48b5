"""Shared fixtures: every test gets a fresh SQLite file in a tmp dir."""
import os

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def db(tmp_path, monkeypatch):
    path = str(tmp_path / "test.db")
    monkeypatch.setenv("MS_DB_PATH", path)
    monkeypatch.delenv("MS_REBUILD_BATCH_DELAY_MS", raising=False)
    monkeypatch.delenv("MS_MAX_SPECTRA", raising=False)
    return path


@pytest.fixture()
def client(db):
    with TestClient(app) as c:
        yield c


def spectrum(sid, peaks, n=256):
    """Deterministic synthetic spectrum: sum of square peaks."""
    v = [0.0] * n
    for center, width, height in peaks:
        for i in range(max(0, center - width), min(n, center + width + 1)):
            v[i] += height
    return {"spectrum_id": sid, "intensities": v}


def query_vec(peaks, n=256):
    v = [0.0] * n
    for center, width, height in peaks:
        for i in range(max(0, center - width), min(n, center + width + 1)):
            v[i] += height
    return {"intensities": v}


def wait_scheme_status(client, scheme_id, want=("active", "abandoned", "failed"), timeout=10.0):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        for s in client.get("/schemes").json()["schemes"]:
            if s["scheme_id"] == scheme_id and s["status"] in want:
                return s
        time.sleep(0.05)
    raise AssertionError(f"scheme {scheme_id} did not reach {want} in {timeout}s")
