"""Masking, rebuild switching, concurrency and failure semantics."""
from app import rebuild, store


def _mask_and_wait(client, channels):
    r = client.post("/scheme/rebuild", json={"masked_channels": channels})
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "building"
    return body["scheme_id"]


def test_rebuild_switches_scheme_and_applies_mask(client):
    from conftest import query_vec, spectrum, wait_scheme_status

    # A and B differ ONLY in channel 3; C only in channel 3.
    client.post("/spectra", json=spectrum("A", [(10, 2, 100), (3, 0, 5)]))
    client.post("/spectra", json=spectrum("B", [(10, 2, 100), (3, 0, 50)]))
    client.post("/spectra", json=spectrum("C", [(3, 0, 999)]))

    before = client.post("/query", json=query_vec([(10, 2, 100)])).json()
    assert before["scheme_id"] == 1
    ids = [x["spectrum_id"] for x in before["results"]]
    assert ids == ["A", "B", "C"]  # C is orthogonal to the query (score 0)
    # The channel-3 difference separates A from B before masking.
    assert before["results"][0]["score"] > before["results"][1]["score"]

    sid = _mask_and_wait(client, [3])
    done = wait_scheme_status(client, sid)
    assert done["status"] == "active"
    assert done["indexed_count"] == 2  # C is all-zero after masking -> skipped

    after = client.post("/query", json=query_vec([(10, 2, 100)])).json()
    assert after["scheme_id"] == sid
    assert after["masked_channels"] == [3]
    ids = [x["spectrum_id"] for x in after["results"]]
    assert "C" not in ids
    # A and B now identical after masking -> tie broken by id.
    assert ids[:2] == ["A", "B"]


def test_library_spectrum_zero_after_mask_is_skipped(client):
    from conftest import query_vec, spectrum, wait_scheme_status

    sid = _mask_and_wait(client, [0, 1, 2])
    wait_scheme_status(client, sid)
    # This spectrum only has signal in masked channels: stored but not indexed.
    r = client.post("/spectra", json=spectrum("GHOST", [(1, 0, 42)]))
    assert r.status_code == 200
    assert r.json()["indexed"] is False
    r = client.post("/query", json=query_vec([(50, 2, 1)]))
    assert "GHOST" not in [x["spectrum_id"] for x in r.json()["results"]]
    # ...and it is still in the raw library.
    assert client.get("/scheme/active").json()["library_size"] == 1


def test_query_zero_after_mask_is_rejected(client):
    from conftest import query_vec, wait_scheme_status

    sid = _mask_and_wait(client, [10, 11, 12])
    wait_scheme_status(client, sid)
    r = client.post("/query", json=query_vec([(11, 0, 5)]))
    assert r.status_code == 422


def test_replace_with_zero_after_mask_removes_old_vector(client):
    from conftest import query_vec, spectrum, wait_scheme_status

    client.post("/spectra", json=spectrum("S", [(50, 0, 10)]))  # channel 50 only
    sid = _mask_and_wait(client, [50])
    wait_scheme_status(client, sid)
    # Re-ingest S: under the active mask it is all-zero -> dropped from index.
    r = client.post("/spectra", json=spectrum("S", [(50, 0, 10)]))
    assert r.json()["indexed"] is False
    r = client.post("/query", json=query_vec([(50, 0, 10)]))
    assert r.status_code == 422  # query itself is all-zero under the mask
    r = client.post("/query", json=query_vec([(7, 2, 1)]))
    assert "S" not in [x["spectrum_id"] for x in r.json()["results"]]


def test_rebuild_same_mask_is_noop(client):
    r = client.post("/scheme/rebuild", json={"masked_channels": []})
    assert r.status_code == 202
    assert r.json()["status"] == "already_active"


def test_invalid_mask_rejected(client):
    assert client.post("/scheme/rebuild", json={"masked_channels": [256]}).status_code == 422
    assert client.post("/scheme/rebuild", json={"masked_channels": [-1]}).status_code == 422
    assert client.post("/scheme/rebuild", json={"masked_channels": [1.5]}).status_code == 422


def test_rebuild_includes_spectra_ingested_during_rebuild(client, monkeypatch):
    from conftest import query_vec, spectrum, wait_scheme_status

    monkeypatch.setenv("MS_REBUILD_BATCH_DELAY_MS", "30")
    for i in range(600):  # several rebuild batches
        client.post("/spectra", json=spectrum(f"PRE-{i:04d}", [(i % 200, 1, 1 + (i % 7))]))

    r = client.post("/scheme/rebuild", json={"masked_channels": [0, 1]})
    sid = r.json()["scheme_id"]

    # Ingest while the rebuild is (very likely) still running.
    client.post("/spectra", json=spectrum("DURING", [(150, 2, 77)]))
    done = wait_scheme_status(client, sid, timeout=30.0)
    assert done["status"] == "active"

    q = client.post("/query", json=query_vec([(150, 2, 77)])).json()
    assert q["scheme_id"] == sid
    assert q["results"][0]["spectrum_id"] == "DURING"
    assert q["results"][0]["score"] > 0.999999
    # Every pre-existing spectrum made it into the new index as well, except
    # PRE-0000/0200/0400 whose only signal sits in masked channels 0-1.
    assert client.get("/scheme/active").json()["indexed_count"] == 601 - 3


def test_late_rebuild_does_not_override_newer_scheme(db):
    """Two rebuilds requested off the same base; the slower one must lose."""
    from app import config

    store.init_db(db)
    store.upsert_spectrum(db, "S1", [1.0] * 256, config.max_spectra())

    a = rebuild.request_rebuild(db, [5])   # candidate A, base = scheme 1
    b = rebuild.request_rebuild(db, [6])   # candidate B, same base
    assert a["base_scheme_id"] == b["base_scheme_id"] == 1

    # B finishes first and activates; A is late and must not overwrite it.
    assert rebuild.run_rebuild(db, b["scheme_id"]) == "active"
    assert rebuild.run_rebuild(db, a["scheme_id"]) == "abandoned"

    active = store.get_active_scheme(db)
    assert active["scheme_id"] == b["scheme_id"]
    assert active["masked_channels"] == [6]
    statuses = {s["scheme_id"]: s["status"] for s in store.list_schemes(db)}
    assert statuses[a["scheme_id"]] == "abandoned"
    assert statuses[b["scheme_id"]] == "active"


def test_failed_rebuild_keeps_old_index(db, monkeypatch):
    from app import config

    store.init_db(db)
    store.upsert_spectrum(db, "S1", [0.0, 1.0] + [0.0] * 254, config.max_spectra())
    before = store.read_index_snapshot(db)

    req = rebuild.request_rebuild(db, [1])
    monkeypatch.setattr(
        rebuild, "_apply_rows",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert rebuild.run_rebuild(db, req["scheme_id"]) == "failed"

    after = store.read_index_snapshot(db)
    assert after.scheme_id == before.scheme_id
    assert after.ids == before.ids
    statuses = {s["scheme_id"]: s["status"] for s in store.list_schemes(db)}
    assert statuses[req["scheme_id"]] == "failed"
    assert statuses[before.scheme_id] == "active"


def test_persistence_across_restart(db):
    """A brand-new app instance over the same file keeps data + index."""
    from fastapi.testclient import TestClient

    from app.main import app
    from conftest import query_vec, spectrum

    with TestClient(app) as c1:
        c1.post("/spectra", json=spectrum("KEEP", [(33, 2, 12)]))
        sid = _mask_and_wait(c1, [0])
        from conftest import wait_scheme_status

        wait_scheme_status(c1, sid)

    # Simulate a restart: new client (lifespan re-runs) on the same file.
    with TestClient(app) as c2:
        q = c2.post("/query", json=query_vec([(33, 2, 12)])).json()
        assert q["scheme_id"] == sid
        assert q["results"][0]["spectrum_id"] == "KEEP"
        assert q["results"][0]["score"] > 0.999999
        assert c2.get("/scheme/active").json()["masked_channels"] == [0]


def test_two_app_instances_share_one_database(db):
    """Two API instances on one SQLite file behave like two processes."""
    from fastapi.testclient import TestClient

    from app.main import app
    from conftest import query_vec, spectrum, wait_scheme_status

    with TestClient(app) as c1, TestClient(app) as c2:
        c1.post("/spectra", json=spectrum("SHARED", [(70, 2, 5)]))
        q = c2.post("/query", json=query_vec([(70, 2, 5)])).json()
        assert q["results"][0]["spectrum_id"] == "SHARED"

        r = c1.post("/scheme/rebuild", json={"masked_channels": [9]})
        sid = r.json()["scheme_id"]
        wait_scheme_status(c2, sid)  # instance 2 observes instance 1's rebuild
        q = c2.post("/query", json=query_vec([(70, 2, 5)])).json()
        assert q["scheme_id"] == sid
        assert q["masked_channels"] == [9]


def test_queries_use_old_scheme_while_rebuild_runs(client, monkeypatch):
    from conftest import query_vec, spectrum, wait_scheme_status

    monkeypatch.setenv("MS_REBUILD_BATCH_DELAY_MS", "50")
    for i in range(400):
        client.post("/spectra", json=spectrum(f"P{i:03d}", [(100 + i % 50, 1, 2)]))

    r = client.post("/scheme/rebuild", json={"masked_channels": [100]})
    sid = r.json()["scheme_id"]
    # While building, queries must still be served by the old scheme (id 1).
    q = client.post("/query", json=query_vec([(120, 1, 2)])).json()
    assert q["scheme_id"] == 1
    wait_scheme_status(client, sid, timeout=30.0)
    q = client.post("/query", json=query_vec([(120, 1, 2)])).json()
    assert q["scheme_id"] == sid
