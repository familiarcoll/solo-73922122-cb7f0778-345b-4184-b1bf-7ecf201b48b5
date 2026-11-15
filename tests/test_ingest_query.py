"""Core ingest/query behaviour."""
import math


def test_ingest_then_query_returns_top3(client):
    from conftest import query_vec, spectrum

    client.post("/spectra", json=spectrum("A", [(10, 2, 100), (100, 2, 50)]))
    client.post("/spectra", json=spectrum("B", [(10, 2, 90), (100, 2, 60)]))
    client.post("/spectra", json=spectrum("C", [(10, 2, 10), (200, 2, 200)]))
    client.post("/spectra", json=spectrum("D", [(220, 2, 300)]))

    r = client.post("/query", json=query_vec([(10, 2, 95), (100, 2, 55)]))
    assert r.status_code == 200
    body = r.json()
    assert body["scheme_id"] == 1
    assert body["masked_channels"] == []
    ids = [x["spectrum_id"] for x in body["results"]]
    assert ids == ["A", "B", "C"]  # D is orthogonal, ranks last, only 3 returned
    scores = [x["score"] for x in body["results"]]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_identical_query_scores_one(client):
    from conftest import query_vec, spectrum

    client.post("/spectra", json=spectrum("X", [(50, 3, 10), (150, 3, 20)]))
    r = client.post("/query", json=query_vec([(50, 3, 10), (150, 3, 20)]))
    top = r.json()["results"][0]
    assert top["spectrum_id"] == "X"
    assert math.isclose(top["score"], 1.0, abs_tol=1e-9)


def test_ties_broken_by_spectrum_id(client):
    from conftest import query_vec, spectrum

    # Same shape -> identical normalized vectors -> identical cosine.
    client.post("/spectra", json=spectrum("ZULU", [(30, 2, 5), (90, 2, 9)]))
    client.post("/spectra", json=spectrum("ALPHA", [(30, 2, 5), (90, 2, 9)]))
    client.post("/spectra", json=spectrum("MIKE", [(30, 2, 5), (90, 2, 9)]))
    r = client.post("/query", json=query_vec([(30, 2, 5), (90, 2, 9)]))
    ids = [x["spectrum_id"] for x in r.json()["results"]]
    assert ids == ["ALPHA", "MIKE", "ZULU"]
    scores = [x["score"] for x in r.json()["results"]]
    assert len(set(scores)) == 1  # exact tie


def test_same_id_replaces(client):
    from conftest import query_vec, spectrum

    client.post("/spectra", json=spectrum("S", [(10, 2, 100)]))
    r = client.post("/spectra", json=spectrum("S", [(200, 2, 100)]))
    assert r.json()["library_size"] == 1  # replaced, not added
    r = client.post("/query", json=query_vec([(200, 2, 100)]))
    assert r.json()["results"][0]["spectrum_id"] == "S"
    assert math.isclose(r.json()["results"][0]["score"], 1.0, abs_tol=1e-9)
    r = client.post("/query", json=query_vec([(10, 2, 100)]))
    assert r.json()["results"][0]["score"] < 1e-9  # old vector is gone


def test_ingest_validation_errors(client):
    from conftest import spectrum

    assert client.post("/spectra", json=spectrum("B", [(1, 1, 1)], n=255)).status_code == 422
    bad = spectrum("B", [(1, 1, 1)])
    bad["intensities"][7] = -0.5
    assert client.post("/spectra", json=bad).status_code == 422
    bad2 = spectrum("B", [(1, 1, 1)])
    bad2["intensities"][7] = float("nan")
    # NaN is not valid JSON for the client encoder; send a raw body instead.
    raw = '{"spectrum_id":"B","intensities":[' + ",".join(
        "NaN" if i == 7 else "1.0" for i in range(256)
    ) + "]}"
    r = client.post("/spectra", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert client.post("/spectra", json={"spectrum_id": "", "intensities": [0.0] * 256}).status_code == 422


def test_max_spectra_cap(client, monkeypatch):
    from conftest import spectrum

    monkeypatch.setenv("MS_MAX_SPECTRA", "3")
    for i in range(3):
        r = client.post("/spectra", json=spectrum(f"S{i}", [(i + 1, 1, 1)]))
        assert r.status_code == 200
    r = client.post("/spectra", json=spectrum("S3", [(10, 1, 1)]))
    assert r.status_code == 409
    # Replacing an existing id is still allowed at the cap.
    r = client.post("/spectra", json=spectrum("S0", [(20, 1, 1)]))
    assert r.status_code == 200
    assert r.json()["library_size"] == 3


def test_query_on_empty_library(client):
    from conftest import query_vec

    r = client.post("/query", json=query_vec([(5, 2, 1)]))
    assert r.status_code == 200
    assert r.json()["results"] == []
