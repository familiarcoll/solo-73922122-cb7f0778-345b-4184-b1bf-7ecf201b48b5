"""Mass-spec candidate lookup API.

Ingest 256-channel non-negative spectra into a persistent standard library
(max 5000, same id replaces), mask bad channels + L2-normalize, and answer
top-3 cosine-similarity queries for unknown samples. Mask changes trigger
an asynchronous index rebuild with atomic, consistent switching.
"""
from __future__ import annotations

import math
import threading
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, field_validator

from . import config, core, rebuild, store


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_db(config.db_path())
    yield


app = FastAPI(
    title="Mass-Spec Candidate Lookup",
    version="1.0.0",
    description=(
        "Standard-spectrum library for mass-spectrometry data engineers. "
        "Ingest 256-channel non-negative intensity spectra (max 5000, same "
        "spectrum id replaces), query unknown samples for the top-3 cosine "
        "candidates, and rebuild the persisted index when the bad-channel "
        "mask changes. All state lives in a shared SQLite database."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------- models


def _sanitize(obj):
    """Replace non-finite floats so the 422 body stays valid JSON."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return repr(obj)
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    # The default handler echoes the offending input verbatim; a NaN/Infinity
    # input would make json.dumps fail and turn the 422 into a 500.
    return JSONResponse(
        status_code=422,
        content={"detail": _sanitize(jsonable_encoder(exc.errors()))},
    )


class SpectrumIn(BaseModel):
    spectrum_id: str = Field(min_length=1, max_length=128, examples=["STD-0001"])
    intensities: list[float] = Field(
        min_length=core.N_CHANNELS, max_length=core.N_CHANNELS
    )

    @field_validator("intensities")
    @classmethod
    def _valid_intensities(cls, v):
        core.validate_intensities(v)  # ValueError subclass -> HTTP 422
        return v


class QueryIn(BaseModel):
    intensities: list[float] = Field(
        min_length=core.N_CHANNELS, max_length=core.N_CHANNELS
    )

    @field_validator("intensities")
    @classmethod
    def _valid_intensities(cls, v):
        core.validate_intensities(v)
        return v


class SchemeIn(BaseModel):
    masked_channels: list[int] = Field(
        default_factory=list,
        description="Bad channels to mask before L2 normalization (0..255).",
        examples=[[3, 17, 200]],
    )

    @field_validator("masked_channels")
    @classmethod
    def _valid_mask(cls, v):
        return list(core.validate_mask(v))


# ---------------------------------------------------------------- routes


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/spectra", summary="Ingest (or replace) a library spectrum")
def ingest_spectrum(body: SpectrumIn):
    """Upsert one spectrum. Same `spectrum_id` replaces the stored vector.

    The spectrum is indexed under the currently active scheme in the same
    transaction, so it is searchable as soon as this call succeeds. A
    spectrum that is all-zero after masking is stored but skipped from the
    index (`indexed: false`).
    """
    try:
        res = store.upsert_spectrum(
            config.db_path(),
            body.spectrum_id,
            body.intensities,
            config.max_spectra(),
        )
    except store.LibraryFullError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {
        "spectrum_id": res.spectrum_id,
        "indexed": res.indexed,
        "scheme_id": res.scheme_id,
        "library_size": res.library_size,
        "max_spectra": config.max_spectra(),
    }


@app.post("/query", summary="Find top-3 candidates for an unknown spectrum")
def query_spectrum(body: QueryIn):
    """Process the unknown spectrum with the *same* active scheme (mask +
    L2 normalization) and return the top-3 library candidates by cosine
    similarity. Ties are ordered by ascending spectrum id.

    The scheme, the indexed vectors and the returned scores are read from
    one committed snapshot. An unknown spectrum that is all-zero after
    masking is rejected with 422.
    """
    snap = store.read_index_snapshot(config.db_path())
    vec = core.normalize(np.asarray(body.intensities, dtype=np.float64), snap.masked_channels)
    if vec is None:
        raise HTTPException(
            status_code=422,
            detail="query spectrum is all-zero after masking the bad channels",
        )
    top = core.top_k_cosine(vec, snap.ids, snap.matrix, k=3)
    return {
        "scheme_id": snap.scheme_id,
        "masked_channels": list(snap.masked_channels),
        "results": [
            {"spectrum_id": sid, "score": score} for sid, score in top
        ],
    }


@app.get("/scheme/active", summary="Show the active processing scheme")
def active_scheme():
    return store.get_active_scheme(config.db_path())


@app.get("/schemes", summary="List all schemes (poll rebuild progress here)")
def schemes():
    return {"schemes": store.list_schemes(config.db_path())}


@app.post(
    "/scheme/rebuild",
    status_code=202,
    summary="Change the bad-channel mask and rebuild the index",
)
def rebuild_scheme(body: SchemeIn):
    """Start an asynchronous rebuild for a new bad-channel mask.

    While the rebuild runs, ingests and queries keep working and queries
    keep using the old scheme. On success the new index — including every
    spectrum ingested during the rebuild — is activated atomically; a
    failed or late rebuild never disturbs the active index. Poll
    `GET /schemes` for the status (`building` → `active` / `abandoned` /
    `failed`).
    """
    req = rebuild.request_rebuild(config.db_path(), body.masked_channels)
    if req is None:
        active = store.get_active_scheme(config.db_path())
        return {
            "scheme_id": active["scheme_id"],
            "masked_channels": active["masked_channels"],
            "status": "already_active",
            "detail": "requested mask is already the active scheme; no rebuild needed",
        }
    thread = threading.Thread(
        target=rebuild.run_rebuild,
        args=(config.db_path(), req["scheme_id"], config.rebuild_batch_delay_ms()),
        daemon=True,
    )
    thread.start()
    return req
