"""Pure numerical core: validation, masking, L2 normalization, cosine top-k."""
from __future__ import annotations

import numpy as np

N_CHANNELS = 256


class SpectrumValidationError(ValueError):
    """Raised when an intensity vector does not meet the instrument contract."""


def validate_intensities(values) -> np.ndarray:
    """Coerce to a (256,) float64 array of finite, non-negative values."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (N_CHANNELS,):
        raise SpectrumValidationError(
            f"expected {N_CHANNELS} channels, got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise SpectrumValidationError("intensities must be finite numbers")
    if np.any(arr < 0):
        raise SpectrumValidationError("intensities must be non-negative")
    return arr


def validate_mask(channels) -> tuple[int, ...]:
    """Validate bad-channel list; returns sorted unique tuple of ints."""
    if channels is None:
        return ()
    seen = set()
    for c in channels:
        if isinstance(c, bool) or not isinstance(c, (int, np.integer)):
            raise SpectrumValidationError(f"channel index must be an integer, got {c!r}")
        if not 0 <= int(c) < N_CHANNELS:
            raise SpectrumValidationError(
                f"channel index out of range [0, {N_CHANNELS}): {c}"
            )
        seen.add(int(c))
    return tuple(sorted(seen))


def normalize(intensities: np.ndarray, masked_channels) -> np.ndarray | None:
    """Mask bad channels then L2-normalize.

    Returns None when the vector is all-zero after masking (caller decides
    whether to skip a library spectrum or reject an unknown query).
    """
    v = np.array(intensities, dtype=np.float64, copy=True)
    if masked_channels:
        v[np.asarray(tuple(masked_channels), dtype=np.int64)] = 0.0
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return None
    return v / norm


def top_k_cosine(
    query_vec: np.ndarray,
    lib_ids: list[str],
    lib_matrix: np.ndarray,
    k: int = 3,
) -> list[tuple[str, float]]:
    """Cosine similarity against L2-normalized library rows (dot == cosine).

    Ties are broken by ascending spectrum_id for a deterministic ranking.
    """
    if not lib_ids:
        return []
    scores = lib_matrix @ query_vec
    order = sorted(range(len(lib_ids)), key=lambda i: (-scores[i], lib_ids[i]))
    return [(lib_ids[i], float(scores[i])) for i in order[:k]]
