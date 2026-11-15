"""Generate deterministic synthetic example spectra into examples/spectra/."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "spectra")
N = 256


def gaussian_peak(center, width, height):
    x = np.arange(N, dtype=np.float64)
    return height * np.exp(-0.5 * ((x - center) / width) ** 2)


def make_spectrum(peaks, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    v = np.zeros(N, dtype=np.float64)
    for c, w, h in peaks:
        v += gaussian_peak(c, w, h)
    if noise:
        v += rng.uniform(0.0, noise, N)
    return np.round(v, 6).tolist()


SPECTRA = {
    "Caffeine": [(40, 3, 900), (95, 4, 500), (150, 3, 750), (210, 5, 300)],
    "Theobromine": [(42, 3, 850), (120, 4, 600), (150, 3, 700), (222, 4, 250)],
    "Nicotine": [(60, 3, 1000), (61, 2, 400), (180, 4, 350)],
    "Glucose": [(20, 4, 300), (80, 4, 800), (160, 4, 800), (240, 4, 300)],
    "Fructose": [(21, 4, 320), (80, 4, 780), (163, 4, 820), (238, 4, 280)],
}


def artifact_only():
    """Signal strictly inside channels 10-12: skipped once those are masked."""
    v = [0.0] * N
    v[10], v[11], v[12] = 400.0, 999.0, 400.0
    return v


def main():
    os.makedirs(OUT, exist_ok=True)
    for i, (name, peaks) in enumerate(SPECTRA.items()):
        doc = {
            "spectrum_id": f"STD-{i + 1:04d}-{name}",
            "intensities": make_spectrum(peaks, noise=1.0, seed=100 + i),
        }
        with open(os.path.join(OUT, f"{doc['spectrum_id']}.json"), "w") as fh:
            json.dump(doc, fh, indent=1)
    doc = {"spectrum_id": f"STD-{len(SPECTRA) + 1:04d}-Artifact-Only",
           "intensities": artifact_only()}
    with open(os.path.join(OUT, f"{doc['spectrum_id']}.json"), "w") as fh:
        json.dump(doc, fh, indent=1)
    # Unknown sample: noisy caffeine with a slightly shifted minor peak.
    unknown = make_spectrum(
        [(40, 3, 880), (95, 4, 510), (151, 3, 730), (210, 5, 290)],
        noise=6.0,
        seed=7,
    )
    with open(os.path.join(HERE, "unknown.json"), "w") as fh:
        json.dump({"intensities": unknown}, fh, indent=1)
    print(f"wrote {len(SPECTRA) + 1} library spectra + 1 unknown to {OUT}")


if __name__ == "__main__":
    main()
