"""Generate inverse-CDF golden samples and engine snapshots (SPEC §6)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
if __package__ is None:
    sys.path.insert(0, str(REPO))

import numpy as np

from evt.extract import extract_peaks_over_threshold
from evt.serialize import to_dict
from evt.series import make_extreme_series
from evt.types import ExtremeSample
from validation.engine_eval import evaluate_sample
from validation.inverse_cdf import gev_quantile, gpd_excess_quantile, uniform_open

SAMPLES = ROOT / "fixtures" / "samples"
SNAPSHOTS = ROOT / "fixtures" / "snapshots"
CATALOG = ROOT / "fixtures" / "catalog.json"

GEV_LOCATION = 10.0
GEV_SCALE = 2.0
GPD_SCALE = 1.5
GPD_THRESHOLD = 0.0
GPD_RATE = 2.0

# SPEC §6.5 coverage (compact cartesian of representative cells).
GEV_CASES: list[tuple[str, int, float]] = [
    ("gev_n15_xi-0.3", 15, -0.3),
    ("gev_n15_xi0", 15, 0.0),
    ("gev_n30_xi-0.01", 30, -0.01),
    ("gev_n30_xi0", 30, 0.0),
    ("gev_n30_xi0.01", 30, 0.01),
    ("gev_n30_xi0.3", 30, 0.3),
    ("gev_n100_xi-0.3", 100, -0.3),
    ("gev_n100_xi0", 100, 0.0),
    ("gev_n100_xi0.3", 100, 0.3),
]
GPD_CASES: list[tuple[str, int, float]] = [
    ("gpd_n15_xi-0.3", 15, -0.3),
    ("gpd_n15_xi0", 15, 0.0),
    ("gpd_n30_xi-0.01", 30, -0.01),
    ("gpd_n30_xi0", 30, 0.0),
    ("gpd_n30_xi0.01", 30, 0.01),
    ("gpd_n30_xi0.3", 30, 0.3),
    ("gpd_n100_xi-0.3", 100, -0.3),
    ("gpd_n100_xi0", 100, 0.0),
    ("gpd_n100_xi0.3", 100, 0.3),
]


def _stable_seed(name: str) -> int:
    # Deterministic across processes (builtin hash is salted).
    acc = 2166136261
    for byte in name.encode("utf-8"):
        acc ^= byte
        acc = (acc * 16777619) % 2**32
    return int(acc)


def _bm_sample(values: np.ndarray) -> ExtremeSample:
    return ExtremeSample(
        raw_values=values,
        transformed_values=values,
        timestamps=None,
        method="BM",
        threshold=None,
        excesses=None,
        n_source=int(values.size),
        exposure=None,
        exceedance_rate=None,
    )


def _pot_sample(excesses: np.ndarray) -> ExtremeSample:
    raw = excesses + GPD_THRESHOLD
    return ExtremeSample(
        raw_values=raw,
        transformed_values=raw,
        timestamps=None,
        method="POT",
        threshold=GPD_THRESHOLD,
        excesses=excesses,
        n_source=int(excesses.size) * 5,
        exposure=float(excesses.size) / GPD_RATE,
        exceedance_rate=GPD_RATE,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _declustered_sample() -> ExtremeSample:
    """Clustered exceedances so run declustering materially reduces the count."""
    values = np.array(
        [0.0, 6.0, 7.0, 6.5, 1.0, 0.5, 8.0, 9.0, 8.5, 2.0, 1.0, 7.5, 8.2, 0.0],
        dtype=np.float64,
    )
    timestamps = np.arange(
        np.datetime64("2000-01-01"),
        np.datetime64("2000-01-15"),
        dtype="datetime64[D]",
    ).astype("datetime64[ns]")
    series = make_extreme_series(values, timestamps=timestamps, tail="high")
    return extract_peaks_over_threshold(
        series, threshold=5.0, decluster_method="run", run_length=0
    )


def generate() -> None:
    catalog: list[dict[str, Any]] = []
    SAMPLES.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)

    for name, n, xi in GEV_CASES:
        rng = np.random.default_rng(_stable_seed(name))
        p = uniform_open(rng, n)
        values = gev_quantile(p, GEV_LOCATION, GEV_SCALE, xi)
        sample = _bm_sample(values)
        _write_json(SAMPLES / f"{name}.json", to_dict(sample))
        snapshot = evaluate_sample(sample, "GEV")
        snapshot["true_params"] = {"location": GEV_LOCATION, "scale": GEV_SCALE, "shape": xi}
        _write_json(SNAPSHOTS / f"{name}.json", snapshot)
        catalog.append({"name": name, "model": "GEV"})

    for name, n, xi in GPD_CASES:
        rng = np.random.default_rng(_stable_seed(name))
        p = uniform_open(rng, n)
        excesses = gpd_excess_quantile(p, GPD_SCALE, xi)
        sample = _pot_sample(excesses)
        _write_json(SAMPLES / f"{name}.json", to_dict(sample))
        snapshot = evaluate_sample(sample, "GPD")
        snapshot["true_params"] = {"location": None, "scale": GPD_SCALE, "shape": xi}
        _write_json(SNAPSHOTS / f"{name}.json", snapshot)
        catalog.append({"name": name, "model": "GPD"})

    declustered = _declustered_sample()
    decluster_name = "pot_declustered_run"
    _write_json(SAMPLES / f"{decluster_name}.json", to_dict(declustered))
    snapshot = evaluate_sample(declustered, "GPD")
    snapshot["true_params"] = None
    _write_json(SNAPSHOTS / f"{decluster_name}.json", snapshot)
    catalog.append(
        {
            "name": decluster_name,
            "model": "GPD",
            "notes": "run declustering reduces exceedance count versus n_source",
        }
    )
    _write_json(CATALOG, {"fixtures": catalog})


if __name__ == "__main__":
    generate()
    print(f"Wrote samples to {SAMPLES}")
    print(f"Wrote snapshots to {SNAPSHOTS}")
