"""Frozen result types. No validation or conversion lives here (SPEC §3)."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

type Tail = Literal["high", "low"]
type SampleMethod = Literal["BM", "POT"]
type ReturnLevelMethod = Literal["GEV", "GPD"]


@dataclass(frozen=True, slots=True)
class ExtremeSeries:
    """Input observations. Construct via ``series.py``; validate via ``validate.py``."""

    values: np.ndarray
    timestamps: np.ndarray | None
    tail: Tail


@dataclass(frozen=True, slots=True)
class ExtremeSample:
    """Extracted extremes. BM leaves threshold/excesses/exceedance_rate as None."""

    raw_values: np.ndarray
    transformed_values: np.ndarray
    timestamps: np.ndarray | None
    method: SampleMethod
    threshold: float | None
    excesses: np.ndarray | None
    n_source: int
    exposure: float | None
    exceedance_rate: float | None


@dataclass(frozen=True, slots=True)
class GEVFit:
    location: float
    scale: float
    shape: float
    log_likelihood: float
    aic: float
    n_extremes: int
    converged: bool


@dataclass(frozen=True, slots=True)
class GPDFit:
    scale: float
    shape: float
    log_likelihood: float
    aic: float
    n_extremes: int
    converged: bool


@dataclass(frozen=True, slots=True)
class ReturnLevelResult:
    estimate: float
    return_period: float
    confidence_level: float | None
    lower: float | None
    upper: float | None
    extrapolation_factor: float
    method: ReturnLevelMethod


@dataclass(frozen=True, slots=True)
class DiagnosticResult:
    qq_empirical: np.ndarray
    qq_model: np.ndarray
    pp_empirical: np.ndarray
    pp_model: np.ndarray
    ks_distance: float


MODEL_TYPES: tuple[type[Any], ...] = (
    ExtremeSeries,
    ExtremeSample,
    GEVFit,
    GPDFit,
    ReturnLevelResult,
    DiagnosticResult,
)
