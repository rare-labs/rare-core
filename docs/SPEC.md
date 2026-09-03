# rare-core V0.1 — Specification

**Engine name:** rare-core  
**Python import:** `evt` (`from evt import ...`)  
**Version:** 0.1  
**Schema version:** `"0.1"` (JSON serialization)  
**Python:** ≥ 3.14  
**Runtime dependencies:** NumPy ≥ 2.3, SciPy ≥ 1.16  

This document is the authoritative functional and numerical specification for V0.1.

---

## 1. Scope

### 1.1 In scope

Stationary univariate extreme-value analysis for upper and lower tails:

| Capability | Description |
|------------|-------------|
| Block maxima/minima | Extract extremes from contiguous blocks |
| POT | Peaks over / below threshold with optional declustering |
| Declustering | Run-length and time-window methods |
| GEV MLE | Maximum-likelihood fit to block maxima (transformed to upper tail) |
| GPD MLE | Maximum-likelihood fit to threshold excesses (`floc=0`) |
| Return levels | Parametric estimates for specified return periods |
| Bootstrap CI | Parametric bootstrap percentile intervals for return levels |
| Diagnostics | QQ/PP coordinates and descriptive KS distance |
| Serialization | Structured, JSON-serializable results |
| Validation | Comparison to pinned Extremes.jl and pyextremes on identical samples |

### 1.2 Out of scope

- Automatic BM-vs-POT or threshold selection
- Nonstationary / covariate models
- Bayesian or probability-weighted-moment (PWM) estimation
- Multivariate or spatial extremes
- Intensity–duration–frequency (IDF) curves
- MCP, REST, DuckDB, WASM, or GUI layers
- Domain-specific semantics (rainfall, power, finance, insurance)

---

## 2. Runtime constraints

1. **Python ≥ 3.14.**
2. **Runtime dependencies:** NumPy and SciPy only.
3. **No** Julia, pyextremes, pandas, xarray, or plotting libraries at runtime.
4. Core accepts NumPy-compatible numeric 1-D arrays and optional `datetime64[ns]` timestamps.
5. **Shape convention:** ξ (`xi`) in all public API and serialized output.
6. **SciPy GEV:** shape parameter `c` is translated as `xi = -c` at the adapter boundary (`fit/scipy_adapter.py`) only.
7. **No silent data mutation:** non-finite values, duplicate timestamps, and invalid timestamp ordering are validation errors.
8. **Reproducibility:** all stochastic procedures accept an explicit integer `seed`.

---

## 3. Data model

All types are immutable (`@dataclass(frozen=True)`). Serialized JSON includes `"schema_version": "0.1"`.

### 3.1 `ExtremeSeries`

Input time series or observations.

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| `values` | `float64` 1-D array | yes | finite, length ≥ 1 |
| `timestamps` | `datetime64[ns]` 1-D array | no | same length as `values`; strictly increasing; no duplicates |
| `tail` | `"high"` \| `"low"` | yes | — |

### 3.2 `ExtremeSample`

Extracted extremes ready for fitting.

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| `raw_values` | `float64` 1-D | yes | original scale |
| `transformed_values` | `float64` 1-D | yes | upper-tail orientation for fitting |
| `timestamps` | `datetime64[ns]` 1-D | no | aligned with values |
| `method` | `"BM"` \| `"POT"` | yes | — |
| `threshold` | `float` | POT only | original scale |
| `excesses` | `float64` 1-D | POT only | non-negative |
| `n_source` | `int` | yes | original observation count; ≥ len(extremes) |
| `exposure` | `float` | no | elapsed years from timestamps |
| `exceedance_rate` | `float` | POT only | declustered exceedances per exposure year (λ) |

**Invariants**

- BM: `threshold`, `excesses`, `exceedance_rate` are null.
- POT: `threshold`, `excesses` are present; all excesses ≥ 0.
- `len(raw_values) == len(transformed_values)`.

### 3.3 `GEVFit`

| Field | Type | Notes |
|-------|------|-------|
| `location` | `float` | μ |
| `scale` | `float` | σ > 0 |
| `shape` | `float` | ξ |
| `log_likelihood` | `float` | canonical, sum of log pdf |
| `aic` | `float` | `2k - 2·LL`, k = 3 |
| `n_extremes` | `int` | sample size used |
| `converged` | `bool` | see §5.2 |

### 3.4 `GPDFit`

| Field | Type | Notes |
|-------|------|-------|
| `scale` | `float` | σ > 0 |
| `shape` | `float` | ξ |
| `log_likelihood` | `float` | canonical |
| `aic` | `float` | k = 2 |
| `n_extremes` | `int` | — |
| `converged` | `bool` | — |

### 3.5 `ReturnLevelResult`

| Field | Type | Notes |
|-------|------|-------|
| `estimate` | `float` | original scale |
| `return_period` | `float` | T (same time unit as λ or blocks) |
| `confidence_level` | `float` \| null | e.g. 0.95; null for point-only |
| `lower` | `float` \| null | bootstrap lower bound |
| `upper` | `float` \| null | bootstrap upper bound |
| `extrapolation_factor` | `float` | T·b (GEV) or λ·T (POT) |
| `method` | `"GEV"` \| `"GPD"` | — |

### 3.6 `DiagnosticResult`

| Field | Type | Notes |
|-------|------|-------|
| `qq_empirical` | 1-D array | empirical quantiles |
| `qq_model` | 1-D array | model quantiles |
| `pp_empirical` | 1-D array | empirical probabilities |
| `pp_model` | 1-D array | model probabilities |
| `ks_distance` | `float` | max \|F_emp − F_model\| on evaluation grid |

---

## 4. Public API

### 4.1 Validation and construction

```python
def validate_series(values, timestamps, tail) -> None: ...
def ExtremeSeries(values, timestamps=None, tail="high") -> ExtremeSeries: ...
```

Raises `ValidationError` with stable `code` (see §8).

### 4.2 Extraction

```python
def extract_block_maxima(
    series: ExtremeSeries,
    *,
    block_size: int,
    min_blocks: int = 1,
) -> ExtremeSample: ...

def extract_peaks_over_threshold(
    series: ExtremeSeries,
    *,
    threshold: float,
    decluster_method: Literal["run", "window"] | None = None,
    run_length: int = 1,
    window: np.timedelta64 | None = None,
) -> ExtremeSample: ...
```

#### Block maxima (BM)

1. Partition `values` into contiguous blocks of length `block_size`.
2. **Drop** the trailing incomplete block.
3. Per block: maximum if `tail=="high"`, minimum if `tail=="low"`.
4. Transform to upper-tail orientation (§4.3).
5. Set `n_source = len(series.values)`.
6. Raise `ExtractionError` if resulting block count < `min_blocks`.

#### Peaks over threshold (POT)

1. Candidate exceedances: `values > threshold` (high) or `values < threshold` (low).
2. Optional declustering (§4.2.1).
3. Cluster representative: maximum (high) or minimum (low) on original scale.
4. Compute excesses on transformed upper-tail scale.
5. If timestamps present:  
   `exposure_years = (t_max - t_min) / DAYS_PER_YEAR`  
   with `DAYS_PER_YEAR = 365.25` (constant in code).
6. `exceedance_rate = n_declustered / exposure_years` when exposure is defined.

##### 4.2.1 Declustering

| Method | Parameter | Algorithm |
|--------|-----------|-----------|
| `run` | `run_length: int` | Group consecutive exceedances separated by ≤ `run_length` non-exceedances; one extreme per group |
| `window` | `window: timedelta64` | Requires timestamps; cluster exceedances within `window` of each other; one extreme per cluster |

### 4.3 Tail transform

```python
def transform_to_upper_tail(values, tail) -> np.ndarray: ...
def inverse_transform_from_upper_tail(values, tail) -> np.ndarray: ...
```

- **High tail:** identity.
- **Low tail:** `x' = -x` (user must ensure resulting support is valid for GEV/GPD).

Return levels are computed on the transformed scale, then mapped back via `inverse_transform_from_upper_tail`.

### 4.4 Fitting

```python
def fit_gev(sample: ExtremeSample) -> GEVFit: ...
def fit_gpd(sample: ExtremeSample) -> GPDFit: ...
```

- GEV: fit `sample.transformed_values` via `scipy.stats.genextreme.fit`.
- GPD: fit `sample.excesses` via `scipy.stats.genpareto.fit(..., floc=0)`.
- Minimum sample sizes: GEV `n ≥ 3`, GPD `n ≥ 2`; else `FitError`.
- Recompute `log_likelihood` and `aic` using `likelihood.py` (§6).

### 4.5 Return levels

```python
def gev_return_level(
    fit: GEVFit,
    return_period: float,
    *,
    blocks_per_period: float = 1.0,
) -> ReturnLevelResult: ...

def gpd_return_level(
    fit: GPDFit,
    sample: ExtremeSample,
    return_period: float,
) -> ReturnLevelResult: ...
```

#### GEV (block maxima)

Target CDF level:

\[
F(z_T) = 1 - \frac{1}{T \cdot b}
\]

where `T = return_period`, `b = blocks_per_period`.

Let \(p = F(z_T)\).

| Condition | Return level (transformed scale) |
|-----------|----------------------------------|
| ξ ≠ 0 | \(z = \mu + \frac{\sigma}{\xi}\left((-\ln p)^{-\xi} - 1\right)\) |
| ξ → 0 | \(z = \mu - \sigma \ln(-\ln p)\) |

`extrapolation_factor = T * b`.

#### GPD (POT)

Requires `sample.threshold` and `sample.exceedance_rate` (λ). Let \(m = \lambda T\).

| Condition | Return level (high tail, original scale) |
|-----------|------------------------------------------|
| ξ ≠ 0 | \(z = u + \frac{\sigma}{\xi}(m^{\xi} - 1)\) |
| ξ → 0 | \(z = u + \sigma \ln m\) |

Apply inverse tail transform for low-tail series.

`extrapolation_factor = m`.

Raise `ValidationError` if POT metadata is missing.

### 4.6 Parametric bootstrap

```python
def bootstrap_return_levels(
    sample: ExtremeSample,
    fit: GEVFit | GPDFit,
    return_periods: Sequence[float],
    *,
    n_samples: int = 500,
    confidence_level: float = 0.95,
    blocks_per_period: float = 1.0,
    seed: int,
) -> list[ReturnLevelResult]: ...
```

**Algorithm**

1. `rng = np.random.default_rng(seed)`.
2. For each bootstrap replicate `b = 1 … n_samples`:
   - Simulate from fitted model (GEV or GPD) with sample size `n_extremes`.
   - Refit using the same MLE routine.
   - On refit failure: retry up to 10 times; else raise `BootstrapError`.
   - Compute each requested return level.
3. Percentile CI: lower = quantile(α/2), upper = quantile(1 − α/2) with α = 1 − `confidence_level`.

### 4.7 Diagnostics

```python
def diagnostic_qq_pp(
    sample: ExtremeSample,
    fit: GEVFit | GPDFit,
    *,
    n_points: int | None = None,
) -> DiagnosticResult: ...
```

- Plotting position: \(p_i = (i - 0.5) / n\) for sorted order statistics.
- Model quantiles and CDF from engine formulas (consistent with return levels).
- KS distance: \(\max_i |i/n - F(x_{(i)})|\).

### 4.8 Serialization

```python
def to_dict(obj) -> dict: ...
def from_dict(data: dict, type_hint) -> T: ...
```

All dataclass types round-trip through JSON. Arrays encode as lists; timestamps as ISO-8601 strings or int nanoseconds (document chosen format in `serialize.py`).

---

## 5. Numerical strategy

### 5.1 SciPy adapters

| Model | SciPy call | Engine parameters |
|-------|------------|-------------------|
| GEV | `genextreme.fit(x)` → `(c, loc, scale)` | `(xi=-c, mu=loc, sigma=scale)` |
| GPD | `genpareto.fit(excesses, floc=0)` → `(c, loc, scale)` | `(xi=c, sigma=scale)` |

**Only** `fit/scipy_adapter.py` may import SciPy fit routines for parameter estimation.

### 5.2 Convergence

`converged = True` when:

- SciPy `fit` completes without exception,
- `scale > 0`,
- all parameters finite.

### 5.3 Canonical likelihood

All AIC values and validation comparisons use `likelihood.py`:

```python
# GEV: scipy.stats.genextreme.logpdf(x, c=-xi, loc=mu, scale=sigma)
# GPD: scipy.stats.genpareto.logpdf(y, c=xi, loc=0, scale=sigma)
log_likelihood = sum(logpdf)
aic = 2 * k - 2 * log_likelihood
```

### 5.4 Limits

Use explicit ξ → 0 branches in return-level and inverse-CDF code to avoid catastrophic cancellation near Gumbel/exponential limits.

---

## 6. Validation

### 6.1 Strategy

Compare **identical pre-extracted extreme samples** across rare-core, Extremes.jl, and pyextremes. Do not compare full preprocessing pipelines.

### 6.2 Pinned references

| Library | Commit (2026-09-03) |
|---------|---------------------|
| Extremes.jl | `fa757a2acb58669067ceef7bf7a3f1ecc6cfe2dc` |
| pyextremes | `81f943e15f4f06246dc0870a14aa3915398d0e6d` |

### 6.3 Fixture generation

Golden samples are generated once via **inverse-CDF** formulas independent of fitting code. Reference scripts fit committed fixture JSON and write snapshot outputs.

### 6.4 Acceptance tolerances

| Quantity | Tolerance |
|----------|-----------|
| location, scale | `max(1e-3, 0.005 × \|ref\|)` |
| shape ξ | `0.005` absolute |
| return levels (T ∈ {2, 10, 50}) | `0.005 × \|ref\|` relative |
| total log-likelihood | `1e-4` absolute |

If a mismatch exceeds tolerance, document root cause before changing tolerance.

### 6.5 Required fixture coverage

- ξ ∈ {−0.3, −0.01, 0, 0.01, 0.3} (approximate limits and tails)
- Sample sizes n ∈ {15, 30, 100}
- BM and POT metadata variants
- Declustering that materially reduces exceedance count

---

## 7. Error codes

| Code | Exception | When |
|------|-----------|------|
| `NON_FINITE_VALUES` | `ValidationError` | NaN/Inf in values |
| `INVALID_SHAPE` | `ValidationError` | array not 1-D |
| `TIMESTAMP_LENGTH_MISMATCH` | `ValidationError` | timestamps length ≠ values |
| `DUPLICATE_TIMESTAMPS` | `ValidationError` | — |
| `UNORDERED_TIMESTAMPS` | `ValidationError` | not strictly increasing |
| `INSUFFICIENT_BLOCKS` | `ExtractionError` | BM blocks < min_blocks |
| `NO_EXCEEDANCES` | `ExtractionError` | POT found zero exceedances |
| `MISSING_POT_METADATA` | `ValidationError` | return level without λ or u |
| `FIT_FAILED` | `FitError` | SciPy fit failure |
| `INSUFFICIENT_SAMPLE` | `FitError` | n below minimum |
| `BOOTSTRAP_FAILED` | `BootstrapError` | too many refit failures |

---

## 8. JSON schema sketch

```json
{
  "schema_version": "0.1",
  "type": "GEVFit",
  "location": 1.23,
  "scale": 0.45,
  "shape": 0.1,
  "log_likelihood": -12.34,
  "aic": 30.68,
  "n_extremes": 50,
  "converged": true
}
```

Full schema definitions will live in `src/evt/serialize.py` docstrings as implementation lands.

---

## 9. Acceptance checklist (release gate)

- [ ] All §1.1 capabilities implemented
- [ ] All §1.2 items absent
- [ ] Python ≥ 3.14; runtime deps NumPy + SciPy only
- [ ] No silent input mutation
- [ ] Explicit seeds on stochastic paths
- [ ] Golden validation passes §6.4 tolerances
- [ ] Public API documented in README and this spec
- [ ] `AGENTS.md` and `ARCHITECTURE.md` aligned with implementation
