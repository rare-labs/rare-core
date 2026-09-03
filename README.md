# rare-core

Lightweight, Python-native **stationary univariate extreme-value analysis** engine — useful on its own and designed as the deterministic substrate for future MCP, DuckDB, and web-agent layers.

Install **`rare-core`**. Import the public API as **`evt`**.

**Status:** V0.1 specification and scaffold. Implementation in progress.

## Quick links

| Audience | Start here |
|----------|------------|
| Users | [Installation](#installation) → [Quick start](#quick-start) |
| Contributors | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) → [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) |
| AI agents | [AGENTS.md](AGENTS.md) |

## Features (V0.1)

- Block maxima / minima extraction
- Peaks over threshold (POT) with run and time-window declustering
- GEV and GPD maximum-likelihood fitting
- Return levels and return periods
- Parametric-bootstrap return-level uncertainty
- QQ/PP diagnostic coordinates and KS distance
- JSON-serializable results
- Cross-validation against pinned [Extremes.jl](https://github.com/JuliaClimate/Extremes.jl) and [pyextremes](https://github.com/georgebv/pyextremes) reference outputs

## Explicitly out of scope (V0.1)

Automatic BM-vs-POT or threshold selection; nonstationary models; Bayesian/PWM estimation; multivariate/spatial extremes; IDF curves; MCP/REST/DuckDB/WASM/GUI; domain-specific rainfall/finance semantics.

See [docs/SPEC.md](docs/SPEC.md) for the full specification.

## Requirements

- **Python ≥ 3.14**
- **Runtime dependencies:** NumPy and SciPy only (no pandas, xarray, Julia, or pyextremes at runtime)

## Installation

```bash
git clone https://github.com/rare-labs/rare-core.git
cd rare-core
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

## Quick start

> API below reflects the V0.1 target. Implementation lands in Phase 3–5 of the [implementation plan](docs/IMPLEMENTATION_PLAN.md).

```python
import numpy as np
from evt import (
    ExtremeSeries,
    extract_block_maxima,
    fit_gev,
    gev_return_level,
    bootstrap_return_levels,
)

values = np.load("data/observations.npy")  # float64 1-D array
series = ExtremeSeries(values=values, timestamps=None, tail="high")

sample = extract_block_maxima(series, block_size=365, min_blocks=20)
fit = fit_gev(sample)

rl_10 = gev_return_level(fit, return_period=10.0)
rl_10_ci = bootstrap_return_levels(
    sample, fit, return_periods=[10.0], n_samples=500, seed=42
)[0]

print(rl_10.estimate, rl_10_ci.lower, rl_10_ci.upper)
```

## Conventions

| Topic | Convention |
|-------|------------|
| Shape parameter | **ξ (xi)** everywhere in public API and docs |
| SciPy GEV shape | `xi = -c` at the SciPy adapter boundary only |
| Input arrays | NumPy-compatible 1-D `float64`; optional `datetime64[ns]` timestamps |
| Data integrity | Non-finite values, duplicate timestamps, and invalid ordering → explicit errors (no silent drops) |
| Reproducibility | All stochastic procedures accept an explicit `seed` |

## Project layout

```
rare-core/
├── AGENTS.md                 # Instructions for AI coding agents
├── README.md                 # This file
├── docs/
│   ├── SPEC.md               # V0.1 functional & numerical specification
│   ├── ARCHITECTURE.md       # Modules, SOLID/DRY, extension points
│   └── IMPLEMENTATION_PLAN.md
├── src/evt/                  # Import package (`import evt`); distribution is rare-core
├── tests/
└── validation/               # Golden fixtures & reference comparison
```

## Development

```bash
pytest
ruff check src tests
mypy src/evt
```

Reference validation (optional, requires Julia + pyextremes):

```bash
pip install -e ".[reference]"
# See validation/README.md when available
```

## Validation

Golden fixtures are pre-extracted extreme samples. References fit **identical arrays** — not each library's full preprocessing pipeline.

Pinned reference commits (inspected 2026-09-03):

- Extremes.jl: `fa757a2acb58669067ceef7bf7a3f1ecc6cfe2dc`
- pyextremes: `81f943e15f4f06246dc0870a14aa3915398d0e6d`

Acceptance tolerances are defined in [docs/SPEC.md](docs/SPEC.md#validation).

## License

MIT — see [LICENSE](LICENSE).
