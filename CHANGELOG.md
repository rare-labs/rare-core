# Changelog

## 0.1.0 — 2026-09-03

First release of **rare-core**, a NumPy/SciPy-only stationary univariate
extreme-value engine imported as `evt`.

- Block maxima/minima and peaks-over-threshold extraction, including run and
  time-window declustering.
- GEV and GPD maximum-likelihood fitting with a single canonical likelihood.
- Return levels (explicit ξ → 0 branches), parametric bootstrap percentile CIs,
  and QQ/PP/KS diagnostics.
- JSON round-trip serialization (`schema_version` `"0.1"`).
- Golden inverse-CDF fixtures compared in CI against committed engine snapshots.
  Optional Extremes.jl and pyextremes runners are documented under `validation/`.
