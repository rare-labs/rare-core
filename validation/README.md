# Validation harness

Cross-library validation for rare-core V0.1.

## Principle

Compare **identical pre-extracted extreme samples**. Each library fits the same
`raw_values` / `excesses` arrays from committed fixture JSON — not independent
preprocessing.

Golden samples are generated with inverse-CDF formulas in
`validation/inverse_cdf.py`. That module must not import `evt.fit`.

## Pinned references

| Library | Commit |
|---------|--------|
| Extremes.jl | `fa757a2acb58669067ceef7bf7a3f1ecc6cfe2dc` |
| pyextremes | `81f943e15f4f06246dc0870a14aa3915398d0e6d` |

## Layout

```
validation/
├── README.md
├── generate_fixtures.py     # inverse-CDF samples + engine snapshots
├── inverse_cdf.py           # GEV/GPD quantiles independent of fitting
├── engine_eval.py           # live engine fit + T ∈ {2, 10, 50}
├── compare.py               # SPEC §6.4 tolerances
├── run_pyextremes.py        # optional pyextremes runner
├── run_extremes_jl.jl       # optional Extremes.jl runner
└── fixtures/
    ├── catalog.json
    ├── samples/             # ExtremeSample JSON (inputs)
    ├── snapshots/           # engine outputs — CI source of truth
    └── references/          # optional pyextremes / extremes_jl JSON
```

`validation/output/` is gitignored scratch space.

## Workflows

### Normal CI

```bash
pytest
python -m validation.compare
```

Compares the live engine to **committed engine snapshots**. Missing
`fixtures/references/{pyextremes,extremes_jl}/` files are skipped. Julia and
pyextremes are not required.

### Regenerate engine snapshots

From the repository root, after an intentional sample or engine change:

```bash
python -m validation.generate_fixtures
python -m validation.compare
```

Review diffs under `validation/fixtures/` and commit them only when the change
is intended.

### Optional cross-library references

```bash
pip install -r validation/requirements-reference.txt
python -m validation.run_pyextremes
```

```bash
julia -e 'using Pkg; Pkg.add("JSON"); Pkg.add(url="https://github.com/jojal5/Extremes.jl", rev="fa757a2acb58669067ceef7bf7a3f1ecc6cfe2dc")'
julia validation/run_extremes_jl.jl
```

Then `python -m validation.compare` also checks those JSON files when present.
The GitHub workflow `.github/workflows/reference.yml` is manual
(`workflow_dispatch`) and is not a required CI check.

## Tolerances

See [docs/SPEC.md](../docs/SPEC.md#64-acceptance-tolerances).

Do not widen tolerances without documented root-cause analysis.
