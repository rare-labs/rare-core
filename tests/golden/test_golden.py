"""Golden fixture comparison: live engine vs committed snapshots."""

from pathlib import Path

from validation.compare import compare_all
from validation.inverse_cdf import gev_quantile, gpd_excess_quantile

from evt.return_levels import gev_quantile as engine_gev_q
from evt.return_levels import gpd_excess_quantile as engine_gpd_q


def test_golden_engine_matches_committed_snapshots() -> None:
    failures = compare_all()
    assert failures == [], "\n".join(failures)


def test_inverse_cdf_matches_engine_quantiles() -> None:
    import numpy as np

    p = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    np.testing.assert_allclose(gev_quantile(p, 10.0, 2.0, 0.2), engine_gev_q(p, 10.0, 2.0, 0.2))
    np.testing.assert_allclose(gev_quantile(p, 0.0, 1.0, 0.0), engine_gev_q(p, 0.0, 1.0, 0.0))
    np.testing.assert_allclose(gpd_excess_quantile(p, 1.5, -0.2), engine_gpd_q(p, 1.5, -0.2))
    np.testing.assert_allclose(gpd_excess_quantile(p, 1.0, 0.0), engine_gpd_q(p, 1.0, 0.0))


def test_declustered_fixture_reduces_count() -> None:
    import json

    from evt.serialize import from_dict
    from evt.types import ExtremeSample

    root = Path(__file__).resolve().parents[2] / "validation" / "fixtures" / "samples"
    sample = from_dict(
        json.loads((root / "pot_declustered_run.json").read_text(encoding="utf-8")),
        ExtremeSample,
    )
    assert sample.n_source > int(sample.raw_values.size)
