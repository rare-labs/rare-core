"""Public surface matches SPEC §4."""

import evt

EXPECTED = {
    "BootstrapError",
    "DiagnosticResult",
    "ErrorCode",
    "EVTError",
    "ExtractionError",
    "ExtremeSample",
    "ExtremeSeries",
    "FitError",
    "GEVFit",
    "GPDFit",
    "ReturnLevelResult",
    "SerializationError",
    "ValidationError",
    "__version__",
    "bootstrap_return_levels",
    "diagnostic_qq_pp",
    "extract_block_maxima",
    "extract_peaks_over_threshold",
    "fit_gev",
    "fit_gpd",
    "from_dict",
    "gev_return_level",
    "gpd_return_level",
    "inverse_transform_from_upper_tail",
    "make_extreme_series",
    "to_dict",
    "transform_to_upper_tail",
    "validate_series",
}


def test_public_exports_match_spec() -> None:
    assert set(evt.__all__) == EXPECTED
    for name in EXPECTED:
        assert hasattr(evt, name)
