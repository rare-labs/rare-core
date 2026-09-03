"""rare-core — stationary univariate extreme-value analysis engine.

Import this package as ``evt`` (``from evt import ...``). The distribution
name on PyPI / pip is ``rare-core``.
"""

from evt.errors import (
    BootstrapError,
    ErrorCode,
    EVTError,
    ExtractionError,
    FitError,
    SerializationError,
    ValidationError,
)
from evt.extract import extract_block_maxima, extract_peaks_over_threshold
from evt.serialize import from_dict, to_dict
from evt.series import make_extreme_series
from evt.transform import inverse_transform_from_upper_tail, transform_to_upper_tail
from evt.types import (
    DiagnosticResult,
    ExtremeSample,
    ExtremeSeries,
    GEVFit,
    GPDFit,
    ReturnLevelResult,
)
from evt.validate import validate_series

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "BootstrapError",
    "DiagnosticResult",
    "ErrorCode",
    "EVTError",
    "extract_block_maxima",
    "extract_peaks_over_threshold",
    "ExtractionError",
    "ExtremeSample",
    "ExtremeSeries",
    "FitError",
    "from_dict",
    "inverse_transform_from_upper_tail",
    "make_extreme_series",
    "GEVFit",
    "GPDFit",
    "ReturnLevelResult",
    "SerializationError",
    "to_dict",
    "transform_to_upper_tail",
    "validate_series",
    "ValidationError",
]
