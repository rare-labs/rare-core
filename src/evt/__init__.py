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
from evt.serialize import from_dict, to_dict
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
    "ExtractionError",
    "ExtremeSample",
    "ExtremeSeries",
    "FitError",
    "from_dict",
    "GEVFit",
    "GPDFit",
    "ReturnLevelResult",
    "SerializationError",
    "to_dict",
    "validate_series",
    "ValidationError",
]
