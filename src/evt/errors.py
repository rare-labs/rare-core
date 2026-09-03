"""Typed exception hierarchy with stable machine-readable codes (SPEC §7)."""

from enum import StrEnum


class ErrorCode(StrEnum):
    """Stable error codes. Do not rename — tests and agents depend on these strings."""

    NON_FINITE_VALUES = "NON_FINITE_VALUES"
    INVALID_SHAPE = "INVALID_SHAPE"
    TIMESTAMP_LENGTH_MISMATCH = "TIMESTAMP_LENGTH_MISMATCH"
    DUPLICATE_TIMESTAMPS = "DUPLICATE_TIMESTAMPS"
    UNORDERED_TIMESTAMPS = "UNORDERED_TIMESTAMPS"
    INVALID_TAIL = "INVALID_TAIL"
    INVALID_SAMPLE = "INVALID_SAMPLE"
    INSUFFICIENT_BLOCKS = "INSUFFICIENT_BLOCKS"
    NO_EXCEEDANCES = "NO_EXCEEDANCES"
    MISSING_POT_METADATA = "MISSING_POT_METADATA"
    FIT_FAILED = "FIT_FAILED"
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
    BOOTSTRAP_FAILED = "BOOTSTRAP_FAILED"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    UNKNOWN_TYPE = "UNKNOWN_TYPE"
    INVALID_PAYLOAD = "INVALID_PAYLOAD"


class EVTError(Exception):
    """Base error for rare-core. Every subclass carries a stable ``code``."""

    def __init__(self, message: str, *, code: ErrorCode | str) -> None:
        super().__init__(message)
        if isinstance(code, ErrorCode):
            self.code = code
            return
        try:
            self.code = ErrorCode(code)
        except ValueError as exc:
            raise EVTError(
                f"unknown error code {code!r}",
                code=ErrorCode.INVALID_PAYLOAD,
            ) from exc


class ValidationError(EVTError):
    """Invalid user input. Input is never silently dropped or mutated."""


class ExtractionError(EVTError):
    """Extraction produced an unusable sample (too few blocks or exceedances)."""


class FitError(EVTError):
    """Maximum-likelihood fit failed or sample is below the minimum size."""


class BootstrapError(EVTError):
    """Parametric bootstrap exhausted refit retries."""


class SerializationError(EVTError):
    """JSON/dict conversion failed (schema, type, or payload)."""
