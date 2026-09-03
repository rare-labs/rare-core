"""JSON-compatible dict encoding for frozen model types.

Arrays become lists of Python floats. ``datetime64[ns]`` timestamps become
lists of integer nanoseconds since the Unix epoch (UTC). That encoding is
lossless for nanosecond resolution and survives ``json.dumps`` / ``json.loads``.

Every payload includes ``schema_version`` (from ``constants.SCHEMA_VERSION``)
and ``type`` (the dataclass name).
"""

from dataclasses import fields, is_dataclass
from types import UnionType
from typing import Any, TypeAliasType, Union, cast, get_args, get_origin, get_type_hints

import numpy as np

from evt.constants import SCHEMA_VERSION
from evt.errors import ErrorCode, SerializationError
from evt.types import MODEL_TYPES

_TYPE_BY_NAME: dict[str, type] = {cls.__name__: cls for cls in MODEL_TYPES}
_TIMESTAMP_FIELDS = frozenset({"timestamps"})


def _resolve_hint(hint: object) -> object:
    if isinstance(hint, TypeAliasType):
        return _resolve_hint(hint.__value__)
    origin = get_origin(hint)
    if origin is Union or origin is UnionType:
        non_none = [arg for arg in get_args(hint) if arg is not type(None)]
        if len(non_none) == 1:
            return _resolve_hint(non_none[0])
    return hint


def _encode(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "M":
            as_ns = value.astype("datetime64[ns]", copy=False)
            return [int(np.datetime64(item, "ns").view(np.int64)) for item in as_ns]
        return [float(item) for item in np.asarray(value, dtype=np.float64)]
    if isinstance(value, np.bool_ | bool):
        return bool(value)
    if isinstance(value, np.floating | float):
        return float(value)
    if isinstance(value, np.integer | int):
        return int(value)
    if isinstance(value, str):
        return value
    raise SerializationError(
        f"cannot encode value of type {type(value).__name__}",
        code=ErrorCode.INVALID_PAYLOAD,
    )


def _decode_float_array(raw: object, *, name: str) -> np.ndarray:
    if not isinstance(raw, list):
        raise SerializationError(
            f"{name} must be a list of numbers",
            code=ErrorCode.INVALID_PAYLOAD,
        )
    try:
        return np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SerializationError(
            f"{name} must be a list of numbers",
            code=ErrorCode.INVALID_PAYLOAD,
        ) from exc


def _decode_timestamps(raw: object, *, name: str) -> np.ndarray:
    if not isinstance(raw, list):
        raise SerializationError(
            f"{name} must be a list of integer nanoseconds",
            code=ErrorCode.INVALID_PAYLOAD,
        )
    try:
        ints = np.asarray(raw, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise SerializationError(
            f"{name} must be a list of integer nanoseconds",
            code=ErrorCode.INVALID_PAYLOAD,
        ) from exc
    return ints.astype("datetime64[ns]", copy=False)


def _decode_field(name: str, raw: object, hint: object) -> object:
    if raw is None:
        return None
    core = _resolve_hint(hint)
    origin = get_origin(core)
    if core is np.ndarray or name in _TIMESTAMP_FIELDS:
        if name in _TIMESTAMP_FIELDS:
            return _decode_timestamps(raw, name=name)
        return _decode_float_array(raw, name=name)
    if core is float:
        if not isinstance(raw, int | float) or isinstance(raw, bool):
            raise SerializationError(
                f"{name} must be a number",
                code=ErrorCode.INVALID_PAYLOAD,
            )
        return float(raw)
    if core is int:
        if not isinstance(raw, int) or isinstance(raw, bool):
            raise SerializationError(
                f"{name} must be an integer",
                code=ErrorCode.INVALID_PAYLOAD,
            )
        return int(raw)
    if core is bool:
        if not isinstance(raw, bool):
            raise SerializationError(
                f"{name} must be a boolean",
                code=ErrorCode.INVALID_PAYLOAD,
            )
        return raw
    if core is str or origin is not None:
        if not isinstance(raw, str):
            raise SerializationError(
                f"{name} must be a string",
                code=ErrorCode.INVALID_PAYLOAD,
            )
        return raw
    raise SerializationError(
        f"unsupported field type for {name}",
        code=ErrorCode.INVALID_PAYLOAD,
    )


def to_dict(obj: object) -> dict[str, Any]:
    """Encode a model dataclass to a JSON-ready dict."""
    if not is_dataclass(obj) or type(obj) not in _TYPE_BY_NAME.values():
        raise SerializationError(
            f"unsupported type {type(obj).__name__}",
            code=ErrorCode.UNKNOWN_TYPE,
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "type": type(obj).__name__,
    }
    for field in fields(obj):
        payload[field.name] = _encode(getattr(obj, field.name))
    return payload


def from_dict[T](data: dict[str, Any], type_hint: type[T]) -> T:
    """Decode a dict produced by ``to_dict`` (or equivalent JSON) into ``type_hint``."""
    if type_hint not in _TYPE_BY_NAME.values():
        raise SerializationError(
            f"unsupported type {type_hint.__name__}",
            code=ErrorCode.UNKNOWN_TYPE,
        )
    if not isinstance(data, dict):
        raise SerializationError("payload must be an object", code=ErrorCode.INVALID_PAYLOAD)
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SerializationError(
            f"unsupported schema_version {version!r}; expected {SCHEMA_VERSION!r}",
            code=ErrorCode.SCHEMA_MISMATCH,
        )
    declared = data.get("type")
    if declared != type_hint.__name__:
        raise SerializationError(
            f"payload type {declared!r} does not match {type_hint.__name__}",
            code=ErrorCode.UNKNOWN_TYPE,
        )
    hints = get_type_hints(type_hint)
    kwargs: dict[str, Any] = {}
    cls: Any = type_hint
    for field in fields(cls):
        if field.name not in data:
            raise SerializationError(
                f"missing field {field.name!r}",
                code=ErrorCode.INVALID_PAYLOAD,
            )
        kwargs[field.name] = _decode_field(field.name, data[field.name], hints[field.name])
    return cast(T, cls(**kwargs))
