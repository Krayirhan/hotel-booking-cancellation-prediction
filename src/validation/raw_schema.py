"""validation/raw_schema.py — Pandera schema builders and high-level validators.

Responsibilities
----------------
- ``build_raw_schema``       — Pandera schema for raw hotel bookings CSV
- ``build_processed_schema`` — Pandera schema for post-preprocess features
- ``build_inference_schema`` — Pandera schema for API inference payload
- ``validate_raw_data``      — Validate raw DataFrame (raises/returns SchemaErrors)
- ``validate_processed_data``— Validate processed DataFrame
- ``validate_inference_payload`` — Validate inference payload
- ``generate_reference_stats`` — Compute per-column reference statistics
- ``get_schema_fingerprint`` — SHA-256 fingerprint of column names + dtypes
- ``basic_schema_checks``    — Lightweight Pandas-only schema checks (no Pandera)
- ``validate_target_labels`` — Assert target column contains only allowed values
- ``null_ratio_report``      — Per-column null ratio summary

All symbols are re-exported from ``src.data_validation`` for backwards compat.
Physical code lives in ``src/data_validation.py``; this module is a focal-point
import alias for schema-related concerns.
"""

from ..data_validation import (  # noqa: F401
    build_inference_schema,
    build_processed_schema,
    build_raw_schema,
    generate_reference_stats,
    get_schema_fingerprint,
    validate_inference_payload,
    validate_processed_data,
    validate_raw_data,
    basic_schema_checks,
    validate_target_labels,
    null_ratio_report,
)

__all__ = [
    "build_raw_schema",
    "build_processed_schema",
    "build_inference_schema",
    "validate_raw_data",
    "validate_processed_data",
    "validate_inference_payload",
    "generate_reference_stats",
    "get_schema_fingerprint",
    "basic_schema_checks",
    "validate_target_labels",
    "null_ratio_report",
]
