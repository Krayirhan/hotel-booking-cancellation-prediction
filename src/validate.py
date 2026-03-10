"""
validate.py — Basic data schema checks (backward-compat re-export façade).

All three functions have been merged into ``src/data_validation.py`` alongside
the Pandera-based schema validators so that all data contract logic lives in
one module.  This file re-exports them to preserve existing imports:

    from src.validate import basic_schema_checks, validate_target_labels, null_ratio_report

For new code, prefer importing directly from data_validation:

    from src.data_validation import basic_schema_checks, validate_target_labels, null_ratio_report
"""

from .data_validation import (  # noqa: F401  (re-export)
    basic_schema_checks,
    validate_target_labels,
    null_ratio_report,
)

__all__ = ["basic_schema_checks", "validate_target_labels", "null_ratio_report"]
