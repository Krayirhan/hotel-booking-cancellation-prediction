"""validation/anomaly.py — Row-level anomaly and data quality checks.

Responsibilities
----------------
- ``AnomalyReport``  / ``detect_row_anomalies``    — Domain-aware row anomaly scan
- ``DuplicateReport``/ ``detect_duplicates``        — Exact / partial duplicate rows
- ``assert_no_nans_after_imputation``               — Post-imputation NaN guard
- ``CardinalityReport`` / ``detect_unseen_categories`` — Novel category detection
- ``OutputValidationReport`` / ``validate_model_output`` — [0,1] output range check
- ``VolumeReport``   / ``validate_data_volume``     — Expected row-count range check
- ``StalenessReport``/ ``check_data_staleness``     — File modification-time check
- ``validate_row_counts``                           — Train/calibration/test split check

Hotel-domain anomaly rules
--------------------------
+-------------------+-------------------------------+
| Rule              | Condition                     |
+-------------------+-------------------------------+
| zero_guests       | adults=0 AND children=0 AND   |
|                   | babies=0                      |
| negative_adr      | adr < -10                     |
| extreme_stay      | weekend + week nights > 365   |
| extreme_lead_time | lead_time > 800               |
| extreme_adults    | adults > 50                   |
+-------------------+-------------------------------+

All symbols are re-exported from ``src.data_validation``.
Physical code lives in ``src/data_validation.py``; this module is a thematic
import alias for anomaly/quality-related concerns.
"""

from ..data_validation import (  # noqa: F401
    AnomalyReport,
    CardinalityReport,
    DuplicateReport,
    OutputValidationReport,
    StalenessReport,
    VolumeReport,
    assert_no_nans_after_imputation,
    check_data_staleness,
    detect_duplicates,
    detect_row_anomalies,
    detect_unseen_categories,
    validate_data_volume,
    validate_model_output,
    validate_row_counts,
)

__all__ = [
    "AnomalyReport",
    "detect_row_anomalies",
    "DuplicateReport",
    "detect_duplicates",
    "assert_no_nans_after_imputation",
    "CardinalityReport",
    "detect_unseen_categories",
    "OutputValidationReport",
    "validate_model_output",
    "VolumeReport",
    "validate_data_volume",
    "StalenessReport",
    "check_data_staleness",
    "validate_row_counts",
]
