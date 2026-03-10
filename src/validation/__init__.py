"""validation/__init__.py — Logical split of data_validation.py.

This package provides thematic entry points into the validation framework.
All implementations remain in ``src/data_validation.py``; this package is
the stable public API surface.

Thematic sub-modules
--------------------
- ``src.validation.raw_schema``  — Pandera schema builders + high-level validate_*()
- ``src.validation.drift``       — Distribution, label, correlation, skew, PSI / JS
- ``src.validation.anomaly``     — Row anomalies, duplicates, cardinality, volume, staleness

Full re-export (backward compatibility)
---------------------------------------
Everything exported here is identical to importing directly from
``src.data_validation``.  Old call sites continue to work unchanged.

Example
-------
    # New, preferred import style
    from src.validation.raw_schema import validate_raw_data
    from src.validation.drift import compute_psi

    # Old style — still works
    from src.data_validation import validate_raw_data, compute_psi
"""

from ..data_validation import (  # noqa: F401
    # ── Schema builders ──
    build_raw_schema,
    build_processed_schema,
    build_inference_schema,
    # ── High-level validators ──
    validate_raw_data,
    validate_processed_data,
    validate_inference_payload,
    # ── Distribution ──
    DistributionReport,
    validate_distributions,
    generate_reference_stats,
    # ── Row-level anomaly ──
    AnomalyReport,
    detect_row_anomalies,
    # ── Duplicates ──
    DuplicateReport,
    detect_duplicates,
    # ── Post-imputation ──
    assert_no_nans_after_imputation,
    # ── Label drift ──
    LabelDriftReport,
    detect_label_drift,
    # ── Cardinality ──
    CardinalityReport,
    detect_unseen_categories,
    # ── Model output ──
    OutputValidationReport,
    validate_model_output,
    # ── Volume + Staleness ──
    VolumeReport,
    validate_data_volume,
    StalenessReport,
    check_data_staleness,
    # ── Schema fingerprint ──
    get_schema_fingerprint,
    # ── Correlation drift ──
    CorrelationDriftReport,
    detect_correlation_drift,
    generate_reference_correlations,
    # ── Reference categories ──
    generate_reference_categories,
    # ── Training-serving skew ──
    SkewReport,
    detect_training_serving_skew,
    # ── Row count ──
    validate_row_counts,
    # ── Feature importance drift ──
    ImportanceDriftReport,
    detect_feature_importance_drift,
    # ── PSI / JS ──
    PSIReport,
    compute_psi,
    # ── Validation profile ──
    ValidationProfileReport,
    run_validation_profile,
    # ── Basic schema checks (merged from validate.py) ──
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
    "DistributionReport",
    "validate_distributions",
    "generate_reference_stats",
    "AnomalyReport",
    "detect_row_anomalies",
    "DuplicateReport",
    "detect_duplicates",
    "assert_no_nans_after_imputation",
    "LabelDriftReport",
    "detect_label_drift",
    "CardinalityReport",
    "detect_unseen_categories",
    "OutputValidationReport",
    "validate_model_output",
    "VolumeReport",
    "validate_data_volume",
    "StalenessReport",
    "check_data_staleness",
    "get_schema_fingerprint",
    "CorrelationDriftReport",
    "detect_correlation_drift",
    "generate_reference_correlations",
    "generate_reference_categories",
    "SkewReport",
    "detect_training_serving_skew",
    "validate_row_counts",
    "ImportanceDriftReport",
    "detect_feature_importance_drift",
    "PSIReport",
    "compute_psi",
    "ValidationProfileReport",
    "run_validation_profile",
    "basic_schema_checks",
    "validate_target_labels",
    "null_ratio_report",
]
