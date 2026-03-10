"""validation/drift.py — Distribution drift detection utilities.

Responsibilities
----------------
- ``DistributionReport`` / ``validate_distributions``   — σ-based mean drift
- ``LabelDriftReport``   / ``detect_label_drift``        — Target rate shift
- ``CorrelationDriftReport`` / ``detect_correlation_drift`` — Feature-pair Pearson drift
- ``generate_reference_correlations``                    — Build reference correlation dict
- ``SkewReport``         / ``detect_training_serving_skew`` — Per-request serving skew
- ``ImportanceDriftReport`` / ``detect_feature_importance_drift`` — SHAP rank drift
- ``PSIReport``          / ``compute_psi``               — PSI and Jensen-Shannon divergence
- ``ValidationProfileReport`` / ``run_validation_profile`` — All-in-one severity-aware check

Drift severity thresholds (defaults)
-------------------------------------
+-------------------+----------+----------+
| Metric            | Warn     | Block    |
+-------------------+----------+----------+
| Mean drift (σ)    |  3 σ     |  3 σ     |
| Label drift (Δ)   | 0.10     | 0.10     |
| Correlation (Δ)   | 0.20     | 0.20     |
| PSI               | 0.10     | 0.25     |
| JS divergence     | 0.05     | 0.15     |
+-------------------+----------+----------+

All symbols are re-exported from ``src.data_validation``.
Physical code lives in ``src/data_validation.py``; this module is a thematic
import alias for drift-related concerns.
"""

from ..data_validation import (  # noqa: F401
    CorrelationDriftReport,
    DistributionReport,
    ImportanceDriftReport,
    LabelDriftReport,
    PSIReport,
    SkewReport,
    ValidationProfileReport,
    compute_psi,
    detect_correlation_drift,
    detect_feature_importance_drift,
    detect_label_drift,
    detect_training_serving_skew,
    generate_reference_correlations,
    generate_reference_stats,
    run_validation_profile,
    validate_distributions,
)

__all__ = [
    "DistributionReport",
    "validate_distributions",
    "generate_reference_stats",
    "LabelDriftReport",
    "detect_label_drift",
    "CorrelationDriftReport",
    "detect_correlation_drift",
    "generate_reference_correlations",
    "SkewReport",
    "detect_training_serving_skew",
    "ImportanceDriftReport",
    "detect_feature_importance_drift",
    "PSIReport",
    "compute_psi",
    "ValidationProfileReport",
    "run_validation_profile",
]
