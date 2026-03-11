"""
explain.py

Model explainability: feature importance + SHAP values.

İki seviye:
1. Permutation Importance — her zaman çalışır, Pipeline dostu, model-agnostik.
2. SHAP values — isteğe bağlı (shap kuruluysa), lokal açıklanabilirlik.

Neden Permutation Importance?
- Sklearn Pipeline ile doğrudan çalışır (preprocess dahil)
- Feature interaction'ları da yakalar
- Basit, güvenilir, kurumsal raporlamaya uygun

Neden SHAP?
- Her satır için bireysel açıklama (lokal)
- Regülatör uyumluluk / model audit gereksinimleri
- Feature interaction yönünü gösterir (pozitif/negatif etki)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance as sklearn_perm_importance

from .utils import get_logger

logger = get_logger("explain")


def compute_permutation_importance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    n_repeats: int = 10,
    seed: int = 42,
    scoring: str = "roc_auc",
) -> Dict[str, Any]:
    """
    Permutation importance hesaplar (Pipeline dostu).

    Returns:
        Dict with method, scoring, n_features, ranking (sorted desc)
    """
    result = sklearn_perm_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=seed,
        scoring=scoring,
    )

    feature_names = list(X_test.columns)
    importances = result.importances_mean
    stds = result.importances_std
    sorted_idx = np.argsort(importances)[::-1]

    ranking = []
    for i in sorted_idx:
        ranking.append(
            {
                "feature": feature_names[i],
                "importance_mean": float(importances[i]),
                "importance_std": float(stds[i]),
            }
        )

    return {
        "method": "permutation_importance",
        "scoring": scoring,
        "n_repeats": n_repeats,
        "n_features": len(feature_names),
        "ranking": ranking,
    }


def _aggregate_shap_to_original(
    shap_values: np.ndarray,
    transformed_names: List[str],
    original_features: List[str],
) -> Dict[str, np.ndarray]:
    """
    OneHotEncoded SHAP değerlerini orijinal feature'lara aggregate et.

    verbose_feature_names_out=False ile:
    - Numeric: "lead_time" → "lead_time"
    - Categorical: "hotel_Resort Hotel" → "hotel"
    """
    result: Dict[str, np.ndarray] = {
        f: np.zeros(shap_values.shape[0]) for f in original_features
    }
    # En uzun isimler önce → partial prefix collision önlenir
    sorted_originals = sorted(original_features, key=len, reverse=True)

    for i, tname in enumerate(transformed_names):
        matched = False
        normalized_names = [tname]
        # ColumnTransformer default names look like "num__lead_time".
        # Strip transformer prefix so matching works with original columns.
        if "__" in tname:
            normalized_names.insert(0, tname.split("__", 1)[1])

        for orig in sorted_originals:
            if any(
                name == orig or name.startswith(f"{orig}_") for name in normalized_names
            ):
                result[orig] += shap_values[:, i]
                matched = True
                break
        if not matched:
            logger.debug(
                f"SHAP: could not map transformed feature '{tname}' to original"
            )

    return result


def compute_shap_values(
    model: Any,
    X_sample: pd.DataFrame,
    *,
    max_samples: int = 500,
) -> Optional[Dict[str, Any]]:
    """
    SHAP values hesaplar (shap kuruluysa).

    Pipeline destekli:
    - preprocessor + clf ayrıştırılır
    - Tree modeller → TreeExplainer (hızlı)
    - Linear modeller → KernelExplainer (yavaş, sample tabanlı)

    Returns:
        Dict with SHAP ranking, or None if shap unavailable
    """
    try:
        import shap
    except ImportError:
        logger.info("shap not installed — skipping SHAP values.")
        return None

    try:
        preprocessor = model.named_steps.get("preprocess")
        clf = model.named_steps.get("clf")

        if preprocessor is None or clf is None:
            logger.warning(
                "Model is not a standard Pipeline with preprocess+clf. Skipping SHAP."
            )
            return None

        X_sub = X_sample.iloc[:max_samples].copy()
        X_transformed = preprocessor.transform(X_sub)

        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            feature_names = [f"f_{i}" for i in range(X_transformed.shape[1])]

        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        if hasattr(clf, "feature_importances_"):
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_transformed)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            background = shap.sample(X_transformed, min(50, len(X_transformed)))
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_transformed, nsamples=100)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        original_features = list(X_sub.columns)
        agg_shap = _aggregate_shap_to_original(
            shap_values, feature_names, original_features
        )

        ranking = []
        for feat, values in sorted(
            agg_shap.items(), key=lambda x: -np.mean(np.abs(x[1]))
        ):
            ranking.append(
                {
                    "feature": feat,
                    "mean_abs_shap": float(np.mean(np.abs(values))),
                    "mean_shap": float(np.mean(values)),
                }
            )

        return {
            "method": "shap",
            "explainer_type": type(explainer).__name__,
            "n_samples": len(X_sub),
            "n_original_features": len(original_features),
            "ranking": ranking,
        }

    except Exception as exc:
        logger.warning(f"SHAP computation failed: {exc}")
        return None


def save_explainability_report(report: Dict[str, Any], out_path: Path) -> None:
    """Explainability raporunu JSON olarak kaydeder."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved explainability report → {out_path}")
