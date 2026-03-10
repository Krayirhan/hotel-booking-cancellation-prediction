"""CLI: train command — with optional MLflow experiment tracking."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from ..calibration import calibrate_frozen_classifier
from ..config import ExperimentConfig, Paths
from ..data_validation import (
    generate_reference_stats,
    validate_processed_data,
    validate_row_counts,
    generate_reference_categories,
    generate_reference_correlations,
    run_validation_profile,
)
from ..experiment_tracking import ExperimentTracker
from ..io import read_parquet, write_parquet
from ..split import stratified_split
from ..train import train_candidate_models
from ..utils import get_logger, sha256_file
from ._helpers import copy_to_latest, json_read, json_write, mark_latest, new_run_id

logger = get_logger("cli.train")


def _load_splits(paths: Paths, cfg: ExperimentConfig):
    """
    Load persisted splits from disk (written by `split` command).
    Falls back to in-memory split if files don't exist yet (backward compat).
    """
    train_path = paths.data_processed / "train.parquet"
    cal_path = paths.data_processed / "cal.parquet"
    test_path = paths.data_processed / "test.parquet"

    if train_path.exists() and cal_path.exists() and test_path.exists():
        train_fit_df = read_parquet(train_path)
        cal_df = read_parquet(cal_path)
        test_df = read_parquet(test_path)
        logger.info(
            f"Loaded persisted splits | train={len(train_fit_df)} "
            f"cal={len(cal_df)} test={len(test_df)}"
        )
        return train_fit_df, cal_df, test_df

    # Fallback: in-memory split (backward compat for existing workflows)
    logger.warning(
        "Persisted splits not found. Falling back to in-memory split. "
        "Run 'python main.py split' first for full reproducibility."
    )
    dataset_path = paths.data_processed / "dataset.parquet"
    df = read_parquet(dataset_path)
    train_df, test_df = stratified_split(df, cfg.target_col, cfg.test_size, cfg.seed)
    train_fit_df, cal_df = stratified_split(
        train_df, cfg.target_col, test_size=0.20, seed=cfg.seed
    )

    # Persist fallback splits so downstream commands (evaluate/predict) can rely
    # on canonical files in fresh CI workspaces.
    write_parquet(train_fit_df, train_path)
    write_parquet(cal_df, cal_path)
    write_parquet(test_df, test_path)
    logger.info(
        f"Persisted fallback splits | train={len(train_fit_df)} cal={len(cal_df)} test={len(test_df)}"
    )

    return train_fit_df, cal_df, test_df


def cmd_train(paths: Paths, cfg: ExperimentConfig, run_id: Optional[str] = None) -> str:
    run_id = run_id or new_run_id()

    train_fit_df, cal_df, test_df = _load_splits(paths, cfg)
    logger.info(
        f"Split sizes | train_fit={len(train_fit_df)} cal={len(cal_df)} test={len(test_df)}"
    )

    # ── Satır sayısı tutarlılığı ──
    row_check = None
    dataset_path = paths.data_processed / "dataset.parquet"
    if dataset_path.exists():
        dataset_df = read_parquet(dataset_path)
        row_check = validate_row_counts(
            dataset_rows=len(dataset_df),
            train_rows=len(train_fit_df),
            cal_rows=len(cal_df),
            test_rows=len(test_df),
        )
        del dataset_df  # free memory

    # ── Eğitim verisi şema doğrulaması ──
    logger.info("Validating training data schema before model fitting...")
    validate_processed_data(
        train_fit_df,
        target_col=cfg.target_col,
        raise_on_error=(cfg.validation.processed_schema.severity == "hard_fail"),
    )
    # ── ValidationProfile (policy-aware) ──
    profile = run_validation_profile(
        train_fit_df,
        target_col=cfg.target_col,
        policy=cfg.validation,
        phase="train",
    )
    if not profile.passed:
        raise ValueError(
            f"Validation profile FAILED [train]: hard_failures={profile.hard_failures}"
        )

    # ── HPO entegrasyonu: önceki HPO sonuçlarını kullan ──
    hpo_result_path = paths.reports_metrics / "latest_hpo_result.json"
    hpo_params = None
    if hpo_result_path.exists():
        hpo_params = json_read(hpo_result_path)
        logger.info(
            f"HPO results found | model_type={hpo_params.get('model_type')} "
            f"best_score={hpo_params.get('best_score', 'N/A')}"
        )
    else:
        logger.info("No HPO results found. Using default hyperparameters.")

    candidates = train_candidate_models(
        train_fit_df,
        cfg.target_col,
        cfg.seed,
        cfg.cv_folds,
        include_challenger=cfg.model.include_challenger,
        hpo_params=hpo_params,
    )

    run_model_dir = paths.models / run_id
    run_metrics_dir = paths.reports_metrics / run_id
    run_model_dir.mkdir(parents=True, exist_ok=True)
    run_metrics_dir.mkdir(parents=True, exist_ok=True)

    # Deferred row count write (computed before training)
    if row_check is not None:
        json_write(run_metrics_dir / "row_count_check.json", row_check)

    X_cal = cal_df.drop(columns=[cfg.target_col])
    y_cal = cal_df[cfg.target_col].astype(int).values

    model_registry: Dict[str, str] = {}
    model_checksums: Dict[str, str] = {}
    cv_summary: Dict[str, Any] = {}
    calibration_report: Dict[str, Any] = {}

    # ── Optional MLflow tracking ───────────────────────────────────
    tracker = ExperimentTracker()
    with tracker.start_run(run_name=f"train_{run_id}"):
        tracker.log_params(
            {
                "run_id": run_id,
                "seed": cfg.seed,
                "cv_folds": cfg.cv_folds,
                "test_size": cfg.test_size,
                "target_col": cfg.target_col,
                "include_challenger": cfg.model.include_challenger,
                "git_sha": os.getenv("GITHUB_SHA")
                or os.getenv("CI_COMMIT_SHA")
                or "local",
            }
        )

        for model_name, result in candidates.items():
            model_path = run_model_dir / f"{model_name}.joblib"
            joblib.dump(result.model, model_path)
            model_registry[model_name] = f"models/{run_id}/{model_name}.joblib"
            model_checksums[model_name] = sha256_file(str(model_path))
            cv_summary[model_name] = {
                "roc_auc_mean": float(result.cv_scores.mean()),
                "roc_auc_std": float(result.cv_scores.std()),
                "cv_folds": int(cfg.cv_folds),
            }
            logger.info(f"Saved model -> {model_path}")

            # MLflow: log CV metrics per model
            tracker.log_metrics(
                {
                    f"{model_name}_roc_auc_mean": float(result.cv_scores.mean()),
                    f"{model_name}_roc_auc_std": float(result.cv_scores.std()),
                }
            )

            calibration_report[model_name] = {}
            for method in ("sigmoid", "isotonic"):
                try:
                    cal_res = calibrate_frozen_classifier(
                        fitted_model=result.model,
                        X_cal=X_cal,
                        y_cal=y_cal,
                        method=method,
                    )
                    cal_name = f"{model_name}_calibrated_{method}"
                    cal_path = run_model_dir / f"{cal_name}.joblib"
                    joblib.dump(cal_res.calibrated_model, cal_path)
                    model_registry[cal_name] = f"models/{run_id}/{cal_name}.joblib"
                    model_checksums[cal_name] = sha256_file(str(cal_path))
                    calibration_report[model_name][method] = cal_res.metrics
                    logger.info(f"Saved calibrated model -> {cal_path}")

                    # MLflow: log calibration metrics
                    for mk, mv in cal_res.metrics.items():
                        if isinstance(mv, (int, float)):
                            tracker.log_metric(f"{cal_name}_{mk}", float(mv))
                except Exception as e:
                    calibration_report[model_name][method] = {"error": str(e)}
                    logger.exception(
                        f"Calibration failed for {model_name}/{method}: {e}"
                    )

        json_write(run_metrics_dir / "model_registry.json", model_registry)
        json_write(run_metrics_dir / "model_checksums.json", model_checksums)
        json_write(run_metrics_dir / "cv_summary.json", cv_summary)
        json_write(run_metrics_dir / "calibration_metrics.json", calibration_report)
        # Data lineage — reference persisted split files
        train_path = paths.data_processed / "train.parquet"
        cal_path_ref = paths.data_processed / "cal.parquet"
        test_path_ref = paths.data_processed / "test.parquet"
        json_write(
            run_metrics_dir / "data_lineage.json",
            {
                "run_id": run_id,
                "train_path": str(train_path),
                "train_sha256": (
                    sha256_file(str(train_path)) if train_path.exists() else None
                ),
                "cal_path": str(cal_path_ref),
                "cal_sha256": (
                    sha256_file(str(cal_path_ref)) if cal_path_ref.exists() else None
                ),
                "test_path": str(test_path_ref),
                "test_sha256": (
                    sha256_file(str(test_path_ref)) if test_path_ref.exists() else None
                ),
                "train_rows": len(train_fit_df),
                "cal_rows": len(cal_df),
                "test_rows": len(test_df),
                "git_sha": os.getenv("GITHUB_SHA") or os.getenv("CI_COMMIT_SHA"),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
        )

        # MLflow: log metric artifacts
        for artifact_name in (
            "cv_summary.json",
            "calibration_metrics.json",
            "model_registry.json",
        ):
            tracker.log_artifact(run_metrics_dir / artifact_name)

    first_result = next(iter(candidates.values()))
    feature_spec_payload = {
        "run_id": run_id,
        "schema_version": cfg.contract.feature_schema_version,
        **first_result.feature_spec.to_dict(),
        "dtypes": first_result.feature_dtypes,
    }
    run_feature_spec = run_metrics_dir / "feature_spec.json"
    json_write(run_feature_spec, feature_spec_payload)
    copy_to_latest(run_feature_spec, paths.reports / "feature_spec.json")

    # ── Schema contract artifact kopyala (versioned schema metadata) ──
    schema_contract_src = paths.reports_metrics / "schema_contract.json"
    if schema_contract_src.exists():
        run_schema_contract = run_metrics_dir / "schema_contract.json"
        copy_to_latest(schema_contract_src, run_schema_contract)

    # ── Referans istatistikleri kaydet (dağılım drift kontrolü için) ──
    ref_stats = generate_reference_stats(
        train_fit_df,
        numeric_cols=first_result.feature_spec.numeric,
    )
    run_ref_stats = run_metrics_dir / "reference_stats.json"
    json_write(run_ref_stats, ref_stats)
    copy_to_latest(run_ref_stats, paths.reports_metrics / "reference_stats.json")
    logger.info(
        f"Reference stats saved for {len(ref_stats)} numeric features → {run_ref_stats}"
    )

    # ── Referans kategoriler kaydet (unseen category kontrolü için) ──
    ref_cats = generate_reference_categories(
        train_fit_df,
        categorical_cols=first_result.feature_spec.categorical,
    )
    run_ref_cats = run_metrics_dir / "reference_categories.json"
    json_write(run_ref_cats, ref_cats)
    copy_to_latest(run_ref_cats, paths.reports_metrics / "reference_categories.json")

    # ── Referans korelasyonlar kaydet (correlation drift için) ──
    ref_corr = generate_reference_correlations(
        train_fit_df,
        numeric_cols=first_result.feature_spec.numeric,
        target_col=cfg.target_col,
        top_k=15,
    )
    run_ref_corr = run_metrics_dir / "reference_correlations.json"
    json_write(run_ref_corr, ref_corr)
    copy_to_latest(run_ref_corr, paths.reports_metrics / "reference_correlations.json")

    # ── Feature importance kaydet (importance drift için) ──
    try:
        first_model = first_result.model
        clf = first_model.named_steps.get("clf")
        if hasattr(clf, "feature_importances_"):
            feat_names = first_result.feature_spec.all_features
            importance = dict(
                zip(feat_names, [float(v) for v in clf.feature_importances_])
            )
        elif hasattr(clf, "coef_"):
            feat_names = first_result.feature_spec.all_features
            # For linear models use absolute coefficient values
            preprocessor = first_model.named_steps.get("preprocess")
            if preprocessor is not None:
                try:
                    out_names = preprocessor.get_feature_names_out()
                    importance = dict(
                        zip(
                            [str(n) for n in out_names],
                            [float(abs(v)) for v in clf.coef_[0]],
                        )
                    )
                except Exception:
                    importance = dict(
                        zip(
                            feat_names,
                            [float(abs(v)) for v in clf.coef_[0][: len(feat_names)]],
                        )
                    )
            else:
                importance = dict(
                    zip(
                        feat_names,
                        [float(abs(v)) for v in clf.coef_[0][: len(feat_names)]],
                    )
                )
        else:
            importance = {}
        if importance:
            run_importance = run_metrics_dir / "feature_importance.json"
            json_write(run_importance, importance)
            global_importance = paths.reports_metrics / "feature_importance.json"
            prev_importance = paths.reports_metrics / "feature_importance.prev.json"
            if global_importance.exists():
                copy_to_latest(global_importance, prev_importance)
            copy_to_latest(
                run_importance, paths.reports_metrics / "feature_importance.json"
            )
            logger.info(f"Feature importance saved: {len(importance)} features")
    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")

    model_registry_pointer = (
        Path("reports") / "metrics" / run_id / "model_registry.json"
    ).as_posix()
    mark_latest(
        paths.models,
        run_id,
        extra={"model_registry": model_registry_pointer},
    )
    mark_latest(paths.reports_metrics, run_id)

    logger.info(f"Training completed. run_id={run_id}")
    return run_id
