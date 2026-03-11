"""
experiment_tracking.py

Optional MLflow experiment tracking integration.

Design:
- ExperimentTracker provides a clean interface for logging
- When MLflow is not installed or MLFLOW_TRACKING_URI is not set, all operations are no-ops
- This ensures the pipeline works identically with or without MLflow

Usage:
    tracker = ExperimentTracker()
    with tracker.start_run(run_name="train_20260217"):
        tracker.log_params({"seed": 42, "cv_folds": 5})
        tracker.log_metric("roc_auc_mean", 0.85)
        tracker.log_artifact("reports/metrics/cv_summary.json")
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import get_logger

logger = get_logger("experiment_tracking")


class ExperimentTracker:
    """
    Thin wrapper around MLflow with graceful fallback.

    Activation conditions (both required):
    1. mlflow package is installed
    2. MLFLOW_TRACKING_URI env var is set OR explicit tracking_uri passed

    When inactive, all methods are silent no-ops.
    """

    def __init__(
        self,
        experiment_name: str = "hotel_booking_cancellation_prediction",
        tracking_uri: Optional[str] = None,
    ):
        self._active = False
        self._mlflow: Any = None
        self._sklearn_mod: Any = None

        try:
            import mlflow
            import mlflow.sklearn

            self._mlflow = mlflow
            self._sklearn_mod = mlflow.sklearn

            uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
            if uri:
                mlflow.set_tracking_uri(uri)
                mlflow.set_experiment(experiment_name)
                self._active = True
                logger.info(
                    f"MLflow tracking active | uri={uri} experiment={experiment_name}"
                )
            else:
                logger.info(
                    "MLflow installed but MLFLOW_TRACKING_URI not set — tracking disabled."
                )
        except ImportError:
            logger.info("MLflow not installed — experiment tracking disabled.")

    @property
    def active(self) -> bool:
        return self._active

    @contextmanager
    def start_run(self, run_name: Optional[str] = None, nested: bool = False):
        """Context manager for an MLflow run (no-op when inactive)."""
        if self._active:
            with self._mlflow.start_run(run_name=run_name, nested=nested):
                yield
        else:
            yield

    def log_param(self, key: str, value: Any) -> None:
        if self._active:
            try:
                self._mlflow.log_param(key, value)
            except Exception as exc:
                logger.warning(f"MLflow log_param failed: {exc}")

    def log_params(self, params: Dict[str, Any]) -> None:
        if self._active:
            try:
                safe = {k: str(v)[:250] for k, v in params.items()}
                self._mlflow.log_params(safe)
            except Exception as exc:
                logger.warning(f"MLflow log_params failed: {exc}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        if self._active:
            try:
                self._mlflow.log_metric(key, value, step=step)
            except Exception as exc:
                logger.warning(f"MLflow log_metric failed: {exc}")

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        if self._active:
            try:
                self._mlflow.log_metrics(metrics, step=step)
            except Exception as exc:
                logger.warning(f"MLflow log_metrics failed: {exc}")

    def log_artifact(self, path: str | Path) -> None:
        if self._active:
            try:
                self._mlflow.log_artifact(str(path))
            except Exception as exc:
                logger.warning(f"MLflow log_artifact failed: {exc}")

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        if self._active:
            try:
                self._sklearn_mod.log_model(model, artifact_path)
            except Exception as exc:
                logger.warning(f"MLflow log_model failed: {exc}")

    def set_tag(self, key: str, value: str) -> None:
        if self._active:
            try:
                self._mlflow.set_tag(key, value)
            except Exception as exc:
                logger.warning(f"MLflow set_tag failed: {exc}")
