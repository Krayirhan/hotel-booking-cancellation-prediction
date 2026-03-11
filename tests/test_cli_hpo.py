"""Tests for src/cli/hpo.py — cmd_hpo."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.config import ExperimentConfig, Paths


@pytest.fixture()
def cfg() -> ExperimentConfig:
    return ExperimentConfig()


@pytest.fixture()
def paths(tmp_path: Path) -> Paths:
    p = Paths(project_root=tmp_path)
    p.data_processed.mkdir(parents=True, exist_ok=True)
    p.models.mkdir(parents=True, exist_ok=True)
    p.reports_metrics.mkdir(parents=True, exist_ok=True)
    return p


def _make_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 150
    return pd.DataFrame(
        {
            "lead_time": rng.integers(0, 100, n),
            "adr": rng.uniform(50, 300, n),
            "arrival_date_month": rng.choice(["January", "March", "July"], n),
            "is_canceled": rng.integers(0, 2, n),
        }
    )


def _make_hpo_result():
    result = MagicMock()
    result.model_type = "XGBoost"
    result.n_trials = 5
    result.n_completed = 5
    result.best_score = 0.84
    result.best_params = {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 100}
    result.best_model = MagicMock()
    return result


class TestCmdHpo:
    def test_raises_if_dataset_missing(self, paths, cfg):
        from src.cli.hpo import cmd_hpo

        with pytest.raises(FileNotFoundError, match="Processed dataset not found"):
            cmd_hpo(paths, cfg, n_trials=3)

    def test_cmd_hpo_success(self, paths, cfg):
        from src.cli.hpo import cmd_hpo

        df = _make_dataset()
        df.to_parquet(paths.data_processed / "dataset.parquet", index=False)

        mock_result = _make_hpo_result()

        with (
            patch("src.cli.hpo.run_hpo", return_value=mock_result),
            patch("src.cli.hpo.ExperimentTracker") as mock_tracker_cls,
            patch("src.cli.hpo.json_write"),
        ):
            mock_tracker = MagicMock()
            mock_tracker_cls.return_value = mock_tracker
            mock_tracker.start_run.return_value.__enter__ = MagicMock(return_value=None)
            mock_tracker.start_run.return_value.__exit__ = MagicMock(return_value=False)

            run_id = cmd_hpo(paths, cfg, n_trials=3, run_id="hpo-run-001")

        assert run_id == "hpo-run-001"

    def test_cmd_hpo_generates_run_id_if_none(self, paths, cfg):
        from src.cli.hpo import cmd_hpo

        df = _make_dataset()
        df.to_parquet(paths.data_processed / "dataset.parquet", index=False)

        mock_result = _make_hpo_result()

        with (
            patch("src.cli.hpo.run_hpo", return_value=mock_result),
            patch("src.cli.hpo.ExperimentTracker") as mock_tracker_cls,
            patch("src.cli.hpo.json_write"),
        ):
            mock_tracker = MagicMock()
            mock_tracker_cls.return_value = mock_tracker
            mock_tracker.start_run.return_value.__enter__ = MagicMock(return_value=None)
            mock_tracker.start_run.return_value.__exit__ = MagicMock(return_value=False)

            run_id = cmd_hpo(paths, cfg, n_trials=3)

        assert run_id is not None
        assert len(run_id) > 0

    def test_cmd_hpo_writes_report_to_run_dir(self, paths, cfg):
        from src.cli.hpo import cmd_hpo

        df = _make_dataset()
        df.to_parquet(paths.data_processed / "dataset.parquet", index=False)

        mock_result = _make_hpo_result()
        written_payloads = {}

        def capture_json_write(payload, path):
            written_payloads[str(path)] = payload

        with (
            patch("src.cli.hpo.run_hpo", return_value=mock_result),
            patch("src.cli.hpo.ExperimentTracker") as mock_tracker_cls,
            patch("src.cli.hpo.json_write") as mock_write,
        ):
            mock_tracker = MagicMock()
            mock_tracker_cls.return_value = mock_tracker
            mock_tracker.start_run.return_value.__enter__ = MagicMock(return_value=None)
            mock_tracker.start_run.return_value.__exit__ = MagicMock(return_value=False)

            cmd_hpo(paths, cfg, n_trials=3, run_id="hpo-run-002")

        # Verify that json_write was called at least once
        assert mock_write.called, "json_write should have been called"
        # json_write(path, payload) — path is args[0], payload is args[1]
        has_run_id = any(
            len(c.args) >= 2
            and isinstance(c.args[1], dict)
            and c.args[1].get("run_id") == "hpo-run-002"
            for c in mock_write.call_args_list
        )
        assert has_run_id, (
            f"run_id=hpo-run-002 not found as payload in json_write calls: {[(c.args[0], c.args[1] if len(c.args) > 1 else None) for c in mock_write.call_args_list]}"
        )

    def test_cmd_hpo_passes_n_trials_to_run_hpo(self, paths, cfg):
        from src.cli.hpo import cmd_hpo

        df = _make_dataset()
        df.to_parquet(paths.data_processed / "dataset.parquet", index=False)

        mock_result = _make_hpo_result()

        with (
            patch("src.cli.hpo.run_hpo", return_value=mock_result) as mock_run_hpo,
            patch("src.cli.hpo.ExperimentTracker") as mock_tracker_cls,
            patch("src.cli.hpo.json_write"),
        ):
            mock_tracker = MagicMock()
            mock_tracker_cls.return_value = mock_tracker
            mock_tracker.start_run.return_value.__enter__ = MagicMock(return_value=None)
            mock_tracker.start_run.return_value.__exit__ = MagicMock(return_value=False)

            cmd_hpo(paths, cfg, n_trials=7, run_id="hpo-run-003")

        call_kwargs = mock_run_hpo.call_args
        assert call_kwargs.kwargs.get("n_trials") == 7 or (
            call_kwargs.args and 7 in call_kwargs.args
        )
