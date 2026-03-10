from __future__ import annotations

import numpy as np
import pandas as pd
from types import SimpleNamespace

import src.cli.train as cli_train
from src.config import ExperimentConfig, Paths


def _paths(tmp_path) -> Paths:
    p = Paths(project_root=tmp_path)
    p.data_processed.mkdir(parents=True, exist_ok=True)
    p.models.mkdir(parents=True, exist_ok=True)
    p.reports_metrics.mkdir(parents=True, exist_ok=True)
    return p


def _cfg() -> ExperimentConfig:
    return ExperimentConfig()


def _splits():
    df = pd.DataFrame(
        {
            "f1": [1, 2, 3, 4],
            "f2": [10, 20, 30, 40],
            "is_canceled": [0, 1, 0, 1],
        }
    )
    return df.copy(), df.copy(), df.copy()


def _tracker():
    class _Ctx:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    return SimpleNamespace(
        start_run=lambda **kwargs: _Ctx(),
        log_params=lambda *_args, **_kwargs: None,
        log_metrics=lambda *_args, **_kwargs: None,
        log_metric=lambda *_args, **_kwargs: None,
        log_artifact=lambda *_args, **_kwargs: None,
    )


def _result_with_model(model):
    feature_spec = SimpleNamespace(
        numeric=["f1", "f2"],
        categorical=[],
        all_features=["f1", "f2", "f3"],
        to_dict=lambda: {"numeric": ["f1", "f2"], "categorical": []},
    )
    return SimpleNamespace(
        model=model,
        cv_scores=np.array([0.8, 0.9]),
        feature_spec=feature_spec,
        feature_dtypes={"f1": "int64", "f2": "int64"},
    )


def _patch_common(
    monkeypatch, *, result, profile_passed=True, calibrate_func=None, copy_mock=None
):
    monkeypatch.setattr(cli_train, "_load_splits", lambda *_args, **_kwargs: _splits())
    monkeypatch.setattr(
        cli_train, "validate_processed_data", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        cli_train,
        "run_validation_profile",
        lambda *_args, **_kwargs: SimpleNamespace(
            passed=profile_passed,
            hard_failures=([] if profile_passed else ["schema"]),
        ),
    )
    monkeypatch.setattr(
        cli_train, "train_candidate_models", lambda *_args, **_kwargs: {"m1": result}
    )
    monkeypatch.setattr(
        cli_train, "generate_reference_stats", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        cli_train, "generate_reference_categories", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        cli_train, "generate_reference_correlations", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(cli_train, "ExperimentTracker", lambda: _tracker())
    monkeypatch.setattr(cli_train.joblib, "dump", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli_train, "sha256_file", lambda *_args, **_kwargs: "sha")
    monkeypatch.setattr(cli_train, "json_write", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli_train, "mark_latest", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cli_train, "copy_to_latest", copy_mock or (lambda *_args, **_kwargs: None)
    )
    monkeypatch.setattr(
        cli_train,
        "calibrate_frozen_classifier",
        calibrate_func
        or (lambda **_kwargs: SimpleNamespace(calibrated_model=object(), metrics={})),
    )


def test_cmd_train_validation_profile_fail(monkeypatch, tmp_path):
    paths = _paths(tmp_path)
    cfg = _cfg()
    result = _result_with_model(
        SimpleNamespace(named_steps={"clf": object(), "preprocess": None})
    )
    _patch_common(monkeypatch, result=result, profile_passed=False)

    try:
        cli_train.cmd_train(paths, cfg, run_id="r-fail")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Validation profile FAILED [train]" in str(exc)


def test_cmd_train_calibration_metrics_error_schema_contract_and_importance(
    monkeypatch, tmp_path
):
    paths = _paths(tmp_path)
    cfg = _cfg()
    (paths.reports_metrics / "schema_contract.json").write_text("{}", encoding="utf-8")
    (paths.reports_metrics / "feature_importance.json").write_text(
        "{}", encoding="utf-8"
    )

    clf = SimpleNamespace(coef_=np.array([[0.1, -0.2, 0.3]]))
    pre = SimpleNamespace(get_feature_names_out=lambda: np.array(["f1", "f2", "f3"]))
    model = SimpleNamespace(named_steps={"clf": clf, "preprocess": pre})
    result = _result_with_model(model)

    def _calib(**kwargs):
        method = kwargs.get("method")
        if method == "sigmoid":
            return SimpleNamespace(
                calibrated_model=object(), metrics={"ece": 0.1, "note": "ok"}
            )
        raise RuntimeError("calibration failed")

    copy_calls = []

    def copy_mock(src, dst):
        copy_calls.append((str(src), str(dst)))

    _patch_common(
        monkeypatch,
        result=result,
        profile_passed=True,
        calibrate_func=_calib,
        copy_mock=copy_mock,
    )

    out = cli_train.cmd_train(paths, cfg, run_id="r-imp")
    assert out == "r-imp"
    assert any("schema_contract.json" in c[0] for c in copy_calls)


def test_cmd_train_importance_coef_fallback_on_feature_name_error(
    monkeypatch, tmp_path
):
    paths = _paths(tmp_path)
    cfg = _cfg()

    clf = SimpleNamespace(coef_=np.array([[0.4, 0.2, -0.1]]))

    class _Pre:
        def get_feature_names_out(self):
            raise RuntimeError("no names")

    model = SimpleNamespace(named_steps={"clf": clf, "preprocess": _Pre()})
    result = _result_with_model(model)
    _patch_common(monkeypatch, result=result, profile_passed=True)
    out = cli_train.cmd_train(paths, cfg, run_id="r-coef-fallback")
    assert out == "r-coef-fallback"


def test_cmd_train_importance_no_supported_attrs(monkeypatch, tmp_path):
    paths = _paths(tmp_path)
    cfg = _cfg()

    model = SimpleNamespace(named_steps={"clf": SimpleNamespace(), "preprocess": None})
    result = _result_with_model(model)
    _patch_common(monkeypatch, result=result, profile_passed=True)
    out = cli_train.cmd_train(paths, cfg, run_id="r-no-imp")
    assert out == "r-no-imp"


def test_cmd_train_importance_coef_with_no_preprocessor(monkeypatch, tmp_path):
    paths = _paths(tmp_path)
    cfg = _cfg()

    clf = SimpleNamespace(coef_=np.array([[0.5, -0.4, 0.1]]))
    model = SimpleNamespace(named_steps={"clf": clf, "preprocess": None})
    result = _result_with_model(model)
    _patch_common(monkeypatch, result=result, profile_passed=True)
    out = cli_train.cmd_train(paths, cfg, run_id="r-coef-nopre")
    assert out == "r-coef-nopre"


def test_cmd_train_importance_outer_exception_is_swallowed(monkeypatch, tmp_path):
    paths = _paths(tmp_path)
    cfg = _cfg()

    model = SimpleNamespace(named_steps=42)  # no .get -> triggers outer exception path
    result = _result_with_model(model)
    _patch_common(monkeypatch, result=result, profile_passed=True)
    out = cli_train.cmd_train(paths, cfg, run_id="r-imp-exc")
    assert out == "r-imp-exc"


def test_cmd_train_marks_latest_model_registry_as_relative_path(monkeypatch, tmp_path):
    paths = _paths(tmp_path)
    cfg = _cfg()

    model = SimpleNamespace(named_steps={"clf": object(), "preprocess": None})
    result = _result_with_model(model)
    _patch_common(monkeypatch, result=result, profile_passed=True)

    seen: dict[str, object] = {}

    def _capture_mark_latest(base_dir, run_id, extra=None):
        if base_dir == paths.models:
            seen["run_id"] = run_id
            seen["extra"] = extra or {}

    monkeypatch.setattr(cli_train, "mark_latest", _capture_mark_latest)

    out = cli_train.cmd_train(paths, cfg, run_id="r-rel-path")
    assert out == "r-rel-path"
    assert seen["run_id"] == "r-rel-path"
    assert seen["extra"]["model_registry"] == (
        "reports/metrics/r-rel-path/model_registry.json"
    )
