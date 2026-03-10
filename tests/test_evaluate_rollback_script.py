import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/evaluate_rollback.py")


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SCRIPT), *args]
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        cmd, check=False, text=True, capture_output=True, env=merged_env
    )


def test_no_report_defaults_to_no_rollback(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    p = _run("--report-path", str(missing))
    assert p.returncode == 0
    assert "rollback_required=False" in p.stdout
    assert "rollback_reasons=['none']" in p.stdout


def test_profit_drop_triggers_rollback(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"alerts": {"profit_drop": True}}, ensure_ascii=True),
        encoding="utf-8",
    )
    p = _run("--report-path", str(report))
    assert p.returncode == 0
    assert "rollback_required=True" in p.stdout
    assert "profit_drop" in p.stdout


def test_data_drift_and_action_rate_triggers_rollback(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "alerts": {
                    "data_drift": True,
                    "action_rate_deviation": True,
                }
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    p = _run("--report-path", str(report))
    assert p.returncode == 0
    assert "rollback_required=True" in p.stdout
    assert "data_drift+action_rate_deviation" in p.stdout


def test_data_volume_anomaly_does_not_trigger_rollback(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"alerts": {"data_volume_anomaly": True}}, ensure_ascii=True),
        encoding="utf-8",
    )
    p = _run("--report-path", str(report))
    assert p.returncode == 0
    assert "rollback_required=False" in p.stdout
    assert "rollback_reasons=['none']" in p.stdout
    assert "non_rollback_signals=['data_volume_anomaly']" in p.stdout


def test_fail_on_rollback_returns_42(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"alerts": {"prediction_drift": True}}, ensure_ascii=True),
        encoding="utf-8",
    )
    p = _run("--report-path", str(report), "--fail-on-rollback")
    assert p.returncode == 42


def test_github_output_written(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"alerts": {"prediction_drift": True}}, ensure_ascii=True),
        encoding="utf-8",
    )
    out = tmp_path / "gh_output.txt"
    p = _run("--report-path", str(report), env={"GITHUB_OUTPUT": str(out)})
    assert p.returncode == 0
    text = out.read_text(encoding="utf-8")
    assert "rollback_required=true" in text
    assert "rollback_reasons=prediction_drift" in text
    assert "non_rollback_signals=none" in text
