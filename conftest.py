"""Root conftest.py with import, isolation, and immutable-artifact guards."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

# Guarantee project root is on sys.path.
_ROOT_PATH = Path(__file__).resolve().parent
_ROOT_STR = str(_ROOT_PATH)
if _ROOT_STR not in sys.path:
    sys.path.insert(0, _ROOT_STR)

IMMUTABLE_REPO_ARTIFACTS = (
    _ROOT_PATH / "models" / "latest.json",
    _ROOT_PATH / "reports" / "feature_spec.json",
)

_IMMUTABLE_BASELINE_HASH: dict[Path, str] = {}
_IMMUTABLE_BASELINE_BYTES: dict[Path, bytes] = {}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pytest_sessionstart(session: pytest.Session) -> None:
    """Capture pre-test snapshots for tracked immutable artifacts."""
    missing = [path for path in IMMUTABLE_REPO_ARTIFACTS if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        # Warn instead of hard-fail so fresh clones / CI without model artifacts
        # can still run the unit test suite.
        import warnings

        warnings.warn(
            f"Immutable artifact(s) not found — drift guard disabled: {joined}",
            stacklevel=1,
        )
        # Only guard artifacts that actually exist.
        effective = [p for p in IMMUTABLE_REPO_ARTIFACTS if p.exists()]
    else:
        effective = list(IMMUTABLE_REPO_ARTIFACTS)

    for path in effective:
        payload = path.read_bytes()
        _IMMUTABLE_BASELINE_BYTES[path] = payload
        _IMMUTABLE_BASELINE_HASH[path] = _sha256_bytes(payload)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fail run if tracked immutable artifacts drifted during tests."""
    changed: list[Path] = []
    for path in IMMUTABLE_REPO_ARTIFACTS:
        if not path.exists():
            changed.append(path)
            continue
        current = _sha256_bytes(path.read_bytes())
        if current != _IMMUTABLE_BASELINE_HASH.get(path, ""):
            changed.append(path)

    if not changed:
        return

    # Restore files to baseline to keep workspace stable for subsequent runs.
    for path in changed:
        baseline = _IMMUTABLE_BASELINE_BYTES.get(path)
        if baseline is not None:
            path.write_bytes(baseline)

    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    message = (
        "Immutable tracked artifacts changed during tests. "
        "Tests must use temp copies or monkeypatched Paths."
    )
    if reporter:
        reporter.write_sep("=", message, red=True)
        for path in changed:
            reporter.write_line(f"changed: {path}", red=True)
    else:
        print(message)
        for path in changed:
            print(f"changed: {path}")

    session.exitstatus = 1


@pytest.fixture(scope="session")
def immutable_repo_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Path]:
    """Provide session-scoped copies for tests needing writable artifacts."""
    base = tmp_path_factory.mktemp("immutable_repo_artifacts")
    copies: dict[str, Path] = {}
    for source in IMMUTABLE_REPO_ARTIFACTS:
        target = base / source.name
        target.write_bytes(source.read_bytes())
        copies[source.name] = target
    return copies


@pytest.fixture(autouse=True)
def _isolate_test_db_and_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use an isolated SQLite DB per test and avoid flaky parallel CPU detection."""
    db_path = tmp_path / "dashboard_test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path.as_posix()}")
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
