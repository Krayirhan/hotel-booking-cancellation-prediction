"""Retention cleanup for local model artifacts.

Policy:
- Keep the newest N run directories under ``models/``.
- Always keep the run referenced by ``models/latest.json``.
- Delete only directories older than ``max_age_days`` and outside keep set.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

RUN_ID_RE = re.compile(r"^\d{8}_\d{6}$")


@dataclass(frozen=True)
class CleanupConfig:
    models_dir: Path
    latest_json: Path
    keep_runs: int
    max_age_days: int
    apply: bool


def _parse_args() -> CleanupConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default="models", type=Path)
    parser.add_argument("--latest-json", default="models/latest.json", type=Path)
    parser.add_argument("--keep-runs", default=20, type=int)
    parser.add_argument("--max-age-days", default=30, type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Delete directories")
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be deleted",
    )
    args = parser.parse_args()
    return CleanupConfig(
        models_dir=args.models_dir,
        latest_json=args.latest_json,
        keep_runs=max(args.keep_runs, 0),
        max_age_days=max(args.max_age_days, 0),
        apply=bool(args.apply),
    )


def _load_latest_run_id(latest_json: Path) -> str | None:
    if not latest_json.exists():
        return None
    try:
        payload = json.loads(latest_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    run_id = payload.get("run_id")
    if isinstance(run_id, str) and RUN_ID_RE.match(run_id):
        return run_id
    return None


def _run_dirs(models_dir: Path) -> list[Path]:
    if not models_dir.exists():
        return []
    return [
        p
        for p in models_dir.iterdir()
        if p.is_dir() and RUN_ID_RE.match(p.name) is not None
    ]


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def main() -> int:
    cfg = _parse_args()
    models_dir = cfg.models_dir
    runs = _run_dirs(models_dir)
    if not runs:
        print(f"[clean-artifacts] no run directories found under: {models_dir}")
        return 0

    latest_run_id = _load_latest_run_id(cfg.latest_json)
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=cfg.max_age_days)

    sorted_runs = sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)
    keep_set = {p.name for p in sorted_runs[: cfg.keep_runs]}
    if latest_run_id:
        keep_set.add(latest_run_id)

    to_delete: list[Path] = []
    kept: list[tuple[Path, str]] = []

    for run_dir in sorted_runs:
        mtime = datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc)
        if run_dir.name in keep_set:
            kept.append((run_dir, "kept_by_policy"))
            continue
        if mtime >= cutoff:
            kept.append((run_dir, "kept_recent"))
            continue
        to_delete.append(run_dir)

    mode = "APPLY" if cfg.apply else "DRY-RUN"
    print(f"[clean-artifacts] mode={mode}")
    print(
        f"[clean-artifacts] models_dir={models_dir} keep_runs={cfg.keep_runs} "
        f"max_age_days={cfg.max_age_days} latest_run_id={latest_run_id or '-'}"
    )
    print(
        f"[clean-artifacts] total_runs={len(sorted_runs)} delete_candidates={len(to_delete)}"
    )

    for run_dir in to_delete:
        st = run_dir.stat()
        print(
            f"  delete: {run_dir} | modified={_fmt_ts(st.st_mtime)} "
            f"| size_bytes={sum(f.stat().st_size for f in run_dir.rglob('*') if f.is_file())}"
        )

    if cfg.apply:
        for run_dir in to_delete:
            shutil.rmtree(run_dir, ignore_errors=False)
        print(f"[clean-artifacts] deleted={len(to_delete)}")
    else:
        print("[clean-artifacts] dry-run only, no files deleted")

    if kept:
        preview = kept[:5]
        print("[clean-artifacts] kept examples:")
        for run_dir, reason in preview:
            print(f"  keep: {run_dir.name} ({reason})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
