"""Evaluate rollback criteria from latest monitoring report.

This utility is used by CI workflows and can also be run locally.
It writes GitHub Action outputs when GITHUB_OUTPUT is present.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _evaluate_reasons(alerts: dict[str, Any]) -> list[str]:
    reasons: list[str] = []

    # Critical model/business regressions that should trigger automated rollback.
    if bool(alerts.get("profit_drop", False)):
        reasons.append("profit_drop")
    if bool(alerts.get("prediction_drift", False)):
        reasons.append("prediction_drift")
    if bool(alerts.get("data_drift", False)) and bool(
        alerts.get("action_rate_deviation", False)
    ):
        reasons.append("data_drift+action_rate_deviation")

    return reasons


def _evaluate_non_rollback_signals(alerts: dict[str, Any]) -> list[str]:
    signals: list[str] = []
    # Operational signal only: this should not trigger policy rollback by itself.
    if bool(alerts.get("data_volume_anomaly", False)):
        signals.append("data_volume_anomaly")
    return signals


def _write_github_output(
    output_path: Path,
    *,
    rollback_required: bool,
    rollback_reasons: list[str],
    non_rollback_signals: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(f"rollback_required={'true' if rollback_required else 'false'}\n")
        fh.write(
            "rollback_reasons="
            + (",".join(rollback_reasons) if rollback_reasons else "none")
            + "\n"
        )
        fh.write(
            "non_rollback_signals="
            + (",".join(non_rollback_signals) if non_rollback_signals else "none")
            + "\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate whether rollback is required from monitoring report."
    )
    parser.add_argument(
        "--report-path",
        default="reports/monitoring/latest_monitoring_report.json",
        help="Path to monitoring report JSON",
    )
    parser.add_argument(
        "--fail-on-rollback",
        action="store_true",
        help="Exit with status 42 when rollback is required.",
    )
    args = parser.parse_args()

    report_path = Path(args.report_path)
    rollback_required = False
    rollback_reasons: list[str] = []
    non_rollback_signals: list[str] = []

    if not report_path.exists():
        print(f"No monitoring report found: {report_path}. Skipping rollback check.")
    else:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        alerts = payload.get("alerts", {})
        if not isinstance(alerts, dict):
            alerts = {}
        print(f"alerts={alerts}")
        rollback_reasons = _evaluate_reasons(alerts)
        non_rollback_signals = _evaluate_non_rollback_signals(alerts)
        rollback_required = bool(rollback_reasons)

    print(f"rollback_required={rollback_required}")
    print(f"rollback_reasons={rollback_reasons if rollback_reasons else ['none']}")
    print(
        f"non_rollback_signals={non_rollback_signals if non_rollback_signals else ['none']}"
    )

    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        _write_github_output(
            Path(github_output),
            rollback_required=rollback_required,
            rollback_reasons=rollback_reasons,
            non_rollback_signals=non_rollback_signals,
        )

    if args.fail_on_rollback and rollback_required:
        return 42
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
