"""
main.py — Thin CLI entry point.

Entry-point design
------------------
main.py is intentionally thin: it parses argv and delegates to sub-modules.
Two real entry points exist for this project:

  1. CLI (training / evaluation / batch inference):
       python main.py <command>  [options]
     Each command maps to a function in ``src/cli/`` or a top-level
     ``src/*.py`` module (train.py, evaluate.py, hpo.py, …).

  2. API server (online inference + dashboard):
       uvicorn src.api:app --host 0.0.0.0 --port 8000
     or via the convenience wrapper:
       python main.py serve-api
     The FastAPI app is defined in ``src/api.py``; its lifespan startup /
     shutdown logic lives in ``src/api_lifespan.py``.

All command implementations live in src/cli/ submodules.
ML utilities: src/train.py, src/evaluate.py, src/hpo.py, src/explain.py.
Experiment tracking: src/experiment_tracking.py.

Commands:
  preprocess          Raw data → processed dataset
  train               Train baseline + challenger models (+ optional MLflow)
  evaluate            Evaluate models, pick champion, generate explainability
  predict             Batch inference using decision policy
  monitor             Data/prediction drift + outcome monitoring
  serve-api           Start FastAPI serving endpoint
  promote-policy      Promote a run's policy to a deployment slot
  rollback-policy     Rollback to previous policy backup
  retry-webhook-dlq   Retry failed webhook deliveries
  hpo                 Hyperparameter optimization with Optuna
  explain             Standalone model explainability (permutation + SHAP)
"""

import argparse

from src.config import ExperimentConfig, Paths
from src.utils import set_seed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DS Project CLI")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("preprocess")
    sub.add_parser("split")

    p_train = sub.add_parser("train")
    p_train.add_argument("--run-id", type=str, default=None)

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--run-id", type=str, default=None)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--input", type=str, default=None)
    p_pred.add_argument("--policy", type=str, default=None)
    p_pred.add_argument("--run-id", type=str, default=None)

    p_mon = sub.add_parser("monitor")
    p_mon.add_argument("--input", type=str, default=None)
    p_mon.add_argument("--outcome", type=str, default=None)
    p_mon.add_argument("--actual-col", type=str, default=None)
    p_mon.add_argument("--run-id", type=str, default=None)

    p_api = sub.add_parser("serve-api")
    p_api.add_argument("--host", type=str, default="0.0.0.0")
    p_api.add_argument("--port", type=int, default=8000)

    p_promote = sub.add_parser("promote-policy")
    p_promote.add_argument("--run-id", type=str, required=True)
    p_promote.add_argument("--slot", type=str, default="default")

    p_rollback = sub.add_parser("rollback-policy")
    p_rollback.add_argument("--slot", type=str, default="default")

    p_retry_dlq = sub.add_parser("retry-webhook-dlq")
    p_retry_dlq.add_argument("--url", type=str, default=None)

    p_hpo = sub.add_parser("hpo")
    p_hpo.add_argument("--n-trials", type=int, default=50)
    p_hpo.add_argument("--run-id", type=str, default=None)

    p_explain = sub.add_parser("explain")
    p_explain.add_argument("--run-id", type=str, default=None)
    p_explain.add_argument("--sample-size", type=int, default=500)

    return p


def main() -> None:
    paths = Paths()
    cfg = ExperimentConfig()
    set_seed(cfg.seed)

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "preprocess":
        from src.cli.preprocess import cmd_preprocess

        cmd_preprocess(paths, cfg)

    elif args.command == "split":
        from src.cli.split import cmd_split

        cmd_split(paths, cfg)

    elif args.command == "train":
        from src.cli.train import cmd_train

        cmd_train(paths, cfg, run_id=args.run_id)

    elif args.command == "evaluate":
        from src.cli.evaluate import cmd_evaluate

        cmd_evaluate(paths, cfg, run_id=args.run_id)

    elif args.command == "predict":
        from src.cli.predict import cmd_predict

        cmd_predict(
            paths,
            cfg,
            input_path=args.input,
            policy_path=args.policy,
            run_id=args.run_id,
        )

    elif args.command == "monitor":
        from src.cli.monitor import cmd_monitor

        cmd_monitor(
            paths,
            cfg,
            input_path=args.input,
            outcome_path=args.outcome,
            actual_col=args.actual_col,
            run_id=args.run_id,
        )

    elif args.command == "serve-api":
        from src.cli.serve import cmd_serve_api

        cmd_serve_api(
            host=args.host,
            port=args.port,
            graceful_shutdown_seconds=cfg.api.graceful_shutdown_seconds,
        )

    elif args.command == "promote-policy":
        from src.cli.policy import cmd_promote_policy

        cmd_promote_policy(paths, run_id=args.run_id, slot=args.slot)

    elif args.command == "rollback-policy":
        from src.cli.policy import cmd_rollback_policy

        cmd_rollback_policy(paths, slot=args.slot)

    elif args.command == "retry-webhook-dlq":
        from src.cli.policy import cmd_retry_webhook_dlq

        cmd_retry_webhook_dlq(paths, webhook_url=args.url)

    elif args.command == "hpo":
        from src.cli.hpo import cmd_hpo

        cmd_hpo(paths, cfg, n_trials=args.n_trials, run_id=args.run_id)

    elif args.command == "explain":
        from src.cli.explain import cmd_explain

        cmd_explain(paths, cfg, run_id=args.run_id, sample_size=args.sample_size)


if __name__ == "__main__":
    main()
