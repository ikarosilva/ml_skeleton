#!/usr/bin/env python3
"""
Inspect an MLflow run by ID: print parameters, metrics, and (for parent HPO runs)
a table of all child trial runs.

Usage:
  python scripts/inspect_mlflow_run.py <run_id>
  python scripts/inspect_mlflow_run.py <run_id> [tracking_uri]
  MLFLOW_RUN_ID=<run_id> python scripts/inspect_mlflow_run.py

Optional env:
  MLFLOW_RUN_ID       Run ID (if not passed as first argument)
  MLFLOW_TRACKING_URI Tracking URI (default: http://localhost:5000)
"""

import os
import sys


def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        print("Usage: inspect_mlflow_run.py <run_id> [tracking_uri]", file=sys.stderr)
        print("   or: MLFLOW_RUN_ID=<run_id> inspect_mlflow_run.py", file=sys.stderr)
        sys.exit(1)

    tracking_uri = (
        sys.argv[2]
        if len(sys.argv) > 2
        else os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    # --- Fetch and print run summary ---
    try:
        run = mlflow.get_run(run_id)
    except Exception as e:
        print(f"Failed to get run {run_id}: {e}", file=sys.stderr)
        sys.exit(2)

    info = run.info
    print("Run ID:", info.run_id)
    print("Experiment ID:", info.experiment_id)
    print("Status:", info.status)
    print("Start time:", info.start_time)
    if getattr(info, "duration_ms", None) is not None and info.duration_ms:
        print("Duration (s):", info.duration_ms / 1000)
    print()

    print("--- Parameters ---")
    params = run.data.params or {}
    best_trial = params.get("best_trial_number")
    if best_trial is not None:
        print(f"  best_trial_number: {best_trial}  ← winning trial (find child with tag trial_number={best_trial}; seed in run name seed_*)")
    for k, v in sorted(params.items()):
        if k == "best_trial_number":
            continue
        print(f"  {k}: {v}")
    print()

    print("--- Metrics ---")
    for k, v in sorted(run.data.metrics.items()):
        print(f"  {k}: {v}")
    print()

    # --- Child runs (e.g. HPO trials) ---
    children = client.search_runs(
        experiment_ids=[info.experiment_id],
        filter_string=f'tags."mlflow.parentRunId" = "{run_id}"',
        order_by=["attributes.start_time ASC"],
        max_results=500,
    )

    if not children:
        return

    print(f"Child runs (trials): {len(children)}")
    print()

    # Table: trial, roc_auc, val_mae, val_accuracy, val_recall, val_f1, val_loss, epochs, lr, dropout
    # Support both val_* (current) and best_val_* (legacy) metric names
    def _m(r, key, legacy=None):
        leg = legacy or key.replace("val_", "best_val_", 1)
        m = r.data.metrics
        return m.get(key) or m.get(leg) or m.get(f"classifier/{key}") or m.get(f"classifier/{leg}")

    print("| Trial | roc_auc | val_mae | val_accuracy | val_recall | val_f1 | val_loss | epochs | lr | dropout |")
    print("|-------|--------|---------|--------------|------------|--------|---------|--------|-----|--------|")
    for r in children:
        m = r.data.metrics
        p = r.data.params
        name = r.info.run_name or r.info.run_id[:8]
        roc = m.get("roc_auc") or m.get("best_val_roc_auc") or m.get("classifier/roc_auc")
        roc_s = f"{roc:.4f}" if roc is not None else "—"
        v_mae = _m(r, "val_mae")
        v_acc = _m(r, "val_accuracy")
        v_rec = _m(r, "val_recall")
        v_f1 = _m(r, "val_f1")
        v_loss = _m(r, "val_loss")
        mae = f"{v_mae:.4f}" if v_mae is not None else "—"
        acc = f"{v_acc:.2%}" if v_acc is not None else "—"
        recall = f"{v_rec:.2%}" if v_rec is not None else "—"
        f1 = f"{v_f1:.4f}" if v_f1 is not None else "—"
        loss = f"{v_loss:.4f}" if v_loss is not None else "—"
        ep = str(m.get("epochs_completed")) if m.get("epochs_completed") is not None else "—"
        lr = str(p.get("learning_rate")) if p.get("learning_rate") is not None else "—"
        do = str(p.get("dropout")) if p.get("dropout") is not None else "—"
        print(f"| {name} | {roc_s} | {mae} | {acc} | {recall} | {f1} | {loss} | {ep} | {lr} | {do} |")

    # Best trial by val_mae (and show roc_auc)
    by_mae = [r for r in children if _m(r, "val_mae") is not None]
    if by_mae:
        best = min(by_mae, key=lambda r: _m(r, "val_mae"))
        m = best.data.metrics
        v_mae = _m(best, "val_mae")
        roc = m.get("roc_auc") or m.get("best_val_roc_auc") or m.get("classifier/roc_auc")
        print()
        print(
            "Best trial (lowest val_mae):",
            best.info.run_name,
            "| val_mae =",
            v_mae,
            "| val_accuracy =",
            _m(best, "val_accuracy"),
            "| val_recall =",
            _m(best, "val_recall"),
            "| roc_auc =",
            roc,
            "| val_f1 =",
            _m(best, "val_f1"),
        )


if __name__ == "__main__":
    main()
