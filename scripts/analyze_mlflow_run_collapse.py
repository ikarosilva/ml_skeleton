#!/usr/bin/env python3
"""
Analyze an MLflow run (or HPO parent and its trials) to identify parameters
associated with model collapse (e.g. recall==1, precision==0).

Usage:
  python scripts/analyze_mlflow_run_collapse.py <run_id> [tracking_uri]
  MLFLOW_RUN_ID=552cc98190374fa785c90dbda368c655 python scripts/analyze_mlflow_run_collapse.py
"""

import os
import sys


def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        print("Usage: analyze_mlflow_run_collapse.py <run_id> [tracking_uri]", file=sys.stderr)
        sys.exit(1)
    tracking_uri = (
        sys.argv[2]
        if len(sys.argv) > 2
        else os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    try:
        run = client.get_run(run_id)
    except Exception as e:
        print(f"Failed to get run {run_id}: {e}", file=sys.stderr)
        print("Ensure MLflow server is running and tracking_uri is correct.", file=sys.stderr)
        sys.exit(2)

    info = run.info
    params = run.data.params or {}
    metrics = run.data.metrics or {}

    # Check if this run itself has recall==1 (single training run)
    # Support both val_* (current) and best_val_* (legacy) metric names
    recall = metrics.get("val_recall") or metrics.get("best_val_recall") or metrics.get("classifier/val_recall") or metrics.get("classifier/best_val_recall")
    precision = metrics.get("val_precision") or metrics.get("best_val_precision") or metrics.get("classifier/val_precision") or metrics.get("classifier/best_val_precision")
    accuracy = metrics.get("val_accuracy") or metrics.get("best_val_accuracy") or metrics.get("classifier/val_accuracy") or metrics.get("classifier/best_val_accuracy")

    print("=" * 70)
    print("MLflow run analysis: parameters leading to recall==1 (model collapse)")
    print("=" * 70)
    print(f"Run ID: {info.run_id}")
    print(f"Run name: {getattr(info, 'run_name', None) or '(none)'}")
    print(f"Experiment ID: {info.experiment_id}")
    print()

    # Single run summary
    print("--- This run: metrics ---")
    for k in sorted(metrics.keys()):
        v = metrics[k]
        if v is not None:
            print(f"  {k}: {v}")
    print()
    print("--- This run: parameters ---")
    for k in sorted(params.keys()):
        print(f"  {k}: {params[k]}")
    print()

    # Collapse diagnosis for this run
    if recall is not None:
        recall_f = float(recall) if not isinstance(recall, (int, float)) else recall
        if recall_f >= 0.99:
            print(">>> COLLAPSE DETECTED: recall ≈ 1 (model predicts positive for almost all samples)")
            print("    Typical cause: always-predict-positive; precision and F1 are low.")
    if precision is not None:
        prec_f = float(precision)
        if prec_f < 0.2 and recall is not None and float(recall) >= 0.99:
            print(">>> Low precision + recall=1 → model is predicting positive everywhere.")
    print()

    # Child runs (HPO trials)
    children = client.search_runs(
        experiment_ids=[info.experiment_id],
        filter_string=f'tags."mlflow.parentRunId" = "{run_id}"',
        order_by=["attributes.start_time ASC"],
        max_results=500,
    )

    if not children:
        print("No child runs (this is a single run, not an HPO parent).")
        print()
        print("--- Parameter sets associated with collapse (this run only) ---")
        if recall is not None and float(recall) >= 0.99:
            print("Parameters for this collapsed run:")
            for k, v in sorted(params.items()):
                print(f"  {k}: {v}")
        return

    # Analyze trials: which have recall==1 vs not
    collapsed = []
    ok = []
    def _get(m, key, legacy):
        return m.get(key) or m.get(legacy) or m.get(f"classifier/{key}") or m.get(f"classifier/{legacy}")

    for r in children:
        m = r.data.metrics
        rec = _get(m, "val_recall", "best_val_recall")
        if rec is not None:
            rec_f = float(rec)
            if rec_f >= 0.99:
                collapsed.append((r, rec_f, _get(m, "val_precision", "best_val_precision"), _get(m, "val_accuracy", "best_val_accuracy"), _get(m, "val_f1", "best_val_f1")))
            else:
                ok.append((r, rec_f, _get(m, "val_precision", "best_val_precision"), _get(m, "val_accuracy", "best_val_accuracy"), _get(m, "val_f1", "best_val_f1")))

    print(f"Child runs (trials): {len(children)}")
    print(f"  Collapsed (recall ≥ 0.99): {len(collapsed)}")
    print(f"  Not collapsed: {len(ok)}")
    print()

    if not collapsed:
        print("No trials with recall≥0.99 in this HPO run.")
        return

    print("--- Collapsed trials (recall ≥ 0.99) ---")
    for r, rec, prec, acc, f1 in collapsed[:20]:
        name = r.info.run_name or r.info.run_id[:8]
        print(f"  Trial {name}: recall={rec:.4f}, precision={prec}, accuracy={acc}, f1={f1}")
        p = r.data.params
        for key in ["learning_rate", "dropout", "batch_size", "training_label_noise", "training_seed"]:
            if key in p:
                print(f"    {key}={p[key]}")
    print()

    # Aggregate: compare param ranges collapsed vs ok
    def get_param(runs, key):
        return [r.data.params.get(key) for r, *_ in runs if r.data.params.get(key) is not None]

    def to_floats(vals):
        out = []
        for x in vals:
            if x is None:
                continue
            try:
                out.append(float(x))
            except (ValueError, TypeError):
                pass
        return out

    param_keys = set()
    for r, *_ in collapsed + ok:
        param_keys.update((r.data.params or {}).keys())
    param_keys = sorted(param_keys)

    print("--- Parameter comparison: collapsed vs not collapsed ---")
    for key in param_keys:
        c_vals = get_param(collapsed, key)
        o_vals = get_param(ok, key)
        if not c_vals and not o_vals:
            continue
        c_num = to_floats(c_vals)
        o_num = to_floats(o_vals)
        if c_num or o_num:
            c_str = f"collapsed n={len(c_num)}: {min(c_num):.6f} .. {max(c_num):.6f}" if c_num else "collapsed: (no numeric)"
            o_str = f"ok n={len(o_num)}: {min(o_num):.6f} .. {max(o_num):.6f}" if o_num else "ok: (no numeric)"
            print(f"  {key}:")
            print(f"    {c_str}")
            print(f"    {o_str}")
        else:
            # Categorical
            from collections import Counter
            cc = Counter(c_vals)
            co = Counter(o_vals)
            print(f"  {key}: collapsed {dict(cc)} | ok {dict(co)}")

    # Key findings: interpret which params are associated with collapse
    print("--- Key findings (parameters associated with recall=1 collapse) ---")
    from collections import Counter
    findings = []
    for key in param_keys:
        c_vals = get_param(collapsed, key)
        o_vals = get_param(ok, key)
        if not c_vals and not o_vals:
            continue
        c_num = to_floats(c_vals)
        o_num = to_floats(o_vals)
        if c_num and o_num:
            c_lo, c_hi = min(c_num), max(c_num)
            o_lo, o_hi = min(o_num), max(o_num)
            if key == "dropout" and o_hi > c_hi * 1.2:
                findings.append("  • dropout: OK runs use higher max dropout (%.4f vs %.4f). Low dropout → more collapse." % (o_hi, c_hi))
            elif key == "adam_weight_decay" and o_lo < c_lo and c_lo > 1e-5:
                findings.append("  • adam_weight_decay: some OK runs have 0 weight decay; collapsed tend to have positive decay.")
        else:
            cc = Counter(str(x) for x in c_vals)
            co = Counter(str(x) for x in o_vals)
            n_c, n_o = len(collapsed), len(ok)
            for val in set(cc) | set(co):
                c_count = cc.get(val, 0)
                o_count = co.get(val, 0)
                if c_count + o_count < 5:
                    continue
                # % of trials with this value that collapsed
                pct_c = 100 * c_count / (c_count + o_count) if (c_count + o_count) else 0
                if key == "binary_use_pos_weight" and val == "True" and c_count > o_count * 2:
                    findings.append("  • binary_use_pos_weight=True: strongly associated with collapse (%d collapsed vs %d ok). Consider False or tune pos_weight." % (c_count, o_count))
                elif key == "chunk_aggregation" and val == "max" and c_count > o_count * 2:
                    findings.append("  • chunk_aggregation=max: strongly associated with collapse (%d collapsed vs %d ok). Prefer mean." % (c_count, o_count))
                elif key == "class_weight_strategy" and val == "inverse" and c_count > o_count * 2:
                    findings.append("  • class_weight_strategy=inverse: associated with collapse (%d collapsed vs %d ok). Try sqrt_inverse or none." % (c_count, o_count))
                elif key == "clip_grad" and val == "False" and c_count > o_count * 2:
                    findings.append("  • clip_grad=False: strongly associated with collapse (%d collapsed vs %d ok). Enable clip_grad." % (c_count, o_count))
                elif key == "adam_decoupled_weight_decay" and val == "True" and o_count > c_count:
                    findings.append("  • adam_decoupled_weight_decay=True: only in OK runs (%d). May be protective." % o_count)
                elif key == "adam_amsgrad" and val == "True" and o_count > c_count * 2:
                    findings.append("  • adam_amsgrad=True: more common in OK runs (%d ok vs %d collapsed). Consider enabling." % (o_count, c_count))
    for line in findings:
        print(line)
    if not findings:
        print("  (No strong categorical associations detected; review numeric ranges above.)")
    print()
    print("--- Recommendation ---")
    print("  Recall=1 usually means the model predicts class 1 for (almost) all samples.")
    print("  In this study: prefer chunk_aggregation=mean, clip_grad=True, binary_use_pos_weight=False")
    print("  or milder class_weight_strategy (sqrt_inverse); consider higher dropout and amsgrad.")
    print("  Check: learning_rate (too high?), dropout (too low?), pos_weight/loss, early stopping.")


if __name__ == "__main__":
    main()
