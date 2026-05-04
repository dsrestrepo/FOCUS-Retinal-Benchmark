import numpy as np
import pandas as pd

from retina_bench.evaluation.calibration import evaluate_calibration
from retina_bench.evaluation.performance import evaluate_performance
from retina_bench.evaluation.subgroups import make_binary_subgroup


def _gap(values):
    values = [v for v in values if pd.notna(v)]
    if len(values) < 2:
        return np.nan
    return float(np.max(values) - np.min(values))


def evaluate_robustness(df, source_dataset=None, target_cols=['pred', 'label'], robustness_specs=None, return_details=False):
    """
    Evaluates robustness across binary metadata slices such as camera or quality.
    """
    metrics = {}
    detail_records = []

    if df.empty or not robustness_specs:
        return (metrics, detail_records) if return_details else metrics

    label_col = target_cols[1] if len(target_cols) > 1 else "label"
    pred_col = target_cols[0] if target_cols else "pred"
    prob_col = "prob"

    for spec in robustness_specs:
        col = spec.get("column") if isinstance(spec, dict) else spec
        if col not in df.columns:
            continue

        groups, meta = make_binary_subgroup(df, spec)
        metric_name = meta.get("name", col)
        if groups is None:
            metrics[f"{metric_name}_robustness_skipped"] = True
            continue

        subgroup_metrics = []
        grouped = df.assign(_subgroup=groups).dropna(subset=["_subgroup"]).groupby("_subgroup")
        for group_name, group_df in grouped:
            y_true = group_df[label_col].values
            y_pred = group_df[pred_col].values
            y_prob = group_df[prob_col].values if prob_col in group_df.columns else y_pred

            perf = evaluate_performance(y_true, y_pred, y_prob)
            calib = evaluate_calibration(y_true, y_prob)
            rec = {
                "metric_type": "robustness",
                "attribute": metric_name,
                "attribute_display": meta.get("display_name", metric_name),
                "source_column": meta.get("column", col),
                "group": group_name,
                "n": int(len(group_df)),
                **perf,
                **calib,
            }
            subgroup_metrics.append(rec)
            detail_records.append(rec)

        if len(subgroup_metrics) < 2:
            continue

        metrics[f"{metric_name}_n_min"] = int(min(m["n"] for m in subgroup_metrics))
        for metric in ["accuracy", "auc", "auprc", "f1", "ece"]:
            metrics[f"{metric_name}_{metric}_gap"] = _gap([m.get(metric) for m in subgroup_metrics])

    return (metrics, detail_records) if return_details else metrics
