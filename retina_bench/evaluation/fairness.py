import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from retina_bench.evaluation.subgroups import make_binary_subgroup


def _binary_rates(group_df, target_col, pred_col, prob_col):
    y_true = pd.to_numeric(group_df[target_col], errors="coerce")
    y_pred = pd.to_numeric(group_df[pred_col], errors="coerce")
    y_prob = pd.to_numeric(group_df[prob_col], errors="coerce") if prob_col in group_df.columns else y_pred

    valid = y_true.notna() & y_pred.notna()
    y_true = y_true[valid].astype(int)
    y_pred = y_pred[valid].astype(int)
    y_prob = y_prob[valid]

    if len(y_true) == 0:
        return {}

    metrics = {
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_pred == 1)),
        "accuracy": float(np.mean(y_true == y_pred)),
    }

    positives = y_true == 1
    negatives = y_true == 0
    metrics["tpr"] = float(np.mean(y_pred[positives] == 1)) if positives.sum() else np.nan
    metrics["fpr"] = float(np.mean(y_pred[negatives] == 1)) if negatives.sum() else np.nan

    try:
        if y_true.nunique() == 2 and y_prob.notna().all():
            metrics["auc"] = float(roc_auc_score(y_true, y_prob.astype(float)))
        else:
            metrics["auc"] = np.nan
    except Exception:
        metrics["auc"] = np.nan

    return metrics


def _gap(values):
    values = [v for v in values if pd.notna(v)]
    if len(values) < 2:
        return np.nan
    return float(np.max(values) - np.min(values))


def _max_valid(values):
    values = [v for v in values if pd.notna(v)]
    if not values:
        return np.nan
    return float(np.max(values))

def compute_fairness(df, target_col='label', pred_col='pred', prob_col='prob', demographic_cols=None):
    """
    Computes fairness and disparity metrics across different demographic subgroups.
    For each subgroup, we calculate disparities (e.g. max gap in accuracy or FPR) 
    in predictions.
    """
    return_details = False
    if isinstance(demographic_cols, dict):
        return_details = demographic_cols.get("_return_details", False)
        demographic_cols = demographic_cols.get("attributes", [])

    if demographic_cols is None or df.empty:
        return ({}, []) if return_details else {}
    
    fairness_metrics = {}
    detail_records = []
    
    for spec in demographic_cols:
        col = spec.get("column") if isinstance(spec, dict) else spec
        if col not in df.columns:
            continue

        groups, meta = make_binary_subgroup(df, spec)
        metric_name = meta.get("name", col)
        if groups is None:
            fairness_metrics[f"{metric_name}_skipped"] = True
            continue

        subgroup_metrics = []
        for group_name, group_df in df.assign(_subgroup=groups).dropna(subset=["_subgroup"]).groupby("_subgroup"):
            rates = _binary_rates(group_df, target_col, pred_col, prob_col)
            if not rates:
                continue
            rates["attribute"] = metric_name
            rates["group"] = group_name
            subgroup_metrics.append(rates)
            detail_records.append({
                "metric_type": "fairness",
                "attribute": metric_name,
                "attribute_display": meta.get("display_name", metric_name),
                "source_column": meta.get("column", col),
                "group": group_name,
                **rates,
            })

        if len(subgroup_metrics) < 2:
            continue

        fairness_metrics[f"{metric_name}_n_min"] = int(min(m["n"] for m in subgroup_metrics))
        fairness_metrics[f"{metric_name}_demographic_parity_gap"] = _gap([m["positive_rate"] for m in subgroup_metrics])
        fairness_metrics[f"{metric_name}_accuracy_gap"] = _gap([m["accuracy"] for m in subgroup_metrics])
        fairness_metrics[f"{metric_name}_equal_opportunity_gap"] = _gap([m["tpr"] for m in subgroup_metrics])
        fairness_metrics[f"{metric_name}_fpr_gap"] = _gap([m["fpr"] for m in subgroup_metrics])
        fairness_metrics[f"{metric_name}_equalized_odds_gap"] = _max_valid([
            fairness_metrics[f"{metric_name}_equal_opportunity_gap"],
            fairness_metrics[f"{metric_name}_fpr_gap"],
        ])
        fairness_metrics[f"{metric_name}_auc_gap"] = _gap([m["auc"] for m in subgroup_metrics])
        fairness_metrics[f"{metric_name}_acc_parity_gap"] = fairness_metrics[f"{metric_name}_accuracy_gap"]

    return (fairness_metrics, detail_records) if return_details else fairness_metrics
