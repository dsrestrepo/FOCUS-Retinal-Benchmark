import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retina_bench.evaluation.calibration import evaluate_calibration
from retina_bench.evaluation.fairness import compute_fairness
from retina_bench.evaluation.performance import evaluate_performance
from retina_bench.evaluation.robustness import evaluate_robustness
from scripts.analyze_benchmark import (
    get_dataset_df,
    load_dataset_eval_config,
    merge_metadata,
)


FT_FILENAME_RE = re.compile(
    r"^train_model-(?P<train_model>.+?)_train_dataset-(?P<train_dataset>.+?)"
    r"_train_task-(?P<train_task>.+?)_train_strategy-(?P<train_strategy>.+?)"
    r"__test_model-(?P<test_model>.+?)_test_dataset-(?P<test_dataset>.+?)"
    r"_test_task-(?P<test_task>.+?)_test_strategy-(?P<test_strategy>.+?)"
    r"_split-(?P<split>.+?)\.csv$"
)


CORE_METRICS = ["auc", "auprc", "accuracy", "f1", "ece"]


def model_to_slug(model):
    return str(model).replace("/", "_").replace(" ", "_")


def parse_ft_filename(path, slug_to_model):
    match = FT_FILENAME_RE.match(path.name)
    if not match:
        return None

    meta = match.groupdict()
    train_model = slug_to_model.get(meta["train_model"])
    test_model = slug_to_model.get(meta["test_model"])
    if train_model is None or test_model is None:
        return None

    meta["train_model"] = train_model
    meta["test_model"] = test_model
    return meta


def normalize_result_columns(df):
    df = df.copy()
    if "id" not in df.columns and "image_id" in df.columns:
        df["id"] = df["image_id"]
    elif "id" not in df.columns and "file" in df.columns:
        df["id"] = df["file"]

    if "label" not in df.columns and "ground_truth" in df.columns:
        df["label"] = df["ground_truth"]

    if "pred" not in df.columns:
        if "prediction" in df.columns:
            df["pred"] = df["prediction"]
        elif "prediction_text" in df.columns:
            df["pred"] = df["prediction_text"].apply(parse_binary_prediction_text)
            df["pred"] = pd.to_numeric(df["pred"], errors="coerce")

    if "prob" not in df.columns:
        if "prob_1" in df.columns:
            df["prob"] = df["prob_1"]
        elif "prob_yes" in df.columns:
            df["prob"] = df["prob_yes"]
        elif "pred" in df.columns:
            df["prob"] = df["pred"]

    for col in ["label", "pred", "prob"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_binary_prediction_text(value):
    text = str(value).strip().lower()
    match = re.search(r"\b(yes|no)\b", text)
    if not match:
        return np.nan
    return 1 if match.group(1) == "yes" else 0


def iter_ft_files(sft_dir, grpo_dir, slug_to_model):
    roots = [
        ("sft", Path(sft_dir)),
        ("grpo", Path(grpo_dir)),
    ]
    for ft_method, root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("**/*.csv")):
            meta = parse_ft_filename(path, slug_to_model)
            if meta is None:
                continue
            if meta["train_task"] != meta["test_task"]:
                continue
            meta["ft_method"] = ft_method
            meta["path"] = path
            yield meta


def compute_metrics_for_file(meta, data_dir, dataset_eval_config, metadata_cache):
    df = pd.read_csv(meta["path"])
    df = normalize_result_columns(df)

    test_dataset = meta["test_dataset"]
    if test_dataset not in metadata_cache:
        metadata_cache[test_dataset] = get_dataset_df(test_dataset, data_dir)

    demo_df = metadata_cache[test_dataset]
    if not demo_df.empty:
        merged = merge_metadata(df, demo_df, test_dataset)
        if merged.empty:
            print(f"Warning: metadata merge produced no rows for {meta['path']}")
        else:
            df = merged

    required = {"label", "pred", "prob"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        print(f"Warning: skipping {meta['path']} because columns are missing: {missing}")
        return None, []

    valid = df["label"].notna() & df["pred"].notna() & df["prob"].notna()
    df = df[valid].copy()

    y_true = df["label"].values
    y_pred = df["pred"].values
    y_prob = df["prob"].values

    perf = evaluate_performance(y_true, y_pred, y_prob)
    calib = evaluate_calibration(y_true, y_prob)

    dataset_cfg = dataset_eval_config.get(test_dataset, {})
    fairness_attrs = (dataset_cfg.get("fairness", {}) or {}).get("attributes", [])
    fairness, fairness_details = compute_fairness(
        df,
        target_col="label",
        pred_col="pred",
        prob_col="prob",
        demographic_cols={"attributes": fairness_attrs, "_return_details": True},
    )

    robustness_attrs = (dataset_cfg.get("robustness", {}) or {}).get("attributes", [])
    robustness = evaluate_robustness(
        df,
        source_dataset=test_dataset,
        target_cols=["pred", "label"],
        robustness_specs=robustness_attrs,
        return_details=False,
    )

    record = {
        "ft_method": meta["ft_method"],
        "train_model": meta["train_model"],
        "train_dataset": meta["train_dataset"],
        "train_task": meta["train_task"],
        "train_strategy": meta["train_strategy"],
        "test_model": meta["test_model"],
        "test_dataset": meta["test_dataset"],
        "test_task": meta["test_task"],
        "test_strategy": meta["test_strategy"],
        "split": meta["split"],
        "result_file": str(meta["path"]),
        "n": len(df),
        "is_in_domain_dataset": meta["train_dataset"] == meta["test_dataset"],
        "is_ood_dataset": meta["train_dataset"] != meta["test_dataset"],
    }
    record.update(perf)
    record.update(calib)
    record.update(fairness)
    record.update(robustness)

    subgroup_records = []
    for detail in fairness_details:
        subgroup_records.append({**record, **detail})

    return record, subgroup_records


def adapter_label(row):
    return (
        f"{row['ft_method'].upper()} | {row['train_model'].split('/')[-1]} | "
        f"train {row['train_dataset']}/{row['train_task']} ({row['train_strategy']})"
    )


def load_baseline_metrics(path):
    path = Path(path)
    if not path.exists():
        print(f"Warning: baseline metrics file not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    needed = {"model", "dataset", "task"}
    if not needed.issubset(df.columns):
        print(f"Warning: baseline metrics file lacks expected columns: {path}")
        return pd.DataFrame()
    return df


def add_baseline_comparison(metrics_df, baseline_df):
    if metrics_df.empty or baseline_df.empty:
        return metrics_df

    baseline_cols = ["model", "dataset", "task"]
    available_metrics = [metric for metric in CORE_METRICS if metric in baseline_df.columns]
    baseline = baseline_df[baseline_cols + available_metrics].copy()
    baseline = baseline.rename(
        columns={
            "model": "test_model",
            "dataset": "test_dataset",
            "task": "test_task",
            **{metric: f"baseline_{metric}" for metric in available_metrics},
        }
    )

    merged = metrics_df.merge(
        baseline,
        on=["test_model", "test_dataset", "test_task"],
        how="left",
    )
    for metric in available_metrics:
        if metric in merged.columns:
            merged[f"delta_{metric}"] = merged[metric] - merged[f"baseline_{metric}"]
    return merged


def save_ft_heatmap(df, metric, output_file, title):
    if df.empty or metric not in df.columns:
        return
    plot_df = df.dropna(subset=[metric]).copy()
    if plot_df.empty:
        return
    plot_df["adapter"] = plot_df.apply(adapter_label, axis=1)
    plot_df["test"] = plot_df["test_dataset"] + "/" + plot_df["test_task"]
    pivot = plot_df.pivot_table(index="adapter", columns="test", values=metric, aggfunc="mean")
    pivot = pivot.dropna(how="all")
    if pivot.empty:
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(14, 1.8 * len(pivot.columns) + 5), max(5, 0.45 * len(pivot) + 2)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", linewidths=0.4, cbar_kws={"label": metric})
    plt.title(title)
    plt.xlabel("Test dataset / task")
    plt.ylabel("Fine-tuned adapter")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight", dpi=200)
    plt.close()


def save_ft_plots(metrics_df, output_dir):
    for metric in CORE_METRICS:
        save_ft_heatmap(
            metrics_df,
            metric,
            output_dir / "plots" / "performance" / f"heatmap_{metric}.png",
            f"FT {metric.upper()} by adapter and test task",
        )
        delta_metric = f"delta_{metric}"
        if delta_metric in metrics_df.columns:
            save_ft_heatmap(
                metrics_df,
                delta_metric,
                output_dir / "plots" / "performance_delta" / f"heatmap_{delta_metric}.png",
                f"FT change vs baseline: {metric.upper()}",
            )

    fairness_cols = [
        c for c in metrics_df.columns
        if c.endswith(("demographic_parity_gap", "equalized_odds_gap", "accuracy_gap", "auc_gap"))
        and not c.startswith("image_quality_")
    ]
    for metric in fairness_cols:
        save_ft_heatmap(
            metrics_df,
            metric,
            output_dir / "plots" / "fairness" / f"heatmap_{metric}.png",
            f"FT fairness: {metric}",
        )

    robustness_cols = [
        c for c in metrics_df.columns
        if c.startswith("image_quality_") and c.endswith("_gap")
    ]
    for metric in robustness_cols:
        save_ft_heatmap(
            metrics_df,
            metric,
            output_dir / "plots" / "robustness" / f"heatmap_{metric}.png",
            f"FT robustness: {metric}",
        )


def write_outputs(metrics_df, subgroup_df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "aggregated_metrics_all.csv", index=False)
    metrics_df.to_csv(output_dir / "aggregated_metrics.csv", index=False)
    metrics_df[metrics_df["is_in_domain_dataset"]].to_csv(output_dir / "in_domain_metrics.csv", index=False)
    metrics_df[metrics_df["is_ood_dataset"]].to_csv(output_dir / "ood_metrics.csv", index=False)
    if not subgroup_df.empty:
        subgroup_df.to_csv(output_dir / "subgroup_metrics_all.csv", index=False)
    save_ft_plots(metrics_df, output_dir)

    for ft_method, group in metrics_df.groupby("ft_method"):
        method_dir = output_dir / ft_method
        method_dir.mkdir(parents=True, exist_ok=True)
        group.to_csv(method_dir / "metrics.csv", index=False)
        group[group["is_in_domain_dataset"]].to_csv(method_dir / "in_domain_metrics.csv", index=False)
        group[group["is_ood_dataset"]].to_csv(method_dir / "ood_metrics.csv", index=False)
        if not subgroup_df.empty:
            subgroup_df[subgroup_df["ft_method"] == ft_method].to_csv(
                method_dir / "subgroup_metrics.csv",
                index=False,
            )
        save_ft_plots(group, method_dir)


def main():
    parser = argparse.ArgumentParser(description="Analyze SFT/GRPO adapter evaluations")
    parser.add_argument("--sft_dir", default="results/evals_sft")
    parser.add_argument("--grpo_dir", default="results/evals_grpo")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--fundus_config", default="config/fundus_datasets.yaml")
    parser.add_argument("--baseline_metrics", default="results/analysis/aggregated_metrics.csv")
    parser.add_argument("--output_dir", default="results/analysis_ft")
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()

    slug_to_model = {model_to_slug(model): model for model in args.models}
    dataset_eval_config = load_dataset_eval_config(args.fundus_config)
    metadata_cache = {}

    records = []
    subgroup_records = []
    for meta in iter_ft_files(args.sft_dir, args.grpo_dir, slug_to_model):
        record, details = compute_metrics_for_file(meta, args.data_dir, dataset_eval_config, metadata_cache)
        if record is None:
            continue
        records.append(record)
        subgroup_records.extend(details)

    metrics_df = pd.DataFrame(records)
    subgroup_df = pd.DataFrame(subgroup_records)
    if metrics_df.empty:
        print("No FT result files matched the requested models.")
        return

    baseline_df = load_baseline_metrics(args.baseline_metrics)
    metrics_df = add_baseline_comparison(metrics_df, baseline_df)

    write_outputs(metrics_df, subgroup_df, Path(args.output_dir))
    print(f"FT analysis saved to {args.output_dir}")


if __name__ == "__main__":
    main()
