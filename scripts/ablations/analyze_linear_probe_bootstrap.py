#!/usr/bin/env python3
"""Summarize linear-probe training bootstraps and paired family comparisons."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from retina_bench.evaluation.calibration import evaluate_calibration
from retina_bench.evaluation.fairness import compute_fairness
from retina_bench.evaluation.robustness import evaluate_robustness
from scripts.analyze_benchmark import (
    get_dataset_df,
    load_dataset_eval_config,
    merge_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--fundus-config", default="config/fundus_datasets.yaml"
    )
    parser.add_argument(
        "--baseline-metrics",
        help=(
            "Optional paper metric CSV used as the full-training reference "
            "when legacy bootstrap files do not contain bootstrap_index=-1."
        ),
    )
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with Path(path).open() as handle:
        return yaml.safe_load(handle) or {}


def model_slug(model_id: str) -> str:
    return (
        str(model_id)
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def load_predictions(
    config: dict, input_dir: Path
) -> tuple[pd.DataFrame, list[dict]]:
    frames = []
    missing = []
    for model in config["models"]:
        for evaluation in config["evaluations"]:
            path = input_dir / (
                f"{evaluation['dataset']}_{evaluation['task']}_"
                f"{model_slug(model['id'])}_bootstrap.csv"
            )
            if not path.exists():
                missing.append(
                    {
                        "model": model["id"],
                        "dataset": evaluation["dataset"],
                        "task": evaluation["task"],
                        "expected_path": str(path),
                    }
                )
                continue
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame(), missing
    return pd.concat(frames, ignore_index=True), missing


def finite_mean(values: list[object]) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    return float(numeric.mean()) if numeric.notna().any() else np.nan


def metric_rows(
    predictions: pd.DataFrame,
    data_dir: str,
    dataset_eval_config: dict,
) -> pd.DataFrame:
    rows = []
    evaluation_columns = [
        "model",
        "model_display",
        "family",
        "domain",
        "dataset",
        "task",
    ]
    seed_columns = [
        "bootstrap_index",
        "seed",
    ]
    metadata_cache: dict[str, pd.DataFrame] = {}
    for evaluation_keys, evaluation_group in predictions.groupby(
        evaluation_columns, sort=False
    ):
        evaluation_record = dict(zip(evaluation_columns, evaluation_keys))
        dataset = evaluation_record["dataset"]
        if dataset not in metadata_cache:
            metadata_cache[dataset] = get_dataset_df(dataset, data_dir)
        metadata = metadata_cache[dataset]
        if not metadata.empty:
            merged = merge_metadata(evaluation_group, metadata, dataset)
            if not merged.empty:
                evaluation_group = merged

        eval_config = dataset_eval_config.get(dataset, {})
        fairness_specs = (
            (eval_config.get("fairness", {}) or {}).get("attributes", [])
        )
        robustness_specs = (
            (eval_config.get("robustness", {}) or {}).get("attributes", [])
        )

        for seed_keys, group in evaluation_group.groupby(
            seed_columns, sort=False
        ):
            record = {
                **evaluation_record,
                **dict(zip(seed_columns, seed_keys)),
            }
            label = pd.to_numeric(
                group["label"], errors="coerce"
            ).to_numpy()
            pred = pd.to_numeric(group["pred"], errors="coerce").to_numpy()
            prob = pd.to_numeric(group["prob"], errors="coerce").to_numpy()
            fairness = compute_fairness(
                group,
                target_col="label",
                pred_col="pred",
                prob_col="prob",
                demographic_cols=fairness_specs,
            )
            robustness = evaluate_robustness(
                group,
                source_dataset=dataset,
                target_cols=["pred", "label"],
                robustness_specs=robustness_specs,
            )
            fairness_gap = finite_mean(
                [
                    value
                    for key, value in fairness.items()
                    if key.endswith(
                        ("equalized_odds_gap", "accuracy_gap", "auc_gap")
                    )
                ]
            )
            quality_gap = finite_mean(
                [
                    value
                    for key, value in robustness.items()
                    if key.startswith("image_quality_")
                    and key.endswith("_gap")
                ]
            )
            record.update(
                {
                    "n_test": len(group),
                    "accuracy": accuracy_score(label, pred),
                    "auc": roc_auc_score(label, prob),
                    "auprc": average_precision_score(label, prob),
                    "ece": evaluate_calibration(label, prob)["ece"],
                    "fairness_gap": fairness_gap,
                    "quality_gap": quality_gap,
                }
            )
            rows.append(record)
    return pd.DataFrame(rows)


def stratified_test_sample(
    labels: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    sampled = []
    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        sampled.append(
            rng.choice(class_indices, size=len(class_indices), replace=True)
        )
    indices = np.concatenate(sampled)
    rng.shuffle(indices)
    return indices


def comparison_effect(
    auc_by_model: dict[str, float],
    model_metadata: dict[str, dict],
    comparison: str,
) -> float:
    if comparison == "vlm_vs_vm":
        left = [
            auc
            for model, auc in auc_by_model.items()
            if model_metadata[model]["family"] == "vlm"
        ]
        right = [
            auc
            for model, auc in auc_by_model.items()
            if model_metadata[model]["family"] == "vm"
        ]
    elif comparison == "ophthalmic_vs_general":
        left = [
            auc
            for model, auc in auc_by_model.items()
            if model_metadata[model]["domain"] == "ophthalmic"
        ]
        right = [
            auc
            for model, auc in auc_by_model.items()
            if model_metadata[model]["domain"] == "general"
        ]
    else:
        raise ValueError(f"Unknown comparison: {comparison}")
    return float(np.mean(left) - np.mean(right))


def paired_bootstrap_comparisons(
    predictions: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_metadata = {model["id"]: model for model in config["models"]}
    n_replicates = int(config.get("n_test_bootstraps", 2000))
    base_seed = int(config.get("base_seed", 3199)) + 100_000
    comparison_rows = []
    replicate_rows = []

    for evaluation_index, evaluation in enumerate(config["evaluations"]):
        dataset = evaluation["dataset"]
        task = evaluation["task"]
        subset = predictions[
            (predictions["dataset"] == dataset)
            & (predictions["task"] == task)
        ].copy()
        probability_matrices: dict[str, pd.DataFrame] = {}
        common_indices: set[int] | None = None
        for model in model_metadata:
            model_frame = subset[subset["model"] == model]
            matrix = model_frame.pivot(
                index="bootstrap_index",
                columns="dataset_index",
                values="prob",
            ).sort_index()
            matrix = matrix[matrix.index >= 0]
            probability_matrices[model] = matrix
            model_indices = set(int(value) for value in matrix.columns)
            common_indices = (
                model_indices
                if common_indices is None
                else common_indices & model_indices
            )

        common = sorted(common_indices or [])
        if not common:
            raise RuntimeError(f"No shared test rows for {dataset}/{task}.")
        label_map = (
            subset.drop_duplicates("dataset_index")
            .set_index("dataset_index")["label"]
        )
        labels = pd.to_numeric(label_map.loc[common], errors="raise").to_numpy(
            dtype=int
        )
        available_seeds = sorted(
            set.intersection(
                *[
                    set(int(value) for value in matrix.index)
                    for matrix in probability_matrices.values()
                ]
            )
        )
        if not available_seeds:
            raise RuntimeError(
                f"No shared training bootstrap indices for {dataset}/{task}."
            )

        rng = np.random.default_rng(base_seed + evaluation_index)
        effects = {
            "vlm_vs_vm": np.empty(n_replicates, dtype=float),
            "ophthalmic_vs_general": np.empty(n_replicates, dtype=float),
        }
        for replicate in range(n_replicates):
            train_bootstrap = int(rng.choice(available_seeds))
            sampled = stratified_test_sample(labels, rng)
            sampled_labels = labels[sampled]
            auc_by_model = {}
            for model, matrix in probability_matrices.items():
                probabilities = (
                    matrix.loc[train_bootstrap, common]
                    .to_numpy(dtype=float)[sampled]
                )
                auc_by_model[model] = roc_auc_score(
                    sampled_labels, probabilities
                )
            for comparison in effects:
                effect = comparison_effect(
                    auc_by_model, model_metadata, comparison
                )
                effects[comparison][replicate] = effect
                replicate_rows.append(
                    {
                        "dataset": dataset,
                        "task": task,
                        "comparison": comparison,
                        "replicate": replicate,
                        "train_bootstrap": train_bootstrap,
                        "auc_difference": effect,
                    }
                )

        for comparison, values in effects.items():
            lower, upper = np.quantile(values, [0.025, 0.975])
            p_lower = (np.count_nonzero(values <= 0.0) + 1) / (
                len(values) + 1
            )
            p_upper = (np.count_nonzero(values >= 0.0) + 1) / (
                len(values) + 1
            )
            comparison_rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "comparison": comparison,
                    "auc_difference": float(np.mean(values)),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "p_bootstrap": min(1.0, 2.0 * min(p_lower, p_upper)),
                    "probability_positive": float(np.mean(values > 0.0)),
                    "n_replicates": len(values),
                }
            )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["p_holm"] = np.nan
    for comparison, indices in comparisons.groupby("comparison").groups.items():
        ordered = comparisons.loc[indices, "p_bootstrap"].sort_values()
        adjusted = {}
        running_max = 0.0
        count = len(ordered)
        for rank, (index, value) in enumerate(ordered.items()):
            running_max = max(running_max, (count - rank) * float(value))
            adjusted[index] = min(1.0, running_max)
        for index, value in adjusted.items():
            comparisons.loc[index, "p_holm"] = value
    return comparisons, pd.DataFrame(replicate_rows)


def format_mean_sd(mean: float, sd: float) -> str:
    return f"{mean:.3f} ± {sd:.3f}"


def write_compact_tables(
    seed_metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    config: dict,
    output_dir: Path,
) -> None:
    model_order = [model["display_name"] for model in config["models"]]
    summary = (
        seed_metrics.groupby(
            ["dataset", "task", "model_display"], as_index=False
        )
        .agg(
            auc_mean=("auc", "mean"),
            auc_sd=("auc", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            ece_mean=("ece", "mean"),
            ece_sd=("ece", "std"),
            n_train_bootstraps=("bootstrap_index", "nunique"),
        )
    )
    summary.to_csv(
        output_dir / "linear_probe_bootstrap_model_summary.csv", index=False
    )
    auc_wide = summary.pivot(
        index=["dataset", "task"],
        columns="model_display",
        values=["auc_mean", "auc_sd"],
    )
    primary = comparisons[
        comparisons["comparison"] == "vlm_vs_vm"
    ].set_index(["dataset", "task"])

    markdown = [
        "| Dataset/task | "
        + " | ".join(model_order)
        + " | ΔAUROC VLM−VM [95% CI] | Holm p |",
        "|---|" + "---:|" * (len(model_order) + 2),
    ]
    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l" + "r" * (len(model_order) + 2) + "}",
        r"\toprule",
        "Dataset/task & "
        + " & ".join(model_order)
        + r" & $\Delta$AUROC VLM--VM [95\% CI] & Holm $p$ \\",
        r"\midrule",
    ]
    for evaluation in config["evaluations"]:
        key = (evaluation["dataset"], evaluation["task"])
        label = f"{evaluation['dataset'].upper()}/{evaluation['task']}"
        values = [
            format_mean_sd(
                auc_wide.loc[key, ("auc_mean", model)],
                auc_wide.loc[key, ("auc_sd", model)],
            )
            for model in model_order
        ]
        comparison = primary.loc[key]
        effect = (
            f"{comparison.auc_difference:+.3f} "
            f"[{comparison.ci_lower:+.3f}, {comparison.ci_upper:+.3f}]"
        )
        p_value = f"{comparison.p_holm:.3f}"
        markdown.append(
            f"| {label} | " + " | ".join(values + [effect, p_value]) + " |"
        )
        latex.append(
            f"{label.replace('_', r'\_')} & "
            + " & ".join(values + [effect, p_value])
            + r" \\"
        )
    caption = (
        "Representative linear-probe variability. Model entries are mean ± "
        "SD AUROC across class-stratified training bootstraps. The family "
        "difference and confidence interval combine paired training and test "
        "bootstrap variation; positive values favor VLMs."
    )
    markdown.extend(["", f"*{caption}*"])
    latex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            r"\label{tab:linear_probe_bootstrap}",
            r"\end{table*}",
        ]
    )
    (output_dir / "linear_probe_bootstrap_table.md").write_text(
        "\n".join(markdown) + "\n"
    )
    (output_dir / "linear_probe_bootstrap_table.tex").write_text(
        "\n".join(latex) + "\n"
    )


AXIS_METRICS = ["auc", "accuracy", "ece", "fairness_gap", "quality_gap"]
AXIS_LABELS = {
    "auc": "ΔAUROC",
    "accuracy": "ΔAccuracy",
    "ece": "ΔECE",
    "fairness_gap": "ΔFairness gap",
    "quality_gap": "ΔQuality gap",
}
TASK_LABELS = {
    "binary_dr": "Binary DR",
    "referable_dr": "Referable DR",
    "glaucoma": "Glaucoma",
    "all_tasks": "All tasks",
}


def bootstrap_metric_deltas(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "model",
        "model_display",
        "family",
        "domain",
        "dataset",
        "task",
    ]
    reference = (
        seed_metrics[seed_metrics["bootstrap_index"] == -1][
            [*identity, *AXIS_METRICS]
        ]
        .rename(
            columns={
                metric: f"{metric}_reference" for metric in AXIS_METRICS
            }
        )
        .drop_duplicates(identity)
    )
    bootstraps = seed_metrics[seed_metrics["bootstrap_index"] >= 0].copy()
    deltas = bootstraps.merge(
        reference, on=identity, how="left", validate="many_to_one"
    )
    for metric in AXIS_METRICS:
        deltas[f"delta_{metric}"] = (
            deltas[metric] - deltas[f"{metric}_reference"]
        )
    return deltas


def add_external_reference_metrics(
    seed_metrics: pd.DataFrame, baseline_path: str | None
) -> pd.DataFrame:
    """Fill missing full-training references from the paper analysis CSV."""
    identity = ["model", "dataset", "task"]
    expected = seed_metrics[identity].drop_duplicates()
    existing = seed_metrics[
        seed_metrics["bootstrap_index"] == -1
    ][identity].drop_duplicates()
    missing = expected.merge(
        existing.assign(_has_reference=True), on=identity, how="left"
    )
    missing = missing[missing["_has_reference"].isna()].drop(
        columns="_has_reference"
    )
    if missing.empty:
        return seed_metrics
    if not baseline_path:
        raise RuntimeError(
            "Bootstrap files lack full-training references. Provide "
            "--baseline-metrics for legacy result analysis."
        )

    baseline = pd.read_csv(baseline_path)
    if "method" in baseline.columns:
        baseline = baseline[baseline["method"] == "linear_probing"].copy()
    fairness_columns = [
        column
        for column in baseline.columns
        if column.endswith(("equalized_odds_gap", "accuracy_gap", "auc_gap"))
        and not column.startswith("image_quality_")
    ]
    quality_columns = [
        column
        for column in baseline.columns
        if column.startswith("image_quality_") and column.endswith("_gap")
    ]
    baseline["fairness_gap"] = (
        baseline[fairness_columns]
        .apply(pd.to_numeric, errors="coerce")
        .mean(axis=1)
        if fairness_columns
        else np.nan
    )
    baseline["quality_gap"] = (
        baseline[quality_columns]
        .apply(pd.to_numeric, errors="coerce")
        .mean(axis=1)
        if quality_columns
        else np.nan
    )

    metadata = seed_metrics[
        ["model", "model_display", "family", "domain"]
    ].drop_duplicates("model")
    references = missing.merge(
        baseline[
            [
                "model",
                "dataset",
                "task",
                "auc",
                "accuracy",
                "ece",
                "fairness_gap",
                "quality_gap",
            ]
        ].drop_duplicates(identity),
        on=identity,
        how="left",
        validate="one_to_one",
    ).merge(metadata, on="model", how="left", validate="many_to_one")
    absent = references[["auc", "accuracy", "ece"]].isna().all(axis=1)
    if absent.any():
        unavailable = references.loc[absent, identity].to_dict("records")
        raise RuntimeError(
            f"Paper references are unavailable for: {unavailable}"
        )
    references["bootstrap_index"] = -1
    references["seed"] = -1
    references["n_test"] = np.nan
    references["auprc"] = np.nan
    return pd.concat([seed_metrics, references], ignore_index=True, sort=False)


def summarize_axis_deltas(
    deltas: pd.DataFrame, task: str | None
) -> pd.DataFrame:
    subset = deltas if task is None else deltas[deltas["task"] == task]
    seed_group = [
        "model",
        "model_display",
        "family",
        "domain",
        "bootstrap_index",
    ]
    delta_columns = [f"delta_{metric}" for metric in AXIS_METRICS]
    # Macro-average dataset/task configurations within each training bootstrap.
    seed_macro = (
        subset.groupby(seed_group, as_index=False)[delta_columns].mean()
    )
    aggregations = {}
    for column in delta_columns:
        aggregations[f"{column}_mean"] = (column, "mean")
        aggregations[f"{column}_sd"] = (column, "std")
        aggregations[f"{column}_n"] = (column, "count")
    return (
        seed_macro.groupby(
            ["model", "model_display", "family", "domain"], as_index=False
        )
        .agg(**aggregations)
    )


def format_delta_cell(mean: float, sd: float) -> str:
    if pd.isna(mean):
        return "--"
    return f"{mean:+.3f} ± {sd:.3f}"


def axis_markdown_table(
    summary: pd.DataFrame, config: dict, title: str
) -> str:
    model_order = {
        model["id"]: index for index, model in enumerate(config["models"])
    }
    table = summary.assign(
        _order=summary["model"].map(model_order)
    ).sort_values("_order")
    header = (
        "| Model | "
        + " | ".join(AXIS_LABELS[metric] for metric in AXIS_METRICS)
        + " |"
    )
    separator = "|---|" + "---:|" * len(AXIS_METRICS)
    lines = [f"### {title}", "", header, separator]
    for row in table.itertuples(index=False):
        cells = []
        for metric in AXIS_METRICS:
            cells.append(
                format_delta_cell(
                    getattr(row, f"delta_{metric}_mean"),
                    getattr(row, f"delta_{metric}_sd"),
                )
            )
        lines.append(
            f"| {row.model_display} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines)


def write_axis_delta_tables(
    seed_metrics: pd.DataFrame, config: dict, output_dir: Path
) -> None:
    deltas = bootstrap_metric_deltas(seed_metrics)
    deltas.to_csv(
        output_dir / "linear_probe_bootstrap_metric_deltas.csv", index=False
    )

    task_order = list(
        dict.fromkeys(evaluation["task"] for evaluation in config["evaluations"])
    )
    markdown_tables = []
    for task in [*task_order, None]:
        key = task or "all_tasks"
        summary = summarize_axis_deltas(deltas, task)
        summary.to_csv(
            output_dir / f"linear_probe_bootstrap_{key}_axes.csv",
            index=False,
        )
        markdown = axis_markdown_table(
            summary, config, TASK_LABELS.get(key, key)
        )
        (output_dir / f"linear_probe_bootstrap_{key}_table.md").write_text(
            markdown + "\n"
        )
        markdown_tables.append(markdown)

    note = (
        "*Cells are mean signed changes ± SD relative to the corresponding "
        "full-training probe across 10 class-stratified training bootstraps. "
        "Results are macro-averaged across applicable datasets (and tasks in "
        "the final table). Positive AUROC/accuracy changes indicate improvement; "
        "positive ECE/fairness/quality-gap changes indicate degradation. "
        "Fairness and quality use only datasets with the required metadata.*"
    )
    (
        output_dir / "linear_probe_bootstrap_axis_tables.md"
    ).write_text("\n\n".join(markdown_tables + [note]) + "\n")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions, missing = load_predictions(config, input_dir)
    pd.DataFrame(missing).to_csv(output_dir / "missing_results.csv", index=False)
    if missing and not args.allow_missing:
        raise SystemExit(
            f"{len(missing)} expected result files are missing. Re-run with "
            "--allow-missing for partial inspection."
        )
    if predictions.empty:
        raise SystemExit("No bootstrap prediction files were found.")

    dataset_eval_config = load_dataset_eval_config(args.fundus_config)
    seed_metrics = metric_rows(
        predictions, args.data_dir, dataset_eval_config
    )
    seed_metrics = add_external_reference_metrics(
        seed_metrics, args.baseline_metrics
    )
    seed_metrics.to_csv(
        output_dir / "linear_probe_bootstrap_seed_metrics.csv", index=False
    )
    comparisons, replicates = paired_bootstrap_comparisons(predictions, config)
    comparisons.to_csv(
        output_dir / "linear_probe_bootstrap_comparisons.csv", index=False
    )
    replicates.to_csv(
        output_dir / "linear_probe_bootstrap_comparison_replicates.csv",
        index=False,
    )
    write_axis_delta_tables(seed_metrics, config, output_dir)
    print(f"Wrote bootstrap analysis to {output_dir}")


if __name__ == "__main__":
    main()
