#!/usr/bin/env python3
"""Analyze zero-shot VLM label-prompt sensitivity across benchmark axes."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from retina_bench.evaluation.calibration import evaluate_calibration  # noqa: E402
from retina_bench.evaluation.fairness import compute_fairness  # noqa: E402
from retina_bench.evaluation.performance import evaluate_performance  # noqa: E402
from retina_bench.evaluation.robustness import evaluate_robustness  # noqa: E402
from scripts.analyze_benchmark import (  # noqa: E402
    get_dataset_df,
    load_dataset_eval_config,
    merge_metadata,
    normalize_result_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--ablation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--fundus-config", default="config/fundus_datasets.yaml"
    )
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open() as handle:
        return yaml.safe_load(handle) or {}


def model_slug(model_id: str) -> str:
    return str(model_id).replace("/", "_").replace(" ", "_")


def dataset_task_pairs(config: dict) -> list[tuple[str, str]]:
    return [
        (dataset, task)
        for dataset in config["datasets"]
        for task in config.get("dataset_tasks", {}).get(
            dataset, config["tasks"]
        )
    ]


def locate_result(
    baseline_dir: Path,
    ablation_dir: Path,
    dataset: str,
    task: str,
    variant: str,
    model_id: str,
) -> Path:
    slug = model_slug(model_id)
    if variant == "base":
        return baseline_dir / f"{dataset}_{task}_zero_shot_{slug}.csv"
    return ablation_dir / (
        f"{dataset}_{task}_zero_shot_{variant}_{slug}.csv"
    )


def mean_columns(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return frame[columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def formatted(value: float, delta: float, is_base: bool) -> str:
    if pd.isna(value):
        return "--"
    if is_base or pd.isna(delta):
        return f"{value:.3f}"
    return f"{value:.3f} ({delta:+.3f})"


def latex_escape(value: str) -> str:
    return str(value).replace("&", r"\&").replace("_", r"\_")


def build_summary(configuration_metrics: pd.DataFrame, config: dict) -> pd.DataFrame:
    fairness_columns = [
        column
        for column in configuration_metrics.columns
        if column.endswith(("equalized_odds_gap", "accuracy_gap", "auc_gap"))
        and not column.startswith("image_quality_")
        and configuration_metrics[column].notna().any()
    ]
    quality_columns = [
        column
        for column in configuration_metrics.columns
        if column.startswith("image_quality_")
        and column.endswith("_gap")
        and configuration_metrics[column].notna().any()
    ]
    source = configuration_metrics.copy()
    source["fairness_gap"] = mean_columns(source, fairness_columns)
    source["quality_gap"] = mean_columns(source, quality_columns)
    summary = (
        source.groupby(["model", "prompt_variant"], as_index=False, sort=False)
        .agg(
            auc=("auc", "mean"),
            ece=("ece", "mean"),
            fairness_gap=("fairness_gap", "mean"),
            quality_gap=("quality_gap", "mean"),
            n_dataset_tasks=("auc", "size"),
            n_fairness_dataset_tasks=("fairness_gap", "count"),
            n_quality_dataset_tasks=("quality_gap", "count"),
        )
    )
    base = (
        summary[summary["prompt_variant"] == "base"]
        .set_index("model")[["auc", "ece", "fairness_gap", "quality_gap"]]
        .add_suffix("_base")
    )
    summary = summary.join(base, on="model")
    for metric in ["auc", "ece", "fairness_gap", "quality_gap"]:
        summary[f"{metric}_delta"] = (
            summary[metric] - summary[f"{metric}_base"]
        )

    model_order = {
        model: index for index, model in enumerate(config["models"])
    }
    variant_names = [
        config.get("base_variant", {}).get("name", "base"),
        *[variant["name"] for variant in config["prompt_variants"]],
    ]
    variant_order = {
        variant: index for index, variant in enumerate(variant_names)
    }
    summary["model_name"] = summary["model"].map(
        config.get("model_display_names", {})
    ).fillna(summary["model"].map(lambda value: str(value).split("/")[-1]))
    prompt_display_names = {
        config.get("base_variant", {}).get("name", "base"): config.get(
            "base_variant", {}
        ).get("display_name", "Base"),
        **{
            variant["name"]: variant.get("display_name", variant["name"])
            for variant in config["prompt_variants"]
        },
    }
    summary["prompt_name"] = summary["prompt_variant"].map(
        prompt_display_names
    )
    summary["_model_order"] = summary["model"].map(model_order)
    summary["_variant_order"] = summary["prompt_variant"].map(variant_order)
    return (
        summary.sort_values(["_model_order", "_variant_order"])
        .drop(columns=["_model_order", "_variant_order"])
        .reset_index(drop=True)
    )


def write_tables(summary: pd.DataFrame, output_dir: Path) -> None:
    summary.to_csv(
        output_dir / "vlm_zero_shot_prompt_sensitivity_table.csv",
        index=False,
    )
    markdown = [
        "| Model | Prompt | AUROC ↑ | ECE ↓ | Fairness gap ↓ | Quality gap ↓ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Prompt & AUROC $\uparrow$ & ECE $\downarrow$ & "
        r"Fairness gap $\downarrow$ & Quality gap $\downarrow$ \\",
        r"\midrule",
    ]
    models = list(dict.fromkeys(summary["model"]))
    for model_index, model in enumerate(models):
        rows = summary[summary["model"] == model]
        for row_index, row in enumerate(rows.itertuples(index=False)):
            model_name = row.model_name if row_index == 0 else ""
            is_base = row.prompt_variant == "base"
            values = [
                formatted(row.auc, row.auc_delta, is_base),
                formatted(row.ece, row.ece_delta, is_base),
                formatted(
                    row.fairness_gap, row.fairness_gap_delta, is_base
                ),
                formatted(row.quality_gap, row.quality_gap_delta, is_base),
            ]
            markdown.append(
                f"| {model_name} | {row.prompt_name} | "
                + " | ".join(values)
                + " |"
            )
            latex.append(
                f"{latex_escape(model_name)} & "
                f"{latex_escape(row.prompt_name)} & "
                + " & ".join(values)
                + r" \\"
            )
        if model_index != len(models) - 1:
            latex.append(r"\midrule")
    latex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Zero-shot VLM sensitivity to three task-label prompt "
            r"variations. Values are macro-averaged across applicable "
            r"dataset--task evaluations; parentheses show changes from the "
            r"same model's paper prompt. All rows use L2-normalized image/text "
            r"embeddings, cosine-similarity logits scaled by 100, and a "
            r"two-class softmax. Fairness and quality use the same composite "
            r"metadata-backed gaps as the main analysis.}",
            r"\label{tab:vlm_zero_shot_prompt_sensitivity}",
            r"\end{table*}",
        ]
    )
    (output_dir / "vlm_zero_shot_prompt_sensitivity_table.md").write_text(
        "\n".join(markdown) + "\n"
    )
    (output_dir / "vlm_zero_shot_prompt_sensitivity_table.tex").write_text(
        "\n".join(latex) + "\n"
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    baseline_dir = Path(args.baseline_dir)
    ablation_dir = Path(args.ablation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_eval_config = load_dataset_eval_config(args.fundus_config)
    metadata_cache: dict[str, pd.DataFrame] = {}
    variants = [
        config.get("base_variant", {}).get("name", "base"),
        *[variant["name"] for variant in config["prompt_variants"]],
    ]
    metric_rows = []
    missing_rows = []

    for model_id in config["models"]:
        for dataset, task in dataset_task_pairs(config):
            for variant in variants:
                path = locate_result(
                    baseline_dir,
                    ablation_dir,
                    dataset,
                    task,
                    variant,
                    model_id,
                )
                if not path.exists():
                    missing_rows.append(
                        {
                            "model": model_id,
                            "dataset": dataset,
                            "task": task,
                            "prompt_variant": variant,
                            "expected_path": str(path),
                        }
                    )
                    continue
                frame = normalize_result_columns(pd.read_csv(path))
                valid = (
                    frame["label"].notna()
                    & frame["pred"].notna()
                    & frame["prob"].notna()
                    & frame["prob"].between(0.0, 1.0)
                )
                frame = frame.loc[valid].copy()
                performance = evaluate_performance(
                    frame["label"], frame["pred"], frame["prob"]
                )
                calibration = evaluate_calibration(
                    frame["label"], frame["prob"]
                )
                if dataset not in metadata_cache:
                    metadata_cache[dataset] = get_dataset_df(
                        dataset, args.data_dir
                    )
                metadata = metadata_cache[dataset]
                metric_frame = frame
                if not metadata.empty:
                    merged = merge_metadata(frame, metadata, dataset)
                    if not merged.empty:
                        metric_frame = merged
                eval_config = dataset_eval_config.get(dataset, {})
                fairness = compute_fairness(
                    metric_frame,
                    target_col="label",
                    pred_col="pred",
                    prob_col="prob",
                    demographic_cols=(
                        (eval_config.get("fairness", {}) or {}).get(
                            "attributes", []
                        )
                    ),
                )
                robustness = evaluate_robustness(
                    metric_frame,
                    source_dataset=dataset,
                    target_cols=["pred", "label"],
                    robustness_specs=(
                        (eval_config.get("robustness", {}) or {}).get(
                            "attributes", []
                        )
                    ),
                )
                metric_rows.append(
                    {
                        "model": model_id,
                        "dataset": dataset,
                        "task": task,
                        "prompt_variant": variant,
                        "source_file": str(path),
                        "n": len(frame),
                        **performance,
                        **calibration,
                        **fairness,
                        **robustness,
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(
        output_dir / "vlm_zero_shot_prompt_configuration_metrics.csv",
        index=False,
    )
    missing = pd.DataFrame(missing_rows)
    missing.to_csv(output_dir / "missing_results.csv", index=False)
    if not metrics.empty:
        write_tables(build_summary(metrics, config), output_dir)
    print(f"Wrote {len(metrics)} configuration rows to {output_dir}.")
    if missing_rows:
        message = f"{len(missing_rows)} expected result files are missing."
        if args.allow_missing:
            print(f"Warning: {message}")
        else:
            raise SystemExit(
                message + " Re-run with --allow-missing for partial analysis."
            )


if __name__ == "__main__":
    main()
