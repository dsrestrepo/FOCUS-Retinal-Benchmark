#!/usr/bin/env python3
"""Analyze MLLM prompt sensitivity and optional parsing/scoring protocols."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from retina_bench.ablations.mllm_prompt_parsing import (
    dataset_task_pairs,
    evaluate_protocol,
    load_yaml,
    model_slug,
    parse_text,
)
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
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--ablation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--data-dir",
        help="Prepared dataset root used to merge demographic and quality metadata.",
    )
    parser.add_argument(
        "--fundus-config",
        default="config/fundus_datasets.yaml",
        help="Dataset evaluation config defining fairness and robustness attributes.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Analyze completed files and list missing grid entries instead of failing.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Also write the row-level long table (large and normally unnecessary).",
    )
    return parser.parse_args()


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
        return baseline_dir / f"{dataset}_{task}_{slug}.csv"
    return ablation_dir / f"{dataset}_{task}_{variant}_{slug}.csv"


def write_compact_table(metrics: pd.DataFrame, output_dir: Path) -> None:
    prompt_rows = metrics[
        metrics["parsing_protocol"] == "regex_binary_normalized"
    ].copy()
    prompt_summary = (
        prompt_rows.groupby("prompt_variant", as_index=False)
        .agg(
            auc=("auc", "mean"),
            ece=("ece", "mean"),
            accuracy=("overall_accuracy", "mean"),
            parse_error_rate=("parse_error_rate", "mean"),
            n_configurations=("auc", "size"),
        )
        .rename(columns={"prompt_variant": "setting"})
    )
    prompt_summary.insert(0, "panel", "Prompt")

    response_rows = metrics[metrics["prompt_variant"] == "base"].copy()
    response_summary = (
        response_rows.groupby("parsing_protocol", as_index=False)
        .agg(
            auc=("auc", "mean"),
            ece=("ece", "mean"),
            accuracy=("overall_accuracy", "mean"),
            parse_error_rate=("parse_error_rate", "mean"),
            n_configurations=("auc", "size"),
        )
        .rename(columns={"parsing_protocol": "setting"})
    )
    response_summary.insert(0, "panel", "Answer extraction/scoring")

    table = pd.concat([prompt_summary, response_summary], ignore_index=True)
    table.to_csv(output_dir / "prompt_response_sensitivity_table.csv", index=False)

    labels = {
        "base": "Base",
        "role": "Expert role",
        "instruction_first": "Instruction first",
        "regex_binary_normalized": "Regex + binary normalized",
        "regex_raw_yes": "Regex + raw p(yes)",
        "strict_regex_binary": "Strict regex + binary normalized",
        "first_token_binary": "First-token argmax + binary normalized",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Setting & AUROC $\uparrow$ & ECE $\downarrow$ & Accuracy$^\dagger$ $\uparrow$ "
        r"& Invalid (\%) $\downarrow$ \\",
        r"\midrule",
    ]
    for panel, panel_frame in table.groupby("panel", sort=False):
        lines.append(rf"\multicolumn{{5}}{{l}}{{\textbf{{{panel}}}}} \\")
        for row in panel_frame.itertuples(index=False):
            label = labels.get(row.setting, str(row.setting).replace("_", " "))
            lines.append(
                f"{label} & {row.auc:.3f} & {row.ece:.3f} & "
                f"{row.accuracy:.3f} & {100.0 * row.parse_error_rate:.1f} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.extend(
        [
            r"\end{tabular}",
            r"\caption{Macro-average MLLM prompt and answer-extraction/scoring sensitivity. "
            r"Prompt variants use the permissive regex and binary-normalized scores; "
            r"response protocols use the base prompt. Invalid denotes failure to produce "
            r"a hard decision under the indicated extraction rule. "
            r"$^\dagger$Invalid decisions are counted as incorrect.}",
            r"\label{tab:mllm_prompt_response_sensitivity}",
            r"\end{table}",
        ]
    )
    (output_dir / "prompt_response_sensitivity_table.tex").write_text(
        "\n".join(lines) + "\n"
    )


def decision_rule_counts(frame: pd.DataFrame, method: str) -> dict:
    labels = pd.to_numeric(
        frame.get("ground_truth", frame.get("label")), errors="coerce"
    )
    prediction = pd.Series(float("nan"), index=frame.index, dtype=float)

    if method == "first_token_argmax":
        yes = pd.to_numeric(
            frame.get("prob_yes", pd.Series(float("nan"), index=frame.index)),
            errors="coerce",
        )
        no = pd.to_numeric(
            frame.get("prob_no", pd.Series(float("nan"), index=frame.index)),
            errors="coerce",
        )
        captured = yes.notna() & no.notna() & ((yes + no) > 0)
        prediction.loc[captured] = (yes.loc[captured] >= no.loc[captured]).astype(
            float
        )
    elif method == "response_regex_first":
        prediction = frame["prediction_text"].map(
            lambda value: parse_text(value, "first_label")
        )
    elif method == "response_regex_strict":
        prediction = frame["prediction_text"].map(
            lambda value: parse_text(value, "strict_label")
        )
    else:
        raise ValueError(f"Unknown response decision method: {method}")

    labelled = labels.notna()
    captured = labelled & prediction.notna()
    correct = captured & (prediction == labels)
    ambiguous = 0
    if method == "response_regex_first":
        ambiguous = int(
            frame.loc[labelled, "prediction_text"]
            .map(
                lambda value: {
                    match.lower()
                    for match in re.findall(
                        r"\b(yes|no)\b", str(value), flags=re.IGNORECASE
                    )
                }
                == {"yes", "no"}
            )
            .sum()
        )
    return {
        "method": method,
        "n_total": int(labelled.sum()),
        "n_captured": int(captured.sum()),
        "n_correct": int(correct.sum()),
        "n_ambiguous": ambiguous,
    }


METHOD_ORDER = {
    "first_token_argmax": 0,
    "response_regex_first": 1,
    "response_regex_strict": 2,
}
METHOD_LABELS = {
    "first_token_argmax": "First-token yes/no argmax",
    "response_regex_first": "First yes/no regex",
    "response_regex_strict": "Strict full-response regex",
}


def summarize_accuracy_counts(
    counts: pd.DataFrame, group_columns: list[str] | None = None
) -> pd.DataFrame:
    group_columns = group_columns or []
    table = (
        counts.groupby([*group_columns, "method"], as_index=False)[
            ["n_total", "n_captured", "n_correct", "n_ambiguous"]
        ]
        .sum()
    )
    table["accuracy_captured"] = table["n_correct"] / table["n_captured"]
    # Treat an unavailable decision as an error rather than evaluating only
    # the easy-to-parse subset.
    table["accuracy_full_set"] = table["n_correct"] / table["n_total"]
    table["coverage"] = table["n_captured"] / table["n_total"]
    table["error_rate"] = 1.0 - table["coverage"]
    table["ambiguous_rate"] = table["n_ambiguous"] / table["n_total"]
    table["_method_order"] = table["method"].map(METHOD_ORDER)
    return table


def write_accuracy_coverage_table(counts: pd.DataFrame, output_dir: Path) -> None:
    table = summarize_accuracy_counts(counts)
    table = table.sort_values("_method_order").drop(columns="_method_order")
    table.to_csv(output_dir / "response_accuracy_coverage_table.csv", index=False)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Decision rule & Captured Acc. $\uparrow$ & Full-set Acc. $\uparrow$ & "
        r"Coverage (\%) $\uparrow$ & Error (\%) $\downarrow$ & "
        r"Both labels (\%) $\downarrow$ \\",
        r"\midrule",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"{METHOD_LABELS[row.method]} & {row.accuracy_captured:.3f} & "
            f"{row.accuracy_full_set:.3f} & {100.0 * row.coverage:.1f} & "
            f"{100.0 * row.error_rate:.1f} & "
            f"{100.0 * row.ambiguous_rate:.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Pooled base-prompt accuracy and coverage under first-token "
            r"and generated-response decision rules. Full-set accuracy counts "
            r"uncaptured responses as errors.}",
            r"\label{tab:mllm_response_accuracy_coverage}",
            r"\end{table}",
        ]
    )
    (output_dir / "response_accuracy_coverage_table.tex").write_text(
        "\n".join(lines) + "\n"
    )


def display_model_name(model_id: str) -> str:
    name = str(model_id).lower()
    if "medgemma-1.5-4b" in name:
        return "MedGemma-1.5-4B"
    if "medgemma-27b" in name:
        return "MedGemma-27B"
    if "medgemma-4b" in name:
        return "MedGemma-4B"
    if "gemma-3-27b" in name:
        return "Gemma-3-27B"
    if "qwen3-vl-8b" in name:
        return "Qwen3-VL-8B"
    if "llava" in name:
        return "LLaVA-NeXT-8B"
    return str(model_id).split("/")[-1]


PROMPT_LABELS = {
    "base": "Base",
    "role": "Expert role",
    "instruction_first": "Instruction first",
}


def _mean_columns(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return frame[columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def _format_value_delta(value: float, delta: float, base: bool) -> str:
    if pd.isna(value):
        return "--"
    if base or pd.isna(delta):
        return f"{value:.3f}"
    return f"{value:.3f} ({delta:+.3f})"


def write_prompt_axes_table(
    configuration_metrics: pd.DataFrame, output_dir: Path
) -> None:
    """Write the prompt-only table using the same composite gaps as the paper."""
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

    table_source = configuration_metrics.copy()
    table_source["fairness_gap"] = _mean_columns(
        table_source, fairness_columns
    )
    table_source["quality_gap"] = _mean_columns(table_source, quality_columns)

    table = (
        table_source.groupby(
            ["model", "prompt_variant"], as_index=False, sort=False
        )
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
        table[table["prompt_variant"] == "base"]
        .set_index("model")[["auc", "ece", "fairness_gap", "quality_gap"]]
        .add_suffix("_base")
    )
    table = table.join(base, on="model")
    for metric in ["auc", "ece", "fairness_gap", "quality_gap"]:
        table[f"{metric}_delta"] = table[metric] - table[f"{metric}_base"]

    model_order = {
        model: index
        for index, model in enumerate(dict.fromkeys(configuration_metrics["model"]))
    }
    prompt_order = {
        variant: index
        for index, variant in enumerate(
            ["base", *dict.fromkeys(configuration_metrics["prompt_variant"])]
        )
    }
    table["model_name"] = table["model"].map(display_model_name)
    table["_model_order"] = table["model"].map(model_order)
    table["_prompt_order"] = table["prompt_variant"].map(prompt_order)
    table = table.sort_values(["_model_order", "_prompt_order"]).reset_index(
        drop=True
    )
    table.drop(columns=["_model_order", "_prompt_order"]).to_csv(
        output_dir / "mllm_prompt_axes_sensitivity_table.csv", index=False
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Prompt & AUROC $\uparrow$ & ECE $\downarrow$ & "
        r"Fairness gap $\downarrow$ & Quality gap $\downarrow$ \\",
        r"\midrule",
    ]
    models = list(dict.fromkeys(table["model"]))
    for model_index, model in enumerate(models):
        model_rows = table[table["model"] == model]
        for row_index, row in enumerate(model_rows.itertuples(index=False)):
            model_label = row.model_name if row_index == 0 else ""
            is_base = row.prompt_variant == "base"
            lines.append(
                f"{model_label} & {PROMPT_LABELS.get(row.prompt_variant, row.prompt_variant)} & "
                f"{_format_value_delta(row.auc, row.auc_delta, is_base)} & "
                f"{_format_value_delta(row.ece, row.ece_delta, is_base)} & "
                f"{_format_value_delta(row.fairness_gap, row.fairness_gap_delta, is_base)} & "
                f"{_format_value_delta(row.quality_gap, row.quality_gap_delta, is_base)} \\\\"
            )
        if model_index != len(models) - 1:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{MLLM sensitivity to two information-preserving prompt "
            r"variations. Values are macro-averaged across applicable dataset--task "
            r"evaluations; parentheses show the change from the same model's base "
            r"prompt. AUROC summarizes cross-dataset discrimination and ECE "
            r"calibration. Fairness and quality use the same composite demographic "
            r"and image-quality gaps as the main analysis. To isolate prompting "
            r"from response parsing, all rows use binary-normalized first-token "
            r"yes/no probabilities and their argmax decision.}",
            r"\label{tab:mllm_prompt_axes_sensitivity}",
            r"\end{table*}",
        ]
    )
    (output_dir / "mllm_prompt_axes_sensitivity_table.tex").write_text(
        "\n".join(lines) + "\n"
    )


def write_accuracy_coverage_by_model_table(
    counts: pd.DataFrame, output_dir: Path
) -> None:
    model_order = {
        model: index for index, model in enumerate(dict.fromkeys(counts["model"]))
    }
    table = summarize_accuracy_counts(counts, ["model"])
    table["model_name"] = table["model"].map(display_model_name)
    table["_model_order"] = table["model"].map(model_order)
    table = (
        table.sort_values(["_model_order", "_method_order"])
        .drop(columns=["_model_order", "_method_order"])
        .reset_index(drop=True)
    )
    table.to_csv(
        output_dir / "response_accuracy_coverage_by_model_table.csv", index=False
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Model & Decision rule & Captured Acc. $\uparrow$ & "
        r"Full-set Acc. $\uparrow$ & Coverage (\%) $\uparrow$ & "
        r"Error (\%) $\downarrow$ & Both labels (\%) $\downarrow$ \\",
        r"\midrule",
    ]
    models = list(dict.fromkeys(table["model"]))
    for model_index, model in enumerate(models):
        model_rows = table[table["model"] == model]
        for row_index, row in enumerate(model_rows.itertuples(index=False)):
            model_label = row.model_name if row_index == 0 else ""
            lines.append(
                f"{model_label} & {METHOD_LABELS[row.method]} & "
                f"{row.accuracy_captured:.3f} & {row.accuracy_full_set:.3f} & "
                f"{100.0 * row.coverage:.1f} & "
                f"{100.0 * row.error_rate:.1f} & "
                f"{100.0 * row.ambiguous_rate:.3f} \\\\"
            )
        if model_index != len(models) - 1:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Base-prompt accuracy and coverage by MLLM. Counts are "
            r"pooled across applicable dataset--task evaluations for each model; "
            r"full-set accuracy counts uncaptured responses as errors.}",
            r"\label{tab:mllm_response_accuracy_coverage_by_model}",
            r"\end{table*}",
        ]
    )
    (
        output_dir / "response_accuracy_coverage_by_model_table.tex"
    ).write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    baseline_dir = Path(args.baseline_dir)
    ablation_dir = Path(args.ablation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = ["base", *config["prompt_variants"]]
    metrics_rows = []
    prompt_axis_rows = []
    accuracy_count_rows = []
    prediction_frames = []
    missing_rows = []
    dataset_eval_config = load_dataset_eval_config(args.fundus_config)
    metadata_cache: dict[str, pd.DataFrame] = {}

    for model_id in config["models"]:
        for dataset, task in dataset_task_pairs(config):
            for variant in variants:
                path = locate_result(
                    baseline_dir, ablation_dir, dataset, task, variant, model_id
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

                source = pd.read_csv(path)
                prompt_metrics, prompt_parsed = evaluate_protocol(
                    source,
                    text_parser="first_token_argmax",
                    score_strategy="binary_renorm",
                )
                prompt_group = prompt_parsed.copy()
                if args.data_dir:
                    if dataset not in metadata_cache:
                        metadata_cache[dataset] = get_dataset_df(
                            dataset, args.data_dir
                        )
                    metadata = metadata_cache[dataset]
                    if not metadata.empty:
                        merged = merge_metadata(prompt_group, metadata, dataset)
                        if merged.empty:
                            print(
                                "Warning: metadata merge produced no rows for "
                                f"{path}; fairness/quality gaps will be unavailable."
                            )
                        else:
                            prompt_group = merged

                valid = (
                    prompt_group["label"].notna()
                    & prompt_group["pred"].notna()
                    & prompt_group["prob"].notna()
                    & prompt_group["prob"].between(0.0, 1.0)
                )
                prompt_group = prompt_group.loc[valid].copy()
                eval_config = dataset_eval_config.get(dataset, {})
                fairness_specs = (
                    (eval_config.get("fairness", {}) or {}).get(
                        "attributes", []
                    )
                )
                robustness_specs = (
                    (eval_config.get("robustness", {}) or {}).get(
                        "attributes", []
                    )
                )
                fairness = compute_fairness(
                    prompt_group,
                    target_col="label",
                    pred_col="pred",
                    prob_col="prob",
                    demographic_cols=fairness_specs,
                )
                robustness = evaluate_robustness(
                    prompt_group,
                    source_dataset=dataset,
                    target_cols=["pred", "label"],
                    robustness_specs=robustness_specs,
                )
                prompt_axis_rows.append(
                    {
                        "model": model_id,
                        "dataset": dataset,
                        "task": task,
                        "prompt_variant": variant,
                        "source_file": str(path),
                        **prompt_metrics,
                        **fairness,
                        **robustness,
                    }
                )
                if variant == "base":
                    for method in [
                        "first_token_argmax",
                        "response_regex_first",
                        "response_regex_strict",
                    ]:
                        accuracy_count_rows.append(
                            {
                                "model": model_id,
                                "dataset": dataset,
                                "task": task,
                                **decision_rule_counts(source, method),
                            }
                        )
                protocols = config.get(
                    "response_protocols", config.get("parsing_protocols", {})
                )
                for protocol_name, protocol in protocols.items():
                    metrics, parsed = evaluate_protocol(
                        source,
                        text_parser=protocol["text_parser"],
                        score_strategy=protocol["score"],
                    )
                    metadata = {
                        "model": model_id,
                        "dataset": dataset,
                        "task": task,
                        "prompt_variant": variant,
                        "parsing_protocol": protocol_name,
                        "text_parser": protocol["text_parser"],
                        "score_strategy": protocol["score"],
                        "source_file": str(path),
                    }
                    metrics_rows.append({**metadata, **metrics})
                    keep = [
                        column
                        for column in [
                            "dataset_index",
                            "id",
                            "label",
                            "prediction_text",
                            "pred",
                            "prob_yes",
                            "prob_no",
                            "prob",
                        ]
                        if column in parsed.columns
                    ]
                    if args.save_predictions:
                        prediction_frames.append(parsed[keep].assign(**metadata))

    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(
        output_dir / "prompt_response_protocol_metrics.csv", index=False
    )
    if not metrics_frame.empty:
        write_compact_table(metrics_frame, output_dir)
    prompt_axis_frame = pd.DataFrame(prompt_axis_rows)
    prompt_axis_frame.to_csv(
        output_dir / "mllm_prompt_axes_configuration_metrics.csv", index=False
    )
    if not prompt_axis_frame.empty:
        write_prompt_axes_table(prompt_axis_frame, output_dir)
    if accuracy_count_rows:
        accuracy_counts = pd.DataFrame(accuracy_count_rows)
        write_accuracy_coverage_table(accuracy_counts, output_dir)
        write_accuracy_coverage_by_model_table(accuracy_counts, output_dir)
    if args.save_predictions and prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            output_dir / "prompt_response_protocol_predictions.csv", index=False
        )
    pd.DataFrame(missing_rows).to_csv(
        output_dir / "missing_results.csv", index=False
    )

    print(f"Wrote {len(metrics_rows)} metric rows to {output_dir}")
    if missing_rows:
        message = f"{len(missing_rows)} expected result files are missing."
        if args.allow_missing:
            print(f"Warning: {message}")
        else:
            raise SystemExit(message + " Re-run with --allow-missing for partial analysis.")


if __name__ == "__main__":
    main()
