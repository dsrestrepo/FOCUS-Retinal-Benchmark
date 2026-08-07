#!/usr/bin/env python3
"""Compare rebuttal LoRA variants with the existing paper LoRA setting."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from retina_bench.ablations.mllm_prompt_parsing import (  # noqa: E402
    evaluate_protocol,
    load_yaml,
)


FT_FILENAME_RE = re.compile(
    r"^train_model-(?P<train_model>.+?)_train_dataset-(?P<train_dataset>.+?)"
    r"_train_task-(?P<train_task>.+?)_train_strategy-(?P<train_strategy>.+?)"
    r"__test_model-(?P<test_model>.+?)_test_dataset-(?P<test_dataset>.+?)"
    r"_test_task-(?P<test_task>.+?)_test_strategy-(?P<test_strategy>.+?)"
    r"_split-(?P<split>.+?)\.csv$"
)
PANEL_ORDER = ["rank", "alpha", "target_modules"]
REFERENCE_VALUES = {
    "rank": "16",
    "alpha": "16",
    "target_modules": "all_linear",
}
SETTING_ORDER = {
    "rank": ["8", "16", "32"],
    "alpha": ["8", "16", "32"],
    "target_modules": ["query_value", "all_attention", "all_linear"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ablation-dir", required=True)
    parser.add_argument(
        "--reference-dir",
        required=True,
        help="Existing paper all-task/all-prompt LoRA evaluation directory.",
    )
    parser.add_argument(
        "--baseline-dir",
        required=True,
        help="Existing untrained MLLM evaluation directory.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-strategy", default="all_prompts")
    return parser.parse_args()


def collect(root: Path, source: str, allowed_strategies: set[str]) -> list[dict]:
    rows = []
    for path in root.rglob("*.csv"):
        match = FT_FILENAME_RE.match(path.name)
        if not match:
            continue
        metadata = match.groupdict()
        if metadata["train_strategy"] not in allowed_strategies:
            continue
        frame = pd.read_csv(path)
        metrics, _ = evaluate_protocol(
            frame, text_parser="first_label", score_strategy="binary_renorm"
        )
        rows.append(
            {
                **metadata,
                **metrics,
                "variant": metadata["train_strategy"],
                "source": source,
                "source_file": str(path),
            }
        )
    return rows


def display_model(model_slug: str) -> str:
    if "medgemma-27b" in model_slug.lower():
        return "MedGemma-27B"
    if "gemma-3-27b" in model_slug.lower():
        return "Gemma-3-27B"
    return model_slug.split("_")[-1]


def model_slug(model_id: str) -> str:
    return str(model_id).replace("/", "_").replace(" ", "_")


def collect_baselines(config: dict, root: Path) -> list[dict]:
    rows = []
    dataset_tasks = config.get("dataset_tasks", {})
    all_tasks = config.get("tasks", [])
    for model in config["models"]:
        slug = model_slug(model)
        for dataset in config["eval_datasets"]:
            for task in dataset_tasks.get(dataset, all_tasks):
                path = root / f"{dataset}_{task}_{slug}.csv"
                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing untrained-model result: {path}"
                    )
                frame = pd.read_csv(path)
                metrics, _ = evaluate_protocol(
                    frame,
                    text_parser="first_label",
                    score_strategy="binary_renorm",
                )
                rows.append(
                    {
                        "test_model": slug,
                        "test_dataset": dataset,
                        "test_task": task,
                        "baseline_auc": metrics["auc"],
                        "baseline_ece": metrics["ece"],
                    }
                )
    return rows


def build_rebuttal_table(
    rebuttal_rows: list[dict],
    reference_rows: list[dict],
    baseline_rows: list[dict],
    variant_config: dict[str, dict],
) -> pd.DataFrame:
    expanded = []
    for row in rebuttal_rows:
        variant = variant_config[row["variant"]]
        expanded.append(
            {
                **row,
                "panel": variant["panel"],
                "setting": str(variant["value"]),
            }
        )
    for row in reference_rows:
        for panel in PANEL_ORDER:
            expanded.append(
                {
                    **row,
                    "panel": panel,
                    "setting": REFERENCE_VALUES[panel],
                }
            )

    frame = pd.DataFrame(expanded)
    coverage_sets = []
    for _, group in frame.groupby(
        ["train_model", "panel", "setting"], sort=False
    ):
        coverage_sets.append(
            set(zip(group["test_dataset"], group["test_task"]))
        )
    common_pairs = set.intersection(*coverage_sets)
    if not common_pairs:
        raise RuntimeError(
            "The LoRA settings do not share any dataset/task evaluations."
        )
    frame = frame[
        frame.apply(
            lambda row: (row["test_dataset"], row["test_task"])
            in common_pairs,
            axis=1,
        )
    ].copy()

    baselines = pd.DataFrame(baseline_rows)
    frame = frame.merge(
        baselines,
        on=["test_model", "test_dataset", "test_task"],
        how="left",
        validate="many_to_one",
    )
    if frame[["baseline_auc", "baseline_ece"]].isna().any().any():
        raise RuntimeError(
            "Some LoRA evaluations could not be matched to an untrained baseline."
        )
    frame["model"] = frame["train_model"].map(display_model)
    frame["domain"] = frame.apply(
        lambda row: "ID"
        if row["test_dataset"] == row["train_dataset"]
        else "OOD",
        axis=1,
    )
    macro = (
        frame.groupby(["panel", "model", "setting", "domain"], as_index=False)
        .agg(
            auc=("auc", "mean"),
            ece=("ece", "mean"),
            baseline_auc=("baseline_auc", "mean"),
            baseline_ece=("baseline_ece", "mean"),
        )
    )
    macro["delta_auc"] = macro["auc"] - macro["baseline_auc"]
    macro["delta_ece"] = macro["ece"] - macro["baseline_ece"]
    table = macro.pivot(
        index=["panel", "model", "setting"],
        columns="domain",
        values=[
            "auc",
            "ece",
            "delta_auc",
            "delta_ece",
            "baseline_auc",
            "baseline_ece",
        ],
    ).reset_index()
    table.columns = [
        "_".join(part for part in column if part).lower()
        if isinstance(column, tuple)
        else column
        for column in table.columns
    ]
    for column in [
        "auc_id",
        "ece_id",
        "auc_ood",
        "ece_ood",
        "delta_auc_id",
        "delta_ece_id",
        "delta_auc_ood",
        "delta_ece_ood",
    ]:
        if column not in table:
            table[column] = float("nan")

    panel_rank = {value: index for index, value in enumerate(PANEL_ORDER)}
    model_rank = {"Gemma-3-27B": 0, "MedGemma-27B": 1}
    table["_panel_order"] = table["panel"].map(panel_rank)
    table["_model_order"] = table["model"].map(model_rank).fillna(99)
    table["_setting_order"] = table.apply(
        lambda row: SETTING_ORDER[row["panel"]].index(row["setting"]), axis=1
    )
    return (
        table.sort_values(["_panel_order", "_model_order", "_setting_order"])
        .drop(columns=["_panel_order", "_model_order", "_setting_order"])
        .reset_index(drop=True)
    )


def write_latex_table(table: pd.DataFrame, path: Path) -> None:
    panel_labels = {
        "rank": "(a) LoRA rank",
        "alpha": "(b) LoRA alpha",
        "target_modules": "(c) Target modules",
    }
    setting_labels = {
        "all_linear": "All linear (reference)",
        "all_attention": "All attention (q/k/v/o)",
        "query_value": "Query/value (original LoRA)",
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Setting & ID AUROC $\uparrow$ & ID ECE $\downarrow$ & "
        r"OOD AUROC $\uparrow$ & OOD ECE $\downarrow$ \\",
        r"\midrule",
    ]
    for panel in PANEL_ORDER:
        lines.append(
            rf"\multicolumn{{6}}{{l}}{{\textbf{{{panel_labels[panel]}}}}} \\"
        )
        panel_frame = table[table["panel"] == panel]
        for row in panel_frame.itertuples(index=False):
            setting = setting_labels.get(row.setting, row.setting)
            if panel in {"rank", "alpha"} and row.setting == "16":
                setting = "16 (reference)"
            lines.append(
                f"{row.model} & {setting} & "
                f"{value_delta(row.auc_id, row.delta_auc_id)} & "
                f"{value_delta(row.ece_id, row.delta_ece_id)} & "
                f"{value_delta(row.auc_ood, row.delta_auc_ood)} & "
                f"{value_delta(row.ece_ood, row.delta_ece_ood)} \\\\"
            )
        if panel != PANEL_ORDER[-1]:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{One-factor-at-a-time LoRA sensitivity. ID is the held-out "
            r"BRSET test set; OOD is the macro-average over the five external "
            r"dataset--task pairs shared by every setting. Parentheses show the change "
            r"from the corresponding "
            r"untrained model. The reference uses the paper LoRA hyperparameters "
            r"with the earlier all-task training protocol.}",
            r"\label{tab:lora_sensitivity}",
            r"\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def value_delta(value: float, delta: float) -> str:
    return f"{value:.3f} ({delta:+.3f})"


def write_markdown(table: pd.DataFrame, path: Path) -> None:
    panel_labels = {
        "rank": "(a) LoRA rank",
        "alpha": "(b) LoRA alpha",
        "target_modules": "(c) Target modules",
    }
    setting_labels = {
        "all_linear": "All linear (reference)",
        "all_attention": "All attention (q/k/v/o)",
        "query_value": "Query/value",
    }
    sections = []
    for panel in PANEL_ORDER:
        lines = [
            f"### {panel_labels[panel]}",
            "",
            "| Model | Setting | ID AUROC ↑ | ID ECE ↓ | "
            "OOD AUROC ↑ | OOD ECE ↓ |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in table[table["panel"] == panel].itertuples(index=False):
            setting = setting_labels.get(row.setting, row.setting)
            if panel in {"rank", "alpha"} and row.setting == "16":
                setting = "16 (reference)"
            lines.append(
                f"| {row.model} | {setting} | "
                f"{value_delta(row.auc_id, row.delta_auc_id)} | "
                f"{value_delta(row.ece_id, row.delta_ece_id)} | "
                f"{value_delta(row.auc_ood, row.delta_auc_ood)} | "
                f"{value_delta(row.ece_ood, row.delta_ece_ood)} |"
            )
        sections.append("\n".join(lines))
    sections.append(
        "*Parentheses show the change from the corresponding untrained model. "
        "ID is the macro-average over the three BRSET tasks; OOD is the "
        "macro-average over the five external dataset–task evaluations shared "
        "by every setting. The reference uses the paper LoRA "
        "hyperparameters with the earlier all-task training protocol.*"
    )
    path.write_text("\n\n".join(sections) + "\n")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    variant_names = {variant["name"] for variant in config["variants"]}
    variant_config = {variant["name"]: variant for variant in config["variants"]}

    rebuttal_rows = collect(Path(args.ablation_dir), "rebuttal", variant_names)
    reference_rows = collect(
        Path(args.reference_dir), "paper_reference", {args.reference_strategy}
    )
    baseline_rows = collect_baselines(config, Path(args.baseline_dir))

    models = {model.replace("/", "_").replace(" ", "_") for model in config["models"]}
    train_datasets = set(config["train_datasets"])
    reference_rows = [
        row
        for row in reference_rows
        if row["train_model"] in models
        and row["train_dataset"] in train_datasets
        and row["train_task"] in {"all", "all_tasks"}
    ]
    if not reference_rows:
        raise SystemExit(
            "No paper LoRA reference results matched. Check --reference-dir and "
            "--reference-strategy (expected all-task/all-prompt evaluations)."
        )
    if not rebuttal_rows:
        raise SystemExit("No rebuttal LoRA evaluation CSVs matched the configured variants.")

    expected_models = {
        model.replace("/", "_").replace(" ", "_") for model in config["models"]
    }
    expected_variants = {
        (model, variant)
        for model in expected_models
        for variant in variant_names
    }
    observed_variants = {
        (row["train_model"], row["variant"]) for row in rebuttal_rows
    }
    missing_variants = sorted(expected_variants - observed_variants)
    if missing_variants:
        raise SystemExit(
            "LoRA rebuttal evaluations are incomplete; missing model/variant "
            f"combinations: {missing_variants}"
        )
    observed_references = {row["train_model"] for row in reference_rows}
    if observed_references != expected_models:
        raise SystemExit(
            "Paper LoRA references are incomplete; missing models: "
            f"{sorted(expected_models - observed_references)}"
        )

    table = build_rebuttal_table(
        rebuttal_rows, reference_rows, baseline_rows, variant_config
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "lora_sensitivity_table.csv", index=False)
    write_latex_table(table, output_dir / "lora_sensitivity_table.tex")
    write_markdown(table, output_dir / "lora_sensitivity_table.md")
    print(
        f"Wrote the {len(table)}-row, three-panel LoRA rebuttal table to "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
