#!/usr/bin/env python3
"""Analyze task-specific LoRA sensitivity with the paper scoring protocol."""

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
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def paper_metrics(path: Path) -> dict:
    frame = pd.read_csv(path)
    metrics, _ = evaluate_protocol(
        frame,
        text_parser="first_label",
        score_strategy="raw_yes",
    )
    return metrics


def collect_adapters(
    root: Path, allowed_strategies: set[str]
) -> list[dict]:
    rows = []
    for path in root.rglob("*.csv"):
        match = FT_FILENAME_RE.match(path.name)
        if not match:
            continue
        metadata = match.groupdict()
        if metadata["train_strategy"] not in allowed_strategies:
            continue
        if metadata["test_strategy"] != "base":
            continue
        rows.append(
            {
                **metadata,
                **paper_metrics(path),
                "source_file": str(path),
            }
        )
    return rows


def model_slug(model_id: str) -> str:
    return str(model_id).replace("/", "_").replace(" ", "_")


def display_model(model_slug_value: str) -> str:
    if "medgemma-27b" in model_slug_value.lower():
        return "MedGemma-27B"
    if "gemma-3-27b" in model_slug_value.lower():
        return "Gemma-3-27B"
    return model_slug_value.split("_")[-1]


def evaluation_pairs(config: dict, task: str) -> list[tuple[str, str]]:
    datasets = config["eval_datasets"]
    dataset_tasks = config.get("dataset_tasks", {})
    all_tasks = config.get("tasks", [])
    return [
        (dataset, task)
        for dataset in datasets
        if task in dataset_tasks.get(dataset, all_tasks)
    ]


def collect_baselines(config: dict, baseline_dir: Path) -> list[dict]:
    rows = []
    for model in config["models"]:
        slug = model_slug(model)
        for task in config["train_tasks"]:
            for dataset, test_task in evaluation_pairs(config, task):
                path = baseline_dir / f"{dataset}_{test_task}_{slug}.csv"
                if not path.exists():
                    raise FileNotFoundError(f"Missing base-model result: {path}")
                rows.append(
                    {
                        "test_model": slug,
                        "test_dataset": dataset,
                        "test_task": test_task,
                        **paper_metrics(path),
                    }
                )
    return rows


def expected_keys(
    config: dict, strategies: set[str]
) -> set[tuple[str, str, str, str]]:
    keys = set()
    for model in config["models"]:
        slug = model_slug(model)
        for task in config["train_tasks"]:
            for strategy in strategies:
                for dataset, _ in evaluation_pairs(config, task):
                    keys.add((slug, task, strategy, dataset))
    return keys


def filter_and_validate(
    rows: list[dict],
    config: dict,
    strategies: set[str],
    source: str,
) -> list[dict]:
    models = {model_slug(model) for model in config["models"]}
    tasks = set(config["train_tasks"])
    train_dataset = config["train_dataset"]
    filtered = [
        row
        for row in rows
        if row["train_model"] in models
        and row["train_dataset"] == train_dataset
        and row["train_task"] in tasks
        and row["test_task"] == row["train_task"]
    ]
    observed = {
        (
            row["train_model"],
            row["train_task"],
            row["train_strategy"],
            row["test_dataset"],
        )
        for row in filtered
    }
    missing = sorted(expected_keys(config, strategies) - observed)
    if missing:
        raise RuntimeError(
            f"{source} results are incomplete; missing {len(missing)} "
            f"model/task/setting/dataset entries. First entries: {missing[:10]}"
        )
    return filtered


def build_table(
    variants: list[dict],
    references: list[dict],
    baselines: list[dict],
    config: dict,
) -> pd.DataFrame:
    variant_config = {
        variant["name"]: variant for variant in config["variants"]
    }
    expanded = []
    for row in variants:
        variant = variant_config[row["train_strategy"]]
        expanded.append(
            {
                **row,
                "panel": variant["panel"],
                "setting": str(variant["value"]),
            }
        )
    for row in references:
        for panel in PANEL_ORDER:
            expanded.append(
                {
                    **row,
                    "panel": panel,
                    "setting": REFERENCE_VALUES[panel],
                }
            )

    frame = pd.DataFrame(expanded)
    baseline = pd.DataFrame(baselines).rename(
        columns={"auc": "baseline_auc", "ece": "baseline_ece"}
    )
    frame = frame.merge(
        baseline[
            [
                "test_model",
                "test_dataset",
                "test_task",
                "baseline_auc",
                "baseline_ece",
            ]
        ],
        on=["test_model", "test_dataset", "test_task"],
        how="left",
        validate="many_to_one",
    )
    frame["model"] = frame["train_model"].map(display_model)
    frame["domain"] = frame["test_dataset"].map(
        lambda dataset: "ID"
        if dataset == config["train_dataset"]
        else "OOD"
    )
    macro = (
        frame.groupby(
            ["panel", "model", "setting", "domain"], as_index=False
        )
        .agg(
            auc=("auc", "mean"),
            ece=("ece", "mean"),
            baseline_auc=("baseline_auc", "mean"),
            baseline_ece=("baseline_ece", "mean"),
            n_evaluations=("auc", "size"),
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
            "n_evaluations",
        ],
    ).reset_index()
    table.columns = [
        "_".join(part for part in column if part).lower()
        if isinstance(column, tuple)
        else column
        for column in table.columns
    ]
    panel_rank = {value: index for index, value in enumerate(PANEL_ORDER)}
    model_rank = {"Gemma-3-27B": 0, "MedGemma-27B": 1}
    table["_panel_order"] = table["panel"].map(panel_rank)
    table["_model_order"] = table["model"].map(model_rank).fillna(99)
    table["_setting_order"] = table.apply(
        lambda row: SETTING_ORDER[row["panel"]].index(row["setting"]),
        axis=1,
    )
    return (
        table.sort_values(
            ["_panel_order", "_model_order", "_setting_order"]
        )
        .drop(
            columns=["_panel_order", "_model_order", "_setting_order"]
        )
        .reset_index(drop=True)
    )


def value_delta(value: float, delta: float) -> str:
    return f"{value:.3f} ({delta:+.3f})"


def write_markdown(table: pd.DataFrame, path: Path) -> None:
    labels = {
        "rank": "LoRA rank",
        "alpha": "LoRA alpha",
        "target_modules": "Target modules",
        "all_linear": "All linear (paper)",
        "all_attention": "All attention (q/k/v/o)",
        "query_value": "Query/value",
    }
    sections = []
    for panel in PANEL_ORDER:
        lines = [
            f"### {labels[panel]}",
            "",
            "| Model | Setting | ID AUROC ↑ | ID ECE ↓ | "
            "OOD AUROC ↑ | OOD ECE ↓ |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in table[table["panel"] == panel].itertuples(index=False):
            setting = labels.get(row.setting, row.setting)
            if panel in {"rank", "alpha"} and row.setting == "16":
                setting = "16 (paper)"
            lines.append(
                f"| {row.model} | {setting} | "
                f"{value_delta(row.auc_id, row.delta_auc_id)} | "
                f"{value_delta(row.ece_id, row.delta_ece_id)} | "
                f"{value_delta(row.auc_ood, row.delta_auc_ood)} | "
                f"{value_delta(row.ece_ood, row.delta_ece_ood)} |"
            )
        sections.append("\n".join(lines))
    sections.append(
        "*Values in parentheses are changes from the corresponding "
        "untrained model. ID is the BRSET binary-DR evaluation; OOD is the "
        "macro-average of external binary-DR evaluations. Raw first-token "
        "p(yes) scoring matches the paper analysis.*"
    )
    path.write_text("\n\n".join(sections) + "\n")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    variant_names = {variant["name"] for variant in config["variants"]}
    variants = filter_and_validate(
        collect_adapters(Path(args.ablation_dir), variant_names),
        config,
        variant_names,
        "Ablation",
    )
    references = filter_and_validate(
        collect_adapters(Path(args.reference_dir), {"base"}),
        config,
        {"base"},
        "Paper reference",
    )
    baselines = collect_baselines(config, Path(args.baseline_dir))
    table = build_table(variants, references, baselines, config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        output_dir / "lora_sensitivity_paper_aligned_table.csv",
        index=False,
    )
    write_markdown(
        table,
        output_dir / "lora_sensitivity_paper_aligned_table.md",
    )
    print(f"Wrote paper-aligned LoRA sensitivity analysis to {output_dir}")


if __name__ == "__main__":
    main()
