#!/usr/bin/env python3
"""Generate method-split summary tables and a LaTeX table with task subtables.

Outputs:
- paper/tables/model_type_method_summary.csv
- paper/tables/task_model_type_method_summary.csv
- paper/tables/model_type_method_task_table.tex
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[1]
BASE_METRICS = ROOT / "results" / "analysis" / "aggregated_metrics.csv"
RELIABILITY_MATRIX = ROOT / "paper" / "tables" / "model_type_reliability_matrix.csv"
TABLES = ROOT / "paper" / "tables"
FIGURES = ROOT / "paper" / "figures"

TASK_ORDER = ["binary_dr", "referable_dr", "glaucoma"]
MODEL_TYPE_ORDER = [
    "cv_general",
    "cv_ophthalmo",
    "vlm_general",
    "vlm_ophthalmo",
    "mllm_general",
    "mllm_medical",
]
METHOD_ORDER = ["linear_probing", "zero_shot", "base"]

TASK_DISPLAY = {
    "binary_dr": "Binary DR",
    "referable_dr": "Referable DR",
    "glaucoma": "Glaucoma",
}
MODEL_TYPE_DISPLAY = {
    "cv_general": "General VM",
    "cv_ophthalmo": "Ophthalmic VM",
    "vlm_general": "General VLM-encoders",
    "vlm_ophthalmo": "Ophthalmic VLM-encoders",
    "mllm_general": "General MLLMs",
    "mllm_medical": "Medical MLLMs",
}

RELIABILITY_DISPLAY_MAP = {
    "General Vision Encoder Model (VM)": "General VM",
    "Ophthalmic Vision Encoder Model (VM)": "Ophthalmic VM",
    "General Dual Encoder Vision Language Model (VLM-encoders)": "General VLM-encoders",
    "Ophthalmic Dual Encoder Vision Language Model (VLM-encoders)": "Ophthalmic VLM-encoders",
    "General Multimodal LLM (MLLMs)": "General MLLMs",
    "Medical Multimodal LLM (MLLMs)": "Medical MLLMs",
}
METHOD_DISPLAY = {
    "base": "Zero-shot prompting",
    "linear_probing": "Linear probing",
    "zero_shot": "Zero-shot",
}

METRICS = ["accuracy", "auc", "auprc", "ece"]
RELIABILITY_METRICS = ["fairness", "quality", "shift"]
METRIC_DISPLAY = {
    "accuracy": "Accuracy",
    "auc": "AUROC",
    "auprc": "AUPRC",
    "ece": "ECE",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def format_metric(value: float) -> str:
    return f"{value:.3f}"


def format_metric_cell(value: float, best_value: float, higher_is_better: bool) -> str:
    is_best = value == best_value
    if not higher_is_better:
        is_best = value == best_value
    text = format_metric(value)
    return f"\\textbf{{{text}}}" if is_best else text


def prepare_metrics() -> pd.DataFrame:
    if not BASE_METRICS.exists():
        raise FileNotFoundError(f"Missing {BASE_METRICS}")
    df = pd.read_csv(BASE_METRICS)
    df["task_display"] = df["task"].map(TASK_DISPLAY).fillna(df["task"])
    df["model_type_display"] = df["model_type"].map(MODEL_TYPE_DISPLAY).fillna(df["model_type"])
    df["method_display"] = df["method"].map(METHOD_DISPLAY).fillna(df["method"])
    return df


def save_table(df: pd.DataFrame, name: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / name, index=False)


def method_label(row: pd.Series) -> str:
    if str(row.get("model_type", "")).startswith("mllm"):
        return "Zero-shot prompting"
    return METHOD_DISPLAY.get(row.get("method"), str(row.get("method")))


def add_reliability_scores(summary: pd.DataFrame, metrics: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    scored = summary.copy()
    fairness_cols = [
        c
        for c in metrics.columns
        if c.endswith(("equalized_odds_gap", "accuracy_gap", "auc_gap"))
        and not c.startswith("image_quality_")
        and metrics[c].notna().any()
    ]
    quality_cols = [
        c
        for c in metrics.columns
        if c.startswith("image_quality_") and c.endswith("_gap") and metrics[c].notna().any()
    ]
    extras = metrics[group_cols].drop_duplicates().copy()
    if fairness_cols:
        fairness = metrics.groupby(group_cols)[fairness_cols].mean().mean(axis=1).rename("fairness_gap")
        extras = extras.merge(fairness.reset_index(), on=group_cols, how="left")
    else:
        extras["fairness_gap"] = np.nan
    if quality_cols:
        quality = metrics.groupby(group_cols)[quality_cols].mean().mean(axis=1).rename("quality_gap")
        extras = extras.merge(quality.reset_index(), on=group_cols, how="left")
    else:
        extras["quality_gap"] = np.nan
    dataset_auc = metrics.groupby([*group_cols, "dataset"], as_index=False)["auc"].mean()
    shift = dataset_auc.groupby(group_cols)["auc"].agg(lambda s: 1.0 - (s.max() - s.min())).rename("shift")
    extras = extras.merge(shift.reset_index(), on=group_cols, how="left")

    fairness_fill = extras["fairness_gap"].median() if extras["fairness_gap"].notna().any() else 0.0
    quality_fill = extras["quality_gap"].median() if extras["quality_gap"].notna().any() else 0.0
    extras["fairness"] = 1.0 - extras["fairness_gap"].fillna(fairness_fill)
    extras["quality"] = 1.0 - extras["quality_gap"].fillna(quality_fill)
    extras[RELIABILITY_METRICS] = extras[RELIABILITY_METRICS].clip(0.0, 1.0)
    return scored.merge(extras[[*group_cols, *RELIABILITY_METRICS]], on=group_cols, how="left")


def build_latex_table(overall: pd.DataFrame, task: pd.DataFrame) -> str:
    reliability = {}
    if RELIABILITY_MATRIX.exists():
        rel = pd.read_csv(RELIABILITY_MATRIX)
        rel = rel.rename(
            columns={
                "age_fairness": "fairness",
                "quality_robustness": "quality",
                "shift_stability": "shift",
            }
        )
        for row in rel.itertuples():
            display = RELIABILITY_DISPLAY_MAP.get(str(row.model_type_display), str(row.model_type_display))
            reliability[display] = {
                "fairness": float(row.fairness),
                "quality": float(row.quality),
                "shift": float(row.shift),
            }
    lines = []
    lines.append("% Auto-generated by make_method_table_figure.py")
    lines.append("\\begin{table*}[t]")
    lines.append("  \\caption{Method-aware mean metrics by model type and task. Higher is better except for ECE.}")
    lines.append("  \\label{tab:model-type-results}")
    lines.append("  \\centering")
    lines.append("  \\scriptsize")
    lines.append("  \\setlength{\\tabcolsep}{5pt}")
    lines.append("  \\begin{tabular}{lccccccccc}")
    lines.append("    \\toprule")
    lines.append("    Task & Model type & Method & Accuracy & AUROC & AUPRC & ECE & Fairness & Quality & Shift \\\\")
    lines.append("    \\midrule")

    task_ordered = task.copy()
    task_ordered["task"] = pd.Categorical(task_ordered["task"], TASK_ORDER, ordered=True)
    task_ordered["model_type"] = pd.Categorical(task_ordered["model_type"], MODEL_TYPE_ORDER, ordered=True)
    task_ordered["method"] = pd.Categorical(task_ordered["method"], METHOD_ORDER, ordered=True)
    task_ordered = task_ordered.sort_values(["task", "model_type", "method"])

    for task_name in TASK_ORDER:
        task_rows = task_ordered[task_ordered["task"] == task_name]
        if task_rows.empty:
            continue
        lines.append("    \\multicolumn{10}{l}{\\textbf{" + TASK_DISPLAY[task_name] + "}} \\\\")
        best_accuracy = task_rows["accuracy"].max()
        best_auc = task_rows["auc"].max()
        best_auprc = task_rows["auprc"].max()
        best_ece = task_rows["ece"].min()
        best_fairness = task_rows["fairness"].max()
        best_quality = task_rows["quality"].max()
        best_shift = task_rows["shift"].max()
        for row in task_rows.itertuples():
            row_series = pd.Series(row._asdict())
            lines.append(
                "    \\quad  & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                    row.model_type_display,
                    method_label(row_series),
                    format_metric_cell(float(row.accuracy), float(best_accuracy), True),
                    format_metric_cell(float(row.auc), float(best_auc), True),
                    format_metric_cell(float(row.auprc), float(best_auprc), True),
                    format_metric_cell(float(row.ece), float(best_ece), False),
                    format_metric_cell(float(row.fairness), float(best_fairness), True) if np.isfinite(row.fairness) else "--",
                    format_metric_cell(float(row.quality), float(best_quality), True) if np.isfinite(row.quality) else "--",
                    format_metric_cell(float(row.shift), float(best_shift), True) if np.isfinite(row.shift) else "--",
                )
            )
        lines.append("    \\midrule")

    overall_ordered = overall.copy()
    overall_ordered["model_type"] = pd.Categorical(overall_ordered["model_type"], MODEL_TYPE_ORDER, ordered=True)
    overall_ordered["method"] = pd.Categorical(overall_ordered["method"], METHOD_ORDER, ordered=True)
    overall_ordered = overall_ordered.sort_values(["model_type", "method"])

    lines.append("    \\multicolumn{10}{l}{\\textbf{Overall}} \\\\")
    best_accuracy = overall_ordered["accuracy"].max()
    best_auc = overall_ordered["auc"].max()
    best_auprc = overall_ordered["auprc"].max()
    best_ece = overall_ordered["ece"].min()
    best_fairness = overall_ordered["fairness"].max()
    best_quality = overall_ordered["quality"].max()
    best_shift = overall_ordered["shift"].max()
    for row in overall_ordered.itertuples():
        row_series = pd.Series(row._asdict())
        lines.append(
            "    \\quad  & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                row.model_type_display,
                method_label(row_series),
                format_metric_cell(float(row.accuracy), float(best_accuracy), True),
                format_metric_cell(float(row.auc), float(best_auc), True),
                format_metric_cell(float(row.auprc), float(best_auprc), True),
                format_metric_cell(float(row.ece), float(best_ece), False),
                format_metric_cell(float(row.fairness), float(best_fairness), True) if np.isfinite(row.fairness) else "--",
                format_metric_cell(float(row.quality), float(best_quality), True) if np.isfinite(row.quality) else "--",
                format_metric_cell(float(row.shift), float(best_shift), True) if np.isfinite(row.shift) else "--",
            )
        )
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def build_method_tables(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = (
        metrics.groupby(["model_type", "model_type_display", "method", "method_display"], as_index=False)[METRICS]
        .mean()
        .sort_values(["model_type", "method"])
    )

    task = (
        metrics.groupby(
            ["task", "task_display", "model_type", "model_type_display", "method", "method_display"],
            as_index=False,
        )[METRICS]
        .mean()
        .sort_values(["task", "model_type", "method"])
    )

    overall = add_reliability_scores(overall, metrics, ["model_type", "method"])
    task = add_reliability_scores(task, metrics, ["task", "model_type", "method"])

    return overall, task


def write_latex_table(overall: pd.DataFrame, task: pd.DataFrame) -> None:
    table_tex = build_latex_table(overall, task)
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "model_type_method_task_table.tex").write_text(table_tex, encoding="utf-8")


def main() -> None:
    metrics = prepare_metrics()
    overall, task = build_method_tables(metrics)
    save_table(overall, "model_type_method_summary.csv")
    save_table(task, "task_model_type_method_summary.csv")
    write_latex_table(overall, task)
    print("Wrote model_type_method_summary.csv, task_model_type_method_summary.csv, model_type_method_task_table.tex")


if __name__ == "__main__":
    main()
