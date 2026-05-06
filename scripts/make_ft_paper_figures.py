#!/usr/bin/env python3
"""Create paper-facing summaries for SFT language-model fine-tuning.

The FT analysis pipeline writes row-level metrics to results/analysis_ft.
This helper distills those rows into compact CSV tables and, when
matplotlib is installed, a small set of manuscript-ready plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SUMMARY_METRICS = [
    "auc",
    "auprc",
    "accuracy",
    "f1",
    "ece",
    "delta_auc",
    "delta_auprc",
    "delta_accuracy",
    "delta_f1",
    "delta_ece",
]

SIGNIFICANCE_DIR = Path("results/analysis_ft/significance")


def significance_table(path: Path, group_cols: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["ft_method"] == "sft"].copy()
    if df.empty:
        return None
    records = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = {col: value for col, value in zip(group_cols, keys)}
        for _, row in group.iterrows():
            metric = str(row["metric"])
            record[f"{metric}_delta"] = row.get("mean_delta", np.nan)
            record[f"{metric}_ci_low"] = row.get("ci_low", np.nan)
            record[f"{metric}_ci_high"] = row.get("ci_high", np.nan)
            record[f"{metric}_q"] = row.get("q_boot", np.nan)
        records.append(record)
    return pd.DataFrame(records).sort_values(group_cols)


def short_model_name(model_id: str) -> str:
    return str(model_id).split("/")[-1]


def as_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def mean_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    cols = [c for c in SUMMARY_METRICS if c in df.columns]
    return (
        df.groupby(group_cols, dropna=False)[cols]
        .mean()
        .reset_index()
        .sort_values(group_cols)
    )


def write_tables(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    table_df = df.copy()
    table_df = table_df[table_df["ft_method"] == "sft"].copy()
    table_df["train_model_short"] = table_df["train_model"].map(short_model_name)
    table_df["domain"] = np.where(as_bool_series(table_df["is_ood_dataset"]), "OOD", "in_domain")

    domain_sig = significance_table(SIGNIFICANCE_DIR / "domain_significance.csv", ["domain"])
    model_sig = significance_table(SIGNIFICANCE_DIR / "model_significance.csv", ["train_model_short", "domain"])
    task_sig = significance_table(SIGNIFICANCE_DIR / "task_significance.csv", ["train_task", "domain"])

    (domain_sig if domain_sig is not None else mean_summary(table_df, ["domain"])).to_csv(
        output_dir / "lm_ft_domain_summary.csv", index=False
    )
    (model_sig if model_sig is not None else mean_summary(table_df, ["train_model_short", "domain"])).to_csv(
        output_dir / "lm_ft_model_summary.csv", index=False
    )
    (task_sig if task_sig is not None else mean_summary(table_df, ["train_task", "domain"])).to_csv(
        output_dir / "lm_ft_task_summary.csv", index=False
    )


def try_load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local env
        print(f"Skipping FT plots because matplotlib is unavailable: {exc}")
        return None
    return plt


def save_domain_bar(df: pd.DataFrame, figure_dir: Path, plt) -> None:
    summary = mean_summary(df[df["ft_method"] == "sft"].copy(), ["domain"])
    summary = summary.sort_values("domain")

    x = np.arange(len(summary))
    colors = summary["domain"].map({"in_domain": "#2C7FB8", "OOD": "#F03B20"}).fillna("#777777")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x, summary["delta_auc"], color=colors)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "-") for d in summary["domain"]], fontsize=9)
    ax.set_ylabel("Mean AUROC delta vs. base")
    ax.set_title("Language model fine-tuning transfer effect")
    ax.grid(axis="y", color="#E6E6E6")
    fig.tight_layout()
    fig.savefig(figure_dir / "lm_ft_delta_auc_domain.png", dpi=220)
    plt.close(fig)


def save_delta_heatmap(df: pd.DataFrame, figure_dir: Path, plt) -> None:
    plot_df = df[df["ft_method"] == "sft"].copy()
    plot_df["adapter"] = (
        plot_df["train_model"].map(short_model_name)
        + " | "
        + plot_df["train_task"]
    )
    pivot = plot_df.pivot_table(index="adapter", columns="domain", values="delta_auc", aggfunc="mean")
    pivot = pivot.reindex(columns=[c for c in ["in_domain", "OOD"] if c in pivot.columns])
    if pivot.empty:
        return

    height = max(4.5, 0.23 * len(pivot))
    fig, ax = plt.subplots(figsize=(6.8, height))
    data = pivot.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 0.01
    vmax = max(vmax, 0.01)
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([c.replace("_", "-") for c in pivot.columns], fontsize=9)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=6.5)
    ax.set_title("Mean AUROC delta by SFT adapter")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Delta AUROC")
    fig.tight_layout()
    fig.savefig(figure_dir / "lm_ft_delta_auc_heatmap.png", dpi=240)
    plt.close(fig)


def write_plots(df: pd.DataFrame, figure_dir: Path) -> None:
    plt = try_load_matplotlib()
    if plt is None:
        return

    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_df = df.copy()
    plot_df["domain"] = np.where(as_bool_series(plot_df["is_ood_dataset"]), "OOD", "in_domain")
    save_domain_bar(plot_df, figure_dir, plt)
    save_delta_heatmap(plot_df, figure_dir, plt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SFT language-model fine-tuning analysis for the paper")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/analysis_ft/aggregated_metrics.csv"),
        help="FT aggregate metrics CSV written by scripts/analyze_ft_benchmark.py",
    )
    parser.add_argument("--table-dir", type=Path, default=Path("paper/tables"))
    parser.add_argument("--figure-dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()

    if not args.metrics.exists():
        raise FileNotFoundError(f"Missing FT metrics file: {args.metrics}")

    df = pd.read_csv(args.metrics)
    required = {"ft_method", "train_model", "train_task", "is_ood_dataset", "delta_auc"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"FT metrics file is missing columns: {missing}")

    write_tables(df, args.table_dir)
    write_plots(df, args.figure_dir)
    print(f"Wrote SFT paper tables to {args.table_dir}")


if __name__ == "__main__":
    main()
