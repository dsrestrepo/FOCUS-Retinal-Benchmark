#!/usr/bin/env python3
"""Generate paper figures and summary tables for the FOCUS benchmark.

The available result bundle contains aggregate metrics, subgroup gaps, and
coverage metadata. It does not contain raw predictions, so matrix figures here
summarize model/dataset/task performance rather than reconstructing confusion
matrices.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results" / "analysis"
PAPER_TABLES = ROOT / "paper" / "tables"
PAPER_FIGURES = ROOT / "paper" / "figures"
DASHBOARD_DATA = ROOT / "dashboard" / "data"

TASK_ORDER = ["binary_dr", "referable_dr", "glaucoma"]
FAMILY_ORDER = ["cv", "vlm", "mllm"]
MODEL_TYPE_ORDER = [
    "cv_general",
    "cv_ophthalmo",
    "vlm_general",
    "vlm_ophthalmo",
    "mllm_general",
    "mllm_medical",
]
DATASET_ORDER = [
    "brset",
    "mbrset",
    "idrid",
    "messidor_2",
    "rfmid",
    "rfmid_2",
    "jsiec1000",
    "refuge",
    "g1020",
    "papila",
]
RELIABILITY_AXES = [
    "auc",
    "auprc",
    "accuracy",
    "calibration",
    "age_fairness",
    "quality_robustness",
]

DATASET_META = {
    "brset": {"country": "Brazil", "region": "Latin America", "camera": "standard fundus"},
    "mbrset": {"country": "Brazil", "region": "Latin America", "camera": "mobile fundus"},
    "papila": {"country": "Spain", "region": "Europe", "camera": "standard fundus"},
    "rfmid": {"country": "India", "region": "South Asia", "camera": "standard fundus"},
    "rfmid_2": {"country": "India", "region": "South Asia", "camera": "standard fundus"},
    "idrid": {"country": "India", "region": "South Asia", "camera": "standard fundus"},
    "messidor_2": {"country": "France", "region": "Europe", "camera": "standard fundus"},
    "refuge": {"country": "China", "region": "East Asia", "camera": "standard fundus"},
    "g1020": {"country": "Germany", "region": "Europe", "camera": "standard fundus"},
    "jsiec1000": {"country": "China", "region": "East Asia", "camera": "standard fundus"},
}
DATASET_DISPLAY = {
    "brset": "BRSET",
    "mbrset": "mBRSET",
    "papila": "PAPILA",
    "rfmid": "RFMiD",
    "rfmid_2": "RFMiD 2",
    "idrid": "IDRiD",
    "messidor_2": "Messidor-2",
    "refuge": "REFUGE",
    "g1020": "G1020",
    "jsiec1000": "JSIEC1000",
}
TASK_DISPLAY = {
    "binary_dr": "Binary DR",
    "referable_dr": "Referable DR",
    "glaucoma": "Glaucoma",
}
MODEL_TYPE_DISPLAY = {
    "cv_general": "General Vision Encoder Model (VM)",
    "cv_ophthalmo": "Ophthalmic Vision Encoder Model (VM)",
    "vlm_general": "General Dual Encoder Vision Language Model (VLM-encoders)",
    "vlm_ophthalmo": "Ophthalmic Dual Encoder Vision Language Model (VLM-encoders)",
    "mllm_general": "General Multimodal LLM (MLLMs)",
    "mllm_medical": "Medical Multimodal LLM (MLLMs)",
}
FAMILY_DISPLAY = {
    "cv": "Vision Encoder Model (VM)",
    "vlm": "Dual Encoder Vision Language Model (VLM-encoders)",
    "mllm": "Multimodal LLM (MLLMs)",
}
METHOD_DISPLAY = {
    "base": "Base",
    "linear_probing": "Linear probe",
    "zero_shot": "Zero-shot",
}
PALETTE = {
    "cv": "#4477AA",
    "vlm": "#228833",
    "mllm": "#CC6677",
    "cv_general": "#4477AA",
    "cv_ophthalmo": "#66CCEE",
    "vlm_general": "#88CCAA",
    "vlm_ophthalmo": "#117733",
    "mllm_general": "#DD8899",
    "mllm_medical": "#AA3377",
    "base": "#CC6677",
    "linear_probing": "#4477AA",
    "zero_shot": "#DDCC77",
    "general": "#88CCEE",
    "domain_specialized": "#AA3377",
    "standard fundus": "#4477AA",
    "mobile fundus": "#CC6677",
}


def display_model(name: str) -> str:
    aliases = {
        "facebook/dinov2-large": "DINOv2-L",
        "facebook/dinov3-vitl16-pretrain-lvd1689m": "DINOv3-L",
        "google/vit-large-patch16-224": "ViT-L/16",
        "VisionFM-Fundus": "VisionFM",
        "YukunZhou/RETFound_dinov2_meh": "RETFound DINOv2 MEH",
        "YukunZhou/RETFound_dinov2_shanghai": "RETFound DINOv2 SH",
        "YukunZhou/RETFound_mae_meh": "RETFound MAE MEH",
        "YukunZhou/RETFound_mae_natureCFP": "RETFound CFP",
        "YukunZhou/RETFound_mae_natureOCT": "RETFound OCT",
        "YukunZhou/RETFound_mae_shanghai": "RETFound MAE SH",
        "Qwen/Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
        "google/gemma-3-27b-it": "Gemma-3-27B",
        "llava-hf/llama3-llava-next-8b-hf": "LLaVA-NeXT-8B",
        "google/medgemma-1.5-4b-it": "MedGemma 1.5 4B",
        "google/medgemma-27b-it": "MedGemma 27B",
        "google/medgemma-4b-it": "MedGemma 4B",
        "google/medsiglip-448": "MedSigLIP",
        "google/siglip2-base-patch16-224": "SigLIP2-B",
        "openai/clip-vit-base-patch32": "CLIP-B/32",
        "EyeCLIP": "EyeCLIP",
        "FLAIR": "FLAIR",
        "RET-CLIP": "RET-CLIP",
    }
    return aliases.get(name, name.split("/")[-1])


def mllm_parameter_billions(name: str) -> float | None:
    aliases = {
        "Qwen/Qwen3-VL-8B-Instruct": 8.0,
        "google/gemma-3-27b-it": 27.0,
        "llava-hf/llama3-llava-next-8b-hf": 8.0,
        "google/medgemma-1.5-4b-it": 4.0,
        "google/medgemma-27b-it": 27.0,
        "google/medgemma-4b-it": 4.0,
    }
    return aliases.get(name)


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["country"] = df["dataset"].map(lambda x: DATASET_META[x]["country"])
    df["region"] = df["dataset"].map(lambda x: DATASET_META[x]["region"])
    df["camera"] = df["dataset"].map(lambda x: DATASET_META[x]["camera"])
    df["domain"] = np.where(df["model_type"].str.endswith(("general",)), "general", "domain_specialized")
    df["model_display"] = df["model"].map(display_model)
    df["dataset_display"] = df["dataset"].map(DATASET_DISPLAY)
    df["task_display"] = df["task"].map(TASK_DISPLAY)
    df["model_type_display"] = df["model_type"].map(MODEL_TYPE_DISPLAY)
    df["family_display"] = df["family"].map(FAMILY_DISPLAY)
    df["method_display"] = df["method"].map(METHOD_DISPLAY).fillna(df["method"])
    df["model_method"] = df["model_display"] + " (" + df["method_display"] + ")"
    df["model_method_short"] = df["model_display"]
    return df


def rounded(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric = df.select_dtypes(include="number").columns
    df[numeric] = df[numeric].round(3)
    return df


def save_table(df: pd.DataFrame, name: str) -> None:
    for root in [PAPER_TABLES, DASHBOARD_DATA]:
        root.mkdir(parents=True, exist_ok=True)
        df.to_csv(root / name, index=False)


def savefig(fig: plt.Figure, name: str, *, pdf: bool = False) -> None:
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    stem = Path(name).with_suffix("")
    suffix = ".pdf" if pdf else Path(name).suffix or ".png"
    fig.savefig(PAPER_FIGURES / f"{stem.name}{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def order_existing(values: list[str], order: list[str]) -> list[str]:
    present = set(values)
    ordered = [v for v in order if v in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def annotate_heatmap(ax: plt.Axes, data: pd.DataFrame, fmt: str = ".2f") -> None:
    for y, row in enumerate(data.index):
        for x, col in enumerate(data.columns):
            val = data.loc[row, col]
            if pd.isna(val):
                continue
            color = "white" if float(val) > 0.74 else "#222222"
            ax.text(x + 0.5, y + 0.5, format(float(val), fmt), ha="center", va="center", fontsize=10.5, fontweight="bold", color=color)


def plot_reliability_matrix(metrics: pd.DataFrame) -> None:
    df = metrics.groupby(["model_type", "model_type_display"], as_index=False).agg(
        accuracy=("accuracy", "mean"),
        auc=("auc", "mean"),
        auprc=("auprc", "mean"),
        calibration=("ece", lambda s: 1 - s.mean()),
        age_fairness=("age_equalized_odds_gap", lambda s: 1 - s.mean()),
        quality_robustness=("image_quality_accuracy_gap", lambda s: 1 - s.mean()),
        mobile_robustness=("auc", "mean"),
    )
    mobile = (
        metrics.groupby(["model_type", "camera"])["auc"]
        .mean()
        .unstack()
        .assign(mobile_robustness=lambda x: 1 - (x.get("standard fundus") - x.get("mobile fundus")).clip(lower=0))
    )
    df = df.drop(columns=["mobile_robustness"]).merge(mobile[["mobile_robustness"]], left_on="model_type", right_index=True, how="left")
    ranges = metrics.groupby(["model_type", "dataset"])["auc"].mean().groupby("model_type").agg(lambda s: 1 - (s.max() - s.min()))
    df = df.merge(ranges.rename("shift_stability"), left_on="model_type", right_index=True, how="left")
    df["model_type"] = pd.Categorical(df["model_type"], MODEL_TYPE_ORDER, ordered=True)
    df = df.sort_values("model_type")
    cols = ["accuracy", "auc", "auprc", "calibration", "age_fairness", "quality_robustness", "mobile_robustness", "shift_stability"]
    labels = ["Accuracy", "AUROC", "AUPRC", "1 - ECE", "1 - age EO gap", "1 - quality gap", "1 - mobile drop", "1 - dataset range"]
    mat = df.set_index("model_type_display")[cols]
    fig, ax = plt.subplots(figsize=(10.4, 3.9))
    sns.heatmap(mat, ax=ax, cmap="viridis", vmin=0.45, vmax=1.0, cbar_kws={"label": "Higher is better"}, linewidths=0.8, linecolor="white")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.tick_params(axis="y", rotation=0)
    annotate_heatmap(ax, mat, ".2f")
    savefig(fig, "model_type_reliability_matrix.png")
    save_table(rounded(df[["model_type_display", *cols]]), "model_type_reliability_matrix.csv")


def plot_task_leaderboards(metrics: pd.DataFrame) -> None:
    grouped = (
        metrics.groupby(["task", "family", "model_type", "model_display", "method_display", "model_method"], as_index=False)
        .agg(auc=("auc", "mean"), auprc=("auprc", "mean"), ece=("ece", "mean"))
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.3), sharex=True)
    for ax, task in zip(axes, TASK_ORDER):
        subset = grouped[grouped["task"] == task].sort_values("auc", ascending=False).head(12).iloc[::-1]
        colors = subset["family"].map(PALETTE).tolist()
        ax.barh(subset["model_method"], subset["auc"], color=colors, height=0.72)
        ax.set_title(TASK_DISPLAY[task])
        ax.set_xlabel("Mean AUROC")
        ax.set_xlim(0.35, 1.0)
        ax.grid(axis="x", color="#E7E7E7")
        ax.grid(axis="y", visible=False)
        for y, value in enumerate(subset["auc"]):
            ax.text(value + 0.01, y, f"{value:.2f}", va="center", fontsize=10, fontweight="bold")
    handles = [Line2D([0], [0], marker="s", linestyle="", color=PALETTE[f], label=FAMILY_DISPLAY[f], markersize=8) for f in FAMILY_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    savefig(fig, "task_arena_leaderboards.png")


def plot_model_dataset_heatmaps(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 7.8), sharex=False)
    for ax, family in zip(axes, FAMILY_ORDER):
        subset = metrics[metrics["family"] == family].copy()
        model_order = (
            subset.groupby("model_method")["auc"].mean().sort_values(ascending=False).index.tolist()
        )
        model_order = model_order[:14]
        table = subset[subset["model_method"].isin(model_order)].pivot_table(
            index="model_method",
            columns="dataset",
            values="auc",
            aggfunc="mean",
        )
        table = table.reindex(model_order).reindex(columns=[d for d in DATASET_ORDER if d in table.columns])
        table = table.rename(columns=DATASET_DISPLAY)
        sns.heatmap(
            table,
            ax=ax,
            cmap="mako",
            vmin=0.35,
            vmax=0.95,
            linewidths=0.45,
            linecolor="white",
            cbar=family == "mllm",
            cbar_kws={"label": "Mean AUROC"},
        )
        ax.set_title(FAMILY_DISPLAY[family])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", labelsize=7.1)
    savefig(fig, "model_dataset_auc_heatmap.png")


def plot_fine_grained_performance(metrics: pd.DataFrame) -> None:
    grouped = (
        metrics.groupby(["task", "model_type", "model_type_display", "model_display", "method_display", "model_method"], as_index=False)
        .agg(auc=("auc", "mean"), ece=("ece", "mean"))
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 7.2), sharex=True)
    for ax, task in zip(axes, TASK_ORDER):
        subset = grouped[grouped["task"] == task].sort_values("auc", ascending=False).head(18).iloc[::-1]
        ax.barh(subset["model_method"], subset["auc"], color=subset["model_type"].map(PALETTE), height=0.68)
        ax.set_title(TASK_DISPLAY[task])
        ax.set_xlim(0.35, 1.0)
        ax.set_xlabel("Mean AUROC")
        ax.grid(axis="x", color="#E8E8E8")
        ax.grid(axis="y", visible=False)
    handles = [Line2D([0], [0], marker="s", linestyle="", color=PALETTE[t], label=MODEL_TYPE_DISPLAY[t], markersize=8) for t in MODEL_TYPE_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    savefig(fig, "fine_grained_model_performance.pdf", pdf=True)


def prepare_reliability(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics.groupby(["family", "model_type", "model_display", "method_display", "model_method"], as_index=False).agg(
        accuracy=("accuracy", "mean"),
        auc=("auc", "mean"),
        auprc=("auprc", "mean"),
        ece=("ece", "mean"),
        age_equalized_odds_gap=("age_equalized_odds_gap", "mean"),
        image_quality_accuracy_gap=("image_quality_accuracy_gap", "mean"),
    )
    grouped["calibration"] = 1 - grouped["ece"]
    grouped["age_fairness"] = 1 - grouped["age_equalized_odds_gap"].fillna(grouped["age_equalized_odds_gap"].mean())
    grouped["quality_robustness"] = 1 - grouped["image_quality_accuracy_gap"].fillna(grouped["image_quality_accuracy_gap"].mean())
    for col in RELIABILITY_AXES:
        grouped[col] = grouped[col].clip(0, 1)
    return grouped


def radar_axis(ax: plt.Axes, rows: pd.DataFrame, title: str) -> None:
    labels = ["AUROC", "AUPRC", "Accuracy", "1-ECE", "Age fairness", "Quality robust."]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0.35, 1.0)
    ax.set_yticks([0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.50", "0.75", "1.00"], fontsize=6)
    ax.grid(color="#D9D9D9")
    for _, row in rows.iterrows():
        vals = [row[c] for c in RELIABILITY_AXES]
        vals += vals[:1]
        color = PALETTE[row["model_type"]]
        ax.plot(angles, vals, lw=1.3, color=color, alpha=0.95)
        ax.fill(angles, vals, color=color, alpha=0.08)
    ax.set_title(title, pad=14)


def plot_reliability_spiders(metrics: pd.DataFrame) -> None:
    reliability = prepare_reliability(metrics)
    save_table(rounded(reliability), "model_reliability_radar_values.csv")

    top = reliability.sort_values("auc", ascending=False).head(6)
    fig = plt.figure(figsize=(6.5, 5.6))
    ax = fig.add_subplot(111, projection="polar")
    radar_axis(ax, top, "Top recovered configurations")
    handles = [Line2D([0], [0], color=PALETTE[r.model_type], lw=2, label=r.model_method) for r in top.itertuples()]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.04))
    savefig(fig, "top_model_reliability_spider.png")

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8), subplot_kw={"projection": "polar"})
    for ax, family in zip(axes, FAMILY_ORDER):
        rows = reliability[reliability["family"] == family].sort_values("auc", ascending=False).head(5)
        radar_axis(ax, rows, FAMILY_DISPLAY[family])
        handles = [Line2D([0], [0], color=PALETTE[r.model_type], lw=2, label=r.model_method) for r in rows.itertuples()]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.34), fontsize=6.2, ncol=1)
    savefig(fig, "arena_spider_grid.png")

    for family in FAMILY_ORDER:
        rows = reliability[reliability["family"] == family].sort_values("auc", ascending=False).head(5)
        fig = plt.figure(figsize=(5.8, 5.1))
        ax = fig.add_subplot(111, projection="polar")
        radar_axis(ax, rows, f"{FAMILY_DISPLAY[family]} top configurations")
        handles = [Line2D([0], [0], color=PALETTE[r.model_type], lw=2, label=r.model_method) for r in rows.itertuples()]
        fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=6.7, bbox_to_anchor=(0.5, -0.05))
        savefig(fig, f"{family}_top_model_spider.png")


def plot_domain_method_shift_panels(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2))
    ax = axes[0, 0]
    subset = metrics.groupby(["model_type", "model_type_display"], as_index=False)["auc"].mean()
    subset["model_type"] = pd.Categorical(subset["model_type"], MODEL_TYPE_ORDER, ordered=True)
    subset = subset.sort_values("model_type")
    ax.bar(subset["model_type_display"], subset["auc"], color=subset["model_type"].map(PALETTE))
    ax.set_ylabel("Mean AUROC")
    ax.set_ylim(0.55, 0.86)
    ax.set_title("Model taxonomy")
    ax.tick_params(axis="x", rotation=35)

    ax = axes[0, 1]
    method = metrics.groupby(["family", "method_display"], as_index=False)["auc"].mean()
    sns.barplot(
        data=method,
        x="family",
        y="auc",
        hue="method_display",
        order=FAMILY_ORDER,
        palette=[PALETTE["base"], PALETTE["linear_probing"], PALETTE["zero_shot"]],
        ax=ax,
    )
    ax.set_xticks(range(len(FAMILY_ORDER)), [FAMILY_DISPLAY[f] for f in FAMILY_ORDER])
    ax.set_xlabel("")
    ax.set_ylabel("Mean AUROC")
    ax.set_title("Interface x adaptation")

    ax = axes[1, 0]
    camera = metrics.groupby(["model_type", "model_type_display", "camera"], as_index=False)["auc"].mean()
    cam = camera.pivot(index=["model_type", "model_type_display"], columns="camera", values="auc").reset_index()
    cam["mobile_drop"] = (cam.get("standard fundus") - cam.get("mobile fundus")).clip(lower=0)
    cam["model_type"] = pd.Categorical(cam["model_type"], MODEL_TYPE_ORDER, ordered=True)
    cam = cam.sort_values("model_type")
    ax.bar(cam["model_type_display"], cam["mobile_drop"], color=cam["model_type"].map(PALETTE))
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_ylabel("Standard - mobile AUROC")
    ax.set_title("Mobile-camera sensitivity")
    ax.tick_params(axis="x", rotation=35)

    ax = axes[1, 1]
    ranges = metrics.groupby(["model_type", "model_type_display", "dataset"])["auc"].mean().groupby(["model_type", "model_type_display"]).agg(["min", "max"]).reset_index()
    ranges["range"] = ranges["max"] - ranges["min"]
    ranges["model_type"] = pd.Categorical(ranges["model_type"], MODEL_TYPE_ORDER, ordered=True)
    ranges = ranges.sort_values("model_type")
    ax.bar(ranges["model_type_display"], ranges["range"], color=ranges["model_type"].map(PALETTE))
    ax.set_ylabel("Dataset AUROC range")
    ax.set_title("Dataset-shift proxy")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    savefig(fig, "domain_method_shift_panels.png")


def plot_vlm_adaptation(metrics: pd.DataFrame) -> None:
    df = metrics[metrics["family"] == "vlm"].groupby(["task", "model_display", "method_display"], as_index=False)["auc"].mean()
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.8), sharex=True)
    for ax, task in zip(axes, TASK_ORDER):
        subset = df[df["task"] == task].copy()
        order = subset.groupby("model_display")["auc"].max().sort_values(ascending=False).index.tolist()
        sns.barplot(data=subset, x="auc", y="model_display", hue="method_display", order=order, ax=ax, palette={"Linear probe": PALETTE["linear_probing"], "Zero-shot": PALETTE["zero_shot"]})
        ax.set_title(TASK_DISPLAY[task])
        ax.set_xlabel("Mean AUROC")
        ax.set_ylabel("")
        ax.set_xlim(0.35, 1.0)
        if ax is not axes[-1]:
            ax.legend_.remove()
    axes[-1].legend(title="")
    savefig(fig, "vlm_adaptation_comparison.pdf", pdf=True)


def plot_dataset_family_metric_panels(metrics: pd.DataFrame) -> None:
    grouped = metrics.groupby(["dataset", "dataset_display", "family"], as_index=False)[["auc", "auprc", "accuracy", "ece"]].mean()
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 7.4), sharex=True)
    for ax, metric, label in zip(axes.ravel(), ["auc", "auprc", "accuracy", "ece"], ["AUROC", "AUPRC", "Accuracy", "ECE"]):
        sns.lineplot(
            data=grouped,
            x="dataset_display",
            y=metric,
            hue="family",
            marker="o",
            hue_order=FAMILY_ORDER,
            palette=PALETTE,
            ax=ax,
        )
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=40)
        if ax is not axes[0, 0]:
            ax.legend_.remove()
        else:
            handles, labels_ = ax.get_legend_handles_labels()
            ax.legend(handles, [FAMILY_DISPLAY.get(x, x) for x in labels_], title="")
    savefig(fig, "dataset_family_metric_panels.png")


def plot_mobile_sensitivity(metrics: pd.DataFrame) -> None:
    camera = metrics.groupby(["model_type", "model_type_display", "camera"], as_index=False)["auc"].mean()
    table = camera.pivot(index=["model_type", "model_type_display"], columns="camera", values="auc").reset_index()
    table["mobile_drop"] = table.get("standard fundus") - table.get("mobile fundus")
    table["model_type"] = pd.Categorical(table["model_type"], MODEL_TYPE_ORDER, ordered=True)
    table = table.sort_values("model_type")
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    ax.barh(table["model_type_display"], table["mobile_drop"], color=table["model_type"].map(PALETTE))
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_xlabel("Standard fundus AUROC - mBRSET AUROC")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#E5E5E5")
    ax.grid(axis="y", visible=False)
    savefig(fig, "mobile_camera_sensitivity.png")
    save_table(rounded(table), "mobile_sensitivity_by_model_type.csv")


def plot_auc_ece_tradeoff(metrics: pd.DataFrame) -> None:
    df = metrics.groupby(["model_type", "model_type_display", "model_display", "family"], as_index=False).agg(auc=("auc", "mean"), ece=("ece", "mean"), auprc=("auprc", "mean"))
    fig, ax = plt.subplots(figsize=(8.0, 5.3))
    for model_type, subset in df.groupby("model_type"):
        ax.scatter(subset["ece"], subset["auc"], s=90, color=PALETTE[model_type], label=MODEL_TYPE_DISPLAY[model_type], alpha=0.88, edgecolor="white", linewidth=0.7)
        for row in subset.itertuples():
            if row.auc >= df["auc"].quantile(0.77) or row.ece >= df["ece"].quantile(0.85):
                ax.text(row.ece + 0.004, row.auc, row.model_display, fontsize=6.5)
    ax.set_xlabel("Expected Calibration Error (lower is better)")
    ax.set_ylabel("Mean AUROC")
    ax.set_xlim(0.08, max(0.36, df["ece"].max() + 0.03))
    ax.set_ylim(0.50, 1.00)
    ax.legend(ncol=2, fontsize=7, loc="lower left")
    ax.grid(color="#E8E8E8")
    savefig(fig, "auc_ece_tradeoff.png")


def plot_mllm_size_tradeoff(metrics: pd.DataFrame) -> None:
    df = metrics[metrics["family"] == "mllm"].groupby(["model", "model_display", "model_type", "model_type_display"], as_index=False).agg(auc=("auc", "mean"), ece=("ece", "mean"), auprc=("auprc", "mean"))
    df["parameters_b"] = df["model"].map(mllm_parameter_billions)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for model_type, subset in df.groupby("model_type"):
        ax.scatter(
            subset["ece"],
            subset["auc"],
            s=80 + subset["parameters_b"].fillna(6) * 18,
            color=PALETTE[model_type],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.8,
            label=MODEL_TYPE_DISPLAY[model_type],
        )
        for row in subset.itertuples():
            ax.text(row.ece + 0.004, row.auc, f"{row.model_display}\n{row.parameters_b:.0f}B", fontsize=7)
    ax.set_xlabel("Expected Calibration Error (lower is better)")
    ax.set_ylabel("Mean AUROC")
    ax.set_xlim(0.14, max(0.40, df["ece"].max() + 0.04))
    ax.set_ylim(0.55, 0.88)
    ax.legend(loc="lower left")
    fig.savefig(PAPER_FIGURES / "mllm_size_domain_tradeoff.png", bbox_inches="tight", facecolor="white")
    fig.savefig(PAPER_FIGURES / "mllm_size_domain_tradeoff.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_dataset_shift_sensitivity(metrics: pd.DataFrame) -> None:
    ranges = (
        metrics.groupby(["model_type", "model_type_display", "dataset"])["auc"]
        .mean()
        .groupby(["model_type", "model_type_display"])
        .agg(["min", "mean", "max"])
        .reset_index()
    )
    ranges["range"] = ranges["max"] - ranges["min"]
    ranges["model_type"] = pd.Categorical(ranges["model_type"], MODEL_TYPE_ORDER, ordered=True)
    ranges = ranges.sort_values("model_type")
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.barh(ranges["model_type_display"], ranges["range"], color=ranges["model_type"].map(PALETTE), height=0.72)
    for y, row in enumerate(ranges.itertuples()):
        ax.text(row.range + 0.006, y, f"{row.range:.2f}", va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Range of mean AUROC across datasets")
    ax.set_ylabel("")
    ax.set_xlim(0, max(0.28, ranges["range"].max() + 0.06))
    ax.grid(axis="x", color="#E7E7E7")
    ax.grid(axis="y", visible=False)
    savefig(fig, "dataset_shift_sensitivity.png")
    save_table(rounded(ranges), "dataset_shift_sensitivity.csv")


def plot_calibration_family_grids(metrics: pd.DataFrame) -> None:
    for family in FAMILY_ORDER:
        subset = metrics[metrics["family"] == family].copy()
        model_order = subset.groupby("model_method")["auc"].mean().sort_values(ascending=False).index.tolist()
        fig, axes = plt.subplots(3, 1, figsize=(12.6, 10.4), sharex=False)
        for ax, task in zip(axes, TASK_ORDER):
            task_rows = subset[subset["task"] == task]
            table = task_rows.pivot_table(index="model_method", columns="dataset", values="ece", aggfunc="mean")
            table = table.reindex([m for m in model_order if m in table.index])
            table = table.reindex(columns=[d for d in DATASET_ORDER if d in table.columns])
            table = table.rename(columns=DATASET_DISPLAY)
            sns.heatmap(
                table,
                ax=ax,
                cmap="rocket_r",
                vmin=0.05,
                vmax=0.42,
                linewidths=0.5,
                linecolor="white",
                cbar=ax is axes[-1],
                cbar_kws={"label": "ECE"},
            )
            ax.set_title(TASK_DISPLAY[task], loc="left")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelsize=7)
            ax.tick_params(axis="x", rotation=38)
        savefig(fig, f"calibration_{family}_grid.pdf", pdf=True)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    grouped = metrics.groupby(["family", "task", "dataset_display"], as_index=False)["ece"].mean()
    for ax, family in zip(axes, FAMILY_ORDER):
        table = grouped[grouped["family"] == family].pivot(index="task", columns="dataset_display", values="ece")
        table = table.reindex(TASK_ORDER).rename(index=TASK_DISPLAY)
        sns.heatmap(table, ax=ax, cmap="rocket_r", vmin=0.05, vmax=0.35, linewidths=0.6, linecolor="white", cbar=family == "mllm", cbar_kws={"label": "ECE"})
        ax.set_title(FAMILY_DISPLAY[family])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45)
    savefig(fig, "calibration_model_type_grids.pdf", pdf=True)


def plot_method_task_metric_panels(metrics: pd.DataFrame) -> None:
    df = metrics.groupby(["family", "method_display", "task_display"], as_index=False)[["auc", "auprc", "ece"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    for ax, metric, label in zip(axes, ["auc", "auprc", "ece"], ["AUROC", "AUPRC", "ECE"]):
        sns.barplot(data=df, x="task_display", y=metric, hue="method_display", ax=ax, palette=[PALETTE["base"], PALETTE["linear_probing"], PALETTE["zero_shot"]])
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=20)
        if ax is not axes[-1]:
            ax.legend_.remove()
    axes[-1].legend(title="")
    savefig(fig, "method_task_metric_panels.png")


def copy_existing_analysis_figures() -> None:
    mapping = {
        ANALYSIS / "plots" / "performance" / "heatmap_auc.png": PAPER_FIGURES / "model_type_task_auc_heatmap.png",
        ANALYSIS / "plots" / "performance" / "bar_auc.png": PAPER_FIGURES / "model_level_auc_leaderboard.png",
    }
    for src, dst in mapping.items():
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)


def write_additional_tables(metrics: pd.DataFrame) -> None:
    save_table(rounded(metrics.groupby(["family", "model_type"], as_index=False)[["accuracy", "auc", "auprc", "ece"]].mean()), "model_type_summary.csv")
    save_table(rounded(metrics.groupby(["dataset", "task"], as_index=False)[["accuracy", "auc", "auprc", "ece"]].mean()), "dataset_task_auc.csv")
    save_table(rounded(metrics.groupby(["model_display", "family", "model_type"], as_index=False)[["accuracy", "auc", "auprc", "ece"]].mean().sort_values("auc", ascending=False)), "model_level_summary.csv")
    med = metrics.groupby(["family", "domain"], as_index=False)[["accuracy", "auc", "auprc", "ece"]].mean()
    save_table(rounded(med), "medical_vs_general_summary.csv")

    dataset_summary = metrics.groupby(["dataset", "country", "camera"], as_index=False).agg(
        runs=("auc", "size"),
        models=("model", "nunique"),
        auc=("auc", "mean"),
        auprc=("auprc", "mean"),
        ece=("ece", "mean"),
    )
    save_table(rounded(dataset_summary.sort_values("auc", ascending=False)), "dataset_summary.csv")

    coverage = metrics.groupby(["dataset", "task"], as_index=False).agg(
        runs=("auc", "size"),
        families=("family", "nunique"),
        model_types=("model_type", "nunique"),
        models=("model", "nunique"),
        mean_auc=("auc", "mean"),
        mean_auprc=("auprc", "mean"),
        mean_ece=("ece", "mean"),
    )
    save_table(rounded(coverage.sort_values(["dataset", "task"])), "dataset_task_coverage.csv")

    dataset_family = metrics.groupby(["dataset_display", "family"], as_index=False)[
        ["auc", "auprc", "accuracy", "ece"]
    ].mean()
    save_table(rounded(dataset_family.sort_values(["dataset_display", "family"])), "dataset_family_metric_summary.csv")

    model_dataset = metrics.pivot_table(
        index=["family", "model_type", "model_display", "method", "model_method"],
        columns="dataset",
        values="auc",
        aggfunc="mean",
    ).reset_index()
    dataset_cols = [dataset for dataset in DATASET_ORDER if dataset in model_dataset.columns]
    model_dataset = model_dataset[["family", "model_type", "model_display", "method", *dataset_cols, "model_method"]]
    save_table(rounded(model_dataset.sort_values(["family", "model_type", "model_display", "method"])), "model_dataset_auc.csv")

    save_table(rounded(metrics.groupby(["family"], as_index=False)[["accuracy", "auc", "auprc", "ece"]].mean()), "family_summary.csv")
    save_table(rounded(metrics.groupby(["family", "model_type"], as_index=False)[["ece"]].mean()), "calibration_summary.csv")
    save_table(rounded(metrics.groupby(["country", "family"], as_index=False)[["auc", "auprc", "accuracy", "ece"]].mean()), "country_family_summary.csv")

    axis_rows = []
    axis_groups = [
        ("interface", "vision_only", metrics["family"] == "cv"),
        ("interface", "vision_text", metrics["family"].isin(["vlm", "mllm"])),
        ("generation", "generative", metrics["family"] == "mllm"),
        ("generation", "non_generative", metrics["family"].isin(["cv", "vlm"])),
        ("training_domain", "domain_specialized", metrics["domain"] == "domain_specialized"),
        ("training_domain", "general", metrics["domain"] == "general"),
    ]
    for axis, group, mask in axis_groups:
        row = {"axis": axis, "group": group}
        row.update(metrics.loc[mask, ["accuracy", "auc", "auprc", "ece"]].mean().to_dict())
        axis_rows.append(row)
    save_table(rounded(pd.DataFrame(axis_rows)), "axis_summary.csv")

    coverage_path = ANALYSIS / "result_coverage.csv"
    if coverage_path.exists():
        coverage_df = pd.read_csv(coverage_path)
        coverage_summary = coverage_df.groupby("family", as_index=False).agg(
            found=("found", "sum"),
            expected=("found", "size"),
        )
        coverage_summary["missing"] = coverage_summary["expected"] - coverage_summary["found"]
        coverage_summary["coverage"] = coverage_summary["found"] / coverage_summary["expected"]
        save_table(rounded(coverage_summary), "coverage_summary.csv")

    fairness_cols = [
        "age_demographic_parity_gap",
        "age_equalized_odds_gap",
        "age_auc_gap",
    ]
    available_fairness_cols = [col for col in fairness_cols if col in metrics.columns]
    if available_fairness_cols:
        save_table(
            rounded(metrics.groupby(["family", "model_type"], as_index=False)[available_fairness_cols].mean()),
            "fairness_summary.csv",
        )
    audit = pd.DataFrame(
        [
            {
                "attribute": "age",
                "available": bool(available_fairness_cols and metrics[available_fairness_cols].notna().any().any()),
                "rows": int(metrics[available_fairness_cols].notna().any(axis=1).sum()) if available_fairness_cols else 0,
                "note": "Age subgroup metrics are present when dataset metadata and sample-size thresholds permit.",
            },
            {
                "attribute": "sex",
                "available": False,
                "rows": 0,
                "note": "No sex subgroup rows are present in the recovered result bundle; aggregated rows mark sex as skipped or missing.",
            },
        ]
    )
    save_table(audit, "fairness_attribute_audit.csv")

    gap_cols = [
        "age_demographic_parity_gap",
        "age_equalized_odds_gap",
        "age_accuracy_gap",
        "age_auc_gap",
        "image_quality_accuracy_gap",
        "image_quality_auc_gap",
        "image_quality_auprc_gap",
        "image_quality_ece_gap",
    ]
    available_gap_cols = [col for col in gap_cols if col in metrics.columns]
    save_table(rounded(metrics.groupby("family", as_index=False)[available_gap_cols].mean()), "gap_summary.csv")

    save_table(
        rounded(metrics.groupby(["family", "domain", "method"], as_index=False)[["auc", "auprc", "accuracy", "ece"]].mean()),
        "method_by_domain_summary.csv",
    )
    save_table(
        rounded(metrics.groupby(["family", "method"], as_index=False)[["accuracy", "auc", "auprc", "ece"]].mean()),
        "method_summary.csv",
    )

    method_task = metrics.copy()
    method_task["method_group"] = (
        method_task["family"].str.upper()
        + " "
        + method_task["method"].str.replace("_", " ")
        + "\n"
        + method_task["domain"].str.replace("_", " ")
    )
    save_table(
        rounded(method_task.groupby(["task", "method_group"], as_index=False)[["auc", "auprc", "accuracy", "ece"]].mean()),
        "method_task_metric_summary.csv",
    )

    mllm = metrics[metrics["family"] == "mllm"].copy()
    if not mllm.empty:
        mllm_summary = mllm.groupby(["family", "model_type", "model", "model_display", "method"], as_index=False).agg(
            runs=("auc", "size"),
            datasets=("dataset", "nunique"),
            tasks=("task", "nunique"),
            accuracy=("accuracy", "mean"),
            auc=("auc", "mean"),
            auprc=("auprc", "mean"),
            ece=("ece", "mean"),
            age_equalized_odds_gap=("age_equalized_odds_gap", "mean"),
            age_auc_gap=("age_auc_gap", "mean"),
            image_quality_accuracy_gap=("image_quality_accuracy_gap", "mean"),
            image_quality_auc_gap=("image_quality_auc_gap", "mean"),
        )
        mllm_summary["model_method"] = mllm_summary["model_display"] + " (" + mllm_summary["method"].map(METHOD_DISPLAY).fillna(mllm_summary["method"]) + ")"
        mllm_summary["parameters_b"] = mllm_summary["model"].map(mllm_parameter_billions)
        save_table(rounded(mllm_summary), "mllm_model_summary.csv")

    mobile_pairs = []
    mobile_group = metrics.groupby(["model_type", "model", "method", "task", "camera"], as_index=False).agg(
        auc=("auc", "mean"),
        ece=("ece", "mean"),
    )
    for keys, group in mobile_group.groupby(["model_type", "model", "method", "task"]):
        by_camera = group.set_index("camera")
        if {"mobile fundus", "standard fundus"}.issubset(by_camera.index):
            mobile_auc = float(by_camera.loc["mobile fundus", "auc"])
            standard_auc = float(by_camera.loc["standard fundus", "auc"])
            mobile_ece = float(by_camera.loc["mobile fundus", "ece"])
            standard_ece = float(by_camera.loc["standard fundus", "ece"])
            mobile_pairs.append(
                {
                    "model_type": keys[0],
                    "model": keys[1],
                    "method": keys[2],
                    "task": keys[3],
                    "mobile_auc": mobile_auc,
                    "standard_auc": standard_auc,
                    "mobile_auc_gap": standard_auc - mobile_auc,
                    "mobile_ece": mobile_ece,
                    "standard_ece": standard_ece,
                    "mobile_ece_delta": mobile_ece - standard_ece,
                }
            )
    if mobile_pairs:
        save_table(rounded(pd.DataFrame(mobile_pairs)), "mobile_sensitivity.csv")

    save_table(
        metrics[["family", "model_type", "model", "method"]]
        .drop_duplicates()
        .sort_values(["family", "model_type", "model", "method"]),
        "model_inventory.csv",
    )

    model_task = metrics.pivot_table(
        index=["family", "model_type", "model_display", "method", "model_method"],
        columns="task",
        values="auc",
        aggfunc="mean",
    ).reset_index()
    task_cols = [task for task in TASK_ORDER if task in model_task.columns]
    save_table(
        rounded(model_task[["family", "model_type", "model_display", "method", *task_cols, "model_method"]]),
        "model_task_auc.csv",
    )

    save_table(
        rounded(metrics.groupby(["family", "model_type"], as_index=False)[
            ["accuracy", "auc", "auprc", "ece", "age_equalized_odds_gap", "image_quality_accuracy_gap"]
        ].mean()),
        "reliability_tradeoff.csv",
    )
    save_table(
        rounded(metrics.groupby(["family", "model_type"], as_index=False)[
            ["image_quality_accuracy_gap", "image_quality_auc_gap", "image_quality_ece_gap"]
        ].mean()),
        "robustness_summary.csv",
    )

    task_family = metrics.pivot_table(index="task", columns="family", values="auc", aggfunc="mean").reset_index()
    save_table(rounded(task_family[["task", *[family for family in FAMILY_ORDER if family in task_family.columns]]]), "task_family_auc.csv")
    task_model_type = metrics.pivot_table(index="model_type", columns="task", values="auc", aggfunc="mean").reset_index()
    save_table(rounded(task_model_type[["model_type", *[task for task in TASK_ORDER if task in task_model_type.columns]]]), "task_model_type_auc.csv")

    dataset_by_type = metrics.groupby(["model_type", "dataset"], as_index=False).agg(auc=("auc", "mean"), ece=("ece", "mean"))
    shift_rows = []
    for model_type, group in dataset_by_type.groupby("model_type"):
        shift_rows.append(
            {
                "model_type": model_type,
                "mean_auc": group["auc"].mean(),
                "dataset_auc_std": group["auc"].std(ddof=0),
                "dataset_auc_range": group["auc"].max() - group["auc"].min(),
                "mean_ece": group["ece"].mean(),
            }
        )
    save_table(rounded(pd.DataFrame(shift_rows)), "dataset_shift_by_model_type.csv")


def main() -> None:
    setup_style()
    metrics = pd.read_csv(ANALYSIS / "aggregated_metrics_all.csv")
    metrics = add_metadata(metrics)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)

    write_additional_tables(metrics)
    plot_reliability_matrix(metrics)
    plot_task_leaderboards(metrics)
    plot_model_dataset_heatmaps(metrics)
    plot_fine_grained_performance(metrics)
    plot_reliability_spiders(metrics)
    plot_domain_method_shift_panels(metrics)
    plot_vlm_adaptation(metrics)
    plot_dataset_family_metric_panels(metrics)
    plot_mobile_sensitivity(metrics)
    plot_auc_ece_tradeoff(metrics)
    plot_mllm_size_tradeoff(metrics)
    plot_dataset_shift_sensitivity(metrics)
    plot_calibration_family_grids(metrics)
    plot_method_task_metric_panels(metrics)
    copy_existing_analysis_figures()
    print(f"Wrote figures to {PAPER_FIGURES}")
    print(f"Wrote tables to {PAPER_TABLES}")


if __name__ == "__main__":
    main()
