#!/usr/bin/env python3
"""Generate readable paper figures without requiring matplotlib.

The original deep paper script uses matplotlib/seaborn and produced several
dense figures that are difficult to read at manuscript scale. This script uses
PIL directly to make simpler, wider, paper-facing summaries and FT plots from
the current CSV outputs.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BASE_METRICS = ROOT / "results" / "analysis" / "aggregated_metrics.csv"
FT_METRICS = ROOT / "results" / "analysis_ft" / "aggregated_metrics.csv"
FT_DOMAIN_SIGNIFICANCE = ROOT / "results" / "analysis_ft" / "significance" / "domain_significance.csv"
FT_MODEL_SIGNIFICANCE = ROOT / "results" / "analysis_ft" / "significance" / "model_significance.csv"
FT_TASK_SIGNIFICANCE = ROOT / "results" / "analysis_ft" / "significance" / "task_significance.csv"
RESULTS_DIR = ROOT / "results" / "evals"
FIGURES = ROOT / "paper" / "figures"
TABLES = ROOT / "paper" / "tables"
EXPORT_SCALE = 2
EXPORT_DPI = 300
TRIM_OUTPUTS = {
    "auc_ece_tradeoff.png",
    "arena_spider_grid.png",
    "arena_spider_grid.pdf",
    "mllm_size_domain_tradeoff.png",
    "mllm_size_domain_tradeoff.pdf",
}

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

TASK_DISPLAY = {
    "binary_dr": "Binary DR",
    "referable_dr": "Referable DR",
    "glaucoma": "Glaucoma",
}
FAMILY_DISPLAY = {
    "cv": "Vision Encoder Model (VM)",
    "vlm": "Dual Encoder Vision Language Model (VLM-encoders)",
    "mllm": "Multimodal LLM (MLLMs)",
}
DATASET_DISPLAY = {
    "brset": "BRSET",
    "mbrset": "mBRSET",
    "idrid": "IDRiD",
    "messidor_2": "Messidor-2",
    "rfmid": "RFMiD",
    "rfmid_2": "RFMiD 2",
    "jsiec1000": "JSIEC1000",
    "refuge": "REFUGE",
    "g1020": "G1020",
    "papila": "PAPILA",
}
MODEL_TYPE_DISPLAY = {
    "cv_general": "General Vision Encoder Model (VM)",
    "cv_ophthalmo": "Ophthalmic Vision Encoder Model (VM)",
    "vlm_general": "General Dual Encoder Vision Language Model (VLM-encoders)",
    "vlm_ophthalmo": "Ophthalmic Dual Encoder Vision Language Model (VLM-encoders)",
    "mllm_general": "General Multimodal LLM (MLLMs)",
    "mllm_medical": "Medical Multimodal LLM (MLLMs)",
}
METHOD_DISPLAY = {
    "base": "Zero-shot prompting",
    "linear_probing": "Linear probing",
    "zero_shot": "Zero-shot",
}
METHOD_SHORT = {
    "base": "ZS Prompt",
    "linear_probing": "LP",
    "zero_shot": "ZS",
}
TASK_COLORS = {
    "binary_dr": "#4477AA",
    "referable_dr": "#228833",
    "glaucoma": "#CC6677",
}
RELIABILITY_AXES = [
    ("auc", "AUROC"),
    ("auprc", "AUPRC"),
    ("accuracy", "Accuracy"),
    ("calibration", "1-ECE"),
    ("fairness", "Fairness"),
    ("quality_robustness", "Quality"),
]
COLORS = {
    "cv": "#4477AA",
    "vlm": "#228833",
    "mllm": "#CC6677",
    "cv_general": "#4477AA",
    "cv_ophthalmo": "#66CCEE",
    "vlm_general": "#88CCAA",
    "vlm_ophthalmo": "#117733",
    "mllm_general": "#DD8899",
    "mllm_medical": "#AA3377",
    "linear_probing": "#4477AA",
    "zero_shot": "#DDCC77",
    "base": "#CC6677",
    "vlm_linear_probing": "#117733",
    "vlm_zero_shot": "#DDCC77",
    "vlm_base": "#CC6677",
    "sft": "#2C7FB8",
    "in_domain": "#2C7FB8",
    "OOD": "#F03B20",
}
TRADEOFF_COLORS = {
    "vlm_general_linear_probing": "#2C7FB8",
    "vlm_general_zero_shot": "#7FCDBB",
    "vlm_ophthalmo_linear_probing": "#D95F0E",
    "vlm_ophthalmo_zero_shot": "#FDD0A2",
}

PALETTE = [
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
    "#AEC7E8",
    "#FFBB78",
    "#98DF8A",
    "#FF9896",
    "#C5B0D5",
    "#C49C94",
    "#F7B6D2",
    "#C7C7C7",
    "#DBDB8D",
    "#9EDAE5",
]


def color_for_key(key: str) -> str:
    if key in COLORS:
        return COLORS[key]
    idx = abs(hash(key)) % len(PALETTE)
    return PALETTE[idx]


def mllm_parameter_billions(model: str) -> float | None:
    params = {
        "Qwen/Qwen3-VL-8B-Instruct": 8.0,
        "google/gemma-3-27b-it": 27.0,
        "llava-hf/llama3-llava-next-8b-hf": 8.0,
        "google/medgemma-1.5-4b-it": 4.0,
        "google/medgemma-27b-it": 27.0,
        "google/medgemma-4b-it": 4.0,
    }
    return params.get(str(model))


def display_model(model: str) -> str:
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
    }
    return aliases.get(str(model), str(model).split("/")[-1])


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


F = {
    "title": font(34, True),
    "subtitle": font(25, True),
    "axis": font(20, False),
    "tick": font(17, False),
    "small": font(14, False),
    "value": font(22, True),
    "bar_value": font(20, True),
}


def rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def lerp(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * max(0.0, min(1.0, t))))


def blend(c1: str, c2: str, t: float) -> tuple[int, int, int]:
    a, b = rgb(c1), rgb(c2)
    return tuple(lerp(x, y, t) for x, y in zip(a, b))


def auc_color(v: float, vmin: float = 0.45, vmax: float = 0.95) -> tuple[int, int, int]:
    return blend("#f7fbff", "#08519c", (v - vmin) / (vmax - vmin))


def ece_color(v: float, vmin: float = 0.05, vmax: float = 0.42) -> tuple[int, int, int]:
    return blend("#e5f5e0", "#de2d26", (v - vmin) / (vmax - vmin))


def diverging_color(v: float, vmax: float) -> tuple[int, int, int]:
    if pd.isna(v):
        return rgb("#F5F5F5")
    if abs(v) < 1e-12:
        return rgb("#F7F7F7")
    if v > 0:
        return blend("#F7F7F7", "#2166AC", abs(v) / vmax)
    return blend("#F7F7F7", "#B2182B", abs(v) / vmax)


def text_size(draw: ImageDraw.ImageDraw, text: str, ft: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=ft)
    return box[2] - box[0], box[3] - box[1]


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, ft, fill="#222222") -> None:
    w, h = text_size(draw, text, ft)
    draw.text((xy[0] - w / 2, xy[1] - h / 2), text, font=ft, fill=fill)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, ft, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).split("\n"):
        current = ""
        for word in paragraph.split():
            candidate = word if not current else f"{current} {word}"
            if text_size(draw, candidate, ft)[0] <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines or [""]


def multiline_center(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    text: str,
    ft,
    max_width: int,
    fill="#222222",
    line_spacing: int = 6,
) -> None:
    lines = wrap_text(draw, text, ft, max_width)
    heights = [text_size(draw, line, ft)[1] for line in lines]
    total_h = sum(heights) + line_spacing * (len(lines) - 1)
    y = center[1] - total_h / 2
    for line, height in zip(lines, heights):
        w, _ = text_size(draw, line, ft)
        draw.text((center[0] - w / 2, y), line, font=ft, fill=fill)
        y += height + line_spacing


def multiline_left_center(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, float],
    text: str,
    ft,
    max_width: int,
    fill="#222222",
    line_spacing: int = 3,
) -> None:
    lines = wrap_text(draw, text, ft, max_width)
    heights = [text_size(draw, line, ft)[1] for line in lines]
    total_h = sum(heights) + line_spacing * (len(lines) - 1)
    y = xy[1] - total_h / 2
    for line, height in zip(lines, heights):
        draw.text((xy[0], y), line, font=ft, fill=fill)
        y += height + line_spacing


def save(img: Image.Image, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    if name in TRIM_OUTPUTS:
        bg = Image.new(img.mode, img.size, "white")
        diff = ImageChops.difference(img, bg)
        bbox = diff.getbbox()
        if bbox:
            margin = 18
            left = max(0, bbox[0] - margin)
            top = max(0, bbox[1] - margin)
            right = min(img.width, bbox[2] + margin)
            bottom = min(img.height, bbox[3] + margin)
            img = img.crop((left, top, right, bottom))
    if EXPORT_SCALE != 1:
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        img = img.resize((img.width * EXPORT_SCALE, img.height * EXPORT_SCALE), resampling)
    if path.suffix.lower() == ".pdf":
        img.convert("RGB").save(path, "PDF", resolution=EXPORT_DPI)
    else:
        img.save(path, dpi=(EXPORT_DPI, EXPORT_DPI))


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def prepare_base() -> pd.DataFrame:
    df = pd.read_csv(BASE_METRICS)
    df["model_display"] = df["model"].map(display_model)
    df["model_type_display"] = df["model_type"].map(MODEL_TYPE_DISPLAY)
    df["family_display"] = df["family"].map(FAMILY_DISPLAY)
    df["dataset_display"] = df["dataset"].map(DATASET_DISPLAY)
    df["task_display"] = df["task"].map(TASK_DISPLAY)
    df["method_display"] = df["method"].map(METHOD_DISPLAY).fillna(df["method"])
    df["method_short"] = df["method"].map(METHOD_SHORT).fillna(df["method"])
    df["model_method"] = df["model_display"] + " (" + df["method_short"] + ")"
    return df


def prepare_ft() -> pd.DataFrame:
    df = pd.read_csv(FT_METRICS)
    df = df[df["ft_method"] == "sft"].copy()
    df["domain"] = np.where(as_bool(df["is_ood_dataset"]), "OOD", "in_domain")
    df["train_model_display"] = df["train_model"].map(display_model)
    df["train_task_display"] = df["train_task"].map(TASK_DISPLAY)
    return df


def draw_x_axis(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    xmin: float,
    xmax: float,
    ticks: Iterable[float],
    label: str,
) -> None:
    left, top, right, bottom = plot
    draw.line((left, bottom, right, bottom), fill="#333333", width=2)
    for tick in ticks:
        x = left + (tick - xmin) / (xmax - xmin) * (right - left)
        draw.line((x, top, x, bottom), fill="#E8E8E8", width=2)
        text_center(draw, (int(x), bottom + 26), f"{tick:.2f}", F["small"], "#333333")
    text_center(draw, ((left + right) // 2, bottom + 60), label, F["axis"])


def hbar_panel(
    draw: ImageDraw.ImageDraw,
    panel: tuple[int, int, int, int],
    title: str,
    labels: list[str],
    values: list[float],
    colors: list[str],
    xmin: float,
    xmax: float,
    x_label: str = "Mean AUROC",
) -> None:
    x0, y0, x1, y1 = panel
    multiline_center(draw, ((x0 + x1) // 2, y0 + 28), title, F["subtitle"], x1 - x0 - 24, line_spacing=3)
    label_w = 300 if any(text_size(draw, str(label), F["small"])[0] > 190 for label in labels) else 220
    plot = (x0 + label_w + 28, y0 + 82, x1 - 38, y1 - 75)
    ticks = np.linspace(xmin, xmax, 5)
    draw_x_axis(draw, plot, xmin, xmax, ticks, x_label)
    left, top, right, bottom = plot
    n = max(1, len(labels))
    row_h = (bottom - top) / n
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        cy = top + row_h * (i + 0.5)
        bar_h = min(28, row_h * 0.62)
        x_val = left + (value - xmin) / (xmax - xmin) * (right - left)
        multiline_left_center(draw, (x0 + 8, cy), label, F["small"], label_w - 18)
        draw.rounded_rectangle((left, cy - bar_h / 2, x_val, cy + bar_h / 2), radius=3, fill=rgb(color))
        draw.text((x_val + 8, cy - 13), f"{value:.2f}", font=F["bar_value"], fill="#222222")


def reliability_scores(base: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    agg = base.groupby(group_cols, as_index=False).agg(
        accuracy=("accuracy", "mean"),
        auc=("auc", "mean"),
        auprc=("auprc", "mean"),
        ece=("ece", "mean"),
    )
    fairness_cols = [
        c
        for c in base.columns
        if c.endswith(("equalized_odds_gap", "accuracy_gap", "auc_gap"))
        and not c.startswith("image_quality_")
        and base[c].notna().any()
    ]
    robustness_cols = [
        c
        for c in base.columns
        if c.startswith("image_quality_") and c.endswith("_gap") and base[c].notna().any()
    ]
    if fairness_cols:
        fairness = base.groupby(group_cols)[fairness_cols].mean().mean(axis=1).rename("fairness_gap")
        agg = agg.merge(fairness, left_on=group_cols, right_index=True, how="left")
    else:
        agg["fairness_gap"] = np.nan
    if robustness_cols:
        robustness = base.groupby(group_cols)[robustness_cols].mean().mean(axis=1).rename("quality_gap")
        agg = agg.merge(robustness, left_on=group_cols, right_index=True, how="left")
    else:
        agg["quality_gap"] = np.nan

    agg["calibration"] = 1.0 - agg["ece"]
    fairness_fill = agg["fairness_gap"].median() if agg["fairness_gap"].notna().any() else 0.0
    robustness_fill = agg["quality_gap"].median() if agg["quality_gap"].notna().any() else 0.0
    agg["fairness"] = 1.0 - agg["fairness_gap"].fillna(fairness_fill)
    agg["quality_robustness"] = 1.0 - agg["quality_gap"].fillna(robustness_fill)
    for col, _ in RELIABILITY_AXES:
        agg[col] = agg[col].clip(0.0, 1.0)
    return agg


def score_color(v: float) -> tuple[int, int, int]:
    return blend("#fff7ec", "#2c7fb8", (v - 0.45) / 0.55)


def plot_model_type_reliability_matrix(base: pd.DataFrame) -> None:
    scores = reliability_scores(base, ["model_type", "model_type_display"])
    scores["model_type"] = pd.Categorical(scores["model_type"], MODEL_TYPE_ORDER, ordered=True)
    scores = scores.sort_values("model_type")
    matrix = scores.set_index("model_type_display")[[col for col, _ in RELIABILITY_AXES]]
    matrix.columns = [label for _, label in RELIABILITY_AXES]
    img = Image.new("RGB", (1650, 760), "white")
    draw = ImageDraw.Draw(img)
    draw_heatmap(
        draw,
        (55, 55, 1595, 700),
        "Model-Type Reliability Matrix",
        matrix,
        score_color,
        ".2f",
        value_font=F["subtitle"],
    )
    save(img, "model_type_reliability_matrix.png")


def radar_points(center: tuple[int, int], radius: int, values: list[float], vmin: float = 0.35) -> list[tuple[float, float]]:
    pts = []
    n = len(values)
    for i, value in enumerate(values):
        angle = -math.pi / 2 + 2 * math.pi * i / n
        r = radius * (max(vmin, min(1.0, float(value))) - vmin) / (1.0 - vmin)
        pts.append((center[0] + r * math.cos(angle), center[1] + r * math.sin(angle)))
    return pts


def draw_radar_panel(
    draw: ImageDraw.ImageDraw,
    panel: tuple[int, int, int, int],
    title: str,
    rows: pd.DataFrame,
    color_col: str,
    show_legend: bool = True,
    show_method_legend: bool = False,
) -> None:
    x0, y0, x1, y1 = panel
    center = ((x0 + x1) // 2, y0 + 275)
    radius = min((x1 - x0) // 2 - 85, 185)
    multiline_center(draw, ((x0 + x1) // 2, y0 + 32), title, F["subtitle"], x1 - x0 - 28, line_spacing=3)
    labels = [label for _, label in RELIABILITY_AXES]
    for frac, label in [(0.0, "0.35"), (0.5, "0.68"), (1.0, "1.00")]:
        r = int(radius * frac)
        pts = [
            (center[0] + r * math.cos(-math.pi / 2 + 2 * math.pi * i / len(labels)),
             center[1] + r * math.sin(-math.pi / 2 + 2 * math.pi * i / len(labels)))
            for i in range(len(labels))
        ]
        draw.line(pts + [pts[0]], fill=rgb("#DDDDDD"), width=1)
        if frac > 0:
            draw.text((center[0] + 5, center[1] - r - 10), label, font=F["small"], fill="#666666")
    for i, label in enumerate(labels):
        angle = -math.pi / 2 + 2 * math.pi * i / len(labels)
        end = (center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle))
        draw.line((center[0], center[1], end[0], end[1]), fill=rgb("#E5E5E5"), width=1)
        lx = center[0] + (radius + 44) * math.cos(angle)
        ly = center[1] + (radius + 34) * math.sin(angle)
        text_center(draw, (int(lx), int(ly)), label, F["small"])

    legend_y = y1 - 125
    for idx, row in enumerate(rows.itertuples()):
        values = [getattr(row, col) for col, _ in RELIABILITY_AXES]
        pts = radar_points(center, radius, values)
        color = color_for_key(str(getattr(row, color_col)))
        draw.line(pts + [pts[0]], fill=rgb(color), width=4)
        if show_legend:
            lx = x0 + 20 + (idx % 2) * ((x1 - x0) // 2)
            ly = legend_y + (idx // 2) * 28
            draw.line((lx, ly + 8, lx + 30, ly + 8), fill=rgb(color), width=4)
            label = getattr(row, "model_method", getattr(row, "model_type_display", "model"))
            draw.text((lx + 38, ly), str(label)[:42], font=F["small"], fill="#222222")
    if show_legend and show_method_legend:
        draw.text((x0 + 20, y1 - 20), "LP=Linear probing, ZS=Zero-shot, ZS Prompt=Zero-shot prompting", font=F["small"], fill="#555555")


def plot_reliability_spiders(base: pd.DataFrame) -> None:
    rel = reliability_scores(
        base,
        ["family", "model_type", "model_type_display", "model_display", "method", "method_short", "model_method"],
    )
    rel["vlm_method_color"] = np.where(
        rel["family"] == "vlm",
        "vlm_" + rel["method"].astype(str),
        rel["model_type"].astype(str),
    )
    top = rel.sort_values("auc", ascending=False).head(8)
    img = Image.new("RGB", (900, 820), "white")
    draw = ImageDraw.Draw(img)
    draw_radar_panel(draw, (45, 45, 855, 775), "Top Configurations Across Metrics", top, "model_method")
    save(img, "top_model_reliability_spider.png")
    save(img, "top_model_reliability_spider.pdf")

    img = Image.new("RGB", (1800, 690), "white")
    draw = ImageDraw.Draw(img)
    for i, family in enumerate(FAMILY_ORDER):
        rows = rel[rel["family"] == family].sort_values("auc", ascending=False).head(6)
        draw_radar_panel(
            draw,
            (40 + i * 590, 25, 570 + i * 590, 660),
            FAMILY_DISPLAY[family],
            rows,
            "model_method",
            show_legend=True,
            show_method_legend=False,
        )
    save(img, "arena_spider_grid.png")
    save(img, "arena_spider_grid.pdf")

    for family in FAMILY_ORDER:
        rows = rel[rel["family"] == family].sort_values("auc", ascending=False)
        img = Image.new("RGB", (900, 820), "white")
        draw = ImageDraw.Draw(img)
        draw_radar_panel(draw, (45, 45, 855, 775), f"{FAMILY_DISPLAY[family]} Top Configurations", rows, "model_method")
        save(img, f"{family}_top_model_spider.png")
        save(img, f"{family}_top_model_spider.pdf")


def parse_binary_prediction_text(value: object) -> float:
    text = str(value).strip().lower()
    match = re.search(r"\b(yes|no)\b", text)
    if not match:
        return np.nan
    if match.group(1) == "yes":
        return 1.0
    return 0.0


def normalize_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "label" not in df.columns and "ground_truth" in df.columns:
        df["label"] = df["ground_truth"]
    if "pred" not in df.columns:
        if "prediction" in df.columns:
            df["pred"] = df["prediction"]
        elif "prediction_text" in df.columns:
            df["pred"] = df["prediction_text"].apply(parse_binary_prediction_text)
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


def model_slug_variants(model: str) -> list[str]:
    slash_slug = str(model).replace("/", "_")
    strict_slug = slash_slug.replace("-", "_")
    alnum_slug = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in slash_slug).strip("_")
    return list(dict.fromkeys([slash_slug, strict_slug, alnum_slug]))


def locate_result_file(row: pd.Series) -> Path | None:
    dataset = row["dataset"]
    task = row["task"]
    method = row["method"]
    for slug in model_slug_variants(row["model"]):
        candidates: list[Path] = []
        if method == "linear_probing":
            candidates.extend(
                [
                    RESULTS_DIR / f"{dataset}_{task}_linear_probing_{slug}.csv",
                    RESULTS_DIR / f"cv_{dataset}_{task}_linear_probing_{slug}.csv",
                ]
            )
        elif method == "zero_shot":
            candidates.append(RESULTS_DIR / f"{dataset}_{task}_zero_shot_{slug}.csv")
        elif method == "base":
            candidates.append(RESULTS_DIR / f"{dataset}_{task}_{slug}.csv")
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def calibration_points(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[tuple[float, float]]:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_prob) & ((y_true == 0) | (y_true == 1))
    y_true = y_true[valid]
    y_prob = np.clip(y_prob[valid], 0.0, 1.0)
    if len(y_true) < 10 or len(np.unique(y_true)) < 2:
        return []
    binids = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)
    points: list[tuple[float, float]] = []
    for i in range(n_bins):
        mask = binids == i
        if mask.sum() >= 2:
            points.append((float(y_prob[mask].mean()), float(y_true[mask].mean())))
    return points


def load_curve_for_row(row: pd.Series, cache: dict[str, list[tuple[float, float]]]) -> list[tuple[float, float]]:
    path = locate_result_file(row)
    if path is None:
        return []
    key = str(path)
    if key not in cache:
        try:
            df = normalize_result_columns(pd.read_csv(path))
            cache[key] = calibration_points(df["label"].to_numpy(), df["prob"].to_numpy())
        except Exception:
            cache[key] = []
    return cache[key]


def plot_calibration_curve_sheets(base: pd.DataFrame) -> None:
    curve_colors = ["#4477AA", "#228833", "#CC6677", "#AA3377", "#DDCC77"]
    cache: dict[str, list[tuple[float, float]]] = {}
    ranked = (
        base.groupby(["model_type", "model_method"], as_index=False)["auc"]
        .mean()
        .sort_values(["model_type", "auc"], ascending=[True, False])
    )

    for model_type in MODEL_TYPE_ORDER:
        top_methods = ranked[ranked["model_type"] == model_type]["model_method"].head(5).tolist()
        if not top_methods:
            continue
        subset = base[(base["model_type"] == model_type) & (base["model_method"].isin(top_methods))].copy()
        if subset.empty:
            continue

        task_panels = [
            (task, [dataset for dataset in DATASET_ORDER if not subset[(subset["dataset"] == dataset) & (subset["task"] == task)].empty])
            for task in TASK_ORDER
        ]
        task_panels = [(task, datasets) for task, datasets in task_panels if datasets]
        if not task_panels:
            continue

        panel_w, panel_h = 210, 285
        left0, top0 = 135, 120
        x_gap, y_gap = 16, 72
        right_margin = 60
        max_cols = max(len(datasets) for _, datasets in task_panels)
        legend_h = 48 + 28 * len(top_methods)
        width = max(920, int(left0 + max_cols * panel_w + (max_cols - 1) * x_gap + right_margin))
        height = int(top0 + len(task_panels) * panel_h + (len(task_panels) - 1) * y_gap + legend_h + 40)

        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        multiline_center(
            draw,
            (width // 2, 42),
            f"Calibration Curves: {MODEL_TYPE_DISPLAY[model_type]}",
            F["title"],
            width - 90,
            line_spacing=4,
        )
        for r, (task, datasets) in enumerate(task_panels):
            y = top0 + r * (panel_h + y_gap)
            draw.text((24, y + panel_h // 2 - 16), TASK_DISPLAY[task], font=F["axis"], fill="#222222")
            for c, dataset in enumerate(datasets):
                x = left0 + c * (panel_w + x_gap)
                multiline_center(
                    draw,
                    (x + panel_w // 2, y + 18),
                    DATASET_DISPLAY[dataset],
                    F["small"],
                    panel_w - 18,
                    line_spacing=2,
                )
                plot = (x + 34, y + 52, x + panel_w - 22, y + panel_h - 48)
                pl, pt, pr, pb = plot
                draw.line((pl, pb, pr, pt), fill="#AAAAAA", width=1)
                draw.line((pl, pb, pr, pb), fill="#333333", width=1)
                draw.line((pl, pt, pl, pb), fill="#333333", width=1)
                text_center(draw, ((pl + pr) // 2, pb + 22), "Pred.", F["small"])
                rows = subset[(subset["dataset"] == dataset) & (subset["task"] == task)]
                for idx, method in enumerate(top_methods):
                    row_match = rows[rows["model_method"] == method]
                    if row_match.empty:
                        continue
                    pts = load_curve_for_row(row_match.iloc[0], cache)
                    if len(pts) < 2:
                        continue
                    coords = [
                        (
                            pl + px * (pr - pl),
                            pb - py * (pb - pt),
                        )
                        for px, py in pts
                    ]
                    draw.line(coords, fill=rgb(curve_colors[idx % len(curve_colors)]), width=2)
                    for cx, cy in coords:
                        draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=rgb(curve_colors[idx % len(curve_colors)]))
        legend_y = top0 + len(task_panels) * panel_h + (len(task_panels) - 1) * y_gap + 35
        for i, method in enumerate(top_methods):
            y = legend_y + i * 28
            draw.line((160, y + 8, 200, y + 8), fill=rgb(curve_colors[i % len(curve_colors)]), width=4)
            draw.text((214, y), method, font=F["tick"], fill="#222222")
        save(img, f"calibration_curves_{model_type}.png")


def plot_model_level_auc_leaderboard(base: pd.DataFrame) -> None:
    axis_font = font(20, False)
    legend_font = font(18, False)
    label_font = font(18, False)
    grouped = base.groupby(
        ["model_type", "model_type_display", "model_display", "method_display", "task"],
        as_index=False,
    )["auc"].mean()
    ranking = (
        grouped.groupby(["model_type", "model_type_display", "model_display", "method_display"], as_index=False)["auc"]
        .mean()
        .sort_values(["model_type", "auc"], ascending=[True, False])
    )
    img = Image.new("RGB", (1800, 1940), "white")
    draw = ImageDraw.Draw(img)
    text_center(draw, (900, 36), "Model-Type-Stratified AUROC Leaderboards", F["title"])
    xmin, xmax = 0.35, 1.00
    for p, model_type in enumerate(MODEL_TYPE_ORDER):
        rows = ranking[ranking["model_type"] == model_type].head(5)
        if rows.empty:
            continue
        col = p % 2
        row_i = p // 2
        panel = (55 + col * 880, 90 + row_i * 590, 850 + col * 880, 640 + row_i * 590)
        x0, y0, x1, y1 = panel
        multiline_center(draw, ((x0 + x1) // 2, y0 + 34), MODEL_TYPE_DISPLAY[model_type], F["subtitle"], x1 - x0 - 40, line_spacing=3)
        plot = (x0 + 250, y0 + 92, x1 - 35, y1 - 65)
        left, top, right, bottom = plot
        for tick in [0.4, 0.6, 0.8, 1.0]:
            x = left + (tick - xmin) / (xmax - xmin) * (right - left)
            draw.line((x, top, x, bottom), fill="#E8E8E8", width=2)
            text_center(draw, (int(x), bottom + 26), f"{tick:.1f}", axis_font)
        draw.line((left, bottom, right, bottom), fill="#333333", width=2)
        row_h = (bottom - top) / max(1, len(rows))
        for i, model_row in enumerate(rows.itertuples()):
            cy = top + row_h * (i + 0.5)
            label = f"{model_row.model_display}\n{model_row.method_display}"
            draw.text((x0 + 8, cy - 12), label, font=label_font, fill="#222222")
            task_values = grouped[
                (grouped["model_type"] == model_type)
                & (grouped["model_display"] == model_row.model_display)
                & (grouped["method_display"] == model_row.method_display)
            ].set_index("task")["auc"]
            for j, task in enumerate(TASK_ORDER):
                if task not in task_values.index:
                    continue
                value = float(task_values.loc[task])
                y = cy + (j - 1) * 16
                xv = left + (value - xmin) / (xmax - xmin) * (right - left)
                draw.rounded_rectangle((left, y - 5, xv, y + 5), radius=2, fill=rgb(TASK_COLORS[task]))
                draw.text((xv + 6, y - 13), f"{value:.2f}", font=label_font, fill="#222222")
    legend_y = 1870
    text_center(draw, (900, legend_y - 32), "Task", axis_font)
    for i, task in enumerate(TASK_ORDER):
        x = 600 + i * 220
        draw.rectangle((x, legend_y, x + 34, legend_y + 18), fill=rgb(TASK_COLORS[task]))
        draw.text((x + 44, legend_y - 2), TASK_DISPLAY[task], font=legend_font, fill="#222222")
    draw.text(
        (70, legend_y - 28),
        "Method key: Linear probing, Zero-shot, Zero-shot prompting",
        font=legend_font,
        fill="#333333",
    )
    save(img, "model_level_auc_leaderboard.png")
    save(img, "model_level_auc_leaderboard.pdf")


def plot_mllm_size_tradeoff(base: pd.DataFrame) -> None:
    rows = base[base["family"] == "mllm"].copy()
    if rows.empty:
        return
    summary = rows.groupby(["model", "model_display", "model_type"], as_index=False).agg(
        auc=("auc", "mean"),
        ece=("ece", "mean"),
        accuracy=("accuracy", "mean"),
    )
    summary["params_b"] = summary["model"].map(mllm_parameter_billions)
    summary = summary.dropna(subset=["params_b"])
    if summary.empty:
        return

    img = Image.new("RGB", (1120, 830), "white")
    draw = ImageDraw.Draw(img)
    text_center(draw, (560, 34), "MLLM Size, Domain Specialization, and Reliability", F["title"])
    plot = (120, 105, 1040, 590)
    left, top, right, bottom = plot
    xmin, xmax = 3.0, 30.0
    ymin, ymax = 0.55, 0.90
    for tick in [4, 8, 16, 27]:
        x = left + (tick - xmin) / (xmax - xmin) * (right - left)
        draw.line((x, top, x, bottom), fill="#E8E8E8", width=2)
        text_center(draw, (int(x), bottom + 24), str(tick), F["small"])
    for tick in [0.55, 0.65, 0.75, 0.85]:
        y = bottom - (tick - ymin) / (ymax - ymin) * (bottom - top)
        draw.line((left, y, right, y), fill="#E8E8E8", width=2)
        draw.text((58, y - 10), f"{tick:.2f}", font=F["small"], fill="#333333")
    draw.line((left, bottom, right, bottom), fill="#333333", width=2)
    draw.line((left, top, left, bottom), fill="#333333", width=2)
    text_center(draw, ((left + right) // 2, bottom + 62), "Approximate parameter count (B)", F["axis"])
    text_center(draw, (55, (top + bottom) // 2), "AUROC", F["axis"])
    label_offsets = {
        "MedGemma 4B": (-4, -28),
        "MedGemma 1.5 4B": (-4, 6),
        "Gemma-3-27B": (4, 12),
    }
    for row in summary.itertuples():
        x = left + (float(row.params_b) - xmin) / (xmax - xmin) * (right - left)
        y = bottom - (float(row.auc) - ymin) / (ymax - ymin) * (bottom - top)
        color = COLORS[row.model_type]
        radius = int(13 + 38 * max(0.0, min(0.35, 0.35 - float(row.ece))) / 0.35)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=rgb(color), outline="#222222", width=2)
        dx, dy = label_offsets.get(row.model_display, (4, -9))
        draw.text((x + radius + 8 + dx, y + dy), row.model_display, font=F["small"], fill="#222222")
    legend_items = [
        ("mllm_general", MODEL_TYPE_DISPLAY["mllm_general"]),
        ("mllm_medical", MODEL_TYPE_DISPLAY["mllm_medical"]),
    ]
    for i, (key, label) in enumerate(legend_items):
        x = 150 + i * 420
        y = 700
        draw.ellipse((x, y - 10, x + 20, y + 10), fill=rgb(COLORS[key]), outline="#222222")
        multiline_left_center(draw, (x + 30, y), label, F["small"], 360)
    save(img, "mllm_size_domain_tradeoff.png")
    save(img, "mllm_size_domain_tradeoff.pdf")


def plot_model_type_task_auc(base: pd.DataFrame) -> None:
    summary = base.groupby(["task", "model_type", "model_type_display"], as_index=False)["auc"].mean()
    img = Image.new("RGB", (2400, 760), "white")
    draw = ImageDraw.Draw(img)
    text_center(draw, (1200, 34), "AUROC by Task and Model Type", F["title"])
    for i, task in enumerate(TASK_ORDER):
        rows = summary[summary["task"] == task].sort_values("auc", ascending=False)
        panel = (35 + i * 785, 75, 770 + i * 785, 705)
        hbar_panel(
            draw,
            panel,
            TASK_DISPLAY[task],
            rows["model_type_display"].tolist(),
            rows["auc"].tolist(),
            [COLORS[m] for m in rows["model_type"]],
            0.50,
            0.90,
        )
    save(img, "model_type_task_auc_heatmap.png")


def plot_task_arena_leaderboards(base: pd.DataFrame) -> None:
    grouped = base.groupby(["task", "family", "model_method"], as_index=False)["auc"].mean()
    img = Image.new("RGB", (1350, 1650), "white")
    draw = ImageDraw.Draw(img)
    text_center(draw, (675, 34), "Task-Specific AUROC Leaderboards", F["title"])
    for i, task in enumerate(TASK_ORDER):
        rows = grouped[grouped["task"] == task].sort_values("auc", ascending=False).head(10)
        panel = (55, 80 + i * 520, 1295, 555 + i * 520)
        hbar_panel(
            draw,
            panel,
            TASK_DISPLAY[task],
            rows["model_method"].tolist(),
            rows["auc"].tolist(),
            [COLORS[f] for f in rows["family"]],
            0.35,
            1.00,
        )
    save(img, "task_arena_leaderboards.png")


def draw_heatmap(
    draw: ImageDraw.ImageDraw,
    panel: tuple[int, int, int, int],
    title: str,
    matrix: pd.DataFrame,
    color_fn,
    fmt: str,
    value_fill: str = "#111111",
    value_font=None,
    label_font=None,
    label_w_override: int | None = None,
    col_label_gap: int = 18,
    row_label_align: str = "left",
) -> None:
    x0, y0, x1, y1 = panel
    multiline_center(draw, ((x0 + x1) // 2, y0 + 28), title, F["subtitle"], x1 - x0 - 24, line_spacing=3)
    rows = list(matrix.index)
    cols = list(matrix.columns)
    label_font = label_font or F["small"]
    label_w = label_w_override or (355 if any(text_size(draw, str(row), label_font)[0] > 220 for row in rows) else 245)
    cell_w = max(42, (x1 - x0 - label_w - 25) / max(1, len(cols)))
    cell_h = max(32, (y1 - y0 - 95) / max(1, len(rows)))
    grid_x = int(x0 + label_w)
    grid_y = int(y0 + 68)
    for j, col in enumerate(cols):
        text_center(draw, (int(grid_x + cell_w * (j + 0.5)), grid_y - col_label_gap), str(col), label_font)
    for i, row in enumerate(rows):
        y = grid_y + i * cell_h
        if row_label_align == "right":
            rw, rh = text_size(draw, str(row), label_font)
            draw.text((grid_x - rw - 8, y + cell_h / 2 - rh / 2), str(row), font=label_font, fill="#222222")
        else:
            multiline_left_center(draw, (x0 + 4, y + cell_h / 2), str(row), label_font, label_w - 12)
        for j, col in enumerate(cols):
            x = grid_x + j * cell_w
            v = matrix.loc[row, col]
            fill = rgb("#FFFFFF") if pd.isna(v) else color_fn(float(v))
            draw.rectangle((x, y, x + cell_w - 2, y + cell_h - 2), fill=fill, outline="#FFFFFF")
            if not pd.isna(v):
                text_center(draw, (int(x + cell_w / 2), int(y + cell_h / 2)), format(float(v), fmt), value_font or F["value"], value_fill)


def plot_model_dataset_auc(base: pd.DataFrame) -> None:
    img = Image.new("RGB", (1750, 2200), "white")
    draw = ImageDraw.Draw(img)
    text_center(draw, (875, 36), "Model-by-Dataset AUROC", F["title"])
    for k, family in enumerate(FAMILY_ORDER):
        subset = base[base["family"] == family]
        order = subset.groupby("model_method")["auc"].mean().sort_values(ascending=False).head(10).index
        mat = subset[subset["model_method"].isin(order)].pivot_table(
            index="model_method", columns="dataset", values="auc", aggfunc="mean"
        )
        mat = mat.reindex(order).reindex(columns=[d for d in DATASET_ORDER if d in mat.columns])
        mat = mat.rename(columns=DATASET_DISPLAY)
        draw_heatmap(
            draw,
            (45, 80 + k * 700, 1705, 720 + k * 700),
            FAMILY_DISPLAY[family],
            mat,
            auc_color,
            ".2f",
            "#111111",
        )
    save(img, "model_dataset_auc_heatmap.png")


def plot_auc_ece_tradeoff(base: pd.DataFrame) -> None:
    grouped = base.groupby(["family", "model_type", "model_display", "method_display"], as_index=False).agg(
        auc=("auc", "mean"), ece=("ece", "mean")
    )
    img = Image.new("RGB", (2700, 890), "white")
    draw = ImageDraw.Draw(img)
    xmin, xmax = 0.05, 0.42
    ymin, ymax = 0.48, 1.00
    for k, family in enumerate(FAMILY_ORDER):
        x0, y0, x1, y1 = 45 + k * 875, 20, 850 + k * 875, 850
        multiline_center(draw, ((x0 + x1) // 2, y0 + 24), FAMILY_DISPLAY[family], F["subtitle"], x1 - x0 - 30, line_spacing=3)
        plot = (x0 + 115, y0 + 88, x1 - 45, y1 - 95)
        draw_x_axis(draw, plot, xmin, xmax, [0.10, 0.20, 0.30, 0.40], "ECE")
        left, top, right, bottom = plot
        for tick in [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
            y = bottom - (tick - ymin) / (ymax - ymin) * (bottom - top)
            draw.line((left, y, right, y), fill="#E8E8E8", width=2)
            draw.text((x0 + 20, y - 9), f"{tick:.2f}", font=F["small"], fill="#333333")
        draw.text((left - 92, top - 42), "AUROC", font=F["axis"], fill="#222222")
        subset = grouped[grouped["family"] == family].sort_values("auc", ascending=False)
        points = []
        if family == "vlm":
            legend = [
                ("General LP", TRADEOFF_COLORS["vlm_general_linear_probing"]),
                ("General ZS", TRADEOFF_COLORS["vlm_general_zero_shot"]),
                ("Ophthalmic LP", TRADEOFF_COLORS["vlm_ophthalmo_linear_probing"]),
                ("Ophthalmic ZS", TRADEOFF_COLORS["vlm_ophthalmo_zero_shot"]),
            ]
            for n, (legend_label, legend_color) in enumerate(legend):
                lx = x0 + 112 + (n % 2) * 220
                ly = top + 8 + (n // 2) * 26
                draw.ellipse((lx, ly, lx + 14, ly + 14), fill=rgb(legend_color), outline="white", width=2)
                draw.text((lx + 20, ly - 3), legend_label, font=F["small"], fill="#222222")
        for row in subset.itertuples():
            px = left + (row.ece - xmin) / (xmax - xmin) * (right - left)
            py = bottom - (row.auc - ymin) / (ymax - ymin) * (bottom - top)
            color_key = f"{row.model_type}_{row.method_display.lower().replace('-', '_').replace(' ', '_')}"
            color = rgb(TRADEOFF_COLORS.get(color_key, COLORS[row.model_type]))
            draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=color, outline="white", width=2)
            label = row.model_display
            if row.method_display == "Zero-shot":
                label += " ZS"
            elif row.method_display in {"Linear probe", "Linear probing"}:
                label += " LP"
            points.append({"x": px, "y": py, "label": label})

        def spread(items: list[dict[str, float | str]], min_y: float, max_y: float, step: float) -> list[float]:
            if not items:
                return []
            ys = [float(item["y"]) for item in sorted(items, key=lambda item: float(item["y"]))]
            out = []
            for y in ys:
                out.append(min(max(y, min_y), max_y) if not out else min(max(y, out[-1] + step), max_y))
            overflow = out[-1] - max_y
            if overflow > 0:
                out = [y - overflow for y in out]
                for i in range(1, len(out)):
                    out[i] = max(out[i], out[i - 1] + step)
            return out

        for side in ["left", "right"]:
            items = [p for p in points if (p["x"] <= (left + right) / 2) == (side == "left")]
            items = sorted(items, key=lambda item: float(item["y"]))
            label_ys = spread(items, top + (72 if family == "vlm" else 8), bottom - 26, 35)
            for point, label_y in zip(items, label_ys):
                label = str(point["label"])
                lw, lh = text_size(draw, label, F["small"])
                if side == "left":
                    lx = left + 8
                    line_end = lx + lw + 4
                else:
                    lx = right - lw - 8
                    line_end = lx - 4
                ly = label_y - lh / 2
                draw.line((point["x"], point["y"], line_end, label_y), fill="#B8B8B8", width=1)
                draw.rectangle((lx - 3, ly - 2, lx + lw + 3, ly + lh + 2), fill="white")
                draw.text((lx, ly), label, font=F["small"], fill="#222222")
    save(img, "auc_ece_tradeoff.png")


def plot_calibration_family_grid(base: pd.DataFrame, family: str, filename: str) -> None:
    subset = base[base["family"] == family]
    order = subset.groupby("model_method")["auc"].mean().sort_values(ascending=False).head(10).index
    mat = subset[subset["model_method"].isin(order)].pivot_table(index="model_method", columns="dataset", values="ece", aggfunc="mean")
    mat = mat.reindex(order).reindex(columns=[d for d in DATASET_ORDER if d in mat.columns]).rename(columns=DATASET_DISPLAY)
    h = 190 + 44 * len(mat)
    img = Image.new("RGB", (1750, h), "white")
    draw = ImageDraw.Draw(img)
    draw_heatmap(draw, (40, 35, 1710, h - 35), f"{FAMILY_DISPLAY[family]} Calibration (ECE)", mat, ece_color, ".2f")
    save(img, filename)


def plot_calibration_model_type(base: pd.DataFrame) -> None:
    mat = base.pivot_table(index="model_type", columns="task", values="ece", aggfunc="mean")
    mat = mat.reindex(MODEL_TYPE_ORDER).rename(index=MODEL_TYPE_DISPLAY)
    mat = mat.reindex(columns=TASK_ORDER).rename(columns=TASK_DISPLAY)
    img = Image.new("RGB", (1050, 575), "white")
    draw = ImageDraw.Draw(img)
    draw_heatmap(draw, (40, 35, 1010, 535), "Calibration by Model Type and Task (ECE)", mat, ece_color, ".2f")
    save(img, "calibration_model_type_grids.pdf")


def plot_dataset_family_metric_panels(base: pd.DataFrame) -> None:
    grouped = base.groupby(["dataset", "family"], as_index=False).agg(auc=("auc", "mean"), ece=("ece", "mean"))
    img = Image.new("RGB", (1700, 950), "white")
    draw = ImageDraw.Draw(img)
    text_center(draw, (850, 34), "Dataset-Level Performance and Calibration", F["title"])
    for p, (metric, label, xmin, xmax) in enumerate([("auc", "AUROC", 0.45, 0.95), ("ece", "ECE", 0.05, 0.40)]):
        x0, y0, x1, y1 = (60, 85 + p * 425, 1640, 460 + p * 425)
        text_center(draw, ((x0 + x1) // 2, y0 + 20), label, F["subtitle"])
        plot = (x0 + 150, y0 + 55, x1 - 35, y1 - 60)
        draw_x_axis(draw, plot, xmin, xmax, np.linspace(xmin, xmax, 5), label)
        left, top, right, bottom = plot
        datasets = [d for d in DATASET_ORDER if d in set(grouped["dataset"])]
        row_h = (bottom - top) / len(datasets)
        bar_h = row_h / 4
        for i, dataset in enumerate(datasets):
            cy = top + row_h * (i + 0.5)
            draw.text((x0 + 6, cy - 10), DATASET_DISPLAY[dataset], font=F["tick"], fill="#222222")
            for j, fam in enumerate(FAMILY_ORDER):
                row = grouped[(grouped["dataset"] == dataset) & (grouped["family"] == fam)]
                if row.empty:
                    continue
                value = float(row.iloc[0][metric])
                y = cy + (j - 1) * bar_h
                xv = left + (value - xmin) / (xmax - xmin) * (right - left)
                draw.rectangle((left, y - bar_h / 2, xv, y + bar_h / 2), fill=rgb(COLORS[fam]))
        for j, fam in enumerate(FAMILY_ORDER):
            lx = x1 - 520
            ly = y0 + 8 + j * 25
            draw.rectangle((lx, ly, lx + 18, ly + 18), fill=rgb(COLORS[fam]))
            draw.text((lx + 25, ly - 1), FAMILY_DISPLAY[fam], font=F["small"], fill="#222222")
    save(img, "dataset_family_metric_panels.png")


def plot_domain_method_shift_panels(base: pd.DataFrame) -> None:
    img = Image.new("RGB", (1700, 820), "white")
    draw = ImageDraw.Draw(img)
    text_center(draw, (850, 34), "Interface and Adaptation Summary", F["title"])
    mt = base.groupby(["model_type", "model_type_display"], as_index=False)["auc"].mean()
    mt["model_type"] = pd.Categorical(mt["model_type"], MODEL_TYPE_ORDER, ordered=True)
    mt = mt.sort_values("model_type")
    hbar_panel(
        draw,
        (55, 85, 870, 750),
        "Model Type Mean AUROC",
        mt["model_type_display"].tolist(),
        mt["auc"].tolist(),
        [COLORS[m] for m in mt["model_type"].astype(str)],
        0.55,
        0.85,
    )
    vlm = base[base["family"] == "vlm"].groupby(["task", "method"], as_index=False)["auc"].mean()
    panel = (915, 85, 1645, 750)
    x0, y0, x1, y1 = panel
    multiline_center(
        draw,
        ((x0 + x1) // 2, y0 + 28),
        "Dual Encoder Vision Language Model (VLM-encoders) Method Effect",
        F["subtitle"],
        x1 - x0 - 20,
        line_spacing=3,
    )
    for j, method in enumerate(["linear_probing", "zero_shot"]):
        lx = x0 + 205 + j * 205
        ly = y0 + 105
        draw.rectangle((lx, ly, lx + 18, ly + 18), fill=rgb(COLORS[method]))
        draw.text((lx + 23, ly - 3), METHOD_DISPLAY[method], font=F["small"], fill="#222222")
    plot = (x0 + 115, y0 + 155, x1 - 45, y1 - 75)
    draw_x_axis(draw, plot, 0.35, 0.90, [0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "Mean AUROC")
    left, top, right, bottom = plot
    row_h = (bottom - top) / len(TASK_ORDER)
    for i, task in enumerate(TASK_ORDER):
        cy = top + row_h * (i + 0.5)
        draw.text((x0 + 8, cy - 11), TASK_DISPLAY[task], font=F["tick"], fill="#222222")
        for j, method in enumerate(["linear_probing", "zero_shot"]):
            val = float(vlm[(vlm["task"] == task) & (vlm["method"] == method)]["auc"].mean())
            y = cy + (j - 0.5) * 24
            xv = left + (val - 0.35) / (0.90 - 0.35) * (right - left)
            draw.rounded_rectangle((left, y - 10, xv, y + 10), radius=3, fill=rgb(COLORS[method]))
            draw.text((xv + 8, y - 13), f"{val:.2f}", font=F["bar_value"], fill="#222222")
    save(img, "domain_method_shift_panels.png")


def plot_vlm_adaptation(base: pd.DataFrame) -> None:
    df = base[base["family"] == "vlm"].groupby(["task", "model_display", "method"], as_index=False)["auc"].mean()
    img = Image.new("RGB", (1400, 1450), "white")
    draw = ImageDraw.Draw(img)
    multiline_center(draw, (700, 42), "Dual Encoder Vision Language Model (VLM-encoders): Linear Probing vs Zero-Shot", F["title"], 1320, line_spacing=4)
    for i, task in enumerate(TASK_ORDER):
        task_df = df[df["task"] == task]
        order = task_df.groupby("model_display")["auc"].max().sort_values(ascending=False).index.tolist()
        panel = (60, 80 + i * 450, 1340, 485 + i * 450)
        x0, y0, x1, y1 = panel
        text_center(draw, ((x0 + x1) // 2, y0 + 22), TASK_DISPLAY[task], F["subtitle"])
        plot = (x0 + 185, y0 + 58, x1 - 55, y1 - 62)
        draw_x_axis(draw, plot, 0.35, 1.00, [0.4, 0.6, 0.8, 1.0], "Mean AUROC")
        left, top, right, bottom = plot
        row_h = (bottom - top) / max(1, len(order))
        for r, model in enumerate(order):
            cy = top + row_h * (r + 0.5)
            draw.text((x0 + 6, cy - 10), model, font=F["tick"], fill="#222222")
            for j, method in enumerate(["linear_probing", "zero_shot"]):
                rows = task_df[(task_df["model_display"] == model) & (task_df["method"] == method)]
                if rows.empty:
                    continue
                val = float(rows["auc"].iloc[0])
                y = cy + (j - 0.5) * 20
                xv = left + (val - 0.35) / 0.65 * (right - left)
                draw.rectangle((left, y - 8, xv, y + 8), fill=rgb(COLORS[method]))
    save(img, "vlm_adaptation_comparison.pdf")


def plot_fine_grained_performance(base: pd.DataFrame) -> None:
    grouped = base.groupby(["task", "model_type", "model_method"], as_index=False)["auc"].mean()
    img = Image.new("RGB", (1500, 1900), "white")
    draw = ImageDraw.Draw(img)
    text_center(draw, (750, 34), "Fine-Grained Model Performance", F["title"])
    for i, task in enumerate(TASK_ORDER):
        rows = grouped[grouped["task"] == task].sort_values("auc", ascending=False).head(14)
        panel = (50, 80 + i * 600, 1450, 640 + i * 600)
        hbar_panel(
            draw,
            panel,
            TASK_DISPLAY[task],
            rows["model_method"].tolist(),
            rows["auc"].tolist(),
            [COLORS[m] for m in rows["model_type"]],
            0.35,
            1.00,
        )
    save(img, "fine_grained_model_performance.pdf")


def plot_ft_delta_summary(ft: pd.DataFrame) -> None:
    ft = ft[ft["ft_method"] == "sft"].copy()
    summary = (
        ft.groupby(["domain"], as_index=False)[["delta_auc", "delta_accuracy", "delta_ece"]]
        .mean()
        .sort_values("domain")
    )
    summary["ft_method"] = "sft"
    if FT_DOMAIN_SIGNIFICANCE.exists():
        sig = pd.read_csv(FT_DOMAIN_SIGNIFICANCE)
        sig = sig[sig["ft_method"] == "sft"].copy()
        for metric in ["auc", "accuracy", "ece"]:
            delta_col = f"delta_{metric}"
            for col in ["ci_low", "ci_high", "p_boot", "q_boot"]:
                summary[f"{delta_col}_{col}"] = np.nan
            rows = sig[sig["metric"] == metric]
            for row in rows.itertuples():
                mask = summary["domain"] == row.domain
                summary.loc[mask, delta_col] = row.mean_delta
                summary.loc[mask, f"{delta_col}_ci_low"] = row.ci_low
                summary.loc[mask, f"{delta_col}_ci_high"] = row.ci_high
                summary.loc[mask, f"{delta_col}_p_boot"] = row.p_boot
                summary.loc[mask, f"{delta_col}_q_boot"] = row.q_boot

    axis_font = font(24, False)
    tick_font = font(22, False)
    value_font = font(20, True)
    img = Image.new("RGB", (1500, 660), "white")
    draw = ImageDraw.Draw(img)
    text_center(draw, (750, 34), "Language Model Fine-Tuning Deltas", F["title"])
    metrics = [
        ("delta_auc", "Delta AUROC", -0.008, 0.010),
        ("delta_accuracy", "Delta Accuracy", -0.005, 0.040),
        ("delta_ece", "Delta ECE", -0.045, 0.010),
    ]
    labels = [r.domain.replace("_", "-") for r in summary.itertuples()]
    for p, (metric, title, ymin, ymax) in enumerate(metrics):
        x0, y0, x1, y1 = 45 + p * 485, 90, 470 + p * 485, 590
        text_center(draw, ((x0 + x1) // 2, y0 + 22), title, F["subtitle"])
        plot = (x0 + 55, y0 + 60, x1 - 30, y1 - 85)
        left, top, right, bottom = plot
        def y_from_value(v: float) -> float:
            clipped = min(max(float(v), ymin), ymax)
            return bottom - (clipped - ymin) / (ymax - ymin) * (bottom - top)

        for tick in np.linspace(ymin, ymax, 5):
            y = y_from_value(tick)
            draw.line((left, y, right, y), fill="#E8E8E8", width=2)
            draw.text((x0 + 4, y - 8), f"{tick:.3f}", font=tick_font, fill="#333333")
        zero_y = y_from_value(0)
        draw.line((left, zero_y, right, zero_y), fill="#333333", width=2)
        group_w = (right - left) / len(summary)
        for i in range(len(summary)):
            row = summary.iloc[i]
            value = float(row[metric])
            cx = left + group_w * (i + 0.5)
            bar_w = group_w * 0.55
            yv = y_from_value(value)
            y_top, y_bottom = sorted([zero_y, yv])
            draw.rectangle((cx - bar_w / 2, y_top, cx + bar_w / 2, y_bottom), fill=rgb(COLORS[row["domain"]]))
            ci_low = row.get(f"{metric}_ci_low", np.nan)
            ci_high = row.get(f"{metric}_ci_high", np.nan)
            if np.isfinite(ci_low) and np.isfinite(ci_high):
                y_low = y_from_value(ci_low)
                y_high = y_from_value(ci_high)
                draw.line((cx, y_high, cx, y_low), fill="#222222", width=3)
                draw.line((cx - 8, y_high, cx + 8, y_high), fill="#222222", width=3)
                draw.line((cx - 8, y_low, cx + 8, y_low), fill="#222222", width=3)
            q_boot = row.get(f"{metric}_q_boot", np.nan)
            if np.isfinite(q_boot) and q_boot < 0.05:
                draw.text((cx - 5, min(y_top, yv) - 34), "*", font=F["subtitle"], fill="#222222")
            draw.text((cx - 28, y1 - 70), labels[i], font=axis_font, fill="#222222")
            draw.text((cx - 30, y_top - 26 if value >= 0 else y_bottom + 6), f"{value:.3f}", font=value_font, fill="#222222")
    save(img, "lm_ft_delta_summary.png")


def plot_ft_model_delta_heatmap(ft: pd.DataFrame) -> None:
    ft = ft[ft["ft_method"] == "sft"].copy()
    if FT_MODEL_SIGNIFICANCE.exists():
        sig = pd.read_csv(FT_MODEL_SIGNIFICANCE)
        sig = sig[sig["ft_method"] == "sft"].copy()
        sig["train_model_display"] = sig["train_model"].map(display_model)
        summary = (
            sig[sig["metric"].isin(["auc", "ece"])]
            .pivot_table(index=["train_model_display", "domain"], columns="metric", values="mean_delta", aggfunc="first")
            .reset_index()
            .rename(columns={"auc": "delta_auc", "ece": "delta_ece"})
        )
    else:
        summary = ft.groupby(["train_model_display", "domain"], as_index=False)[["delta_auc", "delta_ece"]].mean()
    rows = []
    models = summary.groupby("train_model_display")["delta_auc"].mean().sort_values(ascending=False).index
    for model in models:
        rows.append(model)
    matrix = pd.DataFrame(index=rows)
    for metric in ["delta_auc", "delta_ece"]:
        for domain in ["in_domain", "OOD"]:
            col = f"{domain.replace('_', '-')}\n{metric.replace('delta_', 'Delta ')}"
            vals = []
            for model in rows:
                sub = summary[(summary["train_model_display"] == model) & (summary["domain"] == domain)]
                vals.append(np.nan if sub.empty else float(sub[metric].iloc[0]))
            matrix[col] = vals
    vmax = float(np.nanmax(np.abs(matrix.to_numpy()))) or 0.01
    vmax = max(vmax, 0.015)
    h = 200 + 30 * len(matrix)
    img = Image.new("RGB", (1220, h), "white")
    draw = ImageDraw.Draw(img)
    def color(v: float) -> tuple[int, int, int]:
        return diverging_color(v, vmax)
    text_center(draw, (610, 42), "Language Model Fine-Tuning by Model", F["title"])
    draw_heatmap(
        draw,
        (45, 92, 1175, h - 45),
        "",
        matrix,
        color,
        ".3f",
        label_font=font(18, False),
        label_w_override=225,
        col_label_gap=34,
        row_label_align="right",
    )
    save(img, "lm_ft_model_delta_heatmap.png")


def plot_ft_task_delta_summary(ft: pd.DataFrame) -> None:
    ft = ft[ft["ft_method"] == "sft"].copy()
    if FT_TASK_SIGNIFICANCE.exists():
        sig = pd.read_csv(FT_TASK_SIGNIFICANCE)
        sig = sig[sig["ft_method"] == "sft"].copy()
        summary = (
            sig[sig["metric"].isin(["auc", "accuracy", "ece"])]
            .pivot_table(index=["train_task", "domain"], columns="metric", values="mean_delta", aggfunc="first")
            .reset_index()
            .rename(columns={"auc": "delta_auc", "accuracy": "delta_accuracy", "ece": "delta_ece"})
        )
    else:
        summary = ft.groupby(["train_task", "domain"], as_index=False)[["delta_auc", "delta_accuracy", "delta_ece"]].mean()
    img = Image.new("RGB", (1450, 535), "white")
    draw = ImageDraw.Draw(img)
    text_center(draw, (725, 34), "Language Model Fine-Tuning Deltas by Task", F["title"])
    panel = (55, 85, 1395, 455)
    x0, y0, x1, y1 = panel
    plot = (x0 + 145, y0 + 60, x1 - 45, y1 - 95)
    draw_x_axis(draw, plot, -0.02, 0.04, [-0.02, 0.00, 0.02, 0.04], "Delta AUROC")
    left, top, right, bottom = plot
    row_h = (bottom - top) / len(TASK_ORDER)
    for i, task in enumerate(TASK_ORDER):
        cy = top + row_h * (i + 0.5)
        draw.text((x0 + 6, cy - 10), TASK_DISPLAY[task], font=F["tick"], fill="#222222")
        for j, domain in enumerate(["in_domain", "OOD"]):
            sub = summary[(summary["train_task"] == task) & (summary["domain"] == domain)]
            if sub.empty:
                continue
            val = float(sub["delta_auc"].iloc[0])
            zero = left + (0 + 0.02) / 0.06 * (right - left)
            xv = left + (val + 0.02) / 0.06 * (right - left)
            y = cy + (j - 0.5) * 24
            draw.rectangle((min(zero, xv), y - 9, max(zero, xv), y + 9), fill=rgb(COLORS[domain]))
            draw.text((max(zero, xv) + 8, y - 13), f"{val:.3f}", font=F["bar_value"], fill="#222222")
    lx = x1 - 250
    for k, domain in enumerate(["in_domain", "OOD"]):
        ly = y0 + 18 + k * 26
        draw.rectangle((lx, ly, lx + 18, ly + 18), fill=rgb(COLORS[domain]))
        draw.text((lx + 26, ly - 1), domain.replace("_", "-"), font=F["small"], fill="#222222")
    save(img, "lm_ft_task_delta_summary.png")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    base = prepare_base()
    ft = prepare_ft()

    plot_model_level_auc_leaderboard(base)
    plot_model_type_reliability_matrix(base)
    plot_reliability_spiders(base)
    plot_mllm_size_tradeoff(base)
    plot_model_type_task_auc(base)
    plot_task_arena_leaderboards(base)
    plot_model_dataset_auc(base)
    plot_auc_ece_tradeoff(base)
    plot_calibration_family_grid(base, "cv", "calibration_cv_grid.pdf")
    plot_calibration_family_grid(base, "vlm", "calibration_vlm_grid.pdf")
    plot_calibration_family_grid(base, "mllm", "calibration_mllm_grid.pdf")
    plot_calibration_model_type(base)
    plot_calibration_curve_sheets(base)
    plot_dataset_family_metric_panels(base)
    plot_domain_method_shift_panels(base)
    plot_vlm_adaptation(base)
    plot_fine_grained_performance(base)
    plot_ft_delta_summary(ft)
    plot_ft_model_delta_heatmap(ft)
    plot_ft_task_delta_summary(ft)
    print(f"Wrote paper figures to {FIGURES}")


if __name__ == "__main__":
    main()
