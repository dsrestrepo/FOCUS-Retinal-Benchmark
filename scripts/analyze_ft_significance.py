#!/usr/bin/env python3
"""Paired uncertainty estimates for SFT/GRPO adapter deltas.

The FT aggregate table stores one row per adapter/test pair plus deltas
against the matching base MLLM. This script reopens the raw prediction files,
aligns the base and adapter outputs by image id, and estimates uncertainty for
the paired difference on the same examples.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METRICS = ["auc", "accuracy", "ece"]
LOWER_IS_BETTER = {"ece"}


def parse_binary_prediction_text(value: object) -> float:
    text = str(value).strip().lower()
    match = re.search(r"\b(yes|no)\b", text)
    if not match:
        return np.nan
    return 1.0 if match.group(1) == "yes" else 0.0


def normalize_result_columns(df: pd.DataFrame) -> pd.DataFrame:
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


def model_to_slug(model: object) -> str:
    return str(model).replace("/", "_").replace(" ", "_")


def short_model_name(model: object) -> str:
    return str(model).split("/")[-1]


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def locate_base_result(row: pd.Series, base_dir: Path) -> Path | None:
    slug = model_to_slug(row["test_model"])
    direct = base_dir / f"{row['test_dataset']}_{row['test_task']}_{slug}.csv"
    if direct.exists():
        return direct

    variants = [
        slug,
        slug.replace("-", "_"),
        re.sub(r"[^A-Za-z0-9_]+", "_", slug).strip("_"),
    ]
    for variant in dict.fromkeys(variants):
        path = base_dir / f"{row['test_dataset']}_{row['test_task']}_{variant}.csv"
        if path.exists():
            return path

    pattern = f"{row['test_dataset']}_{row['test_task']}_*{slug.split('_')[-1]}*.csv"
    matches = sorted(base_dir.glob(pattern))
    return matches[0] if matches else None


def load_prediction_file(path: Path, prefix: str) -> pd.DataFrame:
    df = normalize_result_columns(pd.read_csv(path))
    required = {"label", "pred", "prob"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} is missing normalized columns: {missing}")

    if "id" in df.columns:
        key = df["id"].astype(str)
    else:
        key = pd.Series(np.arange(len(df)), index=df.index).astype(str)

    out = pd.DataFrame(
        {
            "_key": key,
            f"label_{prefix}": df["label"],
            f"pred_{prefix}": df["pred"],
            f"prob_{prefix}": df["prob"],
        }
    )
    out["_occurrence"] = out.groupby("_key", sort=False).cumcount()
    return out


def merge_base_ft(base_path: Path, ft_path: Path) -> tuple[pd.DataFrame, int]:
    base = load_prediction_file(base_path, "base")
    ft = load_prediction_file(ft_path, "ft")
    merged = ft.merge(base, on=["_key", "_occurrence"], how="inner")

    valid = (
        merged["label_ft"].notna()
        & merged["label_base"].notna()
        & merged["pred_ft"].notna()
        & merged["pred_base"].notna()
        & merged["prob_ft"].notna()
        & merged["prob_base"].notna()
    )
    merged = merged[valid].copy()
    label_mismatch = int((merged["label_ft"] != merged["label_base"]).sum())
    merged = merged[merged["label_ft"] == merged["label_base"]].copy()
    return merged, label_mismatch


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)

    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = average_ranks(score)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return np.nan
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y_true[order]
    precision = np.cumsum(y_sorted == 1) / np.arange(1, len(y_sorted) + 1)
    return float(precision[y_sorted == 1].sum() / n_pos)


def f1_score_binary(y_true: np.ndarray, pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tp = int(((y_true == 1) & (pred == 1)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    denom = 2 * tp + fp + fn
    return np.nan if denom == 0 else float(2 * tp / denom)


def expected_calibration_error(y_true: np.ndarray, score: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=int)
    score = np.clip(np.asarray(score, dtype=float), 0.0, 1.0)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan

    binids = np.minimum((score * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if not mask.any():
            continue
        ece += abs(float(score[mask].mean()) - float(y_true[mask].mean())) * mask.sum() / len(y_true)
    return float(ece)


def compute_metric(metric: str, y_true: np.ndarray, pred: np.ndarray, score: np.ndarray) -> float:
    if metric == "auc":
        return roc_auc(y_true, score)
    if metric == "auprc":
        return average_precision(y_true, score)
    if metric == "accuracy":
        return float(np.mean(np.asarray(y_true, dtype=int) == np.asarray(pred, dtype=int)))
    if metric == "f1":
        return f1_score_binary(y_true, pred)
    if metric == "ece":
        return expected_calibration_error(y_true, score)
    raise ValueError(f"Unknown metric: {metric}")


def bootstrap_p_value(samples: np.ndarray) -> float:
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return np.nan
    lower = (float((samples <= 0).sum()) + 1.0) / (len(samples) + 1.0)
    upper = (float((samples >= 0).sum()) + 1.0) / (len(samples) + 1.0)
    return float(min(1.0, 2.0 * min(lower, upper)))


def paired_bootstrap_delta(
    y_true: np.ndarray,
    pred_base: np.ndarray,
    score_base: np.ndarray,
    pred_ft: np.ndarray,
    score_ft: np.ndarray,
    metric: str,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    base_value = compute_metric(metric, y_true, pred_base, score_base)
    ft_value = compute_metric(metric, y_true, pred_ft, score_ft)
    delta = ft_value - base_value

    samples: list[float] = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        base_boot = compute_metric(metric, y_true[idx], pred_base[idx], score_base[idx])
        ft_boot = compute_metric(metric, y_true[idx], pred_ft[idx], score_ft[idx])
        sample = ft_boot - base_boot
        if np.isfinite(sample):
            samples.append(float(sample))

    sample_arr = np.asarray(samples, dtype=float)
    if len(sample_arr) == 0:
        ci_low = ci_high = p_boot = np.nan
    else:
        ci_low, ci_high = np.percentile(sample_arr, [2.5, 97.5])
        p_boot = bootstrap_p_value(sample_arr)

    direction = -1.0 if metric in LOWER_IS_BETTER else 1.0
    return {
        "base_metric": base_value,
        "ft_metric": ft_value,
        "delta": delta,
        "improvement": direction * delta,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_boot": p_boot,
        "boot_valid": int(len(sample_arr)),
    }


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    q_values = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna()
    if valid.empty:
        return q_values

    ordered = valid.sort_values()
    n = len(ordered)
    adjusted = ordered.to_numpy() * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q_values.loc[ordered.index] = adjusted
    return q_values


def add_q_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["q_boot"] = np.nan
    for metric, idx in df.groupby("metric").groups.items():
        df.loc[idx, "q_boot"] = benjamini_hochberg(df.loc[idx, "p_boot"])
    return df


def summarize_deltas(
    comparisons: pd.DataFrame,
    group_cols: list[str],
    n_boot: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    if comparisons.empty:
        return pd.DataFrame()

    for keys, group in comparisons.groupby(group_cols + ["metric"], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = keys[:-1]
        metric = keys[-1]
        deltas = group["delta"].dropna().to_numpy(dtype=float)
        improvements = group["improvement"].dropna().to_numpy(dtype=float)
        if len(deltas) == 0:
            continue

        boot = np.asarray(
            [rng.choice(deltas, size=len(deltas), replace=True).mean() for _ in range(n_boot)],
            dtype=float,
        )
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
        record = {col: value for col, value in zip(group_cols, key_values)}
        record.update(
            {
                "metric": metric,
                "n_comparisons": int(len(deltas)),
                "mean_delta": float(deltas.mean()),
                "median_delta": float(np.median(deltas)),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "p_boot": bootstrap_p_value(boot),
                "win_rate": float((improvements > 0).mean()) if len(improvements) else np.nan,
            }
        )
        records.append(record)

    summary = pd.DataFrame(records)
    if summary.empty:
        return summary

    summary["q_boot"] = np.nan
    for metric, idx in summary.groupby("metric").groups.items():
        summary.loc[idx, "q_boot"] = benjamini_hochberg(summary.loc[idx, "p_boot"])
    return summary


def analyze(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    ft_metrics = pd.read_csv(args.ft_metrics)
    if args.models:
        ft_metrics = ft_metrics[ft_metrics["train_model"].isin(args.models)].copy()

    ft_metrics["domain"] = np.where(bool_series(ft_metrics["is_ood_dataset"]), "OOD", "in_domain")
    rng = np.random.default_rng(args.seed)
    records: list[dict[str, object]] = []

    for _, row in ft_metrics.iterrows():
        ft_path = Path(row["result_file"])
        base_path = locate_base_result(row, args.base_dir)
        if base_path is None or not ft_path.exists():
            records.append(
                {
                    "ft_method": row["ft_method"],
                    "train_model": row["train_model"],
                    "train_dataset": row["train_dataset"],
                    "train_task": row["train_task"],
                    "test_model": row["test_model"],
                    "test_dataset": row["test_dataset"],
                    "test_task": row["test_task"],
                    "domain": row["domain"],
                    "metric": "__missing_file__",
                    "n_pairs": 0,
                    "missing_base_file": base_path is None,
                    "missing_ft_file": not ft_path.exists(),
                }
            )
            continue

        try:
            merged, label_mismatch = merge_base_ft(base_path, ft_path)
        except Exception as exc:
            records.append(
                {
                    "ft_method": row["ft_method"],
                    "train_model": row["train_model"],
                    "train_dataset": row["train_dataset"],
                    "train_task": row["train_task"],
                    "test_model": row["test_model"],
                    "test_dataset": row["test_dataset"],
                    "test_task": row["test_task"],
                    "domain": row["domain"],
                    "metric": "__read_error__",
                    "n_pairs": 0,
                    "error": str(exc),
                    "base_result_file": str(base_path),
                    "ft_result_file": str(ft_path),
                }
            )
            continue

        if merged.empty:
            continue

        y_true = merged["label_ft"].to_numpy(dtype=int)
        pred_base = merged["pred_base"].to_numpy(dtype=float)
        score_base = merged["prob_base"].to_numpy(dtype=float)
        pred_ft = merged["pred_ft"].to_numpy(dtype=float)
        score_ft = merged["prob_ft"].to_numpy(dtype=float)

        for metric in args.metrics:
            stats = paired_bootstrap_delta(
                y_true,
                pred_base,
                score_base,
                pred_ft,
                score_ft,
                metric,
                args.n_boot,
                rng,
            )
            records.append(
                {
                    "ft_method": row["ft_method"],
                    "train_model": row["train_model"],
                    "train_model_short": short_model_name(row["train_model"]),
                    "train_dataset": row["train_dataset"],
                    "train_task": row["train_task"],
                    "train_strategy": row.get("train_strategy", ""),
                    "test_model": row["test_model"],
                    "test_dataset": row["test_dataset"],
                    "test_task": row["test_task"],
                    "test_strategy": row.get("test_strategy", ""),
                    "domain": row["domain"],
                    "metric": metric,
                    "n_pairs": int(len(merged)),
                    "n_positive": int((y_true == 1).sum()),
                    "n_negative": int((y_true == 0).sum()),
                    "label_mismatch": label_mismatch,
                    "base_result_file": str(base_path),
                    "ft_result_file": str(ft_path),
                    **stats,
                }
            )

    comparisons = add_q_values(pd.DataFrame(records))
    comparisons = comparisons[~comparisons["metric"].astype(str).str.startswith("__")].copy()

    summaries = {
        "domain_significance": summarize_deltas(
            comparisons,
            ["ft_method", "domain"],
            args.n_boot,
            rng,
        ),
        "model_significance": summarize_deltas(
            comparisons,
            ["ft_method", "train_model", "train_model_short", "domain"],
            args.n_boot,
            rng,
        ),
        "task_significance": summarize_deltas(
            comparisons,
            ["ft_method", "train_task", "domain"],
            args.n_boot,
            rng,
        ),
    }
    return comparisons, summaries


def write_outputs(
    comparisons: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    output_dir: Path,
    paper_table_dir: Path | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(output_dir / "comparison_significance.csv", index=False)
    for name, df in summaries.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)

    if paper_table_dir is not None:
        paper_table_dir.mkdir(parents=True, exist_ok=True)
        domain = summaries["domain_significance"].copy()
        if not domain.empty:
            keep = ["ft_method", "domain", "metric", "n_comparisons", "mean_delta", "ci_low", "ci_high", "p_boot", "q_boot", "win_rate"]
            domain[keep].round(4).to_csv(
                paper_table_dir / "ft_lora_significance_domain_summary.csv",
                index=False,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate paired uncertainty for FT-vs-base MLLM deltas")
    parser.add_argument("--ft_metrics", type=Path, default=Path("results/analysis_ft/aggregated_metrics.csv"))
    parser.add_argument("--base_dir", type=Path, default=Path("results/evals"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/analysis_ft/significance"))
    parser.add_argument("--paper_table_dir", type=Path, default=Path("paper/tables"))
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, choices=["auc", "auprc", "accuracy", "f1", "ece"])
    parser.add_argument("--models", nargs="*", default=None, help="Optional train_model ids to include")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    comparisons, summaries = analyze(args)
    write_outputs(comparisons, summaries, args.output_dir, args.paper_table_dir)
    print(f"Wrote FT significance analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
