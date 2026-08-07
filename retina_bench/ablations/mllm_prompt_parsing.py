"""Shared utilities for the MLLM prompt and answer-parsing ablations."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from retina_bench.mllms import prompts as paper_prompts


ROLE_HEADER = (
    "You are an expert ophthalmologist evaluating a retinal fundus photograph."
)
EXACT_ANSWER_FRAGMENT = 'Respond only with "yes" or "no"'


def load_yaml(path: str | Path) -> dict:
    with Path(path).open() as handle:
        return yaml.safe_load(handle) or {}


def model_slug(model_id: str) -> str:
    return str(model_id).replace("/", "_").replace(" ", "_")


def dataset_task_pairs(config: dict, dataset_key: str = "datasets") -> list[tuple[str, str]]:
    datasets = config.get(dataset_key, config.get("datasets", []))
    tasks = config.get("tasks", [])
    allowed = config.get("dataset_tasks", {})
    return [
        (dataset, task)
        for dataset in datasets
        for task in allowed.get(dataset, tasks)
    ]


def get_ground_truth_column(task: str) -> str:
    columns = {
        "binary_dr": "DR_2_Class",
        "referable_dr": "Task_Referable",
        "glaucoma": "Task_Glaucoma",
    }
    try:
        return columns[task]
    except KeyError as exc:
        raise ValueError(f"Unsupported binary ablation task: {task}") from exc


def _move_answer_instruction_first(prompt: str) -> str:
    lines = prompt.splitlines()
    instruction = []
    body = []
    for line in lines:
        if EXACT_ANSWER_FRAGMENT.lower() in line.lower():
            instruction.append(line.strip())
        else:
            body.append(line)

    if not instruction:
        instruction = ['Respond only with "yes" or "no" (without additional commentary).']

    return "\n\n".join(
        [
            " ".join(part for part in instruction if part),
            "\n".join(body).strip(),
        ]
    ).strip()


def get_prompt_func(dataset: str, task: str, variant: str) -> Callable:
    """Wrap the paper's base prompt without changing its implementation."""
    base_func = paper_prompts.get_prompt_func(dataset, task, "base")
    if base_func is None:
        raise ValueError(f"No paper base prompt for {dataset}/{task}")

    if variant == "base":
        return base_func
    if variant == "role":
        return lambda row: f"{ROLE_HEADER}\n\n{base_func(row)}"
    if variant == "instruction_first":
        return lambda row: _move_answer_instruction_first(base_func(row))
    raise ValueError(f"Unknown prompt variant: {variant}")


_LABEL_RE = re.compile(r"\b(yes|no)\b", flags=re.IGNORECASE)
_STRICT_LABEL_RE = re.compile(r"^\s*(yes|no)\s*[.!]?\s*$", flags=re.IGNORECASE)


def parse_text(value: object, strategy: str) -> float:
    text = "" if value is None else str(value)
    if strategy == "strict_label":
        match = _STRICT_LABEL_RE.match(text)
    elif strategy == "first_label":
        match = _LABEL_RE.search(text)
    else:
        raise ValueError(f"Unknown text parser: {strategy}")
    if not match:
        return np.nan
    return 1.0 if match.group(1).lower() == "yes" else 0.0


def probability_score(frame: pd.DataFrame, strategy: str) -> pd.Series:
    yes_values = frame.get("prob_yes", pd.Series(np.nan, index=frame.index))
    yes = pd.to_numeric(yes_values, errors="coerce")
    if strategy == "raw_yes":
        return yes
    if strategy == "binary_renorm":
        no_values = frame.get("prob_no", pd.Series(np.nan, index=frame.index))
        no = pd.to_numeric(no_values, errors="coerce")
        denominator = yes + no
        return yes.div(denominator.where(denominator > 0))
    raise ValueError(f"Unknown probability score: {strategy}")


def hard_prediction(frame: pd.DataFrame, strategy: str) -> pd.Series:
    if strategy != "first_token_argmax":
        return frame["prediction_text"].map(
            lambda value: parse_text(value, strategy)
        )

    yes_values = frame.get("prob_yes", pd.Series(np.nan, index=frame.index))
    no_values = frame.get("prob_no", pd.Series(np.nan, index=frame.index))
    yes = pd.to_numeric(yes_values, errors="coerce")
    no = pd.to_numeric(no_values, errors="coerce")
    available = yes.notna() & no.notna() & ((yes + no) > 0)
    prediction = pd.Series(np.nan, index=frame.index, dtype=float)
    prediction.loc[available] = (yes.loc[available] >= no.loc[available]).astype(float)
    return prediction


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    # Put probability 1.0 in the final bin.
    assignments = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    ece = 0.0
    for bin_idx in range(bins):
        mask = assignments == bin_idx
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(
            float(labels[mask].mean()) - float(probabilities[mask].mean())
        )
    return ece


def evaluate_protocol(
    frame: pd.DataFrame,
    text_parser: str,
    score_strategy: str,
) -> tuple[dict, pd.DataFrame]:
    """Apply one parser/score protocol and return metrics plus row predictions."""
    parsed = frame.copy()
    parsed["label"] = pd.to_numeric(
        parsed.get("ground_truth", parsed.get("label")), errors="coerce"
    )
    parsed["pred"] = hard_prediction(parsed, text_parser)
    parsed["prob"] = probability_score(parsed, score_strategy)

    label_mask = parsed["label"].notna()
    pred_mask = label_mask & parsed["pred"].notna()
    score_mask = label_mask & parsed["prob"].notna()
    score_mask &= parsed["prob"].between(0.0, 1.0)
    valid_mask = pred_mask & score_mask

    labels_score = parsed.loc[valid_mask, "label"].to_numpy(dtype=float)
    probabilities = parsed.loc[valid_mask, "prob"].to_numpy(dtype=float)

    auc = np.nan
    auprc = np.nan
    if len(np.unique(labels_score)) == 2:
        auc = roc_auc_score(labels_score, probabilities)
        auprc = average_precision_score(labels_score, probabilities)

    metrics = {
        "n_total": int(label_mask.sum()),
        "n_text_parsed": int(pred_mask.sum()),
        "n_scored": int(score_mask.sum()),
        "n_valid": int(valid_mask.sum()),
        "text_parse_rate": float(pred_mask.sum() / label_mask.sum())
        if label_mask.any()
        else np.nan,
        "parse_error_rate": float(1.0 - pred_mask.sum() / label_mask.sum())
        if label_mask.any()
        else np.nan,
        "score_coverage": float(score_mask.sum() / label_mask.sum())
        if label_mask.any()
        else np.nan,
        "valid_rate": float(valid_mask.sum() / label_mask.sum())
        if label_mask.any()
        else np.nan,
        "auc": float(auc),
        "auprc": float(auprc),
        "accuracy": float(
            accuracy_score(
                parsed.loc[pred_mask, "label"], parsed.loc[pred_mask, "pred"]
            )
        )
        if pred_mask.any()
        else np.nan,
        "overall_accuracy": float(
            (
                parsed.loc[pred_mask, "label"]
                == parsed.loc[pred_mask, "pred"]
            ).sum()
            / label_mask.sum()
        )
        if label_mask.any()
        else np.nan,
        "ece": expected_calibration_error(labels_score, probabilities)
        if valid_mask.any()
        else np.nan,
        "brier": float(brier_score_loss(labels_score, probabilities))
        if valid_mask.any()
        else np.nan,
    }
    return metrics, parsed


def finite_or_none(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None
