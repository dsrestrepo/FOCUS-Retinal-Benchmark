#!/usr/bin/env python3
"""Evaluate additive zero-shot VLM label-prompt variants efficiently."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from retina_bench.core.data import RetinaDataset  # noqa: E402
from retina_bench.vlms.models import get_vlm_model  # noqa: E402


GROUND_TRUTH_COLUMNS = {
    "binary_dr": "DR_2_Class",
    "referable_dr": "Task_Referable",
    "glaucoma": "Task_Glaucoma",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--grid-index", type=int, required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
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


def variant_output_path(
    output_dir: Path,
    dataset: str,
    task: str,
    variant: str,
    model_id: str,
) -> Path:
    return output_dir / (
        f"{dataset}_{task}_zero_shot_{variant}_{model_slug(model_id)}.csv"
    )


def prompt_pair(variant: dict, task: str) -> tuple[str, str]:
    task_prompts = (variant.get("prompts", {}) or {}).get(task)
    if not task_prompts:
        raise ValueError(
            f"Prompt variant {variant.get('name')} has no prompts for {task}."
        )
    try:
        return str(task_prompts[0]), str(task_prompts[1])
    except KeyError:
        # YAML parsers may preserve explicitly quoted numeric keys as strings.
        return str(task_prompts["0"]), str(task_prompts["1"])


def write_atomic(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.csv")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def evaluate_pair(
    model,
    config: dict,
    model_id: str,
    dataset_name: str,
    task: str,
    dataset_path: str,
    output_dir: Path,
    overwrite: bool,
) -> None:
    variants = config["prompt_variants"]
    output_paths = {
        variant["name"]: variant_output_path(
            output_dir, dataset_name, task, variant["name"], model_id
        )
        for variant in variants
    }
    pending_variants = [
        variant
        for variant in variants
        if overwrite or not output_paths[variant["name"]].exists()
    ]
    if not pending_variants:
        print(f"[skip] {model_id} | {dataset_name}/{task}")
        return

    split = str(config.get("split", "test"))
    dataset = RetinaDataset(dataset_path, dataset_name, split=split)
    ground_truth_column = GROUND_TRUTH_COLUMNS[task]
    batch_size = int(config.get("batch_size", 16))
    logit_scale = float(config.get("logit_scale", 100.0))

    flattened_prompts = [
        prompt
        for variant in pending_variants
        for prompt in prompt_pair(variant, task)
    ]
    with torch.no_grad():
        text_embeddings = model.get_text_embeddings(flattened_prompts)
    text_embeddings = text_embeddings.reshape(
        len(pending_variants), 2, text_embeddings.shape[-1]
    )

    rows_by_variant: dict[str, list[dict]] = {
        variant["name"]: [] for variant in pending_variants
    }
    indices = [
        index
        for index in range(len(dataset))
        if not pd.isna(dataset.get_row(index).get(ground_truth_column))
    ]
    description = f"{model_slug(model_id)} {dataset_name}/{task}"
    for start in tqdm(range(0, len(indices), batch_size), desc=description):
        batch_indices = indices[start : start + batch_size]
        images = [dataset.get_image(index) for index in batch_indices]
        rows = [dataset.get_row(index) for index in batch_indices]
        with torch.no_grad():
            image_embeddings = model.get_image_embeddings(images)
            # V x B x 2: every variant compares the same image embeddings with
            # its negative and positive label-text embeddings.
            logits = torch.einsum(
                "bd,vcd->vbc", image_embeddings, text_embeddings
            )
            probabilities = torch.softmax(
                logits * logit_scale, dim=-1
            ).cpu().numpy()
            predictions = np.argmax(probabilities, axis=-1)

        for variant_index, variant in enumerate(pending_variants):
            negative_prompt, positive_prompt = prompt_pair(variant, task)
            destination_rows = rows_by_variant[variant["name"]]
            for row_index, (dataset_index, row) in enumerate(
                zip(batch_indices, rows)
            ):
                destination_rows.append(
                    {
                        "dataset_index": dataset_index,
                        "id": row.get(
                            "image_id", row.get("file", str(dataset_index))
                        ),
                        "ground_truth": int(row[ground_truth_column]),
                        "prediction": int(
                            predictions[variant_index, row_index]
                        ),
                        "prob_0": float(
                            probabilities[variant_index, row_index, 0]
                        ),
                        "prob_1": float(
                            probabilities[variant_index, row_index, 1]
                        ),
                        "prompt_variant": variant["name"],
                        "negative_prompt": negative_prompt,
                        "positive_prompt": positive_prompt,
                    }
                )

    for variant in pending_variants:
        name = variant["name"]
        write_atomic(pd.DataFrame(rows_by_variant[name]), output_paths[name])
        print(f"[done] {output_paths[name]}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    models = config["models"]
    if not 0 <= args.grid_index < len(models):
        raise IndexError(
            f"Grid index {args.grid_index} is outside [0, {len(models) - 1}]."
        )
    model_id = models[args.grid_index]
    print(f"Loading {model_id} once for all VLM prompt variants.")
    model = get_vlm_model(model_id, device="cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset, task in dataset_task_pairs(config):
        evaluate_pair(
            model=model,
            config=config,
            model_id=model_id,
            dataset_name=dataset,
            task=task,
            dataset_path=args.dataset_path,
            output_dir=output_dir,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
