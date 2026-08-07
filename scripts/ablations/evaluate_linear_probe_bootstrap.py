#!/usr/bin/env python3
"""Refit representative frozen-feature linear probes on training bootstraps."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from retina_bench.core.data import RetinaDataset
from retina_bench.cv.models import get_cv_model
from retina_bench.vlms.models import get_vlm_model


TARGET_COLUMNS = {
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


def load_config(path: str) -> dict:
    with Path(path).open() as handle:
        return yaml.safe_load(handle) or {}


def model_slug(model_id: str) -> str:
    return (
        str(model_id)
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def row_id(row: pd.Series, index: int) -> str:
    return str(row.get("image_id", row.get("file", index)))


def extract_features(
    model,
    dataset: RetinaDataset,
    batch_size: int,
    description: str,
) -> tuple[np.ndarray, list[pd.Series]]:
    features: list[np.ndarray] = []
    rows: list[pd.Series] = []
    for start in tqdm(
        range(0, len(dataset), batch_size), desc=description
    ):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        images = [dataset.get_image(index) for index in indices]
        with torch.no_grad():
            embeddings = model.get_image_embeddings(images).cpu().numpy()
        features.append(embeddings)
        rows.extend(dataset.get_row(index) for index in indices)
    if not features:
        return np.empty((0, 0)), rows
    return np.vstack(features), rows


def labels_for_task(rows: list[pd.Series], task: str) -> np.ndarray:
    target = TARGET_COLUMNS[task]
    return np.asarray(
        [
            int(row.get(target)) if pd.notna(row.get(target)) else -1
            for row in rows
        ],
        dtype=int,
    )


def stratified_bootstrap_indices(
    labels: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    sampled = []
    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        sampled.append(
            rng.choice(class_indices, size=len(class_indices), replace=True)
        )
    indices = np.concatenate(sampled)
    rng.shuffle(indices)
    return indices


def result_is_complete(path: Path, n_bootstraps: int) -> bool:
    if not path.exists():
        return False
    try:
        indices = set(
            pd.read_csv(path, usecols=["bootstrap_index"])[
                "bootstrap_index"
            ].astype(int)
        )
    except Exception:
        return False
    return indices == {-1, *range(n_bootstraps)}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    models = config["models"]
    if not 0 <= args.grid_index < len(models):
        raise IndexError(
            f"Grid index {args.grid_index} is outside [0, {len(models) - 1}]."
        )
    model_config = models[args.grid_index]
    model_id = model_config["id"]
    family = model_config["family"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if family == "vm":
        model = get_cv_model(
            model_id,
            device="cuda",
            pooling=str(config.get("pooling", "cls")),
        )
    elif family == "vlm":
        model = get_vlm_model(model_id, device="cuda")
    else:
        raise ValueError(f"Unsupported model family: {family}")

    batch_size = int(config.get("batch_size", 16))
    n_bootstraps = int(config.get("n_train_bootstraps", 10))
    base_seed = int(config.get("base_seed", 3199))
    evaluations_by_dataset: dict[str, list[str]] = {}
    for evaluation in config["evaluations"]:
        evaluations_by_dataset.setdefault(evaluation["dataset"], []).append(
            evaluation["task"]
        )

    for dataset_name, tasks in evaluations_by_dataset.items():
        train_dataset = RetinaDataset(
            args.dataset_path,
            dataset_name,
            split=str(config.get("split_train", "train")),
        )
        test_dataset = RetinaDataset(
            args.dataset_path,
            dataset_name,
            split=str(config.get("split_test", "test")),
        )
        train_features, train_rows = extract_features(
            model,
            train_dataset,
            batch_size,
            f"{model_config['display_name']} {dataset_name} train",
        )
        test_features, test_rows = extract_features(
            model,
            test_dataset,
            batch_size,
            f"{model_config['display_name']} {dataset_name} test",
        )

        for task in tasks:
            output_path = output_dir / (
                f"{dataset_name}_{task}_{model_slug(model_id)}_bootstrap.csv"
            )
            if (
                result_is_complete(output_path, n_bootstraps)
                and not args.overwrite
            ):
                print(f"[skip] {output_path}")
                continue
            if output_path.exists():
                print(f"[replace incomplete/legacy result] {output_path}")

            train_labels_all = labels_for_task(train_rows, task)
            test_labels_all = labels_for_task(test_rows, task)
            train_valid = train_labels_all != -1
            test_valid = test_labels_all != -1
            train_x = train_features[train_valid]
            train_y = train_labels_all[train_valid]
            test_x = test_features[test_valid]
            test_y = test_labels_all[test_valid]
            test_indices = np.flatnonzero(test_valid)

            if len(np.unique(train_y)) != 2 or len(np.unique(test_y)) != 2:
                raise RuntimeError(
                    f"{dataset_name}/{task} requires two classes in train and test."
                )

            prediction_frames = []
            for bootstrap_index in [-1, *range(n_bootstraps)]:
                is_reference = bootstrap_index == -1
                seed = (
                    base_seed - 1
                    if is_reference
                    else base_seed + bootstrap_index
                )
                if is_reference:
                    sampled = np.arange(len(train_y))
                else:
                    rng = np.random.default_rng(seed)
                    sampled = stratified_bootstrap_indices(train_y, rng)
                classifier = LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=seed,
                )
                classifier.fit(train_x[sampled], train_y[sampled])
                probability = classifier.predict_proba(test_x)[:, 1]
                prediction = classifier.predict(test_x)
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "model": model_id,
                            "model_display": model_config["display_name"],
                            "family": family,
                            "domain": model_config["domain"],
                            "dataset": dataset_name,
                            "task": task,
                            "bootstrap_index": bootstrap_index,
                            "seed": seed,
                            "resample": (
                                "full_train"
                                if is_reference
                                else "stratified_bootstrap"
                            ),
                            "dataset_index": test_indices,
                            "id": [
                                row_id(test_rows[index], index)
                                for index in test_indices
                            ],
                            "label": test_y,
                            "pred": prediction,
                            "prob": probability,
                        }
                    )
                )
            temporary_path = output_path.with_suffix(".partial.csv")
            pd.concat(prediction_frames, ignore_index=True).to_csv(
                temporary_path, index=False
            )
            os.replace(temporary_path, output_path)
            print(f"[done] {output_path}")


if __name__ == "__main__":
    main()
