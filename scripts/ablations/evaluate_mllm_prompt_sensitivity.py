#!/usr/bin/env python3
"""Run additive MLLM prompt or LoRA evaluations with one model load per job."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from retina_bench.ablations.mllm_prompt_parsing import (  # noqa: E402
    dataset_task_pairs,
    get_ground_truth_column,
    get_prompt_func,
    load_yaml,
    model_slug,
)
from retina_bench.core.data import RetinaDataset  # noqa: E402
from retina_bench.mllms.evaluate import get_model_class  # noqa: E402
from retina_bench.mllms.models import GptOssLLM  # noqa: E402


TOKENS = ["yes", "no", "Yes", "No"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--grid-index",
        type=int,
        help="Select model x prompt_variants from the config (for Slurm arrays).",
    )
    parser.add_argument("--model-id")
    parser.add_argument("--prompt-variant", choices=["base", "role", "instruction_first"])
    parser.add_argument("--dataset-key", default="datasets")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--train-dataset")
    parser.add_argument("--train-task")
    parser.add_argument("--train-strategy")
    parser.add_argument(
        "--evaluation-task",
        choices=["binary_dr", "referable_dr", "glaucoma"],
        help="Evaluate only dataset pairs supporting this task.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-flash-attn", action="store_true")
    return parser.parse_args()


def resolve_grid(args: argparse.Namespace, config: dict) -> tuple[str, str]:
    if args.grid_index is None:
        if not args.model_id or not args.prompt_variant:
            raise ValueError(
                "Provide --grid-index or both --model-id and --prompt-variant."
            )
        return args.model_id, args.prompt_variant

    grid = [
        (model_id, variant)
        for model_id in config["models"]
        for variant in config["prompt_variants"]
    ]
    if not 0 <= args.grid_index < len(grid):
        raise IndexError(
            f"Grid index {args.grid_index} is outside [0, {len(grid) - 1}]."
        )
    return grid[args.grid_index]


def output_path(
    output_dir: Path,
    model_id: str,
    dataset: str,
    task: str,
    variant: str,
    args: argparse.Namespace,
    split: str,
) -> Path:
    slug = model_slug(model_id)
    if args.adapter_path:
        required = [args.train_dataset, args.train_task, args.train_strategy]
        if not all(required):
            raise ValueError(
                "Adapter evaluation requires --train-dataset, --train-task, "
                "and --train-strategy."
            )
        name = (
            f"train_model-{slug}_train_dataset-{args.train_dataset}_"
            f"train_task-{args.train_task}_train_strategy-{args.train_strategy}"
            f"__test_model-{slug}_test_dataset-{dataset}_test_task-{task}_"
            f"test_strategy-{variant}_split-{split}.csv"
        )
    else:
        name = f"{dataset}_{task}_{variant}_{slug}.csv"
    return output_dir / name


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(
        path, mode="a", header=not path.exists(), index=False
    )


def evaluate_pair(
    model,
    args: argparse.Namespace,
    config: dict,
    model_id: str,
    dataset_name: str,
    task: str,
    prompt_variant: str,
) -> None:
    split = str(config.get("split", "test"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_path(
        output_dir, model_id, dataset_name, task, prompt_variant, args, split
    )
    partial_path = final_path.with_suffix(".partial.csv")

    if final_path.exists() and not args.overwrite:
        print(f"[skip] Completed output exists: {final_path}")
        return
    if args.overwrite:
        final_path.unlink(missing_ok=True)
        partial_path.unlink(missing_ok=True)

    completed_indices: set[int] = set()
    if partial_path.exists():
        partial = pd.read_csv(partial_path, usecols=["dataset_index"])
        completed_indices = set(partial["dataset_index"].astype(int))
        print(f"[resume] {len(completed_indices)} rows from {partial_path}")

    dataset = RetinaDataset(args.dataset_path, dataset_name, split=split)
    prompt_func = get_prompt_func(dataset_name, task, prompt_variant)
    gt_col = get_ground_truth_column(task)
    batch_size = int(config.get("batch_size", 16))
    max_new_tokens = int(config.get("max_new_tokens", 128))

    pending = [
        index
        for index in range(len(dataset))
        if index not in completed_indices
        and not pd.isna(dataset.get_row(index).get(gt_col))
    ]
    print(
        f"[eval] {model_id} | {dataset_name}/{task} | {prompt_variant} | "
        f"{len(pending)} pending"
    )

    for start in tqdm(range(0, len(pending), batch_size)):
        rows = []
        for index in pending[start : start + batch_size]:
            row = dataset.get_row(index)
            prompt = prompt_func(row)
            error = ""
            try:
                result = model.generate(
                    prompt=prompt,
                    image=dataset.get_image(index),
                    max_new_tokens=max_new_tokens,
                    return_logits=True,
                    tokens=TOKENS,
                )
                prediction_text = result.get("text", "")
                token_probs = result.get("token_probs", {}) or {}
            except Exception as exc:  # Preserve failures for coverage reporting.
                prediction_text = "ERROR"
                token_probs = {}
                error = f"{type(exc).__name__}: {exc}"

            rows.append(
                {
                    "dataset_index": index,
                    "id": row.get("image_id", row.get("file", str(index))),
                    "ground_truth": int(row[gt_col]),
                    "prediction_text": prediction_text,
                    "prompt": prompt,
                    "prompt_variant": prompt_variant,
                    "prob_yes": max(
                        token_probs.get("yes", 0.0) or 0.0,
                        token_probs.get("Yes", 0.0) or 0.0,
                    ),
                    "prob_no": max(
                        token_probs.get("no", 0.0) or 0.0,
                        token_probs.get("No", 0.0) or 0.0,
                    ),
                    "error": error,
                }
            )
        append_rows(partial_path, rows)

    # Atomic completion marker: a .csv means the full dataset/task finished.
    os.replace(partial_path, final_path)
    print(f"[done] {final_path}")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    model_id, prompt_variant = resolve_grid(args, config)
    quantization = config.get("quantization")

    model_class = get_model_class(model_id)
    if model_class == GptOssLLM and quantization is None:
        quantization = "4b"
    print(
        f"Loading {model_id} once (quantization={quantization}, "
        f"adapter={args.adapter_path or 'none'})"
    )
    model = model_class(
        model_id=model_id,
        quantization=quantization,
        use_flash_attention=args.use_flash_attn,
        adapter_path=args.adapter_path,
    )
    if args.adapter_path and not hasattr(model.model, "peft_config"):
        raise RuntimeError(
            f"The requested adapter was not attached to {model_id}: {args.adapter_path}"
        )

    pairs = dataset_task_pairs(config, args.dataset_key)
    if args.evaluation_task:
        pairs = [
            (dataset, task)
            for dataset, task in pairs
            if task == args.evaluation_task
        ]
    for dataset, task in pairs:
        evaluate_pair(
            model, args, config, model_id, dataset, task, prompt_variant
        )


if __name__ == "__main__":
    main()
