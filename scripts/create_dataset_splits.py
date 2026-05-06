#!/usr/bin/env python3
"""Materialize reproducible benchmark split IDs for prepared fundus manifests."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_FUNDUS_CONFIG = ROOT / "config" / "fundus_datasets.yaml"
DEFAULT_MODEL_CONFIG = ROOT / "config" / "analysis_config.yaml"
DEFAULT_ENV = ROOT / "config" / "paths.env"
DEFAULT_SPLIT_DIR = ROOT / "Split_Data"
TASK_COLUMNS = ["DR_2_Class", "Task_DR_Binary", "Task_Referable", "Task_Glaucoma"]
NAMED_SPLITS = {"train", "val", "test"}
DEFAULT_TEST_SIZE = 0.4
DEFAULT_SEED = 42
MANIFEST_STORAGE_FALLBACKS = {
    "brset": "BRSET/brset",
    "mbrset": "mBRSET/mbrset",
}


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def output_name(dataset_key: str) -> str:
    return f"labels_splits_{dataset_key}.csv"


def manifest_path_for(dataset_key: str, dataset_cfg: dict, data_dir: Path) -> Path:
    storage_dir = dataset_cfg.get("download", {}).get("storage_dir")
    if not storage_dir:
        storage_dir = MANIFEST_STORAGE_FALLBACKS.get(dataset_key, dataset_key)
    return data_dir / storage_dir / "prepared" / "manifest.csv"


def split_count(n_rows: int, test_size: float) -> int:
    if n_rows <= 1:
        return 0
    return min(n_rows - 1, max(1, round(n_rows * test_size)))


def stratification_key(df: pd.DataFrame) -> pd.Series | None:
    strat_cols = []
    for col in TASK_COLUMNS:
        if col not in df.columns:
            continue
        values = df[col]
        valid = values.dropna()
        if valid.nunique() > 1:
            strat_cols.append(col)

    if not strat_cols:
        return None

    key = df[strat_cols].astype("string").fillna("missing").agg("|".join, axis=1)
    counts = key.value_counts()
    if len(counts) > 1 and counts.min() >= 2:
        return key

    # Combined multi-task labels can be too sparse. Fall back to the first
    # usable single task label so each class can appear in train and test.
    for col in strat_cols:
        key = df[col].astype("string").fillna("missing")
        counts = key.value_counts()
        if len(counts) > 1 and counts.min() >= 2:
            return key
    return None


def make_train_test_split(df: pd.DataFrame, test_size: float, seed: int) -> pd.DataFrame:
    if not 0 < test_size < 1:
        raise ValueError("--test-size must be between 0 and 1")

    df = df.copy().reset_index(drop=True)
    key = stratification_key(df)
    test_indices: list[int] = []

    if key is None:
        test_indices = list(df.sample(n=split_count(len(df), test_size), random_state=seed).index)
    else:
        keyed = df.assign(_strat_key=key)
        for group_idx, (_, group) in enumerate(keyed.groupby("_strat_key", sort=True, dropna=False)):
            n_test = split_count(len(group), test_size)
            if n_test == 0:
                continue
            test_indices.extend(group.sample(n=n_test, random_state=seed + group_idx).index.tolist())

    out = df.copy()
    out["split"] = "train"
    out.loc[test_indices, "split"] = "test"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fundus-config", type=Path, default=DEFAULT_FUNDUS_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--datasets", nargs="*", help="Dataset keys to materialize. Defaults to configured benchmark datasets.")
    parser.add_argument(
        "--include-native-splits",
        action="store_true",
        help="Deprecated compatibility flag. Native split datasets are exported by default.",
    )
    parser.add_argument(
        "--skip-native-splits",
        action="store_true",
        help="Do not export datasets whose prepared manifests already contain train/val/test splits.",
    )
    parser.add_argument(
        "--verify-loading",
        action="store_true",
        help="After writing split files, instantiate RetinaDataset for train/test/all and print row counts.",
    )
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE, help="Test fraction for manifests without native splits.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic split seed for manifests without native splits.")
    parser.add_argument(
        "--legacy-full-split",
        action="store_true",
        help="For unsplit manifests, reproduce the old full-train/full-test behavior. This is not recommended for linear probing.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    env = {**os.environ, **load_env(args.env_file)}
    data_dir = (args.data_dir or Path(env.get("DATA_PATH", "data"))).expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fundus_cfg = yaml.safe_load(args.fundus_config.read_text())
    model_cfg = yaml.safe_load(args.model_config.read_text()) if args.model_config.exists() else {}
    dataset_keys = args.datasets or model_cfg.get("datasets") or [
        key for key, value in fundus_cfg.get("datasets", {}).items() if value.get("status") == "included"
    ]

    print(f"[data] {data_dir}")
    print(f"[splits] {args.output_dir}")
    print(f"[mode] canonical benchmark split IDs; real deterministic splits for unsplit manifests (seed={args.seed})")

    failures = []
    for dataset_key in dataset_keys:
        if dataset_key in {"brset", "mbrset"}:
            print(f"[skip] {dataset_key}: fixed split file already maintained manually")
            continue

        dataset_cfg = fundus_cfg.get("datasets", {}).get(dataset_key)
        if not dataset_cfg:
            print(f"[skip] {dataset_key}: missing from fundus config")
            continue

        manifest_path = manifest_path_for(dataset_key, dataset_cfg, data_dir)
        out_path = args.output_dir / output_name(dataset_key)
        if out_path.exists() and not args.force:
            print(f"[skip] {dataset_key}: {out_path} already exists")
            continue
        if not manifest_path.exists():
            failures.append((dataset_key, f"missing prepared manifest: {manifest_path}"))
            continue

        df = pd.read_csv(manifest_path)
        if "image_id" not in df.columns:
            failures.append((dataset_key, f"manifest missing image_id: {manifest_path}"))
            continue

        split_values = df["split"].astype("string").str.lower().str.strip() if "split" in df.columns else pd.Series(pd.NA, index=df.index)
        split_values = split_values.replace({"training": "train", "validation": "val", "valid": "val", "nan": pd.NA, "none": pd.NA})
        existing = set(split_values.dropna().unique())

        base = pd.DataFrame({"dataset": dataset_key, "image_id": df["image_id"].astype(str)})
        if "image_path" in df.columns:
            base["image_path"] = df["image_path"].astype(str)
        for col in TASK_COLUMNS:
            if col in df.columns:
                base[col] = df[col]

        named = existing.intersection(NAMED_SPLITS)
        if named:
            if args.skip_native_splits:
                print(f"[skip] {dataset_key}: prepared manifest already has named splits {sorted(named)}")
                continue
            out = base.copy()
            out["split"] = split_values.fillna("all")
        elif args.legacy_full_split:
            out = pd.concat(
                [
                    base.assign(split="train"),
                    base.assign(split="test"),
                ],
                ignore_index=True,
            )
        else:
            out = make_train_test_split(base, test_size=args.test_size, seed=args.seed)

        preferred = ["dataset", "image_id", "split", "image_path"]
        columns = [col for col in preferred if col in out.columns] + [col for col in out.columns if col not in preferred]
        out = out[columns]
        out.to_csv(out_path, index=False)
        counts = out["split"].value_counts().to_dict()
        print(f"[write] {dataset_key}: {len(out)} rows -> {out_path} {counts}")

    if failures:
        print("\nFailures:")
        for dataset_key, reason in failures:
            print(f"- {dataset_key}: {reason}")
        return 1

    if args.verify_loading:
        import retina_bench.core.data as data_module
        from retina_bench.core.data import RetinaDataset

        data_module.SPLIT_DATA_DIR = args.output_dir
        print("\n[verify] RetinaDataset split counts")
        for dataset_key in dataset_keys:
            if dataset_key in {"brset", "mbrset"}:
                continue
            dataset_cfg = fundus_cfg.get("datasets", {}).get(dataset_key)
            if not dataset_cfg:
                continue
            manifest_path = manifest_path_for(dataset_key, dataset_cfg, data_dir)
            if not manifest_path.exists():
                continue
            counts = {}
            try:
                for split in ("train", "test", "all"):
                    counts[split] = len(RetinaDataset(data_dir, dataset_key, split=split))
            except Exception as exc:
                failures.append((dataset_key, f"load verification failed: {exc}"))
                continue
            print(f"[verify] {dataset_key}: {counts}")

    if failures:
        print("\nFailures:")
        for dataset_key, reason in failures:
            print(f"- {dataset_key}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
