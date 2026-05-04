#!/usr/bin/env python3
"""Download open fundus benchmark datasets from the project manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "fundus_datasets.yaml"
DEFAULT_ENV = ROOT / "config" / "paths.env"
DEFAULT_SECRET_ENV = ROOT / ".env"
DEFAULT_SPLIT_DIR = ROOT / "Split_Data"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
DATA_SUFFIXES = IMAGE_SUFFIXES | {".csv", ".xlsx", ".xls", ".json", ".txt", ".zip"}
MANIFEST_COLUMNS = [
    "dataset",
    "split",
    "image_path",
    "image_id",
    "label",
    "Task_DR_Binary",
    "Task_Referable",
    "Task_Glaucoma",
    "label_source",
    "source",
]


def load_env(path: Path) -> dict[str, str]:
    env = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def merged_env(*paths: Path) -> dict[str, str]:
    env = dict(os.environ)
    for path in paths:
        env.update(load_env(path))
    return env


def run(cmd: list[str], dry_run: bool, redactions: set[str] | None = None) -> None:
    redactions = redactions or set()
    printable = ["********" if part in redactions else part for part in cmd]
    print("+", " ".join(printable))
    if not dry_run:
        subprocess.run(cmd, check=True)


def urlopen_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "retina-bench-dataset-downloader"})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def download_url(url: str, output: Path, dry_run: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        print(f"[skip] {output} already exists")
        return
    print(f"[download] {url} -> {output}")
    if dry_run:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "retina-bench-dataset-downloader"})
    with urllib.request.urlopen(request) as response:
        with output.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def maybe_extract_zip(path: Path, target_dir: Path, dry_run: bool) -> None:
    if path.suffix.lower() != ".zip":
        return
    marker = target_dir / ".extracted"
    if marker.exists():
        print(f"[skip] {path.name} already extracted")
        return
    print(f"[extract] {path} -> {target_dir}")
    if dry_run:
        return
    with zipfile.ZipFile(path) as archive:
        archive.extractall(target_dir)
    marker.write_text(path.name + "\n")


def target_dir_for(dataset_key: str, dataset: dict, output_root: Path) -> Path:
    download = dataset.get("download", {})
    return output_root / download.get("storage_dir", dataset_key)


def is_prepared(dataset: dict, target_dir: Path) -> bool:
    prepared_paths = dataset.get("download", {}).get("prepared_paths", [])
    if not prepared_paths:
        return has_downloaded_data(target_dir)
    return all((target_dir / path).exists() for path in prepared_paths)


def download_is_complete(target_dir: Path) -> bool:
    return has_downloaded_data(target_dir)


def prepare_is_complete(dataset: dict, target_dir: Path) -> bool:
    prepared_paths = dataset.get("download", {}).get("prepared_paths", [])
    return bool(prepared_paths) and all((target_dir / path).exists() for path in prepared_paths)


def has_downloaded_data(target_dir: Path) -> bool:
    if not target_dir.exists():
        return False
    if (target_dir / ".download_complete").exists():
        return True
    data_files = [
        path
        for path in target_dir.rglob("*")
        if path.is_file()
        and path.name != "MANUAL_DOWNLOAD.md"
        and path.suffix.lower() in DATA_SUFFIXES
    ]
    return bool(data_files)


def mark_complete(target_dir: Path, dry_run: bool) -> None:
    if dry_run:
        return
    (target_dir / ".download_complete").write_text("ok\n")


def mark_prepared(target_dir: Path, dry_run: bool) -> None:
    if dry_run:
        return
    prepared_dir = target_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    (prepared_dir / ".preprocess_complete").write_text("ok\n")


def image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def csv_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".csv")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with path.open(newline="", encoding=encoding) as handle:
                rows = []
                for row in csv.DictReader(handle):
                    rows.append({normalize_cell(key): normalize_cell(value) for key, value in row.items()})
                return rows
        except UnicodeDecodeError:
            continue
    with path.open(newline="", encoding="latin-1", errors="replace") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append({normalize_cell(key): normalize_cell(value) for key, value in row.items()})
        return rows


def write_manifest(target_dir: Path, rows: list[dict[str, str]], dry_run: bool) -> None:
    manifest = target_dir / "prepared" / "manifest.csv"
    print(f"[prepare] write {len(rows)} rows -> {manifest}")
    if dry_run:
        return
    add_standard_tasks(rows)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(MANIFEST_COLUMNS)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    mark_prepared(target_dir, dry_run)


def write_rows_csv(path: Path, rows: list[dict[str, str]], dry_run: bool) -> None:
    print(f"[prepare] write {len(rows)} rows -> {path}")
    if dry_run:
        return
    add_standard_tasks(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_csv_if_missing(source: Path, dest: Path, dry_run: bool) -> None:
    if dest.exists():
        print(f"[skip] {dest} already exists")
        return
    print(f"[prepare] copy {source} -> {dest}")
    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)


def copy_split_file_if_missing(dataset_key: str, target_dir: Path, dry_run: bool) -> None:
    split_sources = {
        "brset": DEFAULT_SPLIT_DIR / "labels_splits_brset.csv",
        "mbrset": DEFAULT_SPLIT_DIR / "labels_splits_mbrset.csv",
    }
    source = split_sources.get(dataset_key)
    if source is None:
        return

    dest = target_dir / "labels_splits.csv"
    if dest.exists():
        print(f"[skip] {dest} already exists")
        return
    if not source.exists():
        raise RuntimeError(
            f"Missing split file for {dataset_key}: expected {source}. "
            f"Place the benchmark split CSV in Split_Data before preprocessing."
        )
    copy_csv_if_missing(source, dest, dry_run)


def find_first(root: Path, name: str) -> Path | None:
    for path in root.rglob(name):
        return path
    return None


def find_image_dir(root: Path, preferred_names: list[str]) -> Path | None:
    for preferred_name in preferred_names:
        for path in root.rglob(preferred_name):
            if path.is_dir() and any(child.suffix.lower() in IMAGE_SUFFIXES for child in path.rglob("*")):
                return path
    for path in root.rglob("*"):
        if path.is_dir() and any(child.suffix.lower() in IMAGE_SUFFIXES for child in path.rglob("*")):
            return path
    return None


def resize_images_to_224(source_dir: Path, dest_dir: Path, dry_run: bool) -> None:
    images = [path for path in source_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES]
    if not images:
        raise RuntimeError(f"No images found under {source_dir}")
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"[skip] {dest_dir} already exists and is not empty")
        return

    print(f"[prepare] resizing {len(images)} images from {source_dir} into {dest_dir}")
    if dry_run:
        return

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to create images_224. Install pillow in the download environment.") from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        output_name = image_path.with_suffix(".jpg").name
        output_path = dest_dir / output_name
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
            image.save(output_path, quality=95)


def manifest_from_images(dataset_key: str, image_paths: list[Path], split: str = "", label: str = "", label_source: str = "") -> list[dict[str, str]]:
    return [
        {
            "dataset": dataset_key,
            "split": split,
            "image_path": str(path),
            "image_id": path.stem,
            "label": label,
            "label_source": label_source,
            "source": "",
        }
        for path in image_paths
    ]


def sanitize_column(name: object) -> str:
    text = str(name).strip()
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text or "unnamed"


def normalize_cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\xa0", " ").strip()
    return "" if text.lower() == "nan" else text


def read_excel_records(path: Path) -> list[dict[str, str]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas and openpyxl are required to preprocess PAPILA clinical metadata.") from exc

    frame = pd.read_excel(path)
    frame = frame.rename(columns={column: sanitize_column(column) for column in frame.columns})
    frame = frame.where(pd.notnull(frame), "")
    return [
        {str(key): normalize_cell(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def read_papila_clinical_records(path: Path) -> list[dict[str, str]]:
    """Read PAPILA clinical spreadsheets using the layout from HelpCode/utils.py."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas and openpyxl are required to preprocess PAPILA clinical metadata.") from exc

    frame = pd.read_excel(path, index_col=0)
    if "ID" in frame.index:
        frame = frame.drop(["ID"], axis=0)
    if frame.empty:
        return []

    frame.columns = [sanitize_column(column) for column in frame.iloc[0, :]]
    frame.columns.name = "ID"
    frame = frame[~frame.index.isna()].copy()
    frame = frame.where(pd.notnull(frame), "")

    records = []
    for patient_id, row in frame.iterrows():
        record = {str(key): normalize_cell(value) for key, value in row.items()}
        record["patient_id"] = normalize_cell(patient_id)
        record["patient_num"] = numeric_token(record["patient_id"])
        records.append(record)
    return records


def candidate_patient_id(row: dict[str, str], fallback_idx: int) -> str:
    for key in ("patient_id", "patient", "id", "subject_id", "subject", "code", "identifier"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return str(fallback_idx + 1)


def candidate_diagnosis(row: dict[str, str]) -> str:
    for key in ("diagnosis", "diagnostic", "diagnostico", "class", "label", "tag"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    clinical_values = [
        value
        for key, value in row.items()
        if key not in {"patient_id", "patient_num"}
    ]
    return str(clinical_values[2]).strip() if len(clinical_values) > 2 else ""


def numeric_token(text: str) -> str:
    match = re.search(r"\d+", text)
    return str(int(match.group(0))) if match else ""


def papila_eye_from_stem(stem: str) -> str:
    upper = stem.upper()
    if re.search(r"(^|[^A-Z])OD([^A-Z]|$)", upper) or upper.endswith("OD"):
        return "OD"
    if re.search(r"(^|[^A-Z])OS([^A-Z]|$)", upper) or upper.endswith("OS"):
        return "OS"
    return ""


def split_papila_images(images: list[Path]) -> dict[str, list[Path]]:
    by_eye = {"OD": [], "OS": [], "": []}
    for image in images:
        by_eye[papila_eye_from_stem(image.stem)].append(image)
    for key in by_eye:
        by_eye[key] = sorted(by_eye[key])
    return by_eye


def match_papila_image(row: dict[str, str], eye: str, row_idx: int, images_by_eye: dict[str, list[Path]]) -> Path | None:
    candidates = images_by_eye.get(eye) or []
    patient_id = candidate_patient_id(row, row_idx)
    patient_digits = numeric_token(patient_id)
    if patient_digits:
        for image in candidates:
            if numeric_token(image.stem) == patient_digits:
                return image
    normalized_id = re.sub(r"[^0-9a-zA-Z]+", "", patient_id).lower()
    if normalized_id:
        for image in candidates:
            if normalized_id in re.sub(r"[^0-9a-zA-Z]+", "", image.stem).lower():
                return image
    if row_idx < len(candidates):
        return candidates[row_idx]
    all_images = images_by_eye.get("", [])
    if row_idx < len(all_images):
        return all_images[row_idx]
    return None


def build_stem_lookup(root: Path) -> dict[str, Path]:
    if not root.exists():
        return {}
    return {path.stem: path for path in sorted(path for path in root.rglob("*") if path.is_file() and path.name != ".DS_Store")}


def find_by_stem(lookup: dict[str, Path], stem: str) -> str:
    if stem in lookup:
        return str(lookup[stem])
    normalized = re.sub(r"[^0-9a-zA-Z]+", "", stem).lower()
    for key, path in lookup.items():
        if re.sub(r"[^0-9a-zA-Z]+", "", key).lower() == normalized:
            return str(path)
    return ""


def prepare_physionet_fundus(dataset_key: str, target_dir: Path, raw_dir: Path, dry_run: bool) -> None:
    prepare_kind = "brset" if dataset_key == "brset" else "mbrset"
    label_name = "labels_brset.csv" if prepare_kind == "brset" else "labels_mbrset.csv"
    source_image_names = ["fundus_photos", "images"] if prepare_kind == "brset" else ["images", "fundus_photos"]

    label_source = find_first(raw_dir, label_name)
    if label_source is None and (target_dir / label_name).exists():
        label_source = target_dir / label_name
    if label_source is None:
        raise RuntimeError(f"Could not find {label_name} under {raw_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)
    copy_csv_if_missing(label_source, target_dir / label_name, dry_run)
    copy_split_file_if_missing(dataset_key, target_dir, dry_run)

    if not (target_dir / "images_224").exists():
        image_source = find_image_dir(raw_dir, source_image_names)
        if image_source is None and (target_dir / "images").exists():
            image_source = target_dir / "images"
        if image_source is None:
            raise RuntimeError(f"Could not find source images under {raw_dir}")
        resize_images_to_224(image_source, target_dir / "images_224", dry_run)
    else:
        print(f"[skip] {target_dir / 'images_224'} already exists")

    rows = manifest_from_physionet_labels(dataset_key, target_dir, target_dir / label_name)
    write_manifest(target_dir, rows, dry_run)


def manifest_from_physionet_labels(dataset_key: str, target_dir: Path, label_csv: Path) -> list[dict[str, str]]:
    rows = []
    image_root = target_dir / "images_224"
    for label_row in read_csv_rows(label_csv):
        if dataset_key == "brset":
            image_id = label_row.get("image_id", "")
            patient_id = label_row.get("patient_id", "")
            image_path = image_root / f"{image_id}.jpg"
        else:
            image_file = label_row.get("file", "")
            image_id = Path(image_file).stem
            patient_id = label_row.get("patient", "")
            image_path = image_root / image_file
        if not image_path.exists():
            continue
        record = {
            "dataset": dataset_key,
            "split": "",
            "image_path": str(image_path),
            "image_id": image_id,
            "patient_id": patient_id,
            "label": infer_primary_label({**label_row, "dataset": dataset_key}),
            "label_source": str(label_csv),
            "source": label_csv.name,
        }
        record.update({f"label_{key}": value for key, value in label_row.items() if key})
        rows.append(record)
    if not rows:
        raise RuntimeError(f"Could not build {dataset_key} manifest from {label_csv}")
    return rows


def extract_nested_zips(target_dir: Path, dry_run: bool) -> None:
    for zip_path in sorted(target_dir.rglob("*.zip")):
        if zip_path.name == "RFMiD2_0.zip":
            continue
        extract_dir = zip_path.parent / zip_path.stem
        marker = extract_dir / ".extracted"
        if marker.exists() or image_files(extract_dir) or csv_files(extract_dir):
            print(f"[skip] {zip_path} already extracted")
            continue
        print(f"[prepare] extract nested zip {zip_path} -> {extract_dir}")
        if dry_run:
            continue
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)
        marker.write_text(zip_path.name + "\n")


def rows_from_label_csv(dataset_key: str, csv_path: Path, image_root: Path, split: str) -> list[dict[str, str]]:
    rows = []
    images_by_stem = {path.stem: path for path in image_files(image_root)}
    normalized_images_by_stem = {normalize_image_stem(path.stem): path for path in image_files(image_root)}
    for label_row in read_csv_rows(csv_path):
        image_id = (
            label_row.get("ID")
            or label_row.get("id")
            or label_row.get("Image")
            or label_row.get("imageID")
            or label_row.get("imageId")
            or label_row.get("ImageID")
            or label_row.get("imageName")
            or label_row.get("Image name")
            or label_row.get("filename")
            or label_row.get("Filename")
            or label_row.get("file")
            or label_row.get("image")
            or label_row.get("image_id")
            or label_row.get("name")
            or ""
        )
        image_id = str(image_id).strip()
        image_path = images_by_stem.get(Path(image_id).stem)
        if image_path is None:
            image_path = normalized_images_by_stem.get(normalize_image_stem(Path(image_id).stem))
        if image_path is None and image_id:
            for suffix in IMAGE_SUFFIXES:
                candidate = image_root / f"{image_id}{suffix}"
                if candidate.exists():
                    image_path = candidate
                    break
        if image_path is None:
            continue
        record = {
            "dataset": dataset_key,
            "split": split,
            "image_path": str(image_path),
            "image_id": image_path.stem,
            "label": infer_primary_label({**label_row, "dataset": dataset_key}),
            "label_source": str(csv_path),
            "source": csv_path.name,
        }
        record.update({f"label_{key}": value for key, value in label_row.items() if key})
        rows.append(record)
    return rows


def normalize_image_stem(stem: str) -> str:
    normalized = str(stem).strip()
    for suffix in ("_PP", "_pp", "-PP", "-pp"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def infer_primary_label(label_row: dict[str, str]) -> str:
    dr_label = infer_binary_dr_label(label_row)
    if dr_label != "":
        return dr_label
    for key in (
        "Disease_Risk",
        "Disease Risk",
        "binaryLabels",
        "binaryLabel",
        "BinaryLabels",
        "glaucoma",
        "Glaucoma",
        "diagnosis",
        "Diagnosis",
        "label",
    ):
        value = str(label_row.get(key, "")).strip()
        if value:
            return value
    return ""


def infer_binary_dr_label(label_row: dict[str, str]) -> str:
    dataset = str(label_row.get("dataset", "")).strip().lower()

    binary_dr_keys = ("DR", "dr", "DR_2_Class", "DR_2_class", "binary_dr", "Binary DR")
    for key in binary_dr_keys:
        value = get_label_value(label_row, key)
        if value != "":
            return "1" if parse_numeric_label(value) > 0 else "0"

    graded_dr_keys = [
        "DR_ICDR",
        "DR ICDR",
        "ICDR",
        "final_icdr",
        "Retinopathy grade",
        "retinopathy grade",
        "DR Grade",
        "DR grade",
        "adjudicated_dr_grade",
        "adjudicated DR grade",
        "dr_grade",
        "retinopathy_grade",
        "DR grade ICDR",
    ]
    if dataset in {"messidor_2", "messidor-2", "messidor"}:
        graded_dr_keys.extend(["diagnosis", "Diagnosis"])

    for key in graded_dr_keys:
        value = get_label_value(label_row, key)
        if value != "":
            return "1" if parse_numeric_label(value) > 0 else "0"
    return ""


def add_standard_tasks(rows: list[dict[str, str]]) -> None:
    for row in rows:
        row["Task_DR_Binary"] = infer_task_dr_binary(row)
        row["Task_Referable"] = infer_task_referable(row)
        row["Task_Glaucoma"] = infer_task_glaucoma(row)


def get_label_value(row: dict[str, str], *names: str) -> str:
    for name in names:
        for key in (name, f"label_{name}", sanitize_column(name), f"label_{sanitize_column(name)}"):
            value = str(row.get(key, "")).strip()
            if value and value.lower() != "nan":
                return value
    return ""


def infer_task_dr_binary(row: dict[str, str]) -> str:
    binary_dr = infer_binary_dr_label(row)
    if binary_dr != "":
        return binary_dr

    dataset = str(row.get("dataset", "")).strip().lower()
    if dataset in {"brset", "mbrset", "rfmid", "rfmid_2", "idrid", "messidor_2", "jsiec1000"}:
        label = get_label_value(row, "label")
        if label != "":
            return "1" if parse_numeric_label(label) > 0 else "0"
    return ""


def infer_task_referable(row: dict[str, str]) -> str:
    dataset = str(row.get("dataset", "")).strip().lower()

    jsiec_class = get_label_value(row, "class_folder", "class_name", "source").lower()
    if dataset == "jsiec1000" and jsiec_class:
        if jsiec_class in {"dr2", "dr3", "1.0.dr2", "1.1.dr3"} or "suspected_pdr" in jsiec_class or "suspected pdr" in jsiec_class:
            return "1"
        if "dr1" in jsiec_class or get_label_value(row, "binary_dr") != "":
            return "0"

    grade_keys = [
        "DR_ICDR",
        "DR ICDR",
        "ICDR",
        "final_icdr",
        "Retinopathy grade",
        "DR Grade",
        "adjudicated_dr_grade",
        "dr_grade",
        "retinopathy_grade",
    ]
    if dataset in {"messidor_2", "messidor-2", "messidor"}:
        grade_keys.extend(["diagnosis", "Diagnosis"])
    grade = get_label_value(row, *grade_keys)
    edema = get_label_value(
        row,
        "macular_edema",
        "final_edema",
        "Risk of macular edema",
        "adjudicated_dme",
        "DME",
        "ME",
        "CME",
    )
    if grade != "" or edema != "":
        grade_positive = parse_numeric_label(grade) >= 2 if grade != "" else False
        edema_positive = parse_numeric_label(edema) > 0 if edema != "" else False
        return "1" if grade_positive or edema_positive else "0"
    return ""


def infer_task_glaucoma(row: dict[str, str]) -> str:
    dataset = str(row.get("dataset", "")).strip().lower()

    for key in ("increased_cup_disc", "increased_cdr", "binaryLabels", "binaryLabel", "glaucoma", "Glaucoma"):
        value = get_label_value(row, key)
        if value != "":
            return "1" if parse_numeric_label(value) > 0 else "0"

    diagnosis_keys = ["clinical_diagnosis"]
    if dataset == "papila":
        diagnosis_keys.extend(["diagnosis", "Diagnosis"])
    diagnosis = get_label_value(row, *diagnosis_keys)
    if diagnosis != "":
        numeric = parse_numeric_label(diagnosis)
        if numeric in {0.0, 1.0, 2.0}:
            return "1" if numeric > 0 else "0"
        normalized = diagnosis.lower()
        if "glaucoma" in normalized or "suspicious" in normalized or "suspect" in normalized:
            return "1"
        if "healthy" in normalized or "normal" in normalized:
            return "0"

    class_text = " ".join(
        get_label_value(row, key).lower()
        for key in ("class_name", "class_folder", "source")
    )
    if "possible_glaucoma" in class_text or "possible glaucoma" in class_text or "large_optic_cup" in class_text or "large optic cup" in class_text:
        return "1"
    return ""


def parse_numeric_label(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        normalized = value.strip().lower()
        if normalized in {"yes", "true", "positive", "present", "abnormal"}:
            return 1.0
        if normalized in {"no", "false", "negative", "absent", "normal"}:
            return 0.0
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        return float(match.group(0)) if match else 0.0


def labels_from_manifest_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    label_rows = []
    for row in rows:
        label_row = {
            "dataset": row.get("dataset", ""),
            "split": row.get("split", ""),
            "image_id": row.get("image_id", ""),
            "image_path": row.get("image_path", ""),
            "label": row.get("label", ""),
            "Task_DR_Binary": row.get("Task_DR_Binary", ""),
            "Task_Referable": row.get("Task_Referable", ""),
            "Task_Glaucoma": row.get("Task_Glaucoma", ""),
            "label_source": row.get("label_source", ""),
            "source": row.get("source", ""),
        }
        for key, value in row.items():
            if key.startswith("label_") and key != "label_":
                label_row[key.removeprefix("label_")] = value
        label_rows.append(label_row)
    return label_rows


def prepare_rfmid(target_dir: Path, dry_run: bool) -> None:
    split_specs = [
        ("train", "Training_Set/Training_Set/RFMiD_Training_Labels.csv", "Training_Set/Training_Set/Training"),
        ("validation", "Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv", "Evaluation_Set/Evaluation_Set/Validation"),
        ("test", "Test_Set/Test_Set/RFMiD_Testing_Labels.csv", "Test_Set/Test_Set/Test"),
    ]
    rows = []
    for split, label_rel, image_rel in split_specs:
        label_path = target_dir / label_rel
        image_root = target_dir / image_rel
        if label_path.exists() and image_root.exists():
            rows.extend(rows_from_label_csv("rfmid", label_path, image_root, split))
    if not rows:
        raise RuntimeError(f"Could not build RFMiD manifest under {target_dir}")
    labels = labels_from_manifest_rows(rows)
    write_rows_csv(target_dir / "labels_rfmid.csv", labels, dry_run)
    write_rows_csv(target_dir / "labels.csv", labels, dry_run)
    write_manifest(target_dir, rows, dry_run)


def prepare_rfmid_2(target_dir: Path, dry_run: bool) -> None:
    extract_nested_zips(target_dir, dry_run)
    rows = []
    for split in ["Training_set", "Validation_set", "Test_set"]:
        split_root = target_dir / split
        split_images = image_files(split_root)
        label_csvs = csv_files(split_root)
        if label_csvs:
            before = len(rows)
            for label_csv in label_csvs:
                label_rows = rows_from_label_csv("rfmid_2", label_csv, split_root, split.lower().replace("_set", ""))
                rows.extend(label_rows)
            if len(rows) == before and split_images:
                rows.extend(manifest_from_images("rfmid_2", split_images, split=split.lower().replace("_set", ""), label_source="unmatched_csv"))
        elif split_images:
            rows.extend(manifest_from_images("rfmid_2", split_images, split=split.lower().replace("_set", "")))
    if not rows:
        raise RuntimeError(f"Could not build RFMiD_2 manifest under {target_dir}; nested zips may need extraction.")
    labels = labels_from_manifest_rows(rows)
    write_rows_csv(target_dir / "labels_rfmid_2.csv", labels, dry_run)
    write_rows_csv(target_dir / "labels.csv", labels, dry_run)
    write_manifest(target_dir, rows, dry_run)


def prepare_messidor_2(target_dir: Path, dry_run: bool) -> None:
    label_csv = target_dir / "messidor_data.csv"
    image_root = target_dir / "messidor-2" / "messidor-2" / "preprocess"
    if not image_root.exists():
        candidates = [path for path in target_dir.rglob("preprocess") if path.is_dir() and image_files(path)]
        if candidates:
            image_root = candidates[0]
    if not label_csv.exists() or not image_root.exists():
        raise RuntimeError(f"Could not find Messidor-2 labels/images under {target_dir}")
    rows = rows_from_label_csv("messidor_2", label_csv, image_root, "all")
    if not rows:
        images = image_files(image_root)
        label_rows = read_csv_rows(label_csv)
        rows_by_name = {}
        for row in label_rows:
            image_name = (
                row.get("image_id")
                or row.get("id_code")
                or row.get("image")
                or row.get("Image name")
                or row.get("name")
                or ""
            )
            rows_by_name[normalize_image_stem(Path(str(image_name)).stem)] = row
        for image_path in images:
            label_row = rows_by_name.get(normalize_image_stem(image_path.stem), {})
            record = {
                "dataset": "messidor_2",
                "split": "all",
                "image_path": str(image_path),
                "image_id": image_path.stem,
                "label": infer_primary_label({**label_row, "dataset": "messidor_2"}),
                "label_source": str(label_csv),
                "source": label_csv.name,
            }
            record.update({f"label_{key}": value for key, value in label_row.items() if key})
            rows.append(record)
    labels = labels_from_manifest_rows(rows)
    write_rows_csv(target_dir / "labels_messidor_2.csv", labels, dry_run)
    write_rows_csv(target_dir / "labels.csv", labels, dry_run)
    write_manifest(target_dir, rows, dry_run)


def find_dir_with_file(root: Path, file_name: str) -> Path | None:
    for file_path in root.rglob(file_name):
        if file_path.is_file():
            return file_path.parent
    return None


def prepare_g1020(target_dir: Path, dry_run: bool) -> None:
    subset_root = target_dir / "G1020" / "G1020"
    if not subset_root.exists():
        subset_root = target_dir / "G1020"
    if not (subset_root / "G1020.csv").exists():
        discovered_root = find_dir_with_file(target_dir, "G1020.csv")
        if discovered_root is not None:
            subset_root = discovered_root
    image_root = subset_root / "Images"
    if not image_root.exists():
        image_candidates = [
            path
            for path in subset_root.rglob("Images")
            if path.is_dir() and image_files(path)
        ]
        if image_candidates:
            image_root = image_candidates[0]
    label_csv = subset_root / "G1020.csv"
    if not image_root.exists():
        candidates = [path for path in target_dir.rglob("*") if path.is_dir() and path.name.lower() in {"images", "images_square"} and image_files(path)]
        detail = ", ".join(str(path) for path in candidates[:8])
        raise RuntimeError(f"Could not find G1020 image root under {target_dir}. Candidate image dirs: {detail}")
    rows = rows_from_label_csv("g1020", label_csv, image_root, "all") if label_csv.exists() else []
    if not rows:
        rows = manifest_from_images("g1020", image_files(image_root), split="all", label_source=str(label_csv) if label_csv.exists() else "")
    mask_root = subset_root / "Masks"
    square_mask_root = subset_root / "Masks_Square"
    cropped_mask_root = subset_root / "Masks_Cropped"
    for row in rows:
        stem = Path(row["image_id"]).stem
        row["mask_path"] = find_matching_image_file(mask_root, stem)
        row["mask_square_path"] = find_matching_image_file(square_mask_root, stem)
        row["mask_cropped_path"] = find_matching_image_file(cropped_mask_root, stem)
    labels = labels_from_manifest_rows(rows)
    for label_row, row in zip(labels, rows):
        label_row["mask_path"] = row.get("mask_path", "")
        label_row["mask_square_path"] = row.get("mask_square_path", "")
        label_row["mask_cropped_path"] = row.get("mask_cropped_path", "")
    write_rows_csv(target_dir / "labels_g1020.csv", labels, dry_run)
    write_rows_csv(target_dir / "labels.csv", labels, dry_run)
    write_manifest(target_dir, rows, dry_run)


def find_matching_image_file(root: Path, stem: str) -> str:
    if not root.exists():
        return ""
    normalized_stem = normalize_image_stem(stem)
    for path in image_files(root):
        if normalize_image_stem(path.stem) == normalized_stem:
            return str(path)
    return ""


JSIEC_CLASS_NAME_MAP = {
    "0.0.normal": ("0_0", "normal"),
    "0.1.tessellated fundus": ("0_1", "tessellated_fundus"),
    "0.2.large optic cup": ("0_2", "large_optic_cup"),
    "0.3.dr1": ("0_3", "dr1"),
    "1.0.dr2": ("1_0", "dr2"),
    "1.1.dr3": ("1_1", "dr3"),
    "2.0.brvo": ("2_0", "brvo"),
    "2.1.crvo": ("2_1", "crvo"),
    "3.rao": ("3", "rao"),
    "4.rhegmatogenous rd": ("4", "rhegmatogenous_retinal_detachment"),
    "5.0.cscr": ("5_0", "cscr"),
    "5.1.vkh disease": ("5_1", "vkh_disease"),
    "6.maculopathy": ("6", "maculopathy"),
    "7.erm": ("7", "epiretinal_membrane"),
    "8.mh": ("8", "macular_hole"),
    "9.pathological myopia": ("9", "pathological_myopia"),
    "10.0.possible glaucoma": ("10_0", "possible_glaucoma"),
    "10.1.optic atrophy": ("10_1", "optic_atrophy"),
    "11.severe hypertensive retinopathy": ("11", "severe_hypertensive_retinopathy"),
    "12.disc swelling and elevation": ("12", "disc_swelling_and_elevation"),
    "13.dragged disc": ("13", "dragged_disc"),
    "14.congenital disc abnormality": ("14", "congenital_disc_abnormality"),
    "15.0.retinitis pigmentosa": ("15_0", "retinitis_pigmentosa"),
    "15.1.bietti crystalline dystrophy": ("15_1", "bietti_crystalline_dystrophy"),
    "16.peripheral retinal degeneration and break": ("16", "peripheral_retinal_degeneration_and_break"),
    "17.myelinated nerve fiber": ("17", "myelinated_nerve_fiber"),
    "18.vitreous particles": ("18", "vitreous_particles"),
    "19.fundus neoplasm": ("19", "fundus_neoplasm"),
    "20.massive hard exudates": ("20", "massive_hard_exudates"),
    "21.yellow-white spots-flecks": ("21", "yellow_white_spots_flecks"),
    "22.cotton-wool spots": ("22", "cotton_wool_spots"),
    "23.vessel tortuosity": ("23", "vessel_tortuosity"),
    "24.chorioretinal atrophy-coloboma": ("24", "chorioretinal_atrophy_coloboma"),
    "25.preretinal hemorrhage": ("25", "preretinal_hemorrhage"),
    "26.fibrosis": ("26", "fibrosis"),
    "27.laser spots": ("27", "laser_spots"),
    "28.silicon oil in eye": ("28", "silicon_oil_in_eye"),
    "29.0.blur fundus without pdr": ("29_0", "blur_fundus_without_pdr"),
    "29.1.blur fundus with suspected pdr": ("29_1", "blur_fundus_with_suspected_pdr"),
}


def clean_jsiec_class(folder_name: str) -> tuple[str, str]:
    normalized = re.sub(r"\s+", " ", folder_name.strip()).lower()
    if normalized in JSIEC_CLASS_NAME_MAP:
        return JSIEC_CLASS_NAME_MAP[normalized]
    class_id = re.match(r"^[0-9]+(?:\.[0-9]+)?", normalized)
    class_id_text = class_id.group(0).replace(".", "_") if class_id else ""
    class_name = re.sub(r"^[0-9]+(?:\.[0-9]+)?\.?", "", normalized).strip()
    class_name = re.sub(r"[^0-9a-zA-Z]+", "_", class_name).strip("_")
    return class_id_text, class_name or normalized.replace(".", "_")


def jsiec_binary_dr(folder_name: str) -> str:
    normalized = re.sub(r"\s+", " ", folder_name.strip()).lower()
    return "1" if normalized in {"0.3.dr1", "1.0.dr2", "1.1.dr3", "29.1.blur fundus with suspected pdr"} else "0"


def prepare_jsiec1000(target_dir: Path, dry_run: bool) -> None:
    image_root = target_dir / "1000images"
    if not image_root.exists():
        raise RuntimeError(f"Could not find JSIEC1000 image root under {target_dir}")
    rows = []
    for class_dir in sorted(path for path in image_root.iterdir() if path.is_dir() and path.name != "1000images"):
        class_id, class_name = clean_jsiec_class(class_dir.name)
        binary_dr = jsiec_binary_dr(class_dir.name)
        for image_path in image_files(class_dir):
            rows.append(
                {
                    "dataset": "jsiec1000",
                    "split": "all",
                    "image_path": str(image_path),
                    "image_id": image_path.stem,
                    "label": binary_dr,
                    "label_source": "folder",
                    "source": class_dir.name,
                    "class_id": class_id,
                    "class_name": class_name,
                    "class_folder": class_dir.name,
                    "binary_dr": binary_dr,
                }
            )
    if not rows:
        raise RuntimeError(f"Could not build JSIEC1000 manifest under {target_dir}")
    write_rows_csv(target_dir / "labels_jsiec1000.csv", rows, dry_run)
    write_rows_csv(target_dir / "labels.csv", rows, dry_run)
    write_manifest(target_dir, rows, dry_run)


def prepare_idrid(target_dir: Path, dry_run: bool) -> None:
    extract_nested_zips(target_dir, dry_run)
    csv_candidates = [
        path
        for path in csv_files(target_dir)
        if "disease" in path.name.lower() and "grading" in path.name.lower() and "label" in path.name.lower()
    ]
    if not csv_candidates:
        csv_candidates = [
            path
            for path in csv_files(target_dir)
            if "grading" in str(path).lower() and "label" in path.name.lower()
        ]
    if not csv_candidates:
        raise RuntimeError(f"Could not find IDRiD disease grading label CSVs under {target_dir}")

    rows = []
    for label_csv in sorted(csv_candidates):
        lower_path = str(label_csv).lower()
        split = "train" if "train" in lower_path else ("test" if "test" in lower_path else "all")
        image_root = label_csv.parent
        for parent in [label_csv.parent, *label_csv.parents]:
            if parent == target_dir.parent:
                break
            parent_images = image_files(parent)
            if parent_images:
                image_root = parent
                break
        label_rows = rows_from_label_csv("idrid", label_csv, image_root, split)
        if not label_rows:
            images = image_files(image_root)
            labels_by_id = {}
            for label_row in read_csv_rows(label_csv):
                image_id = (
                    label_row.get("Image No")
                    or label_row.get("Image name")
                    or label_row.get("image")
                    or label_row.get("image_id")
                    or label_row.get("ID")
                    or ""
                )
                labels_by_id[Path(str(image_id)).stem] = label_row
            for image_path in images:
                label_row = labels_by_id.get(image_path.stem)
                if label_row is None:
                    continue
                record = {
                    "dataset": "idrid",
                    "split": split,
                    "image_path": str(image_path),
                    "image_id": image_path.stem,
                    "label": infer_primary_label({**label_row, "dataset": "idrid"}),
                    "label_source": str(label_csv),
                }
                record.update({f"label_{key}": value for key, value in label_row.items() if key})
                label_rows.append(record)
        rows.extend(label_rows)

    if not rows:
        raise RuntimeError(f"Could not build IDRiD manifest under {target_dir}")

    labels = labels_from_manifest_rows(rows)
    write_rows_csv(target_dir / "labels_idrid.csv", labels, dry_run)
    write_rows_csv(target_dir / "labels.csv", labels, dry_run)
    write_manifest(target_dir, rows, dry_run)


def prepare_papila(target_dir: Path, dry_run: bool) -> None:
    roots = [path for path in target_dir.rglob("FundusImages") if path.is_dir()]
    if not roots:
        raise RuntimeError(f"Could not find PAPILA FundusImages under {target_dir}")
    fundus_root = roots[0]
    papila_root = fundus_root.parent
    clinical_root = papila_root / "ClinicalData"
    contours_root = papila_root / "ExpertsSegmentations" / "Contours"
    contour_images_root = papila_root / "ExpertsSegmentations" / "ImagesWithContours"

    clinical_files = {
        "OD": clinical_root / "patient_data_od.xlsx",
        "OS": clinical_root / "patient_data_os.xlsx",
    }
    missing = [str(path) for path in clinical_files.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Could not find PAPILA clinical metadata files: {missing}")

    images_by_eye = split_papila_images(image_files(fundus_root))
    contours_by_stem = build_stem_lookup(contours_root)
    contour_images_by_stem = build_stem_lookup(contour_images_root)

    rows = []
    for eye, clinical_file in clinical_files.items():
        clinical_rows = read_papila_clinical_records(clinical_file)
        for row_idx, clinical_row in enumerate(clinical_rows):
            image_path = match_papila_image(clinical_row, eye, row_idx, images_by_eye)
            patient_id = candidate_patient_id(clinical_row, row_idx)
            image_id = image_path.stem if image_path else f"{patient_id}_{eye}"
            diagnosis = candidate_diagnosis(clinical_row)
            record = {
                "dataset": "papila",
                "split": "all",
                "patient_id": patient_id,
                "patient_num": numeric_token(patient_id),
                "eye": eye,
                "eye_side": "right" if eye == "OD" else "left",
                "image_id": image_id,
                "image_path": str(image_path) if image_path else "",
                "label": diagnosis,
                "diagnosis": diagnosis,
                "label_source": str(clinical_file),
                "contour_path": find_by_stem(contours_by_stem, image_id),
                "image_with_contours_path": find_by_stem(contour_images_by_stem, image_id),
            }
            for key, value in clinical_row.items():
                record[f"clinical_{key}"] = value
            rows.append(record)

    missing_images = sum(1 for row in rows if not row["image_path"])
    if missing_images:
        print(f"[warn] PAPILA metadata rows without matched image: {missing_images}")
    if not any(row["image_path"] for row in rows):
        raise RuntimeError(f"Could not match PAPILA clinical metadata to fundus images under {fundus_root}")

    write_rows_csv(target_dir / "labels_papila.csv", rows, dry_run)
    write_rows_csv(target_dir / "labels.csv", rows, dry_run)
    write_manifest(target_dir, rows, dry_run)


def prepare_folder_dataset(dataset_key: str, target_dir: Path, dry_run: bool) -> None:
    images = image_files(target_dir)
    if not images:
        raise RuntimeError(f"No images found under {target_dir}; manual download is still needed.")
    rows = manifest_from_images(dataset_key, images, split="all")
    write_manifest(target_dir, rows, dry_run)


def preprocess_dataset(dataset_key: str, dataset: dict, target_dir: Path, dry_run: bool, force: bool = False) -> None:
    prepare_kind = dataset.get("download", {}).get("prepare")
    if not prepare_kind:
        return
    if prepare_kind in {"brset", "mbrset"}:
        copy_split_file_if_missing(dataset_key, target_dir, dry_run)
    if prepare_is_complete(dataset, target_dir) and not force:
        print(f"[skip] preprocessing already complete for {target_dir}")
        return
    if prepare_kind in {"brset", "mbrset"}:
        prepare_physionet_fundus(dataset_key, target_dir, target_dir / "_raw_physionet", dry_run)
    elif prepare_kind == "papila":
        prepare_papila(target_dir, dry_run)
    elif prepare_kind == "rfmid":
        prepare_rfmid(target_dir, dry_run)
    elif prepare_kind == "rfmid_2":
        prepare_rfmid_2(target_dir, dry_run)
    elif prepare_kind == "messidor_2":
        prepare_messidor_2(target_dir, dry_run)
    elif prepare_kind == "g1020":
        prepare_g1020(target_dir, dry_run)
    elif prepare_kind == "jsiec1000":
        prepare_jsiec1000(target_dir, dry_run)
    elif prepare_kind == "idrid":
        prepare_idrid(target_dir, dry_run)
    elif prepare_kind in {"refuge", "palm"}:
        prepare_folder_dataset(prepare_kind, target_dir, dry_run)
    else:
        raise ValueError(f"Unsupported preprocessing kind for {dataset_key}: {prepare_kind}")


def download_figshare(doi: str, target_dir: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] resolve Figshare DOI {doi} and download listed files")
        return
    encoded_doi = urllib.parse.quote(doi, safe="")
    metadata = urlopen_json(f"https://api.figshare.com/v2/articles?doi={encoded_doi}")
    if isinstance(metadata, list):
        if not metadata:
            raise RuntimeError(f"No Figshare article found for DOI {doi}")
        metadata = metadata[0]
    article_id = metadata["id"]
    article = urlopen_json(f"https://api.figshare.com/v2/articles/{article_id}")
    for file_info in article.get("files", []):
        output = target_dir / file_info["name"]
        download_url(file_info["download_url"], output, dry_run)
        maybe_extract_zip(output, target_dir, dry_run)


def download_zenodo(record_id: str, target_dir: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] resolve Zenodo record {record_id} and download listed files")
        return
    record = urlopen_json(f"https://zenodo.org/api/records/{record_id}")
    for file_info in record.get("files", []):
        key = file_info["key"]
        links = file_info.get("links", {})
        url = links.get("self") or links.get("download")
        if not url:
            print(f"[warn] no download URL for Zenodo file {key}")
            continue
        output = target_dir / key
        download_url(url, output, dry_run)
        maybe_extract_zip(output, target_dir, dry_run)


def download_kaggle(slug: str, target_dir: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] download Kaggle dataset {slug} into {target_dir}")
        return
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI is not installed. Install it and configure ~/.kaggle/kaggle.json, "
            f"then re-run for {slug}."
        )
    target_dir.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "datasets", "download", "-d", slug, "-p", str(target_dir), "--unzip"], dry_run)


def download_physionet(url: str, target_dir: Path, env: dict[str, str], dry_run: bool) -> None:
    raw_dir = target_dir / "_raw_physionet"
    if dry_run:
        print(f"[dry-run] recursively download PhysioNet dataset {url} into {raw_dir}")
        return
    if shutil.which("wget") is None:
        raise RuntimeError("wget is required for PhysioNet recursive downloads.")
    username = env.get("PHYSIONET_USERNAME") or env.get("PHYSION_USERNAME")
    password = env.get("PHYSION_PASSWORD") or env.get("PHYSIONET_PASSWORD")
    if not username:
        raise RuntimeError("PhysioNet username is required. Set PHYSIONET_USERNAME in .env.")
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["wget", "-r", "-N", "-c", "-np", "--user", username]
    redactions = set()
    if password:
        cmd.extend(["--password", password])
        redactions.add(password)
    else:
        cmd.append("--ask-password")
    cmd.extend(["-P", str(raw_dir), url])
    run(cmd, dry_run, redactions=redactions)


def write_manual_note(dataset_key: str, dataset: dict, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    download = dataset.get("download", {})
    note = target_dir / "MANUAL_DOWNLOAD.md"
    note.write_text(
        "\n".join(
            [
                f"# {dataset.get('name', dataset_key)}",
                "",
                "This dataset requires manual download or web-form acceptance.",
                "",
                f"- Source: {download.get('url', 'see dataset documentation')}",
                f"- Expected destination: `{target_dir}`",
                f"- Notes: {dataset.get('notes', '')}",
                "",
            ]
        )
    )
    print(f"[manual] wrote {note}")


def download_dataset(
    dataset_key: str,
    dataset: dict,
    output_root: Path,
    env: dict[str, str],
    dry_run: bool,
    download_only: bool = False,
    preprocess_only: bool = False,
    force_preprocess: bool = False,
) -> None:
    target_dir = target_dir_for(dataset_key, dataset, output_root)
    method = dataset.get("download", {}).get("method", "manual")
    download = dataset.get("download", {})
    print(f"\n== {dataset.get('name', dataset_key)} ({method}) ==")

    if preprocess_only:
        print(f"[skip] download step disabled for {dataset_key}")
    elif download_is_complete(target_dir):
        print(f"[skip] raw data already present in {target_dir}")
    elif method == "figshare":
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
        download_figshare(download["doi"], target_dir, dry_run)
        mark_complete(target_dir, dry_run)
    elif method == "zenodo":
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
        download_zenodo(download["record_id"], target_dir, dry_run)
        mark_complete(target_dir, dry_run)
    elif method == "kaggle":
        download_kaggle(download["slug"], target_dir, dry_run)
        mark_complete(target_dir, dry_run)
    elif method == "physionet":
        download_physionet(download["url"], target_dir, env, dry_run)
        mark_complete(target_dir, dry_run)
    elif method == "manual":
        if dry_run:
            print(f"[dry-run] write manual download note for {dataset_key} in {target_dir}")
        else:
            write_manual_note(dataset_key, dataset, target_dir)
    else:
        raise ValueError(f"Unsupported download method for {dataset_key}: {method}")

    if download_only:
        print(f"[skip] preprocessing disabled for {dataset_key}")
        return

    if download.get("prepare"):
        if download_is_complete(target_dir):
            preprocess_dataset(dataset_key, dataset, target_dir, dry_run, force=force_preprocess)
        elif dry_run:
            print(f"[dry-run] preprocess {dataset_key} after raw data is available")
        else:
            print(f"[manual] raw data not present for {dataset_key}; preprocessing skipped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--secret-env-file", type=Path, default=DEFAULT_SECRET_ENV)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--datasets", nargs="*", help="Dataset keys to download. Defaults to all included datasets.")
    parser.add_argument("--include-optional", action="store_true", help="Also download datasets marked included_optional.")
    parser.add_argument("--download-only", action="store_true", help="Download raw data but skip preprocessing.")
    parser.add_argument("--preprocess-only", action="store_true", help="Skip downloads and only preprocess existing raw data.")
    parser.add_argument("--force-preprocess", action="store_true", help="Overwrite prepared labels/manifests even if they already exist.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.download_only and args.preprocess_only:
        print("--download-only and --preprocess-only cannot be used together", file=sys.stderr)
        return 2
    env = merged_env(args.env_file, args.secret_env_file)
    output_root = args.output_dir or Path(env.get("DATA_PATH", "data"))
    output_root = output_root.expanduser()

    with args.config.open() as handle:
        manifest = yaml.safe_load(handle)

    datasets = manifest["datasets"]
    selected = args.datasets
    if selected is None:
        statuses = {"included"}
        if args.include_optional:
            statuses.add("included_optional")
        selected = [key for key, value in datasets.items() if value.get("status") in statuses]

    print(f"[output] {output_root}")
    failures = []
    for dataset_key in selected:
        if dataset_key not in datasets:
            failures.append((dataset_key, "not found in manifest"))
            continue
        try:
            download_dataset(
                dataset_key,
                datasets[dataset_key],
                output_root,
                env,
                args.dry_run,
                download_only=args.download_only,
                preprocess_only=args.preprocess_only,
                force_preprocess=args.force_preprocess,
            )
        except Exception as exc:
            failures.append((dataset_key, str(exc)))
            print(f"[error] {dataset_key}: {exc}", file=sys.stderr)

    if failures:
        print("\nFailures:")
        for dataset_key, reason in failures:
            print(f"- {dataset_key}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
