import os
import pandas as pd
from PIL import Image
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DATA_DIR = REPO_ROOT / "Split_Data"

MANIFEST_DATASET_PATHS = {
    "papila": ("PAPILA",),
    "rfmid": ("RFMiD",),
    "rfmid_2": ("RFMiD_2",),
    "idrid": ("IDRiD",),
    "messidor_2": ("Messidor-2",),
    "g1020": ("G1020",),
    "refuge": ("REFUGE",),
    "jsiec1000": ("JSIEC1000",),
}

class RetinaDataset:
    def __init__(self, base_dir, dataset_name, split="all", filter_macula=True, filter_diabetes=True):
        self.base_dir = Path(base_dir)
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.df = None
        self.images_dir = None
        self.filter_macula = filter_macula
        self.filter_diabetes = filter_diabetes
        
        self.load_dataset()
        
    def load_dataset(self):
        if self.dataset_name == "brset":
            path = self.base_dir / 'BRSET' / 'brset'
            self.df = pd.read_csv(path / 'labels_brset.csv')
            
            if self.filter_diabetes:
                if 'diabetes' in self.df.columns:
                    self.df = self.df[self.df['diabetes'] == 'yes'].copy()
            
            # Load splits
            split_path = path / 'labels_splits.csv'
            if split_path.exists():
                split_df = pd.read_csv(split_path)
                # Merge on patient_id and image_id to get 'split' column
                self.df = self.df.merge(split_df, on=['patient_id', 'image_id'])
            
            self.images_dir = path / 'images_224'
            # BRSET image filenames usually match image_id + .jpg
            # Ensuring consistency
            if 'image_id' in self.df.columns:
                self.df['image_path'] = self.df['image_id'].apply(lambda x: str(self.images_dir / f"{x}.jpg"))

            self.df['Task_Referable'] = ((self.df['DR_ICDR'] >= 2) | (self.df['macular_edema'] == 1)).astype(int)
        
            # BRSET Tasks
            # 5 Class: DR_ICDR (0-4)
            
            # 3 Class: 0 -> 0, 1-3 -> 1, 4 -> 2
            def map_3_class(val):
                try:
                    v = int(val)
                    if v == 0: return 0
                    if 1 <= v <= 3: return 1
                    if v == 4: return 2
                except: pass
                return -1
        
            if 'DR_ICDR' in self.df.columns:
                self.df['Task_5_Classes'] = self.df['DR_ICDR'].astype(int)
                self.df['Task_3_Classes'] = self.df['DR_ICDR'].apply(map_3_class)
                self.df['DR_2_Class'] = self.df['DR_ICDR'].apply(lambda x: 0 if x == 0 else (1 if 1 <= x <= 4 else -1))
        
            # Glaucoma Proxy: increased_cup_disc
            if 'increased_cup_disc' in self.df.columns:
                self.df['Task_Glaucoma'] = self.df['increased_cup_disc']
        
            # AMD
            if 'amd' in self.df.columns:
                self.df['Task_AMD'] = self.df['amd']
            
             
        elif self.dataset_name == "mbrset":
            path = self.base_dir / 'mBRSET' / 'mbrset'
            self.df = pd.read_csv(path / 'labels_mbrset.csv')
            
            # Load splits
            split_path = path / 'labels_splits.csv'
            if split_path.exists():
                split_df = pd.read_csv(split_path)
                # Merge on patient and file
                self.df = self.df.merge(split_df, on=['patient', 'file'])

            self.images_dir = path / 'images_224'
            
            
            IMG_COL = "file"
            DR_COL = "final_icdr"
            EDEMA_COL = "final_edema"
            
            # Filter Macula Images: Keep only those with .1.jpg or .3.jpg (frontal macula)
            if self.filter_macula:
                macula_regex = re.compile(r'\.[13](\.jpg)?$', re.IGNORECASE)
                self.df = self.df[self.df[IMG_COL].astype(str).apply(lambda x: bool(macula_regex.search(x)))].copy()
            
            # Standardizing Labels
            if EDEMA_COL in self.df.columns:
                self.df[EDEMA_COL] = self.df[EDEMA_COL].astype(str).str.lower().str.strip()
                self.df['edema_bin'] = self.df[EDEMA_COL].map({'yes': 1, 'no': 0}).fillna(0).astype(int)

            self.df[DR_COL] = pd.to_numeric(self.df[DR_COL], errors='coerce')
            self.df = self.df.dropna(subset=[DR_COL])
            
            # Task columns
            self.df['Task_5_Classes'] = self.df[DR_COL].astype(int)
            self.df['Task_Referable'] = ((self.df[DR_COL] >= 2) | (self.df['edema_bin'] == 1)).astype(int)

            # binary DR: 0 -> 0, 1-4 -> 1
            self.df['Task_3_Classes'] = self.df[DR_COL].apply(lambda x: 0 if x == 0 else (1 if 1 <= x <= 4 else -1))
            self.df['DR_2_Class'] = self.df[DR_COL].apply(lambda x: 0 if x == 0 else (1 if 1 <= x <= 4 else -1))
            
            if 'increased_cdr' in self.df.columns:
                self.df['Task_Glaucoma'] = self.df['increased_cdr'].map({'yes': 1, 'Yes': 1, 1: 1, 'no': 0, 'No': 0, 0: 0}).fillna(0).astype(int)
            
            # Image path
            self.df['image_path'] = self.df[IMG_COL].apply(lambda x: str(self.images_dir / x))

        elif self.dataset_name in MANIFEST_DATASET_PATHS:
            path = self.base_dir.joinpath(*MANIFEST_DATASET_PATHS[self.dataset_name])
            manifest_path = path / "prepared" / "manifest.csv"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Prepared manifest not found: {manifest_path}")

            self.df = pd.read_csv(manifest_path)
            self.df = self.df.replace(r"^\s*$", pd.NA, regex=True)
            self.images_dir = path
            self._normalize_standard_task_columns()
            self._ensure_eval_splits()

        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        # Filter by split if applicable
        if self.split != "all":
            if "split" in self.df.columns:
                print(f"Filtering dataset for split: {self.split}")
                self.df = self.df[self.df["split"] == self.split].copy()
                if len(self.df) == 0:
                    print(f"Warning: No samples found for split '{self.split}'")
            else:
                print(f"Warning: Split '{self.split}' requested but 'split' column not found in dataset.")

    def _normalize_standard_task_columns(self):
        if "Task_DR_Binary" in self.df.columns and "DR_2_Class" not in self.df.columns:
            self.df["DR_2_Class"] = self.df["Task_DR_Binary"]

        for col in ("Task_DR_Binary", "DR_2_Class", "Task_Referable", "Task_Glaucoma", "label"):
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        if "image_id" not in self.df.columns:
            self.df["image_id"] = self.df.index.astype(str)
        else:
            self.df["image_id"] = self.df["image_id"].astype(str)

        if "image_path" not in self.df.columns:
            raise ValueError(f"Dataset {self.dataset_name} manifest is missing image_path")

    def _ensure_eval_splits(self):
        if "split" not in self.df.columns:
            self.df["split"] = pd.NA

        split_values = self.df["split"].astype("string").str.lower().str.strip()
        split_values = split_values.replace({
            "training": "train",
            "validation": "val",
            "valid": "val",
            "nan": pd.NA,
            "none": pd.NA,
        })
        self.df["split"] = split_values

        split_df = self._load_external_split_file()
        if split_df is not None and self.split != "all":
            merge_key = "image_path" if "image_path" in split_df.columns and "image_path" in self.df.columns else "image_id"
            self.df[merge_key] = self.df[merge_key].astype(str)
            split_df[merge_key] = split_df[merge_key].astype(str)
            self.df = self.df.drop(columns=["split"], errors="ignore").merge(
                split_df[[merge_key, "split"]],
                on=merge_key,
                how="left",
            )
            self.df["split"] = self.df["split"].fillna("all").astype("string").str.lower().str.strip()
            return

        available = set(split_values.dropna().unique())
        has_named_eval_split = bool(available.intersection({"train", "val", "test"}))
        if has_named_eval_split:
            return

        # Some prepared benchmark manifests are intentionally unsplit and use
        # "all"/NaN in the split column. For evaluation, keep the full manifest
        # available instead of creating a synthetic split.
        self.df["split"] = self.split if self.split != "all" else "all"

    def _load_external_split_file(self):
        candidates = [
            SPLIT_DATA_DIR / f"labels_splits_{self.dataset_name}.csv",
            SPLIT_DATA_DIR / f"labels_splits_{self.dataset_name.replace('-', '_')}.csv",
        ]

        for split_path in candidates:
            if not split_path.exists():
                continue
            split_df = pd.read_csv(split_path)
            if "split" not in split_df.columns:
                continue
            if "image_id" not in split_df.columns:
                if "id" in split_df.columns:
                    split_df = split_df.rename(columns={"id": "image_id"})
                else:
                    continue
            split_df = split_df.copy()
            split_df["image_id"] = split_df["image_id"].astype(str)
            if "image_path" in split_df.columns:
                split_df["image_path"] = split_df["image_path"].astype(str)
            split_df["split"] = split_df["split"].astype("string").str.lower().str.strip()
            split_df["split"] = split_df["split"].replace({"validation": "val", "valid": "val"})
            return split_df.dropna(subset=["image_id", "split"])
        return None

    def __len__(self):
        return len(self.df)

    def get_row(self, idx):
        return self.df.iloc[idx]

    def get_image(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        try:
            return Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None
