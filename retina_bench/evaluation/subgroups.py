import re

import numpy as np
import pandas as pd


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_").lower()


def normalize_sex(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"1", "male", "m", "masculino"}:
        return "male"
    if text in {"0", "2", "female", "f", "feminino"}:
        return "female"
    return np.nan


def make_binary_subgroup(df, spec, min_group_size=10):
    if isinstance(spec, str):
        spec = {"column": spec}
    else:
        min_group_size = spec.get("min_group_size", min_group_size)

    column = spec.get("column")
    if not column or column not in df.columns:
        return None, {}

    name = spec.get("name", column)
    kind = spec.get("type", "auto")
    values = df[column]

    if kind in {"age", "numeric"} or (kind == "auto" and pd.api.types.is_numeric_dtype(values)):
        numeric = pd.to_numeric(values, errors="coerce")
        valid = numeric.dropna()
        if valid.empty:
            return None, {}

        threshold = spec.get("threshold")
        if threshold is None:
            threshold = float(valid.median())
        labels = spec.get("labels") or [f"<={threshold:g}", f">{threshold:g}"]

        groups = pd.Series(np.nan, index=df.index, dtype=object)
        groups.loc[numeric <= threshold] = labels[0]
        groups.loc[numeric > threshold] = labels[1]

    elif kind == "sex":
        groups = values.map(normalize_sex)
        labels = ["female", "male"]
        threshold = None

    else:
        clean = values.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA})
        valid = clean.dropna()
        if valid.empty:
            return None, {}

        positive_values = spec.get("positive_values") or spec.get("reference_values")
        if positive_values:
            positives = {str(v).strip().lower() for v in positive_values}
            positive_label = spec.get("positive_label", "reference")
            negative_label = spec.get("negative_label", "other")
            groups = pd.Series(np.nan, index=df.index, dtype=object)
            groups.loc[clean.str.lower().isin(positives)] = positive_label
            groups.loc[clean.notna() & ~clean.str.lower().isin(positives)] = negative_label
            labels = [negative_label, positive_label]
        else:
            counts = valid.value_counts()
            if len(counts) < 2:
                return None, {}
            top = counts.index[0]
            groups = pd.Series(np.nan, index=df.index, dtype=object)
            groups.loc[clean == top] = f"{top}"
            groups.loc[clean.notna() & (clean != top)] = f"not_{top}"
            labels = [f"{top}", f"not_{top}"]
        threshold = None

    counts = groups.dropna().value_counts()
    if len(counts) < 2 or counts.min() < min_group_size:
        return None, {
            "name": name,
            "column": column,
            "skipped": True,
            "reason": "fewer_than_two_groups_or_too_small",
            "counts": counts.to_dict(),
        }

    return groups, {
        "name": safe_name(name),
        "display_name": name,
        "column": column,
        "type": kind,
        "threshold": threshold,
        "counts": counts.to_dict(),
        "labels": labels,
    }
