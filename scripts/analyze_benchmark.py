import os
import yaml
import json
import argparse
import re
import pandas as pd
import numpy as np
from pathlib import Path
import sys

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    plt = None
    sns = None
    PLOTTING_AVAILABLE = False

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retina_bench.evaluation.performance import evaluate_performance
from retina_bench.evaluation.calibration import evaluate_calibration
from retina_bench.evaluation.fairness import compute_fairness
from retina_bench.evaluation.robustness import evaluate_robustness

from retina_bench.core.data import RetinaDataset


def short_model_name(model):
    return str(model).split("/")[-1]


def safe_filename(value):
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_").lower()


def model_slug_variants(model):
    slash_slug = str(model).replace("/", "_")
    strict_slug = slash_slug.replace("-", "_")
    alnum_slug = re.sub(r"[^A-Za-z0-9_]+", "_", slash_slug).strip("_")
    return list(dict.fromkeys([slash_slug, strict_slug, alnum_slug]))


def load_dataset_eval_config(path):
    if not path or not Path(path).exists():
        return {}
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return {
        name: (entry or {}).get("evaluation", {})
        for name, entry in (cfg.get("datasets", {}) or {}).items()
    }

def normalize_result_columns(df):
    df = df.copy()
    if 'id' not in df.columns and 'image_id' in df.columns:
        df['id'] = df['image_id']
    elif 'id' not in df.columns and 'file' in df.columns:
        df['id'] = df['file']
        
    if 'label' not in df.columns and 'ground_truth' in df.columns:
        df['label'] = df['ground_truth']
    
    if 'pred' not in df.columns:
        if 'prediction' in df.columns:
            df['pred'] = df['prediction']
        elif 'prediction_text' in df.columns:
            df['pred'] = df['prediction_text'].apply(parse_binary_prediction_text)
            df['pred'] = pd.to_numeric(df['pred'], errors='coerce')
            
    if 'prob' not in df.columns:
        if 'prob_1' in df.columns:
            df['prob'] = df['prob_1']
        elif 'prob_yes' in df.columns:
            df['prob'] = df['prob_yes']
        elif 'pred' in df.columns:
            df['prob'] = df['pred']

    for col in ["label", "pred", "prob"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_binary_prediction_text(value):
    text = str(value).strip().lower()
    match = re.search(r"\b(yes|no)\b", text)
    if not match:
        return np.nan
    return 1 if match.group(1) == "yes" else 0


def merge_metadata(res_df, demo_df, dataset):
    if 'id' not in res_df.columns:
        return res_df

    if dataset == "mbrset" and "file" in demo_df.columns:
        demo_merge_id = "file"
    elif "image_id" in demo_df.columns:
        demo_merge_id = "image_id"
    elif "file" in demo_df.columns:
        demo_merge_id = "file"
    else:
        return res_df

    result_columns = set(res_df.columns)

    def metadata_for_merge():
        metadata = demo_df.copy()
        overlapping = (set(metadata.columns) & result_columns) - {demo_merge_id}
        if overlapping:
            metadata = metadata.drop(columns=list(overlapping))
        metadata["_merge_id"] = metadata[demo_merge_id].astype(str)
        return metadata

    numeric_ids = pd.to_numeric(res_df['id'], errors='coerce')
    if dataset == 'mbrset' and numeric_ids.notna().all():
        metadata = demo_df.copy()
        overlapping = set(metadata.columns) & result_columns
        if overlapping:
            metadata = metadata.drop(columns=list(overlapping))
        demo_by_index = metadata.reset_index().rename(columns={'index': '_row_index'})
        res_by_index = res_df.copy()
        res_by_index['_row_index'] = numeric_ids.astype(int)
        merged = res_by_index.merge(demo_by_index, on='_row_index', how='inner')
        merged = merged.drop(columns=['_row_index'])
        if not merged.empty:
            return merged

    id_as_str = res_df['id'].astype(str)
    demo_with_key = metadata_for_merge()
    res_with_key = res_df.copy()
    res_with_key['_merge_id'] = id_as_str
    merged = res_with_key.merge(demo_with_key, on='_merge_id', how='inner').drop(columns=['_merge_id'])
    if not merged.empty:
        return merged

    if dataset == 'mbrset' and not id_as_str.str.endswith('.jpg').all():
        res_with_ext = res_df.copy()
        res_with_ext['_merge_id'] = id_as_str + '.jpg'
        merged = res_with_ext.merge(demo_with_key, on='_merge_id', how='inner').drop(columns=['_merge_id'])

    return merged

def get_dataset_df(dataset, data_dir):
    try:
        ds = RetinaDataset(data_dir, dataset, split="test")
        # Ensure we return only necessary base metadata to merge efficiently
        return ds.df
    except Exception as e:
        print(f"Failed to load metadata dataset {dataset}: {e}")
        return pd.DataFrame()

def load_results(results_dir, data_dir, datasets, tasks, models, dataset_tasks=None):
    all_results_df = pd.DataFrame()
    dataset_tasks = dataset_tasks or {}
    
    for dataset in datasets:
        demo_df = get_dataset_df(dataset, data_dir)
        results_data = []
        
        for task in dataset_tasks.get(dataset, tasks):
            for model in models:
                possible_files = []
                for model_slug in model_slug_variants(model):
                    # Check for possible filenames from CV, VLM, MLLM endpoints
                    possible_files.extend([
                        ("linear_probing", Path(results_dir) / f"{dataset}_{task}_linear_probing_{model_slug}.csv"),
                        ("zero_shot", Path(results_dir) / f"{dataset}_{task}_zero_shot_{model_slug}.csv"),
                        ("base", Path(results_dir) / f"{dataset}_{task}_{model_slug}.csv"),
                        ("linear_probing", Path(results_dir) / f"cv_{dataset}_{task}_linear_probing_{model_slug}.csv")
                    ])
                
                seen_paths = set()
                for method, p in possible_files:
                    if p in seen_paths:
                        continue
                    seen_paths.add(p)
                    if p.exists():
                        try:
                            # Read csv instead of json
                            df_temp = pd.read_csv(p)
                            df_temp['dataset'] = dataset
                            df_temp['task'] = task
                            df_temp['model'] = model
                            df_temp['method'] = method
                            df_temp = normalize_result_columns(df_temp)

                            if not demo_df.empty:
                                merged_df = merge_metadata(df_temp, demo_df, dataset)
                                if merged_df.empty:
                                    print(f"Warning: metadata merge produced no rows for {p}")
                                else:
                                    df_temp = merged_df

                            results_data.extend(df_temp.to_dict('records'))
                        except Exception as e:
                            print(f"Error reading {p}: {e}")

        res_df = pd.DataFrame(results_data)
        
        all_results_df = pd.concat([all_results_df, res_df], ignore_index=True)
                
    return all_results_df

def compute_overall_metrics(df, model_groups_rev, fairness_attrs, dataset_eval_config=None):
    records = []
    subgroup_records = []
    dataset_eval_config = dataset_eval_config or {}
    
    for (dataset, task, model, method), group in df.groupby(['dataset', 'task', 'model', 'method']):
        group = group.copy()
        for col in ["label", "pred", "prob"]:
            if col not in group.columns:
                group[col] = np.nan
            group[col] = pd.to_numeric(group[col], errors="coerce")

        valid = group["label"].notna() & group["pred"].notna() & group["prob"].notna()
        group = group[valid]
        y_true = group['label'].values
        y_pred = group['pred'].values
        y_prob = group['prob'].values
        
        m_type = model_groups_rev.get(model, "unknown")
        
        # 1. Performance
        perf = evaluate_performance(y_true, y_pred, y_prob)
        
        # 2. Calibration
        calib = evaluate_calibration(y_true, y_prob)
        
        # 3. Fairness
        dataset_cfg = dataset_eval_config.get(dataset, {})
        target_demo = (dataset_cfg.get("fairness", {}) or {}).get("attributes")
        if target_demo is None:
            target_demo = fairness_attrs.get(dataset, [])
        fairness, fairness_details = compute_fairness(
            group,
            target_col='label',
            pred_col='pred',
            prob_col='prob',
            demographic_cols={"attributes": target_demo, "_return_details": True}
        )

        # 4. Robustness
        robustness_attrs = (dataset_cfg.get("robustness", {}) or {}).get("attributes", [])
        robustness = evaluate_robustness(
            group,
            source_dataset=dataset,
            target_cols=['pred', 'label'],
            robustness_specs=robustness_attrs,
            return_details=False
        )
        
        base_record = {
            'dataset': dataset,
            'task': task,
            'model_type': m_type,
            'model': model,
            'method': method
        }
        
        # Merge all metric outputs
        base_record.update(perf)
        base_record.update(calib)
        base_record.update(fairness)
        base_record.update(robustness)
        
        records.append(base_record)

        for detail in fairness_details:
            subgroup_records.append({**base_record, **detail})
        
    return pd.DataFrame(records), pd.DataFrame(subgroup_records)


def save_heatmap(df, metric, output_file, title, benchmark_with_task=True):
    if not PLOTTING_AVAILABLE:
        return
    if df.empty or metric not in df.columns:
        return
    plot_df = df.copy()
    plot_df["model_label"] = plot_df["model"].map(short_model_name) + " (" + plot_df["method"].astype(str) + ")"
    if benchmark_with_task:
        plot_df["benchmark"] = plot_df["dataset"].astype(str) + "/" + plot_df["task"].astype(str)
    else:
        plot_df["benchmark"] = plot_df["dataset"].astype(str)
    pivot = plot_df.pivot_table(index="model_label", columns="benchmark", values=metric, aggfunc="mean")
    pivot = pivot.dropna(how="all")
    if pivot.empty:
        return

    height = max(5, 0.42 * len(pivot) + 2.0)
    width = max(14, 1.8 * len(pivot.columns) + 5)
    plt.figure(figsize=(width, height))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", linewidths=0.4, cbar_kws={"label": metric})
    plt.title(title)
    plt.xlabel("Dataset / task")
    plt.ylabel("Model")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight", dpi=200)
    plt.close()


def benchmark_palette(plot_df):
    if not PLOTTING_AVAILABLE:
        return {}
    brset_colors = sns.color_palette("Blues", n_colors=max(3, plot_df["benchmark"].nunique() + 1))[1:]
    mbrset_colors = sns.color_palette("Oranges", n_colors=max(3, plot_df["benchmark"].nunique() + 1))[1:]
    fallback_colors = sns.color_palette("Greens", n_colors=max(3, plot_df["benchmark"].nunique() + 1))[1:]
    palette = {}
    counters = {"brset": 0, "mbrset": 0, "other": 0}
    for benchmark in sorted(plot_df["benchmark"].unique()):
        dataset = str(benchmark).split("/")[0]
        if dataset == "brset":
            palette[benchmark] = brset_colors[counters["brset"] % len(brset_colors)]
            counters["brset"] += 1
        elif dataset == "mbrset":
            palette[benchmark] = mbrset_colors[counters["mbrset"] % len(mbrset_colors)]
            counters["mbrset"] += 1
        else:
            palette[benchmark] = fallback_colors[counters["other"] % len(fallback_colors)]
            counters["other"] += 1
    return palette


def save_grouped_bar(df, metric, output_file, title, benchmark_with_task=True):
    if not PLOTTING_AVAILABLE:
        return
    if df.empty or metric not in df.columns:
        return
    plot_df = df.dropna(subset=[metric]).copy()
    if plot_df.empty:
        return
    plot_df["model_label"] = plot_df["model"].map(short_model_name) + " (" + plot_df["method"].astype(str) + ")"
    if benchmark_with_task:
        plot_df["benchmark"] = plot_df["dataset"].astype(str) + "/" + plot_df["task"].astype(str)
    else:
        plot_df["benchmark"] = plot_df["dataset"].astype(str)
    height = max(7, 0.45 * plot_df["model_label"].nunique() + 2)
    plt.figure(figsize=(16, height))
    sns.barplot(
        data=plot_df,
        y="model_label",
        x=metric,
        hue="benchmark",
        palette=benchmark_palette(plot_df),
        orient="h",
    )
    plt.title(title)
    plt.xlabel(metric)
    plt.ylabel("Model")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Dataset/task")
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight", dpi=200)
    plt.close()


def save_analysis_plots(metrics_df, output_path):
    plots_dir = output_path / "plots"
    performance_dir = plots_dir / "performance"
    fairness_dir = plots_dir / "fairness"
    robustness_dir = plots_dir / "robustness"
    for directory in [performance_dir, fairness_dir, robustness_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    core_metrics = ["auc", "auprc", "accuracy", "f1", "ece"]
    for metric in core_metrics:
        save_heatmap(metrics_df, metric, performance_dir / f"heatmap_{metric}.png", f"{metric.upper()} by model and benchmark")
        if metric in {"auc", "accuracy", "ece"}:
            save_grouped_bar(metrics_df, metric, performance_dir / f"bar_{metric}.png", f"{metric.upper()} by model")

    for task, task_df in metrics_df.groupby("task", dropna=True):
        task_slug = safe_filename(task)
        task_performance_dir = performance_dir / "by_task" / task_slug
        task_performance_dir.mkdir(parents=True, exist_ok=True)
        for metric in core_metrics:
            save_heatmap(
                task_df,
                metric,
                task_performance_dir / f"heatmap_{metric}_{task_slug}.png",
                f"{metric.upper()} by model for {task}",
                benchmark_with_task=False,
            )
            if metric in {"auc", "accuracy", "ece"}:
                save_grouped_bar(
                    task_df,
                    metric,
                    task_performance_dir / f"bar_{metric}_{task_slug}.png",
                    f"{metric.upper()} by model for {task}",
                    benchmark_with_task=False,
                )

    fairness_gap_cols = [
        c for c in metrics_df.columns
        if c.endswith(("demographic_parity_gap", "equalized_odds_gap", "accuracy_gap", "auc_gap"))
        and not c.startswith(("camera_", "image_quality_", "focus_", "illumination_"))
    ]
    for metric in fairness_gap_cols:
        save_heatmap(metrics_df, metric, fairness_dir / f"heatmap_{metric}.png", f"Fairness: {metric}")

    for task, task_df in metrics_df.groupby("task", dropna=True):
        task_slug = safe_filename(task)
        task_fairness_dir = fairness_dir / "by_task" / task_slug
        task_fairness_dir.mkdir(parents=True, exist_ok=True)
        for metric in fairness_gap_cols:
            save_heatmap(
                task_df,
                metric,
                task_fairness_dir / f"heatmap_{metric}_{task_slug}.png",
                f"Fairness: {metric} for {task}",
                benchmark_with_task=False,
            )

    robustness_gap_cols = [
        c for c in metrics_df.columns
        if c.startswith("image_quality_") and c.endswith("_gap")
    ]
    for metric in robustness_gap_cols:
        save_heatmap(metrics_df, metric, robustness_dir / f"heatmap_{metric}.png", f"Robustness: {metric}")

    for task, task_df in metrics_df.groupby("task", dropna=True):
        task_slug = safe_filename(task)
        task_robustness_dir = robustness_dir / "by_task" / task_slug
        task_robustness_dir.mkdir(parents=True, exist_ok=True)
        for metric in robustness_gap_cols:
            save_heatmap(
                task_df,
                metric,
                task_robustness_dir / f"heatmap_{metric}_{task_slug}.png",
                f"Robustness: {metric} for {task}",
                benchmark_with_task=False,
            )


def save_coverage_report(metrics_df, expected_rows, output_path):
    if not expected_rows:
        return
    expected_df = pd.DataFrame(expected_rows)
    observed_cols = ["dataset", "task", "model", "method"]
    observed_df = metrics_df[observed_cols].drop_duplicates() if not metrics_df.empty else pd.DataFrame(columns=observed_cols)
    coverage = expected_df.merge(observed_df.assign(found=True), on=observed_cols, how="left")
    coverage["found"] = coverage["found"].fillna(False)
    coverage.to_csv(output_path / "result_coverage.csv", index=False)


def save_loaded_result_inventory(raw_df, output_path):
    if raw_df.empty:
        return
    inventory = (
        raw_df.groupby(["dataset", "task", "model", "method"], dropna=False)
        .size()
        .reset_index(name="n_rows")
    )
    inventory.to_csv(output_path / "loaded_result_inventory.csv", index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to analysis config yaml")
    parser.add_argument("--results_dir", required=True, help="Directory containing eval results")
    parser.add_argument("--data_dir", required=False, default="data", help="Data directory")
    parser.add_argument("--fundus_config", required=False, default="config/fundus_datasets.yaml", help="Dataset metadata/config yaml")
    parser.add_argument("--output_dir", required=True, help="Directory to save analysis")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    datasets = config.get("datasets", [])
    tasks = config.get("tasks", [])
    dataset_tasks = config.get("dataset_tasks", {})
    
    # Flatten models
    all_models = []
    model_groups_rev = {}
    expected_rows = []
    method_defaults = {"cv": ["linear_probing"], "vlm": ["zero_shot", "linear_probing"], "mllm": ["base"]}
    for group, models in config.get("model_groups", {}).items():
        all_models.extend(models)
        for m in models:
            model_groups_rev[m] = group
            family_prefix = group.split("_")[0]
            methods = method_defaults.get(family_prefix, ["base"])
            for dataset in datasets:
                for task in dataset_tasks.get(dataset, tasks):
                    for method in methods:
                        expected_rows.append({
                            "dataset": dataset,
                            "task": task,
                            "model": m,
                            "method": method,
                            "model_type": group,
                            "family": family_prefix,
                        })
            
    print("Loading results...")
    raw_df = load_results(args.results_dir, args.data_dir, datasets, tasks, all_models, dataset_tasks)
    
    if raw_df.empty:
        print("No result found in", args.results_dir)
        return
    
    print("Computing metrics...")
    save_loaded_result_inventory(raw_df, output_path)
    fairness_attrs = config.get("fairness_attributes", {})
    dataset_eval_config = load_dataset_eval_config(args.fundus_config)
    metrics_df, subgroup_df = compute_overall_metrics(raw_df, model_groups_rev, fairness_attrs, dataset_eval_config)
    
    # Assign family mapping for grouping
    family_map = {
        'cv_general': 'cv', 'cv_ophthalmo': 'cv',
        'vlm_general': 'vlm', 'vlm_ophthalmo': 'vlm',
        'mllm_general': 'mllm', 'mllm_medical': 'mllm'
    }
    metrics_df['family'] = metrics_df['model_type'].map(lambda x: family_map.get(x, 'unknown'))
    raw_df['family'] = raw_df['model'].apply(lambda x: family_map.get(model_groups_rev.get(x), 'unknown'))
    if not subgroup_df.empty:
        subgroup_df['family'] = subgroup_df['model_type'].map(lambda x: family_map.get(x, 'unknown'))
    
    # Save partitioned CSVs
    for family in ['cv', 'vlm', 'mllm']:
        f_df = metrics_df[metrics_df['family'] == family]
        if not f_df.empty:
            family_dir = output_path / family
            family_dir.mkdir(parents=True, exist_ok=True)
            f_df.to_csv(family_dir / "metrics.csv", index=False)
            save_analysis_plots(f_df, family_dir)

            sf_df = subgroup_df[subgroup_df['family'] == family] if not subgroup_df.empty else pd.DataFrame()
            if not sf_df.empty:
                sf_df.to_csv(family_dir / "subgroup_metrics.csv", index=False)

    # Plot Calibration curves
    if PLOTTING_AVAILABLE:
        from sklearn.calibration import calibration_curve
        for (dataset, task, family), group in raw_df.groupby(['dataset', 'task', 'family']):
            if family == 'unknown': continue
            
            plt.figure(figsize=(8, 8))
            ax = plt.gca()
            ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            
            for (model, method), model_group in group.groupby(['model', 'method']):
                y_true = model_group['label'].values
                y_prob = model_group['prob'].values
                try:
                    y_true = y_true.astype(int)
                    y_prob = y_prob.astype(float)
                    mask = (y_true == 0) | (y_true == 1)
                    
                    if sum(mask) > 10:
                        prob_true, prob_pred = calibration_curve(y_true[mask], y_prob[mask], n_bins=10)
                        label_name = f"{model.split('/')[-1]} ({method})"
                        ax.plot(prob_pred, prob_true, "s-", label=label_name)
                except Exception:
                    pass
                    
            ax.set_ylabel("Fraction of positives")
            ax.set_xlabel("Mean predicted value")
            ax.set_title(f"Calibration Diagram: {dataset.upper()} - {task}")
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize='small')
            plt.tight_layout()
            
            family_dir = output_path / family / "plots" / "calibration"
            family_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(family_dir / f"calibration_{dataset}_{task}.png", bbox_inches="tight", dpi=200)
            plt.close()
        
    metrics_df.to_csv(output_path / "aggregated_metrics_all.csv", index=False)
    metrics_df.to_csv(output_path / "aggregated_metrics.csv", index=False)
    if not subgroup_df.empty:
        subgroup_df.to_csv(output_path / "subgroup_metrics_all.csv", index=False)
    save_coverage_report(metrics_df, expected_rows, output_path)
    save_analysis_plots(metrics_df, output_path)
    
    print(f"Analysis saved to {output_path}")

if __name__ == "__main__":
    main()
