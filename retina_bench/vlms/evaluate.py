import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image

from retina_bench.core.data import RetinaDataset
from retina_bench.vlms.models import get_vlm_model
from retina_bench.vlms.prompts import get_zero_shot_prompts

def linear_probing_eval(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    print(f"Loading {args.dataset_name} for Linear Probing...")
    train_dataset = RetinaDataset(args.dataset_path, args.dataset_name, split="train")
    test_dataset = RetinaDataset(args.dataset_path, args.dataset_name, split="test")

    model = get_vlm_model(args.model_id, device="cuda")

    # Mapping target cols
    gt_cols = {
        "binary_dr": "DR_2_Class",
        "referable_dr": "Task_Referable",
        "glaucoma": "Task_Glaucoma"
    }
    gt_col = gt_cols.get(args.task, "Task_Referable")

    def extract_features(dataset):
        features, labels = [], []
        indices = list(range(len(dataset)))
        for i in tqdm(range(0, len(indices), args.batch_size), desc="Extracting features..."):
            batch_indices = indices[i:i+args.batch_size]
            images = [dataset.get_image(idx) for idx in batch_indices]
            batch_labels = [dataset.get_row(idx).get(gt_col, -1) for idx in batch_indices]
            
            with torch.no_grad():
                emb = model.get_image_embeddings(images).cpu().numpy()
            
            features.append(emb)
            labels.extend(batch_labels)
        return np.vstack(features), np.array(labels)

    train_features, train_labels = extract_features(train_dataset)
    test_features, test_labels = extract_features(test_dataset)
    
    train_labels = np.array([int(l) if pd.notnull(l) else -1 for l in train_labels])
    test_labels = np.array([int(l) if pd.notnull(l) else -1 for l in test_labels])

    # Filter valid labels (no -1)
    train_mask = train_labels != -1
    test_mask = test_labels != -1
    train_features, train_labels = train_features[train_mask], train_labels[train_mask]
    test_features, test_labels = test_features[test_mask], test_labels[test_mask]

    if len(np.unique(train_labels)) < 2:
        print("Error: less than 2 classes found in train set. Skipping.")
        return []

    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(train_features, train_labels)

    preds = clf.predict(test_features)
    probs = clf.predict_proba(test_features)[:, 1] if len(np.unique(train_labels)) > 1 else preds

    results = []
    test_indices = list(range(len(test_dataset)))
    test_indices = [idx for idx, label in zip(test_indices, test_mask) if label]
    
    for i, idx in enumerate(test_indices):
        results.append({
            "id": test_dataset.get_row(idx).get("image_id", str(idx)),
            "ground_truth": test_labels[i],
            "prediction": preds[i],
            "prob_1": probs[i]
        })

    return results

def zero_shot_eval(args):
    test_dataset = RetinaDataset(args.dataset_path, args.dataset_name, split="test")
    model = get_vlm_model(args.model_id, device="cuda")
    
    gt_cols = {
        "binary_dr": "DR_2_Class",
        "referable_dr": "Task_Referable",
        "glaucoma": "Task_Glaucoma"
    }
    gt_col = gt_cols.get(args.task, "Task_Referable")
    prompts_map = get_zero_shot_prompts(args.task)
    
    text_prompts = [prompts_map[0], prompts_map[1]]
    with torch.no_grad():
        text_embeds = model.get_text_embeddings(text_prompts) # Shape: (2, embedding_dim)

    results = []
    indices = list(range(len(test_dataset)))
    
    for i in tqdm(range(0, len(indices), args.batch_size), desc="Zero-Shot evaluation..."):
        batch_indices = indices[i:i+args.batch_size]
        images = [test_dataset.get_image(idx) for idx in batch_indices]
        batch_rows = [test_dataset.get_row(idx) for idx in batch_indices]
        
        with torch.no_grad():
            image_embeds = model.get_image_embeddings(images) # Shape: (B, embedding_dim)
            # cosine similarity is dot product if normalized
            logits = image_embeds @ text_embeds.T # Shape: (B, 2)
            probs = torch.softmax(logits * 100.0, dim=-1).cpu().numpy() # scale by temperature
            preds = np.argmax(probs, axis=-1)
        
        for j, row in enumerate(batch_rows):
            ground_truth = row.get(gt_col, -1)
            if pd.isna(ground_truth):
                continue
            results.append({
                "id": row.get("image_id", str(batch_indices[j])),
                "ground_truth": int(ground_truth),
                "prediction": preds[j],
                "prob_1": probs[j][1]
            })

    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate VLMs on Retina Datasets")
    parser.add_argument("--dataset_path", type=str, required=True, help="Base path to datasets")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=["referable_dr", "binary_dr", "glaucoma"])
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--method", type=str, choices=["zero_shot", "linear_probing"], required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--split", type=str, default="test")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.dataset_name}_{args.task}_{args.method}_{args.model_id.replace('/', '_')}.csv")

    if args.method == "zero_shot":
        results = zero_shot_eval(args)
    else:
        results = linear_probing_eval(args)

    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print("Evaluation failed or yielded no results.")

if __name__ == "__main__":
    main()
