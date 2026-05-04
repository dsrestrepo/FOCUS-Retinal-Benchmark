import sys, os
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from retina_bench.core.data import RetinaDataset
from retina_bench.cv.models import get_cv_model

def linear_probing_eval(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    print(f"Loading {args.dataset_name} for Linear Probing with CV model...")
    # Load dataset structure
    try:
        train_dataset = RetinaDataset(args.dataset_path, args.dataset_name, split="train")
        test_dataset = RetinaDataset(args.dataset_path, args.dataset_name, split="test")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

    model = get_cv_model(args.model_id, device="cuda", pooling=args.pooling)

    # Mapping target cols depending on task definition
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
            batch_labels = []
            
            for idx in batch_indices:
                row = dataset.get_row(idx)
                # handle variations in dataset structure
                label = row.get(gt_col, -1) 
                batch_labels.append(label)
            
            with torch.no_grad():
                emb = model.get_image_embeddings(images).cpu().numpy()
            
            features.append(emb)
            labels.extend(batch_labels)
            
        if not features:
            return np.array([]), np.array([])
        return np.vstack(features), np.array(labels)

    print("Extracting training features...")
    train_features, train_labels = extract_features(train_dataset)
    print("Extracting testing features...")
    test_features, test_labels = extract_features(test_dataset)
    
    if len(train_features) == 0 or len(test_features) == 0:
        print("No features extracted.")
        return []

    # Filter out missing labels (e.g. represented as -1, NaN or None in the dataset wrapper)
    # Be robust against None
    train_labels = np.array([int(l) if pd.notnull(l) else -1 for l in train_labels])
    test_labels = np.array([int(l) if pd.notnull(l) else -1 for l in test_labels])

    train_mask = train_labels != -1
    test_mask = test_labels != -1
    
    train_features = train_features[train_mask]
    train_labels = train_labels[train_mask]
    test_features = test_features[test_mask]
    test_labels = test_labels[test_mask]
    
    if len(np.unique(train_labels)) < 2:
        print("Error: less than 2 classes found in train set. Skipping.")
        return []

    print("Training Logistic Regression classifier (Linear Probe)...")
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(train_features, train_labels)

    print("Predicting on test set...")
    preds = clf.predict(test_features)
    probs = clf.predict_proba(test_features)[:, 1] if len(np.unique(train_labels)) > 1 else preds
    
    # Store Results
    results = []
    test_indices = list(range(len(test_dataset)))
    valid_test_indices = [idx for idx, mask_val in zip(test_indices, test_mask) if mask_val]
    
    for i, idx in enumerate(valid_test_indices):
        row = test_dataset.get_row(idx)
        results.append({
            "id": row.get("image_id", str(idx)),
            "ground_truth": test_labels[i],
            "prediction": preds[i],
            "prob_1": probs[i]
        })

    try:
        acc = accuracy_score(test_labels, preds)
        auc = roc_auc_score(test_labels, probs)
        print(f"\nLinear Probing Results for {args.model_id} on {args.dataset_name}/{args.task}:")
        print(f"Accuracy: {acc:.4f} | AUC-ROC: {auc:.4f}\n")
    except Exception as e:
        print(f"Metrics calculation error: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate CV models on Retina Datasets")
    parser.add_argument("--dataset_path", type=str, required=True, help="Base path to datasets")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=["referable_dr", "binary_dr", "glaucoma"])
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--method", type=str, choices=["linear_probing", "zero_shot"], required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "gap"], help="Pooling method for 3D outputs")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    sanitized_model_id = args.model_id.replace('/', '_').replace('-', '_')
    output_file = os.path.join(args.output_dir, f"cv_{args.dataset_name}_{args.task}_{args.method}_{sanitized_model_id}.csv")
    
    if args.method == "linear_probing":
        results = linear_probing_eval(args)
    else:
        print(f"Method '{args.method}' is not natively supported for purely visual CV encoders. Defaulting back to linear_probing.")
        results = linear_probing_eval(args)

    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print("Evaluation failed or yielded no results.")

if __name__ == "__main__":
    main()
