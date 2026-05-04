import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import argparse
import os
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# Import models
from retina_bench.mllms.models import (
    QwenVLM, LlavaVLM, LlamaVLM, GemmaVLM,
    OpenAIVLM, GeminiVLM, GptOssLLM, Gemma4VLM, KimiVLM
)
from retina_bench.core.data import RetinaDataset
import retina_bench.mllms.prompts as prompts

def sanitize_filename_part(value):
    return str(value).replace("/", "_").replace(" ", "_")

def get_model_class(model_id):
    model_id_lower = model_id.lower()
    if "qwen" in model_id_lower: return QwenVLM
    if "llava" in model_id_lower: return LlavaVLM
    if "llama" in model_id_lower and "vision" in model_id_lower: return LlamaVLM # Llama 3.2 Vision
    if "llama" in model_id_lower: return GptOssLLM # Llama text only? Or check mllama
    if "gemma-4" in model_id_lower: return Gemma4VLM
    if "gemma" in model_id_lower: return GemmaVLM
    if "kimi" in model_id_lower: return KimiVLM
    if "gpt" in model_id_lower and "oss" not in model_id_lower: return OpenAIVLM
    if "gemini" in model_id_lower: return GeminiVLM
    # Fallback/Default
    if "mllama" in model_id_lower: return LlamaVLM
    return GptOssLLM

def main():
    parser = argparse.ArgumentParser(description="Evaluate VLMs on Retina Datasets")
    parser.add_argument("--dataset_path", type=str, required=True, help="Base path to datasets")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, 
                        choices=["referable_dr", "binary_dr", "dr_5", "dr_3", "glaucoma", "amd"])
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--quantization", type=str, default=None, choices=["4b", "8b", "16b"])
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter to load")
    parser.add_argument("--train_dataset_name", type=str, default=None, help="Dataset used to train the adapter")
    parser.add_argument("--train_task", type=str, default=None, help="Task used to train the adapter")
    parser.add_argument("--train_strategy", type=str, default=None, help="Prompt/training strategy used to train the adapter")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on (train, val, test, all)")
    parser.add_argument("--prompt_strategy", type=str, default="base", choices=["base", "cot", "role"], help="Prompting strategy to use")
    
    args = parser.parse_args()
    
    # 1. Load Dataset
    print(f"Loading {args.dataset_name} ({args.split} split) from {args.dataset_path}...")
    dataset = RetinaDataset(args.dataset_path, args.dataset_name, split=args.split)
    print(f"Loaded {len(dataset)} samples.")

    # 2. Config Task
    # Define task configs for each dataset
    
    # Common prompts map
    # Note: We use specific prompt functions for each dataset/task combination
    
    common_tasks = {
        "referable_dr": {
            "prompt_func": prompts.GENERIC_REFERABLE_DR_PROMPT,
            "gt_col": "Task_Referable"
        },
        "binary_dr": {
            "prompt_func": prompts.GENERIC_BINARY_DR_PROMPT,
            "gt_col": "DR_2_Class"
        },
        "glaucoma": {
            "prompt_func": prompts.GENERIC_GLAUCOMA_PROMPT,
            "gt_col": "Task_Glaucoma"
        }
    }

    # BRSET Configs
    brset_tasks = {
        **common_tasks,
        "referable_dr": {
            "prompt_func": prompts.BRSET_REFERABLE_DR_PROMPT,
            "gt_col": "Task_Referable"
        },
        "binary_dr": {
            "prompt_func": prompts.BRSET_BINARY_DR_PROMPT,
            "gt_col": "DR_2_Class"
        },
        "dr_5": {
            "prompt_func": prompts.BRSET_5_CLASS_DR_PROMPT,
            "gt_col": "DR_ICDR"
        },
        "dr_3": {
            "prompt_func": prompts.BRSET_3_CLASS_DR_PROMPT,
            "gt_col": "Task_3_Classes"
        },
        "glaucoma": {
            "prompt_func": prompts.BRSET_GLAUCOMA_PROMPT,
            "gt_col": "Task_Glaucoma"
        },
        "amd": {
            "prompt_func": prompts.BRSET_AMD_PROMPT,
            "gt_col": "Task_AMD"
        }
    }

    # mBRSET Configs
    mbrset_tasks = {
        **common_tasks,
        "referable_dr": {
            "prompt_func": prompts.mBRSET_REFERABLE_DR_PROMPT,
            "gt_col": "Task_Referable"
        },
        "binary_dr": {
            "prompt_func": prompts.mBRSET_BINARY_DR_PROMPT,
            "gt_col": "DR_2_Class"
        },
        "dr_5": {
            "prompt_func": prompts.mBRSET_5_CLASS_DR_PROMPT,
            "gt_col": "Task_5_Classes"
        },
        "dr_3": {
            "prompt_func": prompts.mBRSET_3_CLASS_DR_PROMPT,
            "gt_col": "Task_3_Classes"
        },
        "glaucoma": {
            "prompt_func": prompts.mBRSET_GLAUCOMA_PROMPT,
            "gt_col": "Task_Glaucoma"
        }
    }

    if args.dataset_name == "brset":
        task_config = brset_tasks
    elif args.dataset_name == "mbrset":
        task_config = mbrset_tasks
    else:
        task_config = common_tasks
    
    config = task_config.get(args.task)
    if not config:
        raise ValueError(f"Task {args.task} is not supported for dataset {args.dataset_name} (or check logic)")

    # Select prompt function based on strategy
    # Try dynamic prompt dispatcher first
    prompt_func = prompts.get_prompt_func(args.dataset_name, args.task, args.prompt_strategy)
    
    if prompt_func is None:
        # Fallback to static config
        print(f"Strategy '{args.prompt_strategy}' not explicitly found for {args.task}, using default/base configuration.")
        prompt_func = config["prompt_func"]
    else:
        print(f"Using prompt strategy: {args.prompt_strategy}")

    gt_col = config["gt_col"]

    # 3. Load Model
    print(f"Loading model {args.model_id}...")
    ModelClass = get_model_class(args.model_id)
    
    # Handle GptOssLLM specific default like in other scripts
    quantization = args.quantization
    if ModelClass == GptOssLLM and quantization is None:
         # Default to 4b for text LLMs to avoid OOM
         quantization = "4b" 

    model = ModelClass(
        model_id=args.model_id,
        quantization=quantization,
        use_flash_attention=args.use_flash_attn,
        adapter_path=args.adapter_path
    )

    # 4. Inference Loop
    results = []
    
    # Batch indices
    indices = list(range(len(dataset)))
    
    os.makedirs(args.output_dir, exist_ok=True)
    model_slug = sanitize_filename_part(args.model_id)
    if args.adapter_path and args.train_dataset_name and args.train_task and args.train_strategy:
        train_dataset = sanitize_filename_part(args.train_dataset_name)
        train_task = sanitize_filename_part(args.train_task)
        train_strategy = sanitize_filename_part(args.train_strategy)
        eval_dataset = sanitize_filename_part(args.dataset_name)
        eval_task = sanitize_filename_part(args.task)
        eval_strategy = sanitize_filename_part(args.prompt_strategy)
        split = sanitize_filename_part(args.split)
        output_name = (
            f"train_model-{model_slug}_train_dataset-{train_dataset}_train_task-{train_task}_"
            f"train_strategy-{train_strategy}__test_model-{model_slug}_test_dataset-{eval_dataset}_"
            f"test_task-{eval_task}_test_strategy-{eval_strategy}_split-{split}.csv"
        )
    else:
        strategy_suffix = f"_{args.prompt_strategy}" if args.prompt_strategy != "base" else ""
        output_name = f"{args.dataset_name}_{args.task}{strategy_suffix}_{model_slug}.csv"
    output_file = os.path.join(args.output_dir, output_name)

    for i in tqdm(range(0, len(indices), args.batch_size)):
        # Standard slice
        batch_indices = indices[i : i + args.batch_size]

        batch_prompts = []
        batch_images = []
        batch_rows = []

        for idx in batch_indices:
            row = dataset.get_row(idx)
            ground_truth = row.get(gt_col, -1)
            if pd.isna(ground_truth):
                continue
            img = dataset.get_image(idx)
            
            # Construct prompt
            text_prompt = prompt_func(row)
            
            batch_prompts.append(text_prompt)
            batch_images.append(img)
            batch_rows.append(row)

        # Run Inference
        try:
            # Basic generation
            # Some models (GPT-OSS) don't support images, VLM wrapper handles it by ignoring image arg if needed
            # Use individual generate instead of batch to get logits correctly
            for j in range(len(batch_prompts)):
                try:
                    res_dict = model.generate(
                        prompt=batch_prompts[j],
                        image=batch_images[j],
                        max_new_tokens=128,
                        return_logits=True,
                        tokens=['yes', 'no', 'Yes', 'No']
                    )
                    out_text = res_dict["text"]
                    out_probs = res_dict.get("token_probs", {})
                except Exception as e:
                    print(f"Error generating for image {batch_indices[j]}: {e}")
                    out_text = "ERROR"
                    out_probs = {}

                res = {
                    "id": batch_rows[j].get("image_id", batch_rows[j].get("file", str(batch_indices[j]))),
                    "ground_truth": int(batch_rows[j].get(gt_col, -1)),
                    "prediction_text": out_text,
                    "prompt": batch_prompts[j],
                    "prob_yes": max(out_probs.get("yes", 0.0) or 0.0, out_probs.get("Yes", 0.0) or 0.0),
                    "prob_no": max(out_probs.get("no", 0.0) or 0.0, out_probs.get("No", 0.0) or 0.0)
                }
                results.append(res)
                
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            # Save partial?
            continue
            
    # 5. Save Results
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
