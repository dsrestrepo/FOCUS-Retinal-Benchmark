import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import os
import torch
import argparse
import re
import logging
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoProcessor
from trl import GRPOConfig, GRPOTrainer
from retina_bench.core.data import RetinaDataset
from retina_bench.mllms.evaluate import get_model_class
import retina_bench.mllms.prompts as prompts

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_data_generator(dataset, dataset_name, tasks_list, strategy="cot"):
    """Yields samples for RLVR training."""
    
    gt_map = {
        "referable_dr": "Task_Referable",
        "binary_dr": "DR_2_Class",
        "glaucoma": "Task_Glaucoma",
    }


    for i in range(len(dataset)):
        row = dataset.get_row(i)
        img = dataset.get_image(i)
        if img is None: continue
        
        for task in tasks_list:
            gt_col = gt_map.get(task)
            if not gt_col: continue

            gt_val = row.get(gt_col, -1) 
            response = None
            
            # Check if valid binary label exists
            if gt_val == 1: response = "yes"
            elif gt_val == 0: response = "no"

            if response is None:
                continue

            strategies_to_use = ["base_grpo", "role_grpo", "cot"] if strategy == "all" else [strategy]

            for strat in strategies_to_use:
                prompt_func = prompts.get_prompt_func(dataset_name, task, strat)
                if not prompt_func: continue
                
                text_input = prompt_func(row)
                
                # Construct the prompt structure
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": text_input}
                        ]
                    }
                ]
                
                yield {
                    "prompt": conversation,
                    "answer": response,
                    "image": img 
                }

# Reward Functions
def extract_answer(completion: str) -> str:
    """Extract the answer (yes/no) from the completion."""
    if not isinstance(completion, str):
        return None
    if not completion:
        return None

    completion_lower = completion.lower()

    # Specific patterns for "Answer: yes/no" as requested in CoT prompts
    final_answers = [
        r'answer:\s*(yes|no)',
        r'answer:\s*\**\s*(yes|no)\s*\**',
        r'the final answer is\s*(yes|no)',
        r'final answer:\s*(yes|no)',
        r'\**final answer:\**\s*(yes|no)',
        r'answer is\s*(yes|no)',
        r'answer\s*(yes|no)',
    ]
    
    for pattern in final_answers:
        match = re.search(pattern, completion_lower)
        if match:
            return match.group(1)
            
    return None

def correctness_reward_func(prompts, completions, answer, **kwargs):
    """
    Reward function that checks if the completion contains the correct answer.
    """
    rewards = []
    for completion, ground_truth in zip(completions, answer):
        # Ensure completion is a string
        if isinstance(completion, list):
            completion = completion[0]["content"] if completion and isinstance(completion[0], dict) else str(completion)
        
        extracted_answer = extract_answer(completion)
        ground_truth_lower = ground_truth.lower()
        
        if extracted_answer and extracted_answer == ground_truth_lower:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return rewards

# TODO: format_reward_func can be added for checking if the output format is correct (e.g., contains "Answer: yes/no") to encourage better adherence to instructions, but we will focus on correctness for now as per instruction.


def main():
    parser = argparse.ArgumentParser(description="RLVR (GRPO) for MedGemma")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="google/medgemma-4b-it") 
    parser.add_argument("--output_dir", type=str, default="results/rlvr_medgemma")
    
    # GRPO specific args
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    
    # Training args
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--quantization", type=str, default="16b", choices=["4b", "8b", "16b"])
    
    # Data args
    parser.add_argument("--dataset_name", type=str, default="both", choices=["brset", "mbrset", "both"])
    parser.add_argument("--task", type=str, default="all", choices=["referable_dr", "binary_dr", "glaucoma", "all"])
    parser.add_argument("--prompt_strategy", type=str, default="base_grpo", choices=["base", "cot", "role", "base_grpo", "role_grpo", "all"])

    args = parser.parse_args()
    
    logger.info(f"Loading processor and model: {args.model_id}")

    # Use models.py class to load
    ModelClass = get_model_class(args.model_id)
    logger.info(f"Using model class: {ModelClass.__name__}")
    
    device_map = "auto"
    if os.environ.get("LOCAL_RANK") is not None:
        device_map = {"": int(os.environ.get("LOCAL_RANK"))}

    # Disable flash attention specifically for Llama to avoid SDPA padding issues in GRPO
    # Other models will continue to use flash attention normally
    use_fa = not ("llama" in args.model_id.lower() and "llava" not in args.model_id.lower())

    vlm_wrapper = ModelClass(
        model_id=args.model_id,
        quantization=args.quantization,
        device="cuda",
        use_flash_attention=use_fa,
        device_map=device_map
    )
    
    model = vlm_wrapper.model
    processor = vlm_wrapper.processor

    # Enable gradients for LoRA
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # LoRA Config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        task_type="CAUSAL_LM", 
        bias="none",
        lora_dropout=0.05,
    )
    
    # Load Datasets
    logger.info("Loading Datasets...")
    ds_list = []
    
    names_to_load = ["brset", "mbrset"] if args.dataset_name == "both" else [args.dataset_name]
    
    for d_name in names_to_load:
        try:
           d_set = RetinaDataset(args.dataset_path, d_name, split="train")
           logger.info(f"Loaded {d_name}: {len(d_set)} samples")
           ds_list.append((d_set, d_name))
        except Exception as e:
           logger.error(f"Skipping {d_name} due to error: {e}")

    if args.task == "all":
        TASKS = ["referable_dr", "binary_dr", "glaucoma"]
    else:
        TASKS = [args.task]
        
    logger.info(f"Training on tasks: {TASKS}")

    def gen():
        for d_set, d_name in ds_list:
            yield from get_data_generator(d_set, d_name, TASKS, strategy=args.prompt_strategy)
        
    train_dataset = Dataset.from_generator(gen)
    
    # GRPO Config
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        fp16=False,
        bf16=True,
        # Dataloader optimization
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none", 
        remove_unused_columns=False, 
        use_vllm=False, 
        beta=args.beta,
    )
    
    # Remove data_collator as GRPOTrainer manages it internally or doesn't support override in this version
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[correctness_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=processor
    )
    
    logger.info("Starting RLVF training...")
    trainer.train()
    
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
