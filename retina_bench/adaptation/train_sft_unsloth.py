# Ensure Unsloth is imported first
from unsloth import FastVisionModel
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
import argparse
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from trl import SFTConfig
from retina_bench.core.data import RetinaDataset
from retina_bench.mllms.evaluate import get_model_class
import retina_bench.mllms.prompts as prompts

def get_data_generator(dataset, dataset_name, tasks_list, strategy="base"):
    """Yields samples for SFT training using the specified prompt strategy."""
    
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
            
            strategies_to_use = ["base", "role", "cot"] if strategy == "all" else [strategy]

            for strat in strategies_to_use:
                prompt_func = prompts.get_prompt_func(dataset_name, task, strat)
                if not prompt_func: continue
                
                text_input = prompt_func(row)
                
                final_response = response
                
                yield {
                    "image": img,
                    "text": text_input,
                    "label": final_response
                }

def main():
    parser = argparse.ArgumentParser(description="SFT Fine-tuning for MedGemma")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="google/medgemma-27b-it") 
    parser.add_argument("--output_dir", type=str, default="results/sft_medgemma")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--quantization", type=str, default="16b", choices=["4b", "8b", "16b"])
    
    # Task/Prompt args to align with evaluation
    # Allow "both" as a choice since the script loads both anyway if requested
    parser.add_argument("--dataset_name", type=str, default="both", choices=["brset", "mbrset", "both"])
    parser.add_argument("--task", type=str, default="all", choices=["referable_dr", "binary_dr", "glaucoma", "all"])
    parser.add_argument("--prompt_strategy", type=str, default="base_grpo", choices=["base", "cot", "role", "base_grpo", "role_grpo", "all"])

    args = parser.parse_args()
    
    # Try mapping to local huggingface cache path for offline mode support
    import os
    hf_home = os.environ.get("HF_HOME", os.path.expandvars("$WORK/.cache/huggingface"))
    normalized_id = "models--" + args.model_id.replace("/", "--")
    snapshot_base = os.path.join(hf_home, "hub", normalized_id, "snapshots")
    
    if os.path.exists(snapshot_base):
        snapshots = [s for s in os.listdir(snapshot_base) if not s.startswith('.')]
        if snapshots:
            args.model_id = os.path.join(snapshot_base, snapshots[0])
            print(f"OFFLINE MODE: Using local snapshot dir for Unsloth: {args.model_id}")
    
    print("Loading processor and model via UNSLOTH: ", args.model_id)

    print(f"Loading Unsloth model ID: {args.model_id}")
    device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))} if os.environ.get("LOCAL_RANK") else "auto"

    model, processor = FastVisionModel.from_pretrained(
        model_name=args.model_id,
        load_in_4bit=(args.quantization == "4b"),
        use_gradient_checkpointing="unsloth", 
        device_map=device_map,
    )
    
    # 3. LoRA Config using Unsloth directly
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0, # optimized requires 0
        bias="none",
    )
    
    print(f"DEBUG: Model type: {type(model)}")
    print(f"DEBUG: Model config type: {type(model.config)}")
    if hasattr(model.config, "vision_config"):
        # print(f"DEBUG: Model has vision_config: {model.config.vision_config}")
        print("DEBUG: Model has vision_config attribute")
    else:
        print("DEBUG: Model does NOT have vision_config attribute")
    
    # 4. Prepare Dataset
    print(f"Loading Datasets for task {args.task}...")
    
    # Load requested datasets
    ds_list = []
    names_to_load = ["brset", "mbrset"] if args.dataset_name == "both" else [args.dataset_name]
    for d_name in names_to_load:
        try:
           d_set = RetinaDataset(args.dataset_path, d_name, split="train")
           print(f"Loaded {d_name}: {len(d_set)} samples")
           ds_list.append((d_set, d_name))
        except Exception as e:
           print(f"Skipping {d_name} due to error: {e}")

    # Define tasks we are training on
    if args.task == "all":
        TASKS = ["referable_dr", "binary_dr", "glaucoma"]
    else:
        TASKS = [args.task]
        
    print(f"Training on tasks: {TASKS}")

    def gen():
        for d_set, d_name in ds_list:
            yield from get_data_generator(d_set, d_name, TASKS, strategy=args.prompt_strategy)
        
    train_dataset = Dataset.from_generator(gen)
    
    # 5. Data Collator for formatting
    def collate_fn(examples):
        texts = []
        # images for Gemma3 processor must be list of lists of PIL images for batched inputs
        processed_images = [] 
        
        for ex in examples:
            # Create standard chat structure
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": ex['text']}
                    ]
                },
                {
                    "role": "assistant", 
                    "content": [
                        {"type": "text", "text": ex['label']}
                    ]
                }
            ]
            
            # Apply chat template (returns string)
            text_prompt = processor.apply_chat_template(
                messages, 
                add_generation_prompt=False, 
                tokenize=False
            )
            texts.append(text_prompt)
            
            # For batch processing, images arg takes a list (batch) of list (images per seq)
            # ex['image'] is a single PIL image
            if ex['image'] is not None:
                processed_images.append([ex['image']])
            else:
                processed_images.append([]) # Empty list if no image, though unexpected for this task

        # Use processor to handle tokenization and image processing
        inputs = processor(
            text=texts,
            images=processed_images,
            padding=True,
            return_tensors="pt", 
        )
        
        # Copy input_ids to labels
        labels = inputs["input_ids"].clone()

        # Gather standard image tokens to mask across different models
        image_token_ids = []
        
        # Qwen tokens
        for token in ["<|image_pad|>", "<|vision_pad|>", "<|vision_start|>", "<|vision_end|>"]:
            if token in processor.tokenizer.vocab:
                image_token_ids.append(processor.tokenizer.convert_tokens_to_ids(token))
            elif hasattr(processor.tokenizer, "added_tokens_encoder") and token in processor.tokenizer.added_tokens_encoder:
                image_token_ids.append(processor.tokenizer.added_tokens_encoder[token])
                
        # LLaVA / LLaMA tokens
        if "<image>" in processor.tokenizer.vocab:
            image_token_ids.append(processor.tokenizer.convert_tokens_to_ids("<image>"))
        elif hasattr(processor.tokenizer, "added_tokens_encoder") and "<image>" in processor.tokenizer.added_tokens_encoder:
            image_token_ids.append(processor.tokenizer.added_tokens_encoder["<image>"])
            
        # Gemma / Medgemma tokens
        if "boi_token" in processor.tokenizer.special_tokens_map:
            image_token_ids.append(
                processor.tokenizer.convert_tokens_to_ids(
                    processor.tokenizer.special_tokens_map["boi_token"]
                )
            )
        image_token_ids.append(262144) # known MedGemma image patch
        
        # General processor image_token_id
        if hasattr(processor, "image_token_id") and processor.image_token_id is not None:
            if isinstance(processor.image_token_id, int):
                image_token_ids.append(processor.image_token_id)
                
        # Mask tokens that are not used in the loss computation
        if processor.tokenizer.pad_token_id is not None:
            labels[labels == processor.tokenizer.pad_token_id] = -100
        
        for token_id in set(image_token_ids):
            if token_id is not None:
                labels[labels == token_id] = -100
        
        inputs["labels"] = labels
        
        return inputs    
    
    # 6. Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",        # Save at the end of each epoch instead of every 100 steps
        save_total_limit=3,           # Keep last 3 checkpoints
        fp16=False,
        bf16=True,
        # Dataloader optimization
        dataloader_num_workers=4,
        # Gradient Checkpointing - critical for memory
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        #ddp_find_unused_parameters=False, # Important for multi-gpu if using DDP
        remove_unused_columns=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=processor,
        data_collator=collate_fn
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model adapter...")
    trainer.save_model(args.output_dir)
    print(f"Adapter saved to {args.output_dir}. You can load this using the --adapter_path argument in future scripts.")

if __name__ == "__main__":
    main()
