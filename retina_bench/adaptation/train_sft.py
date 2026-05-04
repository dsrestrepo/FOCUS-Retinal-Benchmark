import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import os
import torch
import argparse
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer, SFTConfig
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
    
    print(f"Loading processor and model: {args.model_id}")

    # Use models.py class to load
    ModelClass = get_model_class(args.model_id)
    print(f"Using model class: {ModelClass.__name__}")
    
    # Handle DDP device map
    device_map = None
    if os.environ.get("LOCAL_RANK") is not None:
        device_map = {"": int(os.environ.get("LOCAL_RANK"))}
        print(f"DDP Training: Using device_map={device_map}")
    else:
        # Single GPU Training
        device_map = "auto"
        print("Single GPU Training: Using device_map='auto'")

    vlm_wrapper = ModelClass(
        model_id=args.model_id,
        quantization=args.quantization,
        device="cuda",
        use_flash_attention=True,
        device_map=device_map
    )
    
    model = vlm_wrapper.model
    processor = vlm_wrapper.processor

    # Use right padding to avoid issues during training
    if hasattr(processor.tokenizer, "padding_side"):
        processor.tokenizer.padding_side = "right"
    
    # Enable gradient checkpointing manually on the model
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    print(f"Model class: {type(model)}")
    print(f"Model config: {model.config}")
    
    # 3. LoRA Config
    # Target modules depend on architecture. q_proj, v_proj, etc.
    # Updated to match notebook config
    # modules_to_save breaks for Llama 3 vision due to specific embedding/head ties
    modules_to_save = ["lm_head", "embed_tokens"]
    is_pure_llama = "llama" in args.model_id.lower() and "llava" not in args.model_id.lower()
    if is_pure_llama:
        modules_to_save = None

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        task_type="CAUSAL_LM", 
        bias="none",
        lora_dropout=0.05,
        modules_to_save=modules_to_save,
    )
    
    # Manually prepare model for LoRA training since we aren't using SFTTrainer anymore 
    model = prepare_model_for_kbit_training(model)
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()

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
    training_args = SFTConfig(
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
        dataset_text_field="text", 
        packing=False,
        #ddp_find_unused_parameters=False, # Important for multi-gpu if using DDP
        dataset_kwargs={"skip_prepare_dataset": True}, # Skip TRL's internal dataset processing/checks
        remove_unused_columns=False
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        # Pass the tokenizer explicitly for SFTTrainer checks as we have custom collation/process
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
