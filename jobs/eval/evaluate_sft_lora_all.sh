#!/bin/bash
#SBATCH --job-name=eval_sft_all
#SBATCH --account=dtn@h100
#SBATCH --partition=gpu_p6
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/eval_sft_all.out
#SBATCH --error=logs/eval_sft_all.err

module purge
module load arch/h100
module load miniforge/24.9.0
module load gcc/11.4.1
module load cuda/12.4.1
module load cudnn/9.2.0.82-cuda
module load nccl/2.21.5-1-cuda

eval "$(conda shell.bash hook)"
conda activate llms

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$PYTHONPATH:.

set -o allexport
source config/paths.env
set +o allexport
source jobs/common_paths.sh

CHECKPOINT_ROOT="results/checkpoints_lora_all"
EVAL_ROOT="$SFT_ALL_EVAL_DIR"
TRAIN_TASK_LABEL="all_tasks"
TRAIN_STRATEGY_LABEL="all_prompts"

echo "Starting all-task SFT adapter evaluation..."

MODELS=$(python -c 'import yaml; print(" ".join(yaml.safe_load(open("config/mllm_config.yaml"))["models"]))')
TRAIN_DATASETS=$(python -c 'import yaml; cfg=yaml.safe_load(open("config/mllm_config.yaml")); print(" ".join(cfg.get("lora_all_train_datasets", cfg.get("train_datasets", cfg["datasets"]))))')
EVAL_STRATEGIES=$(python -c 'import yaml; print(" ".join(yaml.safe_load(open("config/mllm_config.yaml"))["strategies"]))')
QUANTIZATION=$(python -c 'import yaml; print(yaml.safe_load(open("config/mllm_config.yaml"))["quantization"])')
USE_UNSLOTH=$(python -c 'import yaml; print(yaml.safe_load(open("config/mllm_config.yaml")).get("use_unsloth", False))')

eval_pairs_for_train_dataset() {
    python -c 'import sys, yaml
train_dataset = sys.argv[1]
cfg = yaml.safe_load(open("config/mllm_config.yaml"))
all_tasks = cfg.get("tasks", [])
dataset_tasks = cfg.get("dataset_tasks", {})
train_tasks = set(dataset_tasks.get(train_dataset, all_tasks))
eval_datasets = cfg.get("eval_datasets_ft", cfg["datasets"])
eval_tasks = set(cfg.get("eval_tasks_ft", all_tasks))
pairs = []
for dataset in eval_datasets:
    for task in dataset_tasks.get(dataset, all_tasks):
        if task in train_tasks and task in eval_tasks:
            pairs.append(f"{dataset}:{task}")
print(" ".join(pairs))' "$1"
}

for model in $MODELS; do
    model_slug=$(echo "$model" | sed 's/\//_/g')
    for train_dataset in $TRAIN_DATASETS; do
        if [ "$USE_UNSLOTH" = "True" ]; then
            ADAPTER_NAME="unsloth_sft_all_${model_slug}_${train_dataset}_${TRAIN_STRATEGY_LABEL}"
        else
            ADAPTER_NAME="sft_all_${model_slug}_${train_dataset}_${TRAIN_STRATEGY_LABEL}"
        fi
        ADAPTER_PATH="$CHECKPOINT_ROOT/$ADAPTER_NAME"

        if [ ! -d "$ADAPTER_PATH" ]; then
            echo "Warning: Adapter path $ADAPTER_PATH does not exist, skipping..."
            continue
        fi

        ADAPTER_OUTPUT_DIR="$EVAL_ROOT/$ADAPTER_NAME"
        EVAL_PAIRS=$(eval_pairs_for_train_dataset "$train_dataset")

        for eval_pair in $EVAL_PAIRS; do
            eval_dataset="${eval_pair%%:*}"
            eval_task="${eval_pair#*:}"
            for eval_strategy in $EVAL_STRATEGIES; do
                echo "----------------------------------------------------------------"
                echo "Running all-task SFT Eval: $model + adapter trained on $train_dataset/all_tasks | Eval: $eval_dataset/$eval_task | Eval Strategy: $eval_strategy"
                echo "Output: $ADAPTER_OUTPUT_DIR"
                echo "----------------------------------------------------------------"

                time python retina_bench/mllms/evaluate.py \
                    --dataset_path "$DATA_PATH" \
                    --dataset_name "$eval_dataset" \
                    --task "$eval_task" \
                    --model_id "$model" \
                    --adapter_path "$ADAPTER_PATH" \
                    --train_dataset_name "$train_dataset" \
                    --train_task "$TRAIN_TASK_LABEL" \
                    --train_strategy "$TRAIN_STRATEGY_LABEL" \
                    --prompt_strategy "$eval_strategy" \
                    --quantization "$QUANTIZATION" \
                    --output_dir "$ADAPTER_OUTPUT_DIR" \
                    --batch_size 16 \
                    --split "test" \
                    --use_flash_attn || echo "Error running eval on $model adapter $ADAPTER_NAME with $eval_dataset/$eval_task/$eval_strategy"

                echo "----------------------------------------------------------------"
            done
        done
    done
done

echo "All-task SFT adapter evaluation completed."
