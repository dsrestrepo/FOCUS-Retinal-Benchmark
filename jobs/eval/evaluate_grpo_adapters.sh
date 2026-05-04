#!/bin/bash
#SBATCH --job-name=eval_grpo_medgemma
#SBATCH --account=dtn@h100
#SBATCH --partition=gpu_p6
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/eval_grpo.out
#SBATCH --error=logs/eval_grpo.err

# Clean modules
module purge
module load arch/h100
# Load required modules
module load miniforge/24.9.0
module load gcc/11.4.1
module load cuda/12.4.1
module load cudnn/9.2.0.82-cuda
module load nccl/2.21.5-1-cuda

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate llms

# Environment Configuration
export HF_HOME="$WORK/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Add project root to PYTHONPATH to ensure src module can be imported
export PYTHONPATH=$PYTHONPATH:.

# Source central paths
set -o allexport
source config/paths.env
set +o allexport

echo "Starting Evaluation Job on H100 for Adapters..."

# Load configs dynamically
MODELS=$(python -c 'import yaml; print(" ".join(yaml.safe_load(open("config/mllm_config.yaml"))["models"]))')
TRAIN_DATASETS=$(python -c 'import yaml; cfg=yaml.safe_load(open("config/mllm_config.yaml")); print(" ".join(cfg.get("train_datasets", cfg["datasets"])))')
TRAIN_TASKS=$(python -c 'import yaml; cfg=yaml.safe_load(open("config/mllm_config.yaml")); print(" ".join(cfg.get("train_tasks", cfg["tasks"])))')
TRAIN_STRATEGIES=$(python -c 'import yaml; print(" ".join(yaml.safe_load(open("config/mllm_config.yaml"))["grpo_strategies"]))')
EVAL_STRATEGIES=$(python -c 'import yaml; print(" ".join(yaml.safe_load(open("config/mllm_config.yaml"))["strategies"]))')
QUANTIZATION=$(python -c 'import yaml; print(yaml.safe_load(open("config/mllm_config.yaml"))["quantization"])')
USE_UNSLOTH=$(python -c 'import yaml; print(yaml.safe_load(open("config/mllm_config.yaml")).get("use_unsloth", False))')
EVAL_ROOT="${OUTPUT_DIR%/*}/evals_grpo"

eval_datasets_for_task() {
    python -c 'import sys, yaml
task = sys.argv[1]
cfg = yaml.safe_load(open("config/mllm_config.yaml"))
datasets = cfg.get("eval_datasets_ft", cfg["datasets"])
dataset_tasks = cfg.get("dataset_tasks", {})
print(" ".join(d for d in datasets if task in dataset_tasks.get(d, cfg["tasks"])))' "$1"
}

for model in $MODELS; do
    model_slug=$(echo $model | sed 's/\//_/g')
    for train_dataset in $TRAIN_DATASETS; do
        for train_task in $TRAIN_TASKS; do
            for train_strategy in $TRAIN_STRATEGIES; do
                # Target the adapter trained with train_strategy
                if [ "$USE_UNSLOTH" = "True" ]; then
                    ADAPTER_NAME="unsloth_grpo_${model_slug}_${train_dataset}_${train_task}_${train_strategy}"
                else
                    ADAPTER_NAME="grpo_${model_slug}_${train_dataset}_${train_task}_${train_strategy}"
                fi
                ADAPTER_PATH="$CHECKPOINTS_DIR/$ADAPTER_NAME"

                if [ ! -d "$ADAPTER_PATH" ]; then
                    echo "Warning: Adapter path $ADAPTER_PATH does not exist, skipping..."
                    continue
                fi

                ADAPTER_OUTPUT_DIR="$EVAL_ROOT/$ADAPTER_NAME"
                EVAL_DATASETS=$(eval_datasets_for_task "$train_task")
                eval_task="$train_task"

                for eval_dataset in $EVAL_DATASETS; do
                    for eval_strategy in $EVAL_STRATEGIES; do
                        echo "----------------------------------------------------------------"
                        echo "Running GRPO Eval: $model + adapter trained on $train_dataset/$train_task ($train_strategy) | Eval: $eval_dataset/$eval_task | Eval Strategy: $eval_strategy"
                        echo "Output: $ADAPTER_OUTPUT_DIR"
                        echo "----------------------------------------------------------------"

                        time python retina_bench/mllms/evaluate.py \
                            --dataset_path "$DATA_PATH" \
                            --dataset_name "$eval_dataset" \
                            --task "$eval_task" \
                            --model_id "$model" \
                            --adapter_path "$ADAPTER_PATH" \
                            --train_dataset_name "$train_dataset" \
                            --train_task "$train_task" \
                            --train_strategy "$train_strategy" \
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
    done
done

echo "Job completed."
