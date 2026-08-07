#!/bin/bash
#SBATCH --job-name=ft_grpo_all
#SBATCH --account=dtn@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/ft_grpo_all.out
#SBATCH --error=logs/ft_grpo_all.err

module purge
module load arch/h100
module load miniforge/24.9.0
module load gcc/11.4.1
module load cuda/12.4.1
module load cudnn/9.2.0.82-cuda
module load nccl/2.21.5-1-cuda

eval "$(conda shell.bash hook)"
conda activate llms

export HF_HOME="$WORK/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$PYTHONPATH:.

set -o allexport
source config/paths.env
set +o allexport

CHECKPOINT_ROOT="results/checkpoints_lora_all"
TRAIN_TASK="all"
TRAIN_STRATEGY="all"
TRAIN_STRATEGY_LABEL="all_prompts"
mkdir -p "$CHECKPOINT_ROOT"

echo "Starting all-task/all-prompt GRPO LoRA training job..."

MODELS=$(python -c 'import yaml; print(" ".join(yaml.safe_load(open("config/mllm_config.yaml"))["models"]))')
DATASETS=$(python -c 'import yaml; cfg=yaml.safe_load(open("config/mllm_config.yaml")); print(" ".join(cfg.get("lora_all_train_datasets", cfg.get("train_datasets", cfg["datasets"]))))')
QUANTIZATION=$(python -c 'import yaml; print(yaml.safe_load(open("config/mllm_config.yaml"))["quantization"])')
USE_UNSLOTH=$(python -c 'import yaml; print(yaml.safe_load(open("config/mllm_config.yaml")).get("use_unsloth", False))')

echo "Models: $MODELS"
echo "Train datasets: $DATASETS"
echo "Quantization: $QUANTIZATION"

for model in $MODELS; do
    model_slug=$(echo "$model" | sed 's/\//_/g')
    for dataset in $DATASETS; do
        if [ "$USE_UNSLOTH" = "True" ]; then
            out_dir="$CHECKPOINT_ROOT/unsloth_grpo_all_${model_slug}_${dataset}_${TRAIN_STRATEGY_LABEL}"
            TRAINING_SCRIPT="retina_bench/adaptation/train_grpo_unsloth.py"
        else
            out_dir="$CHECKPOINT_ROOT/grpo_all_${model_slug}_${dataset}_${TRAIN_STRATEGY_LABEL}"
            TRAINING_SCRIPT="retina_bench/adaptation/train_grpo.py"
        fi

        if [[ "$model" == *"27b"* || "$model" == *"Llama-3.2-11B"* ]]; then
            BATCH_SIZE=1
            GRAD_ACCUM=8
            NUM_GEN=2
            EPOCHS=1
        else
            BATCH_SIZE=2
            GRAD_ACCUM=4
            NUM_GEN=2
            EPOCHS=1
        fi

        echo "Running all-task GRPO: Model=$model | Dataset=$dataset | Output=$out_dir"

        MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

        torchrun \
            --nnodes=$SLURM_NNODES \
            --nproc_per_node=4 \
            --rdzv_id=$SLURM_JOB_ID \
            --rdzv_backend=c10d \
            --rdzv_endpoint=$MASTER_ADDR:29500 \
            "$TRAINING_SCRIPT" \
            --dataset_path "$DATA_PATH" \
            --model_id "$model" \
            --output_dir "$out_dir" \
            --dataset_name "$dataset" \
            --task "$TRAIN_TASK" \
            --prompt_strategy "$TRAIN_STRATEGY" \
            --quantization "$QUANTIZATION" \
            --epochs "$EPOCHS" \
            --num_generations "$NUM_GEN" \
            --batch_size "$BATCH_SIZE" \
            --grad_accum "$GRAD_ACCUM" \
            --lora_r 16 \
            --lr 2e-6
    done
done

echo "All-task GRPO LoRA training completed."
