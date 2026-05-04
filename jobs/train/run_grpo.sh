#!/bin/bash
#SBATCH --job-name=ft_grpo
#SBATCH --account=dtn@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=18:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=logs/ft_grpo.out
#SBATCH --error=logs/ft_grpo.err

# Clean modules
module purge
module load arch/h100
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
# Ensure correct import paths
export PYTHONPATH=$PYTHONPATH:.

# Centralized Path variables
set -o allexport
source config/paths.env
set +o allexport

echo "Starting GRPO Training Job..."

# Load configs dynamically from yaml
MODELS=$(python -c 'import yaml; print(" ".join(yaml.safe_load(open("config/mllm_config.yaml"))["models"]))')
DATASETS=$(python -c 'import yaml; cfg=yaml.safe_load(open("config/mllm_config.yaml")); print(" ".join(cfg.get("train_datasets", cfg["datasets"])))')
TASKS=$(python -c 'import yaml; cfg=yaml.safe_load(open("config/mllm_config.yaml")); print(" ".join(cfg.get("train_tasks", cfg["tasks"])))')
STRATEGIES=$(python -c 'import yaml; print(" ".join(yaml.safe_load(open("config/mllm_config.yaml"))["grpo_strategies"]))')
QUANTIZATION=$(python -c 'import yaml; print(yaml.safe_load(open("config/mllm_config.yaml"))["quantization"])')
USE_UNSLOTH=$(python -c 'import yaml; print(yaml.safe_load(open("config/mllm_config.yaml")).get("use_unsloth", False))')

for model in $MODELS; do
    for dataset in $DATASETS; do
        for task in $TASKS; do
            for strategy in $STRATEGIES; do
                model_slug=$(echo $model | sed 's/\//_/g')
                
                if [ "$USE_UNSLOTH" = "True" ]; then
                    out_dir="$CHECKPOINTS_DIR/unsloth_grpo_${model_slug}_${dataset}_${task}_${strategy}"
                    TRAINING_SCRIPT="retina_bench/adaptation/train_grpo_unsloth.py"
                else
                    out_dir="$CHECKPOINTS_DIR/grpo_${model_slug}_${dataset}_${task}_${strategy}"
                    TRAINING_SCRIPT="retina_bench/adaptation/train_grpo.py"
                fi

                # Dynamic batch sizing logic to handle large vs small models natively
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
                
                echo "Running GRPO ($QUANTIZATION): Model=$model | Dataset=$dataset | Task=$task | Strategy=$strategy"
                
                # Setup multi-node torchrun networking
                MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
                
                torchrun \
                    --nnodes=$SLURM_NNODES \
                    --nproc_per_node=4 \
                    --rdzv_id=$SLURM_JOB_ID \
                    --rdzv_backend=c10d \
                    --rdzv_endpoint=$MASTER_ADDR:29500 \
                    $TRAINING_SCRIPT \
                    --dataset_path "$DATA_PATH" \
                    --model_id "$model" \
                    --output_dir "$out_dir" \
                    --dataset_name "$dataset" \
                    --task "$task" \
                    --prompt_strategy "$strategy" \
                    --quantization "$QUANTIZATION" \
                    --epochs "$EPOCHS" \
                    --num_generations "$NUM_GEN" \
                    --batch_size "$BATCH_SIZE" \
                    --grad_accum "$GRAD_ACCUM" \
                    --lora_r 16 \
                    --lr 2e-6
            done
        done
    done
done

echo "Training Job Completed."
