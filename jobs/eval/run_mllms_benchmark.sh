#!/bin/bash
#SBATCH --job-name=eval_retina
#SBATCH --account=dtn@h100
#SBATCH --partition=gpu_p6
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/eval_mllms.out
#SBATCH --error=logs/eval_mllms.err

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

# Centralized Path variables
set -o allexport
source config/paths.env
if [ -f .env ]; then
    source .env
fi
set +o allexport

export PYTHONPATH=$PYTHONPATH:.

# Load configuration dynamically
MODELS=$(python -c 'import yaml; print(" ".join(yaml.safe_load(open("config/mllm_config.yaml"))["models"]))')
DATASETS=$(python -c 'import yaml; print(" ".join(yaml.safe_load(open("config/mllm_config.yaml"))["datasets"]))')
TASKS=$(python -c 'import yaml; print(" ".join(yaml.safe_load(open("config/mllm_config.yaml"))["tasks"]))')
DATASET_TASKS=$(python -c 'import yaml; cfg=yaml.safe_load(open("config/mllm_config.yaml")); print(" ".join(f"{d}:{t}" for d in cfg["datasets"] for t in cfg.get("dataset_tasks", {}).get(d, cfg["tasks"])))')
STRATEGIES=$(python -c 'import yaml; print(" ".join(yaml.safe_load(open("config/mllm_config.yaml"))["strategies"]))')
QUANTIZATION=$(python -c 'import yaml; print(yaml.safe_load(open("config/mllm_config.yaml"))["quantization"])')

echo "Starting Zero-Shot Benchmarking Evaluation Job on H100..."
echo "Models: $MODELS"
echo "Tasks: $TASKS"
echo "Datasets: $DATASETS"
echo "Dataset/task pairs: $DATASET_TASKS"
echo "Strategies: $STRATEGIES"
echo "Quantization: $QUANTIZATION"

for dataset_task in $DATASET_TASKS; do
    dataset="${dataset_task%%:*}"
    task="${dataset_task#*:}"
    for model in $MODELS; do
        for strategy in $STRATEGIES; do
            
            echo "----------------------------------------------------------------"
            echo "Running Eval: Dataset=$dataset | Task=$task | Model=$model | Strategy=$strategy | Quantization=$QUANTIZATION"
            echo "----------------------------------------------------------------"
            
            time python retina_bench/mllms/evaluate.py \
                --dataset_path "$DATA_PATH" \
                --dataset_name "$dataset" \
                --task "$task" \
                --model_id "$model" \
                --prompt_strategy "$strategy" \
                --quantization "$QUANTIZATION" \
                --output_dir "$OUTPUT_DIR" \
                --batch_size 16 \
                --split "test" \
                --use_flash_attn || echo "Error running eval on $model with $dataset/$task/$strategy"
            
            echo "----------------------------------------------------------------"
        done
    done
done

echo "Job completed."
