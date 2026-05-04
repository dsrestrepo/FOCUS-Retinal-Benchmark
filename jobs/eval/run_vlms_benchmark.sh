#!/bin/bash
#SBATCH --job-name=eval_vlms
#SBATCH --account=dtn@h100
#SBATCH --partition=gpu_p6
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/eval_vlms.out
#SBATCH --error=logs/eval_vlms.err

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

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=$PYTHONPATH:.

MODELS=$(python -c "import yaml; print(' '.join(yaml.safe_load(open('config/vlms_config.yaml'))['models']))")
DATASETS=$(python -c "import yaml; print(' '.join(yaml.safe_load(open('config/vlms_config.yaml'))['datasets']))")
TASKS=$(python -c "import yaml; print(' '.join(yaml.safe_load(open('config/vlms_config.yaml'))['tasks']))")
DATASET_TASKS=$(python -c "import yaml; cfg=yaml.safe_load(open('config/vlms_config.yaml')); print(' '.join(f'{d}:{t}' for d in cfg['datasets'] for t in cfg.get('dataset_tasks', {}).get(d, cfg['tasks'])))")
EVAL_METHODS=$(python -c "import yaml; print(' '.join(yaml.safe_load(open('config/vlms_config.yaml'))['eval_methods']))")

echo "Starting Evaluation Job on H100..."
echo "Models: $MODELS"
echo "Tasks: $TASKS"
echo "Datasets: $DATASETS"
echo "Dataset/task pairs: $DATASET_TASKS"
echo "Eval Methods: $EVAL_METHODS"

for dataset_task in $DATASET_TASKS; do
    dataset="${dataset_task%%:*}"
    task="${dataset_task#*:}"
    for method in $EVAL_METHODS; do
        for model in $MODELS; do
            echo "----------------------------------------------------------------"
            echo "Running Evaluation: $model | Dataset: $dataset | Task: $task | Method: $method"
            echo "----------------------------------------------------------------"
            
            time python retina_bench/vlms/evaluate.py \
                --dataset_path "$DATA_PATH" \
                --dataset_name "$dataset" \
                --task "$task" \
                --model_id "$model" \
                --method "$method" \
                --output_dir "$OUTPUT_DIR" \
                --batch_size 16 \
                --split "test" || echo "Error running $model on $dataset/$task/$method"
            
            echo "----------------------------------------------------------------"
        done
    done
done

echo "Job completed."
