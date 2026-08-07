#!/bin/bash
#SBATCH --job-name=lp_bootstrap
#SBATCH --account=dtn@h100
#SBATCH --partition=gpu_p6
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --array=0-15%2
#SBATCH --output=logs/lp_bootstrap_%A_%a.out
#SBATCH --error=logs/lp_bootstrap_%A_%a.err

set -euo pipefail

REPO_ROOT=${FOCUS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO_ROOT"
if [ ! -f config/paths.env ]; then
    echo "Missing $REPO_ROOT/config/paths.env. Submit from the repository root or set FOCUS_REPO_ROOT."
    exit 2
fi

module purge
module load arch/h100
module load miniforge/24.9.0
module load gcc/11.4.1
module load cuda/12.4.1
module load cudnn/9.2.0.82-cuda
module load nccl/2.21.5-1-cuda

eval "$(conda shell.bash hook)"
conda activate llms

set -o allexport
source config/paths.env
if [ -f .env ]; then
    source .env
fi
set +o allexport

export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="${PYTHONPATH:-}:."

CONFIG=${LINEAR_PROBE_BOOTSTRAP_CONFIG:-config/ablations/linear_probe_bootstrap.yaml}
BOOTSTRAP_ROOT=${LINEAR_PROBE_BOOTSTRAP_DIR:-"${OUTPUT_DIR%/*}/ablations/linear_probe_bootstrap_random_init"}
GRID_INDEX=${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array.}

EXPECTED_GRID_SIZE=$(python -c 'import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); print(len(c["models"]))' "$CONFIG")
if [ "$GRID_INDEX" -ge "$EXPECTED_GRID_SIZE" ]; then
    echo "Array index $GRID_INDEX exceeds the configured grid size $EXPECTED_GRID_SIZE."
    exit 2
fi

python scripts/ablations/evaluate_linear_probe_bootstrap_random_init.py \
    --config "$CONFIG" \
    --grid-index "$GRID_INDEX" \
    --dataset-path "$DATA_PATH" \
    --output-dir "$BOOTSTRAP_ROOT"
