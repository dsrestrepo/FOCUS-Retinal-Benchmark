#!/bin/bash
#SBATCH --job-name=mllm_prompt_abl
#SBATCH --account=dtn@h100
#SBATCH --partition=gpu_p6
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --array=0-11%6
#SBATCH --output=logs/mllm_prompt_abl_%A_%a.out
#SBATCH --error=logs/mllm_prompt_abl_%A_%a.err

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

export HF_HOME="$WORK/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${PYTHONPATH:-}:."

set -o allexport
source config/paths.env
if [ -f .env ]; then
    source .env
fi
set +o allexport

CONFIG=${ABLATION_CONFIG:-config/ablations/mllm_prompt_parsing.yaml}
ABLATION_ROOT=${MLLM_PROMPT_ABLATION_DIR:-"${OUTPUT_DIR%/*}/ablations/mllm_prompt_sensitivity"}
GRID_INDEX=${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array.}

EXPECTED_GRID_SIZE=$(python -c 'import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); print(len(c["models"])*len(c["prompt_variants"]))' "$CONFIG")
if [ "$GRID_INDEX" -ge "$EXPECTED_GRID_SIZE" ]; then
    echo "Array index $GRID_INDEX exceeds the configured grid size $EXPECTED_GRID_SIZE."
    exit 2
fi

echo "Prompt sensitivity array index: $GRID_INDEX / $((EXPECTED_GRID_SIZE - 1))"
echo "Config: $CONFIG"
echo "Output: $ABLATION_ROOT"

python scripts/ablations/evaluate_mllm_prompt_sensitivity.py \
    --config "$CONFIG" \
    --grid-index "$GRID_INDEX" \
    --dataset-path "$DATA_PATH" \
    --output-dir "$ABLATION_ROOT" \
    --use-flash-attn
