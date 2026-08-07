#!/bin/bash
#SBATCH --job-name=lora_paper_eval
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
#SBATCH --output=logs/lora_paper_eval_%A_%a.out
#SBATCH --error=logs/lora_paper_eval_%A_%a.err

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

export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${PYTHONPATH:-}:."

set -o allexport
source config/paths.env
set +o allexport

CONFIG=${LORA_PAPER_ABLATION_CONFIG:-config/ablations/lora_sensitivity_paper_aligned.yaml}
CHECKPOINT_ROOT=${LORA_PAPER_ABLATION_CHECKPOINT_DIR:-"${CHECKPOINTS_DIR%/*}/checkpoints_lora_sensitivity_paper_aligned"}
EVAL_ROOT=${LORA_PAPER_ABLATION_EVAL_DIR:-"${OUTPUT_DIR%/*}/evals_lora_sensitivity_paper_aligned"}
GRID_INDEX=${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array.}

GRID_LINE=$(python -c 'import sys,yaml
c=yaml.safe_load(open(sys.argv[1]))
grid=[(m,t,v) for m in c["models"] for t in c["train_tasks"] for v in c["variants"]]
i=int(sys.argv[2])
if i < 0 or i >= len(grid): raise SystemExit(f"grid index {i} outside 0..{len(grid)-1}")
m,t,v=grid[i]
print("\t".join([m,t,v["name"],str(c["train_dataset"])]))' "$CONFIG" "$GRID_INDEX")

IFS=$'\t' read -r MODEL TRAIN_TASK VARIANT TRAIN_DATASET <<< "$GRID_LINE"
MODEL_SLUG=${MODEL//\//_}
ADAPTER_NAME="sft_paper_ablation_${MODEL_SLUG}_${TRAIN_DATASET}_${TRAIN_TASK}_${VARIANT}"
ADAPTER_PATH="$CHECKPOINT_ROOT/$ADAPTER_NAME"
ADAPTER_OUTPUT_DIR="$EVAL_ROOT/$ADAPTER_NAME"

if [ ! -f "$ADAPTER_PATH/adapter_config.json" ]; then
    echo "Completed adapter not found: $ADAPTER_PATH"
    exit 3
fi

python scripts/ablations/evaluate_mllm_prompt_sensitivity.py \
    --config "$CONFIG" \
    --model-id "$MODEL" \
    --prompt-variant base \
    --dataset-key eval_datasets \
    --evaluation-task "$TRAIN_TASK" \
    --dataset-path "$DATA_PATH" \
    --output-dir "$ADAPTER_OUTPUT_DIR" \
    --adapter-path "$ADAPTER_PATH" \
    --train-dataset "$TRAIN_DATASET" \
    --train-task "$TRAIN_TASK" \
    --train-strategy "$VARIANT" \
    --use-flash-attn
