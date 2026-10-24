#!/bin/bash
#SBATCH --job-name=lora_paper_train
#SBATCH --account=dtn@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --array=0-11%6
#SBATCH --output=logs/lora_paper_train_%A_%a.out
#SBATCH --error=logs/lora_paper_train_%A_%a.err

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

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${PYTHONPATH:-}:."

set -o allexport
source config/paths.env
set +o allexport
source jobs/common_paths.sh

CONFIG=${LORA_PAPER_ABLATION_CONFIG:-config/ablations/lora_sensitivity_paper_aligned.yaml}
CHECKPOINT_ROOT=${LORA_PAPER_ABLATION_CHECKPOINT_DIR:-"${CHECKPOINTS_DIR%/*}/checkpoints_lora_sensitivity_paper_aligned"}
GRID_INDEX=${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array.}
mkdir -p "$CHECKPOINT_ROOT"

GRID_LINE=$(python -c 'import sys,yaml
c=yaml.safe_load(open(sys.argv[1]))
grid=[(m,t,v) for m in c["models"] for t in c["train_tasks"] for v in c["variants"]]
i=int(sys.argv[2])
if i < 0 or i >= len(grid): raise SystemExit(f"grid index {i} outside 0..{len(grid)-1}")
m,t,v=grid[i]
print("\t".join([m,t,v["name"],str(v["lora_r"]),str(v["lora_alpha"]),str(v["lr"]),v["target_modules"],str(c["train_dataset"]),str(c["prompt_strategy"]),str(c["quantization"]),str(c["epochs"])]))' "$CONFIG" "$GRID_INDEX")

IFS=$'\t' read -r MODEL TRAIN_TASK VARIANT LORA_R LORA_ALPHA LR TARGET_MODULES TRAIN_DATASET TRAIN_STRATEGY QUANTIZATION EPOCHS <<< "$GRID_LINE"
MODEL_SLUG=${MODEL//\//_}
OUT_DIR="$CHECKPOINT_ROOT/sft_paper_ablation_${MODEL_SLUG}_${TRAIN_DATASET}_${TRAIN_TASK}_${VARIANT}"

if [ -f "$OUT_DIR/adapter_config.json" ]; then
    echo "[skip] Completed adapter exists: $OUT_DIR"
    exit 0
fi
if [ -d "$OUT_DIR" ]; then
    echo "Refusing to overwrite incomplete adapter directory: $OUT_DIR"
    exit 3
fi

BATCH_SIZE=1
GRAD_ACCUM=8
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "Model=$MODEL Dataset=$TRAIN_DATASET Task=$TRAIN_TASK Variant=$VARIANT"
echo "r=$LORA_R alpha=$LORA_ALPHA lr=$LR target_modules=$TARGET_MODULES"

torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node=4 \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:29500" \
    scripts/ablations/train_sft_lora_ablation.py \
    --dataset_path "$DATA_PATH" \
    --model_id "$MODEL" \
    --output_dir "$OUT_DIR" \
    --dataset_name "$TRAIN_DATASET" \
    --task "$TRAIN_TASK" \
    --prompt_strategy "$TRAIN_STRATEGY" \
    --quantization "$QUANTIZATION" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_target_modules "$TARGET_MODULES" \
    --lr "$LR"
