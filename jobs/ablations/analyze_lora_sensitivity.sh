#!/bin/bash
#SBATCH --job-name=lora_sens_analysis
#SBATCH --account=dtn@cpu
#SBATCH --partition=compil
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/lora_sens_analysis_%j.out
#SBATCH --error=logs/lora_sens_analysis_%j.err

set -euo pipefail

REPO_ROOT=${FOCUS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO_ROOT"
if [ ! -f config/paths.env ]; then
    echo "Missing $REPO_ROOT/config/paths.env. Submit from the repository root or set FOCUS_REPO_ROOT."
    exit 2
fi

module purge
module load miniforge/24.9.0
eval "$(conda shell.bash hook)"
conda activate llms

export PYTHONPATH="${PYTHONPATH:-}:."

set -o allexport
source config/paths.env
set +o allexport

CONFIG=${LORA_ABLATION_CONFIG:-config/ablations/lora_sensitivity.yaml}
EVAL_ROOT=${LORA_ABLATION_EVAL_DIR:-"${OUTPUT_DIR%/*}/evals_lora_sensitivity"}
REFERENCE_ROOT=${LORA_REFERENCE_EVAL_DIR:-"${OUTPUT_DIR%/*}/evals_sft_all"}
BASELINE_ROOT=${LORA_BASELINE_EVAL_DIR:-"${OUTPUT_DIR%/*}/evals"}
ANALYSIS_ROOT=${LORA_ABLATION_ANALYSIS_DIR:-"${OUTPUT_DIR%/*}/analysis_lora_sensitivity"}

python scripts/ablations/analyze_lora_sensitivity.py \
    --config "$CONFIG" \
    --ablation-dir "$EVAL_ROOT" \
    --reference-dir "$REFERENCE_ROOT" \
    --baseline-dir "$BASELINE_ROOT" \
    --output-dir "$ANALYSIS_ROOT"
