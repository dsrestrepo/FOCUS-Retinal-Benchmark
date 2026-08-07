#!/bin/bash
#SBATCH --job-name=vlm_prompt_analysis
#SBATCH --account=dtn@cpu
#SBATCH --partition=compil
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/vlm_prompt_analysis_%j.out
#SBATCH --error=logs/vlm_prompt_analysis_%j.err

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

CONFIG=${VLM_PROMPT_ABLATION_CONFIG:-config/ablations/vlm_zero_shot_prompt_sensitivity.yaml}
BASELINE_ROOT=${VLM_PROMPT_BASELINE_DIR:-"$OUTPUT_DIR"}
ABLATION_ROOT=${VLM_PROMPT_ABLATION_DIR:-"${OUTPUT_DIR%/*}/ablations/vlm_zero_shot_prompt_sensitivity"}
ANALYSIS_ROOT=${VLM_PROMPT_ANALYSIS_DIR:-"${OUTPUT_DIR%/*}/analysis_vlm_zero_shot_prompt_sensitivity"}

python scripts/ablations/analyze_vlm_zero_shot_prompt_sensitivity.py \
    --config "$CONFIG" \
    --baseline-dir "$BASELINE_ROOT" \
    --ablation-dir "$ABLATION_ROOT" \
    --output-dir "$ANALYSIS_ROOT" \
    --data-dir "$DATA_PATH"
