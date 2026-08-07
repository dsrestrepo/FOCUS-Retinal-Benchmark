#!/bin/bash
#SBATCH --job-name=mllm_parse_abl
#SBATCH --account=dtn@cpu
#SBATCH --partition=compil
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/mllm_parse_abl_%j.out
#SBATCH --error=logs/mllm_parse_abl_%j.err

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

CONFIG=${ABLATION_CONFIG:-config/ablations/mllm_prompt_parsing.yaml}
ABLATION_ROOT=${MLLM_PROMPT_ABLATION_DIR:-"${OUTPUT_DIR%/*}/ablations/mllm_prompt_sensitivity"}
ANALYSIS_ROOT=${MLLM_PROMPT_ANALYSIS_DIR:-"${OUTPUT_DIR%/*}/analysis_prompt_parsing_sensitivity"}

python scripts/ablations/analyze_mllm_parsing_sensitivity.py \
    --config "$CONFIG" \
    --baseline-dir "$OUTPUT_DIR" \
    --ablation-dir "$ABLATION_ROOT" \
    --output-dir "$ANALYSIS_ROOT" \
    --data-dir "$DATA_PATH" \
    --fundus-config config/fundus_datasets.yaml
