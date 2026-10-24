#!/bin/bash
#SBATCH --job-name=lp_boot_subset
#SBATCH --account=dtn@cpu
#SBATCH --partition=compil
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/lp_boot_subset_%j.out
#SBATCH --error=logs/lp_boot_subset_%j.err

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
source jobs/common_paths.sh

CONFIG=${LINEAR_PROBE_SUBSET_CONFIG:-config/ablations/linear_probe_bootstrap_completed_subset.yaml}
BOOTSTRAP_ROOT=${LINEAR_PROBE_BOOTSTRAP_DIR:-"${RESULTS_ROOT}/ablations/linear_probe_bootstrap_random_init"}
ANALYSIS_ROOT=${LINEAR_PROBE_SUBSET_ANALYSIS_DIR:-"${RESULTS_ROOT}/analysis_linear_probe_bootstrap_random_init_subset"}
BASELINE_METRICS=${LINEAR_PROBE_BASELINE_METRICS:-"$ANALYSIS_DIR/aggregated_metrics_all.csv"}

python scripts/ablations/analyze_linear_probe_bootstrap.py \
    --config "$CONFIG" \
    --input-dir "$BOOTSTRAP_ROOT" \
    --output-dir "$ANALYSIS_ROOT" \
    --data-dir "$DATA_PATH" \
    --fundus-config config/fundus_datasets.yaml \
    --baseline-metrics "$BASELINE_METRICS"
