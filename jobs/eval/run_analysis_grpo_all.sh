#!/bin/bash
#SBATCH --job-name=analyze_grpo_all
#SBATCH --account=dtn@cpu
#SBATCH --partition=compil
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/analyze_grpo_all.out
#SBATCH --error=logs/analyze_grpo_all.err

module purge
module load miniforge/24.9.0

eval "$(conda shell.bash hook)"
conda activate llms

set -o allexport
source config/paths.env
set +o allexport
source jobs/common_paths.sh

export PYTHONPATH=$PYTHONPATH:.

echo "Starting all-task GRPO analysis job..."

MODELS=$(python -c 'import yaml; print(" ".join(yaml.safe_load(open("config/mllm_config.yaml"))["models"]))')

python scripts/analyze_ft_benchmark.py \
    --sft_dir "${RESULTS_ROOT}/no_sft" \
    --grpo_dir "${GRPO_ALL_EVAL_DIR}" \
    --data_dir "$DATA_PATH" \
    --fundus_config config/fundus_datasets.yaml \
    --baseline_metrics "${ANALYSIS_DIR}/aggregated_metrics.csv" \
    --output_dir "${GRPO_ALL_ANALYSIS_DIR}" \
    --models $MODELS

echo "Estimating paired all-task GRPO-vs-base uncertainty..."

if [ -f "${GRPO_ALL_ANALYSIS_DIR}/aggregated_metrics.csv" ]; then
    python scripts/analyze_ft_significance.py \
        --ft_metrics "${GRPO_ALL_ANALYSIS_DIR}/aggregated_metrics.csv" \
        --base_dir "${OUTPUT_DIR}" \
        --output_dir "${GRPO_ALL_ANALYSIS_DIR}/significance" \
        --paper_table_dir paper/tables/grpo_all \
        --metrics auc accuracy ece \
        --n_boot "${FT_SIGNIFICANCE_BOOTSTRAPS:-1000}"
else
    echo "No all-task GRPO aggregate metrics were written; skipping significance."
fi

echo "All-task GRPO analysis completed."
