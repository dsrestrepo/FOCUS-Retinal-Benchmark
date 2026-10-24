#!/bin/bash
#SBATCH --job-name=analyze_sft_results
#SBATCH --account=dtn@cpu
#SBATCH --partition=compil
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/analyze_sft_results.out
#SBATCH --error=logs/analyze_sft_results.err

module purge
module load miniforge/24.9.0

eval "$(conda shell.bash hook)"
conda activate llms

set -o allexport
source config/paths.env
set +o allexport
source jobs/common_paths.sh

export PYTHONPATH=$PYTHONPATH:.

echo "Starting SFT-only analysis job..."

MODELS=$(python -c 'import yaml; print(" ".join(yaml.safe_load(open("config/mllm_config.yaml"))["models"]))')

python scripts/analyze_ft_benchmark.py \
    --sft_dir "${SFT_EVAL_DIR}" \
    --grpo_dir "${RESULTS_ROOT}/no_grpo" \
    --data_dir "$DATA_PATH" \
    --fundus_config config/fundus_datasets.yaml \
    --baseline_metrics "${ANALYSIS_DIR}/aggregated_metrics.csv" \
    --output_dir "${SFT_ANALYSIS_DIR}" \
    --models $MODELS

echo "Estimating paired SFT-vs-base uncertainty..."

if [ -f "${SFT_ANALYSIS_DIR}/aggregated_metrics.csv" ]; then
    python scripts/analyze_ft_significance.py \
        --ft_metrics "${SFT_ANALYSIS_DIR}/aggregated_metrics.csv" \
        --base_dir "${OUTPUT_DIR}" \
        --output_dir "${SFT_ANALYSIS_DIR}/significance" \
        --paper_table_dir paper/tables/sft \
        --metrics auc accuracy ece \
        --n_boot "${FT_SIGNIFICANCE_BOOTSTRAPS:-1000}"
else
    echo "No SFT aggregate metrics were written; skipping significance."
fi

echo "SFT-only analysis completed."
