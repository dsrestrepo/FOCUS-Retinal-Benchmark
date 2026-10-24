#!/bin/bash
#SBATCH --job-name=analyze_ft_results
#SBATCH --account=dtn@cpu
#SBATCH --partition=compil
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/analyze_ft_results.out
#SBATCH --error=logs/analyze_ft_results.err

module purge
module load miniforge/24.9.0

eval "$(conda shell.bash hook)"
conda activate llms

set -o allexport
source config/paths.env
set +o allexport
source jobs/common_paths.sh

export PYTHONPATH=$PYTHONPATH:.

echo "Starting fine-tuned model analysis job..."

MODELS=$(python -c 'import yaml; print(" ".join(yaml.safe_load(open("config/mllm_config.yaml"))["models"]))')

python scripts/analyze_ft_benchmark.py \
    --sft_dir "${SFT_EVAL_DIR}" \
    --grpo_dir "${GRPO_EVAL_DIR}" \
    --data_dir "$DATA_PATH" \
    --fundus_config config/fundus_datasets.yaml \
    --baseline_metrics "${ANALYSIS_DIR}/aggregated_metrics.csv" \
    --output_dir "${FT_ANALYSIS_DIR}" \
    --models $MODELS

echo "Estimating paired FT-vs-base uncertainty..."

python scripts/analyze_ft_significance.py \
    --ft_metrics "${FT_ANALYSIS_DIR}/aggregated_metrics.csv" \
    --base_dir "${OUTPUT_DIR}" \
    --output_dir "${FT_ANALYSIS_DIR}/significance" \
    --paper_table_dir paper/tables \
    --metrics auc accuracy ece \
    --n_boot "${FT_SIGNIFICANCE_BOOTSTRAPS:-1000}"

echo "Job completed."
