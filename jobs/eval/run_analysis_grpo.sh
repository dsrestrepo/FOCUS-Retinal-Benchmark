#!/bin/bash
#SBATCH --job-name=analyze_grpo_results
#SBATCH --account=dtn@cpu
#SBATCH --partition=compil
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/analyze_grpo_results.out
#SBATCH --error=logs/analyze_grpo_results.err

module purge
module load miniforge/24.9.0

eval "$(conda shell.bash hook)"
conda activate llms

set -o allexport
source config/paths.env
set +o allexport

export PYTHONPATH=$PYTHONPATH:.

echo "Starting GRPO-only analysis job..."

MODELS=$(python -c 'import yaml; print(" ".join(yaml.safe_load(open("config/mllm_config.yaml"))["models"]))')

python scripts/analyze_ft_benchmark.py \
    --sft_dir results/no_sft \
    --grpo_dir results/evals_grpo \
    --data_dir "$DATA_PATH" \
    --fundus_config config/fundus_datasets.yaml \
    --baseline_metrics results/analysis/aggregated_metrics.csv \
    --output_dir results/analysis_grpo \
    --models $MODELS

echo "Estimating paired GRPO-vs-base uncertainty..."

if [ -f results/analysis_grpo/aggregated_metrics.csv ]; then
    python scripts/analyze_ft_significance.py \
        --ft_metrics results/analysis_grpo/aggregated_metrics.csv \
        --base_dir results/evals \
        --output_dir results/analysis_grpo/significance \
        --paper_table_dir paper/tables/grpo \
        --metrics auc accuracy ece \
        --n_boot "${FT_SIGNIFICANCE_BOOTSTRAPS:-1000}"
else
    echo "No GRPO aggregate metrics were written; skipping significance."
fi

echo "GRPO-only analysis completed."
