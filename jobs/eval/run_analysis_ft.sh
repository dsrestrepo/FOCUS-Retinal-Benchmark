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

export PYTHONPATH=$PYTHONPATH:.

echo "Starting fine-tuned model analysis job..."

MODELS=$(python -c 'import yaml; print(" ".join(yaml.safe_load(open("config/mllm_config.yaml"))["models"]))')

python scripts/analyze_ft_benchmark.py \
    --sft_dir results/evals_sft \
    --grpo_dir results/evals_grpo \
    --data_dir "$DATA_PATH" \
    --fundus_config config/fundus_datasets.yaml \
    --baseline_metrics results/analysis/aggregated_metrics.csv \
    --output_dir results/analysis_ft \
    --models $MODELS

echo "Estimating paired FT-vs-base uncertainty..."

python scripts/analyze_ft_significance.py \
    --ft_metrics results/analysis_ft/aggregated_metrics.csv \
    --base_dir results/evals \
    --output_dir results/analysis_ft/significance \
    --paper_table_dir paper/tables \
    --metrics auc accuracy ece \
    --n_boot "${FT_SIGNIFICANCE_BOOTSTRAPS:-1000}"

echo "Job completed."
