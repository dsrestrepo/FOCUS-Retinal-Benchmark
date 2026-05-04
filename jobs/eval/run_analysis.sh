#!/bin/bash
#SBATCH --job-name=analyze_results
#SBATCH --account=dtn@cpu
#SBATCH --partition=compil
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/analyze_results.out
#SBATCH --error=logs/analyze_results.err

# Clean modules
module purge

# Load required modules
module load miniforge/24.9.0

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate llms

# Centralized Path variables
set -o allexport
source config/paths.env
set +o allexport

export PYTHONPATH=$PYTHONPATH:.

echo "Starting Results Analysis Job..."

# Run the analysis script
python scripts/analyze_benchmark.py \
    --config config/analysis_config.yaml \
    --results_dir "$OUTPUT_DIR" \
    --data_dir "$DATA_PATH" \
    --fundus_config config/fundus_datasets.yaml \
    --output_dir results/analysis

echo "Job completed."
