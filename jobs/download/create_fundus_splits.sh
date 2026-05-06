#!/bin/bash
#SBATCH --job-name=fundus_splits
#SBATCH --account=dtn@cpu
#SBATCH --partition=compil
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/create_fundus_splits.out
#SBATCH --error=logs/create_fundus_splits.err

set -euo pipefail

module purge
module load miniforge/24.9.0
module load gcc/11.4.1

eval "$(conda shell.bash hook)"
conda activate llms

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${SLURM_SUBMIT_DIR:-$REPO_DIR}"
mkdir -p logs Split_Data

set -o allexport
set +u
source config/paths.env
if [ -f .env ]; then
    source .env
fi
set -u
set +o allexport

export PYTHONPATH="${PYTHONPATH:-}:."

echo "Creating reproducible fundus train/val/test split files..."
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "DATA_PATH: ${DATA_PATH}"
echo "=========================================="

python scripts/create_dataset_splits.py "$@"

echo "=========================================="
echo "Fundus split creation completed."
