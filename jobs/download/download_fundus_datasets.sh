#!/bin/bash
#SBATCH --job-name=download_fundus
#SBATCH --account=dtn@cpu
#SBATCH --partition=compil
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/download_fundus.out
#SBATCH --error=logs/download_fundus.err

# Use an internet-enabled CPU partition for downloading datasets.

set -euo pipefail

module purge
module load miniforge/24.9.0
module load gcc/11.4.1

eval "$(conda shell.bash hook)"
conda activate llms

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${SLURM_SUBMIT_DIR:-$REPO_DIR}"
mkdir -p logs

set -o allexport
set +u
source config/paths.env
if [ -f .env ]; then
    source .env
fi
set -u
set +o allexport

export HF_HUB_OFFLINE=0
export PYTHONPATH="${PYTHONPATH:-}:."

# Needed for Kaggle mirrors and 224x224 fundus image preparation.
python -m pip install -q kaggle pillow pyyaml

echo "Starting fundus dataset downloads on CPU (compil partition)..."
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "DATA_PATH: ${DATA_PATH}"
echo "=========================================="

python scripts/download_fundus_datasets.py "$@"

echo "=========================================="
echo "Fundus dataset download job completed."
