#!/bin/bash
#SBATCH --job-name=download_models
#SBATCH --account=dtn@cpu
#SBATCH --partition=compil
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/download_models.out
#SBATCH --error=logs/download_models.err

# Use an internet-enabled CPU partition for downloading models.

# Clean modules
module purge

# Load basic modules (adjust version if needed, matching cpu_example.sh)
module load miniforge/24.9.0
module load gcc/11.4.1

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Activate conda environment
conda activate llms

# Use $WORK for model cache to avoid quota issues in $HOME
export HF_HOME="$WORK/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"

# Authentication
# Ensure HF_TOKEN is loaded from .env config
set -o allexport
source config/paths.env
if [ -f .env ]; then
    source .env
fi
set +o allexport

# IMPORTANT: Ensure offline mode is DISABLED for downloading
export HF_HUB_OFFLINE=0

echo "Starting model downloads on CPU (compil partition)..."
echo "Job ID: $SLURM_JOB_ID"
echo "HF_HOME: $HF_HOME"
echo "=========================================="

# Ensure logs directory exists
mkdir -p logs

set -e  # Exit on error

# Verify environment variables
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN environment variable is not set. Assuming it is loaded from .env file by the python script."
fi

# Run the download script
# Move to the directory where the job was submitted (repository root)
cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$PYTHONPATH:.

python retina_bench/mllms/download.py

echo "=========================================="
echo "Download job completed."
