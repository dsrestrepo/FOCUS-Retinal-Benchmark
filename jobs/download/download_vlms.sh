#!/bin/bash
#SBATCH --job-name=download_vlms
#SBATCH --account=dtn@cpu
#SBATCH --partition=compil
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/download_vlms.out
#SBATCH --error=logs/download_vlms.err

# Use an internet-enabled CPU partition for downloading models.

# Clean modules
module purge
module load miniforge/24.9.0
module load gcc/11.4.1

# Initialize conda for bash
eval "$(conda shell.bash hook)"
conda activate llms

# Install benchmark dependencies, CLIP, and vendored FLAIR during environment
# setup (README.md / install_packages.sh). Downloading must not change the pins.


set -o allexport
source config/paths.env
if [ -f .env ]; then
    source .env
fi
set +o allexport
source jobs/common_paths.sh

export HF_HUB_OFFLINE=0

echo "Starting VLM downloads on CPU (compil partition)..."
mkdir -p logs

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$PYTHONPATH:.

python retina_bench/vlms/download.py

echo "VLM Download job completed."
