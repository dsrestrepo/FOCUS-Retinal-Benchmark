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

# Install required dependencies for VLMs (including CLIP for EyeCLIP)
pip install gdown timm ftfy regex git+https://github.com/openai/CLIP.git

# Install FLAIR
if [ -d "ext_repos/FLAIR" ]; then
    pip install -e ext_repos/FLAIR
fi

export HF_HOME="$WORK/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"

set -o allexport
source config/paths.env
if [ -f .env ]; then
    source .env
fi
set +o allexport

export HF_HUB_OFFLINE=0

echo "Starting VLM downloads on CPU (compil partition)..."
mkdir -p logs

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$PYTHONPATH:.

python retina_bench/vlms/download.py

echo "VLM Download job completed."
