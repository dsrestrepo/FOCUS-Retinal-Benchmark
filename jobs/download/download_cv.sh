#!/bin/bash
#SBATCH --job-name=download_cv
#SBATCH --account=dtn@cpu
#SBATCH --partition=compil
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/download_cv.out
#SBATCH --error=logs/download_cv.err

# Use an internet-enabled CPU partition for downloading models.

module purge
module load miniforge/24.9.0
module load gcc/11.4.1

eval "$(conda shell.bash hook)"
conda activate llms

export HF_HOME="$WORK/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"

set -o allexport
source config/paths.env
if [ -f .env ]; then
    source .env
fi
set +o allexport

export HF_HUB_OFFLINE=0
export PYTHONPATH=$PYTHONPATH:.

# Example: Run entirely or specify models:
# python retina_bench/cv/download.py --models RETFound EyeFM
python retina_bench/cv/download.py "$@"
