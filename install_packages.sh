#!/bin/bash
#SBATCH --job-name=install_dependencies
#SBATCH --account=dtn@cpu
#SBATCH --partition=compil
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=logs/install_dependencies.out
#SBATCH --error=logs/install_dependencies.err
#SBATCH --hint=nomultithread

# Clean modules
module purge

# Load required modules
module load miniforge/24.9.0
module load gcc/11.4.1
module load cuda/12.4.1
module load cudnn/9.2.0.82-cuda
module load nccl/2.21.5-1-cuda

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate llms

# Create logs directory
mkdir -p logs

echo "Starting flash-attn installation..."
echo "Job ID: $SLURM_JOB_ID"
echo "Python: $(which python)"
echo "NVCC: $(nvcc --version)"

# Set environment variables to help with compilation
# Limit parallel jobs to avoid OOM or thrashing on shared nodes
#export MAX_JOBS=4
#export FLASH_ATTENTION_FORCE_BUILD="TRUE"
# Force compatibility with PyTorch's default ABI (0)
#export GLIBCXX_USE_CXX11_ABI=0 

# Install with verbose output to track progress
#pip uninstall -y flash-attn
#pip install flash-attn --no-build-isolation --no-cache-dir -v

#pip install google-genai==1.29.0

echo See dependencied and version in the envieronment:
python --version

# Force uninstall vllm to prevent conflicts
#pip uninstall -y vllm

# Re-install PyTorch with correct CUDA 12.4 dependencies to fix shared library errors
#pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall

# Install correct dependencies from the fixed requirements file
#pip install -r requirements.txt

#pip install --upgrade peft

#pip install kernels triton 

# update requiriments.txt file
#pip freeze > requirements.txt

echo "Installation complete."
