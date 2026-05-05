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

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

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

echo "Installing FOCUS benchmark GPU dependencies..."
echo "Job ID: $SLURM_JOB_ID"
echo "Python: $(which python)"
echo "NVCC: $(nvcc --version)"
echo "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"

python --version
python -m pip --version

echo "Installing PyTorch CUDA 12.4 stack used in the benchmark experiments..."
python -m pip install \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

echo "Installing non-GPU-pinned Python dependencies..."
python -m pip install -r requirements.txt

echo "Installing flash-attn after PyTorch is available..."
export MAX_JOBS="${MAX_JOBS:-4}"
python -m pip install flash-attn==2.8.3 --no-build-isolation

echo "Installing GPU packages tied to the PyTorch stack..."
python -m pip install \
    xformers==0.0.29.post3 \
    triton==3.2.0 \
    torchao==0.16.0

echo "Installed GPU stack:"
python - <<'PY'
import importlib
import torch

print(f"torch={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
for name in ("torchvision", "torchaudio", "flash_attn", "xformers", "triton"):
    try:
        mod = importlib.import_module(name)
        print(f"{name}={getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{name}=IMPORT_ERROR: {exc}")
PY

python -m pip check || true

echo "Installation complete."
