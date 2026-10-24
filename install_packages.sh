#!/bin/bash
#SBATCH --job-name=install_dependencies
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

CONDA_ENV_NAME="${CONDA_ENV_NAME:-llms}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12.12}"
export MAX_JOBS="${MAX_JOBS:-4}"

if [[ "${SKIP_MODULES:-0}" != "1" ]] && type module >/dev/null 2>&1; then
    module purge
    module load "${MINIFORGE_MODULE:-miniforge/24.9.0}"
    module load "${GCC_MODULE:-gcc/11.4.1}"
    module load "${CUDA_MODULE:-cuda/12.8.1}"
    # PyTorch wheels supply their matching cuDNN and NCCL runtime libraries.
fi

eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
    conda create -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${CONDA_ENV_NAME}"

echo "Installing FOCUS benchmark GPU dependencies..."
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Python: $(which python)"
echo "NVCC: $(nvcc --version 2>/dev/null || echo unavailable)"
echo "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"

python --version
python -m pip --version

echo "Installing the recorded Torch 2.10 / TorchVision 0.25 stack and benchmark dependencies..."
python -m pip install -r reproducibility/requirements.lock.txt

echo "Installing vendored FLAIR without changing the resolved dependency versions..."
python -m pip install --no-deps -e ./ext_repos/FLAIR

echo "Installing flash-attn after PyTorch and build prerequisites are available..."
python -m pip install flash-attn==2.8.3 --no-build-isolation

echo "Installed GPU stack:"
python - <<'PY'
import importlib
import torch

print(f"torch={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
for name in ("torchvision", "flash_attn", "triton"):
    mod = importlib.import_module(name)
    print(f"{name}={getattr(mod, '__version__', 'unknown')}")
PY

python -m pip check

echo "Installation complete."
