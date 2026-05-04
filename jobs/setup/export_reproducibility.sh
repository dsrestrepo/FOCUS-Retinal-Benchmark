#!/bin/bash
#SBATCH --job-name=export_repro
#SBATCH --account=dtn@cpu
#SBATCH --partition=compil
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/export_reproducibility.out
#SBATCH --error=logs/export_reproducibility.err
#SBATCH --hint=nomultithread

set -euo pipefail

# Run from the repository root, even when submitted from another directory.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs reproducibility

module purge
module load miniforge/24.9.0
module load gcc/11.4.1
module load cuda/12.4.1
module load cudnn/9.2.0.82-cuda
module load nccl/2.21.5-1-cuda

eval "$(conda shell.bash hook)"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-llms}"
conda activate "${CONDA_ENV_NAME}"

OUT_DIR="${OUT_DIR:-reproducibility}"
mkdir -p "${OUT_DIR}"

REQ_FILE="${OUT_DIR}/requirements.lock.txt"
ENV_FILE="${OUT_DIR}/environment.lock.yml"
EXPLICIT_FILE="${OUT_DIR}/conda-explicit-spec.txt"
RUNTIME_FILE="${OUT_DIR}/python-runtime.json"
REPORT_FILE="${OUT_DIR}/environment-report.txt"

echo "Exporting reproducibility files for conda environment: ${CONDA_ENV_NAME}"

python -m pip freeze --all > "${REQ_FILE}"
repo_root="$(pwd)"
python - "${REQ_FILE}" "${repo_root}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
repo_root = sys.argv[2].rstrip("/")
text = path.read_text()
text = text.replace(repo_root + "/", "")
text = text.replace("-e ext_repos/FLAIR", "-e ./ext_repos/FLAIR")
path.write_text(text)
PY
conda env export --no-builds | sed '/^prefix: /d' > "${ENV_FILE}"
conda list --explicit > "${EXPLICIT_FILE}"

python - <<'PY' > "${RUNTIME_FILE}"
import json
import os
import platform
import sys

payload = {
    "python_version": sys.version,
    "python_implementation": platform.python_implementation(),
    "platform": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
    "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY

{
    echo "FOCUS benchmark reproducibility export"
    echo "Generated at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "Slurm job id: ${SLURM_JOB_ID:-not running under Slurm}"
    echo
    echo "Git commit:"
    git rev-parse HEAD 2>/dev/null || echo "not available"
    echo
    echo "Git status:"
    git status --short 2>/dev/null || echo "not available"
    echo
    echo "Python:"
    python --version
    echo
    echo "Pip:"
    python - <<'PY'
import pip
print(f"pip {pip.__version__}")
PY
    echo
    echo "Conda:"
    conda --version
    echo
    echo "Loaded modules:"
    module list 2>&1
    echo
    echo "CUDA compiler:"
    nvcc --version 2>/dev/null || echo "nvcc not available"
    echo
    echo "GPU inventory:"
    nvidia-smi 2>/dev/null || echo "nvidia-smi not available on this node"
    echo
    echo "Pip dependency check:"
    python -m pip check || true
} > "${REPORT_FILE}"

echo "Wrote:"
echo "  ${REQ_FILE}"
echo "  ${ENV_FILE}"
echo "  ${EXPLICIT_FILE}"
echo "  ${RUNTIME_FILE}"
echo "  ${REPORT_FILE}"
