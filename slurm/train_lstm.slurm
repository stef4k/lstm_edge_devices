#!/bin/bash
#SBATCH --job-name=lstm_har
#SBATCH --output=slurm/logs/lstm_%j.out
#SBATCH --error=slurm/logs/lstm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
# Uncomment the line below if a GPU partition is available:
##SBATCH --partition=gpu
##SBATCH --gres=gpu:1

# ── Activate Python environment ──────────────────────────────────────────────
# Adjust CONDA_BASE and ENV_NAME to match your setup.
# Example for a standard conda installation:
#   source ~/miniconda3/etc/profile.d/conda.sh
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null)}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${LSTM_ENV:-sensefi39}"

# ── Navigate to project root ─────────────────────────────────────────────────
# Submit the job from the project root, or set SLURM_PROJECT_DIR explicitly.
PROJECT_DIR="${LSTM_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${PROJECT_DIR}"

# ── Create log directory if it doesn't exist ─────────────────────────────────
mkdir -p slurm/logs

# ── Run ──────────────────────────────────────────────────────────────────────
echo "Job started at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "Python: $(python --version)"
echo "PyTorch / CUDA:"
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())"

python scripts/run_lstm.py \
    --config configs/config.yaml \
    --output_dir results/lstm_run_${SLURM_JOB_ID}

echo "Job finished at: $(date)"
