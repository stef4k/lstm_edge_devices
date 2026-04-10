#!/bin/bash
#SBATCH --job-name=lstm_har
#SBATCH --chdir=/gpfs/workdir/balazsk/lstm_edge_devices
#SBATCH --output=/gpfs/workdir/balazsk/lstm_edge_devices/slurm/logs/lstm_%j.out
#SBATCH --error=/gpfs/workdir/balazsk/lstm_edge_devices/slurm/logs/lstm_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# ── Activate Python environment ──────────────────────────────────────────────
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null)}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${LSTM_ENV:-sensefi39}"

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
