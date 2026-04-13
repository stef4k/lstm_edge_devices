#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# no_empty.slurm
#
# Train the LSTM on the 4-class setup (walk / stand / sit / lying) without the
# empty class, using the w100/s50 window configuration that achieved the highest
# overall accuracy (≈ 59.5 %) in the grid search (grid_w100_s50_job536784).
#
# This mirrors the original paper's class set, which does not include an
# empty-room category.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=lstm_no_empty
#SBATCH --chdir=/gpfs/workdir/balazsk/lstm_edge_devices
#SBATCH --output=/gpfs/workdir/balazsk/lstm_edge_devices/slurm/logs/no_empty_%j.out
#SBATCH --error=/gpfs/workdir/balazsk/lstm_edge_devices/slurm/logs/no_empty_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

OUTPUT_DIR="results/no_empty_w100_s50_job${SLURM_JOB_ID}"

# ── Activate environment ─────────────────────────────────────────────────────
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null)}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${LSTM_ENV:-sensefi39}"

# ── Info ─────────────────────────────────────────────────────────────────────
echo "========================================================"
echo "Job started at : $(date)"
echo "Node           : $SLURMD_NODENAME"
echo "Job ID         : $SLURM_JOB_ID"
echo "Config         : configs/config_no_empty.yaml"
echo "Output dir     : $OUTPUT_DIR"
echo "Python         : $(python --version)"
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
echo "========================================================"

# ── Run ──────────────────────────────────────────────────────────────────────
python scripts/run_lstm.py \
    --config     configs/config_no_empty.yaml \
    --output_dir "$OUTPUT_DIR"

echo "========================================================"
echo "Job finished at: $(date)"
echo "========================================================"
