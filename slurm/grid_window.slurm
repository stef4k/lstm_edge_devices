#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# grid_window.slurm
#
# SLURM array job: sweep window / stride sizes to diagnose the accuracy gap
# vs. the paper (paper: window=20 frames, stride=10).
#
# Candidate configurations
# ┌──────┬───────┬──────────────────────────────────────────────────────────┐
# │ Task │ Win   │ Notes                                                    │
# ├──────┼───────┼──────────────────────────────────────────────────────────┤
# │  0   │  20   │ Paper-equivalent frame count (0.2 s at 100 Hz)          │
# │  1   │  50   │ 0.5 s — intermediate, many windows                      │
# │  2   │ 100   │ 1 s — our original notebook value, ~4k windows          │
# └──────┴───────┴──────────────────────────────────────────────────────────┘
#
# Data-leakage guarantee (unchanged from the baseline run):
#   build_windows() segments every recording with the SAME window/stride.
#   make_cv_folds() then assigns ALL windows that originated from a given
#   experiment exclusively to train OR test — never both.  No window index
#   crosses the fold boundary regardless of how windows overlap within a
#   single recording.
#
# Experiment split is identical to the baseline (config.yaml unchanged):
#   test activity exp = {exp1, exp2, exp3, exp4, exp8} (one per fold)
#   test empty  exp   = {exp5_empty or exp6_empty} (22 sampled windows)
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=lstm_grid_window
#SBATCH --chdir=/gpfs/workdir/balazsk/lstm_edge_devices
#SBATCH --output=/gpfs/workdir/balazsk/lstm_edge_devices/slurm/logs/grid_%A_%a.out
#SBATCH --error=/gpfs/workdir/balazsk/lstm_edge_devices/slurm/logs/grid_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-2

# ── Grid definition ──────────────────────────────────────────────────────────
# Index  0     1     2
WINDOWS=(20    50   100)
STRIDES=(10    25    50)

WINDOW=${WINDOWS[$SLURM_ARRAY_TASK_ID]}
STRIDE=${STRIDES[$SLURM_ARRAY_TASK_ID]}

OUTPUT_DIR="results/grid_w${WINDOW}_s${STRIDE}_job${SLURM_ARRAY_JOB_ID}"

# ── Activate environment ─────────────────────────────────────────────────────
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null)}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${LSTM_ENV:-sensefi39}"

# ── Info ─────────────────────────────────────────────────────────────────────
echo "========================================================"
echo "Job started at : $(date)"
echo "Node           : $SLURMD_NODENAME"
echo "Array job ID   : $SLURM_ARRAY_JOB_ID  task: $SLURM_ARRAY_TASK_ID"
echo "Window         : $WINDOW samples  Stride: $STRIDE samples"
echo "Output dir     : $OUTPUT_DIR"
echo "Python         : $(python --version)"
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
echo "========================================================"

# ── Run ──────────────────────────────────────────────────────────────────────
python scripts/run_lstm.py \
    --config    configs/config.yaml \
    --window    "$WINDOW" \
    --stride    "$STRIDE" \
    --output_dir "$OUTPUT_DIR"

echo "========================================================"
echo "Job finished at: $(date)"
echo "========================================================"
