#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=patchtst_etth1
#SBATCH --array=0-3
#SBATCH --output=logs/etth1_%A_%a.out
#SBATCH --error=logs/etth1_%A_%a.err
#
# Step 9 -- reproduce the ETTh1 row at all four horizons, one array task each.
#
#     mkdir -p logs
#     sbatch scripts/train_slurm.sh
#
# Watch:   squeue -u $USER
# Collate: python experiments/collate.py
#
# Each task is independent, so all four run concurrently if the queue allows.
# ETTh1 is small (8209 training windows, a 16-dim model), so a single horizon
# is well under the 4h limit -- the wall time is mostly queue wait.

set -euo pipefail

HORIZONS=(96 192 336 720)
PRED_LEN=${HORIZONS[$SLURM_ARRAY_TASK_ID]}

REPO=$HOME/patchtst-replication
VENV=$HOME/venvs/patchtst-env

cd "$REPO"

# -u so prints reach the .out file as they happen rather than at exit.
export PYTHONUNBUFFERED=1
# Keep BLAS single-threaded; we asked for 4 cpus and want them for the loader.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "host=$(hostname)  task=$SLURM_ARRAY_TASK_ID  pred_len=$PRED_LEN"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$VENV/bin/python" -u experiments/run_etth1.py \
    --pred-len "$PRED_LEN" \
    --device cuda \
    --num-workers 4 \
    --tag base

echo "done pred_len=$PRED_LEN"
