#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=patchtst_sweep
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --array=0-97
#
# Step 13 -- the seven ETTh1 sweeps, one array task per run.
#
#     mkdir -p logs
#     sbatch scripts/sweep_slurm.sh                    # all 98
#     sbatch --array=0-7 scripts/sweep_slurm.sh        # just Table 7
#
# The array index maps to a job through scripts/sweep_jobs.py, so the mapping
# is one place and `python scripts/sweep_jobs.py` prints it for a human.
#
# Watch:   squeue -u $USER
# Collate: python experiments/collate_sweep.py
#
# Memory is 32G rather than the 16G of train_slurm.sh because the no-patching
# cells of Table 7 build a 336-token (channel-independent) or 2352-token
# (channel-mixing) attention matrix. The paper reports '-' for cells that ran
# out of memory on a 48GB A40 even at batch size 1, so a CUDA OOM in that group
# reproduces their result rather than breaking ours -- collate_sweep.py reports
# those cells as OOM instead of dropping them.

set -euo pipefail

REPO=$HOME/patchtst-replication
VENV=$HOME/venvs/patchtst-env

cd "$REPO"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

JOB_ARGS=$("$VENV/bin/python" scripts/sweep_jobs.py --args "$SLURM_ARRAY_TASK_ID")

echo "host=$(hostname)  task=$SLURM_ARRAY_TASK_ID"
echo "args: $JOB_ARGS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Word splitting on $JOB_ARGS is what we want here: it is a flag string we
# generated ourselves, one line, no user input and no spaces inside a value.
# shellcheck disable=SC2086
"$VENV/bin/python" -u experiments/run_etth1.py $JOB_ARGS --device cuda
