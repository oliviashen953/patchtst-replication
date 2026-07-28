#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --job-name=patchtst_ssl
#SBATCH --output=logs/ssl_%A_%a.out
#SBATCH --error=logs/ssl_%A_%a.err
#
# Step 12 -- masked self-supervised pretraining, then three downstream arms.
#
# Two submissions, because everything downstream needs the same pretrained
# checkpoint and must wait for it:
#
#     mkdir -p logs
#     PRE=$(sbatch --parsable --array=0 scripts/ssl_slurm.sh)
#     sbatch --dependency=afterok:$PRE --array=1-12 scripts/ssl_slurm.sh
#     python experiments/collate_ssl.py
#
# Task 0 is the pretraining run; tasks 1-12 are linear probing, fine-tuning,
# and the from-scratch control at each of the four horizons.

set -euo pipefail

CONFIGS=(
  "--stage pretrain --epochs 100"
  "--stage linear_probe --pred-len 96"
  "--stage linear_probe --pred-len 192"
  "--stage linear_probe --pred-len 336"
  "--stage linear_probe --pred-len 720"
  "--stage finetune --pred-len 96"
  "--stage finetune --pred-len 192"
  "--stage finetune --pred-len 336"
  "--stage finetune --pred-len 720"
  "--stage scratch --pred-len 96"
  "--stage scratch --pred-len 192"
  "--stage scratch --pred-len 336"
  "--stage scratch --pred-len 720"
)

ARGS=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

REPO=$HOME/patchtst-replication
VENV=$HOME/venvs/patchtst-env
cd "$REPO"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "host=$(hostname)  task=$SLURM_ARRAY_TASK_ID  args=$ARGS"

# shellcheck disable=SC2086
"$VENV/bin/python" -u experiments/run_ssl.py $ARGS \
    --device cuda \
    --num-workers 4

echo "done task $SLURM_ARRAY_TASK_ID"
