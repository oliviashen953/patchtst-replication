#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=patchtst_cgm
#SBATCH --array=0-3
#SBATCH --output=logs/cgm_%A_%a.out
#SBATCH --error=logs/cgm_%A_%a.err
#
# Step 11 -- the 2x2: channel independence vs mixing, with and without
# informative meal/bolus drivers.
#
#     mkdir -p logs && sbatch scripts/cgm_slurm.sh
#     python experiments/collate_cgm.py

set -euo pipefail

CONFIGS=(
  "ci_drivers|"
  "mix_drivers|--channel-mixing"
  "ci_control|--zero-drivers"
  "mix_control|--channel-mixing --zero-drivers"
)

ENTRY=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
TAG=${ENTRY%%|*}
ARGS=${ENTRY#*|}

REPO=$HOME/patchtst-replication
VENV=$HOME/venvs/patchtst-env
cd "$REPO"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "host=$(hostname)  task=$SLURM_ARRAY_TASK_ID  tag=$TAG  args=$ARGS"

# shellcheck disable=SC2086
"$VENV/bin/python" -u experiments/run_cgm.py $ARGS --device cuda --tag "$TAG"

echo "done $TAG"
