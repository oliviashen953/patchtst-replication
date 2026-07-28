#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=patchtst_abl
#SBATCH --array=0-12
#SBATCH --output=logs/abl_%A_%a.out
#SBATCH --error=logs/abl_%A_%a.err
#
# Step 10 -- the ablations.
#
#     mkdir -p logs && sbatch scripts/ablation_slurm.sh
#     python experiments/collate_ablation.py
#
# Three questions, 13 runs:
#
#   A. Does channel independence matter?   4 runs, channel-mixing at each H
#   B. Does RevIN matter?                  4 runs, --no-revin at each H
#   C. Does a longer lookback help?        5 runs, L in {96,192,336,512,720} at H=96
#
# C is the important one. The paper's sharpest claim is that PatchTST IMPROVES
# with a longer lookback while other Transformers degrade -- that is the whole
# argument for why patching matters, since patching is what makes a long lookback
# affordable. Reproducing that curve says more than matching any single MSE.

set -euo pipefail

# each entry: TAG|EXTRA_ARGS
CONFIGS=(
  "mix_h96|--pred-len 96 --channel-mixing"
  "mix_h192|--pred-len 192 --channel-mixing"
  "mix_h336|--pred-len 336 --channel-mixing"
  "mix_h720|--pred-len 720 --channel-mixing"
  "norevin_h96|--pred-len 96 --no-revin"
  "norevin_h192|--pred-len 192 --no-revin"
  "norevin_h336|--pred-len 336 --no-revin"
  "norevin_h720|--pred-len 720 --no-revin"
  "look_L96|--pred-len 96 --seq-len 96"
  "look_L192|--pred-len 96 --seq-len 192"
  "look_L336|--pred-len 96 --seq-len 336"
  "look_L512|--pred-len 96 --seq-len 512"
  "look_L720|--pred-len 96 --seq-len 720"
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
"$VENV/bin/python" -u experiments/run_etth1.py \
    $ARGS \
    --device cuda \
    --num-workers 4 \
    --tag "$TAG"

echo "done $TAG"
