# Resume notes

State as of **2026-07-28 12:40**. Delete this file before publishing, or keep it
as a working log — it is not part of the tutorial.

## Session

```
Claude Code session: 017oSXR2fK33zTnSjUCwFXMS
https://claude.ai/code/session_017oSXR2fK33zTnSjUCwFXMS
previous:            016hqEA1UFAX87UR4DuaBt1u
```

## Where everything is

```
~/patchtst-replication      this repo, 18 commits, local only (NO remote yet)
~/PatchTST_upstream         official code, reference/answer-key only
~/venvs/patchtst-env        python3.11 + torch 2.13.0+cu130
~/Downloads/patchtst_2211.14730.pdf
```

Activate with:

```bash
cd ~/patchtst-replication && source ~/venvs/patchtst-env/bin/activate
```

## Status: steps 1-12 all written; step 10 and 11 results are in

All nine checks pass:

```bash
for s in 01 02 03 04 05 06 07 08 12; do python tests/check_step$s.py | tail -1; done
```

## Step 12 (self-supervised) — what is left

The code, the tests and `tutorial/STEP12_pretrain.md` are committed. What is
NOT yet in `RESULTS.md` is the 13-run result table, because the runs are still
queued. Submit:

```bash
mkdir -p logs
PRE=$(sbatch --parsable --array=0 scripts/ssl_slurm.sh)
sbatch --dependency=afterok:$PRE --array=1-12 scripts/ssl_slurm.sh
python experiments/collate_ssl.py
```

Task 0 is pretraining (100 epochs, masked reconstruction); tasks 1-12 are
linear probing / fine-tuning / from-scratch at H = 96, 192, 336, 720. Results
land in `experiments/results_ssl/`; the JSONs are tracked, the `.pt`
checkpoints beside them are not.

The claim to check: on ETTh1 the paper's own Table 12 shows pretraining winning
only at H=96 and *losing* at 336 and 720, because ETTh1 is small and pretraining
sees no extra data. The `scratch` arm is the like-for-like control the paper
lacks — its `Sup.` column is the L=336 PatchTST/42 row, a different geometry
and model size.

## Older jobs (done)

| job | what | collate with |
|---|---|---|
| `17403656` | Step 10 ablations, 13 runs | `python experiments/collate_ablation.py` |
| `17403688` | Step 11 CGM 2x2, 4 runs | `python experiments/collate_cgm.py` |

## Results already in hand (Step 9, ETTh1)

Three of four horizons reproduce. Full detail in `RESULTS.md`.

| H | ours | paper (PatchTST/42) | Δ |
|---:|---:|---:|---:|
| 96 | 0.3720 | 0.375 | −0.0030 |
| 192 | 0.4110 | 0.414 | −0.0030 |
| 336 | 0.4391 | 0.431 | +0.0081 |
| 720 | 0.4981 | 0.449 | +0.0491 |

**H=720 is unresolved.** Two hypotheses tested and both failed:

1. `lradj type3` — rejected. Bit-identical result, because both schedules are
   identical through epoch 3 and the minimum is at epoch 2. Also worse at every
   other horizon, so cosine stays.
2. `weight_decay=0` (matching upstream's plain Adam) — rejected. Worse at every
   horizon.

Three different optimizer/schedule configs land within 0.0013 at H=720, so the
gap is structural, not tuning.

**Open lead, not yet run.** Validation is 2-3x harder than test and the gap grows
with horizon (1.8x at H=96, 3.0x at H=720). Checkpoints are selected on
validation, so H=720 picks epoch 2 using a split that disagrees with test. The
diagnostic: log test MSE per epoch (for inspection only, never for selection) and
see whether the test-optimal epoch is far from the val-optimal one. If so, the
problem is checkpoint *selection*, not training.

## Everything is n=1

No seeds, no error bars anywhere. Differences under ~0.01 MSE are not separable
from seed noise. Do not claim the H=96/192 "wins" over the paper without running
several seeds first.

## To publish

The repo has no remote. Two steps:

1. **You:** create an empty repo via the GitHub web UI (no README, no license, no
   gitignore — this repo has all three). GitHub API auth is not available from
   the HPC, only SSH push.
2. **Then:**

```bash
cd ~/patchtst-replication
git remote add origin git@github.com:oliviashen953/patchtst-replication.git
git push -u origin main
```

Before publishing, consider deleting this file.

## Known loose ends

- `collate_ssl.py` has not been run against a full 13-run sweep yet.
- The CPU smoke test path in `tutorial/STEP09_reproduce.md` is slow on a login
  node (~8 min/epoch); use the GPU jobs instead.
