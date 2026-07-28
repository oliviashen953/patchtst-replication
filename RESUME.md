# Resume notes

State as of **2026-07-28 13:45**. Delete this file before publishing, or keep it
as a working log — it is not part of the tutorial.

## Session

```
Claude Code session: 017oSXR2fK33zTnSjUCwFXMS
https://claude.ai/code/session_017oSXR2fK33zTnSjUCwFXMS
previous:            016hqEA1UFAX87UR4DuaBt1u
```

## Where everything is

```
~/patchtst-replication      this repo, 33 commits, local only (NO remote yet)
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

## Step 12 (self-supervised) — DONE, committed `f9f2c42`

Full write-up is in `RESULTS.md`. Re-run with:

```bash
mkdir -p logs
PRE=$(sbatch --parsable --array=0 scripts/ssl_slurm.sh)
sbatch --dependency=afterok:$PRE --array=1-12 scripts/ssl_slurm.sh
python experiments/collate_ssl.py
```

**The headline.** Selecting checkpoints on validation, pretraining looked like it
won at 3 of 4 horizons. Hold the selection rule fixed across arms and it never
wins — `scratch` is best at every horizon. `scratch` H=336 lost **0.2189 MSE to
selection alone** (validation kept epoch 13; test's best was epoch 2), and that
single error was the entire 0.6405 outlier and the entire "pretraining helps at
336" verdict.

**Next move, NOT yet done: re-run Step 9 with `fit(..., probe_set=test_set)`.**
The +0.0491 H=720 supervised gap has the same suspected cause, and Step 12 has now
measured what that cause is worth. This supersedes the "open lead" note below.

### How it got there (kept as the working log)

**First sweep DONE 2026-07-28 13:12** — `17423134` (smoke), `17423191` (pretrain),
`17423192` (1-12), all 13 COMPLETED, exit 0. `collate_ssl.py` runs clean.
**But the numbers are not yet trustworthy**, so nothing has gone into `RESULTS.md`:

- 6 of 12 runs select their best checkpoint at **epoch 0 or 1** of 20-30, and val
  MSE rises monotonically after.
- Train loss falls steadily in all 16 phases (e.g. scratch h96 0.4462 -> 0.2552),
  so this is overfitting, not under-training. Do **not** raise the LR.
- Val MSE is 0.73-2.13 while test MSE is 0.38-0.64: validation is far harder than
  test, and selection happens on validation.
- `scratch` H=336 (0.6405) is worse than H=720 (0.6084) — non-monotonic in
  horizon. That one outlier alone produces the "pretraining helps at 336" verdict
  (probe-scratch -0.1785, ~10x every other delta).

**Instrumented rerun `17426202` (array 1-12) — all 12 COMPLETED 13:34, and every
one reproduces the first sweep's test MSE exactly (Δ=0.0e+00, 12/12).** `fit()` takes
an optional `probe_set` and logs per-epoch TEST mse into `history.probe_mse` —
diagnostic only, selection still reads validation. Training is bit-identical with
the probe on (verified; the probe loader needs its own RNG generator, because
DataLoader draws from the global stream on every iterator creation even when
`shuffle=False`). Pretraining was **not** repeated — the array reuses
`results_ssl/pretrain_etth1_mask40.pt`, so the transferred weights are unchanged.
The first sweep's JSONs are preserved in `experiments/results_ssl_prerun/`; the
rerun's test MSEs should reproduce them exactly, which is the check that the
probe really is non-invasive.

The question it answers: is the val-optimal epoch far from the test-optimal one?
Each run now prints both, plus how much MSE selecting on val leaves on the table.
Same diagnostic the H=720 supervised gap needs.

Task 0 is pretraining (100 epochs, masked reconstruction); tasks 1-12 are
linear probing / fine-tuning / from-scratch at H = 96, 192, 336, 720. Results
land in `experiments/results_ssl/`; the 13 real JSONs are tracked, the `.pt`
checkpoints and the `smoke_*` JSONs beside them are not. Tasks 1-12 can be re-run
alone as long as `results_ssl/pretrain_etth1_mask40.pt` still exists.

The claim checked: on ETTh1 the paper's own Table 12 shows pretraining winning
only at H=96 and *losing* at 336 and 720, because ETTh1 is small and pretraining
sees no extra data. The `scratch` arm is the like-for-like control the paper
lacks — its `Sup.` column is the L=336 PatchTST/42 row, a different geometry
and model size. Our answer is stronger than the paper's: pretraining wins nowhere.

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

**Open lead — the diagnostic now EXISTS, it just has not been pointed at Step 9.**
Validation is 2-3x harder than test and the gap grows with horizon (1.8x at H=96,
3.0x at H=720). Checkpoints are selected on validation, so H=720 picks epoch 2
using a split that disagrees with test. Step 12 ran exactly this diagnostic on its
own arms and the answer was yes, badly — up to 0.2189 MSE lost to selection at
H=336, and validation stopping at epoch 0 while test improved to epoch 19 at
H=720. Pass `probe_set=test_set` in `experiments/run_etth1.py` and re-run the
Step 9 array; that is the single highest-value job left in this repo.

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

- `collate_ssl.py` had a real bug (smoke runs matched real ones on (stage,
  pred_len)); fixed in `f9f2c42`. Watch for the same pattern in the other
  collate scripts.
- The CPU smoke test path in `tutorial/STEP09_reproduce.md` is slow on a login
  node (~8 min/epoch); use the GPU jobs instead.
