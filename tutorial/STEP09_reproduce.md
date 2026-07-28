# Step 9 — Reproduce the ETTh1 benchmark

This step is different from 1–8. There are no `TODO(you)` gaps: everything is
written. What Step 9 asks of you is **judgment** — run it, read the numbers, and
work out whether a gap is your bug or a documented deviation.

## Which column you're actually competing against

Our config is `L=336`, `P=16`, `S=8` → **42 patches**. So the comparison is
**PatchTST/42**, not PatchTST/64 (which uses `L=512`).

Paper Table 3, ETTh1, PatchTST/42:

| H | MSE | MAE |
|---:|---:|---:|
| 96 | 0.375 | 0.399 |
| 192 | 0.414 | 0.421 |
| 336 | 0.431 | 0.436 |
| 720 | 0.449 | 0.466 |

Comparing against the wrong column is a fast route to concluding your
implementation is broken when it isn't. PatchTST/64 is better at every horizon,
because it sees a longer history.

## The detail that decides whether this works

**ETTh1 does not use the paper's default model size.** Appendix A.1.4:

> For very small datasets (ILI, ETTh1, ETTh2), a reduced size of parameters is
> used (H = 4, D = 16 and F = 128) to mitigate the possible overfitting.

So `n_heads=4`, `d_model=16`, `d_ff=128` — against defaults of 16 / 128 / 256.
ETTh1 has only 8,209 training windows; the default-size model overfits it badly.

This is the single most likely reason a from-scratch attempt misses the target,
and it's one sentence buried in an appendix. `experiments/run_etth1.py` already
encodes it in `ETTH1_MODEL`.

## Deviations I chose, recorded honestly

Three places we differ from the released script. Written down rather than hidden,
because they may cost a little accuracy and you should be able to explain a gap:

1. **LR schedule.** The official ETTh1 script uses `lradj type3`; we use cosine
   annealing.
2. **Dropout.** The paper *text* says 0.2 everywhere, but the released ETTh1
   script uses **0.3**. We follow the script, since that's what produced the
   published numbers. (A real paper/code discrepancy — worth noting in your
   writeup.)
3. **Early stopping.** Their default patience is 100 over 100 epochs, so it never
   fires; they train the full 100 and keep the best-validation checkpoint. We do
   the same.

## Running it

Smoke-test on CPU first, a few epochs, to confirm the plumbing:

```bash
python experiments/run_etth1.py --pred-len 96 --epochs 2 --device cpu --tag smoke
```

Then the real thing — **this wants a GPU, not the login node**:

```bash
mkdir -p logs
sbatch scripts/train_slurm.sh      # array 0-3, one horizon per task
squeue -u $USER
```

Four independent tasks, so they run concurrently if the queue allows. ETTh1 is
small, so each is well under the 4-hour limit; wall-clock is mostly queue wait.

Then:

```bash
python experiments/collate.py --tag base
```

`--tag` is not optional once Step 10 has run: the ablations write into the same
`experiments/results/`, and without a tag the script would average them into the
baseline. It refuses rather than blend, and lists the tags it found.

## Reading the result

`collate.py` prints your MSE/MAE beside the paper's with a delta, and flags
anything past 0.02.

- **within 0.02** — a faithful replication. That's the outcome to want.
- **0.02–0.05** — close; most likely the LR-schedule deviation above.
- **beyond 0.05** — something is wrong. Check in this order:
  1. **model size** — `d_model=16`, not 128
  2. comparing against /42, not /64
  3. **double normalization** — targets are already standardized *and* the model
     denormalizes; if you added another, MSE will be wildly off
  4. the split — 8209/2785/2785 windows at H=96

## What to expect emotionally

Your first run will probably not match. That is completely normal and it is the
most instructive part of any replication — the debugging is where you learn what
the paper actually did, as opposed to what it says.

You have an advantage most replications don't: eight independently-tested
components. If the number is off, the bug is almost certainly in the *protocol*
(schedule, split, normalization, model size), not in the architecture — because
the architecture is already verified piece by piece.
