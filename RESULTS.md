# Results

## ETTh1, multivariate, lookback L=336 (PatchTST/42)

First reproduction run. SLURM array on one A40 per horizon, ~2 minutes each.
Commit `1ae39ed`, seed 2021, single seed per horizon.

| H | ours MSE | paper MSE | Δ | ours MAE | paper MAE | Δ | best epoch |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | **0.3720** | 0.375 | **−0.0030** | **0.3942** | 0.399 | **−0.0048** | 51 |
| 192 | **0.4110** | 0.414 | **−0.0030** | **0.4165** | 0.421 | **−0.0045** | 19 |
| 336 | 0.4391 | 0.431 | +0.0081 | 0.4358 | 0.436 | −0.0002 | 10 |
| 720 | 0.4981 | 0.449 | +0.0491 | 0.4987 | 0.466 | +0.0327 | 2 |
| | | | **+0.0128** | | | **+0.0058** | |

Reference values are PatchTST/42 from Table 3 of the paper. **/42, not /64** —
our L=336 with P=16, S=8 gives 42 patches; PatchTST/64 uses L=512 and is stronger
at every horizon, so comparing against it would be wrong.

Three of four horizons reproduce within 0.01 MSE, and 96/192 come in slightly
*below* the published values. H=720 is off by +0.049.

## Why H=720 misses

Not a generic "hyperparameter" shrug — the validation curves say exactly what
happens. Training loss falls monotonically at every horizon, while validation MSE
turns upward earlier and earlier:

| H | val start | val min | val @ep99 | best epoch |
|---:|---:|---:|---:|---:|
| 96 | 0.978 | 0.677 | 0.677 | 51 |
| 192 | 1.153 | 0.931 (ep 20) | 1.022 | 19 |
| 336 | 1.325 | 1.182 (ep 10) | 1.382 | 10 |
| 720 | 1.573 | 1.499 (ep 2) | **2.031** | 2 |

That is overfitting, and it scales with the horizon for a structural reason: the
flatten head is `n_patches × d_model → pred_len`, so its parameters grow linearly
with the horizon while the number of training windows *shrinks* (a longer horizon
consumes more of each split).

| H | head params | train windows | ratio |
|---:|---:|---:|---:|
| 96 | 81,742 | 8209 | ~10 |
| 720 | 501,694 | 7585 | ~66 |

## Two hypotheses tested, both falsified

Recorded because negative results are results, and because the second one only
looked plausible until it was run.

### Hypothesis 1 — the learning-rate schedule. REJECTED.

The official ETTh1 script uses `lradj type3` (constant for 3 epochs, then ×0.9
per epoch → ~0.18× by epoch 20, ~0.013× by epoch 50). Cosine over `T_max=100` is
nearly flat early (~0.91× at epoch 20). The prediction was that type3's decay
would act as implicit regularization and rescue H=720.

It did not. H=720 came back **bit-identical**: best epoch 2, best val 1.4990,
test 0.4981. The two schedules are identical through epoch 3, and the minimum is
at epoch 2, so the schedule never gets a chance to act. type3 was also *worse* at
every other horizon (96: 0.3801, 192: 0.4137, 336: 0.4422).

The mechanism itself is real — type3 pulls H=720's val at epoch 99 from 2.0109
down to 1.7032 — it simply does not matter, because best-checkpointing already
banked epoch 2.

**Refined finding: H=720 overfits within two epochs.** No learning-rate schedule
can address something that fast.

### Hypothesis 2 — weight decay. REJECTED.

Upstream `exp_main.py` uses plain `optim.Adam(params, lr=...)` with no weight
decay; this repo defaults to `1e-3`. A real unintended deviation, so worth
matching regardless of predicted direction.

Setting `weight_decay=0` was **worse at every horizon**: 0.3778 / 0.4171 /
0.4478 / 0.4994. Our `1e-3` was helping. Matching upstream here costs accuracy.

### What the two failures establish

| config | H=720 MSE |
|---|---:|
| cosine + wd=1e-3 | 0.4981 |
| type3 + wd=1e-3 | 0.4981 |
| cosine + wd=0 | 0.4994 |

Three quite different optimizer/schedule setups land within 0.0013 of each other,
against the paper's 0.449. The gap is **insensitive to training hyperparameters**,
so the cause is structural rather than a matter of tuning.

## The open lead: validation and test disagree

| H | best val MSE | test MSE | ratio |
|---:|---:|---:|---:|
| 96 | 0.676 | 0.372 | 1.8× |
| 192 | 0.930 | 0.411 | 2.3× |
| 336 | 1.182 | 0.439 | 2.7× |
| 720 | 1.499 | 0.498 | **3.0×** |

Validation is 2–3× harder than test, and the divergence grows with the horizon.
Checkpoints are selected on validation — so at H=720 the model is chosen at epoch
2 on the basis of a split that disagrees sharply with the one we report.

If validation is a poor proxy for test here, epoch 2 may simply be a bad choice
*for test*, which would explain why no training-dynamics fix helps: the problem
would be **which checkpoint is selected**, not how the model is trained.

Diagnostic, not yet run: log test MSE per epoch (purely for inspection, never for
selection) and check whether the test-optimal epoch is far from the val-optimal
one.

## Verification

Every reported number was audited rather than trusted:

- All SLURM tasks `COMPLETED` with exit `0:0` and empty stderr.
- `best_val_mse == min(val_mse)` and `best_epoch == argmin(val_mse)` exactly.
- Each saved checkpoint was **reloaded from disk and re-evaluated**, reproducing
  its reported test MSE to 5 decimal places. This confirms the reported score
  comes from the best-validation weights rather than the final ones.
- The config recorded in each result JSON confirms the reduced ETTh1 model
  (`d_model=16, n_heads=4, d_ff=128`) was actually used, not the defaults.

## Known deviations from the official setup

Recorded rather than hidden, since they may explain part of the gap.

1. **LR schedule** — cosine annealing here; `lradj type3` officially.
2. **Dropout** — the paper text states 0.2 everywhere, but the released
   `etth1.sh` uses 0.3. We follow the script, since that is what produced the
   published numbers. A genuine paper/code discrepancy.
3. **Early stopping** — their default patience is 100 over 100 epochs, so it
   never fires; they train the full 100 and keep the best-validation
   checkpoint. We match this.
4. **Single seed.** One run per horizon, no error bars. The paper does not
   report variance either, but a serious comparison would need several seeds.

## Reproducing

```bash
mkdir -p logs
sbatch scripts/train_slurm.sh     # array 0-3, one horizon each
python experiments/collate.py
```
