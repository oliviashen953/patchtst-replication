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

The learning-rate schedule then decides how much damage that does. The official
ETTh1 script uses `lradj type3` — constant for 3 epochs, then ×0.9 every epoch,
so ~0.17× the initial rate by epoch 20 and ~0.007× by epoch 50. That decay is
strong implicit regularization: the model largely stops moving before it can
overfit hard.

This replication uses cosine annealing over `T_max=100`, which is nearly flat
early — still ~0.9× initial at epoch 20. So on the high-capacity H=720 case we
keep training aggressively straight through the overfitting region. Best-epoch
checkpointing still fires, but the minimum reached is worse because the model
rockets past the good region instead of settling into it.

**Testable prediction:** implementing `lradj type3` should substantially improve
H=720 and barely move H=96. Not yet run.

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
