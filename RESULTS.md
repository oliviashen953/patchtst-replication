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
so whatever causes it is not a matter of tuning. The next section names a
candidate — and Step 12 later found that candidate doing exactly this, at scale,
on a different set of runs.

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

The diagnostic: log test MSE per epoch (purely for inspection, never for
selection) and check whether the test-optimal epoch is far from the val-optimal
one.

**Update — Step 12 ran exactly this diagnostic on a different set of runs, and
the answer was yes, emphatically.** On the self-supervised arms, selecting on
this validation split cost up to 0.2189 MSE, and at H=720 one arm's validation
picked epoch 0 while its test error kept falling to epoch 19. See
[Step 12](#step-12--masked-self-supervised-pretraining-and-what-a-validation-split-can-hide).

That makes "the H=720 gap is a property of my implementation" the *less* likely
reading. It is now more likely a property of the benchmark. **The confirming run
— Step 9 with `--probe-test` — is queued but has not landed, so this section
still reports the val-selected numbers and the gap above stands as measured.**
Re-run it with:

```bash
PROBE=1 sbatch scripts/train_slurm.sh
```

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
sbatch scripts/train_slurm.sh              # array 0-3, one horizon each
python experiments/collate.py --tag base   # --tag is required once Step 10 has run
```

---

# Step 10 — Ablations

13 runs, single seed, ~2 min each on an A40. Differences below 0.002 MSE are
reported as ties.

## A. Channel independence vs channel mixing (ETTh1, L=336)

| H | independent | mixing | Δ | verdict |
|---:|---:|---:|---:|---|
| 96 | 0.3720 | 0.3719 | −0.0000 | tie |
| 192 | 0.4110 | 0.4101 | −0.0009 | tie |
| 336 | 0.4391 | 0.4348 | −0.0042 | **mixing** |
| 720 | 0.4981 | **0.4540** | **−0.0441** | **mixing** |

**This does not reproduce the paper's Table 7**, which finds channel
independence better. Here it is a tie at short horizons and mixing wins clearly
at long ones.

The H=720 number is the striking part. Channel mixing gives 0.4540 against the
paper's 0.449 — a gap of +0.005, where our channel-independent run sits at
+0.049. So the horizon that would not replicate under channel independence
very nearly replicates under mixing.

Two readings, and this run cannot distinguish them:

1. Channel mixing genuinely helps on ETTh1 at long horizons, and the paper's
   ablation does not cover this exact configuration (reduced ETTh1 model,
   L=336, our schedule).
2. Our channel-*independent* path is limited in some way we have not found, and
   mixing incidentally routes around it.

Reading 2 deserves weight precisely because it would also explain the Step 9
H=720 gap, which survived two other hypotheses. Before treating this as a
finding it needs multiple seeds and a direct check of the channel-independent
path against the official implementation.

## B. RevIN (ETTh1, L=336)

| H | with RevIN | without | Δ | verdict |
|---:|---:|---:|---:|---|
| 96 | 0.3720 | 0.3745 | +0.0025 | RevIN helps |
| 192 | 0.4110 | 0.4165 | +0.0055 | RevIN helps |
| 336 | 0.4391 | 0.4463 | +0.0072 | RevIN helps |
| 720 | 0.4981 | 0.4970 | −0.0011 | tie |

RevIN helps at three of four horizons, with the benefit growing from 96 to 336.
Small in absolute terms, consistent with the paper treating it as a component
rather than a contribution.

## C. Lookback sweep (ETTh1, H=96)

| L | test MSE | trend |
|---:|---:|---|
| 96 | 0.3873 | |
| 192 | 0.3801 | better |
| 336 | **0.3720** | better |
| 512 | 0.3731 | flat |
| 720 | 0.3829 | **worse** |

**Partial reproduction.** MSE improves steadily from L=96 to L=336 (0.3873 →
0.3720), which supports the paper's claim over that range. But it flattens at
512 and degrades at 720, whereas the paper reports monotone gains — its
PatchTST/64 (L=512) beats PatchTST/42 (L=336) on ETTh1.

Two plausible causes, untested: the reduced ETTh1 model (d_model=16) may lack
the capacity to exploit a very long history, and a longer lookback consumes
training windows, so L=720 trains on fewer examples.

The direction of the paper's central claim is reproduced; its monotonicity is
not.

---

# Step 11 — Channel independence on CGM

Synthetic CGM (`patchtst/cgm.py`), 4 h lookback → 1 h ahead, seed 2021,
`data_seed=0`, one run per cell. RMSE in mg/dL on the CGM channel, denormalized.

## The 2×2

| | channel-indep | mixing | Δ | Δ relative |
|---|---:|---:|---:|---:|
| drivers informative | 18.074 | **16.985** | −1.090 | **−6.03%** |
| drivers zeroed (control) | 4.249 | **4.089** | −0.161 | −3.78% |

MAE tracks RMSE (9.111 → 8.575 informative; 3.249 → 3.123 control).

**The prediction was (b) beats (a) *and* (d) does not beat (c). Half of it
held.** Mixing wins with informative drivers, as expected — but it also wins in
the control, where the causal link has been removed from the generator. So the
advantage cannot be attributed to meal/bolus information alone.

## Reading the control correctly

The two rows are **not on the same difficulty scale**, so the absolute Δs should
not be compared directly. Setting `meal_effect = bolus_effect = 0` removes the
excursions from the series itself, not just from the model's inputs — which is
why the control column sits at ~4 mg/dL against ~18. A model predicting a
smooth circadian baseline has a much easier job.

Relative gain is the fairer comparison: **6.03% informative vs 3.78% control.**
A margin of roughly 2.3 points survives the control and is the largest share of
the effect that can be called causal.

## The capacity confound is real and measurable

Mixing is not a free change. It folds `C` into the sequence axis, which
lengthens the position table:

| | parameters |
|---|---:|
| channel-independent | 43,416 |
| channel-mixing | 45,336 (+4.4%) |

That 4.4% is the mechanism behind the control-row win. Without the control the
whole 1.090 mg/dL would have been reported as evidence that the model exploits
meals — and roughly 60% of it would have been an artifact of a bigger model.
This is the chapter's actual lesson.

## What this supports

A residual advantage for channel mixing survives a capacity control on data
where the causal driver provably exists (`corr(meal[t], cgm[t+45min]) = +0.218`
against a +0.006 control). That is consistent with channel independence
discarding usable information when channels are causally linked rather than
parallel — the opposite of the ETTh1 regime in Step 10A, where the channels are
sensors on one transformer.

**What it does not support:** anything about real CGM, real patients, or
clinical utility. Single seed, one architecture size, synthetic data from a
generator written to *have* the structure under test, and a 2.3-point relative
margin that no seed variance has been measured against. Treat it as a motivated
hypothesis for testing on real data under the appropriate agreement, not as a
result about glucose.

## Reproducing

```bash
mkdir -p logs
sbatch scripts/cgm_slurm.sh      # array 0-3, one cell each, ~75 s
python experiments/collate_cgm.py
```

---

# Step 12 — Masked self-supervised pretraining, and what a validation split can hide

ETTh1, paper §4.2 geometry (L=512, non-overlapping P=S=12, d_model 128, 3 layers,
2.4M params), mask ratio 0.4, seed 2021, one run per cell. Four arms share one
pretrained checkpoint: `pretrain` (100 epochs, masked reconstruction, no labels),
`linear_probe` (frozen backbone, 20 epochs), `finetune` (10 probe + 20 end-to-end),
and `scratch` — identical architecture and budget from random init, which is the
like-for-like control the paper's Table 12 does not have.

Pretraining works as advertised: masked reconstruction MSE 0.9098 → 0.4261 over
100 epochs, still improving at epoch 99.

## The result, selecting checkpoints on validation

| H | scratch | lin. probe | finetune | probe−scr | ft−scr | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 96 | 0.3854 | 0.3816 | 0.3829 | −0.0037 | −0.0025 | pretraining helps |
| 192 | 0.4041 | 0.4182 | 0.4357 | +0.0141 | +0.0316 | pretraining hurts |
| 336 | **0.6405** | 0.4620 | 0.4568 | −0.1785 | −0.1837 | pretraining helps |
| 720 | 0.6084 | 0.5920 | 0.5848 | −0.0163 | −0.0236 | pretraining helps |

Read it and stop there and the story is "pretraining helps at three horizons out
of four." **That story is wrong**, and the tell is the bolded cell: `scratch` at
H=336 is *worse* than `scratch` at H=720. Error is supposed to grow with horizon.
One cell being non-monotonic is not a result, it is a symptom.

## The symptom

Every arm's per-epoch validation curve was already in the result JSONs. Six of the
twelve runs pick their best checkpoint at **epoch 0 or 1** out of 20–30, and
validation MSE rises monotonically afterwards.

The first guess — undertrained, raise the learning rate — is refuted by the
training curves stored in the `.pt` files. Train loss falls steadily in all
sixteen phases (`scratch` H=96: 0.4462 → 0.2552, a 43% drop). Optimization is
fine. Train falls while validation rises: this is overfitting.

But then the numbers stop adding up. Validation MSE runs 0.73–2.13 while test MSE
on the same models runs 0.38–0.64. **Validation is two to three times harder than
test.** Checkpoints are selected on validation. So what is actually being measured?

## The diagnostic

`fit()` gained an optional `probe_set`: it records per-epoch test MSE into
`history.probe_mse` and *nothing reads it back* — selection still consults
validation only. That makes one question answerable after the fact: is the epoch
validation picked the same epoch test would have picked?

A caveat about instrumentation. `DataLoader` draws one number from the global RNG
every time an iterator is created — even with `shuffle=False`, to seed workers —
so probing each epoch shifts the training shuffle from the next epoch onward. The
probe loader needs its own `torch.Generator`. With that, all twelve reruns
reproduce their original test MSE to the last digit (Δ = 0.0e+00, twelve for
twelve), which is the proof that the instrument did not disturb the measurement.

| H | arm | val-selected | test-oracle | cost | val epoch | oracle epoch |
|---:|---|---:|---:|---:|---:|---:|
| 96 | scratch | 0.3854 | 0.3745 | 0.0109 | 0 | 2 |
| 96 | linear_probe | 0.3816 | 0.3771 | 0.0045 | 17 | 14 |
| 96 | finetune | 0.3829 | 0.3829 | 0.0000 | 0 | 0 |
| 192 | scratch | 0.4041 | 0.4041 | 0.0000 | 1 | 1 |
| 192 | linear_probe | 0.4182 | 0.4113 | 0.0068 | 14 | 18 |
| 192 | finetune | 0.4357 | 0.4231 | 0.0126 | 1 | 0 |
| 336 | scratch | **0.6405** | **0.4215** | **0.2189** | 13 | 2 |
| 336 | linear_probe | 0.4620 | 0.4320 | 0.0300 | 2 | 17 |
| 336 | finetune | 0.4568 | 0.4568 | 0.0000 | 0 | 0 |
| 720 | scratch | 0.6084 | 0.4259 | 0.1825 | 5 | 0 |
| 720 | linear_probe | 0.5920 | 0.5010 | 0.0910 | 0 | 19 |
| 720 | finetune | 0.5848 | 0.5238 | 0.0610 | 1 | 0 |

**The oracle column selects on test and is therefore not a result.** It is an
upper bound whose only job is to test whether the first table's verdict survives
a change of selection rule.

It does not. `scratch` H=336 loses 0.2189 MSE to selection alone — validation
keeps epoch 13, test's best was epoch 2. That single selection error is the whole
0.6405 outlier, and the outlier is the whole "pretraining helps at 336" verdict.
At H=720 linear probing is worse: validation says stop at epoch 0 while test error
falls all the way to epoch 19. The two splits point in opposite directions.

## The verdict flips

| H | scratch | lin. probe | finetune | winner |
|---:|---:|---:|---:|---|
| 96 | **0.3745** | 0.3771 | 0.3829 | scratch |
| 192 | **0.4041** | 0.4113 | 0.4231 | scratch |
| 336 | **0.4215** | 0.4320 | 0.4568 | scratch |
| 720 | **0.4259** | 0.5010 | 0.5238 | scratch |

Hold the selection rule fixed across arms and **self-supervised pretraining never
beats training from scratch on ETTh1, at any horizon.** The `scratch` column also
becomes monotonic in horizon, as it should have been all along.

This is not a refutation of the paper's method. It is a statement about the
dataset: ETTh1 is small, its validation split disagrees with its test split more
and more as the horizon grows, and pretraining on it sees no data the supervised
model does not already see. The paper's own Table 12 says something compatible —
pretraining wins only at H=96 there, and loses at 336 and 720.

## What this costs the rest of the repo

The same mechanism is the standing explanation for the Step 9 H=720 gap
(+0.0491 against the paper), where validation was already known to be 1.8–3.0×
harder than test with the ratio growing in the horizon. That lead is no longer
speculative: on the Step 12 arms, selecting on this validation split costs up to
0.2189 MSE. Re-running Step 9 with the probe is the obvious next move.

## Two artifacts of mine that were wrong

Worth recording alongside the two in Step 10, because both would have shipped
silently:

1. **`collate_ssl.py` reported a smoke test as a result.** Lookups were keyed on
   `(stage, pred_len)`, and `smoke_ft.json` — a 3-epoch throwaway — carries
   `stage='finetune'`, `pred_len=96` exactly like the real run. Whichever the
   glob yielded first won. The H=96 fine-tuning cell read 0.3768 when the real
   run was 0.3829. Fixed by excluding `smoke_*` at load time instead of trusting
   glob order.
2. **The first version of the probe was not inert** — the `DataLoader` RNG draw
   above. Caught only because the toy check compared probed and unprobed val
   curves and found them differing at 1e-5 by epoch 2.

## Reproducing

```bash
mkdir -p logs
PRE=$(sbatch --parsable --array=0 scripts/ssl_slurm.sh)     # 100 epochs, ~6 min
sbatch --dependency=afterok:$PRE --array=1-12 scripts/ssl_slurm.sh
python experiments/collate_ssl.py
```

Downstream arms reuse `results_ssl/pretrain_etth1_mask40.pt`, so re-running
tasks 1–12 alone is valid as long as that checkpoint is present.
