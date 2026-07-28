# Step 11 — Does channel independence still make sense on CGM?

This chapter asks the one question the earlier steps kept raising.

## The question

ETTh1's seven channels are parallel measurements of one transformer — six load
sensors and an oil temperature. None *causes* another. Channel independence is a
reasonable assumption there, and Step 10 confirms it wins.

CGM is structurally different. A typical feature set is

```
cgm, meal, bolus, cgm_diff, hour_sin, hour_cos
```

and `meal` and `bolus` are **causal drivers** of `cgm`. Eating at 12:00 makes
glucose rise by 12:30; a bolus makes it fall.

Under channel independence the model forecasting `cgm` is *structurally forbidden*
from looking at `meal` — not discouraged, not regularized away, simply unable.
Recall from Step 5 that the property comes from putting channels in the batch
axis, where no attention can cross. Applied here, that deletes your most
informative input.

Whether the assumption transfers from ETT-style data to CGM is genuinely open.
This chapter tests it.

## The design

A 2×2, because a one-sided comparison would not be interpretable:

| | drivers informative | drivers zeroed (control) |
|---|---|---|
| channel-independent | (a) | (c) |
| channel-mixing | (b) | (d) |

**Prediction:** (b) beats (a). And (d) does *not* beat (c).

The control is the point. Channel mixing adds parameters (a longer position
table), so it could win for the boring reason of being a bigger model. Re-running
with the causal link removed separates "exploits meal/bolus" from "has more
capacity." If mixing wins in both, the causal story is unsupported.

## The data

**No patient data is in this repository.** Real CGM corpora (DCLP3, UVA/Padova)
are under data-use agreements and cannot be redistributed, so this uses a seeded
synthetic generator (`patchtst/cgm.py`). That keeps the repo public and exactly
reproducible.

It is **not** a physiological model. It produces a trace where meals raise
glucose with a ~45-minute delay and boluses lower it with a ~75-minute delay, on
a circadian baseline with autoregressive wander. Sanity figures at the default
settings:

```
mean 127.5   sd 29.6   range 44-262 mg/dL
TIR 70-180: 92.8%   <70: 0.8%   >180: 6.4%
corr(meal[t], cgm[t+45min]) = +0.218      control: +0.006
```

That correlation is what makes the experiment meaningful. An earlier draft of the
generator produced +0.06 against a +0.01 control — too weak for the 2×2 to detect
anything, which would have made the whole chapter vacuous. Worth checking before
trusting any result here.

## Running it

```bash
mkdir -p logs && sbatch scripts/cgm_slurm.sh
python experiments/collate_cgm.py
```

Metrics are reported as RMSE/MAE in **mg/dL on the CGM channel**, denormalized —
standardized MSE is not interpretable for anyone reading a glucose result.

## What this can and cannot support

**Can:** a statement about whether a Transformer with channel independence can
exploit a known causal driver, on data where that driver provably exists.

**Cannot:** anything about real CGM, real patients, or clinical utility. Synthetic
data, single seed, one architecture size, and a generator written to *have* the
causal structure being tested. Establishing this on real data would need the
data, under the appropriate agreement, with patient-level evaluation.

The honest framing: this is a controlled demonstration that the architectural
assumption is load-bearing, and a motivation for testing it properly on real
data — not evidence about glucose.
