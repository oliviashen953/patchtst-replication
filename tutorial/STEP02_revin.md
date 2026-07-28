# Step 2 — RevIN: reversible instance normalization

## The paper, first

PatchTST §4.2 mentions this in one sentence — "we apply reversible instance
normalization" — and cites Kim et al. (ICLR 2022) (https://openreview.net/pdf?id=cGDAkQo1C0p)". 
One sentence, but the ablation in Table 7 shows it is worth a lot. Do not skip it.

**The problem is distribution shift.** A long series drifts. The mean oil
temperature in July is not the mean in December. A model that learned July's
absolute levels sees December and degrades — not because the *shape* changed but
because the *level* did.

**The fix is to remove the level, then put it back.** For each individual
lookback window and each channel separately:

```
normalize     x̂ = γ · (x − μ) / √(σ² + ε) + β
              ↓
            model
              ↓
denormalize   y = (ŷ − β)/γ · √(σ² + ε) + μ
```

where `μ`, `σ²` are computed over **the time axis of that one window**, and
`γ`, `β` are learnable per-channel affine parameters.

"Instance" means per-window, not per-dataset — this is not BatchNorm, and the
statistics are not running averages. "Reversible" is the whole trick: the exact
statistics removed on the way in are restored on the way out, so the network
only ever has to learn shape.

The subtle part: `μ` and `σ` have shape `[batch, 1, channels]`, so they
broadcast over **any** length. That is why the same statistics taken from a
336-step input can be reapplied to a 96-step forecast.

## Your task

Two gaps in `patchtst/revin.py`.

**`normalize(x)`** — `x` is `[batch, time, channels]`, so time is `dim=1`:

```python
if self.subtract_last:
    center = x[:, -1:, :]
else:
    center = x.mean(dim=1, keepdim=True)
var    = x.var(dim=1, keepdim=True, unbiased=False)
stdev  = torch.sqrt(var + self.eps)
```

`keepdim=True` is what keeps them at `[batch, 1, channels]` so they broadcast.
Store both on `self._center` / `self._stdev` with `.detach()` — you do not want
gradients flowing through the statistics. Then subtract, divide, and apply the
affine if `self.affine`.

**`denormalize(x)`** — invert in **reverse order**. Affine first, then scale,
then center:

```python
if self.affine:
    x = (x - self.bias) / (self.weight + self.eps * self.eps)
x = x * self._stdev
x = x + self._center
```

## Check

```bash
python tests/check_step02.py
```

The key test is the round trip: `denormalize(normalize(x)) == x` to floating
point. It also checks that normalized output really has ≈0 mean and ≈1 std per
window, that the statistics are per-window (two windows with very different
levels both normalize to the same place), that `affine=False` still round-trips,
and that `subtract_last` centers on the final value.

## Note on the `eps * eps`

That guard in `denormalize` looks paranoid. It matters: `self.weight` is
*learnable*, so nothing stops training from driving a channel's weight toward 0,
and you would divide by it. Adding `ε²` (not `ε`) keeps the correction far below
the scale of a healthy weight while still being a hard floor.

## Note on ordering

Getting the inverse order wrong is the classic RevIN bug, and it is quiet — the
model still trains, the loss still falls, the forecasts are just subtly wrong.
Affine is applied *last* during normalize, so it must be undone *first* during
denormalize. The round-trip test exists precisely to catch this.
