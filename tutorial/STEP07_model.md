# Step 7 — Assemble PatchTST

## The paper, first

Figure 1 — the whole architecture. Every piece already exists; this step only
wires them together.

```
x  [B, L, C]
  → RevIN.normalize          (inside the backbone)
  → Patchify                 [B, C, N, P]
  → fold C into the batch    [B*C, N, P]
  → PatchEncoder             [B*C, N, d_model]
  → unfold C back out        [B, C, N, d_model]
  → FlattenHead              [B, pred_len, C]
  → RevIN.denormalize        ← the one real subtlety
ŷ  [B, pred_len, C]
```

Worth pausing on how little there is. Two ideas — patching and channel
independence — plus an ordinary Transformer and a single linear layer. No new
attention mechanism, no decoder, no autoregression. The paper's argument is that
getting the input representation right is what matters, and the assembled model
is the evidence: there's nowhere else for the performance to be hiding.

## The denormalize step

The backbone normalized the input, so the head's output is in **normalized
units**. It has to be pushed back into real units before it can be compared with
targets.

This is why the backbone *holds* the RevIN object rather than this file making
its own: `denormalize()` reuses the mean and standard deviation that
`normalize()` stored on that same instance. A fresh `RevIN` would have no
statistics and would raise.

And recall from Step 2 that the statistics are `[B, 1, C]` while the forecast is
`[B, pred_len, C]` — they broadcast even though `pred_len ≠ seq_len`. That's the
property you were careful to preserve, and this is where it pays off.

**Forget this step and nothing crashes.** Your model predicts standardized values
while the loss compares them against real ones. Training "works," the curve goes
down, and the MSE is meaningless. That class of bug — silent, plausible, wrong —
is why `check_step07` tests the output *level*, not just its shape.

## Your task

One gap in `patchtst/model.py`, three lines:

```python
        encoded = self.backbone(x)              # [B, C, n_patches, d_model]
        forecast = self.head(encoded)           # [B, pred_len, C]

        if self.backbone.revin is not None:
            forecast = self.backbone.revin.denormalize(forecast)

        return forecast
```

## Check

```bash
python tests/check_step07.py
```

Beyond shapes, it:

- feeds a series sitting at **+1000** and requires the forecast to come back near
  1000, not near 0 — the denormalize test
- then shifts to **−500** and requires the forecast level to follow, proving the
  statistics are per-instance rather than baked in
- confirms gradients reach every trainable parameter
- wires all four paper horizons (96/192/336/720)
- runs a **real ETTh1 batch** end to end

That last one is the milestone: an actual forecast tensor from actual data.

## Note on the untrained MSE

The check prints an MSE from randomly-initialised weights. Expect something poor
— that's correct and expected. Step 8 adds the training loop; Step 9 is where the
number should start approaching the paper's.

Don't read anything into the untrained value beyond "the plumbing works."
