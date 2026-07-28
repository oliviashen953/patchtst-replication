# Step 5 — Channel independence

## The paper, first

Section 4.2. This is the paper's second contribution, and the more surprising one.

**Channel mixing** — what older multivariate Transformers do — takes all `C`
variables at each timestep and projects them jointly into one token. Every token
is a blend of temperature, humidity, pressure, and so on.

**PatchTST does the opposite.** Each channel is forecast *separately*, through
*one shared backbone*. No attention ever crosses channels.

That should sound wrong. Isn't the whole point of a multivariate model to exploit
cross-variable structure? Table 7's ablation says otherwise, and there are two
reasons it wins:

1. **Less overfitting.** Channel mixing must learn a joint embedding whose
   parameter count grows with `C`. On Traffic — 862 channels — that's a large and
   mostly spurious thing to fit.
2. **Shared statistics.** Because the backbone is shared, every channel's data
   trains the *same* weights. You effectively get `C`× more training sequences
   for one parameter set.

## The implementation is a reshape

Here's the part worth appreciating:

```
[B, C, N, P]  ──fold C into batch──▶  [B*C, N, P]
              ──shared encoder─────▶  [B*C, N, d_model]
              ──unfold────────────▶   [B, C, N, d_model]
```

The encoder receives `B*C` independent patch sequences and has **no way to know**
which channel any of them came from. It's structural — not a penalty term, not a
mask. The information simply isn't present. One line buys the entire property.

This is why Step 3's axis order mattered. Had patching flattened channels into the
token vector, none of this would be available.

## Your task

Two gaps in `patchtst/backbone.py`.

**Fold** — the line that creates channel independence:

```python
flat = patches.contiguous().reshape(batch * self.n_channels, self.n_patches, -1)
```

⚠️ **`.contiguous()` is required.** Without it you get *"view size is not
compatible with input tensor's size and stride"* — `patches` is still the
non-contiguous view `unfold` returned in Step 3. This is the payoff for having
noticed that then.

Then run the encoder: `encoded = self.encoder(flat)`.

**Unfold** — put the channel axis back:

```python
return encoded.reshape(batch, self.n_channels, self.n_patches, self.d_model)
```

The argument **order** matters and the bug is silent. You folded as
`(batch, channels)`, so you must unfold in that same order. Swap them and the
tensor still has the right *shape* — only the channels are scrambled.

## Check

```bash
python tests/check_step05.py
```

Shapes, yes — but the important test pushes a constant-valued channel through and
verifies it comes back in the **same slot**, which catches exactly the scrambling
bug above. It also confirms channels genuinely don't interact: perturb channel 0's
input and every other channel's output must be bit-identical.

That last test is the real proof of channel independence. If it fails, information
is leaking across channels somewhere.

## Note on where this is questionable

The paper is careful here and you should be too. Channel independence assumes the
channels are **parallel series**. If one channel *causally drives* another — an
insulin dose driving glucose, a meal driving a glucose rise — then forbidding
cross-channel attention is a genuine modelling restriction, not a free win.

That's not a criticism of the paper; on ETT/weather/traffic the assumption is
reasonable. But it's precisely the question Step 11 revisits on CGM data, where
`meal` and `bolus` are causal drivers of `cgm`. Keep it in mind.
