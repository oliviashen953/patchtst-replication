# Step 4 — Patch embedding, position encoding, Transformer encoder

## The paper, first

Section 4.1, "Transformer Encoder."

Step 3 gave you patches: `[B, C, N, P]`, where each token is a raw vector of `P`
timesteps. A Transformer can't eat that — it wants tokens of width `d_model`.
Three stages fix it:

```
patches  [.., N, P]  ──Linear(P → d_model)──▶  [.., N, d_model]
                     ──+ W_pos───────────────▶  [.., N, d_model]
                     ──Transformer encoder───▶  [.., N, d_model]
```

**Notice what this file does *not* know about: channels.** Nothing here mentions
`C`. That's deliberate. Step 5 folds the channel axis into the batch *before*
calling the encoder, so the encoder only ever sees a stack of independent
univariate patch sequences. Keeping it channel-agnostic is exactly what makes
channel independence a two-line reshape rather than an architectural rewrite.

**The position encoding is learnable**, not the fixed sinusoidal one from
*Attention Is All You Need*. It's a plain `[n_patches, d_model]` parameter added
to the embedded patches. That's defensible here: there are only ~42 positions and
every one is seen on every forward pass, so the usual argument for sinusoidal
(extrapolating to unseen lengths) doesn't apply.

**The encoder is deliberately vanilla.** No new attention mechanism. That *is*
the argument: if patching and channel independence are the real contributions,
they should work with an ordinary encoder. The paper isn't claiming a better
Transformer — it's claiming a better input representation.

## Your task

Four gaps in `patchtst/encoder.py`.

**`PatchEmbedding.__init__`** — one `nn.Linear(patch_len, d_model)`, named
`self.projection`.

**`LearnablePositionEncoding.__init__`** — an `nn.Parameter` of shape
`[n_patches, d_model]` named `self.W_pos`. Initialise it *small*:

```python
self.W_pos = nn.Parameter(torch.empty(n_patches, d_model))
nn.init.uniform_(self.W_pos, -0.02, 0.02)
```

A large init would swamp the patch embeddings early in training.

**`LearnablePositionEncoding.forward`** — one line. `W_pos` is
`[n_patches, d_model]` and `x` is `[..., n_patches, d_model]`, so it broadcasts
over every leading dimension with no reshaping:

```python
return self.dropout(x + self.W_pos)
```

**`PatchEncoder.forward`** — run the three stages in order: `self.embed`, then
`self.position`, then `self.encoder`. Return the result.

No attention mask. Every patch may attend to every other patch — this is an
*encoder*, not a causal decoder. We're not generating the forecast step by step.

## Check

```bash
python tests/check_step04.py
```

It verifies shapes at each stage, that `W_pos` is registered and learnable and
correctly shaped, that the encoder is permutation-**sensitive** (shuffle the
patches and the output must change — proving the position encoding is actually
doing something), and that a non-contiguous input from Step 3's `unfold` passes
through without error.

## Note on why `nn.Linear` accepts a non-contiguous view

The patches arriving from Step 3 are still the strided view that `unfold`
returned. `nn.Linear` handles that fine — it only needs the last axis to be the
feature axis, and it does not reshape. It's `reshape()` that will complain, which
is Step 5's problem.

## Note on `norm_first=True`

Pre-norm (LayerNorm before attention rather than after) trains more stably at
depth and is what the paper's setup uses. It's a one-flag decision in
`nn.TransformerEncoderLayer`, but worth knowing you made it deliberately rather
than by accident.
