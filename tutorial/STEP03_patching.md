# Step 3 — Patching (the core of the paper)

## The paper, first

Section 4.1 and Figure 2. This is the idea the title is about.

A vanilla Transformer on a time series makes **one token per timestep**. Two
things go wrong:

**1. Cost.** Self-attention is O(N²) in token count. A 336-step lookback with
one token per step means 336² ≈ 113k attention entries — per head, per layer.
You cannot afford a long history, so you don't use one.

**2. Semantics.** A single reading carries almost nothing on its own. The title's
analogy: a character is not a word. You want tokens that *mean* something.

**Patching fixes both at once.** Chop the series into windows of length `P`,
stepping by `S`, and make each window one token:

```
L = 16,  P = 4,  S = 4  (no overlap)        L = 16,  P = 4,  S = 2  (50% overlap)
[ 1  2  3  4]                               [ 1  2  3  4]
[ 5  6  7  8]                               [ 3  4  5  6]
[ 9 10 11 12]                               [ 5  6  7  8]
[13 14 15 16]                               [ 7  8  9 10]  ...
→ 4 tokens                                  → more tokens, smoother coverage
```

Token count falls from `L` to about `(L−P)/S + 1`. At the paper's ETTh1 setting
(`L=336, P=16, S=8`) that is **336 → 42 tokens**, so attention gets ~64× cheaper.

Follow the consequence, because it is the actual argument of the paper: cheaper
attention → you can afford a long lookback → and a long lookback is where most
of the accuracy comes from. Patching is not primarily a "better representation"
claim. It is what *buys* the long lookback. Table 6's lookback sweep is the
evidence, and you'll reproduce it in Step 10.

## The exact patch count

The official implementation pads the end of the series by replicating the final
value `S` times, which buys exactly one more patch. So:

```
N = floor((L − P) / S) + 2
```

The `+2` is `+1` for the ordinary fencepost and `+1` for the padded patch.
Check yourself: `L=336, P=16, S=8` → `floor(320/8) + 2 = 40 + 2 = 42`. ✓

Keep this exact. `N` sets the input width of the prediction head in Step 6, so
an off-by-one changes the parameter count of the entire model.

## Your task

Two gaps in `patchtst/patching.py`.

**`num_patches(...)`**

```python
base = (seq_len - patch_len) // stride + 1
return base + 1 if pad_end else base
```

**`Patchify.forward(x)`** — the transpose and the pad are written for you. The
one line you add is:

```python
x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
```

`unfold` slides a window of `size` along `dimension`, stepping by `step`, and
appends the window contents as a **new trailing axis**. So
`[B, C, T_padded] → [B, C, n_patches, patch_len]`, which is exactly the target
shape. Return it.

## Check

```bash
python tests/check_step03.py
```

It checks `num_patches` against the paper's own configuration (336/16/8 → 42),
the output shape, and — the one that actually matters — that patch *contents*
are correct: patch `i` must equal `x[i*S : i*S + P]`. It also verifies the
overlap case and that the padded final patch repeats the last value.

## Note on why `unfold` is free

`unfold` returns a **view**, not a copy. It re-describes the existing memory with
new strides; nothing is allocated and nothing is moved. The central operation of
the paper costs zero memory and zero time.

That is also the trap: because it is a view, it is not contiguous. If you later
hit an error like *"view size is not compatible with input tensor's size and
stride"*, you need `.contiguous()` before the reshape. You will meet this in
Step 5.

## Note on the axis order

Output is `[B, C, N, P]` — channels are kept on their **own axis**, not mixed
into the features. That is deliberate and it is what makes Step 5 possible: to
get channel independence you simply fold `C` into the batch, `[B, C, N, P] →
[B·C, N, P]`, and the Transformer never sees across channels. Had we flattened
channels into the token vector here, channel independence would be impossible
later.
