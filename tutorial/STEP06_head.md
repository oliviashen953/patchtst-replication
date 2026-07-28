# Step 6 — The prediction head

## The paper, first

Section 4.1, final paragraph. After Step 5, every channel has been encoded into
`n_patches` tokens of width `d_model`. To turn that into a forecast, PatchTST does
the simplest possible thing:

```
[B, C, N, d_model]  ──flatten last two axes──▶  [B, C, N*d_model]
                    ──Linear(N*d_model → H)──▶  [B, C, H]
                    ──transpose──────────────▶  [B, H, C]
```

Flatten, one linear layer, done. That's the whole head.

Two things are worth noticing.

**It's direct multi-step.** All `H` horizon steps come out of a single forward
pass. No decoder, no autoregression, no teacher forcing — and therefore no error
accumulation along the horizon. This is why Step 1 needed no `label_len`, and it's
a real architectural difference from the Informer/Autoformer family, not a detail.

**It's deliberately boring.** After an elaborate story about patching and channel
independence, the read-out is one `nn.Linear`. That is itself a claim: *if the
representation is right, the head doesn't have to be clever.* Everything
interesting already happened upstream.

## Shared vs individual heads

`individual=False` (default) gives every channel the **same** head weights —
consistent with channel independence, where one shared backbone serves all series,
and it keeps the head's parameter count independent of `C`.

`individual=True` gives each channel its own head, multiplying head parameters by
`C`. On Traffic (862 channels) that's the difference between one head and 862 of
them.

The `individual` branch is written for you. You implement the shared one.

## Your task

Three gaps in `patchtst/head.py`.

**Create the shared head** — one `nn.Linear(in_features, self.pred_len)` named
`self.head`, where `in_features = n_patches * d_model`.

**Apply it** — no loop needed. `nn.Linear` maps the *last* axis, and `flat` is
`[B, C, N*d_model]`, so one call handles every channel at once:

```python
out = self.head(flat)          # [B, C, pred_len]
```

That single line is what "shared across channels" means in practice.

**Transpose** — `out` is `[B, C, pred_len]`, but the rest of the world (and your
Step 1 targets) expect `[B, pred_len, C]`:

```python
return out.transpose(1, 2)
```

## Check

```bash
python tests/check_step06.py
```

Shapes, the shared/individual parameter-count difference, and — the one that
matters — that the transpose is right way round. It builds a case where channel
`c` is driven to a distinctive value and confirms it lands in the channel axis of
the output, not the time axis.

## Note on why the transpose bug is nasty

If `C == pred_len`, a missing or doubled transpose produces a tensor of *exactly
the right shape* with the axes swapped. Every shape assertion passes; only the
numbers are wrong, and your loss just trains to something slightly worse without
ever erroring.

On ETTh1 you're safe by luck — `C=7`, `pred_len=96`, so a mistake fails loudly.
But if you later run a config where they coincide, this is the bug you'll spend an
afternoon on. Hence the explicit test.
