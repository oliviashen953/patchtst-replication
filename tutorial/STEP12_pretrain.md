# Step 12 — Masked self-supervised pretraining

Steps 1–11 built a forecaster and asked what makes it work. This step asks a
different question: **can the model learn something useful without labels at
all?**

## The paper, first

Section 4.2, plus the protocol paragraph in Section 5.3. The recipe is the
masked autoencoder of BERT and MAE, moved onto patches:

```
lookback  [B, L, C]
  ──RevIN──▶ normalized
  ──patch (NON-overlapping)──▶ [B, C, N, P]
  ──mask 40% of patches to zero──▶ [B, C, N, P]
  ──same encoder as Steps 4-5──▶ [B, C, N, d_model]
  ──Linear(d_model → P)──▶ reconstruction [B, C, N, P]
  ──MSE, masked patches only──▶ loss
```

No horizon, no targets, no forecast. Train it to fill in its own holes, then
throw the reconstruction head away and bolt the Step 6 forecasting head on.

Three choices carry the whole section, and the paper argues for each.

**1. Mask patches, not timesteps.** The prior work it compares against
(Zerveas et al., 2021) masks individual time steps. The paper's objection is
that this is *too easy*: a single missing value can be recovered by
interpolating its two neighbours, so a model can score well on the pretext task
without learning anything about the signal. A whole missing patch is a
contiguous 12-step hole, and interpolating across it is a real problem.

**2. Non-overlapping patches.** The supervised model uses `P=16, S=8`, so
neighbouring patches literally share half their values. Under masking that is
fatal — the answer to a masked patch is sitting in the patch next door. Hence
the paper: *"we divide each input sequence into regular non-overlapping patches.
It is for convenience to ensure observed patches do not contain information of
the masked ones."* Representation-learning runs therefore use `L=512, P=12,
S=12` → 42 patches. Same token count as PatchTST/42, different geometry.

**3. A `d_model × P` head, not a per-timestep one.** The paper spends a
paragraph on the arithmetic. If your representation were per-timestep, the
forecasting read-out would need an `(L·D) × (M·T)` matrix — with `L=512, D=128,
M=7, T=720` that is 65,536 × 5,040 ≈ 330M parameters, for a dataset with 8,000
training windows. Per-patch representations keep the pretrain head at `D×P =
128×12 = 1,536` weights. Patching is not only about attention cost; it is what
makes the output layer affordable.

## What "masked" means here, precisely

Masking happens **after RevIN and before the encoder**. Both the input and the
reconstruction target are in normalized units, and `denormalize` is never
called — there is nothing to push back into real units, because the target *is*
the normalized patch. The official code says the same thing in its own idiom:
`RevInCB(dls.vars, denorm=False)`.

Each `(sample, channel)` pair gets its **own** mask. Channel independence again:
the encoder sees `B*C` univariate patch sequences, so there is no reason
channel 3's holes should line up with channel 0's — and if they did, the model
could fill a hole by looking sideways at a correlated channel instead of
learning the sequence.

The loss covers the **masked patches only**. This is the one that bites. Include
the visible 60% and the model drives their error to zero by copying its input,
the total loss falls beautifully, and the masked part never improves.

## Your task

Three gaps in `patchtst/pretrain.py`.

**The mask** — in `random_patch_mask`:

```python
noise = torch.rand(batch, channels, n_patches, ...)
rank  = noise.argsort(-1).argsort(-1)
mask  = rank >= n_keep
```

The double `argsort` is worth pausing on. The first gives you positions in
sorted order; applying it again *inverts that permutation*, turning it into each
position's rank. Keeping every position ranked below `n_keep` therefore removes
an **exact** count — `int(N × 0.6) = 25` kept out of 42 — drawn uniformly at
random. A per-patch coin flip (`torch.rand(...) < 0.4`) gives the right ratio on
average but a different count in every series, and with N=42 that spread is not
small. The official `random_masking` sorts noise for the same reason.

**The loss** — in `masked_reconstruction_loss`:

```python
per_patch = ((recon - target) ** 2).mean(dim=-1)   # [B, C, N]
return (per_patch * mask).sum() / mask.sum()
```

Two reductions, in that order. Mean over `patch_len` first so each patch counts
once whatever `P` is; then divide by the number of **masked** patches, not the
total. Dividing by the total would scale the loss by the mask ratio — harmless-
looking, but it silently rescales your gradients.

**The transfer filter** — in `transfer_backbone`:

```python
if not name.startswith("backbone."): continue
```

That single line is the operational definition of "the learned representation":
RevIN, the patch embedding, the position encoding, three encoder layers. Anything
named `head.*` stays behind, because the pretrain head is `d_model → patch_len`
and the forecasting head is `n_patches·d_model → pred_len`. Different layers,
incompatible shapes.

## Check

```bash
python tests/check_step12.py
```

It pins the four things this step can get silently wrong:

- exactly 17 of 42 patches removed per `(sample, channel)`, and channels not
  sharing one mask;
- the loss reading **zero** when only the visible patches are wrong — the direct
  test for the coasting failure above;
- the overlap leak, made measurable: neighbouring patches share 8 of 16 values
  at `S=8` and 0 of 12 at `S=12`, so the paper's "for convenience" line is a
  number, not a style note;
- the transfer copying the encoder and *not* the head, and a frozen backbone
  actually staying put across three optimizer steps.

## Running it

One pretraining run, then three downstream arms at each of four horizons:

```bash
mkdir -p logs
PRE=$(sbatch --parsable --array=0 scripts/ssl_slurm.sh)
sbatch --dependency=afterok:$PRE --array=1-12 scripts/ssl_slurm.sh
python experiments/collate_ssl.py
```

| arm | what it does | paper |
|---|---|---|
| `pretrain` | masked reconstruction, 100 epochs, no labels | §5.3 |
| `linear_probe` | freeze the backbone, train the head 20 epochs | option (a) |
| `finetune` | 10 epochs probing, then 20 end-to-end | option (b) |
| `scratch` | same architecture, random init, 30 epochs | our control |

The two-phase fine-tune is not fussiness. The paper cites Kumar et al. (2022):
probing before fine-tuning beats fine-tuning directly, because a randomly
initialized head sends large early gradients back through the pretrained
features and distorts exactly what you paid to learn.

## Why `scratch` exists

Table 12 compares its ETTh1 self-supervised numbers against a `Sup.` column of
0.375 / 0.414 / 0.431 / 0.449 — which is, digit for digit, the PatchTST/42 row
of Table 3. That row is `L=336`, overlapping `P=16, S=8`, and the *reduced*
ETTh1 model (`d_model=16, d_ff=128`) from Appendix A.1.4. The self-supervised
runs are `L=512`, non-overlapping `P=S=12`, and the full-size model
(`d_model=128, d_ff=512`, ~1.1M parameters).

So the paper's own supervised-vs-self-supervised comparison on ETTh1 moves the
lookback, the patch geometry *and* the model size at the same time as it moves
the initialization. Any difference is unattributable. `scratch` fixes that:
identical architecture, identical geometry, identical epoch budget, random init
instead of pretrained. It is the only column that can answer "did pretraining
help?".

Keep both comparisons in the output — the paper's row for replication, ours for
the causal claim.

## What to expect

Read the ETTh1 block of Table 12 before you run, because it does not say what a
"pretraining works" story would say:

| H | fine-tune | lin. probe | Sup. |
|---:|---:|---:|---:|
| 96 | **0.366** | 0.371 | 0.375 |
| 192 | 0.431 | **0.411** | 0.414 |
| 336 | 0.450 | 0.445 | **0.431** |
| 720 | 0.472 | 0.487 | **0.449** |

Pretraining wins at H=96, roughly ties at 192, and **loses** at 336 and 720.
The paper is candid about the pattern: *"on large datasets our pre-training
procedure contributes a clear improvement compared to supervised training from
scratch"* — and ETTh1, with 8,033 training windows, is the small end of the
benchmark. Self-supervised pretraining pays for itself when unlabelled data is
plentiful relative to the downstream task. On ETTh1 the same 8,640 rows are used
for both, so pretraining buys no extra data — only a different initialization.

A note on "linear probing" here: the frozen-backbone arm still trains
`42 × 128 × 96 = 516,096` head parameters at H=96 — about a third of the model,
and more than the reduced Step 9 model has in total. "Only the head" is not a
small qualifier when the head is a flatten-and-project.

## Note on what is not replicated

Recorded rather than buried:

1. The official self-supervised code picks its learning rate with an LR-finder
   sweep and trains with `fit_one_cycle`. We use its own `--lr` default of 1e-4
   with this repo's cosine schedule, so all four arms differ in as few ways as
   possible.
2. Its encoder uses BatchNorm; ours is a vanilla PyTorch encoder with
   LayerNorm. That deviation is inherited from Step 4, not introduced here.
3. Validation masking is seeded. Upstream draws a fresh mask every validation
   pass, which makes the curve we select checkpoints on noisy for no benefit.
   Same ratio, same distribution — only the seed is pinned.
4. Everything is n=1, like the rest of this repo. The paper's own robustness
   appendix notes that self-supervised variance is *higher* on the small
   datasets, and ETTh1 is one of them. Treat gaps under ~0.01 as noise.
