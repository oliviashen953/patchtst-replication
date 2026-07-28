# Step 8 — The training loop

## The paper, first

Section 5.2 and Appendix A.2. Nothing conceptually new — this is ordinary
supervised training. But the *protocol* details decide whether your Step 9
numbers are comparable to anyone else's, so they're worth getting exactly right.

**Loss is MSE over the whole horizon at once.** Not per-step, not weighted by how
far ahead the step is. One tensor, one `mse_loss`.

**Adam with a cosine-annealed learning rate.** No warmup.

**Early stopping on validation, never on test.** The test split gets touched
exactly once, at the very end, after reloading the best-validation checkpoint.
Choosing a stopping epoch by watching test is the single most common way
replication numbers end up quietly optimistic — and it's invisible in the
writeup.

**Report MSE *and* MAE.** The benchmark tables list both, and they sometimes
disagree about which model wins.

## A note on scale

The MSE/MAE in the paper's tables are on the **standardized** scale — the data as
your Step 1 scaler left it. That's why ~0.37 is a sensible ETTh1 target rather
than something in the hundreds.

Your model denormalizes its forecast (Step 7) and the Step 1 targets are already
standardized, so both sides are in the same units. **Don't add another
normalization here.** If your Step 9 MSE comes out wildly large, this is the first
thing to check.

## Your task

Three gaps in `patchtst/train.py`.

**`train_epoch`** — one optimizer step:

```python
optimizer.zero_grad()
pred = model(x)
loss = nn.functional.mse_loss(pred, y)
loss.backward()
if grad_clip_norm:
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
optimizer.step()
total += loss.item()
n_batches += 1
```

`zero_grad()` **first**. PyTorch accumulates gradients by default, so forgetting
it silently sums every batch together — your loss goes erratic and you'll blame
the learning rate.

**`evaluate`** — accumulate sums, divide once:

```python
pred = model(x)
sse += ((pred - y) ** 2).sum().item()
sae += (pred - y).abs().sum().item()
count += y.numel()
```

Note: **sums, not a running mean of per-batch means.** Those differ whenever the
last batch is short, and the gap is small enough to look like noise while quietly
making your numbers incomparable.

**`fit`** — best-tracking and early stopping:

```python
improved = metrics["mse"] < history.best_val_mse - config.min_delta
if improved:
    history.best_val_mse = metrics["mse"]
    history.best_epoch = epoch
    best_state = {k: v.detach().cpu().clone()
                  for k, v in model.state_dict().items()}
    epochs_without_improvement = 0
else:
    epochs_without_improvement += 1
    if epochs_without_improvement >= config.patience:
        history.stopped_early = True
        break
```

`.clone()` is load-bearing. `state_dict()` returns references to the *live*
tensors, so without it `best_state` keeps changing as training continues and you
restore the **last** weights instead of the best ones.

## Check

```bash
python tests/check_step08.py
```

It's the most adversarial check so far, because these bugs are all silent:

- `evaluate` is verified against a **hand-computed** case (constant prediction 3,
  constant target 5 → MSE exactly 4.0, MAE exactly 2.0)
- a deliberately uneven loader (5 samples, batch size 2) catches the
  running-mean-of-means mistake
- `train_epoch` is run twice at `lr=0` to confirm gradients **don't** grow — that
  catches a missing `zero_grad()`
- `fit` restores the best model, verified by re-scoring it and matching
  `best_val_mse` — that catches the missing `.clone()`
- early stopping is checked on a frozen model that *cannot* improve
- the checkpoint is written, reloaded, and re-scored identically

## Note on what to expect

You'll smoke-test on CPU with a tiny model. Real ETTh1 runs are Step 9 and want a
GPU via SLURM — 100 epochs over 8,209 windows is not a login-node job.

Don't read anything into the toy MSE values here. All Step 8 proves is that the
loop is correct; whether the *model* is correct is what Step 9 finds out.
