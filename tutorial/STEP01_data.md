# Step 1 — ETTh1: the split, the scaler, the windows

## The paper, first

Section 5.1 and Table 8. ETTh1 is *Electricity Transformer Temperature, hourly*:
17,420 rows, 7 channels, recorded from 2016-07-01. The 7 channels are six load
measurements (`HUFL HULL MUFL MULL LUFL LULL`) plus oil temperature (`OT`).

PatchTST does not invent an evaluation protocol — it inherits the one from
Informer (Zhou et al., AAAI 2021) so its numbers are comparable to every other
row in Table 3. Three rules define it.

**1. Split by time, 12 / 4 / 4 months.** Never shuffle. Hourly data means one
month is `30 * 24 = 720` rows, so:

```
train  rows [0,     8640)     12 months
val    rows [8640, 11520)      4 months
test   rows [11520, 14400)     4 months
```

Note this uses 14,400 of the 17,420 rows. The tail is simply unused. That looks
like a bug the first time you see it — it isn't, and you must not "fix" it, or
your numbers stop being comparable to the paper.

**2. Fit the scaler on training rows only.** Standardize per channel, using
`mean` and `std` computed from rows `[0, 8640)`, then apply those same
statistics to val and test. Fitting on the whole file leaks test information
into training and will *improve* your MSE for an illegitimate reason.

**3. Val and test start `seq_len` rows early.** The first test forecast still
needs a full lookback window behind it. Those borrowed rows are context only,
never targets:

```
              8640
 train  ........|
 val         [--|-------------]        <- starts at 8640 - seq_len
              ^^^ borrowed history
```

## Your task

Three gaps in `patchtst/data.py`.

**`split_borders(n_rows, seq_len)`** — return `{split: (start, end)}`. Train is
`(0, train_end)`. Val and test each subtract `seq_len` from their start:

```python
return {
    "train": (0, train_end),
    "val": (train_end - seq_len, val_end),
    "test": (val_end - seq_len, test_end),
}
```

**`ETTh1Dataset.__len__`** — a window occupies `seq_len + pred_len` rows, so the
number of valid start positions in `R` rows is `R - seq_len - pred_len + 1`.
Clamp at 0.

**`ETTh1Dataset.__getitem__`** — slice out the two windows and wrap them with
`torch.from_numpy`:

```
x = self.data[index                : index + seq_len]
y = self.data[index + seq_len      : index + seq_len + pred_len]
```

## Check

```bash
source ~/venvs/patchtst-env/bin/activate
python tests/check_step01.py
```

It verifies the border arithmetic at two different `seq_len` values, that the
scaler was fit on train only (train mean ≈ 0, and test mean is *not* ≈ 0), the
dataset lengths, the tensor shapes, and — the important one — that `y` really is
the rows immediately following `x`, with no gap and no overlap.

## Note on why there is no `label_len`

If you read the Informer or Autoformer code you will find a third window called
`label_len` — a chunk of history fed to the decoder as a start token. PatchTST
has **no decoder**. It is a *direct multi-step* forecaster: one forward pass
emits all `pred_len` steps at once from a linear head. So there is nothing to
prime, and `label_len` does not exist here. If you ever port code between the two
families, this is the first thing that will confuse you.
