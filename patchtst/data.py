"""Step 1 — ETTh1 loading, splitting, scaling, and sliding windows.

Paper anchor: Section 5.1 and Table 8 (dataset description).

The long-term-forecasting benchmark has a fixed protocol that PatchTST inherits
from Informer/Autoformer. Getting it exactly right matters more than it looks:
if you scale using statistics from the whole file, or let a validation window
peek at training rows, your numbers will beat the paper for the wrong reason.

Three rules that define the protocol
------------------------------------
1. ETTh1 is hourly and is split **12 months / 4 months / 4 months** in time
   order. Never shuffle. 12*30*24 = 8640 train rows, 2880 val, 2880 test.
2. The scaler is fit on **training rows only**, then applied to all three
   splits.
3. Validation and test start `seq_len` rows *early*, so that their first
   forecast still has a full history window behind it. Those borrowed rows are
   context only -- they are never prediction targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ETTh1 is hourly, so one "month" of rows is 30 * 24.
HOURS_PER_MONTH = 30 * 24
TRAIN_MONTHS, VAL_MONTHS, TEST_MONTHS = 12, 4, 4


class StandardScaler:
    """Zero-mean unit-variance scaling, fit on training rows only."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "StandardScaler":
        # Per-channel statistics: values has shape [rows, channels].
        self.mean = values.mean(axis=0)
        self.std = values.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("call fit() before transform()")
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("call fit() before inverse_transform()")
        return values * self.std + self.mean


def split_borders(n_rows: int, seq_len: int) -> dict[str, tuple[int, int]]:
    """Return {split: (start, end)} row ranges for ETTh1.

    `start` is where this split's *data* begins, including the `seq_len` rows of
    borrowed history for val and test. `end` is exclusive.

    Example with seq_len=336:
        train -> (0, 8640)
        val   -> (8640 - 336, 11520)
        test  -> (11520 - 336, 14400)

    Note the splits together use 14400 rows, not all 17420. That is the standard
    benchmark -- the tail of the file is unused. Do not "fix" this; every number
    in the paper's tables assumes it.
    """
    train_end = TRAIN_MONTHS * HOURS_PER_MONTH
    val_end = (TRAIN_MONTHS + VAL_MONTHS) * HOURS_PER_MONTH
    test_end = (TRAIN_MONTHS + VAL_MONTHS + TEST_MONTHS) * HOURS_PER_MONTH

    if test_end > n_rows:
        raise ValueError(
            f"file has {n_rows} rows, benchmark needs {test_end}"
        )

    # TODO(you): fill the three (start, end) pairs.
    #
    #   - train starts at 0 and ends at train_end.
    #   - val and test each start `seq_len` rows BEFORE their own data begins,
    #     so the first window has full history. Their ends are val_end/test_end.
    #
    return {
        "train": (0, train_end),
        "val": (train_end - seq_len, val_end),
        "test": (val_end - seq_len, test_end),
    }


class ETTh1Dataset(Dataset):
    """Sliding windows over one split of ETTh1.

    Each item is a pair:
        x -> [seq_len,  n_channels]   the lookback window
        y -> [pred_len, n_channels]   the horizon that immediately follows

    PatchTST is a *direct multi-step* forecaster: one forward pass emits all
    `pred_len` steps at once. There is no decoder and no autoregression, so
    unlike Informer we need no "label_len" start token here.
    """

    def __init__(
        self,
        csv_path: str,
        split: str,
        seq_len: int = 336,
        pred_len: int = 96,
        scaler: StandardScaler | None = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.split = split
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)

        frame = pd.read_csv(csv_path)
        # Column 0 is the timestamp; the remaining 7 are the channels.
        values = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
        self.n_channels = values.shape[1]

        borders = split_borders(len(values), self.seq_len)

        if scaler is None:
            # Fit on TRAIN ROWS ONLY -- this is the line people get wrong.
            train_start, train_end = borders["train"]
            scaler = StandardScaler().fit(values[train_start:train_end])
        self.scaler = scaler

        start, end = borders[split]
        self.data = scaler.transform(values[start:end]).astype(np.float32)

    def __len__(self) -> int:
        # TODO(you): how many complete (lookback, horizon) pairs fit in self.data?
        #
        # A window occupies seq_len + pred_len rows. If the split has R rows,
        # the number of valid start positions is R - seq_len - pred_len + 1.
        # Return 0 rather than a negative number if the split is too short.
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO(you): slice the two windows out of self.data.
        #
        #   x = rows [index,             index + seq_len)
        #   y = rows [index + seq_len,   index + seq_len + pred_len)
        #
        # Return them as torch tensors via torch.from_numpy(...).
        if index < 0 or index >= len(self):
              raise IndexError(index)
        start = index
        mid = start + self.seq_len
        end = mid + self.pred_len
        return torch.from_numpy(self.data[start:mid]), torch.from_numpy(self.data[mid:end])


def make_datasets(
    csv_path: str,
    seq_len: int = 336,
    pred_len: int = 96,
) -> tuple[ETTh1Dataset, ETTh1Dataset, ETTh1Dataset]:
    """Build train/val/test sharing one scaler fit on the training rows."""
    train = ETTh1Dataset(csv_path, "train", seq_len, pred_len)
    val = ETTh1Dataset(csv_path, "val", seq_len, pred_len, scaler=train.scaler)
    test = ETTh1Dataset(csv_path, "test", seq_len, pred_len, scaler=train.scaler)
    return train, val, test
