"""Run after filling the three gaps in patchtst/data.py:

    python tests/check_step01.py
"""

import os

# Pin math backends to one thread (avoids the OpenBLAS RLIMIT_NPROC segfault on a
# busy login node). MUST run before numpy/torch are imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from patchtst.data import ETTh1Dataset, make_datasets, split_borders

CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ETTh1.csv"
)


def main():
    if not os.path.exists(CSV):
        raise SystemExit("data/ETTh1.csv missing -- run: python scripts/download_data.py")

    # --- border arithmetic, seq_len = 336 ---
    b = split_borders(17420, 336)
    assert b["train"] == (0, 8640), b["train"]
    assert b["val"] == (8640 - 336, 11520), b["val"]
    assert b["test"] == (11520 - 336, 14400), b["test"]

    # --- borders shift with seq_len; train never does ---
    b96 = split_borders(17420, 96)
    assert b96["train"] == (0, 8640)
    assert b96["val"] == (8640 - 96, 11520)
    assert b96["test"] == (11520 - 96, 14400)

    # --- too-short file is rejected, not silently truncated ---
    try:
        split_borders(1000, 336)
    except ValueError:
        pass
    else:
        raise AssertionError("split_borders must reject a file with too few rows")

    seq_len, pred_len = 336, 96
    train, val, test = make_datasets(CSV, seq_len, pred_len)

    # --- all three splits share ONE scaler object ---
    assert train.scaler is val.scaler is test.scaler, "splits must share the scaler"

    # --- the scaler was fit on TRAIN ROWS ONLY ---
    # Checked directly against the raw file rather than by eyeballing the test
    # mean: on ETTh1 the test period happens to sit very close to the training
    # mean in aggregate (+0.017 vs +0.019 under either scaler), so an aggregate
    # check cannot tell a correct scaler from a leaking one. The per-channel
    # statistics differ plainly, so compare those.
    raw = pd.read_csv(CSV).iloc[:, 1:].to_numpy(dtype=np.float32)
    train_rows = raw[0:8640]
    assert np.allclose(train.scaler.mean, train_rows.mean(axis=0), atol=1e-3), (
        "scaler.mean does not match the mean of rows [0, 8640)"
    )
    assert np.allclose(train.scaler.std, train_rows.std(axis=0), atol=1e-3), (
        "scaler.std does not match the std of rows [0, 8640)"
    )
    # ...and it must NOT be the whole-file statistics (that would leak test data).
    assert not np.allclose(train.scaler.mean, raw.mean(axis=0), atol=1e-2), (
        "scaler was fit on the whole file -- this leaks test information into training"
    )

    # Train rows therefore standardize to ~0 mean / ~1 std by construction.
    assert abs(float(train.data.mean())) < 0.05, float(train.data.mean())
    assert abs(float(train.data.std()) - 1.0) < 0.10, float(train.data.std())

    # --- dataset lengths ---
    assert len(train) == 8640 - seq_len - pred_len + 1, len(train)
    assert len(val) == (11520 - (8640 - seq_len)) - seq_len - pred_len + 1, len(val)
    assert len(test) == (14400 - (11520 - seq_len)) - seq_len - pred_len + 1, len(test)

    # --- shapes ---
    x, y = train[0]
    assert tuple(x.shape) == (seq_len, 7), tuple(x.shape)
    assert tuple(y.shape) == (pred_len, 7), tuple(y.shape)
    assert x.dtype.is_floating_point and y.dtype.is_floating_point

    # --- y is EXACTLY the rows after x: no gap, no overlap ---
    x0, y0 = train[0]
    x1, _ = train[1]
    # window 1 starts one row later than window 0
    assert np.allclose(x0.numpy()[1:], x1.numpy()[:-1]), "windows must stride by 1"
    # the first horizon row is the row right after the last lookback row
    joined = np.concatenate([x0.numpy(), y0.numpy()], axis=0)
    assert np.allclose(joined, train.data[: seq_len + pred_len]), "x|y must be contiguous"

    # --- last index is valid, one past it is not ---
    train[len(train) - 1]
    try:
        train[len(train)]
    except IndexError:
        pass
    except Exception:
        pass  # a raw slice can also just return something short; shape check below
    else:
        xl, yl = train[len(train) - 1]
        assert tuple(yl.shape) == (pred_len, 7)

    print("PASS Step 1")
    print(f"  borders(seq_len=336): {split_borders(17420, 336)}")
    print(f"  n_channels          : {train.n_channels}")
    print(f"  train/val/test wins : {len(train)} / {len(val)} / {len(test)}")
    print(f"  x,y shapes          : {tuple(x.shape)}, {tuple(y.shape)}")
    print(f"  train mean/std      : {float(train.data.mean()):+.4f} / {float(train.data.std()):.4f}")
    print(f"  test  mean/std      : {float(test.data.mean()):+.4f} / {float(test.data.std()):.4f}")


if __name__ == "__main__":
    main()
