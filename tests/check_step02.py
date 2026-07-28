"""Run after filling normalize/denormalize in patchtst/revin.py:

    python tests/check_step02.py
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from patchtst.revin import RevIN


def main():
    torch.manual_seed(0)
    B, T, C = 4, 96, 7

    # --- the round trip: this is the whole point of "reversible" ---
    x = torch.randn(B, T, C) * 5.0 + 100.0
    revin = RevIN(C, affine=True)
    back = revin.denormalize(revin.normalize(x))
    assert torch.allclose(back, x, atol=1e-4), (
        f"round trip failed, max err {(back - x).abs().max():.2e} -- "
        "check that denormalize inverts in REVERSE order (affine first)"
    )

    # --- with affine off it must still round-trip exactly ---
    plain = RevIN(C, affine=False)
    back = plain.denormalize(plain.normalize(x))
    assert torch.allclose(back, x, atol=1e-4), "affine=False round trip failed"

    # --- normalized output is ~0 mean / ~1 std along TIME, per window+channel ---
    normed = plain.normalize(x)
    assert normed.mean(dim=1).abs().max() < 1e-4, normed.mean(dim=1).abs().max()
    assert (normed.std(dim=1, unbiased=False) - 1).abs().max() < 1e-3

    # --- statistics are PER INSTANCE, not global ---
    # Two windows at wildly different levels must normalize to the same place.
    low = torch.randn(1, T, C)
    high = low + 1000.0
    pair = torch.cat([low, high], dim=0)
    out = RevIN(C, affine=False).normalize(pair)
    assert torch.allclose(out[0], out[1], atol=1e-3), (
        "windows at different levels did not normalize to the same shape -- "
        "statistics must be computed per window (dim=1), not globally"
    )

    # --- stored stats broadcast to a DIFFERENT horizon length ---
    r = RevIN(C, affine=False)
    r.normalize(torch.randn(B, 336, C))
    y = r.denormalize(torch.randn(B, 96, C))   # pred_len != seq_len
    assert tuple(y.shape) == (B, 96, C), tuple(y.shape)

    # --- statistics must be detached (no grad path through mean/std) ---
    r2 = RevIN(C, affine=False)
    r2.normalize(torch.randn(B, T, C))
    assert not r2._center.requires_grad, "store statistics with .detach()"
    assert not r2._stdev.requires_grad, "store statistics with .detach()"

    # --- subtract_last centers on the FINAL value, so its last row is ~0 ---
    last = RevIN(C, affine=False, subtract_last=True)
    out = last.normalize(x)
    assert out[:, -1, :].abs().max() < 1e-4, "subtract_last must zero the final row"

    # --- denormalize before normalize is an error, not silent garbage ---
    try:
        RevIN(C).denormalize(x)
    except RuntimeError:
        pass
    else:
        raise AssertionError("denormalize() before normalize() must raise")

    # --- the affine parameters are actually learnable ---
    a = RevIN(C, affine=True)
    names = {n for n, _ in a.named_parameters()}
    assert names == {"weight", "bias"}, names

    print("PASS Step 2")
    print(f"  round-trip max err (affine)  : {(revin.denormalize(revin.normalize(x)) - x).abs().max():.2e}")
    print(f"  normalized mean / std        : {float(normed.mean()):+.2e} / {float(normed.std()):.4f}")
    print(f"  learnable params             : {sorted(names)}")


if __name__ == "__main__":
    main()
