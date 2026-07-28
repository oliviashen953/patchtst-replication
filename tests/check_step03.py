"""Run after filling num_patches and Patchify.forward in patchtst/patching.py:

    python tests/check_step03.py
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from patchtst.patching import Patchify, num_patches


def main():
    # --- the paper's own configuration: 336 / 16 / 8 -> 42 tokens ---
    assert num_patches(336, 16, 8, pad_end=True) == 42, num_patches(336, 16, 8, True)
    assert num_patches(336, 16, 8, pad_end=False) == 41

    # --- no-overlap case ---
    assert num_patches(16, 4, 4, pad_end=False) == 4
    assert num_patches(16, 4, 4, pad_end=True) == 5

    # --- the token-count saving that the whole paper rests on ---
    assert 336 / num_patches(336, 16, 8) > 7.9   # ~8x fewer tokens
    # attention cost is quadratic, so the saving is ~64x
    assert (336 ** 2) / (num_patches(336, 16, 8) ** 2) > 63

    # --- reject impossible configurations ---
    for bad in [(8, 16, 8), (16, 4, 0), (16, 0, 4)]:
        try:
            num_patches(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"num_patches{bad} should raise")

    B, T, C = 2, 16, 3
    # Distinct, easily-identified values: channel c, timestep t -> 100*c + t
    x = torch.zeros(B, T, C)
    for c in range(C):
        for t in range(T):
            x[:, t, c] = 100 * c + t

    # --- shape, no overlap, no padding ---
    p = Patchify(patch_len=4, stride=4, pad_end=False)
    out = p(x)
    assert tuple(out.shape) == (B, C, 4, 4), tuple(out.shape)

    # --- CONTENTS: patch i must be x[i*S : i*S + P] ---
    for c in range(C):
        for i in range(4):
            expected = torch.tensor([100 * c + t for t in range(i * 4, i * 4 + 4)],
                                    dtype=torch.float32)
            assert torch.allclose(out[0, c, i], expected), (
                f"channel {c} patch {i}: got {out[0, c, i].tolist()}, "
                f"expected {expected.tolist()}"
            )

    # --- overlapping patches (the paper's setting: S < P) ---
    p2 = Patchify(patch_len=4, stride=2, pad_end=False)
    out2 = p2(x)
    assert tuple(out2.shape) == (B, C, num_patches(16, 4, 2, False), 4)
    # patch 1 starts at t=2 and overlaps patch 0 by 2 steps
    assert torch.allclose(out2[0, 0, 1], torch.tensor([2.0, 3.0, 4.0, 5.0]))
    assert torch.allclose(out2[0, 0, 0, 2:], out2[0, 0, 1, :2]), "overlap mismatch"

    # --- padding replicates the LAST value, buying exactly one extra patch ---
    p3 = Patchify(patch_len=4, stride=4, pad_end=True)
    out3 = p3(x)
    assert tuple(out3.shape) == (B, C, 5, 4), tuple(out3.shape)
    # the padded patch is the final value repeated
    assert torch.allclose(out3[0, 0, -1], torch.full((4,), 15.0)), out3[0, 0, -1].tolist()

    # --- channels stay separated (Step 5 depends on this) ---
    assert not torch.allclose(out[0, 0], out[0, 1]), "channels must not be mixed"

    # --- it is a view: patching costs no memory ---
    assert not out.is_contiguous(), (
        "unfold should return a non-contiguous view -- if this is contiguous "
        "you probably copied the data instead of using unfold"
    )

    # --- works at the real ETTh1 size ---
    real = Patchify(patch_len=16, stride=8, pad_end=True)(torch.randn(8, 336, 7))
    assert tuple(real.shape) == (8, 7, 42, 16), tuple(real.shape)

    print("PASS Step 3")
    print(f"  num_patches(336,16,8)  : {num_patches(336, 16, 8)}   (paper: 42)")
    print(f"  token reduction        : 336 -> {num_patches(336, 16, 8)}  "
          f"({336 / num_patches(336, 16, 8):.1f}x fewer)")
    print(f"  attention cost saving  : {(336 ** 2) / (num_patches(336, 16, 8) ** 2):.0f}x")
    print(f"  [B,T,C] -> [B,C,N,P]   : {tuple(torch.randn(8, 336, 7).shape)} -> {tuple(real.shape)}")


if __name__ == "__main__":
    main()
