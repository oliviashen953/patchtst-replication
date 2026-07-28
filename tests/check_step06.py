"""Run after filling the three gaps in patchtst/head.py:

    python tests/check_step06.py
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from patchtst.head import FlattenHead


def main():
    torch.manual_seed(0)
    B, C, N, D, H = 4, 7, 42, 64, 96

    head = FlattenHead(
        n_channels=C, n_patches=N, d_model=D, pred_len=H, dropout=0.0, individual=False
    )
    head.eval()

    # --- the shared head exists and has the right geometry ---
    assert isinstance(head.head, torch.nn.Linear), "self.head must be nn.Linear"
    assert head.head.in_features == N * D, head.head.in_features
    assert head.head.out_features == H, head.head.out_features

    encoded = torch.randn(B, C, N, D)
    out = head(encoded)

    # --- OUTPUT IS TIME-MAJOR: [B, pred_len, C], not [B, C, pred_len] ---
    assert tuple(out.shape) == (B, H, C), (
        f"expected {(B, H, C)}, got {tuple(out.shape)} -- did you forget "
        "out.transpose(1, 2)?"
    )
    assert torch.isfinite(out).all()

    # --- the transpose is the RIGHT WAY ROUND ------------------------------
    # Drive one channel's encoding to something distinctive and confirm the
    # effect shows up along the CHANNEL axis of the output, not the time axis.
    probe = torch.zeros(1, C, N, D)
    probe[0, 3] = 5.0                     # only channel 3 is non-zero
    result = head(probe)                  # [1, H, C]
    per_channel_spread = result[0].std(dim=0)     # variation over time, per channel
    per_channel_mean = result[0].mean(dim=0)      # mean over time, per channel
    others = torch.cat([per_channel_mean[:3], per_channel_mean[4:]])
    assert (per_channel_mean[3] - others.mean()).abs() > 1e-3, (
        "channel 3's distinctive input did not show up in channel 3 of the "
        "output -- the axes are transposed the wrong way"
    )
    # all non-probed channels share the same (bias-only) response
    assert others.std() < 1e-5, "unprobed channels should all give the same output"
    del per_channel_spread

    # --- rejects wrong input shape ---
    for bad in [
        torch.randn(B, C + 1, N, D),
        torch.randn(B, C, N + 1, D),
        torch.randn(B, C, N, D + 1),
        torch.randn(B, C, N),
    ]:
        try:
            head(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"should reject input of shape {tuple(bad.shape)}")

    # --- individual heads: same output shape, C times the head parameters ---
    indiv = FlattenHead(
        n_channels=C, n_patches=N, d_model=D, pred_len=H, dropout=0.0, individual=True
    )
    indiv.eval()
    assert tuple(indiv(encoded).shape) == (B, H, C)

    n_shared = sum(p.numel() for p in head.parameters())
    n_indiv = sum(p.numel() for p in indiv.parameters())
    assert n_indiv == C * n_shared, (
        f"individual heads should have exactly {C}x the shared parameter count, "
        f"got {n_indiv} vs {n_shared}"
    )

    # --- shared head parameters do NOT grow with C ---
    wide = FlattenHead(
        n_channels=64, n_patches=N, d_model=D, pred_len=H, individual=False
    )
    assert sum(p.numel() for p in wide.parameters()) == n_shared, (
        "shared head parameter count must be independent of the channel count"
    )

    print("PASS Step 6")
    print(f"  [B,C,N,D] -> [B,H,C]  : {(B, C, N, D)} -> {tuple(out.shape)}")
    print(f"  shared head           : Linear({N * D} -> {H}), {n_shared:,} params")
    print(f"  individual heads      : {n_indiv:,} params ({C}x)")
    print(f"  shared count at C=64  : {sum(p.numel() for p in wide.parameters()):,} (unchanged)")


if __name__ == "__main__":
    main()
