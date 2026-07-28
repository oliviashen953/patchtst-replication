"""Run after filling the two gaps in patchtst/backbone.py:

    python tests/check_step05.py
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from patchtst.backbone import ChannelIndependentBackbone


def build(**kw):
    params = dict(
        n_channels=7, seq_len=336, patch_len=16, stride=8,
        d_model=64, n_heads=8, n_layers=2, d_ff=128, dropout=0.0, revin=True,
    )
    params.update(kw)
    model = ChannelIndependentBackbone(**params)
    model.eval()
    return model


def main():
    torch.manual_seed(0)
    B, L, C = 4, 336, 7

    model = build()
    assert model.n_patches == 42, f"expected 42 patches, got {model.n_patches}"

    x = torch.randn(B, L, C)
    out = model(x)

    # --- shape: channel axis restored ---
    assert tuple(out.shape) == (B, C, 42, 64), tuple(out.shape)
    assert torch.isfinite(out).all(), "backbone produced non-finite values"

    # --- rejects wrong input shape ---
    for bad in [torch.randn(B, L, C + 1), torch.randn(B, L + 1, C), torch.randn(B, L)]:
        try:
            model(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"should reject input of shape {tuple(bad.shape)}")

    # --- CHANNELS MUST NOT BE SCRAMBLED -------------------------------------
    # Give each channel a distinctive constant offset. RevIN removes the level,
    # so instead give each channel a distinctive *scale*, which survives.
    probe = torch.randn(1, L, C)
    for c in range(C):
        probe[0, :, c] = torch.sin(torch.linspace(0, (c + 1) * 6.0, L))
    encoded = model(probe)

    # Re-run each channel ALONE through a 1-channel backbone with the same
    # weights, and confirm it lands in slot c of the multichannel output.
    solo = build(n_channels=1)
    solo.load_state_dict(
        {k: v for k, v in model.state_dict().items() if not k.startswith("revin.")},
        strict=False,
    )
    solo.eval()
    for c in range(C):
        alone = solo(probe[:, :, c : c + 1])          # [1, 1, 42, 64]
        assert torch.allclose(alone[0, 0], encoded[0, c], atol=1e-4), (
            f"channel {c} did not land in output slot {c} -- the fold/unfold "
            "reshape order does not match (this is the silent scrambling bug)"
        )

    # --- CHANNELS MUST NOT INTERACT ----------------------------------------
    # The perturbation has to change channel 0's SHAPE, not just its level.
    # RevIN normalizes each channel by its own mean and std, so ANY affine
    # change (+= c, *= k) is erased before the model sees it -- a constant
    # offset here makes the whole test vacuous, since nothing changes and
    # "other channels unchanged" becomes trivially true even for a
    # channel-MIXING model. Replacing the waveform survives normalization.
    perturbed = probe.clone()
    perturbed[0, :, 0] = torch.sign(torch.sin(torch.linspace(0, 40.0, L)))
    after = model(perturbed)
    assert not torch.allclose(after[0, 0], encoded[0, 0], atol=1e-4), (
        "perturbing channel 0 did not change channel 0's own output -- the "
        "probe is not reaching the model"
    )
    for c in range(1, C):
        assert torch.allclose(after[0, c], encoded[0, c], atol=1e-6), (
            f"perturbing channel 0 changed channel {c}'s output -- information "
            "is leaking across channels, so this is NOT channel-independent"
        )

    # --- the shared backbone really is shared: params do not scale with C ----
    wide = build(n_channels=64)
    n_narrow = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("revin."))
    n_wide = sum(p.numel() for n, p in wide.named_parameters() if not n.startswith("revin."))
    assert n_narrow == n_wide, (
        f"encoder params changed with C ({n_narrow} vs {n_wide}) -- the backbone "
        "must be shared across channels"
    )

    # --- revin=False still works ---
    plain = build(revin=False)
    assert plain.revin is None
    assert tuple(plain(x).shape) == (B, C, 42, 64)

    print("PASS Step 5")
    print(f"  [B,L,C] -> [B,C,N,D]  : {(B, L, C)} -> {tuple(out.shape)}")
    print(f"  fold                  : [{B}, {C}, 42, 16] -> [{B * C}, 42, 16]")
    print(f"  channels isolated     : perturbing ch0 left all {C - 1} others bit-identical")
    print(f"  encoder params        : {n_narrow:,} at C=7 and C=64 (shared)")


if __name__ == "__main__":
    main()
