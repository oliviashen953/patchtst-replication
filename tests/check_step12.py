"""Run after filling the gaps in patchtst/pretrain.py:

    python tests/check_step12.py

Checks the four things masked pretraining can get silently wrong: the mask
count, the leak from overlapping patches, the loss ignoring visible patches,
and the weight transfer actually transferring.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from patchtst.model import PatchTST
from patchtst.patching import Patchify
from patchtst.pretrain import (
    MaskedPatchTST,
    freeze_backbone,
    masked_reconstruction_loss,
    random_patch_mask,
    transfer_backbone,
)


def main():
    torch.manual_seed(0)
    B, C, N, P = 8, 7, 42, 12

    # --- the mask removes exactly 40%, per (sample, channel) ---------------
    patches = torch.randn(B, C, N, P)
    masked, mask = random_patch_mask(patches, mask_ratio=0.4)

    assert tuple(mask.shape) == (B, C, N), tuple(mask.shape)
    assert mask.dtype == torch.bool
    per_series = mask.sum(dim=-1)
    expected = N - int(N * 0.6)                   # 42 - 25 = 17
    assert (per_series == expected).all(), (
        f"every (sample, channel) must lose exactly {expected} of {N} patches, "
        f"got {per_series.unique().tolist()} -- a per-patch coin flip gives the "
        "right ratio on average but the wrong count per series"
    )
    print(f"  masked per series      : {expected}/{N} = {expected / N:.1%}")

    # masked patches are zeroed, visible ones are untouched
    assert (masked[mask] == 0).all(), "masked patches must be set to zero"
    assert torch.equal(masked[~mask], patches[~mask]), "visible patches changed"

    # --- channels are masked INDEPENDENTLY --------------------------------
    # If one mask were broadcast across channels, every channel of a sample
    # would share masked positions. Over 8*7 series that is not a coincidence
    # you can hit by chance.
    agree = (mask[:, 0:1] == mask).all(dim=-1).float().mean().item()
    assert agree < 0.9, (
        "channels appear to share one mask; each (sample, channel) needs its own"
    )

    # --- the loss ignores visible patches ---------------------------------
    target = torch.randn(B, C, N, P)
    recon = target.clone()
    recon[mask] += 3.0                            # wrong only where masked
    loss_masked_wrong = masked_reconstruction_loss(recon, target, mask)

    recon2 = target.clone()
    recon2[~mask] += 3.0                          # wrong only where visible
    loss_visible_wrong = masked_reconstruction_loss(recon2, target, mask)

    assert loss_visible_wrong.item() < 1e-8, (
        f"error on VISIBLE patches leaked into the loss ({loss_visible_wrong:.4f}); "
        "the model can then coast on the 60% it can already see"
    )
    assert abs(loss_masked_wrong.item() - 9.0) < 1e-4, loss_masked_wrong.item()
    print(f"  loss, masked wrong     : {loss_masked_wrong.item():.3f} "
          f"(visible wrong: {loss_visible_wrong.item():.1e})")

    # --- WHY THE PATCHES MUST NOT OVERLAP ---------------------------------
    # The paper's stated reason: "observed patches do not contain information of
    # the masked ones". Make that measurable. Patch a ramp two ways and ask how
    # much of a masked patch's content survives in its neighbours.
    series = torch.arange(512.0).reshape(1, 512, 1)

    overlap = Patchify(patch_len=16, stride=8, pad_end=False)(series)   # P=16,S=8
    a, b = overlap[0, 0, 4], overlap[0, 0, 5]
    shared = len(set(a.tolist()) & set(b.tolist()))
    assert shared == 8, shared

    disjoint = Patchify(patch_len=12, stride=12, pad_end=False, align="end")(series)
    c, d = disjoint[0, 0, 4], disjoint[0, 0, 5]
    assert len(set(c.tolist()) & set(d.tolist())) == 0, (
        "P=12, S=12 patches must be disjoint"
    )
    print(f"  neighbour overlap      : {shared}/16 values at S=8, 0/12 at S=12")

    # align='end' keeps the most RECENT timesteps
    assert disjoint[0, 0, -1, -1].item() == 511.0, "last patch must end at t=511"
    assert disjoint.shape[2] == 42, disjoint.shape

    # --- the model returns aligned (recon, target, mask) -------------------
    model = MaskedPatchTST(n_channels=C, seq_len=512, d_model=32, n_heads=4,
                           d_ff=64, dropout=0.0, head_dropout=0.0)
    model.eval()
    x = torch.randn(B, 512, C)
    recon, target, mask = model(x)
    assert recon.shape == target.shape == (B, C, N, P), (
        recon.shape, target.shape
    )
    assert mask.shape == (B, C, N)
    assert not target.requires_grad, "the reconstruction target must be detached"

    # the target is the NORMALIZED input, not the raw one: RevIN ran first
    assert abs(target.mean().item()) < 0.2, target.mean().item()

    # a masked position carries no input signal into the encoder
    zeroed = model.backbone.patchify(model.backbone.revin.normalize(x))
    assert torch.allclose(target, zeroed, atol=1e-5), (
        "target should be the un-masked normalized patches"
    )

    # --- weight transfer moves the backbone and nothing else ---------------
    forecaster = PatchTST(n_channels=C, seq_len=512, pred_len=96, patch_len=12,
                          stride=12, d_model=32, n_heads=4, d_ff=64,
                          pad_end=False, align="end", revin_affine=False)
    before = forecaster.backbone.encoder.embed.state_dict()
    before = {k: v.clone() for k, v in before.items()}
    head_before = {k: v.clone() for k, v in forecaster.head.state_dict().items()}

    copied = transfer_backbone(model.state_dict(), forecaster)
    assert copied > 0
    after = forecaster.backbone.encoder.embed.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before), (
        "the encoder weights did not change -- transfer was a silent no-op"
    )
    for k, v in forecaster.head.state_dict().items():
        assert torch.equal(head_before[k], v), (
            f"forecasting head {k} was overwritten; the pretrain head "
            "reconstructs patches and must not be transferred"
        )
    print(f"  tensors transferred    : {copied} (head excluded)")

    # every transferred tensor matches the source exactly
    src = model.state_dict()
    dst = forecaster.state_dict()
    for name in src:
        if name.startswith("backbone.") and name in dst:
            assert torch.equal(src[name], dst[name]), name

    # --- freezing ---------------------------------------------------------
    frozen = freeze_backbone(forecaster, True)
    trainable = sum(p.numel() for p in forecaster.parameters() if p.requires_grad)
    total = sum(p.numel() for p in forecaster.parameters())
    assert frozen > 0 and trainable == total - frozen
    assert all(not p.requires_grad
               for n, p in forecaster.named_parameters()
               if n.startswith("backbone."))
    assert all(p.requires_grad
               for n, p in forecaster.named_parameters()
               if n.startswith("head."))
    print(f"  linear probe trains    : {trainable:,}/{total:,} params "
          f"({trainable / total:.1%})")

    freeze_backbone(forecaster, False)
    assert all(p.requires_grad for p in forecaster.parameters())

    # --- a frozen backbone really does not move ---------------------------
    freeze_backbone(forecaster, True)
    snapshot = {k: v.clone() for k, v in forecaster.backbone.state_dict().items()}
    opt = torch.optim.Adam(
        [p for p in forecaster.parameters() if p.requires_grad], lr=1e-2
    )
    forecaster.train()
    for _ in range(3):
        opt.zero_grad()
        torch.nn.functional.mse_loss(
            forecaster(torch.randn(4, 512, C)), torch.randn(4, 96, C)
        ).backward()
        opt.step()
    for k, v in forecaster.backbone.state_dict().items():
        assert torch.equal(snapshot[k], v), f"frozen backbone tensor {k} moved"

    print("check_step12: OK")


if __name__ == "__main__":
    main()
