"""Run after filling the four gaps in patchtst/encoder.py:

    python tests/check_step04.py
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from patchtst.encoder import LearnablePositionEncoding, PatchEmbedding, PatchEncoder
from patchtst.patching import Patchify


def main():
    torch.manual_seed(0)
    B, N, P, D = 4, 42, 16, 128

    # --- PatchEmbedding: P -> d_model, leading dims untouched ---
    embed = PatchEmbedding(P, D)
    assert isinstance(embed.projection, torch.nn.Linear), "projection must be nn.Linear"
    assert embed.projection.in_features == P, embed.projection.in_features
    assert embed.projection.out_features == D, embed.projection.out_features

    out = embed(torch.randn(B, N, P))
    assert tuple(out.shape) == (B, N, D), tuple(out.shape)
    # works with an extra leading axis too ([B, C, N, P])
    out4 = embed(torch.randn(B, 7, N, P))
    assert tuple(out4.shape) == (B, 7, N, D), tuple(out4.shape)
    # wrong last dim is rejected rather than silently broadcast
    try:
        embed(torch.randn(B, N, P + 1))
    except ValueError:
        pass
    else:
        raise AssertionError("PatchEmbedding must reject a wrong patch_len")

    # --- LearnablePositionEncoding ---
    pos = LearnablePositionEncoding(N, D, dropout=0.0)
    names = {n for n, _ in pos.named_parameters()}
    assert names == {"W_pos"}, f"expected exactly one parameter W_pos, got {names}"
    assert tuple(pos.W_pos.shape) == (N, D), tuple(pos.W_pos.shape)
    assert pos.W_pos.requires_grad, "W_pos must be learnable"
    scale = pos.W_pos.abs().max().item()
    assert scale < 0.5, (
        f"W_pos initialised too large (max |w| = {scale:.3f}); a big init swamps "
        "the patch embeddings early in training"
    )

    x = torch.randn(B, N, D)
    pos.eval()
    added = pos(x)
    assert tuple(added.shape) == (B, N, D), tuple(added.shape)
    # it must actually ADD W_pos, broadcast over the batch
    assert torch.allclose(added, x + pos.W_pos, atol=1e-6), (
        "forward must return x + W_pos (broadcast over leading dims)"
    )
    # broadcasting works over an extra leading axis
    assert tuple(pos(torch.randn(B, 7, N, D)).shape) == (B, 7, N, D)
    # wrong trailing shape rejected
    try:
        pos(torch.randn(B, N + 1, D))
    except ValueError:
        pass
    else:
        raise AssertionError("position encoding must reject a wrong n_patches")

    # --- PatchEncoder end to end ---
    enc = PatchEncoder(
        patch_len=P, n_patches=N, d_model=D, n_heads=8, n_layers=2, d_ff=256, dropout=0.0
    )
    enc.eval()
    patches = torch.randn(B, N, P)
    encoded = enc(patches)
    assert tuple(encoded.shape) == (B, N, D), tuple(encoded.shape)
    assert torch.isfinite(encoded).all(), "encoder produced non-finite values"

    # --- the position encoding must MATTER: shuffling patches changes the output ---
    perm = torch.randperm(N)
    shuffled = enc(patches[:, perm])
    # undo the permutation on the output before comparing
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(N)
    assert not torch.allclose(shuffled[:, inverse], encoded, atol=1e-5), (
        "output is permutation-invariant -- the position encoding is not being "
        "applied (did you forget self.position in PatchEncoder.forward?)"
    )

    # --- a non-contiguous view straight from Step 3 must pass through ---
    real = Patchify(patch_len=P, stride=8, pad_end=True)(torch.randn(2, 336, 1))
    assert not real.is_contiguous(), "expected a non-contiguous view from unfold"
    single = real[:, 0]                       # [2, 42, 16], still a view
    assert tuple(enc(single).shape) == (2, N, D), "encoder failed on a strided view"

    # --- rejects the wrong rank ---
    try:
        enc(torch.randn(B, 7, N, P))
    except ValueError:
        pass
    else:
        raise AssertionError("PatchEncoder.forward must require a 3-D input")

    n_params = sum(p.numel() for p in enc.parameters())
    print("PASS Step 4")
    print(f"  PatchEmbedding      : Linear({P} -> {D})")
    print(f"  W_pos               : {tuple(pos.W_pos.shape)}, max|w| = {scale:.4f}")
    print(f"  [B,N,P] -> [B,N,D]  : {(B, N, P)} -> {tuple(encoded.shape)}")
    print(f"  encoder params      : {n_params:,}")


if __name__ == "__main__":
    main()
