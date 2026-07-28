"""Step 4 — Patch embedding, positional encoding, Transformer encoder.

Paper anchor: Section 4.1 ("Transformer Encoder").

Where we are
------------
Step 3 turned a batch into patches: [B, C, N, P]. Each patch is a raw vector of
`P` timesteps. A Transformer cannot consume that directly -- it wants tokens of
width `d_model`. So three things happen here:

    patches  [.., N, P]  --Linear(P -> d_model)-->  [.., N, d_model]
                         --+ W_pos-->               [.., N, d_model]
                         --Transformer encoder-->   [.., N, d_model]

Note what is NOT here: nothing in this file knows about channels. That is
deliberate. Step 5 folds the channel axis into the batch before calling this,
so the encoder only ever sees a stack of independent univariate patch
sequences. Keeping the encoder channel-agnostic is what makes channel
independence a two-line reshape instead of an architectural rewrite.

On the positional encoding
--------------------------
The paper uses a *learnable* additive position encoding W_pos, not the fixed
sinusoidal one from "Attention Is All You Need". It is a plain
[n_patches, d_model] parameter added to the embedded patches.

Why learnable is defensible here: there are only ~42 positions, all seen every
forward pass, so there is nothing to extrapolate to -- the argument for
sinusoidal (generalizing to unseen lengths) does not apply.

The encoder itself is a vanilla Transformer encoder. That is the point: the
paper's contribution is the *input representation*, not a new attention
mechanism. If patching and channel independence work, they should work with an
ordinary encoder.
"""

from __future__ import annotations

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """Project raw patches of width `patch_len` to model width `d_model`.

    Input : [..., n_patches, patch_len]
    Output: [..., n_patches, d_model]

    The leading dimensions are untouched, so this works whether you hand it
    [B, C, N, P] or the flattened [B*C, N, P] that Step 5 produces.
    """

    def __init__(self, patch_len: int, d_model: int):
        super().__init__()
        self.patch_len = int(patch_len)
        self.d_model = int(d_model)
        if self.patch_len < 1 or self.d_model < 1:
            raise ValueError("patch_len and d_model must be positive")

        # TODO(you): create the projection.
        #
        # One nn.Linear from patch_len to d_model. Name it self.projection.
        raise NotImplementedError("Step 4: create PatchEmbedding.projection")

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.shape[-1] != self.patch_len:
            raise ValueError(
                f"expected last dim {self.patch_len}, got {patches.shape[-1]}"
            )
        # NOTE: `patches` arrives as a non-contiguous view from Step 3's unfold.
        # nn.Linear handles that fine -- it is reshape() that would complain.
        return self.projection(patches)


class LearnablePositionEncoding(nn.Module):
    """Additive learnable position encoding over the patch axis.

    Input : [..., n_patches, d_model]
    Output: same shape, with W_pos added.
    """

    def __init__(self, n_patches: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.n_patches = int(n_patches)
        self.d_model = int(d_model)
        if self.n_patches < 1 or self.d_model < 1:
            raise ValueError("n_patches and d_model must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.dropout = nn.Dropout(dropout)

        # TODO(you): create the learnable position table.
        #
        # An nn.Parameter of shape [n_patches, d_model], named self.W_pos.
        # Initialise it small -- torch.empty(...) then nn.init.uniform_(..., -0.02, 0.02).
        # A large init would swamp the patch embeddings at the start of training.
        raise NotImplementedError("Step 4: create LearnablePositionEncoding.W_pos")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.n_patches, self.d_model):
            raise ValueError(
                f"expected [..., {self.n_patches}, {self.d_model}], "
                f"got {tuple(x.shape)}"
            )
        # TODO(you): add the position table and apply dropout.
        #
        # self.W_pos is [n_patches, d_model] and x is [..., n_patches, d_model],
        # so it broadcasts over every leading dimension with no reshaping.
        # Return self.dropout(x + self.W_pos).
        raise NotImplementedError("Step 4: implement LearnablePositionEncoding.forward")


class PatchEncoder(nn.Module):
    """Patch embedding + position encoding + vanilla Transformer encoder.

    Input : [..., n_patches, patch_len]
    Output: [..., n_patches, d_model]

    Args:
        patch_len: P, width of one raw patch.
        n_patches: N, how many patches per series (from `num_patches`).
        d_model: model width.
        n_heads: attention heads; must divide d_model.
        n_layers: number of encoder layers.
        d_ff: feed-forward width inside each layer.
        dropout: dropout used in the encoder and after the position encoding.
    """

    def __init__(
        self,
        *,
        patch_len: int = 16,
        n_patches: int = 42,
        d_model: int = 128,
        n_heads: int = 16,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        if n_layers < 1 or d_ff < 1:
            raise ValueError("n_layers and d_ff must be positive")

        self.patch_len = int(patch_len)
        self.n_patches = int(n_patches)
        self.d_model = int(d_model)

        self.embed = PatchEmbedding(patch_len, d_model)
        self.position = LearnablePositionEncoding(n_patches, d_model, dropout)

        # A vanilla encoder -- batch_first so tokens are [batch, seq, feature].
        # GELU and pre-norm follow the paper's setup.
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor is inapplicable with norm_first=True and we have
        # no padding mask anyway (every patch sequence is the same length), so
        # say so explicitly rather than let PyTorch warn about it every run.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 3:
            raise ValueError(
                f"expected [batch, n_patches, patch_len], got {tuple(patches.shape)}"
            )
        # TODO(you): run the three stages in order.
        #
        #   1. x = self.embed(patches)      -> [batch, n_patches, d_model]
        #   2. x = self.position(x)         -> same shape, W_pos added
        #   3. x = self.encoder(x)          -> same shape, attention applied
        #
        # Return x. No mask is needed: every patch may attend to every other
        # patch. This is an ENCODER, not a causal decoder -- we are not
        # generating the forecast one step at a time.
        raise NotImplementedError("Step 4: implement PatchEncoder.forward")
