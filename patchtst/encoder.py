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
        # one nn.Linear from patch_len to d_model
        self.projection = nn.Linear(self.patch_len, self.d_model)
    

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
        self.W_pos = nn.Parameter(torch.empty(self.n_patches, self.d_model))
        nn.init.uniform_(self.W_pos, -0.02, 0.02)
        # the learnable position encoding, compared to fixed position encoding, 
        # is that it is learnable, so it can be updated during training.
        # the reason we use a learnable position encoding is that it is more flexible
        # and can be updated during training.
        # for example, if we have a time series with 100 timesteps,
        #  and we use a fixed position encoding, we will have 100 positions,
        #  and if we have a time series with 1000 timesteps, we will have 1000 positions.
        #  this is not flexible and cannot be updated during training.
        # but if we use a learnable position encoding, 
        # we can have a different position for each timestep,
        # for exa


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
        return self.dropout(x + self.W_pos)


class _Transpose(nn.Module):
    """Swap two axes. Needed only to feed BatchNorm1d, which normalizes over
    dim 1, while our tokens live in [batch, n_patches, d_model]."""

    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0, self.dim1 = dim0, dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim0, self.dim1)


class _MultiheadAttention(nn.Module):
    """Attention with the two upstream options `nn.MultiheadAttention` lacks.

    1. `attn_dropout` separate from residual dropout. Upstream defaults it to
       **0** while using `dropout` everywhere else, so their attention weights
       are never dropped. `nn.TransformerEncoderLayer` has one knob for both.
    2. `res_attention` -- RealFormer-style residual attention (He et al. 2020):
       the pre-softmax scores of layer i are added to those of layer i+1.
       Upstream has this **on** by default.
    """

    def __init__(self, d_model: int, n_heads: int, *, attn_dropout: float = 0.0,
                 proj_dropout: float = 0.0, res_attention: bool = False):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.n_heads = int(n_heads)
        self.d_head = d_model // n_heads
        self.res_attention = bool(res_attention)
        self.scale = self.d_head ** -0.5

        self.W_Q = nn.Linear(d_model, d_model, bias=True)
        self.W_K = nn.Linear(d_model, d_model, bias=True)
        self.W_V = nn.Linear(d_model, d_model, bias=True)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.to_out = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(proj_dropout))

    def forward(self, x: torch.Tensor, prev: torch.Tensor | None = None):
        batch, n, _ = x.shape
        shape = (batch, n, self.n_heads, self.d_head)
        q = self.W_Q(x).view(shape).transpose(1, 2)
        k = self.W_K(x).view(shape).permute(0, 2, 3, 1)
        v = self.W_V(x).view(shape).transpose(1, 2)

        scores = torch.matmul(q, k) * self.scale
        if prev is not None:
            scores = scores + prev
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(batch, n, -1)
        out = self.to_out(out)
        return (out, scores) if self.res_attention else (out, None)


class TSTEncoderLayer(nn.Module):
    """One encoder layer, shaped like upstream's `TSTEncoderLayer`.

    `nn.TransformerEncoderLayer` cannot express three of upstream's defaults:
    BatchNorm instead of LayerNorm, post-norm instead of pre-norm, and residual
    attention. This layer exists so those can be turned on one at a time and
    measured, rather than argued about.
    """

    def __init__(self, *, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.2, attn_dropout: float = 0.0,
                 norm: str = "layer", pre_norm: bool = True,
                 res_attention: bool = False):
        super().__init__()
        if norm not in {"layer", "batch"}:
            raise ValueError(f"norm must be 'layer' or 'batch', got {norm!r}")
        self.pre_norm = bool(pre_norm)
        self.res_attention = bool(res_attention)

        def make_norm():
            if norm == "batch":
                return nn.Sequential(_Transpose(1, 2), nn.BatchNorm1d(d_model),
                                     _Transpose(1, 2))
            return nn.LayerNorm(d_model)

        self.self_attn = _MultiheadAttention(
            d_model, n_heads, attn_dropout=attn_dropout, proj_dropout=dropout,
            res_attention=res_attention)
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = make_norm()

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model))
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = make_norm()

    def forward(self, src: torch.Tensor, prev: torch.Tensor | None = None):
        if self.pre_norm:
            src = self.norm_attn(src)
        attended, scores = self.self_attn(src, prev)
        src = src + self.dropout_attn(attended)
        if not self.pre_norm:
            src = self.norm_attn(src)

        if self.pre_norm:
            src = self.norm_ffn(src)
        src = src + self.dropout_ffn(self.ff(src))
        if not self.pre_norm:
            src = self.norm_ffn(src)
        return (src, scores) if self.res_attention else (src, None)


class TSTEncoder(nn.Module):
    """A stack of TSTEncoderLayer, threading attention scores when asked."""

    def __init__(self, *, n_layers: int, **layer_kwargs):
        super().__init__()
        self.res_attention = bool(layer_kwargs.get("res_attention", False))
        self.layers = nn.ModuleList(
            [TSTEncoderLayer(**layer_kwargs) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = None
        for layer in self.layers:
            x, scores = layer(x, scores if self.res_attention else None)
        return x


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
        impl: str = "torch",
        norm: str = "layer",
        pre_norm: bool = True,
        res_attention: bool = False,
        attn_dropout: float | None = None,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        if n_layers < 1 or d_ff < 1:
            raise ValueError("n_layers and d_ff must be positive")
        if impl not in {"torch", "tst"}:
            raise ValueError(f"impl must be 'torch' or 'tst', got {impl!r}")

        self.patch_len = int(patch_len)
        self.n_patches = int(n_patches)
        self.d_model = int(d_model)

        self.embed = PatchEmbedding(patch_len, d_model)
        self.position = LearnablePositionEncoding(n_patches, d_model, dropout)

        # `torch` cannot express BatchNorm, post-norm, or residual attention, so
        # asking for any of them selects the `tst` stack rather than silently
        # ignoring the request.
        wants_tst = (norm != "layer" or not pre_norm or res_attention
                     or attn_dropout is not None)
        if impl == "torch" and wants_tst:
            impl = "tst"
        self.impl = impl

        if impl == "torch":
            # A vanilla encoder -- batch_first so tokens are [batch, seq,
            # feature]. GELU and pre-norm; note that nn.TransformerEncoderLayer
            # applies `dropout` to the attention weights too, where upstream
            # uses a separate attn_dropout that defaults to 0.
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            # enable_nested_tensor is inapplicable with norm_first=True and we
            # have no padding mask anyway (every patch sequence is the same
            # length), so say so explicitly rather than let PyTorch warn about
            # it every run.
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=n_layers, enable_nested_tensor=False
            )
        else:
            self.encoder = TSTEncoder(
                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                dropout=dropout,
                # Matching the torch path means attention dropout equal to
                # `dropout`; upstream's own default is 0.
                attn_dropout=dropout if attn_dropout is None else attn_dropout,
                norm=norm, pre_norm=pre_norm, res_attention=res_attention,
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
        x = self.embed(patches)
        x = self.position(x)
        return self.encoder(x)
        # Return x. No mask is needed: every patch may attend to every other
        # patch. This is an ENCODER, not a causal decoder -- we are not
        # generating the forecast one step at a time.
