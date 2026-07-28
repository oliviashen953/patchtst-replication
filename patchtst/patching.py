"""Step 3 — Patching. This is the core contribution of the paper.

Paper anchor: Section 4.1 and Figure 2.

The idea
--------
A vanilla Transformer on a time series makes one token per timestep. Two things
go wrong with that:

1. **Cost.** Self-attention is O(N^2) in the token count. With one token per
   step, a 336-step lookback costs 336^2 = 112,896 attention entries. You simply
   cannot afford a long history.
2. **Semantics.** A single timestep carries almost no information on its own.
   The analogy in the paper's title: a character is not a word. You want tokens
   that mean something.

Patching fixes both. Chop the series into (possibly overlapping) windows of
length `P`, stepping by `S`, and make each window one token:

    L=16, P=4, S=4  (no overlap)      L=16, P=4, S=2  (50% overlap)
    [ 1  2  3  4]                     [ 1  2  3  4]
    [ 5  6  7  8]                     [ 3  4  5  6]
    [ 9 10 11 12]                     [ 5  6  7  8]
    [13 14 15 16]                     ...
    -> 4 tokens                       -> more tokens, smoother coverage

Token count drops from `L` to roughly `(L - P)/S + 1`. With the paper's ETTh1
setting (L=336, P=16, S=8) that is 336 -> 42 tokens: attention gets ~64x
cheaper. *That* is why PatchTST can use a long lookback when other Transformers
cannot -- and the long lookback is where most of its accuracy comes from.

On the padding
--------------
The official implementation pads the end of the series by replicating the last
value `S` times, which buys exactly one extra patch. So the count in the paper
is

    N = floor((L - P) / S) + 2

The `+2` is `+1` for the usual "fencepost" and `+1` for that padded patch.
Keeping this exact matters: it sets the input width of the prediction head, so
an off-by-one here changes the parameter count of the whole model.

L is the length of the time series,
P is the patch length,
S is the stride,
pad_end is a boolean indicating whether to pad the end of the time series.

The number of patches is calculated as follows:

base = (seq_len - patch_len) // stride + 1
return base + 1 if pad_end else base

Sanity-check yourself against the paper: seq_len=336, patch_len=16, stride=8, pad_end=True should give 42.

"""

from __future__ import annotations

import torch
from torch import nn


def num_patches(seq_len: int, patch_len: int, stride: int, pad_end: bool = True) -> int:
    """Token count produced by `Patchify` for a given configuration."""
    if patch_len > seq_len:
        raise ValueError(f"patch_len {patch_len} exceeds seq_len {seq_len}")
    if stride < 1 or patch_len < 1:
        raise ValueError("patch_len and stride must be positive")

    # TODO(you): return the number of patches.
    # base is the number of patches without the padded patch
    # we subtract the patch length from the sequence length 
    # and divide by the stride to get the number of patches
    # and add one to account for the first patch
    base = (seq_len - patch_len) // stride + 1
    # if pad_end is True, we add one patch for the padded patch
    # otherwise we return the number of patches without the padded patch
    return base + 1 if pad_end else base



class Patchify(nn.Module):
    """Turn [batch, time, channels] into [batch, channels, n_patches, patch_len].

    Note the axis order of the output. Time has been replaced by a *patch* axis,
    and each token is a length-`patch_len` vector. Channels are kept as their own
    axis rather than mixed -- Step 5 relies on that to fold them into the batch
    and get channel independence.

    Args:
        patch_len: P, the window length of one token.
        stride: S, how far the window advances. S == P gives no overlap;
            S < P gives overlapping patches (the paper uses P=16, S=8).
        pad_end: replicate the final timestep `stride` times to gain one patch.

    Example (1 batch, 10 timesteps, 1 channel; P=2, S=1, no padding):

        >>> x = torch.arange(1., 11.).reshape(1, 10, 1)   # [B=1, T=10, C=1]
        >>> Patchify(patch_len=2, stride=1, pad_end=False)(x).shape
        torch.Size([1, 1, 9, 2])

    The output axes are [B, C, N, P], giving the 9 overlapping patches
    [1,2], [2,3], [3,4], ... [9,10].

    Note the input must be 3-D. A bare [1, 10] is rejected: the channel axis
    is not optional, because channel independence in Step 5 depends on it.
    """

    def __init__(self, patch_len: int = 16, stride: int = 8, pad_end: bool = True):
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.pad_end = bool(pad_end)
        # ReplicationPad1d((left, right)) -- pad only on the right.
        self.pad = nn.ReplicationPad1d((0, self.stride)) if self.pad_end else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [batch, time, channels], got {tuple(x.shape)}")

        # Move channels ahead of time: [B, T, C] -> [B, C, T].
        # nn.ReplicationPad1d and Tensor.unfold both operate on the LAST axis,
        # which now holds time. This transpose is the whole reason for the
        # output's axis order.
        x = x.permute(0, 2, 1)

        if self.pad is not None:
            x = self.pad(x)

        # unfold slides a window of `size` along `dimension`, stepping by `step`,
        # and appends the window contents as a NEW trailing axis:
        #   [B, C, T_padded]  ->  [B, C, n_patches, patch_len]
        #
        # It returns a VIEW, not a copy -- new strides over the same memory, so
        # nothing is allocated. The paper's central operation is free. The flip
        # side is that the result is non-contiguous, which Step 5's reshape will
        # care about.
        return x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
