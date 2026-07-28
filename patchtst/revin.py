"""Step 2 — RevIN: reversible instance normalization.

Paper anchor: PatchTST Section 4.2 ("we apply reversible instance
normalization"), citing Kim et al., ICLR 2022.

The problem
-----------
Long time series drift. The mean level of a channel in July is not the mean
level in December. A model trained on one regime sees a shifted input at test
time and degrades -- classic distribution shift.

The fix
-------
Normalize each *instance* (each individual lookback window, each channel)
by its own statistics before the model sees it, then put the statistics back
on the output:

    normalize:    x_hat = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
    ... model ...
    denormalize:  y     = (y_hat - beta) / gamma * sqrt(var(x) + eps) + mean(x)

`mean` and `var` are computed over the TIME axis only, separately for every
window in the batch and every channel. `gamma` and `beta` are learnable
per-channel affine parameters.

The word "reversible" is the whole trick: the same statistics that were removed
on the way in are restored on the way out, so the model only ever has to learn
the *shape* of the series, not its absolute level.
"""

from __future__ import annotations

import torch
from torch import nn


class RevIN(nn.Module):
    """Reversible instance normalization over the time axis.

    Args:
        n_channels: number of series channels (C).
        eps: numerical floor for the variance.
        affine: learn a per-channel scale and shift.
        subtract_last: center on the final observed value instead of the mean.
            PatchTST exposes this; the default (False) uses the mean.
    """

    def __init__(
        self,
        n_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        subtract_last: bool = False,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.eps = float(eps)
        self.affine = bool(affine)
        self.subtract_last = bool(subtract_last)

        if self.affine:
            self.weight = nn.Parameter(torch.ones(self.n_channels))
            self.bias = nn.Parameter(torch.zeros(self.n_channels))

        # Statistics from the most recent normalize() call, reused by denormalize().
        self._center: torch.Tensor | None = None
        self._stdev: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            return self.normalize(x)
        if mode == "denorm":
            return self.denormalize(x)
        raise ValueError(f"mode must be 'norm' or 'denorm', got {mode!r}")

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, time, channels] -> same shape, per-instance normalized."""
        if x.ndim != 3:
            raise ValueError(f"expected [batch, time, channels], got {tuple(x.shape)}")

        # TODO(you): compute the per-instance statistics over the TIME axis.
        #
        # dim=1 is time. Keep the dim so the tensors broadcast back over x:
        # both should end up shaped [batch, 1, channels].
        #
        #   if self.subtract_last:  center = x[:, -1:, :]
        #   else:                   center = x.mean(dim=1, keepdim=True)
        #
        #   var    = x.var(dim=1, keepdim=True, unbiased=False)
        #   stdev  = torch.sqrt(var + self.eps)
        #
        # Store them on self._center / self._stdev -- denormalize() needs them.
        # Use .detach() so gradients do not flow through the statistics.
        raise NotImplementedError("Step 2: implement normalize")

        # Then: subtract the center, divide by stdev, and if self.affine apply
        #       x = x * self.weight + self.bias

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Undo normalize(), using the statistics it stored.

        x here is the *forecast*, shaped [batch, pred_len, channels]. The
        statistics are [batch, 1, channels], so they broadcast over any horizon
        length -- that is why this works even though pred_len != seq_len.
        """
        if self._center is None or self._stdev is None:
            raise RuntimeError("call normalize() before denormalize()")

        # TODO(you): invert normalize(), in the reverse order.
        #
        #   1. if self.affine: undo the affine step first.
        #        x = (x - self.bias) / (self.weight + self.eps * self.eps)
        #      The extra eps^2 guards against a learned weight collapsing to 0.
        #   2. x = x * self._stdev
        #   3. x = x + self._center
        raise NotImplementedError("Step 2: implement denormalize")
