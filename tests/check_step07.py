"""Run after wiring PatchTST.forward in patchtst/model.py:

    python tests/check_step07.py
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from patchtst.data import make_datasets
from patchtst.model import PatchTST

CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ETTh1.csv"
)


def build(**kw):
    params = dict(
        n_channels=7, seq_len=336, pred_len=96, patch_len=16, stride=8,
        d_model=64, n_heads=8, n_layers=2, d_ff=128, dropout=0.0, revin=True,
    )
    params.update(kw)
    model = PatchTST(**params)
    model.eval()
    return model


def main():
    torch.manual_seed(0)
    B, L, C, H = 4, 336, 7, 96

    model = build()
    assert model.n_patches == 42, model.n_patches

    x = torch.randn(B, L, C)
    y = model(x)

    # --- the shape a forecaster must produce ---
    assert tuple(y.shape) == (B, H, C), (
        f"expected {(B, H, C)}, got {tuple(y.shape)}"
    )
    assert torch.isfinite(y).all(), "forecast contains non-finite values"

    # --- THE DENORMALIZE TEST ------------------------------------------------
    # Feed a series sitting at a large offset. RevIN strips the level on the way
    # in, so if denormalize is missing the forecast comes back near ZERO instead
    # of near the input's level. This is the silent bug: nothing crashes, the
    # model just predicts standardized values forever.
    offset = 1000.0
    shifted = torch.randn(2, L, C) * 3.0 + offset
    pred = model(shifted)
    level = pred.mean().item()
    assert abs(level - offset) < 200.0, (
        f"forecast mean is {level:.1f} but the input sits at {offset:.0f} -- "
        "the forecast is still in normalized units. Did you forget "
        "self.backbone.revin.denormalize(forecast)?"
    )

    # ...and it must track the level, not just be large: shift again, follow it.
    shifted2 = torch.randn(2, L, C) * 3.0 - 500.0
    level2 = model(shifted2).mean().item()
    assert level2 < level - 500, (
        f"forecast level did not follow the input level ({level:.1f} -> "
        f"{level2:.1f}); RevIN statistics are not being reapplied per instance"
    )

    # --- revin=False: no denormalize, and it must still run ------------------
    plain = build(revin=False)
    assert plain.backbone.revin is None
    assert tuple(plain(x).shape) == (B, H, C)

    # --- individual heads path ----------------------------------------------
    indiv = build(individual=True)
    assert tuple(indiv(x).shape) == (B, H, C)
    assert indiv.n_parameters() > model.n_parameters(), (
        "individual heads should add parameters"
    )

    # --- gradients reach every trainable parameter --------------------------
    train_model = build()
    train_model.train()
    out = train_model(torch.randn(2, L, C))
    loss = torch.nn.functional.mse_loss(out, torch.randn(2, H, C))
    loss.backward()
    missing = [
        n for n, p in train_model.named_parameters()
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())
    ]
    assert not missing, f"no/!finite gradient for: {missing[:5]}"

    # --- different horizons wire up correctly -------------------------------
    for h in (96, 192, 336, 720):
        m = build(pred_len=h)
        assert tuple(m(torch.randn(1, L, C)).shape) == (1, h, C), h

    # --- rejects wrong input shape ------------------------------------------
    for bad in [torch.randn(B, L, C + 1), torch.randn(B, L + 1, C), torch.randn(B, L)]:
        try:
            model(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"should reject {tuple(bad.shape)}")

    # --- END TO END on real ETTh1 -------------------------------------------
    if not os.path.exists(CSV):
        raise SystemExit("data/ETTh1.csv missing -- run: python scripts/download_data.py")
    train, _, _ = make_datasets(CSV, seq_len=L, pred_len=H)
    xb = torch.stack([train[i][0] for i in range(8)])
    yb = torch.stack([train[i][1] for i in range(8)])
    real = build()
    pred = real.predict(xb)
    assert tuple(pred.shape) == tuple(yb.shape) == (8, H, C)
    mse = torch.nn.functional.mse_loss(pred, yb).item()

    print("PASS Step 7")
    print(f"  [B,L,C] -> [B,H,C]   : {(B, L, C)} -> {tuple(y.shape)}")
    print(f"  denormalize          : input level {offset:.0f} -> forecast {level:.1f}")
    print(f"  trainable parameters : {model.n_parameters():,}")
    print(f"  real ETTh1 batch     : {tuple(xb.shape)} -> {tuple(pred.shape)}")
    print(f"  untrained MSE        : {mse:.4f}  (random init -- Step 8 fixes that)")


if __name__ == "__main__":
    main()
