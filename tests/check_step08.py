"""Run after filling the three gaps in patchtst/train.py:

    python tests/check_step08.py
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from patchtst.model import PatchTST
from patchtst.train import TrainConfig, evaluate, fit, test, train_epoch


class Tiny(nn.Module):
    """Linear map from a flattened lookback to the horizon -- fast to train."""

    def __init__(self, L=8, C=2, H=4):
        super().__init__()
        self.L, self.C, self.H = L, C, H
        self.net = nn.Linear(L * C, H * C)

    def forward(self, x):
        return self.net(x.flatten(1)).view(-1, self.H, self.C)


def toy(n=64, L=8, C=2, H=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, L, C, generator=g)
    y = x[:, :H, :] * 2.0 + 1.0        # a learnable deterministic map
    return TensorDataset(x, y)


def main():
    torch.manual_seed(0)
    device = torch.device("cpu")

    # --- evaluate(): exact arithmetic against a hand-computed answer --------
    class Const(nn.Module):
        def forward(self, x):
            return torch.full((x.shape[0], 4, 2), 3.0)

    x = torch.zeros(6, 8, 2)
    y = torch.full((6, 4, 2), 5.0)         # every error is exactly -2
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    m = evaluate(Const(), loader, device)
    assert abs(m["mse"] - 4.0) < 1e-9, f"mse should be 4.0, got {m['mse']}"
    assert abs(m["mae"] - 2.0) < 1e-9, f"mae should be 2.0, got {m['mae']}"

    # --- evaluate() must SUM then divide, not average per-batch means ------
    # 5 samples, batch_size 2 -> batches of 2,2,1. A running mean of batch
    # means weights the short final batch wrongly.
    xs = torch.zeros(5, 8, 2)
    ys = torch.zeros(5, 4, 2)
    ys[4] = 10.0                            # only the last (short-batch) sample
    uneven = DataLoader(TensorDataset(xs, ys), batch_size=2)

    class Zero(nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 4, 2)

    got = evaluate(Zero(), uneven, device)["mse"]
    correct = (8 * 100.0) / 40.0            # 8 nonzero targets of 10, 40 values
    assert abs(got - correct) < 1e-9, (
        f"mse {got} != {correct} -- accumulate SUMS and divide once at the end, "
        "not a running mean of per-batch means"
    )

    # --- evaluate() must not leave gradients around ------------------------
    probe = Tiny()
    evaluate(probe, DataLoader(toy(16), batch_size=8), device)
    assert all(p.grad is None for p in probe.parameters()), (
        "evaluate() left gradients -- it must run under no_grad"
    )

    # --- train_epoch(): loss must actually go down -------------------------
    model = Tiny()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    loader = DataLoader(toy(64), batch_size=16, shuffle=True)
    first = train_epoch(model, loader, opt, device)
    for _ in range(20):
        last = train_epoch(model, loader, opt, device)
    assert last < first * 0.5, (
        f"loss barely moved ({first:.4f} -> {last:.4f}) -- check that "
        "zero_grad/backward/step are all present and in the right order"
    )
    assert isinstance(first, float) and first == first, "train_epoch must return a float"

    # --- zero_grad() must be called: gradients must not accumulate ---------
    m2 = Tiny()
    opt2 = torch.optim.SGD(m2.parameters(), lr=0.0)      # lr=0 -> weights frozen
    ldr = DataLoader(toy(32), batch_size=8)
    train_epoch(m2, ldr, opt2, device)
    g_after_one = m2.net.weight.grad.abs().sum().item()
    train_epoch(m2, ldr, opt2, device)
    g_after_two = m2.net.weight.grad.abs().sum().item()
    assert abs(g_after_one - g_after_two) < 1e-4 * max(1.0, g_after_one), (
        "gradient magnitude grew across epochs -- optimizer.zero_grad() is "
        "missing, so gradients are accumulating"
    )

    # --- fit(): history recorded, best tracked, best weights restored ------
    model = Tiny()
    cfg = TrainConfig(epochs=12, batch_size=16, learning_rate=0.05,
                      weight_decay=0.0, patience=100, device="cpu")
    hist = fit(model, toy(64, seed=1), toy(32, seed=2), cfg)

    assert len(hist.train_loss) == 12, len(hist.train_loss)
    assert len(hist.val_mse) == len(hist.val_mae) == 12
    assert hist.best_epoch >= 0, "best_epoch was never set"
    assert abs(hist.best_val_mse - min(hist.val_mse)) < 1e-12, (
        f"best_val_mse {hist.best_val_mse} != min(val_mse) {min(hist.val_mse)}"
    )
    # the restored model must BE the best one, not the last one
    final = evaluate(model, DataLoader(toy(32, seed=2), batch_size=16), device)
    assert abs(final["mse"] - hist.best_val_mse) < 1e-6, (
        f"model scores {final['mse']:.6f} but best was {hist.best_val_mse:.6f} -- "
        "best_state must be .clone()d, otherwise you restore the LAST weights"
    )
    assert hist.val_mse[-1] < hist.val_mse[0], "validation MSE did not improve"

    # --- BEST != LAST: the test that actually catches a missing .clone() ---
    # The check above cannot see the bug when the best epoch happens to be the
    # final one, since "restore best" and "restore last" then agree. Use a
    # learning rate high enough that validation degrades after its peak, so the
    # two differ by a wide margin.
    torch.manual_seed(0)
    m_div = Tiny()
    cfg_div = TrainConfig(epochs=15, batch_size=8, learning_rate=1.0,
                          weight_decay=0.0, min_lr=1.0, patience=100,
                          device="cpu")
    h_div = fit(m_div, toy(32, seed=1), toy(32, seed=2), cfg_div)
    assert h_div.best_epoch < len(h_div.val_mse) - 1, (
        "test setup failed: best epoch is still the last one"
    )
    assert h_div.val_mse[-1] > h_div.best_val_mse * 1.5, (
        "test setup failed: final val MSE is not clearly worse than the best"
    )
    restored = evaluate(m_div, DataLoader(toy(32, seed=2), batch_size=8), device)
    assert abs(restored["mse"] - h_div.best_val_mse) < 1e-6, (
        f"restored model scores {restored['mse']:.4f} but the best epoch scored "
        f"{h_div.best_val_mse:.4f} (last epoch was {h_div.val_mse[-1]:.4f}) -- "
        "best_state must be .clone()d. state_dict() returns references to the "
        "LIVE tensors, so without a copy you restore the final weights."
    )

    # --- cosine schedule actually decays the LR ----------------------------
    assert hist.lr[-1] < hist.lr[0], f"lr did not decay: {hist.lr[0]} -> {hist.lr[-1]}"

    # --- early stopping fires on a model that cannot improve ---------------
    # lr=0.0 freezes the weights while leaving autograd intact, so validation
    # MSE is identical every epoch and never beats the first one.
    #
    # Two traps here, both hit while writing this check:
    #   - do NOT use requires_grad_(False): loss.backward() then has no graph
    #     and raises "element 0 of tensors does not require grad".
    #   - min_lr MUST also be 0. CosineAnnealingLR anneals between base_lr and
    #     eta_min, so base_lr=0 with the default eta_min=1e-5 RAISES the lr
    #     above zero and the weights keep training.
    stuck = Tiny()
    cfg_es = TrainConfig(epochs=50, batch_size=16, learning_rate=0.0,
                         weight_decay=0.0, min_lr=0.0, patience=3, device="cpu")
    hist_es = fit(stuck, toy(32, seed=3), toy(32, seed=4), cfg_es)
    assert hist_es.stopped_early, "early stopping never triggered"
    assert len(hist_es.val_mse) < 50, len(hist_es.val_mse)

    # --- checkpoint round trip ---------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "best.pt")
        cfg_ck = TrainConfig(epochs=3, batch_size=16, learning_rate=0.05,
                             patience=100, device="cpu", checkpoint=path)
        m3 = Tiny()
        fit(m3, toy(32, seed=5), toy(32, seed=6), cfg_ck)
        assert os.path.exists(path), "checkpoint was not written"
        blob = torch.load(path, weights_only=False)
        assert {"state_dict", "config", "history"} <= set(blob)
        reloaded = Tiny()
        reloaded.load_state_dict(blob["state_dict"])
        a = evaluate(m3, DataLoader(toy(32, seed=6), batch_size=16), device)
        b = evaluate(reloaded, DataLoader(toy(32, seed=6), batch_size=16), device)
        assert abs(a["mse"] - b["mse"]) < 1e-9, "reloaded checkpoint scores differently"

    # --- the real PatchTST trains for a couple of steps ---------------------
    real = PatchTST(n_channels=3, seq_len=32, pred_len=8, patch_len=8, stride=4,
                    d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0.0)
    rx = torch.randn(12, 32, 3)
    rset = TensorDataset(rx, rx[:, :8, :])
    rcfg = TrainConfig(epochs=2, batch_size=4, learning_rate=1e-3, device="cpu")
    rhist = fit(real, rset, rset, rcfg)
    assert len(rhist.val_mse) == 2 and all(v == v for v in rhist.val_mse)
    tm = test(real, rset, rcfg)
    assert {"mse", "mae"} == set(tm)

    print("PASS Step 8")
    print(f"  evaluate() exactness  : mse 4.0 / mae 2.0 on a hand-checked case")
    print(f"  train_epoch()         : loss {first:.4f} -> {last:.4f} over 21 epochs")
    print(f"  fit() best tracking   : best epoch {hist.best_epoch}, "
          f"val mse {hist.best_val_mse:.5f}")
    print(f"  cosine lr             : {hist.lr[0]:.2e} -> {hist.lr[-1]:.2e}")
    print(f"  early stopping        : fired after {len(hist_es.val_mse)} epochs "
          f"(patience 3)")
    print(f"  checkpoint round trip : identical val mse after reload")
    print(f"  real PatchTST         : trained 2 epochs, test {tm['mse']:.4f} mse")


if __name__ == "__main__":
    main()
