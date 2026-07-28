"""Step 11 — Applying PatchTST to CGM forecasting.

Why this chapter exists
-----------------------
ETTh1's seven channels are parallel measurements of one system: six load
sensors and an oil temperature, all recorded off the same transformer. None of
them *causes* another in any direct sense. Channel independence is a reasonable
assumption there, and the paper's ablation shows it wins.

CGM is different. A typical feature set is

    cgm, meal, bolus, cgm_diff, hour_sin, hour_cos

and `meal` and `bolus` are **causal drivers** of `cgm`. Eating at 12:00 makes
glucose rise by 12:30; an insulin bolus makes it fall. Under channel
independence, the model forecasting `cgm` is structurally forbidden from
looking at `meal` -- not discouraged, not regularized away, simply unable.

That is a real modelling restriction rather than a free win, and whether it
transfers from ETT-style data to CGM is an open question. This chapter tests
it directly, using the `channel_mixing` switch added for Step 10.

**Prediction:** on ETTh1, channel independence should win (it does, in the
paper and in our Step 10 run). On CGM with informative meal/bolus channels,
channel *mixing* should win, because the drivers carry information the target
channel cannot supply on its own.

On the data
-----------
This repository contains no patient data. Real CGM corpora (DCLP3,
UVA/Padova) are subject to data-use agreements and cannot be redistributed,
so this chapter uses a **seeded synthetic generator**. That keeps the whole
repository public and exactly reproducible.

The generator is deliberately simple and is NOT a physiological model. It
produces a glucose trace driven by meals and boluses with plausible timing, so
that the causal structure under test genuinely exists in the data. Conclusions
here are about the *architecture's* ability to exploit a known causal channel,
not about glucose physiology. Any claim about real CGM would need real data
under the appropriate agreement.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

# 5-minute CGM sampling, matching the usual clinical cadence.
SAMPLES_PER_HOUR = 12
SAMPLES_PER_DAY = 24 * SAMPLES_PER_HOUR

FEATURE_COLS = ["cgm", "meal", "bolus", "cgm_diff", "hour_sin", "hour_cos"]


def generate_cgm(
    n_days: int = 120,
    seed: int = 0,
    meal_effect: float = 1.0,
    bolus_effect: float = 1.0,
) -> np.ndarray:
    """Seeded synthetic CGM with meals and boluses as causal drivers.

    Returns [n_samples, 6] in FEATURE_COLS order.

    Not a physiological model. Glucose responds to meals with a delayed,
    smeared rise and to boluses with a delayed fall, on top of a circadian
    baseline and an autoregressive wander. The point is that `meal` and
    `bolus` genuinely carry information about future `cgm`, so a model that
    can see them should beat one that cannot.

    `meal_effect` / `bolus_effect` scale how strongly the drivers act. Setting
    both to 0 removes the causal link entirely, which is the control condition:
    with uninformative drivers, channel mixing should lose its advantage.
    """
    rng = np.random.default_rng(seed)
    n = n_days * SAMPLES_PER_DAY

    minute_of_day = (np.arange(n) * 5) % (24 * 60)
    hour = minute_of_day / 60.0

    # --- events -----------------------------------------------------------
    meal = np.zeros(n)
    bolus = np.zeros(n)
    for day in range(n_days):
        base = day * SAMPLES_PER_DAY
        # three meals a day at loosely-held times, with varying carb load
        for target_hour, spread in ((7.5, 1.0), (12.5, 1.2), (19.0, 1.3)):
            t = base + int((target_hour + rng.normal(0, spread)) * SAMPLES_PER_HOUR)
            if 0 <= t < n:
                carbs = float(np.clip(rng.normal(55, 20), 10, 120))
                meal[t] += carbs
                # a bolus usually accompanies the meal, sometimes late or missed
                if rng.random() < 0.85:
                    delay = int(rng.integers(-2, 7))          # -10 to +30 min
                    b = t + delay
                    if 0 <= b < n:
                        bolus[b] += carbs / rng.uniform(8, 14)   # carb ratio

    # --- impulse responses ------------------------------------------------
    # meal: rise peaking ~45 min later; bolus: fall peaking ~75 min later
    def kernel(peak_min: float, width_min: float, length_min: int = 300) -> np.ndarray:
        t = np.arange(0, length_min, 5, dtype=float)
        k = np.exp(-0.5 * ((t - peak_min) / width_min) ** 2)
        return k / k.sum()

    meal_k = kernel(45.0, 35.0)
    bolus_k = kernel(75.0, 45.0)

    # Scaled so a ~55 g meal produces a peak excursion around 70-90 mg/dL and a
    # matched bolus removes most of it -- i.e. the magnitudes a real trace shows.
    # Getting this wrong makes the whole chapter vacuous: with a weak causal
    # link there is nothing for channel mixing to exploit and the 2x2 measures
    # noise. (First draft used 3.1/26.0, which gave a meal->cgm correlation of
    # +0.06 against a +0.01 control. Far too weak.)
    rise = np.convolve(meal, meal_k)[:n] * 26.0 * meal_effect
    fall = np.convolve(bolus, bolus_k)[:n] * 200.0 * bolus_effect

    # --- baseline ---------------------------------------------------------
    circadian = 10.0 * np.sin(2 * np.pi * (hour - 3.0) / 24.0)
    wander = np.zeros(n)
    for i in range(1, n):
        wander[i] = 0.995 * wander[i - 1] + rng.normal(0, 1.4)

    cgm = 120.0 + circadian + wander + rise - fall
    cgm = np.clip(cgm, 40.0, 400.0)

    cgm_diff = np.concatenate([[0.0], np.diff(cgm)])
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)

    return np.stack(
        [cgm, meal, bolus, cgm_diff, hour_sin, hour_cos], axis=1
    ).astype(np.float32)


class CGMDataset(Dataset):
    """Sliding windows over synthetic CGM, split 70/15/15 in time order.

    Defaults follow the CGM convention rather than ETT's: 5-minute samples,
    so seq_len=48 is a 4-hour lookback and pred_len=12 is a 1-hour forecast.
    """

    def __init__(
        self,
        data: np.ndarray,
        split: str,
        seq_len: int = 48,
        pred_len: int = 12,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.seq_len, self.pred_len = int(seq_len), int(pred_len)

        n = len(data)
        train_end = int(0.70 * n)
        val_end = int(0.85 * n)

        if mean is None or std is None:
            # Train rows only, same discipline as Step 1.
            mean = data[:train_end].mean(axis=0)
            std = data[:train_end].std(axis=0)
            std[std == 0] = 1.0
        self.mean, self.std = mean, std

        bounds = {
            "train": (0, train_end),
            "val": (train_end - self.seq_len, val_end),
            "test": (val_end - self.seq_len, n),
        }
        start, end = bounds[split]
        self.data = ((data[start:end] - mean) / std).astype(np.float32)

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        mid = index + self.seq_len
        end = mid + self.pred_len
        return (
            torch.from_numpy(self.data[index:mid]),
            torch.from_numpy(self.data[mid:end]),
        )


def make_cgm_datasets(
    n_days: int = 120,
    seed: int = 0,
    seq_len: int = 48,
    pred_len: int = 12,
    meal_effect: float = 1.0,
    bolus_effect: float = 1.0,
) -> tuple[CGMDataset, CGMDataset, CGMDataset]:
    data = generate_cgm(n_days, seed, meal_effect, bolus_effect)
    train = CGMDataset(data, "train", seq_len, pred_len)
    val = CGMDataset(data, "val", seq_len, pred_len, train.mean, train.std)
    test = CGMDataset(data, "test", seq_len, pred_len, train.mean, train.std)
    return train, val, test
