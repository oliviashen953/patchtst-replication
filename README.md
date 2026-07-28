# PatchTST — a from-scratch replication

**Jiaxin (Olivia) Shen** · Ph.D. researcher, UVA Center for Diabetes Technology

A from-scratch reimplementation of:

> Nie, Y., Nguyen, N. H., Sinthong, P. & Kalagnanam, J. (2023).
> *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.*
> ICLR 2023. arXiv:2211.14730

The idea in one line: **stop feeding Transformers one timestep at a time.**
Group timesteps into *patches* and treat each patch as a token — like words
instead of characters. Then forecast every channel **independently** through one
shared backbone.

```
series      x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8 ...     (L timesteps, C channels)
                       |
   patching            v      P=4, S=2
patches     [x1..x4] [x3..x6] [x5..x8] ...           (N tokens per channel)
                       |
   channel-independent v      one shared Transformer, C series in parallel
encoder     h_1 h_2 h_3 ...
                       |
   flatten + linear    v
forecast    y_1 ... y_H
```

Two consequences that make the paper work:

1. **Attention is O(N²).** Patching cuts the token count from `L` to about
   `(L−P)/S + 2`, so the cost drops *quadratically*. That is what lets PatchTST
   afford a long lookback window when other Transformers cannot.
2. **A patch carries local semantics.** One timestep alone means very little;
   a 40-minute window has shape. This is the "64 words" in the title.

## Why I built this

I am replicating it because PatchTST is both the **backbone** and the **baseline**
in a CGM (continuous glucose monitoring) forecasting project I work on. Rather
than import it as a black box, I wanted to derive every piece from the paper.

This repo is a learning record, built step by step. The commit history is the
point — see [tutorial/](tutorial/).

## Relationship to the official code

The official implementation is <https://github.com/yuqinie98/PatchTST>
(Apache-2.0). **No code from it is copied into this repository.** I used the
paper as the specification and the official repo only as an answer key to check
myself against after each step. See [NOTICE.md](NOTICE.md) for attribution.

## Roadmap

| Step | Module | Paper anchor | What gets built |
|------|--------|--------------|-----------------|
| 1 | `data.py` | §5.1, Table 8 | ETTh1 loading, the 12/4/4-month split, train-only scaling, sliding windows |
| 2 | `revin.py` | §4.2, Kim et al. 2022 | Reversible instance normalization |
| 3 | `patching.py` | **§4.1, Fig 2** | The patching operation — the core contribution |
| 4 | `encoder.py` | §4.1 | Positional encoding + vanilla Transformer encoder |
| 5 | `backbone.py` | **§4.2** | Channel independence: `[B,T,C] → [B·C,N,P]` |
| 6 | `head.py` | §4.1 | Flatten + linear prediction head |
| 7 | `model.py` | Fig 1 | Assemble the full model |
| 8 | `train.py` | §5.2 | Training loop, MSE/MAE, early stopping |
| 9 | `experiments/` | **Table 3** | Reproduce ETTh1 at horizons 96/192/336/720 |
| 10 | `ablations/` | **Tables 6–7** | Channel-independence on/off, RevIN on/off, lookback sweep |
| 11 | `cgm/` | — | Apply the finished model to CGM forecasting |

**Status: steps 1–9 complete.** Steps 10 and 11 are not written yet. See
[RESULTS.md](RESULTS.md) for the ETTh1 reproduction — three of four horizons
land within 0.01 MSE of the paper, and the fourth is documented along with two
hypotheses that failed to explain it.

Step 10 matters most. The paper's sharpest claim is that PatchTST *improves*
with a longer lookback while other Transformers get worse. Reproducing that
curve says more than matching a single MSE.

## Data

`ETTh1` (Electricity Transformer Temperature, hourly) — 17,420 rows, 7 channels,
the standard long-term-forecasting benchmark. It is public, from
[zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset).

```bash
python scripts/download_data.py     # fetches data/ETTh1.csv
```

No patient data appears anywhere in this repository.

## Setup

```bash
python3.11 -m venv ~/venvs/patchtst-env
source ~/venvs/patchtst-env/bin/activate
pip install -r requirements.txt
```

## License

My code is [MIT](LICENSE). The original PatchTST is Apache-2.0 and is neither
included nor modified here; see [NOTICE.md](NOTICE.md).
