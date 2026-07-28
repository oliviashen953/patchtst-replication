# Attribution

## The paper

> Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam.
> "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers."
> *International Conference on Learning Representations (ICLR)*, 2023.
> arXiv:2211.14730

All architectural ideas replicated here — patching, channel independence, the
flatten-and-linear head, and the use of RevIN — are theirs.

## The official implementation

<https://github.com/yuqinie98/PatchTST>, distributed under the Apache License 2.0.

**No source from that repository is copied, vendored, or adapted here.** Every
module in `patchtst/` was written from the paper text. The official repository
was consulted only *after* each step, as an answer key, to confirm that my
independent implementation agreed with theirs.

Where my implementation deliberately differs from the official one, the
difference is documented in the corresponding `tutorial/STEP*.md` file.

## Other credits

- **RevIN** — T. Kim, J. Kim, Y. Tae, C. Park, J.-H. Choi, J. Choo.
  "Reversible Instance Normalization for Accurate Time-Series Forecasting
  against Distribution Shift." ICLR 2022.
- **ETT datasets** — H. Zhou et al., "Informer: Beyond Efficient Transformer for
  Long Sequence Time-Series Forecasting," AAAI 2021.
  Data from <https://github.com/zhouhaoyi/ETDataset>.
- The 12/4/4-month ETT split convention and the evaluation protocol follow the
  Informer/Autoformer benchmark lineage that PatchTST compares against.
