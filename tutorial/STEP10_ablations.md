# Step 10 — Ablations

Matching one benchmark row shows your implementation runs. Reproducing the
paper's *argument* shows you understood it. That's what this step is for.

```bash
mkdir -p logs && sbatch scripts/ablation_slurm.sh
python experiments/collate_ablation.py
```

13 runs, ~2 minutes each.

## A. Does channel independence matter?

Paper anchor: Table 7.

`--channel-mixing` folds `C` into the **sequence** instead of the batch, so all
`C×N` patches share one attention window and channel 0's patch can attend to
channel 3's. Same tensor, same numbers — only the axis differs, and that choice
is the entire difference between the two regimes.

Expected: independence wins on ETTh1, where the seven channels are parallel
sensors on one transformer and none causes another.

## B. Does RevIN matter?

Paper anchor: §4.2, one sentence, plus Table 7.

`--no-revin`. Long series drift; without reversible normalization the model must
learn absolute levels rather than shape.

## C. Does a longer lookback help?

**This is the important one.** The paper's sharpest claim is that PatchTST
*improves* with a longer lookback while other Transformers degrade. Follow the
logic: patching cuts token count quadratically → a long lookback becomes
affordable → and the long lookback is where the accuracy comes from. Patching
isn't primarily a "better representation" claim; it's what *buys* the long
history.

So this curve is the actual argument for the method. `L ∈ {96, 192, 336, 512,
720}` at fixed `H=96`.

If MSE falls as `L` rises, you've reproduced the paper's central claim. If it
doesn't, that contradicts the paper and needs investigating before you report
anything.

## Reading the output

`collate_ablation.py` prints all three and states a verdict for each. For C it
identifies the best lookback and says explicitly whether the trend supports or
contradicts the paper.

## Caveat

Single seed throughout. A difference of 0.002 between two configurations is not
separable from seed noise; a difference of 0.02 probably is. Treat small
margins as ties unless you run several seeds.
