# Head-first SLICE + MLP follow-up

The first nonlinear benchmark did not establish a benefit over clipped Ridge.
Joint fine-tuning often worsened validation error after epoch one. That is evidence
against its particular learning-rate/penalty recipe, not a proof that nonlinear
critics cannot improve. This bounded follow-up separates validation tuning from
explicitly invoked test evaluation. It does not change the working PPO pipeline.

## Predeclared recipe

All models use cached five-layer vectors, fixed input centering/scaling, the same
existing sparse-768 compressor, and `768 -> 256 -> ReLU -> 1 -> sigmoid`. Qwen is
not loaded; output LayerNorm is not used. Only the compressor and MLP can change.

1. **Head-only pilot, seed 42:** compare head learning rates `1e-3` and `3e-4`,
   keeping SLICE frozen. Select by lowest validation Brier.
2. **Head-first joint pilot, seed 42:** start each candidate from that selected
   frozen-head checkpoint. Compare compressor learning rates `1e-5, 3e-6` crossed
   with activation penalties `0, 0.001, 0.01` (six candidates). Set the head rate
   to one tenth of the selected head-only learning rate. Both projection and head
   update during fine-tuning. The starting checkpoint, epoch zero, is eligible for
   validation selection: zero means no joint update improved validation.
3. **Lock recipes and repeat with seeds 43 and 44.** Each new seed trains its own
   frozen head, then initializes its joint fit from that seed's best frozen model.
   These seeds do not choose hyperparameters; their epochs still use validation.
4. Save a checksummed selection and checkpoint list. **Stop without computing test
   predictions.** The tune stage never opens test feature or label files.
5. In a separate evaluation submission, verify the lock and evaluate the selected
   frozen/joint models and both raw/clipped Ridge on the same saved test prefixes.

This is 12 neural fits in total: two head pilots, six joint pilots, and four fits
for the additional seeds. Defaults are 40 epochs maximum per fit, early stopping
after eight epochs without improvement, batch size 2,048, AdamW weight decay
0.001 and gradient clipping at 1. MSE is the prediction loss; the declared mean
activation penalty is varied only for joint learning. It has no effect on frozen
features, so head-only pilots use zero penalty. All settings are fixed before test
evaluation. This is a small optimization study, not an exhaustive search.

The unmodified Ridge baseline is fitted once on the same training prefixes with
epsilon 0.01. Its clipped validation/test predictions are reported explicitly;
clipping is not introduced into PPO. Reported neural learning histories include
training MSE, full training objective and validation Brier. Best checkpoints
are saved independently of the final epoch.

## Run validation tuning

Push the changes, then run on Duke:

```bash
cd /usr/xtmp/ms785/rl_ms_project
git pull --ff-only origin v2
source scripts/cluster_env.sh
mkdir -p logs

sbatch tools/diagnostics/submit_tune_slice_mlp.sh \
  outputs/jlens-large-numina-v2 \
  outputs/numina-no-folds-v1 \
  outputs/slice-mlp-tuned-v1
```

One A5000, eight CPU cores and 64 GB RAM are requested. Twelve hours is an
allocation ceiling, not an ETA; the earlier six-fit benchmark took about 15
minutes, but this larger budget and different early stopping can take longer.
There is no new generation or transformer feature extraction. The historical
source-directory name refers only to stored hidden-state arrays; no J-lens runs.

```bash
tail -n 30 logs/slice-mlp-tune-JOB_ID.out
cat outputs/slice-mlp-tuned-v1/status.json
cat outputs/slice-mlp-tuned-v1/validation-report.txt
```

Successful tuning ends with `state: validation_complete`. Inspect the report,
especially whether joint models select epoch zero and whether training/validation
errors diverge. Completed fits are reused only under the same checksummed protocol;
incomplete fits restart from their initialization, not a partly trained optimizer.
The warm-start checkpoint's hash is also checked. Do not edit source or options
between retries/stages. Use a new directory for a changed experiment.

## Evaluate the locked models

After validation tuning succeeds:

```bash
sbatch tools/diagnostics/submit_tune_slice_mlp.sh \
  outputs/jlens-large-numina-v2 \
  outputs/numina-no-folds-v1 \
  outputs/slice-mlp-tuned-v1 \
  --stage evaluate
```

This performs no training and does not change the selection. Results are in
`report.txt` and `summary.json`; `status.json` becomes `complete`. Paired contrasts
include each neural arm versus **raw and clipped Ridge**, plus joint versus frozen
MLP. Bootstrap intervals resample seeds/questions and are exploratory/unadjusted.

Seed 42 contributes to both recipe selection and the reported three-seed mean;
43/44 did not select recipes. All share one initial compressor and the previously
inspected questions/verifier labels. This does not recover an untouched benchmark
or prove global optimality. If fine-tuning wins, an independently locked dataset
or subsequent RL comparison is still required. The new research checkpoints do
not replace package artifacts or old cumulative `A,b` statistics automatically.
