# Pre-prover critic follow-up studies

These studies extend the completed three-seed comparison. They have no results
yet. The original batch-only PPO/ridge/LSTD/GRPO settings remain the default.

## 1. Which layer provides useful value features?

Reuse all ordinary-completion answers from a saved frozen-model dataset. The
recommended source is the existing MATH-specific fit. No new answers are generated
or rescored, and source files are not modified.

For a model with N blocks, capture outputs after blocks ceil(N/3), ceil(2N/3), and
N. For Qwen's 28 blocks these are 10, 19 and 28. The final feature includes the
existing final RMSNorm; intermediate features are residual-stream block outputs.
Each depth is standardized using its own training split. Thus this compares these
three feature extraction conventions, not depth independent of normalization.

Use identical question splits, retained answers, labels and eight prefix positions
per answer: question-only plus seven seeded uniform draws with replacement from
1 through T-1. Features precede the next generated action. A single causal forward
pass captures all three depths; only selected prefix vectors are saved. We do not
claim early-exit generation or reduced actor cost from earlier features.

At every depth fit the existing one-layer sigmoid, two-layer MLP, ten-layer ResNet,
raw linear head and direct ridge diagnostic. Match seeds, learning-rate grids,
maximum 200 epochs and stopping rules across depths. Tune on validation only.
Report test Brier, correctness-prediction accuracy, AUROC, calibration, question-only
predictions, fitting curves and costs. Paired layer-minus-final Brier intervals
bootstrap questions, average squared errors across head seeds, and are exploratory
and unadjusted. Seeds share the dataset. This is supervised value prediction,
not an online PPO or LSTD comparison; lower-depth TD fitting can follow if useful.

The output contains `layers.json`, a `report.txt`/`summary.json`, per-layer head
reports/checkpoints/predictions, sampled feature caches and `review.txt` examples.
Source trajectory hashes and generating-model hashes are checked. Resume only
with unchanged code, source data, model and fitting settings by passing the same
`--out`; completed cached stages are reused.

## 2. Frozen encoder with cumulative raw LSTD(0.99)

Freeze the base encoder for the whole run. For each valid action, let phi be the
raw final-normalized hidden vector with a constant intercept appended. Do not
re-standardize each batch: that would change coordinates underneath old matrices.
Within each trajectory, reset z=0, then accumulate:

```
z = phi + 0.99 * z
A += outer(z, phi - next_phi)
b += z * reward
w = solve(A + epsilon * I, b)
```

Gamma is 1. Only the last action receives Math-Verify's binary reward; terminal
next_phi is zero including the intercept. Eligibility traces do not cross answers.
A and b persist across updates and are **never divided by transition count**.
Epsilon defaults to **0.01**, is added once at solve (not once per batch), and
penalizes the intercept too, matching epsilon*I. `--epsilon 1` is algebraically
identical to initializing the regularized matrix with I and then accumulating
raw contributions. Relative to mean statistics the penalty is epsilon/M.
The previous online alpha=.01 multiplied a mean-statistics system; it is not the
same regularization strength as this raw-sum epsilon=.01.

Use float64 solves, residual/finite checks and no silent pseudoinverse, clipping or
regularization increases. Save weights, statistics and per-update effective
regularization/fit counts/value diagnostics. A numerical failure stops the run
with its log; tiny epsilon is not a guarantee of a stable nonsymmetric system.

Two permanent question folds are sealed from the full training question list.
Every value is predicted using statistics from the other fold, including all
historical batches. Fold membership must not change across updates or a question's
old rewards could leak into its own values. A fold with no observations initially
predicts zero and is explicitly flagged.

Compare `frozen_batch` (clear statistics every update) against
`frozen_cumulative` (never clear statistics). Both use the same frozen encoder,
raw coordinates, epsilon, folds, actor settings and budgets. This is a matched
accumulation ablation, not a direct rerun of the older standardized batch critic.
There is no answer buffer; history is stored as sufficient statistics. Old
policies generated the historical answers, so this is an **uncorrected mixture of
behavior policies**, not an exact on-policy value estimate. Only fresh answers
enter PPO actor updates; GAE lambda remains .95.

## 3. Periodic encoder refresh

Default interval: 10 updates. Before extracting values at updates 11, 21, ...,
copy the current PPO actor's backbone into the frozen encoder, including its final
normalization. Freeze it until the next refresh. All distributed ranks participate
in the FSDP full-state export; the vocabulary head is not copied.

**Reset A and b at every encoder refresh.** Old feature-space matrices cannot be
combined with new ones. This first experiment deliberately does not preserve all
history across refreshes. Retaining history would require saving tokens and
re-extracting all historical features/rebuilding the matrices. That more costly
variant is not implemented here.

Compare three conditions, with matched raw-sum settings:

| Condition | Encoder | Statistics |
|---|---|---|
| frozen_cumulative | Base throughout | Never reset |
| frozen_reset | Base throughout | Reset every 10 updates |
| refreshed_reset | Copy current actor every 10 updates | Reset at the same times |

The last two isolate encoder refresh from the effect of discarding older data.
Question folds remain fixed even after a reset. Encoder versions are checked
against trainer expectations before fitting. Refresh cost is included in elapsed
training and exposed in worker metrics. Within each interval, historical data are
still off-policy as the actor changes; refresh is not an off-policy correction.

## Running and interpreting

Commands are in [SETUP.MD](../SETUP.MD#critic-follow-up-studies). Run a short
refresh smoke first, since local CPU tests cannot validate cluster FSDP copying,
vLLM synchronization or GPU memory. The layer study is independent and can run
in another allocation. Full online defaults are 60 updates, 64 answers/update,
three seeds and two GPUs. Each method starts from the same base actor.

Online outputs include status, per-condition logs and checkpoints, full base/final
test answers, critic metrics and a paired report with accuracy and GPU-hours.
No mid-run actor/optimizer/statistics resume is supported; saved statistics are
audit artifacts, not a promise of resume equivalence. Use a fresh output directory
for an online rerun. Source datasets remain unchanged.

Choose conclusions using all seeds, calibration and value diagnostics, numerical
health, end-to-end cost and final answer accuracy. A better frozen-prefix Brier
score does not establish better PPO; cumulative history may add bias; refresh may
cost more than it gains. Existing test questions have been inspected, so these
are internal ablations, not an untouched benchmark.
