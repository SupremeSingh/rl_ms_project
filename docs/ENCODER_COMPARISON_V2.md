# Expanded mean-pooling comparison: agreed design and proposed data budget

Status: separate workflow implemented and CPU-tested; cluster smoke pending. This
records the user's accepted A–D alternatives and the expanded data scale.
The old `tools/encoder/submit_study.py` still launches the smaller last-token pilot
and must not be presented as launching this comparison.

## Representation alternatives

| Arm | Input readout | Compression | Output width |
|---|---|---|---:|
| A | Mean over available tokens at final layer | None | 1,536 |
| B | Mean over available tokens around two-thirds depth | None | 1,536 |
| C | Mean over available tokens at each of five fixed layers, concatenated | Fresh SLICE | 1,536 |
| D | Same five-layer mean-pooled concatenation | Fresh SLICE | 768 |

No ten-layer compressor. Preserve the existing last-token pipeline as a separate
historical/matched reference, not a fifth interpretation of A–D. Apply A–D to the
same causal and adapted backbones. The user selected one complete LLM2Vec variant
(MNTP then SimCSE); omit standalone bidirectional and MNTP-only arms for this run.
This limits attribution: a combined gain/loss cannot identify the responsible stage.

## Pooling and normalization

Use the historical RMS readout transform (not learned LayerNorm), generalized to
one pooled vector per state/layer: first mean over tokens in FP32, then divide by
`sqrt(mean(vector**2) + model.config.rms_norm_eps)`. RMS-after-pooling and
RMS-before-pooling are different transforms; the primary recipe is the former.

For the 28-block backbone, A uses block 28, B block 19 (`ceil(2*28/3)`), and C/D
use blocks `[28,27,25,21,13]`, zero-based `[27,26,24,20,12]`. At the final block
use the backbone's native final-normalized output, as used by official LLM2Vec
pooling; earlier blocks use their block outputs. This convention applies to every
backbone and every occurrence of the final layer in A/C/D. The extra pooled RMS
is explicit. Do not call these pre-final-norm block means.

Pool all real, already-available question/prompt and answer-prefix tokens,
including original formatting/special tokens, excluding padding. Never include
the current chunk, future answer tokens, reference solutions or rewards. C/D
concatenate the five separately pooled/RMS-normalized vectors, then apply the
existing training-only coordinate centering and single scalar scaling before
SLICE. Freeze those statistics. Do not add normalization after SLICE. A/B use
their RMS-pooled vectors directly with Ridge. Preserve one raw-sum head, epsilon
.01 and an intercept. Historical last-token features remain a separately labelled
reference because the final-norm convention/readout are not identical.

RMS normalization is an established control in this repository, not evidence that
RMS-after-mean-pooling has already demonstrated an improvement. Experiment 09
studied an additional SLICE output LayerNorm, a different transform.

## Adaptation context and distributed SimCSE

Replace isolated 128-token adaptation windows with complete question + available
answer-prefix examples from training questions. Use the same original prompt
format as extraction and sample across answer progress. Do not populate the corpus
with every overlapping prefix: cap/balance examples per question and progress bin,
deduplicate exact strings and report same-question near-duplicate exposure.
Neither outcome labels nor reference solutions enter adaptation text.

Before fixing a context cap, profile actual token lengths (p50/p90/p95/p99/max)
for training/validation prefixes and report coverage at 512/1024/2048/2560/4096.
Keep test data out of this decision. Profile the expanded corpus too; the small
pilot's length distribution is only a preliminary guide. Select a cap from those
measurements and memory trials, not an assumed 128 or 1024. Inputs exceeding the
cap must be explicitly accounted for; do not silently truncate, remove their
question, or discard the prompt. Choose sufficient capacity or separately report
the affected stratum with a declared handling rule before training.

Target four GPUs, eight examples/device, gathered contrastive batch 32,
accumulation 1. Verify the pinned official loss gathers embeddings and gradients
across ranks. Test at the selected length before finalizing the allocation.
Padding/length bucketing needs its own gate: the previous unpadded extraction
sanity did not approve arbitrary padded execution. MNTP uses the official data
pipeline on the same training-only prefix text; report its actual grouping,
truncation, masking and token exposure separately from SimCSE. Longer text,
distributed training and changed batches all require a new smoke. No upstream
trainer or loss patches are authorized by this specification.

## Data scale

The repository's earlier offline preparation recipe (`src/math_rl/data.py`) is
configured for 13,000 questions, eight answers/question, and 120 SLICE epochs.
The package question splitter (`slice_critic/data.py`) allocates 60/20/20 percent.
These are verified configuration values, not a claim that a particular cluster
manifest has already been checked for this expanded comparison.

Target the same scale, using existing offline Numina question assignments if the
saved provenance confirms these counts and boundaries:

| Split | Questions | Answers/question | Shared answers |
|---|---:|---:|---:|
| Training | 7,800 | 8 | 62,400 |
| Validation | 2,600 | 8 | 20,800 |
| Test | 2,600 | 8 | 20,800 |
| Total | 13,000 | 8 | 104,000 |

Relative to the completed encoder pilot: 15.6 times as many training questions,
31.2 times as many training answers, and 52 times as many test questions. The
eight responses per question are correlated; confidence intervals cluster by
question rather than treating responses or prefixes as independent questions.

Do not repartition the 3,000/500/2,000 online pool to reach these counts. Do not
move any historical validation/test family into training. The offline 13,000
question corpus and online pool were designed to be disjoint. Check saved IDs,
family assignments, hashes, actor exposure and actual counts before adopting this
budget. If the existing source does not provide these counts, fail explicitly and
resolve a family-disjoint training-only extension; do not silently relax filters.
Previously inspected test data remains reused evaluation data, not a fresh test.

Generate eight fresh verified answers per selected question from a predeclared
frozen actor and share exact IDs, labels, prefixes and sampling parameters across
all A–D/backbone combinations. The current fixed-policy source has only 500/100/50
questions and cannot be expanded merely by changing the encoder study limits.
Reference solutions must not enter encoder adaptation text or critic inputs.

Encoder adaptation uses training text only; report unique texts/tokens, length
coverage, batch-negative pool and optimization exposure explicitly.

## Adequacy and uncertainty

Primary training, validation and outcome-evaluation states use a fixed schedule
of pre-action answer positions `0,32,64,...` while the trajectory is alive. This
schedule uses generated-token count only, not future text, total answer length,
reward, or semantic-boundary detection. Semantic MATH chunks remain a secondary
control with identical physical prefix truncation. Do not choose primary states
by fractions of completed answer length. For the four-state MC budget, seal a
fixed-offset schedule and explicit handling of answers ending before an offset;
never substitute fractions of the realized future length or duplicate states.

For C/D, use a larger predeclared SLICE training budget, with 120 epochs as the
initial ceiling matching the older configured recipe. At a fixed predeclared
checkpoint schedule (e.g. every five epochs), freeze each candidate compressor,
fit a fresh Ridge head using training states only, and score equal-question raw
Brier on the fixed validation bank. Select the compressor and its matched head
by this full-pipeline validation score. Never accumulate matrices across changing
compressors, and never use validation states to fit Ridge. Retain sigmoid-Brier
curves as a diagnostic, not the selection objective. Report optimizer updates and
train/validation curves; epochs
alone do not demonstrate convergence. With up to eight sampled prefixes/answer,
the training cap is 499,200 sampled prefixes, versus 16,000 in the pilot. Ridge
continues to use all designated training chunk states, one head and raw-sum
epsilon .01. Changing training-state counts changes effective regularization;
report that rather than silently renormalizing the accumulated system.

Target five paired compressor seeds for C/D and at least three independent
encoder-adaptation seeds. A/B have no randomly trained compressor; repeating a
deterministic Ridge solve does not create independent seeds. Reuse the same
question/rollout bank across paired feature comparisons. More adaptation seeds
are not more actor seeds.

Retain equal-question raw Brier as the outcome-prediction metric. Add an
evaluation-only MC bank on 1,000 test questions, four prespecified available
states/question, 128 continuations/state: up to 512,000 shared reference
continuations for one frozen actor, reused across all representations. Include
the prompt state; state selection must not depend on reward or method. Use
available-prefix/fixed-token selection for the primary MC bank and report short
trajectory handling and actual state counts. Keep separate bank/continuation
seeds; MC results never train the compressor or head.

Report both accuracy of probability estimates (MAE/RMSE to MC references) and
sampled-return Brier, with paired question-clustered intervals and predeclared
primary contrasts. Do not treat the raw continuation count as a measure of
independent evidence or ignore finite-MC uncertainty.

This makes the proposed offline comparison much larger than the pilot and
comparable in data scale to the older offline recipe. It does not equal the
earlier three/five-seed actor-training studies in scope: a fixed actor cannot
establish robustness across policies or PPO improvements. A confirmatory
multi-actor or PPO study remains a separate next stage.
