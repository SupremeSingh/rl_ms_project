# Large-N full-readout critic comparison

This is a new **offline value-prediction experiment**, not a change to PPO. A
frozen Qwen2.5-Math-1.5B generates an answer. At a partial answer, each probe
predicts the probability that this continuation eventually receives a correct
Math-Verify label. The actor, tokenizer, lens and feature recipe remain fixed.

## What the preceding experiment established

On the repeatedly inspected MATH subset, complete J-lens concatenation obtained
Brier **0.18800** and correctness-prediction accuracy **72.1%**. Raw concatenation
obtained **0.19337 / 71.3%**, RMS-normalized concatenation **0.18908 / 72.4%**, and
the two-thirds layer **0.18406 / 73.2%**. Thus the direct readout is competitive
with layer concatenation, but has not beaten the single two-thirds layer. Its
advantage over the normalization control was inconclusive. This motivates a
larger test; it does not establish that J-space is the better critic.

## Data and independence

The primary source is [NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5),
restricted to non-synthetic `cn_k12` word problems marked valid by its metadata.
Eligibility requires one decimal-valued reference box consistent with the
dataset's answer field. Explicit image/diagram/proof cases and multiple-answer
requests are excluded conservatively. These checks are not a proof that the
reference is correct. Full reference solutions and an audit sample are saved.

We block question text and number-substitution templates appearing in either
official split of GSM8K or any subject/split of MATH. Within the new corpus, only
one question per such template is selected. Pinned dataset revisions and
deterministic hashes fix the split before generation. This does not eliminate
paraphrases, translated duplicates, or unknown model-pretraining overlap. It is
fresh for our experiments, not certified unseen during pretraining. We have
repeatedly inspected the old benchmark; that is different from demonstrating
that the frozen actor was overtrained on it.

| Split | Distinct question families | Answers per question | Purpose |
|---|---:|---:|---|
| Lens calibration | 512 | 1 | Fit official lenses without correctness labels |
| Screen | 256 | 4 | Check difficulty, parsing and response caps |
| Train | 10,000 | 8 | Fit probe weights |
| Validation | 1,000 | 8 | Select learning rate and stopping epoch |
| Test | 2,000 | 8 | Final locked comparison |

There are **104,000 main-study attempts**, plus 1,536 calibration/screen attempts.
The screen requires 20–80% all-attempt answer accuracy, at most 10% exclusions,
and at most 15% capped answers. Failure stops the dependent jobs before the large
generation. No individual question is retained or removed based on how often
the actor solves it. Insufficient eligible families also stops preparation;
there is no silent reduction in N or relaxation of eligibility.

Generation uses the existing completion prompt, temperature 1, top-p 1, a fixed
seed schedule and a **3,584-token** response limit, reserving 512 prompt tokens
within the model's native **4,096-token total context**. Both preparation and
generation check this limit; no context-extension override is used. The initial
4,096-answer-token configuration failed before generation; corrected protocol v3
respects the shared limit. No planning headers or staged
generation. Correct/incorrect verifier statuses enter fitting; parsing/runtime
failures remain saved and are reported as exclusions. All-attempt answer accuracy
counts these attempts as unsuccessful. Probe metrics are explicitly conditional
on retained answers. A capped but parseable answer retains its verifier label.

## Prefixes and features

Each retained answer contributes eight examples: `L=0` and seven independent
uniform draws with replacement from `1..T-1`, where T is its generated token
length. For T=1, all eight are L=0. This weights answers equally, not long answers
more heavily. Future length determines this sampling distribution but is never
a feature. The state is question plus the first L answer tokens, and the vector
comes from the last observed token, immediately before the next action.

Saved generation tokens are teacher-forced through the same frozen weights in a
separate float32 extraction job. Generation uses bfloat16; feature extraction is
therefore a reproducible higher-precision recomputation, not a capture of the
exact generation-time arithmetic. Hooks save **all 1,536 coordinates** at each
selected layer. A future-token perturbation check verifies causal indexing.
We save sampled-prefix vectors, not vectors for every generated token; a later
LSTD experiment would require every transition's features.

The reverse-geometric distances 1/2/4/8/16 select blocks **28/27/25/21/13**. The
single-layer baseline uses block **19** (two-thirds depth).

| Representation | Input dimensions | Construction |
|---|---:|---|
| Direct J-lens concatenation | 7,680 | Complete pre-softmax readout at five blocks |
| Raw hidden concatenation | 7,680 | Same five unnormalized block outputs |
| RMS hidden concatenation | 7,680 | Same blocks with per-state RMS normalization |
| Two-thirds hidden | 1,536 | Ordinary block-19 output |
| Final hidden | 1,536 | Final output after learned RMSNorm |

We call the pinned official [Anthropic Jacobian Lens](https://github.com/anthropics/jacobian-lens)
`jlens.fit` and `JacobianLens.transport`. Four source lenses transport to block
28; its own transport is identity. The estimator remains the upstream average
Jacobian, including its current/future target-position convention. Calibration
uses first continuations irrespective of rewards, with reward-blind truncated
full contexts up to 512 tokens. This finite calibration corpus and context cap
are limitations, particularly for late prefixes in 3,584-token answers.

For each transported state x after RMS normalization, full vocabulary scores
are `z = U diag(g) x`. Factor `B.T B = C C.T`, with `B = U diag(g)`, and store
`phi = C.T x`. All vocabulary directions contribute; no positive eigenvalue is
discarded. Inner products match the full logits, checked explicitly on examples.
This is our exact compact-coordinate implementation of the complete readout,
not upstream sparse decomposition, selected words, top-k probabilities or softmax.
It preserves the linear score geometry; it does not imply all feature recipes
contain the same information or that trained nonlinear probes must agree.

## Probe fitting and evaluation

Every representation uses the same examples and **three head types**:

- One affine layer with sigmoid: logistic regression, linear in log-odds.
- Two-layer MLP: 256 hidden units and GELU.
- Ten-layer residual MLP: width 256, four two-layer residual blocks between
  input and output layers.

All train with binary cross-entropy, AdamW, weight decay 1e-4, batch size 512,
learning rates 0.0003/0.001, and seeds 42/43/44. Each trial allows 120 epochs,
at least 30 before early stopping and patience 20. Initial output equals the
training success prior. Validation Brier chooses epoch and learning rate per
seed; every trial and curve is retained. A best epoch at the budget limit is
flagged. Representations are centered and given one isotropic scalar scale from
training data only. Equal-width heads have equal parameter counts across the
three concatenation arms, but the single-layer inputs have fewer parameters.

In addition, every representation receives **ridge and LSTD(0.99)**, giving
25 representation/head combinations in total. Both linear methods use every
pre-action token state from the same retained training answers. A separate
teacher-forced pass streams the full trajectories into resumable matrix sums;
we do not interpret gaps between random prefixes as one-step transitions.

With fixed train-only centering/scaling and an appended intercept, LSTD uses
`z_t = 0.99 z_(t-1) + phi_t`, `A += z_t (phi_t - phi_(t+1)).T`, and
`b += z_t r_t`. Gamma is 1. Intermediate rewards are zero; the last action
receives the binary verifier reward, with terminal next features zeroed,
including the intercept. Traces reset between answers; matrices accumulate
across all training answers and shards. Ridge accumulates `phi phi.T` and
`phi R` over exactly the same states using the final return R.

Both solve **raw sums plus 0.01 I**, including the intercept, without dividing
by the number of transitions, sigmoid or clipping. This follows the cumulative
fitting design; it is not a validation-tuned ridge sweep. Normalization is fixed
before accumulation, so feature coordinates never drift. Validation diagnostics
are reported, and linear systems must pass residual checks. Test questions do
not enter any fitted matrices, so online question cross-fitting is unnecessary
here. There is no policy update or historical-policy mixture in this experiment.

Ridge and LSTD have matched all-token weighting; neural probes use eight sampled
prefixes per answer. Consequently, their relative results also reflect different
training-state weighting and objectives. Every head evaluates the same saved test
prefixes. Linear fits are deterministic, not three independent training seeds.
Raw linear values can exceed [0,1]: report that fraction and raw Brier; clipped
cross-entropy is only a secondary diagnostic.

All 15 neural selections and 10 linear fits must finish before reporting test scores. Report:
accuracy at 0.5, Brier, cross-entropy, AUROC, calibration, constant-prior baseline,
counts, exclusions and truncation. Accuracy here means **predicting whether an
answer succeeds**, not generating a more accurate answer. Head-seed metrics are
averaged, not predictions ensembled for headline accuracy.

The two prespecified primary tests compare direct J-lens against two-thirds
hidden, once using ridge and once using LSTD(0.99), on paired Brier differences.
Question-cluster bootstraps keep all prefixes/answers from a question together;
neural-head comparisons average losses over head seeds first.
Use 97.5% intervals for these two contrasts (Bonferroni family coverage 95%);
other head, accuracy and normalization contrasts are descriptive. Thousands of
prefixes from the same question are not thousands of independent questions.
Three head seeds do not measure independent data or lens-fitting variability.

This experiment can support a claim about frozen-policy value predictability
on this selected dataset. It cannot demonstrate causal reasoning, universal
J-space superiority, a cheaper online critic, or improved PPO. The new LSTD
comparison uses actual transitions, but remains offline. It can identify a
candidate feature map for a later matched online LSTD/PPO comparison.

## Execution and cost

Separate Slurm jobs handle preparation, screening, official lens fitting,
four generation shards, four feature-extraction shards, five packing arms,
15 neural fitting tasks, four full-transition accumulation shards, five linear
solver tasks and reporting. Completed question batches and probe trials are
reused; official lens fitting checkpoints after every prompt. An interrupted
probe trial restarts that trial, not the entire study. Do not run two writers
against the same output directory. Linear matrix checkpoints are saved every
64 questions and at shard completion; a restart may replay only the unsaved tail.
Eligibility traces are never resumed across answer boundaries.

Large feature arrays use memory-mapped float32 storage and one arm per fitting
task. Before exclusions, packed features total about **87 GB**, raw sampled
blocks about **31 GB**, plus roughly **12 GB** of linear matrix checkpoints,
temporary checkpoint copies, trajectories and dataset downloads.
Allow at least **200 GB free**, preferably more. Each worker requests 96 GB RAM;
generation/extraction arrays allow four GPUs at once, fitting three. Lens fitting
can run alongside generation. Stage limits are ceilings, not runtime estimates;
the first full run will establish actual costs. Timing files retain completed
and caught-failure invocations; hard kills require reconciliation with Slurm
accounting. The report separates recorded costs from performance; this is not an equal-total-cost
comparison, since J-lens calibration has extra cost. All-token matrix accumulation
adds substantial work: a 7,680-dimensional system has about 25 times the matrix
storage and 125 times the dense solve work of a 1,536-dimensional system. This
comparison does not assume multi-layer J-lens is computationally cheaper.

[Commands and recovery](../SETUP.MD#large-n-full-readout-j-lens-study).

## Follow-up: smaller linear critics

The [PCA compression study](jlens-compression.md) reuses the completed all-token
statistics to compare full J-lens, raw hidden and normalized hidden concatenations
at 768, 1,536 and 3,072 dimensions, with Ridge and LSTD(0.99). No new generation or
encoder pass is needed. Uncompressed and single-layer controls are retained;
validation chooses dimensions before test evaluation.
