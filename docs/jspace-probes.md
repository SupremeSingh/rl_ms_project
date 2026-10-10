# J-space value probes: protocol and implementation

This is an **offline supervised representation study**, not a new PPO run. It
reuses the saved ordinary-completion MATH level 4/5 answers from the frozen base
Qwen2.5-Math-1.5B model. It neither generates answers nor changes source files.
No J-space performance results exist yet.

## Where we use Anthropic's code

The dependency is the official [jacobian-lens repository](https://github.com/anthropics/jacobian-lens),
pinned to `581d398613e5602a5af361e1c34d3a92ea82ba8e`. The runner rejects another
revision or a local unverified install. It calls:

1. `jlens.from_hf(hf, tokenizer, force_bos=False)` on our exact frozen base model.
2. `jlens.fit(...)` with the source block index 18 (completed block 19 of 28),
   target block index 27, 128-token calibration sequences, and the official
   exclusion of the first 16 and last positions. The official estimator sums
   derivative effects over valid current/future target positions, averages over
   source positions, then averages prompts. It is not a self-position-only Jacobian.
3. `JacobianLens.save/load` for the fixed float32 matrix. The upstream fitter
   checkpoints each calibration prompt and resumes interrupted fitting.

No model parameters are optimized. Autograd differentiates activations through
frozen transformer blocks. We use eager attention and float32 for fitting and
feature extraction, without changing our PPO environment. The model weights and
saved token IDs are unchanged, but the newer Transformers/float32 path can produce
small numerical differences from historical feature caches: **all baselines are
re-extracted and refitted in this same environment**.

Calibration defaults to 100 deterministically reserved eligible **training**
questions and their first saved continuation, truncated to 128 tokens. Choice is
independent of rewards and verifier success. These entire questions are excluded
from probe training; validation and test questions never fit the lens. This is a
math-domain calibration, not a reproduction of the paper's generic-web corpus.
Calibration IDs and text are saved. Too-short/skipped calibration samples fail
explicitly rather than silently reducing N. The smoke uses only two prompts and
is not an estimate of final lens quality.

## What our code adds: sparse reconstruction

The official package does **not** implement the paper's sparse decomposition.
`src/math_rl/jspace.py` implements an explicitly approximate nonnegative matching
pursuit extension. It is not the paper's gradient-pursuit implementation and is
not an exact projection onto J-space.

For the Qwen RMSNorm architecture, the vocabulary directions are the rows of
`D_raw = U diag(g) J`, where U is the unembedding and g the final RMSNorm gain.
The remaining RMS normalization divides all token readouts by the same positive
state-dependent scalar. Folding g in therefore preserves the official readout
directions; applying the final norm to the source hidden vector would be wrong.
Rows are unit-normalized for reconstruction. All vocabulary rows are retained,
including special tokens; a zero row cannot contribute positively.

For each causal hidden vector h, start with residual h and reconstruction zero.
Repeat at most 16 times: select the direction with largest positive residual
correlation, add its nonnegative least-squares line-search contribution, and
subtract that contribution from the residual. Repeated selections can add to the
same atom. The reconstruction h_J is a nonnegative combination of at most 16
vocabulary directions in the **original 1,536-dimensional coordinates**. It is
not a 16-dimensional bottleneck, a softmax, or merely J times h. The residual
h - h_J is saved as its own control. Neither component is assumed to capture all
of the model's reasoning.

The random control uses a fixed signed coordinate permutation of this dictionary.
It preserves atom count, row norms and all pairwise inner products while changing
alignment to activations. This is a random-orientation control, not a uniformly
random rotation and not evidence of causality.

## Matched inputs and heads

| Input | Purpose |
|---|---|
| Final normalized hidden vector | Historical baseline, refitted |
| Two-thirds block output | Same-layer full-information control |
| J-space reconstruction | Can verbalizable features predict outcomes alone? |
| J-space + final vector | Can the representation help a small predictor? |
| Two-thirds + final vectors | Control for access to both layers and larger input |
| Residual two-thirds minus J-space | What useful signal is discarded? |
| Random reconstruction | Control for sparse filtering itself |
| Random reconstruction + final | Matched combined-input control |

Default heads are one-layer logistic and two-layer MLP (width 256), with binary
cross-entropy training; direct ridge return regression is included as a diagnostic.
Use `--heads linear_logit mlp2 resnet10` to also fit the ten-layer residual head.
Ridge is supervised return regression, **not LSTD**. We do not need consecutive
state features for this experiment.

Every representation gets the same questions, answers, outcomes, eight prefix
positions, fitting seeds, maximum 200 epochs, LR grid and early stopping. The
normalization mean/scale come from training only. Model/epoch/LR selection uses
validation Brier only. Test scores never select a model. Three default head seeds
share one dataset and one lens; they are not independent lens/dataset replications.
Concatenated inputs double input width and increase parameters; the full two-layer
and random-plus-final controls match that width. Parameter counts and actual
fitting costs are reported instead of claiming equal capacity across every arm.

The reward-blind prefix rule is unchanged: one question-only prefix plus seven
uniform draws with replacement from 1 through T-1, using original question and
response indices. Features are read at prompt_length - 1 + L, before the next
action. All examples are causally masked; extraction checks a sampled full-answer
forward against a truncated-prefix forward. Realized answer length selects sample
locations but is not a critic input. Duplicated prefixes for very short answers
are allowed by the original rule. No final/terminal state is sampled.

## Eligibility, costs and interpretation

The source manifest, model/tokenizer assets, trajectory hashes, dependency
versions, code and calibration are recorded. Resume refuses changed inputs,
configuration, environment or code. Matched feature shards are atomic and hashed.
The strict reference rule is applied to **all splits**, and unparseable/timeout
answers remain excluded for every arm. Full reference-audit details and answer
status counts are saved. Existing labels are not repaired: this remains
conditional on retained verifier outcomes on repeatedly inspected questions.

Report Brier, correctness-prediction accuracy, cross-entropy, AUROC, calibration,
question-only/early-prefix metrics and paired question-bootstrap intervals.
The top-level report compares average per-seed losses, not an ensemble predictor.
Intervals are exploratory and unadjusted. Brier columns pool prefixes whereas
paired differences give each question equal weight. Inspect per-head budget-limit
warnings before treating a weak fit as a representational limitation.

Feature extraction and lens fitting can be costly: a lens prompt entails 1,536
output-coordinate derivatives, batched through the official fitter. Sparse
reconstruction searches the full vocabulary repeatedly. GPU runtime on our
cluster is not yet measured; do the smoke first. Lens stage logs show cost per
prompt; `question-*.json` records per-question extraction time. These timings do
not include model-loading/dictionary construction; cluster elapsed time includes
them. Invocation timings and GPU allocation are recorded separately. A cheap
probe does **not** imply cheap total J-space inference.

`responses.html` displays an outcome-independent small sample of full test
responses alongside the saved concept tokens, coefficients and prefix lengths.
`question-*.json` includes those readouts for every retained sampled answer;
`final/trajectories/` links to all full original responses. Interpret the displayed
words as approximate readouts, not verified explanations of the model's reasoning.

Only a useful, cost-justified offline result would motivate integrating this fixed
feature map into cumulative LSTD and PPO. A frozen base lens does not measure how
J-space changes with post-training; that would need a separate checkpoint study.
