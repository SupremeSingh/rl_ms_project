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

Lens fitting and extraction explicitly disable TF32 matrix multiplication. The
J-space launcher clears `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` inside the container
before Python starts, since the container image can force it on independently
of host settings. Other training launchers are unchanged. The
extraction causal guard replaces every token after a sampled prefix while keeping
the sequence length unchanged, then checks the two-thirds and final features at
that prefix. This avoids confusing sequence-shape numerical differences with
future-token leakage. Its tolerances, precision settings and observed differences
are saved in `causality-check.json`; a failed check still stops extraction.

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

## Reconstruction follow-up

`refine_jspace_probes.py` reuses the official lens and cached hidden vectors from
a completed run. It changes the sparse solver and budget while holding the lens,
layer, labels and selected examples fixed. The bounded defaults use up to 400/100
train/validation questions and four retained answers per question; test uses all
available questions with the same answer limit. Question selection hashes IDs,
independent of outcomes. All original cached prefixes of chosen answers remain.

The original matching method is compared with our projected nonnegative
gradient-pursuit variant. At each iteration the latter selects a previously
unselected atom with maximal residual correlation, adds its nonnegative line-search
coefficient, then takes four joint coefficient refinement steps on the active
dictionary. Refinement uses a projected gradient and backtracking to prevent an
increase in squared reconstruction error. Coefficients may decrease or reach zero.
This is explicitly our implementation, not an upstream API, exact nonnegative
least-squares solve, or bitwise reproduction of the paper's gradient pursuit.
Budgets 16, 25 and 64 count selection iterations; matching may repeat atoms and
gradient pursuit may leave coefficients at zero. Report actual active distinct atoms.

All six candidates and matched random controls are reconstructed on train/validation.
Ridge uses train-only standardization and the existing common alpha grid. A single
J-only validation Brier winner selects solver and budget; `selection.json` is saved
before loading test features. Only that winner, its final-feature concatenation and
matched random controls are tested alongside newly fitted full-feature baselines.
All validation results remain visible. Test questions were previously inspected:
locking this selection prevents further within-run test tuning, not historical bias.

Diagnostics include uncentered residual energy fractions (not conventional explained
variance), active atom counts, runtime shared across sparsity checkpoints, and
readouts linked to full saved answers. The previous probe convergence flags are
copied to an audit without selecting on their test scores. Ridge avoids an
iterative-head optimization-budget confound in this follow-up. Reconstruction
residuals need not be orthogonal to the selected atoms or to a global J-space.

This economical first step does not refit the lens or search new layers. If needed,
a subsequent validation-only experiment should compare calibration context coverage
and nearby source layers, with matched controls and new extraction provenance.

## Geometric multi-layer extension

`geometric_jspace_probes.py` uses reverse ranks b**j starting at 1, stopping at the
block count. For 28 blocks and b=2, zero-based indices are 27/26/24/20/12; outputs
are concatenated in that final-to-earlier order. The last selected layer is not
necessarily the first network block. Base 3 is supported but is a separate fixed
configuration, not silently selected using test results.

The official fitter estimates the earlier source Jacobians jointly, targeting the
last block output. Upstream requires each source to precede the target. At the last
block itself, the derivative of target residual with respect to that residual is I,
with zero influence on other positions at that same depth. Consequently its dictionary
is the normalized row directions of U diag(final RMSNorm gain). No bogus final-layer
fit is requested. All selected features, including final, are pre-final-norm residuals;
this differs from the normalized final baseline in the original single-layer study.

This implementation holds calibration fixed to the original recipe (reserved train
questions, first answer, first 128 tokens). It tests aggregation, not changed context
coverage. Questions and attempts are bounded before reward inspection. One causal
forward pass captures all selected block outputs plus the ordinary two-thirds baseline.
Each block has its own sparse reconstruction and random-orientation control.
The same-shape future-token perturbation check covers every captured block.

All heads are ridge return regressors. Concatenation keeps five complete 1,536-vectors
(7,680 coordinates), train-only standardized independently by coordinate. Ordinary
hidden concatenation and random-dictionary concatenation have matching dimensions.
Per-layer heads use the same data and alpha grid; validation selects the best
individual J-space and hidden layer. These and the ordinary two-thirds baseline
are compared with the fixed concatenation. Test data are loaded by fitting only
after validation choices are saved, although extraction necessarily processes all
splits and the underlying test questions have been inspected in earlier studies.

Pursuit defaults to our nonnegative gradient variant, 25 atoms per layer, fixed
before the run. It is not automatically imported as a winner from an unfinished
refinement study. Costs include multi-source lens fitting, feature extraction and
all ridge fits. In a later LSTD integration, fivefold feature dimension entails
approximately 25-fold dense matrix storage and potentially 125-fold dense solve
work; this experiment does not claim that aggregation remains computationally cheap.
