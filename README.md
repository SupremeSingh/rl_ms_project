# Math RL: Can PPO Use a Cheaper Critic?

**Can an LLM learn with PPO without training a second large transformer to judge its progress?**
We fit a linear value function on compact features from a frozen LLM. Instead of
backpropagating through a large critic, we accumulate matrices and solve for its
weights. PPO still updates the actor normally.

The math stage is complete for now. Our chosen representation is a **frozen,
sparsity-penalized 768-dimensional bottleneck** over several hidden layers.
**Cumulative Ridge is our selected default**: it reached **74.60% Numina accuracy**
versus 73.52% for conventional PPO and 72.75% for GRPO. LSTD remains an alternative
for further temporal-learning research. Next: cheap value-guided
search for Lean proofs of properties of small programs.

[Setup](SETUP.MD#initial-setup) · [Online protocol](docs/numina-online.md) ·
[Experiment history](HISTORY.md) · [Educational articles](docs/articles/README.md)

Results updated **24 October 2026**. Design choices below are not universal superiority claims.

## Start with one answer

“How many servings can I make from 9 avocados if each serving needs 3?”

The **actor** writes an answer token by token. Its state is the question and the
answer prefix; its action is the next token. Once it finishes, **Math-Verify**
checks the final answer: correct earns 1, otherwise 0. It does not certify each
reasoning step.

After “Each serving needs 3, so we divide…”, the **critic** estimates the expected
final reward under continuation of the policy. This is a prediction of eventual
success, not a judgment that the current sentence is mathematically valid.
**GAE** combines these values with rewards to estimate advantages; **PPO** uses
those advantages to adjust the actor’s token probabilities.

The expensive conventional option trains a transformer critic. Our alternative
uses the LLM’s existing representations and fits a small linear head.

## The final pieces

![Final architecture: frozen sparse-768 features, cumulative cross-fitted Ridge or LSTD, and fresh-data PPO actor updates](docs/figures/final-architecture.png)

[Scalable vector diagram](docs/figures/final-architecture.svg). The two matrix
systems are question folds, not separate Ridge and LSTD branches: choose one
fitting method per run. The frozen encoder is an additional forward pass. The
figure illustrates the earlier lambda-0.99 variant; the latest comparison below
uses lambda 0.9.

- **Representation:** Qwen’s 1.5B math model has 28 blocks and 1,536-dimensional
  hidden states. We concatenate blocks **28, 27, 25, 21, 13** (one-based),
  RMS-normalize each, and use training-only centering/scaling: 7,680 inputs.
- **Compression:** a learned row-normalized linear projection plus ReLU produces
  **768 features**. A temporary correctness predictor trains it with binary
  cross-entropy. The sparse version adds `0.01 × mean(feature activations)`;
  the plain version omits this penalty. Both can produce zeros.
- **Online fitting:** freeze the encoder and bottleneck, discard the temporary
  predictor, and start fresh linear value heads. Features use only the prefix
  before each action. The actor changes; the feature coordinates do not.
- **Tools:** VERL runs distributed PPO training; vLLM generates rollouts;
  Math-Verify scores final answers; our code supplies features, critic fitting,
  cross-fitting, audits and comparisons. Slurm schedules cluster experiments.

This is supervised nonlinear compression followed by a linear value function—not
an orthogonal projection, a rate-reduction implementation, or a proven “noise
filter”. Its useful information is learned from outcome labels.

## Accumulate evidence, not weights

Let `x` be the 768 features plus an intercept, so the critic has **769 weights**.
Each question permanently belongs to one of two folds. Its trajectories update
one matrix system; its values come from the other. This prevents its own rewards
from directly fitting the baseline used for its advantages.

```text
Freeze the base encoder and learned bottleneck.
Initialize A and b to zero for each question fold.

For each PPO update:
    Generate fresh answers with the current actor; score their outcomes.
    Extract frozen features for every pre-action prefix.

    For each answer, update its question fold's statistics:
        Ridge: for each prefix x, add outer(x, x) to A and x * final_reward to b.
        LSTD: reset eligibility trace z = 0 at the start of the answer.
              For each transition (x, reward, x_next):
                  z = gamma * critic_lambda * z + x
                  A += outer(z, x - gamma * x_next)
                  b += z * reward
              Use zero x_next at termination, including its intercept.

    Solve (A + epsilon * I) w = b separately for each fold.
    Predict each question's values with the opposite fold's weights.
    Compute GAE and update the actor with PPO on fresh answers only.
    Keep A and b for the next update; keep the encoder and bottleneck frozen.
```

We use **gamma = 1**, **GAE lambda = 0.95**, and **epsilon = 0.01**. The
latest LSTD comparison uses **critic lambda = 0.9**; earlier experiments used 0.99. Statistics are raw sums, never divided by dataset size;
the fixed regularizer therefore weakens relative to accumulating evidence.
No answer buffer is retained for critic replay. Historical policy data remain
in the matrices without off-policy correction: freezing features and cross-fitting
do not make that mixture an exact evaluation of the latest policy.

## Latest result: six methods, five seeds

The completed NuminaMath-1.5 `cn_k12` comparison used **3,000 training, 500
validation and 2,000 test questions**, with question families separated from the
representation study and number-template matches to GSM8K/MATH excluded. This
is not semantic or pretraining decontamination. The test split had already been
inspected in earlier experiments.

**Six methods × five fresh actor seeds × 120 updates** produced 30 complete runs.
Each update generated 16 prompts × four answers: **7,680 training answers per
run**. Every method started from the same base actor, scored **66.70%** on the
same greedy test, and used the same verifier and 2,048-token response cap.
Rollout counts were matched; realized tokens and compute could differ. PPO's
`higher_lr` critic profile was selected on validation at 60 updates and reused
for the 120-update comparison without test-based retuning. The bottleneck was
also frozen after validation-only selection.

The LSTD arms below use **critic lambda 0.9**, with **GAE lambda 0.95** and
**gamma 1**. They must not be confused with the earlier lambda-0.99 studies.
These are final **answer accuracies after policy training**, not critic accuracy.
SD describes variability across the five actor seeds.

| Method | Accuracy ± seed SD | Gain over base | Training GPUh/run | Accounted GPUh/run |
|---|---:|---:|---:|---:|
| **Ridge, sparse 768** | 74.60% ± 0.59 pp | +7.90 pp | 10.35 | 12.19 |
| Conventional transformer-critic PPO | 73.52% ± 2.03 pp | +6.82 pp | 12.51 | 14.32 |
| GRPO | 72.75% ± 0.79 pp | +6.05 pp | 9.37 | 11.36 |
| Ridge, two-thirds layer | 72.54% ± 1.02 pp | +5.84 pp | 10.15 | 12.03 |
| LSTD(0.9), two-thirds layer | 72.40% ± 0.32 pp | +5.70 pp | 9.61 | 11.43 |
| LSTD(0.9), sparse 768 | 70.86% ± 1.56 pp | +4.16 pp | 10.33 | 12.17 |

All six methods improved over the base in every seed. Sparse-768 Ridge had the
highest mean accuracy. Relative to conventional PPO, it used **17.3% less measured
training GPU time** (14.8% less accounted GPU time). Relative to GRPO it used
**10.4% more training GPU time**. Allocated GPU-hours include generation and
feature extraction; they are not FLOPs or hardware-utilization measurements.

Training costs include validation, checkpoints and feature/critic work. Accounted
costs also include initialization and initial/final test evaluation. Prior
representation learning is additional, and the reused PPO calibration cost
**19.59 GPUh**. These are per-run costs, not total project costs.

### Which differences are supported?

Eight primary comparisons were declared. The table shows **paired accuracy
changes in percentage points** and approximate seed/question bootstrap intervals
with Bonferroni adjustment over all eight comparisons. Intervals remain
exploratory because the question split was previously inspected and there are
only five training seeds.

| Comparison | Change (pp) | Adjusted interval (pp) |
|---|---:|---:|
| Sparse Ridge − two-thirds Ridge | +2.06 | [+0.59, +3.53] |
| Sparse LSTD(0.9) − two-thirds LSTD(0.9) | -1.54 | [-4.29, +0.70] |
| Sparse LSTD(0.9) − sparse Ridge | -3.74 | [-6.26, -1.73] |
| Two-thirds LSTD(0.9) − two-thirds Ridge | -0.14 | [-2.04, +1.78] |
| Sparse Ridge − conventional PPO | +1.08 | [-1.22, +3.56] |
| Sparse LSTD(0.9) − conventional PPO | -2.66 | [-5.62, +0.53] |
| Sparse Ridge − GRPO | +1.85 | [+0.08, +3.50] |
| Sparse LSTD(0.9) − GRPO | -1.89 | [-4.37, +0.63] |

**The learned sparse-768 representation helped Ridge:** its +2.06 pp difference
against two-thirds Ridge was positive in every seed, with an adjusted interval
above zero. Sparse Ridge also exceeded GRPO in all five seeds, although that
adjusted interval only narrowly excludes zero. Its +1.08 pp advantage over
conventional PPO remains uncertain. The representation comparison changes both
multi-layer compression and the feature transform; it does not isolate the
sparsity penalty as the cause.

**LSTD(0.9) did not improve on Ridge here.** With sparse features it lost 3.74 pp,
with an adjusted interval below zero. Two-thirds-layer LSTD and Ridge remained
unresolved. This result neither tests lambda 0.99 at the same budget nor proves
that Ridge/LSTD should be selected by horizon length alone. The 60-update,
three-seed pilot is preserved in [HISTORY.md](HISTORY.md); it is not a matched
learning curve for this new seed set.

### Diagnostics and limits

All 30 runs and reporting completed successfully. Conventional PPO passed the
completion, finite-value, nonzero-update and zero-initial-value checks in all
five seeds. In seed **302**, its mean training-target MSE increased from
**0.004713 to 0.004867** (about 3.3%); the other four seeds improved. All seeds
remain in the comparison. This is a training-target-fit warning, not a held-out
critic metric; seed 302 actually achieved PPO's highest answer accuracy (76.75%).
The warning does not by itself explain sparse Ridge's higher overall mean.

Final-answer parse failures ranged from **0.10–0.85%** and truncation from
**1.40–3.75%** across method/seed runs. These diagnostics are not an exhaustive
reward audit. Sparse Ridge generated about **635–656 tokens** per test answer,
versus **423–584** for conventional PPO: cheaper training does not establish
cheaper inference. Accumulated statistics still mix historical policies without
off-policy correction. Five seeds share one selected filter per arm, so they do
not measure representation-training variability.

A compact transcription of the supplied results, including all per-seed
accuracies, primary comparisons and PPO health checks, is saved in
[docs/results/numina-six-method.json](docs/results/numina-six-method.json).
Raw predictions and checkpoints remain in the cluster output directory.

Earlier, a five-seed MATH comparison found cumulative two-thirds-layer
LSTD(0.99) at **67.49% versus GRPO's 59.31%** after a conservative reference audit
retained 291 questions. Its descriptive paired interval was [+3.85, +12.92] pp.
That used a different task and representation; it is historical evidence for the
approach, not evidence about sparse-768 LSTD(0.9).

## Decision: sparse 768, with two fitting options

We choose **frozen sparse 768 plus cumulative Ridge** for the next phase. It had
the highest observed accuracy and reduced measured training cost versus our
conventional PPO baseline. The adjusted accuracy difference versus PPO remains
uncertain; it is a promising engineering choice rather than universal superiority.

The earlier plain-768 ablation found no resolved sparse-versus-plain advantage;
plain 768 was not included in the latest six-method study. Keep that distinction
when attributing gains to sparsity.

The efficiency claim is about **compression**: compared with a 1,536-feature head,
its matrix needs approximately **one-quarter the storage**. Sparse and plain 768
have identical dimensions and use dense operations. Sparse LSTD actually took
slightly more training time than plain LSTD in the earlier 60-update pilot
(**5.32 vs 5.21 GPUh**); we do **not** claim that sparsity itself has made the implementation faster.

| Situation | Starting choice | Why / what to check |
|---|---|---|
| Complete, reasonably short trajectories with reliable final rewards | **Ridge** | Direct regression on observed returns; simple and our highest-scoring Numina candidate. |
| Longer trajectories where temporal structure may help credit assignment | **LSTD(lambda)** | Uses adjacent states and traces; test against Ridge because bootstrapping can add error. |
| Sparse terminal reward and complete long trajectories | **Compare both** | Long horizon alone does not make LSTD better; near-one lambda previously worked better than short traces. |
| Incomplete search branches or a changed search policy | **Redesign targets and boundaries first** | An unfinished proof is not automatically a failure; historical statistics may estimate the wrong continuation policy. |

“Ridge for short horizons, LSTD for long horizons” is our **working design
heuristic**, not a result established by these experiments. For now, use Ridge
as the practical default. Retain LSTD as an experimental alternative; the latest
lambda-0.9 results do not establish a long-horizon advantage.

## Setup and reproduce

This repository targets the existing Duke Slurm/container environment. Start with
[installation and model preparation](SETUP.MD#initial-setup); the online protocol
requires completed representation-study artifacts and its passed smoke test.

```bash
cd /usr/xtmp/ms785/rl_ms_project
source scripts/cluster_env.sh
# Read the completed six-method comparison; no new GPU job is needed.
cat outputs/numina-online-six-lambda09-v3/report.txt
cat outputs/numina-online-six-lambda09-v3/summary.json
```

See the [six-method study commands](docs/numina-online.md#six-method-numina-follow-up-lstd09-ridge-grpo-and-ppo)
for reproduction, [bottleneck protocol](docs/sparse-critic-study.md) for feature
training, and [SETUP.MD](SETUP.MD) for full dependencies, audits and resumption.
Reports, manifests and per-answer audits are saved under the study output directory;
large model/data artifacts must be available separately. Do not modify shared
training code while jobs are running.

## Next: spend search compute where it matters

We are pausing math-dataset experiments and moving toward a small **Lean-based
proof-search system for software properties**. An LLM proposes tactics; Lean
checks them; a cheap critic ranks the resulting proof states. The encoder,
representation and reward targets must be fitted and evaluated for that domain—
math-answer success does not automatically transfer to proof search.

Begin with a fixed tactic-generation budget and compare policy-only search against
Ridge- and LSTD-guided search. Measure verified proofs, latency, generated tokens
and total compute. Learn from search outcomes before adding elaborate tree search
or policy training. Lean remains the correctness authority; the critic only guides
which branch receives effort.

Our vision is to reuse rich LLM representations for inexpensive value learning.
The next question is whether a small critic can buy more verified proofs per unit of compute.


For a poster focused on correctness prediction rather than RL, the
[offline evidence export](docs/probe-poster-evidence.md) collects saved probe
metrics, split counts, prefix-position diagnostics and size/timing accounting.
It requires no new generation or training.
