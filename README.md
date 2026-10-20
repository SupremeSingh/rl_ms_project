# Math RL: Can PPO Use a Cheaper Critic?

**Can an LLM learn with PPO without training a second large transformer to judge its progress?**
We fit a linear value function on compact features from a frozen LLM. Instead of
backpropagating through a large critic, we accumulate matrices and solve for its
weights. PPO still updates the actor normally.

The main math stage is complete; one final comparison extension is prepared. Our chosen representation is a **frozen,
sparsity-penalized 768-dimensional bottleneck** over several hidden layers. Ridge
is our practical starting point for complete, terminal-reward trajectories;
LSTD(0.99) remains the temporal-learning alternative. Next: cheap value-guided
search for Lean proofs of properties of small programs.

[Setup](SETUP.MD#initial-setup) · [Online protocol](docs/numina-online.md) ·
[Experiment history](HISTORY.md) · [Educational articles](docs/articles/README.md)

Results updated **17 October 2026**. Design choices below are not universal superiority claims.

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
fitting method per run. The frozen encoder is an additional forward pass.

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

We use **gamma = 1**, **critic lambda = 0.99** for LSTD, **GAE lambda = 0.95**,
and **epsilon = 0.01**. Statistics are raw sums, never divided by dataset size;
the fixed regularizer therefore weakens relative to accumulating evidence.
No answer buffer is retained for critic replay. Historical policy data remain
in the matrices without off-policy correction: freezing features and cross-fitting
do not make that mixture an exact evaluation of the latest policy.

## What the completed comparison says

The focused online study used **3,000 training, 500 validation and 2,000 test
questions** from eligible NuminaMath-1.5 `cn_k12` families. It excluded
number-template overlaps with the earlier representation study, GSM8K and MATH.
This is not semantic or pretraining decontamination.

Five methods ran on **three actor seeds**, each for **60 updates × 64 answers =
3,840 training answers**. The same test questions, verifier and 2,048-token answer
cap were used throughout. PPO’s critic configuration was selected on validation;
all learned filters were also selected without online test outcomes. The study
matches answer counts, not generated tokens or compute.

The base model scored **66.70%**. These are final **answer accuracies**, not critic
classification scores; SD is across three training seeds.

| Critic used with PPO | Accuracy ± seed SD | Gain over base | Training GPUh/run | Accounted GPUh/run |
|---|---:|---:|---:|---:|
| Ridge, sparse 768 | **70.95% ± 0.79 pp** | +4.25 pp | 5.18 | 6.99 |
| LSTD(0.99), plain 768 | 70.67% ± 0.33 pp | +3.97 pp | 5.21 | 7.03 |
| LSTD(0.99), two-thirds layer | 70.42% ± 0.45 pp | +3.72 pp | **4.89** | **6.72** |
| LSTD(0.99), sparse 768 | 70.25% ± 0.13 pp | +3.55 pp | 5.32 | 7.16 |
| Conventional transformer critic | 68.92% ± 1.74 pp | +2.22 pp | 6.02 | 7.80 |

Training costs include validation, checkpointing and feature/critic work;
accounted costs additionally include initialization and initial/final test
execution. Prior representation learning is extra. PPO calibration cost another
**19.59 GPUh**, reused by this follow-up. GPU-hours measure allocated GPU time,
not hardware utilization or FLOPs.

**The top two mean accuracies used 768-dimensional features.** All methods improved
on the base in every seed. Linear critics used **12–19% less measured training
GPU time** than conventional PPO in this run. However, all four primary paired
comparison intervals include zero, including after the four-comparison correction.
Three seeds do not establish superiority, equivalence or long-run convergence.

Checks confirmed frozen features, cumulative statistics, small linear-system
residuals and passing PPO training-health tests. Parse failures were **0.3–0.6%**
and truncation **2.2–3.4%**. Verifier errors can still remain. Linear-critic policies
produced about **632–664 tokens** per test answer versus **452–515** for PPO;
lower training cost does not establish cheaper inference.

Earlier, a locked five-seed MATH comparison found cumulative two-thirds-layer
LSTD at **67.49% versus GRPO’s 59.31%**, after a conservative reference audit
retained 291 questions. Its paired difference was **+8.18 pp**, with a descriptive
95% interval **[+3.85, +12.92]**. That used an inspected split and a different
representation; it is evidence for the approach, not a sparse-768 result or a
cross-dataset comparison. The full record is in [HISTORY.md](HISTORY.md).

## Decision: sparse 768, with two fitting options

We choose **sparse 768 as the representation for the next phase**. It combines a
compact head with competitive observed learning, and sparse Ridge had the highest
mean score. Plain 768 remains a useful ablation: the study found no statistically
resolved sparse-versus-plain advantage, which is not proof of equivalence.

The efficiency claim is about **compression**: compared with a 1,536-feature head,
its matrix needs approximately **one-quarter the storage**. Sparse and plain 768
have identical dimensions and use dense operations. Sparse LSTD actually took
slightly more training time than plain LSTD here (**5.32 vs 5.21 GPUh**); we do
**not** claim that sparsity itself has made the implementation faster.

| Situation | Starting choice | Why / what to check |
|---|---|---|
| Complete, reasonably short trajectories with reliable final rewards | **Ridge** | Direct regression on observed returns; simple and our highest-scoring Numina candidate. |
| Longer trajectories where temporal structure may help credit assignment | **LSTD(lambda)** | Uses adjacent states and traces; test against Ridge because bootstrapping can add error. |
| Sparse terminal reward and complete long trajectories | **Compare both** | Long horizon alone does not make LSTD better; near-one lambda previously worked better than short traces. |
| Incomplete search branches or a changed search policy | **Redesign targets and boundaries first** | An unfinished proof is not automatically a failure; historical statistics may estimate the wrong continuation policy. |

“Ridge for short horizons, LSTD for long horizons” is our **working design
heuristic**, not a result established by these experiments. For now, use Ridge
as the practical reference and retain LSTD(0.99) as the temporal alternative.

## Setup and reproduce

This repository targets the existing Duke Slurm/container environment. Start with
[installation and model preparation](SETUP.MD#initial-setup); the online protocol
requires completed representation-study artifacts and its passed smoke test.

```bash
cd /usr/xtmp/ms785/rl_ms_project
source scripts/cluster_env.sh
# Read the completed focused comparison; no new GPU job is needed.
cat outputs/numina-online-768-focused-v1/report.txt
cat outputs/numina-online-768-focused-v1/summary.json
```

See the [focused study commands](docs/numina-online.md#focused-five-method-comparison)
for reproduction, [bottleneck protocol](docs/sparse-critic-study.md) for feature
training, and [SETUP.MD](SETUP.MD) for full dependencies, audits and resumption.
Reports, manifests and per-answer audits are saved under the study output directory;
large model/data artifacts must be available separately. Do not modify shared
training code while jobs are running.

## Next: spend search compute where it matters

After the final comparison extension, we plan to pause math-dataset experiments and move toward a small **Lean-based
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


### Final comparison extension (prepared; results pending)

The [six-method follow-up](docs/numina-online.md#six-method-numina-follow-up-lstd09-ridge-grpo-and-ppo)
compares sparse-768 and two-thirds-layer critics fitted with Ridge or LSTD(0.9),
plus GRPO and conventional PPO: five fresh actor seeds, 120 updates each, matched
answer budgets and six concurrent runs. The existing Numina splits and
validation-selected PPO profile are reused. This intentionally tests lambda .9,
not .99, and does not replace the completed results above. Eight primary
comparisons are declared before running; the already-inspected test questions
limit confirmatory claims.
