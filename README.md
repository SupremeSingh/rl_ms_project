# CLEAR: Compressed Linear Evaluation with Accumulated Regression

**Reuse the LLM’s representations. Fit a small value function. Let PPO do the learning.**

**CLEAR** is the overall reinforcement-learning pipeline. It uses the
[**SLICE critic system**](https://github.com/SupremeSingh/SLICE), which compresses
features from several frozen LLM layers into **768 learned coordinates**. A linear Ridge critic estimates the value of each answer prefix.
We update that critic by accumulating matrices and solving a linear system,
instead of training another large transformer with backpropagation.

**Naming:** SLICE supplies the critic representation and prediction system;
CLEAR combines it with cumulative fitting, question cross-fitting, rollout
generation, GAE and PPO actor updates. SLICE is not the name of the full RL pipeline.

In our final Numina comparison, **PPO with sparse-768 Ridge reached 74.60% mean
answer accuracy**, versus **73.52% for conventional PPO** and **72.75% for GRPO**.
It used **17% less training GPU time than PPO**, but **10% more than GRPO**.
The accuracy advantage over PPO remains uncertain; the paired interval over GRPO
is positive, narrowly, after correcting for eight primary comparisons.

[Setup](SETUP.MD#initial-setup) · [Experiment protocol](docs/numina-online.md) ·
[Project history](HISTORY.md) · [First-principles articles](docs/articles/README.md)

Results updated **22 October 2026**. Math experiments are complete for now;
Lean proof search is the next research direction.

## From an answer prefix to a useful critic

Suppose the model is solving “9 avocados, 3 per serving: how many servings?”
After writing “We divide 9 by 3…”, it must choose another token. The **actor**
chooses that token; the **critic** estimates the expected reward if the policy
continues from this prefix. The correct final answer earns 1 from Math-Verify;
otherwise the outcome reward is 0. This is eventual-answer prediction, not proof
that each reasoning step is valid.

PPO uses those token-level values and rewards to calculate advantages with GAE,
then updates the actor. Our change is how the values are learned:

1. **Read frozen features.** A separate frozen Qwen2.5-Math-1.5B encoder processes
   causal prefixes. Read blocks **28, 27, 25, 21, 13** (one-based), RMS-normalize
   each vector, concatenate, and apply training-only centering/scaling.
2. **Compress 7,680 → 768.** A learned row-normalized projection plus ReLU forms
   the bottleneck. Offline training uses correctness cross-entropy plus
   `0.01 × mean(feature activations)`. Freeze this compressor and discard its
   temporary prediction head before PPO begins.
3. **Fit a linear value head.** Append an intercept to obtain 769 features.
   Accumulate Ridge statistics from observed final rewards and solve for the
   weights. The encoder and compressor remain fixed while the actor learns.

The compressor is nonlinear; the value head is linear in its output. “Sparse”
means an activation penalty encourages zeros and small activations—not sparse
projection weights, orthogonality, or a proven removal of semantic noise.
Both encoding and compression still cost compute, included in the online timings.

![CLEAR architecture using the SLICE critic: frozen compressed features, cross-fitted cumulative critics and fresh-data PPO updates](docs/figures/final-architecture.png)

[Vector version](docs/figures/final-architecture.svg). The figure includes the
LSTD alternative at the final comparison’s **critic lambda 0.9**; earlier
studies used 0.99. The selected final configuration is **Ridge**, which has no
eligibility-trace lambda.

## Matrix accumulation: used by both Ridge and LSTD

**Accumulation is not exclusive to LSTD. Our selected sparse-768 Ridge critic
also retains evidence across PPO updates.** Its 74.60% result used this cumulative
fitting approach, not a fresh fit on only the latest batch.

We accumulate the sufficient statistics **A and b**, not the weight vector **w**.
After adding each new batch, we solve again for weights using all accumulated
statistics. Think of A and b as a compact ledger of previous training examples:
the ledger persists, while the fitted weights are recomputed.

| Fitting method | Add to A for each transition/prefix | Add to b | What it learns from |
|---|---|---|---|
| Ridge | `outer(x_t, x_t)` | `x_t * R` | Each prefix paired with its observed final return R |
| LSTD(lambda) | `outer(z_t, x_t - gamma * x_next)` | `z_t * r_t` | Consecutive states and immediate rewards, weighted by eligibility traces |

Here `z_t = gamma * lambda * z_previous + x_t`, reset at the start of each
answer. R is the answer's final binary reward; r_t is zero except at termination.
**Both methods solve `(A + epsilon * I) w = b` after accumulating the new data.**
This equivalence of the solve's form does not make their fitting objectives identical.

Each question belongs permanently to one of two folds. Its rewards fit one head;
its values are predicted by the head fitted on the **other** fold. The compressor
was trained on separate question families. Together these choices limit
own-question reward leakage into the actor’s baseline.

```text
Freeze the encoder, preprocessing and sparse-768 compressor.
Initialize two Ridge systems: A = 0, b = 0.

For each PPO update:
    Generate fresh answers and obtain terminal correctness rewards.
    Encode each pre-action prefix as x = [768 features, 1].

    For every answer with final reward R:
        Select its permanent question-fold system.
        For every prefix x:
            A += outer(x, x)
            b += x * R

    For each fold, solve (A + 0.01 I) w = b.
    Predict each question's values using the opposite fold's weights.
    Use fresh rewards and values in GAE; update the actor with PPO.
    Retain A and b. Keep the encoder and compressor frozen.
```

For both methods, **A is 769 × 769; b and w each have 769 entries**, per fold. There is no per-update averaging, encoder
refresh or answer replay buffer. Old policies remain represented in the accumulated
statistics without off-policy correction: this is not an exact evaluation of the
latest policy. GAE uses **gamma = 1, lambda = 0.95**.

The LSTD alternative instead accumulates eligibility-trace-weighted TD equations
between consecutive prefix states. Traces reset per answer; terminal features
are zero. We compare it as a different fitting method, not an additional layer.
The final comparison does not isolate the causal benefit of accumulation alone:
all four linear-critic arms accumulate statistics.

## The final comparison

We used **3,000 training, 500 validation and 2,000 test questions** from the
eligible numeric `cn_k12` subset of NuminaMath-1.5. Question-family filtering
excluded overlaps with the earlier feature-learning study, GSM8K and MATH using
normalized text and number templates. It does not establish semantic or
pretraining decontamination.

Each method ran **five fresh actor seeds × 120 updates**, with **64 generated
answers per update**: 7,680 training answers per run. All used the same base
actor, reward, prompts and 2,048-token answer cap. Evaluation was greedy.
PPO’s critic configuration was selected on validation in the earlier 60-update
calibration and reused. The learned compressor was also fixed across actor seeds.

| Method | Final answer accuracy | Training GPUh/run | Accounted GPUh/run |
|---|---:|---:|---:|
| **CLEAR: PPO + SLICE sparse-768 Ridge** | **74.60%** | **10.35** | **12.20** |
| Conventional transformer-critic PPO | 73.52% | 12.51 | 14.32 |
| GRPO (no learned critic) | 72.75% | **9.38** | **11.36** |
| PPO + Ridge, two-thirds layer | 72.54% | 10.15 | 12.03 |
| PPO + LSTD(0.9), two-thirds layer | 72.40% | 9.61 | 11.43 |
| PPO + LSTD(0.9), sparse 768 | 70.86% | 10.33 | 12.17 |

These are **actor answer accuracies after training**, not critic classification
accuracies. GRPO is an alternative training algorithm, not a critic. All four
linear-critic methods retain frozen features, cumulative statistics and question
cross-fitting. Equal answer budgets do not imply equal generated tokens or FLOPs.

| Primary comparison | Mean difference | Adjusted paired interval (pp) |
|---|---:|---:|
| Sparse Ridge − two-thirds Ridge | **+2.06 pp** | **[+0.59, +3.53]** |
| Sparse Ridge − GRPO | **+1.85 pp** | **[+0.08, +3.50]** |
| Sparse Ridge − conventional PPO | +1.08 pp | [−1.22, +3.56] |
| Sparse Ridge − sparse LSTD(0.9) | **+3.74 pp** | **[+1.73, +6.26]** |

Intervals use a paired seed/question bootstrap and Bonferroni nominal coverage
for eight primary comparisons. Sparse Ridge beat two-thirds Ridge, GRPO and
sparse LSTD in **all five seeds**. Its PPO advantage was positive in four seeds,
but the interval spans zero. The test split had already been inspected; these
are exploratory results, not an untouched confirmatory benchmark.

**What we conclude:** the learned sparse-768 representation improves Ridge over
the single-layer Ridge control in this setting. Sparse Ridge is our chosen
configuration: highest mean accuracy with lower measured cost than conventional
PPO. This comparison does not isolate the sparsity penalty from multi-layer
compression, prove superiority over PPO, or establish a universal LSTD horizon
rule. Plain-versus-sparse and earlier lambda studies are in [HISTORY.md](HISTORY.md).

## Costs and checks that matter

- **Whole-pipeline costs:** training GPU-hours include feature extraction, fitting,
  validation and checkpoints. Accounted GPU-hours also include initialization and
  initial/final test execution. Against PPO, CLEAR saves about **17% training**
  or **15% accounted GPU time**. Against GRPO it costs about **10% training** or
  **7% accounted GPU time** more. These are allocated GPU times, not FLOP counts.
- **Prior costs:** representation generation/training are additional. PPO
  calibration consumed **19.59 GPUh**, reused here. Neither is included in the
  per-run table; the table is not total development cost.
- **PPO diagnostics:** all runs completed, remained finite and updated parameters.
  Four of five seeds reduced average critic training-target MSE. Seed 302 went
  from **0.004713 to 0.004867** (+3.3%) and failed that fitting check; it was also
  the seed where PPO beat sparse Ridge by 1.40 pp. We retain it in all comparisons.
  These checks do not measure held-out value accuracy.
- **Scope:** one base model, one task subset, five actor seeds and one selected
  compressor. Math-Verify labels need auditing; successful execution is not proof
  of perfect scoring. Training-cost savings do not establish inference savings.

## Reproduce and inspect

**VERL** provides distributed PPO/GRPO training, **vLLM** generates answers,
**Math-Verify** supplies outcome labels, and **Slurm** schedules jobs. This
repository implements frozen feature extraction, compressor training, cumulative
linear critics, cross-fitting and the evaluation/audit pipeline.

Use [SETUP.MD](SETUP.MD#initial-setup) for the pinned cluster/container environment.
The [six-method protocol](docs/numina-online.md#six-method-numina-follow-up-lstd09-ridge-grpo-and-ppo)
records generation limits, data separation, submission and dependencies; the
[bottleneck protocol](docs/sparse-critic-study.md) documents representation training.
Saved models, datasets and upstream study artifacts are required separately.

```bash
cd /usr/xtmp/ms785/rl_ms_project
source scripts/cluster_env.sh
cat outputs/numina-online-six-lambda09-v3/report.txt
cat outputs/numina-online-six-lambda09-v3/summary.json
```

The output directory also contains the sealed manifest, per-run records and
`audit.jsonl`. Do not change shared training code while jobs are running. Earlier
experiments, discarded approaches and the previous Numina comparison are preserved
in [HISTORY.md](HISTORY.md); they are not additional results from this final run.

## Next: AlphaProof-inspired search, at a manageable scale

The next step is a small **Lean proof-search system for properties of small
programs**. An LLM proposes tactics, Lean executes and checks them, and a SLICE-based critic
ranks resulting proof states to allocate the search budget. A score guides
exploration; **Lean, not the critic, certifies a completed proof**.

Start with policy-only search versus Ridge-guided search at matched total compute.
Refit the compressor and value head on proof states and search outcomes rather
than assuming math-answer features transfer. Define the target as success under
a specified continuation/search budget; an unfinished branch is not automatically
a failure. Measure verified proofs, latency, generated tokens and encoding cost.
LSTD can then be tested on tactic-level transitions; neither lambda nor the
benefit of accumulated historical statistics transfers automatically.

This is AlphaProof-inspired, not an AlphaProof reproduction. Our aim is narrower:
**use a small learned critic to find more verified proofs for the same compute.**
