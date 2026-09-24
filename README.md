# Math RL: can PPO use a cheaper critic?

We want to improve a math-solving LLM with reinforcement learning while avoiding
PPO's usual second, large neural network for value prediction. Our candidate is a
small linear critic fitted to hidden features the actor already computes.

**So far: useful small critics, a narrow LSTD advantage, and no demonstrated PPO
cost saving yet.** [SETUP.MD](SETUP.MD) contains installation, phase-by-phase commands,
monitoring and resumption. This page explains the experiment and results.

## One example, from tokens to reinforcement learning

Suppose the question is: **“I have 5 avocados and buy 4 more. Each serving needs 3.
How many servings can I make?”**

1. **Actor:** Qwen generates the answer one token at a time. Its action is the next
   token; its state is the question plus the answer written so far.
2. **Reward:** after generation, Math-Verify checks the final answer. A correct
   answer, 3, earns 1; an incorrect answer earns 0. It does not check every step.
3. **Critic:** after a prefix such as “There are 9 avocados…”, estimate the expected
   final reward if this policy continues. For binary outcomes this is a success
   probability, not a verdict on whether that particular sentence is correct.
4. **PPO:** combines rewards and value estimates into advantages using GAE, then
   adjusts the actor's token probabilities with a clipped policy objective.

Ordinary PPO trains a separate transformer critic. We instead take Qwen's final
normalized hidden vector at the last prefix token, **before the next action**, and
fit a small head. The final answer and future tokens never enter that state vector.

**Ridge** fits each prefix directly to the eventual outcome. **LSTD(lambda)** fits
linear value-consistency equations across consecutive states using eligibility
traces. Both use the same features in our matched comparison. At lambda=1, our
complete episodic, gamma=1 setup matches ridge with the same regularization.
Solving TD equations accurately does not guarantee accurate values.

LSTD's critic-fitting lambda is separate from PPO's actor GAE lambda. GRPO, our
other infrastructure baseline, compares rewards across answers to the same prompt
without training a critic.

## What runs the experiment

- **Qwen/Qwen2.5-Math-1.5B base:** the actor, not the instruct model.
- **VERL + vLLM + PyTorch/FSDP:** RL coordination, generation and distributed updates.
- **Math-Verify:** final-answer scoring in a separate environment.
- **Duke Slurm:** GPU/CPU allocations and unattended jobs.
- **Our code:** prompts, reward integration, feature caches, critic fits and reports.

In online pilots, ordinary unparseable outputs receive zero reward. Offline critic
studies exclude answers without a usable verifier label. Thus their metrics are
**conditional on retained answers**. The online comparison scores ordinary parse
failures and verifier timeouts as zero for every method, retaining their status for
audit; infrastructure errors fail the run.

## Results so far

### Phases 0–3: working RL, not proven improvement

We established generation, backpropagation and checkpoint saving. Plain completion
prompts worked better than chat prompts for this base model. Replacing our strict
format parser with Math-Verify recovered credit for correct answers. The fresh
200-answer audit remains incomplete: only 10 were reviewed, all in agreement.
Checkpoint save/resume equivalence is also not yet established.

The PPO and GRPO pilots each ran ten training iterations: **160 generated answers
for PPO, 320 for GRPO**. These were not equal sampling budgets. On the same 500
held-out questions with greedy decoding:

| Model | Correct | Accuracy |
|---|---:|---:|
| Base Qwen | 423/500 | 84.6% |
| PPO/GAE | 424/500 | 84.8% |
| GRPO | 424/500 | 84.8% |

**Interpretation:** the machinery works; one extra correct answer is not convincing
learning evidence. These questions were held out from our training, not necessarily
from Qwen's pretraining.

### Phase 4: do hidden states contain useful value information?

We froze Qwen and generated **80,000 answers** across 4,000 training, 500 validation
and 500 test questions. We retained **77,351** and cached every response-prefix
state. Supervised heads used eight sampled prefixes per retained answer, including
the question-only state. Splits were by question, not by answer.

| Head | Test Brier ↓ | Correctness-prediction accuracy |
|---|---:|---:|
| Constant baseline | 0.23054 | — |
| One-layer sigmoid head | 0.19268 | 71.3% |
| Two-layer MLP | **0.18888** | **71.8%** |
| Ten-layer residual network | 0.18977 | 71.6% |
| Raw linear head, gradient fitted | 0.19521 | 71.1% |
| Direct ridge | 0.19419 | 71.2% |

Brier is mean squared error against binary outcomes; lower is better. These accuracy
numbers measure **prediction of answer correctness**, not Qwen's math accuracy.

**Interpretation:** linear features are useful and close to larger heads. Ridge
fitting/tuning took about 32 seconds, but that excludes generation and feature
extraction. An early undertrained linear head performed badly; longer fitting and
validation-based stopping corrected that. Parser failures later interrupted data
collection; resumable caches and explicit answer exclusions preserved progress.

### Phases 5–7: LSTD needs a long trace in this setting

We fitted LSTD and matched ridge on **61,876 training answers / 20,931,480 token
transitions**. Both used all consecutive training states; evaluation kept the same
sampled prefixes. Regularization and lambda were selected on validation only.

| Method | Original GSM8K test Brier ↓ |
|---|---:|
| LSTD(0) | 0.22889 |
| LSTD(0.90) | 0.22124 |
| LSTD(0.99) | 0.20222 |
| **LSTD(0.999)** | **0.19751** |
| LSTD(1) / matched ridge | 0.19845 |

**Interpretation:** LSTD(0) solved its equations accurately but predicted poorly.
Lambda=0.999 offered a small advantage; lambda=1 matched ridge as expected. This
ridge result differs from Phase 4 because the fitting weights/protocol differ.

A new generation seed on the **same 500 test questions** reproduced the advantage:
0.19790 versus 0.19887; paired question difference **−0.00098**, descriptive 95%
interval **[−0.00146, −0.00050]**. GSM8K length diagnostics favored medium-length
answers, not the longest.

### Phase 8: transfer to harder MATH

Without refitting, the GSM8K critics were evaluated on **148 MATH-500 level 4/5
questions**, restricted to text-only integer/decimal answers. Of 2,368 answers,
2,278 were retained.

| Critic | Brier ↓ | Correctness-prediction accuracy |
|---|---:|---:|
| LSTD(0.999) | **0.20232** | **69.7%** |
| Ridge | 0.20886 | 69.2% |

The paired difference was **−0.00650**, interval **[−0.00887, −0.00411]**. The
advantage was clear on level 4; the level-5 interval included zero. Answers longer
than 512 tokens favored LSTD, with paired difference **−0.00909**. Question-only
predictions showed no clear advantage.

**Interpretation:** encouraging transfer, particularly during longer answers, but
not proof that deeper reasoning causes the benefit. Level-5 truncation was 7.6%;
90 answers were excluded, and reward auditing remains necessary.

### Phase 9: fitting directly on MATH

With the actor frozen, we fitted on **15,346 retained training answers / 11,685,978
token transitions** and evaluated on **300 MATH level 4/5 test questions**. MATH-500
was excluded; lambda and regularization were selected on validation only.

| Critic | Test Brier ↓ | Correctness-prediction accuracy |
|---|---:|---:|
| LSTD(0) | 0.22304 | 66.1% |
| LSTD(0.90) | 0.21715 | 66.1% |
| LSTD(0.99) | 0.19716 | 69.0% |
| LSTD(0.999) | 0.18574 | 71.4% |
| **LSTD(1) / ridge** | **0.18566** | **71.6%** |

**Interpretation:** validation selected lambda=1 and zero regularization. The two
fits then match algebraically, not just statistically. LSTD's earlier transfer
advantage did not translate into an advantage when fitting directly on MATH.
Ridge remains a strong, simpler candidate. Level-5 truncation was 6.9%, and the
metrics exclude unparseable answers. Unknown difficulty labels initially stopped
loading; recording and excluding those labels fixed the loader.

### Phase 10: lower-lambda stress test

This experiment reused the saved new-seed GSM8K and MATH transfer answers. Each
method's regularization was selected on GSM8K validation only.

| Critic | New-seed GSM8K Brier ↓ | MATH transfer Brier ↓ |
|---|---:|---:|
| LSTD(0.85) | 0.22348 | 0.30172 |
| LSTD(0.90) | 0.22113 | 0.29338 |
| LSTD(0.95) | 0.21585 | 0.27263 |
| LSTD(0.999), earlier matched evaluations | **0.19790** | **0.20232** |
| Ridge | 0.19887 | 0.20886 |

All three lower-lambda paired intervals favored ridge. At lambda=0.85/0.90,
accuracy was 64.0% on GSM8K and 32.9% on MATH, versus ridge's 70.9% and 69.2%.

**Interpretation:** LSTD is not generally better. Near-return-regression LSTD is our
promising variant; stronger short-range bootstrapping was harmful here. These
are exploratory results on repeatedly inspected sets. Intervals are descriptive,
not adjusted for multiple comparisons. Raw Brier across datasets also depends on
their success rates; compare methods within each dataset.

### Phase 11: compare the critics inside PPO — implemented, results pending

We now test whether the small critics help **learning**, not just prediction.
All four methods start from the same base Qwen and use the saved MATH level 4/5
numeric question splits, Math-Verify reward and token budgets.

| Method | How it produces advantages |
|---|---|
| Conventional PPO | Separate trainable transformer critic + GAE |
| PPO-ridge | Linear return regression on actor features + GAE |
| PPO-LSTD(0.99) | Linear TD trace solve on actor features + GAE |
| GRPO | Relative rewards among four answers to each question |

Each update generates **16 questions × 4 answers = 64 answers** for every method.
The default pilot is 30 updates (1,920 training answers per method), one seed, and
runs sequentially on the same two GPUs. Greedy test evaluation happens before and
after training. Reports include paired accuracy changes, per-level results,
truncation, critic fitting time, training time and allocated GPU-hours.

For the two linear methods, each iteration captures detached, pre-token features
from the actor's existing log-probability forward pass. Two question folds ensure
an answer's own reward is never used to fit its value predictions. We fit new
heads on the current batch and discard them after the update: **no replay buffer
and no reuse of the offline weights**. The saved offline run supplies the question
split and model identity, not old-policy training answers.

LSTD's trace is **0.99**, as requested; all PPO variants use actor GAE lambda=0.95.
Both linear fits use fixed regularization 0.01 because these batches are much
smaller than the offline dataset; this is a declared pilot setting, not a tuned
winner. Conventional PPO keeps its critic optimizer across updates. All actors
use the same PPO loss and optimizer settings.

**Interpretation:** this is a comparison of complete critic systems, not an
isolated test of the solver against conventional PPO. The test questions were
already inspected offline, the reward audit remains incomplete, and a short
single-seed pilot cannot establish superiority. LSTD(0.99) lost to ridge offline;
we keep that result visible while testing its online behavior. GPU execution of
this new integration still needs cluster validation. Commands are in
[SETUP.MD](SETUP.MD#phase-11--online-ppo-lstd-ridge-conventional-ppo-and-grpo).

We aim to turn an LLM's own hidden representations into a lightweight critic that makes PPO learning more compute-efficient.
The next step is to test whether ridge or LSTD can preserve or improve math-solving performance while reducing total training cost.
