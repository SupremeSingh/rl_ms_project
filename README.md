# Math RL: can PPO use a cheaper critic?

We want to improve a math-solving LLM with reinforcement learning while avoiding
PPO's usual second, large neural network for value prediction. Our candidate is a
small linear critic fitted to hidden features the actor already computes.

**Strongest result so far: PPO with LSTD(0.99) improved MATH accuracy from 52.3%
to 60.0%, with about 30% less measured training time than our conventional PPO
baseline.** GRPO reached 60.3% at slightly lower cost. This is an encouraging
single-seed pilot, not established superiority over ridge or GRPO.

[SETUP.MD](SETUP.MD) contains installation, phase-by-phase commands,
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

Our conventional PPO baseline trains a separate transformer critic. We instead take Qwen's final
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

### Phase 11: online MATH training — a positive LSTD-PPO pilot

We tested whether the small critics help **learning**, not just prediction.
All four methods started from the same base Qwen and used the saved MATH level 4/5
numeric question splits, Math-Verify reward and token budgets.

| Method | How it produces advantages |
|---|---|
| Conventional PPO | Separate trainable transformer critic + GAE |
| PPO-ridge | Linear return regression on actor features + GAE |
| PPO-LSTD(0.99) | Linear TD trace solve on actor features + GAE |
| GRPO | Relative rewards among four answers to each question |

Each update generated **16 questions × 4 answers = 64 answers** for every method.
The completed pilot ran 30 updates (**1,920 training answers per method**), with
seed 42, sequentially on the same two GPUs. Greedy evaluation used the same 300
test questions before and after training; initial correctness scores were identical.

| Method | Final correct | Accuracy | Gain from 52.3% base | Training minutes ↓ | Allocated GPU-hours ↓ |
|---|---:|---:|---:|---:|---:|
| Conventional PPO | 157/300 | 52.3% | 0.0 pp | 78.3 | 2.611 |
| PPO-ridge | 174/300 | 58.0% | +5.7 pp | 54.3 | 1.809 |
| **PPO-LSTD(0.99)** | **180/300** | **60.0%** | **+7.7 pp** | **54.6** | **1.821** |
| GRPO | 181/300 | 60.3% | +8.0 pp | 53.2 | 1.775 |

**What went well:** LSTD-PPO fixed 39 base-model errors and introduced 16 new
errors, a net gain of **23 correct answers**. Its paired comparison against the
base gave p=0.0027; against conventional PPO, p=0.0052. LSTD fitting took only
**81.5 seconds across all 30 updates**, about 2.5% of its measured training time.
Test truncation fell from 7.7% to 2.0%.

**What remains unresolved:** LSTD's six-answer lead over ridge had p=0.42, and
its one-answer deficit to GRPO had p=1.0. These do not establish an advantage or
equivalence. Conventional PPO fixed 27 answers and broke 27: unchanged accuracy
does not mean unchanged weights or prove that a better-tuned baseline cannot learn.

Training costs include generation, feature transport, fitting, validation and
checkpointing; they exclude setup, initialization and initial/final test evaluation.
Thus the 30% saving is for this measured training workload, not an isolated solver
speedup. Logged peak memory was similar across methods; no memory saving is shown.

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

**Interpretation:** this supports our central idea: a cheaply fitted linear critic
can support useful PPO learning. Offline, LSTD(0.99) predicted less accurately
than ridge; online, it achieved the higher score in this pilot. Value-prediction
error alone therefore does not determine which critic will help PPO most.

This compares complete critic systems, not just solvers: conventional PPO also
differs in architecture, targets, persistence and cross-fitting. Test questions
were already inspected offline, the reward audit remains incomplete, and all
p-values are exploratory and unadjusted. Replication across training seeds,
review of changed answers and checks of conventional PPO's critic learning are
needed before claiming a reliable advantage. Commands are in
[SETUP.MD](SETUP.MD#phase-11--online-ppo-lstd-ridge-conventional-ppo-and-grpo).

### Phase 12: planning — initial results and structured rerun

The 800-answer validation screen found no clear solving benefit: current-prompt
accuracy was 32.3%, planning 31.2%, with a paired difference of −1 percentage
point (95% interval −6.25 to +4.5). Both truncated 6.5% of answers.
Planning used 757 tokens on average versus 744 for the current prompt. The run
took about 53 minutes. This tested solving behavior, not critic quality.

The bounded critic experiment completed in 3 hours 36 minutes: 300 training,
50 validation and 100 test questions, four answers per prompt (3,600 total).
Ordinary versus planning answer accuracy was 30.5% versus 29.5%. LSTD Brier was
0.18994 versus 0.19972; ridge was 0.18594 versus 0.20130. Overall paired intervals
included zero. Fixed-prefix results generally favored ordinary reasoning.

Manual response inspection found missed plan headings and an apparent verifier
false rejection: a concluding `b = 7` was obscured by later numbers. Some generated
code-output blocks were also wrong; these blocks are text, not executed programs.
The post-plan comparison therefore needs qualification, and scoring needs auditing.

**Implemented next: an audited, structured rerun.** The planning prompt requests:

1. **What we know:** given facts, constraints and the quantity to find.
2. **What we will do:** a short proposed method, before calculating the answer.
3. **Solution:** carry out the method and check the calculation.
4. **Final answer:** one explicit numeric conclusion.

The control prompt shares the final-answer format and verifier. Both use the same
bounded question splits, four answers per question and a 2,048-token budget:
**3,600 fresh answers**, followed by separate ridge and LSTD(0.99) fits. The actor
remains frozen; this is not a new PPO or GRPO training run.

| Fix | What it checks |
|---|---|
| Broader boundary detection | Inline, Markdown and step-by-step solution headings; fenced code is ignored and multiple headings are flagged as ambiguous. |
| Section-compliance audit | Missing or empty sections, incorrect order and boxed answers appearing before the solution. |
| Opt-in conclusion selection | Explicit final-answer lines and supported concluding numeric assignments, with Math-Verify checking equivalence. Repeated final-answer lines abstain. |
| Response viewer | Actual generated text, prefix predictions, format flags and old/new verifier details. |

An optional audit rescores the previous saved answers into separate files. Original
labels, features, fitted heads and reports are preserved. The new rule records its
selected text and legacy score; existing PPO commands retain their original rule.
It remains an experimental extractor, not a reasoning verifier or a fail-proof
judge. Requested formatting is measured, not assumed or enforced.

The rerun retains random-prefix evaluation and adds fixed-token and post-plan
diagnostics, coverage, calibration and constant baselines. Only already-generated
tokens enter each value prediction. Test questions remain previously inspected;
different prompts and retained-answer populations limit causal interpretation.

**Status:** 50 targeted pipeline tests and 24 verifier tests passed locally;
structured cluster results are pending. No planning benefit is established.
Instructions and audit/viewer locations are in
[SETUP.MD](SETUP.MD#structured-plan-rerun-and-verification-audit).

We aim to turn an LLM's own hidden representations into a lightweight critic that improves learning without a separately trained transformer critic.
The broader goal is reliable gains per unit of compute, with honest comparisons against conventional PPO, ridge and critic-free GRPO.
