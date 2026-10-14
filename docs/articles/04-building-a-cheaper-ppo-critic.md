# Building a cheaper PPO critic: the experiment so far

*Part 4 of Building a Cheaper Critic · 13 October 2026*

Our project asks a practical question: **can a language model's existing representations support a useful PPO critic without training another large transformer?**

The first three articles explain the ingredients separately. [Article 1](01-from-an-answer-to-an-advantage.md) turns a final answer reward into token-level advantages. [Article 2](02-a-critic-built-from-matrices.md) derives a critic fitted from accumulated matrices. [Article 3](03-reading-a-model-through-jlens.md) explains how a Jacobian lens changes the representation available to a predictor. This article brings those ideas together through the experiments we actually ran.

Our evidence has three distinct stages: predicting correctness with small heads, using linear critics to train an actor, and testing whether J-lens can improve the critic's features. The first two have results. The large J-lens comparison is still underway at the last reported checkpoint.

## 1. A concrete task and a replaceable critic

We started with the base Qwen2.5-Math-1.5B model and mathematical question answering. Given a question, the actor generates an answer one token at a time. Math-Verify scores its final answer. Intermediate rewards are zero; the terminal reward is one for verified success and zero for an ordinary incorrect answer. Parser failures and verifier errors need an explicit handling policy and auditing; they are not interchangeable with demonstrated mathematical failure.

Take the question “I have five avocados and buy four more; each serving needs three. How many servings?” At the prefix “There are nine avocados,” a critic estimates the actor's chance of eventually answering correctly. It does not grade that sentence in isolation.

Our infrastructure separates the responsibilities:

| Component | Role in the experiment |
|---|---|
| Qwen actor | Generates answers and receives policy updates |
| VERL, PyTorch and FSDP | Organize rollout and distributed training operations |
| vLLM | Generates sampled answers efficiently |
| Math-Verify | Checks extracted final answers against references |
| Our code | Captures features, fits critics, constructs comparisons and audits results |
| Slurm | Schedules cluster jobs |

Replacing the critic leaves the policy-learning problem intact. LSTD supplies values; GAE constructs advantages; PPO changes the actor. This is not LSPI, and it is not yet formal proof search.

## 2. First check whether small heads can predict outcomes

We generated 80,000 GSM8K answers from a frozen actor and retained 77,351 with usable verifier labels. We saved representations of question-plus-answer prefixes and split by question, so prefixes of the same problem did not become independent train and test examples.

The large version sampled prefix lengths randomly rather than using only a fixed grid. The retained data supplied 495,008 training prefixes, 61,888 validation prefixes and 61,912 test prefixes. A sampled prefix inherited its observed continuation's final outcome. That outcome is one sample of a return, not the prefix's exact success probability.

| Predictor | Test Brier error ↓ | Correctness-prediction accuracy |
|---|---:|---:|
| Constant baseline | 0.23054 | — |
| One-layer sigmoid probe | 0.19268 | 71.3% |
| Two-layer MLP | 0.18888 | 71.8% |
| Ten-layer residual network | 0.18977 | 71.6% |
| Direct ridge value head | 0.19419 | 71.2% |

The extra nonlinear capacity helped modestly; depth did not produce a large improvement. A simple head could extract useful predictive information from the frozen model. The sigmoid probe is linear in its **logit**; ridge produces a value directly. That distinction matters because our LSTD critic requires a value function linear in its fitted weights.

These numbers concern **predicting whether an answer succeeds**, not how often the actor answers correctly. They motivated the next experiment without establishing that PPO would benefit.

## 3. Why we tested LSTD and long traces

Ridge fits each state's observed eventual return. LSTD fits consistency between neighboring state values and rewards. We therefore reconstructed consecutive token transitions, not transitions between arbitrarily spaced sampled prefixes.

On frozen GSM8K data, LSTD(0) had Brier error 0.22889; LSTD(0.999) reached 0.19751, compared with matched ridge at 0.19845. Longer traces helped in this setting. In a later MATH-specific fit, LSTD(1) and ridge both reached 0.18566, consistent with their algebraic equivalence under our complete-episode, undiscounted setup.

This did not show a universal advantage for TD fitting. Shorter traces sometimes performed substantially worse. Our eventual online configuration used LSTD(0.99); it needed its own policy-learning evaluation rather than inheriting a claim from an offline score.

## 4. The accumulated critic we actually use

The actor changes during PPO, but the selected feature encoder remains frozen. Our established single-layer feature map reads around two-thirds through that encoder: block 19 of 28, with 1,536 coordinates. Adding an intercept gives a 1,537-dimensional linear value head.

The state is the prefix **before** the next token. The terminal feature is zero, including the intercept, because no future reward remains. Feature preprocessing stays fixed so that accumulated matrices use consistent coordinates.

For each answer, reset the eligibility trace and process its transitions:

\[
z_t=\gamma\lambda_{\mathrm{critic}}z_{t-1}+\phi_t,
\]
\[
A\mathrel{+}=z_t(\phi_t-\gamma\phi_{t+1})^\top,
\qquad b\mathrel{+}=z_t r_t.
\]

We use \(\gamma=1\), \(\lambda_{\mathrm{critic}}=0.99\), raw sums, and solve

\[
(A+0.01I)w=b.
\]

There is no division by the number of transitions. Relative to an average-form system, the regularizer therefore shrinks as data accumulate. The solve returns new weights; **we do not add old fitted weight vectors together**.

There is one additional safeguard. Each question is permanently assigned to one of two folds. A head fitted from fold A values answers from fold B, and conversely. Otherwise, the very reward used to judge an answer could first enter the critic fitted to that same answer. Token-level values do not prevent that leakage: all those tokens share the eventual outcome.

Here is the training loop, including that distinction:

```text
FREEZE the base encoder and feature preprocessing
ASSIGN each training question permanently to fold 0 or fold 1
INITIALIZE A[0], A[1], b[0], b[1] to zero

FOR each PPO update:
    D = fresh answers sampled from the current actor

    FOR each answer in D:
        f = its question's fold
        z = 0
        FOR each token transition:
            phi = frozen features of the current prefix, with intercept
            next_phi = frozen features of the next prefix
                       or zero at termination
            r = final verifier reward only on the last transition
            z = 0.99 * z + phi
            A[f] += outer(z, phi - next_phi)
            b[f] += z * r

    FOR each fold f:
        w[f] = solve(A[f] + 0.01 * I, b[f])

    FOR each answer in D from fold f:
        values = w[1-f] applied to its current-prefix features
        advantages = GAE(rewards, values, gamma=1, lambda=0.95)

    UPDATE the actor with PPO using D and those advantages
    KEEP the encoder, A and b for the next update
```

The GAE parameter and LSTD trace parameter serve different purposes. The former combines TD residuals into actor advantages; the latter determines how experience enters the critic's fit.

This retains historical information without retaining a replay buffer of answers for fitting. Actor updates use fresh data only. Nevertheless, the critic's matrices contain experience from earlier actor policies. A frozen encoder fixes the coordinate system; it does **not** remove this off-policy mismatch. Our uncorrected accumulation is an empirical design, not a new guarantee of current-policy TD convergence.

## 5. What the online comparisons showed

We first strengthened the conventional PPO baseline with validation-only selection among critic-fitting settings and numerical health checks. In a three-seed comparison with 60 updates of 64 answers per method and seed, the results were:

| Method | Mean final answer accuracy | Mean training GPU-hours |
|---|---:|---:|
| Conventional PPO | 48.11% | 6.707 |
| PPO with ridge | 55.44% | 3.505 |
| PPO with LSTD | 59.67% | 3.557 |
| GRPO | 58.22% | 3.445 |

The base model scored 52.33%. Conventional PPO's critic passed training-target fitting checks, but its actor still deteriorated. These checks exclude some implementation failures; they do not prove that this was an optimal PPO configuration. LSTD's advantage over GRPO was inconclusive in that comparison.

Subsequent studies compared frozen features, accumulation, encoder refresh and layer depth. Accumulation did not establish a win over fresh-batch fitting: its paired difference was −1.11 percentage points, with an interval spanning zero. Refreshing the encoder also failed to establish an improvement. We chose a frozen encoder and accumulation as a workable design, rather than declaring each component independently superior.

With that design locked, a five-seed comparison found 67.80% for two-thirds-layer LSTD(0.99), versus 59.60% for GRPO. Training cost was 3.605 versus 3.435 GPU-hours per run: about 5% more for LSTD.

A later conservative reference audit retained 291 of the 300 test questions:

| System | Mean answer accuracy after reference filtering |
|---|---:|
| Base model | 52.23% |
| GRPO | 59.31% |
| PPO with accumulated LSTD(0.99) | **67.49%** |

The filtered gap was **8.18 percentage points**, with a descriptive paired seed/question interval of **[3.85, 12.92]**. Every one of the five seed differences favored LSTD.

That is our strongest positive result. Its scope remains specific: inspected MATH questions, one small model, a particular training budget and verifier. Filtering was post hoc, retained existing verifier scores and did not repair historical training rewards. Fresh seeds are not fresh test questions. The result supports further investigation, not a universal ranking of RL algorithms.

## 6. What J-lens changes about the experiment

So far, the successful critic reads ordinary hidden features. The next question is whether a more structured readout can improve that input.

We use the official Jacobian-lens implementation to fit layer-specific transports. Our complete-readout construction retains RMS normalization and all pre-softmax vocabulary-score information in exact compact coordinates. It does not select a sparse list of words.

The proposed layer schedule measures distances 1, 2, 4, 8 and 16 from the end, giving blocks **28, 27, 25, 21 and 13**. Their compact readouts concatenate into 7,680 coordinates. This geometric schedule is our hypothesis, not a prescription from the paper.

An initial bounded study found:

| Representation, with ridge | Test Brier error ↓ | Correctness-prediction accuracy |
|---|---:|---:|
| Complete J-lens concatenation | 0.18800 | 72.1% |
| Raw hidden concatenation | 0.19337 | 71.3% |
| RMS-normalized hidden concatenation | 0.18908 | 72.4% |
| Two-thirds hidden layer | **0.18406** | **73.2%** |

The complete lens improved over raw concatenation. Its difference from normalization-matched concatenation was inconclusive, and the single two-thirds layer remained better. Earlier sparse reconstructions performed worse; removing that restriction was useful, but did not establish a superior critic.

## 7. The large comparison now underway

The new study targets **104,000 main-study answers** from a filtered NuminaMath subset: 10,000 training, 1,000 validation and 2,000 test question families, with eight answers each. Known overlap with our earlier datasets is filtered, although pretraining overlap cannot be ruled out.

The actor stays frozen. We store intermediate representations and compare five feature maps: complete J-lens concatenation, raw concatenation, normalized concatenation, the two-thirds layer and the final layer. The study fits one-layer sigmoid, two-layer MLP and ten-layer residual probes, plus ridge and LSTD(0.99).

Neural probes use random prefixes; the matched ridge/LSTD comparison uses consecutive token transitions. Their training-state weighting consequently differs, even when evaluation prefixes match. Validation chooses settings, and question-separated test data measure generalization. Accuracy is accompanied by probability error and computational cost. The [study protocol](../large-jlens-study.md) records the details.

This is a representation experiment before it is a new PPO experiment. Concatenation also expands the linear system: moving from 1,537 to 7,681 coordinates costs about 25 times the dense matrix storage and roughly 125 times the dense-solve work. We cannot call the replacement cheaper merely because its final head is linear.

The immediate decision is whether a lens representation earns its additional cost. If it does, the next controlled online experiment keeps the successful accumulated LSTD/PPO machinery fixed and changes the critic's features. If it does not, the existing two-thirds-layer critic remains a strong practical baseline. Either outcome informs the longer-term idea of using an inexpensive value estimator to guide verified search, but formal proof search still needs its own data and tests.

Our project has shown that a small fitted value head can support meaningful actor learning, and that its representation matters. Could a complete J-lens readout improve that accuracy–cost trade-off enough to become the next critic—or will ordinary intermediate features remain the more useful choice?
