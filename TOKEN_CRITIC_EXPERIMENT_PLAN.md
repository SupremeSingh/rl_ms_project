# Final protocol: token-level PPO critics with gradient TD versus LSTD

Status: experimental design; no training runs are implied. Stages 0–1 are assumed complete as requested and are included below to make the protocol self-contained. This protocol specifies both an initial controlled comparison and a later practical-efficiency comparison.

## Objective and primary comparison

Starting after the completed Stage 0 infrastructure and Stage 1 verifier audit, build standard PPO with GAE, replace its separate critic with a small probe over actor prefix features, and compare gradient-based TD with LSTD. Determine which approach offers the best held-out accuracy per generated token and per GPU-hour. LSTD is a candidate, not the assumed endpoint.

The core question is: **With the same linear value head, actor features, trajectories, and PPO update, does an LSTD solve improve efficiency over incremental or large-batch semi-gradient TD?**

| ID | Method | Role |
|---|---|---|
| P0 | PPO + GAE + a separate trainable LLM critic | Conventional integration and performance reference |
| T-MLP | PPO + GAE + a two-layer actor-feature probe trained by semi-gradient TD | Original token-level proposal; architecture pilot |
| T-small | PPO + GAE + a linear actor-feature probe, small-minibatch TD | Primary gradient baseline |
| T-large | Same linear probe and data, large-minibatch/full-batch TD | Separates batch-size effects from solver effects |
| L | Same linear probe and data, regularized LSTD | Primary least-squares method |

Final reporting must include T-small, T-large, and L. Retain P0 as a reference if resources permit; it is not the isolated solver comparison. T-MLP need only advance beyond the pilot if it offers a meaningful gain. Do not describe any of these as an exact reproduction of POISE: POISE uses sequence-level cross-rollout reward regression.

## Fixed setup

- Actor: `Qwen/Qwen2.5-Math-1.5B`, pinned model revision; use any Stage 1 format warm start as the common initial checkpoint.
- Framework: existing VERL v0.4.1 setup, commit `8d9e350ea58c7ad4b50dd14d9dcb50577242c55f`; pinned container, CUDA, tokenizer, data, and code revisions.
- Hardware: university GPU cluster. Profile memory before deciding whether the separate-critic baseline needs more GPUs. Compare GPU-hours, not only wall time, if allocation differs.
- Smoke data: GSM8K. Main data: select English DAPO-Math-17K or a difficulty-matched MATH subset once, before method selection.
- Reward: binary correctness under the audited extraction contract. Require the final `Answer:` line. No learned reward model or intermediate correctness rewards.
- Re-audit the verifier on the main dataset: the repository's existing numeric GSM8K parser is not automatically sufficient for symbolic MATH/DAPO answers. Deduplicate train/validation/test prompts and sources.
- Sampling: temperature 1.0, top-p 1.0. One response per prompt for all primary online arms. Stage 1's two-response audit does not impose a two-response training requirement.
- Response caps: 512 smoke; 2,048 main. Increase only in a separately matched experiment after stability.
- Actor LR: start at 1e-6. BF16 model computation. Accumulate critic statistics in FP32, with FP64 diagnostics/solves where necessary.
- Development: one seed. Final comparison: at least three paired seed IDs, identical initial checkpoints, prompt ordering, evaluation prompts, and decoding settings. Online responses will diverge as policies diverge; do not claim identical online datasets.
- Fix PPO clipping, actor optimizer, epochs, loss aggregation, KL treatment, entropy bonus, prompt distribution, and advantage normalization across the final arms. Use sequence-mean/token-mean actor aggregation initially: mean over valid response tokens within each sequence, then mean over sequences. Use uniform valid-transition weighting for critic fitting. This defines a common practical PPO surrogate; do not claim it is an exactly unbiased unnormalized episodic policy gradient.
- Start with gamma = 1, PPO clip = 0.2, two actor epochs per fresh batch, no KL shaping or entropy bonus. Treat these as provisional common development settings, not method-specific tunables. Monitor policy drift and freeze any revisions before the final comparison.
- Primary GAE lambda = 0.95, with one matched lambda = 1 sensitivity check. Do not confuse GAE lambda with the critic's eligibility-trace lambda or ridge coefficient.
- Keep unwhitened advantages. Apply identical masked whitening if used for PPO; measure advantage statistics before whitening. Form standard-critic return targets from raw advantages, never from whitened ones. Default to whitening across all valid response tokens in the global rollout batch, once before PPO minibatching, with an explicit epsilon and population-variance convention shared across arms.

## Shared mathematical and data contract

For action/token a_t = y_t, the state is s_t = (x, y_<t). Values must use only that prefix. The final prompt-token activation predicts the first response action; the activation after y_t predicts the next action. Test this shift explicitly.

Attach the actual sequence reward to the final valid response action. Pad tokens contribute nothing. In the primary experiment, EOS and the generation cap both terminate the bounded-response task; the post-terminal value is zero. Apply the same answer checker at the cap, log truncation separately, and do not silently mix this convention with time-limit bootstrapping for an unbounded task.

GAE uses:

    delta_t = r_t + gamma * (1 - done_t) * V(s_{t+1}) - V(s_t)
    A_t = delta_t + gamma * lambda_GAE * (1 - done_t) * A_{t+1}

Freeze values and advantages throughout the actor's PPO epochs. Detach actor features for critic fitting, so critic losses never update the actor backbone. PPO alone updates the actor.

Read one exact, documented residual-stream location during the actor old-log-probability forward pass. Start with the final transformer block output before final normalization. In the frozen-policy pilot compare it with one middle/late block, using exact module names rather than an ambiguous layer number. Fix the chosen location for both solvers. No future-window pooling, answer features, or entropy features in the primary experiment.

Use a fixed feature transform, shared across arms: normalize each prefix vector, apply a seeded fixed projection, and add an intercept. Start with 128 projected dimensions, and compare 64 and 256 in the frozen-policy pilot. This is an experimental choice, not a claim that random projection preserves all useful information. Keep a full-feature diagnostic on a small subset to detect projection damage. Save transform settings and the projection matrix; do not independently refit PCA/scalers between methods or silently change coordinates across iterations.

Specify normalization as per-vector RMS normalization with a fixed epsilon. Initialize the projection once with a recorded seed, scale, and shape. Append the fraction of response budget already consumed as an explicit scalar to expose the finite horizon; with p projected features plus this scalar and an intercept, the linear solve dimension is d = p + 2. Use this same feature map for both solvers. A raw-hidden-state-only ablation can test the budget scalar, but it is not part of the primary solver comparison.

Treat full prefixes as the true states. One hidden vector is a feature approximation, not a proven sufficient Markov state. Use deterministic forward passes for feature extraction (disable dropout), and maintain causal attention and correct positions under packing/padding. Compute response log probabilities and features on the same actor snapshot; explicitly test first-token, EOS, and packed-sequence boundaries.

For fixed raw advantages, the common PPO objective is:

    rho_it = pi_theta(a_it | s_it) / pi_old(a_it | s_it)
    J = mean_i [ (1/T_i) sum_t min(rho_it*A_it,
                                 clip(rho_it, 1-epsilon, 1+epsilon)*A_it) ]

Only valid stochastic response actions enter the objective. Preserve the sampling distribution used to define pi_old, including any logits processors; forced tokens must be masked or accounted for. The primary temperature=1/top-p=1 setup avoids additional sampling transformations. Validate reward handling if the rollout engine omits EOS from decoded text.

## What changes between iterations

At outer iteration k, the actor pi_theta_k and extracted features are fixed while fitting the critic. Each LSTD fold produces a fresh set of weights w_k from its own A_k and b_k. PPO then changes theta_k to theta_{k+1}; new data and current-model representations define the next fitting problem. Repeating a solve on unchanged data, features, and regularization does not improve the solution.

The initial comparison refits linear critics from zero. It therefore tests fitting efficiency without accumulated critic parameters. This is distinct from the conventional persistent critic in P0. A practical warm-prior experiment below tests whether retaining previous weights changes the conclusion. Increasing predicted values is not evidence of increasing critic accuracy: track held-out error under the corresponding actor checkpoint.

## Stage 0 — Infrastructure and reproducibility (assumed complete)

The common VERL pipeline must load prompts, generate responses, parse and verify answers, compute actor log probabilities, construct advantages/returns, update the actor and any critic, then evaluate and checkpoint. Pin the framework commit, container digest, CUDA/runtime versions, model/tokenizer revision, dataset revisions, and source configuration. Run distributed work on cluster compute nodes and record GPU model/count for each experiment.

Every checkpoint must restore actor weights/optimizer, critic or probe state, feature transforms, any buffer/statistics, random-number states, sampler position, and configuration. Checkpoint at an outer-iteration boundary initially. Compare an interrupted/resumed smoke run with an uninterrupted run using predeclared tolerances for rewards and learning metrics. Nondeterministic kernels and rollout scheduling mean numerical reproducibility is not automatically bitwise reproducibility.

**Gate:** Resume validation passes and artifacts contain sufficient provenance to rerun the experiment. Repeat this gate when introducing a new critic or distributed statistics path.

## Stage 1 — Reward, formatting, and difficulty audit (assumed complete)

Manually label at least 200 generated responses and compare with the automatic verifier. Require at least 99% agreement, initial correctness roughly between 5% and 70%, and at least 10% mixed-reward pairs when sampling twice per prompt. Retain the existing project's engineering gates of at least 95% answer-format rate and at most 5% truncation, subject to a documented re-audit rather than silently changing thresholds.

If formatting fails, adjust the shared prompt or use a small format-only SFT warm start. Apply the same resulting initial checkpoint to every arm. If rewards are almost always zero or one, change task difficulty before training. Preserve held-out evaluation splits, and prevent warm-start data from leaking into them.

Repeat the audit when switching from GSM8K to MATH/DAPO or changing the answer parser/response budget. The two-rollout mixed-pair criterion diagnoses initial reward diversity; the experimental token-level methods themselves use one rollout per prompt.

**Gate:** The verifier is reliable and the task provides informative rewards before training starts.

## Stage 2 — Implement conventional PPO and GAE

**Question:** Can the common trainer perform correct token-level actor–critic updates?

Implement P0 with a separate model initialized from the actor checkpoint and a scalar prefix-value head. Fit the conventional critic to detached GAE-derived return targets using the pinned VERL critic implementation. Record whether value clipping is enabled. This baseline is not the same fitting objective as the experimental semi-gradient TD heads.

Audit the pinned source before adding adapters: value/output alignment, advantage whitening, critic-worker creation, response masks, and update order. Current VERL documentation is guidance only; do not assume the current extension registry exists at the pinned revision.

Start with 16–32 prompts per update and 10–20 smoke updates, using gradient accumulation if needed. Then run a longer one-seed GSM8K learning pilot. Keep P0's actor settings shared with later arms.

Required checks:

1. Hand-computed trajectories verify terminal rewards, padding, discounting, and GAE.
2. With gamma = lambda_GAE = 1, GAE equals observed return minus the pre-action baseline.
3. Changing future response tokens leaves earlier prefix features and baseline predictions unchanged.
4. PPO ratios are approximately one before an update; probabilities use the same tokenization, temperature convention, and valid-action mask as sampling. Log any rollout/training engine discrepancy. Repeated PPO minibatches never recompute old log probabilities or advantages using updated actor weights.
5. Actor and critic gradients reach their intended parameters only.
6. Resume restores the newly added critic state and reproduces a short run within documented tolerance.

**Gate:** Numerical checks pass, no NaNs, critic fitting reduces its loss on a fixed batch, and a controlled training-subset experiment shows that PPO can learn. Held-out gains are measured but are not guaranteed by a short smoke test.

## Stage 3 — Implement the original actor-feature TD critic

**Question:** Can a small token-level probe replace the separate critic?

Implement T-MLP first, using a two-layer head (one hidden layer, e.g. width 128, and a scalar output). The actor features are detached. Train with a semi-gradient TD(0) target:

    target_t = r_t + gamma * (1 - done_t) * stopgrad(V_phi(s_{t+1}))
    loss = mean_valid[(V_phi(s_t) - target_t)^2]

Recompute targets between optimizer steps, but detach them within each step. Do not accidentally implement residual-gradient minimization by backpropagating through both values. Do not add a target-network schedule without documenting it as a separate variant.

Introduce T-small with the same pipeline and a linear head V_w(s) = z(s)^T w. Use SGD-style semi-gradient updates for the clean LSTD comparison; Adam may be an additional practical variant, but is not required. Choose the critic LR and update budget on development data with the same search allowance later given to LSTD regularization.

For the primary comparison adopt the following common batch procedure:

1. Freeze the current actor snapshot and generate a fresh batch.
2. Extract and detach current prefix features during log-probability computation.
3. Split the batch into two prompt-disjoint folds. Fit a critic on each fold and use it to predict values on the other fold. Related responses to one prompt must never cross folds.
4. Compute GAE from those out-of-fold values and the actual trajectory rewards.
5. Run PPO with fixed advantages, then discard this batch's features before gathering with the new actor.

All primary linear TD/LSTD arms use the same folds and initialization rule. Initially refit each linear head from zero on each batch; this avoids giving TD a historical warm start while LSTD sees only fresh data. Initialize the neural MLP with a documented ordinary random initialization, not all-zero weights, which would prevent hidden units from learning distinct features. Later warm-start experiments must give both methods an explicit matched prior/initialization treatment. Cross-fitting adds two head fits but no extra rollouts; profile the cost rather than assuming it is negligible. It is an experimental safeguard against own-outcome overfitting, not a claim that practical PPO requires it or that approximate GAE/PPO is fully unbiased.

For each held-out trajectory use one predictor (trained on the opposite fold) for every V(s_t) and V(s_{t+1}); never switch predictors midway through an answer. Standard P0 keeps its conventional update order. Report this procedural difference when comparing P0 with the experimental family; the controlled TD/LSTD comparison uses exactly the same fitting order.

Do not use old-policy replay yet. Features are frozen within a fit; the actor is not frozen for the entire run.

**Gate:** Prefix causality and gradient isolation pass; probes show useful held-out value predictions; short PPO runs remain stable. If TD(0) struggles on sparse, long-horizon rewards, diagnose it in Stage 4 before buying full runs.

## Stage 4 — Compare critics on identical frozen-policy data

**Question:** Is a failure due to representation, fitting, or the PPO feedback loop?

Freeze the initial actor and collect a cached pilot dataset: start with approximately 512 fitting prompts and 128 validation prompts, expanding only if estimates are too noisy. Keep any final diagnostic test prompts separate. Cache raw token IDs, rewards, prefix features, masks, and actor version. This cache may be reused freely while evaluating this fixed policy.

Compare T-MLP, T-small, T-large, L, and a simple linear Monte Carlo regression control (prefix features -> observed final reward). Implement a small standalone LSTD reference solver for this stage using the equations in Stage 5; distributed online integration follows in Stage 5. The Monte Carlo control diagnoses whether bootstrapping is the bottleneck; it is not another required full RL arm.

For a small diagnostic set, take prefixes at several completion fractions and sample 8–16 independent continuations from each exact prefix with the same remaining response budget. Begin with roughly 32 validation prompts and four prefixes each, omitting already-terminal prefixes. Resume from exact token IDs, not a decode/re-tokenize round trip. Treat average reward as a noisy estimate of prefix value; account for binomial uncertainty. These extra continuations are diagnostic expenditure and must be counted separately from training rollouts. Never use these continuation labels to fit the critic that is being evaluated. At later checkpoints generate new continuation labels under that checkpoint, not the initial policy.

Measure held-out return MSE/Brier-style error, correlation and calibration against continuation success rates, value range, and prediction quality by prefix position, length, and difficulty. Use prompt-level splits and uncertainty estimates: thousands of tokens from one answer are correlated observations.

A constant zero predictor can have small intermediate TD errors on sparse-reward tasks; low TD error alone is not a passing result. If neither neural nor linear critics beat a constant baseline meaningfully, revisit features/reward signal. If full features work but projection fails, enlarge or change the shared projection. If Monte Carlo works but TD(0) fails, prioritize a matched trace experiment before long PPO runs.

**Gate:** Select shared layer and feature dimension on validation only; identify numerically stable critic configurations. If LSTD is competitive rather than superior, it may still advance for an online efficiency test. An unstable solve does not advance merely because its training residual is small.

## Stage 5 — Implement regularized LSTD on the same linear features

**Question:** Does solving the linear TD equations beat iterating toward them?

For n valid transitions in a fitting fold, use z'_t = 0 at terminal transitions and form:

    A = (1/n) sum_t z_t (z_t - gamma*z'_t)^T
    b = (1/n) sum_t z_t r_t
    (A + eta*I) w = b

Use normalized statistics so eta has a consistent meaning when batch size changes. Call a general linear solver, not an explicit matrix inverse or a Cholesky routine assuming A is symmetric positive definite. Record solve residual, condition estimate, regularization, and prediction range. Start with eta in {1e-6, 1e-4, 1e-2} on development data after feature normalization, refining locally if conditioning or validation error warrants it. Sparse terminal rewards and long episodes can make excessive shrinkage particularly damaging. Document whether the intercept is regularized and match that convention in TD. The strict solver comparison uses the same eta; separately tuned eta values belong to the practical-efficiency comparison.

For the matched linear TD arm, include the corresponding shrinkage:

    w <- w + alpha * [mean_t(z_t * delta_t) - eta*w]

On a fixed full batch, its stationary equation is the same regularized LSTD system when the iteration converges. This gives a direct solver sanity check, not a guarantee that an arbitrary SGD step size converges. The neural TD pilot is not expected to share this fixed point.

Stream feature chunks to accumulate A and b. In distributed runs, sum statistics and valid-transition counts before normalizing and solving; do not average independently fitted worker weights. Reset fold statistics each fresh batch. Total dense cost is roughly O(n*d^2 + d^3), where d is the projected feature dimension including the intercept. Measure feature/projection cost as well as matrix assembly and solve time.

Do not clip value predictions in the primary fitting comparison: clipping changes the linear fixed-point interpretation and can hide instability. Log values outside [0,1]; any clipped-value robustness run must apply identically to both arms.

**Gate:** On a small tabular episodic process, unregularized LSTD agrees with known values given sufficient coverage, and regularized LSTD agrees with its analytically specified regularized solution. On a cached batch with a stable TD step size, converged full-batch linear TD and LSTD agree within a declared numerical tolerance. Tests catch terminal masking and regularization normalization errors. Validate the solver in FP64 on deterministic examples first; FP32 implementation differences must be below a recorded tolerance before online use.

## Stage 6 — Separate batch effects, solver effects, and traces

**Question:** Is the advantage from LSTD, bigger gradient batches, or better propagation of sparse rewards?

First hold the actor rollout batch and fitting dataset fixed. Compare TD minibatches of 256 and 4,096 valid transitions, and a full-fold update when practical. LSTD always aggregates the full fitting fold; changing its matrix-assembly chunk size is only a systems optimization, not a statistical batch-size experiment.

Use two views: (a) equal data with enough fitting to characterize each method's best attainable prediction quality; (b) equal critic fitting wall time. Count all TD passes and both cross-fitting solves. Do not deliberately undertrain TD or give one method more data.

Then run short online pilots of T-small, T-large, and L, initially about 32–64 prompts/update for 50–100 updates. Choose a common feasible rollout batch by profiling; increase equally across arms. Keep the main token cap at 2,048 and retain one rollout per prompt. These counts are initial development budgets, not statistical power guarantees.

If trace-based fitting is needed, extend **both** linear critics together:

    e_t = gamma*lambda_critic*e_{t-1} + z_t
    A = mean_t[e_t (z_t - gamma*z'_t)^T]
    b = mean_t[e_t r_t]

For gradient TD(lambda_critic), use the corresponding e_t*delta_t update. Reset traces per trajectory and never shuffle individual transitions within a trace. Compare a small matched set, such as lambda_critic in {0, 0.95, 1}; one is a useful Monte Carlo-related endpoint. Keep lambda_critic distinct from lambda_GAE and eta. Retain TD(0)/LSTD(0) as the clean first comparison.

Test GAE lambda = 1 on the selected critics as a diagnostic: with terminal reward and gamma = 1, advantages become final reward minus prefix value, removing next-state bootstrapping from the advantage estimator. If a method wins only with one GAE setting, report that dependence. At lambda_GAE=0.95 the direct terminal contribution decays rapidly over a long answer; do not interpret poor early-prefix training as proof that actor features lack value information.

Before choosing a practical endpoint, run one matched warm-prior pilot for the strongest gradient configuration and LSTD. Let w_prior come from a prior iteration and keep it fixed during the current fit. Use:

    (A + eta*I) w = b + eta*w_prior
    TD direction = mean(z_t*delta_t) - eta*(w - w_prior)

Initialize gradient fitting at w_prior. This changes the regularization center for both methods, rather than merely giving one method historical initialization. Record that the previous head acts on changed actor features and must be recalibrated; do not import stale feature matrices. For strict cross-fitting use a persistent prompt-fold assignment and keep each fold's head history restricted to its own training fold. Conditional on the actor/history, current held-out rollout randomness must not enter its critic fit. Fit each fold's prior only from its own training history, not a full-batch pooled head. This preserves the practical safeguard against fitting an answer's current outcome before using its baseline.

Report cold-fit results separately even if a warm-prior variant becomes the best practical method. Allow an equal validation tuning budget for eta and gradient training effort. No cross-policy trajectory replay is needed for this prior experiment.

**Gate:** Advance shared settings and three stable primary arms. If T-large matches L, retain that finding; do not force an LSTD narrative. If every token-level variant loses to P0, diagnose calibration and approximate-advantage bias before extending the scope.

## Stage 7 — Three-seed end-to-end comparison

Run T-small, T-large, and L from the common starting actor, not sequentially from one another. Report P0 and T-MLP according to the roles above. Give each method an equal development/tuning allowance and freeze the final configuration before inspecting test results.

Choose the final rollout-token budget from pilot throughput and cluster allocation before launching all seeds; a provisional starting budget is 20 million generated training tokens per run. If pilots show that this cannot resolve learning differences, revise the budget for all arms before the final comparison. Evaluation and diagnostic tokens are separately accounted for, and their schedules are identical. State actual batch-boundary overshoot.

Evaluate periodically at common cumulative-token milestones using fixed decoding settings, a disjoint validation set, and multiple completions per prompt. Choose checkpoints on validation only; run the untouched test set with a fixed protocol. Also plot performance against elapsed GPU-hours on comparable hardware. Different generated lengths and different critic costs make equal update counts an inadequate efficiency comparison.

Required reporting:

- Held-out correctness against generated tokens and GPU-hours, final performance, and training-curve area over a common budget range.
- GPU-hours and generated tokens to a predeclared validation target, where reached; do not assign invented times to methods that fail to reach it.
- Rollout, old-log-probability/features, critic fitting, PPO update, evaluation, and checkpoint time separately.
- Peak GPU/host memory, response length, format/truncation rates, policy entropy, KL/ratio diagnostics, clip fraction, gradient norm, and failures.
- Held-out critic prediction error and calibration, raw advantage variance, and LSTD conditioning/residuals. Lower advantage variance alone is not proof of lower gradient variance or better learning; on a small fixed-checkpoint diagnostic compare gradient-estimate variability if feasible.
- Per-seed results plus uncertainty across training seeds; prompt-level resampling can quantify test-set uncertainty but does not replace training seeds.

A provisional decision rule is no more than a 1 percentage-point loss in held-out accuracy and at least a 10% GPU-hour saving at comparable performance. Declare the margins before final runs and ensure evaluation uncertainty is smaller than the claimed effect; otherwise call the result inconclusive. Select the accuracy/compute tradeoff supported by the evidence, whether that is T-small, T-large, or L. Claims apply to the tested model, data, hardware, and budgets, not a universal optimum.

## Stage 8 — Optional replay or less frequent refitting

This is a follow-on study, not necessary for the main TD-versus-LSTD result. Reusing transitions collected under one frozen actor snapshot is safe for evaluating that snapshot; reusing them after actor updates creates a policy mismatch.

If attempting cross-policy replay, store raw trajectories, behavior log probabilities, actor version, and reward/mask metadata. Refresh current-model features, introduce an explicit off-policy evaluation method, and monitor policy lag and effective sample size. Feature refresh alone does not correct policy mismatch. Do not present a FIFO buffer with plain state-value LSTD as a proven current-policy estimator.

If reducing critic refit frequency beyond the Stage 6 warm-prior pilot, treat that as a separate matched experiment and measure feature/value drift. Prior-centered fitting does not make an old head automatically calibrated to a newly updated actor.

## Checkpoint additions and implementation deliverables

Extend the existing Stage 0 checkpoint to include critic type, weights and optimizer if applicable; projection/scaler; exact feature hook; actor version; fold assignment/RNG; any trace state and accumulated statistics; raw and normalized advantage configuration; solver precision/regularization; and the phase within the rollout-fit-PPO cycle. If a phase cannot be resumed exactly, checkpoint only at explicit iteration boundaries. Save a buffer only when that variant actually uses one.

Implement a shared value-estimation interface returning pre-action values with shape [batch, response_length], plus diagnostics. TD and LSTD must plug into the same GAE/PPO path. Keep numerical reference tests separate from distributed integration tests. No GPU experiments are launched by this planning document.

## Sources and interpretation

- [PPO](https://arxiv.org/abs/1707.06347) and [GAE](https://arxiv.org/abs/1506.02438): shared actor update and advantage estimator.
- [The LLM Already Knows](https://arxiv.org/abs/2509.12886): motivation for value prediction from LLM internal states using a TD-style objective, not proof of this online token-level PPO combination.
- [POISE](https://arxiv.org/abs/2605.07579): actor-feature value estimation during RL; its own estimator is sequence-level and cross-rollout.
- [Least-Squares Policy Iteration](https://jmlr.org/papers/v4/lagoudakis03a.html): projected Bellman evaluation and LSTDQ; the plan uses state-value LSTD, not LSPI's greedy action improvement.
- [Shallow Updates](https://arxiv.org/abs/1705.07461): motivates periodic shallow fitting and a large-batch gradient control.
- [Policy Evaluation with Temporal Differences](https://jmlr.org/papers/v15/dann14a.html): distinguishes fitting objectives and supports comparison rather than an assumed LSTD advantage.
- [VERL configuration](https://verl.readthedocs.io/en/latest/examples/config.html) and [core algorithms](https://verl.readthedocs.io/en/latest/_modules/verl/trainer/ppo/core_algos.html): current orientation only. Confirm all integration behavior against the project's pinned source during implementation.
