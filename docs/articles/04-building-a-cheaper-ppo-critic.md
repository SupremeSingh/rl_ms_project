# Putting the pieces together: toward a cheaper PPO critic

*Part 4 of Building a Cheaper Critic · 14 October 2026*

A language model already performs a great deal of computation before choosing its next token. It interprets the question, relates it to the answer so far, and constructs an internal representation. Could a small value estimator use that computation to help train the model, without repeatedly optimizing another large network?

That is the organizing question of this project. This article develops the proposed system from its parts, rather than reporting experiments. The ambition is to reduce the cost of obtaining useful advantages while retaining PPO as the policy optimizer.

[Article 1](01-from-an-answer-to-an-advantage.md) explains how rewards become advantages. [Article 2](02-a-critic-built-from-matrices.md) explains how linear critics can be fitted from accumulated statistics. [Article 3](03-learning-compact-value-features.md) explains how a learned bottleneck can make rich hidden representations manageable. Together, they suggest separating expensive representation learning from repeated value fitting.

## 1. One answer, three different jobs

Imagine an actor answering a question about sharing apples equally. The state \(s_t\) is the question and the answer prefix; action \(a_t\) is the next token. A completed answer receives an outcome reward. Earlier transitions may receive zero reward.

The actor asks, “Which token should I generate?” The critic asks, “How much reward should I expect if this policy continues?” The policy optimizer asks, “Was this sampled continuation better or worse than expected?”

These jobs are related but distinct. A fluent next-token prediction need not imply a correct solution. A useful value estimate need not reproduce the actor's entire language-modeling computation. And a correct final answer does not tell us exactly which individual token deserves credit.

PPO uses estimated advantages to turn those trajectories into updates. Replacing the critic changes how we estimate those advantages, not the identity of the policy optimizer.

## 2. Build a stable description of each prefix

Let a frozen encoder produce hidden representations \(h_{\ell}(s)\). A fixed feature map can read a single layer or combine several through a trained bottleneck:

\[
x(s)=[h_{\ell_1}(s);\ldots;h_{\ell_K}(s)],
\qquad \phi(s)=[g_\psi(x(s));1].
\]

Here \(g_\psi\) may be a normalized projection followed by ReLU. It may also be the identity or a fixed linear compression. The important condition for accumulation is that the encoder, preprocessing and \(\psi\) remain fixed while their statistics are reused.

The critic is

\[
\widehat V_w(s)=w^\top\phi(s).
\]

This is a small linear fit on top of a potentially powerful nonlinear description. Richness comes from the features; repeated value fitting changes only \(w\).

If the actor is changing while the encoder stays frozen, those are different parameter sets. Frozen-encoder inference may require another forward pass. The representation is not automatically free merely because it originated in the actor's architecture.

## 3. Turn fresh experience into a critic

For each sampled answer, record transitions \((s_t,a_t,r_t,s_{t+1})\). Use a zero next-state feature at genuine termination. For an artificial length cutoff, decide explicitly whether the task ends there or whether bootstrapping is required; the two interpretations produce different targets.

LSTD with eligibility traces accumulates

\[
e_t=\gamma\lambda_c e_{t-1}+\phi_t,
\]
\[
A\leftarrow A+e_t(\phi_t-\gamma\phi_{t+1})^\top,
\qquad b\leftarrow b+e_t r_t.
\]

Reset \(e_t\) between episodes. Then solve

\[
(A+\epsilon I)w=b.
\]

For the apples example, a successful completion contributes its terminal reward to the right-hand side through the accumulated trace. Earlier transitions also enter the matrix describing how current and next-state features relate. The solve seeks values consistent with those relationships.

Ridge provides a useful alternative: accumulate \(\sum_t\phi_t\phi_t^\top\) and \(\sum_t\phi_t G_t\), where \(G_t\) is the observed return. Ridge fits outcomes directly; LSTD fits TD consistency. Neither should be assumed to produce better policy updates merely from its definition.

## 4. Why keep matrices across updates?

Each new batch can add to \(A\) and \(b\). For fixed features and targets defined by the same rules, these sums preserve the information needed for the corresponding least-squares system without storing every past feature vector.

We accumulate statistics, not fitted weights. Adding two independently fitted weight vectors is not equivalent to fitting their combined data.

With raw sums and fixed \(\epsilon\), regularization becomes weaker relative to the accumulated data. If \(n\) transitions have contributed, dividing the equation by \(n\) gives

\[
\left(\frac{A}{n}+\frac{\epsilon}{n}I\right)w=\frac{b}{n}.
\]

That can be useful, but it is a design choice, not a guarantee of stability. LSTD's matrix need not be symmetric or positive definite. Numerical residuals, conditioning and prediction behavior still need checking.

There is also a statistical limitation: old trajectories came from old actor policies. Accumulation produces a mixture of historical-policy data. A frozen encoder fixes coordinates; it does not make old experience on-policy. An uncorrected cumulative critic is therefore an approximation to the current policy's value, not a guaranteed current-policy TD fixed point. Forgetting, resets or principled off-policy corrections are alternative designs with their own trade-offs.

## 5. Keep the baseline from learning the answer it is judging

Suppose the critic sees an answer's final reward, fits to that answer, and immediately supplies its advantages. It may partly memorize the outcome rather than provide an independent expectation. Token-level prediction does not remove this dependence: every prefix belongs to the same rewarded continuation.

One safeguard is question-level cross-fitting. Assign each question permanently to a fold. Fit a critic on one fold and use it to value trajectories in the other. Persistent assignment matters when accumulating statistics across updates; moving a question between folds would reintroduce its historical rewards.

For supervised feature learning, the same principle extends upstream. A bottleneck trained using a question's outcome can leak information even if the final linear head excludes that question. A separately trained, disjoint representation or fold-specific feature training can avoid this direct label reuse. Cross-fitting does not eliminate all dependence from a shared, evolving actor, but it addresses the direct reuse of a trajectory's reward to fit its baseline.

## 6. Values become advantages; advantages change the actor

After fitting the critic, compute values for the current batch and hold them fixed during the actor update. The TD residual is

\[
\delta_t=r_t+\gamma\widehat V(s_{t+1})-\widehat V(s_t).
\]

GAE combines those residuals:

\[
\widehat A_t=\sum_{j\ge0}(\gamma\lambda_a)^j\delta_{t+j}.
\]

The parameter \(\lambda_a\) controls the advantage estimate. It is separate from \(\lambda_c\), which controls the critic's LSTD traces. They need not be equal.

For a toy final transition, suppose the predicted value is \(0.4\), success gives reward one, and the next state is terminal. Its residual is \(1-0.4=0.6\). Failure instead gives \(-0.4\). Earlier tokens receive information through the GAE recursion; this is statistical credit assignment, not proof that a particular token caused the outcome.

PPO forms the probability ratio

\[
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)}
\]

and maximizes the clipped surrogate

\[
\mathbb E_t\left[
\min\left(\rho_t\widehat A_t,
\operatorname{clip}(\rho_t,1-\eta,1+\eta)\widehat A_t\right)
\right].
\]

The actor still needs gradients. The savings we seek concern the critic's repeated optimization, not removal of policy learning. [GAE](https://arxiv.org/abs/1506.02438) and [PPO](https://arxiv.org/abs/1707.06347) describe these distinct parts of the update.

## 7. The complete algorithm: learn the filter, accumulate the critic

We can think of the bottleneck as a **learned noise filter**: it tries to preserve information useful for predicting success while compressing the rest. “Noise” here means variation that is unhelpful for this prediction task. The network is not given a list of irrelevant coordinates, and neither compression nor sparsity guarantees that only irrelevant information is removed.

There are two training stages. First, learn the filter using a separate set of questions. Then freeze it and use its output as the critic's coordinates throughout PPO. The separate representation-training questions keep their outcome labels out of the questions used for the online cross-fitted critics.

### Stage A: learn and freeze the noise filter

```text
GIVEN:
    a frozen language-model encoder E
    a chosen set of encoder layers L
    representation-training questions, separate from PPO questions
    compressed width m and sparsity strength beta

COLLECT answers from a fixed policy on representation-training questions
LABEL each answer with its final success or failure
SAMPLE causal prefixes: question + only the answer tokens already written
SPLIT training and validation data by question

FOR each prefix s:
    x(s) = CONCATENATE normalized hidden vectors from layers L of E(s)
    y(s) = final outcome of that prefix's sampled continuation

FIT input centering and scale on training prefixes only
INITIALIZE bottleneck parameters W, bias and temporary readout a, c

REPEAT for the representation-training budget:
    x, y = a minibatch of training prefixes and outcome labels
    x = APPLY the fixed training centering and scale
    W_unit = NORMALIZE each row of W to unit length
    z = ReLU(W_unit @ x + bias)                # compressed features
    p = sigmoid(a @ z + c)                    # temporary success predictor
    loss = binary_cross_entropy(p, y) + beta * mean(z)
    UPDATE W, bias, a, c by gradient descent

SELECT the filter using validation data, not test outcomes
FREEZE E, the bottleneck and all preprocessing
DISCARD the temporary readout a, c

DEFINE FEATURES(s):
    x = CONCATENATE normalized selected-layer vectors of E(s)
    x = APPLY the frozen training centering and scale
    z = ReLU(W_unit @ x + bias)
    RETURN CONCATENATE(z, 1)                   # include a value intercept
```

The plain bottleneck sets `beta = 0`; the sparse version uses a positive penalty. Both compress to the same width. The penalty encourages economical activation, not sparse weights or orthogonal coordinates. The temporary sigmoid head teaches the filter what to preserve; it is **not** the final LSTD value head.

### Stage B: accumulate LSTD statistics and update the actor with PPO

Let `k = m + 1`, including the intercept. Each fold stores a `k × k` matrix and a length-`k` vector. To keep terminal handling unambiguous, the pseudocode below assumes complete episodes: a cutoff is included as terminal only if the task explicitly defines it that way.

```text
GIVEN:
    an actor policy pi, the frozen FEATURES function
    discount gamma, critic trace lambda_c, GAE trace lambda_a
    a small positive regularizer epsilon

ASSIGN each PPO question permanently to fold 0 or fold 1
FOR f in {0, 1}:
    A[f] = ZERO_MATRIX(k, k)
    b[f] = ZERO_VECTOR(k)

FOR each policy update i:
    pi_old = SNAPSHOT of the current actor
    D = SAMPLE fresh complete answers using pi_old
    RECORD old action log-probabilities and final outcome rewards
    SET intermediate rewards to zero for this terminal-reward task

    FOR f in {0, 1}:
        A_update[f] = ZERO_MATRIX(k, k)
        b_update[f] = ZERO_VECTOR(k)

    FOR each answer in D:
        f = the question's permanent fold
        e = ZERO_VECTOR(k)                    # reset between answers
        phi[t] = FEATURES(prefix before token t), for every action t

        FOR each consecutive token transition t:
            next_phi = phi[t+1] if another action follows
                       else ZERO_VECTOR(k)   # terminal intercept is zero too
            e = gamma * lambda_c * e + phi[t]
            A_update[f] += OUTER(e, phi[t] - gamma * next_phi)
            b_update[f] += e * reward[t]

    FOR f in {0, 1}:
        A[f] += A_update[f]                   # keep historical raw sums
        b[f] += b_update[f]                   # no division by data count
        w[f] = SOLVE(A[f] + epsilon * IDENTITY(k), b[f])

    FOR each answer in D:
        f = the question's fold
        V[t] = DOT(w[1-f], phi[t])            # use the OTHER fold's critic
        V[T] = 0                             # no value after termination
        advantage[T] = 0
        FOR t from T-1 down to 0:
            delta = reward[t] + gamma * V[t+1] - V[t]
            advantage[t] = delta + gamma * lambda_a * advantage[t+1]

    DETACH advantages from all critic computations
    UPDATE pi with PPO's clipped objective on D and old log-probabilities
    KEEP A, b, the encoder and the filter for the next update
```

This formalizes the accumulation idea: **experience accumulates in `A` and `b`; weights are solved again from those accumulated statistics.** We do not add `w_new` to `w_old`, and we do not repeatedly add regularization to the stored matrix. The solve uses the same `epsilon * I` each time, so its strength relative to the growing raw sums decreases.

For the apples example, a trajectory from fold 0 contributes to `A[0]` and `b[0]`. Its own advantages use `w[1]`, which was fitted from fold 1. On a later update, another fold-0 answer adds new statistics to the same system. The encoder and filter retain their meaning throughout, even as the actor changes its answers.

The critic trace `lambda_c` determines how rewards and transitions enter the fit. The GAE trace `lambda_a` determines how the fitted values become actor advantages. Both can propagate information backward through an answer, but they act at different stages and are independently chosen.

There is no replay of old actions in the PPO update. Historical information remains in the critic's matrices, including the historical-policy mismatch discussed above. Changing the nonlinear filter or encoder requires rebuilding or discarding those matrices. This algorithm describes how the pieces can be integrated; it does not establish that the learned filter improves PPO or that accumulating off-policy experience is always beneficial.

## 8. What would make PPO genuinely faster?

A smaller linear system is only one part of the budget. For feature dimension \(m\), dense matrices require \(O(m^2)\) storage; accumulating outer products over \(T\) transitions takes roughly \(O(Tm^2)\) work, and a dense solve costs \(O(m^3)\). Feature extraction, communication and bottleneck inference add their own costs.

The relevant comparison includes rollout generation, encoder passes, critic fitting and actor updates. A method that fits values cheaply but produces poor advantages may need many more updates. A more expensive critic can still be preferable if it reaches a target quality sooner.

The useful objective is therefore time or compute to reach a specified policy quality, together with stability across runs. Offline correctness prediction is an intermediate diagnostic. It cannot establish that replacing the critic improves the policy or lowers total training cost.

The project's broader vision is to reuse a language model's representations, compress the information useful for value estimation, and update a small critic through accumulated experience. Could that separation make PPO substantially more economical while preserving the quality of the decisions it learns?
