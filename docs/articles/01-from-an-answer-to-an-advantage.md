# How a final answer teaches a language model

*Part 1 of Building a Cheaper Critic · 13 October 2026*

Suppose a language model is asked a very small question:

> I have five avocados and buy four more. Each serving of guacamole needs three avocados. How many servings can I make?

The model writes, “There are nine avocados, so we can make three servings.” We check the answer and give it a reward of one.

What, exactly, should it learn from that reward?

The reward does not say which words helped. It does not explain whether the model understood addition, guessed the answer, or copied a familiar pattern. It arrives after the whole answer. Yet training changes probabilities for individual tokens. The machinery between those two facts is the credit-assignment problem: how does an outcome at the end teach decisions made along the way?

## 1. Turn writing into a sequence of decisions

A language model chooses its next token from a probability distribution. A token may be a word, part of a word, a number, or punctuation. We call that distribution the **policy**, written \(\pi_\theta(a\mid s)\), where \(\theta\) denotes the model's parameters.

At a particular point in the answer:

| RL term | In this example |
|---|---|
| State \(s_t\) | The question and all answer tokens written so far |
| Action \(a_t\) | The next token |
| Transition | Append that token to the prefix |
| Episode | One complete attempt at answering |
| Reward | One for verified success; zero for failure |

Writing a token deterministically extends the text. Uncertainty comes from which token the policy samples next. In this simple formulation, intermediate rewards are zero; the final action receives the answer's score.

This is an episodic Markov decision process when the complete context and relevant stopping rules are included in the state. A hidden vector extracted from that context is a **representation** of the state. It is not guaranteed to preserve everything the full state contains.

## 2. Predict the future, not the correctness of a sentence

Imagine stopping the answer after “There are nine avocados.” A critic asks:

> If this particular policy continues from here, what reward should we expect?

With a binary terminal reward and no discounting, that expectation is a probability:

\[
V^\pi(s)=\mathbb{E}[R\mid s,\pi]
=P(\text{verified success}\mid s,\pi).
\]

The superscript matters. A strong model and a weak model can have different chances of completing the same prefix. Value belongs to a state **and a continuation policy**.

It is also not a sentence-level correctness score. A mathematically correct sentence may lead a weak model into a dead end. A model may recover from an earlier mistake. The critic predicts the eventual outcome, including those possibilities.

An answer verifier can check the final mathematical answer without checking every step of reasoning. Its labels are then a training signal, not a certificate that the explanation was sound. This distinction applies beyond mathematics: passing a test suite does not establish that a program is correct on every input. The reward defines what the agent is encouraged to achieve.

## 3. Two ways to teach the critic

The simplest approach waits for the outcome. If an attempt succeeds, assign target one to its observed prefixes; otherwise assign zero. Across many attempts, a predictor can learn which features are associated with success. This is **Monte Carlo value learning**.

One successful continuation does not establish that a prefix has value one. It supplies one noisy observation of its eventual return.

Temporal-difference learning instead uses a relationship between neighboring states:

\[
V^\pi(s_t)=\mathbb{E}[r_t+\gamma V^\pi(s_{t+1})\mid s_t].
\]

This is the Bellman equation. The discount \(\gamma\) sets the weight of future rewards. For this example, set \(\gamma=1\): a correct answer receives the same reward regardless of how many tokens it takes.

TD compares the current estimate with “reward now plus estimated value next”:

\[
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t).
\]

Using an estimate inside the target is called **bootstrapping**. It can reduce dependence on a distant noisy outcome, but it also lets prediction errors enter other predictions. More complicated value learning is not automatically better. The useful starting references are [Sutton and Barto, Chapters 5–6](https://www.incompleteideas.net/book/bookdraft2018mar21.pdf) and [Silver's model-free prediction lecture](https://davidstarsilver.wordpress.com/teaching/).

## 4. An advantage is a comparison with expectations

The actor needs more than a prediction. It needs a signal saying whether its sampled actions worked out better or worse than expected.

Consider a **made-up three-action episode**, grouping text into large steps only to make the arithmetic readable. Real training works at token resolution. Suppose the pre-action values are \(0.4,0.6,0.8\), and the last action succeeds. The terminal state's value is zero because no rewards remain.

| Step | Current value | Immediate reward | Next value | TD residual |
|---|---:|---:|---:|---:|
| 0 | 0.4 | 0 | 0.6 | 0.2 |
| 1 | 0.6 | 0 | 0.8 | 0.2 |
| 2 | 0.8 | 1 | 0 | 0.2 |

Generalized Advantage Estimation, or **GAE**, combines these residuals backward:

\[
\widehat A_t=\delta_t+\gamma\lambda_{\mathrm{GAE}}\widehat A_{t+1}.
\]

For \(\lambda_{\mathrm{GAE}}=0.95\), the advantages are \(0.5705,0.39,0.2\). If the same hypothetical value sequence ended in failure, the last residual would instead be \(-0.8\), and the advantages would be \(-0.332,-0.56,-0.8\).

These are learning signals, not proofs that a particular token caused success or failure. The final outcome changes how favorable the sampled trajectory looks relative to its baseline. The [GAE paper](https://arxiv.org/abs/1506.02438) develops this bias–variance trade-off.

## 5. PPO changes the actor cautiously

Positive advantages encourage sampled actions; negative advantages discourage them. But changing a policy too far using one batch can make the data a poor guide to its new behavior.

PPO compares the new probability of an action with the probability under the policy that generated it:

\[
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)}.
\]

Its clipped objective limits the incentive to push that ratio farther in an already favorable direction:

\[
L(\theta)=\mathbb{E}\!\left[
\min\left(\rho_t\widehat A_t,
\operatorname{clip}(\rho_t,1-c,1+c)\widehat A_t\right)\right].
\]

Here \(c\) is a clipping threshold. Clipping is not a hard guarantee that the whole policy stays within a fixed distance. It is a practical objective used with minibatches and limited optimization epochs. [PPO paper](https://arxiv.org/abs/1707.06347)

The remaining design question concerns the source of \(V\), which supplies the advantages. Must it come from another large, trainable transformer?

## 6. Reuse the understanding, simplify the prediction

The actor's transformer already turns a prefix into a rich hidden vector. We can extract features \(\phi(s)\) and fit:

\[
V_w(s)=w^\top\phi(s).
\]

This function is linear in its fitted weights. It is not linear in the words: the transformer has already performed nonlinear computation.

A small head is attractive because its weights can be trained without backpropagating through the feature encoder. It is not automatically accurate: the chosen representation must expose information about eventual success, and the training examples must cover the states the actor visits. Extracting features still costs a forward pass.

There is also a moving target. Once the actor learns, its chance of succeeding from a given prefix can change. A critic trained on yesterday's policy may be systematically pessimistic—or optimistic—about today's policy.

## 7. Ask two different evaluation questions

First ask whether the critic predicts outcomes on held-out questions. For binary rewards, the Brier score is the mean squared difference between predicted probability and outcome. A prediction of 0.8 on a failed answer incurs error 0.64; a prediction of 0.2 incurs 0.04. Accuracy alone loses this confidence information.

Then ask whether the critic actually helps policy learning. Run otherwise comparable actors with different critics and measure final task performance and total cost. A predictor can have lower average error while making its largest mistakes exactly where the policy needs a useful advantage estimate. Offline prediction and online control are related tests, not interchangeable ones.

A useful comparison also includes an approach without a learned value head. GRPO, for example, constructs relative learning signals from groups of sampled answers. That avoids critic training but changes how experience is collected and used. [DeepSeekMath](https://arxiv.org/abs/2402.03300)

The full path is now visible: a final reward supplies evidence about a trajectory; values describe expectations; GAE converts deviations from those expectations into advantages; PPO uses those advantages to adjust token probabilities. The critic is one replaceable component of that path.

**Further reading:** [GAE](https://arxiv.org/abs/1506.02438), [PPO](https://arxiv.org/abs/1707.06347), and [the worked project study](04-building-a-cheaper-ppo-critic.md). For one way to fit the value head, see [A critic built from accumulated matrices](02-a-critic-built-from-matrices.md).

If a language model already computes useful features while answering, how much additional learning is really necessary to turn those features into a reliable critic?
