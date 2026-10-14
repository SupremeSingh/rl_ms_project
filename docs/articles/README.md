# Building a Cheaper Critic

Four connected explainers, written from first principles for a reader with basic
probability, linear algebra and some familiarity with machine learning. The first three are standalone educational articles with worked examples and open
research questions. The fourth connects their ideas through our actual experiment.

| Article | Central question |
|---|---|
| [1. How a final answer teaches a language model](01-from-an-answer-to-an-advantage.md) | How does a terminal reward become a token-level PPO update? |
| [2. A critic built from accumulated matrices](02-a-critic-built-from-matrices.md) | What do LSTD's matrices contain, and what does accumulation preserve? |
| [3. Reading a model through J-lens](03-reading-a-model-through-jlens.md) | How can derivatives make internal representations readable? |
| [4. Building a cheaper PPO critic: the experiment so far](04-building-a-cheaper-ppo-critic.md) | What did our accumulated-critic experiments establish, and what is the J-lens study testing? |

[Open J-lens 101](jlens-companion.ipynb): a minimal, three-hidden-layer word
predictor learns four country–capital facts. Read its early activations directly,
calculate an average Jacobian, then compare the transported readout. Runs on CPU
with PyTorch and Matplotlib; no downloads, cluster artifacts or official-package
installation. This is a teaching analogue, not an LLM or a paper replication.

## Reading and evidence conventions

The explanatory structure takes inspiration from
[Non-linear Value Learning](https://dillonmsandhu.github.io/deep/rl/2026/09/25/nonlinear-value-learning.html):
one small example, explicit equations and comparisons that distinguish similar
algorithms. The prose and worked examples here are original.

- Toy vectors and values are labeled illustrative, never reported as model measurements.
- Project results come from the [repository record](../../README.md) and the
  reports supplied during the project. Historical test questions were inspected.
- The 104,000-answer [large J-lens study](../large-jlens-study.md) was still
  generating data at the last reported checkpoint. These articles contain no
  invented results from it.
- The paper's findings, our experiments and proposed Lean work are separate claims.
- Statistical intervals are reported with their exploratory limitations.

## Pair the series with study

Read Article 1 with Silver lectures 4, 6 and 7; Article 2 with LSTD and eligibility
traces; revisit Article 1 after GAE and PPO. Read Article 3 while running the
notebook, then Article 4 to connect the theory with the experimental results and training loop.

Primary resources: [Silver's lectures](https://davidstarsilver.wordpress.com/teaching/),
[Sutton–Barto](https://www.incompleteideas.net/book/bookdraft2018mar21.pdf),
[LSTD](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/BradtkeBarto-lstd-1996),
[LSPI](https://jmlr.org/papers/v4/lagoudakis03a.html),
[GAE](https://arxiv.org/abs/1506.02438), [PPO](https://arxiv.org/abs/1707.06347),
[J-space](https://transformer-circuits.pub/2026/workspace/index.html), and
[reliable RL evaluation](https://arxiv.org/abs/2108.13264).

Articles dated 13 October 2026. Publication drafts; external publication has not
been performed. The companion notebook is purely educational; its toy outputs
are not project measurements.
