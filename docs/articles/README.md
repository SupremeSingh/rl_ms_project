# Building a Cheaper Critic

Four connected explainers, written from first principles for a reader with basic
probability, linear algebra and some familiarity with machine learning. The first three are standalone educational articles with worked examples and open
research questions. The fourth connects their ideas into a theoretical design for a cheaper PPO critic.

| Article | Central question |
|---|---|
| [1. How a final answer teaches a language model](01-from-an-answer-to-an-advantage.md) | How does a terminal reward become a token-level PPO update? |
| [2. A critic built from accumulated matrices](02-a-critic-built-from-matrices.md) | What do LSTD's matrices contain, and what does accumulation preserve? |
| [3. Learning compact features for a linear critic](03-learning-compact-value-features.md) | How do learned compression and sparsity make rich representations manageable? |
| [4. Putting the pieces together: toward a cheaper PPO critic](04-building-a-cheaper-ppo-critic.md) | How can frozen features, accumulated statistics and PPO work together? |

Optional earlier interpretability companion: [Open J-lens 101](jlens-companion.ipynb): a minimal, three-hidden-layer word
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
- Articles 3 and 4 explain mechanisms and hypotheses without experimental results.
- Learned sparse bottlenecks are distinguished from the geometric RL paper's exact layer.
- The system design does not establish a speedup or a current-policy convergence guarantee.
- See the [repository record](../../README.md) for experimental evidence and limitations.

## Pair the series with study

Read Article 1 with Silver lectures 4, 6 and 7; Article 2 with LSTD and eligibility
traces; revisit Article 1 after GAE and PPO. Read Article 3 alongside the geometric RL paper, then Article 4 to connect
representation learning with the policy-training loop. The J-lens notebook remains
an optional, separate interpretability exercise.

Primary resources: [Silver's lectures](https://davidstarsilver.wordpress.com/teaching/),
[Sutton–Barto](https://www.incompleteideas.net/book/bookdraft2018mar21.pdf),
[LSTD](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/BradtkeBarto-lstd-1996),
[LSPI](https://jmlr.org/papers/v4/lagoudakis03a.html),
[GAE](https://arxiv.org/abs/1506.02438), [PPO](https://arxiv.org/abs/1707.06347),
[Geometry of Neural RL](https://cs.brown.edu/people/gdk/pubs/geometry_neural_rl.pdf), and
[reliable RL evaluation](https://arxiv.org/abs/2108.13264).

Articles 1–2 dated 13 October 2026; articles 3–4 revised 14 October 2026. Publication drafts; external publication has not
been performed. The companion notebook is purely educational; its toy outputs
are not project measurements.
