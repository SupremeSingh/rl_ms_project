# How do critics compare when the policy, adaptation answers and reference values are shared?

**Not measured yet: this experiment freezes one completed actor, fits the
critics on identical new answers, and uses one common Monte Carlo reference
bank. The main budget is 500 training questions, 50 test questions and up to
25,600 shared reference continuations.**

The arms are token/chunk SLICE + Ridge, token/chunk RMS-normalized two-thirds +
Ridge, a full MC-trained transformer critic, and GRPO's question baseline.
The actor is never optimized. Token and chunk arms share answer-level labels;
their fitted-state weighting remains an explicit difference.

The transformer is fitted to the same observed final returns as Ridge, not to
GAE targets. It is therefore a controlled value-regression benchmark rather than
a PPO training comparison. The prior six-actor audit is retained separately.

## Evidence and scope

[Protocol, limitations and run commands](../docs/FIXED_POLICY_CRITICS.md). Planned
main output: `outputs/fixed-policy-critics-v1`. One fixed source policy, one
transformer fitting seed, and reused inspected Numina questions; no new results
or method-superiority claim yet.
