# How accurately do our critics estimate a partial answer's chance of success?

**Not measured yet: the new six-arm diagnostic will compare saved values with
independent Monte Carlo continuations from each actor. It starts with a small
integration smoke and a 10-question pilot before an optional 50-question,
256-continuation-per-state evaluation.**

The six arms are token/chunk SLICE + cumulative Ridge, token/chunk RMS-normalized
two-thirds + cumulative Ridge, conventional PPO, and GRPO's question baseline
(GRPO has no critic). All panels use shared held-out Numina prefixes, independent
reference outcomes and unclipped predictions. Each actor defines its own value
target; this is a one-seed diagnostic, not a universal method ranking.

The normalized-layer arms require fresh fitting; the older chunk comparison's
raw-layer heads cannot be relabeled as normalized heads. No MC outcome is used
for fitting or model selection. Training saves restorable actor/critic weights,
and the report records the cumulative head's one-update policy lag.

## Evidence and scope

[Protocol, uncertainty and cluster commands](../docs/VALUE_AUDIT.md). Planned
output: `outputs/numina-value-audit-v1`. There are no experimental results to
interpret until the corresponding stages complete.
