# Does an LLM2Vec-adapted Qwen encoder improve SLICE correctness predictions?

**Not measured yet: the offline Numina study compares the original causal Qwen,
an unadapted bidirectional copy and a bidirectional copy adapted with MNTP and
contrastive learning. Each has raw-layer, RMS-layer and SLICE Ridge readouts,
trained at token and chunk prefixes, with identical held-out evaluation states.**

The actor remains frozen; this does not run PPO. There are 18 configurations,
three SLICE seeds and one adaptation seed. The bounded default samples up to eight
states per answer per unit; it does not claim exhaustive all-token fitting.

## Evidence and scope

[Design, caveats and isolated cluster commands](../docs/ENCODER_STUDY.md).
Planned outputs: `outputs/numina-encoder-smoke-v1` and `outputs/numina-encoder-v1`.
No performance conclusion is available before evaluation completes.
