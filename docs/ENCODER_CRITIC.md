# Phase v5: official LLM2Vec features for SLICE + chunk PPO

**Question: does a bidirectional encoder give the same compact chunk critic
better features for value estimation and PPO?** No results yet. This foundation
adds isolated dependency setup, prefix-only feature extraction and sanity gates.
It does not yet adapt the encoder, train a new SLICE or launch encoder PPO.

## Preserved algorithm

The actor remains the causal Qwen model. The planned critic is a separate copy
converted/adapted with the unmodified official LLM2Vec package, then frozen.
Retrain SLICE and its training-only normalization for that encoder. During PPO,
keep encoder/compressor/normalization fixed, use one shared cumulative Ridge
system, accumulate raw A,b without sample-count division, solve with epsilon
.01, and keep fresh chunk-GAE PPO (gamma 1, GAE lambda .95). No old decoder SLICE
checkpoint or accumulated statistics can be reused for the new coordinates.

Official sources: [package](https://github.com/McGill-NLP/llm2vec),
[Qwen2 implementation](https://github.com/McGill-NLP/llm2vec/blob/main/llm2vec/models/bidirectional_qwen2.py),
[paper](https://arxiv.org/abs/2404.05961).
Loading bidirectional weights is only the attention conversion, not proof of
completed MNTP/SimCSE adaptation or useful features. Both official adaptation
stages remain future work, before learning a new SLICE. No upstream code is patched.

## Prefix boundary and feature contract

Before chunk starting at response token p, feed exactly original question IDs
plus answer IDs `[:p]`. Before the first chunk, p=0. Current/future chunks,
reference answers and rewards are never input features. Final termination value
is zero rather than an encoder call. Each prefix is a separate padded batch row,
with its own mask and positions; do not concatenate prefixes or reuse causal KV
caches. No full-answer bidirectional forward followed by intermediate-state slicing.

The adapter captures the last **available** token in selected blocks before final
norm, applies per-block RMS and concatenates in the existing block order
[27,26,24,20,12] for the 28-block model. This preserves the 7,680-coordinate input
format, not the old numerical feature basis. A fresh SLICE learns 768 features.
Last-token pooling avoids changing both attention and pooling in the first
comparison. This uses the official backbone directly rather than the high-level
text `encode` API, preserving original action IDs and avoiding its default
512-token text truncation. All model attention code is the official implementation.

**Documented boundary limitation:** the existing MATH segmenter examines the
completed answer when selecting boundaries. Truncation prevents feature access
to future tokens, but boundary selection itself may depend on later text. Keep
the identical chunker in decoder and encoder arms for the initial comparison;
this controls a design choice, not proof it is harmless. Fixed-token chunking
is a later control if we see a meaningful benefit. Suffix-invariance sanity
checks hold supplied boundaries fixed; they do not claim segmentation invariance.

## Important backend finding

On a tiny randomly initialized Qwen2 model, **unmodified llm2vec 0.2.3 with
Transformers 4.44.2 gave zero first-token sensitivity to a changed later token
with both eager and SDPA**; the causal control also gave zero. Thus the
`is_causal=False` flag alone is insufficient. These backends did not demonstrate
bidirectionality in this local test. Do not use them for an encoder comparison.

The official CUDA FlashAttention-2 backend is the next check. It is **not GPU
validated locally**. The full sanity script requires actual earlier-token
sensitivity, causal-control invariance, suffix exclusion and padding/batch
invariance before `passed:true`. It stops without patching the package if the
backend fails. A failed gate means no compressor training/PPO launch.

FlashAttention requires BF16 here. A future causal control must match encoder
precision/backend; comparing old FP32 decoder results to BF16 encoder results
would conflate precision with encoder effects. Record encoder/model identities,
versions, adaptation data and precision when sealing the next experiment.

## Cluster sanity commands

Push v5 from your Mac, then on the cluster, once jobs using the checkout finish:

```bash
cd /usr/xtmp/ms785/rl_ms_project
git switch v5
git pull --rebase origin v5
source scripts/cluster_env.sh
srun -p compsci-gpu --gres=gpu:a5000:1 --cpus-per-task=4 --mem=32G \
  --time=00:45:00 --pty bash -i
source scripts/cluster_env.sh
export MATH_RL_GPU=1
bash scripts/container_exec.sh bash tools/encoder/setup.sh

# Tiny random models; no model download or GPU use. Prints backend behavior.
bash scripts/container_exec.sh "$PWD/.venv-encoder/bin/python" \
  tools/encoder/check_backend.py

# Actual Qwen checkpoint, official FlashAttention-2, prefix-only feature checks.
bash scripts/container_exec.sh "$PWD/.venv-encoder/bin/python" \
  tools/encoder/sanity_llm2vec.py --model "$PWD/models/qwen-math" \
  --device cuda --attention flash_attention_2 --dtype bf16 \
  --out outputs/encoder-sanity-v1
cat outputs/encoder-sanity-v1/report.json
exit
```

Use a new output directory for every attempt. `.venv-encoder` is separate from
`.venv` and `.venv-critic`; setup refuses overwrites and checks inherited torch
identity. LLM2Vec/Transformers/PEFT are pinned, with no dependency upgrades to the
PPO environment. Inherited VERL/vLLM distributions may declare incompatible
requirements; do not import or run them in this encoder environment. Exchange
feature artifacts with the PPO environment in the next phase. The existing
container must already supply FlashAttention; do not silently compile/replace it.

The final sanity additionally tests the existing Bottleneck and one-head Ridge
interfaces on tiny synthetic data: random 768-wide compressor, 769-dimensional
system, two consecutive updates with retained counts. This is plumbing only,
**not trained SLICE or a prediction-quality comparison**. It writes report.json
even on failure. Share that report before adapting Qwen or launching a study.

## Local validation

`python3 -m pytest -q tests/test_encoder_features.py` tests physical prefix
truncation, unseen suffix exclusion, padding independence, hook cleanup, context
checks, frozen-state requirements and the bidirectionality gate. CPU tests do not
replace the actual-checkpoint CUDA gate or measure RL quality.
