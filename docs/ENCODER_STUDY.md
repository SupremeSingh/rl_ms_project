# Offline Numina encoder comparison

**Question: does adapting an exact copy of Qwen into a bidirectional encoder
make its frozen features more useful for eventual-correctness prediction?**

No results yet. This is offline supervised prediction, not PPO or a Monte Carlo
value audit. Every arm sees responses sampled once from the same unchanged
Qwen2.5-Math-1.5B actor, with the same numeric outcome verifier and question splits.

## Design

| Backbone | Attention | Adaptation |
|---|---|---|
| `decoder` | Original causal | None |
| `bidir-unadapted` | Bidirectional within the supplied prefix | None; isolates the mask change |
| `llm2vec` | Bidirectional within the supplied prefix | MNTP, then unsupervised dropout contrastive training |

Each backbone is evaluated with **raw two-thirds + Ridge**, **RMS-normalized
two-thirds + Ridge**, and **new sparse SLICE-768 + Ridge**, each trained on
**token-selected** or **chunk-selected** prefixes. There are 18 reported
configurations. SLICE uses three compressor seeds (42–44); adaptation uses one
seed (42). Raw/RMS Ridge fits are deterministic and are not counted as three seeds.

All backbones start from the exact same local actor checkpoint, not Alibaba GTE
weights. The actor, tokenizer and its vocabulary never change. The decoder SLICE
baseline is trained afresh on the matched training budget; it is the established
architecture, not the earlier study's fitted compressor or a reproduction of its
online accuracy numbers.

## LLM2Vec-style conversion

This bounded Numina adaptation follows the three ideas in
[LLM2Vec](https://arxiv.org/abs/2404.05961) and its
[MNTP](https://github.com/McGill-NLP/llm2vec/blob/main/experiments/run_mntp.py) /
[contrastive](https://github.com/McGill-NLP/llm2vec/blob/main/experiments/run_simcse.py)
implementations. It is **not the original paper's corpus or hyperparameter replication**.
There is no new dependency on the upstream package or change to the installed
actor-training stack.

- Explicit 4-D eager-attention masks allow attention throughout each available
  prefix. A numerical probe must show that changing a later prefix token changes
  an earlier bidirectional state, while leaving its native causal counterpart fixed.
- Rank-8 low-rank adapters (alpha 16) update attention and MLP projections; original
  weights stay frozen. Adapter parameters/optimizer states use FP32. Learning rate
  is fixed at 1e-4, with gradient clipping and non-reentrant gradient checkpointing.
- MNTP replaces selected tokens (probability .2) with the existing EOS sentinel;
  the original token at position t is predicted from hidden position t−1. There is
  no vocabulary resize and no unchanged-target shortcut.
- Contrastive training compares two attention-dropout (.1) views of the same text,
  using mean-pooled normalized embeddings, in-batch negatives from four distinct
  training questions, symmetric cross-entropy and temperature .05.

Adaptation uses one midpoint answer prefix per **training question**, bounded to
512 total tokens. Outcome labels are not used for adaptation. No validation/test
text enters MNTP or contrastive training. The adapter is then frozen before any
SLICE or Ridge fitting. Fixed short training budgets do not establish convergence.

## Prefixes, fitting and evaluation

Default extraction is bounded: sample up to **eight token prefixes and eight
MATH-chunk starts per answer**, without conditioning on correctness. Chunk fallback
is 32 original tokens. These are sampled token/chunk comparisons, **not all-token
fitting**. `--all-prefixes` is available, but can greatly increase runtime and disk.

Training uses each variant's own selected states. Validation and test use the
**same union of those token/chunk positions for every configuration**, avoiding an
accuracy comparison between different test-state distributions. An encoder sees
only question + tokens strictly before the action; no padding or dummy future
action enters bidirectional extraction. The causal control uses one efficient
whole-answer pass; causal masking makes its selected states equivalent to separate
truncated-prefix passes. Bidirectional extraction must recompute each prefix.

SLICE concatenates RMS-normalized states from blocks 28,27,25,21,13; fits training-only
centering/scalar scaling; and learns a row-normalized ReLU bottleneck with BCE + .01
mean activation penalty. Its epoch is selected by equal-question validation Brier,
then frozen. The two-thirds control is block 19. Ridge uses epsilon .01, FP64
statistics, an intercept and a matrix solve. There is no output LayerNorm, clipping,
online accumulation across PPO updates, or LSTD in this study.

Reports include **equal-question threshold-.5 prediction accuracy**, raw Brier,
out-of-range predictions, per-seed results and exploratory paired question-bootstrap
Brier differences. These measure whether a prefix predicts its sampled answer's
eventual verifier outcome, **not actor problem-solving accuracy or MC-measured true
state values**. Questions were inspected in previous Numina studies. Intervals
average compressor seeds and omit adaptation-seed uncertainty; no universal ranking
or superiority claim follows.

Extraction timing and peak allocated GPU memory are recorded per backbone on the
shared prefix union. They do not separate token versus chunk timing or constitute
a complete allocation ledger. Cache-hit resumes are explicitly labeled. Adaptation
time is additional; fewer head coordinates do not remove encoder forward costs.

## Run without disturbing the active value audit

Push the `encoder` branch from your Mac first. **Do not pull or reinstall into the
checkout currently running the value audit.** Create a separate Duke worktree and
reuse its installed environments read-only:

```bash
cd /usr/xtmp/ms785/rl_ms_project
git fetch origin encoder
git worktree add --detach /usr/xtmp/ms785/rl_ms_encoder origin/encoder
cd /usr/xtmp/ms785/rl_ms_encoder
ln -s /usr/xtmp/ms785/rl_ms_project/models models
ln -s /usr/xtmp/ms785/rl_ms_project/.venv .venv
ln -s /usr/xtmp/ms785/rl_ms_project/.venv-ppo-verifier .venv-ppo-verifier
```

No installation or new model download is needed. The launcher sets an explicit
PYTHONPATH for the new worktree. Its outputs and manifests are separate from the
running audit. Model files and shared environments must remain unchanged.

Start with the complete integration smoke. `--cpu-on-gpu` routes preparation and
reporting through the GPU partition too, avoiding the CPU-partition queue:

```bash
source scripts/cluster_env.sh
python3 scripts/submit_encoder_study.py --mode smoke \
  --source /usr/xtmp/ms785/rl_ms_project/outputs/numina-no-folds-v1 \
  --out outputs/numina-encoder-smoke-v1 --concurrency 3 --cpu-on-gpu
```

The chain is prepare → generate → MNTP → contrastive → three encoder extraction
tasks → three fit tasks → evaluate. Every GPU task requests one A5000 and four CPU
cores. The controller records job IDs/intent, stops on ambiguous submissions and
never automatically resubmits paid work.

```bash
cat outputs/numina-encoder-smoke-v1-submission/jobs.tsv
cat outputs/numina-encoder-smoke-v1/verification.json
cat outputs/numina-encoder-smoke-v1/report.txt
```

Only after `passed: true`, submit the larger study with unchanged code:

```bash
python3 scripts/submit_encoder_study.py --mode full \
  --source /usr/xtmp/ms785/rl_ms_project/outputs/numina-no-folds-v1 \
  --smoke outputs/numina-encoder-smoke-v1 \
  --out outputs/numina-encoder-v1 --concurrency 3 --cpu-on-gpu
```

| Budget | Smoke | Full |
|---|---:|---:|
| Train / validation / test questions | 20 / 10 / 10 | 500 / 100 / 200 |
| Answers/question | 2 | 4 |
| Response-token cap | 512 | 2,048 |
| MNTP steps + contrastive steps | 2 + 2 | 200 + 200 |
| Maximum SLICE epochs | 2 | 30 |
| Sampled prefixes per answer per unit | 8 | 8 |

Smoke only establishes integration. Use its measured extraction cost to judge
full-run feasibility; bidirectional recomputation may dominate. The larger run
is still a bounded study, not proof the adaptation has been trained sufficiently.
`summary.json` retains all 18 configurations, paired contrasts and extraction
records; `status/` records individual stage failures. Inspect failures before
retrying: generated-answer and feature caches are reusable, while adapter/fitting
stages refuse silent retraining of existing checkpoints. Use a new study directory
if changing the recipe.
