# Running the expanded offline encoder comparison

Separate protocol: `encoder-pooled-v2`. The old `submit_study.py` remains the
smaller last-token pilot. This workflow has CPU tests; cluster GPU validation is
pending. It never trains the actor or launches PPO.

## Launch smoke

Commit/push the new files locally first, then on the cluster:

```bash
git pull --rebase origin v5
source scripts/cluster_env.sh
python3 tools/encoder/submit_compare.py --mode smoke \
  --out outputs/encoder-pooled-smoke-v1
```

The standard-library controller discovers a saved offline `questions.json` with
matching package `run.json` and existing 7,800/2,600/2,600 question assignments
under `outputs/` or `data/`. If none or several match, supply
`--questions /absolute/path/to/questions.json`. It does not repartition, download
a substitute, or use the 500-question fixed-policy bank as the expanded corpus.
The source actor defaults to the predeclared actor in
`outputs/fixed-policy-critics-v1`. Existing `.venv` handles generation/verification;
existing `.venv-encoder` handles adaptation/extraction/fitting. No reinstall.

```bash
cat outputs/encoder-pooled-smoke-v1-submission/jobs.tsv
cat outputs/encoder-pooled-smoke-v1/profile.json
cat outputs/encoder-pooled-smoke-v1/distributed-check.json
cat outputs/encoder-pooled-smoke-v1/verification.json
cat outputs/encoder-pooled-smoke-v1/report.txt
```

Use job IDs from the ledger with `sacct`. Logs are in the sibling submission
directory. Dependencies require successful predecessors and cancel invalid
dependents. Intent files precede submission; ambiguous submissions never retry
automatically. Preserve partial studies and check the queue before trying fresh
output names. Do not modify code or environments during a run.

## Launch main only after smoke passes

```bash
python3 tools/encoder/submit_compare.py --mode main \
  --smoke outputs/encoder-pooled-smoke-v1 \
  --out outputs/encoder-pooled-numina-v1
```

Main checks matching smoke source/questions/code identities. Budgets:

| Item | Smoke | Main |
|---|---:|---:|
| Train/validation/test questions | 8/4/4 | 7,800/2,600/2,600 |
| Answers per question | 8 | 8 |
| Adaptation seeds | 1 | 3 |
| Steps per MNTP/SimCSE stage | 2 | 1,000 |
| Compressor seeds | 1 | 5 |
| SLICE epochs | 2 | 120 |
| Full-pipeline validation interval | 1 epoch | 20 epochs |
| Total fits | 8 | 48 |
| MC questions / continuations per state | 4 / 4 | 1,000 / 128 |

The two backbones are causal and complete MNTP+SimCSE LLM2Vec. A/B are pooled
final/two-thirds-layer direct Ridge controls; C/D use pooled five-layer SLICE
at widths 1536/768. See `ENCODER_COMPARISON_V2.md` for normalization and splits.
All methods share answer tokens, fixed 32-token pre-action boundaries and the
same MC references. MC uses answer zero and absolute offsets 0/128/256/512;
answers ending earlier contribute fewer states, without substitutions/duplicates.

## Context and distributed gates

Profile training/validation prefix percentiles and candidate-length coverage.
The implemented context trial uses the entire allowed state length: 512 prompt
tokens + 2048 response tokens = 2560. This preserves full prompts/prefixes rather
than silently truncating a tail. Adaptation samples absolute training prefix
offsets 0/128/256/512/1024/1536/2016, capped at sixteen distinct texts per question.
Line-file integrity and maximum length are checked. The official MNTP grouping
pipeline is retained; SimCSE sees full question/prefix strings.

SimCSE uses four GPUs on one node, eight examples each, accumulation one.
The gate checks the pinned official 32-by-32 contrastive loss, local embedding
gradients against a dense reference, left-padded versus unpadded states, and real
two-view BF16/LoRA/checkpointed backward at full length. A failed memory,
gradient or padding gate stops training. Do not weaken it to force a pass.
MNTP uses one GPU, batch two. Exports store `skip_instruction=False`: all our
available prompt tokens belong in the pool, with no instruction/content separator.
Official trainer/model/loss files remain unmodified.

Intermediate adaptation checkpoints every 250 steps are retained; the endpoint
is predeclared at 1000. They are diagnostics, not test-based selection candidates.
SLICE candidates rebuild Ridge using training states only. Validation picks the
complete compressor/head pair; matrices never carry across changing compressors.
Ridge stays one-head/raw-sum/intercept/epsilon .01.

## Reporting and limitations

The report contains per-fit Brier and MC MAE/RMSE, readout contrasts, and paired
hierarchical question/fit-seed/adaptation-seed bootstrap contrasts for adapted
minus causal. Family adjustment covers four readouts per metric. MC intervals
are conditional on sampled references, not exact-value confidence intervals.
Raw success/sample counts and every continuation batch are saved for audit.
Questions, not tokens or repeated answers, are the independent data clusters.

This is one fixed actor and reused Numina families, not actor-seed replication
or a fresh benchmark. Semantic-chunker secondary reruns are deferred; this is the
fixed-boundary primary comparison. Full workflow GPU behavior is not locally
verified. Main can generate 104,000 answers and up to 512,000 continuations,
with repeated unbatched feature extraction and 48 fits. Storage and GPU costs
are substantial. Each main job has a 24-hour limit. Four-A5000 single-node
availability and wall-clock completion have not been verified locally.
