# One frozen policy, six critic estimators

**Question: which critic best estimates the same policy's value function when
adaptation answers, evaluation prefixes and Monte Carlo reference values match?**

This is a separate offline policy-evaluation experiment. It never updates the
actor. Unlike the earlier six-actor audit, **every panel has exactly the same
x-coordinates and reference outcomes**.

## Locked design

- Reuse the completed `slice-token` actor from `numina-value-audit-v1`, actor seed
  1001, final update 120. This policy is chosen before new fitting/evaluation;
  it is not chosen by the new test scores. No six-actor retraining is needed.
- Keep the base Qwen feature encoder and existing sparse 768 compressor frozen.
- Generate four **new** answers for each training and validation question using
  only that frozen actor. All learned critics receive those exact same answers
  and final verifier rewards. Old mixed-policy training rollouts are not reused.
- Fit token Ridge on every pre-action prefix; chunk Ridge on MATH chunk starts,
  with a 32-token fallback. Both use raw A/b sums, an intercept, epsilon .01 and
  a direct solve. The actor never changes, so these sums have no policy drift.
- Apply per-state RMS normalization to the raw two-thirds features during both
  fitting and scoring. No output LayerNorm or implicit clipping is added.
- Train a full transformer token critic from the same base Qwen checkpoint,
  with a zero scalar head, pooled-token squared error against the **same final
  returns**, AdamW learning rate 5e−5 and at most two epochs. FP32 optimizer/master
  state is sharded over two GPUs; forwards use BF16 and gradient checkpointing.
  Validation selects the epoch, including epoch zero. Training/validation curves
  are saved in `neural.json`; this is a limited recipe, not an optimization ceiling.
- Generate one answer per held-out test question and select up to four causal
  chunk-start prefixes. All six estimators score that bank. Independently resume
  the same frozen actor 128 times per prefix, preserving the 2,048-token total
  response budget and original token IDs. Those outcomes supply **one shared
  reference**, generated once and reused by all six panels.
- GRPO's comparator is an independent four-answer prompt-level mean reward,
  constant across a question's prefixes. It is a baseline, not a value network.

The transformer panel is labeled **“Transformer critic (MC fit)”**. It is not a
PPO-trained actor/critic pair and uses no GAE or PPO clipping. This deliberate
choice holds the learning target fixed instead of confounding feature/head
quality with GAE targets. Chunking here changes which states are fitted, not
the actor's actions or a GAE horizon.

SLICE retains its prior supervised representation training. Matching the new
adaptation answers does not match total historical label/compute budgets across
representations. One source policy and one transformer fitting seed do not
establish a general architecture ranking. The Numina questions were inspected
in earlier studies; this remains a controlled follow-up, not a new benchmark.

## Budgets and outputs

| Phase | Train questions | Validation questions | Test questions | Answers/train or val question | Reference samples/state |
|---|---:|---:|---:|---:|---:|
| Smoke | 8 | 4 | 4 | 4 | 4 |
| Main | 500 | 100 | 50 | 4 | 128 |

Main has 2,000 shared training answers, 400 validation answers, and at most 200
test states × 128 continuations = **25,600 reference completions total**, not six
times that number. Bank answers and four-answer GRPO groups are additional.
Short answers can provide fewer than four boundaries. Dataset sizes and
hyperparameters are fixed before looking at test references.

The main six panels are token/chunk SLICE Ridge, token/chunk RMS-layer Ridge,
the MC-trained transformer critic, and GRPO's prompt baseline. Reports include:

- Raw, equal-question MAE/RMSE and out-of-range predictions.
- One shared MC reference and Wilson reference intervals for every state.
- Exploratory paired question-bootstrap MAE differences on the same references.
  These are unadjusted and conditional on sampled MC outcomes, and omit fit-seed
  uncertainty. Individual metric summaries also include the explicitly labeled
  Jeffreys-Beta reference-uncertainty sensitivity analysis from the earlier audit.
- **Secondary** clipped prediction diagnostics; primary predictions stay raw.
- Progress-bin state/question counts so sparse bins are visible.

Files: `report.txt`, `summary.json`, `verification.json`, `neural.json`,
`ridge-statistics.pt`, `value-scatter.png`, `value-scatter.pdf`,
`error-by-progress.png` and `progress-counts.json`. A passing gate means that the
pipeline completed and artifacts match, not that any critic won.

## What to run

Push these changes to `v3`. On Duke, after the old pilot has finished:

```bash
cd /usr/xtmp/ms785/rl_ms_project
git switch v3
git pull --rebase origin v3
source scripts/cluster_env.sh

python3 scripts/submit_fixed_policy.py --mode smoke \
  --source outputs/numina-value-audit-v1 \
  --out outputs/fixed-policy-smoke-v1 --concurrency 4
```

The existing editable installation already provides the required Torch,
Transformers, vLLM and plotting stack; no new installation/CPU allocation is
required. Every submitted stage uses `compsci-gpu`, including small preparation
and reporting tasks, to avoid waiting for the separate CPU partition. Those
utility tasks reserve a GPU even when their work is CPU-only. Four CPU cores
and 64G host memory are requested for one-GPU tasks; transformer fitting uses
two A5000 GPUs, eight cores and 128G. Concurrency bounds arrays at four tasks.

The smoke exercises shared generation, feature extraction, all four Ridge fits,
real two-GPU transformer fitting/restoration, independent GRPO groups, common MC
references and the final plots. Check completion:

```bash
cat outputs/fixed-policy-smoke-v1-submission/jobs.tsv
cat outputs/fixed-policy-smoke-v1/verification.json
cat outputs/fixed-policy-smoke-v1/report.txt
```

Only after `passed: true`, submit the main experiment in a **different** output:

```bash
python3 scripts/submit_fixed_policy.py --mode main \
  --source outputs/numina-value-audit-v1 \
  --smoke outputs/fixed-policy-smoke-v1 \
  --out outputs/fixed-policy-critics-v1 --concurrency 4
```

Check jobs with `squeue -u "$USER"`. After completion:

```bash
cat outputs/fixed-policy-critics-v1/verification.json
cat outputs/fixed-policy-critics-v1/report.txt
cat outputs/fixed-policy-critics-v1/neural.json
```

The source actor, base model, filter, question selection and runtime are pinned
by hashes. Keep source artifacts and code unchanged while jobs run. Submission
timeouts stop the chain and retain a marker; inspect Slurm before retrying.
Generation caches complete question groups and MC blocks atomically. Extraction
shards can rebuild incomplete feature files. Partial transformer training does
not automatically resume or replace a completed fit; inspect its failure before
planning a fresh fit. Do not compare a smoke ranking with the main study.
