# Numina → GSM8K: representation and full-critic transfer

Offline only: no PPO, LSTD or actor updates. This first study fixes Qwen2.5-Math-1.5B
and tests whether learned features and complete Ridge critics transfer between
math datasets. It does not train on MATH-500.

## Locked comparison

| Arm | Representation training | Ridge training/selection | Test |
|---|---|---|---|
| full_transfer | Numina | Numina train/validation only | GSM8K |
| feature_transfer | Numina | GSM8K train/validation | GSM8K |
| target_slice | GSM8K train | GSM8K train/validation | GSM8K |
| pooled_slice | Equal Numina/GSM8K training-question mixture | GSM8K train/validation | GSM8K |
| combined_768 | Two specialist 768-wide SLICEs; concatenate and train linear 1536→768 combiner on GSM8K train | GSM8K train/validation | GSM8K |
| wide_pooled | Single 1536-wide pooled SLICE | GSM8K train/validation | GSM8K |
| rms_two_thirds | No learned compressor; RMS-normalized block 19 | GSM8K train/validation | GSM8K |

All target arms use the SAME generated target answers and sampled pre-action
prefixes. Full transfer freezes the entire source critic, including normalization,
Ridge epsilon and head, before target evaluation. Feature transfer fits a new
Ridge head; it is NOT zero-shot critic transfer. Only the source specialist/head
in the first arm is source-only; other models in the same job can use target labels.

Each SLICE uses blocks 28,27,25,21,13, RMS-normalized vectors, training-only centering
and one isotropic scalar, row-normalized projection + ReLU, width 768, BCE plus
0.01 mean-activation penalty, AdamW lr 3e-4/wd .001, 30 epochs, validation-selected
epoch, seeds 42/43/44. Pooled representation learning uses the SAME total question
budget as each specialist (half from each domain), rather than twice the data.
The 1536-wide pooled control tests extra capacity but does not exactly match the
ensemble's training compute/parameters. The combined arm consumes TWO specialist
fits plus another training stage; no cost advantage is claimed.

Combiner inputs are concatenated specialist features, centered/scaled from target
training prefixes. A linear 1536→768 map and temporary logistic head are trained
with target BCE, no ReLU or sparsity penalty in the combiner. Freeze the selected
map, discard the temporary head, then fit Ridge. Thus combined features are not
claimed sparse. It preserves a 769×769 final Ridge system, not the original
feature-extraction cost. The wider and raw-layer controls use 1537×1537 systems.

Ridge uses sampled prefixes (NOT our all-token cumulative online fitting), with
training-only centering/isotropic scaling and epsilon selected from [.01,1,100]
on equal-question validation raw Brier. Intercepts are regularized. No output
LayerNorm, clipping or calibration is applied to the fitted head. Report raw and
clipped Brier separately; clipping is a scoring comparator only.

## Data, exclusions and limits

Full preset per domain: 2,000 train / 500 validation / 1,000 test questions,
4 answers/question: **28,000 new answers**, max prompt 512 / answer 1,024 tokens,
temperature 1, top-p 1. Eight uniformly sampled pre-action positions per answer
(or all positions if shorter). This defines a capped task, not unrestricted math.
Question counts are eligible numeric subsets, not the full benchmarks.

GSM8K native train supplies train/validation; native test supplies test. Numina
source tags `gsm8k` and `math` are removed. ALL numeric-eligible native GSM8K
families, including unsampled ones, are blocked from Numina. Deterministic
number-template grouping removes further overlaps across domains and splits.
This is not semantic/paraphrase or pretraining decontamination; Qwen may know the
benchmarks. Exact selected questions and pinned dataset revisions are saved.
Existing historical Numina compressors are NOT reused: their overlap exclusions
cannot retrospectively be assumed sufficient.

Only verifiable correct/incorrect answers enter fitting/evaluation. Other statuses
remain in saved trajectories and per-shard label-status counts; inspect coverage
before interpreting results. Pack fails clearly if any split lacks both classes.
More answers/prefixes do not replace more independent question families.

The source critic is also evaluated on held-out Numina questions for in-domain
context, reported separately from GSM8K transfer scores.

Primary metric: equal-question raw Brier. Also report prefix-weighted raw/clipped
Brier, threshold-0.5 accuracy and regularized matrix condition. Six predeclared
paired comparisons use 4,000 approximate seed/question bootstrap draws, descriptive
95% intervals, exploratory/unadjusted. Three seeds measure compressor/head training
variation on the same generated data, not independent rollout collections. No
automatic winner, calibration guarantee, or PPO-transfer claim.

## Run on Duke

Commit/push from your computer, then on the cluster:

```bash
cd /usr/xtmp/ms785/rl_ms_project
git pull --ff-only origin v2
source scripts/cluster_env.sh
python3 tools/transfer/submit.py --mode smoke \
  --out outputs/slice-transfer-smoke-v1 --shards 4 --concurrency 4
```

This submits prepare (CPU) → generation/extraction (four one-GPU shards) → pack
(CPU) → three one-GPU fitting tasks → evaluate (CPU). A failed prerequisite cancels
downstream jobs. The smoke uses 40/20/20 questions per domain, four answers,
256 answer tokens and two epochs; it is an integration check only. It downloads
pinned data/model if not cached. Generation uses the package's FP32 eager
Transformers backend, prioritizing consistency over throughput. No runtime estimate
is promised before this smoke; the full study can exceed a day depending on queues.

Submission bookkeeping lives BESIDE the output so preparation can require a fresh
output directory:

```bash
cat outputs/slice-transfer-smoke-v1-submission/jobs.tsv
cat outputs/slice-transfer-smoke-v1/status/evaluate-0.json
cat outputs/slice-transfer-smoke-v1/report.txt
```

Once the smoke is complete and its output/coverage looks reasonable:

```bash
python3 tools/transfer/submit.py --mode full \
  --smoke outputs/slice-transfer-smoke-v1 \
  --out outputs/slice-transfer-numina-gsm8k-v1 --shards 4 --concurrency 4
```

Full submission rejects a missing/failed smoke or changed implementation. Code
must remain fixed after preparation. Logs are in `<OUT>-submission/logs/`.
An sbatch timeout stops submission without automatic retries; check squeue/sacct
and sidecar stage JSON before resubmitting (the controller may have accepted it).
Never delete ambiguous submission bookkeeping to bypass this guard.

Generated trajectories/features resume inside an identical generation shard;
retries must not overlap a running task. Fit retries restart that seed from the
beginning unless already locked, in which case they refuse overwriting. Model
and source hashes are checked. Learned representations/checkpoints here are
research artifacts, not yet package-published critics or PPO filter exports.

## Verifier environment and the first failed smoke

The transfer runner uses `.venv-critic` for Qwen/features and the isolated
`.venv-ppo-verifier/bin/python` for the pinned Math-Verify scorer. The general
`.venv` does not need Math-Verify installed. Virtualenv executable paths must not
be resolved through their system-Python symlinks.

The first smoke (generation array 13249766) incorrectly selected `.venv` as the
verifier, so its shards failed before loading Qwen or generating answers.
The corrected preparation stage checks the verifier before model/dataset downloads,
records its provenance, and cancels dependent jobs if this CPU preflight fails.
Generation rechecks the same provenance before loading the encoder. The study
also hashes its Slurm wrapper and submission controller.

Preserve the old prepared directory for diagnosis. After pulling this fix, use
`outputs/slice-transfer-smoke-v2` for a new smoke; the old manifest intentionally
rejects changed source code. No environment reinstall is needed if the isolated
verifier already passes its pinned provenance check. If it fails, inspect that
environment before submitting GPU work; do not add verifier packages to the RL
environment as a workaround.

## Model cache paths across the login host and container

Full preparation job 13250110 failed before generation because the submission
controller resolved the successful smoke's cache path through a host symlink.
The resulting `/usr/project/xtmp/...` alias was not visible in that container;
Transformers then treated the nonexistent local path as an invalid Hub repo ID.
The dataset and prediction logic were not reached.

The controller now preserves model-path spelling, making paths absolute without
resolving symlinks on the login host. This differs from resolving paths inside
the container, where the model is actually loaded. The passed smoke can be reused
only when its sole changed pinned file is the exact known predecessor of the
submission controller; all other runtime hashes must match. The new submission
records the smoke hash, recognized change and model argument in
`<OUT>-submission/smoke-reuse.json`. Unrecognized changes still require a new smoke.

After pulling this fix, preserve the failed full output/sidecar and submit:

```bash
python3 tools/transfer/submit.py --mode full \
  --smoke outputs/slice-transfer-smoke-v2 \
  --out outputs/slice-transfer-numina-gsm8k-v2 --shards 4 --concurrency 4
```

Non-array Slurm logs use a sentinel array-task suffix, not `_0`. Find them with
`ls <OUT>-submission/logs/prepare-*.out`, or read them with
`tail -n 120 <OUT>-submission/logs/prepare-*.out`.
