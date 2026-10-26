# Reproducing the final workflow

The `prod` branch contains SLICE and the final Numina comparison. Exploratory
code is retained in Git history at pre-cleanup commit
`8dee855da8a3c20b67103f10abdc94ae1737140f`, rather than in this workflow.
Nothing here deletes cluster outputs, downloaded data or model weights.

## Replay the final comparison

Use the completed six-method study to capture the exact online data, sparse
filter and actual PPO training profile. This **starts new actors from the base
model**; no trained actor or old test decision enters optimization.

After [upgrading the installation](../SETUP.MD#existing-installation), run:

```bash
source scripts/cluster_env.sh
bash scripts/container_exec.sh "$PWD/.venv/bin/python" -m math_rl \
  --stage import-study \
  --source outputs/numina-online-six-lambda09-v3 \
  --out outputs/numina-prod-replay-v1
```

This verifies data, filter/model hashes, all 30 completed run budgets, and a
consistent actual PPO profile across seeds. It preserves the original manifest,
selection and summary. If the selected profile differed from the profile actually
used, the replay uses the actual profile and records both; it does not silently
claim the selection file described the run correctly. Input files stay untouched.

The new directory is self-contained for online training: `data/`, `filters/`,
`manifest.json`, `selection.json` and reference records. It can be relocated; it
still needs the matching model and runtime. Current model pins and environment/
container records are copied under `runtime/` when available. Preserve the actual
container image, model snapshot and original study on durable storage. A mutable
Docker tag or a hash alone cannot reconstruct a missing image.

Run the short package-to-PPO acceptance in [SETUP](../SETUP.MD#cluster-acceptance-after-this-cleanup).
To repeat the full result afterward:

```bash
bash scripts/submit_numina_online.sh \
  --out outputs/numina-prod-replay-v1 --full --smoke-first --concurrency 6
```

This submits the six-method validation-only smoke, then 30 runs (five actor seeds,
120 updates each), then the report. The original PPO calibration is reused as a
locked validation decision. Every actor/head starts fresh; no buffer or encoder
refresh is used. The pipeline checks that smoke used the selected PPO profile.

```bash
cat outputs/numina-prod-replay-v1/jobs.tsv
cat outputs/numina-prod-replay-v1/report.txt
cat outputs/numina-prod-replay-v1/summary.json
```

The output also includes full generated test answers in `audit.jsonl`, per-run
rollouts/checkpoints, critic diagnostics, and paired intervals. Numerical checks
and matching inputs support reproduction of the recipe, not a guarantee of
bit-identical GPU outputs or identical accuracy percentages. Hardware, libraries
and nondeterministic kernels can change results. Costs include setup/evaluation
separately and do not hide offline feature-learning costs.

## Train a new representation from scratch

This route needs **no historical probe/J-lens directories**. It is a new experiment,
not an exact reconstruction of the previously selected compressor. It uses the
same final online settings with newly trained frozen features.

1. Install environments and pin/download the model using [SETUP](../SETUP.MD#first-installation).
2. In a CPU allocation, prepare numeric Numina data and pin all dataset revisions:

```bash
source scripts/cluster_env.sh
bash scripts/container_exec.sh "$PWD/.venv/bin/python" scripts/prepare_numina.py \
  --out data/numina --offline-questions 13000
```

Preparation excludes numeric-template families from GSM8K/MATH, then separates
13,000 offline questions from 3,000/500/2,000 online train/validation/test questions.
Eligibility is reference-only, never based on generated rewards. Revisions and
checksums are saved before generation. `--revisions revisions.json` reuses explicit
immutable dataset SHAs. This is template separation, not semantic decontamination.

3. Inspect the generated budget/config, then submit offline SLICE training:

```bash
bash scripts/container_exec.sh "$PWD/.venv-critic/bin/python" -m slice_critic \
  preview --config data/numina/critic-config.json
mkdir -p logs
sbatch scripts/submit_slice_train.sh \
  data/numina/critic-config.json outputs/slice-numina-v1
```

The default is 104,000 answers, eight sampled prefixes per retained answer,
3,584 generated tokens and 120 compressor epochs. The package uses Transformers
generation on one GPU; a full rebuild can take several allocations. The Slurm
limit is not an ETA. Cached generation and completed stages resume with the same
command/code/configuration; never bypass a failed identity check. Start with a
smaller offline-question budget when validating a new installation.

4. Once `outputs/slice-numina-v1/status.json` is complete, prepare the online study:

```bash
bash scripts/container_exec.sh "$PWD/.venv/bin/python" -m math_rl \
  --stage prepare --artifact outputs/slice-numina-v1/artifact \
  --data data/numina --out outputs/numina-new-v1
bash scripts/submit_numina_online.sh \
  --out outputs/numina-new-v1 --full --smoke-first --concurrency 6
```

The new route calibrates three declared PPO profiles using validation only before
launching the array. If the selected profile changes, an additional two-update
PPO check verifies that profile. Offline weights/calibration never become PPO
advantages: only the frozen feature basis is exported, and online heads start at
empty statistics. Fitting/selection families are checked for overlap before launch.

## Failure handling

A submission timeout may still have created a Slurm job. Submission stops before
creating dependent jobs without a confirmed ID; check `squeue`, `sacct` and
`jobs.tsv` before retrying. Persistent submission markers prevent duplicate work.
Do not remove markers blindly. Use a new prepared output directory for a new chain.

Actor runs deliberately do not resume partially completed PPO. Keep the failed
case/log for inspection and create a new study for reruns. Completed cases are
reused only when their method, seed, budget, split and PPO profile match.

Data, filters, model and active code are checked before each stage. Do not edit
those inputs while jobs are running. CPU unit tests and a two-update smoke do not
prove long-run stability; the final multi-seed study remains the scientific test.
