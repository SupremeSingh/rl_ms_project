# Setup and experiment commands

For the goal, results and interpretation, read [README.md](README.md).
Run submissions from the repository on Duke's login node. Slurm runs the work on
compute nodes; you may disconnect SSH after submission.

Commands with `JOB_ID` or other uppercase placeholders require your returned job
number. Input-path placeholders such as `CRITICS_JOB_ID` refer to the completed
upstream phase. Keep those artifacts; do not rerun completed generation just to
reproduce a report.

## Initial setup

You need cluster access, repository access, and Singularity or Apptainer (load the
site's container module if needed).

```bash
mkdir -p /usr/xtmp/ms785
cd /usr/xtmp/ms785
git clone git@github.com:SupremeSingh/rl_ms_project.git
cd rl_ms_project
```

For an existing checkout, after publishing local changes:

```bash
cd /usr/xtmp/ms785/rl_ms_project
git pull --ff-only origin main
```

Do not change code tracked by a running experiment: its provenance checks reject
changes. Documentation and unrelated new scripts can be updated independently.

Obtain a CPU allocation for installation and downloads. Skip installation if the
existing environment works.

```bash
srun -p compsci --cpus-per-task=8 --mem=128G --time=02:00:00 --pty bash -i
cd /usr/xtmp/ms785/rl_ms_project
source scripts/cluster_env.sh

# First setup only: download the pinned training container.
"$MATH_RL_RUNTIME" pull "$MATH_RL_IMAGE" \
  docker://verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2
sha256sum "$MATH_RL_IMAGE" > configs/container.sha256

# Project dependencies, local model/data, numerical PPO/GRPO checks.
bash scripts/container_exec.sh bash scripts/setup_environment.sh
bash scripts/container_exec.sh bash scripts/setup_ppo.sh

# Host-side verifier for the manual answer audit.
python3 -m venv .venv-verifier
.venv-verifier/bin/python -m pip install -r requirements-verifier.txt
exit
```

Expect passing tests and resolved training configurations. Inherited dependency
conflicts are recorded in `configs/pip-check.txt`; they are not a substitute for
runtime checks. Training and symbolic verification have separate environments.
Model/data downloads and first-time MATH selection require network access.

Back on the login node, start each session with:

```bash
cd /usr/xtmp/ms785/rl_ms_project
source scripts/cluster_env.sh
mkdir -p logs
```

## Monitoring any job

```bash
squeue -j JOB_ID
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
```

`PD` = queued; `R` = running. Once a job leaves the queue, use `sacct`.
`COMPLETED` with `0:0` means successful execution, not that the experiment proved
an improvement. Read its report. `tail -F` follows a log; Ctrl-C stops following,
not the Slurm job. A requested time limit is not an expected runtime.

## Phase 0 — Runtime and training infrastructure

Installation checks imports and numerical integration. Phase 2's short pilots
exercise actual GPU generation, actor/critic updates and checkpoint saving.
They are the current smoke test using completion prompts and Math-Verify.
The older `submit_resume_check.sh` belongs to the legacy smoke path; successful
checkpoint writing does not establish save/resume equivalence.

## Phase 1 — Audit prompts and rewards

```bash
sbatch scripts/submit_stage1.sh
# After completion:
python3 scripts/review_final_answers.py outputs/stage1-JOB_ID
python3 scripts/review_final_answers.py outputs/stage1-JOB_ID --report
```

Review inputs are `unambiguous correct`, each 0 or 1. `q` pauses; rerunning resumes.
Expected: 200 responses plus a review report. Our recorded audit has only
10 reviewed; do not treat it as a complete audit. Inspect generation errors in
`logs/math-rl-stage1-JOB_ID.out`.

## Phase 2 — PPO and GRPO pilots

```bash
sbatch scripts/submit_pilots.sh
# After completion:
cat outputs/pilots-JOB_ID/status.txt
```

One allocation reserves four A5000 GPUs, two per method. Each pilot runs ten
iterations. All five status codes should be zero. Method logs are `ppo.log` and
`grpo.log` inside the output directory.
These runs verify execution; their sampling budgets differ.

## Phase 3 — Evaluate the trained actors

Pass the **training pilot's** ID, not the new evaluation job's ID:

```bash
sbatch scripts/submit_evaluation.sh PILOT_JOB_ID
# After completion:
cat outputs/evaluation-EVAL_JOB_ID/summary.json
```

Expected: base/PPO/GRPO accuracy on the same 500 held-out questions, paired changes
and truncation rates.

## Phase 4 — Generate frozen-policy data and compare small heads

On a fresh experiment:

```bash
df -h .
sbatch --time=2-00:00:00 scripts/submit_critics.sh --full
# Monitor/read:
cat outputs/critics-JOB_ID/status.json
cat outputs/critics-JOB_ID/report.txt
```

One A5000; plans 80,000 answers on 4,000/500/500 train/validation/test questions.
Generation, feature extraction and head fitting run sequentially. Keep ample disk
space for token-level features. A smaller check omits `--full`.

Reuse the completed full cache, `outputs/critics-CRITICS_JOB_ID`, for later phases.
Selection excludes previously used questions; attempting another full run alongside
all existing manifests may exhaust the eligible pool and correctly fail.

Resume an interrupted compatible run without replacing its output:

```bash
sbatch --time=2-00:00:00 scripts/submit_critics.sh --full \
  --out outputs/critics-OLD_JOB_ID
```

Settings, model and code must match the saved manifest. Do not use legacy migration
scripts on a new failure without first diagnosing it. See [the full protocol](docs/frozen-critics.md).

## Phases 5–6 — Fit LSTD and matched ridge

These CPU-only jobs use the completed full feature cache. On a fresh reproduction,
replace the source path with your Phase 4 output.

```bash
# Optional historical LSTD(0) comparison:
sbatch scripts/submit_lstd.sh outputs/critics-CRITICS_JOB_ID --lambdas 0

# Full trace sweep, including ridge and the lambda=1 equivalence check:
sbatch scripts/submit_lstd.sh outputs/critics-CRITICS_JOB_ID
# After completion:
cat outputs/lstd-JOB_ID/report.txt
```

Expected: validation-selected regularization/lambda, test Brier/accuracy, solver
checks and paired question intervals. Source data stay unchanged.

## Phase 7 — Length diagnostics and a new generation seed

Independent jobs; both use the completed Phase 6 fit:

```bash
sbatch scripts/submit_lstd_analysis.sh outputs/lstd-LSTD_JOB_ID
sbatch scripts/submit_lstd_validation.sh outputs/lstd-LSTD_JOB_ID
# After completion, use their respective job IDs:
cat outputs/lstd-analysis-ANALYSIS_JOB_ID/report.txt
cat outputs/lstd-validation-VALIDATION_JOB_ID/report.txt
```

The first is CPU-only. The second uses one A5000 and samples new answers on the
**same** 500 test questions, keeping critics fixed.

## Phase 8 — Transfer existing critics to harder MATH

```bash
sbatch scripts/submit_lstd_math.sh outputs/lstd-LSTD_JOB_ID
# After completion:
cat outputs/lstd-math-JOB_ID/report.txt
cat outputs/lstd-math-JOB_ID/length-report.txt
```

One A5000. Selects all eligible MATH-500 level 4/5 integer/decimal, text-only
questions, generates 16 answers each and evaluates locked GSM8K critics.
Inspect `review.jsonl` for scorer mistakes. No fitting occurs. Our completed run
used 148 questions. The 2,048-token budget can limit harder answers.

## Phase 9 — Fit critics specifically on MATH

This is independent of Phase 8 and can be submitted alongside it:

```bash
sbatch scripts/submit_math_fit.sh
# Monitor/read:
cat outputs/math-fit-JOB_ID/status.json
cat outputs/math-fit-JOB_ID/report.txt
```

One A5000, up to 1,000 train / 200 validation / 300 test questions, 16 answers each.
Excludes MATH-500; official-test questions never enter fitting. Automatically runs
generation, extraction, normalization, LSTD/ridge fitting and evaluation. Lambda
and regularization use validation only. Inspect `data/review.jsonl` afterward.

The first attempt failed during initialization on an unknown difficulty label,
before generation. The loader now excludes and records unknown labels. Check any
existing run with `sacct` before submitting a duplicate.

Resume a compatible interrupted run:

```bash
sbatch scripts/submit_math_fit.sh --out outputs/math-fit-OLD_JOB_ID
```

For a failed run created with incompatible code, choose a new default output instead.

## Phase 10 — Lower-lambda comparison on saved evaluation answers

Requires the completed Phase 4, Phase 7 generation and Phase 8 outputs:

```bash
sbatch scripts/submit_lstd_traces.sh \
  outputs/critics-CRITICS_JOB_ID \
  outputs/lstd-validation-VALIDATION_JOB_ID \
  outputs/lstd-math-MATH_JOB_ID
# After completion:
cat outputs/lstd-traces-JOB_ID/report.txt
```

CPU-only. Fits lambda 0.85, 0.90, 0.95 and ridge on GSM8K training transitions,
selecting regularization on GSM8K validation only. Scores the same saved answers;
no generation, actor updates or MATH-specific fitting.

## If something fails

For the recent experiment pipelines:

```bash
tail -n 100 slurm-math-fit-JOB_ID.out
cat outputs/math-fit-JOB_ID/status.json
# Substitute the stage named in status.json:
tail -n 100 outputs/math-fit-JOB_ID/generate.log
```

Other log prefixes are `slurm-critics`, `slurm-lstd`, `slurm-lstd-analysis`,
`slurm-lstd-validation`, `slurm-lstd-math` and `slurm-lstd-traces`.
Initialization can fail before `status.json` exists; the Slurm log then contains
the cause. Read the error before resubmitting. Never remove original caches to
silence a provenance mismatch.

The next research phase—PPO with ridge/LSTD critics—is not implemented yet.
See [docs/lstd.md](docs/lstd.md) for equations, controls and limitations.
