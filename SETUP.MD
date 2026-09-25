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

## Phase 11 — Online PPO-LSTD, ridge, conventional PPO and GRPO

Publish these changes, then pull them into the cluster checkout. Reuse the
completed Phase 9 directory; this reads its question split and leaves it unchanged.
Replace `MATH_FIT_JOB_ID` with that run's ID:

```bash
cd /usr/xtmp/ms785/rl_ms_project
git pull --ff-only origin main
source scripts/cluster_env.sh
sbatch scripts/submit_math_comparison.sh outputs/math-fit-MATH_FIT_JOB_ID
```

One job requests **two A5000 GPUs, eight CPUs and 128 GB RAM**, with a two-day time
limit. It installs/checks the existing isolated verifier and runs numerical and
integration tests first, then trains PPO, PPO-ridge, PPO-LSTD(0.99) and GRPO
sequentially. Each starts from the base model: **30 updates × 64 answers**, seed 42.
There is no new dependency to install manually. The new GPU path is not yet
cluster-validated; use `--steps 2` for a shorter first execution if desired.

Monitor and read the result (the new submission prints `JOB_ID`):

```bash
squeue -j JOB_ID
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
cat outputs/math-online-JOB_ID/status.json
cat outputs/math-online-JOB_ID/report.txt
```

Detailed results are in `summary.json`; each method has its own log, actor
checkpoint, and base/final test answers. Linear-critic parameters and fitting
diagnostics are saved for each update. Training time includes feature transfer,
CPU fitting, validation and checkpointing; environment setup is excluded, and
model initialization and test times are recorded separately. GPU-hours count GPUs
allocated while CPU fitting runs too.

The online reward policy is identical across methods: correct = 1; incorrect,
ordinary parse failure or verifier timeout = 0. Other runtime failures stop that
method and remain visible. Independent methods still run; any failed method makes
the overall job fail. Nothing is silently excluded from the online sample budget.

After the first comparison succeeds, repeat with several training seeds:

```bash
sbatch scripts/submit_math_comparison.sh outputs/math-fit-MATH_FIT_JOB_ID \
  --steps 30 --seeds 42 43 44
```

This triples the work; the time limit remains two days. For an individual retry,
use a **new** job/output and `--methods lstd` (or `ppo`, `ridge`, `grpo`). Resume
within a partially trained actor run is deliberately unsupported. Never mix
results with different training budgets, seeds or configurations.

The offline heads are not loaded. Ridge/LSTD refit current actor features on each
fresh batch, using two question folds. LSTD lambda=0.99, linear regularization=0.01,
actor GAE lambda=0.95; no historical replay or separate transformer critic for the
linear methods. This first comparison tests the complete systems, including
these fitting differences, on previously inspected questions.

## If something fails

For the recent experiment pipelines:

```bash
tail -n 100 slurm-math-fit-JOB_ID.out
cat outputs/math-fit-JOB_ID/status.json
# Substitute the stage named in status.json:
tail -n 100 outputs/math-fit-JOB_ID/generate.log
```

Other log prefixes are `slurm-critics`, `slurm-lstd`, `slurm-lstd-analysis`,
`slurm-lstd-validation`, `slurm-lstd-math`, `slurm-lstd-traces` and `slurm-math-online`.
For online failures, inspect `outputs/math-online-JOB_ID/METHOD-seed42.log`.
Initialization can fail before `status.json` exists; the Slurm log then contains
the cause. Read the error before resubmitting. Never remove original caches to
silence a provenance mismatch.

See [docs/lstd.md](docs/lstd.md) for equations, controls and limitations.


## Phase 12 — Planning before solving

Start with the quick validation screen, before committing to full critic fits:

```bash
cd /usr/xtmp/ms785/rl_ms_project
source scripts/cluster_env.sh
sbatch --time=03:00:00 scripts/submit_math_planning.sh outputs/math-fit-SOURCE_JOB_ID --screen
```

This uses 100 validation questions (50 at each difficulty level), four answers
per question and prompt, and 2,048 response tokens: **800 answers total**.
It performs no feature extraction or critic fitting. Expect roughly 20–60 minutes
after allocation as a provisional estimate; the three-hour limit gives headroom.
The source snapshots and base model are checked, and the original run is untouched.

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
cat outputs/math-planning-JOB_ID/status.json
cat outputs/math-planning-JOB_ID/report.txt
```

`summary.json` includes verifier status counts and per-level accuracy. All attempts
remain in the accuracy denominator. `review.jsonl` contains paired questions and
both sets of responses, with changed success rates first. Review these for scoring
errors and whether a brief plan was actually produced. The paired interval
resamples questions within difficulty levels, not individual answers.
This tests solving behavior, not improved initial values. If inconclusive, expand
the screen before full fitting. Resume with the same source and `--screen --out
outputs/math-planning-PREVIOUS_JOB_ID`; never reuse a full-fit output directory.

### Bounded critic comparison (recommended next)

```bash
sbatch --time=1-00:00:00 scripts/submit_math_planning.sh outputs/math-fit-SOURCE_JOB_ID --bounded
```

This runs 300/50/100 train/validation/test questions, balanced by level, with four
answers each under both prompts (3,600 total). It excludes the deterministic
screening sample, preserves the original splits, and generates fresh base-model
answers. Original test questions were already inspected, so this is exploratory.
The prompt requests `Plan:` and `Solution:` sections to identify the end of a plan;
this is a deliberate change from the screening prompt. Missing or ambiguous
headings have no post-plan measurement; they are not excluded from other metrics.

LSTD(0.99) and ridge use all retained training transitions with validation-selected
regularization from 0.0001/0.001/0.01/0.1/1.0. Positive regularization suits this
smaller dataset. Random-prefix evaluation remains unchanged. Additional diagnostics
evaluate pre-action states at 0/32/64/128/256 tokens, and post-plan states against
ordinary prefixes of the same length. Coverage, calibration, constant baselines
and question-level confidence intervals are saved in `checkpoints.json`.
No future tokens are inputs, and terminal states are excluded. Conditioning on
answer survival and detected plans can change the evaluated population.

Budget roughly 6–12 hours, provisionally, with a 24-hour allocation. Unlike the
screen, this includes feature extraction and critic fitting; completion within a
day is not guaranteed. Per-question generation and features support resuming:

```bash
sbatch --time=1-00:00:00 scripts/submit_math_planning.sh outputs/math-fit-SOURCE_JOB_ID \
  --bounded --out outputs/math-planning-PREVIOUS_JOB_ID
```

Keep code and settings unchanged when resuming. Check `status.json` and the final
`report.txt` using the commands above. The final `responses.html` shows up to 24
examples: a seeded random sample plus large question-only ridge errors. It shows
both responses, verifier statuses, detected boundaries and prefix predictions.
Sample indices are for display, not paired stochastic continuations. All responses,
including excluded ones, remain in each condition's `data/trajectories/` directory.

To view the HTML, run this **on your laptop**, replacing `CLUSTER_HOST` with your
usual Duke SSH host and `JOB_ID` with the new job ID:

```bash
scp ms785@CLUSTER_HOST:/usr/xtmp/ms785/rl_ms_project/outputs/math-planning-JOB_ID/responses.html ~/Downloads/
```

Open the downloaded file in a browser. Review plan compliance and verifier errors
before drawing conclusions. This job does not train PPO or GRPO.

### Structured-plan rerun and verification audit

```bash
sbatch --time=1-00:00:00 scripts/submit_math_planning.sh outputs/math-fit-SOURCE_JOB_ID \
  --structured --audit-source outputs/math-planning-OLD_JOB_ID
```

`--structured` implies the bounded 3,600-answer experiment. It requests **What we
know**, **What we will do**, **Solution**, and **Final answer** in that order.
Both prompt conditions share the same final-answer instruction and the opt-in
`math-verify-conclusion-v2` rule. This is a new protocol: do not resume into an old
bounded run. The model is prompted, not grammar-constrained; compliance counts and
viewer examples expose missing sections, duplicate boundaries and premature answers.

Before fresh generation, the optional audit rescoring checks all saved answers from
`OLD_JOB_ID` without changing the original labels, features or reports. Its outputs
are `source-audit/summary.json`, `changes.json` and per-condition audit JSONL files.
It does not relabel old heads or recompute their performance. Fresh heads are fitted
on newly scored trajectories in the new run.

The new rule prioritizes explicit final-answer lines and a narrow class of concluding
numeric assignments. Math-Verify still checks equivalence. Old/new scores, extracted
expressions and selected conclusion text are saved for review. Ambiguous repeated
final-answer lines abstain. This experimental rule is not a semantic reasoning checker;
it can still select an irrelevant expression. Inspect changed and unchanged examples.
Existing PPO and other legacy commands keep their original verifier rule.

Boundary detection now accepts inline `Solution:` and Markdown/step-by-step headings,
ignores fenced code and declines ambiguous multiple solution headings. It does not
infer a plan where no explicit boundary exists. Read the new `report.txt`,
`checkpoints.json` and `responses.html` as above. Include both `--structured` and
`--audit-source` unchanged when resuming with `--out outputs/math-planning-NEW_JOB_ID`.
A completed audit is reused; an interrupted audit restarts its CPU rescoring.
Allow approximately 4–8 hours provisionally, with a 24-hour allocation.

### Full critic comparison (optional larger experiment)

After pulling the updated repository on the cluster, use the completed MATH fit
as the immutable question/model source. This generates new answers from the
**base** Qwen model, not the PPO checkpoints. The source run is left untouched.

```bash
cd /usr/xtmp/ms785/rl_ms_project
source scripts/cluster_env.sh
sbatch scripts/submit_math_planning.sh outputs/math-fit-SOURCE_JOB_ID
```

Replace `SOURCE_JOB_ID` with your completed MATH fitting job. The job runs ordinary
reasoning and brief planning sequentially on one A5000, with 16 samples for each
of the same train/validation/test questions. Both budgets are 2,048 tokens.
LSTD lambda is fixed at 0.99; ridge and LSTD regularization are tuned on validation.
All token transitions are used for fitting and the existing random-prefix
protocol for evaluation. The plan is requested in the prompt, not forced or
postprocessed; inspect responses to check compliance.

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
cat outputs/math-planning-JOB_ID/status.json
cat outputs/math-planning-JOB_ID/report.txt
```

The final report is created only after both conditions finish. Individual
condition folders contain stage logs, trajectories, cached features, critic
reports and `data/review.jsonl`. Allow roughly two full MATH-fitting runs; the
previous run took about 17 hours, so this is a substantial job rather than a
short pilot. Actual cost depends on generated answer lengths.

To additionally test extra space, run the full four-condition experiment in a
new output directory:

```bash
sbatch scripts/submit_math_planning.sh outputs/math-fit-SOURCE_JOB_ID --long-budget
```

This repeats both prompts at 2,048 and 4,096 tokens and may exceed the two-day
allocation. Generation/features are saved per question. Resume an interrupted
experiment with the same options, unchanged code/environment, and its old output:

```bash
sbatch scripts/submit_math_planning.sh outputs/math-fit-SOURCE_JOB_ID \
  --out outputs/math-planning-PREVIOUS_JOB_ID
```

Include `--long-budget` when resuming a four-condition run. Do not run two jobs
writing to the same output directory. Compare prompt conditions on validation
before choosing a PPO configuration; these inspected test results are exploratory.
Answer accuracy counts all attempts; critic metrics exclude unparseable/error
answers and may therefore compare different retained populations. No actor
parameters are trained in this phase. PPO/GRPO planning comparisons come afterward.
