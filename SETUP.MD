# Setup and experiment commands

For the goal, results and interpretation, read [README.md](README.md).
Run submissions from the repository on Duke's login node. Slurm runs the work on
compute nodes; you may disconnect SSH after submission.

**Current next step:** the [calibrated PPO comparison](#calibrated-ppo-comparison).
The buffer and planning are paused. Use the original completed MATH fitting
output as the source; no further offline generation is needed.

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
This path completed the reported single-seed pilot. There is no new dependency
to install manually; use the multi-seed command below for replication.

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

Recommended next: repeat the completed comparison with several training seeds,
keeping the ordinary completion prompt, reward and training budget fixed:

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

### Calibrated PPO comparison

After syncing the updated repository to the cluster:

```bash
cd /usr/xtmp/ms785/rl_ms_project
source scripts/cluster_env.sh
sbatch scripts/submit_math_rigorous.sh outputs/math-fit-12688752
```

One two-GPU allocation runs methods sequentially, with a two-day time limit.
Based on the earlier runs, budget roughly **a day to a day and a half of running
time**, plus queueing; longer critic fitting and diagnostic passes add uncertainty.
The defaults are three 30-update validation-only PPO candidates on seed 17,
then four methods × three seeds × 60 updates (3,840 answers per final run).
There is no buffer and no explicit planning. Linear methods reuse current actor
features, avoiding the frozen encoder's extra pass.

The job selects conventional PPO using validation accuracy only, after critic
health and non-regression checks. It never evaluates the final test set during
calibration. If every candidate fails, it stops and saves diagnostics instead of
launching an expensive comparison with an unvalidated baseline. Final runs start
from the original base model; calibration checkpoints are not reused.

```bash
squeue -j JOB_ID
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
cat outputs/math-rigorous-JOB_ID/status.json
cat outputs/math-rigorous-JOB_ID/selection.json
cat outputs/math-rigorous-JOB_ID/report.txt
cat outputs/math-rigorous-JOB_ID/aggregate.json
cat outputs/math-rigorous-JOB_ID/ppo-health.json
```

`selection.json` appears after calibration. If selection fails, inspect
`calibration.json` and `calibration-NAME.log`. Each PPO run saves
`ppo-critic-audit/rank-*.jsonl`: unrounded learning rates, gradient norms,
pre/post-update MSE on identical raw GAE targets and diagnostic-forward time.
`ppo-health.json` includes an estimate of extra diagnostic wall time (sum of the
slowest rank per update). It is included in reported training cost, not hidden.
The aggregate report also estimates PPO cost excluding those diagnostic forwards;
use both figures when discussing compute savings. A completed job can have `state: needs_review` if a final PPO health check fails;
read this before interpreting aggregate scores. The script fails on subprocess
errors and does not resume partially trained actors.

Per-run `base-test.jsonl` / `final-test.jsonl` contain complete generated responses,
references and verifier statuses for answer review. Calibration instead writes
`base-validation-test.jsonl` / `final-validation-test.jsonl`; despite the shared
historical filename suffix these contain **validation questions only**.
`aggregate.json` gives paired seed/question bootstrap intervals; they are
exploratory, not automatic significance/superiority declarations. All existing
test questions have been inspected previously.

The original Phase 11 command still reproduces the historical pilot defaults.
Use the new script for the calibrated comparison. See
[docs/ppo_baseline.md](docs/ppo_baseline.md) for exact settings and limitations.

### Continue after a calibration launch failure

Ray now uses a short, unique temporary directory. Long candidate names previously
made its Unix socket path exceed the system limit before training could start.
To preserve completed calibration candidates and retry missing ones in a new job:

```bash
sbatch scripts/submit_math_rigorous.sh outputs/math-fit-12688752 \
  --reuse-calibration outputs/math-rigorous-PREVIOUS_JOB_ID
```

Use the same calibration budget. Completed candidates are checked against saved
model/data hashes, training-code/configuration records, budgets and validation
question IDs, then referenced without copying checkpoints or modifying the old
output. Missing candidates restart from base weights. Selection and the full
comparison follow normally. Keep the original directory: the new output links to
its completed candidates and records them in `reused-calibration.json`. This is
whole-candidate reuse, not resume of a partly trained actor or final comparison.
If provenance checks fail, inspect the differences rather than bypassing them.

### Buffer experiment — archived

The buffer was tested with frozen features and matched no-buffer controls.
It did not improve this one-seed pilot and increased cost. No new buffer job is
recommended. Its code, settings (`--critic-replay-*`) and original outputs remain
available for reproducing the result. The active comparison fixes capacity to
zero; “buffer” is the term used in reports and discussion.

## If something fails

For the recent experiment pipelines:

```bash
tail -n 100 slurm-math-fit-JOB_ID.out
cat outputs/math-fit-JOB_ID/status.json
# Substitute the stage named in status.json:
tail -n 100 outputs/math-fit-JOB_ID/generate.log
```

For the calibrated comparison, use `slurm-math-rigorous-JOB_ID.out`.
Other log prefixes are `slurm-critics`, `slurm-lstd`, `slurm-lstd-analysis`,
`slurm-lstd-validation`, `slurm-lstd-math`, `slurm-lstd-traces` and `slurm-math-online`.
For online failures, inspect `outputs/math-online-JOB_ID/METHOD-seed42.log`.
Initialization can fail before `status.json` exists; the Slurm log then contains
the cause. Read the error before resubmitting. Never remove original caches to
silence a provenance mismatch.

See [docs/lstd.md](docs/lstd.md) for equations, controls and limitations.


## Phase 12 — Planning before solving (archived; paused)

These commands document completed development experiments. They are not the
current next step and are not used by Phase 11. No further planning run or critic
fit is recommended from the latest screen.

Historical quick validation screen:

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

### Bounded critic comparison (historical)

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
parameters are trained in this phase. PPO/GRPO planning comparisons were not pursued; this branch is paused.


### Multi-stage planning experiment (historical)

The current protocol (`staged-plan-v2`) removes planning instructions from the
control and accepts unambiguous final answers in prose or boxes. Math-Verify
scores only the final section. The old decimal-only filter is removed.
**Start a fresh run: do not migrate or resume the earlier protocol with these
changed prompts and scoring rules.** Its cached answers can be audited separately.

First rescore the saved control answers without using a GPU:

```bash
git pull
source scripts/cluster_env.sh
sbatch scripts/submit_staged_rescore.sh \
  outputs/math-planning-OLD_JOB_ID/completion-2048
```

This writes `outputs/staged-rescore-RESCORE_JOB_ID/summary.json`, `answers.json`
and `responses.html`. Originals stay unchanged. Completed audit batches are cached;
to resume, repeat the command with `--out outputs/staged-rescore-RESCORE_JOB_ID`.
Inspect flagged extractions manually: multiple numbers may be legitimate problem
context, but may also indicate conflicting or irrelevant conclusions. The audit
is of saved batches, not a new benchmark or a silent relabeling of training data.

Then submit the generation-only screen (can run independently of the audit):

```bash
sbatch --time=06:00:00 scripts/submit_math_planning.sh \
  outputs/math-fit-SOURCE_JOB_ID --multistage --screen
```

This uses 100 validation questions (50 per level), four answers per condition:
**800 attempts**, no feature extraction and no critic fitting. These validation
questions were used in prior screening; this is a development check, not an
untouched evaluation. Control and planning have separate instructions. Both use
controller-injected headings and the same final-section verifier.

Planning budgets: 96 facts + 96 plan + 1,792 solution + 64 final-answer tokens.
Control: 1,984 solution + 64 final-answer tokens. Each stage stops at
`<END_STAGE>`, EOS or its cap; unused budgets do not transfer. Headers add context
tokens but are not sampled actions. Invalid generated token IDs are preserved in
error records and excluded before continuation, without retrying the answer.

```bash
sacct -j SCREEN_JOB_ID --format=JobID,State,ExitCode,Elapsed
cat outputs/math-planning-SCREEN_JOB_ID/status.json
cat outputs/math-planning-SCREEN_JOB_ID/report.txt
```

The screen's `responses.html` contains **all 800 responses**, their final-section
extractions and stage metadata. `summary.json` reports answer accuracy over all
attempts, retained fraction, verifier statuses and per-stage caps/empty content.
`review.jsonl` retains full paired-question records. These results test whether
the protocol produces useful answers and labels, not whether the critic improves.
Review before launching another fit; no automatic fitting job follows.

To resume this new screen after interruption, repeat its command and add
`--out outputs/math-planning-SCREEN_JOB_ID`, keeping code and flags unchanged.
A new Slurm ID does not change that output path. The earlier full-budget command
without `--screen` still exists, but this entire planning branch is now paused.

## Critic follow-up studies

See [the protocol](docs/critic-followups.md) for the layer-depth, cumulative
raw-sum LSTD and periodic encoder-refresh comparisons. Sync the code to the
cluster first, then start from the project directory and source the environment.

```bash
cd /usr/xtmp/ms785/rl_ms_project
source scripts/cluster_env.sh

# One GPU: reuse saved MATH answers; compare three layers and all existing heads.
sbatch scripts/submit_critic_layers.sh outputs/math-fit-12688752

# Two GPUs: short integration check, including actual encoder refresh.
sbatch --time=03:00:00 scripts/submit_critic_followups.sh \
  outputs/math-fit-12688752 --study refresh --steps 3 --seeds 42 \
  --refresh-interval 1
```

After the smoke finishes with `state: complete`, submit the full studies:

```bash
# Matched frozen encoder: fresh batch versus all accumulated statistics.
sbatch scripts/submit_critic_followups.sh outputs/math-fit-12688752 \
  --study cumulative --trace-lambda 0.99 --epsilon 0.01

# Refresh versus reset-only control and permanently frozen cumulative control.
sbatch scripts/submit_critic_followups.sh outputs/math-fit-12688752 \
  --study refresh --trace-lambda 0.99 --epsilon 0.01 --refresh-interval 10
```

Both online jobs default to 60 updates and seeds 42, 43, 44. The second job
includes a frozen cumulative reference again so it is self-contained. Do not
interpret the one-seed smoke as a performance study. Submission wall-time limits
are budgets, not measured runtime predictions for the new pipeline.

Outputs are `outputs/critic-layers-<JOBID>` or
`outputs/critic-followups-<JOBID>`. Inspect `status.json`, then `report.txt` and
`summary.json`. The layer report also links conceptually to the per-layer
`review.txt`, fitted heads, predictions and curves. To resume layer extraction or
head fitting with exactly the same sources/settings/code:

```bash
sbatch scripts/submit_critic_layers.sh outputs/math-fit-12688752 \
  --out outputs/critic-layers-<ORIGINAL_JOBID>
```

Online training does not support mid-run resume; start a fresh output directory.
`--epsilon 1` reproduces a single identity prior on raw sums; it is not added
again at each update. Encoder refresh resets matrices; it does not silently mix
old and new feature spaces or replay old data into PPO actor updates.


## Closing comparison: cumulative frozen critics at two depths

Sync the updated code before submission. Seven variants: conventional PPO with
locked `more_fitting`, plus Ridge/LSTD(0.95)/LSTD(0.99) at two-thirds and final
depth. The `source` supplies questions and model identity; offline heads are not
loaded. No buffer, encoder refresh, or matrix reset. Both layers start fresh.

First exercise all seven paths with a short smoke:

```bash
source scripts/cluster_env.sh
sbatch --time=04:00:00 scripts/submit_critic_followups.sh   outputs/math-fit-12688752 --study layers --steps 3 --seeds 42
```

After that finishes successfully, run the full comparison:

```bash
sbatch --time=3-00:00:00 scripts/submit_critic_followups.sh   outputs/math-fit-12688752 --study layers --steps 60 --seeds 42 43 44   --epsilon 0.01
```

The full study has 21 training runs. Earlier timings suggest roughly 43 hours
for training alone, plus setup and evaluation; the requested three-day limit is
headroom, not a promise or a statement of your partition's permitted maximum.
If your partition disallows that duration, do not shorten the scientific budget;
use a longer-time partition or arrange separate seed jobs before submission.
The study fixes the requested traces per condition; `--trace-lambda` does not
change them. Read `outputs/critic-followups-<JOBID>/report.txt` and `summary.json`.
The summary contains every seed and paired comparison, including same-method
layer comparisons. All intervals are exploratory and unadjusted.


## Final direct LSTD versus GRPO comparison

Sync the updated repository first. The final study locks LSTD(0.99), two-thirds
features, a frozen encoder, accumulated raw matrices, epsilon=0.01 and no resets.
GRPO uses the same 64-answer rollout budget per update. Both start from the base
actor; no fitted offline heads or previous PPO checkpoints are loaded.

Smoke-test the new GRPO/reporting branch:

```bash
source scripts/cluster_env.sh
sbatch --time=03:00:00 scripts/submit_critic_followups.sh \
  outputs/math-fit-12688752 --study grpo_final --steps 3 --seeds 100
```

Once `status.json` says `complete`, run the locked comparison:

```bash
sbatch --time=2-00:00:00 scripts/submit_critic_followups.sh \
  outputs/math-fit-12688752 --study grpo_final \
  --steps 60 --seeds 101 102 103 104 105 --epsilon 0.01
```

Ten runs should require approximately 18 hours of training plus startup and test
evaluation based on earlier timings; the 48-hour allocation supplies headroom.
Read `outputs/critic-followups-<JOBID>/report.txt`, `summary.json` and `audit.jsonl`.
The latter contains every pair of final responses, with disagreements, verifier
failures and capped answers first. Flags are review priorities, not automatic
errors. Inspect a random agreement sample too. Results are a replication across
fresh training seeds on inspected questions, not an untouched benchmark. No
automatic superiority or equivalence claim is made. No mid-training resume.


## Reference-integrity correction (no GPU)

The audit reads complete original reference solutions already preserved in
`data/questions.json`. It does not download a newer dataset revision or change
an existing run. Run after syncing code:

```bash
source scripts/cluster_env.sh
bash scripts/container_exec.sh "$PWD/.venv/bin/python" scripts/audit_math_references.py \
  outputs/critic-followups-12754254 \
  --out outputs/reference-audit-12754254
cat outputs/reference-audit-12754254/report.txt
```

The command needs NumPy, not a GPU or Math-Verify environment. The output must be
new; choose a new output name for a second audit. The policy retains exactly one
balanced numeric reference box, rejects explicit requests for multiple answers,
and verifies agreement with the saved reference. Multiple boxes are conservatively
excluded even if they repeat a single valid answer. `reference-audit.jsonl`
contains all decisions and complete references for manual review. It is a
structural eligibility audit, not automatic mathematical verification.

Filtered base and final answers are copied for every method/seed. `summary.json`
recomputes means, gains and paired exploratory intervals on identical retained
test questions, preserving the original scores of those retained answers.
Costs are unchanged because the models are not rerun. Check `excluded_by_split`:
training/validation exclusions mean historical rewards or selection may also be
affected; evaluation correction cannot undo that. Do not claim retraining was
performed. New dataset creation uses the same stricter policy; existing source
snapshots are never silently rewritten. Keep both original and corrected reports.

## J-space supervised probes on saved MATH answers

This adds a separate offline study; no PPO parameters, old feature caches or saved
answers are changed. Read the [protocol](docs/jspace-probes.md), particularly the
distinction between the official lens and our sparse reconstruction extension.

After syncing the code, prepare the **separate** lens environment once on the
cluster login node (network access required for pinned packages):

```bash
source scripts/cluster_env.sh
bash scripts/container_exec.sh bash scripts/setup_jlens.sh
```

The setup uses `venv --without-pip` and the existing project's pip with an
explicit `--python` target for `.venv-jlens`, so the container does not need
`ensurepip`, `apt`, or `sudo`. If an earlier attempt failed at `ensurepip`, rerun
the setup command above after syncing this fix; no directory deletion is needed.
The existing PPO environment is used only to invoke pip, not as its install target.
Runtime dependencies are installed from public PyPI, followed by the pinned
`jlens` package with `--no-deps`. This avoids trying to download the container's
custom CUDA PyTorch wheel. Setup verifies that the new environment imports the
same PyTorch module and version as the container. The optional NVIDIA pip index
is disabled for this setup. A previous resolver failure can be retried with the
same command after syncing; no deletion is necessary.

Run the small GPU smoke first. It uses two calibration prompts, up to 12/8/8
train/validation/test questions, two answers per question, and two fitting epochs.
It verifies the pipeline, not accuracy:

```bash
sbatch --time=03:00:00 scripts/submit_jspace_probes.sh \
  outputs/math-fit-12688752 --smoke
```

Replace JOB_ID with the returned number:

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
cat outputs/jspace-JOB_ID/status.json
cat outputs/jspace-JOB_ID/report.txt
tail -n 30 outputs/jspace-JOB_ID/lens.log
tail -n 30 outputs/jspace-JOB_ID/extract.log
```

Once the smoke completes, run the full matched comparison:

```bash
sbatch --time=2-00:00:00 scripts/submit_jspace_probes.sh \
  outputs/math-fit-12688752
```

Full defaults: 100 reserved training questions for calibration; all other eligible
saved questions/answers; sparsity at most 16; seeds 42/43/44; 200 maximum epochs;
linear logistic and two-layer MLP heads, plus ridge regression. To also include
the ten-layer ResNet, append `--heads linear_logit mlp2 resnet10`. This increases
fitting work for every representation. There is no promise of completion within
a day: measure the full-width lens backward pass and sparse extraction in the
smoke. A 48-hour allocation is a limit, not an estimate.

Resume a timed-out run using its **original** output directory and identical flags
(add `--smoke` again when resuming a smoke):

```bash
sbatch --time=2-00:00:00 scripts/submit_jspace_probes.sh \
  outputs/math-fit-12688752 --out outputs/jspace-ORIGINAL_JOB_ID
```

Do not edit the code/environment mid-run or reuse the smoke directory for a full
run. Resume checks pinned inputs and the code snapshot. The source directory is
read-only. `lens.log`, `extract.log`, and `fit-REPRESENTATION.log` identify the
failed stage; numerical/infrastructure errors fail explicitly. Saved parse
failures are excluded equally across arms, not silently treated as new labels.

Review `responses.html` after copying it locally, `summary.json` for paired
comparisons, and each representation's `summary.json` for calibration, prefix
metrics, parameter counts and fit-budget warnings. All sparse readouts are in
`question-*.json`. No additional generation or reward-verifier job is necessary.
