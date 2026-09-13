## Current next step: teach the output format with one example

Plain completion is the current candidate. Compare it with the same prompt plus
one fixed, synthetic worked example ending in a numeric box. Both conditions use
32 validation questions, two responses each, temperature 1.0, a 2,048 response-token
budget, and the unchanged numeric-box-v1 verifier. Only generated text is scored;
the example is part of the input, never part of the scored response. No weights
are trained. This is a candidate intervention, not a guaranteed formatting fix.

After pulling on Duke:

```bash
cd /usr/xtmp/ms785/rl_ms_project
source scripts/cluster_env.sh
mkdir -p logs
sbatch scripts/submit_diagnostic.sh
```

The job runs two conditions sequentially on one A5000 (128 responses). Replace JOB_ID:

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode
cat outputs/diagnostic-JOB_ID/*/summary.json
python3 scripts/review_stage1.py outputs/diagnostic-JOB_ID/completion-example-v1-t1.0
```

Outputs preserve old jobs and record the actual prefix and token IDs. The example
can affect reasoning as well as format, so compare correctness too. All conditions
use a common prompt-length eligibility check. The 200-response audit and training
contract promotion remain pending; do not start PPO from a diagnostic result.

## Previous step: prompt style and sampling diagnostic

Compare chat-template versus plain completion prompts, each at temperature 1.0
and 0.6. Both use the same numeric-box-v1 instruction/verifier, seed, 32 validation
questions, two responses per question, and 2,048 response-token budget. The plain
prefix is the instruction followed by `Question: ...` and `Solution:`; it uses no
chat role markers. This tests a candidate prompt, not a claim that chat is broken.
No repetition penalties or custom stopping rules are added.

After pulling these changes on Duke:

```bash
cd /usr/xtmp/ms785/rl_ms_project
source scripts/cluster_env.sh
mkdir -p logs
sbatch scripts/submit_diagnostic.sh
```

One A5000 runs the four conditions sequentially (256 responses total). Replace
JOB_ID with the returned number:

```bash
squeue -j JOB_ID
tail -F logs/math-rl-diagnostic-JOB_ID.out
sacct -j JOB_ID --format=JobID,State,ExitCode
cat outputs/diagnostic-JOB_ID/*/summary.json
```

Outputs are chat-t1.0, chat-t0.6, completion-t1.0, and completion-t0.6. Each response
stores the exact rendered generation prefix and prompt token IDs. The reviewer
now displays that exact prefix. Inspect one condition without GPU allocation:

```bash
python3 scripts/review_stage1.py outputs/diagnostic-JOB_ID/completion-t0.6
```

Compare extraction, reward, truncation, and human coherence/correctness. No
condition is preselected as the winner. These are development diagnostics, not
a completed audit. The training path remains unchanged pending selection and
fresh human audit; previous job outputs and labels remain intact.


## Previous step: aligned numeric-box diagnostic

The prompt now asks for a single boxed numeric answer. The versioned candidate
`numeric-box-v1` accepts one numeric box inside prose, exactly like the saved-response
rescorer. Multiple boxes, malformed boxes, and nonnumeric contents are rejected.
This candidate still needs human auditing; extraction alone cannot detect an
unrelated or contradicted answer. The existing training reward is not promoted yet.

On Duke after pulling these changes:

```bash
cd /usr/xtmp/ms785/rl_ms_project
source scripts/cluster_env.sh
mkdir -p logs
sbatch scripts/submit_diagnostic.sh
```

This now runs only the aligned condition: 32 validation questions, two responses
each, 2,048 response tokens, temperature 1, on one A5000. Existing prepared data
can be reused; the generator replaces the instruction explicitly in memory.
Old output directories and labels remain untouched.

Replace JOB_ID with the submitted number:

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode
cat outputs/diagnostic-JOB_ID/numeric-box-v1-2048/summary.json
python3 scripts/review_stage1.py outputs/diagnostic-JOB_ID/numeric-box-v1-2048
```

Wait for generation to complete before reading/reviewing. Human format/reward
labels must be new for this contract. Do not start PPO based on generation success.
If the diagnostic is satisfactory, the separate audit command is:

```bash
sbatch --gres=gpu:a5000:1 scripts/submit_stage1.sh \
  --mode audit --contract numeric-box-v1 --max-tokens 2048 \
  --out outputs/stage1-numeric-box-v1-2048
```

It uses 100 training-subset questions disjoint from diagnostic validation questions.
Promote the audited contract into training preparation and validation together only
after the audit passes. Earlier boxed/Answer commands below document old controls.

# Math RL · Stages 0–1

A small, inspectable pipeline: **prompt → sample → score → update → save → resume**.

- **Model:** `Qwen/Qwen2.5-Math-1.5B` (base).
- **Data:** GSM8K; 128 training and 32 disjoint validation prompts from its training split.
- **Reward:** 1 when the final line is `Answer: <number>` and numerically correct; 0 otherwise.
- **Hardware:** Duke `compsci-gpu`, one `a6000` (48 GB VRAM), 8 CPUs, 64 GB host RAM.
- **Runtime:** pinned VERL v0.4.1 and container; BF16, 512 response tokens.

Stage 0 proves that the machinery works. Stage 1 checks whether the rewards and
problem difficulty are useful. The official test split stays untouched.

## 1. Copy to Duke

From your local checkout:

```bash
ssh ms785@login.cs.duke.edu 'mkdir -p /usr/xtmp/ms785/rl_ms_project'
rsync -av --exclude=.venv --exclude=__pycache__ --exclude=.pytest_cache \
  --exclude=.DS_Store --exclude='*.egg-info' --exclude=data --exclude=models \
  --exclude=checkpoints --exclude=outputs --exclude=logs \
  ./ ms785@login.cs.duke.edu:/usr/xtmp/ms785/rl_ms_project/
ssh ms785@login.cs.duke.edu
cd /usr/xtmp/ms785/rl_ms_project
```

Keep `.git` for provenance. Build the Python environment on Duke, inside the
container. `/usr/xtmp/ms785` is working storage: back up important results elsewhere.

## 2. Set up once

Use a compute allocation for installation and downloads; login nodes are limited
to one CPU and 4 GB RAM.

```bash
srun -p compsci --cpus-per-task=8 --mem=128G --time=02:00:00 --pty bash -i
cd /usr/xtmp/ms785/rl_ms_project
command -v singularity || command -v apptainer
```

If neither runtime is available, use `module avail` and load the site's listed
Singularity/Apptainer module. Then:

```bash
source scripts/cluster_env.sh
"$MATH_RL_RUNTIME" pull "$MATH_RL_IMAGE" \
  docker://verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2
sha256sum "$MATH_RL_IMAGE" > configs/container.sha256
bash scripts/container_exec.sh bash scripts/setup_environment.sh
exit
```

Setup installs VERL commit `8d9e350ea58c7ad4b50dd14d9dcb50577242c55f`, checks
package compatibility, runs CPU tests, downloads assets, and prints the resolved
training configuration. Keep the container's Torch/vLLM versions together.
Downloads require network access from the allocation.

The wrapper places Apptainer's build temporary files on `/usr/xtmp/ms785` rather
than a potentially RAM-backed `/tmp`, and limits squashfs conversion to two
workers. If a prior pull was killed, inspect the partial SIF and remove or move
it before retrying; never use a partial image for training.

The shared environment uses:

```text
/usr/xtmp/ms785/
  containers/verl-v041.sif
  cache/                         # Hugging Face, pip, container caches
  rl_ms_project/
    .venv/                       # created inside the container
    configs/assets.json          # immutable model/data revisions
    configs/environment.txt      # installed packages
    configs/container.sha256     # checked before container commands
    data/gsm8k/                  # prepared Parquet files
    models/qwen-math/            # initial base checkpoint
```

`MATH_RL_ROOT`, `MATH_RL_IMAGE`, and `MATH_RL_RUNTIME` override defaults. The wrapper
binds these paths and sets the working directory. GPU jobs check BF16 support,
the VERL commit, asset identity, and prompt format before running.

**Updating an existing setup:** rerun `scripts/prepare_data.py` inside the
container allocation to regenerate old prompts. The explicit system message
replaces Qwen's default boxed-answer instruction. Preserve old audits and use a
new output directory. An Instruct manifest is rejected rather than reused.

## 3. Stage 0: prove save/resume

From the checkout on the login node:

```bash
source scripts/cluster_env.sh
bash scripts/submit_resume_check.sh
squeue -u ms785
```

The helper creates `logs/` and submits three jobs: four uninterrupted updates,
two updates, then a dependent restart from step 2 through step 4. Each update
samples 8 prompts × 4 responses and uses VERL's GRPO advantage and clipped loss.
KL, entropy regularization, and weight decay are disabled so learning comes from
reward. This exercises infrastructure; the educational REINFORCE stage comes later.

After all three jobs finish successfully:

```bash
bash scripts/container_exec.sh "$PWD/.venv/bin/python" scripts/report_resume.py \
  checkpoints/resume-control checkpoints/resume-split --atol 0.01 --rtol 0.1
```

The report writes `checkpoints/resume-split/resume_report.json` and exits nonzero
if checks fail: checkpoint files, matching code/configuration, steps 3–4 in the
resumed attempt, prompt order, rewards, loss, and gradient norm. Both runs need a
mixed-reward group and a nonzero gradient. Inspect `logs/` for successful state
restoration and any missing-state warnings.

These fixed tolerances are a smoke check, not proof of bitwise RNG restoration.
VERL console metrics have three-decimal precision; exact rollout agreement is
reported separately. Do not loosen tolerances simply to pass. If all rewards
match within every group, investigate Stage 1 before claiming a learning update.

Each run retains `run.json`, `attempts/*/{run.json,resolved.yaml,console.log}`,
`rollouts/*.jsonl`, and `global_step_*` checkpoints. Keep the parent run directory
with its checkpoints. Fresh runs refuse nonempty directories; resume rejects
code/data/configuration changes. Archive previous runs before repeating the helper.

## 4. Stage 1: audit the reward

Generate **100 prompts × 2 responses** from the initial base model, with temperature
1, top-p 1, and a 512-token limit. No weights are updated.

```bash
mkdir -p logs
sbatch scripts/submit_stage1.sh
squeue -u ms785
```

After generation succeeds, review the 200 responses on the login node (no GPU):

```bash
bash scripts/container_exec.sh "$PWD/.venv/bin/python" \
  scripts/review_stage1.py outputs/stage1
```

Read the complete prompt, reference, and response. Enter `reward format correct`:

| Response | Label |
|---|---|
| Correct `Answer: 42` | `1 1 1` |
| Wrong `Answer: 41` | `0 1 0` |
| Correct final answer, wrong format | `0 0 1` |

`format` requires the entire final nonempty line to contain `Answer:` plus a
signed decimal, optionally with correctly grouped commas. Fractions, units,
boxed notation, trailing prose, and a terminal period fail. `correct` checks the
unambiguous final numeric answer regardless of format; `reward = format × correct`.
Automatic scores are hidden. Enter `q` to pause; rerun to resume saved labels.

```bash
bash scripts/container_exec.sh "$PWD/.venv/bin/python" \
  scripts/report_stage1.py outputs/stage1
```

`outputs/stage1/report.json` passes only with all 200 labels, ≥99% verifier
agreement, 5–70% reward accuracy, ≥10% mixed pairs, ≥95% format rate, and ≤5%
truncation. The last two are provisional engineering thresholds. Incomplete or
failed reports exit nonzero. Inspect disagreements; fix the verifier, prompt,
difficulty, or length budget as appropriate, then re-audit into a new `--out`
directory. Never discard difficult responses to improve agreement.

This decimal verifier covers GSM8K. MATH/DAPO requires a suitable answer parser
and a fresh human audit before training.

## Read the code

- `src/math_rl/{prompts,data,reward}.py`: the input and scoring contract.
- `src/math_rl/{train,provenance}.py`: VERL launch, resume guards, run records.
- `src/math_rl/audit.py`: Stage 1 metrics and gates.
- `configs/smoke.yaml`: experiment settings; `scripts/`: setup, jobs, review, reports.
- `tests/`: CPU checks. Run inside the environment with `python -m pytest -q`.

For the underlying RL loop, read `verl/trainer/ppo/ray_trainer.py` and
`core_algos.py` at the pinned commit. Local checks cover CPU behavior and mocked
launching. Stage 0/1 are complete only after the Duke runs and human audit pass.

### Previous Stage 1 format/length diagnostic (historical)

The previous diagnostic ran all four conditions (Answer/boxed × 512/2048 response tokens) sequentially
on one A5000. Uses the same 32 validation prompts, two responses each, temperature
1. Training prompts and the training reward remain unchanged.

```bash
cd /usr/xtmp/ms785/rl_ms_project
source scripts/cluster_env.sh
mkdir -p logs
sbatch scripts/submit_diagnostic.sh
```

Replace JOB_ID with the submitted number:

```bash
squeue -j JOB_ID
tail -F logs/math-rl-diagnostic-JOB_ID.out
sacct -j JOB_ID --format=JobID,State,ExitCode
cat outputs/diagnostic-JOB_ID/*/summary.json
```

Each condition saves 64 responses, metadata, and summary.json. Compare reward,
automatic format compliance, mixed pairs, truncation, and response lengths.
Format-independent correctness requires human review; null human metrics mean
unreviewed, not zero. Diagnostic runs cannot pass Stage 1.

Review a condition on the login node (no GPU needed):

```bash
python3 scripts/review_stage1.py outputs/diagnostic-JOB_ID/boxed-2048
```

The reviewer displays the selected contract; q pauses and labels persist.
The boxed contract requires exactly one numeric box on the final line, no
other boxes or Answer: markers. It does not accept arbitrary numbers in prose.

After selecting a condition, generate 200 fresh audit responses from 100
training-subset questions, disjoint from diagnostic validation prompts.
For example, only if boxed/2048 is selected:

```bash
sbatch --gres=gpu:a5000:1 scripts/submit_stage1.sh \
  --mode audit --contract boxed --max-tokens 2048 \
  --out outputs/stage1-boxed-2048
python3 scripts/review_stage1.py outputs/stage1-boxed-2048
```

Wait for generation to complete before review. Use the existing container-based
report_stage1.py command with the new path after labeling all 200 responses.
Before PPO, promote the selected contract and length into preparation,
preflight, and training together. This diagnostic does not yet change those
settings. Preserve old outputs; do not reuse their labels for new responses.

### Rescore saved responses without a GPU

Run on the Duke login node after pulling the updated code:

```bash
python3 scripts/rescore_stage1.py outputs/diagnostic-12583711
```

This diagnostic accepts exactly one numeric box inside prose and rejects missing,
malformed, symbolic, or multiple boxes. It checks saved response hashes, prints
per-condition summaries, and saves numeric-box-rescore.json without overwriting
existing reports. Original responses, scores, metadata, and labels are preserved.
No packages or GPU allocation are needed. A single condition directory also works.

Extraction does not establish that a box answers the original question or is
uncontradicted. Human auditing remains necessary; old format/reward labels do not
apply to this rule. Rescoring cannot pass Stage 1 or change the training reward.
