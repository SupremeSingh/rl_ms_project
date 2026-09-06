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
srun -p compsci --cpus-per-task=8 --mem=64G --time=02:00:00 --pty bash -i
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
