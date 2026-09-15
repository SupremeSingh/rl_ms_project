# Math RL

Can a small critic built from an LLM's existing hidden features make PPO cheaper
without hurting learning?

This project explores that question with **Qwen/Qwen2.5-Math-1.5B (base)** and
**GSM8K** math problems. We first establish working RL training and evaluation,
then compare lightweight value predictors, investigate LSTD, and bring the useful
critics back into PPO.

A critic estimates how likely an unfinished answer is to succeed. Reusing features
already computed by the actor could avoid running a separate large critic model.
That potential saving is the research question—not an established result.

## Tools

- **VERL** coordinates PPO/GAE and GRPO training.
- **vLLM** generates answers; **PyTorch/FSDP** handles gradients and distributed training.
- **Math-Verify** scores final answers: correct = 1, incorrect or unparseable = 0.
- **Slurm** reserves Duke's GPUs and runs jobs independently of your SSH connection.
- **Our code** supplies prompts, reward integration, experiment settings and result checks.

## Setup on Duke

You need a Duke cluster account, access to this GitHub repository, and
Singularity or Apptainer available on the cluster. If neither is available,
load the site's container module before continuing.

### 1. Get the repository

On the login node:

```bash
mkdir -p /usr/xtmp/ms785
cd /usr/xtmp/ms785
git clone git@github.com:SupremeSingh/rl_ms_project.git
cd rl_ms_project
```

If you already have the repository, update it instead:

```bash
cd /usr/xtmp/ms785/rl_ms_project
git pull --ff-only origin main
```

### 2. Prepare the environments once

Skip this step if your existing cluster setup works. Obtain a CPU compute
allocation for installation and downloads:

```bash
srun -p compsci --cpus-per-task=8 --mem=128G --time=02:00:00 --pty bash -i
cd /usr/xtmp/ms785/rl_ms_project
source scripts/cluster_env.sh

# Download the training container and record its checksum.
"$MATH_RL_RUNTIME" pull "$MATH_RL_IMAGE" \
  docker://verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2
sha256sum "$MATH_RL_IMAGE" > configs/container.sha256

# Install the project, download Qwen/GSM8K, and check PPO/GRPO integration.
bash scripts/container_exec.sh bash scripts/setup_environment.sh
bash scripts/container_exec.sh bash scripts/setup_ppo.sh

# Separate host environment for the answer audit.
python3 -m venv .venv-verifier
.venv-verifier/bin/python -m pip install -r requirements-verifier.txt
exit
```

Expect passing checks and resolved training configurations. The container and
caches live under `/usr/xtmp/ms785`; the model, data and environments live in the
repository. Training and mathematical parsing use separate environments because
their dependencies conflict.

### 3. Submit and check jobs

Submit jobs from the repository on the login node. `sbatch` returns a job number;
you can then disconnect or turn off your computer.

```bash
squeue -j JOB_ID
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
```

`PD` means queued; `R` means running. After a job leaves the queue, use `sacct`.
A successful process reports `COMPLETED` and `0:0`; its experiment report tells
you whether the scientific checks passed.

## Stages completed so far

### Stage 0 — Working cluster infrastructure

We can load Qwen, generate answers, run backpropagation across A5000 GPUs and save
training checkpoints. The code uses a fixed VERL revision and records the model,
data and environment versions. Checkpoint save/resume equivalence has not yet
been established.

### Stage 1 — Usable prompts and rewards (provisional)

Plain completion prompts worked better than chat-style prompts for this base
model. Math-Verify replaced our overly strict answer-format parser, allowing
correct answers to receive credit without one rigid final-line format.

The fresh audit contains 200 responses. **10 were manually reviewed, with full
agreement on those ten.** We provisionally proceeded; the full audit is incomplete.
Math-Verify checks extracted answers, not the validity of every reasoning step.

To repeat generation and review:

```bash
mkdir -p logs
sbatch scripts/submit_stage1.sh
# After completion, replace JOB_ID with the returned number:
python3 scripts/review_final_answers.py outputs/stage1-JOB_ID
```

### Stage 2 — PPO and GRPO training pilots

Job **12589388** completed both methods in **21 minutes 35 seconds**, using two
A5000s per method concurrently.

| Method | Updates | Training answers | What ran |
|---|---:|---:|---|
| PPO/GAE | 10 | 160 | Actor and separate critic training |
| GRPO | 10 | 320 | Actor training from within-question reward comparisons |

Both passed the execution checks: finite training metrics, nonzero gradients,
informative rewards, saved responses and complete final checkpoints.

To repeat the combined pilot:

```bash
sbatch scripts/submit_pilots.sh
# After completion:
cat outputs/pilots-JOB_ID/status.txt
```

All five status codes should be `0`. This establishes working training, not an
accuracy improvement.

### Stage 3 — Evaluation on 500 held-out questions

Job **12592051** evaluated the original model and both step-10 checkpoints on the
same 500 questions, with identical greedy decoding and Math-Verify scoring.
These questions were held out from our experiments, not necessarily from Qwen's
pretraining. The official GSM8K test set remains reserved.

| Model | Correct | Accuracy | Mistakes fixed vs. base | New mistakes vs. base |
|---|---:|---:|---:|---:|
| Original Qwen | 423/500 | 84.6% | — | — |
| PPO | 424/500 | 84.8% | 10 | 9 |
| GRPO | 424/500 | 84.8% | 13 | 12 |

Both gained just one correct answer overall. **These short runs show no convincing
accuracy improvement**, and do not establish that one algorithm is better.
The scores are automated; sample-answer review remains important.

To evaluate another completed pilot, supply its training job number:

```bash
sbatch scripts/submit_evaluation.sh PILOT_JOB_ID
# After completion, use the new evaluation job number:
cat outputs/evaluation-EVAL_JOB_ID/summary.json
```

We now have functioning training and evaluation pipelines. The next research step
is to measure how accurately and cheaply small heads can predict returns from the
frozen base model's features, before testing LSTD and integrating critics into PPO.
