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

### Stage 4 — Frozen-model value prediction

The large critic run generated 80,000 answers and retained 77,351. On unseen
questions, the one-layer sigmoid head reached 71.3% correctness-prediction
accuracy, the two-layer MLP 71.8%, and the deeper residual network 71.6%.
Direct ridge regression reached 71.2% with 32 seconds of fitting/tuning.
These are predictions of answer success, not improvements in Qwen's math accuracy.
Results exclude parsing failures and require that qualification.

The [frozen critic guide](docs/frozen-critics.md) documents the completed study.

### Stage 5 — Frozen-policy LSTD comparison

Job **12686052** fitted a single linear value head using **61,876 retained training
answers and 20,931,480 token transitions**. Qwen stayed frozen. We cached a hidden
vector at every response-prefix boundary, including the question-only state;
LSTD used consecutive pairs, not just the eight randomly sampled probe prefixes.
The next-state value is zero after EOS or the response-length cap.

| Method | Test Brier (lower is better) | Correctness-prediction accuracy |
|---|---:|---:|
| LSTD(0) | 0.22889 | 64.0% |
| Matched ridge | 0.19845 | 70.9% |

Ridge performed better with the same features and training transitions. The paired
question-level Brier difference, LSTD minus ridge, was **+0.03038**, with a 95%
bootstrap interval of **[+0.02624, +0.03483]**. LSTD was close to the earlier constant
baseline (0.23054). This does not establish that all LSTD variants fail, or that
lightweight critics cannot help PPO. Excluded answers still limit both results.

The detailed diagnostics showed an accurate matrix solve but nearly constant
predictions: 61,837 of 61,912 test prefixes fell in the 0.6–0.7 probability bin.
LSTD's AUROC was 0.654 versus ridge's 0.728. Solving these same equations longer
will not fix that result.

### Stage 6 — LSTD(lambda) sweep

Job **12687136** completed in **1 hour 39 minutes**, using the same 20.9 million
training transitions and one-layer head. Lambda and regularization were selected
on validation data; Qwen stayed frozen.

| Method | Test Brier | Correctness-prediction accuracy |
|---|---:|---:|
| LSTD(0) | 0.22889 | 64.0% |
| LSTD(0.9) | 0.22124 | 64.0% |
| LSTD(0.99) | 0.20222 | 69.5% |
| **LSTD(0.999)** | **0.19751** | **71.2%** |
| LSTD(1) / matched ridge | 0.19845 | 70.9% |

Validation selected lambda=0.999 and regularization=0.001. Its paired Brier
advantage over ridge was **0.00095**, with a descriptive 95% interval for
LSTD-minus-ridge of **[-0.00140, -0.00047]**. Lambda=1 matched ridge, as expected.
The improvement is small and this test set has been inspected repeatedly.

### Stage 7 — Stability checks

The length analysis (**12688086**) found the small LSTD advantage mainly on
129–512-token answers, not the longest answers. A second generation seed on the
same 500 questions (**12688087**) reproduced it: Brier **0.19790 vs 0.19887**,
paired difference **−0.00098**, 95% interval **[−0.00146, −0.00050]**.
These are frozen-critic results, not evidence of better PPO learning.

### Next — Harder MATH questions

Evaluate the locked GSM8K critics on **MATH-500 levels 4 and 5 only**. Use every
eligible text-only problem with a plain integer/decimal answer; save exclusions.
Generate 16 answers each, with the same 2,048-token budget. No refitting or tuning.
This tests transfer to harder math, not whether long reasoning causes an advantage.

After publishing these changes and pulling them on the cluster:

```bash
source scripts/cluster_env.sh
sbatch scripts/submit_lstd_math.sh outputs/lstd-12687136
```

Use the returned job number:

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
cat outputs/lstd-math-JOB_ID/report.txt
```

The report compares Brier errors overall and by difficulty level, with paired
question confidence intervals, success rates, exclusions and truncation. Inspect
`review.jsonl` before claiming improvement. This subset excludes symbolic answers,
fractions and diagrams; difficulty labels do not guarantee reasoning depth.
See the [LSTD guide](docs/lstd.md) for resume and interpretation. PPO integration
follows with conventional, ridge and LSTD critics compared on learning and total cost.


### MATH-specific fitting — submit alongside the transfer check

```bash
sbatch scripts/submit_math_fit.sh
```

One job generates data, caches features, fits LSTD(lambda) and matched ridge, then
writes `outputs/math-fit-JOB_ID/report.txt`. It uses up to **1,000 train / 200
validation / 300 test questions**, all level 4/5, with **16 answers each** (up to
24,000 answers). MATH-500 is excluded; the official test split never enters fitting.
Only integer/decimal, text-only questions qualify. Actual counts are saved.

The actor stays frozen. Lambda and regularization are selected on validation;
test results are reported overall and by level. This asks whether **MATH-trained**
LSTD beats MATH-trained ridge; the other job tests **GSM8K-to-MATH transfer**.
Neither job updates PPO. Both need a verifier audit before interpreting an advantage.

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
cat outputs/math-fit-JOB_ID/status.json
cat outputs/math-fit-JOB_ID/report.txt
# Resume an interrupted fitting pipeline:
sbatch scripts/submit_math_fit.sh --out outputs/math-fit-OLD_JOB_ID
```

### Compare additional LSTD traces on saved answers

CPU-only: fit **lambda 0.85, 0.90, 0.95 and ridge** on the same GSM8K training data,
then evaluate on the saved new-seed GSM8K answers and harder MATH transfer answers.
Regularization is selected on GSM8K validation only. Existing runs stay unchanged.

```bash
sbatch scripts/submit_lstd_traces.sh \
  outputs/critics-12654337 \
  outputs/lstd-validation-12688087 \
  outputs/lstd-math-12688713
# After completion:
cat outputs/lstd-traces-JOB_ID/report.txt
```

The GSM8K evaluation uses new answers to the same test questions. These sets have
already been inspected, so this is an exploratory comparison, not fresh confirmation.
