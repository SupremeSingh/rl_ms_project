# Math RL

Educational PPO critic experiments with **Qwen/Qwen2.5-Math-1.5B base** and pinned
VERL. Current task: run the conventional PPO/GAE pilot.
Stage 1 is **provisionally accepted: 10/200 audit responses reviewed**, with full
agreement on those ten. Its saved audit remains incomplete.
The original GRPO smoke ran, but its zero-reward updates did not establish learning.
See [the experiment plan](TOKEN_CRITIC_EXPERIMENT_PLAN.md).

## Next: PPO/GAE on Duke

After the local changes are pushed to GitHub, update the cluster checkout:

```bash
cd /usr/xtmp/ms785/rl_ms_project
git pull --ff-only origin main
source scripts/cluster_env.sh
mkdir -p logs
```

One-time PPO setup, using a **CPU compute allocation**. This installs a small
verifier environment inside the container and runs the PPO integration checks:

```bash
srun -p compsci --cpus-per-task=2 --mem=8G --time=00:20:00 --pty bash -i
cd /usr/xtmp/ms785/rl_ms_project
source scripts/cluster_env.sh
bash scripts/container_exec.sh bash scripts/setup_ppo.sh
exit
```

Expect passing tests and the resolved PPO configuration. The host `.venv-verifier`
continues to serve the audit. `.venv-ppo-verifier` serves training inside the
container: Math-Verify and Hydra require incompatible ANTLR versions, so they
must run in separate interpreters. Setup leaves the Torch environment intact.

Back on the login node:

```bash
sbatch scripts/submit_ppo.sh
```

This requests **two A5000s, eight CPUs and 128 GB host RAM for up to two hours**.
It starts ten updates, with 16 questions and one sampled answer per question.
FSDP shards the actor and separate critic across the two GPUs; vLLM uses one
inference replica per GPU (tensor parallel size 1). Parameters and optimizer
state move to CPU between phases to reduce GPU memory pressure. CPUs help hold
state and run verification; GPU memory feasibility still needs the first run.

Use the returned job number:

```bash
squeue -j JOB_ID
tail -F logs/math-rl-ppo-JOB_ID.out
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
python3 scripts/report_ppo.py checkpoints/ppo-JOB_ID
```

Expect preflight checks, initial validation, steps 1–10, and actor **and critic**
checkpoints at steps 5 and 10. The report checks finite losses, nonzero actor and
critic gradients, mixed rewards across the run, complete rollout files and all
checkpoint shards. `execution_check_pass: true` means the pilot executed; it does
not establish improved accuracy or complete Stage 2. Paste the report and final
validation metrics before launching a longer learning experiment. Saved rollout
and validation answers are under the checkpoint directory.

To resume an interrupted pilot from its completed step-5 checkpoint:

```bash
sbatch scripts/submit_ppo.sh \
  trainer.default_local_dir=checkpoints/ppo-OLD_JOB_ID \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="$PWD/checkpoints/ppo-OLD_JOB_ID/global_step_5"
```

Use this only after the original job ends. Resume requires the same code, data,
runtime, settings and GPU count. It restores actor/critic optimizers, RNG and
dataloader state through VERL. GPU continuation equivalence—including vLLM
sampling—has not yet been established; checkpoint presence alone cannot prove it.

## What the PPO pilot does

1. Render the **same completion tokens** as Stage 1, without a chat template.
   Sample at temperature 1, top-p 1, up to 2,048 tokens; one answer per question.
2. Decode only the response and score it with the unchanged `math-verify-v1`
   adapter. Correct = 1, incorrect/unparseable = 0. Put this reward on the last
   valid response token. Runtime errors/timeouts stop training instead of silently
   becoming wrong-answer labels. Generated Python is never executed.
3. A separate Qwen transformer, initialized from the same base weights with a
   fresh scalar head, predicts each prefix's value **before the next token**.
4. Compute `delta[t] = reward[t] + gamma * V[t+1] - V[t]` and
   `A[t] = delta[t] + gamma * lambda * A[t+1]`, backwards through the response.
   Gamma = 1, GAE lambda = 0.95. EOS and budget exhaustion are terminal with zero
   future value; padding carries no reward or value.
5. Form detached critic targets `return[t] = raw_A[t] + old_V[t]`. Whiten only
   actor advantages, using pinned VERL's sample-variance whitening. Fit the critic
   for two epochs with value clipping 0.5, learning rate 1e-5 and persistent Adam
   state. Critic errors weight valid tokens equally, divided by the fixed response
   budget and batch size; keep its microbatch at one for this normalization.
6. Update the actor for two epochs with PPO clipping 0.2 and learning rate 1e-6.
   Old log probabilities, advantages and targets stay fixed for this batch.
   Average actor losses within each response, then across responses. No KL reward,
   KL loss, entropy bonus or weight decay is added. The pinned
   implementation's extra dual clip is disabled. Log vLLM/actor probability
   differences to expose inference/training discrepancies.

Validation uses greedy decoding on the existing 32-question development split,
before training and every five updates. Training uses the existing 128-question
split, filtering overlong prompts. Neither uses the fresh audit questions or the
official test set. This tiny development run tests the machinery, not research
performance. Reward scoring runs once per batch in a separate CPU process; its
cost is included in VERL's reward timing.

Implementation follows the pinned
[GAE and losses](https://github.com/verl-project/verl/blob/8d9e350ea58c7ad4b50dd14d9dcb50577242c55f/verl/trainer/ppo/core_algos.py),
[pre-token critic](https://github.com/verl-project/verl/blob/8d9e350ea58c7ad4b50dd14d9dcb50577242c55f/verl/workers/critic/dp_critic.py), and
[training loop](https://github.com/verl-project/verl/blob/8d9e350ea58c7ad4b50dd14d9dcb50577242c55f/verl/trainer/ppo/ray_trainer.py).
Numerical tests exercise those functions directly, including a tiny causal Qwen
critic. Full FSDP/CUDA execution and controlled learning remain cluster checks.

## Stage 1 generation (already run)

Pull the changes from GitHub, then submit from the repository:

```bash
cd /usr/xtmp/ms785/rl_ms_project
git pull --ff-only origin main
source scripts/cluster_env.sh
mkdir -p logs
sbatch scripts/submit_stage1.sh
```

This requests one A5000 for up to an hour. It checks the existing environments,
generates **200 responses to 100 fresh questions**, then scores them with the
separate CPU Math-Verify environment. The pinned GSM8K shuffle excludes the first
160 examples used for earlier train/validation diagnostics. The official test set
is untouched. Audit questions must not become parser-development examples if we
want to report their agreement as held out.

Frozen settings: plain completion prompt, no worked example, temperature 1,
top-p 1, two responses per question, 2,048 response tokens, seed 42. The prompt
asks for a boxed answer, but scoring uses Math-Verify's broader extraction.
Generation and scoring are separate: `reward: null` means not yet scored.

Replace JOB_ID with Slurm's number:

```bash
squeue -j JOB_ID
tail -F logs/math-rl-stage1-JOB_ID.out
sacct -j JOB_ID --format=JobID,State,ExitCode
```

Expect `COMPLETED 0:0`. SSH disconnection does not stop the batch job. If generation
succeeds but scoring fails, rerun only scoring; do not generate a new dataset:

```bash
.venv-verifier/bin/python scripts/rescore_stage1.py outputs/stage1-JOB_ID
```

The rescorer refuses to overwrite existing reports. It prints acceptance,
extraction, mixed-pair and truncation rates, plus parser/error status counts.

## Review and report

On the login node, no GPU needed:

```bash
python3 scripts/review_final_answers.py outputs/stage1-JOB_ID
```

Automatic scores are hidden. Enter **two values**:

- `1 1`: clear, correct final answer.
- `1 0`: clear but wrong final answer.
- `0 0`: missing or ambiguous final answer.

Judge the answer to the original question, regardless of boxes. Wrong reasoning
alone does not invalidate a correct final answer. Note ambiguous continuations
and unresolved contradictions. `q` pauses; the same command resumes. New labels
are stored in `human-labels.jsonl`; old development labels are reused in place
when present. Original responses and scores are never rewritten.

After reviewing all 200 responses:

```bash
python3 scripts/review_final_answers.py outputs/stage1-JOB_ID --report
```

The report separates:

- **Verifier audit:** fresh questions, complete review, at least 99% agreement,
  no runtime errors/timeouts, at most 5% truncation.
- **Difficulty:** 5–70% accepted answers and at least 10% mixed pairs.

A high success rate can pass verifier reliability while failing the provisional
difficulty range. That calls for a harder research dataset, not a different parser.
Extraction coverage is reported; there is no boxed-format gate. False accepts and
false rejects both count against human agreement. Passing this audit does not
validate checkpoint resume or establish PPO learning.

## One-time environments

Skip this section if the existing cluster environments work. Keep training and
verification dependencies separate.

For the CPU verifier on Duke:

```bash
python3 -m venv .venv-verifier
.venv-verifier/bin/python -m pip install -r requirements-verifier.txt
```

For training, obtain a compute allocation before installing/downloading:

```bash
srun -p compsci --cpus-per-task=8 --mem=128G --time=02:00:00 --pty bash -i
cd /usr/xtmp/ms785/rl_ms_project
source scripts/cluster_env.sh
"$MATH_RL_RUNTIME" pull "$MATH_RL_IMAGE" \
  docker://verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2
sha256sum "$MATH_RL_IMAGE" > configs/container.sha256
bash scripts/container_exec.sh bash scripts/setup_environment.sh
exit
```

Load the site's Singularity/Apptainer module first if neither executable is
available. The image and caches live beneath `/usr/xtmp/ms785`; the wrapper checks
the image checksum. Setup pins VERL, prepares the model/data, runs CPU checks,
and saves the package inventory. The inherited container's full `pip check` report
is recorded separately; inherited package conflicts do not prove the used GPU
path works or fails. Preflight and actual GPU runs remain necessary.

## What is frozen and what remains

Math-Verify 0.9.0 and parser dependencies are pinned in requirements-verifier.txt.
The adapter records versions, extraction settings, numerical tolerances and
3-second library timeouts. It parses predictions before comparing them to gold;
no reference-guided search is added. Unanchored extraction can still select an
unrelated number. The library is not a semantic judge, and generated code is not
executed. Fresh human agreement is the acceptance criterion.

The existing Stage 0 training config/reward remains a **legacy smoke path**.
Use `submit_ppo.sh` for the completion/Math-Verify PPO path. Memory profiling,
GPU save/resume validation and evidence of learning remain to be collected.
Old parser implementations and superseded
review commands were removed; their source is recoverable from Git history and
saved reports remain untouched.

To rescore old development data with Math-Verify:

```bash
.venv-verifier/bin/python scripts/rescore_stage1.py \
  outputs/diagnostic-12585475/completion-t1.0
python3 scripts/review_final_answers.py \
  outputs/diagnostic-12585475/completion-t1.0 --report
```

## Code and tests

- `prompts.py`: frozen completion prefix plus legacy smoke contract.
- `generate_stage1.py`: fresh sampling, generation and provenance.
- `math_verify_reward.py`: pinned mathematical extraction/comparison.
- `rescore_stage1.py`: immutable scoring reports.
- `review_final_answers.py` and `audit.py`: blind labels and acceptance gates.
- `train.py`, `provenance.py`, `report_resume.py`: existing smoke/resume machinery.
- `completion_dataset.py`, `ppo_runtime.py`: exact completion inputs and seeded initialization.
- `ppo_reward.py`, `verifier_batch.py`: isolated batch scoring for PPO.
- `ppo_checks.py`, `report_ppo.py`: numerical and cluster execution checks.

Run tests with Math-Verify installed so real-library tests are not skipped:

```bash
.venv-verifier/bin/python -m pip install pytest
PYTHONPATH=src .venv-verifier/bin/python -m pytest -q
```
