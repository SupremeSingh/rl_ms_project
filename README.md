# Math RL

Educational PPO critic experiments with **Qwen/Qwen2.5-Math-1.5B base** and pinned
VERL. Current task: audit Math-Verify on fresh GSM8K responses before building PPO.
The original GRPO smoke ran, but its zero-reward updates did not establish learning.
See [the experiment plan](TOKEN_CRITIC_EXPERIMENT_PLAN.md).

## Next run on Duke

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

The existing Stage 0 training config/reward remains a **legacy smoke path**. Do
not launch it expecting the audited completion/Math-Verify setup: promotion into
training, conventional PPO/GAE implementation, memory profiling and save/resume
validation are next, after the audit. Old parser implementations and superseded
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

Run tests with Math-Verify installed so real-library tests are not skipped:

```bash
.venv-verifier/bin/python -m pip install pytest
PYTHONPATH=src .venv-verifier/bin/python -m pytest -q
```
