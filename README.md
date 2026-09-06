# Math RL: Stage 0

Goal: load -> generate -> score -> update -> save -> resume.

This starter targets Linux x86_64, Slurm, Apptainer, and one BF16-capable NVIDIA
GPU. Start with a 40 GB or 80 GB A100. Adjust the Slurm resource/partition/account
options to the cluster. Other hardware and container runtimes need adaptation.

Use VERL v0.4.1 at commit 8d9e350ea58c7ad4b50dd14d9dcb50577242c55f with the documented
application image below. This is an intentionally fixed older baseline.
The package accepts Python >=3.10 to match that image (replaces the initial >=3.12).

## Environment

Run from math-rl on a machine allowed to pull container images:

```bash
export MATH_RL_IMAGE="$PWD/../verl-v041.sif"
apptainer pull "$MATH_RL_IMAGE" \
  docker://verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2
sha256sum "$MATH_RL_IMAGE" > configs/container.sha256
apptainer shell --bind "$PWD:$PWD" --pwd "$PWD" "$MATH_RL_IMAGE"
```

Inside the image, create a local environment using its existing GPU packages:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --no-deps \
  'verl @ git+https://github.com/verl-project/verl.git@8d9e350ea58c7ad4b50dd14d9dcb50577242c55f'
python -m pip install -e '.[dev]'
python -m pip check
python -m pip freeze > configs/environment.txt
python -m pytest -q
python scripts/prepare_data.py
python -m math_rl.train --cfg job
exit
```

The image supplies VERL's GPU dependencies; do not upgrade Torch or vLLM
independently. Resolve any `pip check` errors before training. The environment
snapshot records resolved versions; the SIF checksum identifies the exact runtime
but does not download it. Keep the SIF for future reproduction.

Data preparation resolves and records Hugging Face commit IDs in
configs/assets.json on first success and reuses them on subsequent runs. It
prepares 128 training and 32 disjoint validation examples from GSM8K's training
split, leaving its test split untouched, and downloads the Qwen2.5-Math-1.5B model. Preprocessing needs network access; training uses local
files. Long prompts may be filtered by VERL, reducing the counts.

## Save and resume

From the host, in math-rl, with MATH_RL_IMAGE still exported:

```bash
mkdir -p logs
sbatch scripts/submit_smoke.sh trainer.total_training_steps=2
```

Wait for successful completion and checkpoint global_step_2. Then:

```bash
ls checkpoints/smoke/global_step_2
ls checkpoints/smoke/global_step_2/actor
sbatch scripts/submit_smoke.sh \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path=checkpoints/smoke/global_step_2 \
  trainer.total_training_steps=4
```

Explicit resume fails if the checkpoint is missing. It should report loading
step 2 and proceed through steps 3 and 4. Keep the same GPU count, model, data,
and configuration. We use a constant learning-rate schedule for this short test.
Use a new trainer.default_local_dir for each genuinely fresh run.

Verify:
- Finite rewards and losses, and nonzero actor gradients in at least one update.
- Sampled answers include valid Answer: lines. This numeric scorer intentionally
  rejects fractions, units, prose after the answer, and boxed answers.
- Actor model, optimizer, extra state files and data.pt exist at step 2.
- Logs confirm model/optimizer/extra restoration and do not warn of missing data.pt.
- The second job produces global_step_4 and retains its optimizer/data state.

Successful resumption is not evidence of bitwise-identical sampled trajectories.
With GRPO, groups whose rewards are all identical have zero task advantages.
If that happens throughout this tiny run, increase the prompt budget before
claiming a successful learning update. Four steps are an infrastructure test,
not an accuracy result.

## Files and VERL map

- data.py constructs prompt/reference records; prepare_data.py downloads assets.
- reward.py supplies a rule-based reward using VERL's custom reward interface.
- smoke.yaml inherits VERL's defaults, choosing GRPO with 8 prompts x 4 responses.
- train.py forwards config and CLI overrides to verl.trainer.main_ppo.
- submit_smoke.sh supplies Slurm resources and the container environment.
- tests/test_reward.py checks the scoring contract.

VERL code worth reading, at the pinned tag:
- verl/trainer/main_ppo.py: configuration and worker setup.
- verl/trainer/ppo/ray_trainer.py: overall training loop and save/load.
- verl/trainer/ppo/core_algos.py: advantages and policy losses.
- verl/workers/reward_manager/naive.py: calls the custom reward.

https://github.com/verl-project/verl/tree/v0.4.1
https://verl.readthedocs.io/en/v0.4.1/start/quickstart.html

Local validation: 12 reward tests pass; the YAML composes against v0.4.1 defaults;
a synthetic dataset record survives a Parquet round trip with its prompt, answer,
and ID preserved. GPU training, full downloads, and runtime installation have
not been executed here and must be verified on the cluster.
