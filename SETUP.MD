# Setup and acceptance

Use the checkout for the intended study on Duke (`v2` for the no-fold experiment).
Do not change its code or environments while
jobs use it. Saved outputs are inputs to reproduction, not disposable caches.
See [README](README.md) for results and [reproduction](docs/REPRODUCING.md) for jobs.

## Existing installation

After syncing this branch, enter a CPU allocation and refresh the editable package
and project metadata. This does not replace pinned Torch/Transformers:

```bash
srun -p compsci --cpus-per-task=8 --mem=128G --time=00:45:00 --pty bash -i
source scripts/cluster_env.sh
bash scripts/container_exec.sh bash scripts/install_critic_package.sh
bash scripts/container_exec.sh "$PWD/.venv/bin/python" \
  scripts/check_slice_release.py --out outputs/slice-prod-release-v1
cat outputs/slice-prod-release-v1/verification.json
exit
```

Require `passed: true`. Use a new output directory on every attempt. The command
builds and tests a wheel outside the checkout; it never publishes the package.

## First installation

You need Apptainer or Singularity, repository access and a compute allocation.
Configure `MATH_RL_ROOT` / `MATH_RL_IMAGE` if the default `/usr/xtmp/ms785` is wrong.

```bash
srun -p compsci --cpus-per-task=8 --mem=128G --time=02:00:00 --pty bash -i
source scripts/cluster_env.sh
"$MATH_RL_RUNTIME" pull "$MATH_RL_IMAGE" \
  docker://verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2
sha256sum "$MATH_RL_IMAGE" > configs/container.sha256
bash scripts/container_exec.sh bash scripts/setup_environment.sh
bash scripts/container_exec.sh bash scripts/setup_ppo.sh
bash scripts/container_exec.sh bash scripts/setup_critic.sh
exit
```

For a historical replay, use the original preserved container and model pins;
do not substitute a freshly pulled tag and claim byte-identical reproduction.
`prepare_assets.py --revision MODEL_COMMIT_SHA` downloads the pinned model.
On first setup only, omitting the revision resolves and records the current SHA.
The container itself must supply the pinned VERL runtime dependencies.

| Environment | Purpose |
|---|---|
| `.venv` | Pinned VERL commit `8d9e350ea58c7ad4b50dd14d9dcb50577242c55f`, PPO and GRPO. |
| `.venv-critic` | Offline SLICE training, with the container's Torch preserved. |
| `.venv-ppo-verifier` | Torch-free, pinned Math-Verify subprocess. |

Do not merge symbolic dependencies into the training environment. Setup records
runtime snapshots and verifies that offline installation preserved container Torch.

## Cluster acceptance after this cleanup

First capture the final study using the [replay command](docs/REPRODUCING.md#replay-the-final-comparison).
Then use the previously verified package artifact to check the refactored PPO code:

```bash
source scripts/cluster_env.sh
mkdir -p logs
sbatch scripts/submit_slice_ppo_smoke.sh \
  outputs/slice-package-smoke-v2/training/artifact \
  outputs/slice-prod-smoke-v1 \
  outputs/numina-prod-replay-v1/data/questions.json
```

When finished:

```bash
cat outputs/slice-prod-smoke-v1/status.json
cat outputs/slice-prod-smoke-v1/verification.json
```

This runs two actor updates each for Ridge and LSTD(0.9), checking frozen features,
finite 769-dimensional heads, two question folds and accumulated statistics.
It is not an accuracy benchmark. The full six-method study has a separate smoke
gate before its array of actor runs.

## Local development

```bash
python -m pip install -e './packages/slice-critic[train,dev]'
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/check_slice_release.py --out outputs/slice-local-release-v1
```

CPU development skips checks requiring the pinned VERL runtime. GitHub CI tests
installed wheels and the supported CPU pipeline; GPU acceptance stays on Duke.
