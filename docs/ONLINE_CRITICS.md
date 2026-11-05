# Seven-arm online critic comparison

Fresh PPO actor runs compare:

| Arm | Feature/critic update |
|---|---|
| SLICE + Ridge | Frozen Qwen/compressor, one cumulative matrix system, epsilon .01 |
| SLICE + LSTD(.05) | Same frozen features and accumulation; trace lambda .05 |
| SLICE + LSTD(.99) | Same frozen features and accumulation; trace lambda .99 |
| Frozen SLICE + MLP | Frozen encoder/compressor; fit MLP to detached fresh GAE returns |
| Trainable SLICE + MLP | Frozen encoder; fit projection and MLP to detached fresh GAE returns |
| Conventional PPO | Transformer critic with the source study's validation-selected profile |
| GRPO | Within-question outcome-normalized advantages; no critic |

Every arm gets 120 updates x 64 answers, 2,000 source test questions, greedy
evaluation, 2,048 response tokens, and two A5000 GPUs. Seeds 501/502/503 give 21
runs. Actors start from the same base model. Gamma=1; **GAE lambda=.95** is fixed;
**LSTD trace lambda** is a separate setting. These are fresh actor seeds on reused,
inspected questions, not a new benchmark. The source's PPO calibration is reused;
its historical tuning and offline filter-learning costs remain additional.

## Neural behavior and fairness limits

The online MLP is `768 -> 256 -> ReLU -> 1 -> sigmoid`. All SLICE arms start with
the exact same offline compressor. Neural heads start fresh (zero final logits,
initial value .5); no offline value head or tuned joint compressor is loaded.
Frozen encoder feature extraction remains FP32. No output LayerNorm or Ridge
clipping is added. The joint arm receives raw concatenated, RMS-normalized block
coordinates; only the central compressor changes. Stored normalization is fixed.

For each fresh rollout, neural values are predicted **before fitting**. The pinned
VERL GAE computes detached advantages and return targets. The neural critic then
fits those returns, while the actor uses the original detached advantages. There
is no recomputation of advantages with the just-fitted neural head. This differs
from the preserved Ridge/LSTD fit-then-predict path, which uses observed rewards
to update its matrices before GAE. It is a comparison of complete critic recipes,
not architecture alone.

Neural fitting uses two epochs, answer minibatches of 16, sequence-mean/token-mean
clipped value loss (clip .5), AdamW weight decay .001 and gradient clipping at 1.
Frozen-head learning rate is 3e-4; joint head/projection rates are 3e-5 / 3e-6,
motivated by the offline tuning but not claimed optimal for GAE targets. There is
no activation penalty online. Sigmoid bounds values, but GAE targets are not
clipped. Neural models retain weights/optimizer state, **not examples or A,b**.
Historical-policy sufficient statistics remain only in the linear arms.

**GPU backend:** one authoritative neural critic and optimizer live on rank 0 of
the existing actor GPU group. Both GPUs still handle actor/encoder work; the small
critic uses one GPU without unsynchronized replicas or an additional allocation.
Blocking RPCs run after actor/encoder offload and before the next actor update.
The Ray driver owns no neural model or GPU context. CUDA prediction/fit replies
are checked; CPU fallback is refused.

The joint arm transports 7,680 coordinates per token, versus 768 for compressed
arms. Features are sent to the owning worker once, kept for one fresh rollout,
and streamed to the GPU in token chunks. Fit RPCs subsequently send only detached
targets/masks; the feature cache is cleared after fitting. Only the final neural
checkpoint is returned to the driver, avoiding full-weight copies every update.
Prediction and fit RPC timings, worker fitting wall time, process peak CUDA memory
(including earlier actor/encoder allocations), and feature bytes are logged.
Transfer overhead remains and is counted in training cost;
we do not claim that this unmeasured implementation is globally optimal.

The report pairs test questions/seeds, retains failed target-fit-improvement gates
as diagnostics rather than discarding seeds, and reports seven predeclared paired
contrasts with approximate Bonferroni bootstrap intervals. Only three seeds and
inspected test data limit inference. Review verifier outputs before claims.

## Prepare a new GPU study

After pushing from your computer, prepare a **new** study on Duke. Source inputs
are copied/checksummed; old actor runs and smoke records are not reused. Current
source code is snapshotted separately from the historical input manifest.

```bash
cd /usr/xtmp/ms785/rl_ms_project
git pull --ff-only origin v2
source scripts/cluster_env.sh
mkdir -p logs

bash scripts/container_exec.sh "$PWD/.venv/bin/python" \
  tools/diagnostics/online_critics.py --stage prepare \
  --source outputs/numina-no-folds-v1 \
  --out outputs/numina-seven-critics-gpu-v1
```

Use a new directory even if you prepared the earlier CPU prototype. The new
protocol records GPU ownership, source files and submission tools separately.

## Parallel smoke first (recommended)

```bash
python3 tools/diagnostics/submit_critic_study.py \
  --out outputs/numina-seven-critics-gpu-v1 \
  --mode smoke --smoke-concurrency 7
```

This submits a seven-task smoke array (up to 14 GPUs) and a CPU gate dependent on
the entire array succeeding. Tasks write separate records; no task writes the
shared gate. The gate re-audits all seven saved runs and checks manifest/result
hashes before publishing `smoke.json`. Slurm availability controls actual overlap.
The smoke runs two real actor updates per arm, evaluating validation only. It
audits matrix dimensions/trace settings, all-question accumulation, neural head
updates, frozen normalization/compressor invariants, and conventional PPO health.
It has not yet been executed on cluster GPUs for these new neural paths.

```bash
cat outputs/numina-seven-critics-gpu-v1/jobs.tsv
cat outputs/numina-seven-critics-gpu-v1/smoke.json
tail -n 30 outputs/numina-seven-critics-gpu-v1/smoke-mlp-joint.log
```

After the smoke passes and its runtime/memory are acceptable:

```bash
python3 tools/diagnostics/start_verified_critic_training.py \
  --out outputs/numina-seven-critics-gpu-v1 \
  --concurrency 6
```

Each task verifies the smoke gate and immutable code/data first. Six concurrent
tasks means up to 12 GPUs, subject to cluster availability. Twelve hours per task
is a ceiling, not an ETA; estimate the total from smoke and first completed runs.
Do not pull new code or edit runtime files during the study. Completed runs can
be reused under this protocol; partial actor training does not resume. On a
submission timeout, inspect the queue before resubmitting to avoid duplicates.

The training submission includes a CPU report dependent on all 21 runs succeeding.
Alternatively, to queue the entire gated chain in one command instead of the
two-step smoke/train procedure above:

```bash
python3 tools/diagnostics/submit_critic_study.py \
  --out outputs/numina-seven-critics-gpu-v1 \
  --mode all --smoke-concurrency 7 --concurrency 6
```

Choose one submission route; do not submit both. Smoke failure prevents the full
array from starting. The stdlib-only submission helper confirms numeric job IDs
and records `jobs.tsv` after every success. A timeout or invalid ID stops the
chain; per-stage markers prevent accidental duplicate submissions. Inspect Slurm
before retrying an uncertain submission. Marker removal is never automatic.

Results are `report.txt` and `summary.json`; report generation refuses missing
runs. All actor answers, verifier statuses, neural audits and final critic weights
remain available. The CUDA unit test and worker/Ray integration still require a
real cluster smoke; local CPU tests cannot certify throughput or GPU memory use.

## Recovery from a completed-gate dependency rejection

The original split submission helper used `afterok:<gate job>` even after the
saved gate had passed. Slurm can reject this once that job is no longer available
to the controller. The additive `start_verified_critic_training.py` helper checks
the saved gate, manifest and pinned files instead and omits this obsolete
dependency. The report still depends on the newly submitted training array.
It does not change any runtime/tool file hashed into an already prepared study.

If the original helper explicitly failed with `Job dependency problem`, use:

```bash
python3 tools/diagnostics/start_verified_critic_training.py \
  --out outputs/numina-seven-critics-gpu-v1 \
  --concurrency 6 --recover-dependency-rejection
```

The flag acknowledges that specific rejection, not an unknown submission timeout.
The helper checks for live and recent `slice-critics-run` jobs before archiving
the unconfirmed old marker. It refuses recovery if it finds any named job or an
already recorded training ID. Accounting queries failing also stop submission.
The marker is preserved under `run-rejected-*`, not deleted. Do not use this flag
for a timeout whose outcome has not been established. No study preparation or
smoke rerun is needed. Future smoke-first studies can use the new helper without
the recovery flag; the one-shot `--mode all` chain remains supported.
