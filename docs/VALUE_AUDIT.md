# Monte Carlo audit of critic values

**Question: do our trained critics accurately estimate the probability of a
correct completion from the same partial answers?**

This is a diagnostic, not another test-accuracy leaderboard. A 2×3 scatter plot
compares unclipped predictions against independent continuation rewards. The
horizontal axis is an estimated reference value, not exact ground truth.

## Six matched arms

| Arm | Features / baseline | Credit unit during training |
|---|---|---|
| `slice-token` | Frozen sparse SLICE, 768 features + intercept | Token |
| `rms-layer-token` | Frozen two-thirds layer, per-state RMS normalization | Token |
| `slice-chunk` | Frozen sparse SLICE, 768 features + intercept | MATH chunk |
| `rms-layer-chunk` | Frozen two-thirds layer, per-state RMS normalization | MATH chunk |
| `ppo` | Trainable transformer critic, validated `higher_lr` profile | Token |
| `grpo` | Four-answer question-level mean reward; no value network | Answer group |

The previous chunk comparison used **raw** layer features. Applying RMS to those
features after fitting would invalidate the old weights. This workflow therefore
trains all six actors afresh with one predeclared seed (1001), 120 updates × 64
answers, the existing Numina splits and selected compressor. It does not claim
to reuse the old actor checkpoints or establish a multi-seed ranking.

Ridge retains one cumulative system, epsilon .01, frozen feature coordinates and
no replay. Gamma is 1 and GAE lambda .95. Chunk fallback uses at most 32 original
tokens. RMS readout normalization is applied **both while fitting and scoring**;
it has no learned gain, no centering and epsilon equal to Qwen's `rms_norm_eps`.
The new option defaults to `none`, preserving earlier recipes.

Only validation questions are evaluated during these new training runs. Neither
the prefix bank nor MC outcomes are used to tune penalties, select checkpoints,
refit heads or update actors. Source representation-training/selection families
remain excluded through the existing preparation audit. Test questions were
previously inspected in earlier studies; this is not an independent benchmark.

## Reference values and shared states

For each state, concatenate the exact question-prompt IDs and original prefix
IDs. Independently continue the panel's frozen actor K times, with temperature
1, top-p 1, top-k −1 and the original total response cap of 2,048. A prefix of
600 response tokens permits another 1,448, not 2,048. Verify the **full answer**
(prefix plus continuation) using the same isolated Math-Verify rule as training.

`reference = number of reward-1 completions / K`

This follows [VinePPO's MC value-analysis approach](https://arxiv.org/html/2410.01679v2#S7),
which used 256 samples per state. GAE lambda does not enter the reference value.
Parse failures and bounded verification timeouts remain zero under the training
reward rule, with separate status counts. Infrastructure errors stop the stage.
Thus we measure verifier-defined reward probability, not infallible correctness.

Questions are ordered by a fixed hash, then assigned round-robin to six bank
actors independently of outcomes. Each assigned actor generates one answer.
Up to four chunk-start positions are selected deterministically across its
progress (including the empty answer prefix); short answers may supply fewer.
No post-terminal state is used. Original action IDs are never re-tokenized.

Every panel scores the **same bank** of states, but its MC continuations come
from its **own actor**. Values can therefore differ between panels even at the
same prefix. This controls the state distribution, not the policy. It is not
the stronger experiment of fitting every critic to one common frozen policy.

The GRPO panel uses an independent four-answer group starting at the question;
its mean is shared across that question's prefixes. Those answers are separate
from all reference samples. Normalized GRPO advantages are not value estimates.

## Checkpoints, timing and uncertainty

Training explicitly saves complete Hugging Face actor snapshots and the PPO
token-critic snapshot. Gates require actual weight files, matching token
vocabularies, model/data/filter hashes, PPO functional checks and the Ridge
solve residual. Config/tokenizer-only directories are rejected.

The final cumulative Ridge head was fitted on rollouts **before** the final
actor update. The audit retains and records this one-update lag. It evaluates
the saved final actor/critic pair; it does not claim the head was refitted to
that exact final policy. PPO's last critic targets also come from the final
pre-update rollouts. Both base encoder and heads remain frozen throughout
the reference generation.

Reports include equal-question MAE/RMSE, unbounded prediction frequency and a
Monte Carlo noise-corrected MSE estimate (which may be negative at finite sample
size). Scatter axes are shared and include out-of-range predictions. At K=256,
the worst-case reference standard error is .03125. Per-state Wilson intervals
are saved even when all sampled outcomes agree.

Question-bootstrap intervals are conditional on observed references. Separate
question/MC sensitivity ranges use Jeffreys Beta draws for reference
uncertainty; those are **not frequentist confidence intervals**. Neither
includes actor-seed uncertainty. An error-versus-prefix-progress plot is also
saved. Low MAE alone does not establish better action ranking or better PPO.

## Run on Duke

Push the completed changes from the Mac first. On Duke, from the repository
root, with no jobs using this checkout:

```bash
git switch v3
git pull --rebase origin v3
source scripts/cluster_env.sh
srun -p compsci --cpus-per-task=8 --mem=128G --time=00:45:00 --pty bash -i
source scripts/cluster_env.sh
bash scripts/container_exec.sh "$PWD/.venv/bin/python" -m pip install 'matplotlib>=3.8,<4'
bash scripts/container_exec.sh bash scripts/install_critic_package.sh
exit
```

Use a **new** study directory. The source supplies the preserved full Numina
data and filter, not old actor weights:

```bash
source scripts/cluster_env.sh
python3 scripts/submit_value_audit.py --mode smoke \
  --source outputs/numina-no-folds-v1 \
  --out outputs/numina-value-audit-v1 --concurrency 4
```

Smoke trains six two-update actors, restores their checkpoints, builds a tiny
bank, samples reference continuations, and renders all six panels. It runs the
entire path; passing is not evidence of estimation accuracy.

```bash
cat outputs/numina-value-audit-v1-submission/jobs.tsv
cat outputs/numina-value-audit-v1/smoke/verification.json
```

Only after smoke reports `passed: true`, launch the matched training and pilot:

```bash
python3 scripts/submit_value_audit.py --mode pilot \
  --out outputs/numina-value-audit-v1 --concurrency 4
```

| Phase | Training | Questions | Prefixes/question | MC samples/state/actor | Maximum reference completions |
|---|---|---:|---:|---:|---:|
| Smoke | 2 updates × 6 actors | 2 | 2 | 4 | 96 |
| Pilot | 120 updates × 6 actors | 10 | 4 | 64 | 15,360 |
| Full | Reuses pilot-trained checkpoints | 50 | 4 | 256 | 307,200 |

Bank trajectories and independent GRPO groups are additional. Training tasks
allocate two A5000 GPUs; bank/prediction/reference tasks allocate one. Concurrency
4 means at most four GPU tasks in that stage (eight GPUs during training, four
during generation). The full reference stage has 24 shard tasks; it is not
submitted automatically. Use pilot timers and response-token counts to decide
whether the full budget is practical; queue time and generation throughput vary.

```bash
cat outputs/numina-value-audit-v1/pilot/verification.json
cat outputs/numina-value-audit-v1/pilot/report.txt
# After reviewing the pilot and its cost:
python3 scripts/submit_value_audit.py --mode full \
  --out outputs/numina-value-audit-v1 --concurrency 4
```

Each phase writes `report.txt`, `summary.json`, `verification.json`,
`value-scatter.png`, `value-scatter.pdf` and `error-by-progress.png`. Copy plots
from the cluster to view them locally. `summary.json` retains predictions,
reference counts, intervals, timings and verifier statuses.

## Failures and recovery

Every submitted job ID and intended command is recorded under the sibling
`-submission/` directory. A failed or ambiguous `sbatch` response stops the chain;
do not immediately resubmit because the scheduler may have accepted the job.
Check `squeue`, `sacct` and the submission sidecar first.

Training never silently resumes partial actors. Generation caches completed
sample blocks atomically, with manifest/bank/seed identity checks. After a
generation failure, inspect its status and rerun that exact stage/task in a GPU
allocation with the same code and manifest. For example, inside a one-A5000
allocation:

```bash
source scripts/cluster_env.sh
export MATH_RL_GPU=1
bash scripts/container_exec.sh env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE \
  "$PWD/.venv/bin/python" -m math_rl.value_audit \
  --stage mc --phase full --task 0 --out outputs/numina-value-audit-v1
```

Full MC task `shard * 6 + arm_index` covers four shards and six arms in the table's
order. Rerunning a generation task reuses existing complete blocks rather than
spending again. After all tasks pass, rerun the report stage in a CPU allocation.
Do not change runtime files, environments, checkpoints or sealed manifests
during queued/running work. Failed allocations and reused source feature
training are additional to the reference-stage timers.
