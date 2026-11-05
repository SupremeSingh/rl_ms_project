# Transferred-feature PPO: Numina SLICE versus RMS two-thirds

Question: does a fixed source-trained representation improve target policy learning
per generated answer/token or allocated online compute? This is a new **online
PPO** experiment, unlike the completed offline prediction studies.

## Locked design

Two arms: frozen Numina-trained SLICE-768 + cumulative Ridge; frozen Qwen block
19 with per-prefix RMS normalization + cumulative Ridge (1536 features). Both
have one head, fresh zero A,b, raw accumulation, epsilon added at solve time,
terminal binary rewards, gamma 1, GAE .95, two PPO actor epochs, lr 1e-6, no buffer,
encoder refresh or backpropagation through the critic. No offline Ridge weights
or statistics initialize PPO. No extra target head-fitting collection is used.
The same reward verifier, prompts and decoding settings apply to both arms.

Use the **source compressor from clarification seed 42**, chosen in advance, not
selected on target test scores. It was fitted after blocking MATH train/test
families from source Numina. The export records source checkpoint identity,
layers, model files and normalization; existing target-trained compressors are
not used. All model/tokenizer files must match `models/qwen-math`, which the
existing GPU preflight checks. A mismatch fails during CPU preparation.

Target questions: the clarification study's 2000 MATH training / 500 validation /
294 eligible MATH-500 test questions (actual counts saved). They have already
been inspected; this is exploratory, not a fresh benchmark. Original selection
was numeric-reference/template-based, not semantic/pretraining decontamination.
No target train/validation/test family overlap is introduced. This study reuses
questions, but samples **fresh answers from each changing actor**.

### Stages

1. CPU prepare verifies the completed clarification source and exports features
   only. Real parallel GPU smoke: two PPO updates per arm, validation only.
2. Eight validation pilots: each arm at epsilon [.01,100,1000,10000], seed 601, 20 updates.
   Select each arm's epsilon by normalized validation accuracy AUC at 0/10/20;
   ties favor smaller epsilon. This is a small fixed online search, not a claim
   of optimal regularization. Upper-boundary selections are flagged for review
   before the main run; the grid does not expand automatically online. Offline epsilon 1000 is not copied automatically.
3. Ten fresh actor runs: two arms × seeds 701–705, **60 updates × 64 answers**.
   Each run uses two A5000 GPUs. Source compressor is identical across all seeds.
4. Paired report: final answer accuracy and normalized accuracy AUC versus update
   count, approximate seed/question bootstrap with two Bonferroni nominal contrasts.
   Other curve comparisons are descriptive. No checkpoint is chosen on test data.

All runs use the existing online **2048-token response cap**, different from
1024-token offline generation. The validation pilots address the online setting;
these are not a literal replay of the offline task. All heads start from scratch.
The layer baseline has RMS normalization but no offline target centering/whitening;
SLICE retains its source-trained centering/scaling. Per-arm epsilon selection
compares these complete recipes rather than claiming identical penalty geometry.

## Early observations without actor restarts

The optional `LearningCurveTrainer` keeps normal validation and, after actor
updates 10/20/30/40/50/60, records a fixed greedy test evaluation. It disables its
hook during nested evaluation and restores training counters/loaders. Normal
PPO/GAE/critic fitting order is preserved. Smoke uses checkpoints 1 and 2.

Each curve records cumulative training response tokens (including stored EOS),
answers, accuracy, truncation, elapsed online seconds including monitoring, and
critic-fitting seconds. Initial encoder/worker setup and base evaluation are
reported separately; final accounted GPUh includes them and final evaluation.
Training-time GPUh includes validation, curve evaluation and checkpointing.
These are wall-time × two reserved GPUs, not GPU-utilization measurements.

Predeclared main endpoints: final accuracy and normalized test accuracy AUC versus
updates over 0–60. Token/GPU-time curves describe efficiency but do not establish a
causal label-saving factor or equivalence. More validation/test calls add cost;
that overhead is retained, not subtracted. Test outputs are reviewed for verifier
failures; all attempts count. No training-target error is used as held-out critic
accuracy. Historical-policy statistics remain uncorrected mixtures.

Source feature training is prior investment; pilot/smoke jobs are additional
costs. Report pilot costs separately from main runs. The former cost audit cannot
assign the prior compressor's isolated training cost from bundled timing records;
no fabricated deployment break-even estimate should be attached to this study.

## Run on Duke

After committing/pushing the new files/runtime changes:

```bash
cd /usr/xtmp/ms785/rl_ms_project
git pull --ff-only origin v2
source scripts/cluster_env.sh
mkdir -p logs
python3 tools/transfer/submit_online_transfer.py --mode smoke \
  --source outputs/transfer-clarify-v1 --out outputs/transfer-ppo-v1 \
  --concurrency 4
```

Wait for the smoke gate:

```bash
cat outputs/transfer-ppo-v1-submission/jobs.tsv
cat outputs/transfer-ppo-v1/smoke.json
```

Then run validation calibration (up to four two-GPU tasks concurrently):

```bash
python3 tools/transfer/submit_online_transfer.py --mode pilot \
  --out outputs/transfer-ppo-v1 --concurrency 4
```

When `selection.json` exists and pilot jobs completed:

```bash
cat outputs/transfer-ppo-v1/selection.json
python3 tools/transfer/submit_online_transfer.py --mode train \
  --out outputs/transfer-ppo-v1 --concurrency 4
```

The ten runs and report form an automatic dependency chain. Final:
`outputs/transfer-ppo-v1/report.txt`, `summary.json`, per-case `curve-STEP.json`
and `curve-STEP-test.jsonl`. Logs/IDs: `<OUT>-submission/`; case detailed logs also
live under the study. No code changes between stages. No actor partial resume;
failed directories are preserved. Submission timeouts stop without blind retries.
CPU gate/selection/report are not GPU allocations. Resource availability controls
overlap; no wall-time guarantee is made before observing this GPU smoke.

These files reuse the current installed packages; no package dependency/version
change is required. GPU smoke remains mandatory because local CPU tests cannot
exercise VERL/FSDP/vLLM integration or GPU memory limits.

After report completion, generate PNG/PDF plots without new model execution:

```bash
bash scripts/container_exec.sh "$PWD/.venv-critic/bin/python" \
  tools/transfer/plot_online_transfer.py --study outputs/transfer-ppo-v1
```

Figures are saved under `figures/learning-efficiency.{png,pdf}`. Mean curves use
matched checkpoints across seeds, not cherry-picked test maxima. Test/validation
monitoring overhead and fixed setup/base evaluation are included in the GPU-time
axis; prior source training and tuning pilots remain separate costs.
