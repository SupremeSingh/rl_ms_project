# Does reasoning-chunk learning improve SLICE PPO?

A focused, predeclared 2×2 experiment. All arms use cumulative **Ridge** so the
comparison does not confound chunking with LSTD trace settings or neural fitting.
The six-method integration smoke already passed; this study adds a matched smoke
using the full-study compressor and both current token/chunk paths.

## Four arms, one question

| Arm | Frozen representation | Fitting states and GAE steps |
|---|---|---|
| slice-token | Existing Numina SLICE, 768 features | Every token |
| slice-chunk | Exactly the same SLICE | Pre-chunk prefixes |
| layer-token | Raw Qwen block 19, 1,536 features | Every token |
| layer-chunk | Exactly the same raw block | Pre-chunk prefixes |

This is the **raw** two-thirds control from the current branch, without added
RMS normalization; do not label it as the earlier normalized-layer transfer arm.
SLICE retains its existing per-block RMS and frozen source normalization. Its
compressor is not retrained for either arm. We use the source study's actual
full trained filter, **not** the small package-smoke artifact.

Fixed in all arms: Qwen2.5-Math-1.5B actor initialization, frozen feature encoder,
one cumulative head, raw A,b sums, epsilon .01 (including intercept), gamma 1,
GAE lambda .95, 16 prompts × 4 answers per update, actor optimizer/learning rate,
two actor PPO epochs, 512-token prompt limit, 2048-token response cap, two A5000
GPUs/run, and no answer replay or critic backpropagation.

Five new actor seeds **901–905** × four arms = **20 runs**. Each run has
**120 updates / 7,680 training answers**. Same prepared Numina question splits
and same ordered **2,000-question test** in every run. Validation occurs every
30 updates; test evaluation is greedy at base and final only. No best-test
checkpoint selection, epsilon sweep, or outcome-dependent seed stopping.
Generated answers naturally diverge after actor updates; we match prompt sampling
seeds/budgets, not force identical answers from different policies.

## What this comparison can establish

Two primary contrasts: **chunk minus token final actor accuracy**, separately
for SLICE and the raw layer. Report mean and every seed difference, approximate
paired seed/question bootstrap intervals (4,000 draws), and Bonferroni nominal
adjustment for the two accuracy contrasts. A difference interval spanning zero
is inconclusive, not proof of equivalence. Five seeds do not guarantee power to
detect small effects; these intervals are approximate, not universal guarantees.

The difference between the two chunking effects is an exploratory interaction.
A larger SLICE effect does not establish that sparsity alone caused it.

The intervention is a **change of temporal granularity**: .95 now decays once per
chunk, fitting weights chunks rather than every token, and .01 has different
relative strength with fewer observations. It preserves our chosen constants;
it does not isolate text segmentation alone from those consequences. A future
matched-horizon or regularization ablation would answer a narrower mechanism
question. No claim of unbiased current-policy evaluation: old-policy statistics
remain uncorrected in both modes.

## Cost and interpretation

The MATH heuristic can fail on malformed/repeated LaTeX. Such answers use fixed
groups of at most **32 original tokens**, with unchanged rewards and every sampled
token retained. The final group can be shorter; EOS counts as a token. Normal
heuristic chunks are unchanged. Per-arm fallback
fractions appear in the report and per-update counts in `chunk-metrics.jsonl`.
A high fallback rate limits interpretation as reasoning-step credit assignment.
The size is declared before preparing the study, saved in its manifest, and held
fixed in both chunk arms. It matches the rough granularity of saved Numina
training answers (512 re-tokenized responses; normal chunk mean 18.2, median 16,
p75 25, p95 40 tokens). This is not a tuned PPO optimum. No failures were found
in those saved samples; the aborted update was not saved and is not covered.

Per arm report training and accounted GPU-hours, actual generated training
response tokens, fitted state counts, and critic-fitting seconds. Training includes
validation and checkpointing; accounted hours also include worker initialization,
base and final evaluation. Both mean tables are per run. Smoke costs are separate;
prior offline compressor training is additional. Reserved wall-time × two GPUs
is runtime-based GPU-hour accounting, not utilization, FLOPs, peak memory, or an
inference speed benchmark. Accounted hours cover the listed model-runtime phases;
they are not a complete Slurm allocation ledger and omit launcher/startup and
controller-audit overhead. Scheduling/hardware variation can affect measured cost.

Possible outcomes:

- Accuracy interval favors chunks: evidence for the chunk recipe in this setup.
- Similar means and lower cost: promising efficiency trade-off, not equivalence.
- Fewer fitted states but unchanged total cost: encoder/generation/actor work
  dominate; do not claim end-to-end speedup from matrix-update counts alone.
- Chunks lose accuracy: retain token learning for this recipe; chunking isn't
  automatically an improvement because it helped VinePPO's different estimator.

Data, model and filter identities are checked before submission work. Full
source data are reused; no new offline corpus is collected. Known compressor
family exclusions are checked when present in the legacy filter; their presence
is recorded. Missing old exclusion metadata is not retrospective proof of
independence. Numeric-template checks are not semantic/pretraining decontamination.
The questions have been inspected in earlier studies, so this is a controlled
follow-up, **not an independent benchmark replication**.

## Run

Commit/push v3, then on Duke:

```bash
cd /usr/xtmp/ms785/rl_ms_project
git fetch origin
git switch v3
git pull --ff-only origin v3
source scripts/cluster_env.sh
mkdir -p logs
python3 scripts/submit_chunk_comparison.py --mode smoke \
  --source outputs/numina-prod-replay-v1 \
  --out outputs/numina-chunk-comparison-32-v1 --concurrency 6 \
  --chunk-fallback-tokens 32
```

CPU prepare → four parallel two-update GPU smokes → CPU gate. Each smoke uses
64 train / 16 validation questions from the same source and the full-study
compressor. Never use a tiny smoke-trained compressor for the main accuracy study.
Logs and IDs live beside the output in `<OUT>-submission/`.

```bash
cat outputs/numina-chunk-comparison-32-v1-submission/jobs.tsv
cat outputs/numina-chunk-comparison-32-v1/smoke.json
```

After `passed: true`:

```bash
python3 scripts/submit_chunk_comparison.py --mode train \
  --out outputs/numina-chunk-comparison-32-v1 --concurrency 6
```

This submits all 20 runs (up to six at a time, **12 GPUs**) and a dependent CPU
report. Four waves of tasks; each task has a 12-hour limit. Past runs suggest
roughly one to two days of execution at this concurrency, but queueing, response
lengths and hardware prevent a promised turnaround. Summed GPU-hours are not
wall-clock duration. Arm order rotates across seeds to reduce scheduling-order
confounding; node types are not perfectly matched pairs.

```bash
cat outputs/numina-chunk-comparison-32-v1-submission/jobs.tsv
cat outputs/numina-chunk-comparison-32-v1/report-0-status.json
cat outputs/numina-chunk-comparison-32-v1/report.txt
```

Outputs: `summary.json` (complete protocol, per-seed results, intervals, costs,
verifier statuses and truncation), `results.csv`, and the readable `report.txt`.
All base/final answers remain in each run directory for review.

Optional plots, with no fitting or new evaluation:

```bash
bash scripts/container_exec.sh "$PWD/.venv-critic/bin/python" \
  scripts/plot_chunk_comparison.py --study outputs/numina-chunk-comparison-32-v1
```

Produces `figures/chunk-comparison.png` and `.pdf`: per-seed accuracy, adjusted
paired differences and mean accuracy versus online GPU-hours. Plotting needs
matplotlib; it is separate from report success.

Prepared studies seal code/data/filter hashes. Do not pull changes while they
are active. Failed/partial actors are preserved and cannot silently resume;
submission markers stop duplicate chains. If sbatch times out, inspect Slurm and
sidecar JSON before retrying—timeout does not prove submission failed.

Preserve earlier studies, including the failed v1 and any v2/v3 prepared with the
whole-answer or 64-token fallback. Prepare a fresh `outputs/numina-chunk-comparison-32-v1` with
the commands above. To choose a different fallback size (e.g. 64 or 128), change
`--chunk-fallback-tokens` in the **smoke** command only; training uses that sealed
setting. The code hash intentionally prevents relabeling old smoke results as
checks of the new fallback policy; rerun all four matched smokes.

## Inspect existing runs without training

```bash
bash scripts/container_exec.sh "$PWD/.venv/bin/python" \
  tools/diagnostics/peek_chunk_study.py \
  --study outputs/numina-chunk-comparison-v1 \
  --study outputs/chunked-pipeline-smoke-v2 \
  --fallback-tokens 32 --max-answers 256 --examples 3
```

Use only paths that exist. This CPU-only tool shows completed updates, recorded
fallback counts and any completed final accuracy, then reapplies MATH segmentation
to a bounded sample from each case's latest saved **training** rollout. It prints
normal chunk-length quantiles and example chunks; it never reads test answers
for size selection. `--status-only` skips tokenizer loading and chunk inspection.
It works with older v3 wrappers by inspecting the pinned upstream splitter directly.

Most VERL dumps contain decoded text, not sampled token IDs: their re-tokenized
lengths are approximate and omit unrecorded EOS. Exact recorded online fallback
counts, when present, are shown separately. These are convenience samples, not
a representative corpus estimate or evidence that one size improves PPO. The
32-token fallback was chosen from the observed training chunk sizes, not test
accuracy; it still needs the real GPU integration check and controlled comparison.
The diagnostic does not change defaults, saved studies, or running actors.

The later unmapped-ByteLevel-ID fix extends the fallback to IDs with no token
string. It changes runtime hashes: do not pull it into an active study or rewrite
its manifest. Preserve successful runs and failed artifacts; any recovery must
record the changed code and run the failed seed afresh. Actor checkpoints are
saved at the final update in this recipe, so saved A,b alone cannot resume a
failed actor. Existing legacy fallback counters include all recorded reasons.

Do not pull runtime changes into an active sealed study. To inspect one safely,
copy only `tools/diagnostics/peek_chunk_study.py`; that tool is outside the sealed
runtime paths. Otherwise synchronize normally before preparing the next study.
