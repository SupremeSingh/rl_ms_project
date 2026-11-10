# Repeat the original six-actor value audit with more seeds and questions

**Question: which estimator tracks its own trained actor's value most accurately
across several actor seeds and a larger reference-evaluation sample?**

This repeats the **original six-actor study**, not the subsequent fixed-policy
regression benchmark. Each method updates its own actor. Conventional PPO uses
the existing GAE-return targets, value clipping/loss and preselected `higher_lr`
profile. No Monte Carlo target replaces its training procedure. The four Ridge
arms retain their existing cumulative raw-sum statistics, one head, epsilon .01
and frozen features. GAE lambda remains .95, gamma 1; the actor uses fresh data.

## Locked scope

| Setting | Original completed pilot | New main repeat |
|---|---:|---:|
| Actor seeds | 1 (1001) | 3 fresh seeds (1101, 1102, 1103) |
| Methods | 6 | Same 6 |
| Updates × answers/run | 120 × 64 | 120 × 64 |
| MC evaluation questions/seed | 10 | 50 |
| Sampled prefixes/question | Up to 4 | Up to 4 |
| MC continuations/state/actor | 64 | 128 |
| Maximum reference completions | 15,360 | 460,800 |

The methods are SLICE/token, RMS-normalized two-thirds/token, SLICE/chunk,
RMS-normalized two-thirds/chunk, conventional PPO, and GRPO. GRPO's plotted
quantity is its independent four-answer question baseline, not a learned critic.
Those baseline samples and prefix-bank generation are additional to the table.

Training delegates directly to the original `value_audit.command` and training
functions. There are **18 fresh actor runs**. Model, compressor and Numina data
are reused; the new training actors and heads are not copied from prior results.
Each run saves actual actor weights, and PPO also saves its transformer critic.

Within a seed, one mixture-generated bank supplies exactly the same states for
all six estimators. The same 50 question IDs are used across the three seeds;
sampled trajectories/prefixes can differ across seeds. Sampling streams are
seed-specific. Selection does not depend on correctness or prediction error.

Each panel's frozen final actor generates its **own** reference continuations.
Thus this measures tracking of each method's learned policy, not architecture
quality under one common policy. All methods preserve temperature 1, top-p 1,
top-k −1, original action IDs, the 2,048-token total response cap, and the existing
verifier rule. Prefix tokens reduce the remaining continuation budget. MC labels
are never fed into training or checkpoint selection. Head/actor timing lag is
recorded as in the original audit.

128 samples estimate a value; they do not make it exact ground truth. The maximum
reference standard error is about **.0442**, down from .0625 with 64 samples.
The larger dataset remains inspected Numina data, not an independent benchmark.

## Statistics and plots

- Per-seed six-panel scatter, raw MAE/RMSE, reference uncertainty and status
  counts are retained under `seed1101/full/`, `seed1102/full/`, `seed1103/full/`.
- The combined plot uses different markers for actor seeds, shared axes, and
  the ideal y=x line. It reports mean per-seed MAE and seed standard deviation.
- Mean per-seed RMSE, actor **validation** accuracy and measured training GPUh
  are reported separately. Actor accuracy is not a value-estimation metric.
- Paired question-level MAE differences resample actor seeds **and** question
  IDs, preserving method pairing rather than treating every prefix as independent.
  Two predeclared primary contrasts compare SLICE/token and RMS-layer/token
  against PPO, using Bonferroni nominal adjustment. Chunk comparisons and GRPO
  versus PPO are exploratory, unadjusted secondary contrasts.
- These approximate intervals are conditional on sampled MC references. Three
  seeds still limit precision; no automatic universal winner is declared. The
  per-seed summaries also retain reference-uncertainty sensitivity diagnostics.
- The new aggregate progress plot uses **first/second/third/fourth sampled prefix**
  and the same complete-case questions across every seed and position. Coverage
  and exclusions are recorded. It does not label those positions as token
  fractions. Primary overall metrics retain all questions, including shorter
  answers. The original per-seed token-bin plots remain secondary and can have
  unequal cross-bin coverage.

The GRPO baseline obtains four additional verified answers for each test question
and actor seed. Learned critics do not get test-question outcome samples before
predicting. Report this query-budget difference when interpreting rankings.

## Commands

Push the new code and docs to `v3`. After any jobs using the checkout finish:

```bash
cd /usr/xtmp/ms785/rl_ms_project
git switch v3
git pull --rebase origin v3
source scripts/cluster_env.sh

python3 scripts/submit_value_repeat.py --mode smoke \
  --source outputs/numina-no-folds-v1 \
  --out outputs/numina-value-repeat-v1 --concurrency 6
```

No new installation is required with the existing editable training environment.
Preparation pins all three child recipes, but smoke trains only six two-update
actors (original smoke seed 19), with four questions and eight MC samples/state.
It tests the original training path, checkpoint restoration, shared banks,
per-actor references and plots. It is not a performance ranking.

```bash
cat outputs/numina-value-repeat-v1-submission/jobs.tsv
cat outputs/numina-value-repeat-v1/smoke.json
cat outputs/numina-value-repeat-v1/seed1101/smoke/report.txt
```

After `smoke.json` reports `passed: true`, submit the main run separately:

```bash
python3 scripts/submit_value_repeat.py --mode main \
  --out outputs/numina-value-repeat-v1 --concurrency 6
```

All stages use `compsci-gpu`, including utility stages, to avoid the separate CPU
partition. Training tasks request **two A5000s**; all other tasks request one.
Concurrency 6 therefore means up to **12 GPUs during actor training**, then six
during reference generation. The main MC stage has 72 shards. This larger run
can take substantially longer than the pilot; queue time and generation length
vary. The 460,800-completion reference budget is explicitly submitted only in
main mode, not automatically after smoke. Prior feature training, smoke jobs,
failed attempts and utility allocations are additional costs.

After completion:

```bash
cat outputs/numina-value-repeat-v1/main/verification.json
cat outputs/numina-value-repeat-v1/main/report.txt
```

Combined plots:

```text
outputs/numina-value-repeat-v1/main/value-scatter.png
outputs/numina-value-repeat-v1/main/value-scatter.pdf
outputs/numina-value-repeat-v1/main/error-by-prefix-order.png
```

Keep code, environments, source data and model checkpoints unchanged while jobs
run. Submission failures stop the chain and retain intent/job-ID records under
the sibling `-submission/` folder; inspect Slurm before retrying. MC stages reuse
atomic sample-block caches when explicitly rerun. Training does not silently
resume partial actors. Each task maps to a seed/method/shard through
`value_repeat.coordinates`; do not change seeds or replace runs based on outcomes.
