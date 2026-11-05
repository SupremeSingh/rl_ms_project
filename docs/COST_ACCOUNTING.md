# Cost accounting from completed transfer runs

Read-only retrospective audit: no model load, generation, feature extraction,
fitting, or new test evaluation. Existing results stay untouched. Main audit
uses host Python standard library so it can call Slurm's `sacct`; optional plots
use scientific Python inside the container. No GPU allocation is requested.

## What we can measure

- Actual Slurm **allocated GPU-hours and CPU-hours**, summed over allocation
  records (never batch/extern steps). GPU counts use aggregate `gres/gpu`, not
  aggregate plus typed GPU counts. Confirmed array tasks are counted separately.
- Recorded Python stage times, reported separately from Slurm times. These omit
  some startup/teardown and must not be added on top of allocation hours.
- Actual attempted answers, response tokens (including EOS), prompt-token volume,
  caps, verifier statuses, retained answers and sampled prefix counts by domain
  and split. Trajectory signatures and question/checkpoint identities are checked.
- Actual token/label volumes for the same nested retained-question subsets used
  in the learning curves, paired with the saved test Brier. Target data collection
  is shared by all arms and counted once in the hardware ledger.
- Algebraic head matrix storage: 769×769 versus 1537×1537, approximately one-quarter
  the dense matrix storage. The nominal cubic solve ratio is not measured speed.

The earlier completed source study is a shared prior investment; its saved
Numina/GSM8K generation is **not billed again** as new clarification generation.
Only the later MATH collection is new. Five compressor seeds are research
replications on shared labels, not five independently collected datasets.

## What was not timed separately

Generation and frozen feature extraction share a job/timer. Each fit job trains
source/target compressors and evaluates many Ridge systems. The older fit jobs
also trained pooled/wide/combined arms. We cannot split those aggregate GPU-hours
fairly by method, seed representation, or Ridge head without new profiling.
This audit does not claim that SLICE is cheaper than the layer baseline by some
retrospectively invented percentage. Frozen Qwen inference, dense bottleneck
work and source learning remain real costs.

No end-to-end label-saving ratio, deployment break-even count, inference
throughput, energy or dollar cost was measured. A point using 100 head-training
questions excludes source feature training and validation labels; target-trained
SLICE additionally uses its larger target representation-training budget. The
entire question pool was actually collected, so retrospective subset token counts
are not proof that a prospective acquisition strategy would spend only that much.

## Run

After committing/pushing these files and pulling on Duke:

```bash
cd /usr/xtmp/ms785/rl_ms_project
git pull --ff-only origin v2
source scripts/cluster_env.sh
mkdir -p logs
sbatch tools/transfer/submit_cost_audit.sh \
  outputs/transfer-clarify-v1 outputs/transfer-cost-audit-v1 \
  --extra-study outputs/slice-transfer-smoke-v1 \
  --extra-study outputs/slice-transfer-smoke-v2 \
  --extra-study outputs/slice-transfer-numina-gsm8k-v1 \
  --extra-study outputs/transfer-clarify-smoke-v1
```

The successful original source full study is discovered from the clarification
manifest. The extra paths add known smoke/failed-run ledgers; preparation failures
need not have created their output directory, provided the sibling submission
ledger exists. This is accounting for these supplied transfer studies, not the
entire project's history. Missing ledgers/accounting are flagged, never treated
as evidence of zero cost. Older requeue attempts may be absent from default sacct.
Duplicate job IDs across ledgers cause a failure rather than double counting.

Slurm and the program measure different things: allocated GPU-hours include CPU
work while a GPU is reserved and do not indicate utilization. CPU-hours include
reserved CPUs on GPU jobs. Summed task time is not wall-clock turnaround. No
pricing is assumed.

When complete:

```bash
cat outputs/transfer-cost-audit-v1/report.txt
```

Outputs: `summary.json`, `report.txt`, and `figures/quality-vs-head-tokens.{png,pdf}`;
`figures/allocation-ledger.{png,pdf}` is added when allocation accounting exists.
The figures use recorded means, not new statistical claims. All accounting rows
and missing-record flags are retained in JSON.

If sacct is unavailable, the report retains dataset usage and flags incomplete
hardware costs. A saved allocation-only `sacct -X -n -P` export with columns
`JobIDRaw,JobID,State,ElapsedRaw,AllocCPUS,AllocTRES,Start,End` can be supplied through
`--sacct-file PATH`. Do not include accounting headers. Model fitting is never
rerun. If plotting alone fails (for example missing matplotlib), the audit
report remains saved; rerun only plotting into a new figure directory:

```bash
bash scripts/container_exec.sh "$PWD/.venv-critic/bin/python" \
  tools/transfer/cost_audit.py \
  --plot-report outputs/transfer-cost-audit-v1/summary.json \
  --out outputs/transfer-cost-audit-v1/figures-retry
```
