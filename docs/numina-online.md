# Numina PPO comparison: frozen learned and single-layer critics

This study tests whether the offline learned bottleneck helps **actor learning**.
It is a new online comparison, not a claim that the offline predictor is already
an improved PPO critic.

## Locked protocol

Nine arms: conventional transformer-critic PPO, plus Ridge and LSTD(0.99) for each
of plain-768, sparse-768, frozen two-thirds hidden state and frozen final hidden
state. All use the same initial actor, prompts, verifier, 512-token prompt limit,
2,048-token response cap, gamma=1, actor GAE lambda=.95 and actor optimizer.
The task ends at EOS or the response cap. Four answers per prompt and 16 prompts
per update give 64 answers. The standard preset uses 60 updates and actor seeds 201–205; the large
preset below extends this to 200 updates with the same seeds.
This matches rollout counts, not realized generated tokens or compute.

Each linear head starts from zero statistics. It uses raw cumulative sums,
epsilon=.01, persistent question cross-fitting, no buffer and no encoder refresh.
The encoder is frozen. Both bottleneck and raw single-layer arms use FP32 eager
feature extraction with TF32 disabled; the PPO actor remains BF16. Final-layer
features include learned final RMSNorm; two-thirds features are the block output.
Bottlenecks reproduce the offline normalized geometric-layer concatenation,
centering, scaling, row-normalized projection and ReLU. Only compressed states
are transferred back for critic fitting. Final values are linear and unclipped.

Select **one checkpoint per bottleneck kind** using minimum saved validation Ridge
Brier, breaking ties by checkpoint key. Freeze that checkpoint across all five
actor seeds and both fitting methods. Do not select by inspected test results.
This measures actor-seed variability conditional on these two selected filters,
not variability over representation-training datasets or seeds.

## Avoid reward leakage

The supervised bottlenecks have already seen the original Numina training labels
and validation labels influenced checkpoint selection. Consequently, NONE of the
source study's question families can enter this online study, including its
calibration, screen and test families. Preparation draws 2,000 training, 300
validation and 1,000 test questions from remaining eligible NuminaMath-1.5 cn_k12
families at the source's pinned dataset revision. It also excludes GSM8K and MATH
families using the source revisions and the existing conservative reference filter.

Question families are deduplicated by normalized text with number substitution.
This is not semantic or pretraining decontamination. Preparation fails if too few
eligible questions remain; it never silently reuses source questions or relaxes
filters. The online split is sealed before learning. No answer-success filtering
is used to choose questions.

Online attempts all count: verifier parse failures/timeouts receive the existing
zero-reward policy, with statuses preserved. Infrastructure errors fail the run.
That differs from the offline retained-answer analysis. All final responses and
extractions are saved for audit; claims still require verifier review.

## Baseline and inference safeguards

A two-update smoke run executes all nine arms on validation evaluation only.
Conventional PPO then gets a separate 30-update calibration at seed 17 using the
three existing profiles: normalized, higher_lr and more_fitting. Selection uses
validation accuracy subject to numerical critic health and validation
non-regression. If none passes, the comparison stops rather than silently using
an unsuitable baseline. This is limited calibration, not an optimal-PPO claim.

The main stage starts every run fresh, with the preset update budget, a final checkpoint and no
test-based checkpoint selection. Validation frequency and evaluation decoding are
shared. PPO critic health, final test answers, truncation, verifier statuses and
linear fit diagnostics are retained. Hash checks lock data, model, filters and code.

## Commands

Sync updated code first. Use the existing pinned PPO environment; if it has not
been installed, run `bash scripts/container_exec.sh bash scripts/setup_ppo.sh`
once before submission, not concurrently inside array jobs.

From the repository root:

```bash
source scripts/cluster_env.sh
bash scripts/submit_numina_online.sh \
  outputs/jlens-large-numina-v2 outputs/numina-sparse-768-v1 \
  --out outputs/numina-online-768-v1
```

This submits CPU preparation followed by a two-GPU integration smoke job. Inspect:

```bash
cat outputs/numina-online-768-v1/prepare-0-status.json
cat outputs/numina-online-768-v1/smoke.json
```

After smoke passes:

```bash
bash scripts/submit_numina_online.sh \
  outputs/jlens-large-numina-v2 outputs/numina-sparse-768-v1 \
  --out outputs/numina-online-768-v1 --full
```

This submits calibration, a 45-run array with at most two concurrent two-GPU jobs,
and a report job. Allow substantial cluster time: earlier runs suggest the study
could consume hundreds of GPU-hours; FP32 feature extraction and new question
lengths make a precise estimate premature. Each training job has a 12-hour ceiling.
Queue time is additional. Smoke logs provide the first hardware-specific check.

```bash
squeue -u "$USER"
cat outputs/numina-online-768-v1/jobs.tsv
cat outputs/numina-online-768-v1/selection.json
cat outputs/numina-online-768-v1/report.txt
```

Completed whole runs are reused when resubmitting `--full`; partial actor runs do
not resume. After diagnosing a failure, preserve/rename its partial case directory
and log before retrying. Do not run concurrent writers for the same study. If
training code/config changes after preparation, use a new output directory.

## Interpretation and cost

Four prespecified primary contrasts compare each learned bottleneck against the
two-thirds baseline under the SAME fit method. The report includes paired
seed/question bootstrap differences, ordinary descriptive 95% intervals and
98.75% intervals (Bonferroni nominal coverage for four primary comparisons).
These approximate intervals with five seeds are not an automatic superiority gate.
All other pairwise comparisons, including sparse versus plain, are exploratory.

Report mean accuracy, seed SD and individual results, plus training GPU-hours and
accounted GPU-hours including initialization and base/final evaluation. Training
includes encoder/filter computation, fitting, validation and checkpoints. Record
PPO calibration cost separately. Saved offline bottleneck trial times are also
reported separately; their generation and feature-extraction cost is NOT included
in online costs. Thus this tests pretrained-feature reuse, not equal total prior
supervision or equal end-to-end development cost. Compare accuracy at the matched
rollout budget and total cost honestly; smaller matrices alone prove no speedup.

Historical-policy critic statistics remain uncorrected off-policy data, even with
a fixed encoder. Both Ridge and LSTD use the same persistence and fold rules.

## Larger-data preset

Use `--large` at preparation in a NEW output directory. The saved manifest locks
this choice; `--full` automatically recovers it and rejects a conflicting preset.

| Budget | Standard | Large |
|---|---:|---:|
| PPO training questions | 2,000 | 3,000 |
| Validation questions | 300 | 500 |
| Test questions | 1,000 | 2,000 |
| Updates per final run | 60 | 200 |
| Answers per final run | 3,840 | 12,800 |
| PPO calibration updates per profile | 30 | 60 |
| Actor seeds | 5 | 5 |

The larger horizon samples 3,200 prompt instances per run (16 per update), enough
to traverse the 3,000-question training pool approximately once. Four continuations
per prompt are correlated samples, not four independent questions. This is still
a bounded-horizon comparison rather than training every method to convergence.
The feature map, matrix accumulation, folds, reward and actor settings are unchanged.

```bash
bash scripts/submit_numina_online.sh \
  outputs/jlens-large-numina-v2 outputs/numina-sparse-768-v1 \
  --out outputs/numina-online-768-large-v1 --large

# After preparation and smoke complete:
cat outputs/numina-online-768-large-v1/smoke.json
bash scripts/submit_numina_online.sh \
  outputs/jlens-large-numina-v2 outputs/numina-sparse-768-v1 \
  --out outputs/numina-online-768-large-v1 --full
```

Large training and calibration jobs request 24 hours each; the 45-run array remains
limited to two simultaneous two-GPU runs. Training work increases about 3.3x and
evaluation grows too; total study cost may exceed 1,000 GPU-hours. These are planning
estimates, not measured runtimes for this protocol. If too few clean question
families remain, preparation fails rather than borrowing bottleneck-training data.

Use the large preset instead of the standard one when choosing this expanded
budget. They are alternative allocations from the same eligible pool, not
independent replications; their split boundaries can overlap. If standard test
results have already been inspected, do not describe the large study as an
untouched confirmation. Never alter files while existing jobs are running.
