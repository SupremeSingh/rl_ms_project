# Transfer clarification: matched Ridge controls, regularization and learning curves

Offline only. Compare RMS-normalized Qwen block 19, frozen Numina-trained sparse
SLICE-768 with a new target Ridge head, and target-trained sparse SLICE-768 with
Ridge. Targets: GSM8K and the **numeric eligible subset of MATH-500**. No ensembles,
PPO, LSTD, output LayerNorm or online accumulation in this study.

## Questions and controls

- Does a frozen source representation work on either target after head adaptation?
- Does Ridge regularization change the ranking? All arms use epsilon
  `[.01, 1, 10, 100, 1000, 10000]`, selected on equal-question validation raw Brier.
  If the largest wins, extend automatically by factors of ten up to 1e10. If the
  ceiling still wins, stop before test. Grid and validation scores are saved.
- How much target head-fitting data is needed? Use nested 100/500/2000-question
  subsets, identical for all arms and seeds; all answers/prefixes from a question
  stay together. The target SLICE is trained on the fixed matched representation
  budget even for a small Ridge subset. Thus these are **head-adaptation curves**,
  not equal-total-target-label-budget comparisons against target-trained SLICE.

Five compressor/shuffle seeds: 42–46. Ridge is deterministic for fixed data and
features; the RMS arm is not five independent trained representations. Generation
is shared across seeds. Source and target compressors use the same architecture,
training recipe and number of retained training families: min(2000, available
source families after new exclusions). Representation validation counts are also
matched. Actual counts are saved in each locked fit. No claim that these are
optimal compressor recipes.

## Data and reuse

Supply the completed `slice-transfer-numina-gsm8k-v2` study. Its checked Numina
and GSM8K cached pre-action features/labels are reused. Old compressor weights
are NOT reused: source compressors are retrained after removing **all MATH native
train and MATH-500 question-template families** from source train/validation.
The original source already excluded native GSM8K numeric families. Remaining
source/target overlap removal is lexical number-template grouping, not semantic
or pretraining decontamination.

MATH training/validation comes exclusively from the seven train subsets of
`EleutherAI/hendrycks_math`, at a pinned common revision. Test is
`HuggingFaceH4/MATH-500`, pinned separately. All MATH-500 families are excluded from
MATH train/validation; selected GSM8K families are also excluded from MATH data.
No MATH-500 labels fit compressors, heads or epsilon. Unsupported nonnumeric,
multiple-answer, diagram and overlength questions are excluded and counted;
**do not report the resulting scores as full 500-question benchmark scores**.
Historical MATH use and inspected GSM8K make this exploratory, not an untouched
benchmark. Numeric eligibility is not proof of reference correctness.

Full MATH preset: 2000 train, 500 validation, all eligible test questions; four
answers per question, same frozen Qwen, prompt, 1024 response-token cap and eight
sampled pre-action prefixes as the source. Only MATH needs new generation (about
10,000 training/validation answers plus eligible test answers). If eligibility
cannot supply the requested counts, preparation fails rather than silently
changing the budget. Retained-prefix evaluations condition on verifiable answers;
inspect shard statuses and coverage, including label balance.

## Fits and metrics

Per-block RMS normalization is identical for SLICE inputs and the two-thirds
control. Every Ridge arm receives training-only centering and one isotropic
scalar, an intercept, and the same epsilon search. Systems use raw sums, not
means; epsilon is added to all coordinates including the intercept. Heads are
solved from one float64 symmetric eigendecomposition per arm/subset, equivalent
to regularized solves. Epsilon per prefix and condition number are recorded.

Source and target SLICEs use the existing BCE + .01 mean-activation recipe,
30 epochs, validation-selected checkpoint, fixed Qwen, five geometric blocks and
768 outputs. No head/compressor tuning on the test set. Norm/weights remain fixed
while fitting each Ridge grid. The study is sampled-prefix regression, distinct
from all-token cumulative critic fitting inside PPO.

Primary: equal-question raw Brier. Also equal-question clipped Brier, prefix
threshold accuracy, ten-bin calibration error on clipped predictions, system
conditioning and a constant predictor fitted on the SAME training subset.
Report every fixed epsilon as a diagnostic and the separately validation-selected
head. Test diagnostics must not become a new selection rule. Twelve planned
paired seed/question-bootstrap intervals (two contrasts × three budgets × two
targets) are exploratory and unadjusted. No automatic superiority gate.

The full study stops at validation. **Inspect validation-report.txt first**, then
explicitly submit test evaluation. Evaluation verifies model/cache/source hashes
and loads test data only after all five seeds are locked. Predictions are saved
so further diagnostic comparisons need no new generation. Data/source code
must remain fixed after preparation. Generation resumes identical completed
answer caches; unfinished fits restart deterministically per seed, and locked
fits reject overwriting.

## Commands

After committing/pushing, on Duke:

```bash
cd /usr/xtmp/ms785/rl_ms_project
git pull --ff-only origin v2
source scripts/cluster_env.sh
python3 tools/transfer/submit_clarify.py --mode smoke \
  --source outputs/slice-transfer-numina-gsm8k-v2 \
  --out outputs/transfer-clarify-smoke-v1 --concurrency 4
```

Smoke uses 40 MATH train / 20 validation / 20 eligible test questions, 20/40 head
budgets and two epochs, reusing source generation settings (including 1024 tokens).
It submits CPU prepare → four GPU MATH shards → CPU pack → five GPU fit tasks
(up to four concurrently) → CPU validation → CPU evaluation. No manual intervention
between smoke stages. This checks real Qwen, verifier, dataset ingestion and fitting;
local tests do not establish GPU success.

```bash
cat outputs/transfer-clarify-smoke-v1-submission/jobs.tsv
cat outputs/transfer-clarify-smoke-v1/status/evaluate-0.json
cat outputs/transfer-clarify-smoke-v1/report.txt
```

Once successful:

```bash
python3 tools/transfer/submit_clarify.py --mode full \
  --source outputs/slice-transfer-numina-gsm8k-v2 \
  --smoke outputs/transfer-clarify-smoke-v1 \
  --out outputs/transfer-clarify-v1 --concurrency 4
```

After all validation jobs finish:

```bash
cat outputs/transfer-clarify-v1/validation-report.txt
python3 tools/transfer/submit_clarify.py --mode evaluate \
  --out outputs/transfer-clarify-v1
```

Final output: `outputs/transfer-clarify-v1/report.txt` and `summary.json`. Submission
IDs/logs are in `<OUT>-submission/`. A timeout stops submission without blind
retries; inspect squeue/sacct before attempting recovery. Non-array log suffixes
may be `4294967294`, so use `logs/prepare-*.out`. Cache model paths are preserved
without resolving login-host symlinks; grading uses the source's isolated verifier.
No dependency reinstall or changes to the PPO pipeline are required.
