# Final actor test evaluation and controlled update-cost benchmark

**Question: which saved RL actor performs best on the shared test split, and
which implementation processes the same training workload fastest and with
least allocated GPU memory on fixed hardware?**

These are separate questions and outputs. Nothing retrains the existing 18
actors or modifies their source manifests. No new MC reference audit is needed.

## Test evaluation

Source: `outputs/numina-value-repeat-v1`. Three seeds 1101–1103, six methods,
final actor checkpoint 120. Each is evaluated on the same frozen 2,000-question
test split with the original completion prompt, greedy n=1, total response cap
2048, original tokenizer and verifier. A fresh base-model test reference is also
evaluated: **38,000 answers including 2,000 base-model answers**, plus smoke.
Prompt lengths over 512 fail rather than silently excluding questions.

Preparation verifies source manifests/data and records complete actor weight
identities. A mandatory eight-question smoke restores all 19 models, generates
answers, and verifies them. Smoke is an integration gate, never a ranking.
Each main task checks its actor weight identity again and writes original IDs,
responses/token IDs, scores, verifier statuses, token lengths and truncation.
Output identities bind the new manifest; no old source-runtime validation is
re-run against changed code. The new runtime itself is sealed and checked.

The report includes every seed accuracy, mean ± sample seed SD, base gain,
response lengths, truncation and evaluation GPUh. The plot shows all three
points per method with mean ± SD, **not learning curves or confidence bands**.
Two locked primary contrasts: chunk SLICE minus token SLICE, and chunk SLICE
minus PPO. Approximate paired seed/question bootstrap uses 10,000 draws and
Bonferroni nominal adjustment for those two contrasts. Three seeds limit
precision; reused inspected test questions are not a fresh benchmark.

Generation and verification timing includes no actor learning; initialization
is separately recorded. GPUh means measured single-GPU elapsed time, not
utilization or a complete allocation ledger. Each task has its own GPU; timing
between independent evaluation tasks is descriptive, not a controlled training
speed comparison.

## Fixed-workload computational benchmark

Preparation generates **64 original-token answers from 16 training questions**
with the common base actor. One fixed workload is reused for every method,
round, and statistics regime; the full benchmark reuses the smoke workload.
No test questions or model outcomes select the workload. Replay preserves the
original token IDs, four answers/question and original rewards/verifier path.

The benchmark launches actual pinned VERL training subprocesses, selecting the
same conventional PPO/GRPO or cumulative Ridge update paths. Its explicit
`benchmark_runtime` substitutes saved actions for rollout generation, disables
rollout log probabilities (old action log probabilities are recomputed), and
synchronizes worker GPU phases. Scientific validation/checkpoint saving is
disabled. No reported benchmark accuracy is meaningful. Source actors are never
overwritten; temporary actors are fresh base copies, not the 18 test actors.

- One Slurm allocation requests **two A5000 GPUs** for the entire benchmark.
  Every subprocess runs sequentially on that allocation; node/device identities
  and assigned devices are recorded. GPU identities are checked between trials.
- Three randomized complete rounds, fixed schedule seed 41017. Every round
  contains six methods × two statistics regimes. **36 trials**, each discards
  the first two warm-up updates and measures the next three.
- The initial regime starts empty Ridge matrices. The accumulated regime starts
  from the corresponding seed-1101 saved update-120 Ridge statistics. All actors
  and optimizers still start fresh; PPO/GRPO have no restored matrices. This
  covers matrix history, **not the entire late-training optimizer/policy state**.
- GPU clocks, utilization, power and used memory are sampled with `nvidia-smi`
  every two seconds; telemetry and every original console log are retained.
- Report round-level mean step times, median round mean and between-round SD;
  plot time and logged peak allocated GPU memory. Allocated memory excludes
  some non-PyTorch reservations; raw device-memory telemetry is supplementary.
- No generation timing is included because actions are replayed. Model loading
  and subprocess elapsed time are retained separately. Original diagnostic PPO
  forwards, offloading, reward computation, feature transport, fitting and
  actual actor updates remain part of the measured implementation.

This measures computational overhead on one small fixed workload, not optimal
throughput, complete RL training cost, convergence, deployment cost or cheapest
time to a target accuracy. All trials are retained. Any failed trial stops the
chain; no automatic outlier removal or retries. The old slow-node measurements
remain unchanged. Both previously troublesome fitz-15 and fitz-21 are excluded
**before submission** for all methods, not selectively after outcomes.

## Run on the cluster

For a launcher-only failure after workload preparation, preserve the failed
directory and start a new benchmark smoke with
`--reuse-workload outputs/numina-cost-smoke-v2`. This checks the prior source,
feature manifest, data and workload hashes, then copies only generated workload
data into the new smoke. It never reuses trial results or treats a failed smoke
as passed. New code gets a new runtime manifest. Ray trial temp directories use
short `/tmp/b-*` paths rather than nesting under Slurm's long `TMPDIR`, avoiding
the 107-byte Linux Unix-socket path limit.

Benchmark timings are written directly by the trainer's metric logger to each
trial's `benchmark-steps.jsonl`, bound to the workload hash. Console output is
retained for diagnostics but is not the timing source: Ray can omit a final
console metric line even when training succeeds. The collector requires every
step in order, rejects duplicates and checks the completed update/answer count.
No missing timing is imputed from overall elapsed time.

Push the code to v3 yourself; do not update code/environments during these jobs.
Use existing editable environments; no dependency changes are required.

```bash
cd /usr/xtmp/ms785/rl_ms_project
git switch v3
git pull --rebase origin v3
source scripts/cluster_env.sh

python3 scripts/submit_test_followup.py --mode smoke \
  --source outputs/numina-value-repeat-v1 \
  --out outputs/numina-final-test-v1 --concurrency 6

python3 scripts/submit_test_followup.py --mode benchmark-smoke \
  --source outputs/numina-value-repeat-v1 \
  --out outputs/numina-cost-smoke-v1
```

Wait for both gates:

```bash
cat outputs/numina-final-test-v1/smoke.json
cat outputs/numina-cost-smoke-v1/verification.json
cat outputs/numina-cost-smoke-v1/report.txt
```

After **both** pass (the benchmark smoke tests six complete fixed-replay update
paths, not speed rankings), submit:

```bash
python3 scripts/submit_test_followup.py --mode main \
  --out outputs/numina-final-test-v1 --concurrency 6

python3 scripts/submit_test_followup.py --mode benchmark \
  --source outputs/numina-value-repeat-v1 \
  --smoke outputs/numina-cost-smoke-v1 \
  --out outputs/numina-controlled-cost-v1
```

Test evaluation uses six concurrent one-GPU tasks. The benchmark reserves two
GPUs and runs serially inside that allocation; do not split it into parallel
array tasks. Test stages request up to four hours/task; full benchmark allocation
up to 12 hours, with actual time likely several hours. These are limits, not
promised completion times. Benchmark preflight should be reviewed before full
submission; GPU execution cannot be validated on a CPU-only local machine.

Read results:

```bash
cat outputs/numina-final-test-v1-submission/jobs.tsv
cat outputs/numina-controlled-cost-v1-submission/jobs.tsv
cat outputs/numina-final-test-v1/report.txt
cat outputs/numina-final-test-v1/verification.json
cat outputs/numina-controlled-cost-v1/report.txt
cat outputs/numina-controlled-cost-v1/verification.json
```

Plots are `test-accuracy.png`/`.pdf` in the test directory and
`controlled-cost.png` in the benchmark directory. Keep the original results
separate from these outputs. Submission markers prevent blind duplicate
submission; inspect Slurm before retrying any uncertain failure.
