# Why do the value-audit and earlier actor-accuracy rankings differ?

**The completed repeat gives SLICE/chunk the lowest mean final-checkpoint MAE among learned critics (0.23661). PPO and SLICE/token have almost identical mean MAE (0.24499 and 0.24496); neither the current nor the earlier comparison establishes a universal winner. Actor validation scores are not new actor test results and are omitted from the results table.**

Audit performed on 2026-11-12 against the local checkout and the user-supplied completed cluster report. No training, evaluation, configuration, or existing results were modified. Raw cluster checkpoints, resolved run configurations, and per-answer results were not available locally for numerical replay.

## What these numbers actually measure

- MAE is measured **after update 120**, not averaged over training updates. For each of 50 held-out questions, errors are averaged over four selected states; questions and then three actor seeds receive equal weight.
- The actors are frozen while 128 continuations per state estimate final verifier reward probability. Across 18 actor runs and 200 states/run, this is 460,800 reference continuations.
- Actor accuracy is a separate greedy validation evaluation. Earlier headline online studies reported accuracy on 2,000 test questions. Validation and test accuracy must not be presented as the same leaderboard.
- Every method sees the same state bank within a seed, but its own trained actor defines its reference values. This measures own-policy value tracking on a common mixture of states, not critic architectures predicting one identical value function.
- GRPO's plotted estimate is an independent four-answer prompt mean, shared across a question's prefixes. It is not a learned prefix critic and requires extra evaluation-question outcome queries.

## Results supplied by the cluster

| Method | Mean MAE | MAE seed SD | Training GPUh/run |
|---|---:|---:|---:|
| SLICE/token | 0.24496 | 0.01864 | 7.886 |
| RMS layer/token | 0.25987 | 0.01332 | 7.021 |
| SLICE/chunk | 0.23661 | 0.01603 | 8.458 |
| RMS layer/chunk | 0.24110 | 0.00658 | 8.014 |
| PPO | 0.24499 | 0.03862 | 9.765 |
| GRPO question baseline | 0.20707 | 0.03320 | 6.070 |

The primary SLICE/token minus PPO MAE contrast is -0.0000315 with adjusted interval [-0.03509, 0.03134]. This is no detected difference, not demonstrated equivalence. SLICE/chunk's smallest observed MAE is a descriptive ranking; inspect the secondary contrasts before a stronger claim.

The current report does not provide a new actor test evaluation or an actor-accuracy confidence interval. Three seeds do not support a universal actor-performance ranking.

Earlier seven-critic test means were Ridge 73.97% and PPO 72.65%, but Ridge-minus-PPO's adjusted interval was [-0.99, 3.82] percentage points. PPO was also above GRPO, both MLP variants, and low-lambda LSTD in that study. Describing PPO as consistently the worst method is not supported by that evidence. The historical five-seed six-method data likewise put PPO above GRPO and both LSTD variants (`docs/results/numina-six-method.json`).

## Source inspection

| Check | Finding |
|---|---|
| Repeat training dispatch | `value_repeat.py` delegates to `value_audit.train`; it does not import `fixed_policy` or the MC-regression transformer workflow. |
| Actual PPO objective | `runner.py` selects GAE; `main.py` creates `AuditedCriticWorker`; token-mode updates call the pinned parent critic update. The trainer's token-mode advantage hook delegates to original VERL GAE. |
| PPO profile | `higher_lr`: critic LR 5e-5, two critic epochs, equal-answer token-mean loss; actor LR 1e-6 and two epochs; gamma 1, GAE lambda .95, value clipping .5. Profile and shared training YAML match the historical seven-critic code at commit `7fa63fe`. |
| New export handling | Full HF actor/critic exports are enabled. The critic architecture metadata is explicitly set to token classification so export restores its scalar value head. This changes export handling, not the token PPO objective. |
| Restore safeguards | PPO load rejects missing, unexpected, or mismatched weights. Gates hash actual model files, data, filter, heads and statistics. Synthetic scalar-head save/load and causal-prefix tests pass. No full GPU numerical round-trip replay was performed locally. |
| Value alignment | Pinned VERL critic slices `[-response_length-1:-1]`; audit reads the last available prefix token. Both are pre-action states. Ridge uses the same causal convention, with an unread dummy future token in the evaluation feature API. |
| Ridge | One cumulative head, frozen features, epsilon .01; float64 solve and saved-system residual check. Evaluation does not refit on MC references. |
| Reference generation | Exact original prompt/prefix IDs, actor-specific checkpoint, temperature 1, and remaining budget `2048 - prefix_length`. Full prefix plus continuation is graded with the training verifier. |
| Aggregation | Per-question MAE then equal seed mean. RMSE is explicitly the mean of per-seed RMSEs. Primary bootstrap pairs seeds and question IDs, conditional on sampled references. |
| Checkpoint timing | Both final critics use data generated before the last actor update; the audit retains this lag. Ridge additionally retains earlier-policy statistics without correction. |

Pinned upstream alignment/loss source: [VERL dp_critic.py at the project commit](https://github.com/volcengine/verl/blob/8d9e350ea58c7ad4b50dd14d9dcb50577242c55f/verl/workers/critic/dp_critic.py). Training computes clipped value loss on GAE returns; the audit's MC continuations are evaluation-only.

## Real differences from earlier studies

1. Actor seeds 1101–1103 replace earlier seed cohorts. Earlier results already showed meaningful between-seed variation; the current MAE seed SD is largest for PPO.
2. Current displayed actor scores are validation scores, not earlier test scores. All 18 actors would need evaluation on the same earlier test split to compare actor accuracy on that split directly.
3. The old token/chunk study's layer control was unnormalized; this audit trains an RMS-normalized layer control. Its old weights cannot be reused after normalization.
4. The audit sets validation frequency to the final step (120) and saves full HF exports. Earlier comparisons validated more frequently and saved different checkpoint contents. Training-time measurements are therefore not interchangeable across studies. Within-study cost comparisons remain measured recipe costs, including PPO's extra diagnostic forward pass, not pure algorithmic or total-lifecycle costs.
5. A separate fixed-policy study fitted a transformer to terminal MC returns without PPO. Its MAE 0.34033 is **not** a conventional PPO result and must not be used as one.

## Why smaller MAE need not yield a better actor

MAE measures absolute probability error over selected states. PPO improves actions using temporal differences, GAE, normalized advantages, and clipped policy updates. Errors shared between consecutive states can partly cancel in temporal differences, while smaller absolute errors can still produce inaccurate advantage signs or noisy credit assignment. These are possible explanations, not mechanisms established by these results. Historical-policy averaging in cumulative Ridge, chunk weighting, and changing GAE horizons are additional differences.

## Validation and remaining checks

Executed locally: `python3 -m pytest -q tests/test_value_audit.py tests/test_value_repeat.py tests/test_ppo_baseline.py` — **26 passed, 2 skipped**. The skips require installed VERL. Upstream pinned source was inspected separately. These tests do not reproduce cluster GPU training or establish that any scientific hypothesis is true.

No confirmed result-changing implementation bug was found in the inspected paths. A complete artifact-level audit still needs the cluster's resolved `run.json` files, per-seed summaries and original outputs. In particular, compare actual saved old/new PPO profile settings and check training-critic predictions against restored HF predictions on identical saved sequences. A passed verification file establishes the implemented integrity checks; it is not a proof that all training/evaluation code is bug-free.
