# Stage 1: validate the verifier and dataset

This stage performs inference and a human audit; it does not update weights.
It uses the initial downloaded checkpoint, never the smoke-test RL checkpoint.

The existing starter uses Qwen2.5-Math-1.5B. The original plan named
Qwen2.5-Math-1.5B (base). This is an explicit deviation introduced in Stage 0;
this audit measures the Instruct checkpoint recorded in configs/assets.json.
Choose one common initial checkpoint for all later methods. If restoring the
base model or doing format-only SFT, reprepare/re-audit the corresponding inputs
and use the resulting common checkpoint for every method.

## Additions

| File | Purpose |
|---|---|
| scripts/generate_stage1.py | Select 100 eligible questions and sample twice each with vLLM. |
| scripts/review_stage1.py | Show each response without its automatic score; collect human labels. |
| src/math_rl/audit.py | Calculate metrics and gates; reject incomplete or mismatched audits. |
| scripts/report_stage1.py | Verify response hashes and write report.json. |
| scripts/submit_stage1.sh | Submit generation using the existing Slurm/Apptainer environment. |
| tests/test_audit.py | Check completion gates, disagreement thresholds, and paired sampling. |

Uses existing prepared data/model and environment. No new dependencies.
100 prompts come from the deterministic shuffled training subset, excluding
prompts longer than 512 tokens with the same tokenizer template. Two responses
per prompt, temperature=1, top_p=1, top_k=-1, max_tokens=512, BF16, seed=42.
The official test set is not used. This small selected audit is not a benchmark
accuracy estimate. Counts and filtering are saved in metadata.json.

## Commands

From math-rl on the cluster host:

```bash
export MATH_RL_IMAGE="$PWD/../verl-v041.sif"
mkdir -p logs
sbatch scripts/submit_stage1.sh
```

Wait for the generation job to finish successfully. It writes
outputs/stage1/responses.jsonl and metadata.json. An existing output directory is
never overwritten; pass --out outputs/stage1-v2 for another generation run.

Then review interactively (no GPU needed):

```bash
apptainer exec --bind "$PWD:$PWD" "$MATH_RL_IMAGE" \
  "$PWD/.venv/bin/python" scripts/review_stage1.py outputs/stage1
```

For each response enter three binary labels: reward format answer_correct.
- reward: would a correct implementation of the declared strict scoring contract
  award 1? This must equal format * answer_correct.
- format: is the entire final non-empty line Answer: followed by a decimal number
  with optional sign and correctly grouped commas? No units, fractions, trailing
  prose, boxed notation, exponent notation, or terminal period.
- answer_correct: does the response have an unambiguous correct final numeric
  answer, irrespective of formatting? This checks final-answer correctness, not
  every reasoning step. Unresolved contradictions do not count as correct.

Examples:
- Correct Answer: 42 -> 1 1 1.
- Wrong Answer: 41 -> 0 1 0.
- Correct final boxed answer 42 without Answer: line -> 0 0 1.

Read the response and reference independently of the automatic scorer. If the
reference itself seems wrong or uncertain, pause and resolve it before labeling.
Use q to pause; rerun the same command to resume. Each completed response is
saved immediately in labels.jsonl. Human scores are never prefilled.

Produce a report, even before labeling is complete:

```bash
apptainer exec --bind "$PWD:$PWD" "$MATH_RL_IMAGE" \
  "$PWD/.venv/bin/python" scripts/report_stage1.py outputs/stage1
```

## Gates

- All generated responses labeled, at least 200 (default 200).
- At least 99% human-versus-verifier reward agreement (at most 2 disagreements
  out of 200). This is observed agreement, not a population-confidence guarantee.
- Initial reward accuracy roughly 5%-70%; also inspect format-independent human
  answer accuracy to distinguish formatting trouble from reasoning difficulty.
- At least 10% of prompt pairs have differing automatic rewards.
- Human format rate >=95% and truncation <=5%: provisional engineering thresholds,
  not numeric requirements stated in the original plan. CLI overrides are
  --min-format and --max-truncation; decide thresholds before examining results.

An incomplete audit has stage1_pass=false. Finish_reason=length counts as
truncation. Mixed pairs mean one success and one failure for the same prompt,
not merely different wording. Agreement and format are provisional until all
labels are complete. Raw responses are hash-checked against generation metadata.

## Failure actions

If verifier agreement fails, inspect disagreement_ids and fix the verifier. Do
not discard difficult or malformed examples. Preserve the original audit and
re-run the corrected pipeline before proceeding.
If format rate is poor, improve the prompt and re-audit. If format-only SFT is
needed, its result becomes the common initial checkpoint for all methods.
If almost every reward is zero or one, change difficulty and re-audit. Do not
hide this by cherry-picking successful questions.
If truncation is high, examine responses and choose a common generation budget;
re-audit and apply that choice to all methods.

## Validation status

17 CPU tests pass (reward + audit). GPU generation, human review, and the Stage 1
gates have not been run here. Synthetic test fixtures are only tests and are not
reported as experimental evidence. Stage 0 success is also not inferred.
