# Encoder pilot investigation — 2026-11-13

The completed `encoder-offline-numina-v2` pilot reported causal / bidirectional /
adapted raw Brier 0.181287 / 0.216223 / 0.233771. These results were supplied from
the cluster, not reproduced locally. They establish a result for the specified
pipeline, not an ordering of the information content of encoder and decoder
architectures. No new training recipe has been launched or substituted.

## Theory and relevant research

Both models receive the same state: question plus available answer prefix. The
last causal token already attends to the whole prefix. Bidirectional attention
lets earlier token representations incorporate later *available* tokens; it does
not provide extra observations. Either architecture can discard task information
in a fixed-dimensional readout. There is no general information-dominance result
for the frozen feature vectors we use. A more expressive architecture is also not
a guarantee about finite-data adaptation or linear decodability after compression.

- [LLM2Vec paper, sections 2–4 and Appendix D](https://arxiv.org/html/2404.05961v2):
  naive mask removal often hurts; adaptation matters. Its published setup uses
  Wikipedia text and mean pooling, with 1,000 steps per stage. SimCSE batches are
  32 for the 1.3B model and 128 for the larger models.
  [Official implementation](https://github.com/McGill-NLP/llm2vec).
- [SimCSE paper](https://aclanthology.org/2021.emnlp-main.552/): two dropout views
  are positives, other batch examples are negatives. Accumulation of separate
  losses does not produce the negatives of one larger contrastive batch.
  [Official implementation](https://github.com/princeton-nlp/SimCSE).
- [NV-Embed, sections 5.2.2–5.2.3](https://arxiv.org/html/2405.17428v3): compares
  causal/bidirectional masks and multiple pooling methods. Bidirectional variants
  improve aggregate MTEB scores in that setup; mean pooling beats last-EOS
  pooling, and learned latent pooling is stronger still. These are embedding
  results, not prefix-value guarantees.
- [BiPRM](https://arxiv.org/html/2508.01682v1): uses reversed reasoning trajectories
  so later reasoning can assess earlier steps. This is retrospective verification,
  not an admissible substitute for prefix-only state values. Using the sampled
  continuation in our critic input would change the problem.
- [Le Critique](https://privileged-value-functions.github.io/): studies stronger
  critics using admissible extra context and adaptive group/value baselines.
  Distinguishes extra critic observations from information downstream of the
  current action. It is related value-model research, not evidence about masks.
- [Generative Actor-Critic](https://arxiv.org/html/2604.10701v1): investigates
  limits of direct scalar value prediction and reasoning before value prediction,
  including conditioning on the current actor. This supports checking target and
  readout adequacy rather than assuming semantic embeddings solve value learning.

## Findings from the current code

1. **Readout mismatch (confirmed).** SimCSE trains final-layer mean pooling;
   `encoder_features.block_features` instead reads only the last token at blocks
   `[27,26,24,20,12]`, before final norm, and RMS-normalizes each block. This is a
   valid frozen-feature experiment, but does not evaluate the trained LLM2Vec
   sequence embedding. Its effect remains unmeasured.
2. **Small negative pool (confirmed configuration).** Single GPU, batch 2,
   accumulation 4. Each SimCSE loss has one other example as a negative, not seven.
   A small contrastive loss would not prove useful value representations.
3. **Small corpus and short context (confirmed).** Cluster adaptation metadata
   records 1,805 distinct 128-token windows: 231,040 corpus token positions.
   Roughly 4,000 window presentations per stage at 500 optimizer steps and an
   effective batch of 8, about 2.2 corpus passes (MNTP repacking can alter this).
   This is very different from broad pretraining or long state-prefix training.
   Rejecting windows that fail exact retokenized length further limits coverage.
4. **SLICE training budget (confirmed upper bound).** 2,000 training answers,
   at most eight prefixes each: at most 16,000 examples. Batch 2,048, 30 epochs
   means at most 240 optimizer updates, not thousands. The 768-by-7,680 projection
   has about 5.9M trainable entries. Whether it underfits requires the curves.
5. **Selection/head mismatch (confirmed).** Compressor checkpoints minimize
   validation sigmoid Brier, but the reported prediction comes from a separately
   fitted unbounded Ridge head. Ridge uses all training prefixes; the compressor
   sees at most eight per answer. Better classifier features need not yield a
   better fixed-epsilon Ridge fit. Preserve raw-sum epsilon .01 as the primary
   recipe; investigate conditioning without silently tuning it on test data.
6. **Incomplete stage ablation.** The MNTP-only checkpoint already exists but is
   not evaluated. Current results cannot say whether MNTP helped and SimCSE hurt.
7. **Controls present in code.** Frozen eval models, physical prefix truncation,
   shared original IDs/boundaries/labels, no cache reuse, actual bidirectionality
   probes, checkpoint hashes, fresh compressors and FP64 Ridge solves. These are
   checks, not evidence that representations or optimization are adequate.
8. **Targets and sampling.** Labels are sampled terminal returns, useful but noisy
   supervision for a conditional success probability. They are not multi-rollout
   reference values. The actor is fixed; adaptation never learns actor-specific
   values. The existing future-dependent chunk-boundary selection limitation is
   retained. Fixed token boundaries are a necessary separate control. Fifty test
   questions and three compressor seeds do not measure adaptation-seed uncertainty.

## Audit the saved run before further training

`tools/encoder/audit_study.py` is CPU-only and never trains or modifies the
original study. It checks runtime/source hashes, aligned labels/boundaries,
feature identities and probes, fit hashes, selected-checkpoint restoration, and
reproduction of reported test scores. It reports:

- actual questions, answers, prefix counts and class rates by split;
- actual SLICE update counts, validation histories, and best/final epoch;
- training/validation/test Brier for both compressor sigmoid and raw Ridge;
- a constant baseline learned from training returns, using equal-question scoring;
- out-of-range predictions and initial-state-only error;
- corpus integrity, exact trainer configs, and recorded early/late training losses.

After committing/pushing the new diagnostic and pulling on the cluster, run on a
compute allocation from the repository root, using the existing encoder environment:

```bash
bash scripts/container_exec.sh "$PWD/.venv-encoder/bin/python" \
  tools/encoder/audit_study.py \
  --study outputs/encoder-offline-numina-v2 \
  --out outputs/encoder-offline-numina-v2-audit.json
```

No existing hashed training source is changed by adding this diagnostic. Audit
output must be fresh and outside the original study directory. Repeated reading
of already-inspected test results is diagnostic only; select new recipes using
training/validation and reserve a fresh evaluation set for a confirmatory study.

## Experiments to distinguish explanations

1. Reuse frozen base, bidirectional, MNTP-only and adapted checkpoints. Compare
   current last-token readout with mean pooling at identical selected layers for
   *all* checkpoints. Add a separate official final-layer mean-embedding control.
   Train fresh SLICE/head artifacts, retain the same width and primary Ridge rule.
2. Check whether representation ordering changes before vs. after SLICE/Ridge.
   Use validation learning curves and a wider SLICE optimizer-step budget. If the
   best epoch is still the final epoch, extend training; a plateau or rising
   validation loss does not justify more epochs by itself.
3. Only then compare encoder adaptation on larger training-only math corpora,
   longer natural text/prefix contexts, and true contrastive batches of 16/32
   where memory permits. Keep the official trainers unmodified. Gradient
   accumulation is not a workaround for contrastive-batch memory requirements.
   Compare equal optimizer steps and equal token/example budgets explicitly.
4. Use nested training-question learning curves (e.g. 500/2,000/5,000 if available)
   with identical data for each arm and more adaptation seeds. The current saved
   source has only 500 training questions, so expansion needs new training-only
   rollouts; it is not a submission flag. Do not promote validation/test questions.
5. Confirm promising variants on shared state prefixes with actor-specific MC
   continuation references and fixed-token chunk controls, then matched multi-seed
   PPO. Do not use retrospective PRM results to justify future-token access.

These tests may confirm a training/readout bottleneck or may preserve the causal
advantage. Either result is informative. Do not change or discard the pilot to
force the expected ordering.
