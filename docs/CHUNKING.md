# v3: VinePPO-style reasoning chunks with learned critics

Status: CPU checks cover segmentation, token alignment, matrix accumulation,
credit assignment, and critic loss masking. Real GPU integration must pass before
launching a new study. Historical token-level results are not chunking results.

## Exactly what comes from VinePPO

Source: [McGill-NLP/VinePPO](https://github.com/McGill-NLP/VinePPO), pinned commit
`9f716e68bb934e71da86c226c1699c36b5553faf`.

- MATH uses `tasks/math_extract_steps_inplace.py`, as selected by its
  `tasks/math_inplace_no_answer_prefix.jsonnet` configuration. It protects LaTeX,
  splits around sentences/newlines, merges short parts, and attempts to divide
  long parts around language/math separators. The 100-character setting is a
  heuristic threshold, **not** a strict fixed chunk size.
- GSM8K uses newline splits, leading-empty-step merging, and final-answer merging
  for the literal `\n#### ` answer prefix. Numina/Qwen's boxed-answer format uses
  **MATH mode**, not that GSM8K-specific answer marker.
- `math_episode_generator_with_mc_advantages.py` maps each response token using
  its **first character's** chunk and assigns terminal EOS the last advantage.
- A chunk's raw advantage is `reward + V(next_chunk) - V(chunk)`, terminal V=0.
  Broadcast to token positions, then normalize advantages before the usual
  token-level clipped PPO loss. Do not average token-level GAE afterward.

We borrow the segmentation and broadcast rules; default GAE .95 deliberately
differs from upstream's one-step advantages. The MATH algorithm is vendored with its MIT notice in the installed package.
Only its md5 dependency is replaced by the identical local helper. Our wrapper
keeps a single chunk for short/blank outputs where the upstream splitter raises
an assertion; other segmentation failures stop the run rather than dropping
answers. Character boundaries inside a token belong to that token's starting
chunk; zero-token chunks are omitted. This makes transitions executable on the
**original sampled token IDs**, without rewriting actions. All tokens count,
including EOS and the last token of a capped response.

Fast tokenizer offsets are required. Response decoding must re-encode to the
same IDs; mismatches raise a clear error rather than changing generated actions
or their PPO probabilities. This strict alignment must be exercised by GPU smoke.

## Our adaptation: learned values, not Monte Carlo VinePPO

We retain the frozen encoder/SLICE, binary outcome verifier, and cumulative Ridge
or LSTD. We do **not** generate the auxiliary continuations that make VinePPO's
value estimator Monte Carlo/unbiased. No unbiasedness or speedup claim transfers
from their paper to this implementation.

For each answer, states are exact prefixes before the first token of each chunk.
A causal encoder pass supplies those states; no hidden states are averaged across
a chunk. Segmentation reads the completed response, as upstream does, but each
critic feature remains a pre-action prefix feature. Chunk boundaries are a text
heuristic, not a learned causal abstraction or a guarantee of a Markov compressed
state. Frozen transformer attention still processes the full answer.

1. Generate 64 fresh answers, split each, and verify the final answer.
2. Extract one frozen feature vector per chunk start; append intercept.
3. Ridge adds one `outer(x,x)` and `x * outcome` per chunk. LSTD uses adjacent
   chunk starts as transitions, resetting traces per answer and using all-zero
   terminal successor (including intercept).
4. Retain **one shared A,b**, raw sums; solve with epsilon **0.01** each update.
5. Compute GAE across chunk TD errors with lambda .95, expand to token positions,
   whiten over valid tokens, and run the unchanged PPO actor loss.

Conventional transformer PPO follows the same chunk credit calculation. Its
critic learns the detached **chunk GAE lambda-return targets**, using only chunk-start
positions in its value loss; the full causal attention mask is unchanged. Its
optimizer/profile is retained, but should be calibrated afresh for an accuracy
study. GRPO remains an answer-level outcome baseline: chunking does not invent
step-specific information for it.

The new default is `experiment.chunking=math`, `experiment.critic_folds=1`,
`algorithm.lam=0.95`. **Chunking and advantage estimation are separate choices.**
For chunk j, gamma=1:

```text
delta[j] = reward[j] + V[j+1] - V[j]
advantage[j] = delta[j] + gae_lambda * advantage[j+1]  # backward recurrence
critic_target[j] = advantage[j] + V[j]
```

The terminal successor and terminal advantage are zero. Broadcast advantages and
lambda-return targets to tokens only **after** this recurrence. The conventional
critic fits targets only at chunk starts; whitening occurs after token broadcast.
`algorithm.lam=0` recovers upstream VinePPO's raw value-difference formula;
`algorithm.lam=1` gives full-return credit. Neither setting is required for
chunking. Our default retains the original GAE hyperparameter .95; one factor of
.95 now represents one reasoning chunk, not one token, so the effective horizon
is longer in token units. LSTD's independent `experiment.critic_lambda` remains
configurable (.99 package default; .9 comparison/smoke recipe) and also operates
in chunk units. SLICE compression and Ridge/LSTD equations are unchanged.
Same epsilon with fewer state updates also changes its effective relative strength.
This is a new training recipe, not an algebraically equivalent optimization.

Offline package training defaults to `TrainingConfig(chunking='math')`: sample
up to eight chunk-start states for compressor fitting/evaluation and fit the
linear head on every retained training chunk. `gsm8k` and `token` are explicit
alternatives. Manifests record state units, splitter commit, and positions.
Standalone scoring can still score an arbitrary exact prefix. Old artifacts can
supply frozen features to the new PPO path; their offline head is never loaded.

## Reproduction and fresh runs

Legacy token PPO requires all of:

```text
experiment.chunking=token
algorithm.lam=0.95
experiment.critic_folds=2   # only for historical two-fold experiments; use 1 for one-head
```

Worker settings follow the driver's chunking via configuration interpolation.
Prepared studies lock their code; do not mutate their manifests to bypass this.
The old `import-study` route intentionally preserves the historical token protocol.
Use artifact-based preparation and a **new** output directory for chunked studies.
All existing results/actors/artifacts remain untouched.

After committing/pushing v3, on Duke:

```bash
cd /usr/xtmp/ms785/rl_ms_project
git pull --ff-only origin v3
source scripts/cluster_env.sh
srun -p compsci --cpus-per-task=8 --mem=32G --time=00:30:00 --pty bash -i
source scripts/cluster_env.sh
bash scripts/container_exec.sh bash scripts/install_critic_package.sh
bash scripts/container_exec.sh "$PWD/.venv/bin/python" -m pytest -q
exit
mkdir -p logs
sbatch scripts/submit_chunked_smoke.sh \
  outputs/slice-package-smoke-v2/training/artifact \
  outputs/chunked-pipeline-smoke-v1 \
  outputs/numina-prod-replay-v1/data/questions.json
```

The artifact and question paths above are previously used cluster inputs. The
script excludes the artifact's fitting/selection families. It runs two real actor
updates for SLICE Ridge/LSTD, raw two-thirds Ridge/LSTD, conventional PPO and
GRPO on two A5000 GPUs, sequentially in one allocation (8-hour time limit, not
a runtime promise). Evaluation is validation only.

When complete:

```bash
cat outputs/chunked-pipeline-smoke-v1/status.json
cat outputs/chunked-pipeline-smoke-v1/pipeline-verification.json
```

Require `passed: true` in **pipeline-verification.json**, not just the earlier
SLICE-only verification.json. Logs are `logs/chunked-ppo-smoke-JOBID.out` and
per-method logs in the output. Audits count chunks separately from response tokens,
check one-head statistics and two actor updates, and retain conventional critic
functional checks. This is integration evidence, not an accuracy ranking.

To test the one-step comparison in a **new** output directory, the underlying
smoke CLI accepts `--gae-lambda 0`; individual PPO commands accept
`algorithm.lam=0`. New full-study manifests may explicitly set `actor_lambda=0`
before their provenance is sealed. Do not change an already prepared manifest.
