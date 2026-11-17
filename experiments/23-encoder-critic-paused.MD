# Does bidirectional encoding improve SLICE's prediction of eventual correctness?

**The completed comparisons favor causal Qwen. In the larger follow-up,
LLM2Vec-adapted Qwen has higher validation Brier error on both direct pooled
readouts across two completed adaptation seeds. SLICE C/D fitting exhausted
host RAM, so that part remains unanswered. We paused this direction on
17 November 2026 to focus on other studies. These encoder studies ran no PPO.**

## The idea, in plain language

The actor writes an answer one step at a time. The critic receives the question
and the answer so far, and estimates the chance that the actor will finish correctly.

We hoped an encoder would make a better representation of that partial reasoning.
A causal model's last token already has access to the whole prefix. An encoder
receives no extra future text: it lets every token incorporate all the available
prefix. For example, an early equation can be represented in light of a later
explanation that has already been generated.

This is a possible representation advantage, not an information advantage or a
guarantee. We tested Qwen converted using LLM2Vec's MNTP and SimCSE recipe, not
a model originally pretrained as an encoder. These text-adaptation objectives
do not directly optimize actor-specific value prediction.

## First completed comparison: causal features also won with SLICE

`outputs/encoder-offline-numina-v2` used 500 training questions with four answers
each, 100 validation questions with four answers each, and 50 test questions with
one answer each. Selected-layer last-token features fed a freshly trained SLICE
compressor and Ridge head. It used three compressor seeds, one encoder adaptation
seed and one fixed actor.

| Backbone | Equal-question test raw Brier ↓ |
|---|---:|
| Causal Qwen | **0.181287** |
| Bidirectional Qwen, without adaptation | 0.216223 |
| Bidirectional Qwen, MNTP + SimCSE adapted | 0.233771 |

This small exploratory result motivated checking pooling, context, contrastive
batch size and execution before drawing a broader conclusion. Its absolute scores
cannot be compared directly with the next table: the protocols and splits differ.

## Larger follow-up: what we changed

Study: `outputs/encoder-vine-unpadded-numina-v1`.
Protocol: `encoder-vine-unpadded-v3`.

- Preserve the historical **10,000 train / 1,000 validation / 2,000 test** questions,
  with eight shared actor answers each: **104,000 answers**.
- Compare causal Qwen with three adapted encoder seeds, **42, 43 and 44**.
  The actor stays fixed; every backbone sees the same answers and prefix states.
- Adapt on 160,000 question-plus-prefix texts. A 2,560-token cap covers all
  profiled adaptation inputs, replacing the pilot's isolated short windows.
- Run 1,000 MNTP and 1,000 SimCSE steps per adaptation seed. SimCSE uses a joint
  pool of 32 examples across four GPUs; accumulation does not enlarge that pool.
- Use the existing VinePPO math splitter and fallback. Inputs contain only the
  prefix, although boundary selection uses completed answers. This limitation
  was explicitly accepted; the splitter is not a newly learned boundary model.
- Use unpadded prefix encoding after diagnostics exposed padding/position
  sensitivity in the tested BF16 setup. Its exact root cause was not established.
  The no-padding execution path passed the distributed check and smoke run.

| Arm | Representation | Final predictor |
|---|---|---|
| A | Mean across tokens at final block 28 | Ridge |
| B | Mean across tokens at block 19, the fixed two-thirds layer | Ridge |
| C | Per-layer means at blocks 28, 27, 25, 21, 13 → SLICE 1536 | Ridge |
| D | Same five-layer input → SLICE 768 | Ridge |

Normalization definitions are shared across backbones. C/D were assigned five
paired compressor seeds and validation checkpoint selection of the complete
SLICE-to-Ridge pipeline. Final evaluation was to include shared MC continuation
references alongside sampled-outcome Brier.

## Results available when we stopped

These are **validation outcome Brier scores**, with questions equally weighted.
Lower is better. Predictions are not clipped. Brier is not a percentage accuracy.

| Readout | Causal | Adapted-42 | Encoder − causal | Adapted-43 | Encoder − causal |
|---|---:|---:|---:|---:|---:|
| A: final-layer mean | **0.152635** | 0.172175 | +0.019539 | 0.170772 | +0.018137 |
| B: two-thirds-layer mean | **0.148712** | 0.165959 | +0.017247 | 0.164974 | +0.016262 |

Both completed encoder seeds are worse on both readouts. Block 19 is better than
the final block for both causal and adapted models. These are descriptive
comparisons; paired uncertainty intervals were not computed for this pause report.

The numbers are transcribed from cluster reports supplied on 17 November 2026.
Raw cluster artifacts remain the underlying record and are not bundled here.

## What finished, and what did not

| Work | Status at pause |
|---|---|
| Actor answers | All 104,000 generated |
| Encoder adaptation | All three seeds completed |
| Feature extraction | All 416,000 files; 32/32 shards completed |
| Direct A/B fits | Causal, adapted-42 and adapted-43 results saved |
| Adapted-44 fits | No completed fits in the saved pause report |
| SLICE C/D fits | Repeated host-RAM failures; no completed comparison |
| MC continuation evaluation | No completed shards |
| Final test report / verification | Not completed |
| Encoder-based PPO | Not run |

C/D fitting retained sampled tensors, concatenated another full matrix, then
created large FP64 normalization temporaries. Slurm confirmed host-memory OOM
against a 64 GB allocation, with about 58.7 GiB recorded maximum RSS for the
inspected job. This is an implementation failure, not a C/D accuracy result.

## Why stop here?

The earlier completed SLICE study favored causal features. The revised larger
direct-pooling study points in the same direction across two adaptation seeds.
We have no observed benefit yet that warrants prioritizing more encoder-specific
compute and debugging. Repeated unpadded prefix extraction also costs substantial
compute; this study establishes no cost saving.

We are **pausing this LLM2Vec conversion direction**, not concluding that all
encoder critics are ineffective. Native encoders, direct value fine-tuning and
the repaired C/D readouts remain untested here. We have no new PPO performance
or MC-reference result from the larger run.

A more useful next question may be whether richer supervision improves the
existing critic. One candidate is HiMo-style training: learn value-relevant
features of later states, then teach a prefix-only model to predict those features.
Value prediction would combine current features with predicted future features.
This is a proposal, not an implemented or validated gain, and does not require
an encoder. A small causal-backbone pilot with matched-capacity controls would
separate this idea from the encoder question. No 85% accuracy or cost reduction
is established.

## Saved artifacts and optional recovery

Cluster root: `/usr/xtmp/ms785/rl_ms_project/`.

- Original study: `outputs/encoder-vine-unpadded-numina-v1/`.
- Pause report: `outputs/encoder-paused-validation-v1/report.txt` and `summary.json`.
- Earlier pilot: `outputs/encoder-offline-numina-v2/report.txt`.

The user confirmed an empty Slurm queue after cancellation. Completed answers,
adaptations, features and heads were preserved. New recovery scripts use
disk-backed training features and bounded-block normalization, reuse successful
fits, and write to a fresh output. Local tests passed; full-scale GPU recovery
has not been run. Recovery requires an explicit decision to resume; this README
requests no more encoder jobs.

[Recovery procedure](../docs/ENCODER_PAUSED_RECOVERY.md) ·
[Comparison design](../docs/ENCODER_COMPARISON_V2.md) ·
[Unpadded execution](../docs/ENCODER_UNPADDED_RUN.md)
