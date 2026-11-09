# Does reasoning-chunk learning improve cumulative Ridge PPO?

**No for the tested SLICE recipe: chunking lowered actor accuracy from 75.51% to 73.54%, a −1.97 pp change with adjusted interval [−3.29, −0.56], while saving 11.7% online training compute. The raw-layer change was inconclusive at −0.51 pp, interval [−2.08, 1.01].**

Four cumulative Ridge arms used **five paired actor seeds (901–905)**,
**120 updates × 64 answers/run**, **2,000 greedy test questions**, a 2,048-token
response cap and **two A5000 GPUs/run**. All used frozen features, one head,
epsilon .01, gamma 1 and GAE lambda .95. The existing Numina data and compressor
were reused. The layer control is the **raw** two-thirds layer, not the
RMS-normalized control from earlier transfer experiments.

| Representation / credit unit | Actor accuracy | Training GPUh/run | Accounted GPUh/run | Fitted states/run |
|---|---:|---:|---:|---:|
| **SLICE / token** | **75.51%** | 9.691 | 11.704 | 5,730,546 |
| SLICE / chunk | 73.54% | 8.554 | 10.521 | 325,538 |
| Raw layer / token | 73.25% | 7.785 | 9.669 | 5,787,715 |
| Raw layer / chunk | 72.74% | 7.846 | 9.874 | 324,779 |

All entries are means across five runs. **Chunk minus token** accuracy:

- **SLICE: −1.97 percentage points**, adjusted interval **[−3.29, −0.56] pp**;
  every seed lost accuracy. Online training compute fell **11.7%**.
- **Raw layer: −0.51 pp**, adjusted interval **[−2.08, +1.01] pp**;
  the accuracy difference is inconclusive, with no measured compute saving.

Chunking fitted roughly **18× fewer states**, without proportional total savings.
Generation, frozen encoder passes and actor optimization still require work.
Fallbacks affected **0.79%** of SLICE/chunk and **0.73%** of layer/chunk training
answers, using groups of at most 32 original tokens when needed.

**Takeaway: retain token-level SLICE + Ridge as our accuracy-focused baseline.**
This chunk recipe traded accuracy for a modest SLICE compute saving. It does not
establish that chunking generally fails: .95 decays per chunk rather than per
token, state weighting changes, and the same epsilon has different relative
strength. We retain learned values, not VinePPO's Monte Carlo value estimator.
Intervals use approximate paired seed/question bootstrap with Bonferroni nominal
adjustment for two primary contrasts; reused questions make this a controlled
follow-up, not an independent benchmark.

**Recovery and costs:** 18 original runs were retained; SLICE/chunk seed 903 and
layer/chunk seed 904 were restarted from update zero after fixing unmapped-token
alignment. The report records both code versions. Accounted GPU-hours add
initialization and base/final evaluation to training; they are not a complete
allocation ledger. New and original smoke checks cost another **0.788** and
**0.792 GPUh**, respectively. Failed attempts and prior offline feature learning
are additional and excluded from the per-run means.

[Protocol and recovery details](../docs/CHUNK_COMPARISON.md) ·
[Recorded results and per-seed differences](../docs/results/numina-chunk-comparison.json).
Cluster report: `outputs/numina-chunk-comparison-32-recovery-v1/report.txt`.

## Evidence and scope

[Recorded report data](../docs/results/numina-chunk-comparison.json). Original cluster study: `outputs/numina-chunk-comparison-32-v1`; replacement report: `outputs/numina-chunk-comparison-32-recovery-v1/report.txt`. Eighteen original successful runs plus two explicitly recorded software-failure recoveries; no outcome-based replacement.
