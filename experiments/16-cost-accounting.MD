# Have we established an end-to-end cost advantage for transferred SLICE?

**Not yet: the supplied transfer-study ledger recorded 53.870 allocated GPUh and 436.264 allocated CPUh across the included studies and failed/smoke attempts. It did not establish a per-method equal-quality saving or a break-even reuse count.**

| Recorded allocation | Hours |
|---|---|
| Included transfer studies: GPU | 53.870 |
| Included transfer studies: CPU | 436.264 |
| Original joint generation/extraction: GPU | 31.309 |
| New clarification generation/extraction: GPU | 20.256 |

Read-only retrospective audit. Original generation/extraction is counted once despite reuse. Five-seed sweeps are research expenditure, not the cost of deploying a single selected critic. Fit jobs jointly train multiple compressors and penalty grids; they do not provide fair per-head or per-method timers. Missing accounting is not zero cost.

For 100 target questions, adaptation used 400 generated answers: 119,725 response tokens for GSM8K and 218,907 for MATH. At 2,000 questions these increased to 8,000 answers and 2,444,191 / 4,137,291 tokens. Those answers were shared across comparison arms; the ledger does not demonstrate that SLICE alone saved their generation cost.

No isolated generation versus extraction timings, critic throughput, GPU utilization, energy, label price or dollar estimates were measured. Lower-dimensional matrices save head-state memory mathematically, but total GPU memory and break-even cost require separate measurement. Only explicitly supplied ledgers are included; this is not total project expenditure.

## Evidence and scope

[Full cost and data ledger](data/cost-report.txt). Cluster: `outputs/transfer-cost-audit-v1/report.txt`. Allocated hardware-hours differ from model-phase timing used in PPO tables.
