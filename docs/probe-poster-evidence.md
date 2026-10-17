# Offline probe poster evidence

This export supports the poster about eventual-answer-correctness prediction from
frozen internal states. It does not use online PPO results, generate new answers,
fit heads, change thresholds, or select representations from test performance.
The test questions were already inspected in preceding work.

From the cluster checkout, after syncing the code:

```bash
source scripts/cluster_env.sh
bash scripts/submit_probe_poster.sh \
  outputs/jlens-large-numina-v2 outputs/numina-sparse-768-v1 \
  --out outputs/probe-poster-evidence-v1
```

This submits one CPU job (8 CPUs, 32 GB; no GPUs). Its two-hour allocation is a
ceiling, not an expected duration. It reads saved metadata, trajectories and
predictions, checks alignment and report consistency, and runs small cached-feature
CPU timing trials. No model download or LLM forward pass is needed.

After successful completion:

```bash
cat outputs/probe-poster-evidence-v1/report.md
```

The Slurm log is beside the output directory as `probe-poster-evidence-v1.slurm-JOB.out`.
Use a new output name for another export; source studies are read-only.

## Deliverables

- `summary.json`: exact train/validation/test retained counts; original all-attempt
  status counts from the source report; reconstructed test exclusions; hashes;
  accuracy, Brier, AUROC, ECE and calibration bins per head; head averages;
  paired question-bootstrap differences; CPU timing and model size accounting.
- `metrics.csv`: overall, question-only, absolute-prefix-position, relative-position
  and capped/uncapped metrics. Relative positions use future trajectory length for
  diagnosis only. Sample counts and outcome prevalence differ by position; these
  comparisons are not causal effects of reasoning progress.
- `accuracy.svg`, `brier.svg`, `prefix-brier.svg`: editable vector charts. Bars are
  descriptive and do not display uncertainty; paired intervals are in the JSON.
- `report.md`: concise table and presentation checklist.

The mean across heads is the mean of their separate metrics. It is not the metric
of an ensemble. Question bootstraps keep all answers/prefixes together and average
head errors first; they do not estimate variability over independent training
corpora. The resnet reference, if its saved outputs exist, is checked against its
source report. It used sampled-prefix training, whereas Ridge used all-token
fitting: that difference must accompany a comparison between these head families.

The 768-dimensional readout has 769 parameters including its intercept, but the
learned compressor also has approximately 5.9 million parameters. Its encoder
forward pass, five-layer capture and compressor are part of the complete probe.
The cached-feature CPU benchmark excludes encoder work, transfers, I/O and fitting.
It can distinguish readout from compression costs, but cannot establish end-to-end
latency or GPU throughput. Matrix storage is counted separately from inference.
Sparse activations do not imply sparse execution kernels.

Before printing, inspect source `audit.jsonl` and `responses.html` and report that
labels are automated and results condition on verifiable answers. A low parse
failure rate alone does not validate the labels. Stronger claims about detecting
mistakes before reasoning require the question-only and early-prefix results.
