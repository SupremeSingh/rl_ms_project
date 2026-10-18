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

## Publication-ready poster figures

`scripts/plot_probe_poster.py` makes two matched, white-background figures with
large text and consistent colors. **Use prediction-data mode below to recompute
metrics from the original saved arrays and trajectories, rather than plotting
copied aggregate scores.** No model weights, GPU, generation or fitting is needed.

```bash
python -m pip install 'matplotlib>=3.8,<4'
python scripts/plot_probe_poster.py \
  --source outputs/jlens-large-numina-v2 \
  --bottlenecks outputs/numina-sparse-768-v1 \
  --out outputs/probe-poster-figures-from-data-v1 --dpi 450
```

This reads `predictions.npz`, packed prefix labels/positions and saved trajectories.
It verifies identities and alignment, recomputes overall and question-only metrics
from the individual predictions, and then plots the recomputed values. It also
recomputes the constant baseline from the training labels. The evidence exporter
checks agreement with saved reports, but does not use their aggregate scores as
plotted inputs. `evidence/` inside the new output contains all recomputed metrics,
audit counts, paired comparisons and source hashes. Use a fresh output directory
for each prediction-data run. Saved per-prefix predictions are required; the
command does not rerun the LLM or the fitted probes.

On the configured cluster container:

```bash
source scripts/cluster_env.sh
bash scripts/container_exec.sh "$PWD/.venv-jlens/bin/python" \
  -m pip install 'matplotlib>=3.8,<4'
bash scripts/container_exec.sh "$PWD/.venv-jlens/bin/python" \
  scripts/plot_probe_poster.py \
  --source outputs/jlens-large-numina-v2 \
  --bottlenecks outputs/numina-sparse-768-v1 \
  --out outputs/probe-poster-figures-from-data-v1 --dpi 450
```

This is CPU-only analysis, not another training job. Matplotlib is also available
through the optional `.[figures]` installation extra. Summary-only plotting needs
just Matplotlib; prediction-data mode additionally uses the existing NumPy/PyTorch
analysis environment.

For the portable saved evidence included with the standalone folder:

```bash
python scripts/plot_probe_poster.py \
  --summary SLICE/results/reported-results.json \
  --out docs/figures/probe-poster
```

The output contains PNG (default 300 DPI), PDF, and editable SVG versions of:

- `prediction-quality-vs-size`: sampled-prefix correctness-prediction accuracy
  against the number of features passed to Ridge. Both axes are linear; the
  panel uses only short method labels and the two axis labels.
  Plain and sparse have the same width and almost identical accuracy, so their
  markers overlap and share the label "SLICE / Plain". Exact scores remain in
  `plotted-values.csv`. Interpretation and study limitations belong in the poster
  caption rather than inside this minimal panel. No head family is
  chosen using these test values. The feature-width comparison excludes the
  compressor parameters and does not imply a runtime speedup.
- `prediction-before-generation`: accuracy and AUROC using only the question,
  before any answer tokens. Shows Final Layer, Five Layers and SLICE: Sparse 768
  only, plus a constant training-prior reference. The size figure keeps all five
  representations.
  A question state is evaluated against multiple sampled answer outcomes;
  15,740 rows do not mean 15,740 independent questions.

Both figures compare only Ridge readouts, avoiding the changed fitting
weights/budgets of the neural reference. Mean bottleneck metrics are across
three heads on shared data, not an ensemble. Error bars are omitted because
marginal intervals are not provided in this summary; do not substitute the
paired difference intervals as per-method error bars. Results are descriptive,
conditional on verifiable answers, and use the inspected test questions.

`plotted-values.csv` preserves exact plotted values, counts and head counts.
`figure-manifest.json` records the input hash, software version and interpretation.
The script does not recompute scores and never needs the original large caches.
Use SVG/PDF for crisp poster placement or the PNG files when the editor requires
raster images. Keep the aspect ratio unchanged when resizing.
