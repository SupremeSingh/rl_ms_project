# Offline prediction demo

Compare SLICE/token, SLICE/chunk, RMS-layer/token, RMS-layer/chunk and a full
transformer critic on identical prefixes with shared Monte Carlo references.
Token/chunk labels describe the states used to fit each head; every head scores
the same evaluation states. RMS means the RMS-normalized two-thirds-layer readout.
The transformer was supervised on terminal returns from one frozen policy. It is
**not a conventional PPO-trained critic**. This demo does not train any model.

## Saved predictions (CPU)

Copy these files from a completed `fixed-policy-critics-v1` output into one local
directory: `manifest.json`, `bank.json`, `reference-*.json`, and
`predictions-{slice-token,slice-chunk,rms-layer-token,rms-layer-chunk,transformer}.json`.
The existing Duke output is `outputs/fixed-policy-critics-v1`. Files must come
from the same study; identities, state coverage and sample counts are checked.

From the installed standalone package folder:

```bash
.venv/bin/python -m slice_critic.demo --study /path/to/fixed-policy-critics-v1 --out demo-output
```

Prints a five-method MAE/RMSE table and the first three states' predictions.
Writes `predictions.csv` and `summary.json` to a **new** directory. The source is
read-only. Metrics use equal-question weighting and unmodified raw predictions.
No GPU, model weights or generation is needed for replay. The folder contains
no pretrained weights or per-state research data; these must be copied from Duke.

## Recompute predictions (trained models required)

Also copy `sparse.pt`, `ridge.pt`, `linear.json`, `neural.json` and the complete
`transformer/` Hugging Face checkpoint. Supply the matching frozen Qwen encoder
snapshot separately (including its tokenizer). Checksums reject substituted models.

```bash
.venv/bin/python -m slice_critic.demo --study /path/to/fixed-policy-critics-v1 \
  --recompute --encoder /path/to/qwen-math --device cuda --out demo-recomputed
```

Loads the encoder once for all four Ridge heads, releases it, then loads the
transformer critic. Uses exact saved prompt/prefix token IDs without truncation.
This path needs the `train` dependencies and sufficient model memory. It has
CPU adapter tests; real checkpoint/GPU recomputation must be checked on Duke.
