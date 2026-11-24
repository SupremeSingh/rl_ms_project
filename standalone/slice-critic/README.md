# SLICE Critic

A standalone Python package for compact value estimation from frozen LLM features.
Includes a configurable compressor (768 features by default), Ridge/LSTD heads,
cumulative fitting, inference, and token or reasoning-chunk states. Chunking
supports MATH and GSM8K, preserves original token IDs, and uses configurable
fixed-token fallbacks (32 tokens by default) when MATH splitting/alignment fails.

Currently supports Qwen2.5-Math-1.5B and numeric math answers. No VERL or Slurm
required. Model weights, trained critic artifacts and datasets are supplied separately.
The adapted VinePPO splitter and its license are included in `src/slice_critic/_vendor/`.

## Setup

Use Python 3.10+ on Linux; a CUDA GPU is recommended for training. From this folder:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[train,dev]'
.venv/bin/python -m pytest -q tests

# Keep symbolic verification dependencies in a separate environment.
python3 -m venv .venv-verifier
.venv-verifier/bin/python -m pip install --no-deps -e .
.venv-verifier/bin/python -m pip install math-verify==0.9.0 latex2sympy2_extended==1.11.0 antlr4-python3-runtime==4.13.2 sympy==1.14.0 mpmath==1.3.0
```

Edit `config.example.json`: provide a local model directory, your verifier Python
path, and a JSONL dataset containing `problem` and numeric-string `answer` fields.
Set `chunking` to `math`, `gsm8k`, or `token`.

```bash
.venv/bin/python -m slice_critic preview --config config.example.json
.venv/bin/python -m slice_critic train --config config.example.json --out critics/my-critic
```

Tests use small synthetic fixtures; no downloads, datasets, GPU or trained weights
are required. They check fitting, inference, chunking and artifact validation.
For an existing five-method offline prediction comparison, see
[the demo setup](examples/README.md). The demo can replay saved predictions on
CPU or run supplied trained checkpoints; weights and research data are not bundled.
