# VinePPO boundaries and unpadded encoding

This supersedes the fixed-boundary/padded-batch execution design in the older
pooled-run notes. Protocol: `encoder-vine-unpadded-v3`. GPU validation pending.

Use the existing VinePPO MATH splitter, including its existing fixed-token
fallback on splitter failures. Encode question plus available answer prefix at
each boundary, never just the isolated chunk. MC uses up to four spread-out
existing boundaries via the repository's boundary selection helper; shorter
answers contribute fewer states. Boundary selection from completed answers is
accepted by the user for this experiment. It is not learned segmentation.

## No padding, including during adaptation

`unpadded_adapter.py` changes wrapper execution only: the official SimCSE collator
still collects eight examples per rank, but tokenizes and encodes each example
separately. Every backbone forward has shape `[1, actual_length]`, an all-one
attention mask and positions starting at zero. Embeddings are concatenated
without detaching their graphs. The unchanged official contrastive loss gathers
eight embeddings from each of four GPUs, retaining 32 global candidates.
It is one joint loss, not gradient accumulation over singleton losses.

This is an explicit execution adapter, not a claim of stock out-of-box batching.
Official installed/upstream model, attention, pooling and loss files remain
unchanged. Only the expanded comparison invokes the adapter; the old pilot does
not. Training dropout is retained. Retaining eight per-example computation
graphs still consumes memory; this is not a promise of lower peak memory.
Multiple singleton forwards are slower than one dense batched forward.

MNTP uses batch one, accumulation two, no maximum-length padding, and the existing
official grouping/collation pipeline. Its effective optimizer batch remains two.
The accumulated MNTP token losses do not have the SimCSE negative-pool issue.
Inputs longer than the sealed context are rejected; no truncation is silently
introduced to make batching work. BF16 and the existing numeric tolerances remain.

## First run the standalone four-GPU gate

After commit/push/pull, submit from the project root:

```bash
source scripts/cluster_env.sh
sbatch --partition=compsci-gpu --gres=gpu:a5000:4 \
  --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=128G --time=01:00:00 \
  --job-name=encoder-unpadded-check \
  --output=outputs/encoder-unpadded-check-v1.log \
  --wrap='export MATH_RL_GPU=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; bash scripts/container_exec.sh "$PWD/.venv-encoder/bin/python" -m torch.distributed.run --standalone --nproc_per_node=4 tools/encoder/distributed_check.py --study outputs/encoder-pooled-smoke-v2 --report outputs/encoder-unpadded-check-v1.json'
```

This reads only the sealed older manifest/profile and verifies identical base
weights; it does not modify or approve that old study. It requires the clean
pinned official checkout. It checks global negatives, gather/parameter gradients,
singleton-group versus individual encoding, and two-view LoRA backward at the
full 2560-token cap. No unrelated padded configuration needs to pass, since zero
padding is enforced by the adapter itself. Failures are not bypassed.

```bash
cat outputs/encoder-unpadded-check-v1.log
cat outputs/encoder-unpadded-check-v1.json
```

## New end-to-end smoke after the gate passes

```bash
python3 tools/encoder/submit_compare.py --mode smoke \
  --questions outputs/jlens-large-numina-v2/questions.json \
  --out outputs/encoder-vine-unpadded-smoke-v1
```

Only after the new complete smoke passes:

```bash
python3 tools/encoder/submit_compare.py --mode main \
  --smoke outputs/encoder-vine-unpadded-smoke-v1 \
  --out outputs/encoder-vine-unpadded-numina-v1
```

Readout A–D, normalization, full-pipeline Ridge selection, historical
10,000/1,000/2,000 split assignments, three adaptation seeds, five SLICE seeds and
the shared MC continuation budget remain as specified. Prior smoke verification
cannot authorize this changed protocol. Mount aliases may be relocated within
the project, with the original artifact/model hashes still enforced.
