# SLICE + chunking + Ridge: figure caption

**SLICE with chunk-level states and cumulative Ridge prediction.** (1) A
reasoning answer is segmented using the existing VinePPO math splitter. Each
chunk-start state contains the question and all preceding chunks. (2) A frozen
causal Qwen backbone supplies the last available token's hidden state at blocks
28, 27, 25, 21 and 13. Per-block RMS normalization, concatenation, and fixed
training-set centering/scalar scaling yield 7,680 coordinates. (3) A previously
trained SLICE projection produces 768 sparse ReLU features; an appended intercept
gives a 769-dimensional design vector. (4) A shared linear head estimates eventual
correctness at each chunk start. The lower fitting path uses verified terminal outcomes to supervise raw-sum
Ridge statistics accumulated across labeled batches. A regularized linear solve
updates the head while the feature map stays fixed. The learned sparse projection
and normalization are established in a separate initial training stage. No future
answer tokens enter prefix features, although the splitter identifies boundaries
from the completed answer. Example chunks are schematic. Raw Ridge predictions
are unbounded. Actor optimization and advantage estimation are outside this figure.

## Files

- `slice-chunk-ridge.svg`: editable vector artwork; text is retained as text.
- `slice-chunk-ridge.png`: high-resolution preview.
- `../../output/pdf/slice-chunk-ridge.pdf`: vector PDF with embedded TrueType fonts.
- Rebuild: `python3 scripts/figures/make_slice_chunk_ridge.py` from the repository root.

This illustrates the causal SLICE/chunk/Ridge pipeline. It does not depict the
paused LLM2Vec mean-pooling comparison, or the proposed HiMo extension. Sparse
activations do not imply a sparse-kernel runtime or an established cost saving.
