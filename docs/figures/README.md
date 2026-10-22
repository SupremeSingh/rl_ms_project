# CLEAR architecture and SLICE critic figures

`final-architecture.svg` is the editable vector source for the current math-stage
architecture; `final-architecture.png` is its README rendering. Labels and routing
were checked against `bottleneck_features.py`, `cumulative_lstd.py`, the Numina
online protocol and the final README. Two systems mean persistent question folds;
Ridge and LSTD are alternative fitting methods. Rewards also enter GAE directly,
as labeled in that box. No gradient updates the encoder or frozen bottleneck.

The historical `ppo-architecture.png` and `cross-fitted-lstd.png` describe earlier
single-layer experiments. They remain linked from HISTORY.md for provenance and
must not be presented as the final sparse-768 architecture. Their generated
illustrations are historical explanatory material, not implementation specifications.

Lean search is future work and is intentionally absent from the current-system
figure. Sparse features do not imply sparse kernels or a measured speedup over
plain features.


## SLICE poster: offline probe system

`slice-system-design.svg` is the editable vector source, and
`slice-system-design.png` is a 3600 × 2000 rendering. This high-level diagram
covers correctness prediction only. Solid arrows show the prefix-to-score
prediction path; dashed arrows supply offline outcome supervision. The
compressor is learned using a temporary classifier and then frozen before Ridge
fitting. The LLM remains frozen throughout. Full answers and references provide
training labels; they are not future inputs to a partial-prefix predictor.
The detailed layer-compression diagram is a separate poster panel.

Suggested caption: **SLICE predicts eventual final-answer correctness from an
available question/answer prefix. A frozen LLM supplies internal representations,
a learned frozen compressor produces 768 features, and a Ridge readout produces
a correctness score. Offline verification labels train the compressor and readout
in two stages; they are not required at prediction time.**


## Novice-facing visual overview

`slice-system-visual.png` is an AI-generated explanatory illustration made with
the built-in image-generation tool. Paper-like sheets depict transformer layers;
five schematic streams depict selected hidden-state vectors, followed by the
SLICE compressor, a schematic 768-feature vector, a linear readout and a
correctness-score dial. Layer positions, marks and the dial are illustrative;
no displayed score is an experimental measurement. The lock denotes fixed LLM
weights. The dial denotes a correctness score, not certified correctness or
proof of calibrated probabilities. No future answer tokens enter prediction.
The precise implementation remains defined by the source code and protocol.
The prompt is stored in `slice-system-visual-prompt.txt`.


## Minimal academic SLICE overview (preferred poster version)

`slice-system-minimal.svg` and its 4000 × 1400 PNG rendering replace the pictorial
style for the poster overview. Flat layer planes show selected depths
schematically, each contributing the last-prefix-token hidden state. Their
combination is reduced from 7,680 to 768 coordinates by the trained projection
and ReLU, followed by a linear correctness readout. Layer positions and vector
entries are schematic. No numerical score, layer interpretation, probability
calibration or runtime claim is implied. This is the prediction path only;
offline supervision belongs in the method text or a separate training diagram.
No 3D shading, decorative objects, physical funnel, gauge or gradients are used.
The earlier overview versions are retained as drafts.


## Narrow-column portrait version

`slice-system-portrait.svg` is a 3:4 vertical re-layout of the minimal academic
prediction diagram. `slice-system-portrait.png` is its 2250 × 3000 rendering.
Read top to bottom: prefix, frozen transformer, combined selected states,
compression, 768 features, linear Ridge readout, eventual-correctness score.
Layer planes and vector cells are schematic. No illustrative numeric outcome
or probability is shown. Use the vector source for print scaling.

The final figure now labels LSTD(0.9), matching the completed six-method Numina
comparison. Ridge remains the selected configuration and has no trace lambda.

Naming: the final architecture depicts CLEAR, the complete RL pipeline. The
SLICE poster figures depict only the critic system and retain the SLICE name.
