# Architecture figures

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
