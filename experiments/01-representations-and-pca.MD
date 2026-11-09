# Do multi-layer hidden states outperform J-lens and single-layer readouts?

**In this offline study, normalized multi-layer Ridge had Brier 0.18917 versus 0.18976 for full J-lens, 0.19702 for the two-thirds layer and 0.21306 for the final layer. PCA compression to 768 features worsened normalized multi-layer Brier to 0.20216.**

| Representation | Width | Ridge accuracy | Ridge Brier |
|---|---|---|---|
| Normalized multi-layer | 7,680 | 72.01% | 0.18917 |
| J-lens concatenation | 7,680 | 71.91% | 0.18976 |
| Two-thirds hidden layer | 1,536 | 70.27% | 0.19702 |
| Final layer | 1,536 | 66.89% | 0.21306 |
| Normalized multi-layer + PCA | 768 | 69.30% | 0.20216 |
| Normalized multi-layer + PCA | 1,536 | 70.14% | 0.19773 |
| Normalized multi-layer + PCA | 3,072 | 71.30% | 0.19229 |

The test set had already been inspected. PCA used training all-token Gram statistics; dimensions were selected on validation. Ridge/LSTD(.99) heads were compared without new generation or PPO updates.

J-lens did not establish an accuracy or cost advantage over the normalized hidden-state representation. This motivated learning a task-specific compressor instead of relying on variance-preserving PCA. These results do not establish that J-lens is generally ineffective or that PCA always fails.

## Evidence and scope

[Full report extract](data/pca-report.txt). Cluster: `outputs/jlens-compression-12950163/report.txt`. Historical representation names are retained only to identify what was tested; J-lens is not used by current SLICE. Prediction accuracy is over retained offline states, not actor problem-solving accuracy.
