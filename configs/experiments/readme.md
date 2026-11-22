# Paper experiment configurations

The repeated source-grouped experiment protocol is configured in
[`../paper.json`](../paper.json). Its `evaluation` section configures
`qualisr-run-experiments`. The feature-family,
PCA, transfer, statistical and latency tools are documented in
[the experiment guide](https://github.com/msu-video-group/QualiSR-Lab/blob/main/experiments/README.md).

`diagnostics/` contains the five single-split configurations for the
feature-importance, SHAP and noise-control illustrations:

| File | Purpose |
| --- | --- |
| `baseline.json` | Four-NR/RLFN-FR/SR/compact-WASD control for the importance illustration |
| `qalign_with_baseline.json` | Same control with Q-Align added |
| `feature_relevance.json` | Fixed 34-feature pool for the SHAP discussion |
| `noise_with_baseline.json` | Gaussian/uniform noise controls added to the baseline |
| `noise_only.json` | Noise-only control |

Set `QUALISR_DATA_ROOT` to the prepared dataset root. The configured dataset
names determine the sample-ID prefixes used to join the feature CSVs.

The PCA-based diagnostics read `pca/vgg_pca5.csv` and `pca/resnet_pca5.csv`.
Prepare those coordinates for the configured seed-42 holdout, fitting on
training rows and transforming validation rows. The noise controls also read
`noise.csv` with five Gaussian and five uniform columns, as defined by the
`noise` feature group in `configs/pipeline.json`. The checkout includes these
files for Set120.

To run a diagnostic into a fresh destination:

```bash
qualisr-run-regressors --config configs/experiments/diagnostics/baseline.json \
  --plots-root /path/to/new/diagnostics
```

`configs/pipeline.json` and the packaged image-free example remain available
for feature extraction and a small installation check.
