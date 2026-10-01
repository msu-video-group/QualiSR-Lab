# Image features

Feature extraction is implemented in [`qualisr/features.py`](../qualisr/features.py). The [main README](../README.md#standalone-tools) contains command examples; use the unified pipeline to preserve dataset sample IDs across stages.

## Configuring features

Set `features.common.fr_metrics` and `features.common.nr_metrics` in the pipeline JSON, or pass comma-separated `--fr-metrics` and `--nr-metrics` lists to `qualisr-extract-features`. Metrics available through [PyIQA](https://github.com/chaofengc/IQA-PyTorch) can be selected without editing the source.

VGG16 and ResNet50 produce SR-image embeddings; `ref-vgg` and `ref-resnet` use the reference selected by `embedding_reference`. Generic `timm` and `ref-timm` groups require encoder mappings (`timm_encoders` in JSON or `--timm-encoders NAME=MODEL` on the CLI). Encoder outputs are flattened before writing to CSV.

The `siglip` group writes three scalar features, rather than an embedding vector:

- `content_fidelity`: cosine similarity between SR and bicubic-upscaled LR embeddings;
- `perceptual_enhancement`: cosine similarity between SR and original LR embeddings;
- `final_rr_score`: `alpha * content_fidelity + (1 - alpha) * perceptual_enhancement`.

These are the implementation's feature names; they do not independently establish perceptual improvement. The `gaussian` and `uniform` groups provide five noise columns each, generated with seed 42 for control experiments.

Artifact statistics are computed separately from existing masks by [`qualisr/statistics.py`](../qualisr/statistics.py). With the default thresholds, `area00` counts values greater than zero; `area05` and `area075` count values at least 0.5 and 0.75. Each count is divided by the number of mask elements. Artifact detection itself is outside this repository.

## Adding external features

Prepare a CSV with one unique `sample_id` per sample and numeric feature columns. IDs must match the dataset loader exactly; all samples in each participating dataset need coverage, with compatible feature columns across datasets. Preserve IDs from a unified extraction output instead of deriving them from row order.

In `regressors.config.features`, register the file and enable its group alongside the existing groups:

```json
{
  "include": ["nr", "custom"],
  "feature_files": {
    "nr": "{features_root}/nr.csv",
    "custom": "{features_root}/custom.csv"
  }
}
```

Use distinct feature names across files. The loader joins by `sample_id`, drops recognized metadata columns, and rejects duplicate or missing sample IDs. Missing numeric values follow the configured imputation policy. Precomputed PCA features must be fitted on the appropriate training partition; see the [experiment guide](../configs/experiments/README.md#pca-and-cross-validation).

## References

| Component | Paper or implementation |
|---|---|
| Q-Align | [Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels](https://arxiv.org/abs/2312.17090) |
| MUSIQ | [MUSIQ: Multi-scale Image Quality Transformer](https://arxiv.org/abs/2108.05997) |
| ARNIQA | [ARNIQA: Learning Distortion Manifold for Image Quality Assessment](https://arxiv.org/abs/2310.14918) |
| UNIQUE | [Uncertainty-Aware Blind Image Quality Assessment in the Laboratory and Wild](https://arxiv.org/abs/2005.13983) |
| PaQ-2-PiQ | [From Patches to Pictures (PaQ-2-PiQ): Mapping the Perceptual Space of Picture Quality](https://arxiv.org/abs/1912.10088) |
| LPIPS | [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924) |
| STLPIPS | [Shift-tolerant Perceptual Similarity Metric](https://arxiv.org/abs/2207.13686) |
| PieAPP | [PieAPP: Perceptual Image-Error Assessment through Pairwise Preference](https://arxiv.org/abs/1806.02067) |
| AHIQ | [Attentions Help CNNs See Better: Attention-based Hybrid Image Quality Assessment Network](https://arxiv.org/abs/2204.10485) |
| SSIM | [Image Quality Assessment: From Error Visibility to Structural Similarity](https://ece.uwaterloo.ca/~z70wang/publications/ssim.html) |
| VGG | [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) |
| ResNet | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) |
| SigLIP | [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) |
| timm | [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models) |
| SPAN | [Swift Parameter-free Attention Network for Efficient Super-Resolution](https://arxiv.org/abs/2311.12770) |
| RLFN | [Residual Local Feature Network for Efficient Super-Resolution](https://arxiv.org/abs/2205.07514) |
| Artifact masks | [Prominence-Aware Artifact Detection and Dataset for Image Super-Resolution](https://arxiv.org/abs/2510.16752) |
