# Regression experiment suite

This directory contains reproducible regression configurations for feature ablations, reference comparisons, embedding representations, dataset transfer, mixed-dataset training, and grouped cross-validation. Every configuration enables only the regressor stage and consumes precomputed feature files; it does not regenerate references, embeddings, PCA outputs, differences, or artifact statistics.

## Directory layout

```text
configs/experiments/
├── features/        # Individual feature families and PCA dimensions
├── fr_references/   # FR-reference comparisons
├── embeddings/      # SR, pseudo-reference, and difference representations
├── datasets/        # Cross-dataset, mixture, and grouped-CV experiments
└── README.md
```

Experiment names include the category path. The suite writes below `plots/experiments/`, preserving the config categories:

```text
plots/
└── experiments/
    ├── features/vgg_pca_005@pca5/
    ├── fr_references/fr_rlfn@pca5/
    ├── embeddings/embedding_difference@pca5/
    └── datasets/cv_all@pca5/
```

## Data setup

The suite expects these datasets and their precomputed feature directories:

- QualiSR-Set120
- dsr-dataset
- ISRGen-QA
- RealSRQ, used only for external validation

Before running the suite on another machine, update each dataset entry's `root` and `features_root` to the corresponding local paths. All participating datasets must expose compatible feature columns and complete `sample_id` coverage.

## Running experiments

Run one configuration with either entry point:

```bash
qualisr-run-regressors --config configs/experiments/features/vgg_pca_005.json
qualisr-run-pipeline --config configs/experiments/features/vgg_pca_005.json
```

Both commands also accept a directory. JSON files are discovered recursively and executed sequentially in deterministic relative-path order:

```bash
# Run one category
qualisr-run-regressors --config configs/experiments/features

# Run the complete suite
qualisr-run-pipeline --config configs/experiments
```

The batch stops at the first failing configuration. Common options such as `--no-plots` and `--plots-root` apply to every run. `--experiment-name` is rejected for multi-config batches because it would make outputs collide; the regressor command similarly rejects explicit shared profiling output files.

## Common protocol

Unless an experiment explicitly varies a setting, configurations use:

- model seed 42 and split seed 42;
- median imputation fitted on training data only, before feature scaling;
- `MinMaxScaler` fitted on training data only;
- Random Forest, XGBoost, and CatBoost with baseline parameters;
- Q-Align excluded from regressor inputs;
- grouping by source/GT identity for internal splits;
- PCA dimension 5 and RLFN as the baseline FR reference;
- all supported regression analyses and plots;
- regressor profiling disabled.

Non-dataset feature, reference, and embedding experiments train on QualiSR-Set120 and dsr-dataset with grouped 20% internal validation splits. ISRGen-QA and RealSRQ are validation-only. The split uses `split_seed`, independently of the model `seed`, and all samples sharing one source/GT image remain on the same side.

Combined metrics macro-average dataset-level correlations. MOS datasets are correlated over all of their validation samples. RealSRQ is marked as Bradley–Terry data, so correlations are computed independently for every GT image series and then averaged; the per-GT values are also saved.

Each run keeps the combined outputs at its root and writes validation-only metrics and analyses under `per_dataset/<dataset>/`. Per-dataset feature importances use permutation importance on that dataset's validation samples. Feature correlations, SHAP, outliers, feature metrics, feature selection, predictions, and plots use the corresponding validation subset.

Optional analyses and plots are independent. If one fails, the command prints a warning containing the part name and exception, then continues with the remaining analyses and plots. Model fitting, prediction, dataset splitting, and primary result-table generation remain strict because failures there invalidate the experiment.

Missing or infinite feature values are handled by the explicitly configured preprocessing step:

```json
"imputation": {
  "enabled": true,
  "strategy": "median"
}
```

The imputer is fitted separately on each training split (and each cross-validation fold), then applied to its validation samples. A run fails explicitly if a feature has no finite training value from which to compute the configured statistic.

## Experiment matrix

| Config | Training datasets | Validation datasets | Features / representation | Varied parameter or held-out group | Purpose |
|---|---|---|---|---|---|
| `features/vgg_pca_005.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from QualiSR-Set120 and dsr-dataset; ISRGen-QA; RealSRQ | VGG SR embeddings only | PCA=5 | Measure the VGG PCA dimensionality effect. |
| `features/vgg_pca_010.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from QualiSR-Set120 and dsr-dataset; ISRGen-QA; RealSRQ | VGG SR embeddings only | PCA=10 | Measure the VGG PCA dimensionality effect. |
| `features/vgg_pca_025.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from QualiSR-Set120 and dsr-dataset; ISRGen-QA; RealSRQ | VGG SR embeddings only | PCA=25 | Measure the VGG PCA dimensionality effect. |
| `features/vgg_pca_050.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from QualiSR-Set120 and dsr-dataset; ISRGen-QA; RealSRQ | VGG SR embeddings only | PCA=50 | Measure the VGG PCA dimensionality effect. |
| `features/vgg_pca_075.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from QualiSR-Set120 and dsr-dataset; ISRGen-QA; RealSRQ | VGG SR embeddings only | PCA=75 | Measure the VGG PCA dimensionality effect. |
| `features/resnet_pca_005.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from QualiSR-Set120 and dsr-dataset; ISRGen-QA; RealSRQ | ResNet SR embeddings only | PCA=5 | Measure the ResNet PCA dimensionality effect. |
| `features/resnet_pca_010.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from QualiSR-Set120 and dsr-dataset; ISRGen-QA; RealSRQ | ResNet SR embeddings only | PCA=10 | Measure the ResNet PCA dimensionality effect. |
| `features/resnet_pca_025.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from QualiSR-Set120 and dsr-dataset; ISRGen-QA; RealSRQ | ResNet SR embeddings only | PCA=25 | Measure the ResNet PCA dimensionality effect. |
| `features/resnet_pca_050.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from QualiSR-Set120 and dsr-dataset; ISRGen-QA; RealSRQ | ResNet SR embeddings only | PCA=50 | Measure the ResNet PCA dimensionality effect. |
| `features/resnet_pca_075.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from QualiSR-Set120 and dsr-dataset; ISRGen-QA; RealSRQ | ResNet SR embeddings only | PCA=75 | Measure the ResNet PCA dimensionality effect. |
| `features/nr.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from QualiSR-Set120 and dsr-dataset; ISRGen-QA; RealSRQ | NR metrics only (Q-Align excluded) | Feature family=NR | Evaluate the baseline NR metrics without Q-Align. |
| `features/fr.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from QualiSR-Set120 and dsr-dataset; ISRGen-QA; RealSRQ | FR metrics only | Reference=RLFN | Evaluate FR metrics with the baseline reference. |
| `features/stats_all.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from QualiSR-Set120 and dsr-dataset; ISRGen-QA; RealSRQ | Artifact statistics only: min, max, mean, median, std, p05, p95, area00, area05, area075 | Subset=all | Measure the selected artifact-statistic subset. |
| `features/stats_six.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from QualiSR-Set120 and dsr-dataset; ISRGen-QA; RealSRQ | Artifact statistics only: max, mean, median, std, p95, area00 | Subset=six | Measure the selected artifact-statistic subset. |
| `features/stats_five.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from QualiSR-Set120 and dsr-dataset; ISRGen-QA; RealSRQ | Artifact statistics only: max, median, std, p95, area00 | Subset=five | Measure the selected artifact-statistic subset. |
| `features/stats_no_std.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from QualiSR-Set120 and dsr-dataset; ISRGen-QA; RealSRQ | Artifact statistics only: max, median, p95, area00 | Subset=no_std | Measure the selected artifact-statistic subset. |
| `features/stats_best4.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from QualiSR-Set120 and dsr-dataset; ISRGen-QA; RealSRQ | Artifact statistics only: max, median, std, area00 | Subset=best4 | Measure the selected artifact-statistic subset. |
| `features/stats_three.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from QualiSR-Set120 and dsr-dataset; ISRGen-QA; RealSRQ | Artifact statistics only: max, median, std | Subset=three | Measure the selected artifact-statistic subset. |
| `features/stats_no_p95.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from QualiSR-Set120 and dsr-dataset; ISRGen-QA; RealSRQ | Artifact statistics only: min, max, mean, median, std, p05, area00, area05, area075 | Subset=no_p95 | Measure the selected artifact-statistic subset. |
| `fr_references/fr_hr.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from QualiSR-Set120 and dsr-dataset; ISRGen-QA; RealSRQ | Full baseline: NR + FR + VGG PCA-5 + ResNet PCA-5 | FR reference=HR/GT | Compare one FR reference while preserving the baseline feature configuration. |
| `fr_references/fr_rlfn.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from QualiSR-Set120 and dsr-dataset; ISRGen-QA; RealSRQ | Full baseline: NR + FR + VGG PCA-5 + ResNet PCA-5 | FR reference=RLFN | Compare one FR reference while preserving the baseline feature configuration. |
| `fr_references/fr_span.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from QualiSR-Set120 and dsr-dataset; ISRGen-QA; RealSRQ | Full baseline: NR + FR + VGG PCA-5 + ResNet PCA-5 | FR reference=SPAN | Compare one FR reference while preserving the baseline feature configuration. |
| `fr_references/fr_bicubic.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from QualiSR-Set120 and dsr-dataset; ISRGen-QA; RealSRQ | Full baseline: NR + FR + VGG PCA-5 + ResNet PCA-5 | FR reference=BICUBIC | Compare one FR reference while preserving the baseline feature configuration. |
| `embeddings/sr_embeddings.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from QualiSR-Set120 and dsr-dataset; ISRGen-QA; RealSRQ | SR-image VGG + ResNet PCA-5 embeddings only | Representation=SR | Evaluate SR-image embeddings without reference embeddings or differences. |
| `embeddings/reference_embeddings.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from QualiSR-Set120 and dsr-dataset; ISRGen-QA; RealSRQ | Pseudo-reference VGG + ResNet shared-PCA-5 embeddings only | Representation=reference | Evaluate baseline-reference embeddings without SR embeddings or differences. |
| `embeddings/embedding_difference.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from QualiSR-Set120 and dsr-dataset; ISRGen-QA; RealSRQ | Precomputed SR–reference VGG + ResNet PCA-5 differences only | Representation=difference | Evaluate embedding differences without raw SR or reference embeddings. |
| `datasets/train_qualisr_set120.json` | QualiSR-Set120 | dsr-dataset; ISRGen-QA | Full baseline: NR + RLFN-FR + VGG PCA-5 + ResNet PCA-5 | Training dataset=QualiSR-Set120 | Train on one complete dataset and validate on the other two. |
| `datasets/train_dsr_dataset.json` | dsr-dataset | QualiSR-Set120; ISRGen-QA | Full baseline: NR + RLFN-FR + VGG PCA-5 + ResNet PCA-5 | Training dataset=dsr-dataset | Train on one complete dataset and validate on the other two. |
| `datasets/train_isrgen_qa.json` | ISRGen-QA | QualiSR-Set120; dsr-dataset | Full baseline: NR + RLFN-FR + VGG PCA-5 + ResNet PCA-5 | Training dataset=ISRGen-QA | Train on one complete dataset and validate on the other two. |
| `datasets/mix_qualisr_dsr.json` | QualiSR-Set120; dsr-dataset | ISRGen-QA; RealSRQ | Full baseline: NR + RLFN-FR + VGG PCA-5 + ResNet PCA-5 | Mixture=QualiSR-Set120 + dsr-dataset | Train on two complete datasets and validate on the remaining dataset plus RealSRQ. |
| `datasets/mix_qualisr_isrgen.json` | QualiSR-Set120; ISRGen-QA | dsr-dataset; RealSRQ | Full baseline: NR + RLFN-FR + VGG PCA-5 + ResNet PCA-5 | Mixture=QualiSR-Set120 + ISRGen-QA | Train on two complete datasets and validate on the remaining dataset plus RealSRQ. |
| `datasets/mix_dsr_isrgen.json` | dsr-dataset; ISRGen-QA | QualiSR-Set120; RealSRQ | Full baseline: NR + RLFN-FR + VGG PCA-5 + ResNet PCA-5 | Mixture=dsr-dataset + ISRGen-QA | Train on two complete datasets and validate on the remaining dataset plus RealSRQ. |
| `datasets/mix_all.json` | QualiSR-Set120; dsr-dataset; ISRGen-QA | RealSRQ | Full baseline: NR + RLFN-FR + VGG PCA-5 + ResNet PCA-5 | Mixture=all training-capable datasets | Train on all three complete datasets and validate externally on RealSRQ. |
| `datasets/qualisr_impact_with_qualisr_without_stats.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from both training datasets; ISRGen-QA; RealSRQ | Full baseline without artifact statistics | QualiSR included | Measure the contribution of QualiSR-Set120 to baseline training. |
| `datasets/qualisr_impact_without_qualisr_without_stats.json` | dsr-dataset | Grouped 20% from dsr-dataset; QualiSR-Set120; ISRGen-QA; RealSRQ | Full baseline without artifact statistics | QualiSR excluded | Match the no-statistics impact experiment while training only on dsr-dataset. |
| `datasets/qualisr_impact_with_qualisr_with_stats.json` | QualiSR-Set120; dsr-dataset | Grouped 20% from both training datasets; ISRGen-QA; RealSRQ | Full baseline + max, median, std, area00 | QualiSR included | Measure the contribution of QualiSR-Set120 when artifact statistics are enabled. |
| `datasets/qualisr_impact_without_qualisr_with_stats.json` | dsr-dataset | Grouped 20% from dsr-dataset; QualiSR-Set120; ISRGen-QA; RealSRQ | Full baseline + max, median, std, area00 | QualiSR excluded | Match the statistics-enabled impact experiment while training only on dsr-dataset. |
| `datasets/cv_qualisr_set120.json` | QualiSR-Set120 | Grouped 5-fold CV on QualiSR-Set120 | Full baseline: NR + RLFN-FR + VGG PCA-5 + ResNet PCA-5 | CV=5 folds | Estimate within-dataset performance without source/GT-image leakage. |
| `datasets/cv_dsr_dataset.json` | dsr-dataset | Grouped 5-fold CV on dsr-dataset | Full baseline: NR + RLFN-FR + VGG PCA-5 + ResNet PCA-5 | CV=5 folds | Estimate within-dataset performance without source/GT-image leakage. |
| `datasets/cv_isrgen_qa.json` | ISRGen-QA | Grouped 5-fold CV on ISRGen-QA | Full baseline: NR + RLFN-FR + VGG PCA-5 + ResNet PCA-5 | CV=5 folds | Estimate within-dataset performance without source/GT-image leakage. |
| `datasets/cv_all.json` | QualiSR-Set120; dsr-dataset; ISRGen-QA | Grouped 5-fold CV on all three datasets | Full baseline: NR + RLFN-FR + VGG PCA-5 + ResNet PCA-5 | CV=5 folds | Estimate mixed-dataset performance without source/GT-image leakage; RealSRQ remains outside CV. |

## Grouped cross-validation

Cross-validation is controlled by the nested regressor block and is disabled by default:

```json
"cross_validation": {
  "enabled": false,
  "n_splits": 5
}
```

The four `cv_*.json` configurations enable five-fold shuffled `GroupKFold` with `split_seed` 42. Groups use the composite `dataset/test_case` identity, so all SR samples derived from one source/GT image remain in the same fold. Feature scaling and regressors are fitted independently inside every fold, and RealSRQ is not included.

Each CV run saves:

- normal analysis outputs under `fold_01/` through `fold_05/`;
- `metadata/cross_validation_folds.csv` with sample-to-fold assignments;
- concatenated out-of-fold predictions;
- per-fold correlations;
- mean and standard deviation of PLCC and SRCC across folds.

## Precomputed feature mappings

The baseline mappings are:

- `nr.csv` and `fr.csv`;
- `stats.csv`;
- `pca/vgg_pca{pca_n}.csv` and `pca/resnet_pca{pca_n}.csv`;
- reference embeddings in `pca/ref_vgg_shared_pca5.csv` and `pca/ref_resnet_shared_pca5.csv`;
- embedding differences in `vgg_diff_pca5.csv` and `resnet_diff_pca5.csv`.

Reference-embedding experiments use the baseline bicubic pseudo-reference. FR experiments use RLFN unless the configuration is one of the explicit FR-reference comparisons.
