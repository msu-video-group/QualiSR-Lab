# Regression experiment suite

This directory contains regression configurations for feature ablations, focused feature-impact studies, reference comparisons, embedding representations, feature combinations, dataset transfer, mixed-dataset training, and grouped cross-validation. Every configuration enables only the regressor stage and consumes precomputed feature files; it does not regenerate references, embeddings, PCA outputs, differences, or artifact statistics.

## Directory layout

```text
configs/experiments/
├── features/        # Individual feature families and PCA dimensions
├── fr_references/   # FR-reference comparisons
├── embeddings/      # SR, pseudo-reference, and difference representations
├── combinations/    # Mixed feature families and seeded noise controls
├── feature_impact/  # Paired SigLIP and Q-Align impact studies
├── reference_transfer/ # Standalone FR-reference and single-dataset transfer runs
├── nr_combinations/ # NR paired with complementary reduced-reference families
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
    ├── combinations/all_rr_features@pca5/
    ├── feature_impact/qalign_with_baseline@pca5/
    ├── reference_transfer/fr_all_pseudo_references@pca5/
    ├── nr_combinations/nr_all_complementary_features@pca5/
    └── datasets/cv_all@pca5/
```

## Data setup

The suite expects these datasets and their precomputed feature directories:

- [QualiSR-Set120](https://huggingface.co/datasets/onryabinin/QualiSR-Set120)
- [DISRQAD](https://huggingface.co/datasets/visualprior/DISRQAD)
- [ISRGen-QA](https://github.com/Lighting-YXLI/ISRGen-QA)
- [RealSRQ](https://github.com/Zhentao-Liu/RealSRQ-KLTSRQA)

Before running the suite on another machine, update each dataset entry's `root` and `features_root` to the corresponding local paths. All participating datasets must expose compatible feature columns and complete `sample_id` coverage; keep configured dataset names unchanged when reusing CSVs because they determine ID prefixes.

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

The batch stops at the first failing configuration.

## Common protocol

Unless an experiment explicitly varies a setting, configurations use:

- model seed 42 and split seed 42;
- median imputation fitted on training data only, before feature scaling;
- `MinMaxScaler` fitted on training data only;
- Random Forest, XGBoost, and CatBoost with baseline parameters;
- Q-Align excluded from regressor inputs except in the broad-pool and dedicated Q-Align impact runs;
- grouping by source/GT identity for internal splits;
- PCA dimension 5 and RLFN as the baseline FR reference;
- all supported regression analyses and plots;
- regressor profiling enabled, saving train/predict runtime and estimated prediction FLOPs under each run's `profiling/` directory.

Non-dataset feature, reference, and embedding experiments train on QualiSR-Set120 and DISRQAD with grouped 20% internal validation splits. ISRGen-QA and RealSRQ are validation-only. The split uses `split_seed`, independently of the model `seed`, and all samples sharing one source/GT image remain on the same side.

Combined metrics macro-average dataset-level correlations. MOS datasets are correlated over all of their validation samples. RealSRQ is marked as Bradley–Terry data, so correlations are computed within each source-image/scale comparison group and then averaged; per-group values are also saved.

Each run keeps the combined outputs at its root and writes validation-only metrics and analyses under `per_dataset/<dataset>/`. Per-dataset feature importances use permutation importance on that dataset's validation samples. Feature correlations, SHAP, outliers, feature metrics, feature selection, predictions, and plots use the corresponding validation subset.

The root of every experiment output also contains `log.txt`, which duplicates the regressor stage's standard output, warnings, and standard error without suppressing terminal output.

Optional analyses and plots are independent. If one fails, the command prints a warning containing the part name and exception, then continues with the remaining analyses and plots. Model fitting, prediction, dataset splitting, and primary result-table generation remain strict because failures there invalidate the experiment.

Missing or infinite feature values are handled by `regressors.config.imputation`:

```json
{"imputation": {"enabled": true, "strategy": "median"}}
```

The imputer is fitted separately on each training split (and each cross-validation fold), then applied to its validation samples. A run fails explicitly if a feature has no finite training value from which to compute the configured statistic.

## Focused SigLIP and Q-Align impact

The five configurations in `feature_impact/` share one control: NR metrics without Q-Align, RLFN-based FR metrics, VGG/ResNet PCA-5 features, and the compact artifact-statistic set `{max, median, std, area00}`. All five use the common grouped training and validation protocol, models, seeds, analyses, plots, and profiling settings.

| Config | Regressor inputs | Purpose |
|---|---|---|
| `baseline.json` | Shared compact-statistics baseline | Provide the matched control for both additions. |
| `siglip_only.json` | SigLIP content fidelity, perceptual enhancement, and final RR score | Measure SigLIP as an individual feature family. |
| `siglip_with_baseline.json` | Shared baseline plus all three SigLIP outputs | Measure the incremental effect of adding SigLIP. |
| `qalign_only.json` | Q-Align only | Measure the standalone Q-Align signal. |
| `qalign_with_baseline.json` | Shared baseline plus Q-Align | Measure the incremental effect of Q-Align and expose its importance relative to a heterogeneous feature pool. |

The Q-Align runs explicitly include Q-Align in metric comparisons and feature-selection candidates. `qalign_with_baseline.json` keeps feature-importance, combined-importance, and SHAP outputs enabled. Inspect `per_dataset/<dataset>/importances/importance_<model>.csv` and the corresponding plots for validation-set permutation importance; root-level importance plots use the models' native importance when available. Compare importances separately for each model and validation dataset; feature dominance is an experimental result, not a property assumed by the configuration.

## Standalone references and single-dataset transfer

The seven configurations in `reference_transfer/` cover two related evaluation groups. The four FR-only runs use the default grouped training on QualiSR-Set120 and DISRQAD, their 20% validation splits, and external validation on ISRGen-QA and RealSRQ. The existing RLFN-only counterpart remains `features/fr.json`.

| Config | Regressor inputs | Purpose |
|---|---|---|
| `fr_bicubic.json` | Six FR metrics computed against bicubic pseudo-references | Evaluate bicubic-based FR metrics as a standalone family. |
| `fr_span.json` | Six FR metrics computed against SPAN pseudo-references | Evaluate SPAN-based FR metrics as a standalone family. |
| `fr_gt_oracle.json` | Six FR metrics computed against true GT images | Provide a full-reference oracle; do not rank it as a reduced-reference method. |
| `fr_all_pseudo_references.json` | FR metrics for RLFN, SPAN, and bicubic together | Test whether the three pseudo-reference views complement one another in one model. |

The three transfer runs copy the existing `datasets/train_*.json` protocol: train on one complete MOS dataset with no internal split, validate on the other two MOS datasets, and use the baseline NR (without Q-Align), RLFN-FR, VGG PCA-5, and ResNet PCA-5 features. They additionally validate on RealSRQ. RealSRQ remains validation-only, and its Bradley–Terry correlations are computed within each source-image/scale group before averaging.

| Config | Training dataset | Validation datasets |
|---|---|---|
| `train_qualisr_set120_with_realsrq.json` | QualiSR-Set120 | DISRQAD, ISRGen-QA, RealSRQ |
| `train_dsr_dataset_with_realsrq.json` | DISRQAD | QualiSR-Set120, ISRGen-QA, RealSRQ |
| `train_isrgen_qa_with_realsrq.json` | ISRGen-QA | QualiSR-Set120, DISRQAD, RealSRQ |

## NR-centered complementary families

The five configurations in `nr_combinations/` isolate what each reduced-reference family contributes when paired with the four principal NR metrics (MUSIQ, ARNIQA, UNIQUE, and PaQ-2-PiQ). Q-Align remains excluded. All runs use the common grouped training and validation protocol and PCA dimension 5. Artifact statistics use the compact `{max, median, std, area00}` set. FR metrics use RLFN, while the precomputed pseudo-GT embeddings and SR–reference differences use the pipeline's baseline bicubic pseudo-reference.

| Config | Regressor inputs | Purpose |
|---|---|---|
| `nr_rlfn_fr.json` | NR + six RLFN-based FR metrics | Measure the complementarity of scalar NR and pseudo-reference FR cues. |
| `nr_artifact_statistics.json` | NR + four artifact statistics | Measure the contribution of compact spatial-artifact summaries. |
| `nr_pseudo_gt_embeddings.json` | NR + bicubic pseudo-GT VGG/ResNet PCA-5 embeddings | Measure whether reference representations complement NR metrics. |
| `nr_embedding_differences.json` | NR + VGG/ResNet PCA-5 SR–reference differences | Measure whether representation differences complement NR metrics. |
| `nr_all_complementary_features.json` | NR + RLFN-FR + pseudo-GT embeddings + embedding differences + four artifact statistics | Evaluate all four complementary families together without raw SR embeddings. |

The tree regressors consume every configured input column. Forward/backward Ridge selection uses all scalar candidates in the two scalar runs and a prespecified cross-family candidate pool in the embedding runs, keeping the exploratory selection report tractable without changing the fitted tree-regressor inputs.

## General feature combinations

The ten configurations in `combinations/` all use the default grouped 20% validation splits from QualiSR-Set120 and DISRQAD, with ISRGen-QA and RealSRQ entirely held out for validation. They keep the same models, seeds, scaling, imputation, correlation aggregation, and profiling settings. Feature inputs and corresponding diagnostics are the main differences.

`all_rr_features.json` loads every feature family already mapped in this suite at PCA=5: all NR metrics (including Q-Align), FR metrics against bicubic/RLFN/SPAN pseudo-references, SigLIP outputs, SR and pseudo-reference VGG/ResNet embeddings, their differences, and all artifact-mask statistics. `all_supported_oracle.json` adds GT-based FR columns to that pool. It is a full-reference oracle comparison, not a reduced-reference candidate, and its correlations must be reported separately. Uncompressed high-dimensional embeddings are outside this suite's PCA-5 baseline. GT-based direct metric comparators may still appear in RR correlation reports but are not RR regressor inputs.

| Config | Regressor inputs | Question |
|---|---|---|
| `all_rr_features.json` | All mapped reduced-reference feature families and all ten artifact statistics | How does the broad feature pool perform? |
| `all_supported_oracle.json` | Broad feature pool plus GT-based FR metrics (oracle) | How does the full-reference oracle compare? |
| `pruned_rr_features.json` | NR (without Q-Align), RLFN-FR, SR and difference embeddings, four artifact statistics | Does a fixed smaller pool compete with the broad one? |
| `nr_fr_artifacts.json` | NR, RLFN-FR, four artifact statistics | Are scalar quality cues sufficient without embeddings? |
| `fr_sr_artifacts.json` | RLFN-FR, SR VGG/ResNet PCA-5, four artifact statistics | Do FR and SR-representation cues complement each other without NR metrics? |
| `nr_fr_reference_difference.json` | NR, RLFN-FR, pseudo-reference and difference embeddings, four artifact statistics | Can reference-relative representations replace raw SR embeddings? |
| `nr_fr_siglip_artifacts.json` | NR, RLFN-FR, SigLIP outputs, four artifact statistics | Does SigLIP add value to compact scalar cues? |
| `minimal_three_signals.json` | LPIPS-VGG+RLFN, VGG SR–reference difference component 0, artifact `area00`; no NR metrics | How far can one signal from each of three RR feature families go? |
| `noise_only.json` | Five Gaussian and five uniform noise columns | What performance can an all-noise input pool appear to achieve? |
| `noise_with_baseline.json` | Baseline NR, RLFN-FR, SR VGG/ResNet PCA-5, four artifact statistics, plus ten noise columns | Which real features exceed the validation-set noise floor? |

The pruned set is explicitly specified in the configuration. The tree regressors use every input column configured for each run. Ridge forward/backward selection analyzes a predefined, cross-family candidate list (all columns for the minimal and noise-only sets), keeping the search size manageable. Selection and comparisons on these validation sets are exploratory; any newly chosen configuration needs an independent evaluation before being reported as a general best combination. The oracle run cannot be included in a reduced-reference ranking.

For each model and validation dataset, `noise_with_baseline` saves `per_dataset/<dataset>/importances/noise_floor_<model>.csv`. The floor is the largest absolute permutation importance among the ten noise columns, using the regressor's default R² scoring. Compare absolute magnitudes in `importance_<model>.csv` against this exploratory threshold. It is not a significance test or transferable cutoff. Interpret it on MOS datasets: RealSRQ scores are comparable only within source-image/scale groups, whereas permutation importance uses whole-dataset R². `noise_only` checks for apparent all-noise correlations; its importances use a different model baseline.

## Dataset experiments

The `datasets/` configurations use the baseline NR (without Q-Align), RLFN-FR, and VGG/ResNet PCA-5 features:

| Config pattern | Training and validation |
|---|---|
| `train_*.json` | Train on one complete MOS dataset; validate on the other two. |
| `mix_*.json` | Train on the named complete datasets; validate on remaining MOS datasets and RealSRQ. `mix_all` validates only on RealSRQ. |
| `qualisr_impact_*.json` | Compare training with and without QualiSR-Set120, with and without compact artifact statistics. Hold out 20% of each training dataset; validate on the remaining datasets. |
| `cv_*.json` | Five-fold grouped cross-validation on the named dataset(s); RealSRQ is excluded. |

## Individual features and representations

- `features/nr.json` and `features/fr.json` isolate NR metrics without Q-Align and RLFN-based FR metrics, respectively.
- `features/vgg_pca_*.json` and `features/resnet_pca_*.json` vary PCA dimensions over 5, 10, 25, 50, and 75.
- `features/stats_*.json` compare artifact-statistic subsets; the exact columns are listed in each configuration's `stats_columns`.
- `fr_references/` changes the FR reference within the baseline feature set. `fr_hr.json` uses GT and is a full-reference comparison.
- `embeddings/` isolates SR embeddings, bicubic-reference embeddings, or signed SR–reference differences.

## PCA and cross-validation

PCA is precomputed: neither the regression runner nor its cross-validation loop refits PCA. The unified PCA stage also processes datasets separately. For an inductive evaluation, fit PCA only on the training partition and apply that same basis to held-out and external data. Reusing CSVs fitted on validation images, or fitting a different basis for each dataset, does not meet that protocol. Fold-specific PCA preparation and orchestration are required for a complete PCA-plus-regressor cross-validation experiment.

Cross-validation is controlled by `regressors.config.cross_validation` and is disabled by default:

```json
{"cross_validation": {"enabled": false, "n_splits": 5}}
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

The root pipeline configuration disables reference-embedding extraction, shared PCA, and embedding differences by default; enable those stages when preparing these files. Reference-embedding experiments expect the baseline bicubic pseudo-reference. FR experiments use RLFN unless the configuration is one of the explicit FR-reference comparisons.
