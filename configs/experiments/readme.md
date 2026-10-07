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
- Q-Align excluded from regressor inputs except in the broad-pool, dedicated Q-Align impact runs and NR complementarity experiments;
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

## Experimental categories

The categories below follow the directory layout. Configuration names and patterns are relative to the directory named in each heading. Every table lists the configuration, regressor inputs, training and validation protocol, and experimental purpose.

For these tables:

- **Default grouped protocol** means training on QualiSR-Set120 and DISRQAD with grouped 20% internal validation splits, plus validation-only ISRGen-QA and RealSRQ. Models, seeds, preprocessing, analyses, plots, profiling, and correlation aggregation follow the common protocol above.
- **Baseline inputs** means NR metrics without Q-Align, RLFN-based FR metrics, and SR VGG/ResNet PCA-5 embeddings.
- **Compact statistics** means the artifact-statistic set `{max, median, std, area00}`; **compact baseline** means baseline inputs plus compact statistics.

Unless a row explicitly includes Q-Align or varies the PCA dimension or reference, Q-Align is excluded, PCA dimension is 5, and FR metrics use RLFN. Pseudo-reference embeddings and SR–reference differences use the precomputed baseline bicubic pseudo-reference. GT-based FR runs are full-reference oracle comparisons and must be reported separately from reduced-reference rankings.

### Individual feature families (`features/`)

These configurations isolate one feature family or vary its representation size or artifact-statistic subset.

| Configuration | Regressor inputs | Training and validation | Purpose |
|---|---|---|---|
| `nr.json` | NR metrics without Q-Align | Default grouped protocol | Measure the standalone NR feature family. |
| `fr.json` | Six RLFN-based FR metrics | Default grouped protocol | Measure the standalone pseudo-reference FR feature family. |
| `vgg_pca_*.json` | SR VGG embeddings at PCA dimensions 5, 10, 25, 50, and 75 | Default grouped protocol | Measure the effect of VGG representation size. |
| `resnet_pca_*.json` | SR ResNet embeddings at PCA dimensions 5, 10, 25, 50, and 75 | Default grouped protocol | Measure the effect of ResNet representation size. |
| `stats_*.json` | Artifact-statistic subsets specified by each configuration's `stats_columns` | Default grouped protocol | Compare artifact-statistic subsets. |

### FR-reference comparisons (`fr_references/`)

These configurations change only the FR reference within the baseline feature set. Unlike the FR-only runs in `reference_transfer/`, they retain NR metrics and SR embeddings.

| Configuration | Regressor inputs | Training and validation | Purpose |
|---|---|---|---|
| `fr_rlfn.json` | Baseline inputs with RLFN-based FR metrics | Default grouped protocol | Provide the baseline reference comparison. |
| `fr_bicubic.json` | Baseline inputs with bicubic-based FR metrics replacing RLFN-FR | Default grouped protocol | Measure the effect of using bicubic pseudo-references. |
| `fr_span.json` | Baseline inputs with SPAN-based FR metrics replacing RLFN-FR | Default grouped protocol | Measure the effect of using SPAN pseudo-references. |
| `fr_hr.json` | Baseline inputs with GT-based FR metrics replacing RLFN-FR | Default grouped protocol | Provide a full-reference oracle comparison. |

### Embedding representations (`embeddings/`)

These configurations isolate VGG/ResNet PCA-5 representations of SR images, bicubic pseudo-references, or their signed differences.

| Configuration | Regressor inputs | Training and validation | Purpose |
|---|---|---|---|
| `sr_embeddings.json` | SR VGG/ResNet PCA-5 embeddings | Default grouped protocol | Measure the standalone SR representation signal. |
| `reference_embeddings.json` | Bicubic pseudo-reference VGG/ResNet PCA-5 embeddings | Default grouped protocol | Measure the standalone reference representation signal. |
| `embedding_difference.json` | Signed VGG/ResNet PCA-5 SR–reference differences | Default grouped protocol | Measure the standalone representation-difference signal. |

### General feature combinations (`combinations/`)

The ten configurations compare mixed feature pools and seeded noise controls under the default grouped protocol. Feature inputs and corresponding diagnostics are the main differences.

`all_rr_features.json` loads every feature family already mapped in this suite at PCA=5: all NR metrics (including Q-Align), FR metrics against bicubic/RLFN/SPAN pseudo-references, SigLIP outputs, SR and pseudo-reference VGG/ResNet embeddings, their differences, and all artifact-mask statistics. `all_supported_oracle.json` adds GT-based FR columns to that pool. It is a full-reference oracle comparison, not a reduced-reference candidate, and its correlations must be reported separately. Uncompressed high-dimensional embeddings are outside this suite's PCA-5 baseline. GT-based direct metric comparators may still appear in RR correlation reports but are not RR regressor inputs.

| Configuration | Regressor inputs | Training and validation | Purpose |
|---|---|---|---|
| `all_rr_features.json` | All mapped reduced-reference feature families and all ten artifact statistics | Default grouped protocol | Evaluate the broad reduced-reference feature pool. |
| `all_supported_oracle.json` | Broad feature pool plus GT-based FR metrics (oracle) | Default grouped protocol | Compare the broad pool with a full-reference oracle. |
| `pruned_rr_features.json` | NR (without Q-Align), RLFN-FR, SR and difference embeddings, compact statistics | Default grouped protocol | Compare a fixed smaller pool with the broad pool. |
| `nr_fr_artifacts.json` | NR, RLFN-FR, compact statistics | Default grouped protocol | Evaluate scalar quality cues without embeddings. |
| `fr_sr_artifacts.json` | RLFN-FR, SR VGG/ResNet PCA-5, compact statistics | Default grouped protocol | Measure the complementarity of FR and SR-representation cues without NR metrics. |
| `nr_fr_reference_difference.json` | NR, RLFN-FR, pseudo-reference and difference embeddings, compact statistics | Default grouped protocol | Test whether reference-relative representations can replace raw SR embeddings. |
| `nr_fr_siglip_artifacts.json` | NR, RLFN-FR, SigLIP outputs, compact statistics | Default grouped protocol | Measure the contribution of SigLIP to compact scalar cues. |
| `minimal_three_signals.json` | LPIPS-VGG+RLFN, VGG SR–reference difference component 0, artifact `area00`; no NR metrics | Default grouped protocol | Evaluate one signal from each of three reduced-reference feature families. |
| `noise_only.json` | Five Gaussian and five uniform noise columns | Default grouped protocol | Check for apparent performance from an all-noise input pool. |
| `noise_with_baseline.json` | Compact baseline plus ten noise columns | Default grouped protocol | Compare real-feature importances with a validation-set noise floor. |

The pruned set is explicitly specified in the configuration. The tree regressors use every input column configured for each run. Ridge forward/backward selection analyzes a predefined, cross-family candidate list (all columns for the minimal and noise-only sets), keeping the search size manageable. Selection and comparisons on these validation sets are exploratory; any newly chosen configuration needs an independent evaluation before being reported as a general best combination. The oracle run cannot be included in a reduced-reference ranking.

For each model and validation dataset, `noise_with_baseline` saves `per_dataset/<dataset>/importances/noise_floor_<model>.csv`. The floor is the largest absolute permutation importance among the ten noise columns, using the regressor's default R² scoring. Compare absolute magnitudes in `importance_<model>.csv` against this exploratory threshold. It is not a significance test or transferable cutoff. Interpret it on MOS datasets: RealSRQ scores are comparable only within source-image/scale groups, whereas permutation importance uses whole-dataset R². `noise_only` checks for apparent all-noise correlations; its importances use a different model baseline.

### Focused SigLIP and Q-Align impact (`feature_impact/`)

The five configurations use the compact baseline as a matched control for standalone and incremental SigLIP and Q-Align studies. All five use the default grouped protocol.

| Configuration | Regressor inputs | Training and validation | Purpose |
|---|---|---|---|
| `baseline.json` | Compact baseline | Default grouped protocol | Provide the matched control for both additions. |
| `siglip_only.json` | SigLIP content fidelity, perceptual enhancement, and final RR score | Default grouped protocol | Measure SigLIP as an individual feature family. |
| `siglip_with_baseline.json` | Compact baseline plus all three SigLIP outputs | Default grouped protocol | Measure the incremental effect of adding SigLIP. |
| `qalign_only.json` | Q-Align only | Default grouped protocol | Measure the standalone Q-Align signal. |
| `qalign_with_baseline.json` | Compact baseline plus Q-Align | Default grouped protocol | Measure the incremental effect of Q-Align and expose its importance relative to a heterogeneous feature pool. |

The Q-Align runs explicitly include Q-Align in metric comparisons and feature-selection candidates. `qalign_with_baseline.json` keeps feature-importance, combined-importance, and SHAP outputs enabled. Inspect `per_dataset/<dataset>/importances/importance_<model>.csv` and the corresponding plots for validation-set permutation importance; root-level importance plots use the models' native importance when available. Compare importances separately for each model and validation dataset; feature dominance is an experimental result, not a property assumed by the configuration.

### Standalone references and single-dataset transfer (`reference_transfer/`)

The seven configurations cover two evaluation groups: standalone FR families under the default grouped protocol, and single-dataset transfer with RealSRQ validation.

#### Standalone FR families

The four FR-only runs vary the reference or combine pseudo-references. The existing RLFN-only counterpart remains `features/fr.json`.

| Configuration | Regressor inputs | Training and validation | Purpose |
|---|---|---|---|
| `fr_bicubic.json` | Six FR metrics computed against bicubic pseudo-references | Default grouped protocol | Evaluate bicubic-based FR metrics as a standalone family. |
| `fr_span.json` | Six FR metrics computed against SPAN pseudo-references | Default grouped protocol | Evaluate SPAN-based FR metrics as a standalone family. |
| `fr_gt_oracle.json` | Six FR metrics computed against true GT images | Default grouped protocol | Provide a full-reference oracle; do not rank it as a reduced-reference method. |
| `fr_all_pseudo_references.json` | FR metrics for RLFN, SPAN, and bicubic together | Default grouped protocol | Test whether the three pseudo-reference views complement one another in one model. |

#### Single-dataset transfer

The three transfer runs copy the existing `datasets/train_*.json` protocol: train on one complete MOS dataset with no internal split, validate on the other two MOS datasets, and use baseline inputs. They additionally validate on RealSRQ. RealSRQ remains validation-only, and its Bradley–Terry correlations are computed within each source-image/scale group before averaging.

| Configuration | Regressor inputs | Training and validation | Purpose |
|---|---|---|---|
| `train_qualisr_set120_with_realsrq.json` | Baseline inputs | Train on complete QualiSR-Set120; validate on DISRQAD, ISRGen-QA, and RealSRQ | Measure transfer from QualiSR-Set120, including RealSRQ. |
| `train_dsr_dataset_with_realsrq.json` | Baseline inputs | Train on complete DISRQAD; validate on QualiSR-Set120, ISRGen-QA, and RealSRQ | Measure transfer from DISRQAD, including RealSRQ. |
| `train_isrgen_qa_with_realsrq.json` | Baseline inputs | Train on complete ISRGen-QA; validate on QualiSR-Set120, DISRQAD, and RealSRQ | Measure transfer from ISRGen-QA, including RealSRQ. |

### NR-centered complementary families (`nr_combinations/`)

The six configurations isolate what each reduced-reference family contributes when paired with the five principal NR metrics (Q-Align, MUSIQ, ARNIQA, UNIQUE, and PaQ-2-PiQ). All runs use the default grouped protocol and use PCA dimension 5, RLFN-FR metrics, bicubic pseudo-reference representations, and compact statistics where applicable.

| Configuration | Regressor inputs | Training and validation | Purpose |
|---|---|---|---|
| `nr.json` | All five NR metrics | Default grouped protocol | Fix the baseline combined NR correlations. |
| `nr_rlfn_fr.json` | NR + six RLFN-based FR metrics | Default grouped protocol | Measure the complementarity of scalar NR and pseudo-reference FR cues. |
| `nr_artifact_statistics.json` | NR + compact statistics | Default grouped protocol | Measure the contribution of compact spatial-artifact summaries. |
| `nr_pseudo_gt_embeddings.json` | NR + bicubic pseudo-reference VGG/ResNet PCA-5 embeddings | Default grouped protocol | Measure whether reference representations complement NR metrics. |
| `nr_embedding_differences.json` | NR + VGG/ResNet PCA-5 SR–reference differences | Default grouped protocol | Measure whether representation differences complement NR metrics. |
| `nr_all_complementary_features.json` | NR + RLFN-FR + pseudo-reference embeddings + embedding differences + compact statistics | Default grouped protocol | Evaluate all four complementary families together without raw SR embeddings. |

The tree regressors consume every configured input column. Forward/backward Ridge selection uses all scalar candidates in the two scalar runs and a prespecified cross-family candidate pool in the embedding runs, keeping the exploratory selection report tractable without changing the fitted tree-regressor inputs.

### Dataset experiments (`datasets/`)

These configurations vary the training data or evaluation protocol while keeping baseline inputs. The QualiSR-Set120 impact runs additionally vary whether compact statistics are included.

| Configuration | Regressor inputs | Training and validation | Purpose |
|---|---|---|---|
| `train_*.json` | Baseline inputs | Train on one complete MOS dataset with no internal split; validate on the other two MOS datasets | Measure single-dataset transfer among MOS datasets. |
| `mix_*.json` | Baseline inputs | Train on the named complete datasets; validate on remaining MOS datasets and RealSRQ. `mix_all` validates only on RealSRQ | Measure the effect of mixed-dataset training. |
| `qualisr_impact_*.json` | Baseline inputs, with or without compact statistics | Train on DISRQAD with or without QualiSR-Set120, holding out a grouped 20% of each training dataset; also validate on remaining datasets | Measure the training contribution of QualiSR-Set120, with and without compact statistics. |
| `cv_*.json` | Baseline inputs | Five-fold grouped cross-validation on the named MOS dataset(s); RealSRQ is excluded | Measure performance across grouped folds. |

## PCA and cross-validation

PCA is precomputed: neither the regression runner nor its cross-validation loop refits PCA. The unified PCA stage also processes datasets separately. For an inductive evaluation, fit PCA only on the training partition and apply that same basis to held-out and external data.

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
