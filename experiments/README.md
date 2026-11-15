# Experiments reported in the QualiSR-Lab paper

Use the commands below from the repository root with Python 3.12 or 3.13.
In [`configs/paper.json`](../configs/paper.json), `datasets` sets the training
and evaluation populations. The `evaluation` section sets the fold counts,
split and model seeds, regressors, thread count and PCA dimensions used by
the grouped runner. The eight-candidate tuning grids and named feature pools
are in [`presets.py`](presets.py). Supply a fresh directory for each run. In the examples,
`/path/to/results` denotes one common, already-created parent directory.

Related operations are subcommands of the same module. For example,
`python -m experiments.ref_selection --help` lists the three Ref analyses.

| Module under `experiments` | Subcommands |
| --- | --- |
| `feature_diagnostics` | `inputs`, `fr-methods`, `wasd-overlap` |
| `ref_selection` | `compare`, `geometry`, `sensitivity` |
| `transfer` | `matrix`, `fit`, `summarize` |
| `musiq_backends` | `fit`, `summarize`, `plot` |
| `profile_cost` | `cached`, `wasd` |
| `backend_cost` | `prepare`, `verify`, `summarize` |

## Prepare the inputs

```bash
python -m pip install -c requirements.txt -e '.[regressors]'
export QUALISR_DATA_ROOT=/absolute/path/to/datasets/iqa
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
DATASET_DIR="$QUALISR_DATA_ROOT/QualiSR-Set120" bash reproduce_pipeline.sh download
mkdir "$QUALISR_DATA_ROOT/QualiSR-Set120/features"
# Copy into a new feature directory. Preserve any user-computed files elsewhere.
cp features/{nr,fr,stats,siglip,vgg,ref_vgg,resnet,ref_resnet}.csv "$QUALISR_DATA_ROOT/QualiSR-Set120/features/"
```

The checkout supplies all eight Set120 inputs. Prepare the other three
datasets using the [snapshot and parser layout table](../dataset/readme.md#paper-dataset-snapshots-and-parser-inputs).
The grouped entry point reads each dataset with its parser, hashes the source
images, and joins the features by `sample_id`. Keep the four dataset names
and ensure each cache contains one row for every parser ID. Preserve filenames,
including extensions, in the original image tree.

| File per dataset | Required contents |
| --- | --- |
| `nr.csv` | MUSIQ, ARNIQA, Q-Align, UNIQUE, PaQ2PiQ |
| `fr.csv` | PSNR, SSIM, LPIPS-VGG, STLPIPS-VGG, PieAPP, AHIQ for bicubic, RLFN and SPAN; add the six `*_gt` columns for the oracle and direct-metric study |
| `vgg.csv`, `ref_vgg.csv`, `resnet.csv`, `ref_resnet.csv` | Raw SR and bicubic-reference coordinates with matching numeric suffixes. Each fold fits its own PCA on training embeddings. |
| `stats.csv` | `min,max,mean,median,std,p05,p95,area00,area05,area075` from the SR heatmaps |
| `siglip.csv` | The three LR/SR scores for the broad feature study: `content_fidelity,perceptual_enhancement,final_rr_score` |

To compute the CSVs, install `.[features,regressors]` with the same constraints.
The supplied [extraction config](../configs/paper_features.json) enables all
eight inputs. In a copy of that config keep just the dataset entries whose
features you want to generate and set their `features_root` to fresh directories.
Supply the three pseudo-references at the actual SR output dimensions and
artifact maps for each output. The Set120 download contains both. Other
datasets require separately prepared references and maps. The unified runner
can generate bicubic/SPAN/RLFN references using its `references` section and
matching scale-specific weights. The detector runs externally.

```bash
qualisr-run-pipeline --config /path/to/paper_features.json --only-section features statistics
qualisr-run-experiments --config configs/paper.json --audit-only --output /path/to/results/input-audit
```

The grouped audit checks sample coverage, raw embedding schemas and source
identity. The broad study also checks SigLIP2 and oracle columns. The
[recorded cache hashes](inputs/paper_features.sha256) identify the original
32 inputs for comparison. User-computed caches can be used directly with the
same protocol. Keep them at the same paths throughout dependent stages.
Some extractor settings for the recorded NR/FR features are unknown and the WASD
checkpoint is private. Fresh features can produce different numerical results.
The cached tabular runs used Python 3.12.16, NumPy 2.2.6, pandas 2.3.3,
SciPy 1.18.0, scikit-learn 1.7.2, CatBoost 1.2.10 and xgboost-cpu 3.2.0.
The live CPU runs used PyIQA 0.1.15.post2, torch 2.11.0+cpu and
torchvision 0.26.0+cpu. Save the actual runtime and input manifests with
your outputs.

The wheel supplies the configs and control labels as resources. To obtain
editable copies in the same layout as these commands:

```bash
python - <<'PYTHON'
from importlib.resources import files
from pathlib import Path
from shutil import copytree
copytree(files("qualisr.configs"), Path("configs"))
copytree(files("experiments").joinpath("inputs"), Path("experiments/inputs"))
PYTHON
```

The bundled image-free example runs with `qualisr-run-regressors --no-plots`.
For grouped experiments restricted to the repository Set120 inputs use
`bash reproduce_pipeline.sh set120 --output /path/to/results/set120`.
The four-dataset experiments below report distinct internal and external
evaluation populations.

## Grouped feature-family comparisons

```bash
export QUALISR_DATA_ROOT=/path/to/datasets/iqa
qualisr-run-experiments --config configs/paper.json \
  --output /path/to/results/grouped
python -m experiments.verify_grouped \
  --run-dir /path/to/results/grouped \
  --report /path/to/results/grouped/RESULTS.md
```

The complete runner evaluates fixed and nested-tuned NR, pseudo-FR, Ref, Diff,
WASD and combined pools with all three backends. Its compact Ridge search is
confined to the three grouped inner folds. `--fold-limit 1 --split-seeds 42`
selects a small pilot; a pilot is not a replacement for the complete matrix.

The five single-split interpretation configurations retained under
[`../configs/experiments/diagnostics/`](../configs/experiments/diagnostics/)
support the importance, SHAP and noise-control illustrations. They do
not supply the primary repeated-evaluation tables. The installed image-free
example and `configs/pipeline.json` remain separate, small workflows.

## Broad feature pool, PCA and preprocessing sensitivity

```bash
python -m experiments.run_feature_study \
  --config configs/paper.json --prior /path/to/results/grouped \
  --control-flags experiments/inputs/set120_controls.csv \
  --stage core --output /path/to/results/core
python -m experiments.summarize_feature_study \
  --run /path/to/results/core --output /path/to/results/core-check
python -m experiments.run_feature_study \
  --config configs/paper.json --prior /path/to/results/grouped \
  --control-flags experiments/inputs/set120_controls.csv \
  --stage pca --output /path/to/results/pca
python -m experiments.summarize_feature_study \
  --run /path/to/results/pca --output /path/to/results/pca-check
```

The core stage tests All RR, family removals, MinMax/StandardScaler,
ResNet-50, nested tuning, the ordinary-image cohort and matched true-HR oracle
controls. The SR-only PCA stage evaluates 2, 5, 10, 25, 50, 75 and 96 components
subject to the available training rank. `--prior` supplies identical base
partitions and audited input hashes; it does not supply fitted preprocessing
to another training fold. `feature_diagnostics inputs`, `feature_diagnostics fr-methods` and
`feature_diagnostics wasd-overlap` provide source, reference and detector diagnostics.

After the grouped run, obtain the quality-oriented direct FR results, including
the matched true-HR/pseudo-reference comparison, and per-method diagnostics:

```bash
python -m experiments.feature_diagnostics inputs --config configs/paper.json \
  --prior /path/to/results/grouped --output /path/to/results/input-diagnostics
python -m experiments.feature_diagnostics fr-methods --prior /path/to/results/grouped \
  --output /path/to/results/fr_by_method.csv
```

## Compact-subset stability across regressors

`compact_stability` extends the original nested Ridge experiment to the three
fixed tree backends, preserving the 12-candidate pool without Q-Align and the
15 original outer / three inner source-grouped partitions. It reads the frozen
training-only transforms from a verified grouped run and checks their exact fit
IDs, sample targets and all original feature hashes before preparing small
matrices. It never writes to the parent run.

```bash
python -m experiments.compact_stability prepare --config configs/paper.json \
  --parent /path/to/verified/grouped --output /path/to/new/prepared
for backend in ridge randomforest xgb catboost; do
  python -m experiments.compact_stability select --prepared /path/to/new/prepared \
    --backend "$backend" --pilot --output "/path/to/new/runs/$backend-pilot"
  python -m experiments.compact_stability select --prepared /path/to/new/prepared \
    --backend "$backend" --output "/path/to/new/runs/$backend-full"
done
python -m experiments.compact_stability summarize --prepared /path/to/new/prepared \
  --runs /path/to/new/runs --output /path/to/new/summary
```

The pilot is the first original fold; the full stage covers the remaining 14.
The pilot and full output names in this example are required by the summarizer.
Complete forward/backward paths use mean three-inner-fold MOS-macro SRCC and
the original deterministic tie rules. Every unique trial is saved; path winners
and final choices are independently reconstructed. Ridge must reproduce its
original histories, subsets and predictions. Outputs compare family counts,
feature inclusion, and Jaccard overlap, plus held-out performance with the
backend's own subset, all 12 inputs, and the original Ridge subset. RealSRQ BT
correlations remain separate from MOS. Across-fold PCA coordinates can rotate;
family stability is therefore primary across folds. Overlap and size-matched
random-overlap expectations are descriptive, with no independence or p-value
claim for the overlapping fold pairs.

## Source transfer, training composition and robustness

```bash
python -m experiments.transfer matrix --config configs/paper.json \
  --control-flags experiments/inputs/set120_controls.csv \
  --output /path/to/results/backends
```

All three tabular backends use the same prepared partitions and a three-candidate
inner search. The five-NR pool (`qalign`) supplies full-source transfer and the
matched Set120/WASD factorial. The four-NR control (`historical` in the result
tables) supplies the paired cohort and model-seed checks.
MOS macro correlations and within-scene/scale Bradley–Terry results are kept
separate. `transfer summarize` verifies partitions, coverage and coefficients.

## Set120-only NR fine-tuning and transfer

```bash
python -m experiments.finetune_musiq --config configs/paper.json \
  --checkpoint /path/to/musiq_koniq_ckpt-e95806b9.pth \
  --reference-samples /path/to/results/core/samples.csv \
  --device cpu --output /path/to/results/musiq
python -m experiments.summarize_musiq /path/to/results/musiq
python -m experiments.musiq_backends fit --config configs/paper.json \
  --parent /path/to/results/musiq \
  --output /path/to/results/backends/musiq-full
python -m experiments.musiq_backends summarize \
  --root /path/to/results/backends/musiq-full \
  --parent /path/to/results/musiq \
  --output /path/to/results/musiq-analysis
```

The first runner fits MUSIQ's final MLP, associated normalizations and MOS head
using grouped training-only epoch selection. The second adds the three fixed
25-feature tabular controls on exactly the same partitions and neural inputs.
Input and output distribution diagnostics do not isolate causes of transfer
differences. Every neural checkpoint and its selected epoch is recorded.

ARNIQA, UNIQUE and PaQ2PiQ use the same Set120-only
outer/inner assignments, seeds and epoch budget as MUSIQ. Their encoders stay
frozen during prediction-head adaptation.

| Metric | Adapted parameters | Native score used in the MOS loss |
| --- | --- | --- |
| ARNIQA | KonIQ linear regressor | Already scaled to 0–1 by PyIQA |
| UNIQUE | Mean row of the bilinear quality head; uncertainty row fixed | Unbounded native mean |
| PaQ2PiQ | Existing quality MLP and BatchNorm affine parameters; running statistics fixed, dropout disabled | Score divided by 100 |
| Q-Align | Final language RMSNorm and five quality-token output rows | `(score - 1) / 4` |

```bash
python -m experiments.finetune_nr --config configs/paper.json \
  --metric arniqa --device cpu --parent /path/to/results/musiq \
  --checkpoint-manifest experiments/inputs/nr_checkpoints.sha256 \
  --output /path/to/results/arniqa
python -m experiments.summarize_nr --root /path/to/results/arniqa \
  --parent /path/to/results/musiq \
  --tabular-parent /path/to/results/backends/musiq-full
```

Repeat both commands for `unique` and `paq2piq` with a distinct output per metric.
Supply the recorded pretrained files beneath `TORCH_HOME`; paths in the
manifest are relative to that directory. Set `HF_HOME` for downloaded model
snapshots and `HF_HUB_OFFLINE=1` for offline execution. The Q-Align adapter
is an optional extension. The paper reports frozen Q-Align and four tuned
metrics; it has no completed tuned Q-Align result.
Q-Align uses FP32 extraction and adaptation; CUDA requires one free GPU and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing Torch. The current model
occupies about 28 GB in FP32 before intermediate tensors.

Head adaptation uses AdamW, learning rate 1e-5, weight decay 0.01, batch size
eight and Huber beta 0.1, with at most ten epochs and patience three. Mean
SRCC over the three training-only inner folds selects the earliest best epoch.
No external labels set the scale, parameters or epoch. Only Set120 features
are cached; external images stream once through all fitted heads. Each image's
original and reconstructed head scores must agree before it is evaluated.

Outputs include exact partitions, selection histories, changed-parameter and
checkpoint digests, paired predictions, per-dataset correlations, separate BT
groups, frozen-versus-tuned deltas and internal-to-external MOS gaps. MUSIQ and
all three existing tabular controls are joined on identical IDs. Cached Q-Align
has a separate common-coverage scope from live Q-Align. The metrics' adapted
parameter counts differ: this is a bounded head-adaptation comparison, not an
equal-capacity or end-to-end fine-tuning comparison. The head-adaptation
contrasts are exploratory. The confirmatory analysis covers the 12 tests below.

## Conditional intervals and paired tests

```bash
python -m experiments.analyze_statistics \
  --base /path/to/results/grouped \
  --core /path/to/results/core --pca /path/to/results/pca \
  --core-verification /path/to/results/core-check/verification.json \
  --pca-verification /path/to/results/pca-check/verification.json \
  --output /path/to/results/feature-statistics
python -m experiments.analyze_backends \
  --root /path/to/results/backends \
  --previous-primary /path/to/results/feature-statistics/primary_tests.csv \
  --output /path/to/results/statistics
```

For the paired BT rows whose defined groups differ, use the same outputs:

```bash
python -m experiments.analyze_bt_pairs --root /path/to/results \
  --grouped /path/to/results/grouped \
  --skipped /path/to/results/feature-statistics/undefined_bt_pairs.csv \
  --output /path/to/results/bt-common
```

The feature analysis computes the three feature-family contrasts for both PLCC
and SRCC, plus the core/PCA/direct-metric intervals. The backend analysis combines
the six raw p-values with the all-backend training-composition and MUSIQ-gap
tests and applies Holm correction to the **12-test family**.
Final analysis uses 10,000 source-cluster draws, seed 20261106,
and averaged saved predictions. Fold SD and conditional bootstrap intervals
have different estimands. `analyze_bt_pairs` restricts paired BT comparisons
to jointly defined groups; `verify_results` independently checks saved metrics.

## CPU latency and quality

`profile_compact` and `profile_all_features` measure batch-one live extractors
and separate transformation/prediction stages for all three heads. Pass
`--backends randomforest xgb catboost` explicitly. Their `--help` documents
required checkpoints, generator sources, model bundles and input manifests.
`profile_cost cached` measures cached pipeline stages and `profile_cost wasd` times the detector.
`backend_cost summarize` checks stage sums, OOF quality and scene intervals;
`plot_backend_frontier` draws the combined frontier. Quality uses the archived
feature caches; live coarse WASD timing does not reproduce the stored maps.

Run the following from the source checkout after completing `grouped` and
`core`. Set `WASD_REPO` to the compatible detector implementation with
`calculate_subjective_masks.py` and `configs/heatmaps-iqa-datasets.yaml`,
`WASD_CHECKPOINT` to its checkpoint and `SIGLIP_DIR` to the local
`google/siglip2-base-patch16-256` snapshot
`3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab`. Install the detector's
runtime and cache the extractor weights before running. A different detector
or machine gives a new timing measurement.

```bash
out=/path/to/results
python -m experiments.backend_cost prepare --grouped "$out/grouped" --output "$out/prior_nr.csv"
python -m experiments.profile_compact --config configs/paper.json --run "$out/core" \
  --wasd-repo "$WASD_REPO" --checkpoint "$WASD_CHECKPOINT" \
  --backends randomforest xgb catboost --output "$out/compact-full"
extended=(--config configs/paper.json --core "$out/core" --inputs "$out/compact-full/inputs.csv"
          --generators realtime_sr --siglip "$SIGLIP_DIR" --wasd-repo "$WASD_REPO"
          --checkpoint "$WASD_CHECKPOINT" --prior-nr "$out/prior_nr.csv" --prior "$out/grouped"
          --backends randomforest xgb catboost)
python -m experiments.profile_all_features "${extended[@]}" --pilot --output "$out/extended-pilot"
python -m experiments.profile_all_features "${extended[@]}" \
  --model-bundle "$out/extended-pilot/models.joblib" --output "$out/extended-full"
python -m experiments.backend_cost summarize --compact "$out/compact-full" --core "$out/core" \
  --extended-pilot "$out/extended-pilot" --extended "$out/extended-full" \
  --prior "$out/grouped" --prior-nr "$out/prior_nr.csv" --output "$out/cost-analysis"
```

### Ref gain: within-source selection and matched cost

```bash
python -m experiments.ref_selection compare \
  --prior /path/to/results/grouped --samples /path/to/results/core/samples.csv \
  --output /path/to/new/ref-selection
python -m experiments.profile_ref_gain \
  --prior /path/to/results/grouped --core /path/to/results/core \
  --output /path/to/new/ref-latency
```

The first analysis joins fixed five-NR and five-NR + bicubic Ref predictions
on identical outputs. It reports source-equal pair accuracy on Set120 and the
multi-output source/scale groups in DISRQAD, with paired source-bootstrap
intervals. Exact MOS ties are excluded and prediction ties earn half credit.
The primary accuracy averages repeated splits; a separate mean-forecast
analysis uses the same mean predictions as the conditional pooled correlations.
Pooled per-fold means/SD and conditional PLCC/SRCC remain separate by dataset.
These within-source comparisons are exploratory.

The second runner loads the saved seed-42/fold-1 heads and preprocessing,
verifies cached prediction parity, and measures the exact two feature pools
on two SR outputs at low/median/high source resolution in each MOS dataset.
It uses offline cached weights, CPU FP32 (including Q-Align), two threads and
three repetitions after one excluded complete warm-up. Run `--pilot
--repetitions 1` first. Both runners require fresh output paths.

The paired latency increment reuses common NR extraction. Standalone Ref
generation and generation amortized over two outputs are reported separately.
Live/cache feature differences are recorded; timing does not establish exact
regeneration of the reported quality results. Input manifests, saved-head/cache
parity, raw stage times, source-reuse parity and median/IQR summaries accompany
the run. No training, checkpoint download or global installation is performed.

For the geometry-matched rows of the within-source table, follow the unrestricted
selection run with these commands:

```bash
python -m experiments.ref_selection geometry --core /path/to/results/core --output /path/to/new/geometry
python -m experiments.ref_selection sensitivity --selection /path/to/new/ref-selection \
  --geometry /path/to/new/geometry --output /path/to/new/geometry-sensitivity
```

## Published comparison values

The [artifact index](artifacts.json) identifies the [reference summaries](results/)
shipped with this release. It covers the grouped feature fits and PCA sweep,
final 12 primary tests, training composition and robustness, NR adaptation,
paired Ref accuracy and latency, compact stability and the CPU frontier.
The files contain aggregate reference values. Each new run writes its own
predictions, fit IDs and input hashes. Grouped selection and feature
preprocessing belong to each training partition. Compare fold mean/SD with
fold estimates, and conditional intervals with statistics on averaged
predictions. Score RealSRQ within comparison groups before averaging.

`reproduce_backends --help` is available as
`python -m experiments.reproduce_backends --help` for a five-NR fold replay.
The optional dedicated-method resource audit uses
`python -m experiments.audit_sriqa_resources --help`. The paper reports
resource and exposure findings for those methods.
