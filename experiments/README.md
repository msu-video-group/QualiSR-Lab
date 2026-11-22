# Paper experiments

This directory contains the QualiSR-Lab paper's experiment runners and
[reference results](results/), indexed with checksums in [artifacts.json](artifacts.json).
Run the Bash examples from the repository root with Python 3.12 or 3.13.
Use a fresh output directory for every run.

## Setup and data

```bash
python -m pip install -c requirements.txt -e '.[regressors]'
export QUALISR_DATA_ROOT=/absolute/path/to/datasets/iqa
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
out=/absolute/path/to/new/results
mkdir "$out"
```

Prepare all four datasets using the [dataset guide](../dataset/readme.md#paper-dataset-snapshots-and-parser-inputs).
Keep the dataset names, image filenames and directory layout in
[`configs/paper.json`](../configs/paper.json): feature CSVs are joined by
`sample_id`, and the input audit hashes source images.

The checkout includes all eight QualiSR-Set120 feature CSVs. To download its
images and copy the features into a new directory:

```bash
DATASET_DIR="$QUALISR_DATA_ROOT/QualiSR-Set120" bash reproduce_pipeline.sh download
mkdir "$QUALISR_DATA_ROOT/QualiSR-Set120/features"
cp features/{nr,fr,stats,siglip,vgg,ref_vgg,resnet,ref_resnet}.csv "$QUALISR_DATA_ROOT/QualiSR-Set120/features/"
```

Each dataset needs one row per parser ID in these files:

| File | Contents |
| --- | --- |
| `nr.csv` | MUSIQ, ARNIQA, Q-Align, UNIQUE, PaQ2PiQ |
| `fr.csv` | PSNR, SSIM, LPIPS-VGG, STLPIPS-VGG, PieAPP, AHIQ for bicubic, RLFN and SPAN; six `*_gt` columns for oracle comparisons |
| `vgg.csv`, `ref_vgg.csv`, `resnet.csv`, `ref_resnet.csv` | Raw SR and bicubic-reference embeddings with matching numeric suffixes |
| `stats.csv` | `min,max,mean,median,std,p05,p95,area00,area05,area075` from artifact maps |
| `siglip.csv` | `content_fidelity,perceptual_enhancement,final_rr_score` for the broad feature study |

To extract new features, install `.[features,regressors]` with the same
constraints and edit a copy of [`configs/paper_features.json`](../configs/paper_features.json).
Set fresh `features_root` paths and supply pseudo-references at the actual SR
dimensions and artifact maps. The QualiSR-Set120 download includes both;
other datasets require separate preparation. Reference generation is described
in the [main README](../README.md#step-0-optional-prepare-reference-images);
the WASD detector runs externally.

```bash
qualisr-run-pipeline --config /path/to/paper_features.json --only-section features statistics
qualisr-run-experiments --config configs/paper.json --audit-only --output "$out/input-audit"
```

[Recorded hashes](inputs/paper_features.sha256) identify the original 32 feature
inputs. Keep cache paths fixed across dependent stages and save runtime/input
manifests. Some original NR/FR extractor settings are unknown and the WASD
checkpoint is private, so newly extracted features may produce different results.

For a small installation check, use `qualisr-run-regressors --no-plots`.
For grouped experiments on QualiSR-Set120 alone, use
`bash reproduce_pipeline.sh set120 --output "$out/QualiSR-Set120"`.

## Evaluation protocol

[`configs/paper.json`](../configs/paper.json) defines dataset roles and evaluation
settings. Main feature comparisons train on QualiSR-Set120 and DISRQAD, with
ISRGen-QA and RealSRQ reserved for external evaluation. Five source-grouped
outer folds use split seeds 42/43/44; three grouped inner folds select settings.
The model seed is 42. Imputation, scaling and PCA are fitted on training rows.
Feature pools and bounded tuning grids are in [`presets.py`](presets.py).

MOS correlations are computed within each dataset, then macro-averaged.
RealSRQ Bradley–Terry (BT) correlations are computed within scene/scale groups
and reported separately. Fold mean/SD and conditional intervals on averaged
predictions describe different quantities.

## Grouped feature comparisons

```bash
qualisr-run-experiments --config configs/paper.json --output "$out/grouped"
python -m experiments.verify_grouped --run-dir "$out/grouped" --report "$out/grouped/RESULTS.md"
```

The runner compares fixed and nested-tuned NR, pseudo-reference FR,
reference embeddings (Ref), signed SR-minus-reference embeddings (Diff), WASD
statistics and combined pools using Random Forest, XGBoost and CatBoost.
Compact Ridge selection uses only inner folds. Add `--fold-limit 1 --split-seeds 42`
for a pilot. The [single-split diagnostics](../configs/experiments/diagnostics/)
support importance, SHAP and noise illustrations separately from this protocol.

## Feature ablations and PCA sensitivity

```bash
for stage in core pca; do
  python -m experiments.run_feature_study --config configs/paper.json \
    --prior "$out/grouped" --control-flags experiments/inputs/set120_controls.csv \
    --stage "$stage" --output "$out/$stage"
  python -m experiments.summarize_feature_study \
    --run "$out/$stage" --output "$out/$stage-check"
done
```

`core` evaluates All RR, family removals, scaler choices, ResNet-50, nested
tuning, the ordinary-image cohort and matched true-HR oracle controls.
`pca` tests 2, 5, 10, 25, 50, 75 and 96 SR components, limited by training rank.
Add `--pca-representation ref` to repeat the sweep for the bicubic-reference
VGG and ResNet embeddings. Each basis is fitted on that reference block alone,
using the same outer training rows; use a separate output directory.
`--prior` preserves partitions and input hashes; preprocessing is fitted per fold.

For a focused comparison of paired VGG/ResNet dimensions, run:

```bash
python -m experiments.focused_pca --config configs/paper.json \
  --prior "$out/grouped" --output "$out/focused-pca"
```

This compares (5,5), (5,50), (25,50), (50,50) and (96,96) for five NR + Ref
and All RR without pseudo-FR scores, using fixed default heads for all three
backends. A separate inner-selected row chooses dimensions by three grouped
training-only folds. Shared paired PCA bases are fitted at 96 components within
each training partition and reused through prefixes. The original VGG-10 /
ResNet-50 basis supplies a separate 5/5 control. Outer models, transformations,
inner predictions and original split/input provenance are retained. `--pilot`
uses the first outer fold; `--workers` permits up to three two-thread model jobs.
`--start-fold` retains original fold numbers while skipping completed outer
folds, for scheduling remaining work with the same partitions.

Use `python -m experiments.feature_diagnostics <subcommand> --help` for
`inputs` (source/reference audit), `fr-methods` (direct FR and oracle comparisons)
and `wasd-overlap` (detector overlap audit).

## Compact-subset stability

```bash
python -m experiments.compact_stability prepare --config configs/paper.json \
  --parent "$out/grouped" --output "$out/prepared"
for backend in ridge randomforest xgb catboost; do
  python -m experiments.compact_stability select --prepared "$out/prepared" \
    --backend "$backend" --pilot --output "$out/compact-runs/$backend-pilot"
  python -m experiments.compact_stability select --prepared "$out/prepared" \
    --backend "$backend" --output "$out/compact-runs/$backend-full"
done
python -m experiments.compact_stability summarize --prepared "$out/prepared" \
  --runs "$out/compact-runs" --output "$out/compact-summary"
```

This uses the original 12 candidates without Q-Align, verified training-only
transforms and the same 15 outer/three inner partitions. The pilot covers the
first fold; the full stage covers the remaining 14. Preserve the output naming
above for the summarizer. Outputs compare selected features, families, Jaccard
overlap and held-out performance. PCA coordinates may rotate across folds;
family recurrence and size-matched random overlap are descriptive, with no
independence claim for overlapping fold pairs.

## Training composition and transfer

```bash
python -m experiments.transfer matrix --config configs/paper.json \
  --control-flags experiments/inputs/set120_controls.csv --output "$out/backends"
```

All three backends share partitions and a three-candidate inner search.
The five-NR pool (`qalign`) covers full-source transfer and the matched
QualiSR-Set120/WASD factorial; the four-NR pool (`historical`) covers cohort and
model-seed controls. The matrix command also verifies and summarizes each run.

## Feature-pool transfer comparison

The separate full-source transfer experiment compares five tuned pools (five NR,
NR + Ref, NR + Ref + Diff, All RR without pseudo-FR scores, and the legacy pool)
with an original three-setting/PCA-5 replay control. It preserves the original
transfer runner. Each tuned pool has eight joint candidates per backend, selected
only on the training datasets. Primary selection uses three grouped inner folds;
a separate sensitivity equally weights grouped and leave-one-training-dataset-out
scores. The latter equals grouped selection when only one dataset is trained.

```bash
python -m experiments.transfer_features fit --config configs/paper.json \
  --pilot --workers 3 --output "$out/transfer-features-pilot"
python -m experiments.transfer_features verify "$out/transfer-features-pilot"
python -m experiments.transfer_features fit --config configs/paper.json \
  --workers 3 --output "$out/transfer-features-full"
python -m experiments.transfer_features verify "$out/transfer-features-full"
```

The pilot covers Set120-only and Set120 + ISRGen-QA training. The full run covers
all seven MOS training combinations; RealSRQ is always evaluation-only. Keep all
fixed-pool results and both selection protocols. The verifier reconstructs every
inner score from binary forecasts, checks partition/fit IDs and candidate winners,
and replays final serialized models. Use `verify --output NEW_DIRECTORY` for a
separate local verification. Three workers use six model threads in total; select
fewer workers on shared hosts when needed. Conditional uncertainty is separate
from these one-refit-per-condition results.

## QualiSR-Set120-only NR adaptation

```bash
python -m experiments.finetune_musiq --config configs/paper.json \
  --checkpoint /path/to/musiq_koniq_ckpt-e95806b9.pth \
  --reference-samples "$out/core/samples.csv" --device cpu --output "$out/musiq"
python -m experiments.summarize_musiq "$out/musiq"
python -m experiments.musiq_backends fit --config configs/paper.json \
  --parent "$out/musiq" --output "$out/backends/musiq-full"
python -m experiments.musiq_backends summarize --root "$out/backends/musiq-full" \
  --parent "$out/musiq" --output "$out/musiq-analysis"

python -m experiments.finetune_nr --config configs/paper.json \
  --metric arniqa --device cpu --parent "$out/musiq" \
  --checkpoint-manifest experiments/inputs/nr_checkpoints.sha256 --output "$out/arniqa"
python -m experiments.summarize_nr --root "$out/arniqa" \
  --parent "$out/musiq" --tabular-parent "$out/backends/musiq-full"
```

Repeat the last two commands for `unique` and `paq2piq`, using separate outputs.
Pretrained files must match the manifest under `TORCH_HOME`; model snapshots
use `HF_HOME`, with `HF_HUB_OFFLINE=1` for offline execution.

MUSIQ adapts its final MLP, normalizations and MOS head. ARNIQA, UNIQUE and
PaQ2PiQ adapt prediction heads with frozen encoders. All use the same grouped
QualiSR-Set120 partitions and training-only epoch selection: AdamW, learning
rate 1e-5, weight decay 0.01, batch size eight, Huber beta 0.1, at most ten
epochs and patience three. The tabular controls use identical partitions and
inputs. Adaptation capacities differ, and these contrasts are exploratory.

The paper reports frozen Q-Align and four adapted metrics. The optional Q-Align
adapter has no completed paper result; it uses FP32 and requires about 28 GB
for the model alone. CUDA execution requires a free GPU and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing Torch.

## Statistical analysis

```bash
python -m experiments.analyze_statistics --base "$out/grouped" \
  --core "$out/core" --pca "$out/pca" \
  --core-verification "$out/core-check/verification.json" \
  --pca-verification "$out/pca-check/verification.json" --output "$out/feature-statistics"
python -m experiments.analyze_backends --root "$out/backends" \
  --previous-primary "$out/feature-statistics/primary_tests.csv" --output "$out/statistics"
python -m experiments.analyze_bt_pairs --root "$out" --grouped "$out/grouped" \
  --skipped "$out/feature-statistics/undefined_bt_pairs.csv" --output "$out/bt-common"
```

The analyses combine feature-family, training-composition and MUSIQ-gap
contrasts into the reported 12-test Holm family. Conditional intervals use
10,000 source-cluster draws, seed 20261106, and averaged saved predictions.
Paired BT comparisons use jointly defined groups. Use
`python -m experiments.verify_results --help` for independent metric checks.

## Within-source Ref comparisons

```bash
python -m experiments.ref_selection compare --prior "$out/grouped" \
  --samples "$out/core/samples.csv" --output "$out/ref-selection"
python -m experiments.ref_selection geometry --core "$out/core" --output "$out/geometry"
python -m experiments.ref_selection sensitivity --selection "$out/ref-selection" \
  --geometry "$out/geometry" --output "$out/geometry-sensitivity"
python -m experiments.profile_ref_gain --prior "$out/grouped" \
  --core "$out/core" --output "$out/ref-latency"
```

The selection analysis pairs five-NR and five-NR + bicubic Ref predictions on
identical outputs in QualiSR-Set120 and DISRQAD. It reports source-equal pair
accuracy with source-bootstrap intervals. Primary accuracy averages repeated
splits, excludes exact MOS ties and gives prediction ties half credit; results
from mean predictions are separate. Geometry sensitivity restricts comparisons
to identical LR inputs and output dimensions. These analyses are exploratory.

The latency runner verifies saved-head prediction parity and measures matched
CPU FP32 costs with two threads, three repetitions and an excluded warm-up.
Use `--pilot --repetitions 1` first. Common NR extraction is reused; standalone
and two-output-amortized Ref costs are reported separately.

## CPU profiling and other tools

Use `python -m experiments.<module> --help` for arguments and prerequisites.
For modules with subcommands, append the subcommand before `--help`.

| Module or subcommand | Purpose |
| --- | --- |
| `profile_compact`, `profile_all_features` | Live batch-one extraction and prediction; pass `--backends randomforest xgb catboost` |
| `profile_cost cached`, `profile_cost wasd` | Cached pipeline stages and detector timing |
| `backend_cost` (`prepare`, `verify`, `summarize`) | Export NR scores, verify timings and combine quality/cost results |
| `plot_backend_frontier` | Plot the CPU quality/latency frontier |
| `reproduce_backends` | Replay a five-NR fold |
| `audit_sriqa_resources` | Audit dedicated-method resources and dataset exposure |

Live profiling requires cached extractor weights, compatible generator sources
and the external WASD implementation/checkpoint. The extended profiler uses
`google/siglip2-base-patch16-256` snapshot
`3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab`.
Quality uses archived features; live/cache differences and coarse WASD timing
do not establish exact regeneration. Hardware and runtime changes produce new
timing measurements. Published artifacts contain aggregate reference values;
new runs save their own predictions, fit IDs and input hashes.
