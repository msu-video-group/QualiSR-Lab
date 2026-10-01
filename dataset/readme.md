# Datasets

## QualiSR-Set120

QualiSR-Set120 contains 40 source images and 120 SR images produced by PASD, SUPIR, and Real-ESRGAN, with subjective mean opinion scores (MOS) in `labels.csv`. It includes LR inputs, HR source images, bicubic/RLFN/SPAN pseudo-references, and precomputed artifact masks.

The source pool is [SR-Ground](https://huggingface.co/datasets/Divotion/SR-Ground), with images originating from FLIVE, KonIQ-10k, and AVA. The 40 sources were selected by k-means clustering, with 10 images from each of four clusters.

## Download

From the repository root, download the images from [Hugging Face](https://huggingface.co/datasets/onryabinin/QualiSR-Set120):

```bash
hf download onryabinin/QualiSR-Set120 --repo-type dataset --local-dir dataset
```

Alternatively, download the [Google Drive archive](https://drive.google.com/file/d/1NeGiwWQECTZMxVhJ5ZALxQ5nzRYkz4-E/view):

```bash
python -m pip install gdown
gdown --fuzzy 'https://drive.google.com/file/d/1NeGiwWQECTZMxVhJ5ZALxQ5nzRYkz4-E/view' -O grounding_dataset.zip
unzip grounding_dataset.zip -d dataset/
```

The repository includes labels and precomputed features. Image files are required for dataset-configured runs; only the default packaged regression example bypasses image loading.

## Layout

The bundled QualiSR-Set120 parser reads this layout (filenames are illustrative):

```text
dataset/
├── labels.csv
├── hr/0000001.png
├── lr/0000001.png
├── sr/
│   ├── PASD/0000001.png
│   ├── SUPIR/0000001.png
│   └── RealESRGAN/0000001.png
├── ref/
│   ├── bicubic/0000001.png
│   ├── rlfn/0000001.png
│   └── span/0000001.png
└── heatmaps/
    ├── PASD/0000001.npy.gz
    ├── SUPIR/0000001.npy.gz
    └── RealESRGAN/0000001.npy.gz
```

The dataset parser also recognizes generated references named `<sr_stem>@<sr_method>@<ref_name>.png`. The standalone feature extractor expects this generated naming convention.

Example label rows:

```csv
test_case,method,score,image
0000001,pasd,0.72,sr/PASD/0000001.png
0000001,supir,0.25,sr/SUPIR/0000001.png
0000001,realesrgan,0.61,sr/RealESRGAN/0000001.png
```

## Selecting datasets

The top-level `datasets` list in a pipeline configuration defines paths and regression roles:

```json
{
  "datasets": [
    {
      "name": "QualiSR-Set120",
      "root": "dataset",
      "features_root": "features",
      "regressors": {"train": true, "validate": true, "test_size": 0.2}
    }
  ]
}
```

`root`, `features_root`, and custom parser paths resolve relative to the config directory, except that a file directly inside a directory named `configs` uses its parent directory. Thus `configs/pipeline.json` resolves these paths from the repository root. Direct Python calls to `load_datasets` use `base_dir`, or the current directory when omitted. Feature-producing stages and regressors expand `{features_root}` separately for each dataset.

Both `train` and `validate` must be explicit booleans:

- Training with validation: set `test_size` in `[0, 1)`. A positive value holds out that fraction of source-image groups; zero uses all samples for training and none for validation.
- Training only: set `test_size: 0`.
- Validation only: omit `test_size`; all samples are used for validation.

Splits group by `test_case` within each dataset, using `regressors.config.split_seed`, independently of the model `seed`. All SR variants of one source must share the same `test_case`.

Multiple datasets are combined in one run. Bundled parsers include [QualiSR-Set120](https://huggingface.co/datasets/onryabinin/QualiSR-Set120), [DISRQAD](https://huggingface.co/datasets/visualprior/DISRQAD), [ISRGen-QA](https://github.com/Lighting-YXLI/ISRGen-QA), and [RealSRQ](https://github.com/Zhentao-Liu/RealSRQ-KLTSRQA); see [`datasets.py`](../qualisr/datasets.py) for additional aliases and parser options. The configured name becomes the dataset prefix in `sample_id`, so preserve its spelling and case when reusing feature CSVs.

## Custom parsers

Custom datasets do not need to follow the QualiSR-Set120 layout. Select a parser file and function in a dataset entry:

```json
{
  "name": "my-dataset",
  "root": "/data/my-dataset",
  "features_root": "/data/features/my-dataset",
  "regressors": {"train": false, "validate": true},
  "parser": {"path": "parsers/my_dataset.py", "function": "parse_dataset"},
  "kwargs": {"labels_fn": "scores.csv"}
}
```

The callable receives the absolute dataset root as its first argument and `kwargs` as keyword arguments. It must return a list of sample dictionaries:

```python
{
    "test_case": "source-image-id",
    "method": "sr-method",
    "lr_path": "lr/source-image-id.png",
    "sr_path": "sr/sr-method/source-image-id.png",
    "score": 0.72,
    "hr_path": "hr/source-image-id.png",  # optional
}
```

The five fields above `hr_path` are required. Image paths resolve relative to the dataset root. LR/SR files, and any supplied HR/reference files, must exist. Optional fields include `ref_paths` (reference name to path), `heatmap_path`, and `rel_path` (SR path relative to `sr/`). The loader supplies `dataset` from the configured name, derives `sample_id`, and rejects duplicate identities.

Provide finite scores with higher values indicating better quality. Built-in parsers and the directory parser min-max normalize finite scores within each dataset to `[0, 1]` (constant scores become `0.5`). Custom parser scores are converted to floats without automatic normalization or range validation; normalize them in the parser when combining datasets.

Scores default to `score_type: "mos"`. RealSRQ uses `bradley_terry` automatically. For a custom Bradley–Terry dataset, set `score_type: "bradley_terry"` on the dataset entry and supply `correlation_group` on each sample to identify an independently scored comparison set. Correlations are computed within those groups and averaged; `test_case` still identifies the source image for splitting.

## Directory-based datasets

For conventional filename matching, provide labels and directories instead of a parser:

```json
{
  "name": "my-directory-dataset",
  "root": "/data/my-dataset",
  "features_root": "/data/features/my-directory-dataset",
  "regressors": {"train": false, "validate": true},
  "labels": "labels.csv",
  "directories": {
    "hr": "hr",
    "lr": "lr",
    "sr": {"method-a": "outputs/method-a"},
    "refs": {"bicubic": "references/bicubic"},
    "heatmaps": {"method-a": "masks/method-a"}
  }
}
```

Labels and directory paths resolve relative to `root`. The labels CSV uses `test_case`, `method`, `score`, and optional `image` columns; override names with an entry-level `columns` mapping. Omit `hr` when no GT images are available.

## License

QualiSR-Set120 includes images derived from third-party datasets. Consult original source licenses for applicable use and redistribution terms; see also [third-party notices](../THIRD_PARTY_NOTICES.md).
