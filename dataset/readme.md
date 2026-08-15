# Dataset

Source image pool: [SR-Ground](https://huggingface.co/datasets/Divotion/SR-Ground).

## 🔎 Overview

This is a sample dataset based on open IQA datasets: [FLIVE](https://github.com/niu-haoran/FLIVE_Database), [KonIQ10k](https://database.mmsp-kn.de/koniq-10k-database.html), [AVA](https://github.com/imfing/ava_downloader). We picked 40 ground-truth images based on K-Means clusterization (10 images for each of the 4 centroids). For each of these images, we provide both SR ([PASD](https://github.com/yangxy/PASD)/[SUPIR](https://github.com/Fanghua-Yu/SUPIR)/[RealESRGAN](https://github.com/xinntao/real-esrgan)) and reference ([RLFN](https://github.com/bytedance/RLFN)/[SPAN](https://github.com/zononhzy/SPAN)/bicubic) images and also artifact masks in `heatmaps/` subdirectory. Each SR image has a Mean Opinion Score assigned to it in `labels.csv`.

---

## ⚙️ How to use

Dataset is available to download for free on [Hugging Face](https://huggingface.co/datasets/onryabinin/QualiSR-Set120) or [GDrive](https://drive.google.com/file/d/1NeGiwWQECTZMxVhJ5ZALxQ5nzRYkz4-E/view?usp=sharing). Below are the commands to download and unpack all the images.

```bash
hf download onryabinin/QualiSR-Set120 --repo-type dataset --local-dir dataset
```

Alternatively:

```bash
gdown 'https://drive.google.com/file/d/1NeGiwWQECTZMxVhJ5ZALxQ5nzRYkz4-E/view?usp=sharing'
unzip grounding_dataset.zip -d dataset/
```

---

## 💿 Dataset Input

QualiSR-Lab does not require custom datasets to copy the QualiSR-Set120 layout. The unified pipeline loads one or more datasets through parsing functions. Each parser returns a list of sample dictionaries with these required fields:

```python
{
    "dataset": "dataset-name",
    "test_case": "source-image-id",
    "method": "sr-method",
    "hr_path": "/absolute/path/to/hr.png",  # optional
    "lr_path": "/absolute/path/to/lr.png",
    "sr_path": "/absolute/path/to/sr.png",
    "score": 0.72,
}
```

Scores must be finite and normalized to `[0, 1]`. Parsers may also return `ref_paths`, `heatmap_path`, `rel_path`, and other metadata. QualiSR-Lab resolves paths, derives a stable dataset-prefixed `sample_id`, and rejects duplicate samples.

The following is the native QualiSR-Set120 layout, handled by the bundled `QualiSR-Set120` parser:

```
dataset/
├── hr/
│   ├── 0000001.png        # GT images, shape: (H, W, 3)
│   └── ...
├── lr/
│   ├── 0000001.png        # LR images, shape: (H/scale, W/scale, 3)
│   └── ...
├── heatmaps/
│   ├── sr_method_1/
│   │   ├── 0000001.npy.gz # Artifact masks, shape: (H, W, 1)
│   │   └── ...
│   ├── ...
│   └── sr_method_N/
│       ├── 0000001.npy.gz
│       └── ...
├── sr/
│   ├── sr_method_1/
│   │   ├── 0000001.png    # SR images, shape: (H, W, 3)
│   │   └── ...
│   ├── ...
│   └── sr_method_N/
│       ├── 0000001.png
│       └── ...
└── ref/
    ├── ref_method_1/
    │   ├── 0000001.png    # Reference (pseudo-GT) images, shape: (H, W, 3)
    │   └── ...
    ├── ...
    └── ref_method_M/
        ├── 0000001.png
        └── ...
```

Its SR images have normalized quality scores in `labels.csv`:

```csv
labels.csv

test_case,method,score,image
0000001,pasd,0.72,sr/PASD/0000001.png
0000001,supir,0.25,sr/SUPIR/0000001.png
0000001,realesrgan,0.61,sr/RealESRGAN/0000001.png
0000002,pasd,0.59,sr/PASD/0000002.png
...
```

If a parser does not provide `heatmap_path`, the pipeline derives it from the SR path. For example, `sr/PASD/0000001.png` maps to `heatmaps/PASD/0000001.npy.gz` below the dataset root.

## Selecting datasets

The default pipeline configuration selects QualiSR-Set120:

```json
"datasets": [
    {
        "name": "QualiSR-Set120",
        "root": "dataset",
        "features_root": "features",
        "regressors": {
            "train": true,
            "validate": true,
            "test_size": 0.2
        }
    }
]
```

`features_root` is resolved relative to the pipeline config and is reused by
feature-producing stages and regressors through `{features_root}` path
templates. Each dataset may therefore keep its feature CSVs in a separate
directory without repeating paths in the regressor section.

For regressors, set `train` and/or `validate` on each dataset:

- A training dataset must define `test_size` in `[0, 1)`. With `test_size: 0`,
  all samples train the regressors and none of that dataset is used for
  validation.
- With both roles enabled and `test_size > 0`, the dataset is split by
  `test_case`; the held-out groups are used for validation.
- A validation-only dataset omits `test_size`, and the entire dataset is used
  for validation.

Multiple entries are combined in one run. Bundled names include
`QualiSR-Set120`, `dsr-dataset`, `ISRGen-QA`, and `RealSRQ`.

A user parser is selected by file path and function name:

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

The callable receives the absolute dataset root as its first argument, receives `kwargs` as keyword arguments, and returns the sample dictionaries described above.

For labels-backed datasets with conventional matching, configure the directories directly instead of writing a parser:

```json
{
    "name": "my-directory-dataset",
    "root": "/data/my-dataset",
    "features_root": "/data/features/my-directory-dataset",
    "regressors": {"train": false, "validate": true},
    "labels": "labels.csv",
    "directories": {
        "hr": "/data/my-dataset/hr",
        "lr": "/data/my-dataset/lr",
        "sr": {"method-a": "/data/my-dataset/outputs/method-a"},
        "refs": {"bicubic": "/data/my-dataset/references/bicubic"},
        "heatmaps": {"method-a": "/data/my-dataset/masks/method-a"}
    }
}
```

The labels CSV defaults to `test_case`, `method`, `score`, and optional `image` columns. Override those names with an entry-level `columns` object.

---

## 🔑 License

This dataset combines data derived from multiple third-party sources.

* Content originating from the included third-party datasets follows the licensing terms of those original sources. In the current project setup, these components are treated as MIT-compatible for redistribution and research use where applicable.

If you plan to redistribute or use this dataset in downstream work, you should verify that your use complies with the licenses of all original data sources.
