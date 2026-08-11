from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from qualisr.datasets import load_dataset, load_datasets


def write_image(path: Path, size: tuple[int, int] = (8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(64, 128, 192)).save(path)


def write_qualisr_dataset(root: Path, method: str = "PASD") -> None:
    rows = []
    for test_case, score in (("0001", 2.0), ("0002", 4.0)):
        for subdir in ("hr", "lr"):
            write_image(root / subdir / f"{test_case}.png")
        write_image(root / "sr" / method / f"{test_case}.png")
        rows.append(
            {
                "test_case": test_case,
                "method": method,
                "score": score,
                "image": f"sr/{method}/{test_case}.png",
            }
        )
    root.mkdir(parents=True, exist_ok=True)
    with (root / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_qualisr_samples_have_stable_ids_and_default_heatmaps(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    write_qualisr_dataset(root)

    samples = load_dataset({"name": "QualiSR-Set120", "root": str(root)})

    assert [sample["score"] for sample in samples] == [0.0, 1.0]
    assert samples[0]["sample_id"] == "QualiSR-Set120/PASD/0001.npy.gz"
    assert samples[0]["heatmap_path"] == str(root / "heatmaps" / "PASD" / "0001.npy.gz")
    assert Path(samples[0]["sr_path"]).is_absolute()


def test_custom_file_parser_and_multiple_datasets(tmp_path: Path) -> None:
    builtin_root = tmp_path / "builtin"
    custom_root = tmp_path / "custom"
    write_qualisr_dataset(builtin_root)
    write_image(custom_root / "lr.png")
    write_image(custom_root / "sr.png")
    parser_path = tmp_path / "parser.py"
    parser_path.write_text(
        """
from pathlib import Path

def parse_dataset(root, score=0.25):
    root = Path(root)
    return [{
        "dataset": "ignored-parser-name",
        "test_case": "case",
        "method": "custom-method",
        "rel_path": "sr.png",
        "lr_path": root / "lr.png",
        "sr_path": root / "sr.png",
        "score": score,
    }]
""",
        encoding="utf-8",
    )

    samples = load_datasets(
        [
            {"name": "QualiSR-Set120", "root": str(builtin_root)},
            {
                "name": "custom",
                "root": str(custom_root),
                "parser": {"path": str(parser_path), "function": "parse_dataset"},
                "kwargs": {"score": 0.75},
            },
        ]
    )

    assert len(samples) == 3
    assert samples[-1]["dataset"] == "custom"
    assert samples[-1]["score"] == 0.75
    assert samples[-1]["sample_id"] == "custom/custom-method/sr.npy.gz"


def test_directory_dataset_uses_explicit_directories(tmp_path: Path) -> None:
    root = tmp_path / "layout"
    hr_dir = tmp_path / "images" / "hr"
    lr_dir = tmp_path / "images" / "lr"
    sr_dir = tmp_path / "images" / "outputs"
    heatmap_dir = tmp_path / "images" / "masks"
    for directory in (hr_dir, lr_dir, sr_dir):
        write_image(directory / "case.png")
    heatmap_dir.mkdir(parents=True)
    with gzip.open(heatmap_dir / "case.npy.gz", "wb") as handle:
        np.save(handle, np.ones((2, 2), dtype=np.float32))
    root.mkdir(parents=True)
    pd.DataFrame([{"test_case": "case", "method": "method-a", "score": 0.5, "image": "case.png"}]).to_csv(
        root / "labels.csv", index=False
    )

    samples = load_dataset(
        {
            "name": "directory-data",
            "root": str(root),
            "labels": "labels.csv",
            "directories": {
                "hr": str(hr_dir),
                "lr": str(lr_dir),
                "sr": {"method-a": str(sr_dir)},
                "heatmaps": {"method-a": str(heatmap_dir)},
            },
        }
    )

    assert len(samples) == 1
    assert samples[0]["sr_path"] == str((sr_dir / "case.png").resolve())
    assert samples[0]["heatmap_path"] == str((heatmap_dir / "case.npy.gz").resolve())


def test_regressor_scores_can_come_directly_from_samples() -> None:
    from qualisr.regressors import load_config, load_scores

    cfg = load_config()
    scores = load_scores(
        cfg,
        samples=[{"sample_id": "dataset/method/case.npy.gz", "score": 0.4}],
    )

    assert scores.to_dict("records") == [{"name": "dataset/method/case.npy.gz", "score": 0.4}]


def test_sample_aware_statistics_uses_sample_id(tmp_path: Path) -> None:
    from qualisr.statistics import main as statistics_main

    heatmap_path = tmp_path / "case.npy.gz"
    with gzip.open(heatmap_path, "wb") as handle:
        np.save(handle, np.arange(4, dtype=np.float32).reshape(2, 2))
    output = tmp_path / "stats.csv"

    statistics_main(
        ["--output", str(output), "--no-progress"],
        samples=[{"sample_id": "data/method/case.npy.gz", "heatmap_path": str(heatmap_path)}],
    )

    frame = pd.read_csv(output)
    assert frame.loc[0, "name"] == "data/method/case.npy.gz"
    assert frame.loc[0, "max"] == 3.0


def test_sample_aware_bicubic_generation_updates_reference_paths(tmp_path: Path) -> None:
    from qualisr.references import generate_references

    root = tmp_path / "dataset"
    write_image(root / "lr.png", size=(4, 4))
    write_image(root / "sr.png", size=(8, 8))
    sample = {
        "dataset": "data",
        "dataset_root": str(root),
        "method": "method-a",
        "lr_path": str(root / "lr.png"),
        "sr_path": str(root / "sr.png"),
        "ref_paths": {},
    }

    stats = generate_references([sample], {"refs": ["bicubic"], "no_progress": True})

    assert stats["bicubic_ok"] == 1
    assert Path(sample["ref_paths"]["bicubic"]).is_file()
