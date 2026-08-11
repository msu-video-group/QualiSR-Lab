from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.decomposition import PCA


def write_image(path: Path, red: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(red, 0, 0)).save(path)


def test_extracts_sr_and_selected_reference_embeddings(tmp_path: Path, monkeypatch) -> None:
    from qualisr import features

    sr_path = tmp_path / "sr.png"
    reference_path = tmp_path / "bicubic.png"
    write_image(sr_path, 10)
    write_image(reference_path, 80)

    model = object()
    transform = object()
    monkeypatch.setattr(features, "require_feature_dependencies", lambda: None)
    monkeypatch.setattr(features, "init_vgg", lambda device: (model, transform))
    monkeypatch.setattr(features, "init_resnet", lambda device: (model, transform))
    monkeypatch.setattr(
        features,
        "init_timm_encoder",
        lambda model_name, device, pretrained=True: (model, transform),
    )
    monkeypatch.setattr(
        features,
        "extract_pretrained_features",
        lambda image, model, transform, device: np.asarray(
            [image.getpixel((0, 0))[0], image.getpixel((0, 0))[0] + 1], dtype=np.float32
        ),
    )

    output = tmp_path / "features.csv"
    features.main(
        [
            "--features",
            "vgg,ref-vgg,resnet,ref-resnet,timm,ref-timm",
            "--embedding-reference",
            "bicubic",
            "--timm-encoders",
            "tiny=model-name",
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
        samples=[
            {
                "sample_id": "dataset/method/case",
                "method": "method",
                "sr_path": str(sr_path),
                "ref_paths": {"bicubic": str(reference_path)},
            }
        ],
    )

    row = pd.read_csv(output).iloc[0]
    for column in ("vgg_00000", "resnet_00000", "tiny_00000"):
        assert row[column] == 10
    for column in ("ref_vgg_00000", "ref_resnet_00000", "ref_tiny_00000"):
        assert row[column] == 80


def test_reference_embeddings_require_a_configured_reference(tmp_path: Path, monkeypatch) -> None:
    from qualisr import features

    monkeypatch.setattr(features, "require_feature_dependencies", lambda: None)
    with pytest.raises(ValueError, match="--embedding-reference"):
        features.main(
            ["--features", "ref-vgg", "--device", "cpu", "--output", str(tmp_path / "out.csv")],
            samples=[],
        )


def test_paired_pca_uses_one_basis_fitted_on_both_training_sets(tmp_path: Path) -> None:
    from qualisr.pca import main as pca_main

    sr = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "set_type": ["train", "train", "test", "test"],
            "vgg_0": [0.0, 2.0, 4.0, 6.0],
            "vgg_1": [1.0, 3.0, 5.0, 7.0],
        }
    )
    reference = pd.DataFrame(
        {
            "sample_id": ["d", "b", "a", "c"],
            "ref_vgg_0": [3.0, 1.0, -1.0, 2.0],
            "ref_vgg_1": [4.0, 2.0, 0.0, 3.0],
        }
    )
    sr_path = tmp_path / "sr.csv"
    reference_path = tmp_path / "reference.csv"
    sr.to_csv(sr_path, index=False)
    reference.to_csv(reference_path, index=False)

    pca_main(
        [
            "--input",
            str(sr_path),
            "--reference-input",
            str(reference_path),
            "--blocks",
            "vgg=vgg_",
            "--reference-blocks",
            "vgg=ref_vgg_",
            "--n-components",
            "2",
            "--fit-column",
            "set_type",
            "--fit-value",
            "train",
            "--output-dir",
            str(tmp_path),
            "--output-template",
            "sr_pca{n}.csv",
            "--reference-output-template",
            "reference_pca{n}.csv",
        ]
    )

    sr_output = pd.read_csv(tmp_path / "sr_pca2.csv")
    reference_output = pd.read_csv(tmp_path / "reference_pca2.csv")
    fit_values = np.concatenate(
        [
            sr.loc[sr["set_type"] == "train", ["vgg_0", "vgg_1"]].to_numpy(),
            reference.loc[reference["sample_id"].isin(["a", "b"]), ["ref_vgg_0", "ref_vgg_1"]].to_numpy(),
        ],
        axis=0,
    )
    expected_pca = PCA(n_components=2, svd_solver="full").fit(fit_values)

    np.testing.assert_allclose(
        sr_output[["vgg_pca_000", "vgg_pca_001"]],
        expected_pca.transform(sr[["vgg_0", "vgg_1"]].to_numpy()),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        reference_output[["vgg_pca_000", "vgg_pca_001"]],
        expected_pca.transform(reference[["ref_vgg_0", "ref_vgg_1"]].to_numpy()),
        atol=1e-6,
    )
    assert reference_output["set_type"].tolist() == ["test", "train", "train", "test"]


def test_embedding_difference_aligns_samples_and_computes_sr_minus_reference(
    tmp_path: Path,
) -> None:
    from qualisr.embedding_difference import main as difference_main

    reference = pd.DataFrame(
        {
            "sample_id": ["b", "a"],
            "ref_vgg_00000": [5.0, 1.0],
            "ref_vgg_00001": [7.0, 2.0],
        }
    )
    sr = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "sr_method": ["one", "two"],
            "vgg_00000": [4.0, 9.0],
            "vgg_00001": [8.0, 10.0],
        }
    )
    reference_path = tmp_path / "reference.csv"
    sr_path = tmp_path / "sr.csv"
    output_path = tmp_path / "difference.csv"
    reference.to_csv(reference_path, index=False)
    sr.to_csv(sr_path, index=False)

    difference_main(
        [
            "--reference-input",
            str(reference_path),
            "--sr-input",
            str(sr_path),
            "--blocks",
            "vgg_diff=ref_vgg_,vgg_",
            "--output",
            str(output_path),
        ]
    )

    output = pd.read_csv(output_path)
    assert output["sample_id"].tolist() == ["a", "b"]
    assert output["sr_method"].tolist() == ["one", "two"]
    assert "vgg_00000" not in output
    assert output["vgg_diff_00000"].tolist() == [3.0, 4.0]
    assert output["vgg_diff_00001"].tolist() == [6.0, 3.0]


def test_embedding_difference_rejects_incompatible_inputs() -> None:
    from qualisr.embedding_difference import compute_differences

    sr = pd.DataFrame({"sample_id": ["a"], "vgg_00000": [1.0]})
    missing_sample = pd.DataFrame({"sample_id": ["b"], "ref_vgg_00000": [1.0]})
    with pytest.raises(ValueError, match="identical sample_id sets"):
        compute_differences(missing_sample, sr, [("vgg_diff", "ref_vgg_", "vgg_")])

    wrong_component = pd.DataFrame({"sample_id": ["a"], "ref_vgg_00001": [1.0]})
    with pytest.raises(ValueError, match="mismatched.*component suffixes"):
        compute_differences(wrong_component, sr, [("vgg_diff", "ref_vgg_", "vgg_")])


def test_pipeline_embedding_difference_section(tmp_path: Path) -> None:
    from qualisr.pipeline import PipelineOptions, run_pipeline

    reference_path = tmp_path / "reference.csv"
    sr_path = tmp_path / "sr.csv"
    output_path = tmp_path / "difference.csv"
    pd.DataFrame({"sample_id": ["a"], "ref_0": [2.0]}).to_csv(reference_path, index=False)
    pd.DataFrame({"sample_id": ["a"], "sr_0": [5.0]}).to_csv(sr_path, index=False)
    config = {
        "embedding_difference": {
            "enabled": True,
            "reference_input": str(reference_path),
            "sr_input": str(sr_path),
            "blocks": ["difference=ref_,sr_"],
            "output": str(output_path),
        }
    }

    run_pipeline(
        config,
        options=PipelineOptions(only_section=["embedding_difference"]),
    )

    assert pd.read_csv(output_path)["difference_0"].tolist() == [3.0]


def test_pipeline_reference_is_common_only(monkeypatch) -> None:
    from qualisr import pipeline

    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        pipeline,
        "run_module_main",
        lambda module_name, argv: calls.append((module_name, argv)),
    )
    pipeline.run_features(
        {
            "common": {"embedding_reference": "span"},
            "groups": {
                "reference": {
                    "features": ["ref-vgg"],
                    "output": "reference.csv",
                }
            },
        }
    )
    assert "--embedding-reference" in calls[0][1]
    assert calls[0][1][calls[0][1].index("--embedding-reference") + 1] == "span"

    with pytest.raises(ValueError, match="features.common"):
        pipeline.run_features(
            {
                "common": {},
                "groups": {
                    "reference": {
                        "features": ["ref-vgg"],
                        "embedding_reference": "rlfn",
                        "output": "reference.csv",
                    }
                },
            }
        )
