"""Dataset parsers and sample loading for QualiSR-Lab.

Each parser returns a list of dictionaries with at least:
  dataset, test_case, method, hr_path, lr_path, sr_path, score

Parsers for Bradley-Terry datasets also return correlation_group, identifying
the independently scored comparison group.

When reference images exist in common reference folders, parser outputs also
include bicubic_path, rlfn_path, span_path, and ref_paths.

Subjective scores are min-max normalized per parsed dataset to [0, 1].
"""

import csv
import glob
import importlib.util
import math
import os
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
REFERENCE_TYPES = ("bicubic", "rlfn", "span")
REFERENCE_SUFFIXES = {"bicubic": "bicubic", "rlfn": "rlfn", "span": "span"}
SCORE_TYPES = ("mos", "bradley_terry")
BUILTIN_DATASETS = (
    "QualiSR-Set120",
    "quality-csv",
    "dsr-dataset",
    "dsr-full-dataset",
    "test-dsr-dataset",
    "test-dsr-dataset-bt",
    "GrMOS-small",
    "ISRGen-QA",
    "RealSRQ",
)


def _abs(path: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def _exists(path: str) -> bool:
    return os.path.isfile(path)


def _clean_row(row: dict[str, Any]) -> dict[str, str]:
    return {
        str(k).strip().lstrip("\ufeff"): "" if v is None else str(v).strip()
        for k, v in row.items()
        if k is not None
    }


def _strip_glob_ext(pattern: str) -> str:
    ext = pattern.replace("*", "")
    return ext if ext.startswith(".") else ""


def _normalize_exts(exts: Iterable[str] | None = None) -> tuple[str, ...]:
    if exts is None:
        return IMAGE_EXTS

    out: list[str] = []
    for raw in exts:
        ext = _strip_glob_ext(str(raw).strip().lower())
        if not ext:
            ext = str(raw).strip().lower()
        if ext and not ext.startswith("."):
            ext = "." + ext
        if ext:
            out.append(ext)
    return tuple(dict.fromkeys(out)) or IMAGE_EXTS


def normalize_sample_scores(
    samples: Iterable[dict[str, Any]],
    score_key: str = "score",
) -> list[dict[str, Any]]:
    """Min-max normalize all finite sample scores at once."""

    out = [dict(sample) for sample in samples]
    values: list[float] = []
    for sample in out:
        if score_key not in sample:
            continue
        try:
            value = float(sample[score_key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    if not values:
        return out

    lo = min(values)
    hi = max(values)
    span = hi - lo
    for sample in out:
        if score_key not in sample:
            continue
        try:
            value = float(sample[score_key])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        sample[score_key] = 0.5 if span <= 0 else (value - lo) / span
    return out


def _method_dir_candidates(method: str) -> list[str]:
    raw = method.strip()
    candidates = [raw, raw.upper(), raw.lower(), raw.capitalize()]
    aliases = {
        "pasd": "PASD",
        "supir": "SUPIR",
        "realesrgan": "RealESRGAN",
        "real-esrgan": "RealESRGAN",
        "real_esrgan": "RealESRGAN",
    }
    key = raw.lower()
    if key in aliases:
        candidates.insert(0, aliases[key])
    return list(dict.fromkeys([c for c in candidates if c]))


class ImageIndex:
    """Small filesystem index for resolving images by name or stem."""

    def __init__(self, root: str, recursive: bool = True, exts: Iterable[str] | None = None) -> None:
        self.root = _abs(root)
        self.recursive = recursive
        self.exts = set(_normalize_exts(exts))
        self.files: list[str] = []
        self.by_name: dict[str, list[str]] = defaultdict(list)
        self.by_stem: dict[str, list[str]] = defaultdict(list)
        self._index()

    def _index(self) -> None:
        if not os.path.isdir(self.root):
            return
        if self.recursive:
            walker = os.walk(self.root)
            for dirpath, _, filenames in walker:
                for filename in filenames:
                    self._add(os.path.join(dirpath, filename))
        else:
            for filename in os.listdir(self.root):
                self._add(os.path.join(self.root, filename))
        self.files.sort()
        for values in self.by_name.values():
            values.sort()
        for values in self.by_stem.values():
            values.sort()

    def _add(self, path: str) -> None:
        if not os.path.isfile(path):
            return
        stem, ext = os.path.splitext(os.path.basename(path))
        if ext.lower() not in self.exts:
            return
        path = _abs(path)
        self.files.append(path)
        self.by_name[os.path.basename(path).lower()].append(path)
        self.by_stem[stem.lower()].append(path)

    def find_name(self, name: str) -> str | None:
        matches = self.by_name.get(os.path.basename(name).lower(), [])
        return matches[0] if matches else None

    def find_stem(self, stem: str, preferred_ext: str = "") -> str | None:
        matches = self.by_stem.get(stem.lower(), [])
        if not matches:
            return None
        preferred_ext = preferred_ext.lower()
        if preferred_ext:
            for path in matches:
                if os.path.splitext(path)[1].lower() == preferred_ext:
                    return path
        return matches[0]

    def find_case(self, test_case: str, preferred_ext: str = "") -> str | None:
        return self.find_stem(test_case, preferred_ext=preferred_ext)


def _resolve_relative_or_abs(path_or_rel: str, root: str) -> str:
    if not path_or_rel:
        raise FileNotFoundError("Empty image path.")
    path = _abs(path_or_rel) if os.path.isabs(path_or_rel) else _abs(os.path.join(root, path_or_rel))
    if not _exists(path):
        raise FileNotFoundError(path)
    return path


def _resolve_by_case(root: str, subdir: str, test_case: str, preferred_ext: str = "") -> str:
    base = os.path.join(root, subdir)
    candidates = []
    if preferred_ext:
        candidates.append(os.path.join(base, f"{test_case}{preferred_ext}"))
    for ext in IMAGE_EXTS:
        candidates.append(os.path.join(base, f"{test_case}{ext}"))
    for candidate in candidates:
        if _exists(candidate):
            return _abs(candidate)
    matches = sorted(glob.glob(os.path.join(base, "**", f"{test_case}.*"), recursive=True))
    matches = [m for m in matches if _exists(m)]
    if matches:
        return _abs(matches[0])
    raise FileNotFoundError(f"Could not resolve {subdir} image for test_case={test_case} under {base}")


def _resolve_by_case_index(
    index: ImageIndex,
    test_case: str,
    preferred_ext: str = "",
    fallback_name: str = "",
) -> str:
    if fallback_name:
        found = index.find_name(fallback_name)
        if found:
            return found
    found = index.find_case(test_case, preferred_ext=preferred_ext)
    if found:
        return found
    raise FileNotFoundError(f"Could not resolve image for test_case={test_case} under {index.root}")


def _resolve_qualisr_sr(root: str, image_rel: str, test_case: str, method: str) -> tuple[str, str]:
    if image_rel:
        path = _resolve_relative_or_abs(image_rel, root)
        return path, image_rel.replace(os.sep, "/")

    sr_root = os.path.join(root, "sr")
    for method_dir in _method_dir_candidates(method):
        for ext in IMAGE_EXTS:
            candidate = os.path.join(sr_root, method_dir, f"{test_case}{ext}")
            if _exists(candidate):
                rel = os.path.relpath(candidate, root).replace(os.sep, "/")
                return _abs(candidate), rel

    matches = sorted(glob.glob(os.path.join(sr_root, "*", f"{test_case}.*")))
    matches = [m for m in matches if _exists(m)]
    if matches:
        rel = os.path.relpath(matches[0], root).replace(os.sep, "/")
        return _abs(matches[0]), rel
    raise FileNotFoundError(f"Could not resolve QualiSR SR for test_case={test_case}, method={method}")


def _reference_dir_candidates(root: str, ref_name: str) -> list[str]:
    aliases = [ref_name]
    if ref_name == "bicubic":
        aliases.extend(["sr_bicubic", "bicubic_sr"])
    elif ref_name == "rlfn":
        aliases.append("rfln")

    candidates: list[str] = []
    for alias in aliases:
        candidates.extend(
            [
                os.path.join(root, "ref", alias),
                os.path.join(root, "refs", alias),
                os.path.join(root, "reference", alias),
                os.path.join(root, "references", alias),
                os.path.join(root, alias),
                os.path.join(root, f"ref_{alias}"),
                os.path.join(root, f"sr_{alias}"),
            ]
        )
    return list(dict.fromkeys([_abs(path) for path in candidates if os.path.isdir(_abs(path))]))


def _find_reference_path(
    sample: dict[str, Any],
    indexes: list[ImageIndex],
    ref_name: str,
    suffix: str,
) -> str | None:
    sr_path = str(sample.get("sr_path") or "")
    sr_name = os.path.basename(sr_path)
    sr_stem, sr_ext = os.path.splitext(sr_name)
    test_case = str(sample.get("test_case") or "")
    method = str(sample.get("method") or "")

    stems: list[str] = []
    if sr_stem and test_case and method:
        stems.append(f"{sr_stem}@{test_case}@{method}@{suffix}")
    if sr_stem and method:
        stems.append(f"{sr_stem}@{method}@{suffix}")
    if test_case and method:
        stems.append(f"{test_case}@{method}@{suffix}")
    if sr_stem:
        if test_case:
            stems.append(f"{sr_stem}@{test_case}@{suffix}")
        stems.append(f"{sr_stem}@{suffix}")
        stems.append(sr_stem)
    if test_case:
        stems.append(f"{test_case}@{suffix}")
        stems.append(test_case)

    names = [sr_name] if sr_name else []

    for index in indexes:
        for name in names:
            found = index.find_name(name)
            if found:
                return found
        for stem in stems:
            found = index.find_stem(stem, preferred_ext=sr_ext)
            if found:
                return found
        for method_dir in _method_dir_candidates(method):
            for stem in [test_case, sr_stem]:
                if not stem:
                    continue
                for ext in ([sr_ext] if sr_ext else []) + list(IMAGE_EXTS):
                    candidate = os.path.join(index.root, method_dir, f"{stem}{ext}")
                    if os.path.isfile(candidate):
                        return _abs(candidate)
    return None


def attach_reference_paths(
    samples: Iterable[dict[str, Any]],
    root: str,
    refs: Iterable[str] = REFERENCE_TYPES,
    suffixes: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Attach existing reference image paths without requiring them.

    Added keys are flat for existing callers (`bicubic_path`, `rlfn_path`,
    `span_path`) and grouped in `ref_paths` for code that wants to iterate.
    Missing references are simply omitted from `ref_paths` and set to None in
    the flat key.
    """

    root = _abs(root)
    suffixes = suffixes or REFERENCE_SUFFIXES
    index_by_ref: dict[str, list[ImageIndex]] = {}
    for ref_name in refs:
        key = ref_name.lower()
        index_by_ref[key] = [
            ImageIndex(path, recursive=True) for path in _reference_dir_candidates(root, key)
        ]

    out: list[dict[str, Any]] = []
    for sample in samples:
        item = dict(sample)
        ref_paths: dict[str, str] = dict(item.get("ref_paths") or {})
        for ref_name in refs:
            key = ref_name.lower()
            flat_key = f"{key}_path"
            found = item.get(flat_key)
            if not found and index_by_ref.get(key):
                found = _find_reference_path(
                    item,
                    index_by_ref[key],
                    ref_name=key,
                    suffix=suffixes.get(key, key),
                )
            if found:
                found = _abs(str(found))
                item[flat_key] = found
                ref_paths[key] = found
            else:
                item.setdefault(flat_key, None)
        item["ref_paths"] = ref_paths
        out.append(item)
    return out


def parse_quality_csv_dataset(
    root: str,
    labels_fn: str = "labels.csv",
    lr_subdir: str = "lr",
    sr_subdir: str = "sr",
    test_case_col: str = "test_case",
    method_col: str = "method",
    score_col: str = "score",
    image_col: str = "image",
    dataset_name: str = "quality_csv",
    exts: Iterable[str] | None = None,
    recursive: bool = True,
    include_refs: bool = True,
) -> list[dict[str, Any]]:
    root = _abs(root)
    labels_path = (
        _abs(labels_fn) if os.path.isabs(os.path.expanduser(labels_fn)) else os.path.join(root, labels_fn)
    )
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"labels CSV not found: {labels_path}")

    normalized_exts = _normalize_exts(exts)
    lr_index = ImageIndex(os.path.join(root, lr_subdir), recursive=recursive, exts=normalized_exts)
    sr_index = ImageIndex(os.path.join(root, sr_subdir), recursive=recursive, exts=normalized_exts)

    samples: list[dict[str, Any]] = []
    with open(labels_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [str(col).strip().lstrip("\ufeff") for col in (reader.fieldnames or [])]
        required = [test_case_col, method_col, score_col]
        missing = [col for col in required if col not in fieldnames]
        if missing:
            raise ValueError(f"labels CSV is missing columns: {missing}")

        for row_idx, raw_row in enumerate(reader, start=2):
            row = _clean_row(raw_row)
            if not any(row.values()):
                continue

            test_case = row.get(test_case_col, "")
            method = row.get(method_col, "")
            score_text = row.get(score_col, "")
            image_rel = row.get(image_col, "") if image_col in row else ""
            if not test_case or not method or not score_text:
                raise ValueError(f"Invalid labels CSV row {row_idx}: {row}")

            try:
                if image_rel:
                    sr_path = _resolve_relative_or_abs(image_rel, root)
                    rel_path = image_rel.replace(os.sep, "/")
                else:
                    sr_path = _resolve_by_case_index(sr_index, test_case)
                    rel_path = os.path.relpath(sr_path, root).replace(os.sep, "/")
                sr_ext = os.path.splitext(sr_path)[1]
                lr_path = _resolve_by_case_index(
                    lr_index,
                    test_case,
                    preferred_ext=sr_ext,
                    fallback_name=os.path.basename(sr_path),
                )
                try:
                    hr_path: str | None = _resolve_by_case(root, "hr", test_case, preferred_ext=sr_ext)
                except FileNotFoundError:
                    hr_path = None
                score = float(score_text)
            except Exception as exc:
                raise RuntimeError(f"Failed to parse labels CSV row {row_idx}: {row}") from exc

            samples.append(
                {
                    "dataset": dataset_name,
                    "test_case": test_case,
                    "method": method,
                    "rel_path": rel_path,
                    "hr_path": hr_path,
                    "lr_path": lr_path,
                    "sr_path": sr_path,
                    "score": score,
                }
            )

    samples = normalize_sample_scores(samples)
    if include_refs:
        samples = attach_reference_paths(samples, root)
    return samples


def _pick_csv_column(
    fieldnames: Iterable[str],
    configured: str,
    fallbacks: Iterable[str],
    label: str,
) -> str:
    fields = list(fieldnames)
    if configured and configured in fields:
        return configured
    for column in fallbacks:
        if column in fields:
            return column
    tried = [configured] if configured else []
    tried.extend([col for col in fallbacks if col not in tried])
    raise ValueError(f"labels CSV is missing {label} column. Tried: {tried}")


_DSR_LR_STEM_RE = re.compile(
    r"^(?P<hr_stem>.+)_(?P<degradation>blur|jpeg|noise|resized)_(?P<scale>x[24])$",
    flags=re.IGNORECASE,
)
_DSR_METHOD_SCALE_RE = re.compile(r"_x[24]_", flags=re.IGNORECASE)
_DSR_FINAL_SCALE_RE = re.compile(r"_(?P<scale>x[24])$", flags=re.IGNORECASE)


def _dsr_scale_from_name(image_name: str) -> str:
    stem = os.path.splitext(os.path.basename(image_name))[0]
    match = _DSR_FINAL_SCALE_RE.search(stem)
    if match is None:
        raise ValueError(f"DSR image filename must end with _<x2|x4>: {image_name}")
    return match.group("scale").lower()


def _resolve_dsr_sr(root: str, sr_index: ImageIndex, image_name: str) -> tuple[str, str]:
    normalized = image_name.replace("\\", "/").strip()
    if not normalized:
        raise FileNotFoundError("Empty DSR SR image name.")

    expanded = os.path.expanduser(os.path.expandvars(normalized))
    candidates = (
        [expanded]
        if os.path.isabs(expanded)
        else [
            os.path.join(root, expanded),
            os.path.join(sr_index.root, expanded),
        ]
    )
    for candidate in dict.fromkeys(candidates):
        if _exists(candidate):
            path = _abs(candidate)
            return path, os.path.relpath(path, root).replace(os.sep, "/")

    found = sr_index.find_name(os.path.basename(normalized))
    if found:
        return found, os.path.relpath(found, root).replace(os.sep, "/")
    raise FileNotFoundError(f"Could not resolve DSR SR image '{image_name}' under {sr_index.root}")


def _resolve_dsr_source_paths(
    sr_path: str,
    hr_index: ImageIndex,
    lr_index: ImageIndex,
) -> tuple[str, str, str, str]:
    """Resolve method, test case, HR, and LR from one DSR SR filename."""

    sr_stem, sr_ext = os.path.splitext(os.path.basename(sr_path))
    matches: list[tuple[str, str, str]] = []
    for separator in _DSR_METHOD_SCALE_RE.finditer(sr_stem):
        lr_stem = sr_stem[separator.end() :]
        lr_path = lr_index.find_stem(lr_stem, preferred_ext=sr_ext)
        if lr_path:
            method = sr_stem[: separator.end() - 1]
            matches.append((method, lr_stem, lr_path))

    if not matches:
        raise FileNotFoundError(
            f"Could not match DSR SR image '{os.path.basename(sr_path)}' to an LR filename "
            f"under {lr_index.root}"
        )
    if len(matches) > 1:
        candidates = [lr_stem for _, lr_stem, _ in matches]
        raise ValueError(f"Ambiguous DSR LR match for '{os.path.basename(sr_path)}': {candidates}")

    method, lr_stem, lr_path = matches[0]
    lr_match = _DSR_LR_STEM_RE.fullmatch(lr_stem)
    if lr_match is None:
        raise ValueError(
            f"DSR LR filename '{os.path.basename(lr_path)}' must end with _<blur|jpeg|noise|resized>_<x4|x2>"
        )

    hr_stem = lr_match.group("hr_stem")
    hr_path = hr_index.find_stem(
        hr_stem,
        preferred_ext=os.path.splitext(lr_path)[1],
    )
    if not hr_path:
        raise FileNotFoundError(f"Could not resolve DSR HR image '{hr_stem}.*' under {hr_index.root}")
    return method, lr_stem, hr_path, lr_path


def parse_dsr_dataset(
    root: str,
    labels_fn: str = "scores/mos.csv",
    hr_subdir: str = "hr",
    lr_subdir: str = "lr",
    sr_subdir: str = "sr",
    image_col: str = "img_name",
    score_col: str = "mos",
    dataset_name: str = "dsr-dataset",
    exts: Iterable[str] | None = None,
    recursive: bool = True,
    scale_split_dirs: bool = True,
    include_refs: bool = True,
) -> list[dict[str, Any]]:
    """Parse DSR datasets whose MOS rows name SR images.

    SR filenames have the form
    ``<method>_<x4|x2>_<source>_<degradation>_<x4|x2>.<ext>``. The parser
    matches the filename suffix against the LR directory, then derives the HR
    stem by removing the degradation and LR-scale suffix. When
    ``scale_split_dirs`` is enabled, LR and SR are resolved from
    ``lr_x2/lr_x4`` and ``sr_x2/sr_x4`` respectively.
    """

    root = _abs(root)
    labels_path = (
        _abs(labels_fn) if os.path.isabs(os.path.expanduser(labels_fn)) else os.path.join(root, labels_fn)
    )
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"DSR MOS CSV not found: {labels_path}")

    normalized_exts = _normalize_exts(exts)
    hr_index = ImageIndex(
        os.path.join(root, hr_subdir),
        recursive=recursive,
        exts=normalized_exts,
    )
    index_scales = ("x2", "x4") if scale_split_dirs else ("",)
    lr_indexes: dict[str, ImageIndex] = {}
    sr_indexes: dict[str, ImageIndex] = {}
    for scale in index_scales:
        suffix = f"_{scale}" if scale else ""
        lr_root = os.path.join(root, f"{lr_subdir}{suffix}")
        sr_root = os.path.join(root, f"{sr_subdir}{suffix}")
        if scale_split_dirs:
            for label, directory in (("LR", lr_root), ("SR", sr_root)):
                if not os.path.isdir(directory):
                    raise FileNotFoundError(
                        f"DSR full {label} scale directory not found: {directory}. "
                        "Run scripts/split_dsr_full_scales.py first."
                    )
        lr_indexes[scale] = ImageIndex(
            lr_root,
            recursive=recursive,
            exts=normalized_exts,
        )
        sr_indexes[scale] = ImageIndex(
            sr_root,
            recursive=recursive,
            exts=normalized_exts,
        )

    samples: list[dict[str, Any]] = []
    with open(labels_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [str(column).strip().lstrip("\ufeff") for column in (reader.fieldnames or [])]
        missing = [column for column in (image_col, score_col) if column not in fieldnames]
        if missing:
            raise ValueError(f"DSR MOS CSV is missing columns: {missing}")

        for row_idx, raw_row in enumerate(reader, start=2):
            row = _clean_row(raw_row)
            if not any(row.values()):
                continue
            image_name = row.get(image_col, "")
            score_text = row.get(score_col, "")
            if not image_name or not score_text:
                raise ValueError(f"Invalid DSR MOS row {row_idx}: {row}")

            try:
                scale = _dsr_scale_from_name(image_name) if scale_split_dirs else ""
                lr_index = lr_indexes[scale]
                sr_index = sr_indexes[scale]
                sr_path, rel_path = _resolve_dsr_sr(root, sr_index, image_name)
                if scale_split_dirs and os.path.commonpath([sr_path, sr_index.root]) != sr_index.root:
                    raise ValueError(f"DSR full SR image must resolve under {sr_index.root}: {sr_path}")
                method, test_case, hr_path, lr_path = _resolve_dsr_source_paths(
                    sr_path,
                    hr_index,
                    lr_index,
                )
                resolved_scale = _dsr_scale_from_name(lr_path)
                if scale_split_dirs and resolved_scale != scale:
                    raise ValueError(
                        f"DSR full SR/LR scale mismatch: SR={scale}, LR={resolved_scale}, image={image_name}"
                    )
                score = float(score_text)
            except Exception as exc:
                raise RuntimeError(f"Failed to parse DSR MOS row {row_idx}: {row}") from exc

            samples.append(
                {
                    "dataset": dataset_name,
                    "test_case": test_case,
                    "method": method,
                    "rel_path": rel_path,
                    "hr_path": hr_path,
                    "lr_path": lr_path,
                    "sr_path": sr_path,
                    "scale": int(resolved_scale[1:]),
                    "score": score,
                }
            )

    samples = normalize_sample_scores(samples)
    if include_refs:
        samples = attach_reference_paths(samples, root)
    return samples


def _resolve_test_dsr_sr(root: str, imgs_root: str, image_rel: str) -> tuple[str, str]:
    normalized = image_rel.replace("\\", "/").strip()
    if not normalized:
        raise FileNotFoundError("Empty SR image path.")
    if os.path.isabs(os.path.expanduser(normalized)):
        path = _resolve_relative_or_abs(normalized, root)
        return path, normalized

    candidates = [
        os.path.join(root, normalized),
        os.path.join(imgs_root, normalized),
    ]
    for candidate in dict.fromkeys(candidates):
        if _exists(candidate):
            path = _abs(candidate)
            try:
                rel_path = os.path.relpath(path, imgs_root).replace(os.sep, "/")
            except ValueError:
                rel_path = os.path.relpath(path, root).replace(os.sep, "/")
            return path, rel_path
    raise FileNotFoundError(f"Could not resolve SR image '{image_rel}' under {imgs_root}")


def _resolve_test_dsr_gt(case_dir: str, test_case: str, sr_path: str) -> str:
    candidates = []
    try:
        candidates.append(os.path.join(case_dir, test_case.split("_")[1] + "_vg_dsr.jpg"))
    except IndexError:
        candidates.append(os.path.join(case_dir, "orig.jpg"))
    # for ext in IMAGE_EXTS:
    #     candidates.append(os.path.join(case_dir, f"{test_case}{ext}"))
    sr_abs = _abs(sr_path)
    for candidate in candidates:
        if _exists(candidate) and _abs(candidate) != sr_abs:
            return _abs(candidate)
    raise FileNotFoundError(f"Could not resolve GT image '{test_case}.*' under {case_dir}")


def parse_test_dsr_dataset(
    root: str,
    labels_fn: str = "mos_no_ref.csv",
    imgs_subdir: str = "imgs",
    image_col: str = "",
    score_col: str = "mean",
    dataset_name: str = "test_dsr_dataset",
    include_refs: bool = True,
) -> list[dict[str, Any]]:
    """Parse test DSR-style datasets with SR paths stored under imgs/<case>/.

    The labels CSV must contain a subjective score column, defaulting to
    `mean`, and an SR image path column named `image_b` or `image_a` unless
    overridden by `image_col`.
    """

    root = _abs(root)
    imgs_root = _abs(os.path.join(root, imgs_subdir))
    labels_path = (
        _abs(labels_fn) if os.path.isabs(os.path.expanduser(labels_fn)) else os.path.join(root, labels_fn)
    )
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"labels CSV not found: {labels_path}")
    if not os.path.isdir(imgs_root):
        raise FileNotFoundError(f"imgs directory not found: {imgs_root}")

    samples: list[dict[str, Any]] = []
    with open(labels_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [str(col).strip().lstrip("\ufeff") for col in (reader.fieldnames or [])]
        sr_col = _pick_csv_column(fieldnames, image_col, ("image_b", "image_a"), "SR image path")
        score_key = _pick_csv_column(fieldnames, score_col, ("mean", "score", "mos"), "score")

        for row_idx, raw_row in enumerate(reader, start=2):
            row = _clean_row(raw_row)
            if not any(row.values()):
                continue

            image_rel = row.get(sr_col, "")
            score_text = row.get(score_key, "")
            if not image_rel or not score_text:
                raise ValueError(f"Invalid test DSR row {row_idx}: {row}")

            try:
                sr_path, rel_path = _resolve_test_dsr_sr(root, imgs_root, image_rel)
                case_rel = os.path.dirname(rel_path).replace("\\", "/")
                test_case = (
                    os.path.basename(case_rel.rstrip("/"))
                    if case_rel
                    else os.path.splitext(os.path.basename(sr_path))[0]
                )
                method = os.path.splitext(os.path.basename(sr_path))[0]
                hr_path = _resolve_test_dsr_gt(os.path.dirname(sr_path), test_case, sr_path)
                score = float(score_text)
            except Exception as exc:
                raise RuntimeError(f"Failed to parse test DSR row {row_idx}: {row}") from exc

            samples.append(
                {
                    "dataset": dataset_name,
                    "test_case": test_case,
                    "method": method,
                    "rel_path": rel_path,
                    "hr_path": hr_path,
                    "lr_path": hr_path,
                    "sr_path": sr_path,
                    "score": score,
                }
            )

    samples = normalize_sample_scores(samples)
    if include_refs:
        samples = attach_reference_paths(samples, root)
    return samples


_TEST_DSR_BT_LABEL_RE = re.compile(
    r"^name=(?P<test_case>.+)_model=Bradley-Terry\.csv$",
    flags=re.IGNORECASE,
)
_TEST_DSR_BT_RESIZE_RE = re.compile(
    r"_resized_x[24]$",
    flags=re.IGNORECASE,
)


def _resolve_test_dsr_bt_case_dir(
    imgs_root: str,
    test_case: str,
) -> tuple[str, str]:
    source_name = _TEST_DSR_BT_RESIZE_RE.sub("", test_case)
    source_stem = os.path.splitext(source_name)[0]
    names = [
        test_case,
        source_name,
        source_stem,
        f"name={test_case}",
        f"name={source_name}",
        f"name={source_stem}",
    ]
    for name in dict.fromkeys(names):
        candidate = os.path.join(imgs_root, name)
        if os.path.isdir(candidate):
            return _abs(candidate), source_name

    wanted = {name.lower() for name in names}
    for dirpath, dirnames, _ in os.walk(imgs_root):
        for dirname in sorted(dirnames):
            if dirname.lower() in wanted:
                return _abs(os.path.join(dirpath, dirname)), source_name

    imgs_index = ImageIndex(imgs_root, recursive=True)
    for stem in (f"{source_stem}_vg_dsr", source_stem):
        found = imgs_index.find_stem(stem)
        if found:
            return _abs(os.path.dirname(found)), source_name
    raise FileNotFoundError(
        f"Could not resolve test-DSR image directory for BT case '{test_case}' under {imgs_root}"
    )


def _resolve_test_dsr_bt_gt(
    case_dir: str,
    test_case: str,
    source_name: str,
) -> str:
    index = ImageIndex(case_dir, recursive=False)
    source_stem = os.path.splitext(source_name)[0]
    stems = [
        f"{source_stem}_vg_dsr",
        f"{source_name}_vg_dsr",
        source_stem,
        source_name,
        f"{test_case}_vg_dsr",
        test_case,
        "orig",
        "gt",
        "hr",
        test_case.split("_")[1] + "_vg_dsr",
    ]
    for stem in dict.fromkeys(stems):
        found = index.find_stem(stem)
        if found:
            return found
    raise FileNotFoundError(f"Could not resolve GT for test-DSR BT case '{test_case}' under {case_dir}")


def _resolve_test_dsr_bt_sr(
    case_dir: str,
    test_case: str,
    source_name: str,
    method: str,
) -> str:
    index = ImageIndex(case_dir, recursive=False)
    stems = [
        method,
        f"name={test_case}_model={method}",
        f"name={source_name}_model={method}",
        f"{test_case}_{method}",
        f"{source_name}_{method}",
    ]
    for stem in dict.fromkeys(stems):
        found = index.find_stem(stem)
        if found:
            return found
    raise FileNotFoundError(
        f"Could not resolve SR method '{method}' for test-DSR BT case '{test_case}' under {case_dir}"
    )


def parse_test_dsr_bt_dataset(
    root: str,
    labels_fn: str = "bt",
    imgs_subdir: str = "imgs",
    category_col: str = "Category",
    score_col: str = "*",
    dataset_name: str = "test-dsr-dataset-bt",
    include_refs: bool = True,
) -> list[dict[str, Any]]:
    """Parse one semicolon-delimited Bradley-Terry score CSV per test-DSR GT."""

    root = _abs(root)
    imgs_root = _abs(os.path.join(root, imgs_subdir))
    labels_path = (
        _abs(labels_fn)
        if os.path.isabs(os.path.expanduser(labels_fn))
        else _abs(os.path.join(root, labels_fn))
    )
    if not os.path.isdir(imgs_root):
        raise FileNotFoundError(f"imgs directory not found: {imgs_root}")
    if os.path.isdir(labels_path):
        label_files = sorted(glob.glob(os.path.join(labels_path, "**", "*.csv"), recursive=True))
    elif os.path.isfile(labels_path):
        label_files = [labels_path]
    else:
        raise FileNotFoundError(f"test-DSR BT labels not found: {labels_path}")
    if not label_files:
        raise FileNotFoundError(f"No test-DSR BT CSV files found under {labels_path}")

    samples: list[dict[str, Any]] = []
    for labels_file in label_files:
        filename_match = _TEST_DSR_BT_LABEL_RE.fullmatch(os.path.basename(labels_file))
        if filename_match is None:
            raise ValueError(
                "test-DSR BT label filename must match "
                f"'name=<img_name>_resized_<x4|x2>_model=Bradley-Terry.csv': "
                f"{labels_file}"
            )
        test_case = filename_match.group("test_case")
        if _TEST_DSR_BT_RESIZE_RE.search(test_case) is None:
            raise ValueError(
                "test-DSR BT label filename must include a resized scale suffix "
                f"before '_model=Bradley-Terry.csv': {labels_file}"
            )
        case_dir, source_name = _resolve_test_dsr_bt_case_dir(
            imgs_root,
            test_case,
        )
        hr_path = _resolve_test_dsr_bt_gt(
            case_dir,
            test_case,
            source_name,
        )

        with open(labels_file, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            fieldnames = [str(column).strip().lstrip("\ufeff") for column in (reader.fieldnames or [])]
            missing = [column for column in (category_col, score_col) if column not in fieldnames]
            if missing:
                raise ValueError(f"test-DSR BT CSV is missing columns {missing}: {labels_file}")

            for row_idx, raw_row in enumerate(reader, start=2):
                row = _clean_row(raw_row)
                if not any(row.values()):
                    continue
                method = row.get(category_col, "")
                score_text = row.get(score_col, "")
                if not method or not score_text:
                    raise ValueError(f"Invalid test-DSR BT row {row_idx} in {labels_file}: {row}")
                try:
                    sr_path = _resolve_test_dsr_bt_sr(
                        case_dir,
                        test_case,
                        source_name,
                        method,
                    )
                    score = float(score_text.replace(",", "."))
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to parse test-DSR BT row {row_idx} in {labels_file}: {row}"
                    ) from exc

                samples.append(
                    {
                        "dataset": dataset_name,
                        "score_type": "bradley_terry",
                        "test_case": test_case,
                        "correlation_group": test_case,
                        "method": method,
                        "rel_path": os.path.relpath(
                            sr_path,
                            imgs_root,
                        ).replace(os.sep, "/"),
                        "hr_path": hr_path,
                        "lr_path": hr_path,
                        "sr_path": sr_path,
                        "score": score,
                    }
                )

    samples = normalize_sample_scores(samples)
    if include_refs:
        samples = attach_reference_paths(samples, root)
    return samples


def parse_qualisr_set120(
    root: str,
    labels_fn: str = "labels.csv",
    test_case_col: str = "test_case",
    method_col: str = "method",
    score_col: str = "score",
    image_col: str = "image",
    dataset_name: str = "QualiSR-Set120",
    include_refs: bool = True,
) -> list[dict[str, Any]]:
    root = _abs(root)
    labels_path = (
        _abs(labels_fn) if os.path.isabs(os.path.expanduser(labels_fn)) else os.path.join(root, labels_fn)
    )
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"labels.csv not found: {labels_path}")

    samples: list[dict[str, Any]] = []
    with open(labels_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [str(col).strip().lstrip("\ufeff") for col in (reader.fieldnames or [])]
        required = [test_case_col, method_col, score_col]
        missing = [col for col in required if col not in fieldnames]
        if missing:
            raise ValueError(f"QualiSR labels are missing columns: {missing}")

        for row_idx, raw_row in enumerate(reader, start=2):
            row = _clean_row(raw_row)
            if not any(row.values()):
                continue
            test_case = row.get(test_case_col, "")
            method = row.get(method_col, "")
            score_text = row.get(score_col, "")
            image_rel = row.get(image_col, "")
            if not test_case or not method or not score_text:
                raise ValueError(f"Invalid QualiSR row {row_idx}: {row}")

            try:
                sr_path, rel_path = _resolve_qualisr_sr(root, image_rel, test_case, method)
                rel_parts = Path(rel_path).parts
                resolved_method = rel_parts[1] if len(rel_parts) > 2 and rel_parts[0].casefold() == "sr" else method
                ext = os.path.splitext(sr_path)[1]
                lr_path = _resolve_by_case(root, "lr", test_case, preferred_ext=ext)
                hr_path: str | None
                try:
                    hr_path = _resolve_by_case(root, "hr", test_case, preferred_ext=ext)
                except FileNotFoundError:
                    hr_path = None
                score = float(score_text)
            except Exception as exc:
                raise RuntimeError(f"Failed to parse QualiSR row {row_idx}: {row}") from exc

            samples.append(
                {
                    "dataset": dataset_name,
                    "test_case": test_case,
                    "method": resolved_method,
                    "rel_path": rel_path,
                    "hr_path": hr_path,
                    "lr_path": lr_path,
                    "sr_path": sr_path,
                    "score": score,
                }
            )
    samples = normalize_sample_scores(samples)
    if include_refs:
        samples = attach_reference_paths(samples, root)
    return samples


def parse_grounding_mos(
    base_path: str,
    labels_fn: str = "mos_scores_small.csv",
    dataset_name: str = "GrMOS-small",
    include_refs: bool = True,
) -> list[dict[str, Any]]:
    base_path = _abs(base_path)
    labels_path = (
        _abs(labels_fn)
        if os.path.isabs(os.path.expanduser(labels_fn))
        else os.path.join(base_path, labels_fn)
    )
    samples: list[dict[str, Any]] = []
    with open(labels_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, raw_row in enumerate(reader, start=2):
            row = _clean_row(raw_row)
            test_case = row.get("test_case", "")
            method = row.get("method", "")
            if not test_case or not method:
                raise ValueError(f"Invalid Grounding MOS row {row_idx}: {row}")
            samples.append(
                {
                    "dataset": dataset_name,
                    "test_case": test_case,
                    "method": method,
                    "rel_path": f"{test_case}/{method}.png",
                    "hr_path": _abs(os.path.join(base_path, test_case, "hr.png")),
                    "lr_path": _abs(os.path.join(base_path, test_case, "lr.png")),
                    "sr_path": _abs(os.path.join(base_path, test_case, f"{method}.png")),
                    "score": float(row["score"]),
                }
            )
    samples = normalize_sample_scores(samples)
    if include_refs:
        samples = attach_reference_paths(samples, base_path)
    return samples


def parse_isrgenqa(
    isrgen_root: str,
    labels_fn: str = "combined_labels.txt",
    dataset_name: str = "ISRGen-QA",
    include_refs: bool = True,
) -> list[dict[str, Any]]:
    isrgen_root = _abs(isrgen_root)
    sr_dir = os.path.join(isrgen_root, "SR")
    lr_dir = os.path.join(isrgen_root, "LR")
    hr_dir = os.path.join(isrgen_root, "HR")
    labels_path = (
        _abs(labels_fn)
        if os.path.isabs(os.path.expanduser(labels_fn))
        else os.path.join(isrgen_root, labels_fn)
    )

    samples: list[dict[str, Any]] = []
    with open(labels_path, encoding="utf-8") as f:
        for row_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("#")]
            if len(parts) < 3:
                raise ValueError(f"Invalid ISRGen-QA row {row_idx}: {line}")
            sr_name, mos_text, lr_name = parts[:3]
            hr_name = "0" + lr_name[-9:-6] + ".png"
            samples.append(
                {
                    "dataset": dataset_name,
                    "test_case": os.path.splitext(lr_name)[0],
                    "method": os.path.splitext(sr_name)[0],
                    "rel_path": sr_name,
                    "hr_path": _abs(os.path.join(hr_dir, hr_name)),
                    "lr_path": _abs(os.path.join(lr_dir, lr_name)),
                    "sr_path": _abs(os.path.join(sr_dir, sr_name)),
                    "score": float(mos_text),
                }
            )
    samples = normalize_sample_scores(samples)
    if include_refs:
        samples = attach_reference_paths(samples, isrgen_root)
    return samples


def parse_realsrq(
    base_path: str,
    labels_fn: str = "subj_BT_score.mat",
    dataset_name: str = "RealSRQ",
    include_refs: bool = True,
) -> list[dict[str, Any]]:
    try:
        import scipy.io as sio
    except Exception as exc:
        raise ImportError("parse_realsrq requires scipy") from exc

    base_path = _abs(base_path)
    labels_path = (
        _abs(labels_fn)
        if os.path.isabs(os.path.expanduser(labels_fn))
        else os.path.join(base_path, labels_fn)
    )
    bt_data = sio.loadmat(labels_path)
    bt_scores = bt_data["score_matrix"]

    scene_types = ["Buildings", "Manmade", "Natural", "People", "Scene", "Text"]
    scenes_per_type = 10
    lr2_methods = ["AIS", "Aplus", "ASDS", "BCI", "CSCN", "SPM", "SRCNN", "SRGAN", "USRnet", "VDSR"]
    lr3_methods = ["AIS", "Aplus", "ASDS", "BCI", "CSCN", "SPM", "SRCNN", "USRnet", "VDSR"]
    lr4_methods = ["Aplus", "ASDS", "BCI", "CSCN", "SRCNN", "SRGAN", "USRnet", "VDSR"]
    lr_scales = [(2, lr2_methods), (3, lr3_methods), (4, lr4_methods)]

    samples: list[dict[str, Any]] = []
    row_idx = 0
    for scene_type in scene_types:
        for scene_num in range(1, scenes_per_type + 1):
            scene_name = f"{scene_type}_{scene_num:03d}"
            hr_path = os.path.join(
                base_path, "HR_LR_pairs", scene_type, f"{scene_type}_HR", f"{scene_name}_HR.png"
            )
            col_idx = 0
            for lr_scale, methods in lr_scales:
                lr_path = os.path.join(
                    base_path,
                    "HR_LR_pairs",
                    scene_type,
                    f"{scene_type}_LR{lr_scale}",
                    f"{scene_name}_LR{lr_scale}.png",
                )
                for method in methods:
                    sr_path = os.path.join(
                        base_path,
                        "SR_results",
                        f"{scene_type}_SR",
                        f"{scene_type}_LR{lr_scale}_SR",
                        f"{scene_num:03d}",
                        f"{scene_name}_LR{lr_scale}_{method}.png",
                    )
                    samples.append(
                        {
                            "dataset": dataset_name,
                            "score_type": "bradley_terry",
                            "test_case": scene_name,
                            "correlation_group": f"{scene_name}_LR{lr_scale}",
                            "method": f"{method}_x{lr_scale}",
                            "rel_path": os.path.relpath(sr_path, base_path).replace(os.sep, "/"),
                            "hr_path": _abs(hr_path),
                            "lr_path": _abs(lr_path),
                            "sr_path": _abs(sr_path),
                            "heatmap_path": _abs(
                                os.path.join(
                                    base_path,
                                    "heatmaps",
                                    f"{Path(sr_path).stem}.npy.gz",
                                )
                            ),
                            "score": float(bt_scores[row_idx, col_idx]),
                        }
                    )
                    col_idx += 1
            row_idx += 1
    samples = normalize_sample_scores(samples)
    if include_refs:
        samples = attach_reference_paths(samples, base_path)
    return samples


def filter_existing(
    samples: Iterable[dict[str, Any]],
    require_hr: bool = False,
    strict: bool = False,
    require_refs: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    missing: list[str] = []
    required_ref_keys = [f"{ref.lower()}_path" for ref in (require_refs or [])]
    for sample in samples:
        required_paths = {
            "lr_path": sample.get("lr_path"),
            "sr_path": sample.get("sr_path"),
        }
        if require_hr:
            required_paths["hr_path"] = sample.get("hr_path")
        for key in required_ref_keys:
            required_paths[key] = sample.get(key)
        bad = [
            str(path) if path else key
            for key, path in required_paths.items()
            if not path or not os.path.isfile(str(path))
        ]
        if bad:
            missing.extend(bad)
            continue
        out.append(sample)
    if strict and missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"Missing {len(missing)} files. First missing files:\n{preview}")
    return out


def parse_iqa_dataset(
    name: str,
    root: str,
    labels_fn: str = "",
    include_refs: bool = True,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    key = name.lower().replace("_", "-")
    if key in ("qualisr", "qualisr-set120", "qualisr120"):
        return parse_qualisr_set120(root, labels_fn or "labels.csv", include_refs=include_refs, **kwargs)
    if key in ("isrgenqa", "isrgen-qa"):
        return parse_isrgenqa(root, labels_fn or "combined_labels.txt", include_refs=include_refs, **kwargs)
    if key in ("realsrq", "realsr-q"):
        return parse_realsrq(root, labels_fn or "subj_BT_score.mat", include_refs=include_refs, **kwargs)
    if key in ("grounding", "grmos", "grmos-small"):
        return parse_grounding_mos(
            root, labels_fn or "mos_scores_small.csv", include_refs=include_refs, **kwargs
        )
    if key in ("dsr-dataset", "dsr-full-dataset"):
        parser_kwargs = dict(kwargs)
        parser_kwargs.setdefault("dataset_name", key)
        if key == "dsr-full-dataset":
            parser_kwargs.setdefault("scale_split_dirs", True)
        return parse_dsr_dataset(
            root,
            labels_fn or "scores/mos.csv",
            include_refs=include_refs,
            **parser_kwargs,
        )
    if key == "test-dsr-dataset-bt":
        return parse_test_dsr_bt_dataset(
            root,
            labels_fn or "bt",
            include_refs=include_refs,
            **kwargs,
        )
    if key in (
        "test-dsr",
        "test-dsr-fr",
        "test-dsr-nr",
        "test-dsr-bt",
        "test-dsr-dataset",
        "testdsr",
        "dsr-test",
        "dsr",
        "test_dsr_dataset",
    ):
        return parse_test_dsr_dataset(
            root, labels_fn or "mos_no_ref.csv", include_refs=include_refs, **kwargs
        )
    if key in ("quality-csv", "quality_csv", "csv"):
        return parse_quality_csv_dataset(root, labels_fn or "labels.csv", include_refs=include_refs, **kwargs)
    raise ValueError(f"Unknown IQA dataset: {name}")


def _resolve_config_path(value: str | os.PathLike[str], base: Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _directory_index_path(
    index: ImageIndex,
    test_case: str,
    image_value: str,
    preferred_ext: str = "",
) -> str:
    if image_value:
        found = index.find_name(os.path.basename(image_value))
        if found:
            return found
    return _resolve_by_case_index(index, test_case, preferred_ext=preferred_ext)


def _mapping_value(mapping: Mapping[str, Any], key: str) -> Any:
    wanted = key.casefold()
    for name, value in mapping.items():
        if str(name).casefold() == wanted:
            return value
    return mapping.get("default")


def parse_directory_dataset(
    *,
    dataset_name: str,
    root: str,
    labels_fn: str,
    directories: Mapping[str, Any],
    columns: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Parse a labels CSV against explicitly configured image directories."""

    root_path = Path(root).resolve()
    labels_path = _resolve_config_path(labels_fn, root_path)
    if not labels_path.is_file():
        raise FileNotFoundError(f"labels CSV not found: {labels_path}")

    columns = columns or {}
    test_case_col = columns.get("test_case", "test_case")
    method_col = columns.get("method", "method")
    score_col = columns.get("score", "score")
    image_col = columns.get("image", "image")

    sr_dirs = directories.get("sr")
    if not isinstance(sr_dirs, Mapping) or not sr_dirs:
        raise ValueError("Directory datasets must define directories.sr as a method-to-directory object")

    def resolve_dir(raw: Any) -> Path:
        return _resolve_config_path(str(raw), root_path)

    sr_indexes = {str(name): ImageIndex(str(resolve_dir(path))) for name, path in sr_dirs.items()}
    lr_dir = directories.get("lr")
    if not lr_dir:
        raise ValueError("Directory datasets must define directories.lr")
    lr_index = ImageIndex(str(resolve_dir(lr_dir)))
    hr_index = ImageIndex(str(resolve_dir(directories["hr"]))) if directories.get("hr") else None

    ref_dirs = directories.get("refs", {})
    if ref_dirs is None:
        ref_dirs = {}
    if not isinstance(ref_dirs, Mapping):
        raise ValueError("directories.refs must be an object")
    ref_indexes = {str(name): [ImageIndex(str(resolve_dir(path)))] for name, path in ref_dirs.items()}

    heatmap_dirs = directories.get("heatmaps", {})
    if isinstance(heatmap_dirs, (str, os.PathLike)):
        heatmap_dirs = {"default": heatmap_dirs}
    if not isinstance(heatmap_dirs, Mapping):
        raise ValueError("directories.heatmaps must be a path or method-to-directory object")

    samples: list[dict[str, Any]] = []
    with labels_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = [str(field).strip().lstrip("\ufeff") for field in (reader.fieldnames or [])]
        missing = [name for name in (test_case_col, method_col, score_col) if name not in fields]
        if missing:
            raise ValueError(f"labels CSV is missing columns: {missing}")

        for row_number, raw_row in enumerate(reader, start=2):
            row = _clean_row(raw_row)
            if not any(row.values()):
                continue
            test_case = row.get(test_case_col, "")
            method = row.get(method_col, "")
            score_text = row.get(score_col, "")
            image_value = row.get(image_col, "") if image_col in row else ""
            if not test_case or not method or not score_text:
                raise ValueError(f"Invalid labels CSV row {row_number}: {row}")

            sr_index = _mapping_value(sr_indexes, method)
            if sr_index is None:
                raise KeyError(f"No SR directory configured for method '{method}'")
            try:
                sr_path = _directory_index_path(sr_index, test_case, image_value)
                sr_ext = os.path.splitext(sr_path)[1]
                lr_path = _directory_index_path(lr_index, test_case, image_value, sr_ext)
                hr_path = (
                    _directory_index_path(hr_index, test_case, image_value, sr_ext)
                    if hr_index is not None
                    else None
                )
                score = float(score_text)
            except Exception as exc:
                raise RuntimeError(f"Failed to parse labels CSV row {row_number}: {row}") from exc

            item: dict[str, Any] = {
                "dataset": dataset_name,
                "test_case": test_case,
                "method": method,
                "rel_path": f"{method}/{os.path.basename(sr_path)}",
                "hr_path": hr_path,
                "lr_path": lr_path,
                "sr_path": sr_path,
                "score": score,
            }
            ref_paths: dict[str, str] = {}
            for ref_name, indexes in ref_indexes.items():
                found = _find_reference_path(
                    item, indexes, ref_name, REFERENCE_SUFFIXES.get(ref_name, ref_name)
                )
                if found:
                    ref_paths[ref_name] = found
                    item[f"{ref_name}_path"] = found
            item["ref_paths"] = ref_paths

            heatmap_dir = _mapping_value(heatmap_dirs, method)
            if heatmap_dir:
                heatmap_root = resolve_dir(heatmap_dir)
                stem = Path(sr_path).stem
                candidates = [heatmap_root / f"{stem}.npy.gz", heatmap_root / f"{stem}.npy"]
                item["heatmap_path"] = str(
                    next((path for path in candidates if path.is_file()), candidates[0])
                )
            samples.append(item)

    return normalize_sample_scores(samples)


def _load_custom_parser(path: Path, function_name: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Custom dataset parser not found: {path}")
    module_name = f"qualisr_user_dataset_{path.stem}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load custom dataset parser: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = getattr(module, function_name, None)
    if not callable(parser):
        raise AttributeError(f"Custom parser '{path}' has no callable '{function_name}'")
    return parser


def _normalized_relative_sr_path(sample: Mapping[str, Any], root: Path) -> Path:
    raw = str(sample.get("rel_path") or "").replace("\\", "/").lstrip("/")
    if raw:
        rel = Path(raw)
    else:
        sr_path = Path(str(sample["sr_path"]))
        try:
            rel = sr_path.resolve().relative_to(root)
        except ValueError:
            rel = Path(sr_path.name)

    parts = list(rel.parts)
    if parts and parts[0].casefold() == "sr":
        parts = parts[1:]
    return Path(*parts)


def normalize_samples(
    samples: Iterable[Mapping[str, Any]],
    *,
    dataset_name: str,
    root: str | Path,
) -> list[dict[str, Any]]:
    """Normalize and validate parser output into the shared pipeline contract."""

    root_path = Path(root).expanduser().resolve()
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_sample in enumerate(samples):
        if not isinstance(raw_sample, Mapping):
            raise TypeError(f"Dataset '{dataset_name}' sample #{index} is not a dictionary")
        sample = dict(raw_sample)
        sample["dataset"] = dataset_name
        for key in ("test_case", "method", "sr_path", "lr_path", "score"):
            if sample.get(key) in (None, ""):
                raise ValueError(f"Dataset '{dataset_name}' sample #{index} is missing '{key}'")

        score = float(sample["score"])
        # if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        #     raise ValueError(f"Dataset '{dataset_name}' sample #{index} has score outside [0, 1]: {score}")
        sample["score"] = score

        for key in ("sr_path", "lr_path", "hr_path", "heatmap_path"):
            value = sample.get(key)
            if value:
                sample[key] = str(_resolve_config_path(str(value), root_path))
        for key in ("sr_path", "lr_path"):
            if not Path(sample[key]).is_file():
                raise FileNotFoundError(f"Dataset '{dataset_name}' sample #{index} {key}: {sample[key]}")
        if sample.get("hr_path") and not Path(sample["hr_path"]).is_file():
            raise FileNotFoundError(f"Dataset '{dataset_name}' sample #{index} hr_path: {sample['hr_path']}")

        ref_paths = dict(sample.get("ref_paths") or {})
        for ref_name in REFERENCE_TYPES:
            flat_path = sample.get(f"{ref_name}_path")
            if flat_path:
                ref_paths.setdefault(ref_name, flat_path)
        sample["ref_paths"] = {
            str(name): str(_resolve_config_path(str(path), root_path))
            for name, path in ref_paths.items()
            if path
        }
        for ref_name, path in sample["ref_paths"].items():
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f"Dataset '{dataset_name}' sample #{index} reference '{ref_name}': {path}"
                )
            sample[f"{ref_name}_path"] = path

        relative_sr = _normalized_relative_sr_path(sample, root_path)
        sample["rel_path"] = relative_sr.as_posix()
        heatmap_rel = relative_sr.with_suffix("")
        heatmap_rel = heatmap_rel.parent / f"{heatmap_rel.name}.npy.gz"
        if not sample.get("heatmap_path"):
            sample["heatmap_path"] = str((root_path / "heatmaps" / heatmap_rel).resolve())
        identity_rel = heatmap_rel
        method = str(sample["method"])
        if not identity_rel.parts or identity_rel.parts[0].casefold() != method.casefold():
            identity_rel = Path(method) / identity_rel
        sample["sample_id"] = f"{sample['dataset']}/{identity_rel.as_posix()}"
        sample["dataset_root"] = str(root_path)

        if sample["sample_id"] in seen:
            raise ValueError(f"Duplicate sample_id: {sample['sample_id']}")
        seen.add(sample["sample_id"])
        normalized.append(sample)
    return normalized


def load_dataset(entry: Mapping[str, Any], base_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Load one built-in, custom-file, or explicit-directory dataset entry."""

    if not isinstance(entry, Mapping):
        raise TypeError("Each datasets entry must be an object")
    name = str(entry.get("name") or "").strip()
    if not name:
        raise ValueError("Each datasets entry must define a non-empty name")
    base_path = Path.cwd() if base_dir is None else Path(base_dir)
    if not entry.get("root"):
        raise ValueError(f"Dataset '{name}' must define root")
    root = _resolve_config_path(str(entry["root"]), base_path)
    if not entry.get("features_root"):
        raise ValueError(f"Dataset '{name}' must define features_root")
    features_root = _resolve_config_path(str(entry["features_root"]), base_path)
    if "regressors" not in entry:
        raise ValueError(f"Dataset '{name}' must define regressors usage")
    regressors = entry["regressors"]
    if not isinstance(regressors, Mapping):
        raise ValueError(f"Dataset '{name}' regressors must be an object")
    for role in ("train", "validate"):
        if not isinstance(regressors.get(role), bool):
            raise ValueError(f"Dataset '{name}' regressors.{role} must be a boolean")
    if regressors["train"]:
        if "test_size" not in regressors:
            raise ValueError(f"Training dataset '{name}' must define regressors.test_size")
        test_size = float(regressors["test_size"])
        if not 0.0 <= test_size < 1.0:
            raise ValueError(f"Dataset '{name}' regressors.test_size must be in [0, 1)")
        if not regressors["validate"] and test_size != 0.0:
            raise ValueError(f"Training-only dataset '{name}' must set regressors.test_size to 0")
    elif "test_size" in regressors:
        raise ValueError(f"Non-training dataset '{name}' must not define regressors.test_size")
    kwargs = entry.get("kwargs", {}) or {}
    if not isinstance(kwargs, Mapping):
        raise ValueError(f"Dataset '{name}' kwargs must be an object")

    parser_cfg = entry.get("parser")
    directory_cfg = entry.get("directories")
    if parser_cfg is not None and directory_cfg is not None:
        raise ValueError(f"Dataset '{name}' cannot define both parser and directories")

    if parser_cfg is not None:
        if not isinstance(parser_cfg, Mapping):
            raise ValueError(f"Dataset '{name}' parser must define path and function")
        parser_path = parser_cfg.get("path")
        function_name = str(parser_cfg.get("function") or "parse_dataset")
        if not parser_path:
            raise ValueError(f"Dataset '{name}' custom parser is missing path")
        parser = _load_custom_parser(_resolve_config_path(str(parser_path), base_path), function_name)
        parsed = parser(str(root), **dict(kwargs))
    elif directory_cfg is not None:
        labels = entry.get("labels")
        if not labels:
            raise ValueError(f"Directory dataset '{name}' must define labels")
        parsed = parse_directory_dataset(
            dataset_name=name,
            root=str(root),
            labels_fn=str(labels),
            directories=directory_cfg,
            columns=entry.get("columns"),
        )
    else:
        parsed = parse_iqa_dataset(
            name,
            str(root),
            labels_fn=str(entry.get("labels") or ""),
            **dict(kwargs),
        )
    if isinstance(parsed, (str, bytes)) or not isinstance(parsed, Sequence):
        raise TypeError(f"Dataset parser '{name}' must return a list of dictionaries")
    samples = normalize_samples(parsed, dataset_name=name, root=root)
    configured_score_type = entry.get("score_type")
    if configured_score_type is not None and configured_score_type not in SCORE_TYPES:
        raise ValueError(
            f"Dataset '{name}' score_type must be one of {SCORE_TYPES}: {configured_score_type}"
        )
    for sample in samples:
        score_type = str(configured_score_type or sample.get("score_type", "mos"))
        if score_type not in SCORE_TYPES:
            raise ValueError(
                f"Dataset '{name}' parser returned unsupported score_type '{score_type}'"
            )
        if score_type == "bradley_terry":
            if not sample.get("correlation_group"):
                raise ValueError(
                    f"Bradley-Terry dataset '{name}' samples must define correlation_group"
                )
        else:
            sample["correlation_group"] = str(sample["test_case"])
        sample["score_type"] = score_type
        sample["features_root"] = str(features_root)
        sample["regressors"] = dict(regressors)
    return samples


def load_datasets(
    entries: Sequence[Mapping[str, Any]],
    base_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load and concatenate all configured datasets."""

    if not entries:
        raise ValueError("Pipeline config must define at least one dataset")
    names: set[str] = set()
    samples: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    for entry in entries:
        name = str(entry.get("name") or "")
        if name in names:
            raise ValueError(f"Duplicate dataset name: {name}")
        names.add(name)
        current = load_dataset(entry, base_dir=base_dir)
        duplicates = sample_ids.intersection(sample["sample_id"] for sample in current)
        if duplicates:
            raise ValueError(f"Duplicate sample_id across datasets: {sorted(duplicates)[0]}")
        sample_ids.update(sample["sample_id"] for sample in current)
        samples.extend(current)
    return samples
