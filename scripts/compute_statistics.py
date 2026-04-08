import argparse
import gzip
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


LOGGER = logging.getLogger("compute_statistics")
DEFAULT_PERCENTILES: Sequence[float] = (5.0, 95.0)
DEFAULT_AREA_THRESHOLDS: Sequence[float] = (0.0, 0.5, 0.75)
DEFAULT_EXTENSIONS: Sequence[str] = (".npy", ".gz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute summary statistics for heatmaps stored as .npy or .gz(.npy) files. "
            "Input dirs can be passed as PREFIX=DIR for stable sample naming."
        )
    )

    parser.add_argument(
        "--heatmap-dirs",
        nargs="+",
        required=True,
        metavar="PREFIX=DIR",
        help=(
            "One or more heatmap directories. "
            "Examples: PASD=/data/heatmaps/pasd SUPIR=/data/heatmaps/supir"
        ),
    )
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--percentiles",
        nargs="+",
        type=float,
        default=list(DEFAULT_PERCENTILES),
        help="Percentiles to include as columns. Example: --percentiles 1 5 95 99",
    )
    parser.add_argument(
        "--area-thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_AREA_THRESHOLDS),
        help="Area ratio thresholds. For t=0 uses heatmap>0, else heatmap>=t.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        help="Allowed file endings. Matched with endswith; defaults: .npy .gz",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan directories recursively.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on the first unreadable/invalid heatmap instead of skipping it.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )

    return parser.parse_args()


def parse_named_directories(specs: Iterable[str]) -> Dict[str, Path]:
    parsed: Dict[str, Path] = {}

    for spec in specs:
        if "=" in spec:
            prefix, raw_path = spec.split("=", 1)
            prefix = prefix.strip()
            raw_path = raw_path.strip()
        else:
            raw_path = spec.strip()
            prefix = Path(raw_path).name

        if not raw_path:
            raise ValueError(f"Invalid --heatmap-dirs value '{spec}': missing path")

        if prefix in parsed:
            raise ValueError(f"Duplicate prefix '{prefix}' in --heatmap-dirs")

        parsed[prefix] = Path(raw_path).expanduser().resolve()

    return parsed


def require_existing_directories(named_dirs: Dict[str, Path]) -> None:
    for prefix, directory in named_dirs.items():
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"Heatmap directory for prefix '{prefix}' does not exist: {directory}")


def normalize_extensions(extensions: Sequence[str]) -> List[str]:
    normalized = []
    for ext in extensions:
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        normalized.append(ext)

    if not normalized:
        raise ValueError("At least one extension must be provided.")

    return sorted(set(normalized))


def is_allowed_heatmap(path: Path, extensions: Sequence[str]) -> bool:
    name = path.name.lower()
    return any(name.endswith(ext) for ext in extensions)


def list_heatmap_files(directory: Path, recursive: bool, extensions: Sequence[str]) -> List[Path]:
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    files = [path for path in iterator if path.is_file() and is_allowed_heatmap(path, extensions)]
    files.sort(key=lambda p: str(p.relative_to(directory)).lower())
    return files


def load_heatmap(path: Path) -> np.ndarray:
    if path.name.lower().endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            heatmap = np.load(handle, allow_pickle=False)
    else:
        heatmap = np.load(path, allow_pickle=False)

    heatmap = np.asarray(heatmap)
    if heatmap.size == 0:
        raise ValueError("empty heatmap")

    return heatmap


def format_percentile_name(percentile: float) -> str:
    rounded = round(percentile)
    if np.isclose(percentile, rounded):
        return f"p{int(rounded):02d}"

    text = str(percentile).replace(".", "_")
    return f"p{text}"


def format_area_name(threshold: float) -> str:
    rounded = round(threshold)
    if np.isclose(threshold, rounded):
        return f"area{int(rounded):02d}"

    text = str(threshold)
    if text.startswith("0."):
        return "area0" + text[2:]

    return "area" + text.replace(".", "_")


def compute_statistics_row(
    name: str,
    heatmap: np.ndarray,
    percentiles: Sequence[float],
    area_thresholds: Sequence[float],
) -> Dict[str, float]:
    row: Dict[str, float] = {
        "name": name,
        "min": float(np.min(heatmap)),
        "max": float(np.max(heatmap)),
        "mean": float(np.mean(heatmap)),
        "median": float(np.median(heatmap)),
        "std": float(np.std(heatmap)),
    }

    for percentile in percentiles:
        row[format_percentile_name(percentile)] = float(np.percentile(heatmap, percentile))

    size = heatmap.size
    for threshold in area_thresholds:
        if np.isclose(threshold, 0.0):
            ratio = float(np.count_nonzero(heatmap > 0.0) / size)
        else:
            ratio = float(np.count_nonzero(heatmap >= threshold) / size)
        row[format_area_name(threshold)] = ratio

    return row


def build_output_columns(percentiles: Sequence[float], area_thresholds: Sequence[float]) -> List[str]:
    base_cols = ["name", "min", "max", "mean", "median", "std"]
    percentile_cols = [format_percentile_name(p) for p in percentiles]
    area_cols = [format_area_name(t) for t in area_thresholds]
    return base_cols + percentile_cols + area_cols


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    named_dirs = parse_named_directories(args.heatmap_dirs)
    require_existing_directories(named_dirs)

    percentiles = list(args.percentiles)
    area_thresholds = list(args.area_thresholds)
    if not percentiles:
        raise ValueError("--percentiles cannot be empty")
    if not area_thresholds:
        raise ValueError("--area-thresholds cannot be empty")

    extensions = normalize_extensions(args.extensions)

    rows: List[Dict[str, float]] = []
    for prefix, directory in named_dirs.items():
        files = list_heatmap_files(directory, recursive=args.recursive, extensions=extensions)
        LOGGER.info("%s: found %d heatmaps in %s", prefix, len(files), directory)

        progress = tqdm(files, desc=prefix or directory.name, unit="file", disable=args.no_progress)
        for path in progress:
            relative_name = path.relative_to(directory).as_posix()
            sample_name = f"{prefix}/{relative_name}" if prefix else relative_name

            try:
                heatmap = load_heatmap(path)
            except Exception as exc:
                message = f"Failed to load heatmap {path}: {exc}"
                if args.strict:
                    raise RuntimeError(message) from exc
                LOGGER.warning(message)
                continue

            try:
                row = compute_statistics_row(
                    name=sample_name,
                    heatmap=heatmap,
                    percentiles=percentiles,
                    area_thresholds=area_thresholds,
                )
            except Exception as exc:
                message = f"Failed to compute stats for {path}: {exc}"
                if args.strict:
                    raise RuntimeError(message) from exc
                LOGGER.warning(message)
                continue

            rows.append(row)

    columns = build_output_columns(percentiles, area_thresholds)
    output_df = pd.DataFrame(rows, columns=columns)
    output_df.sort_values("name", inplace=True)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    LOGGER.info("Saved %d rows to %s", len(output_df), output_path)


if __name__ == "__main__":
    main()
