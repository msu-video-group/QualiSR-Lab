"""Compute signed differences between paired SR and reference embeddings."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("embedding_difference")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute signed SR-minus-reference embedding differences from two CSV files."
    )
    parser.add_argument("--reference-input", required=True, help="Reference embedding CSV.")
    parser.add_argument("--sr-input", required=True, help="SR embedding CSV.")
    parser.add_argument(
        "--blocks",
        nargs="+",
        required=True,
        metavar="OUTPUT=REFERENCE_PREFIX,SR_PREFIX",
        help=(
            "Embedding blocks to subtract. OUTPUT is the output column prefix; "
            "the reference and SR prefixes select corresponding columns."
        ),
    )
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args(argv)


def parse_blocks(specs: Sequence[str]) -> list[tuple[str, str, str]]:
    blocks: list[tuple[str, str, str]] = []
    seen_outputs: set[str] = set()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --blocks value '{spec}'. Expected OUTPUT=REFERENCE_PREFIX,SR_PREFIX."
            )
        output_prefix, input_prefixes = spec.split("=", 1)
        prefixes = input_prefixes.split(",")
        output_prefix = output_prefix.strip().rstrip("_")
        if len(prefixes) != 2:
            raise ValueError(
                f"Invalid --blocks value '{spec}'. Expected exactly two input prefixes."
            )
        reference_prefix, sr_prefix = (prefix.strip() for prefix in prefixes)
        if not output_prefix or not reference_prefix or not sr_prefix:
            raise ValueError(f"Invalid --blocks value '{spec}': prefixes must be non-empty.")
        if output_prefix in seen_outputs:
            raise ValueError(f"Duplicate output prefix '{output_prefix}' in --blocks.")
        seen_outputs.add(output_prefix)
        blocks.append((output_prefix, reference_prefix, sr_prefix))
    return blocks


def load_embedding_csv(path_value: str, label: str) -> tuple[Path, pd.DataFrame]:
    path = Path(path_value).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{label} CSV not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{label} CSV is empty.")
    if "sample_id" not in frame.columns:
        raise ValueError(f"{label} CSV must contain a sample_id column.")
    frame["sample_id"] = frame["sample_id"].astype(str)
    duplicate_mask = frame["sample_id"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate = frame.loc[duplicate_mask, "sample_id"].iloc[0]
        raise ValueError(f"{label} CSV contains duplicate sample_id '{duplicate}'.")
    return path, frame


def aligned_reference_frame(reference_df: pd.DataFrame, sr_df: pd.DataFrame) -> pd.DataFrame:
    reference_ids = set(reference_df["sample_id"])
    sr_ids = set(sr_df["sample_id"])
    if reference_ids != sr_ids:
        missing_reference = sorted(sr_ids - reference_ids)
        missing_sr = sorted(reference_ids - sr_ids)
        raise ValueError(
            "Embedding inputs must contain identical sample_id sets. "
            f"Missing from reference: {missing_reference[:3]}; missing from SR: {missing_sr[:3]}."
        )
    return reference_df.set_index("sample_id").loc[sr_df["sample_id"]].reset_index()


def block_columns(
    frame: pd.DataFrame,
    prefix: str,
    label: str,
    output_prefix: str,
) -> list[str]:
    columns = [column for column in frame.columns if column.startswith(prefix)]
    if not columns:
        raise ValueError(
            f"No {label} columns found for block '{output_prefix}' with prefix '{prefix}'."
        )
    return columns


def compute_differences(
    reference_df: pd.DataFrame,
    sr_df: pd.DataFrame,
    blocks: Sequence[tuple[str, str, str]],
) -> pd.DataFrame:
    aligned_reference = aligned_reference_frame(reference_df, sr_df)
    selected_sr_columns: list[str] = []
    difference_frames: list[pd.DataFrame] = []

    for output_prefix, reference_prefix, sr_prefix in blocks:
        reference_columns = block_columns(
            aligned_reference, reference_prefix, "reference", output_prefix
        )
        sr_columns = block_columns(sr_df, sr_prefix, "SR", output_prefix)
        reference_suffixes = [column[len(reference_prefix) :] for column in reference_columns]
        sr_suffixes = [column[len(sr_prefix) :] for column in sr_columns]
        if reference_suffixes != sr_suffixes:
            raise ValueError(
                f"Block '{output_prefix}' has mismatched reference and SR component suffixes."
            )
        overlap = set(selected_sr_columns).intersection(sr_columns)
        if overlap:
            raise ValueError(f"SR embedding columns selected by multiple blocks: {sorted(overlap)}")
        selected_sr_columns.extend(sr_columns)

        try:
            reference_values = aligned_reference[reference_columns].to_numpy(dtype=np.float64)
            sr_values = sr_df[sr_columns].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Block '{output_prefix}' contains non-numeric values.") from exc

        output_columns = [f"{output_prefix}_{suffix}" for suffix in sr_suffixes]
        difference_frames.append(pd.DataFrame(sr_values - reference_values, columns=output_columns))

    output = sr_df.drop(columns=selected_sr_columns).copy()
    return pd.concat([output, *difference_frames], axis=1)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    blocks = parse_blocks(args.blocks)
    reference_path, reference_df = load_embedding_csv(args.reference_input, "Reference input")
    sr_path, sr_df = load_embedding_csv(args.sr_input, "SR input")
    output = compute_differences(reference_df, sr_df, blocks)

    output_path = Path(args.output).expanduser().resolve()
    if output_path in {reference_path, sr_path}:
        raise ValueError("--output must not overwrite either embedding input CSV.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    LOGGER.info("Saved %d embedding-difference rows to %s", len(output), output_path)


if __name__ == "__main__":
    main()
