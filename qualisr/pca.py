import argparse
import logging
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit

LOGGER = logging.getLogger("apply_pca_image_features")
DEFAULT_BLOCKS: Sequence[str] = ("vgg=vgg_", "resnet=resnet_")
PATH_COLUMNS: Sequence[str] = ("sr_path", "gt_path", "lr_path")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply PCA to high-dimensional feature blocks (e.g. vgg_*, resnet_*) "
            "in CSV files created by qualisr-extract-features."
        )
    )

    parser.add_argument("--input", required=True, help="Input CSV from qualisr-extract-features")
    parser.add_argument(
        "--reference-input",
        default=None,
        help=(
            "Optional reference-embedding CSV. When provided, PCA is fit on stacked "
            "SR and reference training rows and the same basis transforms both inputs."
        ),
    )
    parser.add_argument(
        "--n-components",
        nargs="+",
        required=True,
        type=int,
        help="One or more PCA dimensions to generate (example: --n-components 5 10 25 50)",
    )
    parser.add_argument(
        "--blocks",
        nargs="+",
        default=list(DEFAULT_BLOCKS),
        metavar="NAME=PREFIX",
        help=(
            "Feature blocks to reduce. NAME is used in output column names, "
            "PREFIX selects columns by startswith. Default: vgg=vgg_ resnet=resnet_"
        ),
    )
    parser.add_argument(
        "--reference-blocks",
        nargs="+",
        default=None,
        metavar="NAME=PREFIX",
        help=(
            "Blocks in --reference-input corresponding by NAME to --blocks. "
            "Required with --reference-input."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output files. Default: input file directory.",
    )
    parser.add_argument(
        "--output-template",
        default="{stem}_pca{n}.csv",
        help="Output file template. Available fields: {stem}, {n}",
    )
    parser.add_argument(
        "--reference-output-template",
        default=None,
        help=(
            "Output template for the transformed reference CSV. Available fields: {stem}, {n}. "
            "Default: --output-template evaluated with the reference input stem."
        ),
    )
    parser.add_argument(
        "--keep-original-blocks",
        action="store_true",
        help="Keep original block columns (vgg_*/resnet_*) in output.",
    )
    parser.add_argument(
        "--fit-column",
        default=None,
        help="Optional column name for selecting rows to fit PCA.",
    )
    parser.add_argument(
        "--fit-value",
        default=None,
        help="Optional value for --fit-column (example: train). If omitted, fit on all rows.",
    )
    parser.add_argument(
        "--disable-auto-split",
        action="store_true",
        help=(
            "Disable automatic grouped train/test split when --fit-column is not provided. "
            "If set, PCA is fit on all rows (unless --fit-column is used)."
        ),
    )
    parser.add_argument(
        "--split-column",
        default="set_type",
        help="Column name to write automatic split labels into (default: set_type).",
    )
    parser.add_argument(
        "--train-label",
        default="train",
        help="Label used for train rows in automatic split.",
    )
    parser.add_argument(
        "--test-label",
        default="test",
        help="Label used for test rows in automatic split.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of GT groups assigned to test set in automatic split.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for automatic grouped train/test split.",
    )
    parser.add_argument(
        "--group-column",
        default="gt_path",
        help="Primary column for GT grouping in automatic split (default: gt_path).",
    )
    parser.add_argument(
        "--group-fallback-column",
        default="sr_filename",
        help=(
            "Fallback column used when --group-column is missing/empty. "
            "Default uses SR filename stem."
        ),
    )
    parser.add_argument(
        "--svd-solver",
        choices=("auto", "full", "covariance_eigh", "arpack", "randomized"),
        default="full",
        help="scikit-learn PCA solver.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )

    return parser.parse_args(argv)


def parse_blocks(block_specs: Sequence[str]) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    seen_names = set()

    for spec in block_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --blocks value '{spec}'. Expected NAME=PREFIX.")

        name, prefix = spec.split("=", 1)
        name = name.strip()
        prefix = prefix.strip()

        if not name or not prefix:
            raise ValueError(f"Invalid --blocks value '{spec}'. NAME and PREFIX must be non-empty.")

        if name in seen_names:
            raise ValueError(f"Duplicate block name '{name}' in --blocks.")

        seen_names.add(name)
        blocks.append((name, prefix))

    return blocks


def get_fit_mask(df: pd.DataFrame, fit_column: str, fit_value: str) -> np.ndarray:
    if fit_column not in df.columns:
        raise ValueError(f"--fit-column '{fit_column}' does not exist in input CSV.")

    if fit_value is None:
        raise ValueError("--fit-value is required when --fit-column is used.")

    mask = df[fit_column].astype(str) == str(fit_value)
    if not mask.any():
        raise ValueError(
            f"No rows match fit filter: {fit_column} == {fit_value}. "
            "Cannot fit PCA on empty subset."
        )

    return mask.to_numpy()


def validate_n_components(n_components: Sequence[int]) -> list[int]:
    parsed = sorted(set(n_components))
    if not parsed:
        raise ValueError("No n_components provided.")

    invalid = [n for n in parsed if n <= 0]
    if invalid:
        raise ValueError(f"n_components must be > 0. Invalid values: {invalid}")

    return parsed


def normalize_group_key(value: object) -> str | None:
    if pd.isna(value):
        return None

    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        return None

    return raw


def fallback_group_key_from_value(value: object) -> str | None:
    normalized = normalize_group_key(value)
    if normalized is None:
        return None

    return Path(normalized).stem


def relativize_path_value(value: object) -> object:
    if pd.isna(value):
        return value

    raw = str(value).strip()
    if not raw:
        return value

    path = Path(raw).expanduser()
    if not path.is_absolute():
        return raw

    try:
        return Path(os.path.relpath(path, start=Path.cwd())).as_posix()
    except ValueError:
        return path.as_posix()


def relativize_path_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in PATH_COLUMNS:
        if column in df.columns:
            df[column] = df[column].map(relativize_path_value)
    return df


def build_group_keys(
    df: pd.DataFrame,
    group_column: str,
    group_fallback_column: str,
) -> np.ndarray:
    group_keys: list[str | None] = [None] * len(df)

    if group_column in df.columns:
        for i, value in enumerate(df[group_column]):
            group_keys[i] = normalize_group_key(value)
    else:
        LOGGER.warning(
            "Group column '%s' is missing. Falling back to '%s'.",
            group_column,
            group_fallback_column,
        )

    if group_fallback_column not in df.columns and any(key is None for key in group_keys):
        raise ValueError(
            f"Some rows have no group key from '{group_column}', and fallback column "
            f"'{group_fallback_column}' does not exist."
        )

    if group_fallback_column in df.columns:
        fallback_values = df[group_fallback_column].tolist()
        for i, key in enumerate(group_keys):
            if key is not None:
                continue
            group_keys[i] = fallback_group_key_from_value(fallback_values[i])

    unresolved = [i for i, key in enumerate(group_keys) if key is None]
    if unresolved:
        raise ValueError(
            f"Failed to derive GT group keys for {len(unresolved)} rows. "
            f"Check '{group_column}' or '{group_fallback_column}'."
        )

    return np.asarray(group_keys, dtype=object)


def make_grouped_split(
    df: pd.DataFrame,
    split_column: str,
    group_column: str,
    group_fallback_column: str,
    test_size: float,
    split_seed: int,
    train_label: str,
    test_label: str,
) -> tuple[np.ndarray, pd.Series, np.ndarray]:
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"--test-size must be in (0, 1). Got {test_size}.")

    group_keys = build_group_keys(df, group_column, group_fallback_column)
    unique_groups = np.unique(group_keys)
    if unique_groups.size < 2:
        raise ValueError(
            "Automatic split requires at least 2 distinct GT groups. "
            f"Found {unique_groups.size}."
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=split_seed)
    all_indices = np.arange(len(df))
    train_idx, test_idx = next(splitter.split(all_indices, groups=group_keys))

    labels = np.full(len(df), test_label, dtype=object)
    labels[train_idx] = train_label
    labels_series = pd.Series(labels, index=df.index, name=split_column)

    check = pd.DataFrame({"group": group_keys, "split": labels})
    leaks = check.groupby("group")["split"].nunique()
    leaked_groups = leaks[leaks > 1]
    if not leaked_groups.empty:
        raise RuntimeError(
            "Split leakage detected: some GT groups are assigned to both train and test."
        )

    fit_mask = labels_series == train_label
    return fit_mask.to_numpy(), labels_series, group_keys


def load_input_csv(path_value: str, label: str) -> tuple[Path, pd.DataFrame]:
    path = Path(path_value).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{label} CSV not found: {path}")

    LOGGER.info("Loading %s CSV: %s", label, path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{label} CSV is empty.")
    return path, frame


def collect_block_columns(
    df: pd.DataFrame,
    blocks: Sequence[tuple[str, str]],
    label: str,
) -> dict[str, list[str]]:
    block_columns: dict[str, list[str]] = {}
    for block_name, prefix in blocks:
        cols = [col for col in df.columns if col.startswith(prefix)]
        if not cols:
            raise ValueError(
                f"No columns found for {label} block '{block_name}' with prefix '{prefix}'. "
                "Check the configured blocks or input CSV."
            )
        block_columns[block_name] = cols
        LOGGER.info(
            "%s block '%s': %d columns (prefix='%s')",
            label.capitalize(),
            block_name,
            len(cols),
            prefix,
        )
    return block_columns


def validate_paired_samples(sr_df: pd.DataFrame, reference_df: pd.DataFrame) -> None:
    for label, frame in (("SR", sr_df), ("reference", reference_df)):
        if "sample_id" not in frame.columns:
            raise ValueError(f"{label} CSV must contain a sample_id column for paired PCA.")
        duplicate_mask = frame["sample_id"].astype(str).duplicated(keep=False)
        if duplicate_mask.any():
            duplicate = frame.loc[duplicate_mask, "sample_id"].astype(str).iloc[0]
            raise ValueError(f"{label} CSV contains duplicate sample_id '{duplicate}'.")

    sr_ids = set(sr_df["sample_id"].astype(str))
    reference_ids = set(reference_df["sample_id"].astype(str))
    if sr_ids != reference_ids:
        missing_reference = sorted(sr_ids - reference_ids)
        missing_sr = sorted(reference_ids - sr_ids)
        raise ValueError(
            "Paired PCA inputs must contain identical sample_id sets. "
            f"Missing from reference: {missing_reference[:3]}; missing from SR: {missing_sr[:3]}."
        )


def validate_paired_blocks(
    sr_blocks: Sequence[tuple[str, str]],
    reference_blocks: Sequence[tuple[str, str]],
    sr_columns: dict[str, list[str]],
    reference_columns: dict[str, list[str]],
) -> None:
    sr_prefixes = dict(sr_blocks)
    reference_prefixes = dict(reference_blocks)
    if set(sr_prefixes) != set(reference_prefixes):
        raise ValueError(
            "--blocks and --reference-blocks must define the same block names. "
            f"SR={sorted(sr_prefixes)}, reference={sorted(reference_prefixes)}"
        )

    for block_name in sr_prefixes:
        sr_cols = sr_columns[block_name]
        reference_cols = reference_columns[block_name]
        if len(sr_cols) != len(reference_cols):
            raise ValueError(
                f"Paired block '{block_name}' has {len(sr_cols)} SR columns but "
                f"{len(reference_cols)} reference columns."
            )
        sr_suffixes = [column[len(sr_prefixes[block_name]) :] for column in sr_cols]
        reference_suffixes = [
            column[len(reference_prefixes[block_name]) :] for column in reference_cols
        ]
        if sr_suffixes != reference_suffixes:
            raise ValueError(
                f"Paired block '{block_name}' has mismatched component suffixes between "
                "SR and reference inputs."
            )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    input_path, df = load_input_csv(args.input, "SR input")
    paired_mode = args.reference_input is not None
    if paired_mode != (args.reference_blocks is not None):
        raise ValueError("--reference-input and --reference-blocks must be provided together.")
    if args.reference_output_template is not None and not paired_mode:
        raise ValueError("--reference-output-template requires --reference-input.")

    reference_path: Path | None = None
    reference_df: pd.DataFrame | None = None
    reference_blocks: list[tuple[str, str]] = []
    if paired_mode:
        reference_path, reference_df = load_input_csv(args.reference_input, "reference input")
        reference_blocks = parse_blocks(args.reference_blocks)
        validate_paired_samples(df, reference_df)

    n_values = validate_n_components(args.n_components)
    max_n = max(n_values)

    blocks = parse_blocks(args.blocks)

    fit_mask: np.ndarray
    split_labels: pd.Series | None = None
    if args.fit_column is not None:
        fit_mask = get_fit_mask(df, args.fit_column, args.fit_value)
    elif args.disable_auto_split:
        fit_mask = np.ones(len(df), dtype=bool)
    else:
        fit_mask, split_labels, group_keys = make_grouped_split(
            df=df,
            split_column=args.split_column,
            group_column=args.group_column,
            group_fallback_column=args.group_fallback_column,
            test_size=args.test_size,
            split_seed=args.split_seed,
            train_label=args.train_label,
            test_label=args.test_label,
        )
        df[args.split_column] = split_labels

        train_rows = int((split_labels == args.train_label).sum())
        test_rows = int((split_labels == args.test_label).sum())
        train_groups = int(pd.Series(group_keys[split_labels == args.train_label]).nunique())
        test_groups = int(pd.Series(group_keys[split_labels == args.test_label]).nunique())
        LOGGER.info(
            "Auto split complete (seed=%d): train rows=%d, test rows=%d, train groups=%d, test groups=%d",
            args.split_seed,
            train_rows,
            test_rows,
            train_groups,
            test_groups,
        )

    reference_fit_mask: np.ndarray | None = None
    if reference_df is not None:
        fit_ids = set(df.loc[fit_mask, "sample_id"].astype(str))
        reference_fit_mask = reference_df["sample_id"].astype(str).isin(fit_ids).to_numpy()
        copied_column: str | None = None
        copied_values: pd.Series | None = None
        if args.fit_column is not None:
            copied_column = args.fit_column
            copied_values = df[args.fit_column]
        elif split_labels is not None:
            copied_column = args.split_column
            copied_values = split_labels
        if copied_column is not None and copied_values is not None:
            values_by_id = pd.Series(
                copied_values.to_numpy(),
                index=df["sample_id"].astype(str),
            )
            reference_df[copied_column] = (
                reference_df["sample_id"].astype(str).map(values_by_id).to_numpy()
            )

    LOGGER.info("Rows: %d (fit rows: %d)", len(df), int(fit_mask.sum()))

    block_columns = collect_block_columns(df, blocks, "SR")
    reference_block_columns: dict[str, list[str]] = {}
    if reference_df is not None:
        reference_block_columns = collect_block_columns(reference_df, reference_blocks, "reference")
        validate_paired_blocks(
            blocks,
            reference_blocks,
            block_columns,
            reference_block_columns,
        )

    transformed_blocks: dict[str, np.ndarray] = {}
    transformed_reference_blocks: dict[str, np.ndarray] = {}

    for block_name, cols in block_columns.items():
        block_data = df[cols].to_numpy(dtype=np.float32)
        fit_data = block_data[fit_mask]
        reference_data: np.ndarray | None = None
        if reference_df is not None and reference_fit_mask is not None:
            reference_data = reference_df[reference_block_columns[block_name]].to_numpy(
                dtype=np.float32
            )
            fit_data = np.concatenate([fit_data, reference_data[reference_fit_mask]], axis=0)

        max_allowed = min(fit_data.shape[0], fit_data.shape[1])
        if max_n > max_allowed:
            raise ValueError(
                f"Requested max n_components={max_n} for block '{block_name}', "
                f"but maximum allowed is {max_allowed} (min(n_fit_rows, n_features))."
            )

        LOGGER.info("Fitting PCA for block '%s' with n_components=%d", block_name, max_n)
        pca = PCA(n_components=max_n, svd_solver=args.svd_solver)
        pca.fit(fit_data)

        explained = float(np.sum(pca.explained_variance_ratio_))
        LOGGER.info("Block '%s': total explained variance at n=%d is %.6f", block_name, max_n, explained)

        transformed = pca.transform(block_data).astype(np.float32)
        transformed_blocks[block_name] = transformed
        if reference_data is not None:
            transformed_reference_blocks[block_name] = pca.transform(reference_data).astype(np.float32)

    if args.keep_original_blocks:
        base_df = df.copy()
    else:
        cols_to_drop = []
        for cols in block_columns.values():
            cols_to_drop.extend(cols)
        base_df = df.drop(columns=cols_to_drop)

    reference_base_df: pd.DataFrame | None = None
    if reference_df is not None:
        if args.keep_original_blocks:
            reference_base_df = reference_df.copy()
        else:
            reference_cols_to_drop = []
            for cols in reference_block_columns.values():
                reference_cols_to_drop.extend(cols)
            reference_base_df = reference_df.drop(columns=reference_cols_to_drop)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for n in n_values:
        out_df = base_df.copy()

        for block_name in block_columns:
            values = transformed_blocks[block_name][:, :n]
            pca_cols = [f"{block_name}_pca_{i:03d}" for i in range(n)]
            out_df = pd.concat([out_df, pd.DataFrame(values, columns=pca_cols)], axis=1)

        filename = args.output_template.format(stem=input_path.stem, n=n)
        out_path = output_dir / filename
        reference_out_path: Path | None = None
        if reference_base_df is not None and reference_path is not None:
            reference_template = args.reference_output_template or args.output_template
            reference_filename = reference_template.format(stem=reference_path.stem, n=n)
            reference_out_path = output_dir / reference_filename
            if reference_out_path == out_path:
                raise ValueError(
                    "SR and reference PCA outputs resolve to the same path. "
                    "Set --reference-output-template to a distinct filename."
                )
        out_df = relativize_path_columns(out_df)
        out_df.to_csv(out_path, index=False)
        LOGGER.info("Saved PCA CSV (n=%d): %s", n, out_path)

        if reference_base_df is not None and reference_out_path is not None:
            reference_out_df = reference_base_df.copy()
            for block_name in block_columns:
                values = transformed_reference_blocks[block_name][:, :n]
                pca_cols = [f"{block_name}_pca_{i:03d}" for i in range(n)]
                reference_out_df = pd.concat(
                    [reference_out_df, pd.DataFrame(values, columns=pca_cols)], axis=1
                )

            reference_out_df = relativize_path_columns(reference_out_df)
            reference_out_df.to_csv(reference_out_path, index=False)
            LOGGER.info("Saved reference PCA CSV (n=%d): %s", n, reference_out_path)


if __name__ == "__main__":
    main()
