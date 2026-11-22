#!/usr/bin/env bash
set -euo pipefail

# Launch each image-workflow stage in a separate process, using one recorded
# configuration so that sample IDs, feature settings and splits stay aligned.
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python=${PYTHON:-python}
mode=${1:-cached}
if (( $# )); then shift; fi
plots=${PLOTS_DIR:-$PWD/reproduction/$mode}
dataset=${DATASET_DIR:-$repo/dataset}
source=${DATASET_SOURCE:-hf}
archive=${DATASET_ARCHIVE:-}
runtime=${RUNTIME_CONFIG:-$plots/reproduce_pipeline_config.json}
features=${FEATURES_DIR:-$plots/features}

run() {
    printf '\n+'
    printf ' %q' "$@"
    printf '\n'
    "$@"
}

usage() {
    cat <<'EOF'
Usage: bash reproduce_pipeline.sh [cached|download|images|set120|paper] [options]
  cached     Fit the bundled Set120 regression example (default, image-free).
             Additional arguments go to qualisr-run-regressors, e.g. --no-plots.
  download   Prepare Set120 and print its resolved root. No model execution.
  images     Download/reuse Set120, then launch references, features, PCA,
             embedding differences, statistics and regression separately.
  set120     Download/reuse Set120 and run grouped presets on the repository raw
             feature CSVs. Supply --output NEW_DIR and optional grouped runner flags.
  paper      Run the grouped paper matrix on prepared datasets and feature CSVs.
             Supply --output NEW_DIR. Additional options include --audit-only.

Set PYTHON to a Python 3.12/3.13 interpreter. DATASET_DIR is the download
destination. DATASET_SOURCE accepts hf (default), gdrive, archive or skip.
Use DATASET_ARCHIVE for a local ZIP, TAR or TAR.GZ. DATASET_REVISION overrides
the Set120 Hugging Face revision. PLOTS_DIR and FEATURES_DIR select outputs.
CONFIG selects the input config for images or paper. See experiments/README.md
for the feature-cache contract and commands for the other paper experiments.
EOF
}

case "$mode" in
    -h|--help) usage; exit 0 ;;
    cached|set120|paper) ;;
    download|images) if (( $# )); then usage >&2; exit 2; fi ;;
    *) usage >&2; exit 2 ;;
esac

maybe_install_deps() {
    if [[ ${INSTALL_DEPS:-0} == 1 ]]; then
        case "$mode" in
            images) run "$python" -m pip install -r "$repo/requirements.txt" -e "$repo" ;;
            *) run "$python" -m pip install -c "$repo/requirements.txt" -e "$repo[regressors]" ;;
        esac
    fi
}

dataset_ready() {
    [[ -f $1/labels.csv && -d $1/hr && -d $1/lr && -d $1/sr ]] || return 1
    PYTHONPATH="$repo" "$python" - "$1" <<'PY'
import sys

from qualisr.datasets import parse_qualisr_set120

try:
    rows = parse_qualisr_set120(sys.argv[1], include_refs=False)
except (FileNotFoundError, ValueError) as error:
    sys.exit(str(error))
if len(rows) != 120 or len({r["sr_path"] for r in rows}) != 120 or not all(r["hr_path"] for r in rows):
    sys.exit(f"Incomplete Set120 images under {sys.argv[1]}")
PY
}

locate_dataset() {
    local child
    if dataset_ready "$dataset"; then base=$dataset; return 0; fi
    for child in "$dataset"/*/; do
        if dataset_ready "$child"; then base=$child; return 0; fi
    done
    return 1
}

download_dataset() {
    case "$source" in
        hf|gdrive|skip) ;;
        archive) : "${archive:?Set DATASET_ARCHIVE to a local archive}" ;;
        *) echo "Unknown DATASET_SOURCE: $source" >&2; exit 2 ;;
    esac
    if locate_dataset; then
        echo "Using Set120 dataset: $base"
        return
    fi
    mkdir -p -- "$dataset"
    if [[ -n $archive && $source != skip ]]; then
        case "$archive" in
            *.zip) run unzip -qn "$archive" -d "$dataset" ;;
            *.tar|*.tar.gz|*.tgz) run tar --skip-old-files -xf "$archive" -C "$dataset" ;;
            *) echo "Unsupported dataset archive: $archive" >&2; exit 2 ;;
        esac
    elif [[ $source == hf ]]; then
        hf_repo=${DATASET_REPO:-onryabinin/QualiSR-Set120}
        revision=${DATASET_REVISION:-}
        if [[ -z $revision && $hf_repo == onryabinin/QualiSR-Set120 ]]; then
            revision=dbe874b8167610e43cf121d6e2d47aaafd115217
        fi
        hf_args=()
        if [[ -n $revision ]]; then hf_args+=(--revision "$revision"); fi
        if [[ -f $dataset/labels.csv ]]; then hf_args+=(--exclude labels.csv); fi
        if [[ -f $dataset/readme.md ]]; then hf_args+=(--exclude README.md); fi
        run hf download "$hf_repo" --repo-type dataset "${hf_args[@]}" --local-dir "$dataset"
    elif [[ $source == gdrive ]]; then
        run gdown --fuzzy "${GDRIVE_URL:-https://drive.google.com/file/d/1NeGiwWQECTZMxVhJ5ZALxQ5nzRYkz4-E/view}" -O "$dataset/grounding_dataset.zip"
        run unzip -qn "$dataset/grounding_dataset.zip" -d "$dataset"
    fi
    locate_dataset || { echo "Expected labels.csv and hr/, lr/, sr/ under $dataset or one child." >&2; exit 1; }
    echo "Set120 dataset: $base"
}

run_set120() {
    # Requires images for the content-identity audit, and the original raw CSVs for fold-local PCA.
    exec "$python" - "$repo/configs/paper.json" "$base" "${FEATURES_DIR:-$repo/features}" "$@" <<'PYTHON'
from pathlib import Path
import json
import sys
from tempfile import TemporaryDirectory

from experiments.run_grouped import main

template, dataset, features, *args = sys.argv[1:]
config = json.loads(Path(template).read_text())
(entry,) = [row for row in config["datasets"] if row["name"] == "QualiSR-Set120"]
entry.update(root=str(Path(dataset).resolve()), features_root=str(Path(features).resolve()))
config["datasets"] = [entry]
with TemporaryDirectory() as temporary:
    path = Path(temporary) / "paper-set120.json"
    path.write_text(json.dumps(config))
    main(["--config", str(path), *args])
PYTHON
}

write_runtime_pipeline_config() {
    mkdir -p -- "$plots"
    run "$python" - "${CONFIG:-$repo/configs/pipeline.json}" "$runtime" "$base" "$features" "$plots" "$repo" <<'PY'
import json
import os
import sys
from pathlib import Path

source, target, dataset, features, plots, repo = map(Path, sys.argv[1:])
config = json.loads(source.read_text())
entry, = config["datasets"]
if entry["name"] != "QualiSR-Set120":
    raise ValueError("images expects a single QualiSR-Set120 dataset")
entry.update(root=str(dataset.resolve()), features_root=str(features.resolve()))
refs = config["references"]
refs["enabled"] = os.getenv("RUN_REFERENCE", "0") == "1"
refs["python_exec"] = sys.executable
if "REF_METHODS" in os.environ:
    refs["refs"] = os.environ["REF_METHODS"].split()
for name in ("rlfn", "span"):
    folder = repo / "realtime_sr" / name.upper()
    for key, default in (("script", folder / f"inference-{name.upper()}.py"), ("ckpt", folder / f"{name}-tuned-4x.pth")):
        setting = f"{name}_{key}"
        refs[setting] = os.getenv(setting.upper(), refs.get(setting, str(default)))

groups = config["features"]["groups"]
if "FEATURE_GROUPS" in os.environ:
    selected = os.environ["FEATURE_GROUPS"].replace(",", " ").split()
    unknown = set(selected) - groups.keys()
    if unknown:
        raise ValueError(f"Unknown feature groups: {sorted(unknown)}")
    for name, group in groups.items():
        group["enabled"] = name in selected
common = config["features"]["common"]
if "DEVICE" in os.environ:
    common["device"] = os.environ["DEVICE"]
pca = config["pca"]["common"]
pca.update(group_column="test_case", split_seed=config["regressors"]["config"]["split_seed"],
           test_size=entry["regressors"]["test_size"])
if "PCA_COMPONENTS" in os.environ:
    pca["n_components"] = [int(n) for n in os.environ["PCA_COMPONENTS"].split()]
reg = config["regressors"]["config"]
reg["paths"]["plots_root"] = str(plots.resolve())
if os.getenv("SAVE_SVG") == "1":
    reg.setdefault("plot", {})["save_svg"] = True
if os.getenv("PROFILE") == "1" or os.getenv("PROFILE_FLOPS") == "1":
    common.update(profile=True, profile_flops=os.getenv("PROFILE_FLOPS") == "1")
    config["statistics"]["profile"] = True
    profiles = [g["profile_output"] for g in groups.values() if g.get("enabled", True) and config["features"]["enabled"]]
    if config["statistics"]["enabled"]:
        profiles.append(config["statistics"]["profile_output"])
    reg.setdefault("profiling", {}).update(regressors=True, feature_profile_files=[
        p.replace("{features_root}", str(features.resolve())) for p in profiles
    ])
    for profile in reg["profiling"]["feature_profile_files"]:
        Path(profile).expanduser().parent.mkdir(parents=True, exist_ok=True)
target.parent.mkdir(parents=True, exist_ok=True)
with target.open("x") as stream:
    stream.write(json.dumps(config, indent=2) + "\n")
PY
}

# Each command reloads the same config through the current dataset-aware runner.
# Disabled sections remain disabled; all other stages run in their own process.
maybe_make_references() {
    run "$python" -m qualisr.cli run-pipeline --config "$runtime" --only-section references
}

extract_features() {
    run "$python" -m qualisr.cli run-pipeline --config "$runtime" --only-section features
}

apply_pca() {
    run "$python" -m qualisr.cli run-pipeline --config "$runtime" --only-section pca
}

compute_embedding_differences() {
    run "$python" -m qualisr.cli run-pipeline --config "$runtime" --only-section embedding_difference
}

compute_stats() {
    run "$python" -m qualisr.cli run-pipeline --config "$runtime" --only-section statistics
}

run_regressors() {
    run "$python" -m qualisr.cli run-pipeline --config "$runtime" --only-section regressors
}

main() {
    "$python" -c 'import sys; sys.exit(0 if (3, 12) <= sys.version_info[:2] < (3, 14) else "Python 3.12 or 3.13 is required")'
    maybe_install_deps
    case "$mode" in
        cached) exec "$python" -m qualisr.cli run-regressors --plots-root "$plots" "$@" ;;
        paper) exec "$python" -m experiments.run_grouped --config "${CONFIG:-$repo/configs/paper.json}" "$@" ;;
    esac

    download_dataset
    case "$mode" in
        download) return ;;
        set120) run_set120 "$@" ;;
    esac

    write_runtime_pipeline_config
    maybe_make_references
    extract_features
    apply_pca
    compute_embedding_differences
    compute_stats
    run_regressors

    printf '\nReproducibility pipeline complete.\n'
    printf 'Dataset: %s\nFeatures: %s\nPlots: %s\n' "$base" "$features" "$plots"
}

main "$@"
