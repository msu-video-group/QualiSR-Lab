"""Console entry points for QualiSR-Lab."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence

SCRIPT_COMMANDS = {
    "make-reference": "qualisr.references",
    "extract-features": "qualisr.features",
    "apply-pca": "qualisr.pca",
    "compute-stats": "qualisr.statistics",
    "run-pipeline": "qualisr.pipeline",
}


def _run_module_main(module_name: str, argv: Sequence[str] | None = None) -> None:
    if argv is not None:
        sys.argv = [module_name.rsplit(".", 1)[-1], *argv]
    module = importlib.import_module(module_name)
    module.main()


def make_reference_main() -> None:
    _run_module_main("qualisr.references")


def extract_features_main() -> None:
    _run_module_main("qualisr.features")


def apply_pca_main() -> None:
    _run_module_main("qualisr.pca")


def compute_stats_main() -> None:
    _run_module_main("qualisr.statistics")


def run_pipeline_main() -> None:
    _run_module_main("qualisr.pipeline")


def _print_run_regressors_help() -> None:
    print(
        "usage: qualisr-run-regressors [-h] [--config CONFIG] "
        "[--experiment-name EXPERIMENT_NAME] [--plots-root PLOTS_ROOT] "
        "[--no-plots] [--save-svg] [--profile] [--profile-output PROFILE_OUTPUT] "
        "[--profile-total-output PROFILE_TOTAL_OUTPUT] "
        "[--feature-profile-files FEATURE_PROFILE_FILES ...]\n\n"
        "Run configured QualiSR-Lab regressor experiments.\n\n"
        "options:\n"
        "  -h, --help            show this help message and exit\n"
        "  --config CONFIG       Path to experiment JSON config.\n"
        "  --experiment-name EXPERIMENT_NAME\n"
        "                        Override config experiment_name.\n"
        "  --plots-root PLOTS_ROOT\n"
        "                        Override config paths.plots_root.\n"
        "  --no-plots            Skip plot generation.\n"
        "  --save-svg            Also save generated plots in SVG format.\n"
        "  --profile             Measure regressor runtime/FLOPs and save a profile CSV.\n"
        "  --profile-output PROFILE_OUTPUT\n"
        "                        Output CSV path for regressor profile; implies --profile.\n"
        "  --profile-total-output PROFILE_TOTAL_OUTPUT\n"
        "                        Output CSV path for feature+regressor totals.\n"
        "  --feature-profile-files FEATURE_PROFILE_FILES ...\n"
        "                        Feature profile CSV files to aggregate into regressor totals."
    )


def run_regressors_main() -> None:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        _print_run_regressors_help()
        return

    from qualisr.regressors import main

    main()


def _print_help() -> None:
    commands = "\n".join(f"  {name}" for name in sorted([*SCRIPT_COMMANDS, "run-regressors"]))
    print(
        "QualiSR-Lab command dispatcher\n\n"
        "Usage:\n"
        "  python -m qualisr.cli <command> [args]\n\n"
        "Commands:\n"
        f"{commands}\n\n"
        "Each command also has a console-script alias, for example "
        "`qualisr-run-regressors --help`."
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        return

    command, command_args = args[0], args[1:]
    if command == "run-regressors":
        if any(arg in {"-h", "--help"} for arg in command_args):
            _print_run_regressors_help()
            return

        from qualisr.regressors import main as regressors_main

        regressors_main(command_args)
        return

    module_name = SCRIPT_COMMANDS.get(command)
    if module_name is None:
        valid = ", ".join(sorted([*SCRIPT_COMMANDS, "run-regressors"]))
        raise SystemExit(f"Unknown command '{command}'. Valid commands: {valid}")

    _run_module_main(module_name, command_args)


if __name__ == "__main__":
    main()
