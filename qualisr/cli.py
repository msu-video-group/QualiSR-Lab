"""Console entry points for QualiSR-Lab."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence

SCRIPT_COMMANDS = {
    "make-reference": "qualisr.references",
    "extract-features": "qualisr.features",
    "apply-pca": "qualisr.pca",
    "embedding-difference": "qualisr.embedding_difference",
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


def embedding_difference_main() -> None:
    _run_module_main("qualisr.embedding_difference")


def compute_stats_main() -> None:
    _run_module_main("qualisr.statistics")


def run_pipeline_main() -> None:
    _run_module_main("qualisr.pipeline")


def run_regressors_main() -> None:
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
