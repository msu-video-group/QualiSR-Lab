"""Derive installation metadata from the authoritative requirements pins."""

from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from setuptools import setup

root = Path(__file__).parent
requirements = [
    Requirement(line)
    for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
pins = {canonicalize_name(requirement.name): str(requirement) for requirement in requirements}


def pinned(*names: str) -> list[str]:
    return [pins[canonicalize_name(name)] for name in names]


setup(
    install_requires=pinned(
        "matplotlib", "numpy", "pandas", "pillow", "scikit-learn", "scipy", "tqdm", "huggingface-hub",
    ),
    extras_require={
        "regressors": pinned("catboost", "xgboost"),
        "features": pinned(
            "bitsandbytes", "icecream", "opencv-python", "pyiqa", "timm", "torch", "torchvision", "transformers",
        ),
        "notebook": pinned("ipykernel", "ipython"),
        "dev": pinned("pytest", "ruff"),
        # The complete Linux/CUDA environment, including all transitive pins.
        "all": [str(requirement) for requirement in requirements],
    },
)
