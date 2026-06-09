#!/usr/bin/env python3

from pathlib import Path
import sys

from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import find_packages, setup


def load_runtime_requirements(requirements_path: Path) -> list:
    """Keep install requirements limited to parseable runtime package specs."""

    skipped = {"zip", "ffmpeg"}
    requirements = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line in skipped:
            continue
        requirements.append(line)
    return requirements


pip_dependencies = load_runtime_requirements(Path(__file__).parent / "requirements.txt")
mesh_packages = find_packages("src")

d = generate_distutils_setup(
    packages=sorted(set(mesh_packages)),
    package_dir={"": "src"},
)

d.update(
    {
        "install_requires": pip_dependencies,
        "package_data": {
            "pointnet2_ops": ["_ext-src/include/*", "_ext-src/src/*"],
        },
    }
)

# Newer setuptools may reject the legacy Debian-specific flag that catkin still
# forwards during `make install`. Strip it so install remains compatible.
sys.argv = [arg for arg in sys.argv if not arg.startswith("--install-layout=")]

setup(**d)
