#!/usr/bin/env python

import sys
from pathlib import Path

# Try catkin install
from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import find_packages, setup
from setuptools.command.install import install as _install

ROOT = Path(__file__).resolve().parent


def _strip_unsupported_install_args(argv):
    """Keep setup.py compatible with catkin while handling install-layout ourselves."""
    unsupported_prefixes = ()
    return [
        arg
        for arg in argv
        if not any(arg.startswith(prefix) for prefix in unsupported_prefixes)
    ]


sys.argv = _strip_unsupported_install_args(sys.argv)

with open(ROOT / "requirements.txt", "r") as f:
    pip_dependencies = [
        line.strip()
        for line in f.readlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


class CatkinInstallCommand(_install):
    user_options = _install.user_options + [
        ("install-layout=", None, "installation layout"),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.install_layout = None

    def finalize_options(self):
        super().finalize_options()
        if self.install_layout == "deb":
            self.install_purelib = self.install_purelib.replace(
                "site-packages", "dist-packages"
            )
            self.install_platlib = self.install_platlib.replace(
                "site-packages", "dist-packages"
            )


d = generate_distutils_setup(
    packages=find_packages("src"),
    package_dir={"": "src"},
)

d.update(
    {
        "install_requires": pip_dependencies,
        "cmdclass": {"install": CatkinInstallCommand},
    }
)

setup(**d)
