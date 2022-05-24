# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import platform
from pathlib import Path
from shutil import copy, copytree, rmtree
from typing import List

from pkg_resources import add_activation_listener, normalize_path, require, working_set
from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.test import test as test_command


MODEL_DIR = "stan"
MODEL_TARGET_DIR = os.path.join("prophet", "stan_model")

CMDSTAN_VERSION = "2.26.1"
BINARIES_DIR = "bin"
BINARIES = ["diagnose", "print", "stanc", "stansummary"]
TBB_PARENT = "stan/lib/stan_math/lib"
TBB_DIRS = ["tbb", "tbb_2019_U8"]


def prune_cmdstan(cmdstan_dir: str) -> None:
    """
    Keep only the cmdstan executables and tbb files (minimum required to run a cmdstanpy commands on a pre-compiled model).
    """
    original_dir = Path(cmdstan_dir).resolve()
    parent_dir = original_dir.parent
    temp_dir = parent_dir / "temp"
    if temp_dir.is_dir():
        rmtree(temp_dir)
    temp_dir.mkdir()

    print("Copying ", original_dir, " to ", temp_dir, " for pruning")
    copytree(original_dir / BINARIES_DIR, temp_dir / BINARIES_DIR)
    for f in (temp_dir / BINARIES_DIR).iterdir():
        if f.is_dir():
            rmtree(f)
        elif f.is_file() and f.stem not in BINARIES:
            os.remove(f)
    for tbb_dir in TBB_DIRS:
        copytree(original_dir / TBB_PARENT / tbb_dir, temp_dir / TBB_PARENT / tbb_dir)

    rmtree(original_dir)
    temp_dir.rename(original_dir)

def repackage_cmdstan():
    return os.environ.get("PROPHET_REPACKAGE_CMDSTAN", "").lower() not in ["false", "0"]


def maybe_install_cmdstan_toolchain():
    """Install C++ compilers required to build stan models on Windows machines."""
    import cmdstanpy
    from cmdstanpy.install_cxx_toolchain import main as _install_cxx_toolchain

    try:
        cmdstanpy.utils.cxx_toolchain_path()
    except Exception:
        _install_cxx_toolchain({"version": None, "dir": None, "verbose": True})
        cmdstanpy.utils.cxx_toolchain_path()


def install_cmdstan_deps(cmdstan_dir: Path):
    import cmdstanpy
    from multiprocessing import cpu_count

    if repackage_cmdstan():
        if platform.platform().startswith("Win"):
            maybe_install_cmdstan_toolchain()
        print("Installing cmdstan to", cmdstan_dir)
        if os.path.isdir(cmdstan_dir):
            rmtree(cmdstan_dir)

        if not cmdstanpy.install_cmdstan(
            version=CMDSTAN_VERSION,
            dir=cmdstan_dir.parent,
            overwrite=True,
            verbose=True,
            cores=cpu_count(),
            progress=True,
        ):
            raise RuntimeError("CmdStan failed to install in repackaged directory")


def build_cmdstan_model(target_dir):
    """
    Rebuild cmdstan in the build environment, then use this installation to compile the stan model.
    The stan model is copied to {target_dir}/prophet_model.bin
    The cmdstan files required to run cmdstanpy commands are copied to {target_dir}/cmdstan-{version}.

    Parameters
    ----------
    target_dir: Directory to copy the compiled model executable and core cmdstan files to.
    """
    import cmdstanpy

    cmdstan_dir = (Path(target_dir) / f"cmdstan-{CMDSTAN_VERSION}").resolve()
    install_cmdstan_deps(cmdstan_dir)
    model_name = "prophet.stan"
    target_name = "prophet_model.bin"
    sm = cmdstanpy.CmdStanModel(stan_file=os.path.join(MODEL_DIR, model_name))
    copy(sm.exe_file, os.path.join(target_dir, target_name))

    # Clean up
    for f in Path(MODEL_DIR).iterdir():
        if f.is_file() and f.name != model_name:
            os.remove(f)

    if repackage_cmdstan():
        prune_cmdstan(cmdstan_dir)



def get_backends_from_env() -> List[str]:
    return os.environ.get("STAN_BACKEND", "CMDSTANPY").split(",")


def build_models(target_dir):
    print(f"Compiling cmdstanpy model")
    build_cmdstan_model(target_dir)

    if 'PYSTAN' in get_backends_from_env():
        raise ValueError("PyStan backend is not supported for Prophet >= 1.1")

class BuildPyCommand(build_py):
    """Custom build command to pre-compile Stan models."""

    def run(self):
        if not self.dry_run:
            target_dir = os.path.join(self.build_lib, MODEL_TARGET_DIR)
            self.mkpath(target_dir)
            build_models(target_dir)

        build_py.run(self)


class BuildExtCommand(build_ext):
    """Ensure built extensions are added to the correct path in the wheel."""

    def run(self):
        pass


class DevelopCommand(develop):
    """Custom develop command to pre-compile Stan models in-place."""

    def run(self):
        if not self.dry_run:
            target_dir = os.path.join(self.setup_path, MODEL_TARGET_DIR)
            self.mkpath(target_dir)
            build_models(target_dir)

        develop.run(self)


class TestCommand(test_command):
    user_options = [
        ("test-module=", "m", "Run 'test_suite' in specified module"),
        (
            "test-suite=",
            "s",
            "Run single test, case or suite (e.g. 'module.test_suite')",
        ),
        ("test-runner=", "r", "Test runner to use"),
        ("test-slow", "w", "Test slow suites (default off)"),
    ]
    test_slow = None

    def initialize_options(self):
        super(TestCommand, self).initialize_options()
        self.test_slow = False

    def finalize_options(self):
        super(TestCommand, self).finalize_options()
        if self.test_slow is None:
            self.test_slow = getattr(self.distribution, "test_slow", False)

    """We must run tests on the build directory, not source."""

    def with_project_on_sys_path(self, func):
        # Ensure metadata is up-to-date
        self.reinitialize_command("build_py", inplace=0)
        self.run_command("build_py")
        bpy_cmd = self.get_finalized_command("build_py")
        build_path = normalize_path(bpy_cmd.build_lib)

        # Build extensions
        self.reinitialize_command("egg_info", egg_base=build_path)
        self.run_command("egg_info")

        self.reinitialize_command("build_ext", inplace=0)
        self.run_command("build_ext")

        ei_cmd = self.get_finalized_command("egg_info")

        old_path = sys.path[:]
        old_modules = sys.modules.copy()

        try:
            sys.path.insert(0, normalize_path(ei_cmd.egg_base))
            working_set.__init__()
            add_activation_listener(lambda dist: dist.activate())
            require("%s==%s" % (ei_cmd.egg_name, ei_cmd.egg_version))
            func()
        finally:
            sys.path[:] = old_path
            sys.modules.clear()
            sys.modules.update(old_modules)
            working_set.__init__()


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

setup(
    name="prophet",
    version="1.1",
    description="Automatic Forecasting Procedure",
    url="https://facebook.github.io/prophet/",
    project_urls={
        "Source": "https://github.com/facebook/prophet",
    },
    author="Sean J. Taylor <sjtz@pm.me>, Ben Letham <bletham@fb.com>",
    author_email="sjtz@pm.me",
    license="MIT",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.7",
    zip_safe=False,
    include_package_data=True,
    ext_modules=[Extension("prophet.stan_model", [])],
    cmdclass={
        "build_ext": BuildExtCommand,
        "build_py": BuildPyCommand,
        "develop": DevelopCommand,
        "test": TestCommand,
    },
    test_suite="prophet.tests",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
