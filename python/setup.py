# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import platform
from pathlib import Path
from shutil import copy, copytree, rmtree
from typing import List

from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.editable_wheel import editable_wheel
from wheel.bdist_wheel import bdist_wheel


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

    try:
        cmdstanpy.utils.cxx_toolchain_path()
    except Exception:
        try:
            from cmdstanpy.install_cxx_toolchain import run_rtools_install
        except ImportError:
            # older versions
            from cmdstanpy.install_cxx_toolchain import main as run_rtools_install

        run_rtools_install({"version": None, "dir": None, "verbose": True})
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
    print("Compiling cmdstanpy model")
    build_cmdstan_model(target_dir)

    if "PYSTAN" in get_backends_from_env():
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


class EditableWheel(editable_wheel):
    """Custom develop command to pre-compile Stan models in-place."""

    def run(self):
        if not self.dry_run:
            target_dir = os.path.join(self.project_dir, MODEL_TARGET_DIR)
            self.mkpath(target_dir)
            build_models(target_dir)

        editable_wheel.run(self)


class BDistWheelABINone(bdist_wheel):
    def finalize_options(self):
        bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        _, _, plat = bdist_wheel.get_tag(self)
        return "py3", "none", plat


about = {}
here = Path(__file__).parent.resolve()
with open(here / "prophet" / "__version__.py", "r") as f:
    exec(f.read(), about)

setup(
    version=about["__version__"],
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True,
    ext_modules=[Extension("prophet.stan_model", [])],
    cmdclass={
        "build_ext": BuildExtCommand,
        "build_py": BuildPyCommand,
        "editable_wheel": EditableWheel,
        "bdist_wheel": BDistWheelABINone,
    },
    test_suite="prophet.tests",
)
