# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import platform
import sys
from collections import OrderedDict
from distutils.extension import Extension
from pathlib import Path
from shutil import copy, copytree, rmtree
from typing import List

from pkg_resources import add_activation_listener, normalize_path, require, working_set
from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.test import test as test_command

PLATFORM = "unix"
if platform.platform().startswith("Win"):
    PLATFORM = "win"

MODEL_DIR = os.path.join("stan", PLATFORM)
MODEL_TARGET_DIR = os.path.join("prophet", "stan_model")

# cmdstan utils
CMDSTAN_VERSION = "2.26.1"
BINARIES_DIR = "bin"
BINARIES = ["diagnose", "print", "stanc", "stansummary"]
TBB_PARENT = "stan/lib/stan_math/lib"
TBB_DIRS = ["tbb", "tbb_2019_U8"]


def prune_cmdstan(cmdstan_dir: str) -> None:
    original_dir = Path(cmdstan_dir).resolve()
    parent_dir = original_dir.parent
    temp_dir = parent_dir / "temp"
    if temp_dir.is_dir():
        rmtree(temp_dir)
    temp_dir.mkdir()

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


def get_backends_from_env() -> List[str]:
    return os.environ.get("STAN_BACKEND", "PYSTAN").split(",")


def get_cmdstan_cache() -> str:
    return Path.home().resolve() / ".cmdstan" / f"cmdstan-{CMDSTAN_VERSION}"


def build_cmdstan_model(target_dir):
    import cmdstanpy

    cmdstan_cache = get_cmdstan_cache()
    cmdstan_dir = os.path.join(target_dir, f"cmdstan-{CMDSTAN_VERSION}")

    if os.path.isdir(cmdstan_cache):
        print(f"Found existing cmdstan library at {cmdstan_cache}")
    else:
        cmdstanpy.install_cmdstan(
            version=CMDSTAN_VERSION, dir=cmdstan_cache, overwrite=True
        )

    if os.path.isdir(cmdstan_dir):
        rmtree(cmdstan_dir)
    copytree(cmdstan_cache, cmdstan_dir)
    cmdstanpy.set_cmdstan_path(cmdstan_dir)

    model_name = "prophet.stan"
    target_name = "prophet_model.bin"
    sm = cmdstanpy.CmdStanModel(stan_file=os.path.join(MODEL_DIR, model_name))
    sm.compile()
    copy(sm.exe_file, os.path.join(target_dir, target_name))
    # Add tbb to the $PATH on Windows
    if platform.system() == "Windows":
        libtbb = Path(cmdstan_dir).resolve() / TBB_PARENT / "tbb"
        if libtbb not in os.environ["PATH"]:
            os.environ["PATH"] = ";".join(
                list(
                    OrderedDict.fromkeys(
                        [libtbb] + os.environ.get("PATH", "").split(";")
                    )
                )
            )
    # Clean up
    for f in Path(MODEL_DIR).iterdir():
        if f.is_file() and f.name != model_name:
            os.remove(f)
    prune_cmdstan(cmdstan_dir)


def build_pystan_model(target_dir):
    import pystan

    model_name = "prophet.stan"
    target_name = "prophet_model.pkl"
    with open(os.path.join(MODEL_DIR, model_name)) as f:
        model_code = f.read()
    sm = pystan.StanModel(model_code=model_code)
    with open(os.path.join(target_dir, target_name), "wb") as f:
        pickle.dump(sm, f, protocol=pickle.HIGHEST_PROTOCOL)


def build_models(target_dir):
    for backend in get_backends_from_env():
        print(f"Compiling {backend} model")
        if backend == "CMDSTANPY":
            build_cmdstan_model(target_dir)
        elif backend == "PYSTAN":
            build_pystan_model(target_dir)


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
    version="1.0.1",
    description="Automatic Forecasting Procedure",
    url="https://facebook.github.io/prophet/",
    author="Sean J. Taylor <sjtz@pm.me>, Ben Letham <bletham@fb.com>",
    author_email="sjtz@pm.me",
    license="MIT",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3",
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
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
