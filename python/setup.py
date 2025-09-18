# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import platform
from pathlib import Path
from shutil import copy, copytree, rmtree
from typing import List
import tempfile

from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.editable_wheel import editable_wheel
from wheel.bdist_wheel import bdist_wheel


MODEL_DIR = "stan"
MODEL_TARGET_DIR = os.path.join("prophet", "stan_model")

CMDSTAN_VERSION = "2.33.1"
BINARIES_DIR = "bin"
BINARIES = ["diagnose", "print", "stanc", "stansummary"]
TBB_PARENT = "stan/lib/stan_math/lib"
TBB_DIRS = ["tbb", "tbb_2020.3"]


IS_WINDOWS = platform.platform().startswith("Win")

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


def maybe_install_cmdstan_toolchain() -> bool:
    """Install C++ compilers required to build stan models on Windows machines."""
    import cmdstanpy

    try:
        cmdstanpy.utils.cxx_toolchain_path()
        return False
    except Exception:
        try:
            from cmdstanpy.install_cxx_toolchain import run_rtools_install
        except ImportError:
            # older versions
            from cmdstanpy.install_cxx_toolchain import main as run_rtools_install

        run_rtools_install({"version": None, "dir": None, "verbose": True})
        cmdstanpy.utils.cxx_toolchain_path()
        return True

def install_cmdstan_deps(cmdstan_dir: Path):
    import cmdstanpy
    from multiprocessing import cpu_count

    if repackage_cmdstan():
        if IS_WINDOWS:
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
    
    Raises
    ------
    RuntimeError: If any step in the model building process fails.
    """
    import cmdstanpy
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting CmdStan model build process. Target directory: {target_dir}")

    try:
        target_cmdstan_dir = (Path(target_dir) / f"cmdstan-{CMDSTAN_VERSION}").resolve()
        logger.info(f"Target CmdStan directory: {target_cmdstan_dir}")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger.info(f"Created temporary directory: {tmp_dir}")
            
            # long paths on windows can cause problems during build
            if IS_WINDOWS:
                cmdstan_dir = (Path(tmp_dir) / f"cmdstan-{CMDSTAN_VERSION}").resolve()
                logger.info(f"Windows detected. Using temporary CmdStan directory: {cmdstan_dir}")
            else:
                cmdstan_dir = target_cmdstan_dir
                logger.info(f"Using target CmdStan directory directly: {cmdstan_dir}")

            # Install CmdStan dependencies
            try:
                logger.info("Installing CmdStan dependencies...")
                install_cmdstan_deps(cmdstan_dir)
                logger.info("CmdStan dependencies installed successfully")
            except Exception as e:
                error_msg = f"Failed to install CmdStan dependencies: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

            # Copy Stan model file
            model_name = "prophet.stan"
            source_model_path = os.path.join(MODEL_DIR, model_name)
            target_model_dir = cmdstan_dir.parent.resolve()
            
            try:
                logger.info(f"Copying Stan model from {source_model_path} to {target_model_dir}")
                if not os.path.exists(source_model_path):
                    raise FileNotFoundError(f"Stan model file not found: {source_model_path}")
                temp_stan_file = copy(source_model_path, target_model_dir)
                logger.info(f"Stan model copied successfully to: {temp_stan_file}")
            except Exception as e:
                error_msg = f"Failed to copy Stan model file: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

            # Compile Stan model
            try:
                logger.info(f"Compiling Stan model: {temp_stan_file}")
                sm = cmdstanpy.CmdStanModel(stan_file=temp_stan_file)
                logger.info(f"Stan model compiled successfully. Executable: {sm.exe_file}")
                
                if not os.path.exists(sm.exe_file):
                    raise FileNotFoundError(f"Compiled model executable not found: {sm.exe_file}")
            except Exception as e:
                error_msg = f"Failed to compile Stan model: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

            # Copy compiled model to target directory
            target_name = "prophet_model.bin"
            target_executable_path = os.path.join(target_dir, target_name)
            
            try:
                logger.info(f"Copying compiled model from {sm.exe_file} to {target_executable_path}")
                copy(sm.exe_file, target_executable_path)
                logger.info("Compiled model copied successfully")
            except Exception as e:
                error_msg = f"Failed to copy compiled model: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

            # Copy CmdStan directory on Windows if needed
            if IS_WINDOWS and repackage_cmdstan():
                try:
                    logger.info(f"Copying CmdStan directory from {cmdstan_dir} to {target_cmdstan_dir}")
                    copytree(cmdstan_dir, target_cmdstan_dir)
                    logger.info("CmdStan directory copied successfully")
                except Exception as e:
                    error_msg = f"Failed to copy CmdStan directory: {str(e)}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

        # Clean up temporary files
        try:
            logger.info("Cleaning up temporary files in model directory")
            cleanup_count = 0
            model_dir_path = Path(MODEL_DIR)
            
            if not model_dir_path.exists():
                logger.warning(f"Model directory does not exist: {MODEL_DIR}")
            else:
                for f in model_dir_path.iterdir():
                    if f.is_file() and f.name != model_name:
                        try:
                            os.remove(f)
                            cleanup_count += 1
                            logger.debug(f"Removed temporary file: {f}")
                        except Exception as e:
                            logger.warning(f"Failed to remove temporary file {f}: {str(e)}")
                
                logger.info(f"Cleaned up {cleanup_count} temporary files")
                
        except Exception as e:
            # Don't fail the entire build for cleanup issues
            logger.warning(f"Non-critical error during cleanup: {str(e)}")

        # Prune CmdStan installation if needed
        if repackage_cmdstan():
            try:
                logger.info("Pruning CmdStan installation to reduce size")
                prune_cmdstan(target_cmdstan_dir)
                logger.info("CmdStan installation pruned successfully")
            except Exception as e:
                error_msg = f"Failed to prune CmdStan installation: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

        logger.info("CmdStan model build process completed successfully")
        
    except RuntimeError:
        # Re-raise RuntimeErrors (our custom errors)
        raise
    except Exception as e:
        # Catch any unexpected errors and wrap them
        error_msg = f"Unexpected error during CmdStan model build: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


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
