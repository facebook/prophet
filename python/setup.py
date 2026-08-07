# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import platform
import shutil
import subprocess
import tempfile
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

CMDSTAN_VERSION = "2.37.0"
BINARIES_DIR = "bin"
BINARIES = ["diagnose", "print", "stanc", "stansummary"]
TBB_PARENT = "stan/lib/stan_math/lib"
TBB_DIRS = ["tbb", "tbb_2020.3"]


IS_WINDOWS = platform.platform().startswith("Win")
IS_LINUX = platform.system() == "Linux"
TBB_LIB_DIR = f"cmdstan-{CMDSTAN_VERSION}/stan/lib/stan_math/lib/tbb"


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

    # Drop compile artifacts; only shared libraries are needed at runtime.
    tbb_runtime_dir = temp_dir / TBB_PARENT / "tbb"
    if tbb_runtime_dir.is_dir():
        for f in tbb_runtime_dir.iterdir():
            if f.is_file() and f.suffix in {".o", ".d", ".def"}:
                os.remove(f)

    copy(original_dir / "makefile", temp_dir / "makefile")

    rmtree(original_dir)
    temp_dir.rename(original_dir)


def _patchelf(*args: str) -> None:
    """Run patchelf, required when building Linux wheels."""
    patchelf = shutil.which("patchelf")
    if patchelf is None:
        if os.environ.get("CIBUILDWHEEL"):
            raise RuntimeError(
                "patchelf is required when building Linux wheels so TBB can be "
                "bundled without auditwheel rewriting ELF headers."
            )
        print("patchelf not found; leaving Linux RPATHs unchanged")
        return
    subprocess.check_call([patchelf, *args])


def _is_dynamic_elf(path: Path) -> bool:
    """Return True if path is a dynamically linked ELF (has a .dynamic section)."""
    patchelf = shutil.which("patchelf")
    if patchelf is None:
        return False
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"\x7fELF":
                return False
        # stanc is shipped statically linked; only patch binaries that need TBB.
        result = subprocess.run(
            [patchelf, "--print-rpath", str(path)],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except OSError:
        return False


def fix_linux_rpaths(target_dir: str) -> None:
    """
    Point Stan binaries at the TBB we already ship inside the package.

    CmdStan writes absolute build-directory RPATHs. auditwheel then rewrites
    those ELF headers to vendor TBB into *.libs, but on newer manylinux images
    that rewrite leaves PT_DYNAMIC outside any PT_LOAD and the model binary
    immediately SIGSEGVs (cmdstanpy error -11). Keep the original soname and a
    $ORIGIN-relative RPATH instead, and exclude libtbb from auditwheel.
    """
    if not IS_LINUX:
        return

    if shutil.which("patchelf") is None:
        if os.environ.get("CIBUILDWHEEL"):
            raise RuntimeError(
                "patchelf is required when building Linux wheels so TBB can be "
                "bundled without auditwheel rewriting ELF headers."
            )
        print("patchelf not found; leaving Linux RPATHs unchanged")
        return

    target = Path(target_dir)
    model_bin = target / "prophet_model.bin"
    tbb_lib = target / TBB_LIB_DIR / "libtbb.so.2"

    if not model_bin.exists() or not _is_dynamic_elf(model_bin):
        raise RuntimeError(f"Expected dynamic Stan model binary at {model_bin}")
    _patchelf("--set-rpath", f"$ORIGIN/{TBB_LIB_DIR}", str(model_bin))

    if not tbb_lib.exists() or not _is_dynamic_elf(tbb_lib):
        raise RuntimeError(f"Expected dynamic TBB library at {tbb_lib}")
    _patchelf("--set-rpath", "$ORIGIN", str(tbb_lib))

    # Only the dynamically linked CmdStan helpers need TBB. stanc is static.
    cmdstan_bin_dir = target / f"cmdstan-{CMDSTAN_VERSION}" / BINARIES_DIR
    if cmdstan_bin_dir.is_dir():
        for exe in cmdstan_bin_dir.iterdir():
            if exe.is_file() and os.access(exe, os.X_OK) and _is_dynamic_elf(exe):
                _patchelf(
                    "--set-rpath",
                    "$ORIGIN/../stan/lib/stan_math/lib/tbb",
                    str(exe),
                )


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
    """
    import cmdstanpy

    target_cmdstan_dir = (Path(target_dir) / f"cmdstan-{CMDSTAN_VERSION}").resolve()
    with tempfile.TemporaryDirectory() as tmp_dir:
        # long paths on windows can cause problems during build
        if IS_WINDOWS:
            cmdstan_dir = (Path(tmp_dir) / f"cmdstan-{CMDSTAN_VERSION}").resolve()
        else:
            cmdstan_dir = target_cmdstan_dir

        install_cmdstan_deps(cmdstan_dir)
        model_name = "prophet.stan"
        # note: ensure copy target is a directory not a file.
        temp_stan_file = copy(os.path.join(MODEL_DIR, model_name), cmdstan_dir.parent.resolve())
        sm = cmdstanpy.CmdStanModel(stan_file=temp_stan_file)
        target_name = "prophet_model.bin"
        copy(sm.exe_file, os.path.join(target_dir, target_name))

        # CmdStan also leaves the un-renamed executable next to the .stan file.
        stale_exe = Path(sm.exe_file)
        if stale_exe.exists() and stale_exe.name != target_name:
            try:
                stale_exe.unlink()
            except OSError:
                pass

        if IS_WINDOWS and repackage_cmdstan():
            copytree(cmdstan_dir, target_cmdstan_dir)

    # Clean up
    for f in Path(MODEL_DIR).iterdir():
        if f.is_file() and f.name != model_name:
            os.remove(f)

    if repackage_cmdstan():
        prune_cmdstan(target_cmdstan_dir)
        fix_linux_rpaths(target_dir)


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
