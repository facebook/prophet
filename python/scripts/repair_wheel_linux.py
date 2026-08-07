#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Repair Linux prophet wheels without letting auditwheel rewrite the Stan binary.

CmdStan embeds absolute build-directory RPATHs. Default `auditwheel repair`
vendors libtbb into *.libs and rewrites ELF headers; on manylinux_2_28 that
rewrite corrupts the Stan model binary (PT_DYNAMIC left outside any PT_LOAD),
which then SIGSEGVs at runtime (cmdstanpy error -11).

Instead:
  1. Point dynamic binaries at the libtbb.so.2 already shipped under stan_model/
  2. Run auditwheel with --exclude libtbb.so.2 so it only applies manylinux tags
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def _can_set_rpath(patchelf: str, path: Path) -> bool:
    """False for non-ELF files and static binaries (e.g. stanc)."""
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"\x7fELF":
                return False
        return (
            subprocess.run(
                [patchelf, "--print-rpath", str(path)],
                capture_output=True,
                check=False,
            ).returncode
            == 0
        )
    except OSError:
        return False


def _set_rpath(patchelf: str, path: Path, rpath: str) -> None:
    subprocess.check_call([patchelf, "--set-rpath", rpath, str(path)])


def _fix_bundled_tbb_rpaths(root: Path, patchelf: str) -> None:
    matches = list(root.glob("**/prophet/stan_model/prophet_model.bin"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one prophet_model.bin in wheel, found {matches}"
        )
    model_bin = matches[0]
    stan_model = model_bin.parent

    cmdstan_dirs = sorted(stan_model.glob("cmdstan-*"))
    if len(cmdstan_dirs) != 1:
        raise RuntimeError(
            f"Expected exactly one cmdstan-* directory, found {cmdstan_dirs}"
        )
    cmdstan_dir = cmdstan_dirs[0]
    tbb_rel = f"{cmdstan_dir.name}/stan/lib/stan_math/lib/tbb"
    tbb_lib = stan_model / tbb_rel / "libtbb.so.2"

    if not _can_set_rpath(patchelf, model_bin):
        raise RuntimeError(f"Expected dynamic Stan model binary at {model_bin}")
    _set_rpath(patchelf, model_bin, f"$ORIGIN/{tbb_rel}")

    if not tbb_lib.is_file() or not _can_set_rpath(patchelf, tbb_lib):
        raise RuntimeError(f"Expected dynamic TBB library at {tbb_lib}")
    _set_rpath(patchelf, tbb_lib, "$ORIGIN")

    # diagnose/print/stansummary need TBB; stanc is static and is skipped.
    bin_dir = cmdstan_dir / "bin"
    if bin_dir.is_dir():
        for exe in bin_dir.iterdir():
            if exe.is_file() and _can_set_rpath(patchelf, exe):
                _set_rpath(patchelf, exe, "$ORIGIN/../stan/lib/stan_math/lib/tbb")


def _repack_wheel(extract_dir: Path, dest_wheel: Path) -> None:
    with zipfile.ZipFile(dest_wheel, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(extract_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(extract_dir).as_posix())


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"Usage: {argv[0]} WHEEL DEST_DIR", file=sys.stderr)
        return 2

    wheel = Path(argv[1]).resolve()
    dest_dir = Path(argv[2]).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    patchelf = shutil.which("patchelf")
    if patchelf is None:
        raise RuntimeError("patchelf is required to repair Linux prophet wheels")

    with tempfile.TemporaryDirectory(prefix="prophet-wheel-repair-") as tmp:
        tmp_path = Path(tmp)
        extract_dir = tmp_path / "wheel"
        extract_dir.mkdir()
        with zipfile.ZipFile(wheel) as zf:
            zf.extractall(extract_dir)

        _fix_bundled_tbb_rpaths(extract_dir, patchelf)

        dirty_wheel = tmp_path / wheel.name
        _repack_wheel(extract_dir, dirty_wheel)

        subprocess.check_call(
            [
                "auditwheel",
                "repair",
                "--exclude",
                "libtbb.so.2",
                "-w",
                str(dest_dir),
                str(dirty_wheel),
            ]
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
