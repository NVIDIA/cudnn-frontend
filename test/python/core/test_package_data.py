# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every non-Python file shipped under python/cudnn is listed in pyproject's package-data.

Package discovery only collects ``.py`` files; anything read at run time (NVRTC
kernel sources, checksums, vendored licenses) reaches a wheel only through a
``[tool.setuptools.package-data]`` glob. A missing glob passes every in-tree run
and fails only for wheel users: the SM90 KDA CUDA engine shipped without its
``.cu`` sources and raised FileNotFoundError from an installed wheel.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.L0

_REPO = Path(__file__).resolve().parents[3]
_PKG_ROOT = _REPO / "python"
# Not package data: sources, their bytecode, docs, and the extension built in place.
_NOT_DATA = {".py", ".pyc", ".md", ".so", ".pyd"}


def _package_data():
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        tomllib = pytest.importorskip("tomli")
    pyproject = _REPO / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("needs the source tree (pyproject.toml)")
    with pyproject.open("rb") as f:
        return tomllib.load(f)["tool"]["setuptools"].get("package-data", {})


def test_every_runtime_data_file_is_package_data():
    covered = set()
    for package, patterns in _package_data().items():
        if not package.startswith("cudnn"):
            continue
        package_dir = _PKG_ROOT / package.replace(".", "/")
        for pattern in patterns:
            covered.update(p.resolve() for p in package_dir.glob(pattern) if p.is_file())
    shipped = [p.resolve() for p in (_PKG_ROOT / "cudnn").rglob("*") if p.is_file() and p.suffix not in _NOT_DATA and "__pycache__" not in p.parts]
    missing = sorted(str(p.relative_to(_PKG_ROOT)) for p in shipped if p not in covered)
    assert not missing, f"add these to [tool.setuptools.package-data] in pyproject.toml: {missing}"
