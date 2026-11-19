# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which libcudnn ``cudnn._dlopen_cudnn`` picks for a pip-installed backend."""

import ctypes
import sys
import sysconfig

import pytest

import cudnn

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="Linux dlopen path; Windows uses load_cudnn")


@pytest.fixture
def opened(monkeypatch, tmp_path):
    """Run ``_dlopen_cudnn`` without touching the real backend handle: no
    LD_LIBRARY_PATH, an empty purelib, and a ``ctypes.CDLL`` stub that records
    what would be dlopened."""
    purelib = tmp_path / "purelib"
    purelib.mkdir()
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(sysconfig, "get_path", lambda *args, **kwargs: str(purelib))
    names = []

    class _Lib:
        _handle = 0x1000

    def _cdll(name, *args, **kwargs):
        names.append(str(name))
        return _Lib()

    monkeypatch.setattr(ctypes, "CDLL", _cdll)
    monkeypatch.setattr(cudnn._pybind_module, "_set_dlhandle_cudnn", lambda handle: None)
    return names


@pytest.mark.L0
@pytest.mark.parametrize("sub", ["nvidia/cudnn/lib", "nvidia/cudnn_jit/lib"])
def test_dlopen_finds_wheel_on_sys_path_outside_purelib(opened, monkeypatch, tmp_path, sub):
    # ``pip install --user nvidia-cudnn-cu12``, or the base environment of a
    # ``--system-site-packages`` venv: on sys.path, but not in purelib.
    site = tmp_path / "site"
    lib = site / sub / "libcudnn.so.9"
    lib.parent.mkdir(parents=True)
    lib.touch()
    monkeypatch.setattr(sys, "path", ["", str(tmp_path / "other"), str(site)])
    cudnn._dlopen_cudnn()
    assert opened == [str(lib)]


@pytest.mark.L0
def test_dlopen_falls_back_to_soname_without_wheel(opened, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "path", [str(tmp_path / "other")])
    cudnn._dlopen_cudnn()
    assert opened == ["libcudnn.so.9"]
