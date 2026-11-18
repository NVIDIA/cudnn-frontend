# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of manifest builds and buffer validation."""

import importlib.util
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.L0
_SOURCE = Path(__file__).resolve().parents[3] / "python" / "cudnn"


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def loader(monkeypatch, tmp_path):
    module = _module("cake_loader_test", _SOURCE / "_cake" / "loader.py")
    ffi = ModuleType("tvm_ffi")
    ffi.__version__ = "test"
    ffi.libinfo = SimpleNamespace(find_include_path=lambda: "/ffi", find_dlpack_include_path=lambda: "/dlpack")
    monkeypatch.setitem(sys.modules, "tvm_ffi", ffi)
    monkeypatch.setitem(sys.modules, "tvm_ffi.libinfo", ffi.libinfo)
    monkeypatch.setenv("TVM_FFI_CACHE_DIR", str(tmp_path))
    return module


def _spec(loader, name="kernel_closure", flags=None):
    return loader.JitSpec(
        "cake_test",
        name,
        nvcc="nvcc",
        cuda=loader.CudaVersion(13, 4, "13.4.92"),
        flags=flags or ["-shared"],
        include_dirs=["/source"],
        sources=["kernel.cu"],
        ldflags=["-lcuda"],
    )


def test_manifest_toolchain_preserves_flags_and_floor(loader, monkeypatch):
    build = ModuleType("fixture.cake_build")
    build.__file__ = "/source/cake_test/cake_build.py"
    build._jit_toolchain = lambda name: None
    build.libraries_by_name = lambda: {"kernel": {"arch": "sm_100a"}}
    build.manifest = lambda: {"common": {"nvcc_flags": ["-shared"], "nvcc_flags_if": [{"cuda": ">=13.4", "flags": ["-lineinfo"]}]}}
    monkeypatch.setattr(loader, "_toolkit", lambda *args: ("nvcc", loader.CudaVersion(13, 4, "13.4.92")))
    loader.install(build)
    factory, version = build._jit_toolchain("kernel")
    spec = factory("kernel_closure", [Path("kernel.cu")], extra_cuda_cflags=["-arch=sm_100a"], extra_ldflags=["-lcuda"], extra_include_paths=[])
    assert spec.flags == ["-shared", "-lineinfo", "-arch=sm_100a"]
    assert version().full == "13.4.92"
    assert spec.command(Path("out.so"))[-4:] == ["kernel.cu", "-lcuda", "-o", "out.so"]
    monkeypatch.setattr(loader, "_toolkit", lambda *args: ("nvcc", loader.CudaVersion(13, 2, "13.2.78")))
    with pytest.raises(RuntimeError, match="needs CUDA 13.4"):
        factory("kernel_closure", [], extra_cuda_cflags=[], extra_ldflags=[], extra_include_paths=[])
    with pytest.raises(NotImplementedError, match="unsupported Cake architecture"):
        loader.check_toolkit("sm_90a", "test")


def test_atomic_build_cache_and_failure_cleanup(loader, monkeypatch):
    calls = []

    def compile(command, **kwargs):
        calls.append(command)
        Path(command[-1]).write_bytes(b"library")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(loader.subprocess, "run", compile)
    spec = _spec(loader)
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(lambda _: spec.build(), range(2)))
    assert paths[0] == paths[1]
    assert len(calls) == 1
    assert paths[0].read_bytes() == b"library"
    assert not list(paths[0].parent.glob("*.tmp.so"))
    assert _spec(loader, flags=["-shared", "-O3"]).get_library_path() != paths[0]
    assert _spec(loader, name="different_closure").get_library_path() != paths[0]

    def fail(command, **kwargs):
        Path(command[-1]).write_bytes(b"partial")
        return SimpleNamespace(returncode=1, stderr="compile failure")

    monkeypatch.setattr(loader.subprocess, "run", fail)
    failed = _spec(loader, name="failed")
    with pytest.raises(RuntimeError, match="compile failure"):
        failed.build()
    assert not failed.get_library_path().exists()
    assert not list(paths[0].parent.glob("*.tmp.so"))


def test_no_build_guard_is_nested_and_thread_local(loader):
    spec = _spec(loader)
    with loader.no_build():
        with loader.no_build(), pytest.raises(RuntimeError, match="build the plan first"):
            spec.build()
        assert loader._NO_BUILD.get()
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(loader._NO_BUILD.get).result() is False
    assert loader._NO_BUILD.get() is False


def test_loaded_library_checks_runtime_version(loader, monkeypatch):
    spec = _spec(loader)
    monkeypatch.setattr(spec, "build", lambda: Path("library.so"))
    monkeypatch.setattr(sys.modules["tvm_ffi"], "load_module", lambda path: SimpleNamespace(build_cuda_version=lambda: 13020), raising=False)
    with pytest.raises(RuntimeError, match="CUDART 13020"):
        spec.build_and_load()


def test_import_is_lightweight():
    script = """
import importlib.util, sys
before = set(sys.modules)
spec = importlib.util.spec_from_file_location('cake_loader_test', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert not {'torch', 'tvm_ffi', 'cutlass', 'flashinfer'} & (set(sys.modules) - before)
"""
    subprocess.run([sys.executable, "-c", script, str(_SOURCE / "_cake" / "loader.py")], check=True, timeout=30)


def test_buffer_checks_use_metadata_only():
    buffers = _module("buffer_utils_test", _SOURCE / "_buffer_utils.py")

    class Tensor:
        shape = (2, 3)

        def __init__(self, address):
            self.address = address

        def data_ptr(self):
            return self.address

        def stride(self):
            return (8, 1)

        def element_size(self):
            return 2

        def numel(self):
            return 6

    assert buffers.tensor_byte_span(Tensor(128)) == (128, 150)
    with pytest.raises(ValueError, match="out must not alias input"):
        buffers.check_buffers_disjoint([("out", Tensor(132))], [("input", Tensor(128))])
    buffers.check_buffers_disjoint([("out", Tensor(160))], [("input", Tensor(128)), ("same_input", Tensor(128))])


@pytest.mark.parametrize("name", ["../cake_test", "cake_test.module", "torch", ""])
def test_package_names_are_local(loader, name):
    with pytest.raises(ValueError, match="invalid Cake package"):
        loader.load_package(name)
