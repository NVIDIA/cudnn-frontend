# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU regression tests using real ctypes native-call boundaries, without CUDA."""

import ctypes
import gc
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import weakref

import pytest

pytestmark = pytest.mark.L0

from cudnn.simplified_sparse_attention.cutedsl.kernels import _cuda_preload as preload


class NativeModule:
    def __init__(self, count=1):
        self.count = count
        self.events = []
        self.init_error = 0
        self.load_error = 0
        self.init_prefix = count
        self.missing = None
        self.unloaded = []
        self.callbacks = {
            "cuda_num_binaries": ctypes.CFUNCTYPE(ctypes.c_int32)(self.num_binaries),
            "cuda_init": ctypes.CFUNCTYPE(ctypes.c_int32, preload._LibraryArray)(self.init),
            "cuda_load_to_device": ctypes.CFUNCTYPE(ctypes.c_int32, preload._LibraryArray, ctypes.c_int32)(self.load),
        }

    def lookup(self, name):
        if name == self.missing:
            return 0
        return ctypes.cast(self.callbacks[name], ctypes.c_void_p).value

    def num_binaries(self):
        self.events.append(("count",))
        return self.count

    def init(self, libraries):
        self.events.append(("init",))
        for index in range(self.init_prefix):
            libraries[index] = 0x1000 + index
        return self.init_error

    def load(self, libraries, device):
        self.events.append(("load", tuple(libraries[i] for i in range(self.count)), device))
        return self.load_error


@pytest.fixture
def make_module(monkeypatch):
    unloaded = []

    def unload(address):
        unloaded.append(address)
        return (0,)

    monkeypatch.setattr(
        preload,
        "_cuda_runtime",
        lambda: SimpleNamespace(
            cudaLibraryUnload=unload,
            cudaLibrary_t=lambda address: address,
        ),
    )

    def make(count=1):
        native = NativeModule(count)
        native.unloaded = unloaded
        return SimpleNamespace(engine=native)

    return make


@pytest.mark.parametrize("count", [1, 3])
def test_native_abi_supports_every_library_slot_and_integer_device(make_module, count):
    compiled = make_module(count)
    preload.preload_tvm_ffi_cuda(compiled, 3)
    assert compiled.engine.events == [
        ("count",),
        ("init",),
        ("load", tuple(0x1000 + i for i in range(count)), 3),
    ]
    assert len(compiled._ssa_cuda_preload.libraries) == count
    assert compiled._ssa_preloaded_devices == {3}


def test_repeated_device_is_cached_and_second_device_reuses_libraries(make_module):
    compiled = make_module(2)
    preload.preload_tvm_ffi_cuda(compiled, 0)
    preload.preload_tvm_ffi_cuda(compiled, 0)
    preload.preload_tvm_ffi_cuda(compiled, 1)
    assert compiled.engine.events == [
        ("count",),
        ("init",),
        ("load", (0x1000, 0x1001), 0),
        ("load", (0x1000, 0x1001), 1),
    ]
    assert compiled._ssa_preloaded_devices == {0, 1}


def test_init_return_error_releases_partial_array_and_allows_retry(make_module):
    compiled = make_module(3)
    native = compiled.engine
    native.init_prefix = 2
    native.init_error = 17
    with pytest.raises(RuntimeError, match="initialization failed: 17"):
        preload.preload_tvm_ffi_cuda(compiled, 0)
    assert native.events == [("count",), ("init",)]
    assert native.unloaded == [0x1000, 0x1001]
    assert not hasattr(compiled, "_ssa_cuda_preload")
    assert not hasattr(compiled, "_ssa_preloaded_devices")

    native.init_prefix = 3
    native.init_error = 0
    preload.preload_tvm_ffi_cuda(compiled, 0)
    assert native.events[-1] == ("load", (0x1000, 0x1001, 0x1002), 0)
    assert compiled._ssa_preloaded_devices == {0}


def test_load_return_error_is_not_cached_and_retry_does_not_reload_libraries(make_module):
    compiled = make_module()
    native = compiled.engine
    native.load_error = 101
    with pytest.raises(RuntimeError, match="device 2: 101"):
        preload.preload_tvm_ffi_cuda(compiled, 2)
    assert compiled._ssa_preloaded_devices == set()
    assert native.unloaded == []
    native.load_error = 0
    preload.preload_tvm_ffi_cuda(compiled, 2)
    assert native.events == [
        ("count",),
        ("init",),
        ("load", (0x1000,), 2),
        ("load", (0x1000,), 2),
    ]
    assert compiled._ssa_preloaded_devices == {2}


@pytest.mark.parametrize("count", [0, -1])
def test_invalid_binary_count_never_initializes(make_module, count):
    compiled = make_module(count)
    with pytest.raises(RuntimeError, match="invalid count"):
        preload.preload_tvm_ffi_cuda(compiled, 0)
    assert compiled.engine.events == [("count",)]
    assert not hasattr(compiled, "_ssa_cuda_preload")


@pytest.mark.parametrize("symbol", ["cuda_num_binaries", "cuda_init", "cuda_load_to_device"])
def test_missing_symbol_fails_before_native_calls(make_module, symbol):
    compiled = make_module()
    compiled.engine.missing = symbol
    with pytest.raises(RuntimeError, match=symbol):
        preload.preload_tvm_ffi_cuda(compiled, 0)
    assert compiled.engine.events == []


def test_packed_engine_callable_is_rejected_before_native_calls(make_module):
    compiled = make_module()
    native = compiled.engine
    native.lookup = lambda name: native.callbacks[name]
    with pytest.raises(RuntimeError, match="no valid address"):
        preload.preload_tvm_ffi_cuda(compiled, 0)
    assert native.events == []


@pytest.mark.parametrize("device", [-1, 1 << 31])
def test_invalid_int32_device_never_calls_native_code(make_module, device):
    compiled = make_module()
    with pytest.raises(ValueError, match="device ordinal"):
        preload.preload_tvm_ffi_cuda(compiled, device)
    assert compiled.engine.events == []


def test_libraries_live_with_compiled_graph_reference_then_unload_once(make_module):
    compiled = make_module(3)
    unloaded = compiled.engine.unloaded
    preload.preload_tvm_ffi_cuda(compiled, 0)
    owner_ref = weakref.ref(compiled._ssa_cuda_preload)
    # The actual repair graph cache retains its compiled functions/launcher.
    graph_cache = {"compiled": compiled}
    del compiled
    gc.collect()
    assert owner_ref() is not None
    assert unloaded == []
    graph_cache.clear()
    gc.collect()
    assert owner_ref() is None
    assert unloaded == [0x1000, 0x1001, 0x1002]
    gc.collect()
    assert unloaded == [0x1000, 0x1001, 0x1002]


def test_concurrent_host_preloads_initialize_and_load_once(make_module):
    compiled = make_module()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: preload.preload_tvm_ffi_cuda(compiled, 0), range(12)))
    assert compiled.engine.events == [("count",), ("init",), ("load", (0x1000,), 0)]
