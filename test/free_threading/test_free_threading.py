# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated binding probes, without the GPU/PyTorch test/python conftest.

Run against an installed build with CUDNN_FRONTEND_PYTHON_FREE_THREADED=ON:
    python -m pytest -q test/free_threading
Set CUDNN_TEST_FREE_THREADED_GPU=1 for the CUDA context and graph execution probes.
The same probes run on ordinary Python to cover the no-op lock implementation.
Only the native backend with independent graphs/handles is covered. Keep frontend
logging disabled; Python graph_cache and context fallback concurrency are not qualified.
"""

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.L0


def probe(body, environment=None):
    env = os.environ.copy()
    env.update(environment or {})
    # Never force the GIL off: that would hide a missing module declaration.
    env.pop("PYTHON_GIL", None)
    code = """
import sys
import sysconfig
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

# ctypes uses the secure DLL search path on Windows, not PATH.
dll_dirs = []
if sys.platform == "win32":
    dll_dirs = [os.add_dll_directory(path) for path in os.environ.get("CUDNN_LIBRARY_PATH", "").split(os.pathsep) if path]

def check_gil():
    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        assert not sys._is_gil_enabled(), "an imported extension enabled the GIL"

check_gil()
import cudnn
from cudnn import _pybind_module as native
check_gil()
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code) + textwrap.dedent(body) + "\ncheck_gil()\n"], env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_import_preserves_gil_mode():
    probe("assert cudnn.__file__ and native.__file__")


def test_independent_packs_share_the_cold_type_cache():
    probe("""
workers = 16
barrier = Barrier(workers)

def work(index):
    # No GPU dereference: the exchange protocol reads metadata only.
    pointer = 0x100000 + index * 4096
    source = native.make_operand_buffer(pointer, [2, 16], 2, 32, 0)
    pack = native.VariantPackNative(1)
    barrier.wait(timeout=30)
    for _ in range(1000):
        assert pack.read_operand(0, source)
        assert pack.pointer(0) == pointer
        assert list(pack.shape(0)) == [2, 16]
        assert pack.observed_bytes(0) == 128

with ThreadPoolExecutor(workers) as pool:
    list(pool.map(work, range(workers)))
""")


@pytest.mark.parametrize("same_type", [False, True])
def test_type_cache_saturation_and_reentrant_attribute_lookup(same_type):
    probe(f"same_type = {same_type!r}\n" + """
import ctypes
from threading import local

# Copy FE's real vtable, retaining its version/header and required functions,
# but exercise a producer without the optional direct-DLTensor function.
class API(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32),
                ("prev", ctypes.c_void_p), ("allocator", ctypes.c_void_p),
                ("managed_from", ctypes.c_void_p), ("managed_to", ctypes.c_void_p),
                ("dltensor", ctypes.c_void_p), ("stream", ctypes.c_void_p)]
get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
get_pointer.restype = ctypes.c_void_p
new_capsule = ctypes.pythonapi.PyCapsule_New
new_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
new_capsule.restype = ctypes.py_object
source = native.make_operand_buffer(0x100000, [32], 2, 32, 0)
original = type(source).__dlpack_c_exchange_api__
api = API.from_buffer_copy(ctypes.string_at(get_pointer(original, b"dlpack_exchange_api"), ctypes.sizeof(API)))
api.dltensor = None
capsule = new_capsule(ctypes.addressof(api), b"dlpack_exchange_api", None)
barrier = Barrier(16)
seen = local()

class ProducerType(type):
    def __getattribute__(cls, name):
        if name == "__dlpack_c_exchange_api__" and not getattr(seen, "first", False):
            seen.first = True
            # Holding the native type-cache mutex over attribute lookup would
            # deadlock here, even on a single thread.
            assert native.read_buffer_extent(source) == (0x100000, 128)
            barrier.wait(timeout=30)
        return type.__getattribute__(cls, name)

types = [ProducerType("Producer" + str(i), (), {"__dlpack_c_exchange_api__": capsule}) for i in range(1 if same_type else 16)]
producers = [types[i % len(types)]() for i in range(16)]
def work(producer):
    for _ in range(200):
        assert native.read_buffer_extent(producer) is None
        # Read a valid published entry while other types are being appended.
        assert native.read_buffer_extent(source) == (0x100000, 128)
with ThreadPoolExecutor(16) as pool:
    list(pool.map(work, producers))

if same_type:
    # Two occupied slots: the real buffer and the shared Producer. Duplicate
    # publications would consume the six slots that must still be available.
    class CountLookups(type):
        def __getattribute__(cls, name):
            if name == "__dlpack_c_exchange_api__":
                cls.lookups += 1
            return type.__getattribute__(cls, name)
    for i in range(6):
        cls = CountLookups("Remaining" + str(i), (), {"lookups": 0, "__dlpack_c_exchange_api__": capsule})
        instance = cls()
        assert native.read_buffer_extent(instance) is None
        assert native.read_buffer_extent(instance) is None
        assert cls.lookups == 1, "duplicate publication consumed a cache slot"
""")


@pytest.mark.skipif(os.environ.get("CUDNN_TEST_FREE_THREADED_GPU") != "1", reason="set CUDNN_TEST_FREE_THREADED_GPU=1 on a CUDA host")
def test_native_cold_threads_retain_one_primary_context():
    probe("""
import ctypes
drv = (ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1"))
drv.cuInit.argtypes = [ctypes.c_uint]
drv.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
drv.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
drv.cuDevicePrimaryCtxRelease.argtypes = [ctypes.c_int]
drv.cuDevicePrimaryCtxGetState.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_int)]
assert drv.cuInit(0) == 0
flags, active = ctypes.c_uint(), ctypes.c_int()
assert drv.cuDevicePrimaryCtxGetState(0, ctypes.byref(flags), ctypes.byref(active)) == 0
assert active.value == 0, "probe must own the primary context exclusively"
barrier = Barrier(16)
def work(index):
    ctx = ctypes.c_void_p()
    assert drv.cuCtxGetCurrent(ctypes.byref(ctx)) == 0 and not ctx.value
    barrier.wait(timeout=30)
    for _ in range(20):
        assert native._try_ensure_current_context(0, 0)
        assert drv.cuCtxGetCurrent(ctypes.byref(ctx)) == 0 and ctx.value
        assert drv.cuCtxSetCurrent(None) == 0
    return ctx.value
with ThreadPoolExecutor(16) as pool:
    contexts = list(pool.map(work, range(16)))
assert len(set(contexts)) == 1
assert drv.cuDevicePrimaryCtxRelease(0) == 0
assert drv.cuDevicePrimaryCtxGetState(0, ctypes.byref(flags), ctypes.byref(active)) == 0
assert active.value == 0, "concurrent misses retained the primary context more than once"
""")


@pytest.mark.skipif(os.environ.get("CUDNN_TEST_FREE_THREADED_GPU") != "1", reason="set CUDNN_TEST_FREE_THREADED_GPU=1 on a CUDA host")
def test_independent_graphs_execute_on_independent_streams():
    probe(
        """
import ctypes
drv = (ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1"))
u64, ptr, size = ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t
signatures = {
    "cuMemAlloc_v2": [ctypes.POINTER(u64), size],
    "cuMemFree_v2": [u64],
    "cuMemcpyHtoD_v2": [u64, ptr, size],
    "cuMemcpyDtoH_v2": [ptr, u64, size],
    "cuStreamCreate": [ctypes.POINTER(ptr), ctypes.c_uint],
    "cuStreamSynchronize": [ptr],
    "cuStreamDestroy_v2": [ptr],
}
for name, args in signatures.items():
    getattr(drv, name).argtypes = args
def checked(name, *args):
    status = getattr(drv, name)(*args)
    assert status == 0, (name, status)

barrier = Barrier(4)
def work(index):
    assert native._try_ensure_current_context(0, 0)
    allocations = []
    stream = ptr()
    handle = None
    def allocate(nbytes):
        address = u64()
        checked("cuMemAlloc_v2", ctypes.byref(address), max(nbytes, 1))
        allocations.append(address)
        return address.value
    try:
        checked("cuStreamCreate", ctypes.byref(stream), 1)
        handle = cudnn.create_handle()
        cudnn.set_stream(handle, stream.value)
        graph = cudnn.pygraph(handle=handle, io_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        a = graph.tensor(name="a", dim=[1, 32, 1, 1], stride=[32, 1, 32, 32], data_type=cudnn.data_type.FLOAT)
        b = graph.tensor(name="b", dim=[1, 32, 1, 1], stride=[32, 1, 32, 32], data_type=cudnn.data_type.FLOAT)
        c = graph.add(a, b)
        c.set_output(True).set_data_type(cudnn.data_type.FLOAT)
        barrier.wait(timeout=30)
        graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        assert graph.selected_engine is None, "this probe qualifies the native backend"
        ap, bp, cp = [allocate(128) for _ in range(3)]
        workspace = allocate(graph.get_workspace_size())
        output = (ctypes.c_float * 32)()
        for iteration in range(10):
            av = (ctypes.c_float * 32)(*[float(index + j) for j in range(32)])
            bv = (ctypes.c_float * 32)(*[float(iteration + j) for j in range(32)])
            checked("cuMemcpyHtoD_v2", ap, av, 128)
            checked("cuMemcpyHtoD_v2", bp, bv, 128)
            graph.execute({a: ap, b: bp, c: cp}, workspace, handle=handle)
            checked("cuStreamSynchronize", stream)
            checked("cuMemcpyDtoH_v2", output, cp, 128)
            assert list(output) == [float(index + iteration + 2 * j) for j in range(32)]
    finally:
        if stream.value:
            checked("cuStreamSynchronize", stream)
        if handle is not None:
            cudnn.destroy_handle(handle)
        for address in allocations:
            checked("cuMemFree_v2", address)
        if stream.value:
            checked("cuStreamDestroy_v2", stream)

with ThreadPoolExecutor(4) as pool:
    list(pool.map(work, range(4)))
""",
        {"CUDNN_FRONTEND_LOG_INFO": "0"},
    )
