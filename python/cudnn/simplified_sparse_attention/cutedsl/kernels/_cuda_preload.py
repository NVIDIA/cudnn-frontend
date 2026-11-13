# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Load CuTe BinaryExecutionEngine CUDA modules without launching a kernel.

BinaryExecutionEngine.lookup returns native symbols; MLIR's packed invocation
ABI does not apply. TVM FFI's public .to() is a no-op in the supported runtime,
so these libraries have their own owner, separate from FFI's lazy-init state.
The compiled object (also retained by the repair graph cache) owns this state.
"""

import ctypes
import logging
import operator
import threading
import weakref

_lock = threading.Lock()
_logger = logging.getLogger(__name__)
_LibraryArray = ctypes.POINTER(ctypes.c_void_p)


def _native_function(engine, name, *argument_types):
    address = engine.lookup(name)
    if not isinstance(address, int) or address <= 0:
        raise RuntimeError(f"CuTe native symbol {name!r} has no valid address")
    return ctypes.CFUNCTYPE(ctypes.c_int32, *argument_types)(address)


def _cuda_runtime():
    from cuda.bindings import runtime

    return runtime


def _release_libraries(libraries, unload, library_type, _engine):
    # Keep the engine alive until all of its libraries have been released.
    # Finalizers cannot propagate cleanup failures to the original caller.
    for index, address in enumerate(libraries):
        if address:
            try:
                (error,) = unload(library_type(address))
                if int(error):
                    _logger.warning("CuTe preload CUDA library unload failed: %s", error)
            except Exception:
                _logger.exception("CuTe preload CUDA library unload failed")
            libraries[index] = None


class _LibraryOwner:
    def __init__(self, compiled, count, load):
        runtime = _cuda_runtime()
        self.libraries = (ctypes.c_void_p * count)()
        self.load = load
        self.devices = set()
        self.finalizer = weakref.finalize(
            self,
            _release_libraries,
            self.libraries,
            runtime.cudaLibraryUnload,
            runtime.cudaLibrary_t,
            compiled.engine,
        )
        # CUDA/Python shutdown order is not reliable. Process exit reclaims
        # these resources; ordinary object destruction still unloads them.
        self.finalizer.atexit = False


def preload_tvm_ffi_cuda(compiled, device: int):
    """Resolve native CUDA functions before capture, with no kernel launch.

    One array slot is required for every cuda_num_binaries() entry. Each library
    is initialized once, loaded once per successful device, and kept alive while
    the compiled object is reachable. A failed device load can be retried without
    reinitializing the libraries. Concurrent host callers share the same owner.
    """
    device = operator.index(device)
    if not 0 <= device <= 0x7FFFFFFF:
        raise ValueError(f"CUDA device ordinal does not fit a nonnegative int32: {device}")
    with _lock:
        owner = getattr(compiled, "_ssa_cuda_preload", None)
        if owner is None:
            engine = compiled.engine
            count_function = _native_function(engine, "cuda_num_binaries")
            init = _native_function(engine, "cuda_init", _LibraryArray)
            load = _native_function(engine, "cuda_load_to_device", _LibraryArray, ctypes.c_int32)
            count = count_function()
            if count <= 0:
                raise RuntimeError(f"CuTe cuda_num_binaries returned invalid count {count}")
            owner = _LibraryOwner(compiled, count, load)
            error = init(owner.libraries)
            if error:
                # cuda_init may have loaded a prefix before returning an error.
                owner.finalizer()
                raise RuntimeError(f"TVM-FFI CUDA module initialization failed: {error}")
            compiled._ssa_cuda_preload = owner
            compiled._ssa_preloaded_devices = owner.devices
        if device in owner.devices:
            return
        error = owner.load(owner.libraries, device)
        if error:
            raise RuntimeError(f"TVM-FFI CUDA module load failed on device {device}: {error}")
        owner.devices.add(device)
