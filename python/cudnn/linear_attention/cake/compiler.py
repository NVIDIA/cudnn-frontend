# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVRTC compilation, cubin caching and driver-API plumbing for the frozen CAKE
kernel bodies under ``kernels/``.

The bodies are plain CUDA C++: ``extern "C" __global__`` kernels taking
``__grid_constant__ CUtensorMap`` descriptors, raw pointers and scalars, and
including only ``cuda_bf16.h`` / ``math_constants.h``. NVRTC therefore needs
the CUDA headers and nothing else. Cubins are cached on disk, keyed by source
digest, compile options and NVRTC version.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sys
import threading
from array import array
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from cuda.bindings import driver as cu
from cuda.bindings import nvrtc

from cudnn.frost.device import compute_capability, device_context

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"

# What FlashInfer's nvcc build gave the same bodies (-O3 -std=c++17 -use_fast_math
# -DNDEBUG); NVRTC optimizes at -O3 by default. -default-device is what the
# CAKE exporters compile their generated bodies with under NVRTC.
_BASE_OPTIONS = ("-std=c++17", "-default-device", "--use_fast_math", "-DNDEBUG")

_LOCK = threading.Lock()
_LIBRARIES: Dict[Tuple[str, str, int], "KernelLibrary"] = {}


class CakeCompileError(RuntimeError):
    """The frozen bodies could not be compiled in this environment."""


def _ck(result, what: str):
    if isinstance(result, tuple):
        err, *rest = result
    else:
        err, rest = result, []
    if int(err) != 0:
        raise RuntimeError(f"cudnn.linear_attention.cake: {what} failed: {err}")
    if not rest:
        return None
    return rest[0] if len(rest) == 1 else tuple(rest)


def arch_for_device(device: int) -> str:
    major, minor = compute_capability(device)
    if (major, minor) not in ((10, 0), (10, 3)):
        raise CakeCompileError(f"the frozen CAKE bodies target exact compute capability 10.0 or 10.3, got {major}.{minor}")
    return f"sm_{major}{minor}a"


def cuda_include_dirs() -> Tuple[Path, ...]:
    """The directories NVRTC needs for ``cuda_bf16.h`` (plus the CCCL tree when
    the toolkit ships it separately): a toolkit named by ``CUDA_HOME`` /
    ``CUDA_PATH`` / ``CUDAToolkit_ROOT``, else the pip ``nvidia-cuda-*`` wheels,
    else ``/usr/local/cuda``."""
    candidates = []
    for var in ("CUDA_HOME", "CUDA_PATH", "CUDAToolkit_ROOT"):
        root = os.environ.get(var)
        if root:
            candidates.append(Path(root) / "include")
    for base in sys.path:
        nvidia = Path(base) / "nvidia"
        if nvidia.is_dir():
            for wheel in ("cuda_runtime", "cuda_nvcc", "cuda_cccl"):
                candidates.append(nvidia / wheel / "include")
    candidates.append(Path("/usr/local/cuda/include"))
    for include in candidates:
        if (include / "cuda_bf16.h").is_file():
            dirs = [include]
            for cccl in (include / "cccl", include.parent / "include" / "cccl"):
                if (cccl / "cuda" / "std").is_dir():
                    dirs.append(cccl)
                    break
            return tuple(dirs)
    raise CakeCompileError("CUDA headers (cuda_bf16.h) not found; set CUDA_HOME or install nvidia-cuda-runtime")


def nvrtc_version() -> Tuple[int, int]:
    major, minor = _ck(nvrtc.nvrtcVersion(), "nvrtcVersion")
    return int(major), int(minor)


def cache_dir() -> Path:
    root = os.environ.get("CUDNN_FRONTEND_CAKE_CACHE_DIR")
    if root:
        return Path(root)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "cudnn_frontend" / "cake"


def _program_log(program) -> str:
    size = _ck(nvrtc.nvrtcGetProgramLogSize(program), "nvrtcGetProgramLogSize")
    log = b"\0" * int(size)
    _ck(nvrtc.nvrtcGetProgramLog(program, log), "nvrtcGetProgramLog")
    return log.decode(errors="replace").rstrip("\0\n")


def compile_cubin(source: Path, arch: str) -> bytes:
    """The cubin for one body, from the on-disk cache when the same source has
    been compiled with the same options and NVRTC before."""
    src = source.read_bytes()
    includes = cuda_include_dirs()
    options = [f"--gpu-architecture={arch}", *_BASE_OPTIONS, *(f"-I{path}" for path in includes)]
    version = nvrtc_version()
    digest = hashlib.sha256(b"\0".join([src, json.dumps(options).encode(), repr(version).encode(), b"cake-cubin-v1"])).hexdigest()[:24]
    cached = cache_dir() / f"{source.stem}_{arch}_{digest}.cubin"
    if cached.is_file():
        return cached.read_bytes()

    program = _ck(nvrtc.nvrtcCreateProgram(src, source.name.encode(), 0, [], []), "nvrtcCreateProgram")
    try:
        (err,) = nvrtc.nvrtcCompileProgram(program, len(options), [option.encode() for option in options])
        if int(err) != 0:
            raise CakeCompileError(f"NVRTC failed on {source.name} ({err}):\n{_program_log(program)}")
        size = _ck(nvrtc.nvrtcGetCUBINSize(program), "nvrtcGetCUBINSize")
        cubin = b"\0" * int(size)
        _ck(nvrtc.nvrtcGetCUBIN(program, cubin), "nvrtcGetCUBIN")
    finally:
        nvrtc.nvrtcDestroyProgram(program)

    cached.parent.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(cubin)
    os.replace(tmp, cached)
    return cubin


class KernelLibrary:
    """One compiled body loaded into one device's primary context."""

    def __init__(self, source: Path, arch: str, device: int):
        from cudnn._device import _device_handle

        self.source = source
        self.device = device
        self.cubin = compile_cubin(source, arch)
        # Hold our own reference on the primary context: a module dies with its
        # context, and nothing else need have retained it yet on this device.
        self._device_handle = _device_handle(device)
        self._primary = _ck(cu.cuDevicePrimaryCtxRetain(self._device_handle), "cuDevicePrimaryCtxRetain")
        with device_context(device):
            self.module = _ck(cu.cuModuleLoadData(self.cubin), f"cuModuleLoadData({source.name})")
        self._functions: Dict[str, cu.CUfunction] = {}
        self._dynamic_smem: Dict[str, int] = {}

    def __del__(self):
        try:
            cu.cuDevicePrimaryCtxRelease(self._device_handle)
        except Exception:  # noqa: BLE001 — interpreter teardown may have dropped the driver already
            pass

    def function(self, name: str, dynamic_smem: int = 0) -> cu.CUfunction:
        func = self._functions.get(name)
        if func is not None and self._dynamic_smem.get(name, 0) >= dynamic_smem:
            return func
        # Module handles are only valid with their context current; the calling
        # thread (an autograd worker, say) may have none bound.
        with device_context(self.device):
            if func is None:
                func = _ck(cu.cuModuleGetFunction(self.module, name.encode()), f"cuModuleGetFunction({name})")
                self._functions[name] = func
            if dynamic_smem and self._dynamic_smem.get(name, 0) < dynamic_smem:
                attr = cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
                _ck(cu.cuFuncSetAttribute(func, attr, int(dynamic_smem)), f"cuFuncSetAttribute({name}, dynamic smem {dynamic_smem})")
                self._dynamic_smem[name] = dynamic_smem
        return func


def library(body: str, device: int) -> KernelLibrary:
    """The loaded library for ``kernels/<body>`` on ``device`` (process-wide cache)."""
    arch = arch_for_device(device)
    key = (body, arch, int(device))
    with _LOCK:
        lib = _LIBRARIES.get(key)
        if lib is None:
            lib = KernelLibrary(KERNEL_DIR / body, arch, int(device))
            _LIBRARIES[key] = lib
        return lib


class Params:
    """A ``void*[]`` kernel parameter block: each entry points at one argument's
    storage, which lives here until the launch has copied it."""

    __slots__ = ("_keep", "_pointers")

    def __init__(self):
        self._keep = []
        self._pointers = []

    def ptr(self, address: int) -> "Params":
        value = ctypes.c_void_p(int(address))
        self._keep.append(value)
        self._pointers.append(ctypes.addressof(value))
        return self

    def i32(self, value: int) -> "Params":
        boxed = ctypes.c_int32(int(value))
        self._keep.append(boxed)
        self._pointers.append(ctypes.addressof(boxed))
        return self

    def i64(self, value: int) -> "Params":
        boxed = ctypes.c_int64(int(value))
        self._keep.append(boxed)
        self._pointers.append(ctypes.addressof(boxed))
        return self

    def f32(self, value: float) -> "Params":
        boxed = ctypes.c_float(float(value))
        self._keep.append(boxed)
        self._pointers.append(ctypes.addressof(boxed))
        return self

    def tensor_map(self, tensor_map: cu.CUtensorMap) -> "Params":
        self._keep.append(tensor_map)
        self._pointers.append(int(tensor_map.getPtr()))
        return self

    def address(self) -> int:
        block = (ctypes.c_void_p * len(self._pointers))(*self._pointers)
        self._keep.append(block)
        return ctypes.addressof(block)

    def __len__(self) -> int:
        return len(self._pointers)


def launch(func: cu.CUfunction, grid: Sequence[int], block: Sequence[int], dynamic_smem: int, stream: int, params: Params, what: str) -> None:
    _ck(
        cu.cuLaunchKernel(
            func,
            int(grid[0]),
            int(grid[1]),
            int(grid[2]),
            int(block[0]),
            int(block[1]),
            int(block[2]),
            int(dynamic_smem),
            cu.CUstream(int(stream)),
            params.address(),
            0,
        ),
        f"cuLaunchKernel({what})",
    )


_BF16 = cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
SWIZZLE_128B = cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
SWIZZLE_NONE = cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE


def encode_tiled(
    address: int, global_dim: Sequence[int], global_strides_bytes: Sequence[int], box_dim: Sequence[int], *, swizzle=SWIZZLE_128B, what: str = "tensor"
) -> cu.CUtensorMap:
    """``cuTensorMapEncodeTiled`` for a bf16 tensor, innermost dimension first."""
    rank = len(global_dim)
    return _ck(
        cu.cuTensorMapEncodeTiled(
            _BF16,
            rank,
            int(address),
            [cu.cuuint64_t(int(d)) for d in global_dim],
            [cu.cuuint64_t(int(s)) for s in global_strides_bytes],
            [cu.cuuint32_t(int(b)) for b in box_dim],
            [cu.cuuint32_t(1) for _ in range(rank)],
            cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle,
            cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
            cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        ),
        f"cuTensorMapEncodeTiled({what})",
    )


def memset_d32(address: int, value: int, count: int, stream: int) -> None:
    if count > 0:
        _ck(cu.cuMemsetD32Async(int(address), int(value) & 0xFFFFFFFF, int(count), cu.CUstream(int(stream))), "cuMemsetD32Async")


def memcpy_dtod(dst: int, src: int, nbytes: int, stream: int) -> None:
    if nbytes > 0:
        _ck(cu.cuMemcpyDtoDAsync(int(dst), int(src), int(nbytes), cu.CUstream(int(stream))), "cuMemcpyDtoDAsync")


def memcpy_htod(dst: int, host: array, stream: int) -> None:
    """Upload an ``array.array``; pageable-source copies return once the data is
    staged, so the array may be dropped right after."""
    nbytes = len(host) * host.itemsize
    if nbytes > 0:
        _ck(cu.cuMemcpyHtoDAsync(int(dst), host.buffer_info()[0], nbytes, cu.CUstream(int(stream))), "cuMemcpyHtoDAsync")


def read_device_ints(address: int, count: int, itemsize: int, stream: int) -> Tuple[int, ...]:
    """Synchronously read ``count`` int32/int64 values: an async copy ordered
    after the stream's pending work, then a stream sync. This is the one host
    synchronization the CAKE route needs (its work items are planned on the host)."""
    typecode = "q" if itemsize == 8 else "i"
    host = array(typecode, bytes(count * itemsize))
    if count > 0:
        _ck(cu.cuMemcpyDtoHAsync(host.buffer_info()[0], int(address), count * itemsize, cu.CUstream(int(stream))), "cuMemcpyDtoHAsync")
        _ck(cu.cuStreamSynchronize(cu.CUstream(int(stream))), "cuStreamSynchronize")
    return tuple(int(v) for v in host)


def check_not_capturing(stream: int, what: str) -> None:
    status = _ck(cu.cuStreamIsCapturing(cu.CUstream(int(stream))), "cuStreamIsCapturing")
    if status != cu.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE:
        raise RuntimeError(
            f"{what}: the CAKE route plans its work items on the host and cannot run under CUDA graph capture; pin kda_frost for captured graphs"
        )


def source_digests() -> Dict[str, str]:
    """sha256 of every vendored body, for tests against ``kernels/SHA256SUMS``."""
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(KERNEL_DIR.glob("*.cu"))}


__all__ = [
    "CakeCompileError",
    "KERNEL_DIR",
    "KernelLibrary",
    "Params",
    "SWIZZLE_128B",
    "SWIZZLE_NONE",
    "arch_for_device",
    "cache_dir",
    "check_not_capturing",
    "compile_cubin",
    "cuda_include_dirs",
    "encode_tiled",
    "launch",
    "library",
    "memcpy_dtod",
    "memcpy_htod",
    "memset_d32",
    "nvrtc_version",
    "read_device_ints",
    "source_digests",
]
