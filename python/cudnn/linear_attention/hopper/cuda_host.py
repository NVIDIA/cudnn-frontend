# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host half of the fused sm90 KDA prefill kernel.

The kernel is one ``__global__``; this reproduces its launch through the driver
API, reusing ``linear_attention/cake/compiler.py`` for NVRTC compilation, module
caching and the dynamic shared-memory attribute. The campaign artifact's own
``kda_launch`` is not vendored -- NVRTC compiles device code only -- so the grid
and the residency heuristic are restated here and nowhere else.

``compiler.library()`` resolves bodies against ``cake/kernels``, so this keeps
its own equivalent cache over ``hopper/cuda_kernels`` rather than putting an
unrelated kernel in cake's directory.
"""

from __future__ import annotations

import ctypes
import threading
from pathlib import Path
from typing import Dict, Tuple

from cuda.bindings import driver as cu

from cudnn.frost.device import compute_capability, device_context

from ..cake import compiler

HOPPER_CC = (9, 0)

KERNEL_DIR = Path(__file__).resolve().parent / "cuda_kernels"
KERNEL_BODY = "kda_fused_sm90.cu"
KERNEL_NAME = "kda_fused"
CHUNK = 16
BLOCK = 128

# The campaign kernel targets ~264 resident CTAs; a launch splits each sequence
# into P pieces so N*H*P lands near that without exceeding the number of chunks
# a sequence actually has. Restated from the artifact's kda_launch.
TARGET_CTAS = 264

_LOCK = threading.Lock()
_LIBRARIES: Dict[Tuple[str, int], "compiler.KernelLibrary"] = {}
_SMEM_BYTES: Dict[int, int] = {}


def _arch_for_device(device: int) -> str:
    """``sm_90a`` for Hopper.

    Deliberately NOT ``compiler.arch_for_device``: that one hard-gates to
    compute capability 10.0/10.3 because cake's frozen bodies are Blackwell-only,
    and it raises ``CakeCompileError`` on sm90. Everything else in
    ``cake/compiler`` -- ``compile_cubin``, ``KernelLibrary``, ``Params``,
    ``launch`` -- takes the arch as a parameter and is architecture-agnostic, so
    only this one function needs replacing.
    """
    major, minor = compute_capability(device)
    if (major, minor) != HOPPER_CC:
        raise NotImplementedError(f"the fused sm90 KDA kernel targets compute capability 9.0, got {major}.{minor}")
    return "sm_90a"


def _library(device: int) -> "compiler.KernelLibrary":
    key = (KERNEL_BODY, int(device))
    with _LOCK:
        lib = _LIBRARIES.get(key)
        if lib is None:
            lib = compiler.KernelLibrary(KERNEL_DIR / KERNEL_BODY, _arch_for_device(device), int(device))
            _LIBRARIES[key] = lib
        return lib


def _smem_bytes(device: int) -> int:
    """``sizeof(FusedSmem)`` read out of the compiled module.

    The kernel publishes it as a device global so the launcher never restates
    the struct layout -- a Python copy would drift silently the first time the
    kernel's staging buffers changed.
    """
    cached = _SMEM_BYTES.get(device)
    if cached is not None:
        return cached
    lib = _library(device)
    with _LOCK:
        cached = _SMEM_BYTES.get(device)
        if cached is not None:
            return cached
        with device_context(device):
            ptr, size = compiler._ck(
                cu.cuModuleGetGlobal(lib.module, b"kda_fused_smem_bytes"),
                "cuModuleGetGlobal(kda_fused_smem_bytes)",
            )
            buf = ctypes.c_ulonglong()
            compiler._ck(
                cu.cuMemcpyDtoH(ctypes.addressof(buf), ptr, min(int(size), 8)),
                "cuMemcpyDtoH(kda_fused_smem_bytes)",
            )
        _SMEM_BYTES[device] = int(buf.value)
        return _SMEM_BYTES[device]


def pieces_per_sequence(total_tokens: int, n_seqs: int, n_heads: int) -> int:
    """``P`` from the artifact's launcher: fill the machine, but never split a
    sequence into more pieces than it has chunks."""
    approx = ((total_tokens + n_seqs - 1) // n_seqs + CHUNK - 1) // CHUNK
    p = TARGET_CTAS // max(n_seqs * n_heads, 1)
    if p < 1:
        p = 1
    if p > approx:
        p = approx if approx >= 1 else 1
    return p


def launch(
    device: int,
    stream: int,
    q: int,
    k: int,
    v: int,
    g: int,
    beta: int,
    cu_seqlens: int,
    initial_state: int,
    o: int,
    final_state: int,
    total_tokens: int,
    n_seqs: int,
    n_heads: int,
) -> None:
    """One ``kda_fused`` launch. All tensor arguments are device addresses."""
    smem = _smem_bytes(device)
    func = _library(device).function(KERNEL_NAME, dynamic_smem=smem)
    p = pieces_per_sequence(total_tokens, n_seqs, n_heads)

    params = compiler.Params()
    for address in (q, k, v, g, beta, cu_seqlens, initial_state, o, final_state):
        params.ptr(address)
    for scalar in (n_seqs, n_heads, p):
        params.i32(scalar)

    compiler.launch(
        func,
        grid=(n_heads, n_seqs * p, 1),
        block=(BLOCK, 1, 1),
        dynamic_smem=smem,
        stream=stream,
        params=params,
        what=f"{KERNEL_NAME}(T={total_tokens}, N={n_seqs}, H={n_heads}, P={p})",
    )
