# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host half of the fused sm90 KDA backward kernels.

The backward is four ``__global__``s, not one: ``k_meta`` (chunk table, varlen
only), ``k_prep``, ``k_scan_t`` and ``k_bwd``. Seven of the nine entry points
the launcher can reach are template instantiations, so unlike the forward this
cannot go through ``cake.compiler.compile_cubin``: a templated kernel has no
stable C name, and NVRTC only emits and names one if it is registered with
``nvrtcAddNameExpression`` before the compile and looked up with
``nvrtcGetLoweredName`` after it. That is the documented NVRTC pattern (a
``__global__`` cannot be called from another ``__global__``, so an ``extern
"C"`` wrapper is not an option), and it is why this file has its own compile
path. Everything else -- include discovery, the cubin cache directory, the
driver-API launch plumbing -- is reused from ``cake.compiler``.

The launch sequence, the workspace arena and the kernel-selection rules are
restated from the campaign artifact's ``kda_bwd_launch``, which is host code
and therefore not vendored.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Dict, Tuple

from cuda.bindings import driver as cu
from cuda.bindings import nvrtc

from cudnn.frost.device import compute_capability, device_context

from ..cake import compiler

HOPPER_CC = (9, 0)

KERNEL_DIR = Path(__file__).resolve().parent / "cuda_kernels"
KERNEL_BODY = "kda_bwd_sm90.cu"

# Mirrors of the kernel's own constants. The body static_asserts every one of
# these against the -D below, so a change to its tiling is a compile error
# naming the constant rather than a wrong grid or an undersized allocation.
BT = 64  # chunk length
DH = 128  # head dim (K == V == 128)
BNT = 512  # k_bwd threads
SMEM_PREP = 220160
SMEM_SCAN = 198144
SMEM_BWD = 230656

# What the artifact requests for k_prep/k_scan_t -- a flat cap above both
# SMEM_PREP and SMEM_SCAN rather than each kernel's exact figure.
SMEM_OPTIN = 227000

_ALIGN = 256

# Every entry point the launcher can reach. Registered as NVRTC name
# expressions before the compile; anything missing here simply would not be
# emitted, so the list is the contract.
_NAME_EXPRESSIONS = (
    "k_prep<true>",
    "k_prep<false>",
    "k_scan_t<16, 128, true, true>",
    "k_scan_t<16, 128, true, false>",
    "k_scan_t<32, 256, true, true>",
    "k_scan_t<32, 256, true, false>",
    "k_scan_t<32, 256, false, false>",
    "k_bwd<true>",
    "k_bwd<false>",
)

# -std=c++17 -default-device --use_fast_math -DNDEBUG matches what the CAKE
# bodies use. The artifact's nvcc build also passed --expt-relaxed-constexpr;
# NVRTC has no such option and does not need one, because -default-device
# already compiles the whole translation unit as device code.
_BASE_OPTIONS = ("-std=c++17", "-default-device", "--use_fast_math", "-DNDEBUG")

_DEFINES = (
    "-DKDA_BWD_CHECK_CONSTANTS",
    f"-DKDA_BWD_BT={BT}",
    f"-DKDA_BWD_DH={DH}",
    f"-DKDA_BWD_BNT={BNT}",
    f"-DKDA_BWD_SMEM_PREP={SMEM_PREP}",
    f"-DKDA_BWD_SMEM_SCAN={SMEM_SCAN}",
    f"-DKDA_BWD_SMEM_BWD={SMEM_BWD}",
)

_LOCK = threading.Lock()
_LIBRARIES: Dict[int, "BwdLibrary"] = {}


def _arch_for_device(device: int) -> str:
    """``sm_90a`` for Hopper.

    Deliberately not ``compiler.arch_for_device``: that one hard-gates to
    compute capability 10.0/10.3 because cake's frozen bodies are Blackwell
    only.
    """
    major, minor = compute_capability(device)
    if (major, minor) != HOPPER_CC:
        raise NotImplementedError(f"the sm90 KDA backward kernels target compute capability 9.0, got {major}.{minor}")
    return "sm_90a"


def _compile(source: Path, arch: str) -> Tuple[bytes, Dict[str, str]]:
    """Cubin plus the mangled name of every entry point.

    Both halves are cached together on disk: a cubin without its lowered names
    is useless here, since the templated kernels cannot be looked up by any
    name that appears in the source.
    """
    src = source.read_bytes()
    includes = compiler.cuda_include_dirs()
    options = [
        f"--gpu-architecture={arch}",
        *_BASE_OPTIONS,
        *_DEFINES,
        *(f"-I{path}" for path in includes),
    ]
    version = compiler.nvrtc_version()
    digest = hashlib.sha256(
        b"\0".join(
            [
                src,
                json.dumps(options).encode(),
                json.dumps(_NAME_EXPRESSIONS).encode(),
                repr(version).encode(),
                b"kda-bwd-cubin-v1",
            ]
        )
    ).hexdigest()[:24]
    base = compiler.cache_dir() / f"{source.stem}_{arch}_{digest}"
    cubin_path, names_path = base.with_suffix(".cubin"), base.with_suffix(".names.json")
    if cubin_path.is_file() and names_path.is_file():
        return cubin_path.read_bytes(), json.loads(names_path.read_text())

    program = compiler._ck(nvrtc.nvrtcCreateProgram(src, source.name.encode(), 0, [], []), "nvrtcCreateProgram")
    try:
        for expression in _NAME_EXPRESSIONS:
            compiler._ck(
                nvrtc.nvrtcAddNameExpression(program, expression.encode()),
                f"nvrtcAddNameExpression({expression})",
            )
        (err,) = nvrtc.nvrtcCompileProgram(program, len(options), [o.encode() for o in options])
        if int(err) != 0:
            raise compiler.CakeCompileError(f"NVRTC failed on {source.name} ({err}):\n{compiler._program_log(program)}")
        lowered = {}
        for expression in _NAME_EXPRESSIONS:
            name = compiler._ck(
                nvrtc.nvrtcGetLoweredName(program, expression.encode()),
                f"nvrtcGetLoweredName({expression})",
            )
            lowered[expression] = name.decode() if isinstance(name, bytes) else str(name)
        size = compiler._ck(nvrtc.nvrtcGetCUBINSize(program), "nvrtcGetCUBINSize")
        cubin = b"\0" * int(size)
        compiler._ck(nvrtc.nvrtcGetCUBIN(program, cubin), "nvrtcGetCUBIN")
    finally:
        nvrtc.nvrtcDestroyProgram(program)

    base.parent.mkdir(parents=True, exist_ok=True)
    for path, payload in ((cubin_path, cubin), (names_path, json.dumps(lowered).encode())):
        tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, path)
    return cubin, lowered


class BwdLibrary:
    """The backward module loaded into one device's primary context."""

    def __init__(self, device: int):
        from cudnn._device import _device_handle

        self.device = int(device)
        self.cubin, self.lowered = _compile(KERNEL_DIR / KERNEL_BODY, _arch_for_device(device))
        # Hold our own reference on the primary context: a module dies with its
        # context, and nothing else need have retained it on this device.
        self._device_handle = _device_handle(self.device)
        self._primary = compiler._ck(cu.cuDevicePrimaryCtxRetain(self._device_handle), "cuDevicePrimaryCtxRetain")
        self._functions: Dict[str, cu.CUfunction] = {}
        with device_context(self.device):
            self.module = compiler._ck(cu.cuModuleLoadData(self.cubin), f"cuModuleLoadData({KERNEL_BODY})")
            # Raise the dynamic shared-memory ceiling once per function. Every
            # one of these kernels asks for far more than the 48 KB default.
            for expression in _NAME_EXPRESSIONS:
                want = SMEM_BWD if expression.startswith("k_bwd") else SMEM_OPTIN
                self._set_smem(self._lookup(expression), expression, want)

    def __del__(self):
        try:
            cu.cuDevicePrimaryCtxRelease(self._device_handle)
        except Exception:  # noqa: BLE001 — teardown may have dropped the driver already
            pass

    def _lookup(self, name: str) -> cu.CUfunction:
        func = self._functions.get(name)
        if func is None:
            symbol = self.lowered.get(name, name)
            func = compiler._ck(
                cu.cuModuleGetFunction(self.module, symbol.encode()),
                f"cuModuleGetFunction({name} -> {symbol})",
            )
            self._functions[name] = func
        return func

    @staticmethod
    def _set_smem(func, name: str, nbytes: int) -> None:
        attr = cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        compiler._ck(
            cu.cuFuncSetAttribute(func, attr, int(nbytes)),
            f"cuFuncSetAttribute({name}, dynamic smem {nbytes})",
        )

    def function(self, name: str) -> cu.CUfunction:
        # Module handles are only valid with their context current; the calling
        # thread (an autograd worker, say) may have none bound.
        with device_context(self.device):
            return self._lookup(name)


def library(device: int) -> BwdLibrary:
    with _LOCK:
        lib = _LIBRARIES.get(int(device))
        if lib is None:
            lib = BwdLibrary(int(device))
            _LIBRARIES[int(device)] = lib
        return lib


class _Params(compiler.Params):
    """``compiler.Params`` plus the ``bool`` two of these kernels take."""

    def b(self, value: bool) -> "_Params":
        boxed = ctypes.c_bool(bool(value))
        self._keep.append(boxed)
        self._pointers.append(ctypes.addressof(boxed))
        return self


def _alignup(x: int) -> int:
    return (x + _ALIGN - 1) & ~(_ALIGN - 1)


class _Arena:
    """The artifact's bump allocator: each take() returns the current offset and
    then rounds the cursor up to 256 bytes."""

    def __init__(self, base: int = 0):
        self.base = int(base)
        self.off = 0

    def take(self, count: int, itemsize: int) -> int:
        at = self.base + self.off
        self.off = _alignup(self.off + count * itemsize)
        return at


_I32, _F32, _BF16 = 4, 4, 2


def _chunk_slots(total_tokens: int, n_seqs: int) -> int:
    raw = (total_tokens + BT - 1) // BT + n_seqs - 1
    return raw if raw > 0 else 1


def _layout(workspace: int, total_tokens: int, n_heads: int, n_seqs: int):
    """The workspace arena, in the artifact's exact order. Returns the sub-buffer
    addresses and the total size."""
    ncs = _chunk_slots(total_tokens, n_seqs)
    a = _Arena(workspace)
    cs_t0 = a.take(ncs, _I32)
    cs_len = a.take(ncs, _I32)
    seq_c0 = a.take(n_seqs + 1, _I32)
    seq_nc = a.take(n_seqs + 1, _I32)
    gn2 = a.take(ncs * n_heads * DH, _F32)
    amat = a.take(ncs * n_heads * 64 * 64, _BF16)
    wneg = a.take(total_tokens * n_heads * DH, _BF16)
    uu = a.take(total_tokens * n_heads * DH, _BF16)
    kg = a.take(total_tokens * n_heads * DH, _BF16)
    dvl = a.take(total_tokens * n_heads * DH, _BF16)
    vnew = a.take(total_tokens * n_heads * DH, _BF16)
    dv2 = a.take(total_tokens * n_heads * DH, _BF16)
    hst = a.take(ncs * n_heads * DH * DH, _BF16)
    dhst = a.take(ncs * n_heads * DH * DH, _BF16)
    pmat = a.take(ncs * n_heads * DH * DH, _BF16)
    return (
        dict(
            cs_t0=cs_t0,
            cs_len=cs_len,
            seq_c0=seq_c0,
            seq_nc=seq_nc,
            gn2=gn2,
            amat=amat,
            wneg=wneg,
            uu=uu,
            kg=kg,
            dvl=dvl,
            vnew=vnew,
            dv2=dv2,
            hst=hst,
            dhst=dhst,
            pmat=pmat,
        ),
        ncs,
        a.off,
    )


def workspace_bytes(total_tokens: int, n_heads: int, n_seqs: int) -> int:
    """Scratch the backward needs, mirroring ``kda_bwd_workspace_bytes``."""
    _, _, size = _layout(0, total_tokens, n_heads, n_seqs)
    return size


def launch(
    device: int,
    stream: int,
    workspace: int,
    dO: int,
    q: int,
    k: int,
    v: int,
    g: int,
    beta: int,
    cu_seqlens: int,
    initial_state: int,
    d_final_state: int,
    dq: int,
    dk: int,
    dv: int,
    dg: int,
    dbeta: int,
    d_initial_state: int,
    total_tokens: int,
    n_heads: int,
    n_seqs: int,
) -> None:
    """One backward pass: k_meta (varlen only), k_prep, k_scan_t, k_bwd.

    All tensor arguments are device addresses. Restated from the artifact's
    ``kda_bwd_launch``.
    """
    lib = library(device)
    ws, ncs, _ = _layout(workspace, total_tokens, n_heads, n_seqs)
    T, H, N = int(total_tokens), int(n_heads), int(n_seqs)

    single_sequence = N == 1
    # One packed sequence whose length is a multiple of the chunk size has no
    # partial chunk anywhere in the launch, so every tail predicate folds away
    # at compile time in all three kernels.
    full = single_sequence and (T % BT == 0)

    with device_context(int(device)):
        if not single_sequence:
            params = _Params()
            params.ptr(cu_seqlens).i32(N).i32(ncs)
            for name in ("cs_t0", "cs_len", "seq_c0", "seq_nc"):
                params.ptr(ws[name])
            compiler.launch(
                lib.function("k_meta"),
                grid=(1, 1, 1),
                block=(32, 1, 1),
                dynamic_smem=0,
                stream=stream,
                params=params,
                what=f"k_meta(N={N})",
            )

        prep = _Params()
        for address in (q, k, v, g, beta, dO, ws["cs_t0"], ws["cs_len"], ws["amat"], ws["wneg"], ws["uu"], ws["kg"], ws["pmat"], ws["gn2"], ws["dvl"]):
            prep.ptr(address)
        prep.i32(T).i32(H).b(single_sequence)
        compiler.launch(
            lib.function(f"k_prep<{'true' if full else 'false'}>"),
            grid=(ncs, H, 1),
            block=(512, 1, 1),
            dynamic_smem=SMEM_PREP,
            stream=stream,
            params=prep,
            what=f"k_prep(NCS={ncs}, H={H}, full={full})",
        )

        # Scan variant: the H==8 single-sequence case splits the value dim 16
        # ways at 128 threads; every other single-sequence case 32 ways at 256;
        # varlen always takes the untemplated-tail variant.
        if single_sequence and H == 8:
            scan_name = f"k_scan_t<16, 128, true, {'true' if full else 'false'}>"
            grid, block = (2 * (DH // 16), H, 1), (128, 1, 1)
        elif single_sequence:
            scan_name = f"k_scan_t<32, 256, true, {'true' if full else 'false'}>"
            grid, block = (2 * (DH // 32), H, 1), (256, 1, 1)
        else:
            scan_name = "k_scan_t<32, 256, false, false>"
            grid, block = (2 * (DH // 32), N * H, 1), (256, 1, 1)

        scan = _Params()
        for address in (
            ws["wneg"],
            ws["uu"],
            ws["kg"],
            ws["pmat"],
            ws["dvl"],
            dO,
            ws["gn2"],
            initial_state,
            d_final_state,
            ws["cs_t0"],
            ws["cs_len"],
            ws["seq_c0"],
            ws["seq_nc"],
            ws["hst"],
            ws["dhst"],
            ws["vnew"],
            ws["dv2"],
            d_initial_state,
        ):
            scan.ptr(address)
        scan.i32(T).i32(H)
        compiler.launch(
            lib.function(scan_name),
            grid=grid,
            block=block,
            dynamic_smem=SMEM_SCAN,
            stream=stream,
            params=scan,
            what=f"{scan_name} (T={T}, H={H}, N={N})",
        )

        bwd = _Params()
        for address in (q, k, v, g, beta, dO, ws["amat"], ws["vnew"], ws["dv2"], ws["hst"], ws["dhst"], ws["cs_t0"], ws["cs_len"], dq, dk, dv, dg, dbeta):
            bwd.ptr(address)
        bwd.i32(T).i32(H).b(single_sequence)
        compiler.launch(
            lib.function(f"k_bwd<{'true' if full else 'false'}>"),
            grid=(ncs, H, 1),
            block=(BNT, 1, 1),
            dynamic_smem=SMEM_BWD,
            stream=stream,
            params=bwd,
            what=f"k_bwd(NCS={ncs}, H={H}, full={full})",
        )
