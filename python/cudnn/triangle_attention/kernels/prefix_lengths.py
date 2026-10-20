# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The prefix-mask length derivation in ONE launch: a boolean key mask ``(B, N, 1, 1, S_kv)`` -> each row's True count as int32
``(B, N, 1, 1, 1)``, written into memory the caller owns.

The ``mask_is_prefix=True`` form of ``cudnn.triangle_attention`` serves a caller-ASSERTED prefix mask through the kernel's padding
path: the length of a row is its True count.  The first form of that derivation was two torch kernels -- the mask cast to int32 into a
staging chunk of the workspace (``copy_``) and a same-dtype row reduction (``sum``) -- which cost 7-11 us of device time and ~22 us of
host time per call on the 212-SM Rubin class, launch-bound at every training crop (the staging alone is 4x the mask's bytes).  This
kernel is the single launch that replaces both: no cast pass, no staging, no allocation, nothing read on the host, captured by a
CUDA graph like any other launch, so a replay re-derives the lengths from the mask's CURRENT contents.

**Shape.** One WARP per ``(b, n)`` row, ``WARPS_PER_CTA`` rows per CTA, a grid of ``ceil(B * N / WARPS_PER_CTA)``; the warp walks its
row 32 keys per step -- lane ``i`` loads the byte of key ``32 * step + i`` (a clamped, in-row address; a key past ``S_kv`` votes 0, as
``tile_dsl.mask.kv_mask_keep_words`` does), a ``vote.ballot`` makes the 32 votes one word and ``popc`` adds its bits to the warp's
count.  The count is warp-uniform; lane 0 stores it.  Every row shape is served: any ``S_kv`` (not a multiple of 8, 32 or 128 included;
``S_kv == 0`` -> 0 steps -> 0), any ``(B, N, key)`` strides in elements (a batch slice, a padded row, a strided key view), B * N rows
of any count.  The result is BITWISE ``mask.sum(-1, dtype=torch.int32)`` on every mask -- a non-prefix row counts its True keys, which
is the documented behaviour of the asserted form -- and a row with no True key reads 0.

**Barrier table: NONE** (no mbarrier, no named barrier, no SMEM, no TMA; the ballot's convergence is the warp's own -- the row test
``row < n_rows`` and the step count are warp-uniform, so every lane of a warp reaches every ``vote.ballot``).  **SMEM buffer table:
NONE.**  Nothing here can hang; a wrong grid or stride is a wrong count the bitwise pin catches, never a fault (every address is
clamped into its row).

**Host contract.** The kernel is shape-agnostic: one artifact per (device, arch), compiled on first use through the shared
``compile_cached`` (the persistent object cache; the key is this file's digest + the compile arguments) and called through its
positional tvm-ffi entry with plain integers (the two addresses, the extents, the strides, the raw stream) -- the host cost of a call is
the ffi crossing, no tensor objects are built.  A CuTe-DSL launchable is bound to the CUDA context it was loaded in, so the cache is
per DEVICE and the compile runs under that device; the launch is preceded by ``ensure_current_context`` (the calling thread may be an
autograd worker with an empty context stack).  The DSL floor and the device's target are gated HERE, before the first compile for a
device (:func:`requirement_error`: a typed ``NotImplementedError`` naming the version, never a late codegen error -- AGENTS.md Rule 7).

**Bytes.** ``B * N * S_kv`` bytes read, ``4 * B * N`` written: 590 KiB + 3 KiB at the S = 768 training crop -- a launch, not a
bandwidth problem (the two-op form read the mask, wrote and re-read 4x its bytes).
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver

from cudnn.frost.compiled_cache import compile_cached, positional_entry, template_key

FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]

#: Rows (warps) per CTA: 4 warps = 128 threads, the shape of the sibling byte-stream kernels; one row per warp.
WARPS_PER_CTA = 4
LANES = 32
THREADS_PER_CTA = WARPS_PER_CTA * LANES
#: The exported symbol of the artifact (one per device and arch).
SYMBOL = "cudnn_triangle_attention_prefix_lengths"

__all__ = [
    "LANES",
    "SYMBOL",
    "THREADS_PER_CTA",
    "WARPS_PER_CTA",
    "compile_prefix_lengths",
    "gpu_arch_of",
    "launch_prefix_lengths",
    "prefix_lengths_plan",
    "requirement_error",
]


@cute.kernel
def prefix_lengths_kernel(mask, out, n, s_kv, stride_b, stride_n, stride_s, n_rows, mask_extent):
    """One warp per row ``r = b * n + nn``: ``out[r] = popcount of (mask[b, nn, k] != 0 for k < s_kv)``.

    ``mask`` is an Int8 pointer (the boolean tensor's bytes), ``out`` an Int32 pointer of ``n_rows`` contiguous words; the strides
    are in ELEMENTS (= bytes for the mask).  ``mask_extent`` is the number of mask elements reachable from the base (the array
    view's extent; every address the kernel forms is inside it)."""
    tidx = cutlass.Int64(cute.arch.thread_idx()[0])
    bidx = cutlass.Int64(cute.arch.block_idx()[0])
    warp = tidx // cutlass.Int64(LANES)
    lane = tidx % cutlass.Int64(LANES)
    row = bidx * cutlass.Int64(WARPS_PER_CTA) + warp
    if row < n_rows:  # warp-uniform: the whole warp takes the row or none of it (the ballot below needs every lane)
        b = row // n
        nn = row - b * n
        base = b * stride_b + nn * stride_n
        src = cutlass.make_array_view(cute.make_tensor(mask, cute.make_layout((mask_extent,), stride=(1,))))
        last = s_kv - cutlass.Int64(1)  # the clamp target; with s_kv == 0 the loop below runs 0 steps and no address is formed
        count = cutlass.Int32(0)
        n_steps = (s_kv + cutlass.Int64(LANES - 1)) // cutlass.Int64(LANES)  # warp-uniform
        for step in cutlass.range(n_steps):
            col = cutlass.Int64(step) * cutlass.Int64(LANES) + lane
            in_row = col < s_kv
            col_c = cute.math.min(col, last)  # a lane past the row re-reads the last key and votes 0: the load never leaves the row
            byte = src[base + col_c * stride_s]
            keep = in_row & (byte != cutlass.Int8(0))
            word = cute.arch.vote_ballot_sync(keep)
            count = count + cute.arch.popc(word)
        if lane == cutlass.Int64(0):
            dst = cutlass.make_array_view(cute.make_tensor(out, cute.make_layout((n_rows,), stride=(1,))))
            dst[row] = count


prefix_lengths_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(mask, out, n, s_kv, stride_b, stride_n, stride_s, n_rows, mask_extent, stream: driver.CUstream):
    prefix_lengths_kernel(mask, out, n, s_kv, stride_b, stride_n, stride_s, n_rows, mask_extent).launch(
        grid=((n_rows + cutlass.Int64(WARPS_PER_CTA - 1)) // cutlass.Int64(WARPS_PER_CTA), 1, 1), block=(THREADS_PER_CTA, 1, 1), stream=stream
    )


def gpu_arch_of(capability) -> str:
    """The ``--gpu-arch`` value for a device: the arch-specific ``sm_XYa`` target from Hopper on (the part's own SASS; the ``f``-family
    targets reject some instructions the sibling kernels use, and one spelling per part keeps one cache entry per part), the plain
    ``sm_XY`` below it.  This kernel uses nothing arch-specific -- it compiles for every target the DSL names, the A100 included."""
    major, minor = int(capability[0]), int(capability[1])
    return f"sm_{major}{minor}a" if major >= 9 else f"sm_{major}{minor}"


@lru_cache(maxsize=64)
def compile_prefix_lengths(device_index: int, arch: str):
    """The compiled artifact for ``(device_index, arch)`` -- shape-agnostic (every extent and stride is a runtime Int64), the two
    operands as generic pointers (Int8 mask bytes at byte alignment, Int32 lengths at word alignment), loaded into the CALLING
    device's context (the caller enters ``torch.cuda.device(device_index)`` around the first call -- :func:`prefix_lengths_plan`)."""
    key = template_key(globals(), locals(), "compile_prefix_lengths")
    gmem = cute.AddressSpace.gmem
    return compile_cached(
        _host,
        cute.runtime.make_ptr(cutlass.Int8, 16, gmem, assumed_align=1),  # the TYPE of the parameter; the address is per call
        cute.runtime.make_ptr(cutlass.Int32, 16, gmem, assumed_align=4),
        cutlass.Int64(0),  # n            ) runtime; the zeros pin the TYPE only
        cutlass.Int64(0),  # s_kv         )
        cutlass.Int64(0),  # stride_b     )
        cutlass.Int64(0),  # stride_n     )
        cutlass.Int64(0),  # stride_s     )
        cutlass.Int64(0),  # n_rows       )
        cutlass.Int64(0),  # mask_extent  )
        driver.CUstream(0),
        options=f"--enable-tvm-ffi --gpu-arch {arch}",
        cache_key=key,
        symbol=SYMBOL,
    )


def requirement_error(device_index: int):
    """The typed reason this kernel cannot serve ``device_index`` -- a DSL below the FROST floor (the message names the installed version
    and the floor), or one without the device's target -- else None.  Checked before the first compile for a device (AGENTS.md Rule 7:
    a version problem reads as one, never as a late codegen error)."""
    from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error

    msg = cutedsl_requirement_error("prefix_mask_to_kv_lengths (the fused prefix-length derivation)")
    if msg:
        return msg
    import torch

    return cutedsl_arch_requirement_error(tuple(torch.cuda.get_device_capability(int(device_index))))


@lru_cache(maxsize=64)
def prefix_lengths_plan(device_index: int):
    """``(artifact, entry)`` for a CUDA device: the artifact compiled under that device (a launchable is bound to the context it was
    loaded in) and its positional tvm-ffi entry -- plain integers in, the launch out -- or ``None`` as the entry when the DSL build
    exposes none (the artifact is then called with pointer objects; same launch, more host work).  A DSL the kernel cannot use is a
    typed ``NotImplementedError`` (:func:`requirement_error`); an exception is not cached, so a later process state is re-checked."""
    import torch

    msg = requirement_error(int(device_index))
    if msg:
        raise NotImplementedError(msg)
    with torch.cuda.device(int(device_index)):
        artifact = compile_prefix_lengths(int(device_index), gpu_arch_of(torch.cuda.get_device_capability(int(device_index))))
    return artifact, positional_entry(artifact)


def launch_prefix_lengths(
    mask_ptr: int, out_ptr: int, b: int, n: int, s_kv: int, stride_b: int, stride_n: int, stride_s: int, stream: int, device_index: int
) -> None:
    """Issue the kernel on ``stream`` (a raw handle, int) for ``device_index``: ``b * n`` rows of ``s_kv`` keys at ``mask_ptr`` with the
    three ELEMENT strides (Python ints), the ``b * n`` int32 counts at ``out_ptr`` (contiguous, row-major over ``(b, n)``).  No host read,
    no allocation, no synchronisation; the launch is captured by a CUDA graph.  ``b * n == 0`` issues nothing.  The host path is the
    per-call cost of the form (measured per layer on an x86 host: the ffi entry 2.2 us, the context check 0.3 us), so it stays flat:
    no per-call import, no tensor object, one cached plan lookup."""
    global _ensure_current_context
    n_rows = b * n
    if n_rows == 0:
        return
    artifact, entry = prefix_lengths_plan(device_index)
    if _ensure_current_context is None:
        from cudnn._device import ensure_current_context

        _ensure_current_context = ensure_current_context
    _ensure_current_context(stream, device_index)
    # the mask elements reachable from the base: the last row's last key + 1 (every stride is non-negative in a torch view)
    extent = ((b - 1) * stride_b + (n - 1) * stride_n + (s_kv - 1) * stride_s + 1) if s_kv > 0 else 0
    if entry is not None:
        entry(mask_ptr, out_ptr, n, s_kv, stride_b, stride_n, stride_s, n_rows, extent, stream)
        return
    gmem = cute.AddressSpace.gmem
    artifact(
        cute.runtime.make_ptr(cutlass.Int8, int(mask_ptr), gmem, assumed_align=1),
        cute.runtime.make_ptr(cutlass.Int32, int(out_ptr), gmem, assumed_align=4),
        cutlass.Int64(int(n)),
        cutlass.Int64(int(s_kv)),
        cutlass.Int64(int(stride_b)),
        cutlass.Int64(int(stride_n)),
        cutlass.Int64(int(stride_s)),
        cutlass.Int64(n_rows),
        cutlass.Int64(extent),
        driver.CUstream(int(stream)),
    )


_ensure_current_context = None  # bound on the first launch (`cudnn._device.ensure_current_context`); a per-call import is ~0.5 us
