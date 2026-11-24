# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The zero fill of a COMBINE output as a programmatic launch ahead of a general grouped GEMM.

A COMBINE kernel only adds into its output, so every call zeroes the output before the kernel runs. A driver memset
did that, and a memset is no programmatic predecessor: it starts only once the kernel ahead of it has finished, and
the grouped GEMM behind it only once the memset has. This kernel takes the memset's place for the general grouped
templates (``_initialize_combine_output``). Each thread block

- releases its dependents first (``griddepcontrol.launch_dependents``), so the grouped GEMM's thread blocks can be
  resident and through their prologue while the fill waits. Every role of a general grouped template waits for its
  programmatic predecessor (``griddepcontrol.wait``) before it touches global memory, so none of them reads the routing
  or adds into the output before this fill has completed;
- waits for its own predecessors (``griddepcontrol.wait``) before its first store, since the stream work ahead of it
  may still read or write the output;
- stores zeros over each row of the region, 16 bytes per store.

``CombineZero.takes`` says which regions the kernel writes exactly: a non-empty region of whole 16-byte vectors whose
start and row pitch are 16-byte aligned. The caller zeroes any other region with the memset.

Measured on a 68-SM SM100 part (the fused MoE layer: FC1 GATHER + SwiGLU + FP8 block-128 out, then the fill, then FC2
COMBINE). When the fill was added, against the memset in the same process (5 interleaved rounds, eager and replayed from
a CUDA graph), it saved 1.6-3.5 us per layer at 1, 8, 64 and 512 tokens on qwen35_35b, qwen3_30b, qwen3_next_80b and
qwen35_122b (3.7-6.5% of the layer at 1 token, 0.5-1.7% at 8), and 1.6-2.5 us on dsv3 at 1 and 8 tokens (0.8-1.0% at 1
token); dsv3 at 64 and 512 tokens was within its run-to-run spread. Leaving the output unzeroed (wrong; measured as a
bound) saved 2.4-4.0 us at 1 token: the difference is the one more grid completion the GEMM waits for. The same kernel
without the early release saved 0.7-1.2 us at 1 to 64 tokens on the four smaller models. Re-measured after the later
changes to the general templates, against a source tree that zeroes with the memset (7 interleaved rounds, the engine's
default plans): eager, 0.9-2.4 us per layer at 1 and 8 tokens and 0.3-2.3 us at 64 and 512 on the four smaller models
(2.4-4.7% of the layer at 1 token) and 1.2-1.4 us on dsv3 at 1, 8 and 64 tokens; replayed from a CUDA graph, 1.0-2.5 us
at 1-512 tokens on qwen35_35b and qwen3_30b (5.7-5.9% of the layer at 1 token).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
from cuda.bindings import driver as _cuda
from cutlass.cute.runtime import make_fake_stream

from cudnn.frost.compiled_cache import compile_cached
from cudnn.frost.tile_dsl.tma import st_global_v4

THREADS = 128
VEC_BYTES = 16
# Thread blocks per SM the grid is capped at: a 512-token output of 7168 fp32 columns (14.7 MB) is 14 stores per thread
# on 68 SMs. At 512 tokens, replayed from a CUDA graph, 4 per SM saved 0.4-1.1 us more per layer than 2 per SM on the
# four smaller models above, 1 per SM about 1 us less than 2, and 8 per SM measured like 4 in the layer. Behind another
# COMBINE (FC2 after FC2), 8 per SM took 0.5-1.6% less of the FC2 than 4 per SM on qwen35_35b (512 and 4096 tokens),
# qwen3_30b (2048 and 4096) and qwen3_next_80b (512), and within 0.25% in the other cells at 64-4096 tokens (dsv3, glm5
# and the rest), except dsv3 at 512 tokens, where two separately compiled copies of the 4-per-SM tree differed by 0.6%;
# the layers measured within -0.3% to +0.2% of 4 per SM. An output of at most one store per thread of the capped grid
# (every output at 8 tokens on those models) launches fewer thread blocks; at 1 token the grid shape made no measurable
# difference.
CTAS_PER_SM = 8
_SYMBOL = "frost_moe_combine_zero"
# The kernel's whole source is this file: its digest names the compiled object (cudnn.frost.compiled_cache).
_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


@cute.kernel
def frost_moe_combine_zero(base: cutlass.Int64, pitch: cutlass.Int64, row_vecs: cutlass.Int64, rows: cutlass.Int64):
    # The grouped GEMM behind this launch may start now: each of its roles waits for this grid before it touches global
    # memory.
    nvvm.griddepcontrol("launch_dependents")
    # The output's earlier readers and writers in the stream are this grid's predecessors.
    nvvm.griddepcontrol("wait")
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    gdx, gdy, _ = cute.arch.grid_dim()
    zero = cutlass.Int32(0)
    col0 = cutlass.Int64(bidx) * THREADS + cutlass.Int64(tidx)
    col_step = cutlass.Int64(gdx) * THREADS
    row = cutlass.Int64(bidy)
    while row < rows:
        row_addr = base + row * pitch
        col = col0
        while col < row_vecs:
            st_global_v4(row_addr + col * VEC_BYTES, (zero, zero, zero, zero), cutlass.Int32)
            col += col_step
        row += cutlass.Int64(gdy)


frost_moe_combine_zero.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _launch(
    base: cutlass.Int64,
    pitch: cutlass.Int64,
    row_vecs: cutlass.Int64,
    rows: cutlass.Int64,
    grid_x: cutlass.Int32,
    grid_y: cutlass.Int32,
    stream: _cuda.CUstream,
):
    frost_moe_combine_zero(base, pitch, row_vecs, rows).launch(grid=(grid_x, grid_y, 1), block=(THREADS, 1, 1), use_pdl=True, stream=stream)


class CombineZero:
    """The compiled fill, with the SM count of the device it sizes its grid for."""

    __slots__ = ("_fn", "_max_ctas")

    def __init__(self, fn, sm_count: int):
        self._fn = fn
        self._max_ctas = CTAS_PER_SM * max(1, int(sm_count))

    @staticmethod
    def takes(base: int, pitch: int, width: int, rows: int) -> bool:
        """Whether the kernel writes exactly the region of ``rows`` rows of ``width`` bytes, ``pitch`` bytes apart, from
        the device address ``base``: a non-empty region of whole 16-byte vectors, 16-byte aligned at every row."""
        return rows > 0 and width > 0 and width % VEC_BYTES == 0 and base % VEC_BYTES == 0 and (rows == 1 or pitch % VEC_BYTES == 0)

    def __call__(self, base: int, pitch: int, width: int, rows: int, stream) -> None:
        """Enqueue the fill of a region :meth:`takes` accepts on ``stream`` (a ``CUstream``)."""
        row_vecs = width // VEC_BYTES
        grid_x = min(-(-row_vecs // THREADS), self._max_ctas)
        grid_y = min(rows, max(1, self._max_ctas // grid_x))
        self._fn(base, pitch, row_vecs, rows, grid_x, grid_y, stream)


def build(options: str, sm_count: int) -> "CombineZero | None":
    """The fill compiled with the plan's compile ``options`` (``--enable-tvm-ffi`` and its target), or None when the
    options lack ``--enable-tvm-ffi``: the call passes the addresses as plain integers, which the tvm-ffi entry takes."""
    if "--enable-tvm-ffi" not in options:
        return None
    fn = compile_cached(
        _launch,
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int32(1),
        cutlass.Int32(1),
        make_fake_stream(use_tvm_ffi_env_stream=False),
        options=options,
        cache_key=_SOURCE_DIGEST,
        symbol=_SYMBOL,
    )
    return CombineZero(fn, sm_count)
