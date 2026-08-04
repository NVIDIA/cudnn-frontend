# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This kernel is derived from cuDNN, NVIDIA Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small cuTile glue kernels shared by the chunked GDN/KDA pipelines.

The pipeline hosts stitch their main kernels together with a handful of
element-wise / reduction steps (gradient accumulation, leading-axis sums,
dtype-converting copies, chunk-table building). These entries perform those
steps as plain cuTile launches over DLPack/CAI device buffers with an
explicit stream handle.

``add_inplace`` / ``cast_copy`` treat buffers as FLAT contiguous element
ranges (the callers own the shape bookkeeping) and take the element count
explicitly; ``sum_leading`` takes a 2-D ``[r, m]`` view. Tile-tail loads are
zero-padded and stores clip at the buffer extent.
"""

import cuda.tile as ct

ConstInt = ct.Constant[int]

_TILE = 2048


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def zero_fill(buf, *, stream) -> None:
    """Stream-ordered zero of a whole contiguous buffer (any DLPack/CAI)."""
    from cudnn.frost.buffers import DTYPE_ITEMSIZE, memset_zero_async, probe

    ptr, shape, _strides, dtype, _dev = probe(buf)
    n = DTYPE_ITEMSIZE[dtype]
    for s_ in shape:
        n *= int(s_)
    memset_zero_async(ptr, n, stream)


def reshaped(buf, shape):
    """A ``shape``-d view of a contiguous device buffer: native ``reshape``
    when the object has one (torch & co.), else a fresh DLPack view over the
    same pointer (the kernels derive their index rank from the array rank)."""
    if hasattr(buf, "reshape"):
        return buf.reshape(*shape)
    from cudnn.frost.buffers import DeviceView, probe

    ptr, _shape, _strides, dtype, dev = probe(buf)
    return DeviceView(ptr, shape, dtype, dev)


@ct.kernel
def _add_inplace_kernel(dst, src, TILE: ConstInt):
    pid = ct.bid(0)
    a = ct.load(dst, index=(pid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    b = ct.load(src, index=(pid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(dst, index=(pid,), tile=a + b)


def add_inplace(dst, src, numel: int, *, stream) -> None:
    """``dst += src`` over ``numel`` flat elements (same dtype, contiguous)."""
    ct.launch(stream, (_cdiv(numel, _TILE),), _add_inplace_kernel, (dst, src, _TILE))


@ct.kernel
def _cast_copy_kernel(dst, src, TILE: ConstInt):
    pid = ct.bid(0)
    t = ct.load(src, index=(pid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(dst, index=(pid,), tile=ct.astype(t, dst.dtype))


def cast_copy(dst, src, numel: int, *, stream) -> None:
    """``dst[:] = src`` over ``numel`` flat elements, converting to ``dst``'s
    dtype (a plain copy when the dtypes already match)."""
    ct.launch(stream, (_cdiv(numel, _TILE),), _cast_copy_kernel, (dst, src, _TILE))


@ct.kernel
def _sum_leading_kernel(dst, src, R: ConstInt, ACC: ConstInt, TILE: ConstInt):
    pid = ct.bid(0)
    acc = ct.astype(ct.load(src, index=(0, pid), shape=(1, TILE), padding_mode=ct.PaddingMode.ZERO), ct.float32)
    for r in range(1, R):
        acc = acc + ct.astype(ct.load(src, index=(r, pid), shape=(1, TILE), padding_mode=ct.PaddingMode.ZERO), ct.float32)
    acc = acc.reshape((TILE,))
    if ACC:
        acc = acc + ct.astype(ct.load(dst, index=(pid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO), ct.float32)
    ct.store(dst, index=(pid,), tile=ct.astype(acc, dst.dtype))


def sum_leading(dst, src, r: int, m: int, *, stream, accumulate: bool = False) -> None:
    """Reduce ``src`` (a 2-D ``[r, m]`` row-major buffer) over its leading
    axis into ``dst`` (flat ``[m]``), accumulating in fp32.  ``r`` is a
    compile-time constant (small fan-ins: split partials, head groups).
    ``accumulate`` adds the reduction onto ``dst`` instead of overwriting."""
    ct.launch(stream, (_cdiv(m, _TILE),), _sum_leading_kernel, (dst, src, r, int(accumulate), _TILE))


@ct.kernel
def _build_chunk_table_kernel(cu_seqlens, table, count, offsets, N: ConstInt, CS: ConstInt, BOUND: ConstInt):
    run = 0
    last = N - 1
    ct.store(offsets, (0,), 0)
    for n in range(N):
        s = ct.load(cu_seqlens, (n,), shape=()).item()
        e = ct.load(cu_seqlens, (n + 1,), shape=()).item()
        nc = (e - s + (CS - 1)) // CS
        for i in range(nc):
            ct.store(table, ((run + i) * 2,), n)
            ct.store(table, ((run + i) * 2 + 1,), i)
        if nc > 0:
            last = n
        run = run + nc
        ct.store(offsets, (n + 1,), run)
    # sentinel tail: (last_nonempty_seq, BOUND) decodes to a token range
    # starting at or past the packed end (BOUND * CS >= total), so a consumer
    # launched at the bound grid loads zero-padding and its stores clip — no
    # guard needed in the consuming kernels. The sentinel must reference a
    # NONEMPTY sequence: a zero-length one turns seq-derived divisors to
    # zero inside consumers (device trap).
    for j in range(run, BOUND):
        ct.store(table, (j * 2,), last)
        ct.store(table, (j * 2 + 1,), BOUND)
    ct.store(count, (0,), run)


def build_chunk_table(table, count, offsets, cu_seqlens, n_seqs: int, chunk_size: int, bound: int, *, stream) -> None:
    """Build the per-chunk ``(sequence, intra_chunk)`` index table ON DEVICE
    from ``cu_seqlens`` — no host round-trip, so the launch stays async and
    capture-safe.  ``table`` is a flat int32 buffer of ``2 * bound`` entries
    (``bound = cdiv(total, chunk_size) + n_seqs``, shape-derived); rows past
    the real chunk count are filled with an inert sentinel whose decoded
    token range lies at/past the packed end, so consumers may launch their
    chunk grids at ``bound`` unchanged.  ``count`` (one int32) receives the
    real chunk count; ``offsets`` (int32 ``[n_seqs + 1]``) receives the
    per-sequence chunk prefix (``prepare_chunk_offsets`` semantics)."""
    ct.launch(stream, (1,), _build_chunk_table_kernel, (cu_seqlens, reshaped(table, (2 * bound,)), count, offsets, n_seqs, chunk_size, bound))


@ct.kernel
def _head_group_sum_kernel(dst, src, G: ConstInt, BT: ConstInt, BK: ConstInt):
    t = ct.bid(0)
    h = ct.bid(1)
    k = ct.bid(2)
    acc = ct.astype(ct.load(src, index=(t, h * G, k), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO), ct.float32)
    for g in range(1, G):
        acc = acc + ct.astype(ct.load(src, index=(t, h * G + g, k), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO), ct.float32)
    ct.store(dst, index=(t, h, k), tile=ct.astype(acc, dst.dtype))


def head_group_sum(dst, src, t: int, h: int, g: int, k: int, *, stream) -> None:
    """Grouped-head reduction ``dst[t, h, :] = sum_g src[t, h*g + g', :]``:
    ``src`` is a 3-D ``[t, h*g, k]`` buffer, ``dst`` 3-D ``[t, h, k]``;
    fp32 accumulation, ``g`` consecutive heads per group (compile-time)."""
    BT, BK = 64, 128  # padded loads + clipped stores absorb ragged t/k
    ct.launch(stream, (_cdiv(t, BT), h, _cdiv(k, BK)), _head_group_sum_kernel, (dst, src, g, BT, BK))
