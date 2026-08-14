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

TILE = 2048


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def zero_fill(buf, *, stream) -> None:
    """Stream-ordered zero of a whole contiguous buffer (any DLPack/CAI)."""
    from cudnn.frost.buffers import DTYPE_ITEMSIZE, memset_zero_async, probe

    ptr, shape, _strides, dtype, _dev = probe(buf)
    n = DTYPE_ITEMSIZE[dtype]
    for s_ in shape:
        n *= int(s_)
    memset_zero_async(ptr, n, stream)


def reshaped(buf, target_shape):
    """A ``target_shape``-d DeviceView over the same pointer (contiguous by
    the engine gate's contract; the kernels derive their index rank from the
    array rank)."""
    from cudnn.frost.buffers import DeviceView, probe

    ptr, shape, _strides, dtype, dev = probe(buf)
    return DeviceView(ptr, shape, dtype, dev).reshape(tuple(target_shape))


def dummy(dtype_name: str, bufs):
    """Inert typed view over the workspace's 16-byte ``dummy`` carve, for
    ABSENT optional kernel args (always paired with a flag==0, never
    dereferenced). Dtype-bound so the compiled signature stays stable; the
    library allocates nothing."""
    from cudnn.frost.buffers import DTYPE_ITEMSIZE, DeviceView

    d = bufs["dummy"]
    return DeviceView(d.data_ptr(), (16 // DTYPE_ITEMSIZE[dtype_name],), dtype_name, d.__dlpack_device__()[1])


def opt(t, bufs, dtype_name: str = "float32"):
    """Resolve an optional tensor argument to a non-null cuTile launch arg:
    the buffer if present (contiguous by the engine contract), else an inert
    dummy (paired with a USE_*/HAS_* integer flag). cuTile never accepts None
    in launch args, so this is the required dummy-tensor-plus-flag pattern."""
    if t is None:
        return dummy(dtype_name, bufs)
    return t


def dev_id(buf) -> int:
    """Device ordinal of a DLPack/CAI buffer."""
    from cudnn.frost.buffers import probe

    return probe(buf)[4]


def ensure_cuda_context(stream=0) -> None:
    """Make the calling thread's CUDA driver context current.

    cuTile launches and the autotuner's driver-API timing fail on threads
    whose driver context stack is empty — e.g. autograd backward worker
    threads, where cudaSetDevice alone binds nothing. Prefer the launch
    stream's own context; else retain + set-current the current device's
    primary context (retained only when no context is bound, so at most once
    per thread). Best-effort: never fatal."""
    try:
        from cuda.bindings import driver as drv

        err, cur = drv.cuCtxGetCurrent()
        if err == drv.CUresult.CUDA_SUCCESS and int(cur) != 0:
            return
        if stream:
            err, sctx = drv.cuStreamGetCtx(stream)
            if err == drv.CUresult.CUDA_SUCCESS:
                drv.cuCtxSetCurrent(sctx)
                return
        from cuda.bindings import runtime as rt

        err_d, dev = rt.cudaGetDevice()
        if int(err_d) != 0:
            return
        err, pctx = drv.cuDevicePrimaryCtxRetain(dev)
        if err == drv.CUresult.CUDA_SUCCESS:
            drv.cuCtxSetCurrent(pctx)
    except Exception:  # noqa: BLE001
        pass


@ct.kernel
def add_inplace_kernel(dst, src, TILE: ConstInt):
    pid = ct.bid(0)
    a = ct.load(dst, index=(pid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    b = ct.load(src, index=(pid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(dst, index=(pid,), tile=a + b)


def add_inplace(dst, src, numel: int, *, stream) -> None:
    """``dst += src`` over ``numel`` flat elements (same dtype, contiguous)."""
    ct.launch(stream, (cdiv(numel, TILE),), add_inplace_kernel, (dst, src, TILE))


@ct.kernel
def cast_copy_kernel(dst, src, TILE: ConstInt):
    pid = ct.bid(0)
    t = ct.load(src, index=(pid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(dst, index=(pid,), tile=ct.astype(t, dst.dtype))


def cast_copy(dst, src, numel: int, *, stream) -> None:
    """``dst[:] = src`` over ``numel`` flat elements, converting to ``dst``'s
    dtype (a plain copy when the dtypes already match)."""
    ct.launch(stream, (cdiv(numel, TILE),), cast_copy_kernel, (dst, src, TILE))


@ct.kernel
def sum_leading_kernel(dst, src, R: ConstInt, ACC: ConstInt, TILE: ConstInt):
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
    ct.launch(stream, (cdiv(m, TILE),), sum_leading_kernel, (dst, src, r, int(accumulate), TILE))


@ct.kernel
def build_chunk_table_kernel(cu_seqlens, table, count, offsets, N: ConstInt, CS: ConstInt, BOUND: ConstInt):
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
    ct.launch(stream, (1,), build_chunk_table_kernel, (cu_seqlens, reshaped(table, (2 * bound,)), count, offsets, n_seqs, chunk_size, bound))


@ct.kernel
def head_group_sum_kernel(dst, src, G: ConstInt, BT: ConstInt, BK: ConstInt):
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
    ct.launch(stream, (cdiv(t, BT), h, cdiv(k, BK)), head_group_sum_kernel, (dst, src, g, BT, BK))
