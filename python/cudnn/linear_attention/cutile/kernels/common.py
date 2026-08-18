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

"""Helpers and small kernels shared by the chunked GDN/KDA cuTile pipelines.

Every buffer reaching this module comes from an engine (a variant-pack operand
or a workspace carve), so it is contiguous and needs no probing.
"""

from types import SimpleNamespace

import cuda.tile as ct
from cuda.tile.tune import exhaustive_search

from cudnn.frost.buffers import DTYPE_ITEMSIZE, DeviceView, current_device_id, dtype_name as dtname, memset_zero_async

ConstInt = ct.Constant[int]

TILE = 2048


# --- Host helpers ---------------------------------------------------------------------------------


def next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def zero_fill(buf, *, stream) -> None:
    """Stream-ordered zero of a whole contiguous buffer."""
    memset_zero_async(buf.data_ptr(), int(buf.nbytes), stream)


def dummy(dtype_name: str, bufs):
    """Inert typed view over the workspace's 16-byte ``dummy`` carve, for an
    absent optional kernel arg. Always paired with a flag==0, never read."""
    d = bufs["dummy"]
    return DeviceView(d.data_ptr(), (16 // DTYPE_ITEMSIZE[dtype_name],), dtype_name, d.__dlpack_device__()[1])


def opt(t, bufs, dtype_name: str = "float32"):
    """``t`` if present, else an inert dummy: cuTile takes no None launch arg."""
    if t is None:
        return dummy(dtype_name, bufs)
    return t


# --- Launch tuning --------------------------------------------------------------------------------


# Only the @ct.kernel launch hints (occupancy x num_worker_warps) are explored;
# the grid, args and algorithm are unchanged, so tuning never moves numerics.
launch_hint_cache: dict = {}


def launch_hint_configs(occ_choices, nww_choices=(4, 8)):
    """occupancy x num_worker_warps grid; deduped, default (occ=1,nww=4) first."""
    seen = set()
    cfgs = []
    for occ in occ_choices:
        for nww in nww_choices:
            key = (occ, nww)
            if key in seen:
                continue
            seen.add(key)
            cfgs.append(SimpleNamespace(occupancy=occ, num_worker_warps=nww))
    return cfgs


def autotuned_launch(kernel, cache_key, grid, args, occ_choices=(1, 2, 3, 4), nww_choices=(4, 8), timeout=30, stream=None):
    """Launch ``kernel`` with the best launch hints for ``cache_key``.

    Tune-once/cache/launch over launch hints only (grid, args and signature are
    fixed). Falls back to the base kernel when tuning fails or times out.
    Default config (occ=1, nww=4) is explored first so a no-improvement shape
    keeps the base behaviour. ``cache_key`` is qualified
    by the kernel's own name, so two kernels sharing a key shape stay apart.
    """
    stream = 0 if stream is None else stream
    cache_key = (getattr(getattr(kernel, "_pyfunc", None), "__name__", repr(kernel)), cache_key)
    if cache_key not in launch_hint_cache:
        tuned = None
        try:
            configs = launch_hint_configs(occ_choices, nww_choices)
            with ct.compiler_timeout(timeout):
                result = exhaustive_search(
                    configs,
                    stream,
                    lambda cfg: grid,
                    kernel,
                    lambda cfg: args,
                    lambda cfg: {"occupancy": cfg.occupancy, "num_worker_warps": cfg.num_worker_warps},
                )
            best = result.best.config
            tuned = kernel.replace_hints(occupancy=best.occupancy, num_worker_warps=best.num_worker_warps)
        except Exception:
            tuned = None
        launch_hint_cache[cache_key] = tuned

    tuned = launch_hint_cache[cache_key]
    if tuned is None:
        ct.launch(stream, grid, kernel, args)
    else:
        ct.launch(stream, grid, tuned, args)


# --- Device helpers -------------------------------------------------------------------------------


def exp(x):
    return ct.exp(ct.astype(x, ct.float32))


def exp2(x):
    return ct.exp2(ct.astype(x, ct.float32))


def softplus(x):
    # softplus: where(x <= 20, log1p(exp(x)), x)
    return ct.where(x <= 20.0, ct.log(1.0 + ct.exp(x)), x)


def tf32(a):
    """ct.mma/ct.matmul do not auto-cast fp32 operands to tf32; cast
    explicitly (allow-tf32 matmul semantics)."""
    return ct.astype(a, ct.tfloat32) if a.dtype == ct.float32 else a


def ct_min(a, b):
    # scalar/tile min for runtime ints (builtin `min` is whitelisted; `hasattr` is not).
    return min(a, b)


# --- Kernels --------------------------------------------------------------------------------------


@ct.kernel
def l2norm_fwd_kernel1(x, y, rstd, eps, D, BD: ConstInt):
    # D > 512 path: one row per program, row length D.
    i_t = ct.bid(0)
    cols = ct.arange(BD, dtype=ct.int32)
    mask = cols < D

    b_x = ct.astype(ct.gather(x, (i_t, cols), mask=mask, check_bounds=False, padding_value=0.0), ct.float32)
    b_rstd = ct.rsqrt(ct.sum(b_x * b_x) + eps)
    b_y = b_x * b_rstd
    ct.scatter(y, (i_t, cols), ct.astype(b_y, y.dtype), mask=mask, check_bounds=False)
    ct.scatter(rstd, (i_t,), ct.astype(b_rstd, rstd.dtype))


@ct.kernel
def l2norm_fwd_kernel(x, y, rstd, eps, T, D: ConstInt, BD: ConstInt, BT: ConstInt):
    # D <= 512 path: BT rows per block, BD power-of-2 cols. Block-aligned ->
    # ct.load with block index + ZERO padding.
    i_t = ct.bid(0)
    b_x = ct.astype(ct.load(x, index=(i_t, 0), shape=(BT, BD), padding_mode=ct.PaddingMode.ZERO), ct.float32)
    b_rstd = ct.rsqrt(ct.sum(b_x * b_x, axis=1) + eps)
    b_y = b_x * b_rstd[:, None]
    ct.store(y, index=(i_t, 0), tile=ct.astype(b_y, y.dtype))
    ct.store(rstd, index=(i_t,), tile=ct.astype(b_rstd, rstd.dtype))


@ct.kernel
def l2norm_bwd_kernel1(y, rstd, dy, dy2, dx, eps, D, BD: ConstInt, HAS_DY2: ConstInt):
    i_t = ct.bid(0)
    cols = ct.arange(BD, dtype=ct.int32)
    mask = cols < D

    b_y = ct.astype(ct.gather(y, (i_t, cols), mask=mask, check_bounds=False, padding_value=0.0), ct.float32)
    b_dy = ct.astype(ct.gather(dy, (i_t, cols), mask=mask, check_bounds=False, padding_value=0.0), ct.float32)
    if HAS_DY2:
        b_dy2 = ct.astype(ct.gather(dy2, (i_t, cols), mask=mask, check_bounds=False, padding_value=0.0), ct.float32)
        # Preserve bf16 `dk.add_(dk2)` rounding before the fp32 normalization math.
        b_dy = ct.astype(ct.astype(b_dy + b_dy2, dy.dtype), ct.float32)
    b_rstd = ct.astype(ct.gather(rstd, (i_t,), check_bounds=False, padding_value=0.0), ct.float32).item()

    b_dx = b_dy * b_rstd - ct.sum(b_dy * b_y) * b_y * b_rstd
    ct.scatter(dx, (i_t, cols), ct.astype(b_dx, dx.dtype), mask=mask, check_bounds=False)


@ct.kernel
def l2norm_bwd_kernel(y, rstd, dy, dy2, dx, eps, T, D: ConstInt, BD: ConstInt, BT: ConstInt, HAS_DY2: ConstInt):
    i_t = ct.bid(0)
    b_y = ct.astype(ct.load(y, index=(i_t, 0), shape=(BT, BD), padding_mode=ct.PaddingMode.ZERO), ct.float32)
    b_rstd = ct.astype(ct.load(rstd, index=(i_t,), shape=(BT,), padding_mode=ct.PaddingMode.ZERO), ct.float32)
    b_dy = ct.astype(ct.load(dy, index=(i_t, 0), shape=(BT, BD), padding_mode=ct.PaddingMode.ZERO), ct.float32)
    if HAS_DY2:
        b_dy2 = ct.astype(ct.load(dy2, index=(i_t, 0), shape=(BT, BD), padding_mode=ct.PaddingMode.ZERO), ct.float32)
        b_dy = ct.astype(ct.astype(b_dy + b_dy2, dy.dtype), ct.float32)
    b_dot = ct.sum(b_dy * b_y, axis=1)
    b_dx = b_dy * b_rstd[:, None] - b_dot[:, None] * b_y * b_rstd[:, None]
    ct.store(dx, index=(i_t, 0), tile=ct.astype(b_dx, dx.dtype))


@ct.kernel
def fused_beta_sigmoid_fwd_kernel(x, y, scale, n_elements, BLOCK_SIZE: ConstInt):
    pid = ct.bid(0)
    offs = pid * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)
    mask = offs < n_elements
    b_x = ct.astype(ct.gather(x, offs, mask=mask, check_bounds=False, padding_value=0.0), ct.float32)
    b_y = scale * (1.0 / (1.0 + ct.exp(-b_x)))
    ct.scatter(y, offs, ct.astype(b_y, y.dtype), mask=mask, check_bounds=False)


@ct.kernel
def fused_beta_sigmoid_bwd_kernel(x, dy, dx, scale, n_elements, BLOCK_SIZE: ConstInt):
    pid = ct.bid(0)
    offs = pid * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)
    mask = offs < n_elements
    b_x = ct.astype(ct.gather(x, offs, mask=mask, check_bounds=False, padding_value=0.0), ct.float32)
    b_dy = ct.astype(ct.gather(dy, offs, mask=mask, check_bounds=False, padding_value=0.0), ct.float32)
    b_y = 1.0 / (1.0 + ct.exp(-b_x))
    b_dx = b_dy * scale * b_y * (1.0 - b_y)
    ct.scatter(dx, offs, ct.astype(b_dx, dx.dtype), mask=mask, check_bounds=False)


@ct.kernel
def add_inplace_kernel(dst, src, TILE: ConstInt):
    pid = ct.bid(0)
    a = ct.load(dst, index=(pid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    b = ct.load(src, index=(pid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(dst, index=(pid,), tile=a + b)


@ct.kernel
def cast_copy_kernel(dst, src, TILE: ConstInt):
    pid = ct.bid(0)
    t = ct.load(src, index=(pid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(dst, index=(pid,), tile=ct.astype(t, dst.dtype))


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
    # Sentinel tail: (last_nonempty_seq, BOUND) decodes to a token range at or
    # past the packed end, so a consumer gridded at BOUND loads zero-padding
    # and its stores clip. It must name a NONEMPTY sequence -- a zero-length
    # one turns seq-derived divisors to zero inside consumers (device trap).
    for j in range(run, BOUND):
        ct.store(table, (j * 2,), last)
        ct.store(table, (j * 2 + 1,), BOUND)
    ct.store(count, (0,), run)


@ct.kernel
def head_group_sum_kernel(dst, src, G: ConstInt, BT: ConstInt, BK: ConstInt):
    t = ct.bid(0)
    h = ct.bid(1)
    k = ct.bid(2)
    acc = ct.astype(ct.load(src, index=(t, h * G, k), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO), ct.float32)
    for g in range(1, G):
        acc = acc + ct.astype(ct.load(src, index=(t, h * G + g, k), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO), ct.float32)
    ct.store(dst, index=(t, h, k), tile=ct.astype(acc, dst.dtype))


# --- Launchers ------------------------------------------------------------------------------------


BETA_SIGMOID_BLOCK_SIZE = 2048

# l2norm is a memory-bound row reduction (each row loaded once, reduced,
# written once), so higher occupancy hides DRAM latency; do not force occ=1.
L2NORM_TUNE_OCC = (1, 2, 4, 8)


def l2norm_fwd(x, eps: float = 1e-6, out=None, rstd_out=None, stream=None):
    stream = 0 if stream is None else stream
    x_shape_og = x.shape
    x = x.reshape((-1, x.shape[-1]))
    y = out.reshape(tuple(x.shape))
    T, D = x.shape[0], x.shape[-1]
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, next_power_of_2(D))
    rstd = rstd_out.reshape((T,))
    if D <= 512:
        BT = 32
        grid = (cdiv(T, BT),)
        autotuned_launch(
            l2norm_fwd_kernel,
            ("l2norm_fwd_kernel", D, BD, BT, str(x.dtype), current_device_id()),
            grid,
            (x, y, rstd, float(eps), T, D, BD, BT),
            occ_choices=L2NORM_TUNE_OCC,
            nww_choices=(4,),
            stream=stream,
        )
    else:
        ct.launch(stream, (T,), l2norm_fwd_kernel1, (x, y, rstd, float(eps), D, BD))
    return y.view(x_shape_og), rstd.view(x_shape_og[:-1])


def l2norm_bwd(
    y,
    rstd,
    dy,
    eps: float = 1e-6,
    dy2=None,
    out=None,
    bufs=None,
    stream=None,
):
    stream = 0 if stream is None else stream
    y_shape_og = y.shape
    y = y.reshape(-1, dy.shape[-1])
    dy = dy.reshape(-1, dy.shape[-1])
    dy2_arg = dy2.reshape(-1, dy.shape[-1]) if dy2 is not None else dummy(dtname(dy), bufs)
    dx = out.reshape(tuple(y.shape))
    T, D = y.shape[0], y.shape[-1]
    MAX_FUSED_SIZE = 65536 // y.element_size()
    BD = min(MAX_FUSED_SIZE, next_power_of_2(D))
    rstd_flat = rstd.reshape(-1)
    if D <= 512:
        BT = 32
        grid = (cdiv(T, BT),)
        autotuned_launch(
            l2norm_bwd_kernel,
            ("l2norm_bwd_kernel", D, BD, BT, str(y.dtype), int(dy2 is not None), current_device_id()),
            grid,
            (y, rstd_flat, dy, dy2_arg, dx, float(eps), T, D, BD, BT, int(dy2 is not None)),
            occ_choices=L2NORM_TUNE_OCC,
            nww_choices=(4,),
            stream=stream,
        )
    else:
        ct.launch(
            stream,
            (T,),
            l2norm_bwd_kernel1,
            (y, rstd_flat, dy, dy2_arg, dx, float(eps), D, BD, int(dy2 is not None)),
        )
    return dx.view(y_shape_og)


def fused_beta_sigmoid_fwd(x, scale: float = 1.0, out=None, stream=None):
    stream = 0 if stream is None else stream
    y = out.reshape(tuple(x.shape))
    n = x.numel()
    grid = (cdiv(n, BETA_SIGMOID_BLOCK_SIZE),)
    ct.launch(
        stream,
        grid,
        fused_beta_sigmoid_fwd_kernel,
        (x.reshape((-1,)), y.reshape(-1), float(scale), n, BETA_SIGMOID_BLOCK_SIZE),
    )
    return y


def fused_beta_sigmoid_bwd(x, dy, scale: float = 1.0, out=None, stream=None):
    stream = 0 if stream is None else stream
    dx = out.reshape(tuple(x.shape))
    n = x.numel()
    grid = (cdiv(n, BETA_SIGMOID_BLOCK_SIZE),)
    ct.launch(
        stream,
        grid,
        fused_beta_sigmoid_bwd_kernel,
        (x.reshape((-1,)), dy.reshape((-1,)), dx.reshape(-1), float(scale), n, BETA_SIGMOID_BLOCK_SIZE),
    )
    return dx


def fused_beta_sigmoid(x, scale: float = 1.0, out=None, stream=None):
    """Fused ``scale * sigmoid(x)`` (fp32, written to ``out``)."""
    stream = 0 if stream is None else stream
    return fused_beta_sigmoid_fwd(x, scale, out=out, stream=stream)


def add_inplace(dst, src, numel: int, *, stream) -> None:
    """``dst += src`` over ``numel`` flat elements (same dtype, contiguous)."""
    ct.launch(stream, (cdiv(numel, TILE),), add_inplace_kernel, (dst, src, TILE))


def cast_copy(dst, src, numel: int, *, stream) -> None:
    """``dst[:] = src`` over ``numel`` flat elements, converting to ``dst``'s dtype."""
    ct.launch(stream, (cdiv(numel, TILE),), cast_copy_kernel, (dst, src, TILE))


def sum_leading(dst, src, r: int, m: int, *, stream, accumulate: bool = False) -> None:
    """Reduce ``src`` ``[r, m]`` over its leading axis into ``dst`` ``[m]`` in
    fp32. ``r`` is compile-time (split partials, head groups); ``accumulate``
    adds onto ``dst`` instead of overwriting."""
    ct.launch(stream, (cdiv(m, TILE),), sum_leading_kernel, (dst, src, r, int(accumulate), TILE))


def build_chunk_table(table, count, offsets, cu_seqlens, n_seqs: int, chunk_size: int, bound: int, *, stream) -> None:
    """Build the per-chunk ``(sequence, intra_chunk)`` index table ON DEVICE, so
    the launch stays async and capture-safe. ``table`` holds ``2 * bound``
    int32s; rows past the real chunk count get the inert sentinel above, so
    consumers may grid at ``bound``. ``count`` receives the real chunk count,
    ``offsets`` the per-sequence chunk prefix."""
    ct.launch(stream, (1,), build_chunk_table_kernel, (cu_seqlens, table.reshape((2 * bound,)), count, offsets, n_seqs, chunk_size, bound))


def head_group_sum(dst, src, t: int, h: int, g: int, k: int, *, stream) -> None:
    """``dst[t, h, :] = sum_g src[t, h*g + g', :]`` in fp32, ``g`` consecutive
    heads per group (compile-time)."""
    BT, BK = 64, 128  # padded loads + clipped stores absorb ragged t/k
    ct.launch(stream, (cdiv(t, BT), h, cdiv(k, BK)), head_group_sum_kernel, (dst, src, g, BT, BK))
