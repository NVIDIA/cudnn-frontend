# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""XLA buffer adapters for the existing LayerNorm and RMSNorm launchers."""

from functools import lru_cache

import cutlass
import cutlass.cute as cute

from .bprop.kernels import layernorm_cga_sm100 as bwd_cga
from .bprop.kernels import layernorm_sm100 as bwd
from .config_sm100 import NormVariant, TemplateParams, make_cfg, make_warp_cfg, rowwise_spec
from .dtypes import DTYPE_BYTES, DTYPE_TO_CUTLASS
from .fprop.kernels import layernorm_cga_sm100 as fwd_cga
from .fprop.kernels import layernorm_sm100 as fwd
from .fprop.kernels import layernorm_warp_sm100 as fwd_warp
from .utils import smem_capacity


def _parameters(io_dtype, has_mean, has_beta):
    variant = NormVariant.LAYER_NORM if has_mean else NormVariant.RMS_NORM
    return TemplateParams(variant, io_dtype, has_beta=has_beta)


@lru_cache(maxsize=128)
def make_forward(rows, cols, io_dtype, *, has_mean, has_beta, epsilon, sm_count):
    """Return ``launch(stream, x, gamma, beta, y, mean, rstd)``.

    X and affine operands retain their original shapes in row-major layout;
    Y is [rows, cols], and statistics FP32 [rows]. Mean is None for RMSNorm.
    The unused beta operand may alias gamma.
    ``sm_count`` keys the cache because the shared helpers derive launch geometry
    from the device.
    """
    params = _parameters(io_dtype, has_mean, has_beta)
    spec = rowwise_spec(params.variant, (rows, cols))
    eb, et = DTYPE_BYTES[io_dtype], DTYPE_TO_CUTLASS[io_dtype]
    it = cutlass.Int16 if eb == 2 else cutlass.Int32
    wcfg = make_warp_cfg(params, cols)
    ctas = 0
    if fwd_cga.should_use(cols, rows, eb):
        cga, cs, vector, vpt = fwd_cga.cga_cfg(cols, rows, eb)
        route, host = "cga", fwd_cga._ln_fwd_cga_host
        constants = (cols, cs, cga, fwd_cga._BT, vector, vpt, fwd_cga._KC, it, et, float(cols), epsilon, has_mean, has_beta, rows * cga)
    elif wcfg is not None and fwd_warp.eligible(spec, params, wcfg, has_beta):
        tpr, wn, intra, ldgs, rpc, threads, vector = wcfg
        if fwd_warp._pipe_eligible(cols, wn, has_beta, eb):
            stages = fwd_warp._pipe_stages(cols)
            ctas = fwd_warp._pipe_cap(rows, cols)
            shared = fwd_warp._pipe_smem(cols, has_beta, wn, stages, eb)
            route, host = "warp", fwd_warp._warp_fwd_pipe_host
            constants = (cols, vector, tpr, wn, ldgs, (wn + 1) * 32, et, it, eb, stages, has_mean, has_beta, shared)
        else:
            ctas = fwd_warp._persist_cap((rows + rpc - 1) // rpc, threads, wn, ldgs)
            route, host = "warp", fwd_warp._warp_fwd_host
            constants = (cols, vector, tpr, wn, intra, ldgs, rpc, threads, et, it, intra.bit_length() - 1, intra == 32, has_mean, has_beta, fwd_warp._MBPM)
    else:
        cfg = make_cfg(params, cols)
        route, host = "row", fwd._ln_fwd_host
        constants = (cols, cfg.V, cfg.block_threads, eb, cfg.stage_mode, cfg.vec, has_mean, has_beta)

    @cute.jit
    def launch(stream, x, gamma, beta, y, mean, rstd):
        # The JAX bridge constrains row-major storage; change only CuTe metadata
        # here because eager JAX reshapes can copy the input buffers.
        x = cute.make_tensor(x.iterator, cute.make_layout((rows, cols), stride=(cols, 1)))
        gamma = cute.make_tensor(gamma.iterator, cute.make_layout(cols))
        beta = cute.make_tensor(beta.iterator, cute.make_layout(cols))
        if cutlass.const_expr(route == "cga"):
            flat_x = cute.make_tensor(x.iterator, cute.make_layout(rows * cols))
            flat_y = cute.make_tensor(y.iterator, cute.make_layout(rows * cols))
            host(flat_x, flat_y, gamma, beta, mean, rstd, *constants, stream)
        elif cutlass.const_expr(route == "warp"):
            host(x, y, gamma, beta, mean, rstd, cutlass.Int32(rows), cutlass.Int32(ctas), cutlass.Float32(epsilon), *constants, stream)
        else:
            host(x, y, gamma, beta, mean, rstd, cutlass.Int32(rows), cutlass.Float32(epsilon), *constants, stream)

    return launch


def _finalize_geometry(cols, partitions):
    threads = 128 if cols < 256 else 256
    grid = (cols + threads - 1) // threads
    chunks = min(partitions, max(1, min(32, 256 // grid)))
    return threads, grid, chunks, (partitions + chunks - 1) // chunks


@lru_cache(maxsize=128)
def make_backward(rows, cols, io_dtype, *, has_mean, has_beta, sm_count):
    """Return a launcher and the shape of each FP32 partial-gradient output, or None.

    ABI: ``launch(stream, dy, x, gamma, mean, rstd, dx, dgamma, dbeta,
    dgamma_partials, dbeta_partials)``.
    DY/X/gamma retain their original shapes in row-major layout. DX is
    [rows, cols], dgamma/dbeta [cols], and statistics FP32 [rows].
    Dgamma and optional dbeta MUST be initialized to zero. Scratch
    outputs are fully overwritten. Absent mean, dbeta, and partial buffers are None.
    """
    params = _parameters(io_dtype, has_mean, has_beta)
    eb, et = DTYPE_BYTES[io_dtype], DTYPE_TO_CUTLASS[io_dtype]
    it = cutlass.Int16 if eb == 2 else cutlass.Int32
    wcfg = bwd._bwd_pipe_cfg(cols, eb)
    partitions = 0
    if bwd_cga.should_use(cols, rows, eb):
        cga, cs, vector, vpt, stages = bwd_cga.cga_cfg(cols, rows, eb)
        partitions = min(rows, max(1, 4 * sm_count // cga))
        shared = 2 * stages * bwd_cga._BT * vector * eb + 8 + 2 * (bwd_cga._BT // 32) * 4 + 128
        route, host = "cga", bwd_cga._ln_bwd_cga_host
        constants = (
            cols,
            cs,
            cga,
            bwd_cga._BT,
            vector,
            vpt,
            stages,
            it,
            et,
            float(cols),
            has_mean,
            has_beta,
            partitions * cga,
            *_finalize_geometry(cols, partitions),
            shared,
        )
    elif wcfg is not None and wcfg[1] == 1 and wcfg[3] >= 2 and not has_mean and rows % 2 == 0 and not has_beta:
        tpr, _, ldgs, vector = wcfg
        stages = 2
        rows_per_tile = next((r for r in (8, 4, 2) if rows % r == 0 and 2 * stages * r * cols * eb <= smem_capacity()), 1)
        partitions = min(rows // rows_per_tile, sm_count * 12)
        shared = 2 * stages * rows_per_tile * cols * eb + cols * eb + (2 * stages + 1) * 8 + 64
        route, host = "tiled", bwd._ln_bwd_tiled_host
        constants = (cols, vector, tpr, ldgs, rows_per_tile, et, it, eb, stages, *_finalize_geometry(cols, partitions), shared)
    elif wcfg is not None and bwd._pipe_bwd_eligible(cols, wcfg[1], wcfg[2], wcfg[3], rows, has_beta, eb):
        tpr, wn, ldgs, vector = wcfg
        cache_xd = bwd._bwd_cache_xd(ldgs, vector, rows, cols, has_beta, wn, eb)
        stages = bwd._pipe_bwd_actual_stages(cache_xd)
        partitions = bwd._pipe_bwd_cap(rows, wn, cols, cache_xd, eb)
        shared = bwd._pipe_bwd_smem(cols, wn, stages, eb)
        route, host = "pipe", bwd._ln_bwd_pipe_host
        constants = (
            cols,
            vector,
            tpr,
            wn,
            ldgs,
            (wn + 1) * 32,
            et,
            it,
            eb,
            stages,
            cache_xd,
            has_mean,
            has_beta,
            *_finalize_geometry(cols, partitions),
            shared,
        )
    else:
        cfg = make_cfg(params, cols, staged_rows=2)
        route, host = "row", bwd._ln_bwd_host
        constants = (cols, cfg.V, cfg.block_threads, eb, cfg.stage_mode, cfg.vec, has_mean, has_beta)

    @cute.jit
    def launch(stream, dy, x, gamma, mean, rstd, dx, dgamma, dbeta, dgamma_partials, dbeta_partials):
        layout = cute.make_layout((rows, cols), stride=(cols, 1))
        dy = cute.make_tensor(dy.iterator, layout)
        x = cute.make_tensor(x.iterator, layout)
        gamma = cute.make_tensor(gamma.iterator, cute.make_layout(cols))
        if cutlass.const_expr(route == "row"):
            host(dy, x, gamma, mean, rstd, dx, dgamma, dbeta, cutlass.Int32(rows), *constants, stream)
        elif cutlass.const_expr(route == "cga"):
            flat_dy = cute.make_tensor(dy.iterator, cute.make_layout(rows * cols))
            flat_x = cute.make_tensor(x.iterator, cute.make_layout(rows * cols))
            flat_dx = cute.make_tensor(dx.iterator, cute.make_layout(rows * cols))
            host(
                flat_dy,
                flat_x,
                flat_dx,
                gamma,
                mean,
                rstd,
                dgamma_partials,
                dbeta_partials,
                dgamma,
                dbeta,
                cutlass.Int32(rows),
                cutlass.Int32(partitions),
                *constants,
                stream,
            )
        elif cutlass.const_expr(route == "tiled"):
            host(dy, x, dx, gamma, rstd, dgamma_partials, dgamma, cutlass.Int32(rows), cutlass.Int32(partitions), *constants, stream)
        else:
            host(
                dy,
                x,
                dx,
                gamma,
                mean,
                rstd,
                dgamma_partials,
                dbeta_partials,
                dgamma,
                dbeta,
                cutlass.Int32(rows),
                cutlass.Int32(partitions),
                *constants,
                stream,
            )

    partial_shape = (partitions * cols,) if partitions else None
    return launch, partial_shape
