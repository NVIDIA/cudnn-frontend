# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX-native (XLA custom call) entry point for the fused projection GEMM + RoPE +
dual-direction MXFP8 quantize, built on :func:`cudnn.jax.call`."""

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import cutlass
import cutlass.cute as cute
import cutlass.utils

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.jax import call, row_major_desc as _make_desc
from .api import GemmProjRopeMxfp8Bf16InSm100, GemmProjRopeMxfp8Mxfp8InSm100
from .gemm_proj_rope_mxfp8_bf16in import HEAD_DIM, BLOCK, TILE_M, gemm_proj_rope_mxfp8_host as _bf16in_host
from .gemm_proj_rope_mxfp8_mxfp8in import gemm_proj_rope_mxfp8_host as _mxfp8in_host

_bf16in_grid_cache: dict = {}
_mxfp8in_grid_cache: dict = {}


@cute.jit
def _proj_rope_bf16in_adapter(stream, x, w, cos, sin, qrow, srow, qcol, scol, *, grid_m, num_heads, mac, swizzle_size):
    _bf16in_host(x, w, cos, sin, qrow, srow, qcol, scol, grid_m, num_heads, mac, swizzle_size, stream)


@cute.jit
def _proj_rope_mxfp8in_adapter(
    stream, x, x_scale, w, w_scale, cos, sin, qrow, srow, qcol, scol, *, grid_m, num_heads, mac, swizzle_size, t2r_x8, k_scale_words
):
    _mxfp8in_host(x, x_scale, w, w_scale, cos, sin, qrow, srow, qcol, scol, grid_m, num_heads, mac, swizzle_size, t2r_x8, k_scale_words, stream)


def _as_e8m0_array(scale: Any) -> Any:
    """Present a uint8 E8M0-bit-pattern array as float8_e8m0fnu (free bitcast, jit-safe)."""
    if _convert_to_cutlass_data_type(scale.dtype) is cutlass.Uint8:
        import ml_dtypes

        return scale.view(ml_dtypes.float8_e8m0fnu)
    return scale


def gemm_proj_rope_mxfp8_jax_sm100(
    x: Any,
    w: Any,
    cos: Any,
    sin: Any,
    x_scale: Optional[Any] = None,
    w_scale: Optional[Any] = None,
) -> Tuple[Any, Any, Any, Any]:
    """Projection GEMM + RoPE + dual-direction MXFP8 quantize as an XLA custom call.

    Same contract as the eager wrapper with ``w_out_in=True`` (the only JAX-expressible
    weight layout): ``x [tokens, K]`` / ``w [out, in]`` bfloat16 for the BF16 GEMM, or
    ``float8_e4m3fn`` codes plus E8M0 block scales (``uint8`` bit patterns or
    ``float8_e8m0fnu``) for the MXFP8 GEMM. Returns
    ``(out_fp8_row, out_scales_row, out_fp8_col, out_scales_col)``.
    """
    x_cutlass_dtype = _convert_to_cutlass_data_type(x.dtype)
    tokens = x.shape[0]
    proj_dim = w.shape[0]
    num_heads = proj_dim // HEAD_DIM

    out_types = (
        jax.ShapeDtypeStruct((tokens, num_heads, HEAD_DIM), np.dtype("float8_e4m3fn")),
        jax.ShapeDtypeStruct((tokens, num_heads, HEAD_DIM // BLOCK), np.uint8),
        jax.ShapeDtypeStruct((tokens, num_heads, HEAD_DIM), np.dtype("float8_e4m3fn")),
        jax.ShapeDtypeStruct((tokens // BLOCK, num_heads, HEAD_DIM), np.uint8),
    )

    if x_cutlass_dtype is cutlass.BFloat16:
        if x_scale is not None or w_scale is not None:
            raise ValueError("bf16 inputs must not be given MXFP8 scales (x_scale/w_scale); those are for the float8_e4m3fn path")
        cache_key = (tuple(x.shape), tuple(w.shape))
        entry = _bf16in_grid_cache.get(cache_key)
        if entry is None:
            obj = GemmProjRopeMxfp8Bf16InSm100(
                sample_x=_make_desc(tuple(x.shape), x.dtype, "sample_x"),
                sample_w=_make_desc(tuple(w.shape), w.dtype, "sample_w"),
                sample_cos=_make_desc(tuple(cos.shape), cos.dtype, "sample_cos"),
                sample_sin=_make_desc(tuple(sin.shape), sin.dtype, "sample_sin"),
                sample_out_fp8_row=_make_desc(out_types[0].shape, cutlass.Float8E4M3FN, "sample_out_fp8_row"),
                sample_out_scales_row=_make_desc(out_types[1].shape, cutlass.Uint8, "sample_out_scales_row"),
                sample_out_fp8_col=_make_desc(out_types[2].shape, cutlass.Float8E4M3FN, "sample_out_fp8_col"),
                sample_out_scales_col=_make_desc(out_types[3].shape, cutlass.Uint8, "sample_out_scales_col"),
                w_out_in=True,
            )
            assert obj.check_support()
            mac = cutlass.utils.HardwareInfo().get_max_active_clusters(1)
            entry = (tokens // TILE_M, num_heads, mac, 8)
            _bf16in_grid_cache[cache_key] = entry
        grid_m, heads, mac, swizzle = entry

        return call(
            _proj_rope_bf16in_adapter,
            output_shape_dtype=out_types,
            grid_m=grid_m,
            num_heads=heads,
            mac=mac,
            swizzle_size=swizzle,
        )(x, w, cos, sin)

    if x_cutlass_dtype is cutlass.Float8E4M3FN:
        if x_scale is None or w_scale is None:
            raise ValueError("MXFP8 (float8_e4m3fn) inputs require x_scale and w_scale (E8M0 rowwise block scales)")
        x_scale = _as_e8m0_array(x_scale)
        w_scale = _as_e8m0_array(w_scale)
        cache_key = (tuple(x.shape), tuple(w.shape))
        entry = _mxfp8in_grid_cache.get(cache_key)
        if entry is None:
            obj = GemmProjRopeMxfp8Mxfp8InSm100(
                sample_x_code=_make_desc(tuple(x.shape), x.dtype, "sample_x_code"),
                sample_x_scale=_make_desc(tuple(x_scale.shape), cutlass.Uint8, "sample_x_scale"),
                sample_w_code=_make_desc(tuple(w.shape), w.dtype, "sample_w_code"),
                sample_w_scale=_make_desc(tuple(w_scale.shape), cutlass.Uint8, "sample_w_scale"),
                sample_cos=_make_desc(tuple(cos.shape), cos.dtype, "sample_cos"),
                sample_sin=_make_desc(tuple(sin.shape), sin.dtype, "sample_sin"),
                sample_out_fp8_row=_make_desc(out_types[0].shape, cutlass.Float8E4M3FN, "sample_out_fp8_row"),
                sample_out_scales_row=_make_desc(out_types[1].shape, cutlass.Uint8, "sample_out_scales_row"),
                sample_out_fp8_col=_make_desc(out_types[2].shape, cutlass.Float8E4M3FN, "sample_out_fp8_col"),
                sample_out_scales_col=_make_desc(out_types[3].shape, cutlass.Uint8, "sample_out_scales_col"),
            )
            assert obj.check_support()
            grid_m, t2r_x8, swizzle = obj._grid_params()
            mac = cutlass.utils.HardwareInfo().get_max_active_clusters(1)
            entry = (grid_m, num_heads, mac, swizzle, t2r_x8, int(x.shape[1]) // 128)
            _mxfp8in_grid_cache[cache_key] = entry
        grid_m, heads, mac, swizzle, t2r_x8, k_scale_words = entry

        return call(
            _proj_rope_mxfp8in_adapter,
            output_shape_dtype=out_types,
            grid_m=grid_m,
            num_heads=heads,
            mac=mac,
            swizzle_size=swizzle,
            t2r_x8=t2r_x8,
            k_scale_words=k_scale_words,
        )(x, x_scale, w, w_scale, cos, sin)

    raise ValueError(f"unsupported input dtype {x.dtype}; expected bfloat16 (BF16 GEMM) or float8_e4m3fn (MXFP8 GEMM)")
