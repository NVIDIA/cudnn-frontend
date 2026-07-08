# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Framework-neutral dense GEMM + SwiGLU operation."""

from __future__ import annotations

from typing import Any, Optional

from .. import data_type
from ..gemm.helpers import (
    block_scale_shape,
    require_16_byte_alignment,
    require_block_scale_layout,
    require_cluster_shape,
    require_compact_major,
    require_gemm_inputs,
    require_mma_tiler,
    require_tensor_shape,
)
from ..common.op import Op
from ..common.tensor_desc import TensorDesc

_STANDARD_AB_DTYPES = frozenset(
    {
        data_type.HALF,
        data_type.BFLOAT16,
        data_type.FLOAT,
        data_type.FP8_E4M3,
        data_type.FP8_E5M2,
    }
)
_STANDARD_OUTPUT_DTYPES = frozenset(
    {
        data_type.HALF,
        data_type.BFLOAT16,
    }
)
_STANDARD_AB12_DTYPES = _STANDARD_OUTPUT_DTYPES | {
    data_type.FLOAT,
    data_type.FP8_E4M3,
    data_type.FP8_E5M2,
}
_FP8_DTYPES = frozenset({data_type.FP8_E4M3, data_type.FP8_E5M2})
_BLOCK_SCALED_AB_DTYPES = frozenset({data_type.FP4_E2M1, *_FP8_DTYPES})
_BLOCK_SCALED_OUTPUT_DTYPES = frozenset(
    {
        data_type.HALF,
        data_type.BFLOAT16,
        data_type.FLOAT,
        *_FP8_DTYPES,
    }
)
_BLOCK_SCALE_DTYPES = frozenset({data_type.FP8_E4M3, data_type.FP8_E8M0})
_SWIGLU_BLOCK_COLUMNS = 32
_SWIGLU_COLUMNS_PER_PAIR = 2 * _SWIGLU_BLOCK_COLUMNS
_SWIGLU_MMA_N = tuple(range(_SWIGLU_COLUMNS_PER_PAIR, 257, _SWIGLU_COLUMNS_PER_PAIR))


class GemmSwigluSm100Op(Op):
    """Logical signature and launch configuration for standard GEMM + SwiGLU."""

    def __init__(
        self,
        *,
        a: TensorDesc[Any],
        b: TensorDesc[Any],
        ab12: TensorDesc[Any],
        c: TensorDesc[Any],
        alpha: float = 1.0,
        acc_dtype: data_type = data_type.FLOAT,
        mma_tiler_mn: tuple[int, int] = (128, 128),
        cluster_shape_mn: Optional[tuple[int, int]] = None,
    ) -> None:
        for name, desc in (("a", a), ("b", b), ("ab12", ab12), ("c", c)):
            if not isinstance(desc, TensorDesc):
                raise TypeError(f"{name} must be a TensorDesc, got {type(desc).__name__}")
        if not isinstance(acc_dtype, data_type):
            raise TypeError(f"acc_dtype must be a cudnn.data_type, got {type(acc_dtype).__name__}")

        self.a = a
        self.b = b
        self.ab12 = ab12
        self.c = c

        self.alpha = alpha
        self.acc_dtype = acc_dtype
        self.mma_tiler_mn = tuple(mma_tiler_mn)
        if cluster_shape_mn is None:
            use_2cta_default = len(self.mma_tiler_mn) == 2 and self.mma_tiler_mn[0] == 256
            self.cluster_shape_mn = (2, 2) if use_2cta_default else (1, 1)
        else:
            self.cluster_shape_mn = tuple(cluster_shape_mn)

        self.m: Optional[int] = None
        self.n: Optional[int] = None
        self.k: Optional[int] = None
        self.l: Optional[int] = None
        self.output_n: Optional[int] = None
        self.a_major: Optional[str] = None
        self.b_major: Optional[str] = None
        self.output_major: Optional[str] = None

    def check_support(self) -> bool:
        """Validate the logical signature and resolve its canonical modes."""

        self.m = self.n = self.k = self.l = self.output_n = None
        self.a_major = self.b_major = self.output_major = None

        m, n, k, l = require_gemm_inputs(self.a, self.b)
        if n % 2 != 0:
            raise ValueError(f"N must be even for SwiGLU input/gate pairs, got {n}")
        if n % _SWIGLU_COLUMNS_PER_PAIR != 0:
            raise ValueError(
                f"N must be divisible by {_SWIGLU_COLUMNS_PER_PAIR} because SwiGLU pairs consecutive {_SWIGLU_BLOCK_COLUMNS}-column blocks, got {n}"
            )
        output_n = n // 2

        require_tensor_shape(self.ab12, (m, n, l), label="AB12")
        require_tensor_shape(self.c, (m, output_n, l), label="C")

        a_major = require_compact_major(self.a, "m", "k")
        b_major = require_compact_major(self.b, "n", "k")
        ab12_major = require_compact_major(self.ab12, "m", "n")
        c_major = require_compact_major(self.c, "m", "n")
        if ab12_major != c_major:
            raise ValueError(f"AB12 and C must use the same major mode, got {ab12_major}-major and {c_major}-major")

        self.m, self.n, self.k, self.l = m, n, k, l
        self.output_n = output_n
        self.a_major, self.b_major, self.output_major = a_major, b_major, ab12_major

        self._check_standard_dtypes()
        self.mma_tiler_mn = require_mma_tiler(
            self.mma_tiler_mn,
            allowed_n=_SWIGLU_MMA_N,
        )
        cta_group_size = 2 if self.mma_tiler_mn[0] == 256 else 1
        if cta_group_size == 2 and m % self.mma_tiler_mn[0] != 0:
            raise ValueError(f"M must be divisible by {self.mma_tiler_mn[0]} for 2-CTA MMA, got {m}")
        self.cluster_shape_mn = require_cluster_shape(
            self.cluster_shape_mn,
            cta_group_size=cta_group_size,
        )
        if cta_group_size == 1 and self.cluster_shape_mn != (1, 1):
            raise ValueError("cluster_shape_mn must be (1, 1) with a 128-wide M tile")

        for tensor in (self.a, self.b, self.ab12, self.c):
            require_16_byte_alignment(tensor)
        return True

    def _check_standard_dtypes(self) -> None:
        ab_dtype = self.a.cudnn_dtype
        if ab_dtype not in _STANDARD_AB_DTYPES:
            raise ValueError(f"A dtype must be one of the supported dense GEMM dtypes, got {self.a.dtype}")
        if self.b.cudnn_dtype != ab_dtype:
            raise ValueError(f"A and B must have the same dtype, got {self.a.dtype} and {self.b.dtype}")

        ab12_dtype = self.ab12.cudnn_dtype
        if self.acc_dtype == data_type.FLOAT:
            if ab12_dtype not in _STANDARD_AB12_DTYPES:
                raise ValueError(f"AB12 has unsupported dtype {self.ab12.dtype} for float32 accumulation")
            if ab12_dtype in _FP8_DTYPES:
                raise NotImplementedError("FP8 AB12 output is currently disabled")
        elif self.acc_dtype == data_type.HALF:
            if ab12_dtype not in _STANDARD_OUTPUT_DTYPES:
                raise ValueError(f"AB12 has unsupported dtype {self.ab12.dtype} for float16 accumulation")
            if ab_dtype not in _FP8_DTYPES | {data_type.HALF}:
                raise ValueError(f"A and B dtype {self.a.dtype} is unsupported for float16 accumulation")
        else:
            raise ValueError(f"Accumulator dtype must be float32 or float16, got {self.acc_dtype}")

        if self.c.cudnn_dtype not in _STANDARD_OUTPUT_DTYPES:
            raise ValueError(f"C dtype must be float16 or bfloat16, got {self.c.dtype}")


class BlockScaledGemmSwigluSm100Op(Op):
    """Logical signature for block-scaled GEMM + SwiGLU and quantized outputs."""

    def __init__(
        self,
        *,
        a: TensorDesc[Any],
        b: TensorDesc[Any],
        sfa: TensorDesc[Any],
        sfb: TensorDesc[Any],
        ab12: TensorDesc[Any],
        c: TensorDesc[Any],
        sfc: TensorDesc[Any] | None = None,
        amax: TensorDesc[Any] | None = None,
        norm_const: TensorDesc[Any] | None = None,
        alpha: float = 1.0,
        acc_dtype: data_type = data_type.FLOAT,
        mma_tiler_mn: tuple[int, int] = (128, 128),
        cluster_shape_mn: Optional[tuple[int, int]] = None,
        sf_vec_size: int = 16,
        vector_f32: bool = False,
        ab12_stages: int = 4,
    ) -> None:
        for name, desc in (
            ("a", a),
            ("b", b),
            ("sfa", sfa),
            ("sfb", sfb),
            ("ab12", ab12),
            ("c", c),
        ):
            if not isinstance(desc, TensorDesc):
                raise TypeError(f"{name} must be a TensorDesc, got {type(desc).__name__}")
        for name, desc in (("sfc", sfc), ("amax", amax), ("norm_const", norm_const)):
            if desc is not None and not isinstance(desc, TensorDesc):
                raise TypeError(f"{name} must be a TensorDesc or None, got {type(desc).__name__}")
        if not isinstance(acc_dtype, data_type):
            raise TypeError(f"acc_dtype must be a cudnn.data_type, got {type(acc_dtype).__name__}")
        if isinstance(sf_vec_size, bool) or not isinstance(sf_vec_size, int):
            raise TypeError(f"sf_vec_size must be an int, got {type(sf_vec_size).__name__}")
        if not isinstance(vector_f32, bool):
            raise TypeError(f"vector_f32 must be a bool, got {type(vector_f32).__name__}")
        if isinstance(ab12_stages, bool) or not isinstance(ab12_stages, int):
            raise TypeError(f"ab12_stages must be an int, got {type(ab12_stages).__name__}")

        self.a = a
        self.b = b
        self.sfa = sfa
        self.sfb = sfb
        self.ab12 = ab12
        self.c = c
        self.sfc = sfc
        self.amax = amax
        self.norm_const = norm_const
        self.alpha = alpha
        self.acc_dtype = acc_dtype
        self.requested_mma_tiler_mn = mma_tiler_mn
        self.requested_cluster_shape_mn = cluster_shape_mn
        self.sf_vec_size = sf_vec_size
        self.vector_f32 = vector_f32
        self.ab12_stages = ab12_stages

        self.m: Optional[int] = None
        self.n: Optional[int] = None
        self.k: Optional[int] = None
        self.l: Optional[int] = None
        self.output_n: Optional[int] = None
        self.a_major: Optional[str] = None
        self.b_major: Optional[str] = None
        self.output_major: Optional[str] = None
        self.ab_dtype: Optional[data_type] = None
        self.sf_dtype: Optional[data_type] = None
        self.ab12_dtype: Optional[data_type] = None
        self.c_dtype: Optional[data_type] = None
        self.mma_tiler_mn: Optional[tuple[int, int]] = None
        self.cluster_shape_mn: Optional[tuple[int, int]] = None

    def check_support(self) -> bool:
        """Validate shapes, packed scale layouts, dtypes, and static tiling."""

        self.m = self.n = self.k = self.l = self.output_n = None
        self.a_major = self.b_major = self.output_major = None
        self.ab_dtype = self.sf_dtype = self.ab12_dtype = self.c_dtype = None
        self.mma_tiler_mn = self.cluster_shape_mn = None

        m, n, k, batch = require_gemm_inputs(self.a, self.b)
        if n % _SWIGLU_COLUMNS_PER_PAIR:
            raise ValueError(f"N must be divisible by {_SWIGLU_COLUMNS_PER_PAIR} for SwiGLU block pairs, got {n}")
        output_n = n // 2
        require_tensor_shape(self.ab12, (m, n, batch), label="AB12")
        require_tensor_shape(self.c, (m, output_n, batch), label="C")

        if self.sf_vec_size not in (16, 32):
            raise ValueError(f"sf_vec_size must be 16 or 32, got {self.sf_vec_size}")
        require_tensor_shape(self.sfa, block_scale_shape(m, k, batch, self.sf_vec_size), label="SFA")
        require_tensor_shape(self.sfb, block_scale_shape(n, k, batch, self.sf_vec_size), label="SFB")
        require_block_scale_layout(self.sfa, "SFA")
        require_block_scale_layout(self.sfb, "SFB")

        ab_dtype = self.a.cudnn_dtype
        if ab_dtype not in _BLOCK_SCALED_AB_DTYPES:
            raise ValueError(f"A has unsupported block-scaled dtype {self.a.dtype}")
        if self.b.cudnn_dtype != ab_dtype:
            raise ValueError(f"A and B must have the same dtype, got {self.a.dtype} and {self.b.dtype}")

        sf_dtype = self.sfa.cudnn_dtype
        if sf_dtype not in _BLOCK_SCALE_DTYPES:
            raise ValueError(f"SFA has unsupported scale-factor dtype {self.sfa.dtype}")
        if self.sfb.cudnn_dtype != sf_dtype:
            raise ValueError(f"SFA and SFB must have the same dtype, got {self.sfa.dtype} and {self.sfb.dtype}")
        if ab_dtype in _FP8_DTYPES and (sf_dtype, self.sf_vec_size) != (data_type.FP8_E8M0, 32):
            raise ValueError("FP8 A and B require FP8_E8M0 scales with sf_vec_size=32")
        if ab_dtype == data_type.FP4_E2M1 and (sf_dtype, self.sf_vec_size) == (data_type.FP8_E4M3, 32):
            raise ValueError("FP4 A and B do not support FP8_E4M3 scales with sf_vec_size=32")

        ab12_dtype = self.ab12.cudnn_dtype
        c_dtype = self.c.cudnn_dtype
        if ab12_dtype not in _BLOCK_SCALED_OUTPUT_DTYPES:
            raise ValueError(f"AB12 has unsupported dtype {self.ab12.dtype}")
        if c_dtype not in _BLOCK_SCALED_OUTPUT_DTYPES:
            raise ValueError(f"C has unsupported dtype {self.c.dtype}")
        if self.acc_dtype != data_type.FLOAT:
            raise ValueError(f"Accumulator dtype must be float32, got {self.acc_dtype}")
        if ab_dtype == data_type.FP4_E2M1 and c_dtype in _FP8_DTYPES:
            raise ValueError("FP4 A and B are not compatible with FP8 C")
        if c_dtype == data_type.FLOAT and ab12_dtype == data_type.FLOAT:
            raise NotImplementedError("float32 C and float32 AB12 are disabled because the kernel fails to launch")
        if ab_dtype in _FP8_DTYPES and (c_dtype in _FP8_DTYPES or ab12_dtype in _FP8_DTYPES or ab12_dtype == data_type.FLOAT):
            raise NotImplementedError("MXFP8 inputs require float16 or bfloat16 AB12 and non-FP8 C")

        if c_dtype in _FP8_DTYPES:
            if self.sfc is None or self.norm_const is None:
                raise ValueError("sfc and norm_const are required when C is FP8")
        if self.sfc is not None:
            require_tensor_shape(self.sfc, block_scale_shape(m, output_n, batch, self.sf_vec_size), label="SFC")
            require_block_scale_layout(self.sfc, "SFC")
            if self.sfc.cudnn_dtype != sf_dtype:
                raise ValueError(f"SFC must have the same dtype as SFA, got {self.sfc.dtype} and {self.sfa.dtype}")
        if self.norm_const is not None:
            require_tensor_shape(self.norm_const, (1,), label="norm_const")
            if self.norm_const.cudnn_dtype != data_type.FLOAT:
                raise ValueError(f"norm_const must have float32 dtype, got {self.norm_const.dtype}")
        if ab_dtype == data_type.FP4_E2M1 and c_dtype == data_type.BFLOAT16 and self.amax is None:
            raise ValueError("amax is required when A and B are FP4 and C is bfloat16")
        if self.amax is not None:
            require_tensor_shape(self.amax, (1,), label="amax")
            if self.amax.cudnn_dtype != data_type.FLOAT:
                raise ValueError(f"amax must have float32 dtype, got {self.amax.dtype}")

        a_major = require_compact_major(self.a, "m", "k")
        b_major = require_compact_major(self.b, "n", "k")
        ab12_major = require_compact_major(self.ab12, "m", "n")
        c_major = require_compact_major(self.c, "m", "n")
        if ab12_major != c_major:
            raise ValueError(f"AB12 and C must use the same major mode, got {ab12_major}-major and {c_major}-major")
        if ab_dtype == data_type.FP4_E2M1 and (a_major != "k" or b_major != "k"):
            raise ValueError("FP4 A and B must use K-major layouts")
        if ab_dtype == data_type.FP4_E2M1 and ab12_major != "n":
            raise ValueError("FP4 AB12 and C must use N-major layouts")

        mma_tiler_mn = require_mma_tiler(
            self.requested_mma_tiler_mn,
            allowed_n=_SWIGLU_MMA_N,
        )
        cta_group_size = 2 if mma_tiler_mn[0] == 256 else 1
        default_cluster = (2, 2) if cta_group_size == 2 else (1, 1)
        cluster_shape_mn = require_cluster_shape(
            default_cluster if self.requested_cluster_shape_mn is None else self.requested_cluster_shape_mn,
            cta_group_size=cta_group_size,
        )
        if cluster_shape_mn[0] > 4 or cluster_shape_mn[1] > 4:
            raise ValueError(f"cluster_shape_mn entries must be at most 4 for scale-factor multicast, got {cluster_shape_mn}")
        if m % mma_tiler_mn[0] or n % mma_tiler_mn[1]:
            raise ValueError(f"M and N must be divisible by mma_tiler_mn, got M={m}, N={n}, tile={mma_tiler_mn}")
        if self.ab12_stages <= 0:
            raise ValueError(f"ab12_stages must be positive, got {self.ab12_stages}")

        for tensor in (self.a, self.b, self.ab12, self.c):
            require_16_byte_alignment(tensor)

        self.m, self.n, self.k, self.l = m, n, k, batch
        self.output_n = output_n
        self.a_major, self.b_major, self.output_major = a_major, b_major, ab12_major
        self.ab_dtype, self.sf_dtype = ab_dtype, sf_dtype
        self.ab12_dtype, self.c_dtype = ab12_dtype, c_dtype
        self.mma_tiler_mn = mma_tiler_mn
        self.cluster_shape_mn = cluster_shape_mn
        return True


__all__ = [
    "BlockScaledGemmSwigluSm100Op",
    "GemmSwigluSm100Op",
    "block_scale_shape",
]
