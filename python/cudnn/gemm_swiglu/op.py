# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Framework-neutral dense GEMM + SwiGLU operation."""

from __future__ import annotations

from typing import Any, Optional

from .. import data_type
from .._dense_gemm import (
    require_16_byte_alignment,
    require_cluster_shape,
    require_compact_major,
    require_gemm_inputs,
    require_mma_tiler,
    require_tensor_shape,
)
from .._op import Op
from .._tensor_desc import TensorDesc

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


__all__ = ["GemmSwigluSm100Op"]
