# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Framework-neutral block-scaled dense GEMM + amax operation."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from .. import data_type
from .._dense_gemm import (
    block_scale_shape,
    data_type_bits,
    require_block_scale_layout,
    require_cluster_shape,
    require_compact_major,
    require_gemm_inputs,
    require_mma_tiler,
    require_tensor_shape,
)
from .._op import Op
from .._tensor_desc import TensorDesc

_AB_DTYPES = frozenset(
    {
        data_type.FP4_E2M1,
        data_type.UINT8,
        data_type.FP8_E4M3,
        data_type.FP8_E5M2,
    }
)
_SCALE_DTYPES = frozenset(
    {
        data_type.FP8_E4M3,
        data_type.FP8_E8M0,
        data_type.INT8,
    }
)
_C_DTYPES = frozenset(
    {
        data_type.FLOAT,
        data_type.HALF,
        data_type.BFLOAT16,
        data_type.FP8_E4M3,
        data_type.FP8_E5M2,
        data_type.FP4_E2M1,
        data_type.UINT8,
    }
)
_WIDE_C_DTYPES = frozenset(
    {
        data_type.FLOAT,
        data_type.HALF,
        data_type.BFLOAT16,
    }
)
_FP8_DTYPES = frozenset({data_type.FP8_E4M3, data_type.FP8_E5M2})
_FP4_STORAGE_DTYPES = frozenset({data_type.FP4_E2M1, data_type.UINT8})
_E8M0_STORAGE_DTYPES = frozenset({data_type.FP8_E8M0, data_type.INT8})


def _require_supported_dtype(
    label: str,
    dtype: data_type,
    supported: Iterable[data_type],
) -> None:
    if dtype not in supported:
        raise ValueError(f"{label} has unsupported dtype {dtype}")


def _require_logical_alignment(
    tensor: TensorDesc[Any],
    logical_dtype: data_type,
) -> None:
    bits = 4 if logical_dtype == data_type.FP4_E2M1 else data_type_bits(logical_dtype)
    extent = tensor.shape[tensor.stride_order[0]]
    if extent * bits % 128:
        required_multiple = 128 // bits
        raise ValueError(
            f"{tensor.name or 'Tensor'} contiguous extent must be a multiple of " f"{required_multiple} elements for 16-byte alignment, got {extent}"
        )


class GemmAmaxSm100Op(Op):
    """Complete logical signature and static configuration for GEMM + amax."""

    def __init__(
        self,
        *,
        a: TensorDesc[Any],
        b: TensorDesc[Any],
        sfa: TensorDesc[Any],
        sfb: TensorDesc[Any],
        c: TensorDesc[Any],
        amax: TensorDesc[Any],
        acc_dtype: data_type = data_type.FLOAT,
        mma_tiler_mn: tuple[int, int] = (128, 128),
        cluster_shape_mn: tuple[int, int] = (1, 1),
        sf_vec_size: int = 32,
    ) -> None:
        for name, desc in (
            ("a", a),
            ("b", b),
            ("sfa", sfa),
            ("sfb", sfb),
            ("c", c),
            ("amax", amax),
        ):
            if not isinstance(desc, TensorDesc):
                raise TypeError(f"{name} must be a TensorDesc, got {type(desc).__name__}")
        if not isinstance(acc_dtype, data_type):
            raise TypeError(f"acc_dtype must be a cudnn.data_type, got {type(acc_dtype).__name__}")

        self.a = a
        self.b = b
        self.sfa = sfa
        self.sfb = sfb
        self.c = c
        self.amax = amax
        self.acc_dtype = acc_dtype
        self.requested_mma_tiler_mn = mma_tiler_mn
        self.requested_cluster_shape_mn = cluster_shape_mn
        self.sf_vec_size = sf_vec_size

        self.m: Optional[int] = None
        self.n: Optional[int] = None
        self.k: Optional[int] = None
        self.l: Optional[int] = None
        self.a_major: Optional[str] = None
        self.b_major: Optional[str] = None
        self.c_major: Optional[str] = None
        self.ab_dtype: Optional[data_type] = None
        self.scale_dtype: Optional[data_type] = None
        self.c_dtype: Optional[data_type] = None
        self.mma_tiler_mn: Optional[tuple[int, int]] = None
        self.cluster_shape_mn: Optional[tuple[int, int]] = None

    def check_support(self) -> bool:
        """Validate the complete tensor signature and static configuration."""

        self.m = self.n = self.k = self.l = None
        self.a_major = self.b_major = self.c_major = None
        self.ab_dtype = self.scale_dtype = self.c_dtype = None
        self.mma_tiler_mn = self.cluster_shape_mn = None

        m, n, k, batch = require_gemm_inputs(self.a, self.b)
        require_tensor_shape(self.c, (m, n, batch), label="C")
        require_tensor_shape(self.amax, (1, 1, 1), label="Amax")

        if self.a.cudnn_dtype != self.b.cudnn_dtype:
            raise ValueError(f"A and B must have the same dtype, got {self.a.dtype} and {self.b.dtype}")
        ab_storage_dtype = self.a.cudnn_dtype
        _require_supported_dtype("A and B", ab_storage_dtype, _AB_DTYPES)
        ab_dtype = data_type.FP4_E2M1 if ab_storage_dtype in _FP4_STORAGE_DTYPES else ab_storage_dtype

        if self.sfa.cudnn_dtype != self.sfb.cudnn_dtype:
            raise ValueError(f"SFA and SFB must have the same dtype, got {self.sfa.dtype} and {self.sfb.dtype}")
        scale_storage_dtype = self.sfa.cudnn_dtype
        _require_supported_dtype("SFA and SFB", scale_storage_dtype, _SCALE_DTYPES)
        scale_dtype = data_type.FP8_E8M0 if scale_storage_dtype in _E8M0_STORAGE_DTYPES else scale_storage_dtype

        if self.sf_vec_size not in (16, 32):
            raise ValueError(f"sf_vec_size must be 16 or 32, got {self.sf_vec_size}")
        if scale_dtype == data_type.FP8_E4M3 and self.sf_vec_size == 32:
            raise ValueError("FP8 E4M3 scale factors do not support sf_vec_size=32")
        if ab_dtype in _FP8_DTYPES and self.sf_vec_size == 16:
            raise ValueError("FP8 A and B do not support sf_vec_size=16")
        require_tensor_shape(
            self.sfa,
            block_scale_shape(m, k, batch, self.sf_vec_size),
            label="SFA",
        )
        require_tensor_shape(
            self.sfb,
            block_scale_shape(n, k, batch, self.sf_vec_size),
            label="SFB",
        )
        require_block_scale_layout(self.sfa, "SFA")
        require_block_scale_layout(self.sfb, "SFB")

        c_storage_dtype = self.c.cudnn_dtype
        _require_supported_dtype("C", c_storage_dtype, _C_DTYPES)
        c_dtype = data_type.FP4_E2M1 if c_storage_dtype in _FP4_STORAGE_DTYPES else c_storage_dtype
        if self.amax.cudnn_dtype != data_type.FLOAT:
            raise ValueError(f"Amax must have dtype float32, got {self.amax.dtype}")
        if self.acc_dtype != data_type.FLOAT:
            raise ValueError(f"Accumulator dtype must be float32, got {self.acc_dtype}")

        a_major = require_compact_major(self.a, "m", "k")
        b_major = require_compact_major(self.b, "n", "k")
        c_major = require_compact_major(self.c, "m", "n")

        if ab_dtype == data_type.FP4_E2M1 and (a_major, b_major) != ("k", "k"):
            raise ValueError("FP4 A and B require k-major layouts, got " f"{a_major}-major and {b_major}-major")
        if c_dtype == data_type.FP4_E2M1 and c_major != "n":
            raise ValueError(f"FP4 C requires n-major layout, got {c_major}-major")
        if c_dtype == data_type.FP4_E2M1 and ab_dtype != data_type.FP4_E2M1:
            raise ValueError("FP4 C requires FP4 A and B")
        if ab_dtype in _FP8_DTYPES and c_dtype in _FP8_DTYPES:
            raise NotImplementedError("FP8 A and B with FP8 C are unsupported because the kernel fails to launch")

        mma_tiler_mn = require_mma_tiler(
            self.requested_mma_tiler_mn,
            allowed_m=(128, 256),
            allowed_n=(128, 256),
        )
        if mma_tiler_mn[0] == 256:
            raise NotImplementedError("mma_tiler_mn[0] == 256 currently hangs")
        cluster_shape_mn = require_cluster_shape(
            self.requested_cluster_shape_mn,
            cta_group_size=2 if mma_tiler_mn[0] == 256 else 1,
            max_cluster_size=16,
        )
        if cluster_shape_mn[0] > 4 or cluster_shape_mn[1] > 4:
            raise ValueError("cluster_shape_mn entries must be at most 4 for scale-factor multicast, " f"got {cluster_shape_mn}")

        if ab_dtype == data_type.FP4_E2M1 and mma_tiler_mn[1] == 256 and k <= 128:
            raise ValueError(f"mma_tiler_mn (X, 256) requires K > 128 for FP4, got {k}")
        if mma_tiler_mn == (128, 256) and self.sf_vec_size == 16 and c_dtype in _WIDE_C_DTYPES:
            raise NotImplementedError("mma_tiler_mn (128, 256), sf_vec_size=16, and a 16/32-bit C " "dtype are unsupported because the kernel fails to launch")

        _require_logical_alignment(self.a, ab_dtype)
        _require_logical_alignment(self.b, ab_dtype)
        _require_logical_alignment(self.c, c_dtype)

        self.m, self.n, self.k, self.l = m, n, k, batch
        self.a_major, self.b_major, self.c_major = a_major, b_major, c_major
        self.ab_dtype = ab_dtype
        self.scale_dtype = scale_dtype
        self.c_dtype = c_dtype
        self.mma_tiler_mn = mma_tiler_mn
        self.cluster_shape_mn = cluster_shape_mn
        return True


__all__ = ["GemmAmaxSm100Op", "block_scale_shape"]
