# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Framework-neutral validation shared by dense sReLU fusions."""

from __future__ import annotations

from typing import Any

from . import data_type
from ._dense_gemm import (
    block_scale_shape,
    require_16_byte_alignment,
    require_block_scale_layout,
    require_cluster_shape,
    require_compact_major,
    require_gemm_inputs,
    require_mma_tiler,
    require_tensor_shape,
)
from ._op import Op
from ._tensor_desc import TensorDesc

_AB_DTYPES = frozenset(
    {
        data_type.FP4_E2M1,
        data_type.UINT8,
        data_type.FP8_E4M3,
        data_type.FP8_E5M2,
    }
)
_FP4_DTYPES = frozenset({data_type.FP4_E2M1, data_type.UINT8})
_FP8_DTYPES = frozenset({data_type.FP8_E4M3, data_type.FP8_E5M2})
_WIDE_OUTPUT_DTYPES = frozenset({data_type.HALF, data_type.BFLOAT16, data_type.FLOAT})
_C_DTYPES = _WIDE_OUTPUT_DTYPES | _FP8_DTYPES


class BlockScaledGemmReluSm100Op(Op):
    """Logical tensor signature shared by sReLU forward and backward."""

    def __init__(
        self,
        *,
        a: TensorDesc[Any],
        b: TensorDesc[Any],
        c: TensorDesc[Any],
        d: TensorDesc[Any],
        sfa: TensorDesc[Any],
        sfb: TensorDesc[Any],
        prob: TensorDesc[Any],
        dprob: TensorDesc[Any] | None = None,
        sfd: TensorDesc[Any] | None = None,
        amax: TensorDesc[Any] | None = None,
        norm_const: TensorDesc[Any] | None = None,
        alpha: float = 1.0,
        acc_dtype: data_type = data_type.FLOAT,
        mma_tiler_mn: tuple[int, int] = (256, 256),
        cluster_shape_mn: tuple[int, int] | None = None,
        sf_vec_size: int = 16,
    ) -> None:
        for name, desc in (
            ("a", a),
            ("b", b),
            ("c", c),
            ("d", d),
            ("sfa", sfa),
            ("sfb", sfb),
            ("prob", prob),
        ):
            if not isinstance(desc, TensorDesc):
                raise TypeError(f"{name} must be a TensorDesc, got {type(desc).__name__}")
        for name, desc in (
            ("dprob", dprob),
            ("sfd", sfd),
            ("amax", amax),
            ("norm_const", norm_const),
        ):
            if desc is not None and not isinstance(desc, TensorDesc):
                raise TypeError(f"{name} must be a TensorDesc or None, got {type(desc).__name__}")
        if not isinstance(acc_dtype, data_type):
            raise TypeError(f"acc_dtype must be a cudnn.data_type, got {type(acc_dtype).__name__}")
        if isinstance(sf_vec_size, bool) or not isinstance(sf_vec_size, int):
            raise TypeError(f"sf_vec_size must be an int, got {type(sf_vec_size).__name__}")

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.sfa = sfa
        self.sfb = sfb
        self.prob = prob
        self.dprob = dprob
        self.sfd = sfd
        self.amax = amax
        self.norm_const = norm_const
        self.alpha = alpha
        self.acc_dtype = acc_dtype
        self.mma_tiler_mn = tuple(mma_tiler_mn)
        self.cluster_shape_mn = None if cluster_shape_mn is None else tuple(cluster_shape_mn)
        self.sf_vec_size = sf_vec_size

        self.m: int | None = None
        self.n: int | None = None
        self.k: int | None = None
        self.l: int | None = None
        self.a_major: str | None = None
        self.b_major: str | None = None
        self.output_major: str | None = None

    def check_support(self) -> bool:
        """Validate shapes, dtypes, compact layouts, and static tiling."""

        m, n, k, batch = require_gemm_inputs(self.a, self.b)
        require_tensor_shape(self.c, (m, n, batch), label="C")
        require_tensor_shape(self.d, (m, n, batch), label="D")
        require_tensor_shape(self.prob, (m, 1, batch), label="prob")
        if self.dprob is not None:
            require_tensor_shape(self.dprob, (m, 1, batch), label="dprob")

        if self.sf_vec_size not in (16, 32):
            raise ValueError(f"sf_vec_size must be 16 or 32, got {self.sf_vec_size}")
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
        for label, desc in (("SFA", self.sfa), ("SFB", self.sfb)):
            require_block_scale_layout(desc, label)

        if self.sfd is not None:
            raise NotImplementedError("SFD generation is not implemented by the current sReLU kernels")
        if self.amax is not None:
            require_tensor_shape(self.amax, (1,), label="amax")
        if self.norm_const is not None:
            raise ValueError("norm_const is only used with FP8 D output, which is not implemented")

        a_storage_dtype = self.a.cudnn_dtype
        b_storage_dtype = self.b.cudnn_dtype
        if b_storage_dtype != a_storage_dtype:
            raise ValueError(f"A and B must have the same dtype, got {self.a.dtype} and {self.b.dtype}")
        ab_dtype = data_type.FP4_E2M1 if a_storage_dtype == data_type.UINT8 else a_storage_dtype
        if ab_dtype not in _AB_DTYPES:
            raise ValueError(f"A has unsupported block-scaled GEMM dtype {self.a.dtype}")

        sf_dtype = self.sfa.cudnn_dtype
        if self.sfb.cudnn_dtype != sf_dtype:
            raise ValueError(f"SFA and SFB must have the same dtype, got {self.sfa.dtype} and {self.sfb.dtype}")
        if sf_dtype not in {data_type.FP8_E8M0, data_type.FP8_E4M3}:
            raise ValueError(f"SFA has unsupported scale-factor dtype {self.sfa.dtype}")
        if ab_dtype in _FP8_DTYPES and (sf_dtype, self.sf_vec_size) != (data_type.FP8_E8M0, 32):
            raise ValueError("FP8 inputs require FP8_E8M0 scales with sf_vec_size=32")
        if ab_dtype in _FP4_DTYPES and self.sf_vec_size == 32 and sf_dtype != data_type.FP8_E8M0:
            raise ValueError("FP4 inputs with sf_vec_size=32 require FP8_E8M0 scales")

        if self.c.cudnn_dtype not in _C_DTYPES:
            raise ValueError(f"C has unsupported dtype {self.c.dtype}")
        if self.d.cudnn_dtype in _FP8_DTYPES:
            raise NotImplementedError("FP8 D output is unavailable because SFD generation is not implemented")
        if self.d.cudnn_dtype not in _WIDE_OUTPUT_DTYPES:
            raise ValueError(f"D has unsupported dtype {self.d.dtype}")
        if self.prob.cudnn_dtype != data_type.FLOAT:
            raise ValueError(f"prob must have float32 dtype, got {self.prob.dtype}")
        if self.dprob is not None and self.dprob.cudnn_dtype != data_type.FLOAT:
            raise ValueError(f"dprob must have float32 dtype, got {self.dprob.dtype}")
        if self.acc_dtype != data_type.FLOAT:
            raise ValueError(f"Accumulator dtype must be float32, got {self.acc_dtype}")
        if self.amax is not None and self.amax.cudnn_dtype != data_type.FLOAT:
            raise ValueError(f"amax must have float32 dtype, got {self.amax.dtype}")

        a_major = require_compact_major(self.a, "m", "k")
        b_major = require_compact_major(self.b, "n", "k")
        c_major = require_compact_major(self.c, "m", "n")
        d_major = require_compact_major(self.d, "m", "n")
        if c_major != d_major:
            raise ValueError(f"C and D must use the same major mode, got {c_major}-major and {d_major}-major")
        if ab_dtype in _FP4_DTYPES and (a_major != "k" or b_major != "k"):
            raise ValueError("FP4 A and B must use K-major layouts")

        self.mma_tiler_mn = require_mma_tiler(
            self.mma_tiler_mn,
            allowed_n=(64, 128, 192, 256),
        )
        cta_group_size = 2 if self.mma_tiler_mn[0] == 256 else 1
        cta_tile_m = self.mma_tiler_mn[0] // cta_group_size
        if m % cta_tile_m:
            raise ValueError(f"M must be divisible by CTA_TILE_M={cta_tile_m} because the probability load is not predicated, got {m}")
        if self.cluster_shape_mn is None:
            self.cluster_shape_mn = (2, 1) if cta_group_size == 2 else (1, 1)
        self.cluster_shape_mn = require_cluster_shape(
            self.cluster_shape_mn,
            cta_group_size=cta_group_size,
        )
        if self.cluster_shape_mn[0] > 4 or self.cluster_shape_mn[1] > 4:
            raise ValueError("cluster_shape_mn entries must be at most 4 for scale-factor multicast, " f"got {self.cluster_shape_mn}")

        for tensor in (self.c, self.d):
            require_16_byte_alignment(tensor)
        for tensor in (self.a, self.b):
            if tensor.cudnn_dtype == data_type.UINT8:
                contiguous_extent = tensor.shape[tensor.stride_order[0]]
                if contiguous_extent * 4 % 128 != 0:
                    raise ValueError(
                        f"{tensor.name or 'Tensor'} contiguous extent must be a multiple of 32 logical FP4 elements "
                        f"for 16-byte alignment, got {contiguous_extent}"
                    )
            else:
                require_16_byte_alignment(tensor)

        self.m, self.n, self.k, self.l = m, n, k, batch
        self.ab_dtype = ab_dtype
        self.a_major, self.b_major, self.output_major = a_major, b_major, c_major
        return True


__all__ = ["BlockScaledGemmReluSm100Op"]
