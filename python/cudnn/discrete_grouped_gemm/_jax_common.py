# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared JAX metadata and validation for discrete grouped GEMMs."""

from __future__ import annotations

import os
from typing import Any

import jax.numpy as jnp

from .. import data_type
from ..gemm.helpers import (
    block_scale_shape,
    require_16_byte_alignment,
    require_block_scale_layout,
    require_cluster_shape,
    require_compact_major,
    require_mma_tiler,
    require_tensor_shape,
)
from .._jax import JaxApiBase, JaxTensorDesc
from .._jax.datatypes import jax_to_cudnn_dtype, normalize_jax_dtype
from .._jax.gemm import (
    BLOCK_SCALE_MODE,
    PROBABILITY_MODE,
    gemm_a_mode,
    gemm_b_mode,
    gemm_output_mode,
)
from .._jax.layout import mode_from_layout, to_public_axes

SUPPORTED_COMPUTE_CAPABILITIES = (100, 103, 107)
FIX_PAD_SIZE = 256
MAX_EXPERTS = 1024

FP8_DTYPES = frozenset({data_type.FP8_E4M3, data_type.FP8_E5M2})
AB_DTYPES = frozenset({data_type.FP4_E2M1, *FP8_DTYPES})
SF_DTYPES = frozenset({data_type.FP8_E4M3, data_type.FP8_E8M0})


def _require_dtype(
    desc: JaxTensorDesc, allowed: frozenset[data_type], label: str
) -> data_type:
    dtype = desc.cudnn_dtype
    if dtype not in allowed:
        expected = ", ".join(sorted(str(value) for value in allowed))
        raise ValueError(
            f"{label} dtype must be one of {{{expected}}}, got {desc.dtype}"
        )
    return dtype


class DiscreteGroupedGemmJaxBase(JaxApiBase):
    """Common signature for the stacked-expert JAX binding.

    The Torch API accepts device pointer tables because Torch exposes stable
    allocation addresses.  JAX arrays do not expose such an ABI, and pointer
    tables would hide the referenced buffers from XLA's liveness analysis.
    This binding therefore accepts compact stacked B/SFB arrays as explicit
    custom-call operands.  The existing discrete kernel derives one pointer
    per expert while initializing its TMA descriptors.
    """

    def _initialize_common(
        self,
        sample_a: Any,
        sample_b: Any,
        sample_sfa: Any,
        sample_sfb: Any,
        sample_padded_offsets: Any,
        sample_alpha: Any,
        *,
        acc_dtype: Any | None,
        mma_tiler_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int] | None,
        sf_vec_size: int,
        vector_f32: bool,
        m_aligned: int,
        use_dynamic_sched: bool,
        a_layout: str,
        b_layout: str,
        output_layout: str,
    ) -> None:
        self.a_layout = a_layout
        self.b_layout = b_layout
        self.output_layout = output_layout
        self.a_mode = gemm_a_mode(a_layout)
        self.b_mode = gemm_b_mode(b_layout)
        self.output_mode = gemm_output_mode(output_layout, name="output_layout")
        self.scale_mode = BLOCK_SCALE_MODE
        self.probability_mode = PROBABILITY_MODE
        self.bias_mode = mode_from_layout("LN", kernel_axes="NL")

        self.compute_capability: int | None = None
        self.a_desc = self._to_tensor_desc(sample_a, "sample_a", mode=self.a_mode)
        self.b_desc = self._to_tensor_desc(sample_b, "sample_b", mode=self.b_mode)
        self.sfa_desc = self._to_tensor_desc(
            sample_sfa, "sample_sfa", mode=self.scale_mode
        )
        self.sfb_desc = self._to_tensor_desc(
            sample_sfb, "sample_sfb", mode=self.scale_mode
        )
        self.padded_offsets_desc = self._to_tensor_desc(
            sample_padded_offsets, "sample_padded_offsets"
        )
        self.alpha_desc = self._to_tensor_desc(sample_alpha, "sample_alpha")

        self.acc_dtype = normalize_jax_dtype(acc_dtype, jnp.float32, "acc_dtype")
        self.requested_mma_tiler_mn = mma_tiler_mn
        self.requested_cluster_shape_mn = cluster_shape_mn
        self.sf_vec_size = sf_vec_size
        self.vector_f32 = vector_f32
        self.m_aligned = m_aligned
        self.use_dynamic_sched = use_dynamic_sched
        self.num_cluster_overlap_margin = int(
            os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0")
        )

        self.m = self.n = self.k = self.expert_cnt = None
        self.ab_dtype = self.sf_dtype = None
        self.mma_tiler_mn = self.cluster_shape_mn = None

    def _check_common(self) -> None:
        if self.a_desc.ndim != 3:
            raise ValueError(f"A must have rank 3, got shape {self.a_desc.shape}")
        if self.b_desc.ndim != 3:
            raise ValueError(f"B must have rank 3, got shape {self.b_desc.shape}")

        m, k, a_groups = self.a_desc.shape
        n, b_k, experts = self.b_desc.shape
        if m < 0 or min(n, k, experts) <= 0:
            raise ValueError(
                "M must be non-negative and N, K, and expert count must be "
                f"positive, got {(m, n, k, experts)}"
            )
        if a_groups != 1:
            raise ValueError(
                f"A must flatten all expert rows into a singleton group dimension, got {a_groups}"
            )
        if b_k != k:
            raise ValueError(f"B K dimension must match A, got {b_k} and {k}")
        if experts > MAX_EXPERTS:
            raise ValueError(
                f"expert count must not exceed {MAX_EXPERTS}, got {experts}"
            )
        if m % FIX_PAD_SIZE:
            raise ValueError(
                f"A M dimension must be divisible by {FIX_PAD_SIZE}, got {m}"
            )
        if n % 64:
            raise ValueError(f"B N dimension must be divisible by 64, got {n}")

        require_tensor_shape(
            self.padded_offsets_desc, (experts,), label="padded_offsets"
        )
        require_tensor_shape(self.alpha_desc, (experts,), label="alpha")
        if self.padded_offsets_desc.cudnn_dtype != data_type.INT32:
            raise ValueError(
                f"padded_offsets must have int32 dtype, got {self.padded_offsets_desc.dtype}"
            )
        if self.alpha_desc.cudnn_dtype != data_type.FLOAT:
            raise ValueError(
                f"alpha must have float32 dtype, got {self.alpha_desc.dtype}"
            )

        ab_dtype = _require_dtype(self.a_desc, AB_DTYPES, "A")
        if self.b_desc.cudnn_dtype != ab_dtype:
            raise ValueError(
                f"A and B must have the same dtype, got {self.a_desc.dtype} and {self.b_desc.dtype}"
            )
        sf_dtype = _require_dtype(self.sfa_desc, SF_DTYPES, "SFA")
        if self.sfb_desc.cudnn_dtype != sf_dtype:
            raise ValueError(
                f"SFA and SFB must have the same dtype, got {self.sfa_desc.dtype} and {self.sfb_desc.dtype}"
            )
        if self.sf_vec_size not in (16, 32):
            raise ValueError(f"sf_vec_size must be 16 or 32, got {self.sf_vec_size}")
        if sf_dtype == data_type.FP8_E4M3 and self.sf_vec_size == 32:
            raise ValueError("FP8_E4M3 scale factors require sf_vec_size=16")
        if ab_dtype in FP8_DTYPES and self.sf_vec_size != 32:
            raise ValueError("FP8 A and B require sf_vec_size=32")

        require_tensor_shape(
            self.sfa_desc, block_scale_shape(m, k, 1, self.sf_vec_size), label="SFA"
        )
        require_tensor_shape(
            self.sfb_desc,
            block_scale_shape(n, k, experts, self.sf_vec_size),
            label="SFB",
        )
        require_block_scale_layout(self.sfa_desc, "SFA")
        require_block_scale_layout(self.sfb_desc, "SFB")

        a_major = require_compact_major(self.a_desc, "m", "k")
        b_major = require_compact_major(self.b_desc, "n", "k")
        if a_major != "k" or b_major != "k":
            raise ValueError(
                "The discrete grouped kernels currently require K-major A and B layouts"
            )
        require_16_byte_alignment(self.a_desc)
        require_16_byte_alignment(self.b_desc)

        if jax_to_cudnn_dtype(self.acc_dtype) != data_type.FLOAT:
            raise ValueError(f"acc_dtype must be float32, got {self.acc_dtype}")
        if not isinstance(self.vector_f32, bool):
            raise TypeError(
                f"vector_f32 must be a bool, got {type(self.vector_f32).__name__}"
            )
        if not isinstance(self.use_dynamic_sched, bool):
            raise TypeError(
                f"use_dynamic_sched must be a bool, got {type(self.use_dynamic_sched).__name__}"
            )
        if self.m_aligned != FIX_PAD_SIZE:
            raise ValueError(f"m_aligned must be {FIX_PAD_SIZE}, got {self.m_aligned}")

        mma_tiler_mn = require_mma_tiler(
            self.requested_mma_tiler_mn,
            allowed_m=(128, 256),
            allowed_n=(256,),
        )
        cta_group_size = 2 if mma_tiler_mn[0] == 256 else 1
        default_cluster = (2, 1) if cta_group_size == 2 else (1, 1)
        cluster_shape_mn = require_cluster_shape(
            default_cluster
            if self.requested_cluster_shape_mn is None
            else self.requested_cluster_shape_mn,
            cta_group_size=cta_group_size,
        )
        if cluster_shape_mn[0] > 4 or cluster_shape_mn[1] > 4:
            raise ValueError(
                f"cluster_shape_mn entries must not exceed 4, got {cluster_shape_mn}"
            )
        cluster_tile_m = cluster_shape_mn[0] // cta_group_size * mma_tiler_mn[0]
        if cluster_tile_m not in (128, 256):
            raise ValueError(f"cluster M tile must be 128 or 256, got {cluster_tile_m}")

        self.m, self.n, self.k, self.expert_cnt = m, n, k, experts
        self.ab_dtype, self.sf_dtype = ab_dtype, sf_dtype
        self.mma_tiler_mn, self.cluster_shape_mn = mma_tiler_mn, cluster_shape_mn
        if m > 0:
            self.compute_capability = self._resolve_compute_capability(
                None,
                SUPPORTED_COMPUTE_CAPABILITIES,
                type(self).__name__,
            )

    def _canonical_desc(
        self,
        shape: tuple[int, ...],
        dtype: Any,
        name: str,
        *,
        mode: tuple[int, ...] | None = None,
        init_value: bool | int | float | None = None,
        ptr_assumed_align: int | None = None,
    ) -> JaxTensorDesc:
        return JaxTensorDesc.from_shape(
            to_public_axes(shape, mode),
            dtype,
            name=name,
            mode=mode,
            init_value=init_value,
            ptr_assumed_align=ptr_assumed_align,
        )

    def _workspace_desc(self, workspace_bytes: int) -> JaxTensorDesc:
        if workspace_bytes <= 0:
            raise ValueError(
                f"kernel workspace size must be positive, got {workspace_bytes}"
            )
        return self._canonical_desc(
            (workspace_bytes,),
            jnp.uint8,
            "workspace",
            ptr_assumed_align=128,
        )

    def _materialize_output_desc(
        self,
        desc: JaxTensorDesc | None,
    ) -> Any | None:
        """Materialize an inferred result without launching an empty GEMM."""

        if desc is None:
            return None
        metadata = self._to_shape_dtype_struct(desc)
        if desc.init_value is None:
            return jnp.empty(metadata.shape, dtype=metadata.dtype)
        return jnp.full(metadata.shape, desc.init_value, dtype=metadata.dtype)

    def _check_runtime_common(
        self,
        a_tensor: Any,
        b_tensor: Any,
        sfa_tensor: Any,
        sfb_tensor: Any,
        padded_offsets: Any,
        alpha_tensor: Any,
    ) -> None:
        self._check_tensor_signature(a_tensor, self.a_desc)
        self._check_tensor_signature(b_tensor, self.b_desc)
        self._check_tensor_signature(sfa_tensor, self.sfa_desc)
        self._check_tensor_signature(sfb_tensor, self.sfb_desc)
        self._check_tensor_signature(padded_offsets, self.padded_offsets_desc)
        self._check_tensor_signature(alpha_tensor, self.alpha_desc)


__all__ = [
    "AB_DTYPES",
    "DiscreteGroupedGemmJaxBase",
    "FIX_PAD_SIZE",
    "FP8_DTYPES",
    "MAX_EXPERTS",
    "SF_DTYPES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
]
