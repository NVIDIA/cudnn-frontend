# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Framework-neutral validation for DeepSeek indexer forward."""

from __future__ import annotations

from typing import Any

from ... import data_type
from ..._op import Op
from ..._tensor_desc import TensorDesc

TMA_ALIGN_ELEMENTS = 4
SUPPORTED_COMPUTE_CAPABILITIES = (90, 100, 103, 107)


def _require_compact(
    desc: TensorDesc[Any],
    label: str,
    contiguous_axis: str,
) -> None:
    if not desc.is_compact() or desc.stride[-1] != 1:
        raise ValueError(
            f"{label} must be compact with its canonical {contiguous_axis} axis " f"contiguous, got stride {desc.stride} and stride order {desc.stride_order}"
        )


def _require_desc(name: str, desc: TensorDesc[Any] | None) -> None:
    if desc is not None and not isinstance(desc, TensorDesc):
        raise TypeError(f"{name} must be a TensorDesc or None, got {type(desc).__name__}")


class IndexerForwardOp(Op):
    """Complete tensor signature and static configuration for indexer forward."""

    def __init__(
        self,
        *,
        q: TensorDesc[Any],
        k: TensorDesc[Any],
        weight: TensorDesc[Any],
        output: TensorDesc[Any],
        cu_seqlens_q: TensorDesc[Any] | None = None,
        cu_seqlens_k: TensorDesc[Any] | None = None,
        q_causal_offsets: TensorDesc[Any] | None = None,
        ratio: int = 4,
        qhead_per_kv_head: int | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        m_block_size: int = 128,
        n_block_size: int = 128,
        q_stage: int = 2,
        kv_stage: int = 4,
        sm_scale: float = 1.0,
        target_compute_capability: int = 100,
    ) -> None:
        for name, desc in (("q", q), ("k", k), ("weight", weight), ("output", output)):
            if not isinstance(desc, TensorDesc):
                raise TypeError(f"{name} must be a TensorDesc, got {type(desc).__name__}")
        for name, desc in (
            ("cu_seqlens_q", cu_seqlens_q),
            ("cu_seqlens_k", cu_seqlens_k),
            ("q_causal_offsets", q_causal_offsets),
        ):
            _require_desc(name, desc)

        self.q = q
        self.k = k
        self.weight = weight
        self.output = output
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        self.q_causal_offsets = q_causal_offsets
        self.ratio = int(ratio)
        self.requested_qhead_per_kv_head = qhead_per_kv_head
        self.requested_max_seqlen_q = max_seqlen_q
        self.requested_max_seqlen_k = max_seqlen_k
        self.m_block_size = int(m_block_size)
        self.n_block_size = int(n_block_size)
        self.q_stage = int(q_stage)
        self.kv_stage = int(kv_stage)
        self.sm_scale = float(sm_scale)
        self.target_compute_capability = int(target_compute_capability)

        self.is_varlen: bool | None = None
        self.batch_size: int | None = None
        self.s_q: int | None = None
        self.s_k: int | None = None
        self.h_q: int | None = None
        self.h_kv: int | None = None
        self.head_dim: int | None = None
        self.qhead_per_kv_head: int | None = None

    def check_support(self) -> bool:
        is_varlen = self.cu_seqlens_q is not None or self.cu_seqlens_k is not None
        if (self.cu_seqlens_q is None) != (self.cu_seqlens_k is None):
            raise ValueError("THD input requires both cu_seqlens_q and cu_seqlens_k")
        if self.target_compute_capability not in SUPPORTED_COMPUTE_CAPABILITIES:
            raise ValueError("target_compute_capability must be one of " f"{SUPPORTED_COMPUTE_CAPABILITIES}, got {self.target_compute_capability}")

        for desc, label in ((self.q, "Q"), (self.k, "K"), (self.weight, "W")):
            if desc.cudnn_dtype != data_type.BFLOAT16:
                raise ValueError(f"{label} must have dtype bfloat16, got {desc.dtype}")
        if self.output.cudnn_dtype != data_type.FLOAT:
            raise ValueError(f"Out must have dtype float32, got {self.output.dtype}")

        if is_varlen:
            batch_size, s_q, s_k, h_q, h_kv, head_dim = self._check_varlen_signature()
        else:
            batch_size, s_q, s_k, h_q, h_kv, head_dim = self._check_fixed_signature()
        for desc, label, contiguous_axis in (
            (self.q, "Q", "D"),
            (self.k, "K", "D"),
            (self.weight, "W", "H"),
            (self.output, "Out", "K"),
        ):
            _require_compact(desc, label, contiguous_axis)

        if any(value <= 0 for value in (batch_size, s_q, s_k, h_q, h_kv, head_dim)):
            raise ValueError("IndexerForward dimensions must be positive, got " f"B={batch_size}, S_q={s_q}, S_k={s_k}, H_q={h_q}, H_kv={h_kv}, D={head_dim}")
        if head_dim != 128:
            raise ValueError(f"IndexerForward requires head_dim=128, got {head_dim}")

        qhead_per_kv_head = self.requested_qhead_per_kv_head
        if qhead_per_kv_head is None:
            if h_q % h_kv != 0:
                raise ValueError(f"H_q ({h_q}) must be divisible by H_kv ({h_kv})")
            qhead_per_kv_head = h_q // h_kv
        if qhead_per_kv_head * h_kv != h_q:
            raise ValueError("qhead_per_kv_head * H_kv must equal H_q, got " f"{qhead_per_kv_head} * {h_kv} != {h_q}")
        if qhead_per_kv_head not in (32, 64):
            raise ValueError(f"qhead_per_kv_head must be 32 or 64, got {qhead_per_kv_head}")
        if self.target_compute_capability < 100 and h_kv != 1:
            raise ValueError(f"SM90 IndexerForward requires H_kv=1, got {h_kv}")
        if self.ratio < 1:
            raise ValueError(f"ratio must be at least 1, got {self.ratio}")

        tuning = (self.m_block_size, self.n_block_size, self.q_stage, self.kv_stage)
        if any(value <= 0 for value in tuning):
            raise ValueError(f"IndexerForward tuning values must be positive, got {tuning}")
        if self.target_compute_capability < 100 and tuning != (128, 128, 2, 4):
            raise ValueError("SM90 IndexerForward supports only m_block_size=128, n_block_size=128, " f"q_stage=2, kv_stage=4; got {tuning}")

        if self.q_causal_offsets is not None:
            offsets = self.q_causal_offsets
            if offsets.ndim != 1 or offsets.shape != (batch_size,):
                raise ValueError(f"q_causal_offsets must have shape {(batch_size,)}, got {offsets.shape}")
            if offsets.cudnn_dtype != data_type.INT32:
                raise ValueError(f"q_causal_offsets must have dtype int32, got {offsets.dtype}")
            _require_compact(offsets, "q_causal_offsets", "entry")

        expected_leading_shape = (self.q.shape[0],) if is_varlen else (batch_size, s_q)
        output_seqlen_k = self.output.shape[-1]
        if self.output.shape[:-1] != expected_leading_shape or output_seqlen_k < s_k or output_seqlen_k % TMA_ALIGN_ELEMENTS:
            raise ValueError(
                "Out must have leading shape "
                f"{expected_leading_shape} and an S_k extent >= {s_k} divisible by "
                f"{TMA_ALIGN_ELEMENTS}, got {self.output.shape}"
            )

        self.is_varlen = is_varlen
        self.batch_size = batch_size
        self.s_q = s_q
        self.s_k = s_k
        self.h_q = h_q
        self.h_kv = h_kv
        self.head_dim = head_dim
        self.qhead_per_kv_head = qhead_per_kv_head
        return True

    def _check_fixed_signature(self) -> tuple[int, int, int, int, int, int]:
        if self.q.ndim != 4:
            raise ValueError(f"Q must be 4-D (B, S_q, H_q, D), got {self.q.shape}")
        if self.k.ndim != 4:
            raise ValueError(f"K must be 4-D (B, S_k, H_kv, D), got {self.k.shape}")
        if self.weight.ndim != 3:
            raise ValueError(f"W must be 3-D (B, S_q, H_q), got {self.weight.shape}")
        if self.output.ndim != 3:
            raise ValueError(f"Out must be 3-D, got {self.output.shape}")

        batch_size, s_q, h_q, head_dim = self.q.shape
        k_batch, s_k, h_kv, k_head_dim = self.k.shape
        if k_batch != batch_size:
            raise ValueError(f"Q and K batch dimensions must match, got {batch_size} and {k_batch}")
        if k_head_dim != head_dim:
            raise ValueError(f"Q and K head dimensions must match, got {head_dim} and {k_head_dim}")
        if self.weight.shape != (batch_size, s_q, h_q):
            raise ValueError(f"W must have shape {(batch_size, s_q, h_q)}, got {self.weight.shape}")
        if self.requested_max_seqlen_q not in (None, s_q):
            raise ValueError(f"max_seqlen_q must be omitted or equal S_q={s_q}")
        if self.requested_max_seqlen_k not in (None, s_k):
            raise ValueError(f"max_seqlen_k must be omitted or equal S_k={s_k}")
        return batch_size, s_q, s_k, h_q, h_kv, head_dim

    def _check_varlen_signature(self) -> tuple[int, int, int, int, int, int]:
        if self.q.ndim != 3:
            raise ValueError(f"THD Q must be 3-D (total_q, H_q, D), got {self.q.shape}")
        if self.k.ndim != 3:
            raise ValueError(f"THD K must be 3-D (total_k, H_kv, D), got {self.k.shape}")
        if self.weight.ndim != 2:
            raise ValueError(f"THD W must be 2-D (total_q, H_q), got {self.weight.shape}")
        if self.output.ndim != 2:
            raise ValueError(f"THD Out must be 2-D, got {self.output.shape}")

        if self.cu_seqlens_q is None or self.cu_seqlens_k is None:
            raise RuntimeError("THD cumulative sequence descriptors were not configured")
        for desc, label in ((self.cu_seqlens_q, "cu_seqlens_q"), (self.cu_seqlens_k, "cu_seqlens_k")):
            if desc.ndim != 1 or desc.cudnn_dtype != data_type.INT32:
                raise ValueError(f"{label} must be a 1-D int32 tensor, got shape {desc.shape} and dtype {desc.dtype}")
            _require_compact(desc, label, "entry")
        if self.cu_seqlens_q.shape != self.cu_seqlens_k.shape:
            raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same shape")
        if self.cu_seqlens_q.shape[0] < 2:
            raise ValueError("cu_seqlens_q must contain at least two entries")
        if self.requested_max_seqlen_q is None or self.requested_max_seqlen_k is None:
            raise ValueError("THD input requires max_seqlen_q and max_seqlen_k")

        total_q, h_q, head_dim = self.q.shape
        _, h_kv, k_head_dim = self.k.shape
        if k_head_dim != head_dim:
            raise ValueError(f"Q and K head dimensions must match, got {head_dim} and {k_head_dim}")
        if self.weight.shape != (total_q, h_q):
            raise ValueError(f"THD W must have shape {(total_q, h_q)}, got {self.weight.shape}")
        batch_size = self.cu_seqlens_q.shape[0] - 1
        return batch_size, int(self.requested_max_seqlen_q), int(self.requested_max_seqlen_k), h_q, h_kv, head_dim


__all__ = ["IndexerForwardOp", "SUPPORTED_COMPUTE_CAPABILITIES", "TMA_ALIGN_ELEMENTS"]
