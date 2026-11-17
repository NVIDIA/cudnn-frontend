# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""APIBase integration for SM90 and SM100 DSA sparse-prefill forward."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import torch

from cudnn.api_base import APIBase, TupleDict
from . import _interface_sm100 as _iface_sm100
from ._kv_formats import KV_FORMATS, KvFormat
from ._support_sm90 import sm90_dsa_cutedsl_requirement_error

# SM100 (H, D_qk) -> supported indexer_topk. H96 runs the H64 kernel on two
# overlapping 64-head windows, and H128 D576 (and H128 D512 with
# indexer_topk=2048) on two disjoint ones (see
# ``_interface_sm100._HEAD64_WINDOW_STARTS``).
_SUPPORTED_VARIANTS = {
    (64, 512): (0, 512, 1024, 2048),
    (64, 576): (0, 512, 1024, 2048),
    (96, 512): (0, 512, 1024, 2048),
    (96, 576): (0, 512, 1024, 2048),
    (128, 512): (0, 512, 1024, 2048),
    (128, 576): (0, 512, 1024, 2048),
}
_SM90_SUPPORTED_HEADS = (16, 32, 64, 96, 128)
_SM90_SUPPORTED_HEAD_DIMS = (512, 576)


class SparseAttentionForward(APIBase):
    """Sparse prefill forward for the supported Hopper and Blackwell shapes.

    SM90 ``compile()`` builds a storage-free symbolic launchable; its
    ``execute()`` requires caller-owned outputs and performs no allocation or
    JIT. The SM100 interface retains its concrete-tensor cache. Runtime
    sequence and top-k extents remain dynamic on both architectures.

    The KV pool is a ``(S_kv, D_qk)`` tensor of Q's dtype; a
    ``float8_e4m3fn`` ``(S_kv, D_qk)`` tensor with its FP32 ``(1,)``
    per-tensor descale (``sample_descale_kv``); or, with ``kv_format``, the
    ``torch.uint8`` rows ``(S_kv, record_bytes)`` of a packed cache (formats
    in ``_kv_formats``). ``sample_extra_kv`` declares a second pool, of any
    of these kinds, with its own token ids (``sample_extra_topk_idxs``, and
    optionally ``sample_extra_topk_length``): every query attends to its
    ``topk_idxs`` slots of ``sample_kv`` and its ``extra_topk_idxs`` slots of
    ``sample_extra_kv`` under one softmax. No architecture serves an FP8 or
    packed pool or a second pool yet: declaring one raises
    ``NotImplementedError``.
    """

    def __init__(
        self,
        sample_q: torch.Tensor,  # (total_S_q, H, D_qk) FP16/BF16
        sample_kv: torch.Tensor,  # (total_S_kv, D_qk) of Q's dtype or float8_e4m3fn, or packed uint8 rows; K=V latent
        sample_topk_idxs: torch.Tensor,  # (total_S_q, logical_K) INT32
        sample_attn_sink: Optional[torch.Tensor] = None,  # (H,) FP32
        sample_topk_length: Optional[torch.Tensor] = None,  # (total_S_q,) INT32
        softmax_scale: Optional[float] = None,
        indexer_topk: int = 0,
        *,
        kv_format: Optional[str] = None,  # packed cache format of sample_kv; None: a typed (total_S_kv, D_qk) pool
        sample_extra_kv: Optional[torch.Tensor] = None,  # second pool, indexed by sample_extra_topk_idxs
        extra_kv_format: Optional[str] = None,  # packed cache format of sample_extra_kv
        sample_extra_topk_idxs: Optional[torch.Tensor] = None,  # (total_S_q, extra_K) INT32 token ids in sample_extra_kv
        sample_extra_topk_length: Optional[torch.Tensor] = None,  # (total_S_q,) INT32, clamped to [0, extra_K]
        sample_descale_kv: Optional[torch.Tensor] = None,  # (1,) FP32 per-tensor descale of a float8_e4m3fn sample_kv
        sample_descale_extra_kv: Optional[torch.Tensor] = None,  # (1,) FP32 per-tensor descale of a float8_e4m3fn sample_extra_kv
    ):
        """Capture a sparse-attention declaration for support checking and compilation."""
        super().__init__()
        self.q_desc = self._make_tensor_desc(sample_q, name="sample_q")
        self.kv_desc = self._make_tensor_desc(sample_kv, name="sample_kv")
        self.extra_kv_desc = self._make_tensor_desc(sample_extra_kv, name="sample_extra_kv")
        self.extra_topk_idxs_desc = self._make_tensor_desc(sample_extra_topk_idxs, name="sample_extra_topk_idxs")
        self.extra_topk_length_desc = self._make_tensor_desc(sample_extra_topk_length, name="sample_extra_topk_length")
        self.descale_kv_desc = self._make_tensor_desc(sample_descale_kv, name="sample_descale_kv")
        self.descale_extra_kv_desc = self._make_tensor_desc(sample_descale_extra_kv, name="sample_descale_extra_kv")
        self.kv_format = kv_format
        self.extra_kv_format = extra_kv_format
        self.topk_idxs_desc = self._make_tensor_desc(sample_topk_idxs, name="sample_topk_idxs")
        self.attn_sink_desc = self._make_tensor_desc(sample_attn_sink, name="sample_attn_sink")
        self.topk_length_desc = self._make_tensor_desc(sample_topk_length, name="sample_topk_length")
        self.softmax_scale = None if softmax_scale is None else float(softmax_scale)
        self.indexer_topk = int(indexer_topk)
        self.num_heads: Optional[int] = None
        self.head_dim: Optional[int] = None
        self.head_dim_v: Optional[int] = None
        self.logical_topk: Optional[int] = None
        self._device_major: Optional[int] = None

    def check_support(self) -> bool:
        """Validate the sampled declaration and select its architecture path."""
        self._value_error_if(self.q_desc.ndim != 3, f"Q must be 3-D (total_S_q, H, D_qk), got {self.q_desc.shape}")
        self._value_error_if(self.kv_desc.ndim != 2, f"KV must be 2-D (total_S_kv, D_qk), got {self.kv_desc.shape}")
        self._value_error_if(
            self.topk_idxs_desc.ndim != 2,
            f"topk_idxs must be 2-D (total_S_q, logical_K), got {self.topk_idxs_desc.shape}",
        )

        total_s_q, num_heads, head_dim = self.q_desc.shape
        total_s_kv, kv_head_dim = self.kv_desc.shape
        topk_s_q, logical_topk = self.topk_idxs_desc.shape
        del total_s_kv, kv_head_dim

        self._check_dtype(self.q_desc, [torch.float16, torch.bfloat16], name="Q")
        kv_fmt = self._check_kv_storage(self.kv_desc, self.kv_format, self.descale_kv_desc, name="KV", format_arg="kv_format", descale_arg="descale_kv")
        self._check_dtype(self.topk_idxs_desc, torch.int32, name="topk_idxs")
        self._check_pool_width(self.kv_desc, kv_fmt, head_dim, name="KV", rows="total_S_kv")
        self._value_error_if(topk_s_q != total_s_q, f"topk_idxs first dimension ({topk_s_q}) must match Q ({total_s_q})")
        self._check_extra_kv(total_s_q, head_dim)

        ref_device = self.q_desc.device
        descs = [self.q_desc, self.kv_desc, self.topk_idxs_desc]
        descs += [
            desc
            for desc in (self.extra_kv_desc, self.extra_topk_idxs_desc, self.extra_topk_length_desc, self.descale_kv_desc, self.descale_extra_kv_desc)
            if desc is not None
        ]
        if self.attn_sink_desc is not None:
            descs.append(self.attn_sink_desc)
        if self.topk_length_desc is not None:
            descs.append(self.topk_length_desc)
        self._value_error_if(ref_device.type != "cuda", f"Q must live on CUDA, got {ref_device}")
        self._value_error_if(
            any(desc.device != ref_device for desc in descs),
            f"All inputs must share Q's device {ref_device}, got {[desc.device for desc in descs]}",
        )

        capability = torch.cuda.get_device_capability(ref_device)
        major = capability[0]
        variant = (num_heads, head_dim)
        if major == 9:
            self._value_error_if(
                num_heads not in _SM90_SUPPORTED_HEADS or head_dim not in _SM90_SUPPORTED_HEAD_DIMS,
                f"SM90 SparseAttentionForward supports H in {_SM90_SUPPORTED_HEADS} and D_qk in {_SM90_SUPPORTED_HEAD_DIMS}, got {variant}",
            )
            supported_indexer_topk = (0, 512, 1024, 2048)
        else:
            self._value_error_if(
                variant not in _SUPPORTED_VARIANTS,
                "SM100 SparseAttentionForward supports only (H, D_qk) in " f"{tuple(_SUPPORTED_VARIANTS)}, got {variant}",
            )
            supported_indexer_topk = _SUPPORTED_VARIANTS.get(variant, ())
        if supported_indexer_topk:
            self._value_error_if(
                self.indexer_topk not in supported_indexer_topk,
                f"indexer_topk={self.indexer_topk} is unsupported for (H, D_qk)={variant}; expected one of {supported_indexer_topk}",
            )
        self._value_error_if(self.indexer_topk < 0, f"indexer_topk must be nonnegative, got {self.indexer_topk}")
        self._value_error_if(
            self.indexer_topk > logical_topk,
            f"indexer_topk ({self.indexer_topk}) must not exceed logical K ({logical_topk})",
        )

        if self.attn_sink_desc is not None:
            self._check_dtype(self.attn_sink_desc, torch.float32, name="attn_sink")
            self._value_error_if(
                self.attn_sink_desc.shape != (num_heads,),
                f"attn_sink must have shape {(num_heads,)}, got {self.attn_sink_desc.shape}",
            )
        if self.topk_length_desc is not None:
            self._check_dtype(self.topk_length_desc, torch.int32, name="topk_length")
            self._value_error_if(
                self.topk_length_desc.shape != (total_s_q,),
                f"topk_length must have shape {(total_s_q,)}, got {self.topk_length_desc.shape}",
            )

        self._runtime_error_if(
            major not in (9, 10),
            f"SparseAttentionForward requires an SM90 or SM100-family GPU, found SM{capability[0]}{capability[1]}",
        )
        if major == 9:
            dsl_error = sm90_dsa_cutedsl_requirement_error()
            self._runtime_error_if(dsl_error is not None, dsl_error or "")
            self._runtime_error_if(capability != (9, 0), f"SM90 SparseAttentionForward requires compute capability (9, 0), found {capability}")
        elif major == 10:
            # This also rejects capabilities without an architecture-specific
            # compiler target (for example an unmapped SM10x).
            _iface_sm100._gpu_arch_flag(ref_device)
        arch = f"SM{capability[0]}{capability[1]}"
        self._not_implemented_error_if(
            kv_fmt is not None,
            f"SparseAttentionForward does not serve kv_format={self.kv_format!r} on {arch}; pass a (total_S_kv, D_qk) pool of Q's dtype",
        )
        self._not_implemented_error_if(
            self.kv_desc.dtype == torch.float8_e4m3fn,
            f"SparseAttentionForward does not serve a per-tensor-scaled FP8 KV pool (descale_kv) on {arch}; pass a (total_S_kv, D_qk) pool of Q's dtype",
        )
        self._not_implemented_error_if(
            self.extra_kv_desc is not None and self.indexer_topk != 0,
            "SparseAttentionForward does not serve indexer_topk > 0 (lse_indexer) with a second KV pool (extra_kv); pass indexer_topk=0",
        )
        self._not_implemented_error_if(self.extra_kv_desc is not None, f"SparseAttentionForward does not serve a second KV pool (extra_kv) on {arch}")

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.head_dim_v = 512
        self.logical_topk = logical_topk
        self._device_major = major
        self._is_supported = True
        return True

    def _check_kv_storage(self, desc, kv_format: Optional[str], descale_desc, *, name: str, format_arg: str, descale_arg: str) -> Optional[KvFormat]:
        """Check a pool's dtype against its format and descale; return its packed format, or ``None`` for a typed pool.

        A typed pool holds Q's dtype, or ``float8_e4m3fn`` with an FP32 ``(1,)`` per-tensor descale. A packed pool is the
        ``torch.uint8`` rows of a named format, which carry their own scales.
        """
        if kv_format is None:
            if desc.dtype == torch.float8_e4m3fn:
                self._value_error_if(descale_desc is None, f"a float8_e4m3fn {name} pool needs {descale_arg}, its FP32 (1,) per-tensor descale")
                self._check_dtype(descale_desc, torch.float32, name=descale_arg)
                self._value_error_if(descale_desc.shape != (1,), f"{descale_arg} must have shape (1,), got {descale_desc.shape}")
                return None
            self._value_error_if(descale_desc is not None, f"{descale_arg} is the descale of a float8_e4m3fn {name} pool, but {name} is {desc.dtype}")
            hint = f"; a packed torch.uint8 cache also needs {format_arg}" if desc.dtype == torch.uint8 else ""
            self._check_dtype(desc, self.q_desc.dtype, name=name, extra_error_msg=f"{name} must have same dtype as Q{hint}")
            return None
        fmt = KV_FORMATS.get(kv_format) if isinstance(kv_format, str) else None
        self._value_error_if(fmt is None, f"unknown {format_arg} {kv_format!r}; the packed KV cache formats are {tuple(KV_FORMATS)}")
        self._check_dtype(desc, torch.uint8, name=name, extra_error_msg=f"a {fmt.name} cache is passed as the torch.uint8 tensor of its rows")
        self._value_error_if(descale_desc is not None, f"{descale_arg} does not apply to a {fmt.name} cache, which carries its own scales")
        return fmt

    def _check_pool_width(self, desc, fmt: Optional[KvFormat], head_dim: int, *, name: str, rows: str) -> None:
        """Check a pool against Q's ``D_qk``: its channels when typed, whole rows of its format when packed.

        A packed pool's row pitch, ``stride(0)``, is a multiple of 16 bytes and at least one row: rows may be padded
        apart but never overlap.
        """
        if fmt is None:
            self._value_error_if(desc.shape[1] != head_dim, f"{name} head dimension ({desc.shape[1]}) must match Q ({head_dim})")
            return
        self._value_error_if(
            desc.shape[1] != fmt.record_bytes or desc.stride[1] != 1,
            f"{name} in format {fmt.name} must be the ({rows}, {fmt.record_bytes}) byte rows of the cache, each row's bytes "
            f"contiguous; got shape {desc.shape} with stride {desc.stride}",
        )
        pitch = desc.stride[0]
        self._value_error_if(
            desc.shape[0] > 1 and (pitch < fmt.record_bytes or pitch % 16 != 0),
            f"{name} row pitch stride(0) must be a multiple of 16 bytes and at least {fmt.record_bytes}, got {pitch}",
        )
        self._value_error_if(fmt.head_dim_qk != head_dim, f"{fmt.name} rows hold D_qk={fmt.head_dim_qk} channels, but Q has D_qk={head_dim}")

    def _check_extra_kv(self, total_s_q: int, head_dim: int) -> None:
        """Validate the second pool and its own index table, ``extra_topk_idxs`` with an optional ``extra_topk_length``."""
        desc, idxs_desc, length_desc = self.extra_kv_desc, self.extra_topk_idxs_desc, self.extra_topk_length_desc
        if desc is None:
            given = [
                name
                for name, value in (
                    ("extra_kv_format", self.extra_kv_format),
                    ("extra_topk_idxs", idxs_desc),
                    ("extra_topk_length", length_desc),
                    ("descale_extra_kv", self.descale_extra_kv_desc),
                )
                if value is not None
            ]
            self._value_error_if(bool(given), f"{', '.join(given)} describe a second KV pool, but extra_kv is not given")
            return
        self._value_error_if(
            desc.ndim != 2,
            f"extra_kv must be 2-D: (total_S_extra_kv, D_qk) channels, or the (total_S_extra_kv, record_bytes) rows of a packed cache; got {desc.shape}",
        )
        fmt = self._check_kv_storage(
            desc, self.extra_kv_format, self.descale_extra_kv_desc, name="extra_kv", format_arg="extra_kv_format", descale_arg="descale_extra_kv"
        )
        self._check_pool_width(desc, fmt, head_dim, name="extra_kv", rows="total_S_extra_kv")

        self._value_error_if(idxs_desc is None, "extra_kv needs extra_topk_idxs, its own (total_S_q, extra_K) INT32 token ids")
        self._value_error_if(idxs_desc.ndim != 2, f"extra_topk_idxs must be 2-D (total_S_q, extra_K), got {idxs_desc.shape}")
        self._check_dtype(idxs_desc, torch.int32, name="extra_topk_idxs")
        self._value_error_if(idxs_desc.shape[0] != total_s_q, f"extra_topk_idxs first dimension ({idxs_desc.shape[0]}) must match Q ({total_s_q})")
        if length_desc is not None:
            self._check_dtype(length_desc, torch.int32, name="extra_topk_length")
            self._value_error_if(length_desc.shape != (total_s_q,), f"extra_topk_length must have shape {(total_s_q,)}, got {length_desc.shape}")

    def compile(self) -> None:
        """Compile the SM90 symbolic plan or prime the existing SM100 path."""
        self._ensure_support_checked()
        if self._device_major == 9:
            from . import _interface_sm90 as _iface_sm90

            self._compiled_kernel = _iface_sm90.compile_sparse_attention_forward_sm90(
                self.q_desc.device.index,
                self.q_desc.dtype,
                self.head_dim,
                self.attn_sink_desc is not None,
                self.topk_length_desc is not None,
                self.indexer_topk,
            )
            return
        # The interface compiles from concrete execute-time tensors after
        # stream-ordered normalization and top-k padding.
        self._compiled_kernel = True

    def execute(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        topk_idxs: torch.Tensor,
        *,
        attn_sink: Optional[torch.Tensor] = None,
        topk_length: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        max_logits: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        lse_indexer: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
        extra_kv: Optional[torch.Tensor] = None,
        extra_topk_idxs: Optional[torch.Tensor] = None,
        extra_topk_length: Optional[torch.Tensor] = None,
        descale_kv: Optional[torch.Tensor] = None,
        descale_extra_kv: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Execute a compiled sparse-attention plan.

        On SM90 this is the allocation-free, CUDA Graph-compatible entry:
        ``out``, ``max_logits``, ``lse``, and (when enabled) ``lse_indexer``
        must be preallocated, contiguous, and disjoint from all inputs and
        other outputs. Runtime H, Dqk, dtypes, and optional-operand presence
        must match the compiled declaration; sequence and logical top-k
        extents may change. No workspace is required.

        ``current_stream`` is either ``None`` for PyTorch's current stream or
        a concrete CUDA driver stream on Q's device. SM100 retains its existing
        direct execution behavior. ``extra_kv``, ``extra_topk_idxs``,
        ``extra_topk_length``, ``descale_kv`` and ``descale_extra_kv`` are each
        given exactly when the plan was declared with their ``sample_``
        counterparts.
        """
        if self._compiled_kernel is None:
            raise ValueError("SparseAttentionForward kernel not compiled")
        for name, given, declared in (
            ("extra_kv", extra_kv, self.extra_kv_desc),
            ("extra_topk_idxs", extra_topk_idxs, self.extra_topk_idxs_desc),
            ("extra_topk_length", extra_topk_length, self.extra_topk_length_desc),
            ("descale_kv", descale_kv, self.descale_kv_desc),
            ("descale_extra_kv", descale_extra_kv, self.descale_extra_kv_desc),
        ):
            if (given is not None) != (declared is not None):
                raise ValueError(f"{name} must be given exactly when the plan was declared with sample_{name}")
        scale = self.softmax_scale if softmax_scale is None else float(softmax_scale)
        if self._device_major == 9:
            from . import _interface_sm90 as _iface_sm90

            expected = (self.num_heads, self.head_dim, self.q_desc.dtype, self.kv_desc.dtype, self.topk_idxs_desc.dtype)
            actual = (
                q.shape[1] if q.ndim == 3 else None,
                q.shape[2] if q.ndim == 3 else None,
                q.dtype,
                kv.dtype,
                topk_idxs.dtype,
            )
            if actual != expected:
                raise ValueError(f"SM90 runtime (H, Dqk, Q dtype, KV dtype, index dtype) {actual} does not match the compiled plan {expected}")
            if (attn_sink is not None) != (self.attn_sink_desc is not None):
                raise ValueError("SM90 runtime attn_sink presence must match the compiled plan")
            if (topk_length is not None) != (self.topk_length_desc is not None):
                raise ValueError("SM90 runtime topk_length presence must match the compiled plan")

            return _iface_sm90.sparse_attention_forward_sm90(
                q,
                kv,
                topk_idxs,
                attn_sink=attn_sink,
                topk_length=topk_length,
                softmax_scale=scale,
                indexer_topk=self.indexer_topk,
                out=out,
                max_logits=max_logits,
                lse=lse,
                lse_indexer=lse_indexer,
                current_stream=current_stream,
                compiled_kernel=self._compiled_kernel,
            )
        return _iface_sm100.sparse_attention_forward_sm100(
            q,
            kv,
            topk_idxs,
            attn_sink=attn_sink,
            topk_length=topk_length,
            softmax_scale=scale,
            indexer_topk=self.indexer_topk,
            out=out,
            max_logits=max_logits,
            lse=lse,
            lse_indexer=lse_indexer,
            current_stream=current_stream,
        )


_cache_of_sparse_attention_forward_objects: dict = {}


def sparse_attention_forward_wrapper(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_idxs: torch.Tensor,
    *,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    indexer_topk: int = 0,
    stream: Optional[cuda.CUstream] = None,
    kv_format: Optional[str] = None,
    extra_kv: Optional[torch.Tensor] = None,
    extra_kv_format: Optional[str] = None,
    extra_topk_idxs: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    descale_kv: Optional[torch.Tensor] = None,
    descale_extra_kv: Optional[torch.Tensor] = None,
) -> TupleDict:
    """Allocate, normalize, and run SM90/SM100 sparse-prefill attention.

    ``lse_indexer`` is ``None`` when ``indexer_topk == 0``.  The wrapper
    accepts arbitrary logical K. Blackwell pads indices internally to a
    multiple of 64; Hopper handles the tail directly. It may allocate aligned
    input copies and outputs on ``stream``; use ``SparseAttentionForward`` with
    caller-owned outputs for an allocation-free CUDA Graph capture path.
    When ``topk_length`` is present, rows shorter than ``indexer_topk`` retain
    the source kernel's ``+inf`` ``lse_indexer`` sentinel.

    ``kv_format`` declares a packed cache and ``descale_kv`` the descale of an
    FP8 ``kv``; ``extra_kv``, ``extra_kv_format``, ``extra_topk_idxs``,
    ``extra_topk_length`` and ``descale_extra_kv`` declare a second pool with
    its own token ids, as on ``SparseAttentionForward``. No architecture
    serves these yet.
    """

    scale = 1.0 / math.sqrt(q.shape[-1]) if softmax_scale is None else float(softmax_scale)
    capability = torch.cuda.get_device_capability(q.device) if q.is_cuda else None
    if capability is not None and capability[0] == 9:
        dsl_error = sm90_dsa_cutedsl_requirement_error()
        if dsl_error is not None:
            raise RuntimeError(dsl_error)
        from . import _interface_sm90 as _iface_sm90

        q, kv, topk_idxs, attn_sink, topk_length = _iface_sm90.normalize_wrapper_inputs(q, kv, topk_idxs, attn_sink, topk_length, stream)
    num_heads = int(q.shape[1]) if q.ndim == 3 else None
    head_dim = int(q.shape[2]) if q.ndim == 3 else None
    # The API object owns lifecycle state, not compiled shape descriptors.
    # Execute-time normalization revalidates every tensor, so dynamic
    # sequence/top-k extents and layouts can safely share it. Device identity
    # already fixes capability; querying it again here only adds host overhead.
    key = (
        q.device,
        num_heads,
        head_dim,
        q.dtype,
        kv.dtype,
        topk_idxs.dtype,
        attn_sink is not None,
        topk_length is not None,
        int(indexer_topk),
        None if capability is None else tuple(capability),
        # repr() and getattr() keep a malformed argument hashable, so check_support() reports it.
        repr(kv_format),
        getattr(extra_kv, "dtype", type(extra_kv)),
        repr(extra_kv_format),
        getattr(extra_topk_idxs, "dtype", type(extra_topk_idxs)),
        extra_topk_length is not None,
        getattr(descale_kv, "dtype", type(descale_kv)),
        getattr(descale_extra_kv, "dtype", type(descale_extra_kv)),
    )
    obj = _cache_of_sparse_attention_forward_objects.get(key)
    if obj is None:
        obj = SparseAttentionForward(
            sample_q=q,
            sample_kv=kv,
            sample_topk_idxs=topk_idxs,
            sample_attn_sink=attn_sink,
            sample_topk_length=topk_length,
            indexer_topk=indexer_topk,
            kv_format=kv_format,
            sample_extra_kv=extra_kv,
            extra_kv_format=extra_kv_format,
            sample_extra_topk_idxs=extra_topk_idxs,
            sample_extra_topk_length=extra_topk_length,
            sample_descale_kv=descale_kv,
            sample_descale_extra_kv=descale_extra_kv,
        )
        obj.check_support()
        obj.compile()
        _cache_of_sparse_attention_forward_objects[key] = obj

    outputs = {}
    if capability is not None and capability[0] == 9:
        with _iface_sm100._launch_context(q.device, stream) as (stream, consumer):
            outputs["out"] = torch.empty((q.shape[0], q.shape[1], 512), dtype=q.dtype, device=q.device)
            outputs["max_logits"] = torch.empty((q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
            outputs["lse"] = torch.empty((q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
            outputs["lse_indexer"] = torch.empty_like(outputs["lse"]) if indexer_topk else None
            _iface_sm100._record_stream(tuple(outputs.values()), consumer)
    out, max_logits, lse, lse_indexer = obj.execute(
        q,
        kv,
        topk_idxs,
        attn_sink=attn_sink,
        topk_length=topk_length,
        softmax_scale=scale,
        out=outputs.get("out"),
        max_logits=outputs.get("max_logits"),
        lse=outputs.get("lse"),
        lse_indexer=outputs.get("lse_indexer"),
        current_stream=stream,
        extra_kv=extra_kv,
        extra_topk_idxs=extra_topk_idxs,
        extra_topk_length=extra_topk_length,
        descale_kv=descale_kv,
        descale_extra_kv=descale_extra_kv,
    )
    return TupleDict(out=out, max_logits=max_logits, lse=lse, lse_indexer=lse_indexer)
