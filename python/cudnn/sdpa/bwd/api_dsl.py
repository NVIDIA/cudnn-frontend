# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""cuDNN-frontend adapter over the FROST DSL SDPA backward kernels."""

from __future__ import annotations

import logging
import math
import os
from abc import abstractmethod
from typing import Optional

import torch
from cuda.bindings import driver as cuda

from cudnn.api_base import APIBase, TensorDesc, TupleDict
from cudnn.frost.template_loader import load_template
from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_FP16
from cudnn.sdpa.bwd.config_sm120 import (
    SEQ_KV_TILES as _SM120_KV_TILES,
    SEQ_Q_TILES as _SM120_Q_TILES,
    SUPPORTED_HEAD_DIMS as _SM120_SUPPORTED_HEAD_DIMS,
    TemplateParams as Sm120TemplateParams,
    padded_head_dim as _sm120_padded_head_dim,
)
from cudnn.sdpa.fwd.api_dsl import WorkspaceCarver, _torch_stream_context, ws_align

_SM120_KERNEL_FILE = "bprop_f16_sm120.py"
_SM120_DTYPE_QKV_CODE = {
    torch.bfloat16: DTYPE_BF16,
    torch.float16: DTYPE_FP16,
}
# delta / dq_accum rows are padded to multiples of 128 (the kernel's
# dq_accum layout contract: tile_q must divide 128).
_SM120_ROW_ROUND = 128
# dq_sem is sized for the smallest legal q-tile so one formula covers every
# tile choice; must match the template's fake_dq_sem sizing.
_SM120_MIN_Q_TILE = 32

_logger = logging.getLogger(__name__)


def _load_sm120_kernel_module(params: Sm120TemplateParams):
    """Load one uniquely named backward kernel module per parameter set."""

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", _SM120_KERNEL_FILE)
    return load_template(path, params, tag="sdpa_bwd_sm120")


def _round_up(x: int, a: int) -> int:
    return -(-int(x) // a) * a


class SdpaBwdDsl(APIBase):
    """Implementation-agnostic interface for FROST DSL SDPA-backward kernels."""

    def __init__(
        self,
        sample_q: torch.Tensor | TensorDesc,
        sample_k: torch.Tensor | TensorDesc,
        sample_v: torch.Tensor | TensorDesc,
        sample_o: torch.Tensor | TensorDesc,
        sample_do: torch.Tensor | TensorDesc,
        sample_stats: torch.Tensor | TensorDesc,
        sample_dq: torch.Tensor | TensorDesc,
        sample_dk: torch.Tensor | TensorDesc,
        sample_dv: torch.Tensor | TensorDesc,
        is_causal: bool = False,
        causal_bottom_right: bool = False,
        window_size_left: Optional[int] = None,
        deterministic: bool = False,
        scale_softmax: Optional[float] = None,
        tile_m: Optional[int] = None,
        tile_n: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._warn_experimental_api()
        self._logger.debug("Entering __init__")

        self.q_desc = self._make_tensor_desc(sample_q, name="q")
        self.k_desc = self._make_tensor_desc(sample_k, name="k")
        self.v_desc = self._make_tensor_desc(sample_v, name="v")
        self.o_desc = self._make_tensor_desc(sample_o, name="o")
        self.do_desc = self._make_tensor_desc(sample_do, name="dO")
        self.stats_desc = self._make_tensor_desc(sample_stats, name="stats")
        self.dq_desc = self._make_tensor_desc(sample_dq, name="dQ")
        self.dk_desc = self._make_tensor_desc(sample_dk, name="dK")
        self.dv_desc = self._make_tensor_desc(sample_dv, name="dV")

        self.is_causal = bool(is_causal)
        self.causal_bottom_right = bool(causal_bottom_right)
        self.window_size_left = None if window_size_left is None else int(window_size_left)
        self.deterministic = bool(deterministic)
        self.scale_softmax = scale_softmax
        self.tile_m = None if tile_m is None else int(tile_m)
        self.tile_n = None if tile_n is None else int(tile_n)

        self.batch_size: Optional[int] = None
        self.s_q_max: Optional[int] = None
        self.s_k_max: Optional[int] = None
        self.h_q: Optional[int] = None
        self.h_kv: Optional[int] = None
        self.head_dim: Optional[int] = None
        self.dtype: Optional[torch.dtype] = None
        self._initialize_implementation()
        self._logger.debug("__init__ completed")

    @abstractmethod
    def _initialize_implementation(self) -> None:
        """Initialize state private to specific implementations."""

    @abstractmethod
    def scratch_workspace_bytes(self) -> int:
        """Return the per-execution scratch requirement for this implementation."""

    @abstractmethod
    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        do_tensor: torch.Tensor,
        stats_tensor: torch.Tensor,
        dq_tensor: torch.Tensor,
        dk_tensor: torch.Tensor,
        dv_tensor: torch.Tensor,
        scale_softmax: Optional[float] = None,
        workspace: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Execute the compiled kernel chain using the common operand set."""


class SdpaBwdDslSm120(SdpaBwdDsl):
    """Compile and execute fixed-length SM120/SM121 SDPA backward."""

    def _initialize_implementation(self) -> None:
        # 0 = the kernel's per-head-dim CONFIG default.
        self.q_tile = 0 if self.tile_m is None else int(self.tile_m)
        self.kv_tile = 0 if self.tile_n is None else int(self.tile_n)
        self.compute_capability: Optional[tuple[int, int]] = None
        self._k_mod = None
        self._sq_rounded: Optional[int] = None
        # Kernel-facing head dim. When it differs from D, operands stage through zero-padded compact copies
        self.head_dim_padded: Optional[int] = None
        # name -> staging number-of-elements for each non-BSHD-compact port
        self._staging_numels: dict[str, int] = {}

    @staticmethod
    def _bshd_physical_ok(desc: TensorDesc) -> bool:
        """True when the logical-BHSD desc sits on compact BSHD storage."""

        b, h, s, d = desc.shape
        return tuple(desc.stride) == (s * h * d, d, h * d, 1)

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        from cudnn.sdpa.graph_analyzer import dense_layout_ok

        self._staging_numels = {}
        for desc in (self.q_desc, self.k_desc, self.v_desc, self.o_desc, self.do_desc, self.dq_desc, self.dk_desc, self.dv_desc):
            self._value_error_if(
                desc.ndim != 4,
                f"{desc.name} must be rank-4 (B, H, S, D); got {desc.ndim}",
            )
            self._value_error_if(
                not dense_layout_ok(tuple(desc.shape), tuple(desc.stride)),
                f"{desc.name} must have the head dim innermost-contiguous (stride 1) and "
                f"non-broadcast, non-overlapping strides (any B/H/S order, padded "
                f"strides allowed); got stride {desc.stride} shape {desc.shape}",
            )

        b, h_q, s_q, d_qk = self.q_desc.shape
        _, h_kv, s_kv, _ = self.k_desc.shape
        self._check_tensor_shape(self.k_desc, (b, h_kv, s_kv, d_qk), name="K")
        self._check_tensor_shape(self.v_desc, (b, h_kv, s_kv, d_qk), name="V")
        self._check_tensor_shape(self.o_desc, (b, h_q, s_q, d_qk), name="O")
        self._check_tensor_shape(self.do_desc, (b, h_q, s_q, d_qk), name="dO")
        self._check_tensor_shape(self.dq_desc, tuple(self.q_desc.shape), name="dQ")
        self._check_tensor_shape(self.dk_desc, tuple(self.k_desc.shape), name="dK")
        self._check_tensor_shape(self.dv_desc, tuple(self.v_desc.shape), name="dV")

        for label, val in (("B", b), ("H_q", h_q), ("H_kv", h_kv), ("S_q", s_q), ("S_kv", s_kv), ("D", d_qk)):
            self._value_error_if(int(val) <= 0, f"{label} must be > 0; got {val}")
        self._not_implemented_error_if(
            h_q != h_kv,
            f"SM120 DSL SDPA backward does not implement GQA / MQA; got H_q={h_q}, H_kv={h_kv}",
        )
        self.head_dim_padded = _sm120_padded_head_dim(int(d_qk)) if d_qk % 8 == 0 else None
        self._value_error_if(
            self.head_dim_padded is None,
            f"D ({d_qk}) must be a multiple of 8 and <= {max(_SM120_SUPPORTED_HEAD_DIMS)}",
        )
        # All operands stage when D pads; otherwise only non-BSHD-compact ones.
        for desc in (self.q_desc, self.k_desc, self.v_desc, self.o_desc, self.do_desc, self.dq_desc, self.dk_desc, self.dv_desc):
            if self.head_dim_padded != d_qk or not self._bshd_physical_ok(desc):
                self._staging_numels[desc.name] = math.prod(desc.shape[:-1]) * self.head_dim_padded

        self._value_error_if(
            self.stats_desc.ndim != 4 or tuple(self.stats_desc.shape) != (b, h_q, s_q, 1),
            f"stats must be (B, H_q, S_q, 1); got {tuple(self.stats_desc.shape)}",
        )
        self._value_error_if(
            not self.stats_desc.is_contiguous(),
            f"stats must be contiguous; got stride {self.stats_desc.stride}",
        )
        self._check_dtype(self.stats_desc, torch.float32, name="stats")

        self.dtype = self._check_dtype(self.q_desc, [torch.float16, torch.bfloat16], name="Q")
        for desc in (self.k_desc, self.v_desc, self.o_desc, self.do_desc, self.dq_desc, self.dk_desc, self.dv_desc):
            self._check_dtype(
                desc,
                self.dtype,
                name=desc.name,
                extra_error_msg=f"{desc.name} must match Q",
            )
            self._value_error_if(
                desc.device != self.q_desc.device,
                f"{desc.name} must be on device {self.q_desc.device}, got {desc.device}",
            )
        self._value_error_if(
            self.q_desc.device.type != "cuda",
            f"Q must be a CUDA tensor, got device {self.q_desc.device}",
        )

        self._value_error_if(
            self.q_tile not in (0,) + _SM120_Q_TILES,
            f"q_tile must be one of {(0,) + _SM120_Q_TILES} (0 = per-head-dim default); got {self.q_tile}",
        )
        self._value_error_if(
            self.kv_tile not in (0,) + _SM120_KV_TILES,
            f"kv_tile must be one of {(0,) + _SM120_KV_TILES} (0 = per-head-dim default); got {self.kv_tile}",
        )
        self._value_error_if(
            self.causal_bottom_right and not self.is_causal,
            "causal_bottom_right requires is_causal=True",
        )
        self._value_error_if(
            self.window_size_left is not None and self.window_size_left < 0,
            f"window_size_left must be non-negative, got {self.window_size_left}",
        )

        self._runtime_error_if(not torch.cuda.is_available(), "CUDA is not available")
        self.compute_capability = torch.cuda.get_device_capability(self.q_desc.device)
        self._runtime_error_if(
            self.compute_capability not in {(12, 0), (12, 1)},
            f"SdpaBwdDslSm120 requires SM120 or SM121, found SM{self.compute_capability[0]}{self.compute_capability[1]}",
        )

        if self.scale_softmax is None or self.scale_softmax == 0.0:
            self.scale_softmax = 1.0 / math.sqrt(d_qk)

        self.batch_size = int(b)
        self.s_q_max = int(s_q)
        self.s_k_max = int(s_kv)
        self.h_q = int(h_q)
        self.h_kv = int(h_kv)
        self.head_dim = int(d_qk)
        self._sq_rounded = _round_up(self.s_q_max, _SM120_ROW_ROUND)
        self._is_supported = True

        self._logger.debug("check_support completed successfully")
        return True

    def compile(self) -> None:
        """Compile the shape-specialized SM120 FROST backward template."""

        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return

        params = Sm120TemplateParams(
            dtype_qkv=_SM120_DTYPE_QKV_CODE[self.dtype],
            is_causal=self.is_causal,
            causal_top_left=self.is_causal and not self.causal_bottom_right,
            window_size_left=self.window_size_left,
            deterministic=self.deterministic,
            q_tile=self.q_tile,
            kv_tile=self.kv_tile,
        )
        self._k_mod = _load_sm120_kernel_module(params)
        self._compiled_kernel = self._k_mod.compile(
            compute_capability=self.compute_capability,
            b=self.batch_size,
            qh=self.h_q,
            sq=self.s_q_max,
            skv=self.s_k_max,
            d=self.head_dim_padded,
        )
        self._logger.debug("compile completed")

    def _dq_sem_len(self) -> int:
        """Element count of the dq_sem relay-counter buffer (int32)."""

        return self.batch_size * self.h_q * _round_up(self.s_q_max, _SM120_MIN_Q_TILE) // _SM120_MIN_Q_TILE

    def scratch_workspace_bytes(self) -> int:
        """delta (fp32 [B, H, SQ_r128]) + dq_accum (fp32 flat [B*SQ_r128*H*D])
        + dq_sem (int32 flat [B*H*ceil(SQ/32)], deterministic relay counters)
        + one compact staging copy per non-BSHD-compact operand."""

        self._ensure_support_checked()
        delta_bytes = ws_align(self.batch_size * self.h_q * self._sq_rounded * 4)
        dq_accum_bytes = ws_align(self.batch_size * self._sq_rounded * self.h_q * self.head_dim_padded * 4)
        dq_sem_bytes = ws_align(self._dq_sem_len() * 4)
        staging_bytes = sum(ws_align(numel * self.dtype.itemsize) for numel in self._staging_numels.values())
        return delta_bytes + dq_accum_bytes + dq_sem_bytes + staging_bytes

    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        do_tensor: torch.Tensor,
        stats_tensor: torch.Tensor,
        dq_tensor: torch.Tensor,
        dk_tensor: torch.Tensor,
        dv_tensor: torch.Tensor,
        scale_softmax: Optional[float] = None,
        workspace: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Execute tensors matching the compiled specialization."""

        if self._compiled_kernel is None:
            raise RuntimeError("SdpaBwdDslSm120 kernel is not compiled")

        scale_val = self.scale_softmax if scale_softmax is None or scale_softmax == 0.0 else float(scale_softmax)
        scale_log2 = scale_val * math.log2(math.e)

        carver = WorkspaceCarver(workspace, self.scratch_workspace_bytes(), "sdpa_bwd_sm120")
        delta = carver.take(self.batch_size * self.h_q * self._sq_rounded, torch.float32).reshape(self.batch_size, self.h_q, self._sq_rounded)
        dq_accum = carver.take(self.batch_size * self._sq_rounded * self.h_q * self.head_dim_padded, torch.float32)
        dq_sem = carver.take(self._dq_sem_len(), torch.int32)

        if current_stream is None:
            # Direct call (no dispatch-forwarded stream): fall back to torch's
            # current stream. A stream forwarded from the execute-time handle
            # is respected rather than clobbered.
            current_stream = cuda.CUstream(torch.cuda.current_stream(q_tensor.device).cuda_stream)

        import cutlass

        # Non-compact operands (all operands when D pads) stage through
        # workspace-carved compact copies with zero-filled pad columns.
        d_pad = self.head_dim_padded
        pads = d_pad != self.head_dim

        def _staged_bshd(tensor: torch.Tensor) -> torch.Tensor:
            view = tensor.transpose(1, 2)
            if not pads and view.is_contiguous():
                return view
            b, s, h, _ = view.shape
            staged = carver.take(b * s * h * d_pad, self.dtype).view(b, s, h, d_pad)
            if pads:
                staged[..., self.head_dim :].zero_()
            staged[..., : self.head_dim].copy_(view)
            return staged

        def _staged_out_bshd(tensor: torch.Tensor):
            """(kernel-facing compact BSHD buffer, user view to scatter back into or None)."""
            view = tensor.transpose(1, 2)
            if not pads and view.is_contiguous():
                return view, None
            b, s, h, _ = view.shape
            return carver.take(b * s * h * d_pad, self.dtype).view(b, s, h, d_pad), view

        with _torch_stream_context(current_stream, q_tensor.device):
            q = _staged_bshd(q_tensor)
            k = _staged_bshd(k_tensor)
            v = _staged_bshd(v_tensor)
            o = _staged_bshd(o_tensor)
            do = _staged_bshd(do_tensor)
            dq, dq_user = _staged_out_bshd(dq_tensor)
            dk, dk_user = _staged_out_bshd(dk_tensor)
            dv, dv_user = _staged_out_bshd(dv_tensor)
            lse = stats_tensor.reshape(self.batch_size, self.h_q, self.s_q_max)

            kernels = self._compiled_kernel
            # Three-kernel chain
            kernels.dot(o, do, delta, dq_accum, dq_sem, current_stream)
            kernels.main(
                q,
                k,
                v,
                do,
                lse,
                delta,
                dq_accum,
                dq_sem,
                dk,
                dv,
                cutlass.Float32(scale_log2),
                cutlass.Float32(scale_val),
                current_stream,
            )
            kernels.cvt(dq_accum, dq, cutlass.Float32(scale_val), current_stream)
            for user_view, staged in ((dq_user, dq), (dk_user, dk), (dv_user, dv)):
                if user_view is not None:
                    user_view.copy_(staged[..., : self.head_dim])


def _tensor_signature(tensor: torch.Tensor) -> tuple:
    """(shape, stride, dtype, device) — everything the specialization keys on."""
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)


_wrapper_api_cache: dict[tuple, SdpaBwdDslSm120] = {}


def sdpa_bwd_wrapper_dsl_sm120(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    o_tensor: torch.Tensor,
    do_tensor: torch.Tensor,
    stats_tensor: torch.Tensor,
    is_causal: bool = False,
    causal_bottom_right: bool = False,
    window_size_left: Optional[int] = None,
    deterministic: bool = False,
    scale_softmax: Optional[float] = None,
) -> TupleDict:
    """Run SM120 SDPA backward and return ``TupleDict(dq_tensor=..., dk_tensor=..., dv_tensor=...)``."""

    dq_tensor = torch.empty_strided(q_tensor.shape, q_tensor.stride(), dtype=q_tensor.dtype, device=q_tensor.device)
    dk_tensor = torch.empty_strided(k_tensor.shape, k_tensor.stride(), dtype=k_tensor.dtype, device=k_tensor.device)
    dv_tensor = torch.empty_strided(v_tensor.shape, v_tensor.stride(), dtype=v_tensor.dtype, device=v_tensor.device)

    # check_support()/compile() run only on a miss, so the key must carry the
    # full signature of every operand the specialization depends on (dq/dk/dv
    # are derived from q/k/v above).
    cache_key = (
        _tensor_signature(q_tensor),
        _tensor_signature(k_tensor),
        _tensor_signature(v_tensor),
        _tensor_signature(o_tensor),
        _tensor_signature(do_tensor),
        _tensor_signature(stats_tensor),
        bool(is_causal),
        bool(causal_bottom_right),
        window_size_left,
        bool(deterministic),
        scale_softmax,
    )
    api = _wrapper_api_cache.get(cache_key)
    if api is None:
        api = SdpaBwdDslSm120(
            sample_q=q_tensor,
            sample_k=k_tensor,
            sample_v=v_tensor,
            sample_o=o_tensor,
            sample_do=do_tensor,
            sample_stats=stats_tensor,
            sample_dq=dq_tensor,
            sample_dk=dk_tensor,
            sample_dv=dv_tensor,
            is_causal=is_causal,
            causal_bottom_right=causal_bottom_right,
            window_size_left=window_size_left,
            deterministic=deterministic,
            scale_softmax=scale_softmax,
        )
        api.check_support()
        api.compile()
        _wrapper_api_cache[cache_key] = api

    workspace = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device=q_tensor.device)
    api.execute(
        q_tensor=q_tensor,
        k_tensor=k_tensor,
        v_tensor=v_tensor,
        o_tensor=o_tensor,
        do_tensor=do_tensor,
        stats_tensor=stats_tensor,
        dq_tensor=dq_tensor,
        dk_tensor=dk_tensor,
        dv_tensor=dv_tensor,
        scale_softmax=scale_softmax,
        workspace=workspace,
    )
    return TupleDict(dq_tensor=dq_tensor, dk_tensor=dk_tensor, dv_tensor=dv_tensor)
