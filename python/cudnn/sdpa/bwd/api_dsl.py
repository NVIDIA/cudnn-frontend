# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""cuDNN-frontend adapter over the FROST DSL SDPA backward kernels."""

from __future__ import annotations

import inspect
import logging
import math
import os
from abc import abstractmethod
from contextlib import nullcontext
from functools import lru_cache
from typing import Optional

import torch
from cuda.bindings import driver as cuda

from cudnn.api_base import APIBase, TensorDesc, TupleDict
from cudnn.frost.template_loader import load_template
from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_FP16
from cudnn.sdpa.bwd.config_sm120 import (
    ROW_ROUND as _SM120_ROW_ROUND,
    SEQ_KV_TILES as _SM120_KV_TILES,
    SEQ_Q_TILES as _SM120_Q_TILES,
    SUPPORTED_HEAD_DIMS as _SM120_SUPPORTED_HEAD_DIMS,
    TemplateParams as Sm120TemplateParams,
    padded_head_dim as _sm120_padded_head_dim,
    padded_head_dims as _sm120_padded_head_dims,
)
from cudnn.sdpa.fwd.api_dsl import WorkspaceCarver, _torch_stream_context, ws_align
from cudnn.sdpa.fwd import config_sm80 as _fwd_config_sm80

_SM120_KERNEL_FILE = "sm120/bprop_f16.py"
_SM120_DTYPE_QKV_CODE = {
    torch.bfloat16: DTYPE_BF16,
    torch.float16: DTYPE_FP16,
}
# dq_sem is sized for the smallest legal q-tile so one formula covers every
# tile choice; must match the template's fake_dq_sem sizing.
_SM120_MIN_Q_TILE = 32
# (d_qk, d_v) pairs served by the deterministic two-kernel split; d64
# measured slower than the relay (dS-workspace traffic dominates).
_SM120_DET_2K_HEAD_DIM_PAIRS = ((128, 128), (192, 128), (256, 256))

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
        sample_sink: Optional[torch.Tensor | TensorDesc] = None,
        sample_dsink: Optional[torch.Tensor | TensorDesc] = None,
        sample_bias: Optional[torch.Tensor | TensorDesc] = None,
        sample_dbias: Optional[torch.Tensor | TensorDesc] = None,
        is_causal: bool = False,
        causal_bottom_right: bool = False,
        window_size_left: Optional[int] = None,
        window_size_right: Optional[int] = None,
        deterministic: bool = False,
        scale_softmax: Optional[float] = None,
        tile_m: Optional[int] = None,
        tile_n: Optional[int] = None,
        seq_kv_lens_present: bool = False,
        seq_q_lens_present: bool = False,
        thd: bool = False,
        max_total_seq_len_q: Optional[int] = None,
        max_total_seq_len_kv: Optional[int] = None,
        thd_stats_token_major: bool = False,
        thd_stats_head_stride: Optional[int] = None,
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
        self.sink_desc = self._make_tensor_desc(sample_sink, name="sink") if sample_sink is not None else None
        self.dsink_desc = self._make_tensor_desc(sample_dsink, name="dSink") if sample_dsink is not None else None
        self.bias_desc = self._make_tensor_desc(sample_bias, name="bias") if sample_bias is not None else None
        self.dbias_desc = self._make_tensor_desc(sample_dbias, name="dBias") if sample_dbias is not None else None

        self.is_causal = bool(is_causal)
        self.causal_bottom_right = bool(causal_bottom_right)
        self.window_size_left = None if window_size_left is None else int(window_size_left)
        self.window_size_right = None if window_size_right is None else int(window_size_right)
        self.deterministic = bool(deterministic)
        self.scale_softmax = scale_softmax
        self.tile_m = None if tile_m is None else int(tile_m)
        self.tile_n = None if tile_n is None else int(tile_n)
        self.seq_kv_lens_present = bool(seq_kv_lens_present)
        self.seq_q_lens_present = bool(seq_q_lens_present)
        # THD / varlen: Q/K/V/O/dO and the gradients are PACKED [1, T, H, D] and
        # the per-sequence lengths arrive as tensors at execute.  The declared
        # totals only ever TIGHTEN a buffer-derived token capacity -- they are
        # maxima, so they size the workspace but cannot stand in for the current
        # packed total, which is why the kernels clamp their descriptors on
        # device from cu_*[B].
        self.thd = bool(thd)
        self.max_total_seq_len_q = None if max_total_seq_len_q is None else int(max_total_seq_len_q)
        self.max_total_seq_len_kv = None if max_total_seq_len_kv is None else int(max_total_seq_len_kv)
        # Packed Stats layout: token-major (T, H) -- cuDNN's ragged-Stats recipe
        # -- or head-major (1, QH, T), which is what the FROST forward emits
        # natively.  Both are served; the kernel selects on the compiled fake
        # tensor's static rank, so this only has to reach `compile()`.
        self.thd_stats_token_major = bool(thd_stats_token_major)
        # Head-major only: the caller's declared Stats head stride.  The forward
        # emits a token capacity rounded up to 64, so it is routinely WIDER than
        # the packed total; 0 / None means compact (the packed total itself).
        self.thd_stats_head_stride = 0 if thd_stats_head_stride is None else int(thd_stats_head_stride)

        self.batch_size: Optional[int] = None
        self.s_q_max: Optional[int] = None
        self.s_k_max: Optional[int] = None
        self.h_q: Optional[int] = None
        self.h_kv: Optional[int] = None
        self.head_dim_qk: Optional[int] = None
        self.head_dim_v: Optional[int] = None
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
        seq_q_lens: Optional[torch.Tensor] = None,
        seq_kv_lens: Optional[torch.Tensor] = None,
        sink_tensor: Optional[torch.Tensor] = None,
        dsink_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        dbias_tensor: Optional[torch.Tensor] = None,
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
        self._skv_rounded: Optional[int] = None
        # Kernel-facing head dims per side (QK: Q/K/dQ/dK, V: V/O/dO/dV).
        # Padding by TMA zero-fill
        self.head_dim_qk_padded: Optional[int] = None
        self.head_dim_v_padded: Optional[int] = None
        # name -> baked BSHD (batch, seq, head) strides for each non-compact io tensor
        self._io_strides: dict[str, tuple[int, int, int]] = {}
        # Baked (B, H, S) LSE strides when non-contiguous; None = contiguous
        self._lse_strides: "tuple[int, int, int] | None" = None

    @staticmethod
    def _bshd_physical_ok(desc: TensorDesc) -> bool:
        """True when the logical-BHSD desc sits on compact BSHD storage."""

        b, h, s, d = desc.shape
        return tuple(desc.stride) == (s * h * d, d, h * d, 1)

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        from cudnn.sdpa.graph_analyzer import dense_layout_ok

        self._io_strides = {}
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
        d_v = int(self.v_desc.shape[3])
        self._check_tensor_shape(self.k_desc, (b, h_kv, s_kv, d_qk), name="K")
        self._check_tensor_shape(self.v_desc, (b, h_kv, s_kv, d_v), name="V")
        self._check_tensor_shape(self.o_desc, (b, h_q, s_q, d_v), name="O")
        self._check_tensor_shape(self.do_desc, (b, h_q, s_q, d_v), name="dO")
        self._check_tensor_shape(self.dq_desc, tuple(self.q_desc.shape), name="dQ")
        self._check_tensor_shape(self.dk_desc, tuple(self.k_desc.shape), name="dK")
        self._check_tensor_shape(self.dv_desc, tuple(self.v_desc.shape), name="dV")

        for label, val in (("B", b), ("H_q", h_q), ("H_kv", h_kv), ("S_q", s_q), ("S_kv", s_kv), ("D_QK", d_qk), ("D_V", d_v)):
            self._value_error_if(int(val) <= 0, f"{label} must be > 0; got {val}")
        self._value_error_if(
            h_q % h_kv != 0,
            f"SM120 DSL SDPA backward requires H_q to be a multiple of H_kv (GQA / MQA); got H_q={h_q}, H_kv={h_kv}",
        )
        self._value_error_if(
            d_v > d_qk,
            f"SM120 DSL SDPA backward requires D_QK >= D_V (MLA-style rectangular head dims); got D_QK={d_qk}, D_V={d_v}",
        )
        self._value_error_if(
            d_qk % 8 != 0 or _sm120_padded_head_dim(int(d_qk)) is None,
            f"D_QK ({d_qk}) must be a multiple of 8 and <= {max(_SM120_SUPPORTED_HEAD_DIMS)}",
        )
        self._value_error_if(
            d_v % 8 != 0 or _sm120_padded_head_dim(int(d_v)) is None,
            f"D_V ({d_v}) must be a multiple of 8 and <= {max(_SM120_SUPPORTED_HEAD_DIMS)}",
        )
        self.head_dim_qk_padded, self.head_dim_v_padded = _sm120_padded_head_dims(int(d_qk), int(d_v))
        for desc in (self.q_desc, self.k_desc, self.dq_desc, self.dk_desc, self.v_desc, self.o_desc, self.do_desc, self.dv_desc):
            if self._bshd_physical_ok(desc):
                continue
            b, h, s_, _ = desc.shape
            batch_stride, head_stride, seq_stride, elem_stride = (int(x) for x in desc.stride)
            quantum = 16 // desc.dtype.itemsize
            self._value_error_if(
                elem_stride != 1 or (s_ > 1 and seq_stride % quantum != 0) or (h > 1 and head_stride % quantum != 0) or (b > 1 and batch_stride % quantum != 0),
                f"{desc.name} declares strides the kernel cannot address natively (head dim must be "
                f"innermost-contiguous and batch/seq/head strides 16-byte multiples); got stride {tuple(desc.stride)}",
            )
            self._io_strides[desc.name] = (batch_stride, seq_stride, head_stride)

        self._value_error_if(
            self.stats_desc.ndim != 4 or tuple(self.stats_desc.shape) != (b, h_q, s_q, 1),
            f"stats must be (B, H_q, S_q, 1); got {tuple(self.stats_desc.shape)}",
        )
        self._value_error_if(
            any(st < 0 or (st == 0 and d > 1) for d, st in zip(self.stats_desc.shape, self.stats_desc.stride)),
            f"stats must not broadcast (stride 0 on a size > 1 dim) or have negative strides; got stride {self.stats_desc.stride}",
        )
        self._check_dtype(self.stats_desc, torch.float32, name="stats")
        self._lse_strides = None if self.stats_desc.is_contiguous() else tuple(int(st) for st in self.stats_desc.stride[:3])

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
            self.seq_q_lens_present and not self.seq_kv_lens_present,
            "seq_q_lens_present requires seq_kv_lens_present (per-batch Q lengths are only supported as part of the padding mask)",
        )
        self._value_error_if(
            self.window_size_left is not None and self.window_size_left < 0,
            f"window_size_left must be non-negative, got {self.window_size_left}",
        )
        self._value_error_if(
            self.window_size_right is not None and self.window_size_right < 0,
            f"window_size_right must be non-negative, got {self.window_size_right}",
        )
        self._value_error_if(
            self.window_size_right is not None and not self.is_causal,
            "window_size_right widens the causal diagonal and requires is_causal=True",
        )
        self._value_error_if(
            self.dsink_desc is not None and self.sink_desc is None,
            "dSink output requires a sink logits input",
        )
        for desc in (self.sink_desc, self.dsink_desc):
            if desc is None:
                continue
            self._value_error_if(
                desc.device != self.q_desc.device,
                f"{desc.name} must be on {self.q_desc.device} (with Q); got {desc.device}",
            )
            self._value_error_if(
                tuple(desc.shape) != (1, h_q, 1, 1),
                f"{desc.name} must be (1, H_q, 1, 1) = (1, {h_q}, 1, 1); got {tuple(desc.shape)}",
            )
            self._value_error_if(
                not desc.is_contiguous(),
                f"{desc.name} must be contiguous; got stride {desc.stride}",
            )
            self._check_dtype(desc, torch.float32, name=desc.name)

        self._value_error_if(
            self.dbias_desc is not None and self.bias_desc is None,
            "dBias output requires a bias input",
        )
        for desc in (self.bias_desc, self.dbias_desc):
            if desc is None:
                continue
            self._value_error_if(
                desc.device != self.q_desc.device,
                f"{desc.name} must be on {self.q_desc.device} (with Q); got {desc.device}",
            )
            self._value_error_if(
                tuple(desc.shape) not in ((1, h_q, s_q, s_kv), (b, h_q, s_q, s_kv)),
                f"{desc.name} must be (1|B, H_q, S_q, S_kv) = (1|{b}, {h_q}, {s_q}, {s_kv}); got {tuple(desc.shape)}",
            )
            self._value_error_if(
                not desc.is_contiguous(),
                f"{desc.name} must be contiguous; got stride {desc.stride}",
            )
            self._check_dtype(desc, [self.dtype, torch.float32], name=desc.name)
            # Kernel-side offsets are 32-bit element indices.
            self._value_error_if(
                int(desc.shape[0]) * h_q * s_q * s_kv >= 2**31,
                f"{desc.name} is too large for 32-bit element indexing ({tuple(desc.shape)})",
            )
        if self.bias_desc is not None and self.dbias_desc is not None:
            self._value_error_if(
                tuple(self.dbias_desc.shape) != tuple(self.bias_desc.shape),
                f"dBias must match the bias dims {tuple(self.bias_desc.shape)}; got {tuple(self.dbias_desc.shape)}",
            )
            self._value_error_if(
                self.deterministic and b > 1 and int(self.bias_desc.shape[0]) == 1,
                "deterministic dBias requires a per-batch (B, H_q, S_q, S_kv) bias when B > 1 "
                f"(a broadcast bias reduces over B through unordered atomics); got batch dim 1 with B = {b}",
            )

        self._runtime_error_if(not torch.cuda.is_available(), "CUDA is not available")
        self.compute_capability = torch.cuda.get_device_capability(self.q_desc.device)
        self._runtime_error_if(
            self.compute_capability not in {(12, 0), (12, 1)},
            f"SdpaBwdDslSm120 requires SM120 or SM121, found SM{self.compute_capability[0]}{self.compute_capability[1]}",
        )

        if self.scale_softmax is None:
            self.scale_softmax = 1.0 / math.sqrt(d_qk)

        self.batch_size = int(b)
        self.s_q_max = int(s_q)
        self.s_k_max = int(s_kv)
        self.h_q = int(h_q)
        self.h_kv = int(h_kv)
        self.head_dim_qk = int(d_qk)
        self.head_dim_v = int(d_v)
        # delta - [B, H_q, SQ_r128], dq_accum (relay path only) - [B*SQ_r128*H*D]
        self._sq_rounded = _round_up(self.s_q_max, _SM120_ROW_ROUND)
        # ds_ws (deterministic two kernel path only) - [B, H_q, S_q, SKV_r128]
        self._skv_rounded = _round_up(self.s_k_max, _SM120_ROW_ROUND)
        self.det_2k = self._pick_det_2k()
        self._is_supported = True

        self._logger.debug("check_support completed successfully")
        return True

    def _ds_ws_elems(self) -> int:
        """io-dtype elements of the det_2kernel dS workspace, [B, H_q, S_q, S_kv_r128]."""

        if not self.det_2k:
            return 0
        return self.batch_size * self.h_q * self.s_q_max * self._skv_rounded

    def _pick_det_2k(self) -> bool:
        """Deterministic-mode route: two-kernel dS-workspace split vs the ordered-relay dQ scatter."""

        if not self.deterministic:
            return False
        if (self.head_dim_qk_padded, self.head_dim_v_padded) not in _SM120_DET_2K_HEAD_DIM_PAIRS:
            return False
        # dq2k reads K / writes dQ at the declared head dim (no zero-fill envelope)
        if self.head_dim_qk != self.head_dim_qk_padded:
            return False
        if self.window_size_left is not None or self.seq_kv_lens_present or self.seq_q_lens_present:
            return False
        # dq2k addresses K/dQ as compact BSHD
        if "k" in self._io_strides or "dQ" in self._io_strides:
            return False
        # dq2k's q tile (128 at d_qk <= 128, else 64) must be a multiple of the main kernel's q tile
        if self.q_tile and (128 if self.head_dim_qk_padded <= 128 else 64) % self.q_tile:
            return False
        # The full two-kernel scratch (scratch_workspace_bytes' det_2k branch).
        ws_bytes = (
            ws_align(self.batch_size * self.h_q * self._sq_rounded * 4)
            + ws_align(self.batch_size * self.h_q * self.s_q_max * self._skv_rounded * self.dtype.itemsize)
            + ws_align(self._dbias_accum_elems() * 4)
            + sum(ws_align(elems * self.dtype.itemsize) for elems in self._dkv_ws_elems())
        )
        total_mem = torch.cuda.get_device_properties(self.q_desc.device).total_memory
        return ws_bytes <= total_mem

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
            window_size_right=self.window_size_right,
            deterministic=self.deterministic,
            q_tile=self.q_tile,
            kv_tile=self.kv_tile,
            seq_kv_lens_present=self.seq_kv_lens_present,
            seq_q_lens_present=self.seq_q_lens_present,
            sink_present=self.sink_desc is not None,
            dsink_present=self.dsink_desc is not None,
            det_2kernel=self.det_2k,
            bias_present=self.bias_desc is not None,
            dbias_present=self.dbias_desc is not None,
            bias_is_fp32=self.bias_desc is not None and self.bias_desc.dtype == torch.float32,
            dbias_is_fp32=self.dbias_desc is not None and self.dbias_desc.dtype == torch.float32,
        )
        self._k_mod = _load_sm120_kernel_module(params)
        self._compiled_kernel = self._k_mod.compile(
            compute_capability=self.compute_capability,
            b=self.batch_size,
            qh=self.h_q,
            sq=self.s_q_max,
            skv=self.s_k_max,
            d_qk=self.head_dim_qk,
            kvh=self.h_kv,
            d_v=self.head_dim_v,
            bias_batch=int(self.bias_desc.shape[0]) if self.bias_desc is not None else 0,
            lse_strides=self._lse_strides,
            q_strides=self._io_strides.get("q"),
            k_strides=self._io_strides.get("k"),
            v_strides=self._io_strides.get("v"),
            o_strides=self._io_strides.get("o"),
            do_strides=self._io_strides.get("dO"),
            dq_strides=self._io_strides.get("dQ"),
            dk_strides=self._io_strides.get("dK"),
            dv_strides=self._io_strides.get("dV"),
        )
        from .prepared import build_sm120_spec

        self._prepared = build_sm120_spec(self)
        self._logger.debug("compile completed")

    def _dq_sem_len(self) -> int:
        """Element count of the dq_sem relay-counter buffer (int32)."""

        return self.batch_size * self.h_q * _round_up(self.s_q_max, _SM120_MIN_Q_TILE) // _SM120_MIN_Q_TILE

    def _dkv_ws_elems(self) -> tuple[int, int]:
        """io-dtype elements of the GQA partials buffers (dk_ws, dv_ws);
        (0, 0) for MHA, where they alias the dk/dv outputs."""

        if self.h_q == self.h_kv:
            return (0, 0)
        rows = self.batch_size * self.s_k_max * self.h_q
        return (rows * self.head_dim_qk_padded, rows * self.head_dim_v_padded)

    def _dbias_accum_elems(self) -> int:
        """fp32 elements of the dBias accumulator ([1|B, H_q, S_q, S_kv]);
        0 when no dBias output is requested or output dtype = f32."""

        if self.dbias_desc is None or self.dbias_desc.dtype == torch.float32:
            return 0
        return int(self.dbias_desc.shape[0]) * self.h_q * self.s_q_max * self.s_k_max

    def scratch_workspace_bytes(self) -> int:
        """delta (fp32 [B, H, SQ_r128]) + the dQ scratch — relay path:
        dq_accum (fp32 flat [B*SQ_r128*H*D_QK]) + dq_sem (int32 flat
        [B*H*ceil(SQ/32)], relay counters); det_2kernel path: ds_ws (io
        [B, H_q, SQ, SKV_r128]) — + dbias_accum (fp32 [1|B, H_q, S_q, S_kv],
        io-dtype dBias output only; an fp32 dBias accumulates in place)
        + dk_ws/dv_ws (io [B, SKV, H_q, D_QK] / [B, SKV, H_q, D_V],
        per-q-head partials, GQA only)."""

        self._ensure_support_checked()
        delta_bytes = ws_align(self.batch_size * self.h_q * self._sq_rounded * 4)
        if self.det_2k:
            dq_scratch_bytes = ws_align(self._ds_ws_elems() * self.dtype.itemsize)
        else:
            dq_scratch_bytes = ws_align(self.batch_size * self._sq_rounded * self.h_q * self.head_dim_qk_padded * 4) + ws_align(self._dq_sem_len() * 4)
        dbias_bytes = ws_align(self._dbias_accum_elems() * 4)
        dkv_ws_bytes = sum(ws_align(elems * self.dtype.itemsize) for elems in self._dkv_ws_elems())
        return delta_bytes + dq_scratch_bytes + dbias_bytes + dkv_ws_bytes

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
        seq_q_lens: Optional[torch.Tensor] = None,
        seq_kv_lens: Optional[torch.Tensor] = None,
        sink_tensor: Optional[torch.Tensor] = None,
        dsink_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        dbias_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        """Execute tensors matching the compiled specialization."""

        if self._compiled_kernel is None:
            raise RuntimeError("SdpaBwdDslSm120 kernel is not compiled")

        from cudnn.sdpa.fwd.prepared import facts_of_tensor
        from .prepared import ROLES, execute

        spec = self._prepared
        ws = facts_of_tensor(workspace)
        if ws is None or ws.dtype != "uint8" or not ws.contiguous or ws.span < spec.workspace_bytes or ws.device != (2, spec.device_index):
            raise ValueError(f"sdpa_bwd_sm120 requires {spec.workspace_bytes} bytes of contiguous uint8 workspace on CUDA device {spec.device_index}")
        if current_stream is None:
            current_stream = torch.cuda.current_stream(q_tensor.device).cuda_stream
        facts = dict(
            zip(
                ROLES,
                map(
                    facts_of_tensor,
                    (
                        q_tensor,
                        k_tensor,
                        v_tensor,
                        o_tensor,
                        do_tensor,
                        stats_tensor,
                        dq_tensor,
                        dk_tensor,
                        dv_tensor,
                        seq_q_lens,
                        seq_kv_lens,
                        sink_tensor,
                        dsink_tensor,
                        bias_tensor,
                        dbias_tensor,
                    ),
                ),
            )
        )
        stats = facts["stats"]
        if stats is not None:
            elements = self.batch_size * self.h_q * self.s_q_max
            self._value_error_if(stats.numel != elements, f"stats_tensor must have B*H_q*S_q = {elements} elements; got {stats.numel}")
            self._value_error_if(
                self._lse_strides is None and not stats.contiguous,
                "stats_tensor must be contiguous (the kernel was compiled for a contiguous LSE layout)",
            )
        geometry = tuple((op.shape, op.strides) if op is not None and i < 9 else None for i, op in enumerate(spec.operands))
        if self._lse_strides is None:
            # The established standalone contract accepts any contiguous Stats
            # view with the right element count, checked above.
            geometry = (*geometry[:5], None, *geometry[6:])
        execute(spec, facts, ws.ptr, int(current_stream), scale=scale_softmax, geometry=geometry)


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
    window_size_right: Optional[int] = None,
    deterministic: bool = False,
    scale_softmax: Optional[float] = None,
    seq_q_lens: Optional[torch.Tensor] = None,
    seq_kv_lens: Optional[torch.Tensor] = None,
    sink_token: Optional[torch.Tensor] = None,
    bias_tensor: Optional[torch.Tensor] = None,
) -> TupleDict:
    """Run SM120 SDPA backward and return ``TupleDict(dq_tensor=..., dk_tensor=..., dv_tensor=...)``."""

    dq_tensor = torch.empty_strided(q_tensor.shape, q_tensor.stride(), dtype=q_tensor.dtype, device=q_tensor.device)
    dk_tensor = torch.empty_strided(k_tensor.shape, k_tensor.stride(), dtype=k_tensor.dtype, device=k_tensor.device)
    dv_tensor = torch.empty_strided(v_tensor.shape, v_tensor.stride(), dtype=v_tensor.dtype, device=v_tensor.device)
    dsink_tensor = torch.empty_like(sink_token) if sink_token is not None else None
    dbias_tensor = torch.empty_like(bias_tensor, dtype=torch.float32) if bias_tensor is not None else None

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
        window_size_right,
        bool(deterministic),
        scale_softmax,
        seq_q_lens is not None,
        seq_kv_lens is not None,
        _tensor_signature(sink_token) if sink_token is not None else None,
        _tensor_signature(bias_tensor) if bias_tensor is not None else None,
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
            sample_sink=sink_token,
            sample_dsink=dsink_tensor,
            sample_bias=bias_tensor,
            sample_dbias=dbias_tensor,
            is_causal=is_causal,
            causal_bottom_right=causal_bottom_right,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            deterministic=deterministic,
            scale_softmax=scale_softmax,
            seq_kv_lens_present=seq_kv_lens is not None,
            seq_q_lens_present=seq_q_lens is not None,
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
        seq_q_lens=seq_q_lens,
        seq_kv_lens=seq_kv_lens,
        sink_tensor=sink_token,
        dsink_tensor=dsink_tensor,
        bias_tensor=bias_tensor,
        dbias_tensor=dbias_tensor,
    )
    out = TupleDict(dq_tensor=dq_tensor, dk_tensor=dk_tensor, dv_tensor=dv_tensor)
    if dbias_tensor is not None:
        out["dbias_tensor"] = dbias_tensor
    if dsink_tensor is not None:
        out["dsink_tensor"] = dsink_tensor
    return out


# =============================================================================
# SM80 (A100) backward — SdpaBwdDslSm80 + the sdpa_bwd_wrapper_sm80 entry
# point. Ports the pre-TemplateParams ``SdpabwdSm80`` (bwd/api.py, deleted)
# onto the shared SdpaBwdDsl adapter contract, mirroring the forward port
# (#682): one lowering function (``lower_dsl_bwd``) now drives both backward
# cells. The kernels stay self-caching until the TemplateParams conversion
# (the #689 analogue; issue #604's sym_int THD extents land there).
# =============================================================================

_SM80_BWD_KERNEL_MOD = {}

_SM80_BWD_FLAVOR_DIMS = {
    name: (cfg.D_QK, cfg.D_V)
    for name, cfg in (
        ("gptoss", _fwd_config_sm80.GPTOSS_CFG),
        ("llama", _fwd_config_sm80.LLAMA_CFG),
        ("dsv3", _fwd_config_sm80.DSV3_CFG),
        ("qwen", _fwd_config_sm80.QWEN_CFG),
    )
}
_SM80_BWD_SUPPORTED_FLAVORS = ("gptoss", "llama", "dsv3", "qwen")


def _sm80_bwd_kernel_mod(key: str = "d64"):
    """Lazily import + cache the dedicated d=64 SM80 BPROP kernel module.

    ``"d64"`` (the only key) is the plain-dense d=64 MHA perf variant (~2x
    faster on A100); it supports NO features — its ``backward(**_ignored)``
    silently swallows every feature kwarg, so callers must never rely on the
    signature filter and only select it through
    :func:`_sm80_d64_fast_path_eligible`.  The GENERIC kernel
    (``sm80/bprop_f16``) is a TemplateParams module loaded per-specialization
    via :func:`_load_sm80_bwd_module` instead.
    """
    assert key == "d64", f"generic SM80 bwd kernels load via _load_sm80_bwd_module; got {key!r}"
    if key not in _SM80_BWD_KERNEL_MOD:
        from .kernels.sm80 import bprop_d64_f16 as _mod

        _SM80_BWD_KERNEL_MOD[key] = _mod
    return _SM80_BWD_KERNEL_MOD[key]


def _sm80_d64_fast_path_eligible(*, d_qk, d_v, h_q, h_kv, s_q, s_kv, mask_token, right_bound, causal_bottom_right, bw_kwargs) -> bool:
    """Whether the dedicated d=64 kernel can serve this call EXACTLY.

    The perf variant computes a plain dense MHA backward and nothing else;
    every condition here guards a feature it would silently ignore.
    """
    d64 = _sm80_bwd_kernel_mod("d64")
    if (d_qk, d_v) != (64, 64) or h_q != h_kv:
        return False
    if s_q % d64.M_BLOCK != 0 or s_kv % d64.N_BLOCK != 0:
        return False
    if mask_token != "none" or right_bound != 0 or causal_bottom_right:
        return False
    for feature in ("seq_kv_lens", "seq_len_q", "bias", "sinks", "rope_freqs"):
        if bw_kwargs.get(feature) is not None:
            return False
    if bw_kwargs.get("deterministic"):
        return False
    return True


def _sm80_bwd_pick_flavor(d_qk: int, d_v: int) -> str:
    """Smallest BPROP flavor whose ``(D_QK, D_V)`` envelope covers
    ``(d_qk, d_v)`` (fdqk >= d_qk and fdv >= d_v); the user's heads are padded
    up to the flavor dim.  The kernel supports d_qk != d_v but requires the
    (padded) d_qk >= d_v — the flavor list guarantees this (every flavor has
    fdqk >= fdv, and a d_qk < d_v case lands on an equal-d flavor after pad)."""
    for flavor in _SM80_BWD_SUPPORTED_FLAVORS:
        fdqk, fdv = _SM80_BWD_FLAVOR_DIMS[flavor]
        if d_qk == fdqk and d_v == fdv:
            return flavor
    for flavor in _SM80_BWD_SUPPORTED_FLAVORS:
        fdqk, fdv = _SM80_BWD_FLAVOR_DIMS[flavor]
        if d_qk <= fdqk and d_v <= fdv:
            return flavor
    raise ValueError(f"SM80 BPROP: no flavor envelope covers (D_QK={d_qk}, D_V={d_v}); " f"supported: {_SM80_BWD_FLAVOR_DIMS}.")


@lru_cache(maxsize=128)
def _sm80_thd_plan(
    n_seq,
    h_q,
    h_kv,
    d_qk,
    d_v,
    t_q,
    t_kv,
    max_sq,
    max_skv,
    dtype,
    device,
    is_causal,
    window_size,
    bottom_right,
    has_sink,
    deterministic,
    stats_token_major=False,
):
    """Cache immutable wrapper plans without retaining tensors or runtime pointers.

    Capacities and bounds size each plan's workspace; the prepared compiler
    receives them as runtime scalars and reuses its artifact across plans.
    """

    def desc(h, d, s):
        return TensorDesc(dtype, (n_seq, h, s, d), (s * h * d, d, h * d, 1), (3, 1, 2, 0), device)

    q, k, v, o = desc(h_q, d_qk, t_q), desc(h_kv, d_qk, t_kv), desc(h_kv, d_v, t_kv), desc(h_q, d_v, t_q)
    stats = TensorDesc(torch.float32, (n_seq, h_q, t_q), (h_q * t_q, t_q, 1), (2, 1, 0), device)
    sink = TensorDesc(torch.float32, (h_q,), (1,), (0,), device) if has_sink else None
    wl, wr = window_size
    api = SdpaBwdDslSm80(
        q,
        k,
        v,
        o,
        o,
        stats,
        q,
        k,
        v,
        sample_sink=sink,
        sample_dsink=sink,
        is_causal=is_causal,
        causal_bottom_right=bottom_right,
        window_size_left=wl,
        window_size_right=wr,
        deterministic=deterministic,
        thd=True,
        max_total_seq_len_q=t_q,
        max_total_seq_len_kv=t_kv,
        thd_stats_token_major=stats_token_major,
    )
    assert api.check_support(), "Unsupported configuration"
    # Wrapper buffers may have capacity beyond B * max_sequence_length.
    # Preserve their physical capacity/Stats pitch while separately bounding
    # the launch grid and deterministic counters from the caller's hints.
    api._thd_launch_bounds = (max_sq, max_skv)
    api._initialize_packed_outputs = True
    api.compile()
    return api


def _sm80_thd_backward(
    q, k, v, o, do, lse, *, cu_q, cu_k, scale_softmax, is_causal, window_size, causal_bottom_right, sinks=None, deterministic=False, max_s_kv=None, max_s_q=None
):
    """Allocate wrapper outputs and bind the same prepared chain as graph THD.

    Inputs and gradients are packed [1,T,H,D], Stats is [1,H,T_q], and
    cu_q/cu_k contain B+1 prefixes. Missing sequence bounds use the packed
    capacities, so no device-to-host length read is needed. Smaller caller
    bounds must cover every sequence; validating their contents would sync.
    """
    d_qk, d_v, h_q, h_kv = q.shape[-1], v.shape[-1], q.shape[2], k.shape[2]
    fdqk, fdv = _SM80_BWD_FLAVOR_DIMS[_sm80_bwd_pick_flavor(d_qk, d_v)]
    # Resolve from the user's width before envelope padding (e.g. D=96).
    if scale_softmax is None:
        scale_softmax = 1.0 / math.sqrt(d_qk)
    n_seq = cu_q.numel() - 1
    assert n_seq > 0 and cu_k is not None and cu_k.numel() == n_seq + 1, "cu_seqlens_q / cu_seqlens_k length mismatch"
    t_q, t_kv, dev = q.shape[1], k.shape[1], q.device
    for label, hint in (("max_s_q", max_s_q), ("max_s_kv", max_s_kv)):
        if hint is not None:
            assert int(hint) > 0, f"{label} must be > 0; got {hint}"

    if d_qk < fdqk or d_v < fdv or any(not t.is_contiguous() for t in (q, k, v, o, do)):
        from cudnn.sdpa.packed_copy_sm80 import copy_packed_half

        q, k, v, o, do = copy_packed_half((q, k, v, o, do), (fdqk, fdqk, fdv, fdv, fdv), (True, False, False, True, True), compact=True)

    # A zero-capacity wrapper operand gets one never-live row. Device prefixes
    # still contain the actual totals; no adapter-side degenerate computation.
    def packed(t):
        return t if t.shape[1] else torch.empty((1, 1, *t.shape[2:]), dtype=t.dtype, device=dev)

    q, k, v, o, do = map(packed, (q, k, v, o, do))
    tq_cap, tkv_cap = max(t_q, 1), max(t_kv, 1)
    stats_token_major = False
    if t_q:
        lse_t = lse if lse.dtype == torch.float32 and lse.device == dev else lse.to(dtype=torch.float32, device=dev)
        if tuple(lse_t.shape) == (1, h_q, t_q) and lse_t.stride(1) == 1 and lse_t.stride(2) == h_q:
            # Reuse the graph chain's existing compact token-major Stats ABI.
            # This is a view of the caller's storage, not a layout conversion.
            lse_t = lse_t[0].transpose(0, 1)
            stats_token_major = True
        else:
            lse_t = lse_t.contiguous()
    else:
        lse_t = torch.empty((1, h_q, 1), dtype=torch.float32, device=dev)
    cu_q_t = cu_q.to(dtype=torch.int32, device=dev).contiguous()
    cu_k_t = cu_k.to(dtype=torch.int32, device=dev).contiguous()
    sinks_t = sinks.to(dtype=torch.float32, device=dev).reshape(h_q).contiguous() if sinks is not None else None
    # The cast/fold only writes live rows. Keep the established zeroed dQ and
    # GQA output tails; MHA dK/dV remain direct kernel outputs.
    # The wrapper specialization folds this into workspace initialization.
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    dsink = torch.empty(h_q, dtype=torch.float32, device=dev) if sinks is not None else None
    wl, wr = window_size
    wl = None if wl is None or wl < 0 else int(wl)
    wr = None if wr is None or wr < 0 else int(wr)
    api = _sm80_thd_plan(
        n_seq,
        h_q,
        h_kv,
        fdqk,
        fdv,
        tq_cap,
        tkv_cap,
        min(int(max_s_q), tq_cap) if max_s_q is not None else tq_cap,
        min(int(max_s_kv), tkv_cap) if max_s_kv is not None else tkv_cap,
        q.dtype,
        dev,
        bool(is_causal),
        (wl, wr),
        bool(causal_bottom_right) and (bool(is_causal) or wl is not None),
        sinks is not None,
        bool(deterministic),
        stats_token_major,
    )
    workspace = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device=dev)
    api.execute(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        o.transpose(1, 2),
        do.transpose(1, 2),
        lse_t,
        dq.transpose(1, 2),
        dk.transpose(1, 2),
        dv.transpose(1, 2),
        scale_softmax=scale_softmax,
        workspace=workspace,
        seq_q_lens=cu_q_t,
        seq_kv_lens=cu_k_t,
        sink_tensor=sinks_t,
        dsink_tensor=dsink,
    )
    dQ_k, dK_k, dV_k = dq[:, :t_q], dk[:, :t_kv], dv[:, :t_kv]
    if d_qk < fdqk or d_v < fdv:
        dQ_k, dK_k, dV_k = copy_packed_half((dQ_k, dK_k, dV_k), (d_qk, d_qk, d_v), (True, False, False))
    out = TupleDict(dq_tensor=dQ_k, dk_tensor=dK_k, dv_tensor=dV_k)
    if dsink is not None:
        out["dsink_tensor"] = dsink
    return out


# ---------------------------------------------------------------------------
# Functional wrapper (mirrors the forward surface).
# ---------------------------------------------------------------------------
_cache_of_objects: dict = {}


_SM80_BWD_KERNEL_FILE = "sm80/bprop_f16.py"
# The shared tile_dsl scheduler vocabulary maps identity onto the bwd grid
# decode (NATURAL == plain 3-D == 0, LPT == kv-major == 1).
from cudnn.frost.tile_dsl.constants import SCHED_LPT_L2 as _BWD_SCHED_LPT_L2  # noqa: E402
from cudnn.frost.tile_dsl.constants import SCHED_NATURAL as _BWD_SCHED_NATURAL  # noqa: E402


def _sm80_bwd_sched_policy(*, is_causal: bool, deterministic: bool, thd: bool = False) -> int:
    """Grid order for the SM80 backward (a ``tile_dsl.constants.SCHED_*``).

    Under THD the grid is the plain 3-D (kv_tile, head, sequence) decode over
    the envelope's kv-tile count -- the kv-major remaps assume the dense grid,
    so a packed plan takes the plain order whatever the mask.

    Causal takes the kv-major LPT order WITHIN L2-sized head groups.  The plain
    kv-major LPT over every head at once spreads a head's kv-tiles across the
    whole grid, so its Q / dO / dQ tiles fall out of L2 between visits and the
    dQ atomics turn into DRAM read-modify-writes (1.4x slower than the plain
    3-D grid at 64x8 heads, S=4K); the plain grid in turn leaves a one-CTA-long
    tail that costs 10-17% on small grids.  Grouping keeps both.  Non-causal
    work is uniform per kv-tile, so the plain 3-D grid stays; the deterministic
    relay REQUIRES the plain decode (kv_tile == blockIdx.x).
    """
    if thd:
        return _BWD_SCHED_NATURAL
    return _BWD_SCHED_LPT_L2 if (is_causal and not deterministic) else _BWD_SCHED_NATURAL


def _load_sm80_bwd_module(params):
    """Load one uniquely named backward kernel module per parameter set."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", _SM80_BWD_KERNEL_FILE)
    return load_template(path, params, tag="sdpa_bwd_sm80")


class SdpaBwdDslSm80(SdpaBwdDsl):
    """SM80 (A100) SDPA backward through prepared CuTe-DSL launch chains.

    Follows the SM120 adapter lifecycle (check_support → compile → execute) on
    top of immutable template specializations and per-call pointer bindings.
    Packed capacities remain dynamic host arguments. SM80-only operands (bias →
    dBias, RoPE) arrive as extra optional keywords, as the ``SdpaBwdDsl``
    contract permits.

    Layouts: any dense layout with the head dim innermost-contiguous is
    served. Native flavor widths with vector-aligned outer strides compile
    a pointer chain that reads/writes each declared layout directly. Other
    dense layouts and widths retain carved staging; strided Stats are read
    natively in both paths. Prepared execute requires caller workspace, which
    the graph and convenience wrapper provide outside the launch path.
    """

    def __init__(
        self,
        *args,
        has_bias: bool = False,
        bias_is_fp32: bool = True,
        bias_batch: int = 1,
        has_rope: bool = False,
        rope_max_s: int = 0,
        thd: bool = False,
        max_total_seq_len_q: Optional[int] = None,
        max_total_seq_len_kv: Optional[int] = None,
        thd_stats_token_major: bool = False,
        thd_stats_head_stride: Optional[int] = None,
        **kwargs,
    ) -> None:
        # SM80-only plan-time facts (scratch sizing + template identity); the
        # base contract carries everything else.  The THD keywords are base
        # parameters, spelled out here because the lowering forwards a keyword
        # only when it appears in THIS signature (a bare **kwargs hides them).
        self._has_bias = bool(has_bias)
        self._bias_is_fp32 = bool(bias_is_fp32)
        self._bias_batch = int(bias_batch)
        self._has_rope = bool(has_rope)
        self._rope_max_s = int(rope_max_s)
        super().__init__(
            *args,
            thd=thd,
            max_total_seq_len_q=max_total_seq_len_q,
            max_total_seq_len_kv=max_total_seq_len_kv,
            thd_stats_token_major=thd_stats_token_major,
            thd_stats_head_stride=thd_stats_head_stride,
            **kwargs,
        )

    def _initialize_implementation(self) -> None:
        self.flavor: Optional[str] = None
        self.flavor_d_qk: Optional[int] = None
        self.flavor_d_v: Optional[int] = None
        self.mask_token: Optional[str] = None
        self.swa_window_runtime: int = 0
        self.right_bound_runtime: int = 0
        self._lse_stride: "Optional[tuple[int, int, int]]" = None
        self._prepared = None
        self._native_pointer = False
        self._staged_layout = None
        self._staged_prepared = None
        # THD (packed) plan-time state: token capacities the views bind at,
        # the Stats packing, and the compiled lengths -> cu_seqlens setup launch.
        self._t_q_cap: int = 0
        self._t_kv_cap: int = 0
        self._thd_lse_token_major: bool = False
        self._thd_lse_head_stride: int = 0
        self._thd_token_strides: dict = {}  # port role -> the caller's packed token stride (plan-time)
        self._thd_head_strides: dict = {}  # port role -> the caller's head stride (plan-time; D when compact)

    @staticmethod
    def _thd_total(capacity: int, declared: Optional[int]) -> int:
        """Token capacity tightened by the declared packed total (always a MIN,
        so a stale declaration cannot push a view past the caller's buffer)."""
        return capacity if declared is None else min(capacity, max(int(declared), 0))

    @property
    def thd_total_q(self) -> Optional[int]:
        """Packed Q-token extent the lowering binds the Q/O/dO/dQ views at
        (``min(B * S_max, max_total_seq_len_q)``), or None when not THD."""
        return self._t_q_cap if self.thd else None

    @property
    def thd_total_kv(self) -> Optional[int]:
        """Packed KV-token extent; see :attr:`thd_total_q`."""
        return self._t_kv_cap if self.thd else None

    @staticmethod
    def _packed_bshd(desc: TensorDesc) -> bool:
        """True when a logical-BHSD desc sits on PACKED BSHD rows the kernels can
        bind directly: element stride 1, head stride ``>= D`` and token stride
        ``>= H * head_stride``, each a multiple of 8 fp16/bf16 elements so every
        head base stays 16-byte aligned (the cp.async loads move 16-byte
        chunks).  Neither need be compact: the kernels address a row as
        ``token * token_stride + head * head_stride`` with the port's own
        plan-time strides, so a view into a wider per-token record (a K/V slice
        of an interleaved ``[T, 2, H, D]`` buffer) or a head-interleaved record
        is served at those strides.  The
        batch stride is not consulted -- a ragged port's sequences start at the
        ragged offsets, and the packed view rebuilds that axis from the token
        extent.  The token stride is always checked (the packed view walks every
        token with it, even when the envelope S_max is 1); the head and element
        strides wildcard on a size-1 extent (the analyzer's convention)."""
        _, h, _, d = (int(x) for x in desc.shape)
        st = tuple(int(x) for x in desc.stride)
        hs = st[1] if h > 1 else d
        head_ok = h == 1 or (hs >= d and hs % 8 == 0)
        return head_ok and st[2] >= h * hs and st[2] % 8 == 0 and (d == 1 or st[3] == 1)

    def _checked_lse_view(self, lse_tensor: torch.Tensor) -> torch.Tensor:
        """Validate a caller-provided Stats/LSE buffer and return the
        kernels' (B, H_q, S_q) READ view.

        The logical contract is exactly ``B*H_q*S_q`` fp32 elements.  With a
        strided plan (``_lse_stride``), rebuild the DECLARED layout over the
        caller's storage — the kernels' loads were compiled against exactly
        those strides; a contiguous plan requires a contiguous runtime buffer
        (the compiled packed layout would misread anything else).
        """
        self._value_error_if(
            lse_tensor.device != self.stats_desc.device,
            f"stats must be on the plan's device {self.stats_desc.device}; got {lse_tensor.device} (the kernel binds this pointer directly)",
        )
        self._value_error_if(lse_tensor.dtype != torch.float32, f"stats must be float32; got {lse_tensor.dtype}")
        expected = self.batch_size * self.h_q * self.s_q_max
        self._value_error_if(
            lse_tensor.numel() != expected,
            f"stats must have B*H_q*S_q = {expected} elements; got {lse_tensor.numel()}",
        )
        shape = (self.batch_size, self.h_q, self.s_q_max)
        stride = self._lse_stride
        if stride is None:
            self._value_error_if(not lse_tensor.is_contiguous(), "stats must be contiguous (the plan declared a packed LSE layout)")
            return lse_tensor.view(shape)
        if tuple(lse_tensor.shape) == shape and tuple(lse_tensor.stride()) == stride:
            return lse_tensor
        try:
            return lse_tensor.as_strided(shape, stride, lse_tensor.storage_offset())
        except RuntimeError as exc:
            raise ValueError(
                f"stats backing storage is too small for declared shape {shape}, stride {stride}, and storage_offset {lse_tensor.storage_offset()}"
            ) from exc

    # ------------------------------------------------------------------
    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")
        from cudnn.frost.buffers import cutedsl_requirement_error

        requirement = cutedsl_requirement_error("SdpaBwdDslSm80")
        self._not_implemented_error_if(requirement is not None, requirement)

        for desc in (self.q_desc, self.k_desc, self.v_desc, self.o_desc, self.do_desc):
            self._value_error_if(desc.ndim != 4, f"{desc.name} must be rank-4 (B, H, S, D); got {desc.ndim}")

        b, h_qo, s_qo, d_qk = self.q_desc.shape
        _, h_kv, s_kv, _ = self.k_desc.shape
        _, _, _, d_v = self.v_desc.shape

        self._check_tensor_shape(self.q_desc, (b, h_qo, s_qo, d_qk), name="Q")
        self._check_tensor_shape(self.k_desc, (b, h_kv, s_kv, d_qk), name="K")
        self._check_tensor_shape(self.v_desc, (b, h_kv, s_kv, d_v), name="V")
        self._check_tensor_shape(self.o_desc, (b, h_qo, s_qo, d_v), name="O")
        self._check_tensor_shape(self.do_desc, (b, h_qo, s_qo, d_v), name="dO")
        self._check_tensor_shape(self.dq_desc, (b, h_qo, s_qo, d_qk), name="dQ")
        self._check_tensor_shape(self.dk_desc, (b, h_kv, s_kv, d_qk), name="dK")
        self._check_tensor_shape(self.dv_desc, (b, h_kv, s_kv, d_v), name="dV")

        for label, val in (("B", b), ("H_q", h_qo), ("H_kv", h_kv), ("S_q", s_qo), ("S_kv", s_kv), ("D_QK", d_qk), ("D_V", d_v)):
            self._value_error_if(int(val) <= 0, f"{label} must be > 0; got {val}")
        self._value_error_if(h_qo % h_kv != 0, f"H_q ({h_qo}) must be divisible by H_kv ({h_kv}) for GQA / MQA")

        # The kernel supports d_qk != d_v (split sub-groups) but requires
        # d_qk >= d_v; head dims inside a flavor envelope pad host-side.
        self._value_error_if(d_qk < d_v, f"SM80 BPROP requires D_QK >= D_V; got D_QK={d_qk}, D_V={d_v}")
        max_dqk = max(fdqk for fdqk, _ in _SM80_BWD_FLAVOR_DIMS.values())
        max_dv = max(fdv for _, fdv in _SM80_BWD_FLAVOR_DIMS.values())
        self._value_error_if(
            d_qk > max_dqk or d_v > max_dv,
            f"SM80 BPROP: head dim (D_QK={d_qk}, D_V={d_v}) exceeds supported " f"envelope (D_QK<={max_dqk}, D_V<={max_dv}); larger heads not yet ported.",
        )

        self.dtype = self._check_dtype(self.q_desc, [torch.float16, torch.bfloat16], name="Q")
        for desc in (self.k_desc, self.v_desc, self.o_desc, self.do_desc, self.dq_desc, self.dk_desc, self.dv_desc):
            self._check_dtype(desc, self.dtype, name=desc.name, extra_error_msg=f"{desc.name} must match Q dtype (FP16/BF16)")
        self._check_dtype(self.stats_desc, torch.float32, name="stats")
        stats_shape = tuple(self.stats_desc.shape)
        self._value_error_if(
            stats_shape not in ((b, h_qo, s_qo), (b, h_qo, s_qo, 1)),
            f"stats must be (B, H_q, S_q[, 1]) = ({b}, {h_qo}, {s_qo}[, 1]); got {stats_shape}",
        )
        # Any stats layout is served NATIVELY: the kernels' LSE loads are
        # stride-aware, compiled against the DECLARED (B, H_q, S_q) strides
        # (the trailing size-1 dim of a rank-4 Stats contributes no offset).
        # Contiguous stats keep the packed compact fake (byte-identical).
        self._lse_stride = None if self.stats_desc.is_contiguous() else tuple(int(st) for st in self.stats_desc.stride[:3])

        if self.thd:
            # Packed (THD / ragged) plan.  The descs are the ENVELOPE (B, H,
            # S_max, D); the packed buffers appear at execute as [1, T, H, D]
            # views (the lowering's _thd_view), bound at the capacities below.
            self._value_error_if(
                self.seq_kv_lens_present or self.seq_q_lens_present,
                "SM80 bwd THD: the per-batch lengths become device cu_seqlens; a compiled padding mask is dense-only",
            )
            # The declared totals are REQUIRED because scratch_workspace_bytes()
            # is a build-time function: the fp32 dQ accumulator, the per-query-
            # head dK/dV partials and do_dot are all sized from the packed token
            # totals before any buffer exists.  Undeclared, the capacity is
            # B * S_max -- far more tokens than a packed buffer holds.
            self._value_error_if(
                self.max_total_seq_len_q is None or self.max_total_seq_len_kv is None,
                "SM80 bwd THD: max_total_seq_len_q and max_total_seq_len_kv must be declared "
                "(the packed dQ accumulator, dK/dV partials and do_dot scratch are sized from the token totals at build time)",
            )
            self._value_error_if(self._has_bias, "SM80 bwd THD: bias / dBias is dense-only (a packed graph has no [B, H, S_q, S_kv] bias)")
            self._value_error_if(self._has_rope, "SM80 bwd THD: RoPE is dense-only")
            # No staging leg: the kernels bind the caller's packed rows directly,
            # each port at its own plan-time token stride (the compiled fake
            # carries it).  Anything the row arithmetic cannot express -- a
            # head or token stride below its extent or off 16-byte alignment --
            # is a decline here, not a silent mis-bind.
            self._thd_token_strides = {}
            self._thd_head_strides = {}
            for role, desc in (
                ("q", self.q_desc),
                ("k", self.k_desc),
                ("v", self.v_desc),
                ("o", self.o_desc),
                ("do", self.do_desc),
                ("dq", self.dq_desc),
                ("dk", self.dk_desc),
                ("dv", self.dv_desc),
            ):
                self._value_error_if(
                    not self._packed_bshd(desc),
                    f"SM80 bwd THD: {desc.name} must be packed BSHD rows (element stride 1, head stride >= D and "
                    f"token stride >= H * head stride, each a multiple of 8 elements); got {tuple(desc.stride)} (the packed path has no staging copy)",
                )
                self._thd_token_strides[role] = int(desc.stride[2])
                self._thd_head_strides[role] = int(desc.stride[1]) if int(desc.shape[1]) > 1 else int(desc.shape[3])
            self._t_q_cap = self._thd_total(int(b) * int(s_qo), self.max_total_seq_len_q)
            self._t_kv_cap = self._thd_total(int(b) * int(s_kv), self.max_total_seq_len_kv)
            self._value_error_if(self._t_q_cap <= 0 or self._t_kv_cap <= 0, "SM80 bwd THD: the packed token capacities must be > 0")
            # Stats packing (Rule S1), as the lowering classified it from the
            # ragged declaration: token-major (T, H) or head-major (1, H,
            # head_stride), head_stride 0 == compact == the Q-token capacity.
            # The two are exclusive (Rule 1: overlapping optional declarations
            # are validated as a set).
            self._value_error_if(
                bool(self.thd_stats_token_major) and bool(self.thd_stats_head_stride),
                "SM80 bwd THD: thd_stats_head_stride is head-major-only (token-major (T, H) Stats is compact)",
            )
            self._thd_lse_token_major = bool(self.thd_stats_token_major)
            self._thd_lse_head_stride = 0 if self._thd_lse_token_major else int(self.thd_stats_head_stride or 0)
            # A head stride shorter than the bound extent puts the later heads'
            # rows past the buffer (the kernel reads [0, h, row] at that stride
            # for every row < t_q_cap).
            self._value_error_if(
                bool(self._thd_lse_head_stride) and self._thd_lse_head_stride < self._t_q_cap,
                f"SM80 bwd THD: Stats head stride {self._thd_lse_head_stride} must cover the packed token capacity {self._t_q_cap}",
            )
            # The envelope Stats desc describes the packing, not a dense layout.
            self._lse_stride = None

        self._value_error_if(not torch.cuda.is_available(), "CUDA must be available for SM80 BPROP")
        # Plan-time device parity: the kernels bind the Stats pointer directly
        # (native strided reads), and execute() validates the runtime LSE
        # against stats_desc.device — so a host-side or cross-GPU Stats
        # DECLARATION must be rejected here, before it can anchor that check.
        self._value_error_if(
            self.stats_desc.device != self.q_desc.device,
            f"stats must be on Q's device {self.q_desc.device}; got {self.stats_desc.device}",
        )
        device = self.q_desc.device
        major, minor = torch.cuda.get_device_capability(device)
        self._value_error_if((major, minor) != (8, 0), f"SdpaBwdDslSm80 requires SM80 (A100); found SM{major}{minor} on {device}")

        self._value_error_if(
            self.tile_m is not None or self.tile_n is not None,
            "SM80 BPROP wires no tile knobs; tile_m/tile_n must be unset",
        )

        self.flavor = _sm80_bwd_pick_flavor(d_qk, d_v)
        self.flavor_d_qk, self.flavor_d_v = _SM80_BWD_FLAVOR_DIMS[self.flavor]

        # ---- mask token (same resolution as the forward adapter) ----------
        swa_left = -1 if self.window_size_left is None else int(self.window_size_left)
        swa_right = 0 if self.window_size_right is None else int(self.window_size_right)
        self.right_bound_runtime = 0
        if self.is_causal:
            self.mask_token = "causal" if swa_left < 0 else "causal_swa"
            self.swa_window_runtime = max(0, swa_left) if swa_left >= 0 else 0
            self.right_bound_runtime = max(0, swa_right)
        elif swa_left >= 0:
            self._not_implemented_error_if(swa_right > 0, "SM80 BPROP: non-causal SWA with window_size_right > 0 unsupported")
            self.mask_token = "swa"
            self.swa_window_runtime = swa_left
        else:
            self._not_implemented_error_if(
                swa_right > 0,
                "SM80 BPROP: window_size_right without a left window or is_causal=True has no effect; pass is_causal=True or a left window",
            )
            self.mask_token = "none"
            self.swa_window_runtime = 0
        self._value_error_if(
            self.causal_bottom_right and not (self.is_causal or swa_left >= 0),
            "SM80 BPROP: causal_bottom_right requires is_causal and/or a left window",
        )

        if self.scale_softmax is None:
            self.scale_softmax = 1.0 / math.sqrt(d_qk)

        self.batch_size = int(b)
        self.s_q_max = int(s_qo)
        self.s_k_max = int(s_kv)
        self.h_q = int(h_qo)
        self.h_kv = int(h_kv)
        self.head_dim_qk = int(d_qk)
        self.head_dim_v = int(d_v)

        # Preserve the established d64 selection rule. Strided Stats and THD
        # continue to use the generic chain; prepared compilation changes
        # launch plumbing, not the kernel-selection policy.
        self._use_d64 = _sm80_d64_fast_path_eligible(
            d_qk=self.head_dim_qk,
            d_v=self.head_dim_v,
            h_q=self.h_q,
            h_kv=self.h_kv,
            s_q=self.s_q_max,
            s_kv=self.s_k_max,
            mask_token=self.mask_token,
            right_bound=int(self.right_bound_runtime),
            causal_bottom_right=self.causal_bottom_right,
            bw_kwargs=dict(
                seq_kv_lens=object() if self.seq_kv_lens_present else None,
                seq_len_q=object() if self.seq_q_lens_present else None,
                bias=object() if self._has_bias else None,
                sinks=object() if self.sink_desc is not None else None,
                rope_freqs=object() if self._has_rope else None,
                deterministic=self.deterministic,
            ),
        )
        if self._lse_stride is not None or self.thd:
            self._use_d64 = False  # preserve the dense, packed-Stats selection boundary
        from .prepared_sm80 import native_layouts

        self._native_pointer = native_layouts(self)
        if not self._native_pointer:
            from .staged_sm80 import layout_for

            self._staged_layout = layout_for(self)

        self._is_supported = True
        self._logger.debug("check_support completed")
        return True

    # ------------------------------------------------------------------
    def compile(self) -> None:
        """Plan-time JIT: build the TemplateParams from the plan facts, load
        the specialized module via ``frost.template_loader`` (same seam as the
        SM120 adapter), and compile the full kernel chain for this shape.
        Native plans compile the entire pointer chain here, including packed
        metadata setup or the dense d=64 kernel and its unpermute epilogue. Retained
        staging layouts compile a pointer chain after their existing copies."""
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        from cudnn.sdpa.bwd.config_sm80 import bwd_params_for_flavor

        sched = _sm80_bwd_sched_policy(is_causal=self.is_causal, deterministic=self.deterministic, thd=self.thd)
        # NOTE: the generic pipeline always ran the llama-swept tile point
        # regardless of flavor (the old backward() defaults); the gptoss
        # wide-Q-tile row stays unwired pending an adapter-level perf gate.
        self._params = bwd_params_for_flavor(
            "llama",
            io_bf16=self.dtype == torch.bfloat16,
            d_qk=self.flavor_d_qk,
            d_v=self.flavor_d_v,
            is_causal=self.is_causal,
            has_swa=self.swa_window_runtime > 0 or (self.window_size_left is not None and self.window_size_left >= 0),
            causal_bottom_right=self.causal_bottom_right,
            has_seq_kv_lens=self.seq_kv_lens_present,
            has_seq_q_lens=self.seq_q_lens_present,
            has_bias=self._has_bias,
            bias_is_fp32=self._bias_is_fp32,
            bias_broadcast=self._bias_batch == 1,
            has_sink=self.sink_desc is not None,
            has_rope=self._has_rope,
            deterministic=self.deterministic,
            thd_varlen=self.thd,
            sched_policy=sched,
        )
        # RoPE preconditions the old backward() asserted (the params validator
        # covers d_qk <= 128; these two involve the shape, known only here).
        if self._has_rope:
            self._value_error_if(
                self._rope_max_s < max(self.s_q_max, self.s_k_max),
                f"rope_freqs rows ({self._rope_max_s}) must cover max(S_q={self.s_q_max}, S_kv={self.s_k_max})",
            )
            self._not_implemented_error_if(
                bool(self.s_q_max % self._params.tile_q or self.s_k_max % self._params.tile_kv),
                "SM80 bprop: RoPE requires S_q/S_kv tile-aligned",
            )
        self._kmod = _load_sm80_bwd_module(self._params)
        d64 = _sm80_bwd_kernel_mod("d64") if self._use_d64 else None
        if self._native_pointer:
            from .prepared_sm80 import build_spec

            self._prepared = build_spec(self, d64)
            self._compiled_kernel = self._prepared.artifact
        else:
            from .staged_sm80 import compile_staged

            self._staged_prepared = compile_staged(self, d64)
            self._compiled_kernel = self._staged_prepared.artifact
        self._logger.debug("compile completed")

    def scratch_workspace_bytes(self) -> int:
        """Plan-time bytes for the prepared chain and retained operand staging."""
        self._ensure_support_checked()
        if self._staged_layout is not None:
            return self._staged_layout.workspace_bytes
        from .kernels.sm80.prepared_host import workspace_regions

        return workspace_regions(self)[1]

    # ------------------------------------------------------------------
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
        seq_q_lens: Optional[torch.Tensor] = None,
        seq_kv_lens: Optional[torch.Tensor] = None,
        sink_tensor: Optional[torch.Tensor] = None,
        dsink_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        dbias_tensor: Optional[torch.Tensor] = None,
        rope_freqs: Optional[torch.Tensor] = None,
    ) -> None:
        """Run the compiled SM80 backward on caller workspace. Native operands
        bind directly; the existing staged layouts retain their copies and
        feed the same pointer chain. THD outputs remain row-bounded on device.
        """
        self._logger.debug("Entering execute")
        if self._compiled_kernel is None:
            raise RuntimeError("SdpaBwdDslSm80 is not compiled")

        # Init-time flags are compile-time facts; execute must match them
        # exactly, in both directions (Hard Rule 1).  Under THD the length
        # buffers are the graph's per-batch lengths, consumed by the setup
        # launch rather than a compiled padding mask, so their presence is
        # checked there instead.
        self._value_error_if(self._has_bias != (bias_tensor is not None), "bias presence must match the plan (has_bias)")
        self._value_error_if(self._has_rope != (rope_freqs is not None), "rope_freqs presence must match the plan (has_rope)")
        self._value_error_if((self.sink_desc is not None) != (sink_tensor is not None), "sink presence must match the plan")
        if not self.thd:
            self._value_error_if(self.seq_kv_lens_present != (seq_kv_lens is not None), "seq_kv_lens presence must match the plan")
            self._value_error_if(self.seq_q_lens_present != (seq_q_lens is not None), "seq_q_lens presence must match the plan")

        if self._prepared is not None:
            from .prepared_sm80 import execute_tensors

            execute_tensors(
                self,
                (
                    q_tensor,
                    k_tensor,
                    v_tensor,
                    o_tensor,
                    do_tensor,
                    stats_tensor,
                    dq_tensor,
                    dk_tensor,
                    dv_tensor,
                    seq_q_lens,
                    seq_kv_lens,
                    sink_tensor,
                    dsink_tensor,
                    bias_tensor,
                    dbias_tensor,
                ),
                workspace,
                current_stream,
                scale_softmax,
            )
            return

        from .staged_sm80 import run_staged

        run_staged(
            self,
            (
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                do_tensor,
                stats_tensor,
                dq_tensor,
                dk_tensor,
                dv_tensor,
                seq_q_lens,
                seq_kv_lens,
                sink_tensor,
                dsink_tensor,
                bias_tensor,
                dbias_tensor,
            ),
            workspace,
            current_stream,
            scale_softmax,
            rope_freqs,
        )
        self._logger.debug("execute completed (THD)" if self.thd else "execute completed (d64 fast path)" if self._use_d64 else "execute completed")


_sm80_bwd_cache: dict = {}


def sdpa_bwd_wrapper_sm80(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    o_tensor: torch.Tensor,
    do_tensor: torch.Tensor,
    lse_tensor: torch.Tensor,
    is_causal: bool = False,
    window_size: "tuple[int, int]" = (-1, -1),
    scale_softmax: Optional[float] = None,
    causal_bottom_right: bool = False,
    current_stream: Optional[cuda.CUstream] = None,
    seq_kv_lens: Optional[torch.Tensor] = None,
    seq_len_q: Optional[torch.Tensor] = None,
    bias_tensor: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    rope_freqs: Optional[torch.Tensor] = None,
    cum_seqlen_q_tensor: Optional[torch.Tensor] = None,
    cum_seqlen_k_tensor: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    max_s_kv: Optional[int] = None,
    max_s_q: Optional[int] = None,
) -> TupleDict:
    """SM80 (A100) SDPA backward.

    Returns ``TupleDict(dq_tensor=..., dk_tensor=..., dv_tensor=...
    [, dbias_tensor=...][, dsink_tensor=...])`` — BHSD grads; dBias
    head-major [., H, SQ, SKV] when ``bias_tensor`` is given; dSink (H,)
    fp32 when ``sinks`` is given (stable order: dq, dk, dv, dbias, dsink).
    ALiBi and block_mask are not supported (use the graph API, which routes
    them to the cuDNN backend); bias/dBias remain fully served.

    THD (``cum_seqlen_*``): ``max_s_kv`` may bound the longest per-sequence
    KV length to reduce the launch grid. Without it the packed capacity is
    a safe upper bound, requiring no device-to-host length read. With
    ``deterministic=True``, ``max_s_q`` (an upper bound on the longest
    per-sequence Q length) sizes the dQ relay counter; the packed total is
    used when absent.  Both hints are caller contracts (validating them
    would need the D2H read they exist to avoid): an undersized ``max_s_kv``
    drops KV tiles, an undersized ``max_s_q`` indexes the relay counter out
    of bounds.
    """
    # Rule 7 (python/cudnn/AGENTS.md): this entry reaches the kernel module on its
    # own, so decline by DSL version here instead of surfacing the DSL's own
    # TypeError/ModuleNotFoundError from the template load.
    from cudnn.frost.buffers import cutedsl_requirement_error

    _too_old = cutedsl_requirement_error("sdpa_bwd_wrapper_sm80")
    if _too_old is not None:
        raise NotImplementedError(_too_old)
    # THD / varlen: q/k/v/o/dO are PACKED [1, T, H, D] (BSHD) + cu_seqlens;
    # lse is packed [1, H, T_q].  Dedicated path that skips the dense BHSD
    # transpose + dense grad alloc (mirrors the forward THD branch).
    if cum_seqlen_q_tensor is not None:
        for label, present in (
            ("bias_tensor", bias_tensor is not None),
            ("rope_freqs", rope_freqs is not None),
            ("seq_kv_lens", seq_kv_lens is not None),
            ("seq_len_q", seq_len_q is not None),
        ):
            if present:
                raise NotImplementedError(f"SM80 SDPA THD (cum_seqlen_*) backward does not support {label}; the dense path serves it")
        # A missing/current stream handle does not switch CUDA devices.
        # Compile and launch on the operand device, then restore the caller.
        device_context = (
            nullcontext() if q_tensor.device.type != "cuda" or torch.cuda.current_device() == q_tensor.device.index else torch.cuda.device(q_tensor.device)
        )
        with device_context, _torch_stream_context(current_stream, q_tensor.device):
            return _sm80_thd_backward(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                do_tensor,
                lse_tensor,
                cu_q=cum_seqlen_q_tensor,
                cu_k=cum_seqlen_k_tensor,
                scale_softmax=scale_softmax,
                is_causal=is_causal,
                window_size=window_size,
                causal_bottom_right=causal_bottom_right,
                sinks=sinks,
                deterministic=deterministic,
                max_s_kv=max_s_kv,
                max_s_q=max_s_q,
            )
    if max_s_kv is not None or max_s_q is not None:
        raise ValueError("max_s_kv / max_s_q are THD hints; they require cum_seqlen_q_tensor/cum_seqlen_k_tensor")
    for nm, t in (("Q", q_tensor), ("V", v_tensor), ("O", o_tensor), ("dO", do_tensor)):
        if t.ndim != 4:
            raise ValueError(f"{nm} must be rank-4 BHSD; got {t.ndim}D")

    # Allocate grad outputs in cuDNN-FE BHSD-physical stride order (3,1,2,0):
    # contiguous (B, S, H, D) then transpose to a (B, H, S, D) view.
    b, h_q, s_q, d_qk = q_tensor.shape
    d_v = v_tensor.shape[-1]
    with _torch_stream_context(current_stream, q_tensor.device):
        dq = torch.empty((b, s_q, h_q, d_qk), dtype=q_tensor.dtype, device=q_tensor.device).transpose(1, 2)
        h_kv, s_kv = k_tensor.shape[1], k_tensor.shape[2]
        dk = torch.empty((b, s_kv, h_kv, d_qk), dtype=q_tensor.dtype, device=q_tensor.device).transpose(1, 2)
        dv = torch.empty((b, s_kv, h_kv, d_v), dtype=q_tensor.dtype, device=q_tensor.device).transpose(1, 2)
        # The prepared chain clears its accumulators and copies every auxiliary
        # element, including fully masked rows, into these output buffers.
        dbias = torch.empty_like(bias_tensor, dtype=torch.float32) if bias_tensor is not None else None
        dsink = torch.empty(h_q, dtype=torch.float32, device=q_tensor.device) if sinks is not None else None

    cache_key = (
        q_tensor.shape,
        k_tensor.shape,
        v_tensor.shape,
        q_tensor.stride(),
        k_tensor.stride(),
        v_tensor.stride(),
        o_tensor.shape,
        o_tensor.stride(),
        do_tensor.shape,
        do_tensor.stride(),
        # The compiled kernel is SPECIALIZED on the declared LSE layout
        # (compile()'s lse_stride — native strided reads), so the Stats
        # geometry is part of the plan identity, not just runtime data.
        lse_tensor.shape,
        lse_tensor.stride(),
        q_tensor.dtype,
        is_causal,
        window_size,
        scale_softmax,
        causal_bottom_right,
        deterministic,
        seq_kv_lens is not None,
        seq_len_q is not None,
        bias_tensor is not None,
        (bias_tensor.dtype if bias_tensor is not None else None),
        (tuple(bias_tensor.shape) if bias_tensor is not None else None),
        (tuple(bias_tensor.stride()) if bias_tensor is not None else None),
        sinks is not None,
        rope_freqs is not None,
        (int(rope_freqs.shape[0]) if rope_freqs is not None else 0),
        q_tensor.device,
    )
    sdpa_bwd = _sm80_bwd_cache.get(cache_key)
    if sdpa_bwd is None:
        _logger.debug("sdpa_bwd_wrapper_sm80: building new SdpaBwdDslSm80")
        wl, wr = window_size
        sdpa_bwd = SdpaBwdDslSm80(
            sample_q=q_tensor,
            sample_k=k_tensor,
            sample_v=v_tensor,
            sample_o=o_tensor,
            sample_do=do_tensor,
            sample_stats=lse_tensor,
            sample_dq=dq,
            sample_dk=dk,
            sample_dv=dv,
            sample_sink=sinks,
            sample_dsink=dsink,
            sample_bias=bias_tensor,
            sample_dbias=dbias,
            is_causal=is_causal,
            causal_bottom_right=causal_bottom_right,
            window_size_left=(None if wl is None or wl < 0 else int(wl)),
            window_size_right=(None if wr is None or wr < 0 else int(wr)),
            deterministic=deterministic,
            scale_softmax=scale_softmax,
            seq_kv_lens_present=seq_kv_lens is not None,
            seq_q_lens_present=seq_len_q is not None,
            has_bias=bias_tensor is not None,
            bias_is_fp32=(bias_tensor.dtype == torch.float32 if bias_tensor is not None else True),
            bias_batch=(int(bias_tensor.shape[0]) if bias_tensor is not None else 1),
            has_rope=rope_freqs is not None,
            rope_max_s=(int(rope_freqs.shape[0]) if rope_freqs is not None else 0),
        )
        assert sdpa_bwd.check_support(), "Unsupported configuration"
        sdpa_bwd.compile()
        _sm80_bwd_cache[cache_key] = sdpa_bwd

    with _torch_stream_context(current_stream, q_tensor.device):
        workspace = torch.empty(sdpa_bwd.scratch_workspace_bytes(), dtype=torch.uint8, device=q_tensor.device)

    sdpa_bwd.execute(
        q_tensor=q_tensor,
        k_tensor=k_tensor,
        v_tensor=v_tensor,
        o_tensor=o_tensor,
        do_tensor=do_tensor,
        stats_tensor=lse_tensor,
        dq_tensor=dq,
        dk_tensor=dk,
        dv_tensor=dv,
        dbias_tensor=dbias,
        dsink_tensor=dsink,
        scale_softmax=scale_softmax,
        current_stream=current_stream,
        workspace=workspace,
        seq_kv_lens=seq_kv_lens,
        seq_q_lens=seq_len_q,
        sink_tensor=sinks,
        bias_tensor=bias_tensor,
        rope_freqs=rope_freqs,
    )

    out = TupleDict(dq_tensor=dq, dk_tensor=dk, dv_tensor=dv)
    if dbias is not None:
        out["dbias_tensor"] = dbias
    if dsink is not None:
        out["dsink_tensor"] = dsink
    return out


# ---------------------------------------------------------------------------
# SM100 (Blackwell) large-head-dim backward: the three-stage chain
# ---------------------------------------------------------------------------

_SM100_KERNEL_DIR = "cudnn/sdpa/bwd/kernels"
# Stage 2's descriptor scratch: Q / dO / K / V, clamped on device.
_THD_STAGE2_DESC_SLOTS = 4
_SM100_STAGE2_FILE = "sm100/bprop_d512_f16.py"
# The 2x2-datapath twin of stage 2: one fused cta_group::2 pipeline per pair (64 q rows per CTA, both BMMs on every
# SM, S never leaves the lane's registers) inside the same (4,1,1) cluster, K / V shared across the two pairs by TMA
# multicast.  Same host ABI, workspace format and LSE / do_dot contract as the role-split file; a SIBLING file so the
# 4x1 rendering stays byte-identical (its own FROST_SOURCE_DIGEST, its own config record).
_SM100_STAGE2_FILE_2X2 = "sm100/bprop_d512_f16_2x2.py"
# Which stage-2 datapath the SM100 d512 chain renders.  False = the cga4x1 role split (what ships).  True = the 2x2 twin
# (the Rubin d512 bring-up vehicle; on SM100 the A/B gate for a default flip is >= +3 % stage-2 median on BOTH dense and
# causal at B=1 H=128 S=8192 d=512 bf16, see the twin's docstring).  A module constant read at CALL time (``compile``),
# not a knob and not an env var: it must never differ per plan.  The DQ_SINGLE_LAUNCH precedent (api_dsl_sm107).
STAGE2_2X2: bool = False
# Stage-3 dQ launch shape under GQA (the #1318 ``b_head_group`` arm of the stage-3 template, ported from the cc 10.7 d256
# chain's ``api_dsl_sm107.DQ_SINGLE_LAUNCH``).  True = ONE dQ GEMM launch per head chunk: the dQ rendering takes
# ``MatmulTemplateParams.b_head_group = group`` (its B = K is indexed by ``h // group``, the K head the group's Q heads
# share) over the whole dS chunk and the whole dQ chunk -- what ships.  False = one launch per group MEMBER over every
# ``group``-th Q head (``b_head_group = 1``; the per-member loop this chain ran before): the bitwise pin's twin (the same
# k-tile walk per output tile into the same fp32 accumulator, so identical bits) and the A/B base.  At H_q / H_kv = 16 and
# S = 8K the member launches were sixteen under-one-wave launches of 16 four-CTA clusters on a 37-cluster B200, 128 dQ
# launches per backward.  MHA (group 1) renders and launches identically either way.  THD keeps the per-member loop
# whatever this says (this chain's packed dQ host launches per member; the grouped arm's THD leg is the cc 10.7 d256
# chain's), and so does a budget-limited head chunk that is not a whole number of GQA groups (``_sm100_head_chunk`` may
# hand out any divisor of H_q; the host then maps each Q head to its own KV head, ``prepared_host``).  A module
# constant read at CALL time (``compile``), not a knob and not an env var: it
# must never differ per plan.  Mirrors the host (``prepared_host._dq_launches`` refuses a record that is neither 1 nor the
# group) through ``_dq_b_head_group``, copied off the rendered record -- ONE source of truth.
DQ_SINGLE_LAUNCH: bool = True
# Stage-3 causal K-trim under THD.  True = the packed stage 3 renders the same per-sequence trim the dense path renders
# (``causal_mode`` LO / HI with the diagonal edge; bottom-right's per-sequence diagonal ``S_kv[b] - S_q[b]`` read by the
# kernel from the setup launch's metadata, ``MatmulTemplateParams.thd_causal_bottom_right``) -- what ships.  False = the
# untrimmed rendering this chain used before (``CAUSAL_K_NONE``: every k tile of the group read, masked ones included --
# measured -20 % whole-backward on the dense path with the trim forced off): the bitwise pin's twin and the A/B base.  A
# THD graph with a sliding window or a right-band widening keeps the untrimmed rendering either way (the window edge is
# not taken under THD on this chain: the template's THD arm offers it for the kv-blocked cc 10.7 d256 chain, its
# per-sequence bound has not been validated on the Q-major (512, 512) rows; and that arm takes no constant shift).
# The whole-workspace zero-fill stays in both cases (the 512-row cluster M tile straddles two 256-row stage-2 blocks, so
# the fill is the correctness and the trim the optimization, exactly as on the dense path).  A module constant read at
# CALL time (``compile``), not a knob and not an env var: it must never differ per plan.
THD_STAGE3_TRIM: bool = True
_SM100_MATMUL_FILE = "bprop_matmul_blackwell.py"
# The Rubin line as ``engines._RUBIN`` spells it (cc 10.7 up to the SM100 line's end): the 2x2 twin's ring levers
# (8 chunk stages / 2 cast stages / 325 KiB) follow the line, not the one cc that exists today.
_RUBIN_SM = (107, 119)


def _rubin_line(sm: int) -> bool:
    """True for a device on the Rubin line (``major * 10 + minor`` in ``_RUBIN_SM``)."""
    return _RUBIN_SM[0] <= sm <= _RUBIN_SM[1]


# Workspace budget for S + dS. Above this the head chunk shrinks; the loop then
# runs more launches over the same total work (plan section 5).
_SM100_WS_BUDGET_BYTES = 4 << 30
# THD: the blocked workspace pads every sequence's block to _SM100_WS_BLOCK_ROWS
# rows, so an equal-length packed plan carries B * 128 more rows per head than
# the dense plan of the same lengths -- enough to tip the divisor rule at the
# budget edge (B=1 or B=4 at S=8192: 16 -> 8 heads per chunk, twice the
# launches for the same work; B=4 S=2048: 64 -> 32).  The budget is therefore
# charged on the TOKEN rows (the dense plan's row count at equal lengths), and
# the padding may carry the slab past it by at most budget / this (512 MiB):
# parity with the dense chunk for equal lengths S >= 1024 (pad fraction
# 128 / S <= 1/8); beyond that -- many short sequences against a long kv -- the
# padded slab is charged in full, as before.  `scratch_workspace_bytes` is
# computed from the chosen chunk either way, so the request stays honest
# (Rule 8).
_SM100_WS_THD_PAD_SLACK = 8

# Stage-3 cluster tile by sequence length AND compute capability (`MatmulTemplateParams.cgrp_tile_mn`, the template's
# `_TILE_ROWS`).  The (512, 512) row (cluster 2x2, A multicast to two pairs, one 512-column accumulator) co-resides 34 four-CTA
# clusters on the B200 (136 of 148 SMs, `launch__cluster_max_active`); the (512, 256) row (cluster 2x1, the same per-pair
# 512x256 work and k walk, A read once per N tile instead of multicast) co-resides 74 and keeps every SM busy, and the two are
# BITWISE twins (`test_stage3_small_s_tile_is_bitwise_the_wide_row`).  MEASURED (B200, 1155 MHz SW power cap, cuDNN 9.26.0.51,
# DSL 4.7.0, 2026-10-01; in-process round-robin A/B, every arm built once and bitwise-checked, 3 rounds of CUPTI per-event
# medians per slot, NVML clock sampled, CLEAN slots only): the 2x1 row's residency + wave gain carries where the GEMM is MMA-bound
# at the full clock -- dense S2K dV/dK/dQ 523/527/525 -> 465/464/464 us (-11.5 %, 84.5 % of peak), dense S4K 1959/1968/1964 ->
# 1737/1735/1734 us (-11.6 %, 90.5 % of peak), causal S2K 359/363/363 -> 337/340/339 us (-6.4 %), causal S4K 1191/1204/1200 ->
# 1122/1132/1142 us (-5.6 %), whole backward -4.6 % dense / -2.0..-2.2 % causal; each of these four cells reproduced in THREE
# CLEAN slots (medians of the per-slot stage-3 deltas -11.2 / -11.2 / -6.4 / -5.9 % for dense S2K / S4K / causal S2K / S4K,
# every slot within 0.4 % of its cell's median, stage 2 within +-0.2 %) -- and does NOT at S8K (dense +0.1..+0.8 % on
# stage 3: the row's second DRAM read of A turns a -12 % at base clock into a wash at 1155 MHz; causal +7.5 %, dQ +11.2 %) nor
# at S32K (dense +24 %, causal +32 %: the 148-SM row drags the power-capped clock of the whole chain), and behind it at S8K the
# UNCHANGED stage 2 ran +13-15 % slower in three CLEAN slots (mechanism open; absent at S4K, the only chunk boundary the rule
# serves: stage 2 +0.2 %).  Hence the row is keyed on the PADDED sequence length: (512, 256) up to
# `_SM100_STAGE3_SMALL_S_MAX`, the (512, 512) row above.  The key is max(S_q_pad, S_kv_pad), not S_kv alone: dV / dK walk K = S_q
# and dQ walks K = S_kv, every cell had S_q = S_kv, and the memory-side term that undoes the gain grows with the k walk -- a
# rectangular backward (S_q 32K, S_kv 2K) is unmeasured and takes the shipped row.  S in (4096, 8192) is unmeasured too and
# takes the shipped row.  The rule is mask-blind by measurement (dense and causal move in the same direction at every S).
# BSHD only: the THD leg was validated on the (512, 512) row alone.  The causal zero-fill / loose trim apply to both rows alike
# (neither is causal-tight at 512 M rows, `_causal_k_range`).  COMPUTE CAPABILITY: cc 10.0..10.6 only -- every number above is
# the B200's (148 SMs, 34 vs 74 resident clusters at 231 KiB/CTA; none of it transfers to another SM count or SMEM carveout),
# and the cc 10.7 d512 row inherits this `compile`, so it must keep the (512, 512) row until it is measured on its own board.
# Module constants read at compile time, never knobs or env vars.
_SM100_STAGE3_SMALL_S_TILE = (512, 256)
_SM100_STAGE3_SMALL_S_MAX = 4096
_SM100_STAGE3_SMALL_S_CC = (100, 106)  # inclusive cc range (major * 10 + minor) the (512, 256) row was measured on


def _sm100_stage3_cgrp_tile_mn(s_pad: int, thd: bool, cc: tuple) -> tuple:
    """The stage-3 cluster tile for a backward at this padded max(S_q, S_kv) on a device of compute capability ``cc``
    (a ``(major, minor)`` pair, the chain's `compute_capability(resolve_device(...))`); see `_SM100_STAGE3_SMALL_S_TILE`.
    (512, 256) for BSHD at ``s_pad <= _SM100_STAGE3_SMALL_S_MAX`` on cc 10.0..10.6; (512, 512) for every other cc, for THD
    and for longer sequences.  Mask-blind on purpose (see the constant's comment)."""
    major, minor = cc
    lo, hi = _SM100_STAGE3_SMALL_S_CC
    if lo <= major * 10 + minor <= hi and not thd and s_pad <= _SM100_STAGE3_SMALL_S_MAX:
        return _SM100_STAGE3_SMALL_S_TILE
    return (512, 512)


def _sm100_kernel_path(fname: str) -> str:
    import cudnn.sdpa.bwd.kernels as _k

    return os.path.join(os.path.dirname(_k.__file__), fname)


# The blocked S/dS workspace's row granularity: stage 2's per-CTA store box.
# Kept next to the workspace math that uses it; CFG.WS_BLOCK_ROWS is the same
# number and the stage-2 validator pins it to TILE_M.
_SM100_WS_BLOCK_ROWS = 128


def _sm100_device_clusters(device, cga_m: int) -> int:
    """Clusters that fit the device once, for the THD persistent grid.

    Occupancy-sized rather than work-sized: the work list is a device value, so
    the grid cannot be it.  One CTA per SM is this kernel's design point, so the
    cluster count is simply SM count / CGA width.
    """
    from cudnn.frost import device as _dev

    idx = device.index if hasattr(device, "index") and device.index is not None else 0
    return max(1, _dev.multiprocessor_count(idx) // cga_m)


def _sm100_head_chunk_thd(
    h_q: int, ws_rows: int, s_kv: int, bpe: int, budget: int = _SM100_WS_BUDGET_BYTES, group: int = 1, t_rows: Optional[int] = None
) -> int:
    """``_sm100_head_chunk`` for the BLOCKED workspace.

    Same divisor rule, but a head's slab is ``ws_rows * s_kv`` -- packed q
    tokens rather than ``B * S_q_max``, which is where THD's memory win is.

    ``t_rows`` is the packed token capacity (``ws_rows`` minus the block
    padding).  Given, the budget is charged on it, so an equal-length packed
    plan gets the dense plan's chunk, and the padded slab may exceed the budget
    by at most ``budget // _SM100_WS_THD_PAD_SLACK`` (see the constant).
    Without it the slab is charged in full -- the original rule.
    """
    per_head = 2 * ws_rows * s_kv * bpe
    per_head_tokens = per_head if t_rows is None else 2 * min(t_rows, ws_rows) * s_kv * bpe
    slack = 0 if t_rows is None else budget // _SM100_WS_THD_PAD_SLACK
    cands = [c for c in range(1, h_q + 1) if h_q % c == 0]
    for c in sorted(cands, reverse=True):
        if per_head_tokens * c <= budget and per_head * c <= budget + slack:
            return c
    return 1


def _sm100_head_chunk(b: int, h_q: int, s_q: int, s_kv: int, bpe: int, budget: int = _SM100_WS_BUDGET_BYTES, group: int = 1) -> int:
    """Largest DIVISOR of ``h_q`` whose S+dS chunk fits ``budget``.

    A divisor, not a floor: even chunks mean no ragged tail, so one compiled
    artifact serves every chunk and the host just varies ``head_base``. Returns
    1 when even a single head does not fit -- the caller then requests what one
    head needs and the size is still honest.
    """
    per_head = 2 * b * s_q * s_kv * bpe
    # A GQA group may span several chunks: stage 2 maps each global Q head to
    # its KV head, dK/dV keep one partial per Q head, and the prepared host's
    # dQ handles non-group-aligned chunks head by head. Requiring whole groups
    # would break the budget for long-context GQA models (Gemma 4: Hq=16, Hkv=2).
    cands = [c for c in range(1, h_q + 1) if h_q % c == 0]
    for c in sorted(cands, reverse=True):
        if per_head * c <= budget:
            return c
    return 1


class SdpaBwdDslSm100(SdpaBwdDsl):
    """d in (256, 512] backward on Blackwell, as do_dot -> S/dS -> three GEMMs.

    Why three stages and not one kernel: a fused d=512 backward needs dV
    [128 kv, 512] fp32 = 512 TMEM columns AND dK = 512 more, plus the S/dS
    accumulators, against 512 columns per CTA. It does not fit, so S and dS go
    to a GMEM workspace and the gradients are three plain batched GEMMs over it.

    Sub-512 head dims need no kernel change: the TMA descriptors carry the real
    ``d`` and a box reading past it is HW zero-filled, so the padded lanes
    contribute 0 to both BMMs. We pay the full 512-wide MMA for that.
    """

    @staticmethod
    def _thd_total(capacity: int, declared: Optional[int]) -> int:
        """Token capacity, tightened by the caller's declared packed total.

        Always a MIN: a declaration can only shrink what the buffers can hold,
        so a stale or oversized one cannot push an access outside the caller's
        allocation.  It sizes the workspace; it does NOT make the extents exact,
        because it is a maximum while the row that must read as zero is the
        current ``cu_*[B]`` -- hence the kernels' device-side clamps.
        """
        return capacity if declared is None else min(capacity, max(int(declared), 0))

    @property
    def thd_total_q(self) -> Optional[int]:
        """Packed Q-token extent this instance binds its views at, or None when
        not THD.  The lowering builds the caller's packed views at exactly this
        many tokens, so it reads the number from here rather than re-deriving
        it -- two copies of ``min(B * S_max, declared)`` is one copy too many."""
        return self._t_q_cap if self.thd else None

    @property
    def thd_total_kv(self) -> Optional[int]:
        """Packed KV-token extent; see :attr:`thd_total_q`."""
        return self._t_kv_cap if self.thd else None

    @staticmethod
    def _bshd_physical_ok(desc: TensorDesc) -> bool:
        """True when a logical-BHSD desc sits on compact BSHD storage.

        Same predicate the SM120 adapter uses; defined here rather than shared
        because that one is private to its class and this row's staging decision
        is the only other caller.
        """
        b, h, s, d = (int(x) for x in desc.shape)
        return tuple(int(x) for x in desc.stride) == (s * h * d, d, h * d, 1)

    def _initialize_implementation(self) -> None:
        q_shape = tuple(int(x) for x in self.q_desc.shape)  # logical BHSD
        k_shape = tuple(int(x) for x in self.k_desc.shape)
        self.batch_size, self.h_q, self.s_q_max, self.head_dim_qk = q_shape
        self.h_kv, self.s_k_max = int(k_shape[1]), int(k_shape[2])
        self.head_dim_v = int(tuple(self.v_desc.shape)[3])
        self.dtype = self.q_desc.dtype
        self._bpe = 2
        # attn_scale is OPTIONAL on the graph, so `scale_softmax` arrives None
        # when the caller omits it. Default it here, as the SM120 and SM80
        # adapters do -- without this the row admits the graph, check_support
        # passes, and execute dies on `None * log2(e)`. Found by review on the
        # d512 bring-up PR; regression test `test_default_attn_scale`.  ONLY
        # None defaults: an explicit attn_scale = 0.0 is a valid declared scale
        # the analyzer preserves (uniform P -> dQ = dK = 0, dV = sum(dO) / S_kv);
        # `or == 0.0` here ran it at 1/sqrt(d) (Codex on #1212, same clause in
        # the sm107 adapter; host pin test_sdpa_bwd_dsl_sm107.py::
        # test_explicit_zero_attn_scale_survives_the_adapters).
        if self.scale_softmax is None:
            self.scale_softmax = 1.0 / math.sqrt(self.head_dim_qk)
        # Tile-rounded COMPILE shape. The kernel's grid and workspace are tiled,
        # so a sequence length that is not a multiple runs on the next multiple
        # up and the tail is masked. Q/K/V/dO ride their real TMA extents, whose
        # overshoot is HW zero-filled; only S/dS are actually allocated padded.
        self._sq_pad = -(-self.s_q_max // 256) * 256
        self._skv_pad = -(-self.s_k_max // 128) * 128
        self._is_padded = self._sq_pad != self.s_q_max or self._skv_pad != self.s_k_max
        self._compiled = None
        self._staged_prepared = None
        self._zero_ws = False
        self._thd_lse_token_major = bool(getattr(self, "thd_stats_token_major", False)) and self.thd
        # Head-major Stats only: the caller's head stride, which the compiled
        # artifact binds as the LSE tensor's third EXTENT.  0 = compact.
        self._thd_lse_head_stride = int(getattr(self, "thd_stats_head_stride", 0) or 0) if (self.thd and not self._thd_lse_token_major) else 0
        # GQA / MQA: stage 3 cannot write dK/dV straight to the output, because
        # every Q head in a group contributes to the SAME KV head. It writes one
        # partial per Q head and a separate reduce folds the group. group == 1
        # (MHA) skips both the partial buffers and the reduce entirely.
        self._gqa_group = self.h_q // self.h_kv
        self._qh_chunk = _sm100_head_chunk(self.batch_size, self.h_q, self._sq_pad, self._skv_pad, self._bpe, group=self._gqa_group)
        # The dQ rendering's B head group (`MatmulTemplateParams.b_head_group`), copied off the record `compile()` builds
        # so the prepared host launches exactly what was rendered (`prepared_host.host` -> `_dq_launches`): 1 = one dQ
        # launch per GQA group member (and MHA, and THD), the group = one launch per chunk (`DQ_SINGLE_LAUNCH`).
        self._dq_b_head_group = 1
        # THD overrides both the workspace shape and the chunk below.
        if self.thd:
            # PACKED [1, T, H, D]: the declared shapes carry the ENVELOPE
            # (B, H, S_max, D) and the token capacity is the packed buffers'
            # own extent, tightened by a declared total when there is one.
            self._t_q_cap = self._thd_total(self.s_q_max * self.batch_size, self.max_total_seq_len_q)
            self._t_kv_cap = self._thd_total(self.s_k_max * self.batch_size, self.max_total_seq_len_kv)
            # Blocked workspace rows: every sequence's block is padded up to
            # WS_BLOCK_ROWS, so B blocks cost at most B-1 rows of padding each
            # (see tile_dsl.thd.write_thd_row_offsets for why that granularity).
            self._ws_rows_cap = self._t_q_cap + self.batch_size * _SM100_WS_BLOCK_ROWS
            self._ws_rows_cap = -(-self._ws_rows_cap // _SM100_WS_BLOCK_ROWS) * _SM100_WS_BLOCK_ROWS
            # The head chunk now divides a per-head slab measured in packed rows
            # rather than B * S_max^2 -- the whole point of the blocked layout.
            # The budget is charged on the token rows (`t_rows`), so equal
            # lengths get the dense plan's chunk (see _SM100_WS_THD_PAD_SLACK).
            self._qh_chunk = _sm100_head_chunk_thd(self.h_q, self._ws_rows_cap, self._skv_pad, self._bpe, group=self._gqa_group, t_rows=self._t_q_cap)
        # Retain the grandfathered dense conversion fallback for layouts that
        # cannot use the native TMA pointer host. It stages compact BSHD buffers
        # from caller workspace, decided here from the declarations. Native
        # legal TMA layouts clear these staging lists below; BHSD dO now goes
        # directly to the kernel too. THD retains its compact BSHD boundary.
        self._stage_in = tuple(
            name
            for name, desc in (("q", self.q_desc), ("k", self.k_desc), ("v", self.v_desc), ("o", self.o_desc), ("dO", self.do_desc))
            if not self._bshd_physical_ok(desc)
        )
        self._stage_out = tuple(name for name, desc in (("dQ", self.dq_desc), ("dK", self.dk_desc), ("dV", self.dv_desc)) if not self._bshd_physical_ok(desc))
        from .prepared_sm100 import native_io_layout

        self._prepared = None
        self._prepared_native = all(
            native_io_layout(desc) for desc in (self.q_desc, self.k_desc, self.v_desc, self.o_desc, self.do_desc, self.dq_desc, self.dk_desc, self.dv_desc)
        )
        if self.thd:
            self._prepared_native = self._prepared_native and not (self._stage_in or self._stage_out)
        elif self._prepared_native:
            # TMA descriptors and strided stores address these layouts directly.
            self._stage_in = self._stage_out = ()

    # --- capability backstop -------------------------------------------------
    def check_support(self) -> bool:
        """Re-check what the Capabilities row promised.

        Reaching a raise here means the row lied -- these are backstops, not the
        gate (engine contract section 1). They are ValueError, never assert: an
        assert vanishes under -O and an import-time crash is undebuggable from
        the frontend.
        """
        from cudnn.frost.buffers import cutedsl_requirement_error

        error = cutedsl_requirement_error("SdpaBwdDslSm100")
        if error:
            raise NotImplementedError(error)
        self._value_error_if(self.head_dim_qk != self.head_dim_v, f"SM100 bwd: d_qk must equal d_v; got {self.head_dim_qk} / {self.head_dim_v}")
        self._value_error_if(not (256 < self.head_dim_qk <= 512), f"SM100 bwd: d must be in (256, 512]; got {self.head_dim_qk}")
        self._value_error_if(
            self.head_dim_qk % 8 != 0, f"SM100 bwd: d must be a multiple of 8 (TMA 16-byte innermost extent at 2 B/elem); got {self.head_dim_qk}"
        )
        self._value_error_if(self.h_q % self.h_kv != 0, f"SM100 bwd: h_q ({self.h_q}) must be a multiple of h_kv ({self.h_kv})")
        # No S_q / S_kv tile rule any more: a non-multiple is served by rounding
        # the compile shape up and masking the tail.
        # SWA and bottom-right causal ARE implemented (the tile bounds and the
        # per-cell mask both come from the shared mask helpers). Padding is not:
        # it needs the per-batch kv length, and this kernel threads a scalar.
        if self.thd:
            self._value_error_if(
                self.seq_kv_lens_present or self.seq_q_lens_present, "SM100 bwd: THD carries its lengths in the metadata buffer, not seq_len tensors"
            )
            # The declared totals are REQUIRED, and the reason is the workspace,
            # not the numerics.  scratch_workspace_bytes() is a BUILD-time
            # function: the blocked S/dS row count and delta's row stride are
            # both fixed from the packed token capacity before any buffer
            # exists.  Undeclared, that capacity falls back to B * S_max --
            # more tokens than a packed buffer holds -- and the packed views
            # would read past it.  cuDNN's own backward node sizes its ragged
            # workspaces from the same attribute.
            self._value_error_if(
                self.max_total_seq_len_q is None or self.max_total_seq_len_kv is None,
                "SM100 bwd THD: max_total_seq_len_q and max_total_seq_len_kv must be declared "
                "(the blocked workspace is sized from the packed token totals at build time)",
            )
            # The prepared THD chain binds compact packed BSHD buffers directly.
            # Keep the existing layout boundary: incompatible declarations
            # decline here before compiling or binding a pointer host.
            self._value_error_if(
                bool(self._stage_in or self._stage_out),
                f"SM100 bwd THD: {', '.join(self._stage_in + self._stage_out)} must be BSHD-physical " "(the packed path has no staging copy)",
            )
            self._value_error_if(
                self._thd_lse_token_major and bool(self.thd_stats_head_stride),
                "SM100 bwd THD: thd_stats_head_stride is head-major-only (token-major (T, H) Stats is compact)",
            )
            # A head stride SHORTER than the packed total puts the later heads'
            # rows past the buffer -- the kernel reads [0, h, row] at that
            # stride for every row < t_q.
            self._value_error_if(
                bool(self._thd_lse_head_stride) and self._thd_lse_head_stride < self._t_q_cap,
                f"SM100 bwd THD: Stats head stride {self._thd_lse_head_stride} must cover the packed " f"token total {self._t_q_cap}",
            )
        else:
            self._value_error_if(self.seq_kv_lens_present or self.seq_q_lens_present, "SM100 bwd: padding masks (seq lens) are not implemented")
        self._value_error_if(self.deterministic, "SM100 bwd: deterministic mode is not implemented")
        return True

    def scratch_workspace_bytes(self) -> int:
        """delta + one head chunk of S and dS.

        A pure function of (B, H, S_q, S_kv, dtype) known at build time, with no
        per-execute allocation: the executor carves all of it from the caller's
        buffer, which is what keeps the plan CUDA-graph friendly.
        """
        if self.thd:
            # delta is packed [1, H, T_q]; S/dS are the blocked
            # [H_chunk, R_cap, N] pair; then the metadata buffer and the two
            # descriptor scratches the kernels patch for themselves.
            delta = ws_align(self.h_q * (-(-self._t_q_cap // 128) * 128) * 4)
            ws = ws_align(self._qh_chunk * self._ws_rows_cap * self._skv_pad * self._bpe)
            meta = ws_align((5 * self.batch_size + 5) * 4)
            desc = ws_align(_THD_STAGE2_DESC_SLOTS * 128) + ws_align((self.batch_size + 1) * 128)
            total = delta + 2 * ws + meta + desc
        else:
            delta = ws_align(self.batch_size * self.h_q * (-(-self.s_q_max // 128) * 128) * 4)
            ws = ws_align(self.batch_size * self._qh_chunk * self._sq_pad * self._skv_pad * self._bpe)
            # Stage 2 / stage 3's THD ABI slots ((B,) int32 metadata, one int64
            # descriptor word), dead on the dense path (every read is under
            # const_expr(_THD)) but part of the compiled ABI: borrowed here (Rule 8).
            total = delta + 2 * ws + ws_align(self.batch_size * 4) + ws_align(8)
        for name in self._stage_in + self._stage_out:
            s_len = self.s_k_max if name in ("k", "v", "dK", "dV") else self.s_q_max
            total += ws_align(self.batch_size * s_len * self.h_q * self.head_dim_qk * self._bpe)
        if self._gqa_group > 1:
            # One dK and one dV partial per Q head, reduced to the KV heads at
            # the end. Sized on h_q, not h_kv -- that is the whole point.
            # THD packs the kv axis into ONE batch of `t_kv_cap` tokens, which is
            # the same memory win the blocked S/dS workspace gets: B * S_kv_max
            # tokens become the declared packed total.
            _kv_rows = self._t_kv_cap if self.thd else self.batch_size * self.s_k_max
            total += 2 * ws_align(_kv_rows * self.h_q * self.head_dim_qk * self._bpe)
        return total

    # --- stage-2 template selection -------------------------------------------
    # The row's name: the spec's, the prepared artifact's symbol (``frost_<name>_prepared``, Rule 6) and the launch
    # spec's.  A subclass that is its own engine row (the cc 10.7 d512 row, ``api_dsl_sm107_d512``) overrides it.
    _NAME = "sdpa_bwd_sm100"
    # The template-module cache tag of this row's stage-2 rendering (``load_template``): one per row, because the tag
    # keys the cache and the test fixtures spy on it to learn which stage-2 FILE served a plan.
    _STAGE2_TAG = "sdpa_bwd_sm100_stage2"

    def _stage2_file(self) -> str:
        """The stage-2 kernel FILE this row renders (relative to ``kernels/``): the 4x1 role split unless the module
        constant ``STAGE2_2X2`` selects the twin.  A subclass pins its own file here."""
        return _SM100_STAGE2_FILE_2X2 if STAGE2_2X2 else _SM100_STAGE2_FILE

    def _stage2_record(self, stage2_fields: dict):
        """The stage-2 template record for ``stage2_fields`` (dtype / mask / THD) on this row and device.

        The 4x1 role split takes the base ``TemplateParams`` exactly as it always did (its PTX md5 is pinned).  The 2x2
        twin's ring levers follow the device's SMEM: 4 chunk stages / 1 cast stage fit SM100's 227 KiB, 8 / 2 fill the
        Rubin line's 325 KiB (``_rubin_line``).  The ``TemplateParams2x2`` record is built only on the twin path, so the
        base record (and its digest) is untouched.  A subclass that always renders one arm overrides this."""
        from cudnn.sdpa.bwd.config_sm100 import TemplateParams

        if not STAGE2_2X2:
            return TemplateParams(**stage2_fields)
        from cudnn.sdpa.bwd.config_sm100 import SM107_USABLE_DYN_SMEM_2X2, TemplateParams2x2

        _major, _minor = self._device_cc()
        _rubin = _rubin_line(_major * 10 + _minor)
        return TemplateParams2x2(
            **stage2_fields,
            stages_kv=8 if _rubin else 4,
            cast_stages=2 if _rubin else 1,
            **({"smem_cap_bytes": SM107_USABLE_DYN_SMEM_2X2} if _rubin else {}),
        )

    # --- compilation ---------------------------------------------------------
    def _device_cc(self) -> tuple:
        """``(major, minor)`` of the device Q lives on, resolved like the prepared host resolves its ``--gpu-arch``
        (`prepared_sm100`: `compute_capability(resolve_device(q.device))`).  One seam, so a test can fake the cc the
        plan-time rules see (`_sm100_stage3_cgrp_tile_mn`, the stage-2 datapath levers) without faking the host's target."""
        from cudnn.frost.device import compute_capability, resolve_device

        return tuple(compute_capability(resolve_device(self.q_desc.device)))

    def compile(self) -> None:
        """Plan-time JIT for the whole chain: stage 2's specialized module plus
        the two stage-3 GEMM specializations (dV/dK share one; dQ needs the other
        operand-major and the other causal K-trim direction). Native layouts
        and the existing dense conversion path both compile the complete
        pointer chain here."""
        self._ensure_support_checked()
        if self._compiled is not None:
            return self._compiled
        from cudnn.sdpa.bwd.config_sm100 import (
            CAUSAL_K_HI,
            CAUSAL_K_LO,
            CAUSAL_K_NONE,
            MatmulTemplateParams,
            vec_bytes_epi_for,
        )

        dtype_code = DTYPE_BF16 if self.dtype == torch.bfloat16 else DTYPE_FP16
        stage2_fields = dict(
            dtype_qkv=dtype_code,
            window_right=(self.window_size_right if self.window_size_right is not None else 0) if self.is_causal else None,
            window_left=self.window_size_left,
            bottom_right=self.causal_bottom_right,
            thd_varlen=self.thd,
        )
        # The device's compute capability, resolved once per compile (the stage-3 tile rule keys on it; the stage-2
        # record selection resolves it again inside _stage2_record, the cc 10.7 row's seam).
        cc = self._device_cc()
        stage2_mod = load_template(_sm100_kernel_path(self._stage2_file()), self._stage2_record(stage2_fields), tag=self._STAGE2_TAG)
        # Stage 2's write block = the cluster's q span: 256 on both datapaths (the 2x2 config spells it out; the 4x1
        # config's TILE_M * CTA_MMA is the same number, read through the default).
        gran = getattr(stage2_mod.CFG, "CLUSTER_Q_ROWS", stage2_mod.CFG.TILE_M * stage2_mod.CFG.CTA_MMA)
        lo = CAUSAL_K_LO if self.is_causal else CAUSAL_K_NONE
        hi = CAUSAL_K_HI if self.is_causal else CAUSAL_K_NONE
        vec = vec_bytes_epi_for(self.head_dim_qk, self._bpe)
        # How far past the plain kv <= q diagonal stage 2 actually writes. Band
        # widening and bottom-right alignment both push it out, and they add; a
        # stage-3 trim that ignores them cuts away real data.
        shift = (self.window_size_right or 0) if self.is_causal else 0
        if self.causal_bottom_right:
            shift += self.s_k_max - self.s_q_max
        # A non-zero shift breaks the alignment that lets stage 3 read only what
        # stage 2 wrote, so the skipped region has to be ZEROED first -- see
        # _zero_ws_needed.
        # Zero whenever a causal-family MASK is active, not just when shift != 0,
        # and not "when the trim is active" -- THD has no trim and needs this
        # MORE, not less. THREE independent reasons, any one of which alone
        # would require it:
        #   1. the never-empty clamp in `_causal_k_range` means a structurally
        #      masked M tile still reads one k-tile of workspace, and must see
        #      zeros there;
        #   2. stage 3's cluster M tile (512) is WIDER than stage 2's write
        #      block (`gran`, 256), so a per-tile K range cannot exclude the
        #      skipped region at all -- the zero-fill, not the trim, is what
        #      makes the causal path correct. See `_causal_k_range`.
        #   3. under THD with a sliding window the trim is switched off outright
        #      (below), so stage 3 reads EVERY k tile of the group, skipped ones
        #      included.
        self._zero_ws = self.is_causal
        # THD: the trim is PER SEQUENCE, and the same arithmetic as the dense
        # path's.  Every bound in `_causal_k_range` is a sequence-relative row
        # once the kernel hands it the tile's M base inside its sequence and the
        # sequence's own k count (which it already did, `_thd_group`; the blocked
        # row offset and the packed token base are added to the TMA coordinates
        # after the range is chosen), and stage 2's THD unit masks a 256-row q
        # tile of one sequence with that sequence's lengths -- so the written
        # band per sequence is the dense band with the sequence's diagonal.  The
        # constant part of the shift (the right-band widening) stays a template
        # constant; the bottom-right part `S_kv[b] - S_q[b]` is a per-sequence
        # quantity the kernel reads from the setup launch's metadata
        # (`MatmulTemplateParams.thd_causal_bottom_right`; the template's THD
        # arm `_thd_causal_k_range`, which also keeps an EMPTY range for a tile
        # with no kept cell and stores zeros there).  Measured on the dense
        # path with the trim forced off (the untrimmed twin's exact code shape):
        # -20 % on the whole backward at B=1 H=128 S=8192 d=512 bf16 causal,
        # ~259 -> ~207 TFLOPS, every round; the cost scales with sequence length,
        # so a packed workload of short sequences pays less.  The window edge is
        # not taken under THD on this chain (the template offers it for the
        # kv-blocked cc 10.7 d256 chain; its per-sequence bound has not been
        # validated on the Q-major (512, 512) rows), and the THD arm takes no
        # CONSTANT shift (the right-band widening), so a windowed or right-band
        # packed graph renders UNTRIMMED and pays the k tiles the band skipped.
        # `THD_STAGE3_TRIM = False` is the
        # untrimmed twin for every packed causal graph (the bitwise pin).
        per_seq = False
        if self.thd:
            windowed = self.window_size_left is not None
            right_band = bool(self.window_size_right)
            if self.is_causal and THD_STAGE3_TRIM and not windowed and not right_band:
                shift = 0  # the THD arm takes no constant shift: the diagonal offset is per sequence (thd_causal_bottom_right)
                per_seq = bool(self.causal_bottom_right)
            else:
                lo = hi = CAUSAL_K_NONE
                shift = 0
        # dtype_qkv must match stage 2's: stage 3 reads the S/dS workspace stage
        # 2 wrote, and stores the gradients in the graph's io dtype.
        # The cluster tile by padded max(S_q, S_kv) and device cc (`_sm100_stage3_cgrp_tile_mn`): the (512, 256) row on short
        # BSHD sequences of the SM100 line, the (512, 512) row otherwise -- bitwise twins, so this is a timing choice only.
        tile = _sm100_stage3_cgrp_tile_mn(max(self._sq_pad, self._skv_pad), self.thd, cc)
        mm_lo = load_template(
            _sm100_kernel_path(_SM100_MATMUL_FILE),
            MatmulTemplateParams(
                a_is_m_major=True,
                b_is_n_major=True,
                causal_mode=lo,
                causal_gran=gran,
                causal_shift=shift,
                vec_bytes_epi=vec,
                dtype_qkv=dtype_code,
                thd_varlen=self.thd,
                thd_causal_bottom_right=per_seq,
                cgrp_tile_mn=tile,
            ),
            tag="sdpa_bwd_sm100_mm_lo",
        )
        # dQ = dS . K under GQA: ONE launch per head chunk when the rendering indexes B = K by `h // group` itself
        # (`b_head_group = group`, DQ_SINGLE_LAUNCH), else one launch per group member (`b_head_group = 1`).  THD keeps
        # the per-member loop (this chain's packed dQ host launches per member), and so does a head chunk that is not a
        # whole number of groups (`_sm100_head_chunk` hands out any divisor of H_q under the budget; the host then maps
        # each Q head to its own KV head, so the record must say 1 there).
        # The dense (512, 512) rendering is byte-identical at 1 (every use of the field folds out; the PTX md5 pins).
        p_hi = MatmulTemplateParams(
            a_is_m_major=False,
            b_is_n_major=True,
            causal_mode=hi,
            causal_gran=gran,
            causal_shift=shift,
            vec_bytes_epi=vec,
            dtype_qkv=dtype_code,
            thd_varlen=self.thd,
            b_head_group=self._gqa_group if (DQ_SINGLE_LAUNCH and self._gqa_group > 1 and not self.thd and self._qh_chunk % self._gqa_group == 0) else 1,
            thd_causal_bottom_right=per_seq,
            cgrp_tile_mn=tile,
        )
        # The host launches dQ the way its rendering indexes B: ONE source of truth, the record (`prepared_host._dq_launches`
        # refuses a value that is neither 1 nor the group).
        self._dq_b_head_group = int(p_hi.b_head_group)
        mm_hi = load_template(_sm100_kernel_path(_SM100_MATMUL_FILE), p_hi, tag="sdpa_bwd_sm100_mm_hi")
        if self._prepared_native:
            from .prepared_sm100 import compile_plan

            self._prepared = compile_plan(self, stage2_mod, mm_lo, mm_hi)
            self._compiled = self._prepared
            return self._compiled
        from .prepared_sm100 import compile_staged

        self._staged_prepared = compile_staged(self, stage2_mod, mm_lo, mm_hi)
        self._compiled = self._staged_prepared
        return self._compiled

    # --- execution -----------------------------------------------------------
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
        seq_q_lens: Optional[torch.Tensor] = None,
        seq_kv_lens: Optional[torch.Tensor] = None,
        sink_tensor: Optional[torch.Tensor] = None,
        dsink_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        dbias_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        # Declared so the shared lowering can pass them positionally/by keyword,
        # then refused: the Capabilities row does not claim any of them, so a
        # non-None here means the row and this method disagree.
        for _t_name, _t_val in (("sink", sink_tensor), ("dSink", dsink_tensor), ("bias", bias_tensor), ("dBias", dbias_tensor)):
            self._value_error_if(_t_val is not None, f"SM100 bwd: {_t_name} is not implemented")
        if not self.thd:
            self._value_error_if(seq_q_lens is not None or seq_kv_lens is not None, "SM100 bwd: padding masks (seq lens) are not implemented")

        if self._compiled is None:
            self.compile()
        if self._prepared is not None:
            from .prepared_sm100 import execute_standalone

            return execute_standalone(
                self,
                (q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, stats_tensor, dq_tensor, dk_tensor, dv_tensor, seq_q_lens, seq_kv_lens),
                workspace,
                current_stream,
                scale_softmax,
            )
        from .prepared_sm100 import execute_staged

        return execute_staged(
            self,
            (q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, stats_tensor, dq_tensor, dk_tensor, dv_tensor, seq_q_lens, seq_kv_lens),
            workspace,
            current_stream,
            scale_softmax,
        )
