# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""cuDNN-frontend adapter over the FROST DSL SDPA backward kernels."""

from __future__ import annotations

import inspect
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

_SM120_KERNEL_FILE = "bprop_f16_sm120.py"
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
            any(st == 0 and d > 1 for d, st in zip(self.stats_desc.shape, self.stats_desc.stride)),
            f"stats must not broadcast (stride 0 on a size > 1 dim); got stride {self.stats_desc.stride}",
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

        if self.scale_softmax is None or self.scale_softmax == 0.0:
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

    def _checked_seq_lens(self, seq_lens: torch.Tensor, name: str) -> torch.Tensor:
        """Validate per-batch lengths and return a (B,) int32 view (never a copy/cast)."""
        self._value_error_if(
            seq_lens.device != self.q_desc.device,
            f"{name} must be on {self.q_desc.device} (with Q); got {seq_lens.device}",
        )
        self._value_error_if(
            seq_lens.dtype != torch.int32,
            f"{name} must be int32; got {seq_lens.dtype}",
        )
        self._value_error_if(
            seq_lens.numel() != self.batch_size,
            f"{name} must have B = {self.batch_size} elements; got {seq_lens.numel()}",
        )
        self._value_error_if(
            not seq_lens.is_contiguous(),
            f"{name} must be contiguous (bound to the kernel as a flat (B,) view)",
        )
        return seq_lens.reshape(-1)

    def _checked_bias_view(self, tensor: torch.Tensor, desc: TensorDesc, name: str) -> torch.Tensor:
        """Validate a bias/dBias buffer and return the declared (1|B, H_q, S_q, S_kv) view."""
        shape = tuple(int(x) for x in desc.shape)
        self._value_error_if(
            tensor.dtype != desc.dtype,
            f"{name} must be {desc.dtype} (as declared at build); got {tensor.dtype}",
        )
        self._value_error_if(
            tensor.numel() != math.prod(shape),
            f"{name} must have {math.prod(shape)} elements {shape}; got {tensor.numel()}",
        )
        self._value_error_if(
            not tensor.is_contiguous(),
            f"{name} must be contiguous (the kernel accesses it as the declared {shape} view)",
        )
        return tensor.view(shape)

    def _checked_sinks_1d(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """Validate a sink/dSink buffer and return the kernel's (H_q,) fp32 view."""
        self._value_error_if(
            tensor.dtype != torch.float32,
            f"{name} must be float32; got {tensor.dtype}",
        )
        self._value_error_if(
            tensor.numel() != self.h_q,
            f"{name} must have H_q = {self.h_q} elements; got {tensor.numel()}",
        )
        self._value_error_if(
            not tensor.is_contiguous(),
            f"{name} must be contiguous (bound to the kernel as a flat (H_q,) view)",
        )
        return tensor.view(-1)

    def _checked_lse_view(self, tensor: torch.Tensor) -> torch.Tensor:
        """Validate the Stats buffer and return the kernel's (B, H_q, S_q) LSE view."""
        shape = (self.batch_size, self.h_q, self.s_q_max)
        self._value_error_if(
            tensor.dtype != torch.float32,
            f"stats_tensor must be float32; got {tensor.dtype}",
        )
        self._value_error_if(
            tensor.numel() != math.prod(shape),
            f"stats_tensor must have B*H_q*S_q = {math.prod(shape)} elements; got {tensor.numel()}",
        )
        if self._lse_strides is None:
            self._value_error_if(
                not tensor.is_contiguous(),
                "stats_tensor must be contiguous (the kernel was compiled for a contiguous LSE layout)",
            )
            return tensor.view(shape)
        try:
            return tensor.as_strided(shape, self._lse_strides, tensor.storage_offset())
        except RuntimeError as exc:
            raise ValueError(
                f"stats_tensor backing storage is too small for declared shape {shape}, stride {self._lse_strides}, "
                f"and storage_offset {tensor.storage_offset()}"
            ) from exc

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

        self._value_error_if(
            self.seq_kv_lens_present and seq_kv_lens is None,
            "seq_kv_lens is required by this compiled specialization",
        )
        self._value_error_if(
            not self.seq_kv_lens_present and seq_kv_lens is not None,
            "this specialization was compiled without per-batch KV lengths; construct the API with seq_kv_lens_present=True",
        )
        self._value_error_if(
            self.seq_q_lens_present and seq_q_lens is None,
            "seq_q_lens is required by this compiled specialization",
        )
        self._value_error_if(
            not self.seq_q_lens_present and seq_q_lens is not None,
            "this specialization was compiled without per-batch Q lengths; construct the API with seq_q_lens_present=True",
        )
        self._value_error_if(
            self.dsink_desc is not None and (sink_tensor is None or dsink_tensor is None),
            "sink_tensor and dsink_tensor are required by this compiled specialization",
        )
        self._value_error_if(
            self.dsink_desc is None and dsink_tensor is not None,
            "this specialization was compiled without a dSink output; construct the API with sample_dsink",
        )
        self._value_error_if(
            self.bias_desc is not None and bias_tensor is None,
            "bias_tensor is required by this compiled specialization",
        )
        self._value_error_if(
            self.bias_desc is None and bias_tensor is not None,
            "this specialization was compiled without a bias input; construct the API with sample_bias",
        )
        self._value_error_if(
            self.dbias_desc is not None and dbias_tensor is None,
            "dbias_tensor is required by this compiled specialization",
        )
        self._value_error_if(
            self.dbias_desc is None and dbias_tensor is not None,
            "this specialization was compiled without a dBias output; construct the API with sample_dbias",
        )

        scale_val = self.scale_softmax if scale_softmax is None or scale_softmax == 0.0 else float(scale_softmax)
        scale_log2 = scale_val * math.log2(math.e)

        carver = WorkspaceCarver(workspace, self.scratch_workspace_bytes(), "sdpa_bwd_sm120")
        delta = carver.take(self.batch_size * self.h_q * self._sq_rounded, torch.float32).reshape(self.batch_size, self.h_q, self._sq_rounded)
        dq_accum = None
        dq_sem = None
        ds_ws = None
        if self.det_2k:
            ds_ws = carver.take(self._ds_ws_elems(), self.dtype).view(self.batch_size, self.h_q, self.s_q_max, self._skv_rounded)
        else:
            dq_accum = carver.take(self.batch_size * self._sq_rounded * self.h_q * self.head_dim_qk_padded, torch.float32)
            dq_sem = carver.take(self._dq_sem_len(), torch.int32)
        dbias_accum_elems = self._dbias_accum_elems()
        dbias_accum = carver.take(dbias_accum_elems, torch.float32) if dbias_accum_elems else None

        if current_stream is None:
            # Direct call (no dispatch-forwarded stream): fall back to torch's
            # current stream. A stream forwarded from the execute-time handle
            # is respected rather than clobbered.
            current_stream = cuda.CUstream(torch.cuda.current_stream(q_tensor.device).cuda_stream)

        import cutlass

        def _native_view(view: torch.Tensor, name: str) -> torch.Tensor:
            """Rebind the buffer to the compiled strides: execute-time tensors
            are raw storage laid out as declared at build."""
            self._value_error_if(
                view.data_ptr() % 16 != 0,
                f"{name} base address must be 16-byte aligned (TMA global-address requirement)",
            )
            strides = self._io_strides.get(name)
            if strides is None:
                return view
            b, s, h, d = view.shape
            batch_stride, seq_stride, head_stride = strides
            return view.as_strided((b, s, h, d), (batch_stride, seq_stride, head_stride, 1))

        seq_q_t = self._checked_seq_lens(seq_q_lens, "seq_q_lens") if seq_q_lens is not None else None
        seq_kv_t = self._checked_seq_lens(seq_kv_lens, "seq_kv_lens") if seq_kv_lens is not None else None

        with _torch_stream_context(current_stream, q_tensor.device):
            q = _native_view(q_tensor.transpose(1, 2), "q")
            k = _native_view(k_tensor.transpose(1, 2), "k")
            v = _native_view(v_tensor.transpose(1, 2), "v")
            o = _native_view(o_tensor.transpose(1, 2), "o")
            do = _native_view(do_tensor.transpose(1, 2), "dO")
            dq = _native_view(dq_tensor.transpose(1, 2), "dQ")
            dk = _native_view(dk_tensor.transpose(1, 2), "dK")
            dv = _native_view(dv_tensor.transpose(1, 2), "dV")
            lse = self._checked_lse_view(stats_tensor)

            kernels = self._compiled_kernel
            # dK/dV destinations for the main kernel: MHA writes the user
            # outputs directly; GQA reduces per-q-head workspace partials.
            if self.h_q == self.h_kv:
                dk_ws, dv_ws = dk, dv
            else:
                dkw_elems, dvw_elems = self._dkv_ws_elems()
                dk_ws = carver.take(dkw_elems, self.dtype).view(self.batch_size, self.s_k_max, self.h_q, self.head_dim_qk_padded)
                dv_ws = carver.take(dvw_elems, self.dtype).view(self.batch_size, self.s_k_max, self.h_q, self.head_dim_v_padded)

            bias_view = None
            dbias_view = None
            dbias_dst = None
            if self.bias_desc is not None:
                bias_view = self._checked_bias_view(bias_tensor, self.bias_desc, "bias_tensor")
            if self.dbias_desc is not None:
                dbias_view = self._checked_bias_view(dbias_tensor, self.dbias_desc, "dbias_tensor")
                if dbias_accum is None:
                    # fp32 output: the kernel red.adds into it directly.
                    dbias_view.zero_()
                    dbias_dst = dbias_view
                else:
                    dbias_accum.zero_()
                    dbias_dst = dbias_accum.view(dbias_view.shape)

            # Kernel chain: dot -> main -> [reduce] -> cvt -> [dbias_cvt], or on
            # the det_2kernel route dot -> main -> dq2k -> [reduce] -> [dbias_cvt].
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
                ds_ws,
                dk_ws,
                dv_ws,
                seq_q_t,
                seq_kv_t,
                bias_view,
                dbias_dst,
                cutlass.Float32(scale_log2),
                cutlass.Float32(scale_val),
                current_stream,
            )
            if self.det_2k:
                kernels.dq2k(k, ds_ws, dq, cutlass.Float32(scale_val), current_stream)
            # GQA only
            if kernels.reduce is not None:
                kernels.reduce(dk_ws, dv_ws, dk, dv, current_stream)
            if not self.det_2k:
                kernels.cvt(dq_accum, dq, cutlass.Float32(scale_val), current_stream)
            if kernels.dbias_cvt is not None:
                kernels.dbias_cvt(dbias_accum, dbias_view.view(-1), current_stream)
            if kernels.dsink is not None:
                sink_1d = self._checked_sinks_1d(sink_tensor, "sink_tensor")
                dsink_1d = self._checked_sinks_1d(dsink_tensor, "dsink_tensor")
                kernels.dsink(lse, delta, sink_1d, dsink_1d, seq_q_t, current_stream)


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
    (``bprop_f16_sm80``) is a TemplateParams module loaded per-specialization
    via :func:`_load_sm80_bwd_module` instead.
    """
    assert key == "d64", f"generic SM80 bwd kernels load via _load_sm80_bwd_module; got {key!r}"
    if key not in _SM80_BWD_KERNEL_MOD:
        from .kernels import bprop_d64_f16_sm80 as _mod

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


def _sm80_bwd_pad_last_dim(t: torch.Tensor, new_last: int) -> torch.Tensor:
    """Zero-pad the trailing dim of an fp16/bf16 tensor up to ``new_last``."""
    old_last = t.shape[-1]
    if old_last == new_last:
        return t
    if old_last > new_last:
        raise ValueError(f"_sm80_bwd_pad_last_dim: tensor's last dim {old_last} exceeds target {new_last}")
    pad = torch.zeros((*t.shape[:-1], new_last - old_last), dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=-1).contiguous()


def _sm80_thd_backward(
    q, k, v, o, do, lse, *, cu_q, cu_k, scale_softmax, is_causal, window_size, causal_bottom_right, sinks=None, deterministic=False, max_s_kv=None, max_s_q=None
):
    """THD / varlen backward: q/k/v/o/do are PACKED ``[1, T, H, D]`` (BSHD,
    B==1 — no transpose), ``lse`` is packed ``[1, H, T_q]`` (head-major,
    matching the kernel's THD LSE layout), and cu_q/cu_k are ``[n_seq+1]``
    cumulative seqlens.  Loads the THD template specialization (packed token
    totals are ``cute.sym_int`` dynamics — one artifact per (params, n_seq),
    issue #604) and drives the kernel chain directly (over-provisioned grid;
    GQA reduces over the query-head group).  Returns packed ``[1, T, H, D]``
    dQ/dK/dV (BHSD-equivalent for B==1).

    The over-provisioned grid needs the longest per-sequence KV length:
    pass ``max_s_kv`` (any upper bound works — short tiles early-out) to keep
    the call fully async; without it the wrapper reads it from ``cu_k`` on
    the HOST (a D2H sync — the wrapper-only Rule-3 residual).  ``sinks``
    ((H,) natural-log logits) adds a ``dsink_tensor`` output.  ``deterministic``
    sizes the dQ relay counter from ``max_s_q`` (an upper bound on the
    longest per-sequence Q length, trimmed to the packed total; the packed
    total when absent — no D2H).  Like ``max_s_kv``, the hint is a caller
    contract: the wrapper cannot validate it without a D2H read, and an
    undersized value indexes the relay counter out of bounds.
    """
    d_qk = q.shape[-1]
    d_v = v.shape[-1]
    h_q = q.shape[2]
    h_kv = k.shape[2]
    flavor = _sm80_bwd_pick_flavor(d_qk, d_v)
    fdqk, fdv = _SM80_BWD_FLAVOR_DIMS[flavor]
    # Resolve the default scale from the USER's head dim before padding: the
    # kernel would otherwise derive 1/sqrt(D) from the padded flavor width
    # (e.g. 1/sqrt(128) for a d=96 llama-flavor call) — silently wrong
    # gradients.  Mirrors the forward THD path.
    if scale_softmax is None or scale_softmax == 0.0:
        scale_softmax = 1.0 / math.sqrt(d_qk)
    pad_qk = d_qk < fdqk
    pad_v = d_v < fdv
    if pad_qk:
        q = _sm80_bwd_pad_last_dim(q, fdqk)
        k = _sm80_bwd_pad_last_dim(k, fdqk)
    if pad_v:
        v = _sm80_bwd_pad_last_dim(v, fdv)
        o = _sm80_bwd_pad_last_dim(o, fdv)
        do = _sm80_bwd_pad_last_dim(do, fdv)
    # cuDNN's (is_causal, window_size=(left,right)) → mask params.
    wl, wr = window_size
    has_swa = wl is not None and wl >= 0
    swa = int(wl) if has_swa else 0
    right_bound = int(wr) if (is_causal and wr is not None and wr > 0) else 0
    from cudnn.sdpa.bwd.config_sm80 import bwd_params_for_flavor

    # NOTE: llama-swept tiles always (matching the dense adapter — the gptoss
    # wide-Q-tile row stays unwired pending a perf gate); the flavor picks
    # only the ENVELOPE dims, which must reach the compiled kernel (a flavor
    # name alone would leave the template at its 128/128 defaults while the
    # buffers pad to the envelope — OOB at d=64, wrong grads at 192/256).
    params = bwd_params_for_flavor(
        "llama",
        io_bf16=(q.dtype == torch.bfloat16),
        d_qk=fdqk,
        d_v=fdv,
        is_causal=bool(is_causal),
        has_swa=has_swa,
        causal_bottom_right=bool(causal_bottom_right) and (bool(is_causal) or has_swa),
        has_sink=sinks is not None,
        deterministic=bool(deterministic),
        thd_varlen=True,
        sched_policy=_BWD_SCHED_NATURAL,  # LPT+THD is a future tweak
    )
    mod = _load_sm80_bwd_module(params)
    # Host-side grid math.  n_seq is shape metadata (no sync); the longest KV
    # length comes from the caller's max_s_kv hint, or from a host read of
    # cu_k (the D2H documented above) when no hint is given.
    n_seq = cu_q.numel() - 1
    assert cu_k.numel() == n_seq + 1, "cu_seqlens_q / cu_seqlens_k length mismatch"
    if max_s_kv is not None:
        max_s_kv = int(max_s_kv)
        assert max_s_kv > 0, f"max_s_kv must be > 0; got {max_s_kv}"
    else:
        cu_k_host = cu_k.to(dtype=torch.int32, device="cpu")
        max_s_kv = int((cu_k_host[1:] - cu_k_host[:-1]).max())
    c = mod.compile(1, h_q, h_kv, 0, 0, swa_window=swa, n_batch_logical=n_seq)
    t_q = q.shape[1]
    t_kv = k.shape[1]
    dev = q.device
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    q, k, v, o, do = (t.contiguous() for t in (q, k, v, o, do))
    lse_t = lse.to(dtype=torch.float32, device=dev).contiguous()
    cu_q_t = cu_q.to(dtype=torch.int32, device=dev).contiguous()
    cu_k_t = cu_k.to(dtype=torch.int32, device=dev).contiguous()
    dq_acc = torch.zeros(1, t_q, h_q, fdqk, dtype=torch.float32, device=dev)
    dQ_k = torch.empty(1, t_q, h_q, fdqk, dtype=q.dtype, device=dev)
    # dK/dV write buffers carry the h_q query heads (one slice per query head);
    # MHA: they ARE the outputs.  GQA: reduced over the group below.
    dk_ws = torch.empty(1, t_kv, h_q, fdqk, dtype=q.dtype, device=dev)
    dv_ws = torch.empty(1, t_kv, h_q, fdv, dtype=q.dtype, device=dev)
    dot = torch.empty(1, h_q, t_q, dtype=torch.float32, device=dev)
    dummy_i32 = torch.zeros(1, dtype=torch.int32, device=dev)
    dummy_f32 = torch.zeros(1, dtype=torch.float32, device=dev)
    c.do_dot(_fd_tvm(o), _fd_tvm(do), _fd_tvm(dot), _int32(h_q * t_q), stream)
    dsink = None
    if sinks is not None:
        # Natural-log logits (the kernel applies log2e itself); one warp per
        # (sequence, head) row over that sequence's tokens; zero-init target.
        sinks_b = sinks.to(dtype=torch.float32, device=dev).reshape(h_q).contiguous()
        dsink = torch.zeros(h_q, dtype=torch.float32, device=dev)
        c.dsink(_fd_tvm(lse_t), _fd_tvm(dot), _fd_tvm(sinks_b), _fd_tvm(dsink), _fd_tvm(cu_q_t), _int32(n_seq * h_q), stream)
    # Deterministic relay counter: one int32 per (sequence, head, q-tile),
    # sized from the max_s_q hint (any upper bound) or, without one, from the
    # packed total — host shape metadata, so no D2H read.  Fresh zeros every
    # call: a stale counter deadlocks the relay's equality spin.
    if deterministic:
        # The packed total bounds every per-sequence length, so an over-provisioned
        # hint is trimmed to it; an UNDERSIZED hint is a contract violation the
        # wrapper cannot detect without a D2H read of cu_q (same contract as
        # max_s_kv) -- the relay indexes the counter by q-tile, so it would run
        # off the end of dq_sem.
        q_bound = min(int(max_s_q), t_q) if max_s_q is not None else t_q
        assert q_bound > 0, f"max_s_q must be > 0; got {q_bound}"
        sem_q_stride = (q_bound + params.tile_q - 1) // params.tile_q
        dq_sem = torch.zeros(n_seq * h_q * sem_q_stride, dtype=torch.int32, device=dev)
    else:
        sem_q_stride = 0
        dq_sem = dummy_i32
    _sm80_bwd_call(
        c.main,
        q=q,
        k=k,
        v=v,
        do=do,
        dq_acc=dq_acc,
        dk_ws=dk_ws,
        dv_ws=dv_ws,
        lse=lse_t,
        do_dot=dot,
        seq_kv=dummy_i32,
        bias=dummy_f32,
        dbias=dummy_f32,
        rope_cs=dummy_f32,
        cu_q=cu_q_t,
        cu_k=cu_k_t,
        seq_q=dummy_i32,
        dq_sem=dq_sem,
        n_q_tiles=(t_q + params.tile_q - 1) // params.tile_q,
        scale_log2=float(scale_softmax) * _BWD_LOG2E,
        attn_scale=float(scale_softmax),
        right_bound=right_bound,
        inv_scale=1.0 / float(scale_softmax),
        bias_bstride=0,
        sem_q_stride=sem_q_stride,
        grid_kv_tiles=(max_s_kv + params.tile_kv - 1) // params.tile_kv,
        grid_batch=n_seq,
        stream=stream,
    )
    c.cast(_fd_tvm(dq_acc), _fd_tvm(dQ_k), _int32((t_q * h_q * fdqk) // 2), stream)
    if h_q != h_kv:
        dK_k = torch.empty(1, t_kv, h_kv, fdqk, dtype=q.dtype, device=dev)
        dV_k = torch.empty(1, t_kv, h_kv, fdv, dtype=q.dtype, device=dev)
        c.reduce_k(_fd_tvm(dk_ws), _fd_tvm(dK_k), _int32(t_kv * h_kv * fdqk), stream)
        c.reduce_v(_fd_tvm(dv_ws), _fd_tvm(dV_k), _int32(t_kv * h_kv * fdv), stream)
    else:
        dK_k, dV_k = dk_ws, dv_ws
    if pad_qk:
        dQ_k = dQ_k[..., :d_qk].contiguous()
        dK_k = dK_k[..., :d_qk].contiguous()
    if pad_v:
        dV_k = dV_k[..., :d_v].contiguous()
    out = TupleDict(dq_tensor=dQ_k, dk_tensor=dK_k, dv_tensor=dV_k)
    if dsink is not None:
        out["dsink_tensor"] = dsink
    return out


# ---------------------------------------------------------------------------
# Functional wrapper (mirrors the forward surface).
# ---------------------------------------------------------------------------
_cache_of_objects: dict = {}


_SM80_BWD_KERNEL_FILE = "bprop_f16_sm80.py"
# The shared tile_dsl scheduler vocabulary maps identity onto the bwd grid
# decode (NATURAL == plain 3-D == 0, LPT == kv-major == 1).
from cudnn.frost.tile_dsl.constants import SCHED_LPT as _BWD_SCHED_LPT  # noqa: E402
from cudnn.frost.tile_dsl.constants import SCHED_NATURAL as _BWD_SCHED_NATURAL  # noqa: E402


def _load_sm80_bwd_module(params):
    """Load one uniquely named backward kernel module per parameter set."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", _SM80_BWD_KERNEL_FILE)
    return load_template(path, params, tag="sdpa_bwd_sm80")


def _sm80_bwd_call(
    compiled,
    *,
    q,
    k,
    v,
    do,
    dq_acc,
    dk_ws,
    dv_ws,
    lse,
    do_dot,
    seq_kv,
    bias,
    dbias,
    rope_cs,
    cu_q,
    cu_k,
    seq_q,
    dq_sem,
    n_q_tiles,
    scale_log2,
    attn_scale,
    right_bound,
    inv_scale,
    bias_bstride,
    sem_q_stride,
    grid_kv_tiles,
    grid_batch,
    stream,
):
    """Invoke one compiled main-bprop artifact (the traced ``_bprop_host``
    ABI: 17 tensors, then 9 runtime scalars and the launch stream)."""
    import cutlass
    from cutlass.cute.runtime import from_dlpack as _fd

    def fd(t):
        # The kernels compile with --enable-tvm-ffi, so host-side conversions
        # must produce TVM-FFI tensors regardless of the env latch.
        return _fd(t, enable_tvm_ffi=True)

    compiled(
        fd(q),
        fd(k),
        fd(v),
        fd(do),
        fd(dq_acc),
        fd(dk_ws),
        fd(dv_ws),
        fd(lse),
        fd(do_dot),
        fd(seq_kv),
        fd(bias),
        fd(dbias),
        fd(rope_cs),
        fd(cu_q),
        fd(cu_k),
        fd(seq_q),
        fd(dq_sem),
        cutlass.Int32(n_q_tiles),
        cutlass.Float32(scale_log2),
        cutlass.Float32(attn_scale),
        cutlass.Int32(right_bound),
        cutlass.Float32(inv_scale),
        cutlass.Int32(bias_bstride),
        cutlass.Int32(sem_q_stride),
        cutlass.Int32(grid_kv_tiles),
        cutlass.Int32(grid_batch),
        stream,
    )


_BWD_LOG2E = math.log2(math.e)


def _int32(v):
    import cutlass

    return cutlass.Int32(int(v))


def _fd_tvm(t):
    from cutlass.cute.runtime import from_dlpack as _fd

    return _fd(t, enable_tvm_ffi=True)


class SdpaBwdDslSm80(SdpaBwdDsl):
    """SM80 (A100) SDPA backward via the pre-TemplateParams CuTe-DSL kernels.

    Follows the SM120 adapter lifecycle (check_support → compile → execute) on
    top of the self-caching SM80 kernel modules; the TemplateParams conversion
    (the #689 analogue, with issue #604's sym_int THD extents) swaps the
    kernel seam without touching this contract. SM80-only operands (bias →
    dBias, RoPE) arrive as extra optional keywords, as the ``SdpaBwdDsl``
    contract permits.

    Layouts: any dense layout with the head dim innermost-contiguous is
    served — non-BSHD-compact operands (dense_flex) gather into carved
    staging, a strided stats input is READ natively at its declared strides
    (``Capabilities.strided_stats``; no gather), and head dims inside a
    flavor envelope pad host-side into the same carved buffers (issue #514:
    with a workspace provided, execute allocates nothing).
    """

    def __init__(
        self, *args, has_bias: bool = False, bias_is_fp32: bool = True, bias_batch: int = 1, has_rope: bool = False, rope_max_s: int = 0, **kwargs
    ) -> None:
        # SM80-only plan-time facts (scratch sizing + template identity); the
        # base contract carries everything else.
        self._has_bias = bool(has_bias)
        self._bias_is_fp32 = bool(bias_is_fp32)
        self._bias_batch = int(bias_batch)
        self._has_rope = bool(has_rope)
        self._rope_max_s = int(rope_max_s)
        super().__init__(*args, **kwargs)

    def _initialize_implementation(self) -> None:
        self.flavor: Optional[str] = None
        self.flavor_d_qk: Optional[int] = None
        self.flavor_d_v: Optional[int] = None
        self.mask_token: Optional[str] = None
        self.swa_window_runtime: int = 0
        self.right_bound_runtime: int = 0
        self._lse_stride: "Optional[tuple[int, int, int]]" = None
        self._dummy_cache: dict = {}

    def _checked_lse_view(self, lse_tensor: torch.Tensor) -> torch.Tensor:
        """Validate a caller-provided Stats/LSE buffer and return the
        kernels' (B, H_q, S_q) READ view.

        The logical contract is exactly ``B*H_q*S_q`` fp32 elements.  With a
        strided plan (``_lse_stride``), rebuild the DECLARED layout over the
        caller's storage — the kernels' loads were compiled against exactly
        those strides; a contiguous plan requires a contiguous runtime buffer
        (the compiled packed fake would misread anything else).
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

    def _dummy(self, key: str, device: torch.device, factory) -> torch.Tensor:
        """A cached device-local dummy for a dead ABI slot (AGENTS.md Rule 1:
        no per-execute allocation; the matching has_* Constexpr is False so
        the kernel never reads it)."""
        cache_key = (key, device)
        tensor = self._dummy_cache.get(cache_key)
        if tensor is None:
            tensor = factory()
            self._dummy_cache[cache_key] = tensor
        return tensor

    # ------------------------------------------------------------------
    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

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

        if self.scale_softmax is None or self.scale_softmax == 0.0:
            self.scale_softmax = 1.0 / math.sqrt(d_qk)

        self.batch_size = int(b)
        self.s_q_max = int(s_qo)
        self.s_k_max = int(s_kv)
        self.h_q = int(h_qo)
        self.h_kv = int(h_kv)
        self.head_dim_qk = int(d_qk)
        self.head_dim_v = int(d_v)

        self._is_supported = True
        self._logger.debug("check_support completed")
        return True

    # ------------------------------------------------------------------
    def compile(self) -> None:
        """Plan-time JIT: build the TemplateParams from the plan facts, load
        the specialized module via ``frost.template_loader`` (same seam as the
        SM120 adapter), and compile the full kernel chain for this shape.
        The dedicated plain-dense d=64 fast path keeps its self-caching module
        (dense-only — issue #604 concerns the THD extents, which never route
        there); its JIT happens on the first execute."""
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        from cudnn.sdpa.bwd.config_sm80 import bwd_params_for_flavor

        # Scheduler resolution (the old backward()'s "auto"): kv-major LPT for
        # causal load-balance, the plain 3-D grid otherwise; the deterministic
        # relay REQUIRES the plain decode (kv_tile == blockIdx.x).
        sched = _BWD_SCHED_LPT if (self.is_causal and not self.deterministic) else _BWD_SCHED_NATURAL
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
            thd_varlen=False,
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
        # d64 fast-path gate: every input is plan-time state now.  The
        # dedicated kernel keeps legacy PACKED LSE reads, so a strided Stats
        # declaration routes to the generic (stride-aware) module instead.
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
        if self._lse_stride is not None:
            self._use_d64 = False
        if self._use_d64:
            self._kmod = None
            self._compiled_kernel = True  # d64 self-caches on first execute
        else:
            self._kmod = _load_sm80_bwd_module(self._params)
            self._compiled_kernel = self._kmod.compile(
                b=self.batch_size,
                h=self.h_q,
                h_kv=self.h_kv,
                sq=self.s_q_max,
                skv=self.s_k_max,
                swa_window=int(self.swa_window_runtime),
                rope_max_s=self._rope_max_s,
                n_batch_logical=0,
                lse_stride=self._lse_stride,
            )
        self._logger.debug("compile completed")

    # ------------------------------------------------------------------
    def _bshd_gather_bytes(self, desc) -> int:
        """Bytes to gather ``desc`` into a compact BSHD buffer, or 0 when its
        BSHD transpose is already contiguous."""
        b, h, s, d = desc.shape
        if tuple(desc.stride) == (s * h * d, d, h * d, 1):
            return 0
        return ws_align(b * h * s * d * 2)  # fp16/bf16 only on this row

    def scratch_workspace_bytes(self) -> int:
        """Per-execute scratch (issue #514): dense_flex gathers / head-dim pad
        staging for the five input operands, strided-stats staging, and the
        kernel-internal buffers (``bprop_f16_sm80.scratch_bytes``; the generic
        kernel's set covers the d64 fast path's). All plan-time state — no
        arguments."""
        self._ensure_support_checked()
        from .kernels import bprop_f16_sm80 as _kmod

        elem = 2  # fp16/bf16 — check_support admits no other input dtype
        b, hq, hkv = self.batch_size, self.h_q, self.h_kv
        sq, skv = self.s_q_max, self.s_k_max
        fdqk, fdv = self.flavor_d_qk, self.flavor_d_v
        pad_qk = self.head_dim_qk < fdqk
        pad_v = self.head_dim_v < fdv
        total = 0
        # Pad / gather staging, in execute()'s carve order (Q, K, V, O, dO).
        for desc, s_len, hh, pad, fd in (
            (self.q_desc, sq, hq, pad_qk, fdqk),
            (self.k_desc, skv, hkv, pad_qk, fdqk),
            (self.v_desc, skv, hkv, pad_v, fdv),
            (self.o_desc, sq, hq, pad_v, fdv),
            (self.do_desc, sq, hq, pad_v, fdv),
        ):
            if pad:
                total += ws_align(int(desc.shape[0]) * s_len * hh * fd * elem)
            else:
                total += self._bshd_gather_bytes(desc)
        total += _kmod.scratch_bytes(
            B=b,
            SQ=sq,
            SKV=skv,
            H=hq,
            Hk=hkv,
            d_qk=fdqk,
            d_v=fdv,
            io_bytes=elem,
            deterministic=self.deterministic,
            has_bias=self._has_bias,
            bias_batch=self._bias_batch,
            has_sink=self.sink_desc is not None,
            need_do_dot=True,
        )
        return total

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
        """Run the compiled SM80 backward on the adapter's carved workspace (Rule 1:
        no per-call allocation; the dsink launch passes the dense dummy ``cu``).
        """
        self._logger.debug("Entering execute")
        if self._compiled_kernel is None:
            raise RuntimeError("SdpaBwdDslSm80 is not compiled")

        # Init-time flags are compile-time facts; execute must match them
        # exactly, in both directions (Hard Rule 1).
        self._value_error_if(self._has_bias != (bias_tensor is not None), "bias presence must match the plan (has_bias)")
        self._value_error_if(self._has_rope != (rope_freqs is not None), "rope_freqs presence must match the plan (has_rope)")
        self._value_error_if((self.sink_desc is not None) != (sink_tensor is not None), "sink presence must match the plan")
        self._value_error_if(self.seq_kv_lens_present != (seq_kv_lens is not None), "seq_kv_lens presence must match the plan")
        self._value_error_if(self.seq_q_lens_present != (seq_q_lens is not None), "seq_q_lens presence must match the plan")

        scale_val = self.scale_softmax if (scale_softmax is None or scale_softmax == 0.0) else float(scale_softmax)
        device = q_tensor.device
        launch_stream = self._get_default_stream(current_stream)

        # Per-execute scratch: carved from the caller's workspace when one is
        # provided (the engine lowering passes one sized by
        # scratch_workspace_bytes(); issue #514), otherwise allocated (the
        # standalone wrapper path). Carve order mirrors the sizing order:
        # operand staging first, then the kernel-internal buffers.
        carver = WorkspaceCarver(workspace, self.scratch_workspace_bytes(), "SdpaBwdDslSm80") if workspace is not None else None

        def _take(numel, dtype, zero=False, shape=None):
            if carver is not None:
                t = carver.take(numel, dtype)
                if zero:
                    t.zero_()
            else:
                t = (torch.zeros if zero else torch.empty)(numel, dtype=dtype, device=device)
            return t.view(shape) if shape is not None else t

        with _torch_stream_context(current_stream, device):
            pad_qk = self.head_dim_qk < self.flavor_d_qk
            pad_v = self.head_dim_v < self.flavor_d_v

            def _stage(t: torch.Tensor, pad: bool, fd: int) -> torch.Tensor:
                """Kernel-facing BSHD view of BHSD-logical ``t``: zero-copy when
                the transpose is contiguous, else gathered (and head-dim
                padded) into carved staging — or allocated on the wrapper
                path."""
                view = t.transpose(1, 2)
                d = view.shape[-1]
                if pad:
                    bb, ss, hh, _ = view.shape
                    dst = _take(bb * ss * hh * fd, t.dtype, shape=(bb, ss, hh, fd))
                    dst[..., :d].copy_(view)
                    dst[..., d:].zero_()
                    return dst
                if view.is_contiguous():
                    return view
                dst = _take(t.numel(), t.dtype, shape=view.shape)
                dst.copy_(view)
                return dst

            # BHSD → BSHD, in scratch_workspace_bytes()'s sizing order.
            Q = _stage(q_tensor, pad_qk, self.flavor_d_qk)
            K = _stage(k_tensor, pad_qk, self.flavor_d_qk)
            V = _stage(v_tensor, pad_v, self.flavor_d_v)
            O = _stage(o_tensor, pad_v, self.flavor_d_v)
            dO = _stage(do_tensor, pad_v, self.flavor_d_v)

            # Stats → the kernels' (B, H, S) LSE view; squeeze(-1) is a valid
            # view for any (B, H, S, 1) strides.  A strided declaration binds
            # the caller's storage directly — the kernels compiled stride-aware
            # LSE loads against the declared layout (no gather, no copy).
            lse = stats_tensor.squeeze(-1) if stats_tensor.ndim == 4 else stats_tensor
            lse = self._checked_lse_view(lse)

            # --- d64 fast path: the dedicated plain-dense MHA kernel keeps
            # its legacy self-caching module (dense-only; #604 is THD-only).
            if self._use_d64:
                d64 = _sm80_bwd_kernel_mod("d64")
                res = d64.backward(Q, K, V, dO, O, lse, scale=scale_val, workspace=carver.remaining() if carver is not None else None)
                dQ_k, dK_k, dV_k = res[0], res[1], res[2]
                dq_tensor.copy_(dQ_k.transpose(1, 2))
                dk_tensor.copy_(dK_k.transpose(1, 2))
                dv_tensor.copy_(dV_k.transpose(1, 2))
                self._logger.debug("execute completed (d64 fast path)")
                return

            p_ = self._params
            c = self._compiled_kernel
            b, hq, hkv = self.batch_size, self.h_q, self.h_kv
            sq, skv = self.s_q_max, self.s_k_max
            fdqk, fdv = self.flavor_d_qk, self.flavor_d_v
            gqa = hq != hkv

            # Kernel-internal scratch, in the module scratch_bytes() order.
            dsink_acc = _take(hq, torch.float32, zero=True) if sink_tensor is not None else None
            dq_acc = _take(b * sq * hq * fdqk, torch.float32, zero=True, shape=(b, sq, hq, fdqk))
            dq_ws = _take(b * sq * hq * fdqk, self.dtype, shape=(b, sq, hq, fdqk))
            if p_.deterministic:
                sem_units = max(b * hq * c.sem_q_stride, 1)
                dq_sem = _take(sem_units, torch.int32, zero=True)
            else:
                dq_sem = self._dummy("zero_i32", device, lambda: torch.zeros(1, dtype=torch.int32, device=device))
            # dK/dV: MHA at native dims binds the caller's compact-BSHD views
            # directly (no copy-back, mirroring the SM120 adapter); anything
            # else stages (GQA per-q-head partials, head-dim pads, dense_flex).
            dk_view = dk_tensor.transpose(1, 2)
            dv_view = dv_tensor.transpose(1, 2)
            dk_direct = not gqa and not pad_qk and dk_view.is_contiguous()
            dv_direct = not gqa and not pad_v and dv_view.is_contiguous()
            dk_ws = dk_view if dk_direct else _take(b * skv * hq * fdqk, self.dtype, shape=(b, skv, hq, fdqk))
            dv_ws = dv_view if dv_direct else _take(b * skv * hq * fdv, self.dtype, shape=(b, skv, hq, fdv))
            dk_out = dv_out = None
            if gqa:
                dk_out = _take(b * skv * hkv * fdqk, self.dtype, shape=(b, skv, hkv, fdqk))
                dv_out = _take(b * skv * hkv * fdv, self.dtype, shape=(b, skv, hkv, fdv))
            if bias_tensor is not None:
                self._value_error_if(not bias_tensor.is_contiguous(), "bias must be contiguous")
                self._value_error_if(
                    tuple(bias_tensor.shape) != (self._bias_batch, hq, sq, skv),
                    f"bias must be ({self._bias_batch}, {hq}, {sq}, {skv}); got {tuple(bias_tensor.shape)}",
                )
                bias_b = bias_tensor
                dbias_acc = _take(self._bias_batch * hq * sq * skv, torch.float32, zero=True, shape=(self._bias_batch, hq, sq, skv))
            else:
                bias_dt = torch.float32 if p_.bias_is_fp32 else self.dtype
                bias_b = self._dummy(f"one_{bias_dt}", device, lambda: torch.ones(1, dtype=bias_dt, device=device))
                dbias_acc = self._dummy("zero_f32", device, lambda: torch.zeros(1, dtype=torch.float32, device=device))
            dot = _take(b * hq * sq, torch.float32, shape=(b, hq, sq))
            if rope_freqs is not None:
                # (cos, sin) table build — wrapper-only fusion (the engine row
                # never admits RoPE); per-execute by contract, like the caller
                # passing fresh angle tables.
                d2 = fdqk // 2
                rf = rope_freqs.to(dtype=torch.float32, device=device).reshape(rope_freqs.shape[0], -1)
                self._value_error_if(rf.shape[1] < d2, f"rope_freqs last dim ({rf.shape[1]}) must be >= d_qk//2 ({d2})")
                self._value_error_if(
                    rf.shape[0] != self._rope_max_s, f"rope_freqs rows ({rf.shape[0]}) must equal the compiled rope_max_s ({self._rope_max_s})"
                )
                angles = rf[:, :d2]
                rope_b = torch.stack([angles.cos(), angles.sin()], dim=-1).contiguous()
            else:
                rope_b = self._dummy("zero_f32", device, lambda: torch.zeros(1, dtype=torch.float32, device=device))
            if seq_kv_lens is not None:
                seq_kv_b = seq_kv_lens.reshape(-1)
                self._value_error_if(seq_kv_b.dtype != torch.int32 or not seq_kv_b.is_contiguous(), "seq_kv_lens must be contiguous int32")
            else:
                seq_kv_b = self._dummy("zero_i32", device, lambda: torch.zeros(1, dtype=torch.int32, device=device))
            if seq_q_lens is not None:
                seq_q_b = seq_q_lens.reshape(-1)
                self._value_error_if(seq_q_b.dtype != torch.int32 or not seq_q_b.is_contiguous(), "seq_q_lens must be contiguous int32")
            else:
                seq_q_b = self._dummy("zero_i32", device, lambda: torch.zeros(1, dtype=torch.int32, device=device))
            if sink_tensor is not None:
                sinks_b = sink_tensor.reshape(-1)
                self._value_error_if(sinks_b.dtype != torch.float32 or not sinks_b.is_contiguous(), "sinks must be contiguous fp32")
            cu_dummy = self._dummy("zero_i32", device, lambda: torch.zeros(1, dtype=torch.int32, device=device))

            # --- launch chain: do_dot → (dSink) → main → dQ cast → (GQA reduce)
            c.do_dot(_fd_tvm(O), _fd_tvm(dO), _fd_tvm(dot), _int32(b * hq * sq), launch_stream)
            if sink_tensor is not None:
                c.dsink(_fd_tvm(lse), _fd_tvm(dot), _fd_tvm(sinks_b), _fd_tvm(dsink_acc), _fd_tvm(cu_dummy), _int32(b * hq), launch_stream)
            _sm80_bwd_call(
                c.main,
                q=Q,
                k=K,
                v=V,
                do=dO,
                dq_acc=dq_acc,
                dk_ws=dk_ws,
                dv_ws=dv_ws,
                lse=lse,
                do_dot=dot,
                seq_kv=seq_kv_b,
                bias=bias_b,
                dbias=dbias_acc,
                rope_cs=rope_b,
                cu_q=cu_dummy,
                cu_k=cu_dummy,
                seq_q=seq_q_b,
                dq_sem=dq_sem,
                n_q_tiles=(sq + p_.tile_q - 1) // p_.tile_q,
                scale_log2=scale_val * _BWD_LOG2E,
                attn_scale=scale_val,
                right_bound=int(self.right_bound_runtime),
                inv_scale=1.0 / float(scale_val),
                bias_bstride=0 if self._bias_batch == 1 else hq * sq * skv,
                sem_q_stride=c.sem_q_stride,
                grid_kv_tiles=0,
                grid_batch=0,
                stream=launch_stream,
            )
            c.cast(_fd_tvm(dq_acc), _fd_tvm(dq_ws), _int32((b * sq * hq * fdqk) // 2), launch_stream)
            if gqa:
                c.reduce_k(_fd_tvm(dk_ws), _fd_tvm(dk_out), _int32(b * skv * hkv * fdqk), launch_stream)
                c.reduce_v(_fd_tvm(dv_ws), _fd_tvm(dv_out), _int32(b * skv * hkv * fdv), launch_stream)

            # Copy-backs: slice any d-padding, transpose BSHD → BHSD into the
            # caller's buffers (copy_ casts in place; no .to()). Direct-bound
            # dK/dV already landed in place.
            dq_src = dq_ws[..., : self.head_dim_qk] if pad_qk else dq_ws
            dq_tensor.copy_(dq_src.transpose(1, 2))
            dk_src = dk_out if gqa else dk_ws
            dv_src = dv_out if gqa else dv_ws
            if not dk_direct:
                dk_tensor.copy_((dk_src[..., : self.head_dim_qk] if pad_qk else dk_src).transpose(1, 2))
            if not dv_direct:
                dv_tensor.copy_((dv_src[..., : self.head_dim_v] if pad_v else dv_src).transpose(1, 2))
            if dbias_tensor is not None and bias_tensor is not None:
                dbias_tensor.copy_(dbias_acc)
            if dsink_tensor is not None and dsink_acc is not None:
                dsink_tensor.view(-1).copy_(dsink_acc)
        self._logger.debug("execute completed")


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

    THD (``cum_seqlen_*``): pass ``max_s_kv`` (any upper bound on the
    longest per-sequence KV length) to keep the call fully async; without it
    the wrapper reads the max from ``cu_k`` on the host (a D2H sync).  With
    ``deterministic=True``, ``max_s_q`` (an upper bound on the longest
    per-sequence Q length) sizes the dQ relay counter; the packed total is
    used when absent.  Both hints are caller contracts (validating them
    would need the D2H read they exist to avoid): an undersized ``max_s_kv``
    drops KV tiles, an undersized ``max_s_q`` indexes the relay counter out
    of bounds.
    """
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
        with _torch_stream_context(current_stream, q_tensor.device):
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
    dq = torch.empty((b, s_q, h_q, d_qk), dtype=q_tensor.dtype, device=q_tensor.device).transpose(1, 2)
    h_kv, s_kv = k_tensor.shape[1], k_tensor.shape[2]
    dk = torch.empty((b, s_kv, h_kv, d_qk), dtype=q_tensor.dtype, device=q_tensor.device).transpose(1, 2)
    dv = torch.empty((b, s_kv, h_kv, d_v), dtype=q_tensor.dtype, device=q_tensor.device).transpose(1, 2)
    dbias = torch.zeros_like(bias_tensor, dtype=torch.float32) if bias_tensor is not None else None
    dsink = torch.zeros(h_q, dtype=torch.float32, device=q_tensor.device) if sinks is not None else None

    cache_key = (
        q_tensor.shape,
        k_tensor.shape,
        v_tensor.shape,
        q_tensor.stride(),
        k_tensor.stride(),
        v_tensor.stride(),
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
        (bias_tensor.shape[0] if bias_tensor is not None else None),
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
_SM100_STAGE2_FILE = "bprop_d512_f16_sm100.py"
_SM100_MATMUL_FILE = "bprop_matmul_sm100.py"
# do_dot's inner loop is `n_chunks = D_V // chunk_elems` with no tail, so D_V is
# always passed ROUNDED UP to this and the kernel's padded-head-dim guard covers
# the remainder. 64 keeps 8 threads/row, which is a 2x throughput cliff over 32.
_SM100_DOT_CHUNK_ELEMS = 64
_SM100_DOT_Q_TILE = 128
# Workspace budget for S + dS. Above this the head chunk shrinks; the loop then
# runs more launches over the same total work (plan section 5).
_SM100_WS_BUDGET_BYTES = 4 << 30


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


def _sm100_head_chunk_thd(h_q: int, ws_rows: int, s_kv: int, bpe: int, budget: int = _SM100_WS_BUDGET_BYTES, group: int = 1) -> int:
    """``_sm100_head_chunk`` for the BLOCKED workspace.

    Same divisor rule, but a head's slab is ``ws_rows * s_kv`` -- packed q
    tokens rather than ``B * S_q_max``, which is where THD's memory win is.
    """
    per_head = 2 * ws_rows * s_kv * bpe
    cands = [c for c in range(1, h_q + 1) if h_q % c == 0 and c % group == 0]
    for c in sorted(cands, reverse=True):
        if per_head * c <= budget:
            return c
    return group


def _sm100_head_chunk(b: int, h_q: int, s_q: int, s_kv: int, bpe: int, budget: int = _SM100_WS_BUDGET_BYTES, group: int = 1) -> int:
    """Largest DIVISOR of ``h_q`` whose S+dS chunk fits ``budget``.

    A divisor, not a floor: even chunks mean no ragged tail, so one compiled
    artifact serves every chunk and the host just varies ``head_base``. Returns
    1 when even a single head does not fit -- the caller then requests what one
    head needs and the size is still honest.
    """
    per_head = 2 * b * s_q * s_kv * bpe
    # Under GQA the chunk must also be a MULTIPLE OF THE GROUP, so a chunk's Q
    # heads map onto whole KV heads and the dQ group-slice below stays exact.
    cands = [c for c in range(1, h_q + 1) if h_q % c == 0 and c % group == 0]
    for c in sorted(cands, reverse=True):
        if per_head * c <= budget:
            return c
    return group


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
        # d512 bring-up PR; regression test `test_default_attn_scale`.
        if self.scale_softmax is None or self.scale_softmax == 0.0:
            self.scale_softmax = 1.0 / math.sqrt(self.head_dim_qk)
        # Tile-rounded COMPILE shape. The kernel's grid and workspace are tiled,
        # so a sequence length that is not a multiple runs on the next multiple
        # up and the tail is masked. Q/K/V/dO ride their real TMA extents, whose
        # overshoot is HW zero-filled; only S/dS are actually allocated padded.
        self._sq_pad = -(-self.s_q_max // 256) * 256
        self._skv_pad = -(-self.s_k_max // 128) * 128
        self._is_padded = self._sq_pad != self.s_q_max or self._skv_pad != self.s_k_max
        self._compiled = None
        self._dot_fn = None
        self._reduce_fn = None
        self._zero_ws = False
        self._dummy_desc = None
        self._setup_fn = None
        self._thd_lse_token_major = bool(getattr(self, "thd_stats_token_major", False)) and self.thd
        # GQA / MQA: stage 3 cannot write dK/dV straight to the output, because
        # every Q head in a group contributes to the SAME KV head. It writes one
        # partial per Q head and a separate reduce folds the group. group == 1
        # (MHA) skips both the partial buffers and the reduce entirely.
        self._gqa_group = self.h_q // self.h_kv
        self._qh_chunk = _sm100_head_chunk(self.batch_size, self.h_q, self._sq_pad, self._skv_pad, self._bpe, group=self._gqa_group)
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
            self._qh_chunk = _sm100_head_chunk_thd(self.h_q, self._ws_rows_cap, self._skv_pad, self._bpe, group=self._gqa_group)
        # The kernels read/write BSHD-physical buffers. Any io tensor that is not
        # already one gets a staging copy carved from the workspace -- decided
        # HERE, from the descs' declared strides, so scratch_workspace_bytes()
        # stays an honest build-time function. In practice this is usually just
        # dO: a caller that builds it with torch.randn(o.shape) rather than
        # empty_like(o) loses o's memory format and lands BHSD-contiguous.
        self._stage_in = tuple(
            name
            for name, desc in (("q", self.q_desc), ("k", self.k_desc), ("v", self.v_desc), ("o", self.o_desc), ("dO", self.do_desc))
            if not self._bshd_physical_ok(desc)
        )
        self._stage_out = tuple(name for name, desc in (("dQ", self.dq_desc), ("dK", self.dk_desc), ("dV", self.dv_desc)) if not self._bshd_physical_ok(desc))

    # --- capability backstop -------------------------------------------------
    def check_support(self) -> bool:
        """Re-check what the Capabilities row promised.

        Reaching a raise here means the row lied -- these are backstops, not the
        gate (engine contract section 1). They are ValueError, never assert: an
        assert vanishes under -O and an import-time crash is undebuggable from
        the frontend.
        """
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
            self._value_error_if(self.is_causal, "SM100 bwd: THD with a causal mask is not implemented (the stage-3 K-trim is in absolute workspace rows)")
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
            total = delta + 2 * ws
        for name in self._stage_in + self._stage_out:
            s_len = self.s_k_max if name in ("k", "v", "dK", "dV") else self.s_q_max
            total += ws_align(self.batch_size * s_len * self.h_q * self.head_dim_qk * self._bpe)
        if self._gqa_group > 1:
            # One dK and one dV partial per Q head, reduced to the KV heads at
            # the end. Sized on h_q, not h_kv -- that is the whole point.
            total += 2 * ws_align(self.batch_size * self.s_k_max * self.h_q * self.head_dim_qk * self._bpe)
        return total

    # --- compilation ---------------------------------------------------------
    def compile(self) -> None:
        """Plan-time JIT for the whole chain: stage 2's specialized module plus
        the two stage-3 GEMM specializations (dV/dK share one; dQ needs the other
        operand-major and the other causal K-trim direction).  Stage 1 is
        compiled at execute, where the real O/dO tensors are in hand."""
        self._ensure_support_checked()
        if self._compiled is not None:
            return self._compiled
        from cudnn.sdpa.bwd.config_sm100 import (
            CAUSAL_K_HI,
            CAUSAL_K_LO,
            CAUSAL_K_NONE,
            MatmulTemplateParams,
            TemplateParams,
            vec_bytes_epi_for,
        )
        from cudnn.sdpa.bwd.kernels.bprop_chain_f16_sm120 import dot_do_o_host

        dtype_code = DTYPE_BF16 if self.dtype == torch.bfloat16 else DTYPE_FP16
        stage2_mod = load_template(
            _sm100_kernel_path(_SM100_STAGE2_FILE),
            TemplateParams(
                dtype_qkv=dtype_code,
                window_right=(self.window_size_right if self.window_size_right is not None else 0) if self.is_causal else None,
                window_left=self.window_size_left,
                bottom_right=self.causal_bottom_right,
                thd_varlen=self.thd,
            ),
            tag="sdpa_bwd_sm100_stage2",
        )
        gran = stage2_mod.CFG.TILE_M * stage2_mod.CFG.CTA_MMA
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
        # Zero whenever the trim is ACTIVE, not just when shift != 0. TWO
        # independent reasons, either of which alone would require it:
        #   1. the never-empty clamp in `_causal_k_range` means a structurally
        #      masked M tile still reads one k-tile of workspace, and must see
        #      zeros there;
        #   2. stage 3's cluster M tile (512) is WIDER than stage 2's write
        #      block (`gran`, 256), so a per-tile K range cannot exclude the
        #      skipped region at all -- the zero-fill, not the trim, is what
        #      makes the causal path correct. See `_causal_k_range`.
        self._zero_ws = lo != CAUSAL_K_NONE or hi != CAUSAL_K_NONE
        # dtype_qkv must match stage 2's: stage 3 reads the S/dS workspace stage
        # 2 wrote, and stores the gradients in the graph's io dtype.
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
            ),
            tag="sdpa_bwd_sm100_mm_lo",
        )
        mm_hi = load_template(
            _sm100_kernel_path(_SM100_MATMUL_FILE),
            MatmulTemplateParams(
                a_is_m_major=False,
                b_is_n_major=True,
                causal_mode=hi,
                causal_gran=gran,
                causal_shift=shift,
                vec_bytes_epi=vec,
                dtype_qkv=dtype_code,
                thd_varlen=self.thd,
            ),
            tag="sdpa_bwd_sm100_mm_hi",
        )
        stage2 = stage2_mod.compile(
            b=self.batch_size,
            qh=self.h_q,
            # THD ignores these two: the packed totals and the blocked row total
            # are runtime, so the template binds them symbolically.
            sq=self._sq_pad,
            skv=self._skv_pad,
            qh_chunk=self._qh_chunk,
            d=self.head_dim_qk,
            qh_kv=self.h_kv,
            sq_real=self._t_q_cap if self.thd else self.s_q_max,
            skv_real=self._t_kv_cap if self.thd else self.s_k_max,
            lse_token_major=self._thd_lse_token_major,
        )
        self._compiled = (dot_do_o_host, stage2_mod, stage2, mm_lo, mm_hi)
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
        import cutlass
        from cutlass.cute.runtime import from_dlpack, make_fake_stream

        # Declared so the shared lowering can pass them positionally/by keyword,
        # then refused: the Capabilities row does not claim any of them, so a
        # non-None here means the row and this method disagree.
        for _t_name, _t_val in (("sink", sink_tensor), ("dSink", dsink_tensor), ("bias", bias_tensor), ("dBias", dbias_tensor)):
            self._value_error_if(_t_val is not None, f"SM100 bwd: {_t_name} is not implemented")
        if not self.thd:
            self._value_error_if(seq_q_lens is not None or seq_kv_lens is not None, "SM100 bwd: padding masks (seq lens) are not implemented")

        if self.thd:
            return self._execute_thd(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                do_tensor,
                stats_tensor,
                dq_tensor,
                dk_tensor,
                dv_tensor,
                scale_softmax,
                workspace,
                current_stream,
                seq_q_lens,
                seq_kv_lens,
            )

        dot_host, stage2_mod, stage2, mm_lo, mm_hi = self.compile()
        b, h, sq, skv, d = self.batch_size, self.h_q, self.s_q_max, self.s_k_max, self.head_dim_qk
        scale = self.scale_softmax if scale_softmax is None or scale_softmax == 0.0 else float(scale_softmax)
        scale_log2 = scale * math.log2(math.e)

        # Logical BHSD over BSHD storage -> the [B, S, H, D] view the kernels
        # declare. A permutation, never a copy.
        as_bshd = lambda t: t.permute(0, 2, 1, 3)

        stream = self._get_default_stream(current_stream)
        with _torch_stream_context(current_stream, q_tensor.device):
            carver = WorkspaceCarver(workspace, self.scratch_workspace_bytes(), "sdpa_bwd_sm100")
            # ROW_ROUND: dot_do_o_kernel strides delta by ceil(S_q/128)*128.
            sq_dot = -(-sq // 128) * 128
            delta = carver.take(b * h * sq_dot, torch.float32).reshape(b, h, sq_dot)
            chunk = self._qh_chunk
            # Allocated at the PADDED extent, because that is what stage 2's
            # tiled grid writes; stage 3 consumes a REAL-extent slice, so the
            # padded tail never reaches a GEMM's M/N/K at all.
            sqp, skvp = self._sq_pad, self._skv_pad
            s_ws_full = carver.take(b * chunk * sqp * skvp, self.dtype).view(b, chunk, sqp, skvp)
            ds_ws_full = carver.take(b * chunk * sqp * skvp, self.dtype).view(b, chunk, sqp, skvp)
            s_ws, ds_ws = s_ws_full[:, :, :sq, :skv], ds_ws_full[:, :, :sq, :skv]

            # Layout normalisation. A conforming tensor is used in place (the
            # permute is a view); a non-conforming one gets a BSHD staging buffer
            # from the workspace -- copied IN for reads, and copied back OUT
            # after the chain for the gradients.
            def _stage(t, name, is_out):
                if name not in (self._stage_out if is_out else self._stage_in):
                    return as_bshd(t), None
                bt, ht, st, dd = t.shape[0], t.shape[1], t.shape[2], t.shape[3]
                # `buf` is the BSHD-physical [B, S, H, D] buffer the kernels
                # want; `view` is its logical-BHSD alias, used only to copy
                # against the caller's tensor. Returning `view` here instead of
                # `buf` hands the kernel [B, H, S, D] and it rejects the shape.
                buf = carver.take(int(bt) * int(st) * int(ht) * int(dd), self.dtype).view(int(bt), int(st), int(ht), int(dd))
                view = buf.permute(0, 2, 1, 3)
                if not is_out:
                    view.copy_(t)
                return buf, (view, t)

            q, _ = _stage(q_tensor, "q", False)
            k, _ = _stage(k_tensor, "k", False)
            v, _ = _stage(v_tensor, "v", False)
            o, _ = _stage(o_tensor, "o", False)
            do, _ = _stage(do_tensor, "dO", False)
            dq, dq_back = _stage(dq_tensor, "dQ", True)
            dk, dk_back = _stage(dk_tensor, "dK", True)
            dv, dv_back = _stage(dv_tensor, "dV", True)

            # Under GQA the stage-3 dK/dV GEMMs write ONE PARTIAL PER Q HEAD and
            # a separate reduce folds the group afterwards. Writing straight to
            # dk/dv would have the group's Q heads overwrite each other instead
            # of summing. MHA (group == 1) skips both the buffers and the reduce.
            # Zero the S/dS workspace when the stage-3 K-trim can straddle the
            # boundary of what stage 2 wrote.
            #
            # With shift == 0 it cannot: the trim starts at (m0 // 256) * 256 ==
            # m0, and every q-cluster from there wrote the whole 256-wide M tile,
            # so the skipped tiles are never read (the poisoned-workspace test
            # proves it at 43.8% skipped). A non-zero shift -- band widening or
            # bottom-right -- pulls the start BELOW m0, and then an M tile can
            # straddle the written boundary and read residue. That showed up as a
            # 1-in-6 catastrophic dK/dV (cos 0.0006), not a numerics drift,
            # because it depends on whatever was in the buffer.
            if self._zero_ws:
                # NOT for padding: stage 2 writes its own zeros there. Its kv
                # bound is div_up(REAL S_kv, TILE_N), so the tail tile IS visited
                # and apply_mask_chunk zeroes the columns past the real length;
                # the padded q rows are visited too (the grid is sized on the
                # rounded S_q) and row_scale zeroes them. Only a MASK-SKIPPED
                # tile is genuinely never written.
                s_ws_full.zero_()
                ds_ws_full.zero_()

            gqa = self._gqa_group > 1
            if gqa:
                dk_part = carver.take(b * skv * h * d, self.dtype).view(b, skv, h, d)
                dv_part = carver.take(b * skv * h * d, self.dtype).view(b, skv, h, d)
                # Already [B, S_kv, H_q, D] -- the same BSHD orientation `_stage`
                # hands back, so the `hs` head slice below hits the same axis.
                dk_tgt, dv_tgt = dk_part, dv_part
            else:
                dk_tgt, dv_tgt = dk, dv

            _t = lambda x: from_dlpack(x, assumed_align=16, enable_tvm_ffi=True)
            # STAGE 1, hoisted out of the chunk loop: one streaming pass over O
            # and dO instead of n_chunks passes. Nothing in the loop feeds it.
            #
            # The compiled artifact is CACHED on the adapter. Building it here on
            # every call cost ~250 ms of WALL time per execute while device time
            # stayed at 0.1 ms -- invisible to a device-time benchmark, ruinous
            # for anything that measures the real call.
            if self._dot_fn is None:
                d_padded = -(-d // _SM100_DOT_CHUNK_ELEMS) * _SM100_DOT_CHUNK_ELEMS
                self._dot_fn = cutlass.cute.compile(
                    dot_host,
                    _t(o),
                    _t(do),
                    _t(delta),
                    None,
                    None,
                    _SM100_DOT_Q_TILE,
                    d_padded,
                    d_padded,
                    _SM100_DOT_CHUNK_ELEMS,
                    False,
                    False,
                    make_fake_stream(use_tvm_ffi_env_stream=False),
                    options="--enable-tvm-ffi",
                )
            self._dot_fn(_t(o), _t(do), _t(delta), None, None, stream)

            lse = stats_tensor.reshape(b, h, sq)
            seq_kv = torch.full((b,), skv, device=q.device, dtype=torch.int32)
            # Stage 2's THD ABI slots, unused on this dense path: the metadata
            # buffer rides `seq_kv` (which the dense kernel never reads) and the
            # descriptor array is a 1-element dummy. Cached on the adapter so a
            # per-execute call does not allocate.
            if self._dummy_desc is None or self._dummy_desc.device != q.device:
                self._dummy_desc = torch.zeros(1, dtype=torch.int64, device=q.device)
            desc_words = self._dummy_desc
            for c in range(h // chunk):
                hb = c * chunk
                hs = slice(hb, hb + chunk)
                # STAGE 2: head_base offsets every full-tensor read; the S/dS
                # workspace stays chunk-local at origin.
                # Stage 2 gets the FULL padded workspace and the REAL seq
                # lengths; it computes the tail tile and masks it.
                stage2(
                    q,
                    k,
                    v,
                    do,
                    s_ws_full,
                    ds_ws_full,
                    lse,
                    delta,
                    seq_kv,
                    desc_words,
                    # The trailing 0 is N_THD_UNITS: the persistent grid's
                    # cluster count, which only the THD path uses.
                    (b, h, sqp, skvp, chunk, self.h_kv, sq, skv, 0),
                    float(scale),
                    float(scale_log2),
                    float(scale),
                    hb,
                    0,
                    stream,
                )
                # STAGE 3: consume the chunk workspace, write the full output's
                # head slice. Every operand is a permuted view.
                mm_lo.matmul_bh(
                    s_ws.permute(3, 2, 1, 0),
                    do[:, :, hs, :].permute(3, 1, 2, 0),
                    dv_tgt[:, :, hs, :].permute(1, 3, 2, 0),
                    n_head=chunk,
                    n_batch=b,
                    stream=stream,
                    meta=seq_kv,
                    desc_words=desc_words,
                )
                mm_lo.matmul_bh(
                    ds_ws.permute(3, 2, 1, 0),
                    q[:, :, hs, :].permute(3, 1, 2, 0),
                    dk_tgt[:, :, hs, :].permute(1, 3, 2, 0),
                    n_head=chunk,
                    n_batch=b,
                    stream=stream,
                    meta=seq_kv,
                    desc_words=desc_words,
                )
                # dQ = dS.K. Under GQA the K head is shared by `group` Q heads,
                # so the GEMM runs once per group MEMBER: taking every `group`-th
                # Q head lines A and the output up with the KV heads exactly, and
                # every operand stays a strided view (no expand, no copy).
                kv_lo, kv_n = hb // self._gqa_group, chunk // self._gqa_group
                kvs = slice(kv_lo, kv_lo + kv_n)
                for gi in range(self._gqa_group):
                    a_g = ds_ws[:, gi :: self._gqa_group] if gqa else ds_ws
                    o_g = dq[:, :, hs, :][:, :, gi :: self._gqa_group, :] if gqa else dq[:, :, hs, :]
                    mm_hi.matmul_bh(
                        a_g.permute(2, 3, 1, 0),
                        k[:, :, kvs, :].permute(3, 1, 2, 0),
                        o_g.permute(1, 3, 2, 0),
                        n_head=kv_n,
                        n_batch=b,
                        stream=stream,
                        meta=seq_kv,
                        desc_words=desc_words,
                    )

            if gqa:
                # Fold the Q-head partials onto the KV heads. Reused verbatim
                # from the SM120 chain: arch-neutral, one thread per 16 B output
                # vector, fixed-order fp32 accumulation (so it is deterministic).
                from cudnn.sdpa.bwd.kernels.bprop_chain_f16_sm120 import dkv_reduce_host

                io_dt = cutlass.BFloat16 if self.dtype == torch.bfloat16 else cutlass.Float16
                if self._reduce_fn is None:
                    self._reduce_fn = cutlass.cute.compile(
                        dkv_reduce_host,
                        _t(dk_part),
                        _t(dv_part),
                        _t(dk),
                        _t(dv),
                        d,
                        d,
                        self._gqa_group,
                        io_dt,
                        False,
                        make_fake_stream(use_tvm_ffi_env_stream=False),
                        options="--enable-tvm-ffi",
                    )
                self._reduce_fn(_t(dk_part), _t(dv_part), _t(dk), _t(dv), stream)

            for back in (dq_back, dk_back, dv_back):
                if back is not None:
                    staged, dest = back
                    dest.copy_(staged)

    # --- THD / varlen execution ---------------------------------------------
    def _execute_thd(
        self,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        do_tensor,
        stats_tensor,
        dq_tensor,
        dk_tensor,
        dv_tensor,
        scale_softmax,
        workspace,
        current_stream,
        seq_q_lens,
        seq_kv_lens,
    ) -> None:
        """Packed/varlen chain: setup -> do_dot -> S/dS -> the three GEMMs.

        Q/K/V/O/dO and the gradients arrive as logical ``[1, H, T, D]`` views
        over PACKED storage -- the same orientation the dense path takes, with
        the batch collapsed to one and the sequence axis carrying every
        sequence's tokens end to end.  ``seq_*_lens`` are per-batch lengths
        ``(B,)`` or cu prefixes ``(B+1,)``; which one is a per-side bit in
        ``lens_form``, and the setup launch normalises both into the metadata
        buffer.

        Nothing here reads a length on the host.  The packed totals, the
        per-sequence offsets and the blocked-workspace row offsets are all
        device values (issue #552), which is why the grid is occupancy-sized and
        the kernels clamp their own descriptors.
        """
        import cutlass
        from cutlass.cute.runtime import from_dlpack, make_fake_stream

        from cudnn.sdpa.bwd.kernels.thd_helpers import thd_bwd_setup_host as _thd_setup_host

        self._value_error_if(seq_q_lens is None or seq_kv_lens is None, "SM100 bwd THD: seq_q_lens and seq_kv_lens are required")
        dot_host, stage2_mod, stage2, mm_lo, mm_hi = self.compile()
        b, h, d = self.batch_size, self.h_q, self.head_dim_qk
        chunk, n_kv_cols = self._qh_chunk, self._skv_pad
        scale = self.scale_softmax if scale_softmax is None or scale_softmax == 0.0 else float(scale_softmax)
        scale_log2 = scale * math.log2(math.e)
        as_bshd = lambda t: t.permute(0, 2, 1, 3)

        stream = self._get_default_stream(current_stream)
        with _torch_stream_context(current_stream, q_tensor.device):
            carver = WorkspaceCarver(workspace, self.scratch_workspace_bytes(), "sdpa_bwd_sm100_thd")
            t_q_cap, t_kv_cap = self._t_q_cap, self._t_kv_cap
            t_dot = -(-t_q_cap // 128) * 128
            delta = carver.take(h * t_dot, torch.float32).reshape(1, h, t_dot)
            s_ws = carver.take(chunk * self._ws_rows_cap * n_kv_cols, self.dtype).view(1, chunk, self._ws_rows_cap, n_kv_cols)
            ds_ws = carver.take(chunk * self._ws_rows_cap * n_kv_cols, self.dtype).view(1, chunk, self._ws_rows_cap, n_kv_cols)
            meta = carver.take(5 * b + 5, torch.int32)
            desc2 = carver.take(_THD_STAGE2_DESC_SLOTS * 16, torch.int64)
            # ONE stage-3 scratch for all three GEMMs: each patches it
            # immediately before its own launch, and they are sequential on this
            # stream, so the kernel boundaries serialise the reuse.
            desc3 = carver.take((b + 1) * 16, torch.int64)

            q, k, v = as_bshd(q_tensor), as_bshd(k_tensor), as_bshd(v_tensor)
            o, do = as_bshd(o_tensor), as_bshd(do_tensor)
            dq, dk, dv = as_bshd(dq_tensor), as_bshd(dk_tensor), as_bshd(dv_tensor)

            # enable_tvm_ffi matches the `--enable-tvm-ffi` the artifacts below
            # are compiled with; without it the call boundary rejects the tensor.
            _t = lambda x: from_dlpack(x, assumed_align=16, enable_tvm_ffi=True)
            _i32 = lambda x: x.to(dtype=torch.int32, device=q.device).contiguous()
            ql, kl = _i32(seq_q_lens), _i32(seq_kv_lens)
            # lens_form: bit 0 = Q side is a cu prefix, bit 1 = KV side is.
            lens_form = (1 if ql.numel() == b + 1 else 0) | (2 if kl.numel() == b + 1 else 0)
            # Occupancy-sized persistent grid; the DEVICE live-unit total in the
            # metadata is what actually stops the claim loop.
            gran = stage2_mod.CFG.TILE_M * stage2_mod.CFG.CTA_MMA
            units_bound = (-(-t_q_cap // gran) + b) * h
            n_units = max(1, min(units_bound, _sm100_device_clusters(q.device, stage2_mod.CFG.CGA_M)))

            if self._setup_fn is None:
                self._setup_fn = cutlass.cute.compile(
                    _thd_setup_host,
                    _t(meta),
                    _t(ql),
                    _t(kl),
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    make_fake_stream(use_tvm_ffi_env_stream=False),
                    options="--enable-tvm-ffi",
                )
            self._setup_fn(
                _t(meta),
                _t(ql),
                _t(kl),
                cutlass.Int32(lens_form),
                cutlass.Int32(h),
                cutlass.Int32(b),
                cutlass.Int32(_SM100_WS_BLOCK_ROWS),
                cutlass.Int32(gran),
                cutlass.Int32(n_units),
                stream,
            )

            # STAGE 1, over the packed tokens with a batch extent of 1: delta is
            # a per-ROW quantity, so sequence boundaries do not enter it.
            if self._dot_fn is None:
                d_padded = -(-d // _SM100_DOT_CHUNK_ELEMS) * _SM100_DOT_CHUNK_ELEMS
                self._dot_fn = cutlass.cute.compile(
                    dot_host,
                    _t(o),
                    _t(do),
                    _t(delta),
                    None,
                    None,
                    _SM100_DOT_Q_TILE,
                    d_padded,
                    d_padded,
                    _SM100_DOT_CHUNK_ELEMS,
                    False,
                    False,
                    make_fake_stream(use_tvm_ffi_env_stream=False),
                    options="--enable-tvm-ffi",
                )
            self._dot_fn(_t(o), _t(do), _t(delta), None, None, stream)

            lse = stats_tensor.reshape(t_q_cap, h) if self._thd_lse_token_major else stats_tensor.reshape(1, h, t_q_cap)
            gqa = self._gqa_group > 1
            self._value_error_if(gqa, "SM100 bwd THD: GQA is not implemented yet (the dK/dV partials need packed per-Q-head buffers)")

            for c in range(h // chunk):
                hb = c * chunk
                hs = slice(hb, hb + chunk)
                stage2(
                    q,
                    k,
                    v,
                    do,
                    s_ws,
                    ds_ws,
                    lse,
                    delta,
                    meta,
                    desc2,
                    (b, h, self._ws_rows_cap, n_kv_cols, chunk, self.h_kv, 0, 0, n_units),
                    float(scale),
                    float(scale_log2),
                    float(scale),
                    hb,
                    0,
                    stream,
                )
                # STAGE 3.  `m` is the LONGEST sequence's extent: the grid covers
                # it and a shorter sequence's extra tiles are dropped by its own
                # clipped output descriptor.
                mm_lo.matmul_bh(
                    s_ws.permute(3, 2, 1, 0),
                    do[:, :, hs, :].permute(3, 1, 2, 0),
                    dv[:, :, hs, :].permute(1, 3, 2, 0),
                    n_head=chunk,
                    n_batch=b,
                    stream=stream,
                    meta=meta,
                    desc_words=desc3,
                    grid_m=self.s_k_max,
                )
                mm_lo.matmul_bh(
                    ds_ws.permute(3, 2, 1, 0),
                    q[:, :, hs, :].permute(3, 1, 2, 0),
                    dk[:, :, hs, :].permute(1, 3, 2, 0),
                    n_head=chunk,
                    n_batch=b,
                    stream=stream,
                    meta=meta,
                    desc_words=desc3,
                    grid_m=self.s_k_max,
                )
                mm_hi.matmul_bh(
                    ds_ws.permute(2, 3, 1, 0),
                    k[:, :, hs, :].permute(3, 1, 2, 0),
                    dq[:, :, hs, :].permute(1, 3, 2, 0),
                    n_head=chunk,
                    n_batch=b,
                    stream=stream,
                    meta=meta,
                    desc_words=desc3,
                    grid_m=self.s_q_max,
                )
