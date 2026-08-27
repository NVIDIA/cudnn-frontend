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
    ROW_ROUND as _SM120_ROW_ROUND,
    SEQ_KV_TILES as _SM120_KV_TILES,
    SEQ_Q_TILES as _SM120_Q_TILES,
    SUPPORTED_HEAD_DIMS as _SM120_SUPPORTED_HEAD_DIMS,
    TemplateParams as Sm120TemplateParams,
    padded_head_dim as _sm120_padded_head_dim,
    padded_head_dims as _sm120_padded_head_dims,
)
from cudnn.sdpa.fwd.api_dsl import WorkspaceCarver, _torch_stream_context, ws_align

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
