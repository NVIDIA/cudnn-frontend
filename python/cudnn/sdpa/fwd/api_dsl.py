# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""cuDNN-frontend adapter over the Frost DSL SDPA prefill kernels."""

from __future__ import annotations

import logging
import math
import os
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Hashable, Iterator, Optional

import torch
from cuda.bindings import driver as cuda

from cudnn.api_base import APIBase, TensorDesc, TupleDict
from cudnn.frost.template_loader import load_template
from cudnn.frost.tile_dsl.constants import (
    DTYPE_BF16,
    DTYPE_E4M3,
    DTYPE_E5M2,
    DTYPE_FP16,
    MASK_CAUSAL,
    MASK_NONE,
    MASK_PADDED,
    MASK_SWA,
    SCHED_LPT,
    SCHED_NATURAL,
)
from cudnn.sdpa.fwd.config_sm100 import TemplateParams as Sm100TemplateParams
from cudnn.sdpa.fwd.config_sm120 import (
    SEQ_KV_TILES as _SM120_KV_TILES,
    SEQ_Q_TILES as _SM120_Q_TILES,
    SUPPORTED_HEAD_TILES as _SM120_SUPPORTED_HEAD_TILES,
    TemplateParams as Sm120TemplateParams,
)

_SM100_FLAVORS = (
    (128, 128),
    (192, 128),
    (256, 256),
    (512, 512),
)  # ordered smallest-first: (max D_QK, max D_V) envelope
_SM100_KERNEL_FILES = {
    (512, 512): "prefill_d512_f16_sm100.py",
    (256, 256): "prefill_d256_f16_sm100.py",
    (192, 128): "prefill_d192_d128_f16_sm100.py",
    (128, 128): "prefill_d128_f16_sm100.py",
}
# DTYPE_* codes: E4M3=0, E5M2=1, BF16=2, FP16=3. FP8 inputs (0/1) route to the
# block-scale MXFP8 kernel (d128 only); the output dtype is encoded the same way.
_SM100_DTYPE_QKV_CODE = {
    torch.float8_e4m3fn: DTYPE_E4M3,
    torch.float8_e5m2: DTYPE_E5M2,
    torch.bfloat16: DTYPE_BF16,
    torch.float16: DTYPE_FP16,
}
_SM100_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
# d128 FP8 kernels (E4M3/E5M2 in, BF16/FP16/FP8 out). Block-scale MXFP8 (per-32-block
# E8M0 SF) vs per-tensor FP8 (scalar descales). Selected by the graph op (sdpa_mxfp8 vs
# sdpa_fp8); the f16/bf16 flavors use _SM100_KERNEL_FILES.
_SM100_MXFP8_KERNEL_FILE = "prefill_d128_mxfp8_sm100.py"
_SM100_FP8_KERNEL_FILE = "prefill_d128_fp8_sm100.py"
# Both flavors tile KV in TILE_N=128 columns; the KV tail is only masked when
# the padded/causal mask paths are active (see check_support).
_SM100_TILE_N = 128

_SM120_KERNEL_FILE = "prefill_f16_sm120.py"
_SM120_DTYPE_QKV_CODE = {
    torch.bfloat16: DTYPE_BF16,
    torch.float16: DTYPE_FP16,
}

# Workspace-carve chunk alignment. The contract minimum is 16 bytes; 128 is
# used so the per-sequence O TMA descriptors carved for the THD path satisfy
# the cuTensorMap GMEM alignment (64 B) with margin. torch storage bases are
# 512 B aligned, so 128 B-multiple offsets stay 128 B aligned absolutely.
_WS_ALIGN = 128


@contextmanager
def _torch_stream_context(current_stream: Optional[cuda.CUstream], device: torch.device) -> Iterator[None]:
    """Run PyTorch work on the CUDA stream used for the kernel launch."""
    if current_stream is None:
        yield
        return
    handle = int(current_stream)
    torch_current = torch.cuda.current_stream(device)
    torch_default = torch.cuda.default_stream(device)
    if handle == torch_current.cuda_stream:
        launch_stream = torch_current
    elif handle == torch_default.cuda_stream:
        launch_stream = torch_default
    else:
        launch_stream = torch.cuda.ExternalStream(handle, device=device)
    with torch.cuda.stream(launch_stream):
        yield


def ws_align(nbytes: int) -> int:
    """Round a scratch-chunk size up to the carve alignment (128 B)."""
    return -(-int(nbytes) // _WS_ALIGN) * _WS_ALIGN


class WorkspaceCarver:
    """Carves fixed-size, aligned scratch views out of the CALLER's workspace.

    FROST executor contract (see ``engine._FrostSdpaFwdPlan``): an executor that
    records a non-zero ``workspace_bytes`` is handed the caller's workspace
    buffer (``ExecutionContext.workspace``) at execute and
    carves its per-execute scratch from it instead of allocating. Chunks are
    dealt sequentially at 128-byte relative alignment and never reach beyond
    the buffer; an absent, non-torch, or undersized buffer raises immediately
    with the required size in the message (never silent corruption).
    """

    def __init__(self, workspace, required: int, owner: str):
        if workspace is None:
            raise ValueError(
                f"cudnn.sdpa: {owner} requires a {required}-byte workspace but execute() "
                f"received none; allocate graph.get_workspace_size() bytes (uint8, on the "
                f"graph's device) and pass the buffer to execute()"
            )
        if not (hasattr(workspace, "numel") and hasattr(workspace, "element_size") and hasattr(workspace, "view")):
            raise TypeError(f"cudnn.sdpa: {owner} carves its scratch out of the caller's workspace and needs a torch.Tensor; got {type(workspace).__name__}")
        flat = workspace if workspace.dtype == torch.uint8 else workspace.view(torch.uint8)
        flat = flat.reshape(-1)
        if flat.numel() < required:
            raise ValueError(
                f"cudnn.sdpa: {owner} requires a {required}-byte workspace; the provided "
                f"buffer has only {flat.numel()} bytes (size it with graph.get_workspace_size())"
            )
        if flat.data_ptr() % 16 != 0:
            raise ValueError(f"cudnn.sdpa: {owner} workspace must be at least 16-byte aligned; got data_ptr=0x{flat.data_ptr():x}")
        self._flat = flat
        self._off = 0
        self._owner = owner

    def take(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        """The next scratch chunk: a 1-D ``numel``-element view of ``dtype``."""
        nbytes = int(numel) * dtype.itemsize
        start, end = self._off, self._off + nbytes
        if end > self._flat.numel():
            raise ValueError(f"cudnn.sdpa: {self._owner} workspace overrun: chunk [{start}, {end}) exceeds the {self._flat.numel()}-byte buffer (sizing bug)")
        self._off = start + ws_align(nbytes)
        try:
            return self._flat[start:end].view(dtype)
        except RuntimeError as exc:
            raise ValueError(f"cudnn.sdpa: {self._owner} workspace is not sufficiently aligned for {dtype} scratch: {exc}") from None

    def remaining(self) -> torch.Tensor:
        """The unconsumed tail (uint8) — handed down to a nested carver."""
        return self._flat[self._off :]


def _flavor_tag(flavor: tuple[int, int]) -> str:
    d_qk, d_v = flavor
    return f"d{d_qk}" if d_qk == d_v else f"d{d_qk}_d{d_v}"


def _pick_flavor(d_qk: int, d_v: int) -> tuple[int, int]:
    """Smallest flavor whose envelope covers ``(d_qk, d_v)`` (f16/bf16 only).

    ENVELOPE (zero-padding) semantics: one flavor covers ``d_qk`` and ``d_v``
    with its own max extents — e.g. (192, 128) runs on the d192/d128 kernel. The
    kernel's TMA descriptors are built from the ACTUAL tensor extents while
    the tile box stays the compile-time D, so loads past d_qk / d_v hardware
    zero-fill (adding exact zero terms to every QK^T dot product — S, softmax
    and P·V are bit-identical to the unpadded problem) and O stores past d_v
    are OOB-clipped. FP8/MXFP8 stays exact-match d128 (gated in
    check_support); alignment (d % 8, the TMA 16-byte global-stride rule at
    2 bytes/elem) is also gated in check_support / engines.mismatch.
    """
    for flavor in _SM100_FLAVORS:
        fdqk, fdv = flavor
        if d_qk <= fdqk and d_v <= fdv:
            return flavor
    raise ValueError(
        f"Frost SM100 DSL SDPA: no flavor envelope covers (D_QK={d_qk}, D_V={d_v}); "
        f"largest supported: {_SM100_FLAVORS[-1]} (d128/d192-d128/d256/d512 envelopes)."
    )


def _load_kernel_template(filename: str, params: Hashable, tag: str):
    """Load one uniquely named kernel module per template parameter set."""

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", filename)
    return load_template(path, params, tag=tag)


def _load_sm100_kernel_module(flavor: tuple[int, int], params: Sm100TemplateParams, fp8: bool = False, pertensor: bool = False):
    """Load one SM100 module for the selected flavor and quantization path."""

    tag = _flavor_tag(flavor)
    if fp8:
        filename = _SM100_FP8_KERNEL_FILE if pertensor else _SM100_MXFP8_KERNEL_FILE
        tag = f"sdpa_fwd_sm100_{'fp8' if pertensor else 'mxfp8'}_{tag}"
    else:
        filename = _SM100_KERNEL_FILES[flavor]
        tag = f"sdpa_fwd_sm100_{tag}"
    return _load_kernel_template(filename, params, tag)


def _load_sm120_kernel_module(params: Sm120TemplateParams):
    return _load_kernel_template(_SM120_KERNEL_FILE, params, tag="sdpa_fwd_sm120")


class SdpaFwdDsl(APIBase):
    """Implementation-agnostic interface for FROST DSL SDPA-forward kernels."""

    def __init__(
        self,
        sample_q: torch.Tensor | TensorDesc,
        sample_k: torch.Tensor | TensorDesc,
        sample_v: torch.Tensor | TensorDesc,
        sample_o: torch.Tensor | TensorDesc,
        sample_lse: Optional[torch.Tensor | TensorDesc] = None,
        is_causal: bool = False,
        causal_bottom_right: bool = False,
        window_size_left: Optional[int] = None,
        scale_softmax: Optional[float] = None,
        seq_kv_lens_present: bool = False,
        seq_q_lens_present: bool = False,
        has_sink: bool = False,
        thd: bool = False,
        dtype_o: Optional[torch.dtype] = None,
        pertensor_fp8: bool = False,
        sched_policy: Optional[int] = None,
        tile_m: Optional[int] = None,
        tile_n: Optional[int] = None,
        cga: Optional[int] = None,
    ) -> None:
        """Capture the common SDPA operation and tuning contract.

        Optional operands are accepted by every adapter. A concrete
        implementation that cannot lower one raises :class:`NotImplementedError`
        from ``check_support``.
        """

        super().__init__()
        self._warn_experimental_api()
        self._logger.debug("Entering __init__")

        self.q_desc = self._make_tensor_desc(sample_q, name="q")
        self.k_desc = self._make_tensor_desc(sample_k, name="k")
        self.v_desc = self._make_tensor_desc(sample_v, name="v")
        self.o_desc = self._make_tensor_desc(sample_o, name="o")
        self.lse_desc = self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_lse, name="lse"), 3, "lse")

        self.is_causal = bool(is_causal)
        self.causal_bottom_right = bool(causal_bottom_right)
        # window_size_left is an offset W ("keep k in [q-W, q]"); callers pass W = L - 1 for a cuDNN window length L.
        self.window_size_left = window_size_left
        self.scale_softmax = scale_softmax
        self.seq_kv_lens_present = bool(seq_kv_lens_present)
        # Dense padded-Q trim (cuDNN >= 9.14): q rows >= seq_len_q[b] write
        # O := 0 / LSE := -inf. The per-batch Q lengths are a SEPARATE
        # (B,)-int32 kernel parameter (like cuDNN's SEQLEN_Q pointer and
        # FA's seqused_q) bound directly at execute — no packing, no
        # per-execute copies. Dense-only.
        self.seq_q_lens_present = bool(seq_q_lens_present)
        self.has_sink = bool(has_sink)
        self.thd = bool(thd)
        # MXFP8: FP8 (E4M3/E5M2) Q/K/V in, half (BF16/FP16) O out. dtype_o overrides
        # the output dtype; None inherits Q's dtype. _fp8 is set in check_support once
        # Q's dtype is known.
        self.dtype_o = dtype_o
        self._fp8 = False
        # Per-tensor FP8 (sdpa_fp8) vs block-scale MXFP8 (sdpa_mxfp8); both use FP8 Q/K/V.
        self._pertensor = bool(pertensor_fp8)
        self._device_cc = None  # (major, minor); set in check_support
        # Tuning-knob choice, already validated against the engine's
        # Capabilities domain by the probe (engines.mismatch).
        self.sched_policy = SCHED_NATURAL if sched_policy is None else int(sched_policy)
        self.tile_m = None if tile_m is None else int(tile_m)
        self.tile_n = None if tile_n is None else int(tile_n)
        self.cga = None if cga is None else int(cga)

        self.batch_size: Optional[int] = None
        self.s_q_max: Optional[int] = None
        self.s_k_max: Optional[int] = None
        self.h_q: Optional[int] = None
        self.h_kv: Optional[int] = None
        self.head_dim_qk: Optional[int] = None
        self.head_dim_v: Optional[int] = None
        self.dtype: Optional[torch.dtype] = None
        self._dummy_cache: dict[tuple[str, torch.device], torch.Tensor] = {}
        self._initialize_implementation()
        self._logger.debug("__init__ completed")

    @abstractmethod
    def _initialize_implementation(self) -> None:
        """Initialize state private to specific implementations."""

    @staticmethod
    def _to_bshd(tensor: torch.Tensor) -> torch.Tensor:
        """Return the compact kernel-facing BSHD tensor for logical BHSD."""

        view = tensor.transpose(1, 2)
        return view if view.is_contiguous() else view.contiguous()

    @staticmethod
    def _to_bshd_writable(tensor: torch.Tensor):
        """Return a compact writable BSHD tensor and optional copy-back state."""

        view = tensor.transpose(1, 2)
        if view.is_contiguous():
            return view, False, None
        scratch = torch.empty_like(view, memory_format=torch.contiguous_format)
        return view, True, scratch

    def _dummy(self, key: str, device: torch.device, factory: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Return a cached device-local dummy tensor."""

        cache_key = (key, device)
        tensor = self._dummy_cache.get(cache_key)
        if tensor is None:
            tensor = factory()
            self._dummy_cache[cache_key] = tensor
        return tensor

    def _checked_lse_view(self, lse_tensor: torch.Tensor) -> torch.Tensor:
        """Validate a caller-provided LSE buffer and return the kernel's (B, H_q, S_q) view.

        The kernel WRITES through the returned view, so this must be a true
        view: a silent ``reshape`` copy of a non-contiguous buffer would
        receive the output and be dropped, leaving the caller's LSE unwritten.
        """
        self._value_error_if(
            lse_tensor.dtype != torch.float32,
            f"lse_tensor must be float32; got {lse_tensor.dtype}",
        )
        expected = self.batch_size * self.h_q * self.s_q_max
        self._value_error_if(
            lse_tensor.numel() != expected,
            f"lse_tensor must have B*H_q*S_q = {expected} elements; got {lse_tensor.numel()}",
        )
        self._value_error_if(
            not lse_tensor.is_contiguous(),
            "lse_tensor must be contiguous (the kernel writes through this buffer)",
        )
        return lse_tensor.view(self.batch_size, self.h_q, self.s_q_max)

    def _checked_sinks_1d(self, sinks: torch.Tensor) -> torch.Tensor:
        """Validate caller-provided sink logits and return the kernel's (H_q,) fp32 view.

        Strictly a view: the kernels consume fp32 sinks directly, and an
        implicit ``.to(float32)`` here would allocate and launch a cast kernel
        on the execute hot path (and break CUDA-graph pointer stability).
        """
        self._value_error_if(
            sinks.dtype != torch.float32,
            f"sinks must be float32; got {sinks.dtype}",
        )
        self._value_error_if(
            sinks.numel() != self.h_q,
            f"sinks must have H_q = {self.h_q} elements; got {sinks.numel()}",
        )
        self._value_error_if(
            not sinks.is_contiguous(),
            "sinks must be contiguous (bound to the kernel as a flat (H_q,) view)",
        )
        return sinks.reshape(-1)

    def _checked_seq_lens(self, seq_lens: torch.Tensor, name: str) -> torch.Tensor:
        """Validate caller-provided per-batch lengths and return the kernel's (B,) int32 view.

        Strictly a view: an implicit ``.to(torch.int32)`` here would allocate
        and launch a cast kernel on the execute hot path (and break CUDA-graph
        pointer stability).
        """
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

    def _check_seq_lens_contract(self, seq_q_lens, seq_kv_lens) -> None:
        """Reject seq-length tensors inconsistent with the compiled specialization.

        Like sinks, presence is a compile-time specialization: substituting a
        zeros dummy for a required tensor masks every row (silently wrong
        output), and lengths passed to a specialization compiled without them
        are silently ignored. THD is exempt — it always requires both (they
        source the packed cu_seqlens metadata).
        """
        if self.thd:
            self._value_error_if(
                seq_q_lens is None or seq_kv_lens is None,
                "THD execute requires seq_q_lens and seq_kv_lens",
            )
            return
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
        lse_tensor: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        seq_q_lens: Optional[torch.Tensor] = None,
        seq_kv_lens: Optional[torch.Tensor] = None,
        scale_softmax: Optional[float] = None,
        workspace: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Launch the compiled kernel on the common SDPA operand set.

        This is the keyword contract of the shared engine lowering
        (``lower_dsl_prefill`` in ``fwd/engines.py``), which drives every
        adapter through one ``execute_kwargs`` dict: the arguments above are
        always passed by keyword, and ``workspace`` is included iff
        ``scratch_workspace_bytes()`` is non-zero. Subclasses may extend the
        signature only with additional optional keyword arguments; an adapter
        whose engine capabilities accept FP8/MXFP8 graphs must also accept the
        FP8 operand set the lowering adds for those graphs (``sf_q/sf_k/sf_v``,
        ``descale_q/descale_k/descale_v``, ``scale_o``, ``amax_o``,
        ``amax_s`` — see :meth:`SdpaFwdDslSm100.execute`).
        """


class SdpaFwdDslSm100(SdpaFwdDsl):
    """SM100 (Blackwell) SDPA forward via the FROST DSL template kernels."""

    def _initialize_implementation(self) -> None:
        self.flavor: Optional[tuple[int, int]] = None
        self.mask_flags = 0
        self.swa_window_runtime = 0
        self.thd_stats_head_major = False
        self.thd_stats_head_stride = 0
        self._k_mod = None

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        # Layout gate. DENSE: the kernels always consume canonical BSHD-compact
        # buffers — execute() normalizes via _to_bshd / _to_bshd_writable
        # (zero-copy when the tensor already is BSHD-compact, one gather /
        # scatter copy otherwise) — so the only real requirements are the ones
        # normalization itself needs: head dim innermost-contiguous (stride 1),
        # no zero stride on a size>1 dim (broadcast), and non-overlapping
        # strides (padded / oversized are fine; sub-dense would alias and make
        # the O write-back ill-defined). Any B/H/S stride permutation is
        # accepted. No TMA stride/alignment gate is needed here: the TMA
        # descriptors are built over the normalized compact buffers, never
        # over the caller's strides.
        # THD (ragged) keeps the strict BSHD stride order: the varlen path
        # rebuilds packed [1,T,H,D] views and only that packing is defined.
        from cudnn.sdpa.graph_analyzer import dense_layout_ok

        _REQ = (3, 1, 2, 0)
        for desc_name in ["q_desc", "k_desc", "v_desc", "o_desc"]:
            d = getattr(self, desc_name)
            self._value_error_if(
                d.ndim != 4,
                f"{d.name} must be rank-4 (B, H, S, D); got {d.ndim}",
            )
            _shape, _stride = d.shape, d.stride
            if self.thd:
                _act = tuple(ax for ax in d.stride_order if _shape[ax] != 1)
                _exp = tuple(ax for ax in _REQ if _shape[ax] != 1)
                self._value_error_if(
                    _act != _exp,
                    f"{d.name} must have d, h, s, b stride order (3, 1, 2, 0) for THD (size-1 dims wildcarded); got {d.stride_order} shape {_shape}",
                )
            else:
                self._value_error_if(
                    not dense_layout_ok(_shape, _stride),
                    f"{d.name} must have the head dim innermost-contiguous (stride 1) and "
                    f"non-broadcast, non-overlapping strides (any B/H/S order, padded "
                    f"strides allowed); got stride {_stride} shape {_shape}",
                )

        b, h_qo, s_qo, d_qk = self.q_desc.shape
        _, h_kv, s_kv, _ = self.k_desc.shape
        _, _, _, d_v = self.v_desc.shape

        self._check_tensor_shape(self.q_desc, (b, h_qo, s_qo, d_qk), name="Q")
        self._check_tensor_shape(self.k_desc, (b, h_kv, s_kv, d_qk), name="K")
        self._check_tensor_shape(self.v_desc, (b, h_kv, s_kv, d_v), name="V")
        self._check_tensor_shape(self.o_desc, (b, h_qo, s_qo, d_v), name="O")

        for label, val in (
            ("B", b),
            ("H_q", h_qo),
            ("H_kv", h_kv),
            ("S_q", s_qo),
            ("S_kv", s_kv),
            ("D_QK", d_qk),
            ("D_V", d_v),
        ):
            self._value_error_if(int(val) <= 0, f"{label} must be > 0; got {val}")

        self._value_error_if(
            h_qo % h_kv != 0,
            f"H_q ({h_qo}) must be divisible by H_kv ({h_kv}) for GQA / MQA",
        )

        # Q/K/V dtype: half (BF16/FP16, DTYPE_O == input) or FP8 (E4M3/E5M2 → MXFP8,
        # d128 only, DTYPE_O independent — typically BF16/FP16).
        self.dtype = self._check_dtype(self.q_desc, [torch.float16, torch.bfloat16, *_SM100_FP8_DTYPES], name="Q")
        self._fp8 = self.dtype in _SM100_FP8_DTYPES
        for desc in [self.k_desc, self.v_desc]:
            self._check_dtype(desc, self.dtype, name=desc.name, extra_error_msg=f"{desc.name} must match Q dtype")
        if self._fp8:
            # MXFP8 block-scale input: O may be BF16/FP16 (half) or FP8, decoupled from the input dtype.
            self.dtype_o = self._check_dtype(self.o_desc, [torch.float16, torch.bfloat16, *_SM100_FP8_DTYPES], name="O")
        else:
            self._check_dtype(
                self.o_desc,
                self.dtype,
                name=self.o_desc.name,
                extra_error_msg=f"{self.o_desc.name} must match Q dtype (FP16/BF16 on SM100 DSL)",
            )
            self.dtype_o = self.dtype
        if self.lse_desc is not None:
            self._check_dtype(self.lse_desc, torch.float32, name="LSE")
            self._check_tensor_shape(self.lse_desc, (b, h_qo, s_qo), name="LSE")
            if self.thd:
                stride_h, stride_s = tuple(self.lse_desc.stride[1:])
                token_major = (stride_h, stride_s) == (1, h_qo)
                head_major = not token_major and stride_s == 1 and stride_h >= 1
                self._value_error_if(
                    not token_major and not head_major,
                    f"THD LSE must be packed token-major (stride_h == 1, stride_s == H) "
                    f"or head-major (stride_s == 1, stride_h == head_stride); got stride {self.lse_desc.stride}",
                )
                self.thd_stats_head_major = head_major
                self.thd_stats_head_stride = int(stride_h) if head_major else 0
            else:
                self._value_error_if(not self.lse_desc.is_contiguous(), "LSE must be contiguous on SM100 DSL")

        self._value_error_if(not torch.cuda.is_available(), "CUDA must be available for SM100 DSL SDPA")
        device = self.q_desc.device
        major, minor = torch.cuda.get_device_capability(device)
        # cc10.0 (SM100) and cc10.3 (Blackwell-class) both run these kernels; cc10.3
        # additionally has the fused LDTM.STAT row-max, auto-enabled for MXFP8 in compile().
        self._device_cc = (major, minor)
        self._value_error_if(
            self._device_cc not in ((10, 0), (10, 3)),
            f"SdpaFwdDslSm100 requires cc=10.0 or 10.3 (Blackwell); found SM{major}{minor} on {device}",
        )

        # FP8/MXFP8: exact-match d128 only — the FP8 kernels' SF plumbing and
        # QMMA geometry are not audited for envelope zero-padding.
        self._value_error_if(
            self._fp8 and (int(d_qk), int(d_v)) != (128, 128),
            f"FP8/MXFP8 (E4M3/E5M2 inputs) requires exact D_QK=D_V=128 (no envelope padding); got (D_QK={d_qk}, D_V={d_v})",
        )
        # Envelope alignment gate: the TMA descriptors are built from the
        # actual tensor extents, and cuTensorMapEncodeTiled requires every
        # non-innermost global stride to be a multiple of 16 bytes. For the
        # compact BSHD views the H stride is D * BPE (2 bytes at fp16/bf16),
        # so both head dims must be multiples of 8.
        self._value_error_if(
            int(d_qk) % 8 != 0 or int(d_v) % 8 != 0,
            f"SM100 DSL envelope requires D_QK and D_V to be multiples of 8 "
            f"(TMA 16-byte global-stride constraint at 2 bytes/elem); got "
            f"(D_QK={d_qk}, D_V={d_v})",
        )
        self.flavor = _pick_flavor(d_qk, d_v)
        self._value_error_if(
            self.sched_policy != SCHED_NATURAL,
            f"SM100 DSL SDPA only supports sched_policy={SCHED_NATURAL}",
        )
        for requested, supported, name in (
            (self.tile_m, 128, "tile_m"),
            (self.tile_n, 128, "tile_n"),
            (self.cga, 2, "cga"),
        ):
            self._value_error_if(
                requested is not None and requested != supported,
                f"SM100 DSL SDPA only supports {name}={supported}",
            )

        swa_left = self.window_size_left
        self._value_error_if(
            swa_left is not None and swa_left < 0,
            f"window_size_left must be >= 0; got {swa_left}",
        )
        # The kernels' bottom-right diagonal path supports plain causal only:
        # CAUSAL_BOTTOM_RIGHT requires MASK_CAUSAL and excludes MASK_SWA
        # (see config._validate_knobs).
        self._value_error_if(
            self.causal_bottom_right and not self.is_causal,
            "SM100 DSL SDPA: causal_bottom_right requires is_causal=True",
        )
        self._value_error_if(
            self.causal_bottom_right and swa_left is not None,
            "SM100 DSL SDPA: causal_bottom_right cannot be combined with a left sliding-window (kernel gap)",
        )
        # Backstop for the engines.bottom_right_padded_seq_q gate: with dense
        # per-batch Q lengths the kernel's BR diagonal (anchored at the global
        # S_q) is wrong for any batch with seq_len_q[b] < S_q.
        self._value_error_if(
            self.causal_bottom_right and self.seq_q_lens_present,
            "SM100 DSL SDPA: causal_bottom_right with per-batch seq_len_q is not "
            "supported (kernel anchors the BR diagonal at the global S_q, not "
            "seq_len_q[b])",
        )
        if self.is_causal:
            self.mask_flags = MASK_CAUSAL | (MASK_SWA if swa_left is not None else 0)
            self.swa_window_runtime = swa_left if swa_left is not None else 0
        elif swa_left is not None:
            self.mask_flags = MASK_SWA
            self.swa_window_runtime = swa_left
        else:
            self.mask_flags = MASK_NONE
            self.swa_window_runtime = 0
        if self.thd:
            self.seq_kv_lens_present = True
        if self.seq_kv_lens_present:
            self.mask_flags |= MASK_PADDED
        # Dense padded-Q trim backstops (engines.lower_dsl_prefill never sets
        # these combinations; a direct caller could).
        self._value_error_if(
            self.seq_q_lens_present and self.thd,
            "seq_q_lens_present is dense-only (THD carries per-sequence Q lengths via cu_seqlens)",
        )
        self._value_error_if(
            self.seq_q_lens_present and not self.seq_kv_lens_present,
            "seq_q_lens_present requires seq_kv_lens_present (padding mask)",
        )
        self._value_error_if(
            self.seq_q_lens_present and self._fp8,
            "seq_q_lens_present (dense padded-Q LSE trim) is not plumbed for the FP8/MXFP8 kernels",
        )
        # Dense BR + per-batch Q lengths: the kernels anchor the bottom-right
        # diagonal with the GLOBAL S_q (compute_kv_loop_bounds: causal_diag =
        # seq_kv_len - seqlen_q with the scalar S_q), but cuDNN semantics
        # anchor it at the per-batch (seq_len_q[b], seq_len_kv[b]) corner —
        # batches with seq_len_q[b] < S_q get over-masked. KV-only padding is
        # exact (actual Q length == S_q), so only this combination is gated.
        self._value_error_if(
            self.causal_bottom_right and self.seq_q_lens_present,
            "SM100 DSL SDPA: causal_bottom_right with per-batch seq_len_q (dense "
            "padded-Q trim) is not supported — the kernel anchors the BR diagonal "
            "at the global S_q, not seq_len_q[b]/seq_len_kv[b]",
        )
        # KV-tail correctness: the kernel zero-fills the last KV tile via TMA
        # OOB but only *masks* those columns on the padded / causal paths. A
        # ragged S_kv is safe when a padding mask carries the real lengths, or
        # when the causal diagonal provably covers the tail (kv >= S_kv implies
        # kv > q for every query row). Otherwise the tail columns leak into
        # the softmax and the output is silently wrong.
        if int(s_kv) % _SM100_TILE_N != 0:
            causal_covers_tail = self.is_causal and (self.causal_bottom_right or int(s_qo) <= int(s_kv))
            self._value_error_if(
                not (self.seq_kv_lens_present or causal_covers_tail),
                f"S_kv ({s_kv}) must be a multiple of {_SM100_TILE_N} unless a "
                f"padding mask (seq_len_kv) is provided or the causal mask "
                f"covers the KV tail — the tail is otherwise unmasked on "
                f"SM100 DSL",
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
        self._logger.debug("check_support completed successfully")
        return True

    def compile(self) -> None:
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        # MXFP8 on cc10.3+ fuses the S_acc row-max into the LDTM (the fp8/f16 kernels
        # don't read this flag). Auto-set from the device capability so an SM103 run
        # picks the fused path with no user action.
        mxfp8 = self._fp8 and not self._pertensor
        fused_ldtm_stat = mxfp8 and (self._device_cc == (10, 3))
        sched_policy = self.sched_policy
        if mxfp8 and sched_policy == SCHED_NATURAL and (self.mask_flags & MASK_CAUSAL):
            sched_policy = SCHED_LPT
        params = Sm100TemplateParams(
            dtype_qkv=_SM100_DTYPE_QKV_CODE[self.dtype],
            dtype_o=_SM100_DTYPE_QKV_CODE[self.dtype_o],
            mask_flags=self.mask_flags,
            swa_window=int(self.swa_window_runtime),
            causal_bottom_right=self.causal_bottom_right,
            has_sink=self.has_sink,
            seq_kv_lens_present=self.seq_kv_lens_present,
            seq_q_lens_present=self.seq_q_lens_present,
            sched_policy=sched_policy,
            thd_varlen=self.thd,
            fused_ldtm_stat=fused_ldtm_stat,
        )
        self._k_mod = _load_sm100_kernel_module(self.flavor, params, fp8=self._fp8, pertensor=self._pertensor)
        if self.thd:
            # T (total tokens) is a runtime value, so the per-shape compile is deferred to execute().
            self._compiled_kernel = "thd-deferred"
        elif self._fp8:
            # FP8/MXFP8 kernels are exact-match d128 (gated in check_support);
            # their compile() has no envelope head-dim parameters.
            self._compiled_kernel = self._k_mod.compile(
                b=self.batch_size,
                qh=self.h_q,
                kh=self.h_kv,
                sq=self.s_q_max,
                skv=self.s_k_max,
            )
        else:
            # ENVELOPE: hand the f16/bf16 kernel the ACTUAL head dims so its
            # TMA descriptors carry the real extents (loads past them
            # zero-fill, O stores past d_v clip); the tile box stays the
            # flavor's compile-time D. has_lse=False (no Stats output)
            # compiles the LSE store out — no dummy buffer at any level.
            self._compiled_kernel = self._k_mod.compile(
                b=self.batch_size,
                qh=self.h_q,
                kh=self.h_kv,
                sq=self.s_q_max,
                skv=self.s_k_max,
                d_qk=self.head_dim_qk,
                d_v=self.head_dim_v,
                has_lse=self.lse_desc is not None,
            )
        self._logger.debug("compile completed")

    def scratch_workspace_bytes(self) -> int:
        """Per-execute scratch ``execute()`` carves from its ``workspace``.

        Fixed by the compiled geometry (call after ``check_support()``); 0 when
        the path allocates nothing per execute. This is the api-level share of
        a FROST executor's ``workspace_bytes`` (the engine lowering adds its
        own chunks — dummy LSE, synthesized seq_len_kv — on top; see
        ``engines.lower_dsl_prefill``). When ``execute()`` is called WITHOUT a
        workspace (standalone API use), these buffers are torch-allocated per
        execute instead — the carve path is what the FROST dispatch uses.
        """
        self._ensure_support_checked()
        b, qh = self.batch_size, self.h_q
        if self.thd:
            # [meta(seq_kv, cu_q, cu_k) | o_desc | sinks dummy]
            # No packed-LSE chunk: with a Stats output the kernel writes the
            # caller's ragged Stats buffer directly (token-major (T, H) or
            # head-major (H, head_stride)); without one it compiles with
            # has_lse=False and no LSE buffer exists at all. No slq/slk
            # copies either: the metadata is built host-side from the tolist
            # round-trip and uploaded in one H2D copy. o_desc: 16 int64 per
            # sequence + 16 spare, the per-sequence O TMA descriptors the
            # builder kernel fills.
            return ws_align((3 * b + 2) * 4) + ws_align((b * 16 + 16) * 8) + (0 if self.has_sink else ws_align(qh * 4))
        if self._fp8:
            return 0  # dense FP8/MXFP8: no per-execute scratch (dummies are cached one-time)
        # Dense padded-Q lens bind directly as their own kernel parameter
        # (no combine buffer since the seq_len_q-as-parameter change) — no scratch.
        return 0

    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        lse_tensor: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        seq_q_lens: Optional[torch.Tensor] = None,
        seq_kv_lens: Optional[torch.Tensor] = None,
        scale_softmax: Optional[float] = None,
        current_stream: Optional[cuda.CUstream] = None,
        sf_q: Optional[torch.Tensor] = None,
        sf_k: Optional[torch.Tensor] = None,
        sf_v: Optional[torch.Tensor] = None,
        amax_o: Optional[torch.Tensor] = None,
        descale_q: Optional[torch.Tensor] = None,
        descale_k: Optional[torch.Tensor] = None,
        descale_v: Optional[torch.Tensor] = None,
        scale_o: Optional[torch.Tensor] = None,
        amax_s: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
    ) -> None:
        """Launch the compiled kernel.

        ``workspace``: optional caller-provided scratch buffer (uint8, at
        least ``scratch_workspace_bytes()`` bytes). When given, every
        per-execute scratch buffer (the THD metadata / O-descriptor buffers)
        is carved from it — zero per-execute allocations. When None
        (standalone use), those buffers are torch-allocated as before.
        """
        self._logger.debug("Entering execute")
        if self._compiled_kernel is None:
            raise RuntimeError("SdpaFwdDslSm100 is not compiled")
        # Run on the caller's stream (ExecutionContext.stream, resolved from the
        # execute-time handle); None -> the default stream. Threaded to every
        # kernel launch below (dense / fp8 / mxfp8 / THD).
        current_stream = self._get_default_stream(current_stream)

        scale_val = self.scale_softmax if scale_softmax is None or scale_softmax == 0.0 else float(scale_softmax)
        scale_softmax_log2 = scale_val * math.log2(math.e)

        self._value_error_if(
            self.has_sink and sinks is None,
            "sinks is required by this compiled specialization",
        )
        self._value_error_if(
            not self.has_sink and sinks is not None,
            "this specialization was compiled without sink support; construct the API with has_sink=True",
        )
        self._check_seq_lens_contract(seq_q_lens, seq_kv_lens)
        self._value_error_if(
            self.lse_desc is not None and lse_tensor is None,
            "lse_tensor is required by this compiled specialization",
        )
        # Strict presence contract, both directions: the f16 kernels are
        # compiled with has_lse keyed on sample_lse (no Stats output -> the
        # LSE store is compiled out and there is no LSE slot to bind), and a
        # THD lse_tensor is bound in its DECLARED packed layout (recorded at
        # check_support) — so an lse_tensor without a sample_lse cannot be
        # honored and is rejected rather than silently dropped. The FP8/MXFP8
        # kernels (dense-only) still write an LSE unconditionally; their
        # stats-less write lands in a cached write-only dummy (the FROST
        # dispatch never reaches it: engines.lower_dsl_prefill carves the
        # dummy from the caller's workspace instead).
        self._value_error_if(
            self.lse_desc is None and lse_tensor is not None and not self._fp8,
            "this specialization was compiled without an LSE output; construct the API with sample_lse",
        )
        if self.thd:
            pass  # bound in _execute_thd (declared packed layout)
        elif lse_tensor is not None:
            lse_tensor = self._checked_lse_view(lse_tensor)
        elif self._fp8:
            lse_tensor = self._dummy(
                "lse",
                q_tensor.device,
                lambda: torch.empty((self.batch_size, self.h_q, self.s_q_max), dtype=torch.float32, device=q_tensor.device),
            )

        if self._fp8 and self._pertensor:
            # Per-tensor FP8 (sdpa_fp8): scalar descales fold into the softmax scale
            # (descale_q·descale_k) and o_scale_fused (descale_v·scale_o).
            self._execute_fp8(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                lse_tensor,
                scale_val,
                sinks,
                seq_kv_lens,
                seq_q_lens,
                descale_q,
                descale_k,
                descale_v,
                scale_o,
                amax_s,
                amax_o,
                current_stream,
            )
            return

        if self._fp8:
            # MXFP8 block-scale path (E4M3/E5M2 in, half out). Per-block E8M0 scales
            # dequant in-MMA, so scale_softmax_log2 carries only attn_scale·log2(e).
            self._execute_mxfp8(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                lse_tensor,
                scale_softmax_log2,
                sinks,
                seq_kv_lens,
                seq_q_lens,
                sf_q,
                sf_k,
                sf_v,
                amax_o,
                current_stream,
            )
            return

        if self.thd:
            self._execute_thd(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                scale_softmax_log2,
                sinks,
                seq_kv_lens,
                seq_q_lens,
                lse_tensor=lse_tensor,
                workspace=workspace,
                current_stream=current_stream,
            )
            return

        Q = self._to_bshd(q_tensor)
        K = self._to_bshd(k_tensor)
        V = self._to_bshd(v_tensor)
        O_view, o_needs_copy_back, O_scratch = self._to_bshd_writable(o_tensor)

        device = q_tensor.device
        sinks_t = (
            self._checked_sinks_1d(sinks)
            if sinks is not None
            else self._dummy("sinks", device, lambda: torch.zeros(self.h_q, dtype=torch.float32, device=device))
        )
        seq_kv_t = (
            self._checked_seq_lens(seq_kv_lens, "seq_kv_lens")
            if seq_kv_lens is not None
            else self._dummy("seq_kv", device, lambda: torch.zeros(self.batch_size, dtype=torch.int32, device=device))
        )
        # Dense padded-Q trim: per-batch Q lengths are their OWN kernel
        # parameter (compiled in only when seq_q_lens_present — the kernel
        # signature is specialized on `None`, so the flag-off ABI is
        # unchanged). The caller's (B,)-int32 device tensor is bound directly
        # as a validated view — zero allocations/copies on the execute hot
        # path, stable pointer (CUDA-graph-capture friendly).
        seq_q_t = self._checked_seq_lens(seq_q_lens, "seq_q_lens") if self.seq_q_lens_present else None
        o_desc_dummy = self._dummy("o_desc", device, lambda: torch.zeros(1, dtype=torch.int64, device=device))

        import cutlass

        self._compiled_kernel(
            Q,
            K,
            V,
            O_scratch if o_needs_copy_back else O_view,
            lse_tensor.reshape(self.batch_size, self.h_q, self.s_q_max) if lse_tensor is not None else None,
            sinks_t,
            seq_kv_t,
            o_desc_dummy,
            (self.batch_size, self.h_q, self.h_kv, self.s_q_max, self.s_k_max, 0),
            cutlass.Float32(scale_softmax_log2),
            cutlass.Int32(0),
            seq_q_t,
            stream=current_stream,
        )
        if o_needs_copy_back:
            O_view.copy_(O_scratch)
        self._logger.debug("execute completed")

    def _execute_thd(self, q_buf, k_buf, v_buf, o_buf, scale_softmax_log2, sinks, seq_len_kv, seq_q_lens, lse_tensor=None, workspace=None, current_stream=None):
        """THD / varlen execute: reconstruct the kernel's packed [1, T, H, D] views and metadata buffer from the cuDNN ragged buffers, then launch.

        With a ``workspace`` the metadata buffers (int32 length copies, the
        [seq_kv | cu_q | cu_k] buffer, the per-sequence O TMA descriptors, the
        sinks dummy) are carved from it — zero per-execute allocations;
        without one they are torch-allocated (standalone use). ``lse_tensor``,
        when given, is the caller's ragged Stats buffer, written by the
        kernel directly in its declared layout: token-major packed ``(T, H)``
        in the first ``T*H`` elements, or head-major ``(H, head_stride)``
        with tokens contiguous within each head row; when ``None`` the kernel
        compiles the LSE store out (has_lse=False) and no scratch exists.
        The host round-trip for the runtime totals (t_q / t_kv / unit count)
        is inherent to the lowering — the packed extents are data-dependent —
        and costs one D2H sync per length tensor, no device allocation."""
        import cutlass

        dev = q_buf.device
        slq_v = self._checked_seq_lens(seq_q_lens, "seq_q_lens")
        slk_v = self._checked_seq_lens(seq_len_kv, "seq_kv_lens")
        b = slq_v.numel()
        carver = WorkspaceCarver(workspace, self.scratch_workspace_bytes(), "SdpaFwdDslSm100 (THD)") if workspace is not None else None
        # Metadata buffer: [ seq_kv_lens(B) | cu_seqlens_q(B+1) | cu_seqlens_k(B+1) ],
        # built HOST-side from the (inherent) tolist round-trip and uploaded in
        # ONE H2D copy: a device-side cumsum would allocate its scan-temp
        # storage and launch kernels on the execute hot path.
        meta = carver.take(3 * b + 2, torch.int32) if carver is not None else torch.empty(3 * b + 2, dtype=torch.int32, device=dev)
        slq_host = slq_v.tolist()  # one D2H sync; t_q/t_kv/units are runtime values
        slk_host = slk_v.tolist()
        cu_q_host = [0]
        for n in slq_host:
            cu_q_host.append(cu_q_host[-1] + int(n))
        cu_k_host = [0]
        for n in slk_host:
            cu_k_host.append(cu_k_host[-1] + int(n))
        meta.copy_(torch.tensor(slk_host + cu_q_host + cu_k_host, dtype=torch.int32))
        t_q = cu_q_host[-1]
        t_kv = cu_k_host[-1]

        qh, kh = self.h_q, self.h_kv
        d_qk, d_v = self.head_dim_qk, self.head_dim_v

        # Degenerate total (runtime value, invisible to the plan-time probe):
        # t_q == 0 means no query token exists anywhere, so the packed O/LSE
        # have zero rows — nothing to compute or write. (t_kv == 0 launches
        # normally through the kernel's dead-row path; see the K/V binding
        # below.)
        if t_q == 0:
            self._logger.debug("execute (THD): t_q == 0, nothing to do")
            return
        lse = None
        if lse_tensor is not None:
            if self.thd_stats_head_major:
                head_stride = self.thd_stats_head_stride
                self._value_error_if(
                    head_stride < t_q,
                    f"head-major THD LSE head_stride ({head_stride}) must cover the packed Q token total ({t_q})",
                )
                lse = lse_tensor.as_strided((1, qh, head_stride), (qh * head_stride, head_stride, 1), lse_tensor.storage_offset())
            else:
                # Token-major (TH1, the default): natural packed rank-2 (T, H)
                # view — the kernel's epilogue dispatches on this static rank.
                lse = lse_tensor.as_strided((t_q, qh), (qh, 1), lse_tensor.storage_offset())

        # Per-sequence O TMA descriptors, filled by the kernel's builder pass.
        o_desc = carver.take(b * 16 + 16, torch.int64) if carver is not None else torch.zeros(b * 16 + 16, dtype=torch.int64, device=dev)
        if carver is not None:
            o_desc.zero_()
        # One THD unit per CGA-height slice of each sequence's Q rows.
        cga_tile_m = int(self._k_mod.CGA_TILE_M)
        units = qh * sum((l + cga_tile_m - 1) // cga_tile_m for l in slq_host)

        def _packed(buf, t, h, d):
            return buf.as_strided((1, t, h, d), (t * h * d, h * d, d, 1), buf.storage_offset())

        Q = _packed(q_buf, t_q, qh, d_qk)
        O = _packed(o_buf, t_q, qh, d_v)
        if t_kv == 0:
            # Every query row is dead (all-zero seq_kv_lens): served by the
            # KERNEL's own dead-row path (total_sum <= 0 -> O := 0 and
            # LSE := -inf, or the sink alone — its column keeps the softmax
            # denominator alive), exactly like a live launch's zero-KV
            # sequences — no adapter-side fills on the execute hot path
            # (AGENTS.md Rule 1). A zero-token packed K/V view cannot back a
            # CuTe layout / TMA descriptor, so clamp the packed KV extent to
            # ONE never-dereferenced token (every tile sees kv_left ==
            # kv_right == 0, so no K/V load is ever issued) bound over
            # storage guaranteed large enough: Q backs K (kh*d_qk <=
            # t_q*qh*d_qk) and O backs V (kh*d_v <= t_q*qh*d_v).
            t_kv = 1
            K = q_buf.as_strided((1, 1, kh, d_qk), (kh * d_qk, kh * d_qk, d_qk, 1), q_buf.storage_offset())
            V = o_buf.as_strided((1, 1, kh, d_v), (kh * d_v, kh * d_v, d_v, 1), o_buf.storage_offset())
        else:
            K = _packed(k_buf, t_kv, kh, d_qk)
            V = _packed(v_buf, t_kv, kh, d_v)
        # LSE binding: the caller's ragged Stats buffer in its declared layout
        # when a Stats output exists; None otherwise — the kernel compiles the
        # LSE store out (has_lse=False), so no dummy buffer exists at all.
        LSE = lse
        if sinks is not None:
            sinks_t = self._checked_sinks_1d(sinks)
        elif carver is not None:
            sinks_t = carver.take(qh, torch.float32)
            sinks_t.zero_()
        else:
            sinks_t = torch.zeros(qh, dtype=torch.float32, device=dev)

        fn = self._k_mod.compile(
            b=b,
            qh=qh,
            kh=kh,
            sq=t_q,
            skv=t_kv,
            d_qk=d_qk,
            d_v=d_v,
            # The Stats layout is a per-shape specialization (like d_qk/d_v):
            # has_lse=False compiles the store out; token-major binds the
            # packed rank-2 (T, H) view; head-major carries the caller-declared
            # head-row stride (0 -> compact sq).
            has_lse=lse is not None,
            lse_head_major=lse is not None and self.thd_stats_head_major,
            lse_head_stride=(self.thd_stats_head_stride if (lse is not None and self.thd_stats_head_major) else 0),
        )
        fn(Q, K, V, O, LSE, sinks_t, meta, o_desc, (b, qh, kh, t_q, t_kv, 0), cutlass.Float32(scale_softmax_log2), cutlass.Int32(units), stream=current_stream)
        self._logger.debug("execute (THD) completed")

    @staticmethod
    def _ceil_div(x: int, a: int) -> int:
        return (x + a - 1) // a

    def _reshape_sf(self, sf: torch.Tensor, h: int, n_tiles: int, sf_smem_size: int) -> torch.Tensor:
        """cuDNN F8_128x4 scale-factor tensor (FP8_E8M0, ``[B, H, *, *]``) → the
        kernel's per-tile int8 view ``[B, H, n_tiles, sf_smem_size]``.

        cuDNN packs the 128×4 SF atom contiguously (``F8_128x4`` reordering); a Q/K
        tile is 128 rows × d/32 d-blocks and a V tile is 128 rows × 4 s-blocks, so
        each tile is exactly ``sf_smem_size`` E8M0 bytes and this is a pure reshape.
        """
        b = sf.shape[0]
        flat = sf.contiguous()
        if flat.dtype != torch.int8:
            flat = flat.view(torch.int8)
        if flat.numel() != b * h * n_tiles * sf_smem_size:
            raise ValueError(
                f"MXFP8 SF size mismatch: got {flat.numel()} bytes, expected "
                f"{b}*{h}*{n_tiles}*{sf_smem_size}={b * h * n_tiles * sf_smem_size} "
                f"(shape {tuple(sf.shape)}, n_tiles={n_tiles}, sf_smem_size={sf_smem_size})"
            )
        return flat.reshape(b, h, n_tiles, sf_smem_size)

    def _execute_mxfp8(
        self,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        scale_softmax_log2,
        sinks,
        seq_kv_lens,
        seq_q_lens,
        sf_q,
        sf_k,
        sf_v,
        amax_o,
        current_stream=None,
    ):
        """MXFP8 execute (dense): FP8 Q/K/V + per-32-block E8M0 SF → half O.

        SF tensors come from cuDNN in F8_128x4 layout and are reshaped into the
        kernel's per-tile view; ``Amax_O`` (if requested) is filled with ``max|O|``.
        """
        import cutlass

        if self.thd:
            raise NotImplementedError("Frost MXFP8: THD/varlen execute is not wired yet (dense d128 only for v1)")
        if sf_q is None or sf_k is None or sf_v is None:
            raise ValueError("Frost MXFP8 execute requires sf_q/sf_k/sf_v (block-scale descale tensors)")

        km = self._k_mod
        b, h_q, h_kv = self.batch_size, self.h_q, self.h_kv
        sq, skv = self.s_q_max, self.s_k_max
        device = q_tensor.device

        Q = self._to_bshd(q_tensor)
        K = self._to_bshd(k_tensor)
        V = self._to_bshd(v_tensor)
        O_view, o_needs_copy_back, O_scratch = self._to_bshd_writable(o_tensor)
        O = O_scratch if o_needs_copy_back else O_view

        n_q_tiles = self._ceil_div(sq, _SM100_TILE_N)
        n_kv_tiles = self._ceil_div(skv, _SM100_TILE_N)
        sf_q_v = self._reshape_sf(sf_q, h_q, n_q_tiles, km.SF_SMEM_SIZE_Q)
        sf_k_v = self._reshape_sf(sf_k, h_kv, n_kv_tiles, km.SF_SMEM_SIZE_K)
        sf_v_v = self._reshape_sf(sf_v, h_kv, n_kv_tiles, km.SF_SMEM_SIZE_V)

        lse = lse_tensor.reshape(b, h_q, sq)
        sinks_t = (
            self._checked_sinks_1d(sinks) if sinks is not None else self._dummy("sinks", device, lambda: torch.zeros(h_q, dtype=torch.float32, device=device))
        )
        seq_kv_t = (
            self._checked_seq_lens(seq_kv_lens, "seq_kv_lens")
            if seq_kv_lens is not None
            else self._dummy("seq_kv", device, lambda: torch.zeros(b, dtype=torch.int32, device=device))
        )
        o_desc_dummy = self._dummy("o_desc", device, lambda: torch.zeros(1, dtype=torch.int64, device=device))

        amax_o_buf = amax_o.reshape(-1)[:1] if amax_o is not None else self._dummy("amax_o", device, lambda: torch.zeros(1, dtype=torch.float32, device=device))
        # Must be enqueued on the SAME stream as the kernel launch below, else the
        # reset and the kernel's atomicMax are unordered (and the reset is missing
        # from a CUDA-graph capture taken on the handle's stream).
        with _torch_stream_context(current_stream, device):
            amax_o_buf.zero_()

        self._compiled_kernel(
            Q,
            K,
            V,
            O,
            sf_q_v,
            sf_k_v,
            sf_v_v,
            lse,
            amax_o_buf,
            sinks_t,
            seq_kv_t,
            o_desc_dummy,
            (b, h_q, h_kv, sq, skv, 0),
            cutlass.Float32(scale_softmax_log2),
            cutlass.Int32(0),  # n_thd_units (dense)
            cutlass.Int32(0),  # total_q_sf_tiles (dense — kernel folds it out)
            cutlass.Int32(0),  # total_kv_sf_tiles
            stream=current_stream,
        )
        if o_needs_copy_back:
            O_view.copy_(O)
        self._logger.debug("execute (MXFP8) completed")

    def _execute_fp8(
        self,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        scale_val,
        sinks,
        seq_kv_lens,
        seq_q_lens,
        descale_q,
        descale_k,
        descale_v,
        scale_o,
        amax_s,
        amax_o,
        current_stream=None,
    ):
        """Per-tensor FP8 execute (dense): scalar descales fold into scale_softmax_log2
        (attn·descale_q·descale_k·log2 e) and o_scale_fused (descale_v·scale_o).

        ``Amax_S`` is produced in-kernel (atomicMax of the per-row softmax prob) into the
        caller's pre-zeroed buffer; ``Amax_O`` is ``max|O|/scale_o`` post-kernel (exact
        for half output).
        """
        import cutlass

        if self.thd:
            raise NotImplementedError("Frost per-tensor FP8: THD/varlen execute is not wired yet (dense d128 only)")

        def _scalar(t, default=1.0):
            return float(t.reshape(-1)[0].item()) if t is not None else default

        dq, dk, dv, so = _scalar(descale_q), _scalar(descale_k), _scalar(descale_v), _scalar(scale_o)
        scale_softmax_log2 = scale_val * dq * dk * math.log2(math.e)
        o_scale_fused = dv * so

        b, h_q, h_kv = self.batch_size, self.h_q, self.h_kv
        sq, skv = self.s_q_max, self.s_k_max
        device = q_tensor.device

        Q = self._to_bshd(q_tensor)
        K = self._to_bshd(k_tensor)
        V = self._to_bshd(v_tensor)
        O_view, o_needs_copy_back, O_scratch = self._to_bshd_writable(o_tensor)
        O = O_scratch if o_needs_copy_back else O_view

        lse = lse_tensor.reshape(b, h_q, sq)
        sinks_t = (
            self._checked_sinks_1d(sinks) if sinks is not None else self._dummy("sinks", device, lambda: torch.zeros(h_q, dtype=torch.float32, device=device))
        )
        seq_kv_t = (
            self._checked_seq_lens(seq_kv_lens, "seq_kv_lens")
            if seq_kv_lens is not None
            else self._dummy("seq_kv", device, lambda: torch.zeros(b, dtype=torch.int32, device=device))
        )
        o_desc_dummy = self._dummy("o_desc", device, lambda: torch.zeros(1, dtype=torch.int64, device=device))

        # amax_s / amax_o: the kernel atomicMax'es into these buffers, so they MUST
        # start at 0. amax_o accumulates max|o_scaled| (pre-cast, exact even for FP8 O);
        # dividing by scale_o below yields the pre-quant output amax.
        amax_s_buf = amax_s.reshape(-1)[:1] if amax_s is not None else self._dummy("amax_s", device, lambda: torch.zeros(1, dtype=torch.float32, device=device))
        amax_o_buf = amax_o.reshape(-1)[:1] if amax_o is not None else self._dummy("amax_o", device, lambda: torch.zeros(1, dtype=torch.float32, device=device))
        # Same-stream ordering as MXFP8: the resets must precede the kernel's
        # atomicMax on the launch stream, not on torch's current stream.
        with _torch_stream_context(current_stream, device):
            amax_s_buf.zero_()
            amax_o_buf.zero_()

        self._compiled_kernel(
            Q,
            K,
            V,
            O,
            lse,
            sinks_t,
            seq_kv_t,
            o_desc_dummy,
            (b, h_q, h_kv, sq, skv, 0),
            cutlass.Float32(scale_softmax_log2),
            cutlass.Float32(o_scale_fused),
            cutlass.Int32(0),  # n_thd_units (dense)
            amax_s_buf,
            amax_o_buf,
            stream=current_stream,
        )
        if o_needs_copy_back:
            O_view.copy_(O)
        if amax_o is not None:
            amax_o_buf.div_(max(so, 1e-30))
        self._logger.debug("execute (FP8 per-tensor) completed")


_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TensorSignature:
    """Tensor metadata that changes support or compilation."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class _SdpaFwdCacheKey:
    """Architecture-tagged cache key shared by direct FROST SDPA wrappers."""

    api_type: type[SdpaFwdDsl]
    q: _TensorSignature
    k: _TensorSignature
    v: _TensorSignature
    o: _TensorSignature
    lse: Optional[_TensorSignature]
    is_causal: bool
    causal_bottom_right: bool
    window_size_left: Optional[int]
    scale_softmax: Optional[float]
    seq_q_lens_present: bool
    seq_kv_lens_present: bool
    has_sink: bool
    thd: bool
    sched_policy: Optional[int]
    tile_m: Optional[int]
    tile_n: Optional[int]
    cga: Optional[int]


def _tensor_signature(tensor: torch.Tensor) -> _TensorSignature:
    return _TensorSignature(
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        dtype=tensor.dtype,
        device=tensor.device,
    )


def _optional_tensor_signature(tensor: Optional[torch.Tensor]) -> Optional[_TensorSignature]:
    return None if tensor is None else _tensor_signature(tensor)


def _make_cache_key(
    api_type: type[SdpaFwdDsl],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    *,
    lse: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    causal_bottom_right: bool = False,
    window_size_left: Optional[int] = None,
    scale_softmax: Optional[float] = None,
    seq_q_lens_present: bool = False,
    seq_kv_lens_present: bool = False,
    has_sink: bool = False,
    thd: bool = False,
    sched_policy: Optional[int] = None,
    tile_m: Optional[int] = None,
    tile_n: Optional[int] = None,
    cga: Optional[int] = None,
) -> _SdpaFwdCacheKey:
    return _SdpaFwdCacheKey(
        api_type=api_type,
        q=_tensor_signature(q),
        k=_tensor_signature(k),
        v=_tensor_signature(v),
        o=_tensor_signature(o),
        lse=_optional_tensor_signature(lse),
        is_causal=is_causal,
        causal_bottom_right=causal_bottom_right,
        window_size_left=window_size_left,
        scale_softmax=scale_softmax,
        has_sink=has_sink,
        seq_q_lens_present=seq_q_lens_present,
        seq_kv_lens_present=seq_kv_lens_present,
        thd=thd,
        sched_policy=sched_policy,
        tile_m=tile_m,
        tile_n=tile_n,
        cga=cga,
    )


_cache_of_objects: dict[_SdpaFwdCacheKey, SdpaFwdDsl] = {}


def _get_or_create_api(
    cache_key: _SdpaFwdCacheKey,
    **api_kwargs,
) -> SdpaFwdDsl:
    api = _cache_of_objects.get(cache_key)
    if api is None:
        _logger.debug("Building new %s", cache_key.api_type.__name__)
        api = cache_key.api_type(**api_kwargs)
        api.check_support()
        api.compile()
        _cache_of_objects[cache_key] = api
    return api


def _allocate_lse_tensor(q_tensor: torch.Tensor) -> torch.Tensor:
    if q_tensor.ndim != 4:
        raise ValueError(f"Expected BHSD q_tensor to be rank-4, got {q_tensor.ndim}")
    b, h, s_q, _ = q_tensor.shape
    return torch.empty((b, h, s_q), dtype=torch.float32, device=q_tensor.device)


def sdpa_fwd_wrapper_dsl_sm100(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    is_causal: bool = False,
    window_size_left: Optional[int] = None,
    causal_bottom_right: bool = False,
    scale_softmax: Optional[float] = None,
    seq_kv_lens: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """SM100 SDPA forward; returns ``TupleDict(o_tensor=..., lse_tensor=...)``."""
    if current_stream is not None:
        raise NotImplementedError(
            "sdpa_fwd_wrapper_dsl_sm100: explicit current_stream is not "
            "yet supported. Wrap the call in `with torch.cuda.stream(s):` to "
            "dispatch onto a non-default stream."
        )
    if q_tensor.ndim != 4 or v_tensor.ndim != 4:
        raise ValueError(f"Q and V must be rank-4 BHSD; got Q={q_tensor.ndim}D V={v_tensor.ndim}D")

    b, h_q, s_q, _ = q_tensor.shape
    d_v = v_tensor.shape[-1]
    o_tensor = torch.empty(
        (b, s_q, h_q, d_v),
        dtype=q_tensor.dtype,
        device=q_tensor.device,
    ).transpose(1, 2)
    lse_tensor = _allocate_lse_tensor(q_tensor)

    cache_key = _make_cache_key(
        SdpaFwdDslSm100,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse=lse_tensor,
        is_causal=is_causal,
        causal_bottom_right=causal_bottom_right,
        window_size_left=window_size_left,
        scale_softmax=scale_softmax,
        seq_q_lens_present=False,
        seq_kv_lens_present=seq_kv_lens is not None,
        has_sink=sinks is not None,
    )
    sdpa_fwd = _get_or_create_api(
        cache_key,
        sample_q=q_tensor,
        sample_k=k_tensor,
        sample_v=v_tensor,
        sample_o=o_tensor,
        sample_lse=lse_tensor,
        has_sink=sinks is not None,
        seq_kv_lens_present=seq_kv_lens is not None,
        is_causal=is_causal,
        causal_bottom_right=causal_bottom_right,
        window_size_left=window_size_left,
        scale_softmax=scale_softmax,
    )

    sdpa_fwd.execute(
        q_tensor=q_tensor,
        k_tensor=k_tensor,
        v_tensor=v_tensor,
        o_tensor=o_tensor,
        lse_tensor=lse_tensor,
        scale_softmax=scale_softmax,
        sinks=sinks,
        seq_kv_lens=seq_kv_lens,
        current_stream=current_stream,
    )
    return TupleDict(o_tensor=o_tensor, lse_tensor=lse_tensor)


# ---------------------------------------------------------------------------
# SM120: adapter over the SM120 / SM121 prefill kernel template
# ---------------------------------------------------------------------------


class SdpaFwdDslSm120(SdpaFwdDsl):
    """Compile and execute fixed-length SM120/SM121 SDPA forward.

    Q, K, V, and O use logical ``(B, H, S, D)`` shapes over any dense layout
    with the head dim innermost (``dense_flex``, same envelope as SM100):
    ``execute()`` normalizes to the kernel's compact-BSHD storage via
    ``_to_bshd`` / ``_to_bshd_writable`` — zero-copy when the tensor already
    is BSHD-compact, one gather / scatter copy otherwise. The kernel supports
    FP16/BF16 MHA, GQA, and MQA; head dimensions from 16 through 256 in
    increments of 16; top-left or bottom-right causal masks; left sliding
    windows; optional per-batch query and key/value lengths; optional
    per-Q-head attention-sink logits; and THD (ragged / fully packed
    variable-length) batches, whose per-shape compile is deferred to
    ``execute()`` because the packed token totals are runtime values.

    ``scale_softmax`` is a runtime parameter. Dtype, shape, tile sizes, masks,
    and length-tensor / sink / THD presence are compile-time specializations.
    """

    def _initialize_implementation(self) -> None:
        self.q_tile = _SM120_Q_TILES[0] if self.tile_m is None else self.tile_m
        self.kv_tile = _SM120_KV_TILES[0] if self.tile_n is None else self.tile_n
        self.compute_capability: Optional[tuple[int, int]] = None
        self.head_dim_qk: Optional[int] = None
        self.head_dim_v: Optional[int] = None
        self.thd_stats_head_major = False
        self.thd_stats_head_stride = 0
        self._k_mod = None

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        if self.thd:
            self._value_error_if(self.seq_q_lens_present, "seq_q_lens_present is dense-only (THD carries per-sequence Q lengths via cu_seqlens)")
            self.seq_kv_lens_present = True
        self._value_error_if(
            self.sched_policy is not None and self.sched_policy != SCHED_NATURAL,
            f"SM120 DSL SDPA only supports sched_policy={SCHED_NATURAL}",
        )
        self._value_error_if(
            self.cga is not None and self.cga != 1,
            "SM120 DSL SDPA only supports cga=1",
        )

        # Layout gate (same envelope as SM100): THD keeps the strict BSHD
        # stride order (the varlen path rebuilds packed (1, T, H, D) views
        # and only that packing is defined); dense graphs get the dense_flex
        # relaxation — execute() normalizes to the kernel's compact-BSHD
        # storage via _to_bshd / _to_bshd_writable (zero-copy when already
        # BSHD-compact, one gather / scatter copy otherwise), so only what
        # normalization needs is required: head dim innermost-contiguous
        # (stride 1), non-broadcast, non-overlapping strides, any B/H/S
        # order, padded strides allowed.
        from cudnn.sdpa.graph_analyzer import bshd_layout_ok, dense_layout_ok

        for desc in (self.q_desc, self.k_desc, self.v_desc, self.o_desc):
            self._value_error_if(
                desc.ndim != 4,
                f"{desc.name} must be rank-4 (B, H, S, D); got {desc.ndim}",
            )
            if self.thd:
                self._value_error_if(
                    not bshd_layout_ok(desc.shape, desc.stride),
                    f"THD (ragged) {desc.name} must be BSHD-physical (stride order 3,1,2,0, size-1 dims wildcarded); got stride {desc.stride}",
                )
            else:
                self._value_error_if(
                    not dense_layout_ok(desc.shape, desc.stride),
                    f"{desc.name} must have the head dim innermost-contiguous (stride 1) and "
                    f"non-broadcast, non-overlapping strides (any B/H/S order, padded "
                    f"strides allowed); got stride {desc.stride} shape {desc.shape}",
                )

        b, h_q, s_q, d_q = self.q_desc.shape
        _, h_kv, s_kv, _ = self.k_desc.shape
        d_v = self.v_desc.shape[3]
        self._check_tensor_shape(self.k_desc, (b, h_kv, s_kv, d_q), name="K")
        self._check_tensor_shape(self.v_desc, (b, h_kv, s_kv, d_v), name="V")
        self._check_tensor_shape(self.o_desc, (b, h_q, s_q, d_v), name="O")
        if self.lse_desc is not None:
            self._check_dtype(self.lse_desc, torch.float32, name="LSE")
            self._check_tensor_shape(self.lse_desc, (b, h_q, s_q), name="LSE")
            if self.thd:
                stride_h, stride_s = tuple(self.lse_desc.stride[1:])
                token_major = (stride_h, stride_s) == (1, h_q)
                head_major = not token_major and stride_s == 1 and stride_h >= 1
                self._value_error_if(
                    not token_major and not head_major,
                    f"THD LSE must be packed token-major (stride_h == 1, stride_s == H) "
                    f"or head-major (stride_s == 1, stride_h == head_stride); got stride {self.lse_desc.stride}",
                )
                self.thd_stats_head_major = head_major
                self.thd_stats_head_stride = int(stride_h) if head_major else 0
            else:
                self._value_error_if(not self.lse_desc.is_contiguous(), "LSE must be contiguous on SM120 DSL")

        for label, val in (
            ("B", b),
            ("H_q", h_q),
            ("H_kv", h_kv),
            ("S_q", s_q),
            ("S_kv", s_kv),
            ("D_QK", d_q),
            ("D_V", d_v),
        ):
            self._value_error_if(int(val) <= 0, f"{label} must be > 0; got {val}")
        self._value_error_if(
            h_q % h_kv != 0,
            f"H_q ({h_q}) must be divisible by H_kv ({h_kv}) for GQA / MQA",
        )
        self._value_error_if(
            d_q not in _SM120_SUPPORTED_HEAD_TILES,
            f"D_QK ({d_q}) must be one of {_SM120_SUPPORTED_HEAD_TILES}",
        )
        self._value_error_if(
            d_v not in _SM120_SUPPORTED_HEAD_TILES,
            f"D_V ({d_v}) must be one of {_SM120_SUPPORTED_HEAD_TILES}",
        )

        self.dtype = self._check_dtype(self.q_desc, [torch.float16, torch.bfloat16], name="Q")
        for desc in (self.k_desc, self.v_desc, self.o_desc):
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
            self.q_tile not in _SM120_Q_TILES,
            f"q_tile must be one of {_SM120_Q_TILES}",
        )
        self._value_error_if(
            self.kv_tile not in _SM120_KV_TILES,
            f"kv_tile must be one of {_SM120_KV_TILES}",
        )
        self._value_error_if(
            self.causal_bottom_right and not self.is_causal,
            "causal_bottom_right requires is_causal=True",
        )
        self._value_error_if(
            self.window_size_left is not None and self.window_size_left < 0,
            f"window_size_left must be non-negative, got {self.window_size_left}",
        )
        self._value_error_if(
            self.seq_q_lens_present and not self.seq_kv_lens_present,
            "seq_q_lens_present requires seq_kv_lens_present (padding mask)",
        )

        self._runtime_error_if(not torch.cuda.is_available(), "CUDA is not available")
        self.compute_capability = torch.cuda.get_device_capability(self.q_desc.device)
        self._runtime_error_if(
            self.compute_capability not in {(12, 0), (12, 1)},
            f"SdpaFwdDslSm120 requires SM120 or SM121, found SM{self.compute_capability[0]}{self.compute_capability[1]}",
        )

        import cutlass

        arch = f"sm_{self.compute_capability[0]}{self.compute_capability[1]}"
        smem_capacity_bytes = cutlass.utils.get_smem_capacity_in_bytes(arch)

        def _smem_bytes(kv_tile: int) -> int:
            # One K tile (D_QK wide) + one V tile (D_V wide), aliased with the
            # q_tile x D_V output staging tile.
            return max(kv_tile * (d_q + d_v) * self.dtype.itemsize, self.q_tile * d_v * self.dtype.itemsize) + 16

        if self.tile_n is None:
            # Pick the largest KV tile that fits this device.
            self.kv_tile = next((t for t in _SM120_KV_TILES if _smem_bytes(t) <= smem_capacity_bytes), self.kv_tile)
        self._not_implemented_error_if(
            _smem_bytes(self.kv_tile) > smem_capacity_bytes,
            (
                f"SM120 prefill requires {_smem_bytes(self.kv_tile)} bytes of shared memory for D={d_q}, "
                f"q_tile={self.q_tile}, and kv_tile={self.kv_tile}, but {arch} provides {smem_capacity_bytes} bytes"
            ),
        )

        if self.scale_softmax is None or self.scale_softmax == 0.0:
            self.scale_softmax = 1.0 / math.sqrt(d_q)

        self.batch_size = int(b)
        self.s_q_max = int(s_q)
        self.s_k_max = int(s_kv)
        self.h_q = int(h_q)
        self.h_kv = int(h_kv)
        self.head_dim_qk = int(d_q)
        self.head_dim_v = int(d_v)
        self._is_supported = True

        self._logger.debug("check_support completed successfully")
        return True

    def compile(self) -> None:
        """Compile the shape-specialized SM120 FROST template."""

        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return

        params = Sm120TemplateParams(
            dtype_qkv=_SM120_DTYPE_QKV_CODE[self.dtype],
            is_causal=self.is_causal,
            causal_bottom_right=self.causal_bottom_right,
            window_size_left=self.window_size_left,
            seq_q_lens_present=self.seq_q_lens_present,
            seq_kv_lens_present=self.seq_kv_lens_present,
            has_sink=self.has_sink,
            thd_varlen=self.thd,
            q_tile=self.q_tile,
            kv_tile=self.kv_tile,
        )
        self._k_mod = _load_sm120_kernel_module(params)
        if self.thd:
            # The packed token totals (and max sequence length) are runtime
            # values, so the per-shape compile is deferred to execute().
            self._compiled_kernel = "thd-deferred"
            self._logger.debug("compile completed (THD per-shape compile deferred)")
            return
        self._compiled_kernel = self._k_mod.compile(
            compute_capability=self.compute_capability,
            b=self.batch_size,
            qh=self.h_q,
            kh=self.h_kv,
            sq=self.s_q_max,
            skv=self.s_k_max,
            d_qk=self.head_dim_qk,
            d_v=self.head_dim_v,
            # No sample_lse -> the LSE store is compiled out; execute() then
            # binds no LSE buffer at all (no dummy, no allocation).
            has_lse=self.lse_desc is not None,
        )
        self._logger.debug("compile completed")

    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        lse_tensor: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        seq_q_lens: Optional[torch.Tensor] = None,
        seq_kv_lens: Optional[torch.Tensor] = None,
        scale_softmax: Optional[float] = None,
        workspace: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Execute tensors matching the compiled specialization."""

        if self._compiled_kernel is None:
            raise RuntimeError("SdpaFwdDslSm120 kernel is not compiled")
        self._value_error_if(
            self.has_sink and sinks is None,
            "sinks is required by this compiled specialization",
        )
        self._value_error_if(
            not self.has_sink and sinks is not None,
            "this specialization was compiled without sink support; construct the API with has_sink=True",
        )
        self._check_seq_lens_contract(seq_q_lens, seq_kv_lens)
        self._value_error_if(
            self.lse_desc is not None and lse_tensor is None,
            "lse_tensor is required by this compiled specialization",
        )
        self._value_error_if(
            self.lse_desc is None and lse_tensor is not None,
            "this specialization was compiled without an LSE output; construct the API with sample_lse",
        )
        scale_val = self.scale_softmax if scale_softmax is None or scale_softmax == 0.0 else float(scale_softmax)
        scale_softmax_log2 = scale_val * math.log2(math.e)
        if self.thd:
            self._execute_thd(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                scale_softmax_log2,
                sinks,
                seq_kv_lens,
                seq_q_lens,
                lse_tensor=lse_tensor,
                workspace=workspace,
                current_stream=current_stream,
            )
            return
        lse = self._checked_lse_view(lse_tensor) if lse_tensor is not None else None
        sinks_t = self._checked_sinks_1d(sinks) if sinks is not None else None
        seq_q_lens = (
            self._checked_seq_lens(seq_q_lens, "seq_q_lens")
            if seq_q_lens is not None
            else self._dummy(
                "seq_q_lens",
                q_tensor.device,
                lambda: torch.zeros(self.batch_size, dtype=torch.int32, device=q_tensor.device),
            )
        )
        seq_kv_lens = (
            self._checked_seq_lens(seq_kv_lens, "seq_kv_lens")
            if seq_kv_lens is not None
            else self._dummy(
                "seq_kv_lens",
                q_tensor.device,
                lambda: torch.zeros(self.batch_size, dtype=torch.int32, device=q_tensor.device),
            )
        )
        if current_stream is None:
            # Direct call (no dispatch-forwarded stream): fall back to torch's
            # current stream, as before. A stream forwarded from the execute-time
            # handle is respected rather than clobbered.
            current_stream = cuda.CUstream(torch.cuda.current_stream(q_tensor.device).cuda_stream)

        import cutlass

        q = self._to_bshd(q_tensor)
        k = self._to_bshd(k_tensor)
        v = self._to_bshd(v_tensor)
        o_view, o_needs_copy_back, o_scratch = self._to_bshd_writable(o_tensor)
        o = o_scratch if o_needs_copy_back else o_view
        self._compiled_kernel(
            q,
            k,
            v,
            o,
            lse,
            sinks_t,
            seq_q_lens,
            seq_kv_lens,
            cutlass.Float32(scale_softmax_log2),
            current_stream,
        )
        if o_needs_copy_back:
            o_view.copy_(o_scratch)

    def _execute_thd(
        self, q_buf, k_buf, v_buf, o_buf, scale_softmax_log2, sinks, seq_kv_lens, seq_q_lens, lse_tensor=None, workspace=None, current_stream=None
    ):
        """THD (ragged) execute: packed ``(1, T, H, D)`` views + cu_seqlens.

        Serves the same fully-packed contract as the SM100 THD path
        (``ragged_offset == cumsum(seq_len) * H * D`` from 0, multiplier 1);
        the offsets are re-derived from ``seq_len_q``/``seq_len_kv``. The two
        ``.tolist()`` D2H syncs are inherent — the packed totals and the
        longest sequence's Q length are runtime values that size the
        per-execute compile and grid. ``lse_tensor``, when given, is the
        caller's ragged Stats buffer, in its declared layout: token-major
        packed ``(T, H)`` in the first ``T*H`` elements, or head-major
        ``(H, head_stride)`` with tokens contiguous within each head row.
        """

        b, qh, kh = self.batch_size, self.h_q, self.h_kv
        d_qk, d_v = self.head_dim_qk, self.head_dim_v
        dev = q_buf.device
        carver = WorkspaceCarver(workspace, self.scratch_workspace_bytes(), "SdpaFwdDslSm120 (THD)") if workspace is not None else None

        slq_v = self._checked_seq_lens(seq_q_lens, "seq_q_lens")
        slk_v = self._checked_seq_lens(seq_kv_lens, "seq_kv_lens")
        # [seq_kv(B) | cu_q(B+1) | cu_k(B+1)] — bound as the kernel's
        # seq_kv_lens tensor; the leading B words alias the per-sequence KV
        # lengths so the kernel's existing padded-mask read works unchanged.
        # Built HOST-side from the (inherent) tolist round-trip and uploaded
        # in ONE H2D copy: a device-side cumsum would allocate its scan-temp
        # storage and launch kernels on the execute hot path.
        meta = carver.take(3 * b + 2, torch.int32) if carver is not None else torch.empty(3 * b + 2, dtype=torch.int32, device=dev)
        slq_host = slq_v.tolist()
        slk_host = slk_v.tolist()
        cu_q_host = [0]
        for n in slq_host:
            cu_q_host.append(cu_q_host[-1] + int(n))
        cu_k_host = [0]
        for n in slk_host:
            cu_k_host.append(cu_k_host[-1] + int(n))
        meta.copy_(torch.tensor(slk_host + cu_q_host + cu_k_host, dtype=torch.int32))
        t_q = cu_q_host[-1]
        t_kv = cu_k_host[-1]
        max_sq = max(slq_host) if slq_host else 0

        if t_q == 0:
            return
        lse = None
        if lse_tensor is not None:
            if self.thd_stats_head_major:
                head_stride = self.thd_stats_head_stride
                self._value_error_if(
                    head_stride < t_q,
                    f"head-major THD LSE head_stride ({head_stride}) must cover the packed Q token total ({t_q})",
                )
                lse = lse_tensor.as_strided((qh, head_stride), (head_stride, 1), lse_tensor.storage_offset())
            else:
                lse = lse_tensor.as_strided((t_q, qh), (qh, 1), lse_tensor.storage_offset())

        # Sinks are None-specialized like the LSE when the graph has no sink
        # token.
        sinks_t = self._checked_sinks_1d(sinks) if sinks is not None else None
        seq_q_dummy = self._dummy(
            "seq_q_lens",
            dev,
            lambda: torch.zeros(b, dtype=torch.int32, device=dev),
        )

        def _packed(buf, tokens):
            heads = qh if buf is q_buf or buf is o_buf else kh
            d = d_qk if buf is q_buf or buf is k_buf else d_v
            return buf.as_strided((1, tokens, heads, d), (tokens * heads * d, heads * d, d, 1), buf.storage_offset())

        if t_kv == 0:
            # Every query row is dead (all-zero seq_kv_lens): served by the
            # KERNEL's own dead-row path (row_sum <= 0 -> O := 0 and
            # LSE := -inf, or the sink alone — its column keeps the softmax
            # denominator alive), exactly like a live launch's zero-KV
            # sequences — no adapter-side fills on the execute hot path
            # (AGENTS.md Rule 1). A zero-token packed K/V view cannot back a
            # CuTe layout, so clamp the packed KV extent to ONE
            # never-dereferenced token (every sequence's KV tile range is
            # empty, so no K/V load is ever issued) bound over storage
            # guaranteed large enough: Q backs K (kh*d_qk <= t_q*qh*d_qk) and
            # O backs V (kh*d_v <= t_q*qh*d_v).
            t_kv = 1
            K = q_buf.as_strided((1, 1, kh, d_qk), (kh * d_qk, kh * d_qk, d_qk, 1), q_buf.storage_offset())
            V = o_buf.as_strided((1, 1, kh, d_v), (kh * d_v, kh * d_v, d_v, 1), o_buf.storage_offset())
        else:
            K = _packed(k_buf, t_kv)
            V = _packed(v_buf, t_kv)

        if current_stream is None:
            current_stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)

        import cutlass

        fn = self._k_mod.compile(
            compute_capability=self.compute_capability,
            b=b,
            qh=qh,
            kh=kh,
            sq=t_q,
            skv=t_kv,
            d_qk=d_qk,
            d_v=d_v,
            max_sq=max_sq,
            has_lse=self.lse_desc is not None,
            lse_head_major=self.thd_stats_head_major,
            lse_head_stride=self.thd_stats_head_stride,
        )
        fn(
            _packed(q_buf, t_q),
            K,
            V,
            _packed(o_buf, t_q),
            lse,
            sinks_t,
            seq_q_dummy,
            meta,
            cutlass.Float32(scale_softmax_log2),
            current_stream,
        )

    def scratch_workspace_bytes(self) -> int:
        if self.thd:
            # [meta(seq_kv, cu_q, cu_k)].
            # No packed-LSE chunk: with a Stats output the kernel writes the
            # caller's ragged Stats buffer directly (token-major (T, H) or
            # head-major (H, head_stride)); without one it compiles with
            # has_lse=False and no LSE buffer exists at all. No slq/slk
            # copies either: the metadata is built host-side from the tolist
            # round-trip. No sinks-dummy chunk: the kernel None-specializes
            # on sinks. No O-descriptor chunk: SM120 stores O with plain
            # guarded GMEM stores, so THD needs no per-sequence tensor maps.
            b = self.batch_size
            return ws_align((3 * b + 2) * 4)
        return 0


def sdpa_fwd_wrapper_dsl_sm120(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    is_causal: bool = False,
    causal_bottom_right: bool = False,
    window_size_left: Optional[int] = None,
    scale_softmax: Optional[float] = None,
    seq_q_lens: Optional[torch.Tensor] = None,
    seq_kv_lens: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    q_tile: Optional[int] = None,
    kv_tile: Optional[int] = None,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """SM120 SDPA forward; returns ``TupleDict(o_tensor=..., lse_tensor=...)``."""

    if current_stream is not None:
        raise NotImplementedError(
            "sdpa_fwd_wrapper_dsl_sm120: explicit current_stream is not "
            "yet supported. Wrap the call in `with torch.cuda.stream(s):` to "
            "dispatch onto a non-default stream."
        )
    if q_tensor.ndim != 4 or k_tensor.ndim != 4 or v_tensor.ndim != 4:
        raise ValueError(f"Q, K, and V must be rank-4 BHSD; got Q={q_tensor.ndim}D K={k_tensor.ndim}D V={v_tensor.ndim}D")
    b, h_q, s_q, _ = q_tensor.shape
    d_v = v_tensor.shape[-1]
    o_tensor = torch.empty(
        (b, s_q, h_q, d_v),
        dtype=q_tensor.dtype,
        device=q_tensor.device,
    ).transpose(1, 2)
    lse_tensor = _allocate_lse_tensor(q_tensor)
    cache_key = _make_cache_key(
        SdpaFwdDslSm120,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse=lse_tensor,
        is_causal=is_causal,
        causal_bottom_right=causal_bottom_right,
        window_size_left=window_size_left,
        scale_softmax=scale_softmax,
        seq_q_lens_present=seq_q_lens is not None,
        seq_kv_lens_present=seq_kv_lens is not None,
        has_sink=sinks is not None,
        tile_m=_SM120_Q_TILES[0] if q_tile is None else q_tile,
        tile_n=_SM120_KV_TILES[0] if kv_tile is None else kv_tile,
    )
    sdpa_fwd = _get_or_create_api(
        cache_key,
        sample_q=q_tensor,
        sample_k=k_tensor,
        sample_v=v_tensor,
        sample_o=o_tensor,
        sample_lse=lse_tensor,
        seq_q_lens_present=seq_q_lens is not None,
        seq_kv_lens_present=seq_kv_lens is not None,
        has_sink=sinks is not None,
        is_causal=is_causal,
        causal_bottom_right=causal_bottom_right,
        window_size_left=window_size_left,
        scale_softmax=scale_softmax,
        tile_m=q_tile,
        tile_n=kv_tile,
    )
    sdpa_fwd.execute(
        q_tensor=q_tensor,
        k_tensor=k_tensor,
        v_tensor=v_tensor,
        o_tensor=o_tensor,
        lse_tensor=lse_tensor,
        sinks=sinks,
        seq_q_lens=seq_q_lens,
        seq_kv_lens=seq_kv_lens,
        scale_softmax=scale_softmax,
        current_stream=current_stream,
    )
    return TupleDict(o_tensor=o_tensor, lse_tensor=lse_tensor)
