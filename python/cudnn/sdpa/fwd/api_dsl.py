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
from types import SimpleNamespace
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
    SCHED_LPT,
    SCHED_LPT_L2,
    SCHED_NATURAL,
)
from cudnn.sdpa.fwd.config_sm100 import TemplateParams as Sm100TemplateParams, pack_gqa_supported
from cudnn.sdpa.fwd.config_sm120 import (
    HEAD_TILE_GRANULE as _SM120_HEAD_TILE_GRANULE,
    SEQ_KV_TILES as _SM120_KV_TILES,
    SEQ_Q_TILES as _SM120_Q_TILES,
    SUPPORTED_HEAD_TILE_MAX as _SM120_HEAD_TILE_MAX,
    FP8_HEAD_TILE_GRANULE as _SM120_FP8_HEAD_TILE_GRANULE,
    TemplateParams as Sm120TemplateParams,
    smem_bytes as _sm120_smem_bytes,
)


def dtype_name(buffer) -> str:
    """The buffer's dtype as a bare name, whoever produced it.

    A caller buffer reaches these checks as whatever the graph normalized it
    into, which is a variant-pack slot rather than a torch tensor. Comparing
    ``buffer.dtype is torch.float32`` therefore rejects a perfectly good fp32
    buffer with "must be float32; got float32". Names are the one spelling
    every producer agrees on -- torch prints ``torch.float32``, numpy and the
    slot print ``float32``.
    """
    return str(buffer.dtype).rsplit(".", 1)[-1]


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
# FP8 kernel family; the output dtype is encoded the same way.
_SM100_DTYPE_QKV_CODE = {
    torch.float8_e4m3fn: DTYPE_E4M3,
    torch.float8_e5m2: DTYPE_E5M2,
    torch.bfloat16: DTYPE_BF16,
    torch.float16: DTYPE_FP16,
}
_SM100_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
# FP8 kernels use E4M3/E5M2 inputs and BF16/FP16/FP8 outputs. Block-scale
# Both FP8 paths have exact d128/d128 and d192/d128 kernels.
_SM100_MXFP8_KERNEL_FILES = {
    (128, 128): "prefill_d128_mxfp8_sm100.py",
    (192, 128): "prefill_d192_d128_mxfp8_sm100.py",
}
_SM107_FP8_KERNEL_FILE = "prefill_d128_fp8_sm107.py"
_SM100_FP8_KERNEL_FILES = {
    (128, 128): "prefill_d128_fp8_sm100.py",
    (192, 128): "prefill_d192_d128_fp8_sm100.py",
}


def _sm100_fp8_shapes(pertensor: bool, device_cc: tuple[int, int]) -> frozenset[tuple[int, int]]:
    if device_cc == (10, 7):
        return frozenset({(128, 128)})
    return frozenset({(128, 128), (192, 128)})


# Both flavors tile KV in TILE_N=128 columns; the KV tail is only masked when
# the padded/causal mask paths are active (see check_support).
_SM100_TILE_N = 128

_SM120_KERNEL_FILE = "prefill_f16_sm120.py"
# Per-tensor FP8 kernel (E4M3/E5M2 in, FP16/BF16/FP8 out, mma.sync
# m16n8k32); selected by the graph op (sdpa_fp8) via check_support's dtype.
_SM120_FP8_KERNEL_FILE = "prefill_fp8_sm120.py"
_SM120_DTYPE_QKV_CODE = {
    torch.float8_e4m3fn: DTYPE_E4M3,
    torch.float8_e5m2: DTYPE_E5M2,
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
    if handle in (0, 1, 2):
        # Legacy default / per-thread-default stream sentinels: wrapping one in
        # ExternalStream breaks re-execution on some torch builds (NGC), where
        # every launch after the compile run silently no-ops (all-zero outputs;
        # caught by test_mhas_v2's determinism re-run). Torch work is already
        # ordered against the default stream here, so run in place.
        yield
        return
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


# Causal-balancing scheduler choice: use the L2-GROUPED LPT when ONE head's K+V
# working set fits this budget -- that is the condition under which the
# block-cyclic grouping can actually keep that K/V resident; otherwise plain
# reverse-row LPT.
_SCHED_L2_BUDGET_BYTES = 50 * 1024 * 1024


def _causal_sched_policy(s_kv: int, d_qk: int, d_v: int, elem_bytes: int) -> int:
    """SCHED_LPT_L2 vs SCHED_LPT for a causal graph (see _SCHED_L2_BUDGET_BYTES)."""
    one_head_bytes = int(s_kv) * (int(d_qk) + int(d_v)) * int(elem_bytes)
    return SCHED_LPT_L2 if _SCHED_L2_BUDGET_BYTES >= one_head_bytes else SCHED_LPT


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
    are OOB-clipped. Per-tensor FP8 serves the same envelope under its d128
    tile (dense only, d % 16 at 1 byte/elem); d192 FP8 and MXFP8 use exact
    native shapes. All of it is gated in check_support / engines.mismatch,
    including the f16 alignment rule (d % 8, the TMA 16-byte global-stride
    rule at 2 bytes/elem).
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


def _load_sm100_kernel_module(flavor: tuple[int, int], params: Sm100TemplateParams, fp8: bool = False, pertensor: bool = False, rubin: bool = False):
    """Load one SM100-family module for the selected flavor and quantization
    path.  ``rubin`` routes per-tensor FP8 to the SM107 sibling kernel (the
    dense K=64 FP8 path baked in — see prefill_d128_fp8_sm107.py)."""

    tag = _flavor_tag(flavor)
    if fp8 and pertensor and rubin and flavor == (128, 128):
        filename = _SM107_FP8_KERNEL_FILE
        tag = f"sdpa_fwd_sm107_fp8_{tag}"
    elif fp8:
        filename = _SM100_FP8_KERNEL_FILES[flavor] if pertensor else _SM100_MXFP8_KERNEL_FILES[flavor]
        tag = f"sdpa_fwd_sm100_{'fp8' if pertensor else 'mxfp8'}_{tag}"
    else:
        filename = _SM100_KERNEL_FILES[flavor]
        tag = f"sdpa_fwd_sm100_{tag}"
    return _load_kernel_template(filename, params, tag)


def _load_sm120_kernel_module(params: Sm120TemplateParams, fp8: bool = False):
    if fp8:
        return _load_kernel_template(_SM120_FP8_KERNEL_FILE, params, tag="sdpa_fwd_sm120_fp8")
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
        window_size_right: Optional[int] = None,
        scale_softmax: Optional[float] = None,
        seq_kv_lens_present: bool = False,
        seq_q_lens_present: bool = False,
        cu_seq_q_lens: bool = False,
        cu_seq_kv_lens: bool = False,
        has_sink: bool = False,
        thd: bool = False,
        dtype_o: Optional[torch.dtype] = None,
        pertensor_fp8: bool = False,
        sched_policy: Optional[int] = None,
        tile_m: Optional[int] = None,
        tile_n: Optional[int] = None,
        cga: Optional[int] = None,
        split_kv: Optional[int] = None,
        softmax_precision: Optional[int] = None,
        pack_gqa: Optional[bool] = None,
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
        # window_size_right is the diagonal-band right bound R ("keep k in
        # [.., q+R]", cuDNN diagonal_band_right_bound): the causal upper limit
        # widened by R columns. None = no right band; requires is_causal (the
        # band is the causal diagonal, possibly widened).
        self.window_size_right = window_size_right
        # The ONE canonical band every adapter lowers (same model as the
        # analyzer facts / config TemplateParams): per-side offsets from the
        # diagonal, None = unbounded. is_causal means "right bound 0";
        # window_size_right widens it (check_support validates it requires
        # is_causal). The padding mask stays orthogonal (seq_kv_lens_present).
        self.window_left: Optional[int] = window_size_left
        self.window_right: Optional[int] = (window_size_right or 0) if self.is_causal else None
        self.scale_softmax = scale_softmax
        self.seq_kv_lens_present = bool(seq_kv_lens_present)
        # Dense padded-Q trim (cuDNN >= 9.14): q rows >= seq_len_q[b] write
        # O := 0 / LSE := -inf. The per-batch Q lengths are a SEPARATE
        # (B,)-int32 kernel parameter (like cuDNN's SEQLEN_Q pointer and
        # FA's seqused_q) bound directly at execute — no packing, no
        # per-execute copies. Dense-only.
        self.seq_q_lens_present = bool(seq_q_lens_present)
        # cu_seq_len form (cuDNN 9.24+): the corresponding seq-lens execute
        # argument arrives as a (B+1,)-int32 PREFIX-SUM tensor instead of
        # (B,) per-batch lengths. THD-only today: the setup kernels consume
        # either form on device (issue #552); the dense
        # kernels have no CU read mode yet (check_support rejects).
        self.cu_seq_q_lens = bool(cu_seq_q_lens)
        self.cu_seq_kv_lens = bool(cu_seq_kv_lens)
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
        # Capabilities domain by the probe (engines.mismatch). None means the
        # caller stated NO preference: the graph path always arrives with an
        # explicit value (the heuristic emits complete assignments), so a None
        # here is the standalone-wrapper tier, where compile() derives the
        # policy itself. An explicit value — including NATURAL — is honored
        # verbatim, never re-derived.
        self.sched_policy = None if sched_policy is None else int(sched_policy)
        self.tile_m = None if tile_m is None else int(tile_m)
        self.tile_n = None if tile_n is None else int(tile_n)
        self.cga = None if cga is None else int(cga)
        self.split_kv = 1 if split_kv is None else int(split_kv)
        # Framework axis: no forward kernel serves a softmax-precision choice
        # yet, so anything non-None is rejected in check_support.
        self.softmax_precision = softmax_precision
        self.pack_gqa = bool(pack_gqa) if pack_gqa is not None else False

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

    # -- THD declared-stride binding ------------------------------------------
    # A THD tensor may DECLARE a wider token stride than the packed h*d — e.g.
    # a K/V view of a kv-interleaved [T, 2, H, D] buffer (token stride 2*h*d),
    # the layout torch.nn.attention.varlen users produce by slicing a fused KV
    # projection. The f16 kernels address declared strides NATIVELY
    # (layout-driven offset math + TMA-encoded strides); declarations the
    # hardware cannot express are REJECTED in check_support — no
    # normalization-copy fallback (AGENTS.md Hard Rule 2) — so the router
    # picks an engine that honors them instead.

    @staticmethod
    def _thd_declared(desc: TensorDesc):
        """(token, head, elem) strides a THD tensor declares, and whether they
        are the packed contract (h*d, d, 1)."""
        h, d = desc.shape[1], desc.shape[3]
        st = (int(desc.stride[2]), int(desc.stride[1]), int(desc.stride[3]))
        return st, st == (h * d, d, 1)

    def _thd_check_strides_native(self) -> None:
        """Reject THD stride declarations the kernels cannot address
        natively: TMA's 16-byte global-stride rule — the head dim must be
        innermost-contiguous (elem stride 1) and the token/head strides
        multiples of ``16 // itemsize`` elements (which also keeps every
        per-sequence ragged base 16-byte aligned). Whole-token gaps always
        qualify for supported head dims; sub-token gaps only in 16-byte
        multiples. The strides must also COVER the tensor (head >= d,
        token >= h*head): an overlapping declaration would alias distinct O
        rows onto the same storage (a write race) and is outside the
        kernels' addressing contract."""
        for desc in (self.q_desc, self.k_desc, self.v_desc, self.o_desc):
            (ts, hs, es), _ = self._thd_declared(desc)
            h, d = desc.shape[1], desc.shape[3]
            # The 16-byte TMA rule in this tensor's OWN element units: 8 at
            # 2 B/elem (f16/bf16), 16 at 1 B/elem (fp8), 4 at 4 B/elem.
            quantum = 16 // desc.dtype.itemsize
            self._not_implemented_error_if(
                es != 1 or ts % quantum != 0 or hs % quantum != 0 or hs < d or ts < h * hs,
                f"{desc.name} THD strides {tuple(desc.stride)} are not TMA-expressible "
                f"(head dim must be innermost-contiguous, token/head strides 16-byte — "
                f"{quantum}-element — multiples, and non-overlapping: head stride >= {d}, "
                f"token stride >= heads * head stride)",
            )

    def _thd_check_strides_packed(self) -> None:
        """FP8 THD serves only the packed contract for now (its kernel and
        harness are not audited for declared strides) — decline anything else
        rather than adapt (AGENTS.md Hard Rule 2)."""
        for desc in (self.q_desc, self.k_desc, self.v_desc, self.o_desc):
            _, packed = self._thd_declared(desc)
            self._not_implemented_error_if(
                not packed,
                f"{desc.name}: non-packed THD strides {tuple(desc.stride)} are not supported by the FP8 path yet",
            )

    def _thd_view(self, buf: torch.Tensor, desc: TensorDesc, tokens: int) -> torch.Tensor:
        """The declared-stride ``(1, T, H, D)`` view over a THD buffer's storage.

        Validates the RUNTIME buffer against the declaration before
        reinterpreting its storage: the dtype/device must match what was
        declared, and the base address must be 16-byte aligned — the kernels
        are compiled with ``assumed_align=16`` and TMA requires it of the
        descriptor's global address, so a misaligned slice would fault (or
        worse) instead of erroring here. ``as_strided`` itself rejects views
        that extend past the underlying storage."""
        h, d = desc.shape[1], desc.shape[3]
        (ts, hs, es), _ = self._thd_declared(desc)
        self._value_error_if(
            buf.dtype != desc.dtype or buf.device != desc.device,
            f"{desc.name}: runtime buffer ({buf.dtype}, {buf.device}) does not match its declaration ({desc.dtype}, {desc.device})",
        )
        self._value_error_if(
            buf.data_ptr() % 16 != 0,
            f"{desc.name}: runtime buffer base address must be 16-byte aligned (TMA global-address rule); got data_ptr() % 16 == {buf.data_ptr() % 16}",
        )
        return buf.as_strided((1, tokens, h, d), (max(tokens, 1) * ts, ts, hs, es), buf.storage_offset())

    def _amax_slot(self, tensor, name: str, device: torch.device) -> torch.Tensor:
        """The caller's 1-element amax storage, or a cached dummy.

        view(), not reshape(): the kernel atomicMax'es into whatever this
        returns, and reshape() silently hands back a COPY when the input is not
        contiguous -- the kernel would write the copy and the caller would read
        back the zeros it was reset to.
        """
        if tensor is None:
            return self._dummy(name, device, lambda: torch.zeros(1, dtype=torch.float32, device=device))
        # Both callers re-view the slot as Int32 for the kernel ABI, so a wider
        # element would yield two int32s and the kernel would write only the low
        # word -- the caller then reads a corrupted value.
        if dtype_name(tensor) != "float32":
            raise ValueError(f"{name} must be float32; got {tensor.dtype}")
        try:
            return tensor.view(-1)[:1]
        except RuntimeError as exc:
            raise ValueError(f"{name} must be contiguous — the kernel writes it in place; got strides {tuple(tensor.stride())}") from exc

    def _scale_view(self, t, name: str, device: torch.device) -> torch.Tensor:
        """A per-tensor scale as the kernel's 1-element fp32 device view.

        ``None`` binds a cached 1.0 dummy (identity fold) — the kernels take
        the scale tensors unconditionally so there is exactly one compile
        form and execute never reads a value back to the host (Rule 3)."""
        if t is None:
            return self._dummy("scale_one", device, lambda: torch.ones(1, dtype=torch.float32, device=device))
        self._value_error_if(
            not isinstance(t, torch.Tensor) or t.device.type != "cuda",
            f"{name} must be a CUDA tensor; got {type(t).__name__}",
        )
        self._value_error_if(
            t.dtype != torch.float32 or t.numel() < 1,
            f"{name} must be a 1-element fp32 tensor; got dtype={t.dtype} numel={t.numel()}",
        )
        return t.reshape(-1)[:1]

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

        The logical contract is exactly ``B*H_q*S_q`` fp32 elements. Graph
        Stats commonly arrive as a rank-4 ``(B, H_q, S_q, 1)`` view, which is
        reinterpreted as the declared rank-3 LSE layout without copying. The
        kernel writes through the returned view, so a silent ``reshape`` copy
        of a non-contiguous buffer would leave the caller's Stats unwritten.
        Dense adapters therefore record ``_lse_stride`` and rebuild that
        declared view directly over the caller's storage.
        """
        self._value_error_if(
            dtype_name(lse_tensor) != "float32",
            f"lse_tensor must be float32; got {lse_tensor.dtype}",
        )
        expected = self.batch_size * self.h_q * self.s_q_max
        self._value_error_if(
            lse_tensor.numel() != expected,
            f"lse_tensor must have B*H_q*S_q = {expected} elements; got {lse_tensor.numel()}",
        )
        shape = (self.batch_size, self.h_q, self.s_q_max)
        stride = getattr(self, "_lse_stride", None)
        if stride is None:
            self._value_error_if(
                not lse_tensor.is_contiguous(),
                "lse_tensor must be contiguous (the kernel writes through this buffer)",
            )
            return lse_tensor.view(shape)
        if tuple(lse_tensor.shape) == shape and tuple(lse_tensor.stride()) == stride:
            return lse_tensor
        try:
            return lse_tensor.as_strided(shape, stride, lse_tensor.storage_offset())
        except RuntimeError as exc:
            raise ValueError(
                f"lse_tensor backing storage is too small for declared shape {shape}, stride {stride}, and storage_offset {lse_tensor.storage_offset()}"
            ) from exc

    def _checked_sinks_1d(self, sinks: torch.Tensor) -> torch.Tensor:
        """Validate caller-provided sink logits and return the kernel's (H_q,) fp32 view.

        Strictly a view: the kernels consume fp32 sinks directly, and an
        implicit ``.to(float32)`` here would allocate and launch a cast kernel
        on the execute hot path (and break CUDA-graph pointer stability).
        """
        self._value_error_if(
            dtype_name(sinks) != "float32",
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
            dtype_name(seq_lens) != "int32",
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

    def _checked_cu_seq_lens(self, cu_seq_lens: torch.Tensor, name: str) -> torch.Tensor:
        """Validate a caller-provided (B+1,)-int32 prefix-sum tensor (cu_seq_len form).

        Strictly a view, like :meth:`_checked_seq_lens`. The prefix-sum
        INVARIANTS (non-decreasing; any base — the setup kernel normalizes by
        subtracting element 0) are runtime values and caller contract: a
        validation that needs a device read is not a validation (Rule 3).
        """
        self._value_error_if(
            cu_seq_lens.dtype != torch.int32,
            f"{name} must be int32; got {cu_seq_lens.dtype}",
        )
        self._value_error_if(
            cu_seq_lens.numel() != self.batch_size + 1,
            f"{name} must have B + 1 = {self.batch_size + 1} elements (prefix sums); got {cu_seq_lens.numel()}",
        )
        self._value_error_if(
            not cu_seq_lens.is_contiguous(),
            f"{name} must be contiguous (read as a flat (B+1,) view)",
        )
        return cu_seq_lens.reshape(-1)

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

    # -- KV-split shared helpers (SM100 + SM120 dense split paths) -----------

    def _o_itemsize(self) -> int:
        return 2  # f16 / bf16; the split path is half-precision-O only

    def _combine_dtype_tag(self) -> str:
        # The combine reduces INTO the O dtype: the graph's dtype_o on the
        # quantized rows (half-gated by check_support), Q's dtype elsewhere.
        o_dtype = self.dtype_o if (self._fp8 and self.dtype_o is not None) else self.dtype
        return "bf16" if o_dtype == torch.bfloat16 else "f16"

    def _split_partials(self, workspace, o_like, device, current_stream=None):
        """The split-major (O, LSE) partial buffers, carved from the caller's
        workspace when there is one and torch-allocated otherwise (standalone
        use, matching what the rest of this adapter does).

        The allocation happens ON the launch stream: the caching allocator tags
        a block with the stream it was allocated on, and the kernels that write
        and read these buffers run on ``current_stream``. Allocating on torch's
        current stream instead would leave a later free/reuse unordered against
        those launches."""
        rows = self.split_kv * self.batch_size
        o_shape = (rows, self.s_q_max, self.h_q, self.head_dim_v)
        lse_shape = (rows, self.h_q, self.s_q_max)
        if workspace is None:
            with _torch_stream_context(current_stream, device):
                return (
                    torch.empty(o_shape, dtype=o_like.dtype, device=device),
                    torch.empty(lse_shape, dtype=torch.float32, device=device),
                )
        carver = WorkspaceCarver(workspace, self.scratch_workspace_bytes(), f"{type(self).__name__} (KV split)")
        o_part = carver.take(rows * self.s_q_max * self.h_q * self.head_dim_v, o_like.dtype).view(o_shape)
        lse_part = carver.take(rows * self.h_q * self.s_q_max, torch.float32).view(lse_shape)
        return o_part, lse_part

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
        ``descale_q/descale_k/descale_v``, ``scale_o``, ``amax_o`` — see :meth:`SdpaFwdDslSm100.execute`).
        """


class SdpaFwdDslSm100(SdpaFwdDsl):
    """SM100 (Blackwell) SDPA forward via the FROST DSL template kernels."""

    def _initialize_implementation(self) -> None:
        self.flavor: Optional[tuple[int, int]] = None
        self.thd_stats_head_major = False
        self.thd_stats_head_stride = 0
        self._lse_stride: Optional[tuple[int, int, int]] = None
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

        if self.thd:
            if self.q_desc.dtype in _SM100_FP8_DTYPES:
                # FP8/MXFP8 THD serves only the packed contract (their
                # compile() builds compact fakes); the f16 kernels address
                # declared strides natively.
                self._thd_check_strides_packed()
            else:
                self._thd_check_strides_native()

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

        if self.pack_gqa:
            self._not_implemented_error_if(
                self.thd,
                "PackGQA is dense-only (THD/ragged runs unpacked)",
            )
            self._value_error_if(
                not pack_gqa_supported(int(h_qo), int(h_kv)),
                f"PackGQA requires h_q/h_kv to divide the kernel tile_m; got h_q/h_kv = {int(h_qo)}/{int(h_kv)}",
            )

        # Q/K/V dtype: half (BF16/FP16, DTYPE_O == input) or FP8 (E4M3/E5M2 → MXFP8,
        # d128 only, DTYPE_O independent — typically BF16/FP16).
        self.dtype = self._check_dtype(self.q_desc, [torch.float16, torch.bfloat16, *_SM100_FP8_DTYPES], name="Q")
        self._fp8 = self.dtype in _SM100_FP8_DTYPES
        self._not_implemented_error_if(
            self.pack_gqa and self._fp8 and not self._pertensor,
            "PackGQA is not supported for MXFP8: the F8_128x4 sf_q scale-factor atom "
            "bundles 128 rows of ONE head and is not TMA-gatherable at token granularity "
            "(see the SF layout note in prefill_d128_mxfp8_sm100.py)",
        )
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
                self._value_error_if(
                    not dense_layout_ok((*self.lse_desc.shape, 1), (*self.lse_desc.stride, 1)),
                    f"LSE must use a dense-compatible B/H/S permutation or padded layout "
                    f"with non-broadcast, non-overlapping-by-span strides; got {self.lse_desc.stride}",
                )
                self._lse_stride = None if self.lse_desc.is_contiguous() else tuple(int(stride) for stride in self.lse_desc.stride)

        self._value_error_if(not torch.cuda.is_available(), "CUDA must be available for SM100 DSL SDPA")
        device = self.q_desc.device
        major, minor = torch.cuda.get_device_capability(device)
        # cc10.0 (SM100) and cc10.3 (Blackwell-class) both run these kernels; cc10.3
        # additionally has the fused LDTM.STAT row-max, auto-enabled for MXFP8 in compile().
        self._device_cc = (major, minor)
        # cc10.7 (Rubin) is served by the per-tensor FP8 path only, through the
        # SM107 sibling kernel (dense K=64 FP8; the f16 and MXFP8 kernels'
        # K=32-QMMA / f16-QMMA geometry has not been ported).
        if self._fp8 and self._pertensor:
            _allowed_cc = ((10, 0), (10, 3), (10, 7))
            _allowed_msg = "cc=10.0/10.3 (Blackwell) or 10.7 (Rubin, per-tensor FP8)"
        else:
            _allowed_cc = ((10, 0), (10, 3))
            _allowed_msg = "cc=10.0 or 10.3 (Blackwell; Rubin cc10.7 serves only the per-tensor FP8 d128 path)"
        self._value_error_if(
            self._device_cc not in _allowed_cc,
            f"SdpaFwdDslSm100 requires {_allowed_msg}; found SM{major}{minor} on {device}",
        )

        # FP8 flavor shapes: SM100 serves d128/d128 and d192/d128; Rubin
        # currently serves only per-tensor FP8 d128. Per-tensor FP8 serves
        # the dense ENVELOPE of every flavor it has (TMA zero-padding, like
        # the f16 flavors — exact in FP8, and the descales are scalars, so
        # the envelope is arch-independent): head dims componentwise <= a
        # flavor shape and multiples of 16 (TMA 16-byte global-stride rule
        # at 1 byte/elem). THD stays native-shape (the packed THD compile
        # key carries no head-dim entries — engines.thd_d_shapes) and MXFP8
        # stays exact (SF plumbing not audited for zero-padding).
        fp8_shapes = _sm100_fp8_shapes(self._pertensor, self._device_cc)
        _fp8_envelope_ok = (
            self._pertensor and not self.thd and any(int(d_qk) <= sq and int(d_v) <= sv for sq, sv in fp8_shapes) and int(d_qk) % 16 == 0 and int(d_v) % 16 == 0
        )
        self._value_error_if(
            self._fp8 and (int(d_qk), int(d_v)) not in fp8_shapes and not _fp8_envelope_ok,
            f"{'FP8' if self._pertensor else 'MXFP8'} (E4M3/E5M2 inputs) requires a native shape in {sorted(fp8_shapes)}"
            + (" — or, dense only, its envelope (head dims <= a flavor shape, multiples of 16)" if self._pertensor else " (no envelope padding)")
            + f"; got (D_QK={d_qk}, D_V={d_v})",
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
            self.sched_policy is not None and self.sched_policy not in (SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2),
            f"SM100 DSL SDPA sched_policy must be NATURAL/LPT/LPT_L2 (or None to derive); got {self.sched_policy}",
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
        # softmax_precision values are cudnn.data_type (the knob vocabulary
        # fixed by #692); imported locally — this file otherwise speaks torch
        # dtypes and frost constants only.
        from cudnn import data_type as _cudnn_dtype

        self._value_error_if(
            self.softmax_precision is not None and not (self._fp8 and self._pertensor),
            "softmax_precision is served on the per-tensor FP8 path only (other families run the f32 pipeline)",
        )
        self._value_error_if(
            self.softmax_precision is not None and self.softmax_precision not in (_cudnn_dtype.FLOAT, _cudnn_dtype.HALF),
            f"softmax_precision must be cudnn.data_type.FLOAT or HALF; got {self.softmax_precision}",
        )
        # The f16x2 exponent arm is numerics-changing and lives in the SM107
        # sibling kernel only — honored exactly or declined (mirrors the
        # split engine rows: only sdpa_fwd_prefill_sm107_d128_fp8 declares
        # HALF in its softmax_precisions domain).
        self._value_error_if(
            self.softmax_precision == _cudnn_dtype.HALF and (self._device_cc != (10, 7) or self.flavor != (128, 128)),
            "softmax_precision=HALF is served for per-tensor FP8 d128 on cc10.7 only (FLOAT is the default everywhere)",
        )
        if self.split_kv > 1:
            # Split-KV: partials weighted by the per-split LSE, recombined by
            # split_combine_sm100 (which also owns the FP8 amax of the
            # recombined O). Structural limits mirror mismatch()'s
            # facts x knobs gate so the standalone API declines identically.
            self._not_implemented_error_if(
                self._fp8 and self.dtype_o not in (torch.float16, torch.bfloat16),
                "split_kv > 1 on a quantized graph requires a bf16/fp16 O (the combine reduces half-precision partials)",
            )
            self._not_implemented_error_if(self.thd, "split_kv > 1 is dense-only (THD packs its own flat grid)")
            self._value_error_if(self.has_sink, "split_kv > 1 with an attention sink is not supported")
            self._value_error_if(
                self.seq_kv_lens_present or self.seq_q_lens_present,
                "split_kv > 1 serves unpadded dense graphs only",
            )

        swa_left = self.window_size_left
        self._value_error_if(
            swa_left is not None and swa_left < 0,
            f"window_size_left must be >= 0; got {swa_left}",
        )
        band_right = self.window_size_right
        self._value_error_if(
            band_right is not None and band_right < 0,
            f"window_size_right must be >= 0; got {band_right}",
        )
        self._value_error_if(
            band_right is not None and not self.is_causal,
            "SM100 DSL SDPA: window_size_right widens the causal diagonal and requires is_causal=True",
        )
        self._value_error_if(
            self.causal_bottom_right and not self.is_causal,
            "SM100 DSL SDPA: causal_bottom_right requires is_causal=True",
        )
        if self.thd:
            self.seq_kv_lens_present = True
        self._not_implemented_error_if(
            (self.cu_seq_q_lens or self.cu_seq_kv_lens) and not self.thd,
            "cu_seq_len_* is THD-only (the dense kernels have no CU read mode yet)",
        )
        # Of the FP8/MXFP8 flavors only d128/d128 carries the write_thd_meta
        # THD leg; the d192/d128 siblings are dense-only. The engine specs
        # already route this (their d192 rows declare thd=False); the gate
        # covers direct construction.
        self._not_implemented_error_if(
            self.thd and self._fp8 and (int(d_qk), int(d_v)) != (128, 128),
            f"THD/varlen on the FP8/MXFP8 path requires D_QK=D_V=128 (the d192/d128 " f"kernels are dense-only); got (D_QK={d_qk}, D_V={d_v})",
        )
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
        # KV-tail correctness: the kernel zero-fills the last KV tile via TMA
        # OOB but only *masks* those columns on the padded / causal paths. A
        # ragged S_kv is safe when a padding mask carries the real lengths, or
        # when the causal diagonal provably covers the tail (kv >= S_kv implies
        # kv > q for every query row). Otherwise the tail columns leak into
        # the softmax and the output is silently wrong.
        if int(s_kv) % _SM100_TILE_N != 0:
            # A right-widened band pushes the last unmasked column to
            # (S_q - 1) + R (top-left) or (S_kv - 1) + R (bottom-right), so the
            # KV tail is only provably masked when it stays below S_kv.
            _br = int(self.window_size_right or 0)
            causal_covers_tail = self.is_causal and ((self.causal_bottom_right and _br == 0) or (not self.causal_bottom_right and int(s_qo) + _br <= int(s_kv)))
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
        # None = the standalone-wrapper tier stated no preference: derive the
        # causal-balancing policy here. The graph path never hits this branch —
        # the heuristic emits an explicit policy (the same primary this
        # derivation picks) and it is honored verbatim, NATURAL included.
        sched_policy = self.sched_policy
        if sched_policy is None:
            sched_policy = SCHED_NATURAL
            if self.window_right is not None:
                # Causal: balance the triangular load; pick the LPT variant by working set.
                _, _, s_kv_sched, _ = self.k_desc.shape
                _, _, _, d_qk_sched = self.q_desc.shape
                _, _, _, d_v_sched = self.v_desc.shape
                sched_policy = _causal_sched_policy(
                    s_kv=s_kv_sched,
                    d_qk=d_qk_sched,
                    d_v=d_v_sched,
                    elem_bytes=1 if self._fp8 else 2,
                )
        _pack_g = (self.h_q // self.h_kv) if self.pack_gqa else 1
        lpt_head_group = 1
        if self._fp8 and self.flavor == (192, 128) and not self.thd and (self.batch_size * self.h_q // _pack_g) % 8 == 0:
            lpt_head_group = 8
        lpt_q_tiles = 0
        if self._fp8 and self.flavor == (192, 128) and not self.thd:
            lpt_q_tiles = (self.s_q_max * _pack_g + 511) // 512
        template_window_right = self.window_right
        if (
            self._fp8
            and self._pertensor
            and self.flavor == (192, 128)
            and self.window_left is None
            and self.window_right is None
            and not self.seq_kv_lens_present
        ):
            # CUTLASS DSL 4.7 does not finish lowering the large-shape FP8
            # MASK_NONE x32 path. A right bound of S_kv removes no valid K but
            # selects the equivalent masked-interior lowering.
            template_window_right = self.s_k_max
        from cudnn import data_type as _cudnn_dtype

        params = Sm100TemplateParams(
            dtype_qkv=_SM100_DTYPE_QKV_CODE[self.dtype],
            dtype_o=_SM100_DTYPE_QKV_CODE[self.dtype_o],
            window_left=self.window_left,
            window_right=template_window_right,
            bottom_right=self.causal_bottom_right,
            has_sink=self.has_sink,
            seq_kv_lens_present=self.seq_kv_lens_present,
            seq_q_lens_present=self.seq_q_lens_present,
            sched_policy=sched_policy,
            lpt_head_group=lpt_head_group,
            lpt_q_tiles=lpt_q_tiles,
            thd_varlen=self.thd,
            pack_gqa=self.pack_gqa,
            qh_per_kh=int(self.q_desc.shape[1]) // int(self.k_desc.shape[1]),
            split_kv=self.split_kv,
            fused_ldtm_stat=fused_ldtm_stat,
            softmax_f16=self.softmax_precision == _cudnn_dtype.HALF,
        )
        self._k_mod = _load_sm100_kernel_module(self.flavor, params, fp8=self._fp8, pertensor=self._pertensor, rubin=(self._device_cc == (10, 7)))
        if self.thd:
            # The THD compile key is PLAN-TIME-ONLY (the packed token totals
            # compile as dynamic extents — issue #552), so compile HERE like
            # every dense specialization; execute()'s lru-cached call re-binds
            # this artifact. (The all-KV-zero clamp swaps the K/V strides and
            # mints its own entry on first hit.) The FP8/MXFP8 flavors serve
            # the packed contract — their key carries no stride/head-dim
            # entries (see _thd_compile_kwargs).
            self._compiled_kernel = self._k_mod.compile(**self._thd_compile_kwargs())
        elif self._fp8:
            # has_lse=False (no Stats output) compiles the LSE store out — no
            # dummy buffer at any level (the amax_o atomicMax write is
            # independent). A split REQUIRES the in-kernel LSE: the per-split
            # LSE is the combine weight (the kernel skips its own amax write
            # under a split; the combine reports the amax of the RECOMBINED O
            # instead).
            fp8_kwargs = dict(
                b=self.batch_size,
                qh=self.h_q,
                kh=self.h_kv,
                sq=self.s_q_max,
                skv=self.s_k_max,
                has_lse=(self.lse_desc is not None) or self.split_kv > 1,
                lse_stride=None if self.split_kv > 1 else self._lse_stride,
            )
            if self._pertensor:
                # ENVELOPE (per-tensor only): hand the kernel the ACTUAL head
                # dims so its TMA descriptors carry the real extents (loads
                # past them zero-fill — exact in FP8; O stores past d_v clip).
                # Both fp8 flavor kernels take these; MXFP8 is exact-native
                # (gated in check_support) and takes no head-dim parameters.
                fp8_kwargs.update(d_qk=self.head_dim_qk, d_v=self.head_dim_v)
            self._compiled_kernel = self._k_mod.compile(**fp8_kwargs)
        else:
            # ENVELOPE: hand the f16/bf16 kernel the ACTUAL head dims so its
            # TMA descriptors carry the real extents (loads past them
            # zero-fill, O stores past d_v clip); the tile box stays the
            # flavor's compile-time D. has_lse=False (no Stats output)
            # compiles the LSE store out — no dummy buffer at any level.
            # Split-KV REQUIRES the in-kernel LSE regardless of a Stats
            # output: the per-split LSE is the combine weight.
            self._compiled_kernel = self._k_mod.compile(
                b=self.batch_size,
                qh=self.h_q,
                kh=self.h_kv,
                sq=self.s_q_max,
                skv=self.s_k_max,
                d_qk=self.head_dim_qk,
                d_v=self.head_dim_v,
                has_lse=(self.lse_desc is not None) or self.split_kv > 1,
                lse_stride=None if self.split_kv > 1 else self._lse_stride,
            )
        self._combine_kernel = None
        if self.split_kv > 1:
            # The recombine pass compiles at PLAN time like everything else;
            # execute() only rebinds the partial slabs it carves. On the FP8
            # families the combine also owns the amax of the recombined O
            # (a max over per-split partials would over-report — each split's
            # O is normalized by its own running sum).
            from cudnn.sdpa.fwd.kernels import split_combine_sm100 as _split_combine

            self._combine_kernel = _split_combine.compile(
                b=self.batch_size,
                h=self.h_q,
                sq=self.s_q_max,
                d_v=self.head_dim_v,
                splits=self.split_kv,
                dtype_o=self._combine_dtype_tag(),
                has_lse=self.lse_desc is not None,
                has_amax=self._fp8,
                lse_stride=self._lse_stride,
            )
        self._logger.debug("compile completed")

    def scratch_workspace_bytes(self) -> int:
        """Per-execute scratch ``execute()`` carves from its ``workspace``.

        Fixed by the compiled geometry (call after ``check_support()``); 0 when
        the path allocates nothing per execute. This is the api-level share of
        a FROST executor's ``workspace_bytes`` (the engine lowering adds its
        own chunks — synthesized seq_len_kv — on top; see
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
            # copies either: the metadata is built DEVICE-side by the setup
            # kernel (issue #552). o_desc: 16 int64 per
            # sequence + 16 spare, the per-sequence O TMA descriptors the
            # builder kernel fills.
            # o_desc: 16 int64 per sequence + the dead-unit pad slot; the
            # FP8/MXFP8 flavors carry two more slots for the packed-total-
            # clamped K/V runtime descriptors (see the kernels' THD closures).
            o_desc_slots = b + (3 if self._fp8 else 1)
            return ws_align((3 * b + 2) * 4) + ws_align(o_desc_slots * 16 * 8) + (0 if self.has_sink else ws_align(qh * 4))
        if self._fp8 and self.split_kv == 1:
            return 0  # dense FP8/MXFP8: no per-execute scratch (dummies are cached one-time)
        if self.split_kv > 1:
            # Split-major partial slabs the main kernel writes and the combine
            # pass reduces: O_s [splits*B, S_q, H, d_v] in the O dtype (half —
            # the split path requires a bf16/fp16 O even on the FP8 families)
            # and lse_s [splits*B, H, S_q] fp32. Carved from the caller's
            # workspace — zero per-execute allocations (Hard Rule 1).
            o_bytes = self.split_kv * b * self.s_q_max * qh * self.head_dim_v * self._o_itemsize()
            lse_bytes = self.split_kv * b * qh * self.s_q_max * 4
            return ws_align(o_bytes) + ws_align(lse_bytes)
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
        # Strict presence contract, both directions: every SM100 kernel
        # (f16/bf16, FP8, MXFP8) is compiled with has_lse keyed on sample_lse
        # (no Stats output -> the LSE store is compiled out and there is no
        # LSE slot to bind), and a THD lse_tensor is bound in its DECLARED
        # packed layout (recorded at check_support) — so an lse_tensor without
        # a sample_lse cannot be honored and is rejected rather than silently
        # dropped.
        self._value_error_if(
            self.lse_desc is None and lse_tensor is not None,
            "this specialization was compiled without an LSE output; construct the API with sample_lse",
        )
        if self.thd:
            pass  # bound in _execute_thd (declared packed layout)
        elif lse_tensor is not None:
            lse_tensor = self._checked_lse_view(lse_tensor)

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
                amax_o,
                current_stream,
                workspace=workspace,
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
                workspace=workspace,
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

        o_arg = O_scratch if o_needs_copy_back else O_view
        lse_arg = lse_tensor
        if self.split_kv > 1:
            # Redirect the mainloop into split-major partial slabs, then reduce
            # into the caller's O/LSE with the plan-time-compiled combine pass.
            # Slabs are carved from the caller's workspace (Hard Rule 1);
            # standalone use (no workspace) torch-allocates, like the THD path.
            # No zero-fill: the split kernel writes EVERY (split, batch) slot,
            # emitting O := 0 / lse := -inf for empty split ranges itself.
            s, b, h, sq, dv = self.split_kv, self.batch_size, self.h_q, self.s_q_max, self.head_dim_v
            o_partial, lse_partial = self._split_partials(workspace, o_arg, device, current_stream)
            self._compiled_kernel(
                Q,
                K,
                V,
                o_partial,
                lse_partial,
                sinks_t,
                seq_kv_t,
                o_desc_dummy,
                (b, h, self.h_kv, sq, self.s_k_max, 0),
                cutlass.Float32(scale_softmax_log2),
                cutlass.Int32(0),
                seq_q_t,
                stream=current_stream,
            )
            self._combine_kernel(
                o_partial,
                lse_partial,
                o_arg,
                lse_arg,
                None,
                (b, h, sq, dv),
                cutlass.Int32(s),
                stream=current_stream,
            )
        else:
            self._compiled_kernel(
                Q,
                K,
                V,
                o_arg,
                lse_arg,
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

    def _thd_compile_kwargs(self) -> dict:
        """The THD compile key — PLAN-TIME-ONLY by contract (issue #552).

        The packed token totals are runtime values and compile as DYNAMIC
        extents; everything here (logical batch, heads, head dims, the Stats
        specialization, the declared strides with the batch stride zeroed —
        ``_thd_view``'s batch stride is ``t * token_stride``, a runtime value
        that never steps at batch extent 1) is known when the graph is built,
        so ``compile()`` compiles eagerly and ``_execute_thd``'s lru-cached
        call re-binds the same artifact for every packed total."""

        def _key(desc):
            (ts, hs, es), _ = self._thd_declared(desc)
            return (0, ts, hs, es)

        has_lse = self.lse_desc is not None
        kwargs = dict(
            b=self.batch_size,
            qh=self.h_q,
            kh=self.h_kv,
            # The Stats layout is a per-shape specialization (like d_qk/d_v):
            # has_lse=False compiles the store out; token-major binds the
            # packed rank-2 (T, H) view; head-major carries the caller-declared
            # head-row stride (0 -> compact t_q).
            has_lse=has_lse,
            lse_head_major=has_lse and self.thd_stats_head_major,
            lse_head_stride=(self.thd_stats_head_stride if (has_lse and self.thd_stats_head_major) else 0),
        )
        if self._fp8:
            # The FP8/MXFP8 cells serve only the packed contract at exact
            # d128 (check_support) — no stride/head-dim keys.
            return kwargs
        kwargs.update(
            d_qk=self.head_dim_qk,
            d_v=self.head_dim_v,
            q_stride=_key(self.q_desc),
            k_stride=_key(self.k_desc),
            v_stride=_key(self.v_desc),
            o_stride=_key(self.o_desc),
        )
        return kwargs

    def _thd_unit_envelope(self) -> int:
        """PLAN-TIME upper bound on live THD units:
        ``B * ceil(S_q_declared / CGA_TILE_M) * QH``.

        Every sequence's length is bounded by the declared S_q (the padding
        contract), so this covers ``Σ_b ceil(s_b / tile) * QH``. Units past
        the live total are DEAD by kernel contract (the decode's
        ``batch == n_batch`` sentinel): no loads, no O/LSE writes, one
        empty-mainloop barrier dance each. The grid is host-known at PLAN
        time — execute reads nothing from the lengths — which removes the
        last THD D2H sync and unblocks CUDA-graph capture (issue #552). The
        dead-tile tax mirrors the C++ backend's THD grid strategy; callers
        declaring S_q far above their live totals pay it. The over-launch
        test pads this to pin the dead-unit contract."""
        cga_tile_m = int(self._k_mod.CGA_TILE_M)
        s_q_decl = int(self.q_desc.shape[2])
        return self.batch_size * ((s_q_decl + cga_tile_m - 1) // cga_tile_m) * self.h_q

    def _thd_pack(self, q_buf, k_buf, v_buf, o_buf, sinks, seq_kv_lens, seq_q_lens, workspace, label, current_stream=None, lse_tokens_cap=None):
        """Shared SM100 THD (ragged) packing — issue #552, zero host reads.

        Builds the [seq_kv | cu_q | cu_k] metadata scratch (WRITTEN device-side
        by the kernels' setup launch from the caller's length tensors, returned
        as ``q_lens_dev``/``kv_lens_dev`` + the ``lens_form`` bitmask), the
        per-batch O-descriptor scratch, the plan-time envelope unit count, the
        sinks binding, and the declared-stride ``(1, T, H, D)`` views with
        CAPACITY token extents — Q/O (and a token-major LSE, via
        ``lse_tokens_cap``) share one dynamic token symbol, K/V the other;
        writes/loads never step past the real per-sequence lengths the kernel
        reads from the device metadata, so the over-claim only widens the TMA
        descriptors' bound. The FP8/MXFP8 packed contract declares compact
        strides, so the same views ARE the packed ones. With a ``workspace``
        every scratch chunk is carved from it (zero per-execute allocations);
        torch allocations run on the LAUNCH stream. The prefix-sum invariants
        are caller contract (a validation that needs a device read is not a
        validation — AGENTS.md Rule 3).

        Returns ``None`` when no Q token is addressable (zero capacity —
        all-zero LENGTHS over live storage launch normally: every unit
        decodes dead and neither O nor LSE is touched)."""
        dev = q_buf.device
        b, qh, kh = self.batch_size, self.h_q, self.h_kv
        d_qk, d_v = self.head_dim_qk, self.head_dim_v
        carver = WorkspaceCarver(workspace, self.scratch_workspace_bytes(), label) if workspace is not None else None
        with _torch_stream_context(current_stream, dev):
            meta = carver.take(3 * b + 2, torch.int32) if carver is not None else torch.empty(3 * b + 2, dtype=torch.int32, device=dev)
        q_lens_dev = self._checked_cu_seq_lens(seq_q_lens, "cu_seq_len_q") if self.cu_seq_q_lens else self._checked_seq_lens(seq_q_lens, "seq_q_lens")
        kv_lens_dev = self._checked_cu_seq_lens(seq_kv_lens, "cu_seq_len_kv") if self.cu_seq_kv_lens else self._checked_seq_lens(seq_kv_lens, "seq_kv_lens")
        lens_form = (1 if self.cu_seq_q_lens else 0) | (2 if self.cu_seq_kv_lens else 0)

        (q_ts, _, _), _ = self._thd_declared(self.q_desc)
        (o_ts, _, _), _ = self._thd_declared(self.o_desc)
        t_q = min(q_buf.numel() // q_ts, o_buf.numel() // o_ts)
        if lse_tokens_cap is not None:
            t_q = min(t_q, lse_tokens_cap)
        if t_q == 0:
            return None

        # Per-sequence O TMA descriptors. No zero-init: the kernel's builder
        # pass copies every qword of each sequence's slot from the base
        # descriptor (then patches address/extent) before the fence and
        # before any consumer read, so stale bytes never survive — a fill
        # here is a wasted kernel launch on the execute hot path (Rule 1).
        with _torch_stream_context(current_stream, dev):
            # +2 slots on the FP8/MXFP8 flavors: the packed-total-clamped K/V
            # runtime descriptors the setup kernel writes after the pad slot.
            o_desc_slots = b + (3 if self._fp8 else 1)
            o_desc = carver.take(o_desc_slots * 16, torch.int64) if carver is not None else torch.empty(o_desc_slots * 16, dtype=torch.int64, device=dev)
        # The PLAN-TIME envelope grid — dead units exit by kernel contract.
        units = self._thd_unit_envelope()

        Q = self._thd_view(q_buf, self.q_desc, t_q)
        O = self._thd_view(o_buf, self.o_desc, t_q)
        (k_ts, _, _), _ = self._thd_declared(self.k_desc)
        (v_ts, _, _), _ = self._thd_declared(self.v_desc)
        t_kv = min(k_buf.numel() // k_ts, v_buf.numel() // v_ts)
        if t_kv == 0:
            # No KV storage at all:
            # every query row is dead — served by the KERNEL's own dead-row
            # path (total_sum <= 0 -> O := 0 and LSE := -inf, or the sink
            # alone — its column keeps the softmax denominator alive),
            # exactly like a live launch's zero-KV sequences — no
            # adapter-side fills on the execute hot path (AGENTS.md Rule 1).
            # A zero-token packed K/V view cannot back a CuTe layout / TMA
            # descriptor, so clamp the packed KV extent to ONE
            # never-dereferenced token (every tile sees kv_left ==
            # kv_right == 0, so no K/V load is ever issued). Q backs K
            # (same element type, and kh*d_qk <= t_q*qh*d_qk); V must carry
            # the INPUT element type, which no output buffer guarantees (O's
            # dtype is independent), and Q's storage is not always large
            # enough for kh*d_v — so V binds a cached zero stub (allocated
            # once, off the execute hot path).
            t_kv = 1
            K = q_buf.as_strided((1, 1, kh, d_qk), (kh * d_qk, kh * d_qk, d_qk, 1), q_buf.storage_offset())
            with _torch_stream_context(current_stream, dev):
                V = self._dummy(f"thd_v_stub_{d_v}", dev, lambda: torch.zeros(kh * d_v, dtype=q_buf.dtype, device=dev)).as_strided(
                    (1, 1, kh, d_v), (kh * d_v, kh * d_v, d_v, 1), 0
                )
        else:
            K = self._thd_view(k_buf, self.k_desc, t_kv)
            V = self._thd_view(v_buf, self.v_desc, t_kv)
        if sinks is not None:
            sinks_t = self._checked_sinks_1d(sinks)
        else:
            # Dummy sinks bound on the launch stream (ordering vs the kernel).
            with _torch_stream_context(current_stream, dev):
                if carver is not None:
                    sinks_t = carver.take(qh, torch.float32)
                    sinks_t.zero_()
                else:
                    sinks_t = torch.zeros(qh, dtype=torch.float32, device=dev)
        return SimpleNamespace(
            meta=meta,
            o_desc=o_desc,
            units=units,
            t_q=t_q,
            t_kv=t_kv,
            Q=Q,
            K=K,
            V=V,
            O=O,
            sinks_t=sinks_t,
            q_lens_dev=q_lens_dev,
            kv_lens_dev=kv_lens_dev,
            lens_form=lens_form,
        )

    def _thd_lse_tokens_cap(self, lse_tensor):
        """The LSE buffer's token capacity when it shares the Q/O dynamic token
        symbol — token-major (the default) and COMPACT head-major (declared
        head stride 0, i.e. the token extent itself) both do, so their
        capacity joins the packed-Q floor; head-major with a declared
        head_stride carries its own extent (covering the packed total is
        caller contract — t_q is a device value, Rule 3), so it stays out."""
        if lse_tensor is None or (self.thd_stats_head_major and self.thd_stats_head_stride):
            return None
        return lse_tensor.numel() // self.h_q

    def _thd_lse_view(self, lse_tensor, t_q):
        """The caller's ragged Stats buffer in its declared layout — token-major
        packed rank-2 (T, H) (the default; cuDNN's TH1 ragged Stats recipe) or
        head-major rank-3 (1, QH, head_stride) with head_stride covering the
        packed total (caller contract — t_q is a device value, Rule 3; 0 =
        compact = the token extent itself)."""
        if lse_tensor is None:
            return None
        qh = self.h_q
        if self.thd_stats_head_major:
            head_stride = self.thd_stats_head_stride
            if head_stride == 0:
                head_stride = t_q
            return lse_tensor.as_strided((1, qh, head_stride), (qh * head_stride, head_stride, 1), lse_tensor.storage_offset())
        return lse_tensor.as_strided((t_q, qh), (qh, 1), lse_tensor.storage_offset())

    def _execute_thd(self, q_buf, k_buf, v_buf, o_buf, scale_softmax_log2, sinks, seq_len_kv, seq_q_lens, lse_tensor=None, workspace=None, current_stream=None):
        """THD / varlen execute (f16 kernels): shared packing + launch.

        ``lse_tensor``, when given, is the caller's ragged Stats buffer,
        written by the kernel directly in its declared layout (token-major
        packed ``(T, H)`` or head-major ``(H, head_stride)``); when ``None``
        the kernel compiles the LSE store out (has_lse=False) and no scratch
        exists. Host round-trips (issue #552): NONE — the lengths never reach
        the host (the setup kernel builds the metadata buffer device-side),
        every ragged view binds its buffer's capacity, and the launch grid is
        the plan-time envelope (dead units exit by kernel contract) — the
        execute is fully async and CUDA-graph capturable. No compile is keyed
        on runtime data: the kernels compile with DYNAMIC token extents, so a
        new packed total re-binds the same artifact."""
        import cutlass

        lse_cap = self._thd_lse_tokens_cap(lse_tensor)
        pack = self._thd_pack(
            q_buf, k_buf, v_buf, o_buf, sinks, seq_len_kv, seq_q_lens, workspace, "SdpaFwdDslSm100 (THD)", current_stream=current_stream, lse_tokens_cap=lse_cap
        )
        if pack is None:
            self._logger.debug("execute (THD): no addressable Q token, nothing to do")
            return
        LSE = self._thd_lse_view(lse_tensor, pack.t_q)

        # PLAN-TIME-ONLY compile key (issue #552: keying on the packed totals
        # degenerated into a per-step recompile under continuous batching):
        # this lru-cached call re-binds the artifact compile() already built.
        # The K/V strides are taken from the BOUND views because the
        # all-KV-zero clamp swaps in packed batch-1 views (that rare shape
        # mints its own cache entry); the batch stride is zeroed out of the
        # key (a runtime value the kernel rebuilds symbolically).
        kwargs = self._thd_compile_kwargs()
        kwargs.update(k_stride=(0, *pack.K.stride()[1:]), v_stride=(0, *pack.V.stride()[1:]))
        fn = self._k_mod.compile(**kwargs)
        # The caller's length tensors ride to the setup kernel, which builds
        # the metadata buffer device-side; the form bitmask is a runtime
        # value (no compile key grows). problem_size sq/skv slots are 0 by
        # the THD contract (_host reads the dynamic extents).
        fn(
            pack.Q,
            pack.K,
            pack.V,
            pack.O,
            LSE,
            pack.sinks_t,
            pack.meta,
            pack.o_desc,
            (self.batch_size, self.h_q, self.h_kv, 0, 0, 0),
            cutlass.Float32(scale_softmax_log2),
            cutlass.Int32(pack.units),
            None,
            pack.q_lens_dev,
            pack.kv_lens_dev,
            cutlass.Int32(pack.lens_form),
            stream=current_stream,
        )
        self._logger.debug("execute (THD) completed")

    @staticmethod
    def _ceil_div(x: int, a: int) -> int:
        return (x + a - 1) // a

    def _reshape_sf(self, sf: torch.Tensor, h: int, n_tiles: int, sf_smem_size: int) -> torch.Tensor:
        """cuDNN F8_128x4 scale-factor tensor (FP8_E8M0) → the kernel's per-tile
        int8 view ``[B, H, n_tiles, sf_smem_size]``.

        cuDNN packs the 128×4 SF atom contiguously (``F8_128x4`` reordering); a Q/K
        tile is 128 rows × d/32 d-blocks and a V tile is 128 rows × 4 s-blocks, so
        each tile is exactly ``sf_smem_size`` E8M0 bytes and this is a pure reshape.

        A reordered tensor is an opaque byte layout: callers legally bind it under
        any shape with the right byte count (the graph declares logical
        ``[B, H, s_padded, d/32]`` dims; TE-style producers hand over flat
        ``[B·H·s_padded, 4]`` swizzle output).  So B comes from the graph facts,
        never from ``sf.shape[0]``, and only the total size is validated.
        """
        b = self.batch_size
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

    def _reshape_sf_packed(self, sf: torch.Tensor, h: int, sf_smem_size: int, name: str, current_stream=None) -> torch.Tensor:
        """THD packed SF buffer → the kernel's ``[1, H, T_sf, sf_smem_size]`` int8 view.

        THD SF buffers hold the PACKED per-sequence-TILE-padded layout: per
        head, every sequence's 128-row F8_128x4 SF tiles concatenated in
        cu_seqlens order (tile index = cu_sf_base[b] + local tile — the same
        base the kernel derives device-side from the metadata). The packed
        tile extent ``T_sf`` is a runtime value that must come WITHOUT a
        device read (Rule 3), so it derives from the buffer's byte size —
        the buffer is the packed layout exactly (its head stride is
        ``T_sf * sf_smem_size``, so a larger allocation could not be
        addressed anyway). A zero-sized buffer (zero-capacity KV storage —
        the one-token K/V clamp) binds a one-tile stub: the KV range of
        every tile is empty there, so no SF byte is ever loaded."""
        flat = sf.contiguous()
        if flat.dtype != torch.int8:
            flat = flat.view(torch.int8)
        flat = flat.reshape(-1)
        row = h * sf_smem_size
        if flat.numel() == 0:
            with _torch_stream_context(current_stream, sf.device):
                return self._dummy(f"thd_sf_stub_{name}_{sf_smem_size}", sf.device, lambda: torch.zeros(row, dtype=torch.int8, device=sf.device)).reshape(
                    1, h, 1, sf_smem_size
                )
        if flat.numel() % row != 0:
            raise ValueError(
                f"MXFP8 THD SF buffer {name}: {flat.numel()} bytes is not a whole number of packed "
                f"[H={h} x SF_SMEM={sf_smem_size}] tile rows — THD SF buffers must hold exactly the "
                f"packed per-sequence-TILE-padded layout (Σ_b ceil(S_b/128) tiles per head)"
            )
        return flat.reshape(1, h, flat.numel() // row, sf_smem_size)

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
        workspace=None,
    ):
        """MXFP8 execute: FP8 Q/K/V + per-32-block E8M0 SF → half/FP8 O.

        SF tensors come from cuDNN in F8_128x4 layout and are reshaped into the
        kernel's per-tile view; ``Amax_O`` (if requested) is produced in-kernel
        (atomicMax over the pre-cast fp32 output rows). THD/varlen rides the
        shared packed lowering (``_thd_pack``); there the SF buffers hold the
        PACKED per-sequence-TILE-padded layout ([1, H, Σ_b ceil(S_b/128),
        SF_SMEM] tile sequences in cu_seqlens order — the same TILE base the
        kernel derives device-side from the metadata) and their packed tile
        extent is derived from the buffer size (``_reshape_sf_packed``).
        """
        import cutlass

        if sf_q is None or sf_k is None or sf_v is None:
            raise ValueError("Frost MXFP8 execute requires sf_q/sf_k/sf_v (block-scale descale tensors)")

        km = self._k_mod
        b, h_q, h_kv = self.batch_size, self.h_q, self.h_kv
        sq, skv = self.s_q_max, self.s_k_max
        device = q_tensor.device

        if self.thd:
            # amax_o reset first (launch-stream ordered): the degenerate
            # early-returns below leave the correct 0 (no valid row).
            amax_o_buf = self._amax_slot(amax_o, "amax_o", device)
            with _torch_stream_context(current_stream, device):
                amax_o_buf.zero_()
            lse_cap = self._thd_lse_tokens_cap(lse_tensor)
            pack = self._thd_pack(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                sinks,
                seq_kv_lens,
                seq_q_lens,
                workspace,
                "SdpaFwdDslSm100 (MXFP8 THD)",
                current_stream=current_stream,
                lse_tokens_cap=lse_cap,
            )
            if pack is None:
                self._logger.debug("execute (MXFP8 THD): no addressable Q token, nothing to do")
                return
            sf_q_v = self._reshape_sf_packed(sf_q, h_q, km.SF_SMEM_SIZE_Q, "sf_q", current_stream)
            sf_k_v = self._reshape_sf_packed(sf_k, h_kv, km.SF_SMEM_SIZE_K, "sf_k", current_stream)
            sf_v_v = self._reshape_sf_packed(sf_v, h_kv, km.SF_SMEM_SIZE_V, "sf_v", current_stream)
            LSE = self._thd_lse_view(lse_tensor, pack.t_q)
            # PLAN-TIME-ONLY compile key: re-binds the artifact compile()
            # already built (packed totals + SF tile extents are dynamic).
            fn = km.compile(**self._thd_compile_kwargs())
            fn(
                pack.Q,
                pack.K,
                pack.V,
                pack.O,
                sf_q_v,
                sf_k_v,
                sf_v_v,
                LSE,
                amax_o_buf,
                pack.sinks_t,
                pack.meta,
                pack.o_desc,
                (b, h_q, h_kv, 0, 0, 0),
                cutlass.Float32(scale_softmax_log2),
                cutlass.Int32(pack.units),
                pack.q_lens_dev,
                pack.kv_lens_dev,
                cutlass.Int32(pack.lens_form),
                stream=current_stream,
            )
            self._logger.debug("execute (MXFP8 THD) completed")
            return

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

        # has_lse=False (no Stats output): the store is compiled out; bind None.
        lse = lse_tensor
        sinks_t = (
            self._checked_sinks_1d(sinks) if sinks is not None else self._dummy("sinks", device, lambda: torch.zeros(h_q, dtype=torch.float32, device=device))
        )
        seq_kv_t = (
            self._checked_seq_lens(seq_kv_lens, "seq_kv_lens")
            if seq_kv_lens is not None
            else self._dummy("seq_kv", device, lambda: torch.zeros(b, dtype=torch.int32, device=device))
        )

        amax_o_buf = amax_o.reshape(-1)[:1] if amax_o is not None else self._dummy("amax_o", device, lambda: torch.zeros(1, dtype=torch.float32, device=device))
        # Must be enqueued on the SAME stream as the kernel launch below, else the
        # reset and the kernel's atomicMax are unordered (and the reset is missing
        # from a CUDA-graph capture taken on the handle's stream).
        with _torch_stream_context(current_stream, device):
            amax_o_buf.zero_()

        o_desc_dummy = self._dummy("o_desc", device, lambda: torch.zeros(1, dtype=torch.int64, device=device))
        # Split-KV: the mainloop writes split-major partials (skipping its own
        # amax — each split's O is normalized by its own running sum, so a max
        # over partials over-reports); the combine reduces them into the
        # caller's O/LSE and owns the recombined amax.
        O_dst, lse_dst = O, lse
        if self.split_kv > 1:
            O_dst, lse_dst = self._split_partials(workspace, O, device, current_stream)
        self._compiled_kernel(
            Q,
            K,
            V,
            O_dst,
            sf_q_v,
            sf_k_v,
            sf_v_v,
            lse_dst,
            amax_o_buf,
            sinks_t,
            seq_kv_t,
            o_desc_dummy,
            (b, h_q, h_kv, sq, skv, 0),
            cutlass.Float32(scale_softmax_log2),
            cutlass.Int32(0),
            stream=current_stream,
        )
        if self.split_kv > 1:
            self._combine_kernel(
                O_dst,
                lse_dst,
                O,
                lse,
                amax_o_buf,
                (b, h_q, sq, self.head_dim_v),
                cutlass.Int32(self.split_kv),
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
        amax_o,
        current_stream=None,
        workspace=None,
    ):
        """Per-tensor FP8 execute: scalar descales fold into scale_softmax_log2
        (attn·descale_q·descale_k·log2 e) and o_scale_fused (descale_v·scale_o) —
        both IN-KERNEL from device scales (Rule 3); ``Amax_O`` is produced
        in-kernel and divided by scale_o on device post-kernel. THD/varlen
        rides the shared packed lowering (``_thd_pack``); the scale folding
        and the amax protocol are identical.
        """
        import cutlass

        b, h_q, h_kv = self.batch_size, self.h_q, self.h_kv
        sq, skv = self.s_q_max, self.s_k_max
        device = q_tensor.device

        # Rule 3: the scales stay on device — the kernel loads and folds
        # descale_q*descale_k into the softmax scale and descale_v*scale_o
        # into o_scale_fused; the scalar args carry only the bases.
        # (descale_s/scale_s never reach this layer: the lowering does not
        # forward them, and P is cast with the baked P_CAST_LOG2_SCALE bias.)
        dq_t = self._scale_view(descale_q, "descale_q", device)
        dk_t = self._scale_view(descale_k, "descale_k", device)
        dv_t = self._scale_view(descale_v, "descale_v", device)
        so_t = self._scale_view(scale_o, "scale_o", device)
        scale_softmax_log2 = scale_val * math.log2(math.e)
        o_scale_fused = 1.0

        if self.thd:
            # amax_o reset first (launch-stream ordered): the degenerate
            # early-returns below leave the correct 0 (no valid row).
            amax_o_buf = self._amax_slot(amax_o, "amax_o", device)
            with _torch_stream_context(current_stream, device):
                amax_o_buf.zero_()
            lse_cap = self._thd_lse_tokens_cap(lse_tensor)
            pack = self._thd_pack(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                sinks,
                seq_kv_lens,
                seq_q_lens,
                workspace,
                "SdpaFwdDslSm100 (FP8 THD)",
                current_stream=current_stream,
                lse_tokens_cap=lse_cap,
            )
            if pack is None:
                self._logger.debug("execute (FP8 THD): no addressable Q token, nothing to do")
                return
            LSE = self._thd_lse_view(lse_tensor, pack.t_q)
            # PLAN-TIME-ONLY compile key: re-binds the artifact compile()
            # already built (the packed totals are dynamic extents).
            fn = self._k_mod.compile(**self._thd_compile_kwargs())
            fn(
                pack.Q,
                pack.K,
                pack.V,
                pack.O,
                LSE,
                pack.sinks_t,
                pack.meta,
                pack.o_desc,
                (b, h_q, h_kv, 0, 0, 0),
                cutlass.Float32(scale_softmax_log2),
                cutlass.Float32(o_scale_fused),
                cutlass.Int32(pack.units),
                dq_t,
                dk_t,
                dv_t,
                so_t,
                amax_o_buf,
                pack.q_lens_dev,
                pack.kv_lens_dev,
                cutlass.Int32(pack.lens_form),
                stream=current_stream,
            )
            with _torch_stream_context(current_stream, device):
                if amax_o is not None:
                    # Device divisor: same div_, no readback; scale_o > 0 is
                    # caller contract (None bound a cached 1.0 above).
                    amax_o_buf.div_(so_t)
            self._logger.debug("execute (FP8 per-tensor THD) completed")
            return

        Q = self._to_bshd(q_tensor)
        K = self._to_bshd(k_tensor)
        V = self._to_bshd(v_tensor)
        O_view, o_needs_copy_back, O_scratch = self._to_bshd_writable(o_tensor)
        O = O_scratch if o_needs_copy_back else O_view

        # has_lse=False (no Stats output): the store is compiled out; bind None.
        lse = lse_tensor
        sinks_t = (
            self._checked_sinks_1d(sinks) if sinks is not None else self._dummy("sinks", device, lambda: torch.zeros(h_q, dtype=torch.float32, device=device))
        )
        seq_kv_t = (
            self._checked_seq_lens(seq_kv_lens, "seq_kv_lens")
            if seq_kv_lens is not None
            else self._dummy("seq_kv", device, lambda: torch.zeros(b, dtype=torch.int32, device=device))
        )

        # amax_o: the kernel atomicMax'es into this buffer, so it MUST start
        # at 0. It accumulates max|o_scaled| (pre-cast, exact even for FP8 O);
        # dividing by scale_o below yields the pre-quant output amax.
        amax_o_buf = self._amax_slot(amax_o, "amax_o", device)
        # Same-stream ordering as MXFP8: the reset must precede the kernel's
        # atomicMax on the launch stream, not on torch's current stream.
        with _torch_stream_context(current_stream, device):
            amax_o_buf.zero_()

        o_desc_dummy = self._dummy("o_desc", device, lambda: torch.zeros(1, dtype=torch.int64, device=device))
        # Split-KV: mainloop into split-major partials (the kernel skips its
        # in-kernel amax under a split), combine into the caller's O/LSE with
        # the recombined amax.
        O_dst, lse_dst = O, lse
        if self.split_kv > 1:
            O_dst, lse_dst = self._split_partials(workspace, O, device, current_stream)
        self._compiled_kernel(
            Q,
            K,
            V,
            O_dst,
            lse_dst,
            sinks_t,
            seq_kv_t,
            o_desc_dummy,
            (b, h_q, h_kv, sq, skv, 0),
            cutlass.Float32(scale_softmax_log2),
            cutlass.Float32(o_scale_fused),
            cutlass.Int32(0),
            dq_t,
            dk_t,
            dv_t,
            so_t,
            amax_o_buf,
            stream=current_stream,
        )
        if self.split_kv > 1:
            self._combine_kernel(
                O_dst,
                lse_dst,
                O,
                lse,
                amax_o_buf,
                (b, h_q, sq, self.head_dim_v),
                cutlass.Int32(self.split_kv),
                stream=current_stream,
            )
        if o_needs_copy_back:
            O_view.copy_(O)
        if amax_o is not None:
            # Device divisor: the same div_ as before, minus the readback.
            # scale_o > 0 is caller contract (backend parity); None bound a
            # cached 1.0 above.
            amax_o_buf.div_(so_t)
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
    window_size_right: Optional[int]
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
    window_size_right: Optional[int] = None,
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
        window_size_right=window_size_right,
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
    FP16/BF16 MHA, GQA, and MQA; head dimensions in multiples of 8 through
    256 (ENVELOPE: the kernel compiles at tiles rounded up to 16 and TMA
    zero-fills the pad columns); top-left or bottom-right causal masks; left sliding
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
        self._lse_stride: Optional[tuple[int, int, int]] = None
        self._k_mod = None

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        if self.thd:
            self._value_error_if(self.seq_q_lens_present, "seq_q_lens_present is dense-only (THD carries per-sequence Q lengths via cu_seqlens)")
            self.seq_kv_lens_present = True
        self._not_implemented_error_if(
            (self.cu_seq_q_lens or self.cu_seq_kv_lens) and not self.thd,
            "cu_seq_len_* is THD-only (the dense kernels have no CU read mode yet)",
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
        if self.thd:
            if self._pertensor:
                self._thd_check_strides_packed()
            else:
                self._thd_check_strides_native()

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
                self._value_error_if(
                    not dense_layout_ok((*self.lse_desc.shape, 1), (*self.lse_desc.stride, 1)),
                    f"LSE must use a dense-compatible B/H/S permutation or padded layout "
                    f"with non-broadcast, non-overlapping-by-span strides; got {self.lse_desc.stride}",
                )
                self._lse_stride = None if self.lse_desc.is_contiguous() else tuple(int(stride) for stride in self.lse_desc.stride)

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
            d_q % 8 != 0 or not 0 < d_q <= _SM120_HEAD_TILE_MAX,
            f"D_QK ({d_q}) must be a multiple of 8 (TMA 16-byte global-stride rule at 2 B/elem) and <= {_SM120_HEAD_TILE_MAX}",
        )
        self._value_error_if(
            d_v % 8 != 0 or not 0 < d_v <= _SM120_HEAD_TILE_MAX,
            f"D_V ({d_v}) must be a multiple of 8 (TMA 16-byte global-stride rule at 2 B/elem) and <= {_SM120_HEAD_TILE_MAX}",
        )

        self.dtype = self._check_dtype(self.q_desc, [torch.float16, torch.bfloat16, *_SM100_FP8_DTYPES], name="Q")
        self._fp8 = self.dtype in _SM100_FP8_DTYPES
        if self.pack_gqa:
            self._not_implemented_error_if(
                self.thd,
                "PackGQA is dense-only (THD/ragged runs unpacked)",
            )
            self._value_error_if(
                not pack_gqa_supported(int(h_q), int(h_kv), int(self.q_tile)),
                f"PackGQA requires h_q/h_kv to divide q_tile ({self.q_tile}); got h_q/h_kv = {int(h_q)}/{int(h_kv)}",
            )
        for desc in (self.k_desc, self.v_desc, self.o_desc):
            if self._fp8 and desc is self.o_desc:
                # SDPA_FP8's O dtype is independent of QKV: fp16/bf16 ride the
                # staging epilogue, fp8 the direct quantizing store.
                self._check_dtype(desc, [torch.float16, torch.bfloat16, *_SM100_FP8_DTYPES], name="O")
            else:
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
        if self._fp8:
            self._value_error_if(
                not self._pertensor,
                "SM120 fp8 serves the per-tensor SDPA_FP8 op only (no MXFP8 cell)",
            )
            self._value_error_if(
                d_q % 16 != 0 or d_v % 16 != 0,
                f"SM120 fp8 requires D_QK/D_V multiples of 16, got ({d_q}, {d_v})",
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
            "causal_bottom_right requires is_causal=True (a band graph arrives as is_causal with its right bound)",
        )
        self._value_error_if(
            self.window_size_left is not None and self.window_size_left < 0,
            f"window_size_left must be non-negative, got {self.window_size_left}",
        )
        self._value_error_if(
            self.window_size_right is not None and self.window_size_right < 0,
            f"window_size_right must be >= 0; got {self.window_size_right}",
        )
        self._value_error_if(
            self.window_size_right is not None and not self.is_causal,
            "window_size_right widens the causal diagonal and requires is_causal=True",
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

        # SMEM tiles are sized at the ENVELOPE-padded head tiles (rounded up
        # to the head-tile granule), not the actual dims — the kernel stages
        # full tiles and the TMA zero-fills the pad columns.
        granule = _SM120_FP8_HEAD_TILE_GRANULE if self._fp8 else _SM120_HEAD_TILE_GRANULE
        d_qp = -(-d_q // granule) * granule
        d_vp = -(-d_v // granule) * granule

        def _smem_bytes(kv_tile: int) -> int:
            # FP8 stages a byte per KV element but still writes O in half.
            return _sm120_smem_bytes(d_qp, d_vp, self.q_tile, kv_tile, self.dtype.itemsize, 2 if self._fp8 else self.dtype.itemsize)

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

        self._value_error_if(
            self.sched_policy is not None and self.sched_policy not in (SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2),
            f"SM120 DSL SDPA sched_policy must be NATURAL/LPT/LPT_L2 (or None to derive); got {self.sched_policy}",
        )
        if self.split_kv > 1:
            # The SM120 kernel's inline split chunking + the shared (arch-
            # agnostic, one block per row) split_combine pass. The config
            # backstop additionally bars a split under the LPT remaps —
            # validated at compile via make_cfg, and the heuristic's split
            # sets ride SCHED_NATURAL.
            self._not_implemented_error_if(self._fp8, "SM120 split_kv > 1 is f16/bf16-only (the fp8 kernel has no split path)")
            self._not_implemented_error_if(self.thd, "split_kv > 1 is dense-only (THD packs its own flat grid)")
            self._value_error_if(self.has_sink, "split_kv > 1 with an attention sink is not supported")
            self._value_error_if(
                self.seq_kv_lens_present or self.seq_q_lens_present,
                "split_kv > 1 serves unpadded dense graphs only",
            )
        self._value_error_if(
            self.softmax_precision is not None,
            "SM120 DSL SDPA has no softmax-precision arm yet (softmax_precision must be unset)",
        )

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

        # None = the standalone-wrapper tier stated no preference: derive the
        # causal-balancing policy here. The graph path arrives with an explicit
        # policy from the heuristic and it is honored verbatim, NATURAL included.
        sched_policy = self.sched_policy
        if sched_policy is None:
            sched_policy = SCHED_NATURAL
            if self.window_right is not None:
                # Causal: balance the triangular load; pick the LPT variant by working set.
                _, _, s_kv_sched, _ = self.k_desc.shape
                _, _, _, d_qk_sched = self.q_desc.shape
                _, _, _, d_v_sched = self.v_desc.shape
                sched_policy = _causal_sched_policy(
                    s_kv=s_kv_sched,
                    d_qk=d_qk_sched,
                    d_v=d_v_sched,
                    elem_bytes=1 if self._fp8 else 2,
                )
        params = Sm120TemplateParams(
            dtype_qkv=_SM120_DTYPE_QKV_CODE[self.dtype],
            dtype_o=_SM120_DTYPE_QKV_CODE[self.o_desc.dtype],
            sched_policy=sched_policy,
            window_left=self.window_left,
            window_right=self.window_right,
            bottom_right=self.causal_bottom_right,
            seq_q_lens_present=self.seq_q_lens_present,
            seq_kv_lens_present=self.seq_kv_lens_present,
            has_sink=self.has_sink,
            thd_varlen=self.thd,
            q_tile=self.q_tile,
            kv_tile=self.kv_tile,
            pack_gqa=self.pack_gqa,
            split_kv=self.split_kv,
        )
        self._k_mod = _load_sm120_kernel_module(params, fp8=self._fp8)
        if self.thd:
            # The THD compile key is PLAN-TIME-ONLY (the packed token totals
            # compile as dynamic extents and max_sq is a runtime launch
            # argument — issue #552), so compile HERE like every dense
            # specialization; execute()'s lru-cached call re-binds this
            # artifact. (The all-KV-zero clamp swaps the K/V strides and
            # mints its own entry on first hit.)
            self._compiled_kernel = self._k_mod.compile(**self._thd_compile_kwargs())
            self._logger.debug("compile completed (THD, dynamic token extents)")
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
            # binds no LSE buffer at all (no dummy, no allocation). A split
            # REQUIRES it: the per-split LSE is the combine weight.
            has_lse=(self.lse_desc is not None) or self.split_kv > 1,
            lse_stride=None if self.split_kv > 1 else self._lse_stride,
        )
        self._combine_kernel = None
        if self.split_kv > 1:
            # The recombine pass compiles at PLAN time; execute() only rebinds
            # the partial slabs it carves. The combine kernel is arch-agnostic
            # (one block per (q_row, head, batch), no cluster/TMEM features).
            from cudnn.sdpa.fwd.kernels import split_combine_sm100 as _split_combine

            self._combine_kernel = _split_combine.compile(
                b=self.batch_size,
                h=self.h_q,
                sq=self.s_q_max,
                d_v=self.head_dim_v,
                splits=self.split_kv,
                dtype_o=self._combine_dtype_tag(),
                has_lse=self.lse_desc is not None,
                has_amax=False,
                lse_stride=self._lse_stride,
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
        descale_q: Optional[torch.Tensor] = None,
        descale_k: Optional[torch.Tensor] = None,
        descale_v: Optional[torch.Tensor] = None,
        scale_o: Optional[torch.Tensor] = None,
        amax_o: Optional[torch.Tensor] = None,
        sf_q: Optional[torch.Tensor] = None,
        sf_k: Optional[torch.Tensor] = None,
        sf_v: Optional[torch.Tensor] = None,
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
        if self._fp8:
            self._value_error_if(
                any(t is not None for t in (sf_q, sf_k, sf_v)),
                "SM120 fp8 is per-tensor (scalar descales); block-scale SF tensors are MXFP8-only",
            )
            self._execute_fp8(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                lse_tensor,
                scale_val,
                seq_kv_lens,
                descale_q,
                descale_k,
                descale_v,
                scale_o,
                amax_o,
                sinks=sinks,
                seq_q_lens=seq_q_lens,
                workspace=workspace,
                current_stream=current_stream,
            )
            return
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
        # Split-KV: the kernel's inline chunking writes split-major partials
        # (every (split, batch) slot — empty ranges as O := 0 / lse := -inf);
        # the shared combine reduces them into the caller's O/LSE.
        o_dst, lse_dst = o, lse
        if self.split_kv > 1:
            o_dst, lse_dst = self._split_partials(workspace, o, q_tensor.device, current_stream)
        self._compiled_kernel(
            q,
            k,
            v,
            o_dst,
            lse_dst,
            sinks_t,
            seq_q_lens,
            seq_kv_lens,
            cutlass.Float32(scale_softmax_log2),
            cutlass.Int32(0),  # thd_max_sq: THD-only plan-time envelope grid extent
            None,  # thd_q_lens / thd_kv_lens / thd_lens_form: THD-only, folded out
            None,
            None,
            current_stream,
        )
        if self.split_kv > 1:
            self._combine_kernel(
                o_dst,
                lse_dst,
                o,
                lse,
                None,
                (self.batch_size, self.h_q, self.s_q_max, self.head_dim_v),
                cutlass.Int32(self.split_kv),
                stream=current_stream,
            )
        if o_needs_copy_back:
            o_view.copy_(o_scratch)

    def _execute_fp8(
        self,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        scale_val,
        seq_kv_lens,
        descale_q,
        descale_k,
        descale_v,
        scale_o,
        amax_o,
        sinks=None,
        seq_q_lens=None,
        workspace=None,
        current_stream=None,
    ):
        """Per-tensor FP8 execute: SM100 convention on the SM120 kernel.

        The scales ride as 1-element DEVICE tensors and fold IN-KERNEL
        (Rule 3 — no host readback): ``descale_q*descale_k`` into the softmax
        scale, ``descale_v*scale_o`` into ``o_scale_fused``. Scale_S/Descale_S
        never reach this layer (the lowering does not forward them); P is cast
        with the kernels' baked ``P_CAST_LOG2_SCALE`` bias instead. ``Amax_O``
        is ``max|o_scaled|/scale_o`` post-kernel (pre-cast fp32, so exact for
        every O dtype including the fp8 quantizing store).
        """
        import cutlass

        device = q_tensor.device
        # Rule 3: the scales stay on device — the kernel loads and folds
        # dq*dk into the softmax scale and dv*so into o_scale_fused; the
        # scalar args carry only the bases. None binds a cached 1.0.
        # (descale_s/scale_s never reach this layer: the lowering does not
        # forward them, and P is cast with the baked P_CAST_LOG2_SCALE bias.)
        dq_t = self._scale_view(descale_q, "descale_q", device)
        dk_t = self._scale_view(descale_k, "descale_k", device)
        dv_t = self._scale_view(descale_v, "descale_v", device)
        so_t = self._scale_view(scale_o, "scale_o", device)
        scale_softmax_log2 = scale_val * math.log2(math.e)
        o_scale_fused = 1.0

        self._value_error_if(
            self.lse_desc is not None and lse_tensor is None,
            "lse_tensor is required by this compiled specialization",
        )
        self._value_error_if(
            self.lse_desc is None and lse_tensor is not None,
            "this specialization was compiled without an LSE output; construct the API with sample_lse",
        )
        if current_stream is None:
            current_stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)

        sinks_t = self._checked_sinks_1d(sinks) if sinks is not None else None
        # THD packs the batch away, so the ragged views and the per-execute
        # compile replace the dense buffers; everything else (the folded
        # scalars, the amax protocol) is identical. THD buffers go to
        # _thd_pack RAW: callers may bind ragged storage in any rank (mhas
        # hands flat (T, H, D) buffers), the packed as_strided views read raw
        # storage directly, and the dense _to_bshd_writable normalization
        # must NOT run -- its scratch copy-back is a SEMANTIC (shape-wise)
        # copy that scrambles packed bytes whenever it misreads such a buffer
        # as a non-BSHD dense layout (h > 1; size-1 axes made h == 1 immune).
        pack = None
        q = k = v = o = None
        o_needs_copy_back = False
        seq_kv_t = seq_q_t = None
        if self.thd:
            # A token-major LSE shares the Q/O dynamic token symbol, so its
            # capacity joins their floor; head-major carries its own declared
            # head_stride (covering the packed total is caller contract — t_q
            # is a device value, Rule 3).
            lse_cap = lse_tensor.numel() // self.h_q if (lse_tensor is not None and not self.thd_stats_head_major) else None
            pack = self._thd_pack(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                seq_q_lens,
                seq_kv_lens,
                workspace,
                "SdpaFwdDslSm120 (FP8 THD)",
                current_stream=current_stream,
                lse_tokens_cap=lse_cap,
            )
            if pack is None:
                return
            # The kernel specializes on either ragged-Stats layout:
            #   token-major packed (T, H) or head-major (H, head_stride)
            lse = None
            if lse_tensor is not None:
                if self.thd_stats_head_major:
                    head_stride = self.thd_stats_head_stride
                    lse = lse_tensor.as_strided((self.h_q, head_stride), (head_stride, 1), lse_tensor.storage_offset())
                else:
                    lse = lse_tensor.as_strided((pack.t_q, self.h_q), (self.h_q, 1), lse_tensor.storage_offset())
        else:
            seq_kv_t = (
                self._checked_seq_lens(seq_kv_lens, "seq_kv_lens")
                if seq_kv_lens is not None
                else self._dummy("seq_kv_lens", device, lambda: torch.zeros(self.batch_size, dtype=torch.int32, device=device))
            )
            # Dense padded-Q trim: bind the REAL per-batch Q lengths whenever
            # the graph carries them — a zeroed dummy would trim every row.
            seq_q_t = (
                self._checked_seq_lens(seq_q_lens, "seq_q_lens")
                if seq_q_lens is not None
                else self._dummy("seq_q_lens", device, lambda: torch.zeros(self.batch_size, dtype=torch.int32, device=device))
            )
            q = self._to_bshd(q_tensor)
            k = self._to_bshd(k_tensor)
            v = self._to_bshd(v_tensor)
            o_view, o_needs_copy_back, o_scratch = self._to_bshd_writable(o_tensor)
            o = o_scratch if o_needs_copy_back else o_view
            lse = self._checked_lse_view(lse_tensor) if lse_tensor is not None else None

        # amax_o: the kernel atomicMax'es into this buffer, so it MUST start
        # at 0, reset on the LAUNCH stream (ordering vs the kernel).
        amax_o_buf = self._amax_slot(amax_o, "amax_o", device)
        with _torch_stream_context(current_stream, device):
            amax_o_buf.zero_()

        fn = self._compiled_kernel
        if pack is not None:
            # PLAN-TIME-ONLY compile key (issue #552): this lru-cached call
            # re-binds the artifact compile() already built — the packed
            # totals are dynamic extents and max_sq is a launch argument.
            fn = self._k_mod.compile(**self._thd_compile_kwargs())
        fn(
            pack.Q if pack is not None else q,
            pack.K if pack is not None else k,
            pack.V if pack is not None else v,
            pack.O if pack is not None else o,
            lse,
            sinks_t,
            pack.seq_q_dummy if pack is not None else seq_q_t,
            pack.meta if pack is not None else seq_kv_t,
            amax_o_buf.view(torch.int32),
            cutlass.Float32(scale_softmax_log2),
            cutlass.Float32(o_scale_fused),
            dq_t,
            dk_t,
            dv_t,
            so_t,
            cutlass.Int32(pack.max_sq if pack is not None else 0),
            pack.q_lens_dev if pack is not None else None,
            pack.kv_lens_dev if pack is not None else None,
            cutlass.Int32(pack.lens_form) if pack is not None else None,
            current_stream,
        )
        # Both of these consume what the kernel just wrote, so they belong on
        # the launch stream for the same reason the resets above do.
        with _torch_stream_context(current_stream, device):
            if o_needs_copy_back:
                o_view.copy_(o_scratch)
            if amax_o is not None:
                # Device divisor: same div_, minus the readback; scale_o > 0
                # is caller contract (None bound a cached 1.0 above).
                amax_o_buf.div_(so_t)
        self._logger.debug("execute (SM120 FP8 per-tensor) completed")

    def _thd_compile_kwargs(self) -> dict:
        """The THD compile key — PLAN-TIME-ONLY by contract (issue #552).

        The packed token totals compile as DYNAMIC extents and ``max_sq`` is
        a runtime launch argument, so everything here is known when the graph
        is built: ``compile()`` compiles eagerly and the execute paths'
        lru-cached calls re-bind the same artifact for every packed total."""
        has_lse = self.lse_desc is not None
        kwargs = dict(
            compute_capability=self.compute_capability,
            b=self.batch_size,
            qh=self.h_q,
            kh=self.h_kv,
            d_qk=self.head_dim_qk,
            d_v=self.head_dim_v,
            has_lse=has_lse,
            lse_head_stride=self.thd_stats_head_stride,
        )
        if self._fp8:
            # The FP8 cell serves only the packed contract (no stride keys);
            # its ragged LSE store specializes on either layout.
            kwargs.update(lse_head_major=self.thd_stats_head_major)
            return kwargs

        def _key(desc):
            (ts, hs, es), _ = self._thd_declared(desc)
            return (0, ts, hs, es)

        kwargs.update(
            lse_head_major=self.thd_stats_head_major,
            q_stride=_key(self.q_desc),
            k_stride=_key(self.k_desc),
            v_stride=_key(self.v_desc),
            o_stride=_key(self.o_desc),
        )
        return kwargs

    def _thd_pack(self, q_buf, k_buf, v_buf, o_buf, seq_q_lens, seq_kv_lens, workspace, label, declared_views=False, current_stream=None, lse_tokens_cap=None):
        """Shared THD (ragged) packing: metadata buffer + ``(1, T, H, D)`` views.

        Serves the same fully-packed contract as the SM100 THD path
        (``ragged_offset == cumsum(seq_len) * H * D`` from 0, multiplier 1).
        Host round-trips (issue #552): NONE — the metadata buffer is built
        DEVICE-side by the kernels' setup launch from the caller's length
        tensors (returned as ``q_lens_dev``/``kv_lens_dev`` + the
        ``lens_form`` bitmask), every ragged view binds its buffer's
        CAPACITY (the kernels read the real per-sequence lengths from the
        device metadata; ``lse_tokens_cap`` joins the Q/O floor when the LSE
        shares their dynamic token symbol), and ``max_sq`` is the PLAN-TIME
        declared S_q envelope that sizes the per-sequence grid — tiles past
        a sequence's real length drain without loads or stores. The torch
        allocation runs on ``current_stream`` — the LAUNCH stream — so it is
        ordered against the kernels that consume it.

        Returns ``None`` when no Q token is addressable (zero capacity)."""
        b, qh, kh = self.batch_size, self.h_q, self.h_kv
        d_qk, d_v = self.head_dim_qk, self.head_dim_v
        dev = q_buf.device
        carver = WorkspaceCarver(workspace, self.scratch_workspace_bytes(), label) if workspace is not None else None

        # [seq_kv(B) | cu_q(B+1) | cu_k(B+1)] — bound as the kernel's
        # seq_kv_lens tensor; the leading B words alias the per-sequence KV
        # lengths so the kernel's existing padded-mask read works unchanged.
        # WRITTEN by the setup kernel; the prefix-sum invariants are caller
        # contract (a validation that needs a device read is not a
        # validation — AGENTS.md Rule 3; cu prefixes are normalized by the
        # setup kernel).
        with _torch_stream_context(current_stream, dev):
            meta = carver.take(3 * b + 2, torch.int32) if carver is not None else torch.empty(3 * b + 2, dtype=torch.int32, device=dev)
        q_lens_dev = self._checked_cu_seq_lens(seq_q_lens, "cu_seq_len_q") if self.cu_seq_q_lens else self._checked_seq_lens(seq_q_lens, "seq_q_lens")
        kv_lens_dev = self._checked_cu_seq_lens(seq_kv_lens, "cu_seq_len_kv") if self.cu_seq_kv_lens else self._checked_seq_lens(seq_kv_lens, "seq_kv_lens")
        lens_form = (1 if self.cu_seq_q_lens else 0) | (2 if self.cu_seq_kv_lens else 0)

        def _cap(buf, desc, heads, d):
            # Token CAPACITY under the strides the view will bind: declared
            # (f16, TMA-expressible by check_support) or packed (FP8).
            ts = self._thd_declared(desc)[0][0] if declared_views else heads * d
            return buf.numel() // ts

        # Q/O (and a token-major LSE) bind ONE dynamic token symbol; K/V the
        # other — shared floors.
        t_q = min(_cap(q_buf, self.q_desc, qh, d_qk), _cap(o_buf, self.o_desc, qh, d_v))
        if lse_tokens_cap is not None:
            t_q = min(t_q, lse_tokens_cap)
        t_kv = min(_cap(k_buf, self.k_desc, kh, d_qk), _cap(v_buf, self.v_desc, kh, d_v))

        if t_q == 0:
            return None

        def _packed(buf, tokens, heads, d):
            return buf.as_strided((1, tokens, heads, d), (tokens * heads * d, heads * d, d, 1), buf.storage_offset())

        def _view(buf, desc, tokens, heads, d):
            # declared_views: the f16 kernel addresses declared strides
            # natively (check_support rejected inexpressible ones); the FP8
            # path keeps the packed contract (check_support declined
            # anything else).
            return self._thd_view(buf, desc, tokens) if declared_views else _packed(buf, tokens, heads, d)

        if t_kv == 0:
            # No KV storage at all: every query row is dead — served by the
            # KERNEL's own dead-row path (row_sum <= 0 -> O := 0 and
            # LSE := -inf, or the sink alone — its column keeps the softmax
            # denominator alive), exactly like a live launch's zero-KV
            # sequences — no adapter-side fills on the execute hot path
            # (AGENTS.md Rule 1). A zero-token packed K/V view cannot back a
            # CuTe layout, so clamp the packed KV extent to ONE
            # never-dereferenced token (every sequence's KV tile range is
            # empty, so no K/V load is ever issued). Q backs K (same element
            # type, and kh*d_qk <= t_q*qh*d_qk); V must carry the INPUT
            # element type, which no output buffer guarantees (O's dtype is
            # independent), and Q's storage is not always large enough for
            # kh*d_v — so V binds a cached zero stub (allocated once, off the
            # execute hot path). All-zero LENGTHS over live storage launch
            # normally through the dead-row path.
            t_kv = 1
            K = q_buf.as_strided((1, 1, kh, d_qk), (kh * d_qk, kh * d_qk, d_qk, 1), q_buf.storage_offset())
            with _torch_stream_context(current_stream, dev):
                V = self._dummy(f"thd_v_stub_{d_v}", dev, lambda: torch.zeros(kh * d_v, dtype=q_buf.dtype, device=dev)).as_strided(
                    (1, 1, kh, d_v), (kh * d_v, kh * d_v, d_v, 1), 0
                )
        else:
            K = _view(k_buf, self.k_desc, t_kv, kh, d_qk)
            V = _view(v_buf, self.v_desc, t_kv, kh, d_v)

        # The cached seq_q dummy is allocated/zeroed once, on the launch
        # stream (first-use ordering vs the kernel that reads it).
        with _torch_stream_context(current_stream, dev):
            seq_q_dummy = self._dummy("seq_q_lens", dev, lambda: torch.zeros(b, dtype=torch.int32, device=dev))
        return SimpleNamespace(
            meta=meta,
            t_q=t_q,
            t_kv=t_kv,
            # PLAN-TIME envelope: sizes the per-sequence grid; tiles past a
            # sequence's real length drain without loads or stores.
            max_sq=int(self.q_desc.shape[2]),
            Q=_view(q_buf, self.q_desc, t_q, qh, d_qk),
            K=K,
            V=V,
            O=_view(o_buf, self.o_desc, t_q, qh, d_v),
            seq_q_dummy=seq_q_dummy,
            q_lens_dev=q_lens_dev,
            kv_lens_dev=kv_lens_dev,
            lens_form=lens_form,
        )

    def _execute_thd(
        self, q_buf, k_buf, v_buf, o_buf, scale_softmax_log2, sinks, seq_kv_lens, seq_q_lens, lse_tensor=None, workspace=None, current_stream=None
    ):
        """THD (ragged) execute: packed ``(1, T, H, D)`` views + cu_seqlens.

        ``lse_tensor``, when given, is the caller's ragged Stats buffer, in its
        declared layout: token-major packed ``(T, H)`` in the first ``T*H``
        elements, or head-major ``(H, head_stride)`` with tokens contiguous
        within each head row.
        """

        # Resolve the launch stream BEFORE packing: the metadata upload inside
        # _thd_pack must be ordered against the kernel launch below.
        if current_stream is None:
            current_stream = cuda.CUstream(torch.cuda.current_stream(q_buf.device).cuda_stream)

        # A token-major LSE shares the Q/O dynamic token symbol, so its
        # capacity joins their floor; head-major carries its own declared
        # head_stride (covering the packed total is caller contract — t_q is
        # a device value, Rule 3).
        lse_cap = lse_tensor.numel() // self.h_q if (lse_tensor is not None and not self.thd_stats_head_major) else None
        pack = self._thd_pack(
            q_buf,
            k_buf,
            v_buf,
            o_buf,
            seq_q_lens,
            seq_kv_lens,
            workspace,
            "SdpaFwdDslSm120 (THD)",
            declared_views=True,
            current_stream=current_stream,
            lse_tokens_cap=lse_cap,
        )
        if pack is None:
            return

        lse = None
        if lse_tensor is not None:
            if self.thd_stats_head_major:
                head_stride = self.thd_stats_head_stride
                lse = lse_tensor.as_strided((self.h_q, head_stride), (head_stride, 1), lse_tensor.storage_offset())
            else:
                lse = lse_tensor.as_strided((pack.t_q, self.h_q), (self.h_q, 1), lse_tensor.storage_offset())

        # Sinks are None-specialized like the LSE when the graph has no sink
        # token.
        sinks_t = self._checked_sinks_1d(sinks) if sinks is not None else None

        import cutlass

        # PLAN-TIME-ONLY compile key (issue #552): this lru-cached call
        # re-binds the artifact compile() already built. The K/V strides are
        # taken from the BOUND views because the all-KV-zero clamp swaps in
        # packed batch-1 views (that rare shape mints its own cache entry);
        # the batch stride is zeroed out of the key (a runtime value the
        # kernel rebuilds symbolically).
        kwargs = self._thd_compile_kwargs()
        kwargs.update(k_stride=(0, *pack.K.stride()[1:]), v_stride=(0, *pack.V.stride()[1:]))
        fn = self._k_mod.compile(**kwargs)
        fn(
            pack.Q,
            pack.K,
            pack.V,
            pack.O,
            lse,
            sinks_t,
            pack.seq_q_dummy,
            pack.meta,
            cutlass.Float32(scale_softmax_log2),
            cutlass.Int32(pack.max_sq),
            pack.q_lens_dev,
            pack.kv_lens_dev,
            cutlass.Int32(pack.lens_form),
            current_stream,
        )

    def scratch_workspace_bytes(self) -> int:
        if self.thd:
            # [meta(seq_kv, cu_q, cu_k)].
            # No packed-LSE chunk: with a Stats output the kernel writes the
            # caller's ragged Stats buffer directly (token-major (T, H) or
            # head-major (H, head_stride)); without one it compiles with
            # has_lse=False and no LSE buffer exists at all. No slq/slk
            # copies either: the metadata is built DEVICE-side by the setup
            # kernel (issue #552). No sinks-dummy chunk: the kernel None-specializes
            # on sinks. No O-descriptor chunk: SM120 stores O with plain
            # guarded GMEM stores, so THD needs no per-sequence tensor maps.
            b = self.batch_size
            return ws_align((3 * b + 2) * 4)
        if self.split_kv > 1:
            # Split-major partial slabs (see the SM100 sibling): O_s in the O
            # dtype (half) + lse_s fp32, carved from the caller's workspace.
            b, qh = self.batch_size, self.h_q
            o_bytes = self.split_kv * b * self.s_q_max * qh * self.head_dim_v * self._o_itemsize()
            lse_bytes = self.split_kv * b * qh * self.s_q_max * 4
            return ws_align(o_bytes) + ws_align(lse_bytes)
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


# =============================================================================
# SM80 (A100) adapter — SdpaFwdDslSm80 + the sdpa_fwd_wrapper_sm80 entry point.
#
# The SM80 kernels predate the TemplateParams/load_template pipeline: they
# self-cache one cute.compile artifact per (shape, feature) key inside the
# kernel module (``_compile_cached``'s lru_cache), and take their tile/mask
# configuration as forward() kwargs. This adapter gives them the same
# SdpaFwdDsl lifecycle and keyword contract as the SM100/SM120 adapters so
# ``lower_dsl_prefill`` drives all three identically; converting the kernels
# themselves to TemplateParams form is tracked as follow-up work.
# =============================================================================

from cudnn.sdpa.fwd import config_sm80 as _sm80_config

_SM80_FLAVOR_CFGS = {
    "gptoss": _sm80_config.GPTOSS_CFG,
    "llama": _sm80_config.LLAMA_CFG,
    "dsv3": _sm80_config.DSV3_CFG,
    "qwen": _sm80_config.QWEN_CFG,
}

# (D_QK, D_V) envelope per flavor.
_SM80_FLAVOR_DIMS = {name: (cfg.D_QK, cfg.D_V) for name, cfg in _SM80_FLAVOR_CFGS.items()}

# (tile_m, num_warps, tile_n) per flavor — frozen from the A100 perf sweep.
_SM80_FLAVOR_KNOBS = {name: (cfg.TILE_M, cfg.NUM_WARPS, cfg.TILE_N) for name, cfg in _SM80_FLAVOR_CFGS.items()}

# Causal L2 budget (MiB) for ``sched=lpt_l2`` per flavor.  Larger d_qk
# inflates the per-(B, H) resident set so dsv3 needs a smaller group.
_SM80_FLAVOR_CAUSAL_L2_MIB = {
    "llama": 16,
    "gptoss": 16,
    "dsv3": 8,
    "qwen": 8,
}

# Ascending (D_QK, D_V) order so the flavor pick walks closest-from-above.
_SM80_SUPPORTED_FLAVORS = ("gptoss", "llama", "dsv3", "qwen")

# Flavors that route to the dedicated d=256 kernel (symmetric K+V prefetch);
# all others use the shared generic kernel.
_SM80_D256_FLAVORS = ("qwen",)


def _sm80_pick_flavor(d_qk: int, d_v: int) -> str:
    """Smallest kernel flavor whose ``(D_QK, D_V)`` envelope covers
    ``(d_qk, d_v)``.  Exact-match wins when both axes match; otherwise walk
    gptoss → llama → dsv3 → qwen and pick the first that fits.  Raises if
    nothing fits (heads bigger than the qwen envelope are not supported on
    SM80 yet)."""
    for flavor in _SM80_SUPPORTED_FLAVORS:
        fdqk, fdv = _SM80_FLAVOR_DIMS[flavor]
        if d_qk == fdqk and d_v == fdv:
            return flavor
    for flavor in _SM80_SUPPORTED_FLAVORS:
        fdqk, fdv = _SM80_FLAVOR_DIMS[flavor]
        if d_qk <= fdqk and d_v <= fdv:
            return flavor
    raise ValueError(
        f"SM80 SDPA: no flavor envelope covers (D_QK={d_qk}, D_V={d_v}).  "
        f"Supported envelopes: {_SM80_FLAVOR_DIMS}.  Heads larger than qwen "
        "(256/256) are not yet ported to SM80."
    )


def _sm80_pad_last_dim(t: torch.Tensor, new_last: int) -> torch.Tensor:
    """Zero-pad the trailing dim of a half tensor up to ``new_last``."""
    old_last = t.shape[-1]
    if old_last == new_last:
        return t
    if old_last > new_last:
        raise ValueError(f"_sm80_pad_last_dim: tensor's last dim {old_last} exceeds target {new_last}")
    pad = torch.zeros(
        (*t.shape[:-1], new_last - old_last),
        dtype=t.dtype,
        device=t.device,
    )
    return torch.cat([t, pad], dim=-1).contiguous()


def _sm80_resolve_scheduler(
    *,
    scheduler: str,
    flavor: str,
    is_causal: bool,
    swa_window: int,
    skv: int,
) -> tuple[str, int]:
    """Return ``(sched_token, sched_l2_mib)`` to pass to the kernel."""
    l2_mib = _SM80_FLAVOR_CAUSAL_L2_MIB[flavor]
    if scheduler == "auto":
        if is_causal:
            return "lpt_l2", l2_mib
        if swa_window > 0:
            # SWA heuristic — LPT wins for 1K ≤ SKV ≤ 16K.
            return ("lpt" if 1024 <= skv <= 16384 else "default"), l2_mib
        return "default", l2_mib
    if scheduler in ("natural", "default"):
        return "default", l2_mib
    if scheduler == "lpt":
        return "lpt", l2_mib
    if scheduler == "lpt_l2":
        return "lpt_l2", l2_mib
    raise ValueError(f"SM80 SDPA: scheduler must be 'auto' / 'default' / 'natural' / 'lpt' / 'lpt_l2', got {scheduler!r}")


# --- SM80 template loading ---------------------------------------------------

_LOG2E = math.log2(math.e)

_SM80_KERNEL_FILES = {
    "d256": "prefill_d256_f16_sm80.py",
    "f16": "prefill_f16_sm80.py",
}


def _sm80_load_kernel_module(flavor: str, params):
    """One uniquely-named module per (kernel file, TemplateParams) — the same
    ``frost.template_loader`` mechanism the SM100/SM120 templates use. qwen
    (d=256) routes to the symmetric-K+V-prefetch file; the rest share the
    generic kernel."""
    filename = _SM80_KERNEL_FILES["d256" if flavor in _SM80_D256_FLAVORS else "f16"]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", filename)
    return load_template(path, params, tag=f"sm80_{filename.rsplit('.', 1)[0]}")


def _sm80_sched_policy_int(token: str) -> int:
    return {"default": SCHED_NATURAL, "lpt": SCHED_LPT, "lpt_l2": SCHED_LPT_L2}[token]


def _sm80_call(
    fn,
    *,
    q,
    k,
    v,
    o,
    lse,
    seq_kv,
    seq_q,
    sinks_log2,
    bias,
    cu_q,
    cu_k,
    rope_cs,
    n_kv_tiles,
    scale_log2,
    sq,
    skv,
    d,
    right_bound,
    inv_scale,
    thd_q_tiles,
    n_batch_logical,
    stream,
):
    """Invoke one compiled SM80 artifact (the traced ``_sdpa_host`` ABI:
    12 tensors — LSE may be None-specialized — then 9 runtime scalars and the
    launch stream)."""
    import cutlass
    from cutlass.cute.runtime import from_dlpack as _from_dlpack_raw

    def from_dlpack(t):
        # The kernels compile with --enable-tvm-ffi, so host-side conversions
        # must produce TVM-FFI tensors regardless of the env latch.
        return _from_dlpack_raw(t, enable_tvm_ffi=True)

    fn(
        from_dlpack(q),
        from_dlpack(k),
        from_dlpack(v),
        from_dlpack(o),
        from_dlpack(lse) if lse is not None else None,
        from_dlpack(seq_kv),
        from_dlpack(seq_q),
        from_dlpack(sinks_log2),
        from_dlpack(bias),
        from_dlpack(cu_q),
        from_dlpack(cu_k),
        from_dlpack(rope_cs),
        cutlass.Int32(n_kv_tiles),
        cutlass.Float32(scale_log2),
        cutlass.Int32(sq),
        cutlass.Int32(skv),
        cutlass.Int32(d),
        cutlass.Int32(right_bound),
        cutlass.Float32(inv_scale),
        cutlass.Int32(thd_q_tiles),
        cutlass.Int32(n_batch_logical),
        stream,
    )


class SdpaFwdDslSm80(SdpaFwdDsl):
    """SM80 (A100) SDPA forward via the FROST template kernels.

    Since the TemplateParams conversion this adapter has the same shape as
    its SM100/SM120 siblings end to end: ``check_support`` resolves the
    flavor/mask/scheduler into a plan-time :class:`config_sm80.TemplateParams`,
    ``compile()`` loads the specialized template module and compiles the
    per-shape artifact ONCE (THD packed token extents compile dynamic via
    ``cute.sym_int`` — issue #604's key is gone), and ``execute()`` re-binds
    caller buffers to the cached artifact (a compile-cache miss at execute is
    a bug by contract).

    SM80-only compile axes that have no home in the shared constructor arrive
    as extra keyword-only arguments (``bias_present`` / ``bias_fp32`` /
    ``rope_max_s``); the engine lowering forwards them only when the graph
    declares the operands. ALiBi, block_mask and the score-stat side outputs
    are deliberately NOT served: the capability row declines such graphs and
    the backend takes them.

    Known deviations, pre-existing and tracked rather than introduced here:
    dense GQA expands K/V heads adapter-side until the kernels' native dense
    GQA path is qualified (see ``graph_analyzer.expand_gqa_heads``); an
    off-flavor head dim pads V (and O, via a scratch) host-side; sink logits
    are rescaled to log2 units with one (H,)-element multiply per execute.
    """

    def __init__(self, *args, scheduler: Optional[str] = None, bias_present: bool = False, bias_fp32: bool = False, rope_max_s: int = 0, **kwargs) -> None:
        # SM80-only plan-time axes (see class docstring). ``scheduler`` is the
        # token override for standalone callers; the graph path leaves it None
        # and carries the heuristic's explicit sched_policy knob instead
        # (None = derive via "auto", explicit ints map to their tokens).
        self._scheduler_token = scheduler
        self._bias_present = bool(bias_present)
        self._bias_fp32 = bool(bias_fp32)
        self._rope_max_s = int(rope_max_s)
        super().__init__(*args, **kwargs)

    def _initialize_implementation(self) -> None:
        self.flavor: Optional[str] = None
        self.flavor_d_qk: Optional[int] = None
        self.flavor_d_v: Optional[int] = None
        self.kernel_tile_m: Optional[int] = None
        self.kernel_num_warps: Optional[int] = None
        self.kernel_tile_n: Optional[int] = None
        self.sched_token: Optional[str] = None
        self.sched_l2_mib: Optional[int] = None
        self.mask_token: Optional[str] = None
        self.swa_window_runtime: int = 0
        self.right_bound_runtime: int = 0
        self._k_mod = None
        self._params = None
        self._lse_stride: Optional[tuple[int, int, int]] = None

    # ------------------------------------------------------------------
    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        from cudnn.sdpa.graph_analyzer import dense_layout_ok

        for desc in (self.q_desc, self.k_desc, self.v_desc, self.o_desc):
            self._value_error_if(
                desc.ndim != 4,
                f"{desc.name} must be rank-4 (B, H, S, D); got {desc.ndim}",
            )
            _shape, _stride = tuple(desc.shape), tuple(desc.stride)
            self._value_error_if(
                not dense_layout_ok(_shape, _stride),
                f"{desc.name} must have the head dim innermost-contiguous (stride 1) and "
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

        for label, val in (("B", b), ("H_q", h_qo), ("H_kv", h_kv), ("S_q", s_qo), ("S_kv", s_kv), ("D_QK", d_qk), ("D_V", d_v)):
            self._value_error_if(int(val) <= 0, f"{label} must be > 0; got {val}")

        self._value_error_if(
            h_qo % h_kv != 0,
            f"H_q ({h_qo}) must be divisible by H_kv ({h_kv}) for GQA / MQA",
        )

        max_d_qk = max(fdqk for fdqk, _ in _SM80_FLAVOR_DIMS.values())
        max_d_v = max(fdv for _, fdv in _SM80_FLAVOR_DIMS.values())
        self._value_error_if(
            d_qk > max_d_qk or d_v > max_d_v,
            f"SM80 SDPA: head dim (D_QK={d_qk}, D_V={d_v}) exceeds "
            f"supported envelope (D_QK<={max_d_qk}, D_V<={max_d_v}).  "
            f"Larger heads are not yet ported.",
        )

        self.dtype = self._check_dtype(self.q_desc, [torch.float16, torch.bfloat16], name="Q")
        for desc in (self.k_desc, self.v_desc, self.o_desc):
            self._check_dtype(
                desc,
                self.dtype,
                name=desc.name,
                extra_error_msg=f"{desc.name} must match Q dtype (FP16/BF16 on SM80)",
            )
        self._not_implemented_error_if(
            self._pertensor or self.dtype_o is not None,
            "SM80 SDPA serves f16/bf16 only (no FP8/MXFP8 and no dtype_o override)",
        )
        if self.lse_desc is not None:
            self._check_dtype(self.lse_desc, torch.float32, name="LSE")
            self._check_tensor_shape(self.lse_desc, (b, h_qo, s_qo), name="LSE")
            self._value_error_if(
                not dense_layout_ok((*self.lse_desc.shape, 1), (*self.lse_desc.stride, 1)),
                f"LSE must use a dense-compatible B/H/S permutation or padded layout "
                f"with non-broadcast, non-overlapping-by-span strides; got {self.lse_desc.stride}",
            )
            self._lse_stride = None if self.lse_desc.is_contiguous() else tuple(int(stride) for stride in self.lse_desc.stride)

        self._not_implemented_error_if(
            self.thd or self.cu_seq_q_lens or self.cu_seq_kv_lens,
            "SdpaFwdDslSm80 does not serve packed THD / cu_seq_len graphs; " "sdpa_fwd_wrapper_sm80's varlen path launches them directly",
        )
        self._not_implemented_error_if(
            self.window_size_right is not None and not self.is_causal,
            "SM80 SDPA: window_size_right without is_causal=True has no diagonal to anchor to",
        )
        self._not_implemented_error_if(
            self._rope_max_s and (self.seq_kv_lens_present or self.seq_q_lens_present),
            "SM80 SDPA: RoPE fusion is dense-unpadded-only",
        )
        self._not_implemented_error_if(
            self.split_kv > 1,
            "SM80 SDPA has no split-KV path (no partial slabs, no combine kernel)",
        )
        self._value_error_if(
            self.softmax_precision is not None,
            "SM80 SDPA has no softmax-precision arm yet (softmax_precision must be unset)",
        )

        self._value_error_if(not torch.cuda.is_available(), "CUDA must be available for SM80 SDPA")
        device = self.q_desc.device
        major, minor = torch.cuda.get_device_capability(device)
        self._device_cc = (major, minor)
        self._value_error_if(
            (major, minor) != (8, 0),
            f"SdpaFwdDslSm80 requires SM80 (A100); found SM{major}{minor} on {device}",
        )

        self.flavor = _sm80_pick_flavor(d_qk, d_v)
        self.flavor_d_qk, self.flavor_d_v = _SM80_FLAVOR_DIMS[self.flavor]
        tile_m_default, num_warps_default, tile_n_default = _SM80_FLAVOR_KNOBS[self.flavor]
        self.kernel_tile_m = tile_m_default if self.tile_m is None else int(self.tile_m)
        self.kernel_num_warps = num_warps_default
        self.kernel_tile_n = tile_n_default if self.tile_n is None else int(self.tile_n)
        self._value_error_if(
            self.cga not in (None, 1),
            f"SM80 SDPA has no CGA clustering; cga must be 1 (or unset), got {self.cga}",
        )

        self._value_error_if(
            self.causal_bottom_right and not (self.is_causal or (self.window_size_left is not None and self.window_size_left >= 0)),
            "SM80 SDPA: causal_bottom_right requires is_causal=True and/or a left sliding-window (window_size_left >= 0).",
        )

        swa_left = -1 if self.window_size_left is None else int(self.window_size_left)
        swa_right = 0 if self.window_size_right is None else int(self.window_size_right)
        self.right_bound_runtime = 0
        if self.is_causal:
            self.mask_token = "causal" if swa_left < 0 else "causal_swa"
            self.swa_window_runtime = max(0, swa_left) if swa_left >= 0 else 0
            self.right_bound_runtime = max(0, swa_right)
        elif swa_left >= 0:
            self.mask_token = "swa"
            self.swa_window_runtime = swa_left
        else:
            self.mask_token = "none"
            self.swa_window_runtime = 0

        token = self._scheduler_token
        if token is None:
            # None = no preference anywhere -> "auto" (the adapter derives, a
            # standalone-wrapper convenience). An explicit knob is honored
            # verbatim — NATURAL included — never re-derived.
            if self.sched_policy is None:
                token = "auto"
            else:
                token = {SCHED_NATURAL: "default", SCHED_LPT: "lpt", SCHED_LPT_L2: "lpt_l2"}.get(self.sched_policy)
                self._value_error_if(
                    token is None,
                    f"SM80 SDPA: unsupported sched_policy {self.sched_policy}",
                )
        else:
            _VALID = ("auto", "natural", "default", "lpt", "lpt_l2")
            self._value_error_if(token not in _VALID, f"scheduler must be one of {_VALID}; got {token!r}")
        self.sched_token, self.sched_l2_mib = _sm80_resolve_scheduler(
            scheduler=token,
            flavor=self.flavor,
            is_causal=self.is_causal,
            swa_window=self.swa_window_runtime,
            skv=int(s_kv),
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

    # ------------------------------------------------------------------
    def compile(self) -> None:
        """Load the TemplateParams-specialized module and compile the artifact.

        Plan-time only (Hard Rule 4): every key component here is graph
        declaration or capability data; execute()'s call into the module's
        per-shape lru is a guaranteed hit.
        """
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        from cudnn.sdpa.fwd import config_sm80 as _sm80_cfg

        self._params = _sm80_cfg.TemplateParams(
            io_bf16=(self.dtype == torch.bfloat16),
            d_qk=self.flavor_d_qk,
            d_v=self.flavor_d_v,
            tile_m=self.kernel_tile_m,
            num_warps=self.kernel_num_warps,
            tile_n=self.kernel_tile_n,
            is_causal=self.mask_token in ("causal", "causal_swa"),
            has_swa=self.mask_token in ("swa", "causal_swa"),
            causal_bottom_right=self.causal_bottom_right,
            has_seq_kv_lens=self.seq_kv_lens_present,
            has_seq_q_lens=self.seq_q_lens_present,
            has_sink=self.has_sink,
            has_bias=self._bias_present,
            bias_is_fp32=self._bias_fp32,
            has_rope=self._rope_max_s > 0,
            thd_varlen=False,
            sched_policy=_sm80_sched_policy_int(self.sched_token),
            sched_l2_mib=self.sched_l2_mib,
            has_lse=self.lse_desc is not None,
        )
        self._k_mod = _sm80_load_kernel_module(self.flavor, self._params)
        self._compiled_kernel = self._k_mod.compile(
            b=self.batch_size,
            # Dense GQA is served by adapter-side K/V head expansion until the
            # kernels' native dense-GQA path is qualified (class docstring), so
            # the artifact is compiled against the EXPANDED head count — the
            # shapes execute() actually binds.
            h=self.h_q,
            h_kv=self.h_q,
            sq=self.s_q_max,
            skv=self.s_k_max,
            d=self.head_dim_qk,
            swa_window=int(self.swa_window_runtime),
            rope_max_s=self._rope_max_s,
            lse_stride=self._lse_stride,
        )
        self._logger.debug("compile completed")

    def scratch_workspace_bytes(self) -> int:
        return 0

    # ------------------------------------------------------------------
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
        bias_tensor: Optional[torch.Tensor] = None,
        rope_freqs: Optional[torch.Tensor] = None,
    ) -> None:
        self._logger.debug("Entering execute")
        if self._compiled_kernel is None:
            raise RuntimeError("SdpaFwdDslSm80 is not compiled")
        p = self._params

        # Init-time flags are compile-time specializations; execute must match
        # them exactly, in both directions (Hard Rule 1).
        self._value_error_if(p.has_lse and lse_tensor is None, "compiled with a Stats output but execute() got no lse_tensor")
        self._value_error_if(not p.has_lse and lse_tensor is not None, "lse_tensor provided but the plan compiled the LSE store out")
        self._value_error_if(p.has_bias != (bias_tensor is not None), "bias presence must match the compiled specialization")
        self._value_error_if((p.has_rope) != (rope_freqs is not None), "rope_freqs presence must match the compiled specialization")
        self._value_error_if(p.has_sink != (sinks is not None), "sinks presence must match the compiled specialization")
        self._value_error_if(p.has_seq_kv_lens != (seq_kv_lens is not None), "seq_kv_lens presence must match the compiled specialization")
        self._value_error_if(p.has_seq_q_lens != (seq_q_lens is not None), "seq_q_lens presence must match the compiled specialization")
        # Graph Stats declarations arrive as (B, H, S, 1); the kernels write
        # [B, H, SQ] through the exact declared strides.
        if lse_tensor is not None:
            lse_tensor = self._checked_lse_view(lse_tensor)

        scale_val = self.scale_softmax if (scale_softmax is None or scale_softmax == 0.0) else float(scale_softmax)
        device = q_tensor.device
        launch_stream = self._get_default_stream(current_stream)

        with _torch_stream_context(current_stream, device):
            # BHSD → BSHD views; a dense_flex layout that is not BSHD-physical
            # normalizes with one copy — the same grandfathered normalization
            # the SM100 dense path applies (open cleanup, Hard Rule 2).
            Q = self._to_bshd(q_tensor)
            K = self._to_bshd(k_tensor)
            V = self._to_bshd(v_tensor)
            if self.h_kv != self.h_q:
                # Dense GQA: expand K/V heads until the kernels' native dense
                # GQA path is qualified (see class docstring). BSHD head dim is 2.
                reps = self.h_q // self.h_kv
                K = K.repeat_interleave(reps, dim=2)
                V = V.repeat_interleave(reps, dim=2)

            pad_v = self.head_dim_v < self.flavor_d_v
            if pad_v:
                V = _sm80_pad_last_dim(V, self.flavor_d_v)

            # Output binding: the compiled O ABI is (B, SQ, H, flavor_d_v).
            # Direct-bind the caller's BSHD view when it matches; the padded-V
            # envelope and dense_flex cases go through a scratch + copy-back
            # (both pre-existing normalizations).
            o_view, o_needs_copyback, o_scratch = self._to_bshd_writable(o_tensor)
            if pad_v:
                o_kernel = torch.zeros(self.batch_size, self.s_q_max, self.h_q, self.flavor_d_v, dtype=q_tensor.dtype, device=device)
            elif o_needs_copyback:
                o_kernel = o_scratch
            else:
                o_kernel = o_view
            # DEFENSIVE zero-fill, not load-bearing: the dense epilogue stores
            # every in-bounds row unconditionally; kept so a bound buffer can
            # never surface uninitialized memory if a future path skips rows.
            # (The pad_v scratch above is allocated zeroed already.)
            if (seq_q_lens is not None or seq_kv_lens is not None) and not pad_v:
                o_kernel.zero_()
                if lse_tensor is not None:
                    lse_tensor.zero_()

            # Dummies fill compiled-out ABI slots (Rule 1); the kernel never
            # dereferences them.  Base-class ``_dummy`` (key, device, factory).
            seq_kv_b = (
                self._checked_seq_lens(seq_kv_lens, "seq_kv_lens")
                if seq_kv_lens is not None
                else self._dummy("seq_i32", device, lambda: torch.ones(1, dtype=torch.int32, device=device))
            )
            seq_q_b = (
                self._checked_seq_lens(seq_q_lens, "seq_q_lens")
                if seq_q_lens is not None
                else self._dummy("seq_i32", device, lambda: torch.ones(1, dtype=torch.int32, device=device))
            )
            if sinks is not None:
                # log2-unit rescale: one (H,)-element multiply per execute
                # (pre-existing SM80 contract; the kernels consume log2 units).
                sinks_b = (self._checked_sinks_1d(sinks) * _LOG2E).contiguous()
            else:
                sinks_b = self._dummy("one_f32", device, lambda: torch.ones(1, dtype=torch.float32, device=device))
            if bias_tensor is not None:
                self._value_error_if(
                    bias_tensor.dtype != (torch.float32 if p.bias_is_fp32 else q_tensor.dtype),
                    f"bias dtype must match the compiled specialization; got {bias_tensor.dtype}",
                )
                self._value_error_if(
                    tuple(bias_tensor.shape[-3:]) != (self.h_q, self.s_q_max, self.s_k_max),
                    f"bias trailing dims must be (H, SQ, SKV) = ({self.h_q}, {self.s_q_max}, {self.s_k_max}); got {tuple(bias_tensor.shape)}",
                )
                bias_b = bias_tensor[:1] if bias_tensor.shape[0] != 1 else bias_tensor
                self._value_error_if(not bias_b.is_contiguous(), "bias must be contiguous")
            else:
                bias_dt = q_tensor.dtype
                bias_b = self._dummy(f"one_{bias_dt}", device, lambda: torch.ones(1, dtype=bias_dt, device=device))
            if rope_freqs is not None:
                # (cos, sin) table build — wrapper-only fusion (the engine row
                # never admits RoPE); per-execute by contract, like the caller
                # passing fresh angle tables.
                d2 = self.flavor_d_qk // 2
                rf = rope_freqs.to(dtype=torch.float32, device=device).reshape(rope_freqs.shape[0], -1)
                self._value_error_if(rf.shape[1] < d2, f"rope_freqs last dim ({rf.shape[1]}) must be >= d_qk//2 ({d2})")
                self._value_error_if(
                    rf.shape[0] != self._rope_max_s, f"rope_freqs rows ({rf.shape[0]}) must equal the compiled rope_max_s ({self._rope_max_s})"
                )
                angles = rf[:, :d2]
                rope_b = torch.stack([angles.cos(), angles.sin()], dim=-1).contiguous()
            else:
                rope_b = self._dummy("one_f32", device, lambda: torch.ones(1, dtype=torch.float32, device=device))
            cu_dummy = self._dummy("seq_i32", device, lambda: torch.ones(1, dtype=torch.int32, device=device))

            _sm80_call(
                self._compiled_kernel,
                q=Q,
                k=K,
                v=V,
                o=o_kernel,
                lse=lse_tensor if p.has_lse else None,
                seq_kv=seq_kv_b,
                seq_q=seq_q_b,
                sinks_log2=sinks_b,
                bias=bias_b,
                cu_q=cu_dummy,
                cu_k=cu_dummy,
                rope_cs=rope_b,
                n_kv_tiles=(self.s_k_max + p.tile_n - 1) // p.tile_n,
                scale_log2=scale_val * _LOG2E,
                sq=self.s_q_max,
                skv=self.s_k_max,
                d=self.head_dim_qk,
                right_bound=int(self.right_bound_runtime),
                inv_scale=1.0 / float(scale_val),
                thd_q_tiles=0,
                n_batch_logical=1,
                stream=launch_stream,
            )

            if pad_v:
                o_view.copy_(o_kernel[..., : self.head_dim_v])
            elif o_needs_copyback:
                o_view.copy_(o_scratch)
        self._logger.debug("execute completed")


def _sm80_thd_forward(q, k, v, *, cu_q, cu_k, max_s_q, scale_softmax, is_causal, window_size, causal_bottom_right, bias_tensor, sinks, current_stream=None):
    """THD / varlen forward: q/k/v are PACKED ``[1, T, H, D]`` (already BSHD —
    no transpose), cu_q/cu_k are ``[B+1]`` cumulative seqlens.  Rides the same
    TemplateParams-specialized module as the dense path; the packed token
    extents compile DYNAMIC (``cute.sym_int``), so the compile key is
    plan-time-only and a new token total re-binds the cached artifact (the
    old per-total ``lru`` key — issue #604 — is gone).  Returns packed
    ``[1, T_q, H, D_v]`` O + packed ``[1, H, T_q]`` LSE."""
    from cudnn.sdpa.fwd import config_sm80 as _sm80_cfg

    if bias_tensor is not None:
        raise NotImplementedError("SM80 SDPA THD does not support bias (varlen has no single [1,H,SQ,SKV] bias shape)")
    d_qk = q.shape[-1]
    d_v = v.shape[-1]
    h_q = q.shape[2]
    h_kv = k.shape[2]
    device = q.device
    flavor = _sm80_pick_flavor(d_qk, d_v)
    fdqk, fdv = _SM80_FLAVOR_DIMS[flavor]
    tile_m, num_warps, tile_n = _SM80_FLAVOR_KNOBS[flavor]
    if scale_softmax is None or scale_softmax == 0.0:
        scale_softmax = 1.0 / math.sqrt(d_qk)
    if d_qk < fdqk:
        q = _sm80_pad_last_dim(q, fdqk)
        k = _sm80_pad_last_dim(k, fdqk)
    pad_v = d_v < fdv
    if pad_v:
        v = _sm80_pad_last_dim(v, fdv)
    wl, wr = window_size
    right_bound = wr if (is_causal and wr is not None and wr > 0) else 0

    n_seqs = int(cu_q.numel()) - 1
    if n_seqs < 1:
        raise ValueError("cu_seqlens_q must have >= 2 entries")
    cu_q_t = cu_q.to(dtype=torch.int32, device=device).contiguous()
    cu_k_t = cu_k.to(dtype=torch.int32, device=device).contiguous()

    params = _sm80_cfg.TemplateParams(
        io_bf16=(q.dtype == torch.bfloat16),
        d_qk=fdqk,
        d_v=fdv,
        tile_m=tile_m,
        num_warps=num_warps,
        tile_n=tile_n,
        is_causal=bool(is_causal),
        has_swa=wl is not None and wl >= 0,
        causal_bottom_right=bool(causal_bottom_right),
        has_sink=sinks is not None,
        thd_varlen=True,
        has_lse=True,
    )
    mod = _sm80_load_kernel_module(flavor, params)
    # Off-flavor d_qk was HOST-PADDED to fdqk above, so the compiled fakes and
    # the runtime d must both be the padded width: the kernel derives its Q/K
    # row strides from d_runtime (Q_ROW_STRIDE_E = H * d_runtime), and the
    # zero columns are exact for the QK dot products.  (Same contract as the
    # pre-template forward(), which read d_runtime off the padded shape.)
    fn = mod.compile(
        b=1,
        h=h_q,
        h_kv=h_kv,
        sq=0,
        skv=0,
        d=int(fdqk),
        swa_window=int(max(0, wl)) if wl is not None and wl >= 0 else 0,
        n_batch_logical=n_seqs,
    )

    t_q = q.shape[1]
    o_buf = torch.zeros(1, t_q, h_q, fdv, dtype=q.dtype, device=device)
    lse_buf = torch.zeros(1, h_q, t_q, dtype=torch.float32, device=device)
    sinks_b = (
        (sinks.to(dtype=torch.float32, device=device).reshape(h_q) * _LOG2E).contiguous()
        if sinks is not None
        else torch.ones(1, dtype=torch.float32, device=device)
    )
    dummy_i32 = torch.ones(1, dtype=torch.int32, device=device)
    dummy_f32 = torch.ones(1, dtype=torch.float32, device=device)
    dummy_io = torch.ones(1, dtype=q.dtype, device=device)

    _sm80_call(
        fn,
        q=q,
        k=k,
        v=v,
        o=o_buf,
        lse=lse_buf,
        seq_kv=dummy_i32,
        seq_q=dummy_i32,
        sinks_log2=sinks_b,
        bias=dummy_io,
        cu_q=cu_q_t,
        cu_k=cu_k_t,
        rope_cs=dummy_f32,
        n_kv_tiles=(int(k.shape[1]) + tile_n - 1) // tile_n,
        scale_log2=float(scale_softmax) * _LOG2E,
        sq=int(t_q),
        skv=int(k.shape[1]),
        d=int(fdqk),
        right_bound=int(right_bound),
        inv_scale=1.0 / float(scale_softmax),
        thd_q_tiles=(int(max_s_q) + tile_m - 1) // tile_m,
        n_batch_logical=n_seqs,
        stream=current_stream if current_stream is not None else cuda.CUstream(torch.cuda.current_stream(device).cuda_stream),
    )
    if pad_v:
        o_buf = o_buf[..., :d_v].contiguous()
    return TupleDict(o_tensor=o_buf, lse_tensor=lse_buf)


_sm80_wrapper_cache: dict = {}


def sdpa_fwd_wrapper_sm80(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    is_causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    scale_softmax: Optional[float] = None,
    scale_output: float = 1.0,
    scheduler: str = "auto",
    current_stream: Optional[cuda.CUstream] = None,
    causal_bottom_right: bool = False,
    seq_kv_lens: Optional[torch.Tensor] = None,
    seq_len_q: Optional[torch.Tensor] = None,
    bias_tensor: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    cum_seqlen_q_tensor: Optional[torch.Tensor] = None,
    cum_seqlen_k_tensor: Optional[torch.Tensor] = None,
    max_s_q: Optional[int] = None,
    rope_freqs: Optional[torch.Tensor] = None,
) -> TupleDict:
    """SM80 (A100) SDPA forward.

    Returns ``TupleDict(o_tensor=..., lse_tensor=...)`` matching the DSL
    wrappers' contract.  Dense calls route through :class:`SdpaFwdDslSm80`;
    packed THD calls (``cum_seqlen_*``) ride the same template with dynamic
    token extents.  ALiBi, block_mask and the score-stat side outputs are not
    supported (use the graph API, which routes them to the cuDNN backend).
    """
    if q_tensor.ndim != 4 or v_tensor.ndim != 4:
        raise ValueError(f"Q and V must be rank-4 BHSD; got Q={q_tensor.ndim}D V={v_tensor.ndim}D")
    if scale_output not in (None, 1.0):
        raise NotImplementedError(f"SM80 SDPA: scale_output != 1.0 is not supported yet (got {scale_output})")

    if cum_seqlen_q_tensor is not None:
        if max_s_q is None:
            raise ValueError("THD path requires max_s_q (host int) for the grid")
        if causal_bottom_right and not (is_causal or window_size[0] >= 0):
            raise ValueError("SM80 SDPA: causal_bottom_right requires is_causal=True and/or a left sliding-window (window_size_left >= 0).")
        for label, present in (
            ("rope_freqs", rope_freqs is not None),
            ("seq_kv_lens", seq_kv_lens is not None),
            ("seq_len_q", seq_len_q is not None),
            ('scheduler != "auto"', scheduler not in (None, "auto")),
        ):
            if present:
                raise NotImplementedError(f"SM80 SDPA THD (cum_seqlen_*) path does not support {label}; the dense path serves it")
        with _torch_stream_context(current_stream, q_tensor.device):
            return _sm80_thd_forward(
                q_tensor,
                k_tensor,
                v_tensor,
                cu_q=cum_seqlen_q_tensor,
                cu_k=cum_seqlen_k_tensor,
                max_s_q=max_s_q,
                scale_softmax=scale_softmax,
                is_causal=is_causal,
                window_size=window_size,
                causal_bottom_right=causal_bottom_right,
                bias_tensor=bias_tensor,
                sinks=sinks,
                current_stream=current_stream,
            )

    b, h_q, s_q, _ = q_tensor.shape
    d_v = v_tensor.shape[-1]
    o_tensor = torch.empty(
        (b, s_q, h_q, d_v),
        dtype=q_tensor.dtype,
        device=q_tensor.device,
    ).transpose(1, 2)
    lse_tensor = _allocate_lse_tensor(q_tensor)

    wl, wr = window_size
    if not is_causal and wr >= 0:
        raise NotImplementedError("SM80 SDPA: window_size_right without is_causal=True has no effect; pass is_causal=True or a left window")
    rope_max_s = int(rope_freqs.shape[0]) if rope_freqs is not None else 0
    cache_key = (
        q_tensor.shape,
        k_tensor.shape,
        v_tensor.shape,
        q_tensor.dtype,
        bool(is_causal),
        (wl, wr),
        scale_softmax,
        scheduler,
        bool(causal_bottom_right),
        seq_kv_lens is not None,
        seq_len_q is not None,
        sinks is not None,
        bias_tensor is not None,
        (bias_tensor.dtype if bias_tensor is not None else None),
        rope_max_s,
        q_tensor.device,
    )
    api = _sm80_wrapper_cache.get(cache_key)
    if api is None:
        api = SdpaFwdDslSm80(
            sample_q=q_tensor,
            sample_k=k_tensor,
            sample_v=v_tensor,
            sample_o=o_tensor,
            sample_lse=lse_tensor,
            is_causal=is_causal,
            causal_bottom_right=causal_bottom_right,
            window_size_left=(wl if wl >= 0 else None),
            window_size_right=(wr if (is_causal and wr >= 0) else None),
            scale_softmax=scale_softmax,
            seq_kv_lens_present=seq_kv_lens is not None,
            seq_q_lens_present=seq_len_q is not None,
            has_sink=sinks is not None,
            scheduler=scheduler,
            bias_present=bias_tensor is not None,
            bias_fp32=(bias_tensor is not None and bias_tensor.dtype == torch.float32),
            rope_max_s=rope_max_s,
        )
        api.check_support()
        api.compile()
        _sm80_wrapper_cache[cache_key] = api
    api.execute(
        q_tensor=q_tensor,
        k_tensor=k_tensor,
        v_tensor=v_tensor,
        o_tensor=o_tensor,
        lse_tensor=lse_tensor,
        sinks=sinks,
        seq_q_lens=seq_len_q,
        seq_kv_lens=seq_kv_lens,
        scale_softmax=scale_softmax,
        current_stream=current_stream,
        bias_tensor=bias_tensor,
        rope_freqs=rope_freqs,
    )
    return TupleDict(o_tensor=o_tensor, lse_tensor=lse_tensor)
