"""APIBase wrapper for the lean H=64/D=128 indexer-forward score kernel.

``IndexerForwardLean`` is an additive fast path next to
:class:`cudnn.deepseek_sparse_attention.indexer_forward.IndexerForward`:
``check_support()`` returns ``True`` only for the configuration the lean
schedule is specialized for (``head_dim == 128``, ``qhead_per_kv_head ==
64``, ``h_kv == 1``, BF16 Q/K with BF16/FP32 W and FP32 scores, batched
uniform-length BSHD, and a q-tile count that keeps the persistent grid
saturated). ``indexer_forward_wrapper`` consults the same gate and
dispatches here transparently; everything the gate rejects (THD/varlen
ragged batches in particular — the static reversed-LPT single-wave grid
is built around one uniform triangular work distribution, and per-batch
KV offsets would break the 128-row-aligned dense KV tile sweep) keeps
using the legacy path, which is never modified.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from typing import Optional

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_stream

from cudnn.api_base import APIBase, TupleDict

from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import (
    resolve_stream,
    torch_stream_context,
    validate_q_causal_offsets,
)

from .indexer_fwd_sm100_lean import IndexerForwardSm100Lean

# Lean-path dispatch constants (documented, fixed for now):
#
# LEAN_MIN_WAVES: the lean schedule uses a static single-wave persistent
# grid of min(sm_count, num_m_tiles) CTAs with reversed-LPT tile order.
# Dispatch to lean requires
#     num_m_tiles >= LEAN_MIN_WAVES * sm_count
# where num_m_tiles = s_q // LEAN_TILE_TOKENS is the number of persistent
# q-tiles (metadata-only; sm_count is queried from the target device).
#
# Calibration (B200, 148 SMs, nsys pure-kernel medians, lean vs legacy
# IndexerForwardSm100, BF16 H=64 D=128 MQA, 50 warmup / 100 reps):
#
#   ratio 2, S_k = S_q/2:            ratio 4, S_k = S_q/4:
#     S_q   waves  speedup             S_q   waves  speedup
#     256    0.43   1.27x              1024   1.73   1.40x
#     512    0.86   1.38x              4096   6.92   1.48x
#     768    1.30   1.45x              8192  13.84   1.47x
#    1024    1.73   1.43x
#    1536    2.59   1.37x
#    2048    3.46   1.33x
#    3072    5.19   1.43x
#    4096    6.92   1.47x
#    5120    8.65   1.46x
#    6144   10.38   1.44x
#    8192   13.84   1.44x
#
# The lean schedule won at every measured tile count, including sub-wave
# grids, so the gate is not a measured perf boundary. It is kept at one
# full wave because (a) below one wave the persistent grid no longer
# saturates the GPU, which is the regime the schedule was designed and
# validated for, and (b) the lean path compiles per (S_q, S_k, W dtype,
# sm_scale variant, device) — the schedule is static — so auto-dispatching
# arbitrarily small dynamic shapes would trade a ~1-2 us per-call win for
# a JIT compile per novel shape. Callers who want lean below the gate can
# invoke indexer_forward_lean_wrapper explicitly after lowering
# LEAN_MIN_WAVES.
LEAN_MIN_WAVES = 1
# query tokens per q-tile: the kernel packs TQ=4 tokens x 64 heads into
# its N=256 MMA tile.
LEAN_TILE_TOKENS = 4
# KV rows per MMA tile in the lean kernel: multiples compile with a fully
# static K extent; ragged extents use a dynamic K dim + TMA zero-fill.
_LEAN_KV_TILE_ROWS = 128
# TMA descriptors are built with assumed_align=16. Contiguity does NOT
# imply base-pointer alignment (storage-offset views), so the runtime
# data_ptr() of every kernel operand is checked against this.
_TMA_MIN_ALIGN_BYTES = 16
# Legacy evaluates (q_causal_offset + q_token + 1) in int32 on device;
# the lean host path evaluates the same window in int64 and clamps. The
# two agree only while the legacy intermediate cannot overflow, so the
# dispatcher keeps offsets beyond this bound on the legacy path.
_INT32_MAX = 2**31 - 1
# Bound + eviction policy follow python/cudnn/graph.py's graph_cache
# precedent (maxsize=256).
_LEAN_CACHE_MAXSIZE = 256


class _LruDict:
    """Bounded thread-safe LRU mapping.

    Mirrors the bound of ``graph.py``'s ``graph_cache`` (maxsize 256) with
    true LRU eviction and a lock, since dispatch may be called from
    multiple threads.
    """

    def __init__(self, maxsize: int = _LEAN_CACHE_MAXSIZE):
        self._data: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._maxsize = maxsize

    def get(self, key, default=None):
        with self._lock:
            if key not in self._data:
                return default
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key, value) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)


class IndexerForwardLean(APIBase):
    """SM100 lean APIBase implementation used by ``indexer_forward_lean_wrapper``.

    Computes ``scores[b, i, j] = sm_scale * sum_h relu(q[b,i,h,:] .
    k[b,j,0,:]) * w[b,i,h]`` for ``j`` inside the per-row visibility
    window ``[ks[b, i], ke[b, i])`` (positions inside swept KV tiles but
    outside the window are ``-inf``; the wrapper pre-fills the output so
    tiles the kernel never sweeps also read ``-inf``). The wrapper builds
    ratio-causal windows ``ke = clamp((q_causal_offsets[b] + i + 1) //
    ratio, 0, S_k)`` matching the legacy mask semantics.

    The compiled kernel is batch-size independent (it runs on flattened
    per-batch views); ``execute()`` reads the batch size from the runtime
    tensors, so one compiled instance serves every ``B`` of the same
    ``(S_q, S_k)``.
    """

    def __init__(
        self,
        sample_q: torch.Tensor,  # (B, S_q, H_q, D) BF16
        sample_k: torch.Tensor,  # (B, S_k, H_kv, D) BF16
        sample_w: torch.Tensor,  # (B, S_q, H_q) BF16 or FP32
        sample_out: torch.Tensor,  # (B, S_q, S_k) FP32
        ratio: int = 4,
        qhead_per_kv_head: Optional[int] = None,
        sm_scale: float = 1.0,
    ):
        super().__init__()
        self._kernel = IndexerForwardSm100Lean

        self.q_desc = self._make_tensor_desc(sample_q, name="sample_q")
        self.k_desc = self._make_tensor_desc(sample_k, name="sample_k")
        self.w_desc = self._make_tensor_desc(sample_w, name="sample_w")
        self.o_desc = self._make_tensor_desc(sample_out, name="sample_out")

        self.ratio = int(ratio)
        # sm_scale selects the compiled VARIANT only (whether the epilogue
        # multiply exists at all); the value itself is a runtime kernel
        # argument that callers pass to execute() per call. The cached
        # instance is never mutated after compile.
        self.sm_scale = float(sm_scale)
        self._kernel_applies_sm_scale = False
        self.qhead_per_kv_head = qhead_per_kv_head

        self.batch_size = None  # sample batch size (execute() is B-agnostic)
        self.s_q = None
        self.s_k = None
        self.h_q = None
        self.h_kv = None
        self.head_dim = None
        self.sm_count = None
        self.target_device: Optional[torch.device] = None

    def _unsupported(self, msg: str) -> bool:
        """Soft-fail a lean-path gate: log and report unsupported."""
        self._logger.debug("IndexerForwardLean unsupported: %s", msg)
        self._is_supported = False
        return False

    def check_support(self) -> bool:
        """Lean-path dispatch gate.

        Malformed inputs (rank/shape mismatches between Q/K/W/Out) raise
        ``ValueError`` exactly like ``IndexerForward.check_support``;
        configurations that are merely outside the lean specialization
        return ``False`` so the caller can fall back to the legacy path.

        Descriptors carry no storage pointers, so base-pointer alignment
        (TMA needs 16 B) is a runtime concern checked in ``execute()`` and
        by the dispatcher, not here. Meta-device sample tensors are
        accepted for metadata-only support checks; all real (CUDA)
        descriptors must agree on one device, and compute capability plus
        sm_count are queried from that resolved device (falling back to
        the current device when every descriptor is meta).
        """
        self._logger.debug("Entering check_support")
        self._value_error_if(
            self.q_desc.ndim != 4,
            f"Q must be 4-D (B, S_q, H_q, D), got {self.q_desc.shape}",
        )
        self._value_error_if(
            self.k_desc.ndim != 4,
            f"K must be 4-D (B, S_k, H_kv, D), got {self.k_desc.shape}",
        )
        self._value_error_if(
            self.w_desc.ndim != 3,
            f"W must be 3-D (B, S_q, H_q), got {self.w_desc.shape}",
        )
        self._value_error_if(
            self.o_desc.ndim != 3,
            f"Out must be 3-D (B, S_q, S_k), got {self.o_desc.shape}",
        )

        b, s_q, h_q, d = self.q_desc.shape
        b_k, s_k, h_kv, d_k = self.k_desc.shape
        b_o, s_q_out, s_k_out = self.o_desc.shape
        self._value_error_if(b != b_k, f"Batch size mismatch Q={b} vs K={b_k}")
        self._value_error_if(b != b_o, f"Batch size mismatch Q={b} vs Out={b_o}")
        self._value_error_if(s_q != s_q_out, f"S_q mismatch Q={s_q} vs Out={s_q_out}")
        self._value_error_if(d != d_k, f"Head dim mismatch Q={d} vs K={d_k}")
        self._value_error_if(
            self.w_desc.shape != (b, s_q, h_q),
            f"W must have shape (B, S_q, H_q) = ({b}, {s_q}, {h_q}), got {self.w_desc.shape}",
        )

        qhpkv = self.qhead_per_kv_head if self.qhead_per_kv_head is not None else (h_q // h_kv)
        self._value_error_if(
            qhpkv * h_kv != h_q,
            f"qhead_per_kv_head * h_kv != h_q ({qhpkv} * {h_kv} != {h_q})",
        )
        self.qhead_per_kv_head = qhpkv

        # ---- lean-specialization gates (soft: False -> legacy fallback) ----
        if d != 128:
            return self._unsupported(f"head_dim must be 128, got {d}")
        if qhpkv != 64:
            return self._unsupported(f"qhead_per_kv_head must be 64, got {qhpkv}")
        if h_kv != 1:
            return self._unsupported(f"h_kv must be 1, got {h_kv}")
        if self.q_desc.dtype != torch.bfloat16 or self.k_desc.dtype != torch.bfloat16:
            return self._unsupported(f"Q/K must be bfloat16, got {self.q_desc.dtype}/{self.k_desc.dtype}")
        if self.w_desc.dtype not in (torch.bfloat16, torch.float32):
            return self._unsupported(f"W must be bfloat16 or float32, got {self.w_desc.dtype}")
        if self.o_desc.dtype != torch.float32:
            return self._unsupported(f"Out must be float32, got {self.o_desc.dtype}")
        if not math.isfinite(self.sm_scale):
            return self._unsupported(f"sm_scale must be finite, got {self.sm_scale}")
        if self.ratio < 1:
            return self._unsupported(f"ratio must be >= 1, got {self.ratio}")
        if s_q % LEAN_TILE_TOKENS != 0:
            return self._unsupported(f"S_q must be a multiple of {LEAN_TILE_TOKENS}, got {s_q}")
        if s_k < 1 or s_k_out != s_k:
            return self._unsupported(f"Out column dim must equal S_k >= 1, got S_k={s_k} Out={s_k_out}")
        for desc, name in (
            (self.q_desc, "Q"),
            (self.k_desc, "K"),
            (self.w_desc, "W"),
            (self.o_desc, "Out"),
        ):
            if not desc.is_contiguous():
                return self._unsupported(f"{name} must be contiguous")

        # device resolution: all runtime (CUDA) descriptors on one device;
        # meta descriptors are metadata-only stand-ins and pin nothing.
        devices = {desc.device for desc in (self.q_desc, self.k_desc, self.w_desc, self.o_desc)}
        if any(dev.type not in ("cuda", "meta") for dev in devices):
            return self._unsupported(f"Q/K/W/Out must be CUDA tensors, got devices {sorted(str(dev) for dev in devices)}")
        cuda_devices = {dev for dev in devices if dev.type == "cuda"}
        if len(cuda_devices) > 1:
            return self._unsupported(f"Q/K/W/Out must share one CUDA device, got {sorted(str(dev) for dev in cuda_devices)}")
        if not torch.cuda.is_available():
            return self._unsupported("CUDA is not available")
        if cuda_devices:
            target = next(iter(cuda_devices))
            if target.index is None:
                target = torch.device("cuda", torch.cuda.current_device())
        else:
            target = torch.device("cuda", torch.cuda.current_device())

        # full (major, minor) from the TARGET device (not the current one,
        # and not the process-wide lru-cached device_major()). Every
        # SM10.x minor exposes the tcgen05 UMMA/TMEM/LDTM features and the
        # 227 KB CTA smem carveout this schedule needs; SM90 and SM110+
        # stay on the legacy path.
        major, minor = torch.cuda.get_device_capability(target)
        if major != 10:
            return self._unsupported(f"lean schedule requires SM100-class compute capability, found SM{major}.{minor} on {target}")

        # occupancy gate: keep the static persistent grid saturated
        # (metadata-only; sm_count comes from the target device runtime,
        # not GPU memory)
        num_m_tiles = s_q // LEAN_TILE_TOKENS
        sm_count = torch.cuda.get_device_properties(target).multi_processor_count
        if num_m_tiles < LEAN_MIN_WAVES * sm_count:
            return self._unsupported(
                f"num_m_tiles={num_m_tiles} < LEAN_MIN_WAVES*sm_count="
                f"{LEAN_MIN_WAVES}*{sm_count} — static reversed-LPT grid "
                f"needs >= {LEAN_MIN_WAVES} q-tiles per SM"
            )

        self.batch_size = b
        self.s_q = s_q
        self.s_k = s_k
        self.h_q = h_q
        self.h_kv = h_kv
        self.head_dim = d
        self.sm_count = sm_count
        self.target_device = target
        self._is_supported = True
        return True

    def compile(self) -> None:
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return

        # the sm_scale == 1.0 variant compiles the epilogue multiply out
        # entirely by passing sm_scale=None (an absent optional folds out of
        # the kernel parameter layout at trace time, keeping the production
        # instruction stream identical to the scale-free schedule;
        # _maybe_lean_api keys the cache on this variant choice)
        self._kernel_applies_sm_scale = self.sm_scale != 1.0

        s_q, s_k, h, d = self.s_q, self.s_k, self.h_q, self.head_dim
        # generated code depends only on this key: shapes enter through the
        # fake tensors below (flattened per-batch views — B never appears),
        # the sm_scale variant through the optional scalar, and the device
        # through sm_count/toolchain target. ratio and B are runtime-side.
        compile_key = (self.target_device.index, s_q, s_k, self.w_desc.dtype, self._kernel_applies_sm_scale)
        cached = _lean_compile_cache.get(compile_key)
        if cached is not None:
            self._compiled_kernel = cached
            self._logger.debug("Kernel fetched from compile cache")
            return

        kernel_obj = self._kernel(
            num_heads=self.qhead_per_kv_head,
            head_dim=self.head_dim,
            sm_count=self.sm_count,
        )

        # flattened single-batch views the kernel operates on (B > 1 loops
        # per-batch launches over the same compiled kernel)
        fake_q = self._make_fake_cute_tensor(torch.bfloat16, (s_q * h, d), (d, 1), assumed_align=16)
        if s_k % _LEAN_KV_TILE_ROWS == 0:
            # tile-aligned K: fully static extent (best address codegen)
            fake_k = self._make_fake_cute_tensor(torch.bfloat16, (s_k, d), (d, 1), assumed_align=16)
        else:
            # ragged K: the row count is a dynamic dim, so the TMA
            # descriptor carries the true runtime extent and the trailing
            # partial 128-row KV tile is zero-filled by the TMA hardware
            # instead of tripping the static tile-divisibility check
            fake_k = self._make_fake_cute_compact_tensor(
                torch.bfloat16,
                (s_k, d),
                stride_order=(1, 0),
                assumed_align=16,
                dynamic_mode=0,
                divisibility=1,
            )
        fake_w = self._make_fake_cute_tensor(self.w_desc.dtype, (s_q, h), (h, 1), assumed_align=16)
        fake_ks = self._make_fake_cute_tensor(torch.int32, (s_q,), (1,), assumed_align=16)
        fake_ke = self._make_fake_cute_tensor(torch.int32, (s_q,), (1,), assumed_align=16)
        fake_out = self._make_fake_cute_tensor(torch.float32, (s_q, s_k), (s_k, 1), assumed_align=16)

        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)

        # Compile-failure policy: a config check_support() accepted MUST
        # compile; if it does not, that is a lean-path bug and it raises
        # loudly (a silent legacy fallback here would hide real compiler
        # or schedule bugs). CUDNNFE_DSA_INDEXER_FWD_DISABLE_LEAN is the
        # escape hatch while such a bug is being fixed.
        try:
            _compiled_kernel = cute.compile(
                kernel_obj,
                fake_q,
                fake_k,
                fake_w,
                fake_ks,
                fake_ke,
                fake_out,
                cutlass.Float32(self.sm_scale) if self._kernel_applies_sm_scale else None,
                fake_stream,
                options=compile_options(),
            )
        except Exception as exc:
            raise RuntimeError(
                f"IndexerForwardLean failed to compile a configuration check_support() accepted "
                f"(S_q={s_q}, S_k={s_k}, W dtype={self.w_desc.dtype}, scaled={self._kernel_applies_sm_scale}, "
                f"device={self.target_device}). This is a lean fast-path bug; set "
                f"CUDNNFE_DSA_INDEXER_FWD_DISABLE_LEAN=1 to force the legacy kernel while it is investigated."
            ) from exc

        def tensor_api(q_flat, k, w, ks, ke, out, sm_scale, stream):
            # The kernel sweeps whole 128-column KV tiles of each q-tile's
            # union visibility window: in-window positions get scores,
            # out-of-window positions inside swept tiles get -inf, and
            # never-swept tiles are left untouched. Callers that depend on
            # -inf there must pre-fill the output (the wrapper does).
            return _compiled_kernel(q_flat, k, w, ks, ke, out, sm_scale, stream)

        _lean_compile_cache.put(compile_key, tensor_api)
        self._compiled_kernel = tensor_api
        self._logger.debug("Kernel compiled successfully")

    def execute(
        self,
        q: torch.Tensor,  # (B, S_q, H_q, D) BF16, contiguous
        k: torch.Tensor,  # (B, S_k, 1, D) BF16, contiguous
        w: torch.Tensor,  # (B, S_q, H_q) BF16/FP32, contiguous
        ks: torch.Tensor,  # (B, S_q) INT32 per-row window start (rows contiguous)
        ke: torch.Tensor,  # (B, S_q) INT32 per-row window end (exclusive)
        out: torch.Tensor,  # (B, S_q, S_k) FP32, pre-filled with -inf
        sm_scale: Optional[float] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Run the compiled kernel, one persistent launch per batch entry.

        ``sm_scale`` is a per-call runtime argument (``None`` uses the
        construction-time default); the cached instance itself is never
        mutated here, so concurrent same-shape callers with different
        scales do not race. ``B`` is read from the runtime tensors — the
        compiled kernel operates on flattened per-batch views and is
        batch-size independent.
        """
        self._logger.debug("Entering execute")
        current_stream = resolve_stream(current_stream)
        if self._compiled_kernel is None:
            raise ValueError("IndexerForwardLean kernel not compiled")
        scale_value = float(self.sm_scale if sm_scale is None else sm_scale)
        if not math.isfinite(scale_value):
            raise ValueError(f"sm_scale must be finite, got {scale_value}")
        if scale_value != 1.0 and not self._kernel_applies_sm_scale:
            raise ValueError(
                "IndexerForwardLean was compiled as the sm_scale == 1.0 variant "
                f"(the epilogue multiply is compiled out); got sm_scale={scale_value}. "
                "Build a separate instance for scaled execution."
            )
        b = int(q.shape[0]) if q.ndim == 4 else 0
        s_q, s_k, h_q, d = self.s_q, self.s_k, self.h_q, self.head_dim
        for t, name, shape, dtype in (
            (q, "q", (b, s_q, h_q, d), torch.bfloat16),
            (k, "k", (b, s_k, self.h_kv, d), torch.bfloat16),
            (w, "w", (b, s_q, h_q), self.w_desc.dtype),
            (ks, "ks", (b, s_q), torch.int32),
            (ke, "ke", (b, s_q), torch.int32),
            (out, "out", (b, s_q, s_k), torch.float32),
        ):
            if tuple(t.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(t.shape)}")
            if t.dtype != dtype:
                raise ValueError(f"{name} must have dtype {dtype}, got {t.dtype}")
            if not t.is_cuda or t.device != q.device:
                raise ValueError(f"{name} must be a CUDA tensor on {q.device}, got {t.device}")
            # ks/ke may be batch-expanded views (stride 0 on B); only their
            # per-batch rows must be dense. Everything else is viewed flat.
            if name in ("ks", "ke"):
                if t.stride(-1) != 1:
                    raise ValueError(f"{name} rows must be contiguous, got stride {tuple(t.stride())}")
            elif not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous, got stride {tuple(t.stride())}")
            if t.data_ptr() % _TMA_MIN_ALIGN_BYTES:
                raise ValueError(f"{name} base pointer must be {_TMA_MIN_ALIGN_BYTES}-byte aligned for TMA, got 0x{t.data_ptr():x}")
        sm_scale_arg = cutlass.Float32(scale_value) if self._kernel_applies_sm_scale else None
        # one single-wave persistent launch per batch; K/windows/output are
        # per-batch, so the batch dim folds into sequential launches on the
        # same stream (uniform-length BSHD only — the gate rejects ragged)
        for bi in range(b):
            self._compiled_kernel(
                q[bi].view(s_q * h_q, d),
                k[bi].view(s_k, d),
                w[bi].view(s_q, h_q),
                ks[bi],
                ke[bi],
                out[bi].view(s_q, s_k),
                sm_scale_arg,
                current_stream,
            )


# module-level bounded LRU caches (thread-safe, maxsize per graph.py's
# graph_cache precedent):
#   _dispatch_cache:      verdict key -> compiled IndexerForwardLean | None
#   _lean_compile_cache:  codegen key -> compiled kernel callable (shared
#                         across instances; B and ratio never affect codegen)
#   _window_cache:        (S_q, S_k, ratio, device) -> single-row ks/ke + event
_dispatch_cache = _LruDict()
_lean_compile_cache = _LruDict()
_window_cache = _LruDict()
_MISS = object()
# serializes verdict construction + JIT so concurrent same-config callers
# cannot compile the same kernel twice
_lean_build_lock = threading.Lock()


def _sample_out_like(b: int, s_q: int, s_k: int) -> torch.Tensor:
    """Metadata-only stand-in for the (B, S_q, S_k) FP32 output.

    ``check_support`` only consumes shape/stride/dtype, so describe the
    output on the meta device instead of allocating the real buffer (which
    can be GBs) just to decide dispatch.
    """
    return torch.empty(b, s_q, s_k, dtype=torch.float32, device="meta")


def _q_causal_offsets_within_int32(q_causal_offsets: Optional[torch.Tensor], s_q: int) -> bool:
    """True when the legacy kernel's int32 window arithmetic cannot overflow.

    Legacy computes ``(q_causal_offset + q_token + 1) // ratio`` in Int32 on
    device (``indexer_fwd_sm100.py``), which wraps for offsets near
    INT32_MAX; the lean host path computes the same windows in int64 and
    clamps, which does not. Additive discipline: extreme-but-valid offsets
    where ``offset + S_q + 1 > INT32_MAX`` stay on the legacy path so
    dispatched behavior is unchanged. Negative offsets cannot underflow
    (``offset + i + 1 >= INT32_MIN + 1``) and clamp identically on both
    paths.

    NOTE: this reads the small ``(B,)`` offsets tensor (one device sync),
    and only runs on the offsets path for otherwise lean-eligible configs.
    """
    if q_causal_offsets is None or q_causal_offsets.numel() == 0:
        return True
    off_max = int(q_causal_offsets.max().item())
    return off_max + s_q + 1 <= _INT32_MAX


def _maybe_lean_api(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    ratio: int,
    qhead_per_kv_head: Optional[int],
    sm_scale: float = 1.0,
) -> Optional[IndexerForwardLean]:
    """Return a compiled ``IndexerForwardLean`` when the config is inside the
    lean specialization, else ``None`` (dispatching callers fall back to
    legacy).

    Structure (dispatch exception boundary):

    - Cheap NON-THROWING per-call guards run first and are never cached:
      rank, same-CUDA-device, dtype, batch/W-shape consistency, contiguity,
      base-pointer TMA alignment (a per-pointer property — storage-offset
      views defeat metadata caching), finite ``sm_scale`` (a non-finite
      scale must neither poison nor hit the cached finite-variant
      verdicts), and ``ratio >= 1``. Anything they reject goes to legacy,
      which produces its own error surface for malformed inputs.
    - The cached verdict is keyed ONLY by metadata that determines the
      ``check_support`` verdict and the generated code. ``B``, ``ratio``
      and the exact ``sm_scale`` value are excluded: the kernel compiles
      on flattened per-batch views (B-independent — ``execute()`` reads B
      from the runtime tensors), ratio only shapes the host-built windows,
      and only the ``sm_scale == 1.0`` variant choice reaches codegen.
    - ``check_support`` may still raise ``ValueError`` for cross-tensor
      mismatches (mirroring ``IndexerForward.check_support``); dispatching
      callers catch exactly that and let the legacy path raise its own
      errors. ``compile()`` failures for accepted configs raise
      ``RuntimeError`` and are deliberately NOT caught (see compile()).
    """
    # ---- cheap, non-throwing per-call eligibility (False -> legacy) ----
    if q.ndim != 4 or k.ndim != 4 or w.ndim != 3:
        return None
    if not (q.is_cuda and k.is_cuda and w.is_cuda) or not (q.device == k.device == w.device):
        return None
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or w.dtype not in (torch.bfloat16, torch.float32):
        return None
    b, s_q, h_q, d = q.shape
    if k.shape[0] != b or tuple(w.shape) != (b, s_q, h_q):
        return None
    if not (q.is_contiguous() and k.is_contiguous() and w.is_contiguous()):
        return None
    if (q.data_ptr() % _TMA_MIN_ALIGN_BYTES) or (k.data_ptr() % _TMA_MIN_ALIGN_BYTES) or (w.data_ptr() % _TMA_MIN_ALIGN_BYTES):
        return None
    if not math.isfinite(sm_scale):
        return None
    if int(ratio) < 1:
        return None

    s_k, h_kv = k.shape[1], k.shape[2]
    key = (
        s_q,
        s_k,
        h_q,
        h_kv,
        d,
        w.dtype,
        qhead_per_kv_head,
        float(sm_scale) != 1.0,
        q.device.index,
    )
    hit = _dispatch_cache.get(key, _MISS)
    if hit is not _MISS:
        return hit

    with _lean_build_lock:
        hit = _dispatch_cache.get(key, _MISS)
        if hit is not _MISS:
            return hit
        api = IndexerForwardLean(
            sample_q=q,
            sample_k=k,
            sample_w=w,
            sample_out=_sample_out_like(b, s_q, s_k),
            ratio=ratio,
            qhead_per_kv_head=qhead_per_kv_head,
            sm_scale=sm_scale,
        )
        if api.check_support():
            api.compile()
        else:
            api = None
        _dispatch_cache.put(key, api)
        return api


def _ratio_causal_windows(
    batch: int,
    s_q: int,
    s_k: int,
    ratio: int,
    device,
    q_causal_offsets: Optional[torch.Tensor] = None,
    stream: Optional[cuda.CUstream] = None,
) -> tuple:
    """(B, S_q) int32 window tensors for the legacy ratio-causal mask:
    ``ks = 0``, ``ke[b, i] = clamp((q_causal_offsets[b] + i + 1) // ratio,
    0, S_k)`` (offsets are 0 when omitted; int64 host math, clamped before
    the int32 cast).

    The offset-free windows are cached per ``(S_q, S_k, ratio, device)``
    as single-row tensors and batch-expanded per call (metadata-only).
    They are produced asynchronously on the first caller's stream, so a
    CUDA event recorded right after production is waited on by every
    consuming stream before reuse — the cheapest correct option (one
    ``wait_event`` per hit) versus per-stream rebuilds or a producer-side
    sync. Offset windows are rebuilt each call because the offsets tensor
    contents may change between calls with identical metadata.
    """
    if q_causal_offsets is None:
        key = (s_q, s_k, int(ratio), device.index)
        hit = _window_cache.get(key)
        if hit is None:
            with torch_stream_context(stream):
                ks_row = torch.zeros(1, s_q, dtype=torch.int32, device=device)
                ke_row = (((torch.arange(s_q, device=device) + 1) // int(ratio)).clamp_(0, s_k)).to(torch.int32).view(1, s_q)
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(device))
            hit = (ks_row, ke_row, ready)
            _window_cache.put(key, hit)
        else:
            ks_row, ke_row, ready = hit
            with torch_stream_context(stream):
                torch.cuda.current_stream(device).wait_event(ready)
        return ks_row.expand(batch, s_q), ke_row.expand(batch, s_q)
    with torch_stream_context(stream):
        ks = torch.zeros(1, s_q, dtype=torch.int32, device=device).expand(batch, s_q)
        pos = torch.arange(1, s_q + 1, device=device, dtype=torch.int64).view(1, s_q)
        ke = ((q_causal_offsets.view(batch, 1).to(torch.int64) + pos) // int(ratio)).clamp_(0, s_k).to(torch.int32)
    return ks, ke


def indexer_forward_lean_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    ratio: int = 4,
    qhead_per_kv_head: Optional[int] = None,
    sm_scale: float = 1.0,
    q_causal_offsets: Optional[torch.Tensor] = None,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """High-level wrapper for the lean H=64/D=128 batched BSHD path.

    Allocates and ``-inf``-pre-fills the ``(B, S_q, S_k)`` FP32 score
    buffer (always contiguous — the lean kernel has no TMA-store padding
    constraint), builds the ratio causal windows (folding
    ``q_causal_offsets`` when given, in int64 host math), and runs the
    lean kernel once per batch. Raises ``ValueError`` if the config is
    outside the lean specialization (including non-contiguous or
    non-16-byte-aligned inputs) — ``indexer_forward_wrapper`` performs
    this dispatch automatically and falls back to the legacy path instead.
    All tensor arguments must live on one CUDA device, and ``stream``
    (when given) must belong to that device — the same contract as the
    legacy wrapper (CUstream carries no queryable device identity).

    Returns ``{'scores': (B, S_q, S_k) FP32}``; positions outside the
    ratio causal mask ``j < (q_causal_offsets[b] + i + 1) // ratio`` are
    ``-inf``.
    """
    if not math.isfinite(sm_scale):
        raise ValueError(f"sm_scale must be finite, got {sm_scale}")
    api = _maybe_lean_api(q, k, w, ratio, qhead_per_kv_head, sm_scale=sm_scale)
    if api is None:
        raise ValueError(
            "indexer_forward_lean_wrapper: configuration is outside the lean specialization "
            "(shape/dtype/contiguity/alignment/device gate); use indexer_forward_wrapper instead"
        )
    b, s_q = q.shape[0], q.shape[1]
    s_k = k.shape[1]

    current_stream = resolve_stream(stream)
    q_causal_offsets = validate_q_causal_offsets(q_causal_offsets, int(b), q.device, stream=current_stream)
    ks, ke = _ratio_causal_windows(b, s_q, s_k, ratio, q.device, q_causal_offsets, stream)
    with torch_stream_context(stream):
        # -inf pre-fill: tiles the kernel never sweeps keep -inf (same
        # contract as the legacy wrapper's pre-fill)
        out = torch.full((b, s_q, s_k), float("-inf"), dtype=torch.float32, device=q.device)
    with torch.cuda.nvtx.range("indexer_fwd_lean_kernel"):
        # sm_scale is passed through as a per-call runtime argument; the
        # cached api instance is immutable after compile
        api.execute(q, k, w, ks, ke, out, sm_scale=sm_scale, current_stream=current_stream)
    return TupleDict(scores=out)
