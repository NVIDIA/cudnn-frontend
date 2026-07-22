"""APIBase wrapper for the lean H=64/D=128 indexer-forward score kernel.

``IndexerForwardLean`` is an additive fast path next to
:class:`cudnn.deepseek_sparse_attention.indexer_forward.IndexerForward`:
``check_support()`` returns ``True`` only for the configuration the lean
schedule is specialized for (``head_dim == 128``, ``qhead_per_kv_head ==
64``, ``h_kv == 1``, BF16 Q/K with BF16/FP32 W and FP32 scores, and a
q-tile count that keeps the persistent grid saturated). Two input layouts
are served, both on the *same* compiled kernel (the schedule is a pure
per-row-window sweep — it never sees a batch or segment axis):

* **Uniform BSHD** ``(B, S_q, H, D)``: one persistent launch per batch
  entry, uniform ``ks == 0`` ratio-causal windows. This is the path
  ``indexer_forward_wrapper`` dispatches to transparently.

* **THD / varlen (ragged packed)** ``(T_q, H, D)`` + ``cu_seqlens_q/k``:
  one persistent launch over the whole packed problem, with per-row
  *absolute* compressed-KV windows ``ks = cu_seqlens_k[seg]``,
  ``ke = ks + visible`` (see ``_thd_ratio_causal_windows``, the exact
  integer mirror of gtp ``csa_host_index_math``). The windows carry all
  segment isolation *and* ratio-causal masking, so no kernel change is
  needed. THD scores use **global compressed-KV columns**
  ``(T_q, m_total)`` (a query in segment ``b`` has finite scores only in
  its own segment's column block ``[cu_seqlens_k[b], cu_seqlens_k[b+1])``);
  this differs from the legacy kernel's local ``(total_q, max_seqlen_k)``
  layout, so THD is exposed only through the explicit
  ``indexer_forward_lean_wrapper`` — ``indexer_forward_wrapper`` still
  routes every THD call to the (local-column) legacy path, which is never
  modified.
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
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
    ):
        super().__init__()
        self._kernel = IndexerForwardSm100Lean

        # THD/varlen (ragged packed) mode is a purely additive branch: when
        # ``cu_seqlens_q`` is provided the sample tensors are packed
        # ``(T_q, H, D)`` / ``(m_total, H_kv, D)`` / ``(T_q, H)`` /
        # ``(T_q, m_total)`` and the per-row absolute compressed-KV windows
        # carry all segment isolation + ratio-causal masking. When it is
        # None every code path below is byte-identical to the BSHD schedule
        # this PR shipped.
        self._thd = cu_seqlens_q is not None
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k

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
        if self._thd:
            return self._check_support_thd()
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

    def _check_support_thd(self) -> bool:
        """THD / varlen (ragged packed) dispatch gate.

        Additive to :meth:`check_support`; only reachable when
        ``cu_seqlens_q`` was provided. Packed layout:

            Q   (T_q, H_q, D)          W   (T_q, H_q)
            K   (m_total, H_kv, D)     Out (T_q, m_total)

        Malformed inputs (rank/shape mismatches, missing/ill-typed
        cu_seqlens, non-monotonic offsets, or a ``cu_seqlens[-1]`` that
        disagrees with the packed extents) raise ``ValueError`` — the same
        hard-failure surface ``IndexerForward``/``indexer_fwd`` produce for
        malformed THD input. Configurations that are merely outside the lean
        specialization return ``False`` (the explicit wrapper then raises a
        fall-back-to-legacy error; there is no transparent THD dispatch).
        """
        self._logger.debug("Entering _check_support_thd")
        self._value_error_if(
            self.q_desc.ndim != 3,
            f"THD Q must be 3-D (T_q, H_q, D), got {self.q_desc.shape}",
        )
        self._value_error_if(
            self.k_desc.ndim != 3,
            f"THD K must be 3-D (m_total, H_kv, D), got {self.k_desc.shape}",
        )
        self._value_error_if(
            self.w_desc.ndim != 2,
            f"THD W must be 2-D (T_q, H_q), got {self.w_desc.shape}",
        )
        self._value_error_if(
            self.o_desc.ndim != 2,
            f"THD Out must be 2-D (T_q, m_total), got {self.o_desc.shape}",
        )

        t_q, h_q, d = self.q_desc.shape
        m_total, h_kv, d_k = self.k_desc.shape
        t_q_o, m_total_o = self.o_desc.shape
        self._value_error_if(d != d_k, f"Head dim mismatch Q={d} vs K={d_k}")
        self._value_error_if(
            self.w_desc.shape != (t_q, h_q),
            f"THD W must have shape (T_q, H_q) = ({t_q}, {h_q}), got {self.w_desc.shape}",
        )
        self._value_error_if(
            t_q_o != t_q,
            f"THD Out row dim must equal T_q ({t_q}), got {t_q_o}",
        )
        self._value_error_if(
            m_total_o != m_total,
            f"THD Out column dim must equal m_total = K rows ({m_total}), got {m_total_o}",
        )

        # ---- cu_seqlens structural validation (hard: ValueError) ----
        cu_q, cu_k = self.cu_seqlens_q, self.cu_seqlens_k
        self._value_error_if(
            cu_k is None,
            "THD input requires both cu_seqlens_q and cu_seqlens_k",
        )
        for cu, name in ((cu_q, "cu_seqlens_q"), (cu_k, "cu_seqlens_k")):
            self._value_error_if(cu.dtype != torch.int32, f"{name} must be int32, got {cu.dtype}")
            self._value_error_if(cu.ndim != 1, f"{name} must be 1-D, got {cu.ndim}-D")
            self._value_error_if(cu.stride(0) != 1, f"{name} must be contiguous")
            # device checks mirror the bwd THD gate: reject CPU/foreign-device
            # cu_seqlens HERE with a clear message instead of letting
            # _execute_thd fail later on the internally-built ks/ke tensors.
            self._value_error_if(
                cu.device.type not in ("cuda", "meta"),
                f"{name} must be a CUDA tensor, got device {cu.device}",
            )
            self._value_error_if(
                cu.device.type == "cuda" and self.q_desc.device.type == "cuda" and cu.device != self.q_desc.device,
                f"{name} must be on the same CUDA device as Q ({self.q_desc.device}), got {cu.device}",
            )
        self._value_error_if(
            cu_q.shape[0] != cu_k.shape[0],
            f"cu_seqlens_q ({cu_q.shape[0]}) and cu_seqlens_k ({cu_k.shape[0]}) must have the same length",
        )
        self._value_error_if(
            cu_q.shape[0] < 2,
            f"cu_seqlens must have length batch+1 >= 2, got {cu_q.shape[0]}",
        )
        n_seg = cu_q.shape[0] - 1

        # value validation needs the offsets on host (small int32, one D2H
        # per dispatch verdict — THD verdicts are not shape-cached, so a
        # caller that mutates cu_seqlens contents is always re-validated).
        cu_q_host = cu_q.tolist() if cu_q.device.type != "meta" else None
        cu_k_host = cu_k.tolist() if cu_k.device.type != "meta" else None
        if cu_q_host is not None:
            self._value_error_if(cu_q_host[0] != 0, f"cu_seqlens_q must start at 0, got {cu_q_host[0]}")
            self._value_error_if(
                any(cu_q_host[i + 1] < cu_q_host[i] for i in range(n_seg)),
                f"cu_seqlens_q must be non-decreasing, got {cu_q_host}",
            )
            self._value_error_if(
                cu_q_host[-1] != t_q,
                f"cu_seqlens_q[-1] ({cu_q_host[-1]}) must equal packed T_q ({t_q})",
            )
        if cu_k_host is not None:
            self._value_error_if(cu_k_host[0] != 0, f"cu_seqlens_k must start at 0, got {cu_k_host[0]}")
            self._value_error_if(
                any(cu_k_host[i + 1] < cu_k_host[i] for i in range(n_seg)),
                f"cu_seqlens_k must be non-decreasing, got {cu_k_host}",
            )
            self._value_error_if(
                cu_k_host[-1] != m_total,
                f"cu_seqlens_k[-1] ({cu_k_host[-1]}) must equal packed K rows / m_total ({m_total})",
            )

        qhpkv = self.qhead_per_kv_head if self.qhead_per_kv_head is not None else (h_q // h_kv)
        self._value_error_if(
            qhpkv * h_kv != h_q,
            f"qhead_per_kv_head * h_kv != h_q ({qhpkv} * {h_kv} != {h_q})",
        )
        self.qhead_per_kv_head = qhpkv

        # ---- lean-specialization gates (soft: False -> caller falls back) ----
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
        # The static reversed-LPT grid tiles the packed q axis in TQ-token
        # tiles exactly as BSHD does; T_q (not any per-segment length) is the
        # only q-axis divisibility the schedule needs. Segment boundaries
        # that fall mid-tile are still correct (the per-row window masks each
        # of the tile's 4 tokens to its own segment's KV block); they only
        # widen that one tile's union KV sweep, which is bounded by the
        # larger adjacent segment and negligible for realistic packings.
        if t_q % LEAN_TILE_TOKENS != 0:
            return self._unsupported(f"packed T_q must be a multiple of {LEAN_TILE_TOKENS}, got {t_q}")
        if m_total < 1:
            return self._unsupported(f"packed K rows / m_total must be >= 1, got {m_total}")
        for desc, name in (
            (self.q_desc, "Q"),
            (self.k_desc, "K"),
            (self.w_desc, "W"),
            (self.o_desc, "Out"),
        ):
            if not desc.is_contiguous():
                return self._unsupported(f"{name} must be contiguous")

        # device resolution mirrors the BSHD gate (cu_seqlens do not pin a
        # device; the packed operands do).
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

        major, minor = torch.cuda.get_device_capability(target)
        if major != 10:
            return self._unsupported(f"lean schedule requires SM100-class compute capability, found SM{major}.{minor} on {target}")

        num_m_tiles = t_q // LEAN_TILE_TOKENS
        sm_count = torch.cuda.get_device_properties(target).multi_processor_count
        if num_m_tiles < LEAN_MIN_WAVES * sm_count:
            return self._unsupported(
                f"num_m_tiles={num_m_tiles} < LEAN_MIN_WAVES*sm_count="
                f"{LEAN_MIN_WAVES}*{sm_count} — static reversed-LPT grid "
                f"needs >= {LEAN_MIN_WAVES} q-tiles per SM"
            )

        # THD folds the whole packed problem into ONE launch (S_q = T_q,
        # S_k = m_total); the compiled kernel is the shared BSHD schedule.
        self.batch_size = n_seg
        self.s_q = t_q
        self.s_k = m_total
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
                f"device={self.target_device}). This is a lean fast-path bug. For transparently "
                f"dispatched BSHD calls, set CUDNNFE_DSA_INDEXER_FWD_DISABLE_LEAN=1 to force the legacy "
                f"kernel while it is investigated; explicit indexer_forward_lean_wrapper THD calls have "
                f"no legacy fallback (the global-column THD score layout differs from the legacy "
                f"local-column output) — switch such call sites to indexer_forward_wrapper."
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
        if self._thd:
            self._execute_thd(q, k, w, ks, ke, out, scale_value, current_stream)
            return
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

    def _execute_thd(
        self,
        q: torch.Tensor,  # (T_q, H_q, D) BF16, contiguous
        k: torch.Tensor,  # (m_total, 1, D) BF16, contiguous
        w: torch.Tensor,  # (T_q, H_q) BF16/FP32, contiguous
        ks: torch.Tensor,  # (T_q,) INT32 absolute compressed-KV window start
        ke: torch.Tensor,  # (T_q,) INT32 absolute compressed-KV window end (excl.)
        out: torch.Tensor,  # (T_q, m_total) FP32, pre-filled with -inf
        scale_value: float,
        current_stream: Optional[cuda.CUstream],
    ) -> None:
        """Single persistent launch over the whole packed THD problem.

        ``S_q == T_q`` and ``S_k == m_total`` fold the batch away entirely:
        the kernel is a per-row-window sweep, so the packed problem is just
        one big uniform launch on the shared compiled schedule. Segment
        structure lives only in the ``ks``/``ke`` absolute windows built by
        ``_thd_ratio_causal_windows``.
        """
        s_q, s_k, h_q, d = self.s_q, self.s_k, self.h_q, self.head_dim
        for t, name, shape, dtype in (
            (q, "q", (s_q, h_q, d), torch.bfloat16),
            (k, "k", (s_k, self.h_kv, d), torch.bfloat16),
            (w, "w", (s_q, h_q), self.w_desc.dtype),
            (ks, "ks", (s_q,), torch.int32),
            (ke, "ke", (s_q,), torch.int32),
            (out, "out", (s_q, s_k), torch.float32),
        ):
            if tuple(t.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(t.shape)}")
            if t.dtype != dtype:
                raise ValueError(f"{name} must have dtype {dtype}, got {t.dtype}")
            if not t.is_cuda or t.device != q.device:
                raise ValueError(f"{name} must be a CUDA tensor on {q.device}, got {t.device}")
            if not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous, got stride {tuple(t.stride())}")
            if t.data_ptr() % _TMA_MIN_ALIGN_BYTES:
                raise ValueError(f"{name} base pointer must be {_TMA_MIN_ALIGN_BYTES}-byte aligned for TMA, got 0x{t.data_ptr():x}")
        sm_scale_arg = cutlass.Float32(scale_value) if self._kernel_applies_sm_scale else None
        self._compiled_kernel(
            q.view(s_q * h_q, d),
            k.view(s_k, d),
            w.view(s_q, h_q),
            ks,
            ke,
            out.view(s_q, s_k),
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


def _thd_ratio_causal_windows(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    total_q: int,
    ratio: int,
    stream: Optional[cuda.CUstream] = None,
) -> tuple:
    """(T_q,) int32 ABSOLUTE compressed-KV windows for a ragged packed batch.

    Exact integer mirror of gtp ``csa_host_index_math`` (tcp_rank = 0). For
    the packed query row ``i`` in segment ``sid``:

        sid   = searchsorted(cu_seqlens_q[1:], i, right=True) clamped [0, S-1]
        off   = i - cu_seqlens_q[sid]                 # local position
        m_seg = cu_seqlens_k[sid+1] - cu_seqlens_k[sid]  # compressed KV len
        vis   = min((off + 1) // ratio, m_seg)        # ratio-causal visible
        ks[i] = cu_seqlens_k[sid]                      # absolute segment base
        ke[i] = ks[i] + vis                            # absolute end (excl.)

    ``ks``/``ke`` are absolute columns into the packed compressed-KV buffer,
    so they encode segment isolation AND the ratio causal mask in one shot:
    a row in segment ``b`` can only produce finite scores in columns
    ``[cu_seqlens_k[b], cu_seqlens_k[b+1])`` and only up to its visible
    prefix. ``total_q`` is passed in (== packed ``q.shape[0]``) so the arange
    extent needs no device sync.
    """
    device = cu_seqlens_q.device
    with torch_stream_context(stream):
        cu_q = cu_seqlens_q.to(torch.int64)
        cu_k = cu_seqlens_k.to(torch.int64)
        n_seg = cu_q.shape[0] - 1
        i = torch.arange(total_q, device=device, dtype=torch.int64)
        sid = torch.searchsorted(cu_q[1:], i, right=True).clamp_(max=n_seg - 1)
        off = i - cu_q[sid]
        m_seg = cu_k[sid + 1] - cu_k[sid]
        vis = torch.minimum((off + 1) // int(ratio), m_seg)
        ks = cu_k[sid].to(torch.int32).contiguous()
        ke = (cu_k[sid] + vis).to(torch.int32).contiguous()
    return ks, ke


def _maybe_lean_api_thd(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    ratio: int,
    qhead_per_kv_head: Optional[int],
    sm_scale: float = 1.0,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
) -> Optional[IndexerForwardLean]:
    """Compiled ``IndexerForwardLean`` for a packed THD problem, or ``None``.

    Cheap non-throwing eligibility (packed rank, device, dtype, contiguity,
    TMA alignment, finite scale, ratio) runs first; anything it rejects
    returns ``None`` (the wrapper then raises its fall-back-to-legacy error).
    THD verdicts are deliberately NOT metadata-cached: the accept/reject
    decision depends on the cu_seqlens *contents* (monotonicity, extent
    consistency), so ``check_support`` re-validates them every call. The
    expensive JIT is still shared through ``_lean_compile_cache`` (the codegen
    key is shape-only: T_q, m_total, W dtype, scale variant, device).
    """
    if q.ndim != 3 or k.ndim != 3 or w.ndim != 2:
        return None
    if not (q.is_cuda and k.is_cuda and w.is_cuda) or not (q.device == k.device == w.device):
        return None
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or w.dtype not in (torch.bfloat16, torch.float32):
        return None
    t_q, h_q, d = q.shape
    if tuple(w.shape) != (t_q, h_q):
        return None
    if not (q.is_contiguous() and k.is_contiguous() and w.is_contiguous()):
        return None
    if (q.data_ptr() % _TMA_MIN_ALIGN_BYTES) or (k.data_ptr() % _TMA_MIN_ALIGN_BYTES) or (w.data_ptr() % _TMA_MIN_ALIGN_BYTES):
        return None
    if not math.isfinite(sm_scale):
        return None
    if int(ratio) < 1:
        return None

    m_total = int(k.shape[0])
    with _lean_build_lock:
        api = IndexerForwardLean(
            sample_q=q,
            sample_k=k,
            sample_w=w,
            sample_out=torch.empty(t_q, m_total, dtype=torch.float32, device="meta"),
            ratio=ratio,
            qhead_per_kv_head=qhead_per_kv_head,
            sm_scale=sm_scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )
        if api.check_support():
            api.compile()
            return api
        return None


def _indexer_forward_lean_thd(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    ratio: int,
    qhead_per_kv_head: Optional[int],
    sm_scale: float,
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    q_causal_offsets: Optional[torch.Tensor],
    stream: Optional[cuda.CUstream],
) -> TupleDict:
    """Explicit THD (ragged packed) lean path — GLOBAL compressed-KV columns.

    ``q`` ``(T_q, H, D)`` / ``k`` ``(m_total, H_kv, D)`` / ``w`` ``(T_q, H)``;
    returns ``{'scores': (T_q, m_total) FP32}`` where row ``i`` (in segment
    ``b``) is ``-inf`` everywhere except its own segment's compressed-KV
    column block, ratio-causally masked. ``q_causal_offsets`` is not folded
    on this path yet (documented gate); pass such shapes to
    ``indexer_forward_wrapper`` (legacy) instead.
    """
    if q_causal_offsets is not None:
        raise ValueError(
            "indexer_forward_lean_wrapper: q_causal_offsets is not supported on the THD " "lean path; use indexer_forward_wrapper (legacy) for offset varlen"
        )
    if cu_seqlens_q is None or cu_seqlens_k is None:
        raise ValueError("THD input requires both cu_seqlens_q and cu_seqlens_k")
    api = _maybe_lean_api_thd(
        q,
        k,
        w,
        cu_seqlens_q,
        cu_seqlens_k,
        ratio,
        qhead_per_kv_head,
        sm_scale=sm_scale,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    if api is None:
        raise ValueError(
            "indexer_forward_lean_wrapper: THD configuration is outside the lean specialization "
            "(rank/dtype/contiguity/alignment/H=64-D=128-MQA/saturated-grid gate); "
            "use indexer_forward_wrapper instead"
        )
    total_q = int(q.shape[0])
    m_total = int(k.shape[0])
    current_stream = resolve_stream(stream)
    ks, ke = _thd_ratio_causal_windows(cu_seqlens_q, cu_seqlens_k, total_q, ratio, stream)
    with torch_stream_context(stream):
        out = torch.full((total_q, m_total), float("-inf"), dtype=torch.float32, device=q.device)
    with torch.cuda.nvtx.range("indexer_fwd_lean_thd_kernel"):
        api.execute(q, k, w, ks, ke, out, sm_scale=sm_scale, current_stream=current_stream)
    return TupleDict(scores=out)


def indexer_forward_lean_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    ratio: int = 4,
    qhead_per_kv_head: Optional[int] = None,
    sm_scale: float = 1.0,
    q_causal_offsets: Optional[torch.Tensor] = None,
    stream: Optional[cuda.CUstream] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
) -> TupleDict:
    """High-level wrapper for the lean H=64/D=128 path (BSHD or THD).

    **BSHD** ``q (B, S_q, H, D)`` / ``k (B, S_k, H_kv, D)`` / ``w (B, S_q, H)``
    (``cu_seqlens_*`` omitted): allocates and ``-inf``-pre-fills the
    ``(B, S_q, S_k)`` FP32 score buffer (always contiguous — the lean kernel
    has no TMA-store padding constraint), builds the ratio causal windows
    (folding ``q_causal_offsets`` when given, in int64 host math), and runs
    the lean kernel once per batch. Returns ``{'scores': (B, S_q, S_k)}``;
    positions outside the mask ``j < (q_causal_offsets[b] + i + 1) // ratio``
    are ``-inf``.

    **THD / varlen** ``q (T_q, H, D)`` / ``k (m_total, H_kv, D)`` /
    ``w (T_q, H)`` with ``cu_seqlens_q`` and ``cu_seqlens_k`` (both int32,
    ``batch+1``): one packed launch with per-row absolute compressed-KV
    windows. Returns ``{'scores': (T_q, m_total)}`` in GLOBAL compressed-KV
    columns — row ``i`` (segment ``b``) is ``-inf`` outside its own block
    ``[cu_seqlens_k[b], cu_seqlens_k[b+1])`` and beyond its ratio-causal
    visible prefix. This global-column THD layout differs from the legacy
    ``(total_q, max_seqlen_k)`` local-column output, so ``q_causal_offsets``
    is unsupported here and ``indexer_forward_wrapper`` never auto-routes THD
    to this path (it keeps every THD call on the legacy kernel).

    Raises ``ValueError`` if the config is outside the lean specialization
    (including non-contiguous or non-16-byte-aligned inputs). All tensor
    arguments must live on one CUDA device, and ``stream`` (when given) must
    belong to that device — the same contract as the legacy wrapper
    (CUstream carries no queryable device identity).
    """
    if not math.isfinite(sm_scale):
        raise ValueError(f"sm_scale must be finite, got {sm_scale}")
    if cu_seqlens_q is not None or cu_seqlens_k is not None:
        return _indexer_forward_lean_thd(
            q,
            k,
            w,
            cu_seqlens_q,
            cu_seqlens_k,
            ratio=ratio,
            qhead_per_kv_head=qhead_per_kv_head,
            sm_scale=sm_scale,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            q_causal_offsets=q_causal_offsets,
            stream=stream,
        )
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
