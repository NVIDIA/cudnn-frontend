# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Adapter of the Rubin (SM107) index-list SPARSE d256 forward -- ``kernels/sm107/sparse_d256_f16.py``.

Frontend-only: the gated attention block's sparse stage consumes it; no graph form carries a block-index list, so there is
NO engine row and NO manifest slot (an ``EngineSpec`` whose ``lower`` cannot run is a contract break).  The kernel's claims
live in ONE frozen record, :data:`SPARSE_CAPABILITIES`, spelled in the ``Capabilities`` vocabulary where a field exists, and
:meth:`SparseGqaFwdDslSm107.check_support` is the ENFORCEMENT point: the CuTe DSL version gate first (``sm_107a`` needs the
public 4.8.0 wheel; ``python/cudnn/AGENTS.md`` Rule 7 -- BEFORE the kernel module is imported, so a too-old DSL reads as a
version problem, never as a ``KeyError`` from inside the DSL), then every field of the record as a typed decline.  Every
arm the body does not carry yet (THD, paged pools, the fused epilogue gate, split-KV, bottom-right, a sink, a band, the
decode form, Q-length trimming, log2 stats, a pure caller list) is declined BY NAME; a later commit that lands an arm flips
the record field, the config's wired-arm set and the support-matrix tracker in the same change.

The device does the work (Rule 3): lengths, counts, dead items and the tail block are derived on device from the position
and the per-row ``block_lens``; nothing below reads device memory.  ``execute`` validates and launches (Rule 1): no
conversion, no allocation, the caller's stream.

Stride contract (the gather map is TWO-dimensional: the tokens of EVERY batch are its rows, the token's full row its
columns): the head dim contiguous; the head stride a multiple of 64 elements (a column box start); the token stride a
multiple of 8 elements (TMA's 16-byte rule); the K / V batch stride EQUAL to ``S_kv x token stride`` (or B == 1) -- a padded
batch stride has no 2-D row coordinate; Q / O any dense (B, S, H) strides.  ``block_ids``: int32 ``[B, S_q, top_k]`` (or
``[B x S_q, top_k]``) CONTIGUOUS -- the row stride is the bulk-copy length; ``block_lens``: int32 ``[B, S_q]`` / ``[B x S_q]``
contiguous or None; ``seq_kv_lens``: int32 ``[B]`` contiguous or None (its presence is a compile-time specialization).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cudnn

KERNEL_FILE = "sm107/sparse_d256_f16.py"
_TOPK_MIN, _TOPK_MAX = 4, 512
_BLOCK_SIZE = 4
_D = 256
_GQA_GROUP_MAX = 16
_INT32_MAX = 2**31 - 1


@dataclass(frozen=True)
class SparseCapabilities:
    """What the sparse core serves -- the adapter-owned claims record (no engine row: see the module docstring).  Field names
    follow ``engines.Capabilities`` where the meaning matches; the index-list fields are this record's own."""

    sm_lo: int = 107
    sm_hi: int = 119
    phase: str = "forward"
    d_shapes: frozenset = frozenset({(_D, _D)})
    d_pad_multiple: int = 0  # exact native shape only: the gather box count and the Q box are derived for d = 256
    dtypes: frozenset = frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16})
    kv_cache_dtypes: frozenset = frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16})
    causal: bool = True  # the in-block mask: key <= the token's position (top-left)
    bottom_right: bool = False
    swa: bool = False
    sink: bool = False
    padded: bool = False  # per-batch Q lengths (the dense padding mask) are not carried ...
    kv_lens: bool = True  # ... per-batch KV lengths are (seq_kv_lens: the tail follows the visible range, a 0 length is a dead row)
    thd: bool = False
    paged_kv: bool = False
    decode: bool = False
    epilogue_gate: bool = False
    split_kv: bool = False
    stats_log2: bool = False
    pack_gqa: bool = True  # the work item IS the token's packed query-head group
    index_block_sizes: frozenset = frozenset({_BLOCK_SIZE})
    index_top_k_min: int = _TOPK_MIN
    index_top_k_max: int = _TOPK_MAX  # any multiple of 4 in [4, 512] is served
    index_gqa_group_min: int = 1
    index_gqa_group_max: int = _GQA_GROUP_MAX
    open_block_appended: bool = True  # the kernel appends the open tail block from the position (not a public switch)
    cutedsl_min_version: Tuple[int, int, int] = (4, 8, 0)


SPARSE_CAPABILITIES = SparseCapabilities()

_DTYPE_CODE = {"torch.bfloat16": 2, "torch.float16": 3}
_DTYPE_PUBLIC = {"torch.bfloat16": cudnn.data_type.BFLOAT16, "torch.float16": cudnn.data_type.HALF}


def _shape(t) -> Tuple[int, ...]:
    return tuple(int(x) for x in t.shape)


def _strides(t) -> Tuple[int, ...]:
    return tuple(int(x) for x in t.stride())


class SparseGqaFwdDslSm107:
    """``O, LSE = sparse_sdpa(Q, K, V, block_ids[, block_lens][, seq_kv_lens])`` on cc 10.7 -- the standalone adapter.

    Construct with the operands (framework tensors: ``.shape`` / ``.stride()`` / ``.dtype`` / ``.data_ptr()`` /
    ``.device``), call :meth:`check_support` (typed declines), :meth:`compile` (one compiled artifact per specialization),
    :meth:`execute` (launch on the caller's stream).  The kernel module is imported ONLY after the DSL gate passed.
    """

    def __init__(
        self,
        *,
        q,
        k,
        v,
        o,
        block_ids,
        lse=None,
        block_lens=None,
        seq_kv_lens=None,
        top_k: int,
        block_size: int = _BLOCK_SIZE,
        scale: Optional[float] = None,
        stats_log2: bool = False,
        thd: bool = False,
        paged_kv: bool = False,
        epilogue_gate=None,
        split_kv: int = 1,
        bottom_right: bool = False,
        sink=None,
        window_left: Optional[int] = None,
        window_right: Optional[int] = None,
        seq_q_lens=None,
        list_per_sequence: bool = False,
        include_open_block: bool = True,
        device_cc: Optional[Tuple[int, int]] = None,
    ):
        self.q, self.k, self.v, self.o, self.lse = q, k, v, o, lse
        self.block_ids, self.block_lens, self.seq_kv_lens = block_ids, block_lens, seq_kv_lens
        self.top_k, self.block_size = int(top_k), int(block_size)
        self.scale = (1.0 / math.sqrt(_D)) if scale is None else float(scale)
        self.stats_log2 = bool(stats_log2)
        self.unserved = dict(
            thd=bool(thd),
            paged_kv=bool(paged_kv),
            epilogue_gate=epilogue_gate is not None,
            split_kv=int(split_kv) > 1,
            bottom_right=bool(bottom_right),
            sink=sink is not None,
            sliding_window=window_left is not None or window_right is not None,
            seq_q_lens=seq_q_lens is not None,
            list_per_sequence=bool(list_per_sequence),
            pure_list=not include_open_block,
            stats_log2=self.stats_log2,
        )
        self.device_cc = tuple(device_cc) if device_cc is not None else None
        self._fn = None
        self._module = None
        self._B = self._SQ = self._SKV = self._H = self._KH = self._G = 0
        self._dtype_code = 0

    # --- the claims ------------------------------------------------------------------------------------------------------

    @staticmethod
    def claims() -> SparseCapabilities:
        return SPARSE_CAPABILITIES

    # --- the enforcement point -------------------------------------------------------------------------------------------

    def _device_cc(self) -> Tuple[int, int]:
        if self.device_cc is not None:
            return self.device_cc
        import torch  # the framework is the caller's; imported here, never at module import

        return tuple(torch.cuda.get_device_capability(self.q.device))

    def check_support(self) -> bool:
        """Typed declines in order: the DSL gate (Rule 7, before any kernel import), the arch, the unserved arms by name, the
        dtypes, the shapes, the stride contract, the index list.  Returns True when the record serves the request."""
        from cudnn.frost.tile_dsl.tma import tma_gather4_requirement_error

        cc = self._device_cc()
        msg = tma_gather4_requirement_error(cc)
        if msg is not None:
            raise NotImplementedError(f"sparse d256 forward (sm107): {msg}")
        cc_code = cc[0] * 10 + cc[1]
        if not (SPARSE_CAPABILITIES.sm_lo <= cc_code <= SPARSE_CAPABILITIES.sm_hi):
            raise NotImplementedError(f"sparse d256 forward (sm107): needs a Rubin-line GPU (cc 10.7 .. 11.9); got cc {cc[0]}.{cc[1]}")
        for name, requested in self.unserved.items():
            if requested:
                raise NotImplementedError(f"sparse d256 forward (sm107): {name} is not served by the kernel body yet (the record declines it by name)")

        # the index list's FORM (the kernel's staging is sized by it)
        if self.block_size != _BLOCK_SIZE:
            raise NotImplementedError(
                f"sparse d256 forward (sm107): block_size must be {_BLOCK_SIZE} (one gather transaction = four rows); got {self.block_size}"
            )
        if not (_TOPK_MIN <= self.top_k <= _TOPK_MAX) or self.top_k % 4 != 0:
            raise NotImplementedError(f"sparse d256 forward (sm107): top_k must be a multiple of 4 in [{_TOPK_MIN}, {_TOPK_MAX}]; got {self.top_k}")

        # dtypes: one half dtype for Q / K / V / O, fp32 LSE
        dt = str(self.q.dtype)
        if dt not in _DTYPE_CODE:
            raise NotImplementedError(f"sparse d256 forward (sm107): Q must be bf16 or f16; got {dt}")
        for name, t in (("K", self.k), ("V", self.v), ("O", self.o)):
            if str(t.dtype) != dt:
                raise ValueError(f"sparse d256 forward (sm107): {name} dtype {t.dtype} must equal Q's {dt}")
        if self.lse is not None and str(self.lse.dtype) != "torch.float32":
            raise ValueError(f"sparse d256 forward (sm107): LSE must be float32; got {self.lse.dtype}")

        # shapes: (B, S, H, 256) BSHD; one KV head serves H / KH query heads, 1 <= group <= 16
        qs, ks, vs, os_ = _shape(self.q), _shape(self.k), _shape(self.v), _shape(self.o)
        for name, s in (("Q", qs), ("K", ks), ("V", vs), ("O", os_)):
            if len(s) != 4:
                raise ValueError(f"sparse d256 forward (sm107): {name} must be rank-4 (B, S, H, D); got {s}")
            if s[3] != _D:
                raise NotImplementedError(f"sparse d256 forward (sm107): d_qk = d_v = {_D} exactly (no envelope); got {name} d = {s[3]}")
        B, SQ, H, _ = qs
        Bk, SKV, KH, _ = ks
        if vs != ks:
            raise ValueError(f"sparse d256 forward (sm107): V shape {vs} must equal K shape {ks}")
        if Bk != B or os_ != qs:
            raise ValueError(f"sparse d256 forward (sm107): batch / O shape mismatch: Q {qs}, K {ks}, O {os_}")
        if H % KH != 0:
            raise ValueError(f"sparse d256 forward (sm107): H_q = {H} must be a multiple of H_kv = {KH}")
        G = H // KH
        if not (SPARSE_CAPABILITIES.index_gqa_group_min <= G <= SPARSE_CAPABILITIES.index_gqa_group_max):
            raise NotImplementedError(
                f"sparse d256 forward (sm107): the query-head group H_q / H_kv = {G} must be in [1, {_GQA_GROUP_MAX}] (one 16-row tile per token)"
            )
        if self.lse is not None and _shape(self.lse) != (B, H, SQ):
            raise ValueError(f"sparse d256 forward (sm107): LSE must be (B, H_q, S_q) = {(B, H, SQ)}; got {_shape(self.lse)}")
        if B * SKV > _INT32_MAX or B * SQ * self.top_k > _INT32_MAX or (KH - 1) * _strides(self.k)[2] + _D > _INT32_MAX:
            raise NotImplementedError("sparse d256 forward (sm107): an extent exceeds the Int32 coordinate range")

        # strides
        for name, t in (("Q", self.q), ("K", self.k), ("V", self.v), ("O", self.o)):
            st = _strides(t)
            if st[3] != 1:
                raise ValueError(f"sparse d256 forward (sm107): {name} head dim must be contiguous (stride 1); got strides {st}")
        for name, t in (("K", self.k), ("V", self.v)):
            bs, ss, hs, _ = _strides(t)
            if hs % 64 != 0:
                raise ValueError(f"sparse d256 forward (sm107): {name} head stride must be a multiple of 64 elements (a 128-B column box start); got {hs}")
            if ss % 8 != 0:
                raise ValueError(f"sparse d256 forward (sm107): {name} token stride must be a multiple of 8 elements (TMA's 16-byte rule); got {ss}")
            if B > 1 and bs != SKV * ss:
                raise ValueError(
                    f"sparse d256 forward (sm107): {name} batch stride must be S_kv x the token stride ({SKV} x {ss}) for the 2-D gather map; got {bs}"
                )

        # the index list's tensors
        ids = self.block_ids
        if str(ids.dtype) != "torch.int32":
            raise ValueError(f"sparse d256 forward (sm107): block_ids must be int32; got {ids.dtype}")
        if _shape(ids) not in ((B, SQ, self.top_k), (B * SQ, self.top_k)):
            raise ValueError(f"sparse d256 forward (sm107): block_ids must be [B, S_q, top_k] = {(B, SQ, self.top_k)} or [B x S_q, top_k]; got {_shape(ids)}")
        if not ids.is_contiguous():
            raise ValueError("sparse d256 forward (sm107): block_ids must be contiguous (its row stride is the bulk-copy length)")
        if self.block_lens is not None:
            bl = self.block_lens
            if str(bl.dtype) != "torch.int32" or _shape(bl) not in ((B, SQ), (B * SQ,)) or not bl.is_contiguous():
                raise ValueError(f"sparse d256 forward (sm107): block_lens must be a contiguous int32 [B, S_q] / [B x S_q]; got {bl.dtype} {_shape(bl)}")
        if self.seq_kv_lens is not None:
            kl = self.seq_kv_lens
            if str(kl.dtype) != "torch.int32" or _shape(kl) != (B,) or not kl.is_contiguous():
                raise ValueError(f"sparse d256 forward (sm107): seq_kv_lens must be a contiguous int32 [B]; got {kl.dtype} {_shape(kl)}")
        self._B, self._SQ, self._SKV, self._H, self._KH, self._G = B, SQ, SKV, H, KH, G
        self._dtype_code = _DTYPE_CODE[dt]
        return True

    # --- compile / execute -----------------------------------------------------------------------------------------------

    def template_params(self):
        from cudnn.sdpa.fwd.config_sm100 import TemplateParams

        if not self._G:
            self.check_support()
        return TemplateParams(
            dtype_qkv=self._dtype_code,
            cta_mma=1,
            pack_gqa=True,
            qh_per_kh=self._G,
            qsa_block_topk=self.top_k,
            qsa_block_size=self.block_size,
            seq_kv_lens_present=self.seq_kv_lens is not None,
            stats_log2=self.stats_log2,
        )

    def compile(self):
        """Load the template for this specialization and compile its pointer ABI (one artifact per (dtype, group, top_k,
        KV-lens presence, LSE presence, block_lens presence))."""
        if self._fn is not None:
            return self._fn
        self.check_support()
        from cudnn.frost.template_loader import load_template

        params = self.template_params()
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", KERNEL_FILE)
        self._module = load_template(path, params, tag="sdpa_fwd_sm107_sparse_d256")
        self._fn = self._module.compile(has_lse=self.lse is not None, has_block_lens=self.block_lens is not None)
        return self._fn

    def execute(self, stream=None):
        """Launch on ``stream`` (default: the framework's current stream).  Validation only -- no conversion, no allocation."""
        fn = self.compile()
        import torch
        import cutlass
        import cuda.bindings.driver as cuda_driver
        from cutlass.cute.runtime import make_ptr
        import cutlass.cute as cute

        gmem = cute.AddressSpace.gmem
        half = cutlass.BFloat16 if self._dtype_code == 2 else cutlass.Float16

        def P(t, dtype, align=16):
            return None if t is None else make_ptr(dtype, t.data_ptr(), gmem, assumed_align=align)

        if stream is None:
            stream = cuda_driver.CUstream(torch.cuda.current_stream(self.q.device).cuda_stream)
        q, k, v, o, lse = self.q, self.k, self.v, self.o, self.lse
        seq_kv_lens = self.seq_kv_lens
        if seq_kv_lens is None:
            # Unread by the kernel (SEQ_KV_LENS_PRESENT = 0): the pointer slot is bound to the ids (a valid address).
            seq_kv_lens = self.block_ids
        fn(
            q_ptr=P(q, half),
            k_ptr=P(k, half),
            v_ptr=P(v, half),
            o_ptr=P(o, half),
            lse_ptr=P(lse, cutlass.Float32, 4),
            block_ids_ptr=P(self.block_ids, cutlass.Int32, 16),
            block_lens_ptr=P(self.block_lens, cutlass.Int32, 4),
            seq_kv_lens_ptr=P(seq_kv_lens, cutlass.Int32, 4),
            problem_size=(self._B, self._H, self._KH, self._SQ, self._SKV),
            q_strides=_strides(q)[:3],
            k_strides=_strides(k)[:3],
            v_strides=_strides(v)[:3],
            o_strides=_strides(o)[:3],
            lse_strides=_strides(lse)[:3] if lse is not None else (0, 0, 0),
            scale_softmax_log2=cutlass.Float32(self.scale * math.log2(math.e)),
            stream=stream,
        )
