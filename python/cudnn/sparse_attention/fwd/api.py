# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""APIBase wrapper for the generic sparse-attention forward contract.

Softmax attention restricted to a per-query selected KV subset, covering the
index-driven sparse-attention family — token-level top-k (DeepSeek DSA),
block-level top-k (MiniMax MSA, NSA selection), and micro-block selections
(Qwen QSA) — through one signature. It is the forward counterpart of
:func:`cudnn.deepseek_sparse_attention.sparse_attention_backward_wrapper`
(whose KV-only LSE contract this op produces) and is intended as the
variant-neutral home for index-driven sparse attention, alongside the
mask-driven :mod:`cudnn.block_sparse_attention`.

Contract summary (normative — kernels must match the reference exactly):

* **Index space is storage-native.** ``topk_idxs`` entries index the K/V
  tensors as the kernel receives them: packed THD -> global flat ids in
  ``[0, T_kv)``; BSHD -> within-sequence ids in ``[0, S_kv)`` (the batch
  coordinate comes from the query row). ``-1`` marks an invalid slot.
  Ids must be unique within a row: kernels gather without deduplication, so
  a duplicated id would count its tokens twice (real top-k never emits
  duplicates).
* **``index_granularity``** is the number of consecutive tokens covered by
  one index entry. Entry value ``i`` selects tokens
  ``[i * g, i * g + g)``; the tail entry is clamped to the KV bound.
* **Index scope ``G``.** ``topk_idxs`` is either ``(T_q, topk)`` (one set
  shared by every query head) or ``(T_q, G, topk)`` with
  ``G in {H_kv, H_q}`` (per KV-head group / per query head).
* **LSE is KV-only, base-e, FP32.** ``lse[t, h] = log(sum_j exp(s_j))``
  over the valid selected entries only; ``attn_sink`` participates in the
  softmax denominator but never in the LSE. Rows with no valid entry
  produce ``lse = -inf`` and ``out = 0``.
* **Deterministic always**: same inputs produce bitwise-identical outputs.

Tensor inputs are framework tensors (anything dlpack-compatible — torch,
JAX, numpy): validation runs on canonical descriptors (cutlass dtypes,
adapter devices) and never requires torch. Each registered kernel declares
which frameworks it can execute and fails loudly otherwise.

Execution dispatches to registered device kernels, when the corresponding
module is present in the tree:

* the SM100 DSA sparse-prefill kernel for its envelope (THD, MQA latent
  with K aliased as V, ``D_k in (512, 576)``, shared token-granularity
  indices, FP16/BF16);
* the SM100 GQA substrate kernel for its envelope (``G == H_kv``, block
  granularity in ``(4, 64, 128)`` -- QSA/MSA shapes, BF16 or FP8-per-tensor,
  K/V unaliased). The substrate module's own dispatcher
  (``sparse_attention.fwd.sm100_gqa.dispatch``) has a correctness-safe,
  D2H-validated mechanism to additionally try a tile-batched fast-path
  kernel ahead of its scalar per-row mainloop for ``index_granularity ==
  128``, but this API does not reach it: that fast path measured 1.13x-1.28x
  *slower* than the scalar kernel on every shape tried this round (see
  ``dispatch.py``'s module docstring for the numbers), so it stays off by
  default rather than regressing callers of this wrapper.

Configurations no kernel serves raise ``NotImplementedError`` from
``check_support``. The normative reference
implementation lives with the tests
(``test/python/sparse_attention/sparse_attention_reference.py``), which is
also the oracle every kernel is validated against.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional

import cutlass

from cudnn.api_base import APIBase, TupleDict
from cudnn.tensor_adapter import detect_framework, get_compute_capability, get_data_ptr

# Framework neutrality: this module must be importable without torch (JAX
# processes may not import torch at all), and the public API takes framework
# tensors (anything dlpack-compatible: torch, JAX, numpy) — validation runs on
# canonical descriptors (cutlass dtypes, adapter devices), never on torch
# types. torch appears only function-locally inside torch-only *backends*.
if TYPE_CHECKING:  # pragma: no cover - typing only
    import cuda.bindings.driver as cuda


def _get_dsa_prefill_kernel():
    """Probe for the SM100 DSA sparse-prefill kernel (ships on its own branch).

    Returns the wrapper or ``None``. The generic op registers it as the
    dispatch target for its envelope when present, so this module stays
    landable (and raising ``NotImplementedError``) on trees without the
    kernel.
    """
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_forward import (
            sparse_attention_forward_wrapper as dsa_fwd,
        )
    except ImportError:
        return None
    return dsa_fwd


def _get_gqa_substrate_kernel():
    """Probe for the SM100 GQA substrate kernel (ships on its own branch).

    Serves the ``G = H_kv``, granularity 4/64/128 envelope (QSA / MSA block
    shapes) — structurally a block-sparse gather driven by ``topk_idxs``
    rather than a static mask. Returns the wrapper or ``None`` so this
    module stays import-safe (and the generic op falls through to
    ``NotImplementedError``) on trees without the kernel module.
    """
    try:
        from cudnn.sparse_attention.fwd.sm100_gqa import (
            sparse_attention_forward_wrapper as gqa_fwd,
        )
    except ImportError:
        return None
    return gqa_fwd


class SparseAttentionForward(APIBase):
    """Forward sparse attention over a per-query selected KV subset.

    Layouts (both K and V follow Q's layout):

    * THD (packed varlen): ``q (T_q, H_q, D_k)``, ``k (T_kv, H_kv, D_k)``,
      ``v (T_kv, H_kv, D_v)`` with ``cu_seqlens_q`` mapping query rows to
      sequences. ``topk_idxs`` holds global flat KV ids.
    * BSHD: ``q (B, S_q, H_q, D_k)``, ``k (B, S_kv, H_kv, D_k)``,
      ``v (B, S_kv, H_kv, D_v)``. ``topk_idxs`` holds within-sequence ids
      and leads with the same ``(B, S_q)`` coordinates.

    ``k`` and ``v`` may alias the same storage (MLA-style latents). When
    ``D_k == 576`` and ``D_v == 512`` with aliased storage, ``v`` is the
    leading-512 view of ``k`` — pass it as such; the API does not slice
    implicitly.
    """

    def __init__(
        self,
        sample_q: Any,  # framework tensor (dlpack: torch / JAX / numpy)
        sample_k: Any,
        sample_v: Any,
        sample_topk_idxs: Any,
        sample_topk_length: Optional[Any] = None,
        sample_attn_sink: Optional[Any] = None,
        sample_cu_seqlens_q: Optional[Any] = None,
        index_granularity: int = 1,
        softmax_scale: Optional[float] = None,
    ):
        super().__init__()

        self.q_desc = self._make_tensor_desc(sample_q, name="sample_q", canonical=True)
        self.k_desc = self._make_tensor_desc(sample_k, name="sample_k", canonical=True)
        self.v_desc = self._make_tensor_desc(sample_v, name="sample_v", canonical=True)
        self.topk_idxs_desc = self._make_tensor_desc(sample_topk_idxs, name="sample_topk_idxs", canonical=True)
        self.topk_length_desc = self._make_tensor_desc(sample_topk_length, name="sample_topk_length", canonical=True)
        self.attn_sink_desc = self._make_tensor_desc(sample_attn_sink, name="sample_attn_sink", canonical=True)
        self.cu_seqlens_q_desc = self._make_tensor_desc(sample_cu_seqlens_q, name="sample_cu_seqlens_q", canonical=True)

        self.index_granularity = int(index_granularity)
        self.softmax_scale = softmax_scale

        # Derived in check_support().
        self.is_thd = None
        self.group_scope = None  # G: 1, H_kv, or H_q
        self._dispatch = None  # registered device-kernel wrapper (set in check_support)
        self._dispatch_envelope = None  # "dsa" | "gqa" | None (set in check_support)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def check_support(self) -> bool:
        major, _ = get_compute_capability()
        self._runtime_error_if(
            major < 9,
            f"SparseAttentionForward requires SM90+, found SM{major}",
        )

        q, k, v = self.q_desc, self.k_desc, self.v_desc
        idxs = self.topk_idxs_desc

        # ---- layout ----
        self._value_error_if(
            q.ndim not in (3, 4),
            f"Q must be (T_q, H_q, D_k) or (B, S_q, H_q, D_k), got {q.shape}",
        )
        self.is_thd = q.ndim == 3
        self._value_error_if(
            self.is_thd and self.cu_seqlens_q_desc is None,
            "THD Q (3-D) requires cu_seqlens_q",
        )
        self._value_error_if(
            not self.is_thd and self.cu_seqlens_q_desc is not None,
            "cu_seqlens_q is only valid with packed THD (3-D) Q",
        )
        expected_kv_ndim = 3 if self.is_thd else 4
        self._value_error_if(
            k.ndim != expected_kv_ndim or v.ndim != expected_kv_ndim,
            f"K and V must follow Q's layout ({expected_kv_ndim}-D), got K {k.shape}, V {v.shape}",
        )

        # ---- dtypes (canonical descriptors carry cutlass dtypes) ----
        # FP16/BF16 for the DSA envelope, BF16/FP8-per-tensor for the GQA
        # substrate envelope; envelope predicates below re-check the
        # dtype-per-envelope split, this just admits the union.
        self._check_dtype(q, [cutlass.Float16, cutlass.BFloat16, cutlass.Float8E4M3FN], name="Q")
        self._check_dtype(k, q.dtype, name="K", extra_error_msg="K must have same dtype as Q")
        self._check_dtype(v, q.dtype, name="V", extra_error_msg="V must have same dtype as Q")
        self._check_dtype(idxs, cutlass.Int32, name="topk_idxs")
        if self.topk_length_desc is not None:
            self._check_dtype(self.topk_length_desc, cutlass.Int32, name="topk_length")
        if self.attn_sink_desc is not None:
            self._check_dtype(self.attn_sink_desc, cutlass.Float32, name="attn_sink")
        if self.cu_seqlens_q_desc is not None:
            self._check_dtype(self.cu_seqlens_q_desc, cutlass.Int32, name="cu_seqlens_q")

        # ---- shapes ----
        if self.is_thd:
            t_q, h_q, d_k = q.shape
            t_kv, h_kv, d_k_kv = k.shape
            _, h_kv_v, d_v = v.shape
            lead = (t_q,)
        else:
            b, s_q, h_q, d_k = q.shape
            b_k, s_kv, h_kv, d_k_kv = k.shape
            b_v, _, h_kv_v, d_v = v.shape
            self._value_error_if(
                b_k != b or b_v != b,
                f"K/V batch must match Q batch {b}, got K {k.shape}, V {v.shape}",
            )
            lead = (b, s_q)
        self._value_error_if(
            d_k_kv != d_k,
            f"K head dim must match Q head dim {d_k}, got {d_k_kv}",
        )
        self._value_error_if(
            h_kv_v != h_kv,
            f"V KV-head count must match K ({h_kv}), got {h_kv_v}",
        )
        self._value_error_if(
            k.shape[0 if self.is_thd else 1] != v.shape[0 if self.is_thd else 1],
            f"K and V must share the KV token count, got K {k.shape}, V {v.shape}",
        )
        self._value_error_if(
            h_q % h_kv != 0,
            f"H_q ({h_q}) must be a multiple of H_kv ({h_kv})",
        )

        # ---- index scope ----
        n_lead = len(lead)
        self._value_error_if(
            idxs.ndim not in (n_lead + 1, n_lead + 2),
            f"topk_idxs must be (*lead, topk) or (*lead, G, topk) with lead={lead}, got {idxs.shape}",
        )
        self._value_error_if(
            tuple(idxs.shape[:n_lead]) != lead,
            f"topk_idxs leading dims must match query rows {lead}, got {idxs.shape}",
        )
        if idxs.ndim == n_lead + 1:
            self.group_scope = 1
        else:
            g = idxs.shape[n_lead]
            self._value_error_if(
                g not in (h_kv, h_q),
                f"topk_idxs group dim must be H_kv ({h_kv}) or H_q ({h_q}), got {g}",
            )
            self.group_scope = g

        if self.topk_length_desc is not None:
            expected = lead if self.group_scope == 1 else lead + (self.group_scope,)
            self._value_error_if(
                tuple(self.topk_length_desc.shape) != expected,
                f"topk_length must have shape {expected}, got {self.topk_length_desc.shape}",
            )

        if self.attn_sink_desc is not None:
            self._value_error_if(
                tuple(self.attn_sink_desc.shape) != (h_q,),
                f"attn_sink must have shape ({h_q},), got {self.attn_sink_desc.shape}",
            )

        # ---- config ----
        self._value_error_if(
            self.index_granularity not in (1, 4, 64, 128),
            f"index_granularity must be one of (1, 4, 64, 128), got {self.index_granularity}",
        )

        # ---- devices ----
        ref_device = q.device
        self._value_error_if(
            getattr(ref_device, "type", str(ref_device)) != "cuda",
            f"Q must live on CUDA, got {ref_device}",
        )
        descriptors = [q, k, v, idxs]
        for opt in (self.topk_length_desc, self.attn_sink_desc, self.cu_seqlens_q_desc):
            if opt is not None:
                descriptors.append(opt)
        self._value_error_if(
            any(d.device != ref_device for d in descriptors),
            f"All inputs must share Q's device {ref_device}, got {[d.device for d in descriptors]}",
        )

        # ---- kernel dispatch ----
        # SM100 DSA sparse-prefill kernel envelope: THD, MQA latent
        # (H_kv == 1, K aliased as V), shared indices, token granularity,
        # D_k in {512, 576} (576 splits QK=576 / V=512).
        in_dsa_envelope = (
            self.is_thd
            and self.group_scope == 1
            and self.index_granularity == 1
            and h_kv == 1
            and d_k in (512, 576)
            and d_v == (512 if d_k == 576 else d_k)
            and major == 10
            and q.dtype in (cutlass.Float16, cutlass.BFloat16)
        )
        # SM100 GQA substrate kernel envelope: G == H_kv (per-KV-head-group
        # indices), block granularity 4/64/128 (QSA / MSA shapes), BF16 or
        # FP8-per-tensor. K/V need not alias (separate D_k/D_v storage) —
        # the DSA envelope's aliased-latent requirement doesn't apply here.
        # The two envelopes are mutually exclusive by construction: the DSA
        # envelope pins granularity == 1 and G == 1, the GQA envelope pins
        # granularity in (4, 64, 128) and G == H_kv > 1 (multi-KV-head).
        in_gqa_envelope = (
            not in_dsa_envelope
            and self.group_scope == h_kv
            and h_kv > 1
            and self.index_granularity in (4, 64, 128)
            and major == 10
            and q.dtype in (cutlass.BFloat16, cutlass.Float8E4M3FN)
        )
        assert not (in_dsa_envelope and in_gqa_envelope), "sparse_attention_forward: DSA and GQA-substrate envelopes must be mutually exclusive"

        if in_dsa_envelope:
            kernel = _get_dsa_prefill_kernel()
        elif in_gqa_envelope:
            kernel = _get_gqa_substrate_kernel()
        else:
            kernel = None
        self._not_implemented_error_if(
            kernel is None,
            "sparse_attention_forward: no registered kernel supports this configuration; "
            "the SM100 DSA sparse-prefill kernel serves THD MQA-latent (H_kv=1, K aliased "
            "as V, D_k in (512, 576), G=1, granularity=1, FP16/BF16), and the SM100 GQA "
            "substrate kernel serves G=H_kv, granularity in (4, 64, 128), BF16/FP8-per-tensor "
            "-- when the corresponding module is present",
        )
        self._dispatch = kernel
        self._dispatch_envelope = "dsa" if in_dsa_envelope else ("gqa" if in_gqa_envelope else None)

        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
        # The reference backend has nothing to compile; device backends own
        # their kernel caches once registered.
        self._compiled_kernel = True

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def execute(
        self,
        q: Any,
        k: Any,
        v: Any,
        topk_idxs: Any,
        topk_length: Optional[Any] = None,
        attn_sink: Optional[Any] = None,
        cu_seqlens_q: Optional[Any] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> tuple[Any, Any]:
        if self._compiled_kernel is None:
            self.compile()
        # The CONTRACT is framework-neutral (any dlpack tensor); each KERNEL
        # declares the frameworks it executes, request-or-fail.
        framework = detect_framework(q)
        if framework != "torch":
            raise NotImplementedError(f"the registered kernel currently executes torch tensors only, got {framework!r} inputs")

        if self._dispatch_envelope == "gqa":
            # GQA substrate envelope: K/V are separate storage (no aliasing
            # requirement), multi-KV-head, index-driven block gather.
            result = self._dispatch(
                q,
                k,
                v,
                topk_idxs,
                topk_length=topk_length,
                attn_sink=attn_sink,
                cu_seqlens_q=cu_seqlens_q,
                index_granularity=self.index_granularity,
                softmax_scale=self.softmax_scale,
                stream=current_stream,
            )
            return result["out"], result["lse"]

        # DSA envelope: K/V aliasing is part of this kernel's envelope; the
        # sample descriptors cannot see storage, so verify on the real
        # tensors.
        d_v = v.shape[-1]
        if get_data_ptr(v) != get_data_ptr(k) or v.shape[0] != k.shape[0]:
            raise ValueError("the registered DSA sparse-prefill kernel requires V to alias K's storage (MLA latent); pass v as a view of k")
        result = self._dispatch(
            q,
            k[:, 0, :],
            topk_idxs,
            attn_sink=attn_sink,
            topk_length=topk_length,
            softmax_scale=self.softmax_scale,
            stream=current_stream,
        )
        return result["out"][:, :, :d_v], result["lse"]


# ----------------------------------------------------------------------
# High-level wrapper
# ----------------------------------------------------------------------
_cache_of_SparseAttentionForwardObjects: dict = {}


def sparse_attention_forward_wrapper(
    q: Any,  # framework tensor (dlpack: torch / JAX / numpy); see module docstring
    k: Any,
    v: Any,
    topk_idxs: Any,
    topk_length: Optional[Any] = None,
    index_granularity: int = 1,
    softmax_scale: Optional[float] = None,
    attn_sink: Optional[Any] = None,
    cu_seqlens_q: Optional[Any] = None,
    max_seqlen_q: Optional[int] = None,
    page_table: Optional[Any] = None,
    page_size: Optional[int] = None,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """High-level wrapper. Returns ``{'out', 'lse'}``.

    ``out`` is ``(T_q, H_q, D_v)`` (THD) or ``(B, S_q, H_q, D_v)`` (BSHD) in
    Q's dtype; ``lse`` is the matching ``(..., H_q)`` FP32 tensor holding the
    KV-only, base-e log-sum-exp (``attn_sink`` contributes to the softmax
    denominator but never to the LSE — the convention consumed by
    :func:`cudnn.deepseek_sparse_attention.sparse_attention_backward_wrapper`).

    ``topk_idxs`` uses storage-native ids (see module docstring), ``-1`` for
    invalid slots; ``topk_length`` optionally gives per-row (per-group) valid
    counts. Execution is deterministic: identical inputs produce
    bitwise-identical outputs.

    ``page_table``/``page_size`` (paged KV) and ``max_seqlen_q`` are part of
    the frozen signature but not yet implemented by any kernel.
    """
    if page_table is not None or page_size is not None:
        raise NotImplementedError("paged KV is not implemented yet; pass contiguous K/V")
    if max_seqlen_q is not None and cu_seqlens_q is None:
        raise ValueError("max_seqlen_q is only meaningful with cu_seqlens_q (THD)")

    key = (
        q.dtype,
        q.shape,
        k.shape,
        v.shape,
        topk_idxs.shape,
        None if topk_length is None else topk_length.shape,
        attn_sink is not None,
        None if cu_seqlens_q is None else cu_seqlens_q.shape,
        int(index_granularity),
        softmax_scale,
    )
    obj = _cache_of_SparseAttentionForwardObjects.get(key)
    if obj is None:
        obj = SparseAttentionForward(
            sample_q=q,
            sample_k=k,
            sample_v=v,
            sample_topk_idxs=topk_idxs,
            sample_topk_length=topk_length,
            sample_attn_sink=attn_sink,
            sample_cu_seqlens_q=cu_seqlens_q,
            index_granularity=index_granularity,
            softmax_scale=softmax_scale,
        )
        assert obj.check_support()
        obj.compile()
        _cache_of_SparseAttentionForwardObjects[key] = obj

    out_t, lse_t = obj.execute(
        q,
        k,
        v,
        topk_idxs,
        topk_length=topk_length,
        attn_sink=attn_sink,
        cu_seqlens_q=cu_seqlens_q,
        current_stream=stream,
    )
    return TupleDict(out=out_t, lse=lse_t)
