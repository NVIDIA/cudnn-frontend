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

Backends: ``"default"`` dispatches to registered device kernels — currently
the SM100 DSA sparse-prefill kernel for its envelope (THD, MQA latent with
K aliased as V, ``D_k in (512, 576)``, shared token-granularity indices),
when that module is present in the tree — and raises ``NotImplementedError``
otherwise. ``"reference"`` is an explicit opt-in PyTorch implementation used
for validation; reference-speed by design, never selected implicitly.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from cudnn.api_base import APIBase, TupleDict

# Framework neutrality: this module must be importable without torch (JAX
# processes may not import torch at all). torch is imported function-locally
# on the execute/validate paths only; annotations stay lazy via
# ``from __future__ import annotations``.
if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    import cuda.bindings.driver as cuda

_BACKENDS = ("default", "reference")


def _validate_backend(backend: str) -> None:
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")


def _get_dsa_prefill_kernel():
    """Probe for the SM100 DSA sparse-prefill kernel (ships on its own branch).

    Returns the wrapper or ``None``. The generic op registers it as the
    ``backend="default"`` implementation for its envelope when present, so
    this module stays landable (and raising) on trees without the kernel.
    """
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_forward import (
            sparse_attention_forward_wrapper as dsa_fwd,
        )
    except ImportError:
        return None
    return dsa_fwd


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
        sample_q: torch.Tensor,
        sample_k: torch.Tensor,
        sample_v: torch.Tensor,
        sample_topk_idxs: torch.Tensor,
        sample_topk_length: Optional[torch.Tensor] = None,
        sample_attn_sink: Optional[torch.Tensor] = None,
        sample_cu_seqlens_q: Optional[torch.Tensor] = None,
        index_granularity: int = 1,
        softmax_scale: Optional[float] = None,
        backend: str = "default",
    ):
        super().__init__()
        _validate_backend(backend)

        self.q_desc = self._make_tensor_desc(sample_q, name="sample_q")
        self.k_desc = self._make_tensor_desc(sample_k, name="sample_k")
        self.v_desc = self._make_tensor_desc(sample_v, name="sample_v")
        self.topk_idxs_desc = self._make_tensor_desc(sample_topk_idxs, name="sample_topk_idxs")
        self.topk_length_desc = self._make_tensor_desc(sample_topk_length, name="sample_topk_length")
        self.attn_sink_desc = self._make_tensor_desc(sample_attn_sink, name="sample_attn_sink")
        self.cu_seqlens_q_desc = self._make_tensor_desc(sample_cu_seqlens_q, name="sample_cu_seqlens_q")

        self.index_granularity = int(index_granularity)
        self.softmax_scale = softmax_scale
        self.backend = backend

        # Derived in check_support().
        self.is_thd = None
        self.group_scope = None  # G: 1, H_kv, or H_q
        self._dispatch = None  # device-kernel wrapper for backend="default"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def check_support(self) -> bool:
        import torch

        major, _ = torch.cuda.get_device_capability()
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

        # ---- dtypes ----
        self._check_dtype(q, [torch.float16, torch.bfloat16], name="Q")
        self._check_dtype(k, q.dtype, name="K", extra_error_msg="K must have same dtype as Q")
        self._check_dtype(v, q.dtype, name="V", extra_error_msg="V must have same dtype as Q")
        self._check_dtype(idxs, torch.int32, name="topk_idxs")
        if self.topk_length_desc is not None:
            self._check_dtype(self.topk_length_desc, torch.int32, name="topk_length")
        if self.attn_sink_desc is not None:
            self._check_dtype(self.attn_sink_desc, torch.float32, name="attn_sink")
        if self.cu_seqlens_q_desc is not None:
            self._check_dtype(self.cu_seqlens_q_desc, torch.int32, name="cu_seqlens_q")

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
            ref_device.type != "cuda",
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

        # ---- backend envelope ----
        self._dispatch = None
        if self.backend == "default":
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
            )
            kernel = _get_dsa_prefill_kernel() if in_dsa_envelope else None
            self._not_implemented_error_if(
                kernel is None,
                "sparse_attention_forward has no device kernel registered for this "
                "configuration; the SM100 DSA sparse-prefill kernel serves THD MQA-latent "
                "(H_kv=1, K aliased as V, D_k in (512, 576), G=1, granularity=1) when its "
                'module is present. Pass backend="reference" for the PyTorch path.',
            )
            self._dispatch = kernel

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
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        topk_idxs: torch.Tensor,
        topk_length: Optional[torch.Tensor] = None,
        attn_sink: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._compiled_kernel is None:
            self.compile()
        if self.backend == "default":
            # K is V aliasing is part of this kernel's envelope; the sample
            # descriptors cannot see storage, so verify on the real tensors.
            d_v = v.shape[-1]
            if v.data_ptr() != k.data_ptr() or v.shape[0] != k.shape[0]:
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
        assert self.backend == "reference"
        # The reference path runs ordinary PyTorch ops on the caller's
        # current stream; `current_stream` is accepted for signature parity.
        return _reference_forward(
            q,
            k,
            v,
            topk_idxs,
            topk_length=topk_length,
            attn_sink=attn_sink,
            index_granularity=self.index_granularity,
            softmax_scale=self.softmax_scale,
            group_scope=self.group_scope,
            is_thd=self.is_thd,
        )


# ----------------------------------------------------------------------
# Reference implementation (normative semantics; reference speed)
# ----------------------------------------------------------------------
def _reference_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idxs: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    index_granularity: int,
    softmax_scale: Optional[float],
    group_scope: int,
    is_thd: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch

    g = index_granularity

    if is_thd:
        t_q, h_q, d_k = q.shape
        t_kv, h_kv, _ = k.shape
        d_v = v.shape[-1]
        q_flat, k_flat, v_flat = q, k, v
        idxs = topk_idxs.reshape(t_q, -1, topk_idxs.shape[-1]) if topk_idxs.ndim == 3 else topk_idxs.reshape(t_q, 1, -1)
        kv_bound = torch.full((t_q,), t_kv, dtype=torch.int64, device=q.device)
        kv_base = torch.zeros((t_q,), dtype=torch.int64, device=q.device)
    else:
        b, s_q, h_q, d_k = q.shape
        _, s_kv, h_kv, _ = k.shape
        d_v = v.shape[-1]
        t_q = b * s_q
        q_flat = q.reshape(t_q, h_q, d_k)
        k_flat = k.reshape(b * s_kv, h_kv, d_k)
        v_flat = v.reshape(b * s_kv, h_kv, d_v)
        idxs = topk_idxs.reshape(t_q, -1, topk_idxs.shape[-1])
        if topk_idxs.ndim == 3:  # (B, S_q, topk) -> G = 1
            idxs = idxs.reshape(t_q, 1, -1)
        # BSHD ids are within-sequence; each query row gathers from its own
        # batch's KV segment of the flattened stream.
        batch_of_row = torch.arange(b, device=q.device, dtype=torch.int64).repeat_interleave(s_q)
        kv_bound = torch.full((t_q,), s_kv, dtype=torch.int64, device=q.device)
        kv_base = batch_of_row * s_kv

    n_groups = idxs.shape[1]
    topk_max = idxs.shape[-1]
    heads_per_kv = h_q // h_kv

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d_k)

    length = None
    if topk_length is not None:
        length = topk_length.reshape(t_q, n_groups) if group_scope != 1 else topk_length.reshape(t_q, 1)

    idxs = idxs.to(torch.int64)
    slot = torch.arange(topk_max, device=q.device)
    valid = idxs >= 0  # (T_q, n_groups, topk_max)
    if length is not None:
        valid = valid & (slot.view(1, 1, -1) < length.unsqueeze(-1))

    # Expand granularity-g entries to token ids: entry i covers [i*g, i*g+g).
    token_ids = idxs.unsqueeze(-1) * g + torch.arange(g, device=q.device).view(1, 1, 1, g)
    token_valid = valid.unsqueeze(-1) & (token_ids < kv_bound.view(-1, 1, 1, 1))
    token_ids = token_ids.reshape(t_q, n_groups, topk_max * g)
    token_valid = token_valid.reshape(t_q, n_groups, topk_max * g)
    gather_ids = (token_ids.clamp(min=0) + kv_base.view(-1, 1, 1)).clamp(max=k_flat.shape[0] - 1)

    out_t = torch.zeros(t_q, h_q, d_v, dtype=q.dtype, device=q.device)
    lse_t = torch.full((t_q, h_q), float("-inf"), dtype=torch.float32, device=q.device)

    for h in range(h_q):
        kv_head = h // heads_per_kv
        if group_scope == 1:
            grp = 0
        elif group_scope == h_kv:
            grp = kv_head
        else:  # group_scope == h_q
            grp = h
        ids_h = gather_ids[:, grp, :]  # (T_q, K')
        valid_h = token_valid[:, grp, :]
        kk = k_flat[:, kv_head, :].float()[ids_h]  # (T_q, K', D_k)
        vv = v_flat[:, kv_head, :].float()[ids_h]  # (T_q, K', D_v)
        s = torch.einsum("td,tkd->tk", q_flat[:, h, :].float(), kk) * softmax_scale
        s = torch.where(valid_h, s, torch.full_like(s, float("-inf")))

        row_lse = torch.logsumexp(s, dim=-1)  # -inf on dead rows; KV-only
        denom_lse = row_lse
        if attn_sink is not None:
            denom_lse = torch.logaddexp(row_lse, attn_sink[h].float().expand_as(row_lse))
        p = torch.exp(s - denom_lse.unsqueeze(-1))
        p = torch.where(valid_h, p, torch.zeros_like(p))
        out_t[:, h, :] = torch.einsum("tk,tkd->td", p, vv).to(q.dtype)
        lse_t[:, h] = row_lse

    if not is_thd:
        out_t = out_t.reshape(b, s_q, h_q, d_v)
        lse_t = lse_t.reshape(b, s_q, h_q)

    return out_t, lse_t


# ----------------------------------------------------------------------
# High-level wrapper
# ----------------------------------------------------------------------
_cache_of_SparseAttentionForwardObjects: dict = {}


def sparse_attention_forward_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idxs: torch.Tensor,
    topk_length: Optional[torch.Tensor] = None,
    index_granularity: int = 1,
    softmax_scale: Optional[float] = None,
    attn_sink: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
    backend: str = "default",
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
    the frozen signature but not yet implemented by any backend.
    """
    _validate_backend(backend)
    if page_table is not None or page_size is not None:
        raise NotImplementedError("paged KV is not implemented yet; pass contiguous K/V")
    if max_seqlen_q is not None and cu_seqlens_q is None:
        raise ValueError("max_seqlen_q is only meaningful with cu_seqlens_q (THD)")

    key = (
        backend,
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
            backend=backend,
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
