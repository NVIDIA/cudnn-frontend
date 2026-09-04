# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""APIBase wrapper for the fused indexer + top-k contract.

One fused pass computes indexer scores and selects the top-k entries per
query — never materializing the dense score row (every production stack
today pays a ``(T_q, T_e)`` fp32 logits tensor plus OOM-chunking machinery
between two kernels). One signature covers the shipped indexer family:
DeepSeek DSA / DSv4 (many weighted ReLU heads, token or compressed-entry
keys), GLM-5.x (32 weighted heads, exact top-k), Qwen QSA (4 unweighted
heads over ratio-4 pooled keys), and MiniMax MSA (per-KV-head-group scoring,
exp-free, score max-pooling to blocks).

Scoring (normative): for query row ``t`` and key entry ``e``,

    score[t, g, e] = sum_{h in group g} w[t, h] * f(q[t, h, :] . k[e, gk, :])

with ``f`` = ReLU (``activation="relu"``) or identity (``"none"``),
``w = 1`` when ``weights`` is None, ``gk = 0`` when ``k_index`` has one head
else ``gk = g``. Any softmax/temperature scale folds into ``weights`` (or the
``q_index`` projection) — there is no scale argument.

Selection (normative):

* **Causal bound**: a key entry is a candidate for a query at global token
  position ``p`` iff the entry is fully in the past: entry ``e`` covers
  tokens ``[e * ratio, e * ratio + ratio)``, so the per-row candidate count
  is ``(p + 1) // ratio``. ``q_causal_offsets`` supplies each THD segment's
  global position of its first query row (chunked prefill / decode);
  ``None`` means segments start at token 0.
* **Score pooling**: with ``score_pool > 1`` scores are max-pooled over
  consecutive groups of ``score_pool`` entries before selection; a pooled
  entry is a candidate iff all covered entries are. Emitted ids are pooled
  ids — downstream ``index_granularity = ratio * score_pool``.
* **Forced includes**: the first ``force_first`` and last ``force_last``
  candidate entries of each row are always selected, inside the ``top_k``
  budget; the remaining budget takes the highest-scoring candidates.
* **Deterministic always**: exact top-k by score, ties at the boundary
  resolved toward the smallest entry id.

Output (normative): storage-native global entry ids, **compact and
ascending** — slots ``[0, topk_length[t, g])`` hold the selected ids sorted
ascending, slots beyond are ``-1``; ``logits`` (opt-in) is slot-aligned FP32
scores. Ascending order is a perf feature for the downstream gather
(monotonic KV loads) and, with the fixed tie-break, makes the whole pipeline
bitwise-reproducible.

Tensor inputs are framework tensors (anything dlpack-compatible — torch,
JAX, numpy): validation runs on canonical descriptors and never requires
torch. Execution dispatches to registered device kernels; no kernel is
registered yet, so every configuration raises ``NotImplementedError`` from
``check_support``. The normative reference lives with the tests
(``test/python/sparse_attention/indexer_topk_reference.py``) and is the
oracle every kernel is validated against.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import cutlass

from cudnn.api_base import APIBase, TupleDict

if TYPE_CHECKING:  # pragma: no cover - typing only
    import cuda.bindings.driver as cuda

_ACTIVATIONS = ("relu", "none")


class IndexerTopK(APIBase):
    """Fused indexer scoring + top-k selection over a shared key stream.

    Layouts (K follows Q's layout):

    * THD (packed varlen): ``q_index (T_q, H_i, D_i)``,
      ``k_index (T_e, H_ik, D_i)`` with ``cu_seqlens_q`` / ``cu_seqlens_k``.
      Emitted ids are global flat entry ids (sequences must start
      ``ratio * score_pool``-aligned in the packed key stream).
    * BSHD: ``q_index (B, S_q, H_i, D_i)``, ``k_index (B, S_e, H_ik, D_i)``.
      Emitted ids are within-sequence entry ids.
    """

    def __init__(
        self,
        sample_q_index: Any,  # framework tensor (dlpack: torch / JAX / numpy)
        sample_k_index: Any,
        top_k: int,
        sample_weights: Optional[Any] = None,
        activation: str = "relu",
        head_groups: int = 1,
        ratio: int = 1,
        score_pool: int = 1,
        force_first: int = 0,
        force_last: int = 0,
        sample_cu_seqlens_q: Optional[Any] = None,
        sample_cu_seqlens_k: Optional[Any] = None,
        sample_q_causal_offsets: Optional[Any] = None,
        return_logits: bool = False,
    ):
        super().__init__()
        self.q_desc = self._make_tensor_desc(sample_q_index, name="sample_q_index", canonical=True)
        self.k_desc = self._make_tensor_desc(sample_k_index, name="sample_k_index", canonical=True)
        self.weights_desc = self._make_tensor_desc(sample_weights, name="sample_weights", canonical=True)
        self.cu_seqlens_q_desc = self._make_tensor_desc(sample_cu_seqlens_q, name="sample_cu_seqlens_q", canonical=True)
        self.cu_seqlens_k_desc = self._make_tensor_desc(sample_cu_seqlens_k, name="sample_cu_seqlens_k", canonical=True)
        self.q_causal_offsets_desc = self._make_tensor_desc(sample_q_causal_offsets, name="sample_q_causal_offsets", canonical=True)

        self.top_k = int(top_k)
        self.activation = activation
        self.head_groups = int(head_groups)
        self.ratio = int(ratio)
        self.score_pool = int(score_pool)
        self.force_first = int(force_first)
        self.force_last = int(force_last)
        self.return_logits = bool(return_logits)

        # Derived in check_support().
        self.is_thd = None
        self._dispatch = None  # registered device-kernel wrapper

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def check_support(self) -> bool:
        from cudnn.tensor_adapter import get_compute_capability

        major, _ = get_compute_capability()
        self._runtime_error_if(
            major < 9,
            f"IndexerTopK requires SM90+, found SM{major}",
        )

        q, k = self.q_desc, self.k_desc

        # ---- layout ----
        self._value_error_if(
            q.ndim not in (3, 4),
            f"q_index must be (T_q, H_i, D_i) or (B, S_q, H_i, D_i), got {q.shape}",
        )
        self.is_thd = q.ndim == 3
        self._value_error_if(
            self.is_thd and (self.cu_seqlens_q_desc is None or self.cu_seqlens_k_desc is None),
            "THD q_index (3-D) requires cu_seqlens_q and cu_seqlens_k",
        )
        self._value_error_if(
            not self.is_thd and (self.cu_seqlens_q_desc is not None or self.cu_seqlens_k_desc is not None),
            "cu_seqlens are only valid with packed THD (3-D) q_index",
        )
        self._value_error_if(
            k.ndim != q.ndim,
            f"k_index must follow q_index's layout ({q.ndim}-D), got {k.shape}",
        )

        # ---- dtypes ----
        self._check_dtype(q, [cutlass.Float16, cutlass.BFloat16], name="q_index")
        self._check_dtype(k, q.dtype, name="k_index", extra_error_msg="k_index must have same dtype as q_index")
        if self.weights_desc is not None:
            self._check_dtype(self.weights_desc, cutlass.Float32, name="weights")
        for desc, name in (
            (self.cu_seqlens_q_desc, "cu_seqlens_q"),
            (self.cu_seqlens_k_desc, "cu_seqlens_k"),
            (self.q_causal_offsets_desc, "q_causal_offsets"),
        ):
            if desc is not None:
                self._check_dtype(desc, cutlass.Int32, name=name)

        # ---- shapes ----
        if self.is_thd:
            t_q, h_i, d_i = q.shape
            _, h_ik, d_ik = k.shape
            lead = (t_q,)
        else:
            b, s_q, h_i, d_i = q.shape
            b_k, _, h_ik, d_ik = k.shape
            self._value_error_if(
                b_k != b,
                f"k_index batch must match q_index batch {b}, got {k.shape}",
            )
            lead = (b, s_q)
        self._value_error_if(
            d_ik != d_i,
            f"k_index head dim must match q_index head dim {d_i}, got {d_ik}",
        )
        self._value_error_if(
            self.head_groups < 1 or h_i % self.head_groups != 0,
            f"head_groups ({self.head_groups}) must divide H_i ({h_i})",
        )
        self._value_error_if(
            h_ik not in (1, self.head_groups),
            f"k_index head count must be 1 (shared) or head_groups ({self.head_groups}), got {h_ik}",
        )
        if self.weights_desc is not None:
            self._value_error_if(
                tuple(self.weights_desc.shape) != lead + (h_i,),
                f"weights must have shape {lead + (h_i,)}, got {self.weights_desc.shape}",
            )
        if self.q_causal_offsets_desc is not None:
            self._value_error_if(
                self.q_causal_offsets_desc.ndim != 1,
                f"q_causal_offsets must be 1-D (batch,), got {self.q_causal_offsets_desc.shape}",
            )

        # ---- config ----
        self._value_error_if(
            self.activation not in _ACTIVATIONS,
            f"activation must be one of {_ACTIVATIONS}, got {self.activation!r}",
        )
        self._value_error_if(self.top_k < 1, f"top_k must be >= 1, got {self.top_k}")
        self._value_error_if(self.ratio < 1, f"ratio must be >= 1, got {self.ratio}")
        self._value_error_if(self.score_pool < 1, f"score_pool must be >= 1, got {self.score_pool}")
        self._value_error_if(
            self.force_first < 0 or self.force_last < 0 or self.force_first + self.force_last > self.top_k,
            f"force_first ({self.force_first}) and force_last ({self.force_last}) must be >= 0 and sum to <= top_k ({self.top_k})",
        )

        # ---- devices ----
        ref_device = q.device
        self._value_error_if(
            getattr(ref_device, "type", str(ref_device)) != "cuda",
            f"q_index must live on CUDA, got {ref_device}",
        )
        descs = [q, k]
        for opt in (self.weights_desc, self.cu_seqlens_q_desc, self.cu_seqlens_k_desc, self.q_causal_offsets_desc):
            if opt is not None:
                descs.append(opt)
        self._value_error_if(
            any(d.device != ref_device for d in descs),
            f"All inputs must share q_index's device {ref_device}, got {[d.device for d in descs]}",
        )

        # ---- kernel dispatch ----
        # No device kernel is registered yet (contract bring-up; kernels are
        # sourced via the Kernel Factory campaigns and register here).
        self._not_implemented_error_if(
            self._dispatch is None,
            "indexer_topk: no registered kernel supports this configuration yet",
        )

        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
        # Registered kernels own their compile caches; compilation is
        # deferred to execute-time tensors.
        self._compiled_kernel = True

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def execute(
        self,
        q_index: Any,
        k_index: Any,
        weights: Optional[Any] = None,
        cu_seqlens_q: Optional[Any] = None,
        cu_seqlens_k: Optional[Any] = None,
        q_causal_offsets: Optional[Any] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> tuple[Any, Any, Optional[Any]]:
        if self._compiled_kernel is None:
            self.compile()
        from cudnn.tensor_adapter import detect_framework

        framework = detect_framework(q_index)
        if framework != "torch":
            raise NotImplementedError(f"the registered kernel currently executes torch tensors only, got {framework!r} inputs")
        return self._dispatch(
            q_index,
            k_index,
            weights=weights,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            q_causal_offsets=q_causal_offsets,
            stream=current_stream,
        )


# ----------------------------------------------------------------------
# High-level wrapper
# ----------------------------------------------------------------------
_cache_of_IndexerTopKObjects: dict = {}


def indexer_topk_wrapper(
    q_index: Any,  # framework tensor (dlpack: torch / JAX / numpy); see module docstring
    k_index: Any,
    top_k: int,
    weights: Optional[Any] = None,
    activation: str = "relu",
    head_groups: int = 1,
    ratio: int = 1,
    score_pool: int = 1,
    force_first: int = 0,
    force_last: int = 0,
    cu_seqlens_q: Optional[Any] = None,
    cu_seqlens_k: Optional[Any] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    q_causal_offsets: Optional[Any] = None,
    return_logits: bool = False,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """High-level wrapper. Returns ``{'indices', 'topk_length'[, 'logits']}``.

    ``indices`` is ``(T_q, top_k)`` (``(T_q, G, top_k)`` when
    ``head_groups > 1``) int32 — storage-native global entry ids, compact
    and ascending, ``-1`` beyond ``topk_length``; ``topk_length`` is the
    matching int32 valid-count tensor; ``logits`` (``return_logits=True``)
    is the slot-aligned FP32 score tensor. Outputs plug directly into
    :func:`cudnn.sparse_attention.fwd.sparse_attention_forward_wrapper`
    (``index_granularity = ratio * score_pool``) and the DSA indexer
    backward. Deterministic: identical inputs give bitwise-identical
    outputs.
    """
    if max_seqlen_q is not None and cu_seqlens_q is None:
        raise ValueError("max_seqlen_q is only meaningful with cu_seqlens_q (THD)")
    if max_seqlen_k is not None and cu_seqlens_k is None:
        raise ValueError("max_seqlen_k is only meaningful with cu_seqlens_k (THD)")

    key = (
        q_index.dtype,
        q_index.shape,
        k_index.shape,
        None if weights is None else weights.shape,
        int(top_k),
        activation,
        int(head_groups),
        int(ratio),
        int(score_pool),
        int(force_first),
        int(force_last),
        None if cu_seqlens_q is None else cu_seqlens_q.shape,
        None if cu_seqlens_k is None else cu_seqlens_k.shape,
        q_causal_offsets is not None,
        bool(return_logits),
    )
    obj = _cache_of_IndexerTopKObjects.get(key)
    if obj is None:
        obj = IndexerTopK(
            sample_q_index=q_index,
            sample_k_index=k_index,
            top_k=top_k,
            sample_weights=weights,
            activation=activation,
            head_groups=head_groups,
            ratio=ratio,
            score_pool=score_pool,
            force_first=force_first,
            force_last=force_last,
            sample_cu_seqlens_q=cu_seqlens_q,
            sample_cu_seqlens_k=cu_seqlens_k,
            sample_q_causal_offsets=q_causal_offsets,
            return_logits=return_logits,
        )
        assert obj.check_support()
        obj.compile()
        _cache_of_IndexerTopKObjects[key] = obj

    result = obj.execute(
        q_index,
        k_index,
        weights=weights,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        q_causal_offsets=q_causal_offsets,
        current_stream=stream,
    )
    indices, topk_length, logits = result
    values = {"indices": indices, "topk_length": topk_length}
    if return_logits:
        values["logits"] = logits
    return TupleDict(**values)
