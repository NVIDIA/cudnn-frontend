# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attempted FP8 opt-in adapter for the PR2 DSA envelope (``D_k in
{512,576}``, ``D_v=512``, ``G=1``, ``index_granularity=1``, aliased K/V MLA
latent -- ``dsa_fwd_sm100_head64.py``'s envelope), vendoring KF campaign
``qvsntbkbgh6j9a8my01reexvnw``'s winner ``hist128_rcp_full``
(2.917x vs KF's own baseline, 6.862ms). Vendored source (byte-identical,
not hand-modified): ``kf_dsv4/kernel.py`` -- copy this file alongside this
adapter under a ``kf_dsv4/`` subdirectory before use; not yet vendored into
the tree this round (see "Round-1 status" below for why).

**Round-1 finding: this is a HARD BLOCKER, not a narrow fix.** Read to the
end before assuming the missing vendored copy is the only gap.

The round brief asked to confirm, by direct code reading, whether the
vendored kernel's dense fp8 QK/PV GEMM against a fixed ``(TOPK=2048,
HEAD_DIM=512)`` catalog is fed *only* by shape/dtype downstream of catalog
construction (in which case swapping ``_pack_support_kernel``'s synthetic
index formula for a real per-query gather would be a narrow, isolated fix),
or whether the fixed-window catalog design is baked into the GEMM/epilogue
structure itself. Direct read of ``kernel.py`` (all 617 lines, not just the
``_pack_support_kernel`` excerpt from the round brief) establishes the
latter:

1. **The catalog is not just fixed-window, it is *sequence-global*, not
   even per-Q-tile.** ``run()`` calls ``_pack_launch`` exactly *once* per
   ``seqlen`` (cached on ``pack_key = ("pack_fused_sink", seqlen)``) to
   build one ``support_v`` buffer of shape ``(TOPK, HEAD_DIM)``, used as
   operand B for *every* query token in the sequence via one dense GEMM
   tiled over ``M = seqlen * HEADS``. There is no per-tile or per-row
   catalog-rebuild hook anywhere in the file -- the whole architecture's
   speed trick is "MQA against one shared dense window", not "gather a
   per-row/per-tile selection". This is a stronger assumption than MSA's
   ``uniform_within_tile`` (round-1 MSA adapter, ``gqa_prefill_bf16_msa_kf_sm100.py``)
   -- it is "uniform across the *entire* sequence", which the DSA envelope's
   own contract explicitly does not hold (see point 3 below).

2. **The real per-query ``idxs`` tensor is never used to select which K/V
   rows enter the catalog.** ``_pack_support_kernel`` (kernel.py:62-83)
   computes ``kid = bin_idx`` for ``bin_idx < 1024`` else
   ``kid = seqlen + bin_idx - TOPK`` -- i.e. literally "first 1024 + last
   1024 KV rows of the whole sequence" -- a formula over ``bin_idx`` and
   ``seqlen`` only. ``idxs`` (shape ``(seqlen, TOPK)``, genuinely
   per-token) is consumed *only* by ``_hist_counts_kernel``, which folds
   each raw id into a catalog bin via ``bin_idx = raw & (TOPK - 1)`` (a
   power-of-two bitwise-AND mod-fold) and atomically counts collisions per
   ``(token, bin)`` into ``counts``. ``_qk_fused_kernel``'s epilogue then
   computes ``e = exp2(score) * cnt`` -- i.e. it *reweights* the fixed
   catalog row's score by how many of the token's real ids happened to
   fold into that bin, it never fetches the K/V content at those real ids.
   ``_pv_norm_kernel`` mirrors this: the PV GEMM's B operand is
   ``support_v_mn`` -- the same one fixed, sequence-global catalog -- so a
   bin's output contribution is always the fixed catalog row's V content
   scaled by the (possibly multi-id) count weight, never a per-id-distinct
   V value. This is not a lossy approximation of top-k attention that
   converges to it as an edge case; for two real, distinct ids that
   collide under `raw & (TOPK-1)` and have different true K/V content, the
   correct per-id PV contributions are silently replaced by one arbitrary
   catalog row's contribution counted twice. Mathematically wrong, not
   merely approximate, whenever real ids disperse across a range wider
   than a bin-aligned ``TOPK``-sized window (see point 3) -- and
   structurally not fixable by only touching catalog construction, because
   the epilogue's per-bin scalar weighting scheme (score * cnt) has no
   slot to carry more than one V value per bin at all.

3. **The DSA envelope's real ``topk_idxs`` is confirmed non-tile-uniform,
   non-sequence-uniform, by the frozen contract itself** -- not merely
   "the common case" per the round brief's framing, but this envelope's
   documented design point: ``dsa_fwd_sm100_head64.py``'s own module
   docstring lists "duplicate indices, dynamic per-query lengths" as
   first-class behavior it implements (there is no ``uniform_within_tile``
   precondition anywhere in this envelope's dispatch, unlike the MSA/QSA
   GQA-substrate cells in ``python/cudnn/sparse_attention/fwd/sm100_gqa``).
   ``_interface_sm100.py``'s ``topk_idxs`` contract is ``(total_S_q,
   logical_K)`` int32, arbitrary storage-native ids per row, independently
   selected per query token by an upstream lightning indexer -- i.e.
   exactly the case option (b) of the round brief describes: "if per-row
   selections genuinely vary (the common case), this kernel's
   shared-catalog architecture cannot represent our contract at all
   without a full per-row-gather rewrite".

**Conclusion**: option (a) (rebuild ``support_v`` per Q-tile from a real
gather, gated by a ``validate_uniform`` D2H check mirroring
``gqa_prefill_bf16_msa_kf_sm100.py``'s pattern) does not apply -- there is
no tile-uniformity precondition to validate against, because real DSA
selections vary per query token by construction, and even a per-tile
rebuild would not fix point 2 above (the count-weighted single-catalog-row
epilogue has no representation for multiple *distinct* real V values
selected by different rows of the same tile). This is a genuine
architectural mismatch between "one shared dense catalog, per-token
count-reweighted" (this kernel's actual algorithm -- an approximate
MQA-over-fixed-window scheme suited to whatever synthetic index
distribution KF's own harness generates) and "true per-query top-k
gathered attention" (this envelope's frozen contract). Closing this gap
needs a full replacement of the QK/PV GEMM structure with a genuine
per-row-gathered dense-tile mainloop (i.e. reusing this kernel's fp8
``tcgen05.MmaF8F6F4Op`` GEMM primitives and TMA/TMEM plumbing, but not its
catalog/histogram/count-reweight scheme) -- effectively a new kernel
sharing only low-level building blocks with ``kernel.py``, not an adapter
over it. That is out of scope for a round budgeted as "narrower than a
full rewrite"; documenting the blocker is the honest deliverable here, per
the round brief's own instruction to prefer this over "forcing a fit".

**What this adapter still does**, so the file is a real, wired, opt-in
extension point rather than a dead stub: ``fast_path_eligible`` performs
the structural envelope check (dtype, D_k/D_v, ``G==1``,
``index_granularity==1``) the round brief specified, but the vendored
kernel is not invoked -- ``sparse_attention_forward_wrapper`` always
returns ``None`` (this envelope's uniform "not eligible, fall back" signal,
matching every other opt-in adapter in this tree) regardless of shape,
because the *architectural* blocker above holds independent of any runtime
shape. ``kf_dsv4/`` is intentionally not vendored into the tree this round
(there is nothing runnable for it to back yet) -- a future round that
attempts the real per-row-gather rewrite should vendor
``kernel.py``'s ``tcgen05.MmaF8F6F4Op`` GEMM/epilogue plumbing at that
point, not this scheme's catalog/histogram stages.

**-1 / dead-row handling**: not applicable this round -- there is no
launch path to apply it to. For the future full-rewrite kernel, the
established pattern (``gqa_prefill_bf16_msa_kf_sm100.py``'s
``_alloc_partial_buffers`` / row_group_dead masking, and
``dsa_fwd_sm100_head64.py``'s own O=0/max_logits=-inf/LSE=-inf empty-row
sentinel) is the one to reuse: KV-only base-e FP32 LSE, ``attn_sink`` folded
into the denominator (not the LSE, per this envelope's frozen contract),
dead rows (all slots ``-1``) get ``lse=-inf``, ``out=0``, always
deterministic.

**Round-1 KF-integration status (fill in by Verify)**: N/A for compiles /
launches / correctness / determinism -- there is no code path in this
adapter that reaches device execution (``sparse_attention_forward_wrapper``
returns ``None`` unconditionally). ``default_routing_safe=True``: this
module cannot affect default routing because it is never wired into
``_interface_sm100.py``'s hot path, only behind the ``try_fp8_kf`` opt-in
kwarg documented there, and that kwarg's probe always no-ops back to the
existing BF16 kernel. A future round should treat "is a genuine per-row-
gathered fp8 mainloop worth building here" (not "how do I adapt
``kernel.py``") as the open question.
"""

from __future__ import annotations

from typing import Optional

import torch

_HEAD_DIM_V = 512
_SUPPORTED_HEAD_DIM_K = (512, 576)


def fast_path_eligible(*, d_k: int, d_v: int, g: int, index_granularity: int) -> bool:
    """Structural (shape-only) envelope check for the PR2 DSA cell this
    module targets. Always returns ``False``: the vendored kernel's
    sequence-global fixed-catalog architecture cannot represent this
    envelope's genuinely per-row-varying ``topk_idxs`` contract regardless
    of shape -- see the module docstring's "Conclusion" section. Kept as a
    real function (not inlined into the wrapper) so a future round that
    replaces the vendored scheme with a true per-row-gather kernel has a
    single, obvious place to restore the structural gate without needing
    to re-derive it.
    """
    del d_k, d_v, g, index_granularity  # unused: the blocker is architectural, not shape-dependent
    return False


def sparse_attention_forward_wrapper(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_idxs: torch.Tensor,
    *,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    indexer_topk: int = 0,
    stream=None,
):
    """Always returns ``None`` (never raises) -- see the module docstring's
    "Round-1 finding" for why the vendored KF dsv4 kernel cannot represent
    this envelope's per-row-varying ``topk_idxs`` contract this round. This
    keeps the function wired with the same "probe, return ``None`` on
    ineligibility, caller falls back" contract every other opt-in adapter
    in this tree uses (see ``gqa_prefill_bf16_msa_kf_sm100.sparse_attention_forward_wrapper``),
    so ``_interface_sm100.py``'s ``try_fp8_kf`` opt-in kwarg is a real,
    safe no-op today rather than an unwired stub, and a future round can
    fill in a real kernel here without touching the call site again.

    Note for that future round: unlike the MSA/QSA GQA-substrate adapters
    (which return an ``(out, lse)`` ``TupleDict``), this envelope's call
    site (``_interface_sm100.py``'s ``sparse_attention_forward_sm100``)
    expects a 4-tuple ``(out, max_logits, lse, lse_indexer)`` matching its
    own return contract -- a real implementation must produce that shape,
    not the GQA-substrate ``TupleDict`` shape.
    """
    del q, kv, topk_idxs, attn_sink, topk_length, softmax_scale, indexer_topk, stream
    return None
