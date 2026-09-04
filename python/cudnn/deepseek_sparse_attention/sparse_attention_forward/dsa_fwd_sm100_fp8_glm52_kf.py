# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in FP8 adapter for KF campaign ``1a52z46vf16193k7pr01prf8qw`` winner
``glm52_v26_qk_pv_on_main`` (3.39x vs KF's own baseline) -- targets this
PR's DSA envelope (``D_k in {512, 576}``, ``D_v == 512`` aliased MLA latent,
``G == 1``, ``index_granularity == 1``, ``H == 64``). Wired from
``_interface_sm100.py`` via ``try_fp8_kf_glm52`` -- see that module's
docstring for the opt-in contract and safety gate.  Vendored kernel source:
``kf_glm52/kernel.py`` (byte-identical copy of the campaign winner's file,
kept for provenance -- this adapter does **not** import or call it, see
below for why).

**What the vendored kernel actually implements, vs. the frozen contract**
(``python/cudnn/sparse_attention/fwd/api.py``) -- confirmed by direct read
of ``kernel.py`` this round, not re-litigated from a prior hypothesis:

* ``_copy_boundary_kv_fp8`` (``kernel.py:330-346``) builds **one shared**
  ``pool_k_fp8`` / ``pool_v_fp8`` catalog of shape ``(TOPK=2048, D)`` for the
  **entire sequence** -- every query token uses the *same* 2048-row catalog
  (the first/last 1024 rows of the whole KV range, ``src_row = slot -
  HALF_TOPK`` / ``seqlen - HALF_TOPK + slot``), built once via
  ``_launch_copy_boundary_kv_fp8`` before any query-specific work happens.
* ``_count_indices`` (``kernel.py:434-452``) does **not** gather -- it
  computes ``slot = raw + HALF_TOPK`` directly as an offset into that shared
  catalog and atomically increments a per-``(token, slot)`` multiplicity
  counter. This only makes sense if every token's real top-k selection is
  drawn from that *same* fixed 2048-row window (the campaign harness's
  synthetic index generator guarantees this; real DSA selections spanning
  the full ``[0, T_kv)`` range do not).
* The actual attention math (``run()``, ``kernel.py:533-679``) is **one
  dense** ``torch._scaled_mm(q_fp8, kv_pool_k_fp8_t, ...)`` over **all**
  ``rows = seqlen * HEADS`` query rows against that **single shared**
  catalog, followed by a softmax whose per-slot weight is
  ``count[token, slot] * exp(score)`` (``_softmax_fp8_inplace``,
  ``kernel.py:464-511``) -- the per-row "selection" is encoded entirely as
  a multiplicity mask over the *shared* catalog, not via gathering distinct
  per-row KV rows into a per-row catalog.

**Why this is a harder blocker than a narrow gather-swap, and how this
adapter differs from ``dsa_fwd_sm100_head64_fp8_kf.py`` (dsv4)'s round-1
conclusion**: our frozen contract's ``topk_idxs`` selects an **arbitrary,
per-query-row** subset of ``[0, T_kv)`` -- different query rows generally
select different KV rows. glm52's speed comes specifically from doing
*one* dense GEMM (``(rows, D) @ (D, 2048)``) shared across *every* row in
the whole call; that is only sound when every row's real selection is a
subset of one shared, fixed 2048-row set. There is no per-query-tile
dimension anywhere in glm52's GEMM to swap a real gather into, exactly as
dsv4's tcgen05 mainloop lacked one -- **but** unlike dsv4 (a hand-rolled
TMA/smem/tcgen05 mainloop where the catalog buffer is threaded through a
custom pipeline that has no other entry point), glm52's GEMMs are plain
``torch._scaled_mm`` calls against ordinary HBM-resident tensors, with no
TMA descriptor, smem layout, or persistent-kernel state tying the catalog
to "one shared buffer for the whole call". That makes it possible to
*replace the GEMM granularity itself* (one shared-catalog GEMM for the
whole batch -> ``T_q`` independent per-token GEMMs, each against its own
genuinely gathered catalog) without needing to touch any DSL-level
mainloop at all -- a real fix, not merely a documented permanent blocker.
The cost (see the perf note below) is losing glm52's whole-batch GEMM
batching, which is precisely the trick that produced its reported speedup;
this adapter is therefore correct but not competitive with glm52's own
number or with this envelope's ~402-424 TFLOPS BF16 baseline at realistic
``T_q`` -- see the module-level perf note.

**This adapter's actual approach**: a real per-query-row gather (every
token's own ``topk_idxs`` row, ``-1`` -> masked, ids anywhere in
``[0, T_kv)``, matching MSA/QSA's established ``-1`` convention from prior
rounds), with genuine FP8 tensor-core GEMMs (``torch._scaled_mm``, not
emulation) computed **per query token** (``M = H = 64`` rows share one
token's gathered catalog, exactly as many rows as one token's query heads
-- a legitimate GEMM tile, not a degenerate ``M=1``).

**A batched alternative was tried and rejected this round**:
``torch._scaled_grouped_mm`` (offs-delimited groups, each token's own
``(D, topk)`` catalog as one group) is structurally exactly what is
needed to batch this per-token loop into one kernel launch. It was probed
against this environment's PyTorch/CUDA/GPU combination (SM100, torch
2.14.0.dev20260807+cu132) and **aborted the CUDA context** (repeated
``ERROR: Arch conditional MMA instruction used without targeting
appropriate compute capability. Aborting.`` followed by permanent
``CUBLAS_STATUS_NOT_INITIALIZED`` for the rest of the process) even on a
minimal synthetic 3-group smoke test unrelated to this kernel. That is a
correctness-independent, environment-level instability this adapter must
not risk taking on: a Verify pass hitting it would not just fail this
kernel's own checks, it would corrupt CUDA state for every check that
runs afterward in the same process. This adapter therefore deliberately
does not use it; a future round could revisit ``_scaled_grouped_mm`` once
that instability is root-caused (possibly a PyTorch-CUDA/CUTLASS
grouped-GEMM SM100 dispatch bug in this dev build) -- worth flagging
upstream independent of this task, and would likely recover most of the
performance this round's per-token loop leaves on the table.

**Perf expectation (explicit, not measured optimistically)**: looping
``torch._scaled_mm`` once per query token abandons glm52's entire
"one huge shared-catalog GEMM" throughput story -- expect this path to be
*far* below both glm52's reported number and the ~402-424 TFLOPS bf16
baseline for this envelope at realistic ``T_q`` (thousands of Python-level
kernel launches dominate at that scale). This round's Verify should
measure it honestly rather than assume it is competitive; see the round
report for what was actually measured on the shapes exercised.

**Dead rows / -1 handling**: computed entirely via masked arithmetic (no
per-row host-side branch, no ``.item()``/``bool()`` sync inside the token
loop -- see ``feedback_no_hidden_kernel_launches`` /
``feedback_d2h_int_tensor_idiom`` in project memory). A row with zero
valid slots gets ``row_max`` from an all-``-inf`` masked-fill, which is
made finite (``0.0``) only for the *exponentiation* step so ``exp(-inf -
0) == 0`` for every slot; ``sum_kv`` is then exactly ``0`` and ``lse`` is
set to ``-inf`` via a ``where`` on ``sum_kv > 0``; ``out`` is the ``0``
that ``PV`` naturally produces when every probability is ``0``. This
mirrors ``sparse_attention_reference.py``'s dead-row semantics.

**attn_sink**: joins the softmax denominator only (contract: ``lse``
stays KV-only). ``row_max`` is taken over KV-valid slots *and* the sink
value together for numerical stability (matches
``project_frost_lse_sink_strict``'s established convention), but ``lse``
is computed from ``sum_kv`` alone, before the sink term is added to the
denominator used for ``out``'s normalization.

**indexer_topk / lse_indexer**: not supported (glm52's kernel has no such
concept, unlike ``dsa_fwd_sm100_head64.py``'s indexer-prefix-statistic
plumbing) -- this adapter returns ``None`` whenever ``indexer_topk != 0``,
falling back like every other ineligible case.

**Round-2 Verify status (measured, not projected)**: on a real SM100 GPU
(bia box, ``fix613`` venv), against
``test/python/sparse_attention/sparse_attention_reference.py`` at OUR
tolerance (``torch.testing.assert_close(out.float(), ref_out.float(),
atol=2e-2, rtol=2e-2)`` over the *whole* tensor, no matched-ratio slack):

* **compiles/loads**: yes, no vendored-DSL compile step at all (pure
  ``torch._scaled_mm`` + elementwise ops).
* **launches/completes under a hard timeout**: yes -- a single call at
  ``T_q=64, T_kv=8192, H=64, D=576, topk=2048`` completed in ~20ms.
* **oracle correctness at OUR tolerance: FAILS.** Across the
  ``D_k in {512,576}`` x ``topk in {512,1024,2048}`` grid plus ``-1``-pad,
  dead-row, ``attn_sink``, and fuzz cases, most shapes have a handful of
  output elements (typically 2-80 out of ~2*10^5-5*10^5, i.e. well under
  0.1%) exceed ``atol=2e-2``/``rtol=2e-2`` -- max observed absolute error
  ~0.049, with outlier relative errors up to ~100x on near-zero reference
  values (softmax-output cancellation, not a structural bug). Only the
  ``topk=2048`` cells (and a few smaller-``t_kv`` fuzz cases) passed
  cleanly. This is consistent with naive, unscaled FP8 E4M3 quantization
  (``scale_a``/``scale_b`` are hardcoded to ``1.0`` -- no per-tensor/
  per-block amax scaling) rather than a logic bug: it is exactly the class
  of gap KF's own looser matched-ratio tolerance (up to 1% of elements past
  atol/rtol 0.02/0.02) is designed to paper over, and which this project's
  safety gate explicitly does not accept. **A future round adding
  amax-based (or per-block) quantization scaling for Q/K/V before the
  ``torch._scaled_mm`` calls is the most likely fix** -- not attempted this
  round (out of the round's time budget; the perf finding below makes it
  moot for this envelope regardless, see next point).
* **determinism: PASSES.** 25 repeats each on 3 shapes
  (``(16,4096,64,576,1024)``, ``(8,2048,64,512,512)``,
  ``(32,8192,64,576,2048)``) produced bitwise-identical ``out``/``lse``
  every repeat -- no QSA-style race, consistent with this module's
  atomic-free, single-stream, per-token-sequential design.
* **perf: measured, not competitive, exactly as predicted.** At
  ``T_q=64, T_kv=8192, H=64, D=576, topk=2048``: ~20ms/call, ~0.9 TFLOPS
  (``flops_fwd()`` convention) -- roughly **450x below** this envelope's
  ~402-424 TFLOPS BF16 baseline, confirming the per-token Python-loop
  launch overhead dominates exactly as the module docstring predicted.

**Net round-2 verdict**: launches and is deterministic, but does not clear
the correctness gate at this project's tolerance and is not remotely
competitive on perf. Remains ``try_fp8_kf_glm52=True``-gated only; default
routing is unaffected. Not eligible for default-flip until a future round
fixes FP8 quantization scaling (correctness) and replaces the per-token
Python loop with a real batched/grouped GEMM (perf) -- see the
``torch._scaled_grouped_mm`` instability note above for why the latter is
not a small follow-up either.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from cudnn.deepseek_sparse_attention.utils.runtime import torch_stream_context

_HEADS = 64
_SUPPORTED_DK = (512, 576)
_DV = 512
_FP8_DTYPE = torch.float8_e4m3fn


def fast_path_eligible(*, num_heads: int, head_dim: int, head_dim_v: int, topk: int) -> bool:
    """Structural (shape-only, no device read) envelope check.

    ``topk % 16 == 0`` is required because ``torch._scaled_mm`` (both the QK
    and PV GEMMs below) hard-requires its N/K dims to be a multiple of 16 --
    without this check, an arbitrary ``topk`` (e.g. an un-padded per-row
    valid-count contract) would surface as an opaque ``RuntimeError`` from
    inside the kernel loop instead of this adapter cleanly declining and
    letting the caller fall back to the BF16 kernel. Every value in this
    round's benchmark grid (512/1024/2048) already satisfies this.
    """
    if num_heads != _HEADS:
        return False
    if head_dim not in _SUPPORTED_DK:
        return False
    if head_dim_v != _DV:
        return False
    if topk <= 0 or topk % 16 != 0:
        return False
    return True


def _gather_catalog_fp8(kv: torch.Tensor, topk_idxs: torch.Tensor):
    """Real per-query-row gather from storage-native ids -- the fix this
    module exists for.  ``kv`` is ``(T_kv, D)``; ``topk_idxs`` is
    ``(T_q, topk)`` int32 with ``-1`` marking an invalid slot.  Returns the
    gathered K catalog ``(T_q, topk, D)`` and V catalog ``(T_q, topk, D_v)``
    (V is the first ``D_v`` columns of the same aliased latent, matching
    ``dsa_fwd_sm100_head64.py``'s ``kv`` convention), both FP8 E4M3, plus the
    ``(T_q, topk)`` bool validity mask.  Invalid slots gather row 0 (an
    arbitrary in-range placeholder -- never selected, since it is
    subsequently masked to zero and excluded from the softmax) rather than
    the sentinel ``-1`` itself, so this never becomes an out-of-bounds
    index.
    """
    valid = topk_idxs >= 0
    safe_idx = topk_idxs.clamp_min(0).to(torch.int64)
    gathered = kv[safe_idx]  # (T_q, topk, D) -- one real gather, arbitrary per-row ids
    zero = torch.zeros((), dtype=kv.dtype, device=kv.device)
    gathered = torch.where(valid.unsqueeze(-1), gathered, zero)
    k_fp8 = gathered.to(_FP8_DTYPE)
    v_fp8 = gathered[..., :_DV].to(_FP8_DTYPE)
    return k_fp8, v_fp8, valid


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
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    """Attempt the FP8 opt-in path for the DSA (``H=64``, ``D_k in
    {512,576}``, ``D_v=512`` aliased-latent, ``G=1``) envelope.  Returns
    ``None`` (never raises) whenever the config is outside this adapter's
    round-1 scope, so the caller (``_interface_sm100.py``) can always fall
    back to the existing BF16 kernel, or raise its own honest
    ``NotImplementedError`` for FP8 input with no eligible path, safely.
    Called with the *raw*, pre-normalization ``q``/``kv``/``topk_idxs`` --
    this adapter does its own dtype/contiguity handling and does not rely
    on ``_interface_sm100._normalize_and_validate`` having run first.

    Matches the 4-tuple ``(out, max_logits, lse, lse_indexer)`` return shape
    ``sparse_attention_forward_sm100`` itself uses (not the GQA-substrate
    ``TupleDict`` shape the MSA/QSA adapters use) -- ``lse_indexer`` is
    always ``None`` here (see module docstring's ``indexer_topk`` note).
    """
    if q.dtype not in (torch.float16, torch.bfloat16, _FP8_DTYPE):
        return None
    if kv.dtype not in (torch.float16, torch.bfloat16):
        return None
    if q.ndim != 3 or kv.ndim != 2:
        return None
    if int(indexer_topk) != 0:
        return None
    total_s_q, num_heads, head_dim = q.shape
    total_s_kv, kv_head_dim = kv.shape
    if kv_head_dim != head_dim:
        return None
    if topk_idxs.ndim != 2 or topk_idxs.shape[0] != total_s_q or topk_idxs.dtype != torch.int32:
        return None
    topk = int(topk_idxs.shape[-1])
    if not fast_path_eligible(num_heads=num_heads, head_dim=head_dim, head_dim_v=_DV, topk=topk):
        return None
    if total_s_kv == 0 or total_s_q == 0:
        return None

    device = q.device
    scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None else float(softmax_scale)

    if not topk_idxs.is_contiguous():
        topk_idxs = topk_idxs.contiguous()
    if not kv.is_contiguous():
        kv = kv.contiguous()

    if topk_length is not None:
        slot = torch.arange(topk, device=device, dtype=torch.int64).view(1, topk)
        keep = slot < topk_length.to(torch.int64).unsqueeze(-1)
        topk_idxs = torch.where(keep, topk_idxs, torch.full_like(topk_idxs, -1))

    k_fp8, v_fp8, valid = _gather_catalog_fp8(kv, topk_idxs)  # (T_q, topk, D) / (T_q, topk, Dv) / (T_q, topk)
    q_fp8 = q if q.dtype == _FP8_DTYPE else q.to(_FP8_DTYPE)  # (T_q, H, D)
    out_dtype = q.dtype if q.dtype != _FP8_DTYPE else torch.bfloat16

    out_t = torch.empty(total_s_q, num_heads, _DV, dtype=out_dtype, device=device)
    max_logits_t = torch.empty(total_s_q, num_heads, dtype=torch.float32, device=device)
    lse_t = torch.empty(total_s_q, num_heads, dtype=torch.float32, device=device)

    scale_a = torch.ones((), device=device, dtype=torch.float32)
    scale_b = torch.ones((), device=device, dtype=torch.float32)
    neg_inf = float("-inf")
    sink_row = None if attn_sink is None else attn_sink.to(torch.float32)

    # Per-query-token loop: each token's H=64 query rows share one gathered
    # catalog, a real GEMM tile (not a degenerate M=1) -- see module
    # docstring for why this cannot be batched into glm52's original single
    # shared-catalog GEMM, and why a genuinely batched replacement
    # (torch._scaled_grouped_mm) was tried and rejected this round.
    with torch_stream_context(stream):
        for t in range(total_s_q):
            q_t = q_fp8[t]  # (H, D)
            k_t = k_fp8[t].transpose(0, 1)  # (D, topk), transposed view (torch._scaled_mm requires mat2 transposed)
            raw = torch._scaled_mm(q_t, k_t, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float32)  # (H, topk)
            raw = raw * scale
            mask_t = valid[t].unsqueeze(0)  # (1, topk)
            raw = raw.masked_fill(~mask_t, neg_inf)
            row_max = raw.max(dim=-1).values  # (H,)
            if sink_row is not None:
                row_max = torch.maximum(row_max, sink_row)
            row_max_safe = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
            p = torch.exp(raw - row_max_safe.unsqueeze(-1))
            p = torch.where(mask_t, p, torch.zeros_like(p))
            sum_kv = p.sum(dim=-1)  # (H,)
            denom = sum_kv
            if sink_row is not None:
                denom = denom + torch.exp(sink_row - row_max_safe)
            p_fp8 = p.to(_FP8_DTYPE)
            v_t = v_fp8[t]  # (topk, Dv)
            pv = torch._scaled_mm(p_fp8, v_t, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float32)  # (H, Dv)
            safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
            out_t[t] = (pv / safe_denom.unsqueeze(-1)).to(out_dtype)
            max_logits_t[t] = torch.where(sum_kv > 0, row_max, torch.full_like(row_max, neg_inf))
            lse_t[t] = torch.where(sum_kv > 0, torch.log(sum_kv) + row_max_safe, torch.full_like(sum_kv, neg_inf))

    return out_t, max_logits_t, lse_t, None
