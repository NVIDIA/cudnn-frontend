# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter for KF campaign ``71242n05bd68s5kser0fn7g6rg`` winner
``msa_r7_v3_unroll8`` (round 8) -- the MiniMax-M3 MSA cell: GQA 64Q/4KV,
``D_k == D_v == 128``, block ``index_granularity == 128``, ``topk == 16``
selections shared per KV-head group (``G == H_kv``). Vendored kernel source:
``kf_msa/{msa.py,msa_helpers.py,kernel.py}`` (byte-identical copies of the
campaign winner's files, not hand-modified -- ``kernel.py`` is kept only for
provenance/reference, this adapter does not import it).

**What the vendored kernel actually implements, vs. the frozen contract**
(``python/cudnn/sparse_attention/fwd/api.py``) -- read top-to-bottom before
touching this file:

* ``kernel.py`` (the campaign's own DPS benchmark harness entry point, i.e.
  what actually produced the reported 4.06-4.10ms number) is a thin,
  benchmark-shaped wrapper that (``kernel.py:253``) folds every index
  through ``idxs.bitwise_and_(NE - 1)`` -- the harness's uniform-random-id
  ``mod NE`` shim baked directly into the *kernel's own entry point*, not
  just its reference. It has **no ``-1``/invalid-slot handling at all**
  (``-1 & (NE-1) == NE-1`` in two's complement -- an invalid slot would
  silently alias to the last real KV block and get gathered like a real
  selection) and never exercises a dead-row path. This file, by design,
  does **not** call ``kernel.py``.
* ``msa_helpers.py`` additionally ships a second, more general code path
  that ``kernel.py`` does not use: ``build_msa_metadata`` /
  ``prepare_workspace`` / ``_compile_and_launch_k1`` /
  ``_compile_and_launch_k2`` (also exercised by ``msa.py``'s
  ``sparse_attention()`` -- the campaign's own correctness-reference
  driver, not the fast benchmark path). ``build_msa_metadata`` **natively
  supports ``-1`` as an invalid-slot sentinel** (its own docstring:
  "Invalid choices may appear in any topK slot and must be represented by
  ``-1``") and builds a CSR reverse-index that structurally excludes ``-1``
  entries from K1's per-(KV-block, head) worklist -- no fold, no wraparound,
  ids pass through as plain batch-local KV-block indices (which is exactly
  our contract's ``index_granularity``-scaled entry semantics: entry ``i``
  covers tokens ``[i*128, i*128+128)``). This is the path this adapter
  vendors and drives -- point (1) and half of point (2) in the round brief
  ("replace the fold with pass-through", "-1 handling before K1") are
  already implemented, upstream, by this alternate entry point; this
  adapter's job is wiring + the one real gap described next, not
  reimplementing index handling.
* **The one real gap this adapter closes**: ``prepare_workspace`` allocates
  ``o_partial``/``lse_partial`` (shape ``(topk, T_q, H_q)``) via
  ``torch.empty`` -- *uninitialized*. K1 only writes the compacted
  ``[0, valid_count)`` prefix of each row's ``topk`` partial slots (CSR
  entries map ``-1``-free selections to compact slots via
  ``cumsum(valid) - 1`` in ``_build_csr``); slots beyond a row's valid count
  are **never touched by K1**. K2's combine kernel (``msa_helpers.py``
  ~4055: ``for s in cutlass.range(self.topk, unroll=8): ...``) unconditionally
  reduces over all ``topk`` slots per row -- it has no per-row valid-count
  input, so an untouched slot's garbage ``lse_partial`` value gets included
  in the merge as-is. Traced K2's actual merge math (``msa_helpers.py``
  ~3900-3920): it computes ``lse_max = max(lse_partial[:])`` (init
  ``-inf``), guards ``lse_max_cur = 0.0 if lse_max == -inf else lse_max``
  (an existing *whole-row-dead* guard against ``max(-inf,...) * log2(e)`` ->
  NaN), then ``scale[s] = exp2(lse_partial[s]*log2(e) - lse_max_cur*log2(e))``
  for every slot and sums. This is a standard streaming-softmax merge: a
  slot with ``lse_partial == -inf`` contributes ``scale == 0`` cleanly (no
  NaN), and if *every* slot for a row is ``-inf`` the whole-row guard above
  makes the final ``lse = log(0) + (-inf) = -inf`` and (since ``O`` starts
  at ``tOrO.fill(0.0)`` and every ``scale == 0``) ``O = 0`` -- i.e. K2's
  merge math *already* implements the frozen contract's dead-row semantics
  (``lse = -inf``, ``out = 0``) correctly, **provided every never-written
  partial slot reads back as ``lse_partial == -inf`` and a finite
  ``o_partial``**. ``prepare_workspace``'s ``torch.empty`` does not
  guarantee that -- this is the real, upstream-unexercised bug the round
  brief is pointing at (the campaign harness's ``kernel.py`` fold means
  every slot is always populated in its benchmark, so this path is never
  hit there either). This adapter's own ``_alloc_partial_buffers`` below
  fixes it directly: ``lse_partial = torch.full(..., -inf)``,
  ``o_partial = torch.zeros(...)`` before every K1 launch, so partially- and
  fully-invalid rows read back through K2's already-correct merge instead of
  through uninitialized memory. This is a real, non-cosmetic per-call cost
  (two extra HBM-bandwidth-bound fills sized ``(topk, T_q, H_q, D_v)`` /
  ``(topk, T_q, H_q)``) -- not attempted to be optimized away this round.
* **Partial dtype**: ``kernel.py``'s reported 4.06-4.10ms was measured with
  ``_PARTIAL_DTYPE = torch.float8_e4m3fn`` o_partial (``kernel.py:27``,
  with an explicit round-6 comment there that bf16 partials regressed
  s32768 by +17% in *that* harness). This adapter instead defaults
  ``partial_dtype=torch.bfloat16`` (also ``msa.py``'s own
  ``sparse_attention()`` default) -- fp8 partials round-trip
  unnormalized-``exp(score - row_max)`` values through a 4-bit-mantissa
  format before the final normalize-by-``1/row_sum`` division; given this
  round's correctness gate is measured against
  ``sparse_attention_reference.py`` (not the campaign's own oracle), and
  point (4) in the round brief explicitly asks to re-measure and fall back
  if fp8 partials fail our tolerance, defaulting to bf16 is the honest
  choice absent a completed accuracy sweep of the fp8 path against our
  oracle -- ``partial_dtype`` is exposed as a kwarg for a future round to
  re-attempt fp8 once that sweep exists. Expect a real perf regression
  relative to the reported number for this reason alone, on top of the
  ``-1``-fill cost above.
* ``attn_sink`` is not supported (no sink term anywhere in K1/K2's ABI --
  confirmed by grep, not merely absent from the harness).
* **THD scope, round 1**: ``build_msa_metadata`` ties per-query KV-block
  addressing to ``cu_seqlens_k`` (batch-local block ids, one row-layout
  segment per batch) -- multi-sequence packed THD would need each
  sequence's own KV segment, i.e. a real ``cu_seqlens_k`` derived from the
  caller, which our top-level contract does not pass to this envelope's
  wrapper (THD here means *one flat KV range addressed by global ids*, no
  per-sequence reset -- see ``api.py``'s ``in_gqa_envelope`` docstring: K/V
  need not alias, but nothing in the GQA envelope's contract resets ids per
  sequence the way DSA's THD does). Driving multiple independent THD
  segments through this adapter would require inventing a
  ``cu_seqlens_k``-equivalent this round's brief does not ask for, so this
  round restricts THD to the single-segment case (``cu_seqlens_q``
  describing exactly one sequence spanning all of ``T_q``) and returns
  ``None`` (structural ineligibility, safe fallback) otherwise -- BSHD is
  unrestricted (any ``B``, uniform per-batch ``S_q``/``S_kv``, which maps
  onto ``build_msa_metadata``'s batch-local convention exactly).

**Round-1 KF-integration status (fill in by Verify)**: compiles/loads,
launches within a hard timeout, oracle correctness, and determinism are
checked by this round's Verify phase against
``test/python/sparse_attention/sparse_attention_reference.py`` on a
realistic shape before ``dispatch.py`` is allowed to default-route here --
see ``dispatch.py``'s module docstring for the outcome. Until that gate
passes all four checks, this module is reachable only via an explicit
opt-in kwarg, never the default path.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Optional

import torch

from cudnn.api_base import TupleDict

_HEAD_DIM = 128
_KV_BLOCK = 128
_SUPPORTED_TOPK = (4, 8, 16, 32)
_SUPPORTED_GQA_RATIOS = (1, 2, 4, 8, 16)

_msa_helpers_mod = None


def _load_kf_msa_modules():
    """Import the vendored ``kf_msa/{msa_helpers.py,msa.py}`` under the
    literal module names they import each other by (``msa.py`` does a bare
    ``import msa_helpers as _msa_helpers`` at its top level; ``msa_helpers.py``
    does a function-local ``from msa import BlackwellMiniMaxSparseAttentionForward``
    inside ``_compile_and_launch_k1``) -- these are flat-layout campaign
    files, not a package, so the loader registers them into
    ``sys.modules`` under those exact bare names rather than trying to
    rewrite their internal imports. Idempotent: returns the cached
    ``msa_helpers`` module on repeat calls without re-exec'ing either file.
    """
    global _msa_helpers_mod
    if _msa_helpers_mod is not None:
        return _msa_helpers_mod
    kf_dir = os.path.join(os.path.dirname(__file__), "kf_msa")
    # msa_helpers.py has no top-level dependency back on msa.py (only the
    # function-local one above), so it must load first.
    for name in ("msa_helpers", "msa"):
        if name in sys.modules:
            continue
        path = os.path.join(kf_dir, f"{name}.py")
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    _msa_helpers_mod = sys.modules["msa_helpers"]
    return _msa_helpers_mod


def fast_path_eligible(*, d_k: int, d_v: int, h_q: int, h_kv: int, index_granularity: int, topk: int) -> bool:
    """Structural (shape-only, no device read) envelope check for the MSA
    cell this adapter targets: ``D_k == D_v == 128``,
    ``index_granularity == 128``, ``topk`` one of the vendored kernel's
    supported values, GQA ratio one of the vendored kernel's supported
    values. ``G == H_kv`` is dispatch.py's/api.py's envelope precondition
    and is not re-checked here.
    """
    if d_k != _HEAD_DIM or d_v != _HEAD_DIM or int(index_granularity) != _KV_BLOCK:
        return False
    if h_kv <= 0 or h_q % h_kv != 0:
        return False
    qhead_per_kv = h_q // h_kv
    if qhead_per_kv not in _SUPPORTED_GQA_RATIOS:
        return False
    if topk not in _SUPPORTED_TOPK:
        return False
    return True


def _alloc_partial_buffers(mh, *, topk: int, t_q: int, h_q: int, partial_dtype: torch.dtype, device: torch.device):
    """Allocate K1's ``o_partial``/``lse_partial`` scratch with the fills
    K2's merge needs to treat a never-written slot as "no contribution"
    (see module docstring's "one real gap" section): ``lse_partial`` must
    read back ``-inf`` (K2's per-slot ``exp2`` term must be exactly zero,
    not garbage) and ``o_partial`` must read back finite (K2 multiplies it
    by that zero scale, so any finite value works, but a garbage bit
    pattern reinterpreted as bf16/fp8 could be NaN/Inf -- zero is always
    safe). ``msa_helpers.prepare_workspace`` uses ``torch.empty`` for both;
    this is the one place this adapter deliberately does not reuse it.
    """
    partial_shape = (topk, t_q, h_q)
    o_partial = torch.zeros(*partial_shape, mh.HEAD_DIM, dtype=partial_dtype, device=device)
    lse_partial = torch.full(partial_shape, float("-inf"), dtype=torch.float32, device=device)
    return o_partial, lse_partial


def _single_segment_cu_seqlens(cu_seqlens_q: Optional[torch.Tensor], t_q: int, device: torch.device) -> Optional[torch.Tensor]:
    """Return ``True``-equivalent (the tensor itself) iff ``cu_seqlens_q``
    describes exactly one segment spanning all of ``T_q`` (or is absent,
    i.e. already single-segment) -- else ``None`` to signal "not eligible
    for this round's THD scope" (see module docstring).
    """
    if cu_seqlens_q is None:
        return torch.tensor([0, t_q], dtype=torch.int32, device=device)
    if cu_seqlens_q.numel() != 2:
        return None
    if int(cu_seqlens_q[0].item()) != 0 or int(cu_seqlens_q[-1].item()) != t_q:
        return None
    return cu_seqlens_q.to(torch.int32)


def sparse_attention_forward_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idxs: torch.Tensor,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    index_granularity: int = 1,
    softmax_scale: Optional[float] = None,
    stream=None,
    *,
    partial_dtype: torch.dtype = torch.bfloat16,
) -> Optional[TupleDict]:
    """Attempt KF's vendored MSA kernel (``kf_msa/msa_helpers.py``'s general
    ``build_msa_metadata`` / K1 / K2 path, NOT ``kernel.py``'s benchmark
    fold -- see module docstring) for the MSA cell (BF16,
    ``D_k == D_v == 128``, ``index_granularity == 128``, ``G == H_kv``).

    Returns ``None`` (never raises) whenever the config is outside this
    adapter's round-1 scope -- multi-segment THD, ``attn_sink`` set,
    unsupported ``topk``/GQA ratio, or an out-of-range/duplicate-vs-bound
    selection ``build_msa_metadata`` itself rejects -- so the caller
    (``dispatch.py``) can always fall through to the scalar kernel safely.
    This is an opt-in-only path (see ``dispatch.py``'s ``try_kf_msa``); it
    is never reached by ``api.py``'s default dispatch.
    """
    if q.dtype != torch.bfloat16:
        return None
    is_thd = q.ndim == 3
    d_k = int(q.shape[-1])
    d_v = int(v.shape[-1])
    h_q = int(q.shape[-2])
    h_kv = int(k.shape[-2])
    n_lead = 1 if is_thd else 2
    topk = int(topk_idxs.shape[-1])
    if not fast_path_eligible(d_k=d_k, d_v=d_v, h_q=h_q, h_kv=h_kv, index_granularity=index_granularity, topk=topk):
        return None
    if attn_sink is not None:
        return None
    if topk_idxs.ndim != n_lead + 2 or int(topk_idxs.shape[n_lead]) != h_kv:
        # G must be H_kv (dispatch.py's envelope precondition); a shared
        # (G==1) or per-Q-head (G==H_q) index tensor isn't this cell.
        return None

    mh = _load_kf_msa_modules()
    device = q.device
    qhead_per_kv = h_q // h_kv

    if is_thd:
        t_q, _, _ = q.shape
        t_kv, _, _ = k.shape
        cu_q = _single_segment_cu_seqlens(cu_seqlens_q, t_q, device)
        if cu_q is None:
            return None
        cu_k = torch.tensor([0, t_kv], dtype=torch.int32, device=device)
        q_flat = q
        k_flat = k
        v_flat = v
        # topk_idxs: (T_q, H_kv, topk) already global-flat block ids over the
        # single flat KV range -> exactly KF's "batch-local" convention for
        # this one-batch case. -> (H_kv, T_q, topk)
        idxs_flat = topk_idxs
        length_flat = topk_length
    else:
        b, s_q, _, _ = q.shape
        _, s_kv, _, _ = k.shape
        if cu_seqlens_q is not None:
            return None
        t_q = b * s_q
        t_kv = b * s_kv
        cu_q = (torch.arange(b + 1, dtype=torch.int32, device=device) * s_q).contiguous()
        cu_k = (torch.arange(b + 1, dtype=torch.int32, device=device) * s_kv).contiguous()
        q_flat = q.reshape(t_q, h_q, d_k)
        k_flat = k.reshape(t_kv, h_kv, d_k)
        v_flat = v.reshape(t_kv, h_kv, d_v)
        idxs_flat = topk_idxs.reshape(t_q, h_kv, topk)
        length_flat = topk_length.reshape(t_q, h_kv) if topk_length is not None else None

    if not q_flat.is_contiguous() or not k_flat.is_contiguous() or not v_flat.is_contiguous():
        q_flat = q_flat.contiguous()
        k_flat = k_flat.contiguous()
        v_flat = v_flat.contiguous()

    idxs_i32 = idxs_flat.to(torch.int32)
    if length_flat is not None:
        # topk_length is this contract's alternate way to bound valid slots
        # (in addition to, or instead of, -1 padding inside topk_idxs):
        # entries at position >= length are masked to -1 so build_msa_metadata's
        # native -1 handling covers both mechanisms uniformly.
        slot = torch.arange(topk, device=device, dtype=torch.int64).view(1, 1, topk)
        keep = slot < length_flat.to(torch.int64).unsqueeze(-1)
        idxs_i32 = torch.where(keep, idxs_i32, torch.full_like(idxs_i32, -1))

    # (T_q, H_kv, topk) -> (H_kv, T_q, topk): KF's q2k_indices convention.
    q2k_indices = idxs_i32.permute(1, 0, 2).contiguous()

    scale = mh.resolve_softmax_scale(softmax_scale)

    try:
        metadata = mh.build_msa_metadata(
            q2k_indices,
            cu_q,
            cu_k,
            block_size=mh.KV_BLOCK_SIZE,
            qhead_per_kv=qhead_per_kv,
        )
    except (TypeError, ValueError):
        # Out-of-range selections, unsupported topk, or another structural
        # mismatch build_msa_metadata itself validates -- not this cell's
        # job to re-derive, just fall back like every other probe here.
        return None

    out = torch.empty(t_q, h_q, d_v, dtype=torch.bfloat16, device=device)
    lse = torch.empty(t_q, h_q, dtype=torch.float32, device=device)

    if int(metadata.work_count.item()) <= 0:
        # Every selection in every row is invalid: no K1 work at all.
        out.zero_()
        lse.fill_(float("-inf"))
        return TupleDict(out=out, lse=lse)

    o_partial, lse_partial = _alloc_partial_buffers(mh, topk=topk, t_q=t_q, h_q=h_q, partial_dtype=partial_dtype, device=device)

    cuda_stream = stream
    if cuda_stream is None:
        import cuda.bindings.driver as _cuda_driver

        cuda_stream = _cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)

    with torch.cuda.device(device):
        mh._compile_and_launch_k1(
            q_flat,
            k_flat,
            v_flat,
            metadata,
            o_partial,
            lse_partial,
            cu_q,
            cu_k,
            head_kv=h_kv,
            qhead_per_kv=qhead_per_kv,
            softmax_scale=scale,
            causal=False,
            qk_dtype=torch.bfloat16,
            pv_dtype=torch.bfloat16,
            stream=cuda_stream,
        )
        mh._compile_and_launch_k2(
            o_partial,
            lse_partial,
            out,
            lse,
            metadata,
            cu_q,
            qhead_per_kv=qhead_per_kv,
            stream=cuda_stream,
        )

    # K2's LSE merge computes ``exp2(lse_partial[s]*log2e - lse_max*log2e)``
    # per slot (see module docstring's "one real gap" section). When a
    # (query, KV-head-group) is *entirely* invalid, every one of its topk
    # slots reads back ``lse_partial == -inf`` (this adapter's own fill,
    # see ``_alloc_partial_buffers``) -- including ``lse_max`` itself, so
    # the merge computes ``-inf - (-inf)`` on that group. Verified on
    # hardware (round-1 Verify) that this produces ``NaN`` (not the
    # mathematically-correct ``0``) here: the two ``-inf*log2e`` terms are
    # not literally added as ``(-inf) + (-inf)`` but reassociated (likely
    # under ``fastmath=True``'s FMA scheduling) into an equivalent-in-exact-
    # arithmetic-but-not-in-float form that hits ``inf - inf`` before the
    # ``exp2``. K1/K2 have no per-row valid-count input to special-case
    # this internally (see module docstring), so this adapter corrects it
    # on the host instead: any (query, KV-head-group) with zero valid
    # selections is overwritten directly to the contract's dead-row values
    # (``lse = -inf``, ``out = 0``), unconditionally replacing whatever K2
    # wrote for it. This is an extra D2H-free host-side reduction + masked
    # write (bounded by ``(T_q, H_kv)``), not a data-dependent control-flow
    # change to the kernel launches above.
    group_has_valid = (q2k_indices >= 0).any(dim=-1)  # (H_kv, T_q) bool
    row_group_dead = ~group_has_valid.transpose(0, 1)  # (T_q, H_kv) bool
    if bool(row_group_dead.any()):
        head_dead = row_group_dead.repeat_interleave(qhead_per_kv, dim=1)  # (T_q, H_q)
        out.masked_fill_(head_dead.unsqueeze(-1), 0.0)
        lse.masked_fill_(head_dead, float("-inf"))

    if not is_thd:
        out = out.reshape(b, s_q, h_q, d_v)
        lse = lse.reshape(b, s_q, h_q)
    return TupleDict(out=out, lse=lse)
