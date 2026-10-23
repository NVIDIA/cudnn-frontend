# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Rubin index-list sparse d256 forward (``kernels/sm107/sparse_d256_f16.py``) through its adapter
(``sdpa/fwd/sparse_gqa_sm107.py``): the adapter's typed declines, the source pins, and the kernel's correctness matrix.

Host tier (any box): the claims record agrees with the config's wired-arm set (an arm the body does not carry is declined by
BOTH); every unserved request is a typed decline naming the feature; the DSL gate declines by name below the ``sm_107a``
floor without importing the kernel module; the kernel source keeps the sm107 conventions (derived descriptor version on every
SMEM tile, the module spin constant on every per-tile ring wait, the LOAD wait between the S^T read and the ``s_empty``
arrive, the one-helper ``_item_bounds`` at the item start of every tile-looping role, every ring wait / arrive on a carried
``PipelineState``, the empty-row SELECT after the denominator floor, a per-lane validity mask, no module-level assert, no
atomic); and the multiplicity reference used below agrees with the oracle's visible-set rule on every contract-clean list.

Rubin tier (``requires_rubin``), every cell against an fp32 reference built from the oracle's visible-set rule
(``qsa_visible_mask``: the listed complete blocks inside the causal range plus the open tail, ``-1`` below the count = no
key), with poisoned outputs (no sentinel may survive on a live row, a dead row lands ``O = 0`` / ``LSE = -inf`` exactly) and
two launches bitwise:

* the bring-up ladder (one tile, 2 / 3 / 4 / 17 tiles, ``B x H > 1`` at one tile, the identity bound, a dead sequence through
  per-batch KV lengths, ``block_lens`` absent at ``top_k = 256``, f16, one KV head over 16 query heads);
* the degenerate-input matrix row by row: a query with 0 complete blocks, ``(pos + 1) % 4 == 0`` (no open block), an empty
  selection (tail only; dead where there is no tail either), ``-1`` inside the count (a VALUE: that block is absent), entries
  beyond the count (never read -- bitwise invariant, both list forms, ``block_lens`` given or absent), ids past the sequence
  length and past the tensor (masked key by key), a future block (contributes nothing), a listed id that duplicates another
  entry or the open tail (counted twice -- the documented contract), ``block_lens`` out of range (clamped on device), ``B = 2``
  with a different list per entry, the slab (``inplace_qkv``) K / V layout, ``top_k`` in {4, 64, 256}, MHA (one query head
  per KV head) and the 24/2 geometry, one CTA in the whole grid, dead and live items interleaved in one CTA's item sequence;
* the identity check: with the full list at ``S <= 2051`` the sparse kernel, the dense sm107 d256 kernel and the oracle agree
  within the dense budget (the same function, a different summation order -- not bitwise; both magnitudes reported);
* the adversarial ``S = 2052`` cell: a dominant key planted inside the block the 512-wide list omits for row 2051 -- within
  budget vs the oracle ON the list, outside it vs the oracle on the FULL set (and vs the dense kernel) at that row by the
  plant's margin -- the pin that the list is really applied;
* the recorded index lists of the released checkpoint's first QSA layer (``CUDNN_FROST_QSA_INDEX_LISTS_DIR``; skipped when
  unset), sha256-verified before use, at S = 32768 (bf16) and on the 4096-row prefix (f16);
* the paged read (``paged_kv=True``): the same tokens through page pools -- page 16 / 64 / 48, HND and NHD storage, bf16 and
  f16, B = 2 with different lengths and lists, a block in a partially filled last page, the open block's rows beyond the
  length sitting in that page next to ANOTHER sequence's tokens (or NaN), ``-1`` table entries past the live pages, ids past
  the length / the table, ``block_lens`` present and absent, up to 17 tiles -- BITWISE the dense read of the same tokens and
  within the budget vs the oracle; the typed declines of the paged form (``page_size`` not a multiple of 4, a missing table or
  length, the table / page size on a dense declaration, every malformed pool stride);
* the THD arm (``thd=True``: packed ``[1, T, H, D]`` operands, per-sequence or cumulative lengths, sequence-relative block ids,
  the persistent claim-counter scheduler): every sequence of a packing against ITS OWN reference (the packings [300, 128,
  200] and [2048, 4096, 6144, 8192], a zero-length sequence at the front / in the middle / trailing, a 5-token sequence, a
  sequence with Q tokens and no keys -- every row dead, stored exactly -- a ragged ``S_kv_b < S_q_b``, a capacity tail past
  the packed totals left untouched, ``block_lens`` absent, a small packing whose capacity grid exceeds its live units and an
  all-empty one -- the CTAs at or past the live total exit at entry), a one-sequence packing BITWISE the dense ``B = 1, S = T`` run, the
  cumulative length forms (a prefix sliced from a larger one included) bitwise the lengths form, and on every sequence the
  missed-coordinate signature of a THD port (``LSE == log(n_vis)`` -- an all-zero K operand) ruled out;
* the DECODE form (``bottom_right=True``, ``list_per_sequence=True``, ``split_kv=S``): the ``S_q = 1`` ladder, the dead-sequence
  partials, the paged open block, the gate on the epilogue / in the combine, and the MTP MATRIX of the shared list -- ``S_q`` in
  {1, 2, 3, 4} x every residue of the step-0 position ``pos_0 = kv_len - S_q`` mod 4 x B in {1, 4}: per row the step-0 anchor's
  0 / 1 / 2 appended tail blocks, the rows where the per-row-tail reading hides the block completed after step 0 (exactly
  ``j >= 3 - r`` at ``r < 3``, by exactly that block; natural teeth on a ~60-token sequence, a PLANTED dominant key on a saturated
  512-list sequence whose tail ids sit at list indices 512 / 513), per-row causal (the rows before the plant never see it), a
  degenerate ``S_q - 1``-token sequence, paged bitwise the dense read, split 2 within the derived budget; the step-0 reuse pin
  (the shared list BITWISE the per-token form fed with the replicated step-0 list + the completed block appended explicitly);
  the boundary decline at ``S_q = 5``.

Tolerances: atol 2e-2 on O and LSE, no rtol -- the sparse module's own budget, TIGHTER than the dense sm107 suite's O bar (atol
5e-2 / rtol 3e-2) and equal to its LSE atol; nothing widened, nothing added.
"""

import hashlib
import math
import os
import re
import sys

import pytest
import torch

from frost_test_utils import requires_dsl, requires_rubin

pytestmark = [pytest.mark.L0, requires_dsl]

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORACLE_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "gated_attention_block", "cutedsl")
D = 256
BS = 4  # tokens per selectable block
ATOL = 2e-2  # on O and LSE, no rtol: tighter than the dense sm107 suite's O budget (atol 5e-2 / rtol 3e-2) -- never widened here
# A directory holding ``layer03_S<S>_block_ids.pt`` (+ ``.sha256`` sidecars): the recorded block-id lists of the released
# checkpoint's first QSA layer over one 32K-token text.  Unset -> the real-list cells skip.
INDEX_LISTS_ENV = "CUDNN_FROST_QSA_INDEX_LISTS_DIR"


def _kernel_source() -> str:
    import cudnn.sdpa.fwd.config_sm107 as c107

    path = os.path.join(os.path.dirname(os.path.abspath(c107.__file__)), "kernels", "sm107", "sparse_d256_f16.py")
    return open(path).read()


def _item_common_source() -> str:
    """The visible-set helper both sparse bodies share (sm107/_sparse_item_common.py; the arithmetic that lived in this kernel)."""
    import cudnn.sdpa.fwd.config_sm107 as c107

    path = os.path.join(os.path.dirname(os.path.abspath(c107.__file__)), "kernels", "sm107", "_sparse_item_common.py")
    return open(path).read()


def _adapter_source() -> str:
    import cudnn.sdpa.fwd.sparse_gqa_sm107 as adapter

    return open(adapter.__file__).read()


def _code(src: str) -> str:
    return "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))


def _oracle():
    if _ORACLE_DIR not in sys.path:
        sys.path.insert(0, _ORACLE_DIR)
    import gated_block_qsa_reference as oracle

    return oracle


# ============================================================================ host: the claims record and the declines
def test_claims_record_agrees_with_the_config_wired_arms():
    """Form A: the adapter's record and the config's ``SPARSE_D256_WIRED_ARMS`` are two spellings of one fact -- an arm the
    body does not carry is False in the record AND absent from the wired set; a wired arm must be True in the record."""
    from cudnn.sdpa.fwd.config_sm107 import SPARSE_D256_WIRED_ARMS
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SPARSE_CAPABILITIES as C

    pairs = {
        "epilogue_gate": C.epilogue_gate,
        "thd_varlen": C.thd,
        "paged_kv": C.paged_kv,
        "split_kv": C.split_kv,
        "bottom_right": C.bottom_right,
        "list_per_sequence": C.index_list_per_sequence,
        "seq_q_lens": C.padded,
    }
    for arm, claimed in pairs.items():
        assert claimed == (arm in SPARSE_D256_WIRED_ARMS), f"{arm}: record {claimed}, wired {arm in SPARSE_D256_WIRED_ARMS}"
    assert C.causal and C.kv_lens and C.pack_gqa and C.d_shapes == frozenset({(D, D)}) and C.index_block_sizes == frozenset({BS})
    assert (C.index_top_k_min, C.index_top_k_max, C.index_gqa_group_max) == (4, 512, 16)
    # the decode form: the shared list (bottom-right anchored, S_q <= 4) and the split are the record's decode claim
    assert C.decode == (C.index_list_per_sequence and C.bottom_right and C.split_kv) and C.index_shared_list_max_tokens == 4
    assert C.cutedsl_min_version == (4, 8, 0) and (C.sm_lo, C.sm_hi) == (107, 119)


class _T:
    """A tensor stand-in for the host-side declines (shape / stride / dtype only)."""

    def __init__(self, shape, stride, dtype):
        self.shape, self._stride, self.dtype = shape, stride, dtype

    def stride(self):
        return self._stride

    def is_contiguous(self):
        return True


@pytest.fixture
def sm107_host_target(monkeypatch):
    """Model the compiler target for host contracts; native execution and target-decline tests use their own gates."""
    import cudnn.frost.buffers as buffers

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)


def _operands(B=1, S=8, H=24, KH=2, SKV=None, dtype="torch.bfloat16", top_k=512):
    SKV = S if SKV is None else SKV
    q = _T((B, S, H, D), (S * H * D, H * D, D, 1), dtype)
    k = _T((B, SKV, KH, D), (SKV * KH * D, KH * D, D, 1), dtype)
    ids = _T((B * S, top_k), (top_k, 1), "torch.int32")
    return dict(q=q, k=k, v=k, o=q, block_ids=ids, top_k=top_k, device_cc=(10, 7))


def _pool_strides(KH, P, hnd):
    """Strides of a ``[num_pages, KH, P, D]`` pool view: HND compact, or NHD storage ``[num_pages, P, KH, D]`` viewed as that shape."""
    return (KH * P * D, P * D, D, 1) if hnd else (P * KH * D, D, KH * D, 1)


@pytest.mark.parametrize("operand", ["k", "v"])
@pytest.mark.parametrize("KH,head_stride", [(2, 2**32 + D), (3, 2**30 + D)])
def test_gather_column_range_checks_each_operand(operand, KH, head_stride, sm107_host_target):
    """Both the stride itself and the head-index product must fit the gather descriptor's Int32 column coordinate."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    args = _operands(B=1, S=1, H=KH, KH=KH, top_k=4)
    args[operand] = _T((1, 1, KH, D), (KH * head_stride, KH * head_stride, head_stride, 1), "torch.bfloat16")
    with pytest.raises(NotImplementedError, match="Int32"):
        SparseGqaFwdDslSm107(**args).check_support()


@pytest.mark.parametrize("kind", ["expanded", "strided", "overlapping"])
@pytest.mark.parametrize("thd", [False, True])
def test_workspace_requires_contiguous_backing_storage(kind, thd, sm107_host_target):
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    args = _operands(B=1, S=1, H=2, KH=1, top_k=64)
    if thd:
        args.update(thd=True, seq_q_lens=_T((1,), (1,), "torch.int32"), seq_kv_lens=_T((1,), (1,), "torch.int32"))
    else:
        args.update(split_kv=2)
    plan = SparseGqaFwdDslSm107(**args)
    assert plan.check_support()
    need = plan.scratch_workspace_bytes()
    if kind == "expanded":
        ws = torch.empty(1, dtype=torch.uint8).expand(need)
    elif kind == "strided":
        ws = torch.empty(2 * need, dtype=torch.uint8)[::2]
    else:
        ws = torch.empty(need, dtype=torch.uint8).as_strided((2, need // 2), (1, 1))
    with pytest.raises(ValueError, match="workspace.*contiguous"):
        plan._check_workspace(ws, 1)
    # A contiguous suballocation is valid, but only its remaining bytes count.
    backing = torch.empty(need + 32, dtype=torch.uint8)
    plan._check_workspace(backing[16 : 16 + need], 1)
    with pytest.raises(ValueError, match="workspace of.*bytes"):
        plan._check_workspace(backing[16 : 16 + need - 1], 1)


def _device_validation_plan(operand):
    """Tiny real buffers, with entry interception below: even the RED test never dispatches an invalid pointer."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    q = torch.zeros((1, 4, 2, D), dtype=torch.bfloat16, device="cuda")
    k = torch.zeros((1, 4, 1, D), dtype=q.dtype, device=q.device)
    args = dict(
        q=q,
        k=k,
        v=k.clone(),
        o=torch.empty_like(q),
        lse=torch.empty((1, 2, 4), device=q.device),
        block_ids=torch.zeros((1, 4, 64), dtype=torch.int32, device=q.device),
        block_lens=torch.zeros((1, 4), dtype=torch.int32, device=q.device),
        seq_kv_lens=torch.tensor([4], dtype=torch.int32, device=q.device),
        epilogue_gate=torch.zeros_like(q),
        top_k=64,
        device_cc=(10, 7),
    )
    if operand == "block_table":
        args.update(
            paged_kv=True, page_size=4, k=k.transpose(1, 2), v=k.clone().transpose(1, 2), block_table=torch.zeros((1, 1), dtype=torch.int32, device=q.device)
        )
    if operand == "seq_q_lens":
        args.update(thd=True, seq_q_lens=torch.tensor([4], dtype=torch.int32, device=q.device))
    if operand == "workspace":
        args.update(split_kv=2)
    plan = SparseGqaFwdDslSm107(**args)
    assert plan.check_support()
    if plan.thd or plan.split_kv > 1:
        plan.workspace = torch.empty(plan.scratch_workspace_bytes(), dtype=torch.uint8, device=q.device)
    return plan


@pytest.mark.parametrize("operand", ["q", "k", "v", "o", "lse", "block_ids", "block_lens", "seq_kv_lens", "block_table", "seq_q_lens", "gate", "workspace"])
@pytest.mark.parametrize("binding", ["declared", "rebound"])
def test_execute_rejects_cpu_storage_before_pointer_lowering(operand, binding, monkeypatch, sm107_host_target):
    import cutlass.cute.runtime as runtime

    plan = _device_validation_plan(operand)
    invalid = getattr(plan, operand).cpu()

    def forbidden(*args, **kwargs):
        pytest.fail("invalid operand reached pointer lowering or a kernel entry")

    plan._fns = {False: forbidden, True: forbidden}
    plan._combine = forbidden
    monkeypatch.setattr(runtime, "make_ptr", forbidden)
    if binding == "declared":
        setattr(plan, operand, invalid)
        bindings = {}
    else:
        bindings = {operand: invalid}
    with pytest.raises(ValueError, match=operand + ".*CUDA storage"):
        plan.execute(**bindings)


@pytest.mark.parametrize("operand", ["q", "k", "v", "o", "lse", "block_ids", "block_lens", "seq_kv_lens", "block_table", "seq_q_lens", "gate", "workspace"])
@pytest.mark.parametrize("device", ["cpu", "cuda:1"])
def test_declarations_reject_non_cuda_or_mixed_devices(operand, device, sm107_host_target):
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseOperandDesc

    plan = _device_validation_plan(operand)
    tensor = getattr(plan, operand)
    # No second GPU is touched: descriptor device metadata alone must reject this binding.
    other = SparseOperandDesc(tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, torch.device(device))
    setattr(plan, operand, other)
    with pytest.raises(ValueError, match="CUDA storage|execution device"):
        plan.check_support()


@pytest.mark.parametrize("declared_device", [None, "cuda:1"])
def test_compilation_and_launch_preserve_the_declared_or_resolved_device(declared_device, monkeypatch, sm107_host_target):
    """Intercept context changes/compilation: no second GPU or invalid launch is needed to prove context ownership."""
    from contextlib import contextmanager
    from types import SimpleNamespace
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107, SparseOperandDesc
    import cudnn.frost.template_loader as loader

    args = _operands(B=1, S=1, H=2, KH=1, top_k=4)
    for key in ("q", "k", "v", "o", "block_ids"):
        t = args[key]
        args[key] = SparseOperandDesc(t.shape, t.stride(), t.dtype, declared_device)
    current, compiled, launched = [0], [], []

    @contextmanager
    def device_context(index):
        old, current[0] = current[0], int(index)
        try:
            yield
        finally:
            current[0] = old

    def compile_entry(**kwargs):
        compiled.append(current[0])
        return lambda **kw: None

    monkeypatch.setattr(torch.cuda, "current_device", lambda: current[0])
    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(loader, "load_template", lambda *a, **kw: SimpleNamespace(compile=compile_entry))
    plan = SparseGqaFwdDslSm107(**args)
    plan.compile()
    expected = 0 if declared_device is None else 1
    current[0] = 1 - expected
    plan.compile(has_block_lens=True)
    assert compiled == [expected, expected]
    assert current[0] == 1 - expected

    bindings = {
        key: SparseOperandDesc(args[key].shape, args[key].stride(), args[key].dtype, f"cuda:{1 - expected}") for key in ("q", "k", "v", "o", "block_ids")
    }
    with pytest.raises(ValueError, match="q.*execution device"):
        plan.execute(**bindings)
    # Matching bindings enter the pinned context even when the caller's current device differs.
    bindings = {key: SparseOperandDesc(args[key].shape, args[key].stride(), args[key].dtype, f"cuda:{expected}") for key in bindings}
    monkeypatch.setattr(plan, "_execute_on_device", lambda *a: launched.append(current[0]))
    plan.execute(**bindings)
    assert launched == [expected]
    assert current[0] == 1 - expected


def test_compilation_bypasses_argument_only_caches_across_devices(monkeypatch, sm107_host_target):
    from contextlib import contextmanager
    from functools import lru_cache
    from types import SimpleNamespace
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107, SparseOperandDesc
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine
    import cudnn.frost.compiled_cache as cache
    import cudnn.frost.template_loader as loader

    current, kernels, combines = [0], [], []

    @contextmanager
    def context(index):
        old, current[0] = current[0], int(index)
        try:
            yield
        finally:
            current[0] = old

    @lru_cache(None)
    def kernel(**kwargs):
        kernels.append(current[0])
        return lambda **kw: None

    @lru_cache(None)
    def combine(**kwargs):
        combines.append(current[0])
        return lambda *a: None

    monkeypatch.setattr(torch.cuda, "current_device", lambda: current[0])
    monkeypatch.setattr(torch.cuda, "device", context)
    monkeypatch.setattr(loader, "load_template", lambda *a, **kw: SimpleNamespace(compile=kernel))
    monkeypatch.setattr(split_combine, "compile_ptr", combine)
    monkeypatch.setattr(cache, "positional_entry", lambda owner: owner)
    for device in (0, 1):
        args = _operands(B=1, S=1, H=2, KH=1, top_k=64)
        for key in ("q", "k", "v", "o", "block_ids"):
            t = args[key]
            args[key] = SparseOperandDesc(t.shape, t.stride(), t.dtype, f"cuda:{device}")
        SparseGqaFwdDslSm107(**args, split_kv=2).compile()
    assert kernels == [0, 1]
    assert combines == [0, 1]
    assert current[0] == 0


def _paged_operands(B=1, S=64, H=24, KH=2, P=16, dtype="torch.bfloat16", top_k=512, hnd=True, max_pages=None, extra_pages=2):
    """The paged form's stand-ins: pools of ``B x max_pages + extra_pages`` pages, a ``(B, max_pages)`` table, ``[B]`` lengths."""
    max_pages = -(-S // P) if max_pages is None else max_pages
    num_pages = B * max_pages + extra_pages
    o = _operands(B=B, S=S, H=H, KH=KH, dtype=dtype, top_k=top_k)
    pool = _T((num_pages, KH, P, D), _pool_strides(KH, P, hnd), dtype)
    o.update(
        k=pool,
        v=pool,
        paged_kv=True,
        page_size=P,
        block_table=_T((B, max_pages), (max_pages, 1), "torch.int32"),
        seq_kv_lens=_T((B,), (1,), "torch.int32"),
    )
    return o


def test_adapter_accepts_the_record_and_builds_the_params(sm107_host_target):
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    a = SparseGqaFwdDslSm107(**_operands())
    assert a.check_support() is True
    p = a.template_params()
    assert (p.qsa_block_topk, p.qsa_block_size, p.qh_per_kh, p.pack_gqa, p.cta_mma, p.dtype_qkv) == (512, 4, 12, True, 1, 2)
    assert not p.seq_kv_lens_present and not p.epilogue_gate and not p.thd_varlen and not p.paged_kv and p.page_size == 0


@pytest.mark.parametrize("P", [16, 64, 48])
@pytest.mark.parametrize("hnd", [True, False], ids=["HND", "NHD"])
def test_adapter_accepts_the_paged_form_and_derives_the_pool_geometry(P, hnd, sm107_host_target):
    """``paged_kv=True`` with pools, a table and lengths is served (the record claims it); the template is the paged
    specialization with the lengths present; the per-operand row terms the kernel takes come from the strides: an HND pool
    folds the head into the row (``rows_per_head = P``, column head stride 0), an NHD pool keeps it in the column
    (``rows_per_head = 0``, column head stride D); rows per page = P (NHD) or KH x P (HND); S_kv = the table's capacity."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    B, S, KH = 2, 100, 2
    a = SparseGqaFwdDslSm107(**_paged_operands(B=B, S=S, KH=KH, P=P, hnd=hnd))
    assert a.check_support() is True
    p = a.template_params()
    assert (p.paged_kv, p.page_size, p.seq_kv_lens_present) == (True, P, True)
    max_pages = -(-S // P)
    num_pages, mp, geom_k, geom_v = a._paged
    assert (num_pages, mp) == (B * max_pages + 2, max_pages) and a._SKV == max_pages * P
    s_page, s_head, s_tok, _ = _pool_strides(KH, P, hnd)
    expect = (s_page, s_tok, 0, KH * P, P) if hnd else (s_page, s_tok, D, P, 0)
    assert geom_k == expect and geom_v == expect, (geom_k, expect)


def _decode_operands(B=1, S=1, H=24, KH=2, SKV=8192, dtype="torch.bfloat16", top_k=512, split_kv=1, **paged):
    """The decode form's stand-ins: ``S`` rows per sequence at the END of the sequence (bottom-right), ONE list per sequence
    (``block_ids [B, top_k]``, ``block_lens [B]``), per-batch KV lengths, an optional split; ``paged`` = the paged form's extras."""
    o = (
        _paged_operands(B=B, S=S, H=H, KH=KH, dtype=dtype, top_k=top_k, **paged)
        if paged
        else _operands(B=B, S=S, H=H, KH=KH, SKV=SKV, dtype=dtype, top_k=top_k)
    )
    o.update(
        block_ids=_T((B, top_k), (top_k, 1), "torch.int32"),
        block_lens=_T((B,), (1,), "torch.int32"),
        seq_kv_lens=_T((B,), (1,), "torch.int32"),
        list_per_sequence=True,
        bottom_right=True,
        split_kv=split_kv,
    )
    return o


@pytest.mark.parametrize("split_kv", [1, 2, 17])
@pytest.mark.parametrize("paged", [False, True], ids=["dense", "paged16"])
def test_adapter_accepts_the_decode_form_and_builds_the_params(split_kv, paged, sm107_host_target):
    """The decode form (``list_per_sequence=True, bottom_right=True[, split_kv=S]``): accepted at ``S_q`` 1..4 over a dense tensor
    or paged pools, the three arms in the template record, a split's gate routed to the COMBINE (``epilogue_gate`` False in the
    record while the gate operand is kept), the workspace = the two fp32 partial slabs (+ the final-LSE scratch without an LSE),
    0 unsplit.  Bottom-right with per-token lists is accepted on its own."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    B, H, KH = 2, 24, 2
    for S in (1, 4):
        kw = _decode_operands(B=B, S=S, H=H, KH=KH, split_kv=split_kv, **(dict(P=16) if paged else {}))
        a = SparseGqaFwdDslSm107(**kw)
        assert a.check_support() is True
        tp = a.template_params()
        assert (tp.split_kv, tp.bottom_right, tp.qsa_list_per_sequence, tp.paged_kv, tp.seq_kv_lens_present, tp.epilogue_gate) == (
            split_kv,
            True,
            True,
            paged,
            True,
            False,
        )
        if split_kv > 1:
            o_part, lse_part = B * split_kv * S * H * D * 4, B * split_kv * H * S * 4
            assert (
                a.scratch_workspace_bytes() == -(-o_part // 16) * 16 + -(-lse_part // 16) * 16 + -(-(B * H * S * 4) // 16) * 16
            )  # no LSE declared: + the scratch
            kw2 = dict(kw, lse=_T((B, H, S), (H * S, S, 1), "torch.float32"))
            assert SparseGqaFwdDslSm107(**kw2).scratch_workspace_bytes() == -(-o_part // 16) * 16 + -(-lse_part // 16) * 16
        else:
            assert a.scratch_workspace_bytes() == 0
    # a split with the gate operand: the UNGATED kernel + the gated combine
    kw = _decode_operands(B=B, S=1, H=H, KH=KH, split_kv=split_kv)
    kw.update(epilogue_gate=_T((B, 1, H, D), (H * D, H * D, D, 1), "torch.bfloat16"))
    g = SparseGqaFwdDslSm107(**kw)
    assert g.check_support() is True and g.template_params().epilogue_gate == (split_kv == 1)
    # bottom-right alone: per-token lists at the bottom-right positions
    kw = _operands(S=8)
    kw.update(bottom_right=True, seq_kv_lens=_T((1,), (1,), "torch.int32"))
    b = SparseGqaFwdDslSm107(**kw)
    assert b.check_support() is True
    assert (b.template_params().bottom_right, b.template_params().qsa_list_per_sequence) == (True, False)


def test_adapter_accepts_a_negative_scale_and_hands_it_to_the_kernel_signed(sm107_host_target):
    """A negative attn_scale is SERVED by the sparse core, not declined: the adapter keeps the sign (``scale`` is what the kernel
    multiplies every raw score by before its column max) and never raises the dense line's ``TemplateParams.negate_scores``
    (BMM1 negating Q at |scale|), which the sparse body does not consume -- so neither path drops the sign.  The Rubin cell
    ``test_a_negative_attn_scale_is_served_by_the_signed_scale`` runs the function; this is the host half of that claim."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    a = SparseGqaFwdDslSm107(**_operands(), scale=-0.0625)
    assert a.check_support() and a.scale == -0.0625
    assert a.template_params().negate_scores is False, "the sparse adapter passes the signed scale; negate_scores is the dense line's"


@pytest.mark.parametrize(
    "mutate, exc, word",
    [
        (lambda o: o.update(device_cc=(10, 0)), NotImplementedError, "Rubin-line"),
        # cc 8.0 is refused one gate EARLIER: the gather4 part floor (tile_dsl.requirements, cc 10.0+) answers before the record's
        # Rubin-line check does -- the typed decline names the instruction's floor, not the record
        (lambda o: o.update(device_cc=(8, 0)), NotImplementedError, "requires a cc 10.0+ part"),
        (lambda o: o.update(top_k=501), NotImplementedError, "top_k"),
        (lambda o: o.update(top_k=516), NotImplementedError, "top_k"),
        (lambda o: o.update(block_size=8), NotImplementedError, "block_size"),
        (lambda o: o.update(thd=True), ValueError, "seq_q_lens AND seq_kv_lens"),
        (lambda o: o.update(thd=True, seq_q_lens=_T((3,), (1,), "torch.int32"), seq_kv_lens=_T((2,), (1,), "torch.int32")), ValueError, "same B"),
        (lambda o: o.update(thd=True, seq_q_lens=_T((3,), (1,), "torch.int64"), seq_kv_lens=_T((3,), (1,), "torch.int32")), ValueError, "rank-1"),
        (
            lambda o: o.update(**_operands(B=2), thd=True, seq_q_lens=_T((2,), (1,), "torch.int32"), seq_kv_lens=_T((2,), (1,), "torch.int32")),
            ValueError,
            "batch extent 1",
        ),
        (lambda o: o.update(cu_seq_q_lens=True), NotImplementedError, "cu_seq_lens"),
        (lambda o: o.update(**_paged_operands(P=6)), NotImplementedError, "page_size"),  # the paged arm's one typed decline: a block would straddle pages
        (
            lambda o: o.update(**_paged_operands(), thd=True, seq_q_lens=_T((1,), (1,), "torch.int32")),
            NotImplementedError,
            "thd=True with paged_kv=True",
        ),  # the two wired arms are served one at a time: the packed sequence's K / V row offset composes with a dense tensor only
        # the fused epilogue gate is SERVED (the record claims it); a malformed gate operand is a ValueError naming the gate
        (lambda o: o.update(epilogue_gate=_T((1, 8, 24, D), (8 * 24 * D, 24 * D, D, 1), "torch.float32")), ValueError, "epilogue gate dtype"),
        (lambda o: o.update(epilogue_gate=_T((1, 8, 12, D), (8 * 12 * D, 12 * D, D, 1), "torch.bfloat16")), ValueError, "O-shaped"),
        (lambda o: o.update(epilogue_gate=_T((1, 8, 24, D), (8 * 24 * D, 24 * D, D + 4, 1), "torch.bfloat16")), ValueError, "multiples of 8 elements"),
        (lambda o: o.update(epilogue_gate=_T((1, 8, 24, D), (8 * 24 * D, 24 * D, D, 2), "torch.bfloat16")), ValueError, "contiguous (stride 1)"),
        # the decode form's arms are SERVED (split_kv, bottom_right, list_per_sequence); their unserved compositions decline by name
        (lambda o: o.update(split_kv=18), NotImplementedError, "split_kv=18 exceeds the item's maximum tile count"),
        (lambda o: o.update(split_kv=4, top_k=64, block_ids=_T((8, 64), (64, 1), "torch.int32")), NotImplementedError, "ceil((top_k + 1) / 32) = 3"),
        (lambda o: o.update(split_kv=0), ValueError, "split_kv must be >= 1"),
        (lambda o: o.update(list_per_sequence=True), NotImplementedError, "list_per_sequence=True needs bottom_right=True"),
        (lambda o: o.update(list_per_sequence=True, bottom_right=True), NotImplementedError, "list_per_sequence serves S_q <= 4"),  # _operands: S = 8
        (lambda o: o.update(**_decode_operands(B=1, S=5)), NotImplementedError, "got S_q = 5 -- pass per-token lists"),  # the first refused row count
        (lambda o: o.update(**_decode_operands(B=1, S=5, P=16)), NotImplementedError, "got S_q = 5 -- pass per-token lists"),  # over paged pools too
        (
            lambda o: o.update(**_operands(S=4), list_per_sequence=True, bottom_right=True),
            ValueError,
            "list_per_sequence needs block_ids [B, top_k]",
        ),  # a per-token [B x S_q, top_k] list under the shared-list form
        (
            lambda o: (
                o.update(_operands(S=4)),
                o.update(list_per_sequence=True, bottom_right=True, block_ids=_T((1, 512), (512, 1), "torch.int32"), block_lens=_T((4,), (1,), "torch.int32")),
            ),
            ValueError,
            "list_per_sequence needs block_lens a contiguous int32 [B]",
        ),
        (
            lambda o: o.update(thd=True, seq_q_lens=_T((1,), (1,), "torch.int32"), seq_kv_lens=_T((1,), (1,), "torch.int32"), split_kv=2),
            NotImplementedError,
            "split_kv with thd=True is not served",
        ),
        (
            lambda o: o.update(thd=True, seq_q_lens=_T((1,), (1,), "torch.int32"), seq_kv_lens=_T((1,), (1,), "torch.int32"), bottom_right=True),
            NotImplementedError,
            "bottom_right with thd=True is not served",
        ),
        (
            lambda o: o.update(
                thd=True, seq_q_lens=_T((1,), (1,), "torch.int32"), seq_kv_lens=_T((1,), (1,), "torch.int32"), list_per_sequence=True, bottom_right=True
            ),
            NotImplementedError,
            "list_per_sequence with thd=True is not served",
        ),
        (lambda o: o.update(sink=object()), NotImplementedError, "sink"),
        (lambda o: o.update(window_left=128), NotImplementedError, "sliding_window"),
        (lambda o: o.update(seq_q_lens=object()), NotImplementedError, "seq_q_lens"),
        (lambda o: o.update(include_open_block=False), NotImplementedError, "pure_list"),
        (lambda o: o.update(stats_log2=True), NotImplementedError, "stats_log2"),
        (
            lambda o: o.update(**{k: _T(v.shape, v.stride(), "torch.float32") for k, v in _operands().items() if k in ("q", "k", "v", "o")}),
            NotImplementedError,
            "bf16 or f16",
        ),
        (
            lambda o: o.update(
                q=_T((1, 8, 24, 128), (8 * 24 * 128, 24 * 128, 128, 1), "torch.bfloat16"),
                o=_T((1, 8, 24, 128), (8 * 24 * 128, 24 * 128, 128, 1), "torch.bfloat16"),
            ),
            NotImplementedError,
            "exactly",
        ),
        (
            lambda o: o.update(k=_T((1, 8, 1, D), (8 * D, D, D, 1), "torch.bfloat16"), v=_T((1, 8, 1, D), (8 * D, D, D, 1), "torch.bfloat16")),
            NotImplementedError,
            "group",
        ),
        (
            lambda o: o.update(
                k=_T((1, 8, 2, D), (8 * 2 * D, 2 * D, 100, 1), "torch.bfloat16"), v=_T((1, 8, 2, D), (8 * 2 * D, 2 * D, 100, 1), "torch.bfloat16")
            ),
            ValueError,
            "head stride",
        ),
        (
            lambda o: o.update(
                k=_T((1, 8, 2, D), (8 * 2 * D, 2 * D + 4, D, 1), "torch.bfloat16"), v=_T((1, 8, 2, D), (8 * 2 * D + 4, 2 * D + 4, D, 1), "torch.bfloat16")
            ),
            ValueError,
            "token stride",
        ),
        (lambda o: o.update(block_ids=_T((8, 512), (1024, 1), "torch.int64")), ValueError, "int32"),
        (lambda o: o.update(block_ids=_T((8, 256), (256, 1), "torch.int32")), ValueError, "block_ids must be"),
        # Q is TMA-loaded: a (batch, seq, head) stride off the 16-byte grid (seq stride 24 x 256 + 4) is refused like the gate's
        (lambda o: o.update(q=_T((1, 8, 24, D), (8 * (24 * D + 4), 24 * D + 4, D, 1), "torch.bfloat16")), ValueError, "multiples of 8 elements"),
    ],
)
def test_adapter_declines_every_unserved_request_by_name(mutate, exc, word, sm107_host_target):
    """Every field of the record has a typed decline naming the feature (NotImplementedError = an unserved feature,
    ValueError = a malformed operand); nothing falls through to the kernel."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    o = _operands()
    mutate(o)
    with pytest.raises(exc, match=re.escape(word)):
        SparseGqaFwdDslSm107(**o).check_support()


@pytest.mark.parametrize(
    "mutate, exc, word",
    [
        (lambda o: o.update(page_size=6), NotImplementedError, "positive multiple of 4"),
        (lambda o: o.update(page_size=0), NotImplementedError, "positive multiple of 4"),
        (lambda o: o.update(page_size=-16), NotImplementedError, "positive multiple of 4"),
        (lambda o: o.update(block_table=None), ValueError, "paged_kv needs block_table"),
        (lambda o: o.update(seq_kv_lens=None), ValueError, "paged_kv needs seq_kv_lens"),
        (lambda o: o.update(block_table=_T((1, 4), (4, 1), "torch.int64")), ValueError, "block_table must be int32"),
        (lambda o: o.update(block_table=_T((2, 4), (4, 1), "torch.int32")), ValueError, r"block_table must be a contiguous int32 \(B, max_pages\)"),
        (lambda o: o.update(block_table=_T((4,), (1,), "torch.int32")), ValueError, r"block_table must be a contiguous int32 \(B, max_pages\)"),
        (
            lambda o: o.update(
                k=_T((6, 2, 32, D), _pool_strides(2, 32, True), "torch.bfloat16"), v=_T((6, 2, 32, D), _pool_strides(2, 32, True), "torch.bfloat16")
            ),
            ValueError,
            "page axis is 32",
        ),
        (
            lambda o: o.update(
                k=_T((6, 2, 16, D), (2 * 16 * D, 16 * D, D + 4, 1), "torch.bfloat16"), v=_T((6, 2, 16, D), (2 * 16 * D, 16 * D, D + 4, 1), "torch.bfloat16")
            ),
            ValueError,
            "token stride",
        ),
        (
            lambda o: o.update(
                k=_T((6, 2, 16, D), (2 * 16 * D + 8, 16 * D, D, 1), "torch.bfloat16"), v=_T((6, 2, 16, D), (2 * 16 * D + 8, 16 * D, D, 1), "torch.bfloat16")
            ),
            ValueError,
            "whole rows per page",
        ),
        (
            lambda o: o.update(k=_T((6, 2, 16, D), (16 * D, 16 * D, D, 1), "torch.bfloat16"), v=_T((6, 2, 16, D), (16 * D, 16 * D, D, 1), "torch.bfloat16")),
            ValueError,
            "do not fit the page",
        ),
        (
            lambda o: o.update(
                k=_T((6, 2, 16, D), (16 * 2 * D, 100, 2 * D, 1), "torch.bfloat16"), v=_T((6, 2, 16, D), (16 * 2 * D, 100, 2 * D, 1), "torch.bfloat16")
            ),
            ValueError,
            "multiple of 64 elements",
        ),
        (
            lambda o: o.update(
                k=_T((6, 2, 16, D), (16 * 2 * D, 2 * D + 64, 2 * D, 1), "torch.bfloat16"),
                v=_T((6, 2, 16, D), (16 * 2 * D, 2 * D + 64, 2 * D, 1), "torch.bfloat16"),
            ),
            ValueError,
            "neither a multiple",
        ),
        (
            lambda o: o.update(
                k=_T((6, 2, 16, D), (16 * 2 * D, D, 2 * D, 2), "torch.bfloat16"), v=_T((6, 2, 16, D), (16 * 2 * D, D, 2 * D, 2), "torch.bfloat16")
            ),
            ValueError,
            "head dim must be contiguous",
        ),
        (lambda o: o.update(k=_T((6, 2, 16, D), _pool_strides(2, 16, True), "torch.float16")), ValueError, "dtype"),
    ],
)
def test_adapter_paged_form_checks(mutate, exc, word, sm107_host_target):
    """Every malformed paged request is refused by name before any kernel import: the page size (the one unserved FORM, a
    NotImplementedError), a missing table or length, a malformed table, and every pool stride the 2-D gather map cannot
    express (the page axis, the KV head count, the token / page / head strides, the head dim, the dtype)."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    o = _paged_operands()
    mutate(o)
    with pytest.raises(exc, match=word):
        SparseGqaFwdDslSm107(**o).check_support()


def test_adapter_refuses_the_table_and_the_page_size_on_a_dense_declaration(sm107_host_target):
    """The mirror image: ``block_table`` / ``page_size`` belong to the paged read; a dense declaration refuses them rather than
    ignoring them, and a dense declaration refuses a table at execute (never a silent re-specialization)."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    o = _operands()
    o.update(block_table=_T((1, 2), (2, 1), "torch.int32"))
    with pytest.raises(ValueError, match="block_table given but paged_kv=False"):
        SparseGqaFwdDslSm107(**o).check_support()
    o = _operands()
    o.update(page_size=16)
    with pytest.raises(ValueError, match="page_size=16 given but paged_kv=False"):
        SparseGqaFwdDslSm107(**o).check_support()
    a = SparseGqaFwdDslSm107(**_operands())
    a.check_support()
    with pytest.raises(ValueError, match="block_table was not declared"):
        a._bind("block_table", _T((1, 2), (2, 1), "torch.int32"), a.block_table, required=False)


def test_adapter_accepts_a_packed_thd_request_and_sizes_its_workspace(sm107_host_target):
    """``thd=True`` on packed operands (batch extent 1) with the two length tensors -- per-sequence ``[B]`` or cumulative
    ``[B + 1]`` per flag -- is served: the params carry ``thd_varlen`` (and the per-batch KV lengths the arm forces), the
    workspace is the THD metadata (``4 B + 4`` int32 words, 16-byte rounded), and the dense request keeps ``thd_varlen`` off."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    i32 = "torch.int32"
    for cu_q, cu_kv in ((False, False), (True, False), (False, True), (True, True)):
        o = _operands(B=1, S=600)
        o.update(
            thd=True,
            seq_q_lens=_T((3 + int(cu_q),), (1,), i32),
            seq_kv_lens=_T((3 + int(cu_kv),), (1,), i32),
            cu_seq_q_lens=cu_q,
            cu_seq_kv_lens=cu_kv,
        )
        a = SparseGqaFwdDslSm107(**o)
        assert a.check_support() is True
        p = a.template_params()
        assert p.thd_varlen and p.seq_kv_lens_present and (p.qsa_block_topk, p.qh_per_kh) == (512, 12)
        assert a.scratch_workspace_bytes() == 64, "THD metadata: (4 x 3 + 4) int32 words = 64 B"
    # the same operands without thd: no workspace, no THD specialization
    a = SparseGqaFwdDslSm107(**_operands(B=1, S=600))
    assert a.check_support() and a.scratch_workspace_bytes() == 0 and not a.template_params().thd_varlen


def test_adapter_declines_a_padded_kv_batch_stride_at_B2_but_not_at_B1(sm107_host_target):
    """The gather map is 2-D (tokens of every batch as rows): a batch stride other than S_kv x the token stride has no row
    coordinate -- declined at B >= 2, irrelevant at B == 1."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    o = _operands(B=2)
    k = _T((2, 8, 2, D), (16 * 2 * D, 2 * D, D, 1), "torch.bfloat16")  # batch stride 2x what the extent needs
    o.update(k=k, v=k)
    with pytest.raises(ValueError, match="batch stride"):
        SparseGqaFwdDslSm107(**o).check_support()
    o1 = _operands(B=1)
    k1 = _T((1, 8, 2, D), (16 * 2 * D, 2 * D, D, 1), "torch.bfloat16")
    o1.update(k=k1, v=k1)
    assert SparseGqaFwdDslSm107(**o1).check_support()


def test_adapter_refuses_a_misaligned_operand_base_by_name():
    """Every operand pointer reaches the kernel with ``assumed_align=16`` (a TMA tensor map's base, the bulk-copy source of an ids
    row), so a view that starts a few elements into its storage is refused BY NAME -- at execute for the bound tensors and at
    check_support for a declared tensor; a storage-free declaration has no address yet and is skipped.  Host-only: the rule is
    about addresses, not devices."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseOperandDesc, _require_16b_base

    base = torch.empty(64 + 16, dtype=torch.bfloat16)
    k0 = next(k for k in range(8) if (base.data_ptr() + 2 * k) % 16 == 0)  # the first element on a 16-byte boundary
    _require_16b_base("Q", base[k0 : k0 + 64].view(1, 1, 8, 8))  # aligned: silent
    with pytest.raises(ValueError, match="Q must start at a 16-byte aligned address"):
        _require_16b_base("Q", base[k0 + 1 : k0 + 65].view(1, 1, 8, 8))  # 2 bytes off the boundary
    _require_16b_base("K", SparseOperandDesc((1, 8, 2, 256), (4096, 512, 256, 1), torch.bfloat16))  # no storage: nothing to check
    _require_16b_base("the epilogue gate", None)
    # the call sites: the six pointers of a launch at execute and the declared tensors at check_support
    src = _adapter_source()
    assert src.count("_require_16b_base(name, t)") == 2, "execute and check_support each run the alignment rule over the operand list"
    for name in ('("Q", q)', '("K", k)', '("V", v)', '("O", o)', '("block_ids", ids)', '("the epilogue gate", gate)'):
        assert name in src, f"execute binds {name} through the alignment rule"


def test_adapter_dsl_gate_declines_by_name_before_the_kernel_import(monkeypatch):
    """Rule 7: on a cc 10.7 part a DSL without the ``sm_107a`` target is refused by NAME in check_support, before the kernel
    module is imported (the version state is substituted, not the wheel)."""
    import cudnn.frost.buffers as buffers
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: False)
    with pytest.raises(NotImplementedError, match="sm_107a"):
        SparseGqaFwdDslSm107(**_operands()).check_support()
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.6.2")))
    with pytest.raises(NotImplementedError, match=re.escape("found 4.6.2")):
        SparseGqaFwdDslSm107(**_operands()).check_support()


# ============================================================================ host: source pins of the kernel body
def test_sparse_kernel_source_pins():
    """The sm107 conventions the body must keep: the descriptor version derived (no literal) on EVERY SMEM tile, one module
    spin constant on every per-tile ring wait, the LOAD wait between the S^T read and the ``s_empty`` arrive, the one helper
    at every tile-looping role."""
    src = _kernel_source()
    code = _code(src)
    assert "desc_version=DESC_VERSION" in code and "desc_version=0" not in code and "desc_version=1" not in code
    assert code.count("SmemTile(") == code.count("desc_version=DESC_VERSION") == 4, "every SmemTile takes the module descriptor version"
    # The per-TILE ring waits (the gather warps' kv_empty x 3; the MMA warp's kv_full x 3, s_empty x 2, p_full x 1; the softmax
    # warps' s_full and the in-loop bmm2_done) take the module constant; the per-ITEM waits (q_full, ids_full / ids_empty, the
    # epilogue's bmm2_done) and the exit drains keep the default form.
    assert code.count("spin=SPIN_RING_WAITS") == 11, "every per-tile ring wait takes the module constant, nothing else"
    assert "SPIN_RING_WAITS: bool = False" in code
    # the LOAD wait orders the s_empty arrive after the S^T read (frost-kernels.md section 3)
    i_ld = code.index('s_raw = nvvm.tcgen05_ld("32x32b"')
    i_wait = code.index("nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)", i_ld)
    i_arr = code.index("bars.mb_s_empty[par].arrive()", i_ld)
    assert i_ld < i_wait < i_arr
    # the one helper at the item start of every role that loops over tiles
    for role in ("_gather_warp_group", "_mma_warp_group", "_softmax_warp_group"):
        body = code[code.index(f"def {role}(") :]
        body = body[: body.index("\ndef ", 1) if "\ndef " in body[1:] else len(body)]
        assert "= _item_bounds(" in body, f"{role} must derive its bounds from _item_bounds"
    # no per-item tile-index parity anywhere (the decode tile's arithmetic form)
    assert "_kv_slot(" not in code and "// cutlass.Int32(2)) & cutlass.Int32(1)" not in code
    assert "EXPLICIT_ABI = True" in code
    # the S^T / P^T ring depth is read ONCE from the config and spelled at every ring site -- never a literal 2 (a depth changed at some
    # sites and not others is a stale S^T slot read from the first reuse): every advance() names a depth constant, the four per-tile
    # MBarriers and the sP tile take STAGES_S, and the only other depths are the CFG ring fields
    assert "STAGES_S = CFG.STAGES_S" in code and "RED_SLOTS = STAGES_S + 1" in code
    assert re.search(r"advance\(\w+, \d+\)", code) is None, "every PipelineState advance takes a named ring depth, never a literal"
    assert set(re.findall(r"advance\(\w+, (\w+)\)", code)) == {"STAGES_S", "STAGES_KV", "STAGES_Q", "STAGES_IDS", "SCHEDULER_STAGES"}
    assert code.count("_alloc(STAGES_S), stages=STAGES_S") == 4 and "_alloc(2)" not in code and "stages=2" not in code
    assert "range_constexpr(STAGES_S)" in code and "range_constexpr(2)" not in code and "STAGES_S * pBufferWords" in code


def test_sparse_kernel_thd_arm_source_pins():
    """The THD arm's conventions, no GPU: the scheduler form is a ``const_expr`` arm of ``CFG.THD_VARLEN`` (the persistent
    claim counter vs the CLC loop); a CTA past the live total exits at entry BEFORE any mbarrier init; ONE decode helper
    turns every payload (the first item included) into the item in every role; the three load sites take their coordinates
    from it and nothing else (the ids bulk copy ``item_row``, the Q box ``(out_tok, out_batch)``, the gather rows
    ``kv_row_base`` on the dense arm -- the paged arm's pool row never composes with it: THD x paged is a typed decline) -- no
    dense ``batch x S`` coordinate survives outside the decode; the metadata offsets come from
    ``tile_dsl.thd``'s names, never re-literalled; the setup launch precedes the main launch."""
    code = _code(_kernel_source())
    i_persist = code.index(
        "scheduler_warp_loop_persistent(sched, SCHEDULER_STAGES, is_cga_first_cta, seq_kv_lens_tensor, THD_CTR_OFF(n_batch), THD_LIVE_OFF(n_batch), 1, 1)"
    )
    arm = code[code.rfind("if cutlass.const_expr(CFG.THD_VARLEN == 1):", 0, i_persist) : i_persist]
    assert arm and "\ndef " not in arm, "the persistent scheduler is the THD arm of the scheduler warp"
    assert "scheduler_warp_loop(sched, SCHEDULER_STAGES, is_cga_first_cta, 1)" in code, "the CLC form stays the dense arm"
    assert code.index("exit_if_dead_thd_cluster(seq_kv_lens_tensor, n_batch, 1)") < code.index(
        "bars = make_sparse_bars()"
    ), "the dead-CTA exit precedes every init"
    assert code.count("= _decode_item(") == 8, "the prologue and the loop bottom of each of the four roles decode through the ONE helper"
    assert "_decode_payload" not in code
    assert "tma_q(cutlass.Int32(0), head * cutlass.Int32(G), out_tok, out_batch)" in code
    assert "ids_base + item_row * cutlass.Int32(BLOCK_TOPK)" in code
    assert "base = kv_row_base + key0" in code and "base + cutlass.Int32(r)" in code, "the dense arm's gather row is kv_row_base + 4 blk + r"
    i_dec = code.index("def _decode_item(")
    i_end = code.index("\ndef ", i_dec + 1)
    outside = code[:i_dec] + code[i_end:]
    assert "batch * seqlen_kv" not in outside and "batch * seqlen_q" not in outside, "a dense coordinate outside the decode helper"
    for name in ("THD_CTR_OFF(n_batch)", "THD_LIVE_OFF(n_batch)", "THD_CU_Q_TOTAL_OFF(n_batch)", "THD_META_WORDS(B)"):
        assert name in code, name
    assert re.search(r"4\s*\*\s*n_batch\s*\+\s*[23]\b", code) is None, "the metadata offsets are tile_dsl.thd's, never re-literalled"
    assert code.index("_build_thd_meta_kernel(") < code.index("    _kernel(\n        tma_q_desc,"), "the setup launch precedes the main launch"


def test_sparse_kernel_declares_no_cross_cta_arrive_and_no_cluster_scope():
    src = _code(_kernel_source())
    assert "arrive_on_peer" not in src and "arrive_on_leader" not in src and "MemScope.CLUSTER" not in src
    assert "cta_group=1" in src and "cta_group=2" not in src


def test_sparse_kernel_empty_row_guard_floor_mask_and_hygiene_pins():
    """sdpa-invariants sections 2-5 and 10 on the source, no GPU: the empty-row SELECT (the family grep
    ``grep -L 'row_dead\\|_kv_empty\\|_row_empty'`` must not print this kernel); the denominator floor sits inside the log and
    the reciprocal only and is followed by the ``row_dead`` SELECT (``* inv`` never sees residue: the O select is a SELECT on
    ``dead_cols``, never a multiply by zero); no atomic (no amax output); the mask is ONE per-lane validity predicate with the
    three terms (block >= 0, key <= position, key < the sequence length) -- the key index is per lane and all 16 columns share the
    token's position, so the bit-word chunk mask of the dense tiles does not apply; the loop bounds are derived; and no
    module-level assert in the kernel or the adapter (structural invariants live in the raising config validator)."""
    code = _code(_kernel_source())
    assert "row_dead" in code, "the empty-row guard the family grep looks for"
    assert code.count("cutlass.Float32(1e-30)") == 1 and "TINY = cutlass.Float32(1e-30)" in code, "one floor constant, defined once"
    assert code.count("cute.math.max(l_tot, TINY)") == 2, "the floor on the log and the reciprocal only"
    assert code.count("_select_f32(row_dead, NEG_INF, lse_j)") == 1 and code.count("_select_f32(row_dead, ZERO, inv_j)") == 1
    assert "_select_f32(dead_cols[j], ZERO, cutlass.Float32(o_vals[j]) * inv_cols[j])" in code, "O's empty case is a SELECT, not residue * 0"
    assert "atomic" not in code.lower()
    assert "valid = (blk >= cutlass.Int32(0)) & (key_abs <= pos) & (key_abs < eff_seqlen_kv)" in code
    assert "apply_mask_chunk" not in code, "the mask is per lane (one predicate, 16 selects), not the per-column bit-word form"
    assert code.count("cutlass.range(0, n_tiles, 1, unroll=1)") == 3, "every tile loop runs the derived n_tiles (gather, MMA, softmax)"
    assert "n_tiles = cute.math.max(cutlass.Int32(1)," in code, "a dead item runs ONE clamped tile (never a zero-trip arm)"
    for name, src in (("kernel", code), ("adapter", _code(_adapter_source()))):
        assert re.search(r"^assert\b", src, re.M) is None, f"{name}: no module-level assert"


def test_sparse_kernel_decode_form_source_pins():
    """The decode form's shape in the source (no GPU): the three arms are ``const_expr`` folds of module constants read from CFG
    (the dense rendering traces the prefill program), ONE decode helper carries the split axis (x = tok + S_q x split) and the
    per-sequence list row, the ONE bounds helper carries the anchor, the 0-2 tail ids and the split chunk (tile0 / n_sp, the empty
    chunk a DEAD item through the SAME clamp), every list read offsets by the chunk's first tile, the epilogue stores at the
    split-major partial batch b + s x B through the same store path, the grid is (S_q x SPLIT_KV, KH, B), the compile entry's O
    pointer is fp32 under a split and the per-split LSE is required; no new barrier, SMEM tile or ring wait (the counts stay 15 /
    4 / 11)."""
    code = _code(_kernel_source())
    common = _code(_item_common_source())  # the visible-set arithmetic lives in the shared helper (one helper for both bodies)
    for name in ("SPLIT_KV = CFG.SPLIT_KV", "LIST_PER_SEQUENCE = CFG.LIST_PER_SEQUENCE", "BOTTOM_RIGHT = CFG.BOTTOM_RIGHT"):
        assert name in code, name
    assert code.count("cutlass.const_expr(SPLIT_KV > 1)") >= 5, "the split arm folds at the decode, the bounds, both list-read sites and the store"
    # the per-sequence / bottom-right arms fold in the kernel (the decode, the host, the store) AND in the shared helper (as constexpr parameters)
    assert code.count("cutlass.const_expr(LIST_PER_SEQUENCE == 1)") >= 2 and code.count("cutlass.const_expr(BOTTOM_RIGHT == 1)") == 1
    assert common.count("cutlass.const_expr(list_per_sequence == 1)") >= 2 and common.count("cutlass.const_expr(bottom_right == 1)") == 2
    assert "live_row = tok < eff_seqlen_q" in code, "under the bottom-right diagonal the store predicate is the Q row's liveness"
    # the prefill form keeps its own tail spelling (one tail id at most) so the dense rendering traces the same program as before
    assert (
        "n_tail = _select_i32((n_vis % bs) != zero, 1, 0)" in common
        and "tail = _select_i32((idx == count) & (n_tail != cutlass.Int32(0)), tail_lo, cutlass.Int32(-1))" in common
    )
    assert "split = cute.arch.make_warp_uniform(x // seqlen_q)" in code and "tok = x - split * seqlen_q" in code
    assert "return tok, head, batch, item_row, kv_row_base, eff_seqlen_q, out_tok, out_batch, split" in code
    # the one helper: the anchor / tail count in the shared helper, the chunk cut here (the tcgen05 geometry)
    for frag in (
        "pos = tok + (eff_seqlen_kv - eff_seqlen_q)",
        "anchor = eff_seqlen_kv - eff_seqlen_q",
        "n_tail = cute.math.max((n_vis + bs - one) // bs - tail_lo, zero)",
        "return pos, count, n_tail, tail_lo, dead, n_listed, eff_seqlen_kv",
    ):
        assert frag in common, frag
    i_ib = code.index("def _item_bounds(")
    ib = code[i_ib : code.index("\ndef ", i_ib + 1)]
    for frag in (
        "= visible_bounds(",
        "tile0 = split * per + cute.math.min(split, rem)",
        "dead = dead | (n_sp == zero)",
        "n_tiles = cute.math.max(cutlass.Int32(1), n_sp)",
        "return pos, count, n_tail, tail_lo, dead, n_tiles, eff_seqlen_kv, tile0",
    ):
        assert frag in ib, frag
    assert "tail = _select_i32((rel >= cutlass.Int32(0)) & (rel < n_tail), tail_lo + rel, cutlass.Int32(-1))" in common
    assert "from cudnn.sdpa.fwd.kernels.sm107._sparse_item_common import _select_f32, _select_i32, block_id_at, visible_bounds" in code
    # the list reads offset by the chunk's first tile; the stores at the partial batch
    assert "_gather_block_ids(sIds_raw, slot_base, tile0, w, count, n_tail, tail_lo, dead)" in code
    assert "t_next = tile0 + t_next" in code and "t_abs = tile0 + i" in code
    assert "part_batch = out_batch + split * n_batch" in code
    # All four gated/ungated store arms widen BEFORE forming an address; a narrow row subview can wrap a legal O stride.
    assert code.count("cutlass.Int64(part_batch)") == 4 and "oo[part_batch" not in code
    assert "lse_arr[part_batch, head_base + col, out_tok] = lse_cols[j]" in code  # col = the group's head row (j at one column group)
    # the host: the grid, the partial slabs, the per-sequence list rows, the fp32 pointer + the LSE requirement
    assert "grid_shape = (SQ * SPLIT_KV, KH, B)" in code and "n_o_batches = B * SPLIT_KV" in code
    assert "n_list_rows = B if cutlass.const_expr(LIST_PER_SEQUENCE == 1) else n_q_batches * SQ" in code
    assert "P(cutlass.Float32) if SPLIT_KV > 1 else P(STORAGE_DTYPE)" in code
    assert 'raise ValueError("sparse_d256_f16: split_kv > 1 requires has_lse=True' in code
    # no new ring, tile or wait
    assert code.count("MBarrier(_alloc(") == 13 and code.count("SmemTile(") == 4 and code.count("spin=SPIN_RING_WAITS") == 11


# ============================================================================ the reference (torch; any device)
def _visible_weights(ids, lens, positions, L, SKV, top_k):
    """``[rows, SKV]`` int64: how many times the kernel attends each key of each row -- every list entry below the count that
    is ``>= 0`` and inside the tensor contributes its block's keys once (an id listed twice contributes twice), the open tail
    of the VISIBLE range (``min(pos + 1, L)``) contributes once more, and every key at or past the visible range is removed.
    On a contract-clean list every weight is 0 or 1 and ``weights > 0`` is exactly the oracle's ``qsa_visible_mask``
    (asserted per cell by :func:`_reference`); the multiplicity matters only for the documented duplicate contract."""
    rows, width = ids.shape
    dev = ids.device
    p = positions.to(torch.long)
    kv = torch.full((rows,), int(L), dtype=torch.long, device=dev)
    n_vis = torch.minimum(p + 1, kv).clamp_min(0)
    n_sel_default = torch.minimum(torch.div(p + 1, BS, rounding_mode="floor").clamp_min(0), torch.full_like(p, int(top_k)))
    count = n_sel_default if lens is None else torch.minimum(lens.to(torch.long).clamp_min(0), n_sel_default)
    n_blocks_total = -(-int(SKV) // BS)
    idl = ids.to(torch.long)
    read = torch.arange(width, device=dev)[None, :] < count[:, None]
    live = read & (idl >= 0) & (idl < n_blocks_total)
    target = torch.where(live, idl, torch.full_like(idl, n_blocks_total))
    mult = torch.zeros(rows, n_blocks_total + 1, dtype=torch.long, device=dev)
    mult.scatter_add_(1, target, live.to(torch.long))
    t = torch.arange(int(SKV), device=dev)
    w = mult[:, :n_blocks_total][:, torch.div(t, BS, rounding_mode="floor")]
    tail_start = torch.div(n_vis, BS, rounding_mode="floor") * BS
    w = w + (t[None, :] >= tail_start[:, None]).to(torch.long)
    return torch.where(t[None, :] < n_vis[:, None], w, torch.zeros_like(w))


def _reference(q, k, v, ids, lens, kv_lens, scale, top_k, *, oracle_check=True, row_chunk=1024):
    """fp32 over the half-rounded operands; the oracle's visible set per row (``oracle_check``: every chunk's ``weights > 0`` is
    asserted equal to ``qsa_visible_mask``), natural-log LSE; a dead row is O = 0, LSE = -inf.  Chunked over rows so the
    32K cells fit."""
    oracle = _oracle()
    B, S, H, _ = q.shape
    SKV, KH = k.shape[1], k.shape[2]
    G = H // KH
    dev = q.device
    ids3 = ids.reshape(B, S, -1)
    lens2 = None if lens is None else lens.reshape(B, S)
    ref_o = torch.zeros(B, S, H, D, device=dev, dtype=torch.float32)
    ref_lse = torch.full((B, H, S), float("-inf"), device=dev, dtype=torch.float32)
    for b in range(B):
        L = SKV if kv_lens is None else int(kv_lens[b])
        kk = [k[b, :, g].float() for g in range(KH)]
        vv = [v[b, :, g].float() for g in range(KH)]
        for lo in range(0, S, row_chunk):
            hi = min(lo + row_chunk, S)
            pos = torch.arange(lo, hi, device=dev)
            w = _visible_weights(ids3[b, lo:hi], None if lens2 is None else lens2[b, lo:hi], pos, L, SKV, top_k)
            if oracle_check:
                allowed = oracle.qsa_visible_mask(ids3[b, lo:hi], None if lens2 is None else lens2[b, lo:hi], pos, L, SKV, BS, top_k=top_k)
                assert torch.equal(w > 0, allowed), "the multiplicity reference must agree with the oracle's visible-set rule on this list"
            wf = w.to(torch.float32)
            allowed = w > 0
            for h in range(H):
                sc = (q[b, lo:hi, h].float() @ kk[h // G].t()) * scale
                sc = sc.masked_fill(~allowed, float("-inf"))
                rmax = sc.amax(dim=-1)
                dead = torch.isinf(rmax) & (rmax < 0)
                safe = torch.where(dead, torch.zeros_like(rmax), rmax)
                p = torch.where(allowed, torch.exp(sc - safe[:, None]) * wf, torch.zeros_like(sc))
                den = p.sum(dim=-1)
                oc = (p @ vv[h // G]) / torch.where(dead, torch.ones_like(den), den)[:, None]
                ref_o[b, lo:hi, h] = torch.where(dead[:, None], torch.zeros_like(oc), oc)
                ref_lse[b, h, lo:hi] = torch.where(dead, torch.full_like(den, float("-inf")), safe + torch.log(den))
    return ref_o, ref_lse


def test_visible_weights_match_the_oracle_rule_on_canonical_lists():
    """Host, CPU: on contract-clean lists (full, shuffled, with per-sequence KV lengths, with and without ``block_lens``) the
    multiplicity reference is 0 / 1 and ``weights > 0`` is the oracle's mask; a duplicated id and a listed open block are the
    two multiplicity-2 cases, exactly where the kernel's "counted twice" contract says so."""
    oracle = _oracle()
    g = torch.Generator().manual_seed(3)
    S, top_k = 300, 64
    pos = torch.arange(S)
    for kv_len in (None, 257):
        kvl = None if kv_len is None else torch.tensor([kv_len])[:, None].expand(1, S)
        for kind in ("full", "shuffled"):
            if kind == "full":
                ids, lens = oracle.full_block_ids(pos[None], top_k, BS, kv_lens=kvl)
            else:
                ids, lens = oracle.random_block_ids(pos[None], top_k, BS, generator=g, kv_lens=kvl, shuffle=True)
            L = S if kv_len is None else kv_len
            for given in (lens[0], None):
                w = _visible_weights(ids[0], given, pos, L, S, top_k)
                assert int(w.max()) == 1
                assert torch.equal(w > 0, oracle.qsa_visible_mask(ids[0], given, pos, L, S, BS, top_k=top_k))
    # duplicates: entry 1 of every row with >= 2 entries repeats entry 0 -> block 0's keys weigh 2 (the set rule stays 1)
    ids, lens = oracle.full_block_ids(pos[None], top_k, BS)
    dup = ids[0].clone()
    rows = lens[0] >= 2
    dup[rows, 1] = dup[rows, 0]
    w = _visible_weights(dup, lens[0], pos, S, S, top_k)
    assert torch.equal(w[rows][:, :BS], torch.full((int(rows.sum()), BS), 2, dtype=torch.long))
    assert torch.equal(w > 0, oracle.qsa_visible_mask(dup, lens[0], pos, S, S, BS, top_k=top_k))
    # the open block listed IN PLACE of the last complete block (a contract violation the test-side detector rejects; an entry
    # appended past the derived count would never be read): its tail keys weigh 2, the replaced block's keys 0
    opn = ids[0].clone()
    has_open = (pos + 1) % BS != 0
    sel = has_open & (lens[0] >= 1)
    opn[sel, (lens[0, sel] - 1).long()] = ((pos[sel] + 1) // BS).to(torch.int32)
    w = _visible_weights(opn, lens[0], pos, S, S, top_k)
    tail = (torch.arange(S)[None, :] >= ((pos + 1) // BS * BS)[:, None]) & (torch.arange(S)[None, :] <= pos[:, None])
    assert torch.equal(w[sel][tail[sel]], torch.full((int(tail[sel].sum()),), 2, dtype=torch.long))
    assert torch.equal(w > 0, oracle.qsa_visible_mask(opn, lens[0], pos, S, S, BS, top_k=top_k))
    assert oracle.block_ids_contract_violations(opn, pos, S, block_size=BS, block_lens=lens[0])


# ============================================================================ Rubin: the launch / check machinery
def _stream():
    import cuda.bindings.driver as cuda

    return cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))


def _sentinel(dtype):
    return 1.5e30 if dtype == torch.bfloat16 else 6.0e4  # f16 cannot hold 1.5e30; no attention output reaches 6e4


def _stored_sentinel(dtype) -> float:
    """The sentinel AS THE OUTPUT DTYPE STORES IT: 1.5e30 is not a bf16 value (it lands at 1.4954e30), so a detector that
    compares the output against the Python float never fires on bf16 -- compare against the rounded value instead."""
    return float(torch.full((), _sentinel(dtype), dtype=dtype).float())


def _inputs(B, S, H, KH, dtype, seed=0, SKV=None):
    dev = torch.device("cuda")
    SKV = S if SKV is None else SKV
    g = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randn(B, S, H, D, device=dev, dtype=torch.float32, generator=g).to(dtype)
    k = torch.randn(B, SKV, KH, D, device=dev, dtype=torch.float32, generator=g).to(dtype)
    v = torch.randn(B, SKV, KH, D, device=dev, dtype=torch.float32, generator=g).to(dtype)
    return q, k, v, g


def _lists(kind, B, S, top_k, g, kv_lens=None):
    """``(block_ids int32 [B, S, top_k], block_lens int32 [B, S])`` from the oracle's builders: ``full`` (every complete block,
    in order) or ``shuffled`` (a random subset of ``top_k`` complete blocks in random order; the full set below the bound)."""
    oracle = _oracle()
    dev = torch.device("cuda")
    positions = torch.arange(S, device=dev).expand(B, S)
    kvl = None if kv_lens is None else torch.tensor(kv_lens, device=dev, dtype=torch.long)[:, None]
    if kind == "full":
        ids, lens = oracle.full_block_ids(positions, top_k, BS, kv_lens=kvl)
    else:
        ids, lens = oracle.random_block_ids(positions, top_k, BS, generator=g, kv_lens=kvl, shuffle=True)
    return ids.contiguous(), lens.contiguous()


def _launch(q, k, v, ids, lens, kv_lens, top_k, scale, launches=2, o_shape=None, gate=None):
    """Sentinel-filled O / LSE per launch through the adapter (``check_support`` -> ``compile`` -> ``execute``: the adapter
    compiles nothing on the execute path).  ``gate`` (appended) = the fused epilogue gate's operand, O-shaped in Q's dtype."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    dev = q.device
    B, S, H, _ = q.shape
    seq_kv = None if kv_lens is None else torch.tensor(kv_lens, device=dev, dtype=torch.int32)
    sent = _sentinel(q.dtype)
    outs = []
    for _ in range(launches):
        o = torch.full((B, S, H, D), sent, device=dev, dtype=q.dtype)
        lse = torch.full((B, H, S), sent, device=dev, dtype=torch.float32)
        a = SparseGqaFwdDslSm107(q=q, k=k, v=v, o=o, lse=lse, block_ids=ids, block_lens=lens, seq_kv_lens=seq_kv, top_k=top_k, scale=scale, epilogue_gate=gate)
        assert a.check_support()
        a.compile()  # plan time: the adapter never compiles on the execute path (the declared block_lens presence selects the variant)
        a.execute(stream=_stream())
        torch.cuda.synchronize()
        outs.append((o, lse))
    return outs


@requires_rubin
@pytest.mark.parametrize("mode", ["dense", "split", "thd"])
@pytest.mark.parametrize("gated", [False, True])
def test_valid_bindings_remain_capture_safe_with_changed_input_replays(mode, gated):
    plan = _device_validation_plan({"dense": "o", "split": "workspace", "thd": "seq_q_lens"}[mode])
    if not gated:
        plan.gate = None
    plan.v.fill_(3)
    plan.compile()
    plan.execute()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.execute()
    for value in (4, 6, 8):
        plan.v.fill_(value)
        plan.o.fill_(-17)
        graph.replay()
        torch.cuda.synchronize()
        expected = torch.zeros_like(plan.o)
        expected[:, :3] = value * (0.5 if gated else 1)
        assert torch.equal(plan.o, expected)
        assert torch.isneginf(plan.lse[:, :, 3]).all()
        expected_lse = torch.arange(1, 4, dtype=torch.float32, device=plan.q.device).log().expand(1, 2, 3)
        torch.testing.assert_close(plan.lse[:, :, :3], expected_lse, atol=ATOL, rtol=0)


@requires_rubin
@pytest.mark.parametrize("layout", ["dense", "paged", "thd"])
@pytest.mark.parametrize("bad_id", [2**29, 2**30, 2**31 - 1, -(2**31)])
@pytest.mark.parametrize("softmax_groups", [1, 2])
def test_extreme_block_ids_cannot_wrap_to_live_keys(layout, bad_id, softmax_groups):
    """Block IDs are bounded BEFORE multiplying by four, in both the gather and softmax paths."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    q = torch.zeros((1, 4, 2, D), dtype=torch.bfloat16, device="cuda")
    k = torch.zeros((1, 4, 1, D), dtype=q.dtype, device=q.device)
    v = torch.full_like(k, 3)
    o, lse = torch.full_like(q, -17), torch.full((1, 2, 4), -17.0, device=q.device)
    ids = torch.full((1, 4, 4), -1, dtype=torch.int32, device=q.device)
    ids[0, 3, 0] = bad_id
    lens = torch.zeros((1, 4), dtype=torch.int32, device=q.device)
    lens[0, 3] = 1
    args = dict(q=q, k=k, v=v, o=o, lse=lse, block_ids=ids, block_lens=lens, top_k=4, softmax_groups=softmax_groups)
    if layout == "paged":
        args.update(
            paged_kv=True,
            page_size=4,
            k=k.transpose(1, 2),
            v=v.transpose(1, 2),
            block_table=torch.zeros((1, 1), dtype=torch.int32, device=q.device),
            seq_kv_lens=torch.tensor([4], dtype=torch.int32, device=q.device),
        )
    if layout == "thd":
        args.update(
            thd=True, seq_q_lens=torch.tensor([4], dtype=torch.int32, device=q.device), seq_kv_lens=torch.tensor([4], dtype=torch.int32, device=q.device)
        )
    plan = SparseGqaFwdDslSm107(**args)
    assert plan.check_support()
    if plan.thd:
        plan.workspace = torch.empty(plan.scratch_workspace_bytes(), dtype=torch.uint8, device=q.device)
    plan.compile()
    for _ in range(2):
        o.fill_(-17)
        lse.fill_(-17)
        plan.execute(stream=_stream())
        torch.cuda.synchronize()
        assert torch.equal(o[0, 3], torch.zeros_like(o[0, 3]))
        assert torch.isneginf(lse[0, :, 3]).all()
        assert torch.equal(o[0, :3], torch.full_like(o[0, :3], 3))


@requires_rubin
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("softmax_groups", [1, 2])
@pytest.mark.parametrize("wide_stride", [True, False])
def test_output_address_uses_int64_before_any_subview(gated, softmax_groups, wide_stride):
    """Exercise a >32-bit stride AND a <31-bit stride whose batch-index product overflows; guard the old wrapped addresses."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    H, KH, S = 12, 1, 1
    B = 2 if wide_stride else 5
    stride = (2**32 if wide_stride else 2**30) + H * D
    offset = 0 if wide_stride else 2**31  # keeps even signed wrapped addresses inside this allocation in the RED run
    elements = offset + (B - 1) * stride + H * D
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    if free < elements * 2 + (1 << 30):
        pytest.skip("the wide-output regression needs an 8–12 GiB allocation plus 1 GiB headroom")
    try:
        storage = torch.empty(elements, dtype=torch.bfloat16, device="cuda")
    except torch.OutOfMemoryError:
        pytest.skip("the wide-output backing allocation no longer fits in available GPU memory")
    o = storage.as_strided((B, S, H, D), (stride, H * D, D, 1), storage_offset=offset)
    live_offsets = {offset + b * stride for b in range(B)}
    wrapped_offsets = set()
    for b in range(B):
        unsigned = (b * stride) % 2**32
        signed = unsigned if unsigned < 2**31 else unsigned - 2**32
        wrapped_offsets.update((offset + unsigned, offset + signed))
    guards = [storage[start : start + H * D] for start in sorted(wrapped_offsets - live_offsets)]
    q = torch.zeros((B, S, H, D), dtype=o.dtype, device=o.device)
    k = torch.zeros((B, S, KH, D), dtype=o.dtype, device=o.device)
    v = torch.arange(1, B + 1, device=o.device, dtype=torch.float32).to(o.dtype).view(B, 1, 1, 1).expand_as(k).contiguous()
    lse = torch.empty((B, H, S), device=o.device)
    ids = torch.full((B, S, 4), -1, dtype=torch.int32, device=o.device)
    lens = torch.zeros((B, S), dtype=torch.int32, device=o.device)
    gate = torch.zeros_like(q) if gated else None
    plan = SparseGqaFwdDslSm107(q=q, k=k, v=v, o=o, lse=lse, block_ids=ids, block_lens=lens, top_k=4, epilogue_gate=gate, softmax_groups=softmax_groups)
    assert plan.check_support()
    plan.compile()
    for _ in range(2):
        o.fill_(-17)
        for guard in guards:
            guard.fill_(-17)
        plan.execute(stream=_stream())
        torch.cuda.synchronize()
        expected = v.expand_as(q) * (0.5 if gated else 1)
        assert torch.equal(o, expected), "a valid output row was missed or written at a wrapped address"
        assert torch.equal(lse, torch.zeros_like(lse))
        assert all(torch.equal(guard, torch.full_like(guard, -17)) for guard in guards), "wrapped-address guard was overwritten"


def _check(outs, ref_o, ref_lse):
    """The standard assertions of every cell: finite, no sentinel on a live row, dead rows exact, within the budget on live
    rows, two launches bitwise.  Returns (max|dO|, max|dLSE| over live rows)."""
    (o, lse), *rest = outs
    sent, sent_o = _sentinel(o.dtype), _stored_sentinel(o.dtype)
    of = o.float()
    dead = torch.isinf(ref_lse)  # [B, H, S]
    assert torch.isfinite(of).all(), "non-finite O"
    assert not (of == sent_o).any(dim=-1).any(), "a sentinel survived on an output row"
    assert not (lse == sent).any(), "a sentinel survived in LSE"
    if dead.any():
        assert torch.isinf(lse[dead]).all() and (lse[dead] < 0).all(), "a dead row's LSE must be -inf exactly"
        assert (of.permute(0, 2, 1, 3)[dead] == 0).all(), "a dead row's O must be 0 exactly"
    live = ~dead
    max_o = float((of - ref_o).abs().max())
    max_lse = float((lse[live] - ref_lse[live]).abs().max()) if live.any() else 0.0
    assert max_o <= ATOL, f"O off the reference by {max_o} (budget {ATOL})"
    assert max_lse <= ATOL, f"LSE off the reference by {max_lse} (budget {ATOL})"
    for o2, lse2 in rest:
        assert torch.equal(o, o2) and torch.equal(lse, lse2), "two launches must be bitwise"
    return max_o, max_lse


def _run_cell(*, B, S, H, KH, dtype, top_k, list_kind, kv_lens=None, block_lens=True, seed=0, scale=None):
    q, k, v, g = _inputs(B, S, H, KH, dtype, seed)
    ids, lens = _lists(list_kind, B, S, top_k, g, kv_lens)
    scale = (1.0 / math.sqrt(D)) if scale is None else float(scale)  # appended: a cell may pass the SIGNED scale it wants served
    outs = _launch(q, k, v, ids, lens if block_lens else None, kv_lens, top_k, scale)
    ref_o, ref_lse = _reference(q, k, v, ids, lens if block_lens else None, kv_lens, scale, top_k)
    return _check(outs, ref_o, ref_lse)


def _dense_kernel(q, k, v, scale, kv_lens=None):
    """The dense sm107 d256 prefill kernel (top-left causal, the query head's KV head = head // group) on the same operands
    through the shared explicit pointer ABI: the FUNCTION twin of the sparse kernel at the identity bound and the
    ignore-the-list twin of the adversarial cell.  Returns sentinel-filled (O, LSE) after the launch."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    dev = q.device
    B, S, H, _ = q.shape
    SKV, KH = k.shape[1], k.shape[2]
    code = 2 if q.dtype == torch.bfloat16 else 3
    params = TemplateParams(dtype_qkv=code, window_right=0, qh_per_kh=H // KH, seq_kv_lens_present=kv_lens is not None)
    mod = _load_sm100_kernel_module((D, D), params, rubin=True)
    fn = mod.compile(d_qk=D, d_v=D, has_lse=True, lse_kind="dense")
    gmem = cute.AddressSpace.gmem
    half = cutlass.BFloat16 if code == 2 else cutlass.Float16

    def P(t, cdt, align=16):
        return None if t is None else make_ptr(cdt, t.data_ptr(), gmem, assumed_align=align)

    sent = _sentinel(q.dtype)
    o = torch.full((B, S, H, D), sent, device=dev, dtype=q.dtype)
    lse = torch.full((B, H, S), sent, device=dev, dtype=torch.float32)
    meta = torch.tensor(kv_lens if kv_lens is not None else [SKV] * B, dtype=torch.int32, device=dev)
    fn(
        q_ptr=P(q, half),
        k_ptr=P(k, half),
        v_ptr=P(v, half),
        o_ptr=P(o, half),
        lse_ptr=P(lse, cutlass.Float32, 4),
        sinks_ptr=P(torch.zeros(H, dtype=torch.float32, device=dev), cutlass.Float32),
        meta_ptr=P(meta, cutlass.Int32),
        o_desc_ptr=P(torch.zeros(1, dtype=torch.int64, device=dev), cutlass.Int64),
        problem_size=(B, H, KH, S, SKV, 0),
        q_strides=tuple(q.stride()[:3]),
        k_strides=tuple(k.stride()[:3]),
        v_strides=tuple(v.stride()[:3]),
        o_strides=tuple(o.stride()[:3]),
        lse_strides=tuple(lse.stride()),
        lse_ext=0,
        scale_softmax_log2=cutlass.Float32(scale * math.log2(math.e)),
        n_thd_units=cutlass.Int32(0),
        seq_q_lens_addr=0,
        thd_q_lens_ptr=None,
        thd_kv_lens_ptr=None,
        thd_lens_form=None,
        gate_ptr=None,
        gate_strides=(0, 0, 0),
        stream=_stream(),
    )
    torch.cuda.synchronize()
    return o, lse


def _max_diff(a_o, a_lse, b_o, b_lse, live):
    return float((a_o.float() - b_o.float()).abs().max()), float((a_lse[live] - b_lse[live]).abs().max())


def _paginate(k, v, kv_lens, P, hnd, fill, g, *, minus_one_tail=True, extra_pages=3):
    """Scatter the first ``kv_lens[b]`` tokens of ``k`` / ``v`` ``[B, SKV, KH, D]`` into page pools viewed as
    ``[num_pages, KH, P, D]`` (HND contiguous, or NHD storage ``[num_pages, P, KH, D]`` permuted into that shape) through a
    shuffled ``(B, max_pages)`` block table.  Every row no sequence wrote -- the spare pages, the pages past a sequence's live
    ones, and the TAIL rows of its partially filled last page -- holds ``fill``: NaN (a gathered unwritten row would poison
    O) or, under ``"other"``, the OTHER batch entry's tokens at those positions (a gathered unwritten row would be a REAL key
    of another sequence: finite, plausible, wrong).  Table entries past a sequence's live pages are -1 when
    ``minus_one_tail`` (the page -1 convention), else stale valid pages full of ``fill``.  Returns (k_pool, v_pool, table)."""
    B, SKV, KH, _ = k.shape
    dev = k.device
    max_pages = -(-SKV // P)
    num_pages = B * max_pages + extra_pages
    shape = (num_pages, KH, P, D) if hnd else (num_pages, P, KH, D)
    kp = torch.full(shape, float("nan"), device=dev, dtype=k.dtype)
    vp = torch.full(shape, float("nan"), device=dev, dtype=v.dtype)
    perm = torch.randperm(num_pages, device=dev, generator=g)[: B * max_pages].view(B, max_pages)
    table = perm.to(torch.int32).clone()

    def put(pool, page, t0, t1, rows):  # rows: [t1 - t0, KH, D]
        if hnd:
            pool[page, :, t0:t1] = rows.permute(1, 0, 2)
        else:
            pool[page, t0:t1] = rows

    for b in range(B):
        L = int(kv_lens[b])
        n_live = -(-L // P)
        other = (b + 1) % B
        for p in range(max_pages):
            page = int(perm[b, p])
            lo, hi = p * P, min((p + 1) * P, L)
            if hi > lo:
                put(kp, page, 0, hi - lo, k[b, lo:hi])
                put(vp, page, 0, hi - lo, v[b, lo:hi])
            if fill == "other":  # the unwritten rows of this page: the other sequence's tokens at those positions
                n_fill = P - max(hi - lo, 0)
                pos = (torch.arange(n_fill, device=dev) + max(hi, lo)) % SKV
                put(kp, page, P - n_fill, P, k[other][pos])
                put(vp, page, P - n_fill, P, v[other][pos])
        if minus_one_tail:
            table[b, n_live:] = -1
    if not hnd:
        kp, vp = kp.permute(0, 2, 1, 3), vp.permute(0, 2, 1, 3)
    assert tuple(kp.shape) == (num_pages, KH, P, D)
    return kp, vp, table


def _launch_paged(q, k_pool, v_pool, table, kv_lens, ids, lens, top_k, scale, P, launches=2):
    """Sentinel-filled O / LSE per launch through the adapter's paged form (``paged_kv=True``)."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    dev = q.device
    B, S, H, _ = q.shape
    seq_kv = torch.tensor(kv_lens, device=dev, dtype=torch.int32)
    sent = _sentinel(q.dtype)
    outs = []
    for _ in range(launches):
        o = torch.full((B, S, H, D), sent, device=dev, dtype=q.dtype)
        lse = torch.full((B, H, S), sent, device=dev, dtype=torch.float32)
        a = SparseGqaFwdDslSm107(
            q=q,
            k=k_pool,
            v=v_pool,
            o=o,
            lse=lse,
            block_ids=ids,
            block_lens=lens,
            seq_kv_lens=seq_kv,
            top_k=top_k,
            scale=scale,
            paged_kv=True,
            page_size=P,
            block_table=table,
        )
        assert a.check_support()
        a.compile()
        a.execute(stream=_stream())
        torch.cuda.synchronize()
        outs.append((o, lse))
    return outs


def _assert_bitwise(a, b, what):
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), f"{what}: O / LSE differ from the dense read of the same tokens"


# ============================================================================ Rubin: the ladder
@requires_rubin
@pytest.mark.parametrize(
    "B, S, H, KH, dtype, top_k, list_kind, kv_lens, block_lens",
    [
        pytest.param(1, 4, 24, 2, torch.bfloat16, 512, "full", None, True, id="one-tile-one-block"),
        pytest.param(1, 256, 24, 2, torch.bfloat16, 512, "full", None, True, id="two-tiles"),
        pytest.param(1, 384, 24, 2, torch.bfloat16, 512, "full", None, True, id="three-tiles-ring-depth"),
        pytest.param(1, 512, 24, 2, torch.bfloat16, 512, "full", None, True, id="four-tiles-first-wrap"),
        pytest.param(1, 2176, 24, 2, torch.bfloat16, 512, "full", None, True, id="seventeen-tiles"),
        pytest.param(2, 100, 24, 2, torch.bfloat16, 512, "full", None, True, id="BxH-gt-1-one-tile-tail"),
        pytest.param(2, 2051, 24, 2, torch.bfloat16, 512, "shuffled", None, True, id="identity-bound-shuffled-B2"),
        pytest.param(3, 300, 24, 2, torch.bfloat16, 512, "full", [0, 257, 300], True, id="kv-lens-dead-sequence"),
        pytest.param(1, 1200, 24, 2, torch.bfloat16, 256, "shuffled", None, False, id="topk256-no-block-lens"),
        pytest.param(1, 512, 24, 2, torch.float16, 512, "full", None, True, id="f16"),
        pytest.param(1, 300, 16, 1, torch.bfloat16, 512, "shuffled", None, True, id="group16-one-kv-head"),
    ],
)
def test_sparse_core_ladder(B, S, H, KH, dtype, top_k, list_kind, kv_lens, block_lens):
    max_o, max_lse = _run_cell(B=B, S=S, H=H, KH=KH, dtype=dtype, top_k=top_k, list_kind=list_kind, kv_lens=kv_lens, block_lens=block_lens)
    print(f"\nsparse ladder B={B} S={S} H={H}/{KH} {dtype} top_k={top_k} {list_kind}: max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f} (budget {ATOL})")


# ============================================================================ Rubin: the degenerate-input matrix, row by row
@requires_rubin
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
def test_rows_with_no_complete_block_and_rows_with_no_open_block(dtype):
    """A query with 0 complete blocks (positions 0, 1, 2: the tail only, ``pos + 1`` live keys; position 0 attends itself) and
    a query with ``(pos + 1) % 4 == 0`` (positions 3, 7: complete blocks only, no tail appended) -- one tile, one block."""
    B, S, H, KH, top_k = 1, 8, 24, 2, 512
    q, k, v, g = _inputs(B, S, H, KH, dtype)
    ids, lens = _lists("full", B, S, top_k, g)
    assert lens[0].tolist() == [0, 0, 0, 1, 1, 1, 1, 2]
    scale = 1.0 / math.sqrt(D)
    outs = _launch(q, k, v, ids, lens, None, top_k, scale)
    ref_o, ref_lse = _reference(q, k, v, ids, lens, None, scale, top_k)
    max_o, max_lse = _check(outs, ref_o, ref_lse)
    o, lse = outs[0]
    # position 0: the only visible key is the token itself -> O = V[0] within the budget, LSE = its own score (natural log)
    for h in range(H):
        assert float((o[0, 0, h].float() - v[0, 0, h // (H // KH)].float()).abs().max()) <= ATOL
        own = float(q[0, 0, h].float() @ k[0, 0, h // (H // KH)].float()) * scale
        assert abs(float(lse[0, h, 0]) - own) <= ATOL
    print(f"\nno-complete-block / no-open-block rows {dtype}: max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}")


@requires_rubin
@pytest.mark.parametrize("form", ["3d", "2d"], ids=["ids-B-S-topk", "ids-BS-topk"])
def test_empty_selection_attends_the_tail_only_and_is_dead_without_a_tail(form):
    """``block_lens = 0`` on every row (the list all ``-1``): a row with an open tail attends the tail only (LSE finite); a row
    with ``(pos + 1) % 4 == 0`` has no tail either -> a DEAD item that still stores ``O = 0`` / ``LSE = -inf`` exactly (the
    clamped one-tile item, every lane masked, the row_dead SELECT); both list forms."""
    B, S, H, KH, top_k = 1, 40, 24, 2, 512
    dtype = torch.bfloat16
    q, k, v, g = _inputs(B, S, H, KH, dtype)
    ids = torch.full((B, S, top_k), -1, dtype=torch.int32, device=q.device)
    lens = torch.zeros(B, S, dtype=torch.int32, device=q.device)
    if form == "2d":
        ids, lens = ids.reshape(B * S, top_k), lens.reshape(B * S)
    scale = 1.0 / math.sqrt(D)
    outs = _launch(q, k, v, ids, lens, None, top_k, scale)
    ref_o, ref_lse = _reference(q, k, v, ids, lens, None, scale, top_k)
    pos = torch.arange(S, device=q.device)
    dead_rows = (pos + 1) % BS == 0
    assert torch.equal(torch.isinf(ref_lse[0, 0]), dead_rows)
    max_o, max_lse = _check(outs, ref_o, ref_lse)
    o, lse = outs[0]
    assert torch.isfinite(lse[:, :, ~dead_rows]).all()
    assert (o[0, dead_rows] == 0).all() and torch.isinf(lse[0, :, dead_rows]).all()
    print(f"\nempty selection ({form}): {int(dead_rows.sum())} dead rows exact, max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}")


@requires_rubin
def test_minus_one_inside_the_count_is_a_value_not_a_terminator():
    """A ``-1`` below ``block_lens`` is "no key" (zero rows, -inf) and the remaining entries still count: the kernel matches the
    oracle with those blocks ABSENT -- and differs from the full-list function by more than the budget, so the entry is a
    value, not a terminator the loop stops at."""
    B, S, H, KH, top_k = 1, 600, 24, 2, 512
    dtype = torch.bfloat16
    q, k, v, g = _inputs(B, S, H, KH, dtype)
    ids, lens = _lists("full", B, S, top_k, g)
    holes = ids.clone()
    holes[0, 100:, 1::3] = -1  # every third entry (inside the count on every row >= 100) becomes "no key"; block_lens unchanged
    scale = 1.0 / math.sqrt(D)
    outs = _launch(q, k, v, holes, lens, None, top_k, scale)
    ref_o, ref_lse = _reference(q, k, v, holes, lens, None, scale, top_k)
    max_o, max_lse = _check(outs, ref_o, ref_lse)
    full_o, full_lse = _reference(q, k, v, ids, lens, None, scale, top_k)
    o, lse = outs[0]
    teeth = float((o[0, 100:].float() - full_o[0, 100:]).abs().max())
    teeth_lse = float((lse[0, :, 100:] - full_lse[0, :, 100:]).abs().max())
    assert teeth > ATOL and teeth_lse > ATOL, "removing a third of the keys must move the function beyond the budget"
    print(f"\n-1 inside the count: max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f} vs the holed oracle; {teeth:.4f} / {teeth_lse:.4f} vs the full-list function")


@requires_rubin
def test_entries_beyond_the_count_are_never_read_both_list_forms_and_block_lens_absent():
    """Entries at index >= ``block_lens`` are never read, valid-looking or not: the output is BITWISE invariant to what sits
    there (``-1`` padding vs a garbage fill of real ids, huge ids and negative ids), to the list form (``[B, S, top_k]`` vs
    ``[B x S, top_k]``) and -- on a list whose count is the derived default -- to ``block_lens`` given vs absent."""
    B, S, H, KH, top_k = 1, 300, 24, 2, 512
    dtype = torch.bfloat16
    q, k, v, g = _inputs(B, S, H, KH, dtype)
    ids, lens = _lists("shuffled", B, S, top_k, g)
    scale = 1.0 / math.sqrt(D)
    j = torch.arange(top_k, device=q.device)[None, None, :]
    beyond = j >= lens[:, :, None]
    garbage = torch.tensor([0, 7, 10**6, -5], dtype=torch.int32, device=q.device)[j % 4].expand(B, S, top_k)
    filled = torch.where(beyond, garbage, ids).contiguous()
    assert not torch.equal(filled, ids)
    ((o_a, lse_a),) = _launch(q, k, v, ids, lens, None, top_k, scale, launches=1)
    ((o_b, lse_b),) = _launch(q, k, v, filled, lens, None, top_k, scale, launches=1)
    ((o_c, lse_c),) = _launch(q, k, v, filled.reshape(B * S, top_k), lens.reshape(B * S), None, top_k, scale, launches=1)
    ((o_d, lse_d),) = _launch(q, k, v, ids, None, None, top_k, scale, launches=1)
    ref_o, ref_lse = _reference(q, k, v, ids, lens, None, scale, top_k)
    _check([(o_a, lse_a)], ref_o, ref_lse)
    assert torch.equal(o_a, o_b) and torch.equal(lse_a, lse_b), "a garbage fill beyond the count changed the output"
    assert torch.equal(o_a, o_c) and torch.equal(lse_a, lse_c), "the [B x S, top_k] form changed the output"
    assert torch.equal(o_a, o_d) and torch.equal(lse_a, lse_d), "block_lens absent (the derived count) changed the output"


@requires_rubin
def test_ids_past_the_sequence_length_and_past_the_tensor_are_masked_key_by_key():
    """With ``seq_kv_lens = [150]`` on a 200-token tensor, a listed block inside the tensor but past the length (block 45 =
    tokens 180..183) and a block past the tensor (id 10^6: TMA zero-fills the rows and credits the bytes) contribute nothing;
    the replaced entries' blocks are gone from the function (the oracle sees the same list)."""
    B, S, H, KH, top_k = 1, 200, 24, 2, 512
    kv_lens = [150]
    dtype = torch.bfloat16
    q, k, v, g = _inputs(B, S, H, KH, dtype)
    ids, lens = _lists("full", B, S, top_k, g, kv_lens)
    assert int(ids.max()) == 150 // BS - 1
    bad = ids.clone()
    two = lens[0] >= 2
    last = (lens[0] - 1).long()
    bad[0, two, last[two]] = 45
    bad[0, two, (last - 1)[two]] = 10**6
    scale = 1.0 / math.sqrt(D)
    outs = _launch(q, k, v, bad, lens, kv_lens, top_k, scale)
    ref_o, ref_lse = _reference(q, k, v, bad, lens, kv_lens, scale, top_k)
    max_o, max_lse = _check(outs, ref_o, ref_lse)
    print(f"\nids past the length / the tensor: max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}")


@requires_rubin
def test_a_future_block_in_the_list_contributes_nothing():
    """A listed block strictly past the query's position (``4 blk > pos``) is masked key by key: the kernel matches the oracle,
    which drops every key at or past the visible range."""
    B, S, H, KH, top_k = 1, 64, 24, 2, 512
    dtype = torch.bfloat16
    q, k, v, g = _inputs(B, S, H, KH, dtype)
    ids, lens = _lists("full", B, S, top_k, g)
    pos = torch.arange(S, device=q.device)
    fut = ids.clone()
    rows = pos >= 8
    fut[0, rows, (lens[0, rows] - 1).long()] = (pos[rows] // BS + 2).to(torch.int32)  # a block two past the query's own
    scale = 1.0 / math.sqrt(D)
    outs = _launch(q, k, v, fut, lens, None, top_k, scale)
    ref_o, ref_lse = _reference(q, k, v, fut, lens, None, scale, top_k)
    max_o, max_lse = _check(outs, ref_o, ref_lse)
    print(f"\nfuture block listed: max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}")


@requires_rubin
@pytest.mark.parametrize("which", ["duplicate-id", "open-block-listed"])
def test_duplicates_count_twice_the_documented_contract(which):
    """The documented contract for a list the test tree's detector rejects (``block_ids_contract_violations``; the library's
    host check is FORM-only and never reads a list): an id listed twice is attended twice (the softmax over the multiset),
    and the open block listed in place of a complete block duplicates the appended tail (its keys <= the position count
    twice; its future keys stay masked; an entry appended PAST the derived count is never read).  Pinned against the
    multiplicity reference; the distance to the idempotent set oracle is reported, not asserted.  The counted-twice reading
    is the documented contract; should the idempotent set reading ever be adopted, these two cells flip to the set oracle
    together with the kernel's side of it (a ``blk < floor(n_vis / 4)`` test per entry for the listed open block; a dedupe of
    a repeated id, on the host or in the kernel -- a mask term cannot deduplicate) -- never one without the other."""
    oracle = _oracle()
    B, S, H, KH, top_k = 1, 300, 24, 2, 64
    dtype = torch.bfloat16
    q, k, v, g = _inputs(B, S, H, KH, dtype)
    ids, lens = _lists("full", B, S, top_k, g)
    pos = torch.arange(S, device=q.device)
    ids, lens = ids.clone(), lens.clone()
    if which == "duplicate-id":
        rows = lens[0] >= 2
        ids[0, rows, 1] = ids[0, rows, 0]  # entry 1 repeats entry 0 -> block 0 twice
    else:
        sel = ((pos + 1) % BS != 0) & (lens[0] >= 1)
        ids[0, sel, (lens[0, sel] - 1).long()] = ((pos[sel] + 1) // BS).to(torch.int32)  # the last entry becomes the open block
    assert oracle.block_ids_contract_violations(ids[0], pos, S, block_size=BS, block_lens=lens[0])
    scale = 1.0 / math.sqrt(D)
    outs = _launch(q, k, v, ids, lens, None, top_k, scale)
    ref_o, ref_lse = _reference(q, k, v, ids, lens, None, scale, top_k, oracle_check=False)
    max_o, max_lse = _check(outs, ref_o, ref_lse)
    # the idempotent set reading (the oracle's) for the record
    allowed = oracle.qsa_visible_mask(ids[0], lens[0], pos, S, S, BS, top_k=top_k)
    o, lse = outs[0]
    d_set = 0.0
    for h in range(H):
        sc = (q[0, :, h].float() @ k[0, :, h // (H // KH)].float().t()) * scale
        sc = sc.masked_fill(~allowed, float("-inf"))
        p = torch.softmax(sc, dim=-1)
        d_set = max(d_set, float((o[0, :, h].float() - p @ v[0, :, h // (H // KH)].float()).abs().max()))
    print(f"\n{which}: max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f} vs the multiplicity reference; {d_set:.4f} vs the set oracle (reported)")


@requires_rubin
def test_block_lens_out_of_range_are_clamped_on_device():
    """A negative ``block_lens`` reads as 0 (the tail only; dead where there is no tail), one above the derived count as the
    derived count -- a device min / max, never a fault; the oracle clamps the same way."""
    B, S, H, KH, top_k = 1, 400, 24, 2, 512
    dtype = torch.bfloat16
    q, k, v, g = _inputs(B, S, H, KH, dtype)
    ids, lens = _lists("full", B, S, top_k, g)
    wild = lens.clone()
    wild[0, 1::2] = -7  # odd positions: the rows with (pos + 1) % 4 == 0 among them have no tail either -> dead
    wild[0, 0::2] = 100000
    scale = 1.0 / math.sqrt(D)
    outs = _launch(q, k, v, ids, wild, None, top_k, scale)
    ref_o, ref_lse = _reference(q, k, v, ids, wild, None, scale, top_k)
    max_o, max_lse = _check(outs, ref_o, ref_lse)
    pos = torch.arange(S, device=q.device)
    dead = (pos % 2 == 1) & ((pos + 1) % BS == 0)
    assert torch.equal(torch.isinf(ref_lse[0, 0]), dead) and int(dead.sum()) == S // BS
    print(f"\nblock_lens clamped: {int(dead.sum())} dead rows exact, max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}")


@requires_rubin
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
def test_B2_with_a_different_list_per_batch_entry_reads_its_own_rows(dtype):
    """``B = 2``: a random 64-wide subset per row differs between the entries and so do the K / V rows; each entry matches
    the oracle on ITS list and ITS rows (the gather row coordinate carries ``b x S_kv``), and entry 1's output is far from
    entry 0's function."""
    B, S, H, KH, top_k = 2, 520, 24, 2, 64
    q, k, v, g = _inputs(B, S, H, KH, dtype, seed=5)
    ids, lens = _lists("shuffled", B, S, top_k, g)
    assert not torch.equal(ids[0], ids[1])
    scale = 1.0 / math.sqrt(D)
    outs = _launch(q, k, v, ids, lens, None, top_k, scale)
    ref_o, ref_lse = _reference(q, k, v, ids, lens, None, scale, top_k)
    max_o, max_lse = _check(outs, ref_o, ref_lse)
    o, _ = outs[0]
    assert float((o[1].float() - ref_o[0]).abs().max()) > ATOL
    print(f"\nB=2 different lists {dtype}: max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}")


@requires_rubin
def test_slab_stride_kv_layout_is_bitwise_the_contiguous_layout():
    """The fused-projection slab ``[B, S, (H + 2 KH) x D]`` with Q, K, V as column views (token stride = the slab width, head
    stride D, K / V batch stride = S_kv x the token stride): within budget vs the oracle and BITWISE the run on contiguous
    copies of the same values."""
    B, S, H, KH, top_k = 2, 300, 24, 2, 512
    dtype = torch.bfloat16
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(11)
    n = (H + 2 * KH) * D
    slab = torch.randn(B, S, n, device=dev, dtype=torch.float32, generator=g).to(dtype)
    q = slab[..., : H * D].unflatten(-1, (H, D))
    k = slab[..., H * D : (H + KH) * D].unflatten(-1, (KH, D))
    v = slab[..., (H + KH) * D :].unflatten(-1, (KH, D))
    assert k.stride() == (S * n, n, D, 1) and k.data_ptr() == slab.data_ptr() + H * D * slab.element_size()
    ids, lens = _lists("shuffled", B, S, top_k, g)
    scale = 1.0 / math.sqrt(D)
    outs = _launch(q, k, v, ids, lens, None, top_k, scale)
    ref_o, ref_lse = _reference(q, k, v, ids, lens, None, scale, top_k)
    max_o, max_lse = _check(outs, ref_o, ref_lse)
    ((o_c, lse_c),) = _launch(q.contiguous(), k.contiguous(), v.contiguous(), ids, lens, None, top_k, scale, launches=1)
    assert torch.equal(outs[0][0], o_c) and torch.equal(outs[0][1], lse_c), "the slab layout must read the same values as the contiguous one"
    print(f"\nslab layout: max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}, bitwise the contiguous layout")


@requires_rubin
@pytest.mark.parametrize(
    "top_k, block_lens",
    [pytest.param(4, True, id="topk4"), pytest.param(4, False, id="topk4-no-block-lens"), pytest.param(64, True, id="topk64")],
)
def test_small_top_k_lists(top_k, block_lens):
    """``CFG.BLOCK_TOPK`` in {4, 64}: a 16-B id row (one quad per tile), a random subset per row; with and without the count."""
    max_o, max_lse = _run_cell(B=1, S=300, H=24, KH=2, dtype=torch.bfloat16, top_k=top_k, list_kind="shuffled", block_lens=block_lens)
    print(f"\ntop_k={top_k} block_lens={block_lens}: max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}")


@requires_rubin
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
def test_a_negative_attn_scale_is_served_by_the_signed_scale(dtype):
    """``attn_scale = -1 / sqrt(D)``: the kernel multiplies every raw score by the SIGNED scale before the column max, so the
    function is the reference's softmax over the negated scores within the module's budget -- an ACCEPT cell, not a decline (the
    dense line serves the same request through ``negate_scores``; the sparse adapter passes the sign through).  Teeth: the same
    operands at +scale give a different O, so the sign provably reached the kernel."""
    scale = -1.0 / math.sqrt(D)
    max_o, max_lse = _run_cell(B=1, S=300, H=24, KH=2, dtype=dtype, top_k=512, list_kind="shuffled", seed=11, scale=scale)
    q, k, v, g = _inputs(1, 300, 24, 2, dtype, 11)
    ids, lens = _lists("shuffled", 1, 300, 512, g)
    ((o_neg, lse_neg),) = _launch(q, k, v, ids, lens, None, 512, scale, launches=1)
    ((o_pos, lse_pos),) = _launch(q, k, v, ids, lens, None, 512, -scale, launches=1)
    assert float((o_neg.float() - o_pos.float()).abs().max()) > 10 * ATOL and not torch.equal(lse_neg, lse_pos), "the sign must reach the kernel"
    print(
        f"\nnegative attn_scale {dtype}: max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f} (budget {ATOL}); |O(-s) - O(+s)| {float((o_neg.float() - o_pos.float()).abs().max()):.4f}"
    )


@requires_rubin
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
def test_mha_one_query_head_per_kv_head(dtype):
    """``H_q == H_kv`` (the group is 1: one live row of the 16-row tile): the 24/2 geometry's sibling at the other end of the
    group range (16 is in the ladder)."""
    max_o, max_lse = _run_cell(B=1, S=300, H=2, KH=2, dtype=dtype, top_k=512, list_kind="shuffled")
    print(f"\nMHA G=1 {dtype}: max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}")


@requires_rubin
def test_one_cta_in_the_whole_grid():
    """``B = S = H = H_kv = 1``: one work item, one CTA; the scheduler hands out nothing else; the row attends itself."""
    B, S, H, KH, top_k = 1, 1, 1, 1, 512
    dtype = torch.bfloat16
    q, k, v, g = _inputs(B, S, H, KH, dtype)
    ids, lens = _lists("full", B, S, top_k, g)
    scale = 1.0 / math.sqrt(D)
    outs = _launch(q, k, v, ids, lens, None, top_k, scale)
    ref_o, ref_lse = _reference(q, k, v, ids, lens, None, scale, top_k)
    max_o, max_lse = _check(outs, ref_o, ref_lse)
    o, lse = outs[0]
    assert float((o[0, 0, 0].float() - v[0, 0, 0].float()).abs().max()) <= ATOL
    print(f"\none CTA: max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}")


@requires_rubin
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
def test_dead_and_live_items_interleave_in_one_cta(dtype):
    """``B = 4`` with ``seq_kv_lens = [0, 400, 0, 100]`` at ``S = 400`` (3200 items over the resident CTAs): every CTA's item
    sequence mixes dead one-tile items (the empty sequences) with 1..4-tile items of the live ones -- the carried pipeline
    states across such a sequence (not the systematic sequence probe, which is its own test)."""
    max_o, max_lse = _run_cell(B=4, S=400, H=24, KH=2, dtype=dtype, top_k=512, list_kind="full", kv_lens=[0, 400, 0, 100])
    print(f"\ndead / live interleave {dtype}: max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}")


# ============================================================================ Rubin: the paged read
@requires_rubin
@pytest.mark.parametrize("fill", ["nan", "other"], ids=["unwritten-rows-NaN", "unwritten-rows-other-sequence"])
@pytest.mark.parametrize(
    "P, hnd, dtype",
    [
        pytest.param(16, False, torch.bfloat16, id="page16-NHD-bf16"),
        pytest.param(16, True, torch.bfloat16, id="page16-HND-bf16"),
        pytest.param(64, False, torch.float16, id="page64-NHD-f16"),
        pytest.param(64, True, torch.bfloat16, id="page64-HND-bf16"),
        pytest.param(48, False, torch.bfloat16, id="page48-NHD-bf16"),
    ],
)
def test_paged_read_is_bitwise_the_dense_read_on_the_same_tokens(P, hnd, dtype, fill):
    """``B = 2`` with ``seq_kv_lens = [333, 301]`` on 333 query rows and a random 64-wide list per row: the same tokens
    through page pools (a shuffled table, ``-1`` past the live pages) give BITWISE the O / LSE of the dense read, and both
    are within the budget vs the oracle.  Both lengths end in a partially filled last page (333 = 20 x 16 + 13, 301 = 4 x 64
    + 45, 301 = 6 x 48 + 13) whose live blocks are gathered from it; entry 1's rows at or past position 301 attend the open
    block 75 (tokens 300..303), whose rows 301..303 are UNWRITTEN rows of that page -- NaN, or under ``other`` the OTHER
    sequence's real tokens at those positions -- and entry 0's open block 83 (332..335) has rows past the length in its last
    page too: the length mask keeps every such row out of the gather (bitwise the dense read, whose tensor ends at 333)."""
    B, S, H, KH, top_k = 2, 333, 24, 2, 64
    kv_lens = [333, 301]
    q, k, v, g = _inputs(B, S, H, KH, dtype, seed=3)
    ids, lens = _lists("shuffled", B, S, top_k, g, kv_lens)
    scale = 1.0 / math.sqrt(D)
    dense = _launch(q, k, v, ids, lens, kv_lens, top_k, scale, launches=1)
    kp, vp, table = _paginate(k, v, kv_lens, P, hnd, fill, g)
    assert int(table.max()) < kp.shape[0]  # (-1 entries past the live pages where a length leaves spare pages: page 16 / 64, not 48)
    paged = _launch_paged(q, kp, vp, table, kv_lens, ids, lens, top_k, scale, P)
    ref_o, ref_lse = _reference(q, k, v, ids, lens, kv_lens, scale, top_k)
    max_o, max_lse = _check(paged, ref_o, ref_lse)
    _assert_bitwise(paged[0], dense[0], f"paged page {P} {'HND' if hnd else 'NHD'} {dtype} fill={fill}")
    print(f"\npaged page={P} {'HND' if hnd else 'NHD'} {dtype} fill={fill}: max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}; BITWISE the dense read")


@requires_rubin
def test_paged_other_sequence_rows_in_the_open_block_would_change_the_output_if_gathered():
    """The masked-rows claim has teeth: the pools whose unwritten rows hold the other sequence's tokens and the pools whose
    unwritten rows hold NaN give BITWISE the same output (a gathered unwritten row would differ between them -- a NaN
    poisons O, a real key moves it), on a length (``301``, ``(pos + 1) % 4 == 2`` at the last row) whose open block straddles
    the length inside the last page; and the oracle over the same tokens with those rows PRESENT (length 304) is far from
    both, so the rows do carry weight when they are visible."""
    B, S, H, KH, top_k, P = 2, 304, 24, 2, 64, 16
    dtype = torch.bfloat16
    kv_lens = [304, 301]
    q, k, v, g = _inputs(B, S, H, KH, dtype, seed=9)
    ids, lens = _lists("full", B, S, top_k, g, kv_lens)
    scale = 1.0 / math.sqrt(D)
    g2 = torch.Generator(device=q.device).manual_seed(21)
    kp_n, vp_n, table_n = _paginate(k, v, kv_lens, P, True, "nan", g2)
    g3 = torch.Generator(device=q.device).manual_seed(21)
    kp_o, vp_o, table_o = _paginate(k, v, kv_lens, P, True, "other", g3)
    assert torch.equal(table_n, table_o) and not torch.equal(torch.nan_to_num(kp_n, nan=7.0), torch.nan_to_num(kp_o, nan=7.0))
    out_n = _launch_paged(q, kp_n, vp_n, table_n, kv_lens, ids, lens, top_k, scale, P, launches=1)
    out_o = _launch_paged(q, kp_o, vp_o, table_o, kv_lens, ids, lens, top_k, scale, P, launches=1)
    ref_o, ref_lse = _reference(q, k, v, ids, lens, kv_lens, scale, top_k)
    max_o, max_lse = _check(out_n, ref_o, ref_lse)
    _assert_bitwise(out_n[0], out_o[0], "NaN-filled vs other-sequence-filled unwritten rows")
    # the rows carry weight when visible: entry 1 at length 304 (rows 301..303 present) differs beyond the budget at its last rows
    vis_o, _ = _reference(q, k, v, ids, lens, [304, 304], scale, top_k)
    teeth = float((out_n[0][0][1, 301:].float() - vis_o[1, 301:]).abs().max())
    assert teeth > ATOL, f"rows 301..303 must carry weight when visible (diff {teeth})"
    print(
        f"\nopen-block rows beyond the length: NaN fill == other-sequence fill bitwise; max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}; visible-rows teeth {teeth:.4f}"
    )


@requires_rubin
def test_paged_stale_pages_minus_one_pages_and_ids_past_the_table_are_zero_rows():
    """A table whose entries past the live pages are STALE valid pages (full of NaN) or ``-1``; lists that name a block past the
    length but inside the table (``block 80`` at length 301 -> a stale / -1 page) and a block past the table (``10^6``: the
    page index is clamped for the read and the row takes -1): every such key is masked (the length) or zero-filled (the page
    -1 convention), BITWISE the dense read of the same lists, where those blocks are masked by the length / TMA-OOB."""
    B, S, H, KH, top_k, P = 2, 320, 24, 2, 512, 16
    dtype = torch.bfloat16
    kv_lens = [320, 301]
    q, k, v, g = _inputs(B, S, H, KH, dtype, seed=4)
    ids, lens = _lists("full", B, S, top_k, g, kv_lens)
    bad = ids.clone()
    two = lens >= 2
    last = (lens - 1).long()
    for b in range(B):
        rows = two[b]
        bad[b, rows, last[b, rows]] = 80 if b == 1 else 10**6  # entry 1: a block past its length inside the table; entry 0: past the table
        bad[b, rows, (last[b] - 1)[rows]] = 10**6 if b == 1 else 80
    scale = 1.0 / math.sqrt(D)
    dense = _launch(q, k, v, bad, lens, kv_lens, top_k, scale, launches=1)
    ref_o, ref_lse = _reference(q, k, v, bad, lens, kv_lens, scale, top_k)
    for minus_one in (True, False):
        g2 = torch.Generator(device=q.device).manual_seed(5)
        kp, vp, table = _paginate(k, v, kv_lens, P, True, "nan", g2, minus_one_tail=minus_one)
        assert (int((table == -1).sum()) > 0) == minus_one
        paged = _launch_paged(q, kp, vp, table, kv_lens, bad, lens, top_k, scale, P)
        max_o, max_lse = _check(paged, ref_o, ref_lse)
        _assert_bitwise(paged[0], dense[0], f"minus_one_tail={minus_one}")
        print(
            f"\nstale / -1 pages (minus_one_tail={minus_one}), ids past the length and the table: max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}; BITWISE the dense read"
        )


@requires_rubin
def test_paged_seventeen_tiles_without_block_lens_is_bitwise_the_dense_read():
    """The full list at ``S = 2176`` (17 tiles: the ring wraps five times inside one item), ``block_lens`` ABSENT (the
    derived count), ``seq_kv_lens = [2176, 2051]`` (entry 1's last rows see 513 blocks and keep the open tail of its own
    length), page 16 HND: BITWISE the dense read and within the budget."""
    B, S, H, KH, top_k, P = 2, 2176, 24, 2, 512, 16
    dtype = torch.bfloat16
    kv_lens = [2176, 2051]
    q, k, v, g = _inputs(B, S, H, KH, dtype, seed=6)
    ids, lens = _lists("full", B, S, top_k, g, kv_lens)
    scale = 1.0 / math.sqrt(D)
    dense = _launch(q, k, v, ids, None, kv_lens, top_k, scale, launches=1)
    kp, vp, table = _paginate(k, v, kv_lens, P, True, "nan", g)
    paged = _launch_paged(q, kp, vp, table, kv_lens, ids, None, top_k, scale, P)
    ref_o, ref_lse = _reference(q, k, v, ids, None, kv_lens, scale, top_k)
    max_o, max_lse = _check(paged, ref_o, ref_lse)
    _assert_bitwise(paged[0], dense[0], "17 tiles, block_lens absent")
    print(f"\npaged 17 tiles (block_lens absent): max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}; BITWISE the dense read")


# ============================================================================ Rubin: the identity check vs the dense kernel
@requires_rubin
@pytest.mark.parametrize("S", [512, 2051])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
def test_identity_vs_the_dense_kernel_below_the_bound(S, dtype):
    """With the full list at ``S <= 2051`` the sparse kernel, the dense sm107 d256 kernel (top-left causal) and the oracle
    compute the same function: pairwise within the dense budget on O and LSE.  Not bitwise -- the sparse kernel accumulates
    4-token blocks in list order through the swap-AB body, the dense tile walks 128-key tiles (a different summation order);
    the max diffs and the bitwise flag are reported per cell."""
    B, H, KH, top_k = 1, 24, 2, 512
    q, k, v, g = _inputs(B, S, H, KH, dtype)
    ids, lens = _lists("full", B, S, top_k, g)
    scale = 1.0 / math.sqrt(D)
    outs = _launch(q, k, v, ids, lens, None, top_k, scale)
    ref_o, ref_lse = _reference(q, k, v, ids, lens, None, scale, top_k)
    max_o, max_lse = _check(outs, ref_o, ref_lse)
    o_d, lse_d = _dense_kernel(q, k, v, scale)
    live = ~torch.isinf(ref_lse)
    assert torch.isfinite(o_d.float()).all() and not (o_d.float() == _sentinel(dtype)).any()
    sd_o, sd_lse = _max_diff(outs[0][0], outs[0][1], o_d, lse_d, live)
    dr_o, dr_lse = _max_diff(o_d, lse_d, ref_o, ref_lse, live)
    assert sd_o <= ATOL and sd_lse <= ATOL, f"sparse vs dense {sd_o} / {sd_lse} outside the budget"
    assert dr_o <= ATOL and dr_lse <= ATOL, f"dense vs oracle {dr_o} / {dr_lse} outside the budget"
    bitwise = torch.equal(outs[0][0], o_d) and torch.equal(outs[0][1], lse_d)
    print(
        f"\nidentity S={S} {dtype}: sparse-vs-oracle {max_o:.5f} / {max_lse:.6f}, sparse-vs-dense {sd_o:.5f} / {sd_lse:.6f}, "
        f"dense-vs-oracle {dr_o:.5f} / {dr_lse:.6f}, bitwise {bitwise}"
    )


@requires_rubin
def test_adversarial_s2052_planted_key_inside_the_omitted_block():
    """At ``S = 2052`` query 2051 has 513 complete blocks and the 512-wide full list omits block 512 (tokens 2048..2051, its
    own token's block).  A dominant key is planted there for one query head of each KV head (``k = 1.25 q_row``: the score
    beats every other key by >= 8 nats, the planted key holds > 99.9 % of the full-set mass).  The kernel is within the budget vs
    the oracle ON the list everywhere, and at row 2051 for the planted heads OUTSIDE the budget vs the oracle on the FULL set
    and vs the dense kernel (cos < 0.5, the diff >= 10 budgets) -- a kernel that ignored the list could not pass both.  Both
    magnitudes are reported."""
    B, S, H, KH, top_k = 1, 2052, 24, 2, 512
    G = H // KH
    dtype = torch.bfloat16
    q, k, v, g = _inputs(B, S, H, KH, dtype, seed=7)
    row = 2051
    plants = [(0, 2049), (G, 2050)]  # (query head, planted token inside block 512) -- one per KV head
    for h, tok in plants:
        k[0, tok, h // G] = (1.25 * q[0, row, h].float()).to(dtype)
    ids, lens = _lists("full", B, S, top_k, g)
    assert int(lens[0, row]) == 512 and int(ids[0, row].max()) == 511, "row 2051 lists blocks 0..511 and omits block 512"
    scale = 1.0 / math.sqrt(D)
    # the plant's margin and mass on the full set (fp32 on the half-rounded operands)
    for h, tok in plants:
        sc = (q[0, row, h].float() @ k[0, : row + 1, h // G].float().t()) * scale
        others = torch.cat([sc[:tok], sc[tok + 1 :]])
        margin = float(sc[tok] - others.max())
        mass = float(1.0 / (1.0 + torch.exp(others - sc[tok]).sum()))
        assert margin >= 8.0 and mass > 0.999, f"head {h}: plant margin {margin:.2f} nats, mass {mass:.5f}"
    outs = _launch(q, k, v, ids, lens, None, top_k, scale)
    ref_o, ref_lse = _reference(q, k, v, ids, lens, None, scale, top_k)
    max_o, max_lse = _check(outs, ref_o, ref_lse)
    # the FULL set: every one of the 513 complete blocks (the oracle at width 513) and the dense causal kernel
    oracle = _oracle()
    positions = torch.arange(S, device=q.device).expand(B, S)
    ids_full, lens_full = oracle.full_block_ids(positions, 513, BS)
    assert int(ids_full[0, row].max()) == 512
    full_o, full_lse = _reference(q, k, v, ids_full.contiguous(), lens_full.contiguous(), None, scale, 513)
    o_d, lse_d = _dense_kernel(q, k, v, scale)
    o, lse = outs[0]
    # elsewhere (every row below 2051) the three agree within the budget
    below = slice(0, row)
    assert float((o[0, below].float() - full_o[0, below]).abs().max()) <= ATOL
    assert float((o_d[0, below].float() - full_o[0, below]).abs().max()) <= ATOL
    assert float((lse[0, :, below] - full_lse[0, :, below]).abs().max()) <= ATOL
    report = []
    for h, tok in plants:
        ok = o[0, row, h].float()
        for name, other in (("full-set oracle", full_o[0, row, h]), ("dense kernel", o_d[0, row, h].float())):
            diff = float((ok - other).abs().max())
            cos = float(torch.nn.functional.cosine_similarity(ok, other, dim=0))
            assert cos < 0.5 and diff >= 10 * ATOL, f"head {h} vs {name}: cos {cos:.3f}, diff {diff:.4f} -- the planted block was attended"
            report.append(f"head {h} vs {name}: diff {diff:.4f} cos {cos:.3f}")
        d_lse = float(abs(lse[0, h, row] - full_lse[0, h, row]))
        assert d_lse > ATOL
        report.append(f"head {h} LSE vs the full set: {d_lse:.3f}")
    print(f"\nadversarial S=2052: within budget vs the oracle on the list ({max_o:.5f} / {max_lse:.6f}); row 2051 " + "; ".join(report))


# ============================================================================ Rubin: the recorded lists of the real layer
def _real_lists(S: int):
    """``(block_ids, block_lens)`` of the recorded first-QSA-layer lists at sequence length ``S`` from
    ``$CUDNN_FROST_QSA_INDEX_LISTS_DIR/layer03_S<S>_block_ids.pt``; the sha256 sidecar is verified before use; the test skips
    when the variable is unset or the file is absent."""
    root = os.environ.get(INDEX_LISTS_ENV)
    if not root:
        pytest.skip(f"{INDEX_LISTS_ENV} unset: the recorded real lists are not available here")
    path = os.path.join(root, f"layer03_S{S}_block_ids.pt")
    if not os.path.isfile(path):
        pytest.skip(f"{path} absent")
    side = path + ".sha256"
    if os.path.isfile(side):
        want = open(side).read().split()[0]
        have = hashlib.sha256(open(path, "rb").read()).hexdigest()
        assert have == want, f"{path}: sha256 {have} != the recorded {want}"
    blob = torch.load(path, map_location="cpu", weights_only=True)
    ids, lens = blob["block_ids"], blob["block_lens"]
    assert tuple(ids.shape) == (S, 512) and ids.dtype == torch.int32 and tuple(lens.shape) == (S,)
    assert str(blob.get("meta", {}).get("label", "")).startswith("REAL indices")
    return ids, lens


@requires_rubin
@pytest.mark.parametrize("S, dtype", [pytest.param(32768, torch.bfloat16, id="S32768-bf16"), pytest.param(4096, torch.float16, id="S4096-f16-prefix")])
def test_real_layer_lists(S, dtype):
    """The recorded lists of the released checkpoint's first QSA layer (sorted ascending, ``block_lens = min(512, (t + 1) //
    4)``, the identity below row 2051, 512 of up to 8192 blocks above it) at ``S = 32768`` in bf16 and their 4096-row prefix
    in f16, against the oracle's visible-set rule; every row live, two launches bitwise."""
    B, H, KH, top_k = 1, 24, 2, 512
    ids_cpu, lens_cpu = _real_lists(32768)
    ids_cpu, lens_cpu = ids_cpu[:S], lens_cpu[:S]
    t = torch.arange(S)
    assert torch.equal(lens_cpu, torch.minimum((t + 1) // BS, torch.tensor(512)).to(torch.int32))
    assert int(ids_cpu.max()) < S // BS
    q, k, v, g = _inputs(B, S, H, KH, dtype, seed=1)
    ids = ids_cpu.to(q.device).unsqueeze(0).contiguous()
    lens = lens_cpu.to(q.device).unsqueeze(0).contiguous()
    scale = 1.0 / math.sqrt(D)
    outs = _launch(q, k, v, ids, lens, None, top_k, scale)
    ref_o, ref_lse = _reference(q, k, v, ids, lens, None, scale, top_k)
    assert not torch.isinf(ref_lse).any()
    max_o, max_lse = _check(outs, ref_o, ref_lse)
    print(f"\nreal lists S={S} {dtype}: {int((ids >= 0).sum())} ids, max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}")


# ============================================================================ Rubin: the THD arm (packed sequences)
def _cu(lens, base=0):
    out, acc = [base], base
    for n in lens:
        acc += int(n)
        out.append(acc)
    return out


def _thd_inputs(lens_q, lens_kv, H, KH, dtype, *, top_k=512, list_kind="full", seed=0, extra_cap=0):
    """Packed operands ``Q [1, T_q, H, D]``, ``K / V [1, T_kv, KH, D]`` (``extra_cap`` unused rows past the packed totals) and
    the per-row lists built PER SEQUENCE from the oracle's builders with the sequence's own KV length -- the block ids are the
    sequence's own block numbers; a sequence with no key gets all ``-1`` / count 0; the rows past the totals stay ``-1``."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    T_q, T_kv = sum(lens_q), sum(lens_kv)
    q = torch.randn(1, T_q + extra_cap, H, D, device=dev, dtype=torch.float32, generator=g).to(dtype)
    k = torch.randn(1, T_kv + extra_cap, KH, D, device=dev, dtype=torch.float32, generator=g).to(dtype)
    v = torch.randn(1, T_kv + extra_cap, KH, D, device=dev, dtype=torch.float32, generator=g).to(dtype)
    ids = torch.full((T_q + extra_cap, top_k), -1, dtype=torch.int32, device=dev)
    lens = torch.zeros(T_q + extra_cap, dtype=torch.int32, device=dev)
    oracle = _oracle()
    cu_q = _cu(lens_q)
    for b, (sq, skv) in enumerate(zip(lens_q, lens_kv)):
        if sq == 0:
            continue
        pos = torch.arange(sq, device=dev)[None]
        kvl = torch.tensor([skv], device=dev, dtype=torch.long)[:, None]
        if list_kind == "full":
            ids_b, lens_b = oracle.full_block_ids(pos, top_k, BS, kv_lens=kvl)
        else:
            ids_b, lens_b = oracle.random_block_ids(pos, top_k, BS, generator=g, kv_lens=kvl, shuffle=True)
        ids[cu_q[b] : cu_q[b + 1]] = ids_b[0]
        lens[cu_q[b] : cu_q[b + 1]] = lens_b[0]
    return q, k, v, ids.contiguous(), lens.contiguous()


def _lens_tensor(lens, cu, base=0):
    dev = torch.device("cuda")
    return torch.tensor(_cu(lens, base) if cu else [int(n) for n in lens], device=dev, dtype=torch.int32)


def _launch_thd(q, k, v, ids, lens, lens_q, lens_kv, top_k, scale, *, cu_q=False, cu_kv=False, cu_base=0, launches=2, gate=None):
    """Sentinel-filled packed O ``[1, T_q, H, D]`` / LSE ``(1, H, T_q)`` per launch through the adapter's THD contract; the
    metadata workspace sized by the adapter.  ``gate`` (appended) = the fused epilogue gate's PACKED operand, O-shaped in Q's dtype."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    dev = q.device
    _, T_q, H, _ = q.shape
    seq_q, seq_kv = _lens_tensor(lens_q, cu_q, cu_base), _lens_tensor(lens_kv, cu_kv, cu_base)
    sent = _sentinel(q.dtype)
    outs = []
    for _ in range(launches):
        o = torch.full((1, T_q, H, D), sent, device=dev, dtype=q.dtype)
        lse = torch.full((1, H, T_q), sent, device=dev, dtype=torch.float32)
        a = SparseGqaFwdDslSm107(
            q=q,
            k=k,
            v=v,
            o=o,
            lse=lse,
            block_ids=ids,
            block_lens=lens,
            seq_kv_lens=seq_kv,
            top_k=top_k,
            scale=scale,
            thd=True,
            seq_q_lens=seq_q,
            cu_seq_q_lens=cu_q,
            cu_seq_kv_lens=cu_kv,
            epilogue_gate=gate,
        )
        assert a.check_support()
        a.compile()  # plan time: the adapter never compiles on the execute path
        ws = torch.empty(a.scratch_workspace_bytes(), dtype=torch.uint8, device=dev)
        a.execute(stream=_stream(), workspace=ws)
        torch.cuda.synchronize()
        outs.append((o, lse))
    return outs


def _check_thd(outs, q, k, v, ids, lens, lens_q, lens_kv, scale, top_k):
    """Every sequence against ITS OWN reference (its Q rows, its K / V rows, its lists; a dead reference where it has no
    key) through the standard per-cell assertions; the capacity tail past the packed Q total untouched on every launch;
    no sequence carries the missed-coordinate signature (``LSE == log(n_vis)`` on most rows = an all-zero K operand, the
    THD-port class of bug).  Returns (max|dO|, max|dLSE|) over the live rows of every sequence."""
    (o, lse), *rest = outs
    sent, sent_o = _sentinel(o.dtype), _stored_sentinel(o.dtype)
    H = q.shape[2]
    cu_q, cu_k = _cu(lens_q), _cu(lens_kv)
    t_live = cu_q[-1]
    for oo, ll in outs:
        assert (oo[0, t_live:].float() == sent_o).all() and (ll[0, :, t_live:] == sent).all(), "the capacity tail past the packed Q total was written"
    max_o = max_lse = 0.0
    for b, (sq, skv) in enumerate(zip(lens_q, lens_kv)):
        if sq == 0:
            continue
        lo, hi = cu_q[b], cu_q[b + 1]
        klo, khi = cu_k[b], cu_k[b + 1]
        ids_b = ids[lo:hi]
        lens_b = None if lens is None else lens[lo:hi]
        if skv == 0:
            ref_o = torch.zeros(1, sq, H, D, device=q.device, dtype=torch.float32)
            ref_lse = torch.full((1, H, sq), float("-inf"), device=q.device, dtype=torch.float32)
        else:
            ref_o, ref_lse = _reference(q[:, lo:hi], k[:, klo:khi], v[:, klo:khi], ids_b, lens_b, [skv], scale, top_k)
        mo, ml = _check([(oo[:, lo:hi], ll[:, :, lo:hi]) for oo, ll in outs], ref_o, ref_lse)
        max_o, max_lse = max(max_o, mo), max(max_lse, ml)
        if skv > 0:
            pos = torch.arange(sq, device=q.device)
            sig = torch.log(torch.minimum(pos + 1, torch.full_like(pos, skv)).float())  # the LSE of an all-zero K operand
            lse_b = lse[0, :, lo:hi]
            frac = float(((lse_b - sig[None, :]).abs() < 1e-3).float().mean())
            assert frac < 0.5, f"sequence {b}: LSE == log(n_vis) on {frac:.0%} of the rows -- an all-zero K operand (a dense load coordinate survived the port)"
            if sq >= 2:
                assert (lse_b.std(dim=1) > 0).all(), f"sequence {b}: LSE constant across the rows of a head"
    return max_o, max_lse


_THD_PACKINGS = {
    "three-seqs": dict(lens_q=[300, 128, 200], lens_kv=[300, 128, 200], extra_cap=37),
    "long-four-shuffled": dict(lens_q=[2048, 4096, 6144, 8192], lens_kv=[2048, 4096, 6144, 8192], list_kind="shuffled"),
    "empty-front": dict(lens_q=[0, 300, 200], lens_kv=[0, 300, 200]),
    "empty-middle": dict(lens_q=[300, 0, 200], lens_kv=[300, 0, 200]),
    "empty-trailing": dict(lens_q=[300, 200, 0], lens_kv=[300, 200, 0], extra_cap=37),
    "five-token-seq": dict(lens_q=[5, 300, 128], lens_kv=[5, 300, 128]),
    "tokens-no-keys": dict(lens_q=[300, 128, 200], lens_kv=[300, 0, 200]),
    "ragged-kv-shorter": dict(lens_q=[300, 128], lens_kv=[257, 128]),
    "kv-longer-than-q": dict(lens_q=[128, 300], lens_kv=[300, 300]),
    # 34 live tokens x 2 KV heads = 68 units under a 71-row capacity grid of 142 CTAs: the CTAs at or past the live total exit at
    # entry (the only packing whose grid exceeds its units -- every other one fills the 204 SMs).
    "small-with-dead-ctas": dict(lens_q=[5, 20, 0, 9], lens_kv=[5, 20, 0, 9], list_kind="shuffled", extra_cap=37),
    # Zero live tokens: every CTA exits at entry, nothing is stored -- the whole capacity tail must still hold its sentinel.
    "all-empty": dict(lens_q=[0, 0], lens_kv=[0, 0], list_kind="shuffled", extra_cap=8),
}
_THD_F16_PACKINGS = ("three-seqs", "five-token-seq", "tokens-no-keys")


@requires_rubin
@pytest.mark.parametrize(
    "name, dtype",
    [pytest.param(n, torch.bfloat16, id=f"{n}-bf16") for n in _THD_PACKINGS] + [pytest.param(n, torch.float16, id=f"{n}-f16") for n in _THD_F16_PACKINGS],
)
def test_thd_packing_matches_the_per_sequence_oracle(name, dtype):
    """Every sequence of the packing against its own reference (sequence-relative lists, the sequence's own K / V rows):
    the packings [300, 128, 200] (with a 37-row capacity tail) and [2048, 4096, 6144, 8192] (random 512-subsets above the
    identity bound), a zero-length sequence at the front / in the middle / trailing, a 5-token sequence (never a length-1
    one), a sequence with Q tokens and NO key (every row dead: O = 0 / LSE = -inf stored exactly, its neighbours intact), a
    ragged ``S_kv_b < S_q_b`` (the tail follows the visible range) and ``S_kv_b > S_q_b``, a small packing whose capacity
    grid holds more CTAs than live units (the CTAs past the live total exit at entry) and an all-empty one (every CTA exits,
    nothing stored); two launches bitwise; the capacity tail untouched; no sequence with the missed-coordinate signature."""
    H, KH, top_k = 24, 2, 512
    p = _THD_PACKINGS[name]
    q, k, v, ids, lens = _thd_inputs(
        p["lens_q"], p["lens_kv"], H, KH, dtype, top_k=top_k, list_kind=p.get("list_kind", "full"), extra_cap=p.get("extra_cap", 0)
    )
    scale = 1.0 / math.sqrt(D)
    outs = _launch_thd(q, k, v, ids, lens, p["lens_q"], p["lens_kv"], top_k, scale)
    max_o, max_lse = _check_thd(outs, q, k, v, ids, lens, p["lens_q"], p["lens_kv"], scale, top_k)
    if name == "tokens-no-keys":
        cu_q = _cu(p["lens_q"])
        o, lse = outs[0]
        assert (o[0, cu_q[1] : cu_q[2]] == 0).all() and torch.isneginf(lse[0, :, cu_q[1] : cu_q[2]]).all(), "the keyless sequence is dead on every row"
    print(f"\nTHD {name} {dtype}: lens_q {p['lens_q']} lens_kv {p['lens_kv']}: max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f} (budget {ATOL})")


@requires_rubin
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
def test_thd_one_sequence_is_bitwise_the_dense_B1_run(dtype):
    """A one-sequence packing of T tokens computes the dense ``B = 1, S = T`` function BITWISE: the same gathers in the same
    order under both scheduler forms, only the item-to-CTA assignment differs (a shuffled list, ``block_lens`` given)."""
    T, H, KH, top_k = 600, 24, 2, 512
    q, k, v, ids, lens = _thd_inputs([T], [T], H, KH, dtype, top_k=top_k, list_kind="shuffled", seed=3)
    scale = 1.0 / math.sqrt(D)
    thd_outs = _launch_thd(q, k, v, ids, lens, [T], [T], top_k, scale)
    max_o, max_lse = _check_thd(thd_outs, q, k, v, ids, lens, [T], [T], scale, top_k)
    ((o_d, lse_d),) = _launch(q, k, v, ids.reshape(1, T, top_k), lens.reshape(1, T), None, top_k, scale, launches=1)
    o_t, lse_t = thd_outs[0]
    assert torch.equal(o_t, o_d) and torch.equal(lse_t, lse_d), "the packed one-sequence run must be bitwise the dense B = 1 run"
    print(f"\nTHD one sequence {dtype}: bitwise the dense run; max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}")


@requires_rubin
def test_thd_cumulative_length_forms_are_bitwise_the_lengths_form():
    """The lengths form vs the cumulative forms (Q only, KV only, both -- and a cumulative prefix sliced from a larger one,
    whose first entry the setup normalises away): bitwise equal outputs."""
    H, KH, top_k = 24, 2, 512
    lens_q, lens_kv = [300, 128, 200], [300, 0, 200]
    q, k, v, ids, lens = _thd_inputs(lens_q, lens_kv, H, KH, torch.bfloat16, top_k=top_k, seed=4)
    scale = 1.0 / math.sqrt(D)
    base = _launch_thd(q, k, v, ids, lens, lens_q, lens_kv, top_k, scale, launches=1)
    _check_thd(base, q, k, v, ids, lens, lens_q, lens_kv, scale, top_k)
    o0, lse0 = base[0]
    for cu_q, cu_kv, cu_base in ((True, False, 0), (False, True, 0), (True, True, 0), (True, True, 1000)):
        ((o1, lse1),) = _launch_thd(q, k, v, ids, lens, lens_q, lens_kv, top_k, scale, cu_q=cu_q, cu_kv=cu_kv, cu_base=cu_base, launches=1)
        assert torch.equal(o0, o1) and torch.equal(lse0, lse1), f"cu_q={cu_q} cu_kv={cu_kv} base={cu_base}: the length form changed the output"
    print("\nTHD length forms: lengths == cu_q == cu_kv == cu_both == sliced prefix (bitwise)")


@requires_rubin
def test_thd_block_lens_absent_uses_the_derived_count():
    """``block_lens`` absent under THD: the derived count per row (from the sequence-relative position) -- on full lists
    bitwise the run with the count given, within the budget vs every sequence's reference."""
    H, KH, top_k = 24, 2, 512
    lens_q, lens_kv = [5, 300, 128], [5, 300, 128]
    q, k, v, ids, lens = _thd_inputs(lens_q, lens_kv, H, KH, torch.bfloat16, top_k=top_k, seed=5)
    scale = 1.0 / math.sqrt(D)
    with_lens = _launch_thd(q, k, v, ids, lens, lens_q, lens_kv, top_k, scale, launches=1)
    without = _launch_thd(q, k, v, ids, None, lens_q, lens_kv, top_k, scale)
    max_o, max_lse = _check_thd(without, q, k, v, ids, None, lens_q, lens_kv, scale, top_k)
    assert torch.equal(with_lens[0][0], without[0][0]) and torch.equal(with_lens[0][1], without[0][1])
    print(f"\nTHD block_lens absent: bitwise the given count; max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}")


# ============================================================================ host: the fused epilogue gate's claims and pins
def test_adapter_accepts_the_gate_operand_and_selects_the_gated_specialization(sm107_host_target):
    """The record claims ``epilogue_gate``: a request WITH the gate operand (O-shaped in Q's dtype at its own strides -- here
    the slab's GATE columns at a padded token stride) is served and compiles the gated specialization
    (``TemplateParams.epilogue_gate``); a request without it compiles the ungated one.  The gate's dtype must be Q's."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    o = _operands()
    o.update(epilogue_gate=_T((1, 8, 24, D), (8 * 13312, 13312, D, 1), "torch.bfloat16"))
    a = SparseGqaFwdDslSm107(**o)
    assert a.check_support() is True
    assert a.template_params().epilogue_gate is True
    assert SparseGqaFwdDslSm107(**_operands()).template_params().epilogue_gate is False
    o16 = _operands(dtype="torch.float16")
    o16.update(epilogue_gate=_T((1, 8, 24, D), (8 * 24 * D, 24 * D, D, 1), "torch.float16"))
    assert SparseGqaFwdDslSm107(**o16).template_params().epilogue_gate is True


def test_sparse_kernel_gate_arm_source_pins():
    """The gate arm's shape in the source (no GPU): ONE arrive site per gate barrier (the TMA warp's ``pred=elect_sync()``
    expect_tx of GATE_TX_BYTES, the softmax warps' bare arrive after the last gate read), the gate loaded into the item's
    freed Q^T slot (no sGate SmemTile: the SmemTile count stays 4), the MATH through the shared helpers with the dead-column
    SELECT per element AFTER the fma, sigmoid's 1/2 folded into the scale, the two gate waits on the default (per-item) form
    (the ring-wait constant count stays 11), every gate byte count the BOX (never the slot, never a literal), the gate
    descriptor appended LAST so the ungated rendering's parameter offsets are untouched."""
    code = _code(_kernel_source())
    assert code.count("bars.mb_gate_full.arrive(n_bytes=GATE_TX_BYTES, pred=nvvm.elect_sync())") == 1
    assert code.count("bars.mb_gate_empty.arrive()") == 1
    assert code.count("tma_load_tile(sQ[qg_idx], tma_gate(") == 1, "the gate lands in the item's own Q^T slot"
    assert "sGate" not in code and code.count("SmemTile(") == 4
    assert code.count("spin=SPIN_RING_WAITS") == 11, "the gate waits are per-item waits on the default form"
    assert code.count("gate_epilogue_pairs(") == 1 and "gate_inv_sum(inv_j)" in code and code.count("gate_half_opaque()") == 1
    assert "_select_f32(dead_cols[j], ZERO, gated[j])" in code, "the dead-column SELECT per element, AFTER the gate fma"
    assert "GATE_TX_BYTES = CFG.GATE_TX_BYTES" in code and "n_bytes=Q_SLOT_BYTES" not in code and "n_bytes=6144" not in code
    i_sig = code.index("def _kernel(")
    sig = code[i_sig : code.index(") -> None:", i_sig)]
    # the LAST RUNTIME parameter: a trailing ``cutlass.Constexpr`` (the programmatic-dependent-launch switch, a compile-time specialization
    # the compiled call never carries) may follow it without moving any parameter offset
    runtime_params = [ln for ln in sig.splitlines() if ln.strip() and "cutlass.Constexpr[" not in ln]
    assert (
        runtime_params[-1].strip() == "tma_gate_desc: cutlass.GridConstant[tmap.TensorMap] = None,"
    ), "the gate descriptor is the LAST runtime kernel parameter"
    # the ungated rendering keeps its drains; the gated one consumes mb_gate_empty in ORDER as well as in count: the exit-path wait
    # BEFORE the last item's gate (a parity wait two completions behind a single-stage barrier never returns), then x STAGES_GATE
    assert "cutlass.range_constexpr(CFG.STAGES_GATE + 1)" not in code, "the drain is STAGES_GATE deep once the exit path waits before the last gate"
    assert code.count("cutlass.range_constexpr(CFG.STAGES_GATE)") == 1
    i_exit_wait = code.index("if is_valid <= cutlass.Int32(0):\n                bars.mb_gate_empty.wait(ge_state.phase)")
    i_last_gate = code.index("qgi, qgp = _tma_issue_gate(")
    i_drain = code.index("cutlass.range_constexpr(CFG.STAGES_GATE)")
    assert i_exit_wait < i_last_gate < i_drain, "the exit-path mb_gate_empty wait precedes the last gate's issue, the drain follows the loop"
    # the gate barriers' parities come from PipelineStates advanced by the ring depth like every other ring (P2) -- never a scalar toggle
    assert "_phase ^ cutlass.Int32(1)" not in code, "a gate barrier's parity is a PipelineState advanced by CFG.STAGES_GATE, never a scalar toggle"
    assert code.count("advance(ge_state, CFG.STAGES_GATE)") == 2 and code.count("advance(gf_state, CFG.STAGES_GATE)") == 1
    assert "ge_state = PipelineState.start(phase=1)" in code and "gf_state = PipelineState.start(phase=0)" in code


# ============================================================================ Rubin: the fused epilogue gate
def _ulp_of(x: torch.Tensor, dtype) -> torch.Tensor:
    """One unit in the last place of ``dtype`` at |x| (CPU fp32 math; ``torch.log2`` / ``exp2`` are avoided on the device),
    floored at the dtype's smallest subnormal step so a zero never yields a zero budget (QI01's helper, same derivation)."""
    mant, sub = (7, 2.0**-133) if dtype == torch.bfloat16 else (10, 2.0**-24)
    ax = x.detach().float().cpu().abs().clamp_min(sub)
    _, e = torch.frexp(ax)  # ax = m * 2^e, m in [0.5, 1) -> floor(log2 ax) = e - 1
    return torch.ldexp(torch.ones_like(ax), e - 1 - mant).clamp_min(sub)


def _elementwise_gate(o: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """The block's production stage (5) -- ``O_gated = O * sigmoid(GATE)`` on the HALF-ROUNDED O (the unfused pipeline) --
    the 'unfused-then-gated' reference of the fused arm."""
    from cudnn.gated_attention_block.kernels.elementwise import compile_elementwise_gate, run_elementwise_gate

    B, S, H, _ = o.shape
    r = compile_elementwise_gate(dtype=o.dtype, h=H, d=D, has_gate=True)
    src = o.reshape(B * S, H, D).contiguous()
    gt = gate.reshape(B * S, H, D).contiguous()
    dst = torch.empty_like(src)
    run_elementwise_gate(r, src, gt, dst, stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return dst.view(B, S, H, D)


def _gate_cell(*, B, S, H, KH, dtype, top_k, list_kind, kv_lens=None, block_lens=True, seed=0, gate_scale=2.0):
    """One gated cell: the gated kernel vs (a) the fp32 oracle gated exactly (the sacred budget), (b) the UNGATED kernel's O
    through the production elementwise gate -- 'unfused-then-gated' -- within ONE rounding of the dtype per element (the
    unfused path rounds O to the dtype before the gate; both use the same approximate tanh), (c) the ungated kernel's LSE
    BITWISE (the gate cannot touch it), (d) dead rows exactly 0.  Two gated launches bitwise.  Returns the magnitudes."""
    q, k, v, g = _inputs(B, S, H, KH, dtype, seed)
    ids, lens = _lists(list_kind, B, S, top_k, g, kv_lens)
    lens_arg = lens if block_lens else None
    scale = 1.0 / math.sqrt(D)
    # the gate: O-shaped, a wide pre-sigmoid range so sigmoid spans (0, 1); a slab-like padded token stride on half the cells
    gate_c = (torch.randn(B, S, H, D, device=q.device, dtype=torch.float32, generator=g) * gate_scale).to(dtype)
    if seed % 2 == 1:
        slab = torch.zeros(B, S, H * D + 512, device=q.device, dtype=dtype)
        slab[:, :, : H * D] = gate_c.reshape(B, S, H * D)
        gate = slab[:, :, : H * D].view(B, S, H, D)  # token stride H*D + 512: a column slice, bound with no copy
        assert gate.stride(1) == H * D + 512
    else:
        gate = gate_c
    outs_f = _launch(q, k, v, ids, lens_arg, kv_lens, top_k, scale, gate=gate)
    outs_u = _launch(q, k, v, ids, lens_arg, kv_lens, top_k, scale, launches=1)
    ref_o, ref_lse = _reference(q, k, v, ids, lens_arg, kv_lens, scale, top_k)
    ref_og = ref_o * torch.sigmoid(gate.float())  # the exact sigmoid on the fp32 oracle; a dead row's 0 stays 0
    max_o, max_lse = _check(outs_f, ref_og, ref_lse)
    (o_f, lse_f), (o_u, lse_u) = outs_f[0], outs_u[0]
    assert torch.equal(lse_f, lse_u), "the gate must not touch the LSE: gated and ungated renderings publish it bitwise"
    o_ug = _elementwise_gate(o_u, gate)
    dead = torch.isinf(ref_lse)  # [B, H, S]
    if dead.any():
        assert (o_f.float().permute(0, 2, 1, 3)[dead] == 0).all() and (o_ug.float().permute(0, 2, 1, 3)[dead] == 0).all(), "a dead row gates to exactly 0"
    d = (o_f.float() - o_ug.float()).abs()
    ulps = d.cpu() / _ulp_of(torch.maximum(o_f.float().abs(), o_ug.float().abs()), dtype)
    max_d, max_ulp = float(d.max()), float(ulps.max())
    n_off = int((d > 0).sum())
    assert max_ulp <= 1.0, f"gated vs unfused-then-gated: {max_ulp:.3f} ulp at the element's magnitude (max|d| {max_d:.3e}); the budget is one rounding"
    assert max_d <= ATOL
    return dict(max_o=max_o, max_lse=max_lse, max_d=max_d, max_ulp=max_ulp, n_off=n_off, n=int(d.numel()), dead=int(dead.sum()))


@requires_rubin
@pytest.mark.parametrize(
    "B, S, H, KH, dtype, top_k, list_kind, kv_lens, block_lens, seed",
    [
        pytest.param(1, 512, 24, 2, torch.bfloat16, 512, "full", None, True, 0, id="g12-bf16-s512-full-four-tiles"),
        pytest.param(1, 512, 24, 2, torch.float16, 512, "full", None, True, 1, id="g12-f16-s512-full-slab-gate"),
        pytest.param(1, 2176, 24, 2, torch.bfloat16, 512, "shuffled", None, True, 0, id="g12-bf16-s2176-shuffled-17-tiles"),
        pytest.param(1, 300, 2, 2, torch.bfloat16, 512, "shuffled", None, True, 1, id="g1-mha-bf16-s300"),
        pytest.param(1, 300, 10, 2, torch.bfloat16, 512, "shuffled", None, True, 0, id="g5-odd-bf16-s300"),
        pytest.param(1, 300, 16, 1, torch.bfloat16, 512, "shuffled", None, True, 1, id="g16-cap-bf16-s300"),
        pytest.param(3, 300, 24, 2, torch.bfloat16, 512, "full", [0, 257, 300], True, 0, id="g12-bf16-b3-kv-lens-dead-sequence"),
        pytest.param(1, 300, 24, 2, torch.bfloat16, 4, "shuffled", None, False, 1, id="g12-bf16-top4-no-block-lens"),
        pytest.param(2, 100, 8, 2, torch.bfloat16, 512, "full", None, True, 0, id="g4-bf16-b2-s100-one-tile"),
    ],
)
def test_epilogue_gate_matches_the_oracle_and_the_unfused_path_within_one_rounding(B, S, H, KH, dtype, top_k, list_kind, kv_lens, block_lens, seed):
    """The fused epilogue gate (``epilogue_gate=`` the gate operand): within the budget vs the fp32 oracle gated exactly, within
    ONE rounding of the dtype per element vs the UNGATED kernel's O through the production elementwise gate, the LSE bitwise
    the ungated kernel's, dead rows exactly 0, two launches bitwise -- over the GQA groups 1 / 4 / 5 (odd: the duplicated last
    pair) / 10 / 12 / 16, bf16 and f16, a compact and a slab-strided gate, one to seventeen tiles, a dead sequence, top_k = 4."""
    m = _gate_cell(B=B, S=S, H=H, KH=KH, dtype=dtype, top_k=top_k, list_kind=list_kind, kv_lens=kv_lens, block_lens=block_lens, seed=seed)
    print(
        f"\ngate cell B={B} S={S} H={H}/{KH} {dtype} top_k={top_k} {list_kind}: vs oracle max|dO| {m['max_o']:.5f} max|dLSE| {m['max_lse']:.6f}; "
        f"vs unfused-then-gated max|d| {m['max_d']:.4e} = {m['max_ulp']:.3f} ulp ({m['n_off']} of {m['n']} elements differ); dead rows {m['dead']}"
    )


@requires_rubin
@pytest.mark.parametrize(
    "name, dtype",
    [
        pytest.param("three-seqs", torch.bfloat16, id="three-seqs-bf16-capacity-tail"),
        pytest.param("tokens-no-keys", torch.float16, id="tokens-no-keys-f16-dead-sequence"),
        pytest.param("small-with-dead-ctas", torch.bfloat16, id="small-with-dead-ctas-bf16"),
    ],
)
def test_epilogue_gate_composes_with_thd_within_one_rounding(name, dtype):
    """The fused epilogue gate on the PACKED arm (``epilogue_gate=`` the packed ``[1, T_q, H, D]`` gate operand like O -- here the
    slab's GATE columns at a padded token stride): the gated packed run vs (a) every live sequence's own oracle gated exactly (the
    sacred budget), (b) the UNGATED packed run's O through the production elementwise gate within ONE rounding of the dtype per
    live element, (c) the ungated run's LSE BITWISE, (d) a keyless sequence's rows exactly 0 on both, the capacity tail past the
    packed total untouched, two gated launches bitwise.  The ungated run is held to the per-sequence oracle by ``_check_thd``."""
    H, KH, top_k = 24, 2, 512
    p = _THD_PACKINGS[name]
    q, k, v, ids, lens = _thd_inputs(
        p["lens_q"], p["lens_kv"], H, KH, dtype, top_k=top_k, list_kind=p.get("list_kind", "full"), extra_cap=p.get("extra_cap", 0), seed=7
    )
    T = q.shape[1]
    g = torch.Generator(device=q.device).manual_seed(0x6A7E)
    gate_c = (torch.randn(1, T, H, D, device=q.device, dtype=torch.float32, generator=g) * 2.0).to(dtype)
    slab = torch.zeros(1, T, H * D + 512, device=q.device, dtype=dtype)
    slab[:, :, : H * D] = gate_c.reshape(1, T, H * D)
    gate = slab[:, :, : H * D].view(1, T, H, D)  # token stride H*D + 512: the slab's GATE columns, bound with no copy
    assert gate.stride(1) == H * D + 512
    scale = 1.0 / math.sqrt(D)
    outs_f = _launch_thd(q, k, v, ids, lens, p["lens_q"], p["lens_kv"], top_k, scale, gate=gate)
    outs_u = _launch_thd(q, k, v, ids, lens, p["lens_q"], p["lens_kv"], top_k, scale, launches=1)
    max_o_u, max_lse_u = _check_thd(outs_u, q, k, v, ids, lens, p["lens_q"], p["lens_kv"], scale, top_k)
    (o_f, lse_f), (o_u, lse_u) = outs_f[0], outs_u[0]
    assert torch.equal(lse_f, lse_u), "the gate must not touch the LSE: gated and ungated packed runs publish it bitwise"
    sent, sent_o = _sentinel(dtype), _stored_sentinel(dtype)
    cu_q, cu_k = _cu(p["lens_q"]), _cu(p["lens_kv"])
    t_live = cu_q[-1]
    for oo, ll in outs_f:
        assert (oo[0, t_live:].float() == sent_o).all() and (ll[0, :, t_live:] == sent).all(), "the capacity tail past the packed Q total was written"
    assert torch.equal(outs_f[0][0], outs_f[1][0]) and torch.equal(outs_f[0][1], outs_f[1][1]), "two gated launches must be bitwise"
    o_ug = _elementwise_gate(o_u[:, :t_live], gate[:, :t_live])
    dead = torch.isinf(lse_u[0, :, :t_live])  # [H, t_live]
    if dead.any():
        assert (o_f[0, :t_live].float().permute(1, 0, 2)[dead] == 0).all() and (
            o_ug[0].float().permute(1, 0, 2)[dead] == 0
        ).all(), "a dead row gates to exactly 0"
    d = (o_f[0, :t_live].float() - o_ug[0].float()).abs()
    ulps = d.cpu() / _ulp_of(torch.maximum(o_f[0, :t_live].float().abs(), o_ug[0].float().abs()), dtype)
    max_d, max_ulp, n_off = float(d.max()), float(ulps.max()), int((d > 0).sum())
    assert max_ulp <= 1.0, f"gated vs unfused-then-gated: {max_ulp:.3f} ulp at the element's magnitude (max|d| {max_d:.3e}); the budget is one rounding"
    assert max_d <= ATOL
    max_o = max_lse = 0.0
    for b, (sq, skv) in enumerate(zip(p["lens_q"], p["lens_kv"])):
        if sq == 0 or skv == 0:
            continue  # an empty sequence owns no rows; a keyless one is dead on every row (asserted above)
        lo, hi, klo, khi = cu_q[b], cu_q[b + 1], cu_k[b], cu_k[b + 1]
        ref_o, ref_lse = _reference(q[:, lo:hi], k[:, klo:khi], v[:, klo:khi], ids[lo:hi], None if lens is None else lens[lo:hi], [skv], scale, top_k)
        ref_og = ref_o * torch.sigmoid(gate[:, lo:hi].float())  # the exact sigmoid on the fp32 oracle
        mo, ml = _check([(oo[:, lo:hi], ll[:, :, lo:hi]) for oo, ll in outs_f], ref_og, ref_lse)
        max_o, max_lse = max(max_o, mo), max(max_lse, ml)
    print(
        f"\nTHD gate {name} {dtype}: lens_q {p['lens_q']} lens_kv {p['lens_kv']}: vs oracle gated max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f} "
        f"(ungated run {max_o_u:.5f} / {max_lse_u:.6f}); vs unfused-then-gated max|d| {max_d:.4e} = {max_ulp:.3f} ulp ({n_off} of {d.numel()} "
        f"live elements differ); dead rows {int(dead.sum())}; capacity tail {T - t_live} rows untouched"
    )


# ============================================================================ Rubin: the DECODE form (bottom-right, one list per sequence, split over tiles)
def _decode_lists(kind, B, S_q, top_k, kv_lens, g):
    """ONE list per SEQUENCE at its step-0 position ``pos_0 = kv_len - S_q`` -> ``(block_ids int32 [B, top_k], block_lens int32
    [B])`` from the oracle's builders (``full``: the first complete blocks in order; ``shuffled``: a random ``top_k``-subset of the
    complete blocks in random order); a sequence shorter than the query count has a negative ``pos_0`` and an empty list."""
    oracle = _oracle()
    dev = torch.device("cuda")
    kvl = torch.tensor(kv_lens, device=dev, dtype=torch.long)
    pos0 = kvl - S_q
    if kind == "full":
        ids, lens = oracle.full_block_ids(pos0, top_k, BS, kv_lens=kvl)
    else:
        ids, lens = oracle.random_block_ids(pos0, top_k, BS, generator=g, kv_lens=kvl, shuffle=True)
    return ids.contiguous(), lens.contiguous()


def _reference_decode(q, k, v, ids_seq, lens_seq, kv_lens, scale, top_k, *, pos0_anchor=True):
    """The fp32 oracle of the decode form: row ``j`` of sequence ``b`` sits at ``pos_j = L_b - S_q + j`` (bottom-right) and sees the
    SHARED list anchored at ``pos_0 = L_b - S_q`` (``qsa_visible_mask(..., pos0=)``: the count from ``pos_0``, the tail from the
    step-0 tail start to ``pos_j``; ``pos0_anchor=False`` = the per-row-tail reading, which hides the block completed between
    ``pos_0`` and ``pos_j``); natural-log LSE; a row with ``pos_j < 0`` is dead.  Returns ``(O, LSE, sum_j p_j |v_j|)`` -- the third is
    the row's softmax-weighted mean of ``|V|`` the split budget needs."""
    oracle = _oracle()
    B, S, H, _ = q.shape
    SKV, KH = k.shape[1], k.shape[2]
    G = H // KH
    dev = q.device
    ref_o = torch.zeros(B, S, H, D, device=dev, dtype=torch.float32)
    ref_lse = torch.full((B, H, S), float("-inf"), device=dev, dtype=torch.float32)
    pv_abs = torch.zeros(B, S, H, D, device=dev, dtype=torch.float32)
    for b in range(B):
        L = int(kv_lens[b])
        pos0 = L - S
        pos = torch.arange(S, device=dev) + pos0
        ids_rows = ids_seq[b][None].expand(S, -1)
        lens_rows = None if lens_seq is None else lens_seq[b][None].expand(S)
        allowed = oracle.qsa_visible_mask(ids_rows, lens_rows, pos, L, SKV, BS, top_k=top_k, pos0=pos0 if pos0_anchor else None)
        for h in range(H):
            kk = k[b, :, h // G].float()
            vv = v[b, :, h // G].float()
            sc = (q[b, :, h].float() @ kk.t()) * scale
            sc = sc.masked_fill(~allowed, float("-inf"))
            rmax = sc.amax(dim=-1)
            dead = torch.isinf(rmax) & (rmax < 0)
            safe = torch.where(dead, torch.zeros_like(rmax), rmax)
            p = torch.where(allowed, torch.exp(sc - safe[:, None]), torch.zeros_like(sc))
            den = p.sum(dim=-1)
            den_safe = torch.where(dead, torch.ones_like(den), den)
            oc = (p @ vv) / den_safe[:, None]
            ref_o[b, :, h] = torch.where(dead[:, None], torch.zeros_like(oc), oc)
            ref_lse[b, h] = torch.where(dead, torch.full_like(den, float("-inf")), safe + torch.log(den))
            pv_abs[b, :, h] = (p @ vv.abs()) / den_safe[:, None]
    return ref_o, ref_lse, pv_abs


def _split_vs_unsplit(a_o, b_o, dtype, pv_abs, live, *, gate=None):
    """The DERIVED budget between a split and the unsplit run of one decode cell, per output element: ``ulp_dtype(max(|a|, |b|)) +
    sigmoid(g) x 2 eps_P x sum_j p_j |v_j| / l`` -- the output rounded ONCE from two fp32 values that associate the attention
    differently (one output ulp at the element's binade), and the half-dtype quantisation of P^T (each chunk's P at its OWN running
    max: every ``p_j`` carries a relative error ``<= eps_P = 2^-(mbits + 1)`` that differs between the two paths, so their difference
    is within twice the row's softmax-weighted mean of ``|V|``).  Over the live rows; returns ``(max |diff|, max diff / budget)``."""
    mbits = {torch.bfloat16: 7, torch.float16: 10}[dtype]
    eps_p = 2.0 ** -(mbits + 1)
    a64, b64 = a_o.double(), b_o.double()
    mag = torch.maximum(a64.abs(), b64.abs())
    ulp = _ulp_of(mag.float(), dtype).to(mag.device).double()
    weight = torch.sigmoid(gate.double()) if gate is not None else 1.0
    budget = ulp + weight * (2.0 * eps_p) * pv_abs.double()
    diff = (a64 - b64).abs()
    m = live[:, :, :, None].expand_as(diff)
    return float(diff[m].max()), float((diff[m] / budget[m]).max())


def _launch_decode(q, kv, ids, lens, kv_lens, top_k, scale, *, split=1, launches=2, gate=None, with_lse=True, list_per_sequence=True):
    """Sentinel-filled O / LSE per launch through the adapter's DECODE form (``list_per_sequence=True, bottom_right=True,
    split_kv=split``; ``list_per_sequence=False`` = the per-token bottom-right form, ``block_ids [B, S_q, top_k]``); ``kv`` = ``(k, v)``
    dense or ``(k_pool, v_pool, table, P)`` paged.  A split's workspace is sentinel-filled too: an unwritten partial slot would read
    as a huge live weight and the oracle comparison would catch it."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    dev = q.device
    B, S, H, _ = q.shape
    seq_kv = torch.tensor(kv_lens, device=dev, dtype=torch.int32)
    sent = _sentinel(q.dtype)
    paged = len(kv) == 4
    outs = []
    for _ in range(launches):
        o = torch.full((B, S, H, D), sent, device=dev, dtype=q.dtype)
        lse = torch.full((B, H, S), sent, device=dev, dtype=torch.float32) if with_lse else None
        kw = dict(paged_kv=True, page_size=kv[3], block_table=kv[2]) if paged else {}
        a = SparseGqaFwdDslSm107(
            q=q,
            k=kv[0],
            v=kv[1],
            o=o,
            lse=lse,
            block_ids=ids,
            block_lens=lens,
            seq_kv_lens=seq_kv,
            top_k=top_k,
            scale=scale,
            list_per_sequence=list_per_sequence,
            bottom_right=True,
            split_kv=split,
            epilogue_gate=gate,
            **kw,
        )
        assert a.check_support()
        a.compile()  # plan time: the kernel and, under a split, the combine
        ws = None
        if split > 1:
            ws = torch.full((a.scratch_workspace_bytes() // 4,), 1.5e30, device=dev, dtype=torch.float32)
        a.execute(stream=_stream(), workspace=ws)
        torch.cuda.synchronize()
        outs.append((o, lse, ws))
    return outs


def _decode_cell(*, B, SKV, kv_lens, S_q=1, H=24, KH=2, dtype=torch.bfloat16, top_k=512, list_kind="shuffled", split=1, paged=None, seed=0, block_lens=True):
    """One decode cell: the dense run (and, when ``paged`` = ``(P, hnd)``, the paged run of the same tokens: BITWISE the dense one),
    within the budget vs the oracle anchored at the step-0 position, two launches bitwise; under a split, the unsplit run of the same
    inputs (also within the budget) and the split-vs-unsplit magnitude against the derived two-term budget."""
    q, k, v, g = _inputs(B, S_q, H, KH, dtype, seed, SKV=SKV)
    ids, lens = _decode_lists(list_kind, B, S_q, top_k, kv_lens, g)
    lens_arg = lens if block_lens else None
    scale = 1.0 / math.sqrt(D)
    ref_o, ref_lse, pv_abs = _reference_decode(q, k, v, ids, lens_arg, kv_lens, scale, top_k)
    outs = _launch_decode(q, (k, v), ids, lens_arg, kv_lens, top_k, scale, split=split)
    max_o, max_lse = _check([(o, l) for o, l, _ in outs], ref_o, ref_lse)
    res = dict(max_o=max_o, max_lse=max_lse, dead=int(torch.isinf(ref_lse).sum()))
    if paged is not None:
        P, hnd = paged
        kp, vp, table = _paginate(k, v, kv_lens, P, hnd, "nan", g)
        pg = _launch_decode(q, (kp, vp, table, P), ids, lens_arg, kv_lens, top_k, scale, split=split, launches=1)
        _check([(pg[0][0], pg[0][1])], ref_o, ref_lse)
        _assert_bitwise((pg[0][0], pg[0][1]), (outs[0][0], outs[0][1]), f"decode paged page {P} {'HND' if hnd else 'NHD'} split {split}")
    if split > 1:
        un = _launch_decode(q, (k, v), ids, lens_arg, kv_lens, top_k, scale, split=1, launches=1)
        u_o, u_lse = _check([(un[0][0], un[0][1])], ref_o, ref_lse)
        live = ~torch.isinf(ref_lse).permute(0, 2, 1)  # [B, S, H]
        d, ratio = _split_vs_unsplit(outs[0][0].float(), un[0][0].float(), dtype, pv_abs, live)
        lse_d = float((outs[0][1] - un[0][1])[~torch.isinf(ref_lse)].abs().max()) if live.any() else 0.0
        assert (
            ratio <= 1.0
        ), f"split {split} vs unsplit: {ratio:.3f} of the derived budget (max |dO| {d:.3e}); the two paths differ beyond one rounding + the P quantisation"
        assert lse_d <= ATOL
        res.update(unsplit_max_o=u_o, split_max_d=d, split_ratio=ratio, split_lse_d=lse_d)
    return res


@requires_rubin
@pytest.mark.parametrize(
    "B, SKV, kv_lens, dtype, top_k, list_kind, split, paged, block_lens",
    [
        pytest.param(1, 2052, [2052], torch.bfloat16, 512, "full", 1, None, True, id="B1-S2052-full-list-omits-one-block"),
        pytest.param(1, 2052, [2052], torch.bfloat16, 512, "shuffled", 4, None, True, id="B1-S2052-split4"),
        pytest.param(4, 8192, [8192, 2052, 4097, 7000], torch.bfloat16, 512, "shuffled", 1, (16, True), True, id="B4-S8K-paged16-HND"),
        pytest.param(4, 8192, [8192, 2052, 4097, 7000], torch.bfloat16, 512, "shuffled", 2, (16, True), False, id="B4-S8K-paged16-split2-no-block-lens"),
        pytest.param(4, 8192, [8192, 2052, 4097, 7000], torch.bfloat16, 512, "shuffled", 17, (16, True), True, id="B4-S8K-paged16-split17-one-tile-per-chunk"),
        pytest.param(4, 32768, [32768, 20000, 32768, 4096], torch.bfloat16, 512, "shuffled", 1, (64, False), True, id="B4-S32K-paged64-NHD"),
        pytest.param(4, 32768, [32768, 20000, 32768, 4096], torch.bfloat16, 512, "shuffled", 8, (64, False), True, id="B4-S32K-paged64-split8"),
        pytest.param(1, 32768, [32768], torch.float16, 512, "shuffled", 17, None, True, id="B1-S32K-f16-split17"),
        pytest.param(2, 8192, [8192, 300], torch.bfloat16, 64, "shuffled", 3, None, True, id="B2-top64-split3-max-tiles"),
        pytest.param(4, 2052, [2052, 0, 2052, 1], torch.bfloat16, 512, "full", 4, (16, True), True, id="B4-dead-sequence-and-one-token-sequence-split4"),
    ],
)
def test_decode_form_ladder(B, SKV, kv_lens, dtype, top_k, list_kind, split, paged, block_lens):
    """The decode form at ``S_q = 1`` (one item per (sequence, KV head)): B in {1, 4}, S_kv in {2052, 8K, 32K}, dense and paged (16
    HND / 64 NHD: BITWISE the dense read), bf16 and f16, the full list that omits one block at 2052 and random lists, per-sequence
    KV lengths incl. a dead (0) and a one-token (1: no complete block, the tail only) sequence, ``block_lens`` given and absent;
    splits 1 / 2 / 3 / 4 / 8 / 17 (17 = one tile per chunk at top_k 512, 3 = the maximum at top_k 64): within the budget vs the oracle
    anchored at the step-0 position, split == unsplit within the derived budget, two launches bitwise."""
    r = _decode_cell(B=B, SKV=SKV, kv_lens=kv_lens, dtype=dtype, top_k=top_k, list_kind=list_kind, split=split, paged=paged, block_lens=block_lens)
    extra = ""
    if split > 1:
        extra = f"; unsplit max|dO| {r['unsplit_max_o']:.5f}; split-vs-unsplit max|dO| {r['split_max_d']:.3e} = {r['split_ratio']:.3f} of the derived budget, |dLSE| {r['split_lse_d']:.2e}"
    print(
        f"\ndecode B={B} S_kv={SKV} lens={kv_lens} {dtype} top_k={top_k} {list_kind} split={split} paged={paged}: max|dO| {r['max_o']:.5f} max|dLSE| {r['max_lse']:.6f}; dead rows {r['dead']}{extra}"
    )


@requires_rubin
def test_decode_form_dead_sequence_partials_are_the_combine_identity():
    """``kv_lens = [0, 4096, 0]`` at split 5: the dead sequences' rows are exactly ``0`` / ``-inf`` in O / LSE, AND every partial
    slot the kernel wrote for them in the workspace is exactly ``O_s = 0`` / ``LSE_s = -inf`` (the combine's identity: a dead chunk
    is a dead item, never residue); the live sequence's empty chunks (its 17 tiles over 5 chunks leave none empty here) and the
    unsplit run agree within the budget."""
    B, SKV, kv_lens, S, H, KH, top_k, split = 3, 4096, [0, 4096, 0], 1, 24, 2, 512, 5
    dtype = torch.bfloat16
    q, k, v, g = _inputs(B, S, H, KH, dtype, 11, SKV=SKV)
    ids, lens = _decode_lists("shuffled", B, S, top_k, kv_lens, g)
    scale = 1.0 / math.sqrt(D)
    ref_o, ref_lse, _ = _reference_decode(q, k, v, ids, lens, kv_lens, scale, top_k)
    outs = _launch_decode(q, (k, v), ids, lens, kv_lens, top_k, scale, split=split)
    max_o, max_lse = _check([(o, l) for o, l, _ in outs], ref_o, ref_lse)
    o, lse, ws = outs[0]
    assert (o[0] == 0).all() and (o[2] == 0).all() and torch.isinf(lse[0]).all() and torch.isinf(lse[2]).all()
    o_part = ws[: B * split * S * H * D].view(B * split, S, H, D)
    lse_part = ws[B * split * S * H * D : B * split * S * H * D + B * split * H * S].view(B * split, H, S)
    for s_ in range(split):
        for b in (0, 2):
            assert (o_part[b + s_ * B] == 0).all(), f"dead sequence {b} chunk {s_}: the O partial must be exactly 0"
            assert torch.isinf(lse_part[b + s_ * B]).all() and (lse_part[b + s_ * B] < 0).all(), f"dead sequence {b} chunk {s_}: the LSE partial must be -inf"
        assert torch.isfinite(lse_part[1 + s_ * B]).all(), "the live sequence's chunks are all non-empty at 17 tiles over 5 chunks"
    print(f"\ndead-sequence partials: identity on 2 x {split} slots; max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}")


@requires_rubin
def test_decode_form_other_sequence_rows_in_the_open_block_are_masked():
    """The A5 decode row: ``kv_lens = [98, 37]`` (``L % 4 != 0``: the open block straddles the length inside the last page; SHORT
    sequences, so the 2-3 rows past the length would weigh ~5 % of the softmax mass if gathered) over page-16 pools whose unwritten
    rows hold the OTHER sequence's tokens -- BITWISE the NaN-filled pools and the dense read, split and unsplit; the oracle over the
    same tokens with those rows PRESENT (lengths rounded up to the block end) is far from the kernel, so the rows carry weight when
    visible.  (The long-sequence paged cells of the ladder are bitwise the dense read too; at 4K visible keys three rows weigh too
    little for a teeth assertion.)"""
    B, SKV, kv_lens, S, H, KH, top_k, P = 2, 112, [98, 37], 1, 24, 2, 512, 16
    dtype = torch.bfloat16
    q, k, v, g = _inputs(B, S, H, KH, dtype, 5, SKV=SKV)
    ids, lens = _decode_lists("shuffled", B, S, top_k, kv_lens, g)
    scale = 1.0 / math.sqrt(D)
    ref_o, ref_lse, _ = _reference_decode(q, k, v, ids, lens, kv_lens, scale, top_k)
    for split in (1, 4):
        dense = _launch_decode(q, (k, v), ids, lens, kv_lens, top_k, scale, split=split, launches=1)
        g2 = torch.Generator(device=q.device).manual_seed(31)
        kp_n, vp_n, t_n = _paginate(k, v, kv_lens, P, True, "nan", g2)
        g3 = torch.Generator(device=q.device).manual_seed(31)
        kp_o, vp_o, t_o = _paginate(k, v, kv_lens, P, True, "other", g3)
        assert torch.equal(t_n, t_o)
        out_n = _launch_decode(q, (kp_n, vp_n, t_n, P), ids, lens, kv_lens, top_k, scale, split=split, launches=1)
        out_o = _launch_decode(q, (kp_o, vp_o, t_o, P), ids, lens, kv_lens, top_k, scale, split=split, launches=1)
        max_o, max_lse = _check([(out_n[0][0], out_n[0][1])], ref_o, ref_lse)
        _assert_bitwise((out_n[0][0], out_n[0][1]), (out_o[0][0], out_o[0][1]), f"split {split}: NaN-filled vs other-sequence-filled unwritten rows")
        _assert_bitwise((out_n[0][0], out_n[0][1]), (dense[0][0], dense[0][1]), f"split {split}: paged vs dense")
        vis_o, _, _ = _reference_decode(q, k, v, ids, lens, [100, 40], scale, top_k)
        teeth = float((out_n[0][0].float() - vis_o).abs().max())
        assert teeth > ATOL, f"the open block's rows past the length must carry weight when visible (diff {teeth})"
        print(
            f"\nopen-block rows beyond the length (split {split}): NaN fill == other fill == dense bitwise; max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}; teeth {teeth:.4f}"
        )


@requires_rubin
@pytest.mark.parametrize("split", [1, 4], ids=["fused-epilogue-gate", "gate-in-the-combine-split4"])
def test_decode_form_gate_rides_the_epilogue_or_the_combine(split):
    """The gate on the decode form: unsplit, the kernel's fused epilogue gate; split, the UNGATED kernel + the gated combine (the
    gate applied to the fp32 merged value, one rounding) -- within the budget vs the oracle gated exactly, dead rows exactly 0, the
    gate-free LSE within the budget of the oracle; the split run vs the unsplit fused-gate run within the derived budget scaled by
    sigmoid(g) (the two paths apply the same gate arithmetic before the single rounding)."""
    B, SKV, kv_lens, S, H, KH, top_k = 3, 8192, [8192, 0, 3000], 1, 24, 2, 512
    dtype = torch.bfloat16
    q, k, v, g = _inputs(B, S, H, KH, dtype, 7, SKV=SKV)
    ids, lens = _decode_lists("shuffled", B, S, top_k, kv_lens, g)
    scale = 1.0 / math.sqrt(D)
    gate = (torch.randn(B, S, H, D, device=q.device, dtype=torch.float32, generator=g) * 2.0).to(dtype)
    ref_o, ref_lse, pv_abs = _reference_decode(q, k, v, ids, lens, kv_lens, scale, top_k)
    ref_og = ref_o * torch.sigmoid(gate.float())
    outs = _launch_decode(q, (k, v), ids, lens, kv_lens, top_k, scale, split=split, gate=gate)
    max_o, max_lse = _check([(o, l) for o, l, _ in outs], ref_og, ref_lse)
    o, lse, _ = outs[0]
    assert (o[1] == 0).all() and torch.isinf(lse[1]).all(), "the dead sequence gates to exactly 0"
    if split > 1:
        fused = _launch_decode(q, (k, v), ids, lens, kv_lens, top_k, scale, split=1, launches=1, gate=gate)
        live = ~torch.isinf(ref_lse).permute(0, 2, 1)
        d, ratio = _split_vs_unsplit(o.float(), fused[0][0].float(), dtype, pv_abs, live, gate=gate.float())
        assert ratio <= 1.0, f"gate in the combine vs the fused epilogue gate: {ratio:.3f} of the derived budget (max |dO| {d:.3e})"
        print(
            f"\ngate in the combine (split {split}): vs oracle max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}; vs the fused gate max|dO| {d:.3e} = {ratio:.3f} of the budget"
        )
    else:
        print(f"\nfused epilogue gate on the decode form: max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}")


@requires_rubin
@pytest.mark.parametrize("S_q", [2, 4])
def test_decode_form_mtp_rows_share_the_step0_list(S_q):
    """``S_q`` rows per sequence sharing the step-0 list (the MTP verify rows; 4 single-token items per (sequence, KV head)):
    sequence 0 is SHORT with its step-0 position at ``4k + 2`` (``S_q = 2``) / ``4k + 1`` (``S_q = 4``), so a later row sits at
    ``pos_j = 4k + 3`` and sees the block that completes between ``pos_0`` and ``pos_j`` (two tail ids at ``S_q = 4``); sequence 1's
    step-0 row has NO tail (``pos_0 = 2051``: ``(pos_0 + 1) % 4 == 0``) and its later rows gain the completing block; sequence 2 is
    shorter than the query count (``L = S_q - 1``: row 0 dead, the rest live) -- within the budget vs the oracle anchored at ``pos0``;
    and the per-row-tail reading (``pos0=None``, which hides the completing block from the row at ``pos_j = 4k + 3``) differs beyond
    the budget on that row of the short sequence (the hidden block weighs ~4 / 56 of its mass), so the shared-list semantics have
    teeth.  Split 2 vs split 1 within the derived budget."""
    B, SKV, H, KH, top_k = 3, 2056, 24, 2, 512
    pos0_teeth = 54 if S_q == 2 else 53  # 4k + 2 / 4k + 1: the row at 4k + 3 is row 1 / row 2
    j_teeth = (4 * 13 + 3) - pos0_teeth
    kv_lens = [pos0_teeth + S_q, 2051 + S_q, S_q - 1]
    assert (kv_lens[1] - S_q + 1) % 4 == 0 and 0 < j_teeth < S_q
    dtype = torch.bfloat16
    q, k, v, g = _inputs(B, S_q, H, KH, dtype, 13, SKV=SKV)
    ids, lens = _decode_lists("shuffled", B, S_q, top_k, kv_lens, g)
    scale = 1.0 / math.sqrt(D)
    ref_o, ref_lse, pv_abs = _reference_decode(q, k, v, ids, lens, kv_lens, scale, top_k)
    assert torch.isinf(ref_lse[2, :, 0]).all() and torch.isfinite(ref_lse[2, :, 1:]).all(), "the short sequence: row 0 dead (pos_0 < 0), the rest live"
    outs = _launch_decode(q, (k, v), ids, lens, kv_lens, top_k, scale, split=1)
    max_o, max_lse = _check([(o, l) for o, l, _ in outs], ref_o, ref_lse)
    # teeth: the per-row-tail reading differs on the short sequence's 4k + 3 row (the completing block hidden from it)
    alt_o, alt_lse, _ = _reference_decode(q, k, v, ids, lens, kv_lens, scale, top_k, pos0_anchor=False)
    teeth = float((outs[0][0][0, j_teeth].float() - alt_o[0, j_teeth]).abs().max())
    teeth_lse = float((outs[0][1][0, :, j_teeth] - alt_lse[0, :, j_teeth]).abs().max())
    assert teeth > ATOL and teeth_lse > ATOL, f"the shared-list form must differ from the per-row-tail reading on the 4k + 3 row ({teeth} / {teeth_lse})"
    # the step-0 row of every sequence reads the same under both anchors (no later token has been seen)
    assert float((outs[0][0][:, 0].float() - alt_o[:, 0]).abs().max()) <= ATOL
    sp = _launch_decode(q, (k, v), ids, lens, kv_lens, top_k, scale, split=2, launches=1)
    _check([(sp[0][0], sp[0][1])], ref_o, ref_lse)
    live = ~torch.isinf(ref_lse).permute(0, 2, 1)
    d, ratio = _split_vs_unsplit(sp[0][0].float(), outs[0][0].float(), dtype, pv_abs, live)
    assert ratio <= 1.0
    print(
        f"\nMTP S_q={S_q} lens={kv_lens}: max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}; per-row-tail teeth {teeth:.4f} / {teeth_lse:.4f}; split2 vs unsplit {d:.3e} = {ratio:.3f}"
    )


@requires_rubin
def test_decode_form_bottom_right_with_per_token_lists():
    """``bottom_right=True`` alone (per-token lists at the bottom-right positions ``pos_j = L_b - S_q + j``): ``S_q = 8`` over
    ``kv_lens = [300, 100, 5]`` (the third sequence shorter than the query count: rows 0..2 dead, 3..7 live) -- within the budget vs
    the oracle at the per-row positions (no shared list)."""
    B, SKV, kv_lens, S, H, KH, top_k = 3, 300, [300, 100, 5], 8, 24, 2, 512
    dtype = torch.bfloat16
    q, k, v, g = _inputs(B, S, H, KH, dtype, 17, SKV=SKV)
    oracle = _oracle()
    dev = q.device
    kvl = torch.tensor(kv_lens, device=dev, dtype=torch.long)[:, None]
    positions = (kvl - S) + torch.arange(S, device=dev)[None, :]  # [B, S], negative on the short sequence's first rows
    ids, lens = oracle.random_block_ids(positions, top_k, BS, generator=g, kv_lens=kvl, shuffle=True)
    ids, lens = ids.contiguous(), lens.contiguous()
    scale = 1.0 / math.sqrt(D)
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    sent = _sentinel(dtype)
    outs = []
    for _ in range(2):
        o = torch.full((B, S, H, D), sent, device=dev, dtype=dtype)
        lse = torch.full((B, H, S), sent, device=dev, dtype=torch.float32)
        a = SparseGqaFwdDslSm107(
            q=q,
            k=k,
            v=v,
            o=o,
            lse=lse,
            block_ids=ids,
            block_lens=lens,
            seq_kv_lens=kvl[:, 0].to(torch.int32).contiguous(),
            top_k=top_k,
            scale=scale,
            bottom_right=True,
        )
        assert a.check_support()
        a.compile()
        a.execute(stream=_stream())
        torch.cuda.synchronize()
        outs.append((o, lse))
    # the oracle at the per-row positions
    ref_o = torch.zeros(B, S, H, D, device=dev, dtype=torch.float32)
    ref_lse = torch.full((B, H, S), float("-inf"), device=dev, dtype=torch.float32)
    G = H // KH
    for b in range(B):
        L = kv_lens[b]
        allowed = oracle.qsa_visible_mask(ids[b], lens[b], positions[b], L, SKV, BS, top_k=top_k)
        for h in range(H):
            sc = (q[b, :, h].float() @ k[b, :, h // G].float().t()) * scale
            sc = sc.masked_fill(~allowed, float("-inf"))
            rmax = sc.amax(dim=-1)
            dead = torch.isinf(rmax) & (rmax < 0)
            safe = torch.where(dead, torch.zeros_like(rmax), rmax)
            p = torch.where(allowed, torch.exp(sc - safe[:, None]), torch.zeros_like(sc))
            den = p.sum(dim=-1)
            oc = (p @ v[b, :, h // G].float()) / torch.where(dead, torch.ones_like(den), den)[:, None]
            ref_o[b, :, h] = torch.where(dead[:, None], torch.zeros_like(oc), oc)
            ref_lse[b, h] = torch.where(dead, torch.full_like(den, float("-inf")), safe + torch.log(den))
    assert torch.isinf(ref_lse[2, :, :3]).all() and torch.isfinite(ref_lse[2, :, 3:]).all()
    max_o, max_lse = _check(outs, ref_o, ref_lse)
    print(f"\nbottom-right per-token lists S_q={S} lens={kv_lens}: max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}; dead rows {int(torch.isinf(ref_lse).sum())}")


def test_sparse_kernel_softmax_split_source_pins():
    """The softmax SPLIT's shape in the source (no GPU): every split term is a ``const_expr`` fold of the module constant SOFTMAX_GROUPS (the
    one-column-group rendering traces the pre-split program -- its cubin is byte-identical), the group's named barrier id is ``2 + grp`` as a
    runtime value, the S^T load / the O^T rescale / the epilogue load take the group's column offset, the P^T chunk lands at the group's word
    offset of the lane's row, the per-group column max / sum read ONE group's four warps, and no arrive site, ring wait, tile loop or SmemTile
    was added (the counts stay 11 / 3 / 4); the config's per-group constants are what the body reads."""
    code = _code(_kernel_source())
    for name in (
        "COLS = CFG.COLS_PER_GROUP",
        "SOFTMAX_GROUPS = CFG.SOFTMAX_GROUPS",
        "SOFTMAX_WARPS_PER_GROUP = CFG.SOFTMAX_WARPS_PER_GROUP",
        "SOFTMAX_LANES = CFG.SOFTMAX_LANES",
    ):
        assert name in code, name
    assert code.count("cutlass.const_expr(SOFTMAX_GROUPS > 1)") >= 14, "the split terms fold at one column group"
    assert "grp = (sm_warp // cutlass.Int32(SOFTMAX_WARPS_PER_GROUP)) if cutlass.const_expr(SOFTMAX_GROUPS > 1) else cutlass.Int32(0)" in code
    assert "bar_softmax = (cutlass.Int32(_BAR_SOFTMAX) + grp) if cutlass.const_expr(SOFTMAX_GROUPS > 1) else _BAR_SOFTMAX" in code
    assert code.count("nvvm.barrier_cta_sync(barrier_id=bar_softmax, thread_count=BAR_SOFTMAX_THREADS)") == 2, "the exchange barriers take the group's id"
    assert "s_off = s_off + col0" in code and code.count("o_col = o_col + col0") == 1 and "o_col_e = o_col_e + col0" in code
    assert "p_row_base = p_row_base + grp * cutlass.Int32(P_GROUP_WORDS)" in code, "the group's 16-B chunk of the lane's P^T row"
    assert code.count("red_read = red_read + grp * cutlass.Int32(SOFTMAX_WARPS_PER_GROUP * COLS)") == 1
    assert code.count("red_epi_read = red_epi_read + grp * cutlass.Int32(SOFTMAX_WARPS_PER_GROUP * COLS)") == 1
    assert code.count("for w in cutlass.range_constexpr(SOFTMAX_WARPS_PER_GROUP):") == 2, "a column max / sum combines ONE group's four warps"
    assert "for w in cutlass.range_constexpr(SOFTMAX_WARPS):" not in code
    assert "n_live_cols = (cutlass.Int32(Q_BOX_ROWS) - col0) if cutlass.const_expr(SOFTMAX_GROUPS > 1) else cutlass.Int32(Q_BOX_ROWS)" in code
    assert code.count("spin=SPIN_RING_WAITS") == 11 and code.count("cutlass.range(0, n_tiles, 1, unroll=1)") == 3 and code.count("SmemTile(") == 4
    assert (
        code.count("bars.mb_s_empty[par].arrive()") == 1
        and code.count("bars.mb_p_full[par].arrive()") == 1
        and code.count("bars.mb_tmem_dealloc.arrive()") == 1
    )
    assert "COLS = N_Q" not in code, "the column count per group is the config's, never N_Q"


# ============================================================================ Rubin: the MTP matrix of the shared-list decode form
def _mtp_kv_lens(S_q, r, B):
    """``kv_lens`` with the step-0 position ``pos_0 = kv_len - S_q = 4k + r`` on every live sequence.  ``B = 1``: one saturated-list
    sequence (``8192 + r + S_q``: 2048 complete blocks, 512 selected, 17 tiles); ``B = 4``: the SHORT sequence (``56 + r + S_q``,
    ``k = 14`` -- the 4 keys of one hidden block weigh ~7 % of a row's mass), the IDENTITY-BOUND sequence (``2048 + r + S_q``,
    ``k = 512``: exactly 512 complete blocks at ``r < 3``, 513 at ``r = 3`` so the 512-list omits one), the saturated one, and the
    DEGENERATE sequence of ``S_q - 1`` tokens (``pos_0 = -1``: row 0 dead, the rest tail-only; the whole sequence dead at ``S_q = 1``)."""
    if B == 1:
        return [8192 + r + S_q]
    return [56 + r + S_q, 2048 + r + S_q, 8192 + r + S_q, S_q - 1]


def _mtp_rows(S_q, r):
    """The shared-list arithmetic per row ``j`` (``pos_j = 4k + r + j``): the number of tail blocks the kernel appends,
    ``n_tail(j) = ceil((pos_j + 1) / 4) - floor((pos_0 + 1) / 4)`` (0, 1 or 2); the rows where the per-row-tail reading (the tail from
    the row's OWN ``4 floor((pos_j + 1) / 4)``) hides block ``k`` -- ``r < 3`` and ``pos_j >= 4k + 3``, i.e. ``j >= 3 - r``; and the
    ``4k + 3`` row itself when the step has one.  Returns ``(n_tail, teeth_rows, row_4k3)``."""
    three = 1 if r == 3 else 0
    n_tail = [-(-(r + j + 1) // 4) - three for j in range(S_q)]
    teeth = [j for j in range(S_q) if r < 3 and r + j >= 3]
    row_4k3 = (3 - r) if (r < 3 and 3 - r < S_q) else None
    return n_tail, teeth, row_4k3


def _row_masks(oracle, ids_b, lens_b, L, S_q, SKV, top_k):
    """The oracle's visible sets of one sequence's ``S_q`` rows under the shared-list (``pos0``) and the per-row-tail readings."""
    dev = ids_b.device
    pos0 = L - S_q
    pos = torch.arange(S_q, device=dev) + pos0
    ids_rows = ids_b[None].expand(S_q, -1)
    lens_rows = lens_b[None].expand(S_q)
    vis0 = oracle.qsa_visible_mask(ids_rows, lens_rows, pos, L, SKV, BS, top_k=top_k, pos0=pos0)
    vis_row = oracle.qsa_visible_mask(ids_rows, lens_rows, pos, L, SKV, BS, top_k=top_k)
    return vis0, vis_row


def _plant_margin(q, k, b, row, head, G, pos, allowed, scale):
    """The planted key's score on (``b``, ``row``, ``head``) minus the largest OTHER visible score, in nats (fp32 on the half-rounded
    operands; ``allowed`` = the row's visible set)."""
    sc = (q[b, row, head].float() @ k[b, :, head // G].float().t()) * scale
    sc = sc.masked_fill(~allowed, float("-inf"))
    others = sc.clone()
    others[pos] = float("-inf")
    return float(sc[pos] - others.max())


@requires_rubin
@pytest.mark.parametrize("B", [1, 4], ids=["B1", "B4"])
@pytest.mark.parametrize("r", [0, 1, 2, 3], ids=[f"pos0-mod4-{r}" for r in range(4)])
@pytest.mark.parametrize("S_q", [1, 2, 3, 4], ids=[f"Sq{s}" for s in (1, 2, 3, 4)])
def test_decode_form_mtp_matrix(S_q, r, B):
    """The MTP matrix of the shared-list decode form: ``S_q`` rows per sequence sharing the step-0 list, at every residue
    ``r = pos_0 mod 4`` of the step-0 position and ``B`` in {1, 4} (the lengths of :func:`_mtp_kv_lens`).  Per live sequence and row:
    the residue holds; the oracle's shared-list set past the listed blocks is exactly the ``n_tail(j)`` blocks from the step-0 tail
    start (0, 1 or 2 -- two at ``(r, j)`` in {(1, 3), (2, 2), (2, 3)}, where the ids sit at list indices 512 / 513 of a saturated list);
    the kernel equals the pos0-anchored oracle (the module's budget; dead rows exact; two launches bitwise); the per-row-tail reading
    differs on EXACTLY the rows ``j >= 3 - r`` (``r < 3``), by exactly block ``k``, and coincides (bitwise in the oracle) everywhere
    else -- per-row causal on the coinciding rows; on the SHORT sequence every differing row's kernel output is beyond the budget from
    the per-row-tail oracle (natural teeth); on the saturated sequence a dominant key PLANTED at the ``4k + 3`` row's own token (or the
    last row's, when the step has no ``4k + 3`` row) for one query head is attended by that row (cos > 0.99 with the planted V), never
    by the rows before it (a leak would sit >= 10 budgets off), with a margin >= 8 nats, and -- when it is the ``4k + 3`` row -- sits at
    cos < 0.5 from the per-row-tail oracle that hides block ``k``.  ``B = 1``: split 2 == unsplit within the derived budget; ``B = 4``:
    the paged read (page 16 HND / 64 NHD) bitwise the dense read, and the degenerate ``S_q - 1``-token sequence (row 0 dead, the rest
    tail-only).  The planted rows' O is the planted V itself (entries ~N(0, 1), |O| up to ~3-4): one bf16 output rounding there is
    2^-7 .. 2^-6, so these cells sit at max|dO| ~0.01 against the fp32 oracle -- one rounding of a large value, inside the module's
    budget, not a kernel miss (the ladder's unplanted rows sit at ~5e-4); the seeds are fixed, so the magnitudes are reproducible."""
    H, KH, top_k = 24, 2, 512
    G = H // KH
    dtype = torch.bfloat16
    kv_lens = _mtp_kv_lens(S_q, r, B)
    SKV = max(kv_lens)
    n_tail, teeth_rows, row_4k3 = _mtp_rows(S_q, r)
    q, k, v, g = _inputs(B, S_q, H, KH, dtype, seed=100 + 16 * S_q + 4 * r + B, SKV=SKV)
    # the plant: on the saturated sequence, at the 4k + 3 row's own token (else the last row's), for one query head of KV head 0
    b_long = 0 if B == 1 else 2
    j_plant = row_4k3 if row_4k3 is not None else S_q - 1
    h_plant = 5
    pos_plant = kv_lens[b_long] - S_q + j_plant
    k[b_long, pos_plant, h_plant // G] = q[b_long, :, h_plant].float().sum(0).to(dtype)
    ids, lens = _decode_lists("shuffled", B, S_q, top_k, kv_lens, g)
    # at (S_q, r) = (1, 3) the plant row's own token lies in a block COMPLETE at step 0 (pos_0 + 1 = 4k + 4: no open block), which
    # the shuffled 512-list of 2049 blocks usually omits -- list it, so the row sees its own token through the LIST (both readings)
    blk_plant = pos_plant // BS
    plant_listed = blk_plant < (kv_lens[b_long] - S_q + 1) // BS
    if plant_listed and not bool((ids[b_long] == blk_plant).any()):
        ids[b_long, int(lens[b_long]) - 1] = blk_plant
    scale = 1.0 / math.sqrt(D)
    ref_o, ref_lse, pv_abs = _reference_decode(q, k, v, ids, lens, kv_lens, scale, top_k)
    alt_o, alt_lse, _ = _reference_decode(q, k, v, ids, lens, kv_lens, scale, top_k, pos0_anchor=False)
    outs = _launch_decode(q, (k, v), ids, lens, kv_lens, top_k, scale, split=1)
    max_o, max_lse = _check([(o, l) for o, l, _ in outs], ref_o, ref_lse)
    o, lse, _ = outs[0]
    oracle = _oracle()
    dev = q.device
    t = torch.arange(SKV, device=dev)
    report = []
    for b in range(B):
        L = kv_lens[b]
        pos0 = L - S_q
        if pos0 < 0:
            dead_rows = min(S_q, -pos0)
            assert torch.isinf(ref_lse[b, :, :dead_rows]).all() and torch.isfinite(ref_lse[b, :, dead_rows:]).all()
            report.append(f"seq {b} L={L}: {dead_rows} dead row(s), the rest tail-only")
            continue
        assert pos0 % 4 == r
        vis0, vis_row = _row_masks(oracle, ids[b], lens[b], L, S_q, SKV, top_k)
        tail_start = 4 * ((pos0 + 1) // 4)
        for j in range(S_q):
            pos_j = pos0 + j
            # the tail: every token from the step-0 tail start to the row's own position, covering exactly n_tail(j) blocks; nothing past it
            tail = vis0[j] & (t >= tail_start)
            assert torch.equal(tail, (t >= tail_start) & (t <= pos_j)), f"seq {b} row {j}: the tail must be [{tail_start}, {pos_j}]"
            assert pos_j // 4 - tail_start // 4 + 1 == n_tail[j], f"seq {b} row {j}: {pos_j // 4 - tail_start // 4 + 1} tail blocks, expected {n_tail[j]}"
            assert not vis0[j][t > pos_j].any(), f"seq {b} row {j}: a key past the row's own position is visible"
            # the two readings differ on exactly the teeth rows, by exactly block k; elsewhere the oracles coincide bitwise
            differs = not torch.equal(vis0[j], vis_row[j])
            assert differs == (
                j in teeth_rows
            ), f"seq {b} row {j}: readings {'differ' if differs else 'coincide'}, expected {'teeth' if j in teeth_rows else 'coincidence'}"
            if differs:
                hidden = vis0[j] & ~vis_row[j]
                assert int(hidden.sum()) == 4 and not (vis_row[j] & ~vis0[j]).any() and int(hidden.nonzero().min()) == 4 * (pos0 // 4)
                assert not torch.equal(ref_o[b, j], alt_o[b, j])
            else:
                assert torch.equal(ref_o[b, j], alt_o[b, j]) and torch.equal(ref_lse[b, :, j], alt_lse[b, :, j])
        if L < 100:  # natural teeth on the short sequence: the hidden block's 4 keys weigh more than the budget
            for j in teeth_rows:
                d = float((o[b, j].float() - alt_o[b, j]).abs().max())
                dl = float((lse[b, :, j] - alt_lse[b, :, j]).abs().max())
                assert d > ATOL and dl > ATOL, f"seq {b} row {j} (pos {pos0 + j}): the hidden block k must weigh more than the budget ({d:.4f} / {dl:.4f})"
                report.append(f"seq {b} row {j} teeth {d:.4f}/{dl:.4f}")
        if b == b_long:  # the plant
            margin = _plant_margin(q, k, b, j_plant, h_plant, G, pos_plant, vis0[j_plant], scale)
            assert margin >= 8.0, f"plant margin {margin:.2f} nats"
            v_plant = v[b, pos_plant, h_plant // G].float()
            cos_p = float(torch.nn.functional.cosine_similarity(o[b, j_plant, h_plant].float(), v_plant, dim=0))
            assert cos_p > 0.99, f"the plant row {j_plant} must attend its own token (cos {cos_p:.4f})"
            for j in range(j_plant):  # per-row causal: the rows before the plant row never see it
                assert not vis0[j, pos_plant] and not vis_row[j, pos_plant]
                d = float((o[b, j, h_plant].float() - v_plant).abs().max())
                assert d >= 10 * ATOL, f"row {j} sits at {d:.4f} from the planted V: a future key leaked"
            if j_plant in teeth_rows:  # the 4k + 3 row sees block k, which the per-row-tail reading hides
                assert bool(vis0[j_plant, pos_plant]) and not bool(vis_row[j_plant, pos_plant])
                d = float((o[b, j_plant, h_plant].float() - alt_o[b, j_plant, h_plant]).abs().max())
                cos_a = float(torch.nn.functional.cosine_similarity(o[b, j_plant, h_plant].float(), alt_o[b, j_plant, h_plant], dim=0))
                assert cos_a < 0.5 and d >= 10 * ATOL, f"the 4k + 3 row must see block k: vs the per-row-tail oracle cos {cos_a:.3f}, diff {d:.4f}"
                report.append(
                    f"plant row {j_plant} (4k+3, pos {pos_plant}): margin {margin:.1f} nats, cos {cos_p:.4f} with V, vs per-row-tail cos {cos_a:.3f} diff {d:.3f}"
                )
            else:
                assert bool(vis0[j_plant, pos_plant]) and bool(vis_row[j_plant, pos_plant])
                where = "a listed complete block" if plant_listed else "the open block"
                report.append(f"plant row {j_plant} (last row, pos {pos_plant}, {where}): margin {margin:.1f} nats, cos {cos_p:.4f} with V")
    if B == 1:  # MTP x split
        sp = _launch_decode(q, (k, v), ids, lens, kv_lens, top_k, scale, split=2, launches=1)
        _check([(sp[0][0], sp[0][1])], ref_o, ref_lse)
        live = ~torch.isinf(ref_lse).permute(0, 2, 1)
        d, ratio = _split_vs_unsplit(sp[0][0].float(), o.float(), dtype, pv_abs, live)
        lse_d = float((sp[0][1] - lse)[~torch.isinf(ref_lse)].abs().max())
        assert ratio <= 1.0 and lse_d <= ATOL, f"split 2 vs unsplit: {ratio:.3f} of the derived budget, |dLSE| {lse_d:.2e}"
        extra = f"; split2 vs unsplit {d:.3e} = {ratio:.3f} of the budget"
    else:  # MTP x paged
        P, hnd = (16, True) if S_q % 2 == 0 else (64, False)
        kp, vp, table = _paginate(k, v, kv_lens, P, hnd, "nan", g)
        pg = _launch_decode(q, (kp, vp, table, P), ids, lens, kv_lens, top_k, scale, split=1, launches=1)
        _check([(pg[0][0], pg[0][1])], ref_o, ref_lse)
        _assert_bitwise((pg[0][0], pg[0][1]), (o, lse), f"MTP S_q {S_q} r {r} paged page {P} {'HND' if hnd else 'NHD'}")
        extra = f"; paged {P} {'HND' if hnd else 'NHD'} bitwise the dense read"
    print(
        f"\nMTP matrix S_q={S_q} r={r} B={B} lens={kv_lens}: max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}; n_tail {n_tail}; teeth rows {teeth_rows}; "
        + "; ".join(report)
        + extra
    )


def _per_token_lists_from_shared(ids_seq, lens_seq, kv_lens, S_q):
    """The per-token bottom-right lists that spell the SHARED-list semantics explicitly: row ``j`` of sequence ``b`` = the step-0 list
    (its ``count`` entries, replicated) + the block completed between ``pos_0`` and ``pos_j`` (``floor((pos_j + 1) / 4) - tail_lo``: 0 or
    1 at ``S_q <= 4``) appended at index ``count``; ``block_lens`` = ``count`` + that.  The kernel derives the row's own open block in
    both forms, so the tiles hold the same blocks in the same order.  Needs ``count + 1 <= top_k`` (clamp ``lens_seq`` first)."""
    B, top_k = ids_seq.shape
    ids3 = ids_seq[:, None, :].expand(B, S_q, top_k).clone()
    lens2 = torch.zeros(B, S_q, dtype=torch.int32, device=ids_seq.device)
    for b in range(B):
        L = int(kv_lens[b])
        pos0 = L - S_q
        assert pos0 >= 0
        count = min(max(int(lens_seq[b]), 0), min(top_k, (pos0 + 1) // 4))
        tail_lo = min(pos0 + 1, L) // 4
        for j in range(S_q):
            n_between = (pos0 + j + 1) // 4 - tail_lo
            assert 0 <= n_between <= 1 and count + n_between <= top_k
            if n_between:
                ids3[b, j, count] = tail_lo
            lens2[b, j] = count + n_between
    return ids3.contiguous(), lens2.contiguous()


@requires_rubin
@pytest.mark.parametrize("r", [0, 1, 2, 3], ids=[f"pos0-mod4-{r}" for r in range(4)])
@pytest.mark.parametrize("S_q", [2, 4], ids=["Sq2", "Sq4"])
def test_decode_form_mtp_shared_list_is_bitwise_the_replicated_per_token_list(S_q, r):
    """The step-0 reuse pin: the shared-list form is BITWISE the per-token bottom-right form (``block_ids [B, S_q, top_k]``,
    ``list_per_sequence=False``) fed with the step-0 list replicated on every row plus the block completed between ``pos_0`` and
    ``pos_j`` appended explicitly (:func:`_per_token_lists_from_shared`) -- the drafted rows share ONE list, the appended tail ids
    are exactly the derived ones and the count anchor is the step-0 position, or the two renderings would gather different tiles.
    ``block_lens`` on the shared form is clamped to ``top_k - 2`` so the replicated row has room; sequences at ``56 + r + S_q`` (short)
    and ``8192 + r + S_q`` (a saturated list), both within the budget vs the pos0 oracle."""
    B, H, KH, top_k = 2, 24, 2, 512
    dtype = torch.bfloat16
    kv_lens = [56 + r + S_q, 8192 + r + S_q]
    q, k, v, g = _inputs(B, S_q, H, KH, dtype, seed=200 + 4 * S_q + r, SKV=max(kv_lens))
    ids, lens = _decode_lists("shuffled", B, S_q, top_k, kv_lens, g)
    lens = torch.minimum(lens, torch.full_like(lens, top_k - 2)).contiguous()
    scale = 1.0 / math.sqrt(D)
    ref_o, ref_lse, _ = _reference_decode(q, k, v, ids, lens, kv_lens, scale, top_k)
    shared = _launch_decode(q, (k, v), ids, lens, kv_lens, top_k, scale, split=1)
    max_o, max_lse = _check([(o, l) for o, l, _ in shared], ref_o, ref_lse)
    ids3, lens2 = _per_token_lists_from_shared(ids, lens, kv_lens, S_q)
    appended = int((lens2 > lens[:, None]).sum())
    assert appended == B * len(
        [j for j in range(S_q) if r < 3 and r + j >= 3]
    ), "a block is appended exactly on the rows where the per-row-tail reading would hide one"
    per_tok = _launch_decode(q, (k, v), ids3, lens2, kv_lens, top_k, scale, split=1, launches=1, list_per_sequence=False)
    _check([(per_tok[0][0], per_tok[0][1])], ref_o, ref_lse)
    _assert_bitwise((per_tok[0][0], per_tok[0][1]), (shared[0][0], shared[0][1]), f"S_q {S_q} r {r}: the replicated per-token list vs the shared list")
    print(
        f"\nshared list == replicated per-token list (bitwise) S_q={S_q} r={r} lens={kv_lens}: max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}; rows with an appended block {appended}"
    )


# ============================================================================ the split combine's FORM pin (decode-grid / lean walk / the module's own forms)
# The kernel names the CUDA profiler reports for the combine's forms (the shared combine module's own pins name them the same way).
_COMBINE_DECODE_GRID_KERNEL = "cudnn_kernel__combine_decode_grid_kernel"
_COMBINE_LEAN_WALK_KERNEL = "cudnn_kernel__combine_lean_walk_kernel"
_COMBINE_OWN_FORM_KERNELS = ("cudnn_kernel__combine_kernel", "cudnn_kernel__combine_decode_kernel")  # the wide lean group / serial walk, the decode form
_FORM_KERNEL = {"decode_grid": (_COMBINE_DECODE_GRID_KERNEL,), "lean_walk": (_COMBINE_LEAN_WALK_KERNEL,), "own": _COMBINE_OWN_FORM_KERNELS}
# (B, S_q, split) -> the FORM the sparse adapter pins its combine entry to on cc 10.7 at 24 Q heads (the dense adapter's rule on the
# sparse grid): the serving decode cells (B 1 / B 4 at split 8) take the decode-grid kernel, the B 32 x S_q 1 step at split <= 4 the lean
# group walk, a small grid below split 8 and the mid-sized B 4 x S_q 4 grid at split 7 the module's own forms, an overflow grid past the
# walk's split bound the AUTO entry (= the module's own forms there by construction).
_FORM_CELLS = {
    (1, 1, 8): "decode_grid",
    (4, 1, 8): "decode_grid",
    (1, 2, 16): "decode_grid",
    (1, 1, 17): "decode_grid",
    (32, 1, 4): "lean_walk",
    (32, 1, 8): "auto_overflow",
    (4, 4, 7): "own",
    (1, 1, 4): "own",
}


def _form_value(comb, form):
    return {"decode_grid": comb.DECODE_GRID_AUTO, "auto_overflow": comb.DECODE_GRID_AUTO, "lean_walk": comb.DECODE_GRID_LEAN_WALK, "own": comb.DECODE_GRID_OFF}[
        form
    ]


def test_sparse_combine_form_pin_follows_the_dense_rule_per_grid(sm107_host_target):
    """``_sparse_combine_decode_grid`` -- the FORM the sparse adapter compiles a split plan's combine entry with -- is the dense adapter's
    rule (``prepared._split_combine_decode_grid``) on the sparse grid ``(B, H_q, S_q)``: on cc 10.7 the decode-grid kernel on the small
    grids from split 8 (B 1 / B 4 x 24 heads: the serving decode cells; the 17-split item too), the lean group walk on the overflow grid
    (B 32 x S_q 1) at split <= 4, the module's own forms where the rule forces them (a small grid below split 8, B 4 x S_q 4 at split 7),
    the AUTO entry on an overflow grid past the walk's bound; off cc 10.7 never the walk and never the AUTO entry inside the bound.  The
    adapter reads the same value back (``combine_form()``, after ``check_support`` on the host-modelled target), ``DECODE_GRID_OFF``
    unsplit; the two halves of the rule name every pin."""
    from types import SimpleNamespace

    from cudnn.sdpa.fwd import heuristics as heur
    from cudnn.sdpa.fwd import prepared as prep
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107, _sparse_combine_decode_grid

    H, KH, rubin = 24, 2, (10, 7)
    for (b, s_q, split), form in _FORM_CELLS.items():
        want = _form_value(comb, form)
        assert _sparse_combine_decode_grid(b, H, s_q, split, rubin) == want, (b, s_q, split, form)
        # the ONE source of the pin: the dense adapter's rule on the same (B, H_q, S_q, split) at the same device
        assert prep._split_combine_decode_grid(SimpleNamespace(_device_cc=rubin), b, H, s_q, split) == want, (b, s_q, split)
        walk = heur.decode_combine_lean_walk_selected(b, H, s_q, split, device_cc=rubin)
        forced = heur.decode_combine_force_lean(b, H, s_q, split, device_cc=rubin)
        assert walk is (form == "lean_walk") and forced is (form == "own"), (b, s_q, split, form, walk, forced)
        inside = heur.decode_combine_lean_blocks(b, H, s_q) <= heur.DECODE_COMBINE_MAX_LEAN_BLOCKS
        assert inside is (form != "lean_walk" and form != "auto_overflow"), (b, s_q, split)
        # off the measured line: the module's own forms inside the bound (forced OFF), the AUTO entry outside it (its own forms there)
        for other in ((10, 0), (10, 3), (12, 0), (9, 0)):
            assert _sparse_combine_decode_grid(b, H, s_q, split, other) == (comb.DECODE_GRID_OFF if inside else comb.DECODE_GRID_AUTO), (b, s_q, split, other)
        # the adapter's readback (device_cc (10, 7) in the stand-ins; the target modelled on the host)
        a = SparseGqaFwdDslSm107(**_decode_operands(B=b, S=s_q, H=H, KH=KH, split_kv=split))
        assert a.check_support() is True and a.combine_form() == want, (b, s_q, split, form)
    assert SparseGqaFwdDslSm107(**_decode_operands(B=1, S=1, H=H, KH=KH, split_kv=1)).combine_form() == comb.DECODE_GRID_OFF, "unsplit: no combine"
    # the selector's vocabulary is the combine module's (the pin names ITS constants; a value the module does not know is refused by compile_ptr)
    assert {comb.DECODE_GRID_AUTO, comb.DECODE_GRID_OFF, comb.DECODE_GRID_LEAN_WALK} == {0, 2, 3}


def test_adapter_compiles_the_combine_with_the_pinned_form_and_the_pdl_switch(monkeypatch, sm107_host_target):
    """The split combine's ``compile_ptr`` call carries ``decode_grid=`` (the pinned form) and ``use_pdl=`` (the adapter's switch; False
    here) per plan -- recorded through a stub (the kernel template stubbed too: no device), the form the adapter recorded equal to the
    kwarg; an unsplit plan compiles no combine."""
    from functools import lru_cache
    from types import SimpleNamespace

    import cudnn.frost.compiled_cache as cache
    import cudnn.frost.template_loader as loader
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    seen = []

    @lru_cache(None)
    def kernel(**kwargs):
        return lambda **kw: None

    def combine(**kwargs):
        seen.append(dict(kwargs))
        return lambda *a: None

    kernel_kw = []

    def kernel_recording(**kwargs):
        kernel_kw.append(dict(kwargs))
        return lambda **kw: None

    monkeypatch.setattr(loader, "load_template", lambda *a, **kw: SimpleNamespace(compile=kernel))
    monkeypatch.setattr(comb, "compile_ptr", combine)
    monkeypatch.setattr(cache, "positional_entry", lambda owner: owner)
    for (b, s_q, split), form in _FORM_CELLS.items():
        a = SparseGqaFwdDslSm107(**_decode_operands(B=b, S=s_q, H=24, KH=2, split_kv=split))
        a.compile()
        kw = seen[-1]
        assert kw["decode_grid"] == _form_value(comb, form) == a._combine_form and kw["use_pdl"] is False, (b, s_q, split, kw)
        assert (kw["dtype_o"], kw["dtype_partial"], kw["has_lse"], kw["stats_log2"], kw["gate"], kw["dtype_gate"]) == ("bf16", "f32", True, False, False, None)
    n = len(seen)
    SparseGqaFwdDslSm107(**_decode_operands(B=1, S=1, H=24, KH=2, split_kv=1)).compile()
    assert len(seen) == n, "an unsplit plan compiles no combine"
    # the gated split: the gate rides the combine's entry (dtype_gate named), the form pinned the same way
    kw = _decode_operands(B=4, S=1, H=24, KH=2, split_kv=8)
    kw.update(epilogue_gate=_T((4, 1, 24, D), (24 * D, 24 * D, D, 1), "torch.bfloat16"))
    SparseGqaFwdDslSm107(**kw).compile()
    assert seen[-1]["gate"] is True and seen[-1]["dtype_gate"] == "bf16" and seen[-1]["decode_grid"] == comb.DECODE_GRID_AUTO
    # the switch reaches BOTH compiles: the kernel's compile key (use_pdl=) and the combine's entry (use_pdl=)
    monkeypatch.setattr(loader, "load_template", lambda *a, **kw: SimpleNamespace(compile=kernel_recording))
    for pdl in (False, True):
        SparseGqaFwdDslSm107(**_decode_operands(B=1, S=1, H=24, KH=2, split_kv=8), use_pdl=pdl).compile()
        assert seen[-1]["use_pdl"] is pdl and seen[-1]["decode_grid"] == comb.DECODE_GRID_AUTO
        assert kernel_kw[-1]["use_pdl"] is pdl and kernel_kw[-1]["has_lse"] is True, kernel_kw[-1]  # the declared block_lens variant's compile


def test_adapter_serves_use_pdl_and_declines_it_with_thd_by_name(sm107_host_target):
    """``use_pdl=True`` on the sparse adapter is SERVED on the dense BSHD and paged forms (the decode form split and unsplit, the prefill
    form): the flag is a recorded constructor fact that reaches the kernel's compile key and the combine's entry; ``thd=True`` with the
    switch is a typed decline naming the measured fact (the packed form reads the setup launch's metadata above every role wait).  The
    inversion of the former decline pin.  The kernel source carries the sites: the pair imported from ``tile_dsl.barrier``, ONE wait per
    role that reads a predecessor's output (TMA, gather, MMA, softmax: four), ONE release per CTA after ``fence_mbarrier_init +
    barrier_cta_sync`` and before the role dispatch, the constexpr threaded ``_kernel`` -> every role and ``_host`` -> the launch attribute,
    ``compile(use_pdl=)`` in the key -- and NO new barrier, ring, SMEM tile or ring wait.  ONE source of the THD fact: the adapter raises
    ``pdl_decline_reason(thd=True)`` verbatim and the gated attention block's declaration-time decline carries the same statement -- the two
    sites are pinned equal here, so neither text drifts alone when the kernel's PDL scope moves."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107, pdl_decline_reason

    for kw in (_decode_operands(B=1, S=1, H=24, KH=2, split_kv=8), _decode_operands(B=4, S=1, H=24, KH=2, split_kv=1, P=16), _operands(S=8)):
        off = SparseGqaFwdDslSm107(**kw)
        assert off.use_pdl is False and off.check_support() is True
        on = SparseGqaFwdDslSm107(**kw, use_pdl=True)
        assert on.use_pdl is True and on.check_support() is True
    thd = _operands(B=1, S=64, H=24, KH=2)
    thd.update(thd=True, seq_q_lens=_T((2,), (1,), "torch.int32"), seq_kv_lens=_T((2,), (1,), "torch.int32"))
    assert SparseGqaFwdDslSm107(**thd).check_support() is True
    with pytest.raises(NotImplementedError, match="thd=True with use_pdl=True is not served") as e:
        SparseGqaFwdDslSm107(**thd, use_pdl=True).check_support()
    assert "setup launch" in str(e.value) and "declare use_pdl=False" in str(e.value)
    assert str(e.value) == f"sparse d256 forward (sm107): {pdl_decline_reason(thd=True)}" and pdl_decline_reason(thd=False) is None
    from cudnn.gated_attention_block import GatedAttentionBlockFwd, GatedAttentionBlockGeometry, QsaSpec

    geom = GatedAttentionBlockGeometry(d_model=512, h_q=8, h_kv=2, d_head=D, rope_dim=64, qsa=QsaSpec())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mk = lambda *shape: torch.empty(*shape, dtype=torch.bfloat16, device=dev)  # noqa: E731
    with pytest.raises(NotImplementedError) as eb:  # the block's packed QsaSpec declaration with the switch: the adapter's statement verbatim
        GatedAttentionBlockFwd(
            mk(32, 512),
            mk(geom.n_qkvg, 512),
            mk(D),
            mk(D),
            mk(32, 64),
            mk(32, 64),
            mk(512, 8 * D),
            mk(32, 512),
            geom,
            thd=True,
            num_sequences=2,
            max_seq_len=32,
            token_capacity=32,
            use_pdl=True,
        )
    assert pdl_decline_reason(thd=True) in str(eb.value), str(eb.value)
    code = _code(_kernel_source())
    assert "from cudnn.frost.tile_dsl.barrier import MBarrier, PipelineState, Producer, advance, wait, wait_on_dependent_grids" in code
    assert (
        code.count("        wait_on_dependent_grids()") == 4
    ), "one dependent-grid wait per role that reads a predecessor's output (TMA, gather, MMA, softmax)"
    # A PLACEMENT pin (never a tolerance): the core triggers its dependent grid at its COMPLETION -- no explicit release anywhere.  A
    # release at the prologue (the first form) left the split combine's CTAs resident through the core's whole run and cost 3.2-4.4 us
    # per step at B <= 4 on the cc 10.7 class; the same programmatic edge triggered at completion costs nothing (the header's PDL SITES).
    assert code.count("        launch_dependent_grids()") == 0, "no explicit release: the CTA exit is the trigger (the measured placement)"
    assert "launch_dependent_grids" not in code, "the helper is not even imported: nothing may release above the mainloop"
    assert (
        code.count("if cutlass.const_expr(use_pdl):") == 4
    ), "the four waits, each a const_expr arm, and nothing else (the False rendering traces the program as before)"
    # the former release site -- between the init fence + CTA sync and the role dispatch -- holds no dependent-grid instruction at all
    i_sync = code.index("nvvm.fence_mbarrier_init()\n    nvvm.barrier_cta_sync()")
    i_disp = code.index("if warp_idx < SOFTMAX_WARPS:\n        _softmax_warp_group(")
    assert i_sync < i_disp and "dependent_grids" not in code[i_sync:i_disp]
    # every role's wait precedes its first decode / bounds / issue, and the pre-dispatch THD exit stays ABOVE the init sync, hence above
    # every wait (the declined composition)
    for role in ("_tmaldg_warp_group", "_gather_warp_group", "_mma_warp_group", "_softmax_warp_group"):
        body = code[code.index(f"def {role}(") :]
        body = body[: body.index("\ndef ", 1)]
        assert "use_pdl: cutlass.Constexpr[bool] = False" in body, f"{role} takes the constexpr"
        assert body.index("        wait_on_dependent_grids()") < body.index("= _decode_item("), f"{role}: the wait precedes the first decode"
    assert code.index("exit_if_dead_thd_cluster(seq_kv_lens_tensor, n_batch, 1)") < i_sync
    assert code.count("use_pdl=use_pdl,") == 5, "the four role calls and the launch attribute"
    assert "use_pdl: cutlass.Constexpr[bool] = False,\n) -> None:\n    warp_idx = cute.arch.make_warp_uniform" in code, "the kernel's constexpr, appended last"
    assert "    use_pdl: cutlass.Constexpr[bool] = False,\n    stream: _cuda_driver.CUstream = None," in code, "the host's constexpr before the stream"
    assert "    use_pdl: bool = False,\n) -> Callable:" in code and "        bool(use_pdl),\n        stream=cute.runtime.make_fake_stream" in code
    assert code.count("MBarrier(_alloc(") == 13 and code.count("SmemTile(") == 4 and code.count("spin=SPIN_RING_WAITS") == 11


def _combine_kernels_of(fn):
    """The combine kernel names (``cudnn_kernel__combine*``) the CUDA profiler sees while ``fn`` runs; ``None`` when the node recorded
    no CUDA event at all (CUPTI unavailable: the caller skips the name assertion by name)."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    if not events:
        return None
    return {e.name for e in events if e.name.startswith("cudnn_kernel__combine")}


def _former_combine_entry(gate: bool):
    """The combine entry the sparse adapter compiled BEFORE the form pin: ``compile_ptr`` without ``decode_grid`` (= ``DECODE_GRID_OFF``,
    the module's own three-way rule on every device) and without ``use_pdl`` -- the develop entry, run here over the adapter's own partials."""
    import cudnn.frost.compiled_cache as cache
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    return cache.positional_entry(
        comb.compile_ptr(dtype_o="bf16", dtype_partial="f32", has_lse=True, stats_log2=False, gate=gate, dtype_gate="bf16" if gate else None)
    )


def _pinned_form_cell(B, S_q, split, gate, with_lse, seed, use_pdl=False):
    """One decode cell through the adapter with the pinned combine form (and, ``use_pdl``, the programmatic-dependent-launch form of both
    launches): returns ``(adapter, o, lse, ws, gate_t, ref_o, ref_lse, run)`` -- ``run`` executes the adapter once more into the same O /
    LSE / workspace (the census hook)."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    H, KH, top_k, dtype = 24, 2, 512, torch.bfloat16
    kv_lens = {1: [8192], 4: [8192, 2052, 4097, 7000], 6: [8192, 300, 2052, 4097, 7000, 1024]}[B]
    SKV = 8192
    q, k, v, g = _inputs(B, S_q, H, KH, dtype, seed, SKV=SKV)
    ids, lens = _decode_lists("shuffled", B, S_q, top_k, kv_lens, g)
    scale = 1.0 / math.sqrt(D)
    gate_t = (torch.randn(B, S_q, H, D, device=q.device, dtype=torch.float32, generator=g) * 2.0).to(dtype) if gate else None
    ref_o, ref_lse, _ = _reference_decode(q, k, v, ids, lens, kv_lens, scale, top_k)
    if gate:
        ref_o = ref_o * torch.sigmoid(gate_t.float())
    dev = q.device
    sent = _sentinel(dtype)
    o = torch.full((B, S_q, H, D), sent, device=dev, dtype=dtype)
    lse = torch.full((B, H, S_q), sent, device=dev, dtype=torch.float32) if with_lse else None
    a = SparseGqaFwdDslSm107(
        q=q,
        k=k,
        v=v,
        o=o,
        lse=lse,
        block_ids=ids,
        block_lens=lens,
        seq_kv_lens=torch.tensor(kv_lens, device=dev, dtype=torch.int32),
        top_k=top_k,
        scale=scale,
        list_per_sequence=True,
        bottom_right=True,
        split_kv=split,
        epilogue_gate=gate_t,
        use_pdl=use_pdl,
    )
    assert a.check_support()
    a.compile()
    ws = torch.full((max(a.scratch_workspace_bytes(), 16) // 4,), 1.5e30, device=dev, dtype=torch.float32)

    def run():
        a.execute(stream=_stream(), workspace=ws if split > 1 else None)

    run()
    torch.cuda.synchronize()
    return a, o, lse, ws, gate_t, ref_o, ref_lse, run


_PINNED_FORM_CELLS = [
    pytest.param(1, 1, 8, False, True, "decode_grid", id="B1-Sq1-split8-decode-grid-lse"),
    pytest.param(4, 1, 8, True, True, "decode_grid", id="B4-Sq1-split8-decode-grid-gated-lse"),
    pytest.param(1, 2, 16, False, True, "decode_grid", id="B1-Sq2-split16-decode-grid-shared-list"),
    pytest.param(6, 1, 4, False, False, "lean_walk", id="B6-Sq1-split4-lean-walk-no-lse-output"),
    pytest.param(6, 1, 4, True, True, "lean_walk", id="B6-Sq1-split4-lean-walk-gated-lse"),
    pytest.param(4, 4, 7, False, True, "own", id="B4-Sq4-split7-own-forms"),
    pytest.param(1, 1, 4, True, True, "own", id="B1-Sq1-split4-own-forms-gated"),
    pytest.param(1, 1, 17, False, True, "decode_grid", id="B1-Sq1-split17-decode-grid-fp32-association"),
]


@requires_rubin
@pytest.mark.parametrize("B, S_q, split, gate, with_lse, form", _PINNED_FORM_CELLS)
def test_decode_form_combine_takes_the_pinned_form_and_is_bitwise_the_former_entry(B, S_q, split, gate, with_lse, form):
    """The decode form with the combine's FORM pinned per plan (the decode-grid kernel on the small grids from split 8, the lean group
    walk on the overflow grid at split <= 4, the module's own forms otherwise): within the budget vs the oracle (gated exactly where the
    gate rides the combine), and -- over the SAME fp32 partials the adapter's kernel wrote -- BITWISE the FORMER entry (``compile_ptr``
    without a form, the module's own three-way rule) on O and LSE up to 16 splits, the forms' shared arithmetic; at 17 splits the
    decode-grid kernel associates its eight chain partials differently, so O agrees within one output ulp and the LSE within fp32
    rounding (the former entry there is the module's decode form, itself another association).  Two launches bitwise."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    a, o, lse, ws, gate_t, ref_o, ref_lse, run = _pinned_form_cell(B, S_q, split, gate, with_lse, seed=100 + split)
    assert a.combine_form() == _form_value(comb, form) == a._combine_form, (form, a._combine_form)
    H = 24
    lse_check = lse if with_lse else ws[a._split_workspace_layout()[2] // 4 :][: B * H * S_q].view(B, H, S_q)
    max_o, max_lse = _check([(o, lse_check)], ref_o, ref_lse)
    o1, lse1 = o.clone(), lse_check.clone()
    run()
    torch.cuda.synchronize()
    assert torch.equal(o, o1) and torch.equal(lse_check, lse1), "two launches are not bitwise"
    # the former entry over the adapter's own partials (the workspace the kernel just wrote), into fresh O / LSE buffers
    o_off, lse_off, fin_off, _total = a._split_workspace_layout()
    base = int(ws.data_ptr())
    o2 = torch.full_like(o, _sentinel(o.dtype))
    lse2 = torch.full((B, H, S_q), _sentinel(o.dtype), device=o.device, dtype=torch.float32)
    args = [base + o_off, base + lse_off, int(o2.data_ptr()), int(lse2.data_ptr()), (B, H, S_q, D), split, tuple(o2.stride()), tuple(lse2.stride())]
    if gate:
        args += [int(gate_t.data_ptr()), tuple(gate_t.stride())]
    _former_combine_entry(gate)(*args, int(_stream()))
    torch.cuda.synchronize()
    live = ~torch.isinf(ref_lse)  # [B, H, S]
    if split <= 16:
        assert torch.equal(o2, o1) and torch.equal(lse2, lse1), f"{form} at split {split}: the pinned form differs from the former entry on the same partials"
        verdict = "bitwise the former entry"
    else:
        d = (o2.float() - o1.float()).abs()
        ulps = d.cpu() / _ulp_of(torch.maximum(o2.float().abs(), o1.float().abs()), o.dtype)
        d_lse = float((lse2 - lse1)[live].abs().max())
        assert float(ulps.max()) <= 1.0 and d_lse <= 1e-5, f"split {split}: {float(ulps.max()):.3f} ulp / |dLSE| {d_lse:.2e} vs the former entry"
        verdict = f"within {float(ulps.max()):.3f} output ulp / |dLSE| {d_lse:.2e} of the former entry (two fp32 associations)"
    print(f"\npinned combine form {form} B={B} S_q={S_q} split={split} gate={gate} lse={with_lse}: max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}; {verdict}")


@requires_rubin
@pytest.mark.parametrize("B, S_q, split, gate, with_lse, form", _PINNED_FORM_CELLS[:1] + _PINNED_FORM_CELLS[3:4] + _PINNED_FORM_CELLS[5:7])
def test_decode_form_combine_launches_the_pinned_kernel(B, S_q, split, gate, with_lse, form):
    """The kernel the profiler sees for the combine launch is the pinned form's: ``_combine_decode_grid_kernel`` on the small grid at
    split 8, ``_combine_lean_walk_kernel`` on the overflow grid at split 4, the module's own kernels where the rule forces them."""
    a, o, lse, ws, gate_t, ref_o, ref_lse, run = _pinned_form_cell(B, S_q, split, gate, with_lse, seed=200 + split)
    names = _combine_kernels_of(run)
    if names is None:
        pytest.skip("torch.profiler recorded no CUDA event (CUPTI unavailable on this node): the kernel name is unverified here")
    want = _FORM_KERNEL[form]
    assert len(names) == 1 and any(next(iter(names)).startswith(w) for w in want), (form, names)
    others = {k for f, ks in _FORM_KERNEL.items() if f != form for k in ks}
    assert not any(next(iter(names)).startswith(w) for w in others), (form, names)
    print(f"\npinned combine form {form} B={B} S_q={S_q} split={split}: the profiler saw {sorted(names)}")


# ============================================================================ programmatic dependent launch on the sparse core (use_pdl): bitwise the plain arm
_PDL_CELLS = [
    pytest.param(1, 1, 8, False, True, id="B1-Sq1-split8-decode-grid-lse"),
    pytest.param(4, 1, 8, True, True, id="B4-Sq1-split8-decode-grid-gated-lse"),
    pytest.param(6, 1, 4, True, True, id="B6-Sq1-split4-lean-walk-gated-lse"),
    pytest.param(1, 1, 1, True, True, id="B1-Sq1-unsplit-epilogue-gate-lse"),
    pytest.param(4, 2, 1, False, False, id="B4-Sq2-unsplit-shared-list-no-lse"),
]


@requires_rubin
@pytest.mark.parametrize("B, S_q, split, gate, with_lse", _PDL_CELLS)
def test_decode_form_pdl_arm_is_bitwise_the_plain_arm(B, S_q, split, gate, with_lse):
    """The decode form with ``use_pdl=True`` (the core launched with the attribute and its griddepcontrol sites, the combine's pair) is
    BITWISE the plain arm on O and LSE over the same inputs -- split (the decode-grid kernel and the lean group walk behind it) and unsplit
    (the kernel's own epilogue gate; no combine) -- within the budget vs the oracle, two launches bitwise."""
    off = _pinned_form_cell(B, S_q, split, gate, with_lse, seed=300 + split, use_pdl=False)
    on = _pinned_form_cell(B, S_q, split, gate, with_lse, seed=300 + split, use_pdl=True)
    a_off, o_off, lse_off, ws_off, _g, ref_o, ref_lse, _run_off = off
    a_on, o_on, lse_on, ws_on, _g2, _ref_o, _ref_lse, run_on = on
    assert a_on.use_pdl is True and a_off.use_pdl is False and a_on.combine_form() == a_off.combine_form()
    H = 24
    lse_check_off = lse_off if with_lse else ws_off[a_off._split_workspace_layout()[2] // 4 :][: B * H * S_q].view(B, H, S_q) if split > 1 else None
    lse_check_on = lse_on if with_lse else ws_on[a_on._split_workspace_layout()[2] // 4 :][: B * H * S_q].view(B, H, S_q) if split > 1 else None
    if lse_check_off is not None:
        max_o, max_lse = _check([(o_on, lse_check_on)], ref_o, ref_lse)
        assert torch.equal(lse_check_on, lse_check_off), "the LSE differs with use_pdl"
    else:
        max_o = float((o_on.float() - ref_o).abs().max())
        max_lse = 0.0
        assert max_o <= ATOL
    assert torch.equal(o_on, o_off), f"O differs with use_pdl (max|d| {(o_on.float() - o_off.float()).abs().max().item():.3e})"
    o1 = o_on.clone()
    run_on()
    torch.cuda.synchronize()
    assert torch.equal(o_on, o1), "two switched launches are not bitwise"
    print(f"\npdl arm B={B} S_q={S_q} split={split} gate={gate} lse={with_lse}: bitwise the plain arm; max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}")


@requires_rubin
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
def test_prefill_form_pdl_arm_is_bitwise_the_plain_arm(dtype):
    """The prefill form (top-left, per-token lists, the block's own K / V) with ``use_pdl=True``: bitwise the plain arm on O and LSE at
    S = 384 (three tiles: the ring depth) over 24 / 2 heads, the full list; within the budget vs the oracle."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    B, S, H, KH, top_k = 1, 384, 24, 2, 512
    q, k, v, g = _inputs(B, S, H, KH, dtype, 9)
    ids, lens = _lists("full", B, S, top_k, g)
    scale = 1.0 / math.sqrt(D)
    ref_o, ref_lse = _reference(q, k, v, ids, lens, None, scale, top_k)
    outs = {}
    for pdl in (False, True):
        o = torch.full((B, S, H, D), _sentinel(dtype), device=q.device, dtype=dtype)
        lse = torch.full((B, H, S), _sentinel(dtype), device=q.device, dtype=torch.float32)
        a = SparseGqaFwdDslSm107(q=q, k=k, v=v, o=o, lse=lse, block_ids=ids, block_lens=lens, top_k=top_k, scale=scale, use_pdl=pdl)
        assert a.check_support() and a.use_pdl is pdl
        a.compile()
        a.execute(stream=_stream())
        torch.cuda.synchronize()
        outs[pdl] = (o, lse)
    max_o, max_lse = _check([outs[True]], ref_o, ref_lse)
    _assert_bitwise(outs[True], outs[False], f"prefill {dtype} use_pdl vs plain")
    print(f"\nprefill pdl arm {dtype}: bitwise the plain arm; max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f}")
