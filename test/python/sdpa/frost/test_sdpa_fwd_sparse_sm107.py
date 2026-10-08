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
  missed-coordinate signature of a THD port (``LSE == log(n_vis)`` -- an all-zero K operand) ruled out.

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


def _operands(B=1, S=8, H=24, KH=2, SKV=None, dtype="torch.bfloat16", top_k=512):
    SKV = S if SKV is None else SKV
    q = _T((B, S, H, D), (S * H * D, H * D, D, 1), dtype)
    k = _T((B, SKV, KH, D), (SKV * KH * D, KH * D, D, 1), dtype)
    ids = _T((B * S, top_k), (top_k, 1), "torch.int32")
    return dict(q=q, k=k, v=k, o=q, block_ids=ids, top_k=top_k, device_cc=(10, 7))


def _pool_strides(KH, P, hnd):
    """Strides of a ``[num_pages, KH, P, D]`` pool view: HND compact, or NHD storage ``[num_pages, P, KH, D]`` viewed as that shape."""
    return (KH * P * D, P * D, D, 1) if hnd else (P * KH * D, D, KH * D, 1)


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


def test_adapter_accepts_the_record_and_builds_the_params():
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    a = SparseGqaFwdDslSm107(**_operands())
    assert a.check_support() is True
    p = a.template_params()
    assert (p.qsa_block_topk, p.qsa_block_size, p.qh_per_kh, p.pack_gqa, p.cta_mma, p.dtype_qkv) == (512, 4, 12, True, 1, 2)
    assert not p.seq_kv_lens_present and not p.epilogue_gate and not p.thd_varlen and not p.paged_kv and p.page_size == 0


@pytest.mark.parametrize("P", [16, 64, 48])
@pytest.mark.parametrize("hnd", [True, False], ids=["HND", "NHD"])
def test_adapter_accepts_the_paged_form_and_derives_the_pool_geometry(P, hnd):
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
def test_adapter_accepts_the_decode_form_and_builds_the_params(split_kv, paged):
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


@pytest.mark.parametrize(
    "mutate, exc, word",
    [
        (lambda o: o.update(device_cc=(10, 0)), NotImplementedError, "Rubin-line"),
        (lambda o: o.update(device_cc=(8, 0)), NotImplementedError, "Rubin-line"),
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
    ],
)
def test_adapter_declines_every_unserved_request_by_name(mutate, exc, word):
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
def test_adapter_paged_form_checks(mutate, exc, word):
    """Every malformed paged request is refused by name before any kernel import: the page size (the one unserved FORM, a
    NotImplementedError), a missing table or length, a malformed table, and every pool stride the 2-D gather map cannot
    express (the page axis, the KV head count, the token / page / head strides, the head dim, the dtype)."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    o = _paged_operands()
    mutate(o)
    with pytest.raises(exc, match=word):
        SparseGqaFwdDslSm107(**o).check_support()


def test_adapter_refuses_the_table_and_the_page_size_on_a_dense_declaration():
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


def test_adapter_accepts_a_packed_thd_request_and_sizes_its_workspace():
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


def test_adapter_declines_a_padded_kv_batch_stride_at_B2_but_not_at_B1():
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
    for name in ("SPLIT_KV = CFG.SPLIT_KV", "LIST_PER_SEQUENCE = CFG.LIST_PER_SEQUENCE", "BOTTOM_RIGHT = CFG.BOTTOM_RIGHT"):
        assert name in code, name
    assert code.count("cutlass.const_expr(SPLIT_KV > 1)") >= 5, "the split arm folds at the decode, the bounds, both list-read sites and the store"
    assert code.count("cutlass.const_expr(LIST_PER_SEQUENCE == 1)") >= 4 and code.count("cutlass.const_expr(BOTTOM_RIGHT == 1)") == 3
    assert "live_row = tok < eff_seqlen_q" in code, "under the bottom-right diagonal the store predicate is the Q row's liveness"
    # the prefill form keeps its own tail spelling (one tail id at most) so the dense rendering traces the same program as before
    assert (
        "n_tail = _select_i32((n_vis % bs) != zero, 1, 0)" in code
        and "tail = _select_i32((idx == count) & (n_tail != cutlass.Int32(0)), tail_lo, cutlass.Int32(-1))" in code
    )
    assert "split = cute.arch.make_warp_uniform(x // seqlen_q)" in code and "tok = x - split * seqlen_q" in code
    assert "return tok, head, batch, item_row, kv_row_base, eff_seqlen_q, out_tok, out_batch, split" in code
    # the one helper: anchor / tail count / chunk
    i_ib = code.index("def _item_bounds(")
    ib = code[i_ib : code.index("\ndef ", i_ib + 1)]
    for frag in (
        "pos = tok + (eff_seqlen_kv - eff_seqlen_q)",
        "anchor = eff_seqlen_kv - eff_seqlen_q",
        "n_tail = cute.math.max((n_vis + bs - one) // bs - tail_lo, zero)",
        "tile0 = split * per + cute.math.min(split, rem)",
        "dead = dead | (n_sp == zero)",
        "n_tiles = cute.math.max(cutlass.Int32(1), n_sp)",
        "return pos, count, n_tail, tail_lo, dead, n_tiles, eff_seqlen_kv, tile0",
    ):
        assert frag in ib, frag
    assert "tail = _select_i32((rel >= cutlass.Int32(0)) & (rel < n_tail), tail_lo + rel, cutlass.Int32(-1))" in code
    # the list reads offset by the chunk's first tile; the stores at the partial batch
    assert "_gather_block_ids(sIds_raw, slot_base, tile0, w, count, n_tail, tail_lo, dead)" in code
    assert "t_next = tile0 + t_next" in code and "t_abs = tile0 + i" in code
    assert "part_batch = out_batch + split * n_batch" in code and code.count("oo[part_batch, out_tok, head_base + cutlass.Int32(j), :]") == 2
    assert "lse_arr[part_batch, head_base + cutlass.Int32(j), out_tok] = lse_cols[j]" in code
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


def _run_cell(*, B, S, H, KH, dtype, top_k, list_kind, kv_lens=None, block_lens=True, seed=0):
    q, k, v, g = _inputs(B, S, H, KH, dtype, seed)
    ids, lens = _lists(list_kind, B, S, top_k, g, kv_lens)
    scale = 1.0 / math.sqrt(D)
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
def test_adapter_accepts_the_gate_operand_and_selects_the_gated_specialization():
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
    assert sig.rstrip().endswith("tma_gate_desc: cutlass.GridConstant[tmap.TensorMap] = None,"), "the gate descriptor is the LAST kernel parameter"
    # the ungated rendering keeps its drains, the gated one its own (STAGES_GATE + 1 waits on mb_gate_empty, none on mb_q_empty)
    assert "cutlass.range_constexpr(CFG.STAGES_GATE + 1)" in code


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


def _launch_decode(q, kv, ids, lens, kv_lens, top_k, scale, *, split=1, launches=2, gate=None, with_lse=True):
    """Sentinel-filled O / LSE per launch through the adapter's DECODE form (``list_per_sequence=True, bottom_right=True,
    split_kv=split``); ``kv`` = ``(k, v)`` dense or ``(k_pool, v_pool, table, P)`` paged.  A split's workspace is sentinel-filled too:
    an unwritten partial slot would read as a huge live weight and the oracle comparison would catch it."""
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
            list_per_sequence=True,
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
