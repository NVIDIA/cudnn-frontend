# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Rubin index-list sparse d256 forward (``kernels/sm107/sparse_d256_f16.py``) through its adapter
(``sdpa/fwd/sparse_gqa_sm107.py``): the bring-up ladder of the first body, the adapter's typed declines and the source pins.

Host tier (any box): the claims record agrees with the config's wired-arm set (an arm the body does not carry is declined by
BOTH); every unserved request is a typed decline naming the feature; the DSL gate declines by name below the ``sm_107a``
floor without importing the kernel module; the kernel source keeps the sm107 conventions (derived descriptor version, the
module spin constant, the LOAD wait between the S^T read and the ``s_empty`` arrive, the one-helper ``_item_bounds`` at the
item start of every tile-looping role, every ring wait / arrive on a carried ``PipelineState``).

Rubin tier (``requires_rubin``): the ladder -- one tile, 2 / 3 / 4 / 17 tiles, ``B x H > 1`` at one tile, the full list below
the identity bound, a shuffled list, a dead sequence through per-batch KV lengths, ``block_lens`` absent at ``top_k = 256``,
f16 -- each against an fp32 reference built from the oracle's visible-set rule (``qsa_visible_mask``: the listed complete
blocks inside the causal range plus the open tail, ``-1`` below the count = no key), with poisoned outputs (no sentinel may
survive on a live row, a dead row lands ``O = 0`` / ``LSE = -inf`` exactly) and two launches bitwise.  Tolerances are the
dense suite's (atol 2e-2 on O and LSE), none added.
"""

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
ATOL = 2e-2  # the SDPA stage budget of the dense suite -- never widened here


def _kernel_source() -> str:
    import cudnn.sdpa.fwd.config_sm107 as c107

    path = os.path.join(os.path.dirname(os.path.abspath(c107.__file__)), "kernels", "sm107", "sparse_d256_f16.py")
    return open(path).read()


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
        "seq_q_lens": C.padded,
    }
    for arm, claimed in pairs.items():
        assert claimed == (arm in SPARSE_D256_WIRED_ARMS), f"{arm}: record {claimed}, wired {arm in SPARSE_D256_WIRED_ARMS}"
    assert C.causal and C.kv_lens and C.pack_gqa and C.d_shapes == frozenset({(D, D)}) and C.index_block_sizes == frozenset({4})
    assert (C.index_top_k_min, C.index_top_k_max, C.index_gqa_group_max) == (4, 512, 16)
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


def test_adapter_accepts_the_record_and_builds_the_params():
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    a = SparseGqaFwdDslSm107(**_operands())
    assert a.check_support() is True
    p = a.template_params()
    assert (p.qsa_block_topk, p.qsa_block_size, p.qh_per_kh, p.pack_gqa, p.cta_mma, p.dtype_qkv) == (512, 4, 12, True, 1, 2)
    assert not p.seq_kv_lens_present and not p.epilogue_gate and not p.thd_varlen and not p.paged_kv


@pytest.mark.parametrize(
    "mutate, exc, word",
    [
        (lambda o: o.update(device_cc=(10, 0)), NotImplementedError, "Rubin-line"),
        (lambda o: o.update(device_cc=(8, 0)), NotImplementedError, "Rubin-line"),
        (lambda o: o.update(top_k=501), NotImplementedError, "top_k"),
        (lambda o: o.update(top_k=516), NotImplementedError, "top_k"),
        (lambda o: o.update(block_size=8), NotImplementedError, "block_size"),
        (lambda o: o.update(thd=True), NotImplementedError, "thd"),
        (lambda o: o.update(paged_kv=True), NotImplementedError, "paged_kv"),
        (lambda o: o.update(epilogue_gate=object()), NotImplementedError, "epilogue_gate"),
        (lambda o: o.update(split_kv=2), NotImplementedError, "split_kv"),
        (lambda o: o.update(bottom_right=True), NotImplementedError, "bottom_right"),
        (lambda o: o.update(sink=object()), NotImplementedError, "sink"),
        (lambda o: o.update(window_left=128), NotImplementedError, "sliding_window"),
        (lambda o: o.update(seq_q_lens=object()), NotImplementedError, "seq_q_lens"),
        (lambda o: o.update(list_per_sequence=True), NotImplementedError, "list_per_sequence"),
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
    """The sm107 conventions the body must keep: the descriptor version derived (no literal), one module spin constant on
    every ring wait, the LOAD wait between the S^T read and the ``s_empty`` arrive, the one helper at every tile-looping role."""
    src = _kernel_source()
    code = _code(src)
    assert "desc_version=DESC_VERSION" in code and "desc_version=0" not in code and "desc_version=1" not in code
    assert code.count("spin=SPIN_RING_WAITS") >= 8, "every per-tile ring wait takes the module constant"
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


def test_sparse_kernel_declares_no_cross_cta_arrive_and_no_cluster_scope():
    src = _code(_kernel_source())
    assert "arrive_on_peer" not in src and "arrive_on_leader" not in src and "MemScope.CLUSTER" not in src
    assert "cta_group=1" in src and "cta_group=2" not in src


# ============================================================================ Rubin: the ladder
def _stream():
    import cuda.bindings.driver as cuda

    return cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))


def _reference(q, k, v, ids, lens, kv_lens, scale, top_k):
    """fp32 over the half-rounded operands; the oracle's visible set per row; natural-log LSE; a dead row is O = 0, LSE = -inf."""
    oracle = _oracle()
    B, S, H, _ = q.shape
    SKV, KH = k.shape[1], k.shape[2]
    G = H // KH
    dev = q.device
    ref_o = torch.zeros(B, S, H, D, device=dev, dtype=torch.float32)
    ref_lse = torch.full((B, H, S), float("-inf"), device=dev, dtype=torch.float32)
    for b in range(B):
        L = SKV if kv_lens is None else int(kv_lens[b])
        allowed = oracle.qsa_visible_mask(ids[b], None if lens is None else lens[b], torch.arange(S, device=dev), L, SKV, 4, top_k=top_k)
        for h in range(H):
            kk, vv = k[b, :, h // G].float(), v[b, :, h // G].float()
            sc = (q[b, :, h].float() @ kk.t()) * scale
            sc = sc.masked_fill(~allowed, float("-inf"))
            rmax = sc.amax(dim=-1)
            dead = torch.isinf(rmax) & (rmax < 0)
            safe = torch.where(dead, torch.zeros_like(rmax), rmax)
            p = torch.where(allowed, torch.exp(sc - safe[:, None]), torch.zeros_like(sc))
            den = p.sum(dim=-1)
            oc = (p @ vv) / torch.where(dead, torch.ones_like(den), den)[:, None]
            ref_o[b, :, h] = torch.where(dead[:, None], torch.zeros_like(oc), oc)
            ref_lse[b, h] = torch.where(dead, torch.full_like(den, float("-inf")), safe + torch.log(den))
    return ref_o, ref_lse


def _run_cell(*, B, S, H, KH, dtype, top_k, list_kind, kv_lens=None, block_lens=True, seed=0):
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    oracle = _oracle()
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randn(B, S, H, D, device=dev, dtype=torch.float32, generator=g).to(dtype)
    k = torch.randn(B, S, KH, D, device=dev, dtype=torch.float32, generator=g).to(dtype)
    v = torch.randn(B, S, KH, D, device=dev, dtype=torch.float32, generator=g).to(dtype)
    positions = torch.arange(S, device=dev).expand(B, S)
    kvl = None if kv_lens is None else torch.tensor(kv_lens, device=dev, dtype=torch.long)[:, None]
    if list_kind == "full":
        ids, lens = oracle.full_block_ids(positions, top_k, 4, kv_lens=kvl)
    else:
        ids, lens = oracle.random_block_ids(positions, top_k, 4, generator=g, kv_lens=kvl, shuffle=True)
    ids, lens = ids.contiguous(), lens.contiguous()
    seq_kv = None if kv_lens is None else torch.tensor(kv_lens, device=dev, dtype=torch.int32)
    sent = 1.5e30 if dtype == torch.bfloat16 else 6.0e4
    scale = 1.0 / math.sqrt(D)
    outs = []
    for _ in range(2):
        o = torch.full((B, S, H, D), sent, device=dev, dtype=dtype)
        lse = torch.full((B, H, S), sent, device=dev, dtype=torch.float32)
        a = SparseGqaFwdDslSm107(
            q=q, k=k, v=v, o=o, lse=lse, block_ids=ids, block_lens=lens if block_lens else None, seq_kv_lens=seq_kv, top_k=top_k, scale=scale
        )
        assert a.check_support()
        a.execute(stream=_stream())
        torch.cuda.synchronize()
        outs.append((o, lse))
    (o, lse), (o2, lse2) = outs
    ref_o, ref_lse = _reference(q, k, v, ids, lens if block_lens else None, kv_lens, scale, top_k)
    of = o.float()
    dead = torch.isinf(ref_lse)  # [B, H, S]
    assert torch.isfinite(of).all(), "non-finite O"
    assert not (of == sent).any(dim=-1).any(), "a sentinel survived on an output row"
    assert not (lse == sent).any(), "a sentinel survived in LSE"
    if dead.any():
        assert torch.isinf(lse[dead]).all() and (lse[dead] < 0).all(), "a dead row's LSE must be -inf exactly"
        assert (of.permute(0, 2, 1, 3)[dead] == 0).all(), "a dead row's O must be 0 exactly"
    live = ~dead
    assert (of - ref_o).abs().max().item() <= ATOL
    assert (lse[live] - ref_lse[live]).abs().max().item() <= ATOL
    assert torch.equal(o, o2) and torch.equal(lse, lse2), "two launches must be bitwise"
    return float((of - ref_o).abs().max()), float((lse[live] - ref_lse[live]).abs().max())


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
