# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``qkvg_from_vllm`` -- the loader twin for a serving stack's merged ``qkv_proj.weight`` at a tensor-parallel rank. CPU-only, no GPU.

The twin cuts the rank's ``[q_proj | k_proj | v_proj]`` rows and hands them to ``qkvg_from_hf``, so every cell here is a BITWISE pin:
against ``qkvg_from_hf`` on the same HF tensors (the model's own per-head ``[q_h | gate_h]`` split, the ``(1 + w)`` norm form),
against an independent spelling of the row permutation, and against the shard of the TP 1 ``W_qkvg`` -- at TP 1 and at the rank
slices vLLM's ``QKVParallelLinear`` loads (the q shard by rank, the k / v shard by ``rank // num_kv_head_replicas``) of the 32 / 2 and
24 / 2 geometries; the indexer band in both of its forms; the refusals, each naming the fix.
"""

import functools
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pytest
import torch

from cudnn.gated_attention_block import (
    GatedAttentionBlockGeometry,
    ProjBlock,
    QsaSpec,
    build_fused_qkvg_weight,
    qkvg_from_hf,
    qkvg_from_vllm,
)

pytestmark = pytest.mark.L0

# The two serving geometries at TP 1 (d_head 256, RoPE on the first 64 dims): 32 query heads over 2 KV heads at d_model 4096, and
# 24 over 2 at d_model 2560.  Their rank slices below follow the serving stack's shard rule.
GEOM_32_2 = GatedAttentionBlockGeometry(d_model=4096, h_q=32, h_kv=2, d_head=256, rope_dim=64)
GEOM_24_2 = GatedAttentionBlockGeometry(d_model=2560, h_q=24, h_kv=2, d_head=256, rope_dim=64)
GEOMS = {"32_2": GEOM_32_2, "24_2": GEOM_24_2}
# A small geometry for the arithmetic cells, and its indexer-band twin (a 192-column band keeps the 64-column tile alignment).
GEOM_SMALL = GatedAttentionBlockGeometry(d_model=256, h_q=4, h_kv=2, d_head=64, rope_dim=16)
GEOM_SMALL_BAND = GatedAttentionBlockGeometry(
    d_model=256, h_q=4, h_kv=2, d_head=64, rope_dim=16, qsa=QsaSpec(index_band=True, index_heads=2, index_head_dim=64)
)

# The stage-(1) width of every rank slice (TP, h_q / h_kv -> n_qkvg): (2 h_q + 2 h_kv) x 256.
RANK_WIDTHS = {("32_2", 1): 17408, ("32_2", 2): 8704, ("32_2", 4): 4608, ("32_2", 8): 2560, ("24_2", 1): 13312, ("24_2", 2): 6656, ("24_2", 4): 3584}


class HfLayer(NamedTuple):
    """One HF attention layer's weights at TP 1: the double-width ``q_proj`` (per-head ``[q_h | gate_h]``), ``k_proj``, ``v_proj``, the
    two zero-centered QK-norm weights."""

    q_proj: torch.Tensor
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor


@functools.lru_cache(maxsize=None)
def _hf_layer(name: str, seed: int = 0, dtype: torch.dtype = torch.bfloat16) -> HfLayer:
    """ONE CPU generator per geometry; the projections drawn in fp32 and rounded once into the checkpoint dtype (bf16)."""
    g = GEOMS[name]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    d, hq, hkv, dm = g.d_head, g.h_q, g.h_kv, g.d_model

    def randn(*shape, std):
        return (torch.randn(*shape, generator=gen, dtype=torch.float32) * std).to(dtype)

    return HfLayer(
        q_proj=randn(2 * hq * d, dm, std=0.02),
        k_proj=randn(hkv * d, dm, std=0.02),
        v_proj=randn(hkv * d, dm, std=0.02),
        q_norm=torch.randn(d, generator=gen) * 0.1,
        k_norm=torch.randn(d, generator=gen) * 0.1,
    )


def _merged(q, k, v):
    """The serving stack's merged ``qkv_proj.weight``: ``[q_proj | k_proj | v_proj]`` along the rows, the per-head gate interleave kept."""
    return torch.cat([q, k, v], dim=0).contiguous()


def _rank_slices(hf: HfLayer, g_full: GatedAttentionBlockGeometry, tp: int, rank: int):
    """The HF rows a serving stack's QKV linear gives rank ``rank`` of ``tp``, and the rank's geometry.

    The q shard of rank ``r`` is rows ``[r * 2 h_q_r D, (r + 1) * 2 h_q_r D)`` of ``q_proj`` -- the per-head rows of query heads
    ``[r h_q_r, (r + 1) h_q_r)`` -- and the k / v shard is KV head group ``r // num_kv_head_replicas``: one KV head per rank, replicated
    ``tp / H_kv`` times, when ``tp >= H_kv``; ``H_kv / tp`` heads per rank otherwise.
    """
    d = g_full.d_head
    assert g_full.h_q % tp == 0
    hq_r = g_full.h_q // tp
    if tp >= g_full.h_kv:
        assert tp % g_full.h_kv == 0
        hkv_r, replicas = 1, tp // g_full.h_kv
    else:
        assert g_full.h_kv % tp == 0
        hkv_r, replicas = g_full.h_kv // tp, 1
    kv_rank = rank // replicas
    q = hf.q_proj[rank * 2 * hq_r * d : (rank + 1) * 2 * hq_r * d]
    k = hf.k_proj[kv_rank * hkv_r * d : (kv_rank + 1) * hkv_r * d]
    v = hf.v_proj[kv_rank * hkv_r * d : (kv_rank + 1) * hkv_r * d]
    g_r = GatedAttentionBlockGeometry(d_model=g_full.d_model, h_q=hq_r, h_kv=hkv_r, d_head=d, rope_dim=g_full.rope_dim)
    return q, k, v, g_r, kv_rank


def _permute_rows(w_merged: torch.Tensor, g: GatedAttentionBlockGeometry) -> torch.Tensor:
    """An independent spelling of the permutation: split ``[2 q | kv | kv]``, view the Q block per head, cut ``[:D]`` / ``[D:]``,
    concatenate ``Q | GATE | K | V``."""
    d, hq, hkv, dm = g.d_head, g.h_q, g.h_kv, g.d_model
    q_gate, w_k, w_v = torch.split(w_merged, [2 * hq * d, hkv * d, hkv * d], dim=0)
    per_head = q_gate.reshape(hq, 2 * d, dm)
    return torch.cat([per_head[:, :d].reshape(hq * d, dm), per_head[:, d:].reshape(hq * d, dm), w_k, w_v], dim=0).contiguous()


def _shard_of_the_tp1_weight(w_qkvg_tp1: torch.Tensor, g_full: GatedAttentionBlockGeometry, g_r: GatedAttentionBlockGeometry, rank: int, kv_rank: int):
    """The rank's ``W_qkvg`` read out of the TP 1 one: the rank's query heads from the Q and GATE bands, its KV head group from K and V."""
    d = g_full.d_head
    o_q, o_g, o_k, o_v = g_full.qkvg_offsets
    q_rows = slice(rank * g_r.h_q * d, (rank + 1) * g_r.h_q * d)
    kv_rows = slice(kv_rank * g_r.h_kv * d, (kv_rank + 1) * g_r.h_kv * d)
    return torch.cat(
        [
            w_qkvg_tp1[o_q:o_g][q_rows],
            w_qkvg_tp1[o_g:o_k][q_rows],
            w_qkvg_tp1[o_k:o_v][kv_rows],
            w_qkvg_tp1[o_v:][kv_rows],
        ],
        dim=0,
    )


def _assert_three_equal(got, want):
    for a, b, what in zip(got, want, ("w_qkvg", "w_q_norm", "w_k_norm")):
        assert (a is None) == (b is None), what
        if a is not None:
            assert a.dtype == b.dtype and a.shape == b.shape and a.is_contiguous(), what
            assert torch.equal(a, b), what


# ---------------------------------------------------------------------------
# TP 1 and the rank slices: bitwise qkvg_from_hf, the independent permutation, the shard of the TP 1 weight
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(GEOMS))
def test_qkvg_from_vllm_is_bitwise_qkvg_from_hf_at_tp1(name):
    g = GEOMS[name]
    hf = _hf_layer(name)
    merged = _merged(hf.q_proj, hf.k_proj, hf.v_proj)
    assert merged.shape == (g.n_qkvg, g.d_model) == (RANK_WIDTHS[name, 1], g.d_model)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the layout is fixed: no FutureWarning about an omitted q_gate_layout, ever
        got = qkvg_from_vllm(merged, hf.q_norm, hf.k_norm, g)
    want = qkvg_from_hf(hf.q_proj, hf.k_proj, hf.v_proj, hf.q_norm, hf.k_norm, g)
    _assert_three_equal(got, want)
    w_qkvg, w_q_norm, w_k_norm = got
    assert w_qkvg.shape == merged.shape and w_qkvg.dtype == merged.dtype == torch.bfloat16 and w_qkvg.is_contiguous()
    assert torch.equal(w_qkvg, _permute_rows(merged, g))
    # A permutation of the input: K and V rows untouched, the Q / GATE bands a reordering of the q_proj rows, nothing else.
    o_q, o_g, o_k, o_v = g.qkvg_offsets
    assert torch.equal(w_qkvg[o_k:], merged[2 * g.h_q * g.d_head :])
    assert not torch.equal(w_qkvg[:o_k], merged[: 2 * g.h_q * g.d_head])  # the per-head interleave really moves rows
    assert torch.equal(w_qkvg[:o_k].view(2, g.h_q, g.d_head, g.d_model).transpose(0, 1).reshape(-1, g.d_model), merged[: 2 * g.h_q * g.d_head])
    # The norm weights: (1 + w) formed in fp32, rounded once into the activation dtype.
    assert torch.equal(w_q_norm, (1.0 + hf.q_norm.float()).to(torch.bfloat16)) and torch.equal(w_k_norm, (1.0 + hf.k_norm.float()).to(torch.bfloat16))


@pytest.mark.parametrize("name, tp", [("32_2", 2), ("32_2", 4), ("32_2", 8), ("24_2", 2), ("24_2", 4)])
def test_qkvg_from_vllm_at_the_rank_slices(name, tp):
    """Every rank of ``tp``: the twin on the rank's merged weight equals ``qkvg_from_hf`` on the rank's HF slices bitwise, equals the
    independent permutation, and equals the rank's shard of the TP 1 ``W_qkvg``; the norm weights are the same on every rank."""
    g = GEOMS[name]
    hf = _hf_layer(name)
    w_tp1, w_q_norm_tp1, w_k_norm_tp1 = qkvg_from_hf(hf.q_proj, hf.k_proj, hf.v_proj, hf.q_norm, hf.k_norm, g)
    for rank in range(tp):
        q, k, v, g_r, kv_rank = _rank_slices(hf, g, tp, rank)
        g_r.validate()
        assert g_r.n_qkvg == RANK_WIDTHS[name, tp] and g_r.n_qkvg % 256 == 0  # the forced 256-wide projection tile holds at every shard
        merged = _merged(q, k, v)
        got = qkvg_from_vllm(merged, hf.q_norm, hf.k_norm, g_r)
        _assert_three_equal(got, qkvg_from_hf(q, k, v, hf.q_norm, hf.k_norm, g_r))
        w_qkvg = got[0]
        assert w_qkvg.shape == (g_r.n_qkvg, g.d_model) and w_qkvg.numel() == merged.numel()
        assert torch.equal(w_qkvg, _permute_rows(merged, g_r))
        assert torch.equal(w_qkvg, _shard_of_the_tp1_weight(w_tp1, g, g_r, rank, kv_rank))
        assert torch.equal(got[1], w_q_norm_tp1) and torch.equal(got[2], w_k_norm_tp1)


def test_qkvg_from_vllm_projects_the_values_the_serving_stack_projects():
    """fp64, CPU: ``h @ W^T`` through the merged weight, split as the serving layer splits it (``[2 q | kv | kv]``, the Q block viewed per
    head and chunked into q / gate), equals ``h @ W_qkvg^T`` cut at the block's offsets, band by band."""
    torch.manual_seed(0)
    g = GEOM_SMALL
    d, hq, hkv, dm = g.d_head, g.h_q, g.h_kv, g.d_model
    q, k, v = torch.randn(2 * hq * d, dm), torch.randn(hkv * d, dm), torch.randn(hkv * d, dm)
    merged = _merged(q, k, v)
    w_qkvg, _, _ = qkvg_from_vllm(merged, torch.zeros(d), torch.zeros(d), g, act_dtype=torch.float32)
    t = 6
    h = torch.randn(t, dm).double()
    out_merged = h @ merged.double().t()
    q_gate, k_out, v_out = torch.split(out_merged, [2 * hq * d, hkv * d, hkv * d], dim=-1)
    q_out, gate_out = torch.chunk(q_gate.reshape(t, hq, 2 * d), 2, dim=-1)
    out_block = h @ w_qkvg.double().t()
    o_q, o_g, o_k, o_v = g.qkvg_offsets
    tol = dict(rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(out_block[:, o_q:o_g], q_out.reshape(t, hq * d), **tol)
    torch.testing.assert_close(out_block[:, o_g:o_k], gate_out.reshape(t, hq * d), **tol)
    torch.testing.assert_close(out_block[:, o_k:o_v], k_out, **tol)
    torch.testing.assert_close(out_block[:, o_v:], v_out, **tol)


def test_qkvg_from_vllm_takes_a_non_contiguous_merged_weight():
    """A column slice of a wider buffer is a legitimate load-time input (the row cuts and the per-head view only split the row axis)
    and gives the contiguous result bitwise."""
    torch.manual_seed(0)
    g = GEOM_SMALL
    d, hq, hkv, dm = g.d_head, g.h_q, g.h_kv, g.d_model
    merged = _merged(torch.randn(2 * hq * d, dm), torch.randn(hkv * d, dm), torch.randn(hkv * d, dm))
    sliced = torch.cat([merged, merged], dim=1)[:, :dm]
    assert not sliced.is_contiguous() and torch.equal(sliced, merged)
    w_n = torch.zeros(d)
    _assert_three_equal(qkvg_from_vllm(sliced, w_n, w_n, g, act_dtype=torch.float32), qkvg_from_vllm(merged, w_n, w_n, g, act_dtype=torch.float32))


# ---------------------------------------------------------------------------
# The norm form, the indexer band, the refusals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _GeometryWithNormOffset(GatedAttentionBlockGeometry):
    """The shape of the in-kernel offset form (``norm_weight_offset`` appended LAST); the loaders read it with ``getattr(..., 0.0)``."""

    norm_weight_offset: float = 1.0


@pytest.mark.parametrize("offset", [0.0, 1.0])
def test_qkvg_from_vllm_norm_form_is_derived_from_the_geometry(offset):
    """The norm weights take exactly the form ``qkvg_from_hf`` hands over: ``(1 + w)`` at offset 0.0 (no field, or 0.0), ``w`` as is at
    1.0 -- in fp32 bitwise, and rounded once into bf16."""
    g = GEOM_SMALL if offset == 0.0 else _GeometryWithNormOffset(d_model=256, h_q=4, h_kv=2, d_head=64, rope_dim=16)
    assert getattr(g, "norm_weight_offset", 0.0) == offset
    torch.manual_seed(0)
    d, hq, hkv, dm = g.d_head, g.h_q, g.h_kv, g.d_model
    merged = _merged(torch.randn(2 * hq * d, dm), torch.randn(hkv * d, dm), torch.randn(hkv * d, dm))
    w = torch.randn(d) * 0.05
    for act_dtype in (torch.float32, torch.bfloat16):
        _, w_q_norm, w_k_norm = qkvg_from_vllm(merged, w, w, g, act_dtype=act_dtype)
        expected = ((1.0 + w) if offset == 0.0 else w).to(act_dtype)
        assert torch.equal(w_q_norm, expected) and torch.equal(w_k_norm, expected)
        _, hf_q, hf_k = qkvg_from_hf(
            merged[: 2 * hq * d], merged[2 * hq * d : 2 * hq * d + hkv * d], merged[2 * hq * d + hkv * d :], w, w, g, act_dtype=act_dtype
        )
        assert torch.equal(w_q_norm, hf_q) and torch.equal(w_k_norm, hf_k)
    with pytest.raises(ValueError, match="norm_weight_offset"):
        qkvg_from_vllm(merged, w, w, _GeometryWithNormOffset(d_model=256, h_q=4, h_kv=2, d_head=64, rope_dim=16, norm_weight_offset=0.5))


def test_qkvg_from_vllm_carries_the_indexer_band_in_both_forms():
    """Under ``index_band`` the indexer projection arrives as ``index_qk_proj_weight`` (the HF loader's arm) or appended to the merged
    weight's rows (a stack that packs the indexer into the QKV GEMM); the two are bitwise, both are bitwise ``qkvg_from_hf``; giving
    both, neither, or the appended rows on a geometry without the band is refused with a message naming the fix."""
    torch.manual_seed(0)
    g = GEOM_SMALL_BAND
    d, hq, hkv, dm = g.d_head, g.h_q, g.h_kv, g.d_model
    q, k, v = torch.randn(2 * hq * d, dm), torch.randn(hkv * d, dm), torch.randn(hkv * d, dm)
    w_i = torch.randn(g.qsa.index_band_cols, dm)
    w_n = torch.randn(d) * 0.01
    merged = _merged(q, k, v)
    packed = torch.cat([merged, w_i], dim=0)
    assert w_i.shape[0] == 192 and packed.shape[0] == g.n_qkvg
    want = qkvg_from_hf(q, k, v, w_n, w_n, g, index_qk_proj_weight=w_i, act_dtype=torch.float32)
    separate = qkvg_from_vllm(merged, w_n, w_n, g, act_dtype=torch.float32, index_qk_proj_weight=w_i)
    appended = qkvg_from_vllm(packed, w_n, w_n, g, act_dtype=torch.float32)
    _assert_three_equal(separate, want)
    _assert_three_equal(appended, want)
    assert separate[0].shape == (g.n_qkvg, dm) and torch.equal(separate[0][g.qkvg_offsets[ProjBlock.INDEX] :], w_i)
    assert torch.equal(separate[0], build_fused_qkvg_weight(q, k, v, g, q_gate_layout="per_head", index_qk_proj_weight=w_i))
    with pytest.raises(ValueError, match="given twice"):
        qkvg_from_vllm(packed, w_n, w_n, g, act_dtype=torch.float32, index_qk_proj_weight=w_i)
    with pytest.raises(ValueError, match="index_qk_proj_weight is required"):
        qkvg_from_vllm(merged, w_n, w_n, g, act_dtype=torch.float32)
    with pytest.raises(ValueError, match=r"must be \[192, 256\]"):  # the assembler's band checks propagate
        qkvg_from_vllm(merged, w_n, w_n, g, act_dtype=torch.float32, index_qk_proj_weight=torch.randn(640, dm))
    with pytest.raises(ValueError, match="declares no indexer band"):
        qkvg_from_vllm(merged, w_n, w_n, GEOM_SMALL, act_dtype=torch.float32, index_qk_proj_weight=w_i)
    with pytest.raises(ValueError, match=rf"must have {merged.shape[0]} rows .*; got {packed.shape[0]}"):
        qkvg_from_vllm(packed, w_n, w_n, GEOM_SMALL, act_dtype=torch.float32)  # no band declared: the appended rows are a shape error
    # The row-count message under a band geometry names BOTH admissible forms.
    with pytest.raises(ValueError, match=rf"must have {merged.shape[0]} rows .*or {packed.shape[0]} with the indexer band's 192 rows appended"):
        qkvg_from_vllm(merged[:-1], w_n, w_n, g, act_dtype=torch.float32)


def test_qkvg_from_vllm_rejects():
    g = GEOM_SMALL
    d, hq, hkv, dm = g.d_head, g.h_q, g.h_kv, g.d_model
    merged = torch.zeros(2 * hq * d + 2 * hkv * d, dm)
    w_n = torch.zeros(d)
    with pytest.raises(
        ValueError,
        match=rf"must have {merged.shape[0]} rows = \[{2 * hq * d} q_proj rows \(per-head \[q_h \| gate_h\]\) \| {hkv * d} k_proj \| {hkv * d} v_proj\]",
    ):
        qkvg_from_vllm(merged[:-d], w_n, w_n, g)
    with pytest.raises(ValueError, match=rf"must have {dm} columns"):
        qkvg_from_vllm(torch.zeros(merged.shape[0], dm + 1), w_n, w_n, g)
    with pytest.raises(ValueError, match="2-D"):
        qkvg_from_vllm(merged.flatten(), w_n, w_n, g)
    with pytest.raises(ValueError, match="2-D"):
        qkvg_from_vllm(merged.unsqueeze(0), w_n, w_n, g)
    with pytest.raises(ValueError, match="2-D"):
        qkvg_from_vllm([merged], w_n, w_n, g)
    # The HF loader's own refusals propagate unchanged.
    with pytest.raises(ValueError, match=r"q_norm_weight must be \[d_head=64\]"):
        qkvg_from_vllm(merged, torch.zeros(d + 1), w_n, g)
    with pytest.raises(ValueError, match="k_norm_weight is required"):
        qkvg_from_vllm(merged, w_n, None, g)
    with pytest.raises(ValueError, match="act_dtype"):
        qkvg_from_vllm(merged, w_n, w_n, g, act_dtype=torch.int32)
    # qk_norm=False: no norm weights in, None out; a tensor in a norm slot is refused.
    g_nonorm = GatedAttentionBlockGeometry(d_model=256, h_q=4, h_kv=2, d_head=64, rope_dim=16, qk_norm=False)
    w_qkvg, wq, wk = qkvg_from_vllm(merged, None, None, g_nonorm)
    assert wq is None and wk is None and w_qkvg.shape == (g_nonorm.n_qkvg, dm) and w_qkvg.dtype == torch.bfloat16
    with pytest.raises(ValueError, match="qk_norm=False"):
        qkvg_from_vllm(merged, w_n, None, g_nonorm)
    # A geometry the block cannot express is refused before any row is read.
    with pytest.raises(ValueError, match="divisible"):
        qkvg_from_vllm(merged, w_n, w_n, GatedAttentionBlockGeometry(d_model=256, h_q=3, h_kv=2, d_head=64, rope_dim=16))


# ---------------------------------------------------------------------------
# e4m3 codes: the permutation on the BYTES (a static-scale FP8 W8A8 checkpoint's merged weight), the per-tensor scale untouched
# ---------------------------------------------------------------------------


def _e4m3_layer(name: str):
    """The HF layer's three projections as e4m3 CODES under ONE per-tensor scale (a serving stack's FP8 W8A8 checkpoint: the merged
    qkv weight quantized at its amax), plus that scale."""
    hf = _hf_layer(name)
    merged = _merged(hf.q_proj, hf.k_proj, hf.v_proj).float()
    scale = float(448.0 / merged.abs().amax())
    codes = (merged * scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    rows_q, rows_kv = hf.q_proj.shape[0], hf.k_proj.shape[0]
    return hf, codes, codes[:rows_q], codes[rows_q : rows_q + rows_kv], codes[rows_q + rows_kv :], scale


@pytest.mark.parametrize("name", sorted(GEOMS))
def test_qkvg_from_vllm_permutes_e4m3_codes_as_bytes_and_keeps_the_per_tensor_scale(name):
    """A merged ``qkv_proj.weight`` of e4m3 CODES (one per-tensor scale): ``qkvg_from_vllm`` / ``qkvg_from_hf`` return e4m3 codes, contiguous --
    BITWISE the independent permutation on the uint8 view, bitwise each other, and ``dequant(perm(codes)) == perm(dequant(codes))`` (the
    permutation and the per-tensor dequant commute, so the stack's scale travels unchanged as ``QuantSpec.descale_w_qkvg``); the norm weights
    come back in ``act_dtype`` exactly as for 16-bit values; a non-contiguous merged weight and the shard slices permute the same bytes."""
    g = GEOMS[name]
    hf, codes, q8, k8, v8, scale = _e4m3_layer(name)
    got = qkvg_from_vllm(codes, hf.q_norm, hf.k_norm, g)
    want = qkvg_from_hf(q8, k8, v8, hf.q_norm, hf.k_norm, g)
    _assert_three_equal(got, want)
    w8, w_q_norm, w_k_norm = got
    assert w8.dtype == torch.float8_e4m3fn and w8.shape == codes.shape and w8.is_contiguous() and w_q_norm.dtype == torch.bfloat16
    assert torch.equal(w8.view(torch.uint8), _permute_rows(codes.view(torch.uint8), g)), "the codes' bytes are not the permutation of the input's bytes"
    assert torch.equal(w8.float() * (1.0 / scale), _permute_rows(codes.float() * (1.0 / scale), g)), "dequant and the permutation do not commute"
    assert torch.equal(w_q_norm, (1.0 + hf.q_norm.float()).to(torch.bfloat16)) and torch.equal(w_k_norm, (1.0 + hf.k_norm.float()).to(torch.bfloat16))
    # the bf16 VALUES' permutation is the dequantized codes' permutation, value for value (what a stack that dequantizes first would see)
    w_vals, _, _ = qkvg_from_vllm((codes.float() * (1.0 / scale)).to(torch.bfloat16), hf.q_norm, hf.k_norm, g)
    assert torch.equal(w_vals, (w8.float() * (1.0 / scale)).to(torch.bfloat16))
    # a non-contiguous merged weight -- a COLUMN slice of a wider tensor (the row stride exceeds d_model; a leading-row slice would stay
    # contiguous) -- permutes the same bytes
    wide = torch.empty(codes.shape[0], g.d_model + 16, dtype=torch.float8_e4m3fn)[:, : g.d_model]
    wide.copy_(codes)
    assert not wide.is_contiguous() and torch.equal(wide.view(torch.uint8), codes.view(torch.uint8))
    w8_nc, _, _ = qkvg_from_vllm(wide, hf.q_norm, hf.k_norm, g)
    assert torch.equal(w8_nc.view(torch.uint8), w8.view(torch.uint8))
    # the rank slices (TP 2): the shard of the TP 1 codes, bitwise
    q, k, v, g_r, kv_rank = _rank_slices(HfLayer(q8, k8, v8, hf.q_norm, hf.k_norm), g, 2, 1)
    w8_r, _, _ = qkvg_from_vllm(_merged(q, k, v), hf.q_norm, hf.k_norm, g_r)
    assert torch.equal(w8_r.view(torch.uint8), _shard_of_the_tp1_weight(w8, g, g_r, 1, kv_rank).view(torch.uint8))


def test_qkvg_from_vllm_refuses_mixed_e4m3_and_16_bit_projections_and_keeps_the_band_arm():
    """e4m3 codes beside 16-bit values are refused by name (one per-tensor scale covers every projection of an FP8 checkpoint); the indexer
    band's fifth band permutes as bytes too (appended rows or the separate argument), bitwise."""
    hf, codes, q8, k8, v8, _scale = _e4m3_layer("24_2")
    g = GEOMS["24_2"]
    with pytest.raises(ValueError, match="q_proj_weight, v_proj_weight are e4m3 codes but k_proj_weight are not"):
        qkvg_from_hf(q8, hf.k_proj, v8, hf.q_norm, hf.k_norm, g)
    with pytest.raises(ValueError, match="are e4m3 codes but q_proj_weight, v_proj_weight are not"):
        qkvg_from_hf(hf.q_proj, k8, hf.v_proj, hf.q_norm, hf.k_norm, g)
    gb = GEOM_SMALL_BAND
    d, hq, hkv, dm = gb.d_head, gb.h_q, gb.h_kv, gb.d_model
    gen = torch.Generator(device="cpu").manual_seed(3)
    merged = (torch.randn(2 * hq * d + 2 * hkv * d, dm, generator=gen) * 0.02).to(torch.float8_e4m3fn)
    band = (torch.randn(gb.qsa.index_band_cols, dm, generator=gen) * 0.02).to(torch.float8_e4m3fn)
    w_n = torch.zeros(d)
    sep, _, _ = qkvg_from_vllm(merged, w_n, w_n, gb, index_qk_proj_weight=band)
    packed, _, _ = qkvg_from_vllm(torch.cat([merged, band], 0), w_n, w_n, gb)
    assert sep.dtype == torch.float8_e4m3fn and sep.shape == (gb.n_qkvg, dm) and torch.equal(sep.view(torch.uint8), packed.view(torch.uint8))
    assert torch.equal(sep[gb.qkvg_offsets[4] :].view(torch.uint8), band.view(torch.uint8)), "the band rides below V as the checkpoint stores it"
    with pytest.raises(ValueError, match="index_qk_proj_weight are e4m3 codes but q_proj_weight, k_proj_weight, v_proj_weight are not"):
        qkvg_from_vllm(merged.float().to(torch.bfloat16), w_n, w_n, gb, index_qk_proj_weight=band)


# ---------------------------------------------------------------------------
# The export: the package's name, the lazy `cudnn.gated_attention_block` attribute, the API index row
# ---------------------------------------------------------------------------


def test_qkvg_from_vllm_is_exported_lazily_and_indexed():
    """``import cudnn`` leaves the block's package unimported; ``cudnn.gated_attention_block.qkvg_from_vllm`` then resolves it on first
    attribute access (a fresh interpreter, so this process's imports cannot mask the order); the name is in the package's ``__all__`` and
    in the API index."""
    import cudnn.gated_attention_block as pkg

    assert "qkvg_from_vllm" in pkg.__all__ and pkg.qkvg_from_vllm is qkvg_from_vllm
    probe = (
        "import sys, cudnn\n"
        "assert 'cudnn.gated_attention_block' not in sys.modules, 'the block package must not be imported eagerly'\n"
        "f = cudnn.gated_attention_block.qkvg_from_vllm\n"
        "import cudnn.gated_attention_block as pkg\n"
        "assert f is pkg.qkvg_from_vllm and 'cudnn.gated_attention_block' in sys.modules\n"
        "from cudnn.gated_attention_block import qkvg_from_vllm\n"
        "assert qkvg_from_vllm is f\n"
        "print('lazy export ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and "lazy export ok" in out.stdout, out.stdout + out.stderr
    index = Path(__file__).resolve().parents[2] / "api_index" / "api_index.txt"
    rows = index.read_text(encoding="utf-8").splitlines()
    assert "cudnn.gated_attention_block.qkvg_from_vllm" in rows and "cudnn.gated_attention_block.qkvg_from_hf" in rows
    assert rows == sorted(rows), "api_index.txt is a sorted list of names"
