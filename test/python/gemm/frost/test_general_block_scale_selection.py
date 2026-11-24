# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-default plans of the general block-scale templates (``planning``'s general block-scale rules).

Without a GPU: the picks for representative dense and grouped shapes, with an explicit device (68 SMs and its cluster
occupancy table) and a probe that accepts every plan, through ``plan_config`` / ``plan_configs`` of the sm100 tree. On an
SM100 GPU: the real probe accepts every pick and shortlist entry, a graph with an FP8 block-128 output plans, builds and runs
through the public API at a row count the previous default geometry was refused at, an override-shape dense graph runs other
M in the workspace it declares, and ``recommend`` offers the shortlist."""

from __future__ import annotations

import cudnn
import cudnn.gemm.frost  # noqa: F401  (installs recorder)
import pytest

from gemm_test_utils import requires_sm100, requires_sm100_tree
from cudnn.gemm.frost import planning
from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.gemm.frost.sm100 import compiler as C100

pytestmark = pytest.mark.L0

DT = cudnn.data_type
_BLOCK = 128
# cuOccupancyMaxActiveClusters of the 68-SM part the rules were measured on, cluster sizes 1..16: clusters of 4 and 8
# CTAs keep 64 of its 68 SMs.
_OCC68 = (68, 34, 21, 16, 12, 9, 8, 8, 5, 4, 4, 4, 4, 4, 4, 4)
_DEVICE = planning.DeviceProperties(100, "sm100 test part", 68, 0, occupancy=_OCC68)


def _cfg(m, n, cluster="1x1", cta=1, *, slices=1, swap=False):
    name = f"CONFIG_sm100_{m}x{n}x128_{min(m, 128)}x{n}x32_cluster{cluster}_{cta}ctamma"
    return name + (f"_splitK{slices}" if slices > 1 else "") + ("_swapAB" if swap else "")


def _name(config):
    return config.name + ("_swapAB" if config.swap_ab and not config.name.endswith("_swapAB") else "")


def _moe(E, S, N, K, mode="none", *, top_k=8, gated=False, fp8=False, override=False, t=None):
    """A grouped GEMM with 1x128 token and 128x128 weight fp32 scales; GATHER reads T = S / top_k token rows. ``t``: a dict
    the graph's tensors are recorded in."""
    t = {} if t is None else t
    g = cudnn.pygraph(
        io_data_type=DT.BFLOAT16, intermediate_data_type=DT.FLOAT, compute_data_type=DT.FLOAT, **({"is_override_shape_enabled": True} if override else {})
    )
    rows = S // top_k if mode == "gather" else S
    kb, nb = K // _BLOCK, N // _BLOCK
    tok = t["tok"] = g.tensor(name="token", dim=[1, rows, K], stride=[rows * K, K, 1], data_type=DT.FP8_E4M3)
    sfa = t["sfa"] = g.tensor(name="SFA", dim=[1, rows, kb], stride=[rows * kb, kb, 1], data_type=DT.FLOAT)
    fto = t["fto"] = g.tensor(name="first_token_offset", dim=[E + 1, 1, 1], stride=[1, 1, 1], data_type=DT.INT32)
    tok_d = g.block_scale_dequantize(input=tok, descale=sfa, block_size=[1, _BLOCK])
    routing = {}
    if mode == "gather":
        routing["token_index"] = g.tensor(name="token_index", dim=[1, S, 1], stride=[S, 1, 1], data_type=DT.INT32)
    elif mode == "combine":
        routing = {
            "token_index": g.tensor(name="token_index", dim=[1, S, 1], stride=[S, 1, 1], data_type=DT.INT32),
            "token_ks": g.tensor(name="token_ks", dim=[1, S, 1], stride=[S, 1, 1], data_type=DT.INT32),
            "top_k": top_k,
            "top_k_scores": g.tensor(name="scores", dim=[1, S // top_k, top_k], stride=[S, top_k, 1], data_type=DT.FLOAT),
        }
    outs = []
    for i in range(2 if gated else 1):
        w = t[f"w{i}"] = g.tensor(name=f"w{i}", dim=[E, K, N], stride=[K * N, 1, K], data_type=DT.FP8_E4M3)
        sfb = t[f"sfb{i}"] = g.tensor(name=f"SFB{i}", dim=[E, kb, nb], stride=[kb * nb, 1, kb], data_type=DT.FLOAT)
        w_d = g.block_scale_dequantize(input=w, descale=sfb, block_size=[_BLOCK, _BLOCK])
        mm_mode = getattr(cudnn.moe_grouped_matmul_mode, mode.upper())
        outs.append(g.moe_grouped_matmul(tok_d, w_d, fto, mode=mm_mode, compute_data_type=DT.FLOAT, name=f"moe{i}", **routing))
    y = g.mul(a=g.swish(input=outs[0], name="silu"), b=outs[1], name="mul") if gated else outs[0]
    _finish(g, y, fp8, DT.FLOAT if mode == "combine" else DT.BFLOAT16, t)
    return g


def _dense(M, N, K, *, fp8=False, dynamic=False, override=False, batch=1, gated=False, n_major=False, t=None):
    """A dense GEMM (or, ``gated``, SwiGLU of two GEMMs sharing the token operand) with 1x128 token and 128x128 weight fp32
    scales; ``n_major``: an N-major weight."""
    t = {} if t is None else t
    kw = {"is_dynamic_shape_enabled": True} if dynamic else {"is_override_shape_enabled": True} if override else {}
    g = cudnn.pygraph(io_data_type=DT.BFLOAT16, intermediate_data_type=DT.FLOAT, compute_data_type=DT.FLOAT, **kw)
    kb, nb = K // _BLOCK, N // _BLOCK
    A = t["tok"] = g.tensor(name="A", dim=[batch, M, K], stride=[M * K, K, 1], data_type=DT.FP8_E4M3)
    SFA = t["sfa"] = g.tensor(name="SFA", dim=[batch, M, kb], stride=[M * kb, kb, 1], data_type=DT.FLOAT)
    lhs = g.block_scale_dequantize(input=A, descale=SFA, block_size=[1, _BLOCK])
    outs = []
    for i in range(2 if gated else 1):
        stride = [K * N, N, 1] if n_major else [K * N, 1, K]
        B = t[f"w{i}"] = g.tensor(name=f"B{i}", dim=[batch, K, N], stride=stride, data_type=DT.FP8_E4M3)
        SFB = t[f"sfb{i}"] = g.tensor(name=f"SFB{i}", dim=[batch, kb, nb], stride=[kb * nb, 1, kb], data_type=DT.FLOAT)
        rhs = g.block_scale_dequantize(input=B, descale=SFB, block_size=[_BLOCK, _BLOCK])
        outs.append(g.matmul(A=lhs, B=rhs, name=f"mm{i}"))
    y = g.mul(a=g.swish(input=outs[0], name="silu"), b=outs[1], name="mul") if gated else outs[0]
    _finish(g, y, fp8, DT.BFLOAT16, t)
    return g


def _finish(g, y, fp8, out_dtype, t):
    if fp8:
        t["q"], t["qs"] = g.block_scale_quantize(input=y, block_size=_BLOCK, name="q")
        t["q"].set_output(True).set_data_type(DT.FP8_E4M3)
        t["qs"].set_output(True).set_data_type(DT.FLOAT)
    else:
        t["out"] = y
        y.set_output(True).set_data_type(out_dtype)


@pytest.fixture
def offline(monkeypatch):
    """Plan for ``_DEVICE`` with a probe that accepts every plan: the rules alone, no GPU."""
    monkeypatch.setattr(planning, "current_device_properties", lambda: _DEVICE)
    monkeypatch.setattr(C100, "probe_chain", lambda chain, config: None)


def _pick(graph):
    return _name(C100.plan_config(analyze(graph), dynamic_shapes=C100._graph_dynamic_shapes(graph), shapes_fixed=C100._graph_shapes_fixed(graph)))


def _shortlist(graph):
    return [_name(c) for c in C100.plan_configs(analyze(graph), dynamic_shapes=C100._graph_dynamic_shapes(graph), shapes_fixed=C100._graph_shapes_fixed(graph))]


# Grouped templates: (graph, expected pick). E experts over S routed rows: S / E rows per expert.
_MOE_PICKS = {
    # A gated pair (NONE or GATHER): the 16-wide swapped tile up to 16 rows per expert -- up to 1 row at a K of 4096 bytes or
    # more -- the 32-wide one up to 32, the 64-wide one up to 56, then 128x128 (GATHER 1x1; NONE by the NONE cluster rule).
    "gated_fp8_r1": (lambda: _moe(256, 8, 2048, 7168, "gather", gated=True, fp8=True), _cfg(128, 16, swap=True)),
    "gated_fp8_r2_k7168": (lambda: _moe(256, 512, 2048, 7168, "gather", gated=True, fp8=True), _cfg(128, 32, swap=True)),
    "gated_fp8_r16_k7168": (lambda: _moe(256, 4096, 2048, 7168, "gather", gated=True, fp8=True), _cfg(128, 32, swap=True)),
    "gated_fp8_r16_k4096": (lambda: _moe(512, 8192, 1024, 4096, "gather", gated=True, fp8=True), _cfg(128, 32, swap=True)),
    "gated_fp8_r16_k3072": (lambda: _moe(256, 4096, 1024, 3072, "gather", gated=True, fp8=True), _cfg(128, 16, swap=True)),
    "gated_fp8_r16_k2048": (lambda: _moe(256, 4096, 512, 2048, "gather", gated=True, fp8=True), _cfg(128, 16, swap=True)),
    "gated_fp8_r32": (lambda: _moe(128, 4096, 1536, 4096, "gather", gated=True, fp8=True), _cfg(128, 32, swap=True)),
    "gated_fp8_r33": (lambda: _moe(128, 4224, 1536, 4096, "gather", gated=True, fp8=True), _cfg(128, 64, swap=True)),
    "gated_fp8_r40_k2048": (lambda: _moe(512, 20480, 512, 2048, "gather", top_k=10, gated=True, fp8=True), _cfg(128, 64, swap=True)),
    "gated_fp8_r56": (lambda: _moe(256, 14336, 2048, 4096, "gather", gated=True, fp8=True), _cfg(128, 64, swap=True)),
    "gated_fp8_r57": (lambda: _moe(256, 14592, 2048, 4096, "gather", gated=True, fp8=True), _cfg(128, 128)),
    "gated_fp8_r128": (lambda: _moe(256, 32768, 2048, 7168, "gather", gated=True, fp8=True), _cfg(128, 128)),
    "gated_fp8_r80_odd_n_tiles": (lambda: _moe(512, 40960, 640, 2560, "gather", top_k=10, gated=True, fp8=True), _cfg(128, 128)),
    "gated_none_r128": (lambda: _moe(256, 32768, 2048, 7168, gated=True), _cfg(128, 128, "1x2")),
    # One GEMM (NONE or GATHER): a token tile of at least twice the rows per expert -- 16-wide up to 8 (up to 1 at a K of 2560
    # bytes or more), 32-wide up to 16.
    "none_r8": (lambda: _moe(16, 128, 4096, 2048), _cfg(128, 16, swap=True)),
    "none_r9": (lambda: _moe(16, 144, 4096, 2048), _cfg(128, 32, swap=True)),
    "none_r1_k7168": (lambda: _moe(256, 256, 4096, 7168), _cfg(128, 16, swap=True)),
    "none_r2_k2560": (lambda: _moe(256, 512, 1280, 2560), _cfg(128, 32, swap=True)),
    "gather_r8_k7168": (lambda: _moe(384, 3072, 6144, 7168, "gather", top_k=6), _cfg(128, 32, swap=True)),
    "gather_r16": (lambda: _moe(256, 4096, 4096, 7168, "gather"), _cfg(128, 32, swap=True)),
    # Non-swapped: NONE pairs CTAs along N (1x2) from 128 rows per expert at a K of 1536 bytes or more; GATHER never pairs.
    "none_r17": (lambda: _moe(16, 272, 4096, 2048), _cfg(128, 128)),
    "none_r127_k1536": (lambda: _moe(16, 2032, 4096, 1536), _cfg(128, 128)),
    "none_r128_k1536": (lambda: _moe(16, 2048, 4096, 1536), _cfg(128, 128, "1x2")),
    "none_r256_k1024": (lambda: _moe(8, 2048, 1024, 1024), _cfg(128, 128)),
    "none_fp8_r256_k1536": (lambda: _moe(8, 2048, 1024, 1536, fp8=True), _cfg(128, 128, "1x2")),
    "gather_r17": (lambda: _moe(16, 272, 4096, 2048, "gather"), _cfg(128, 128)),
    "gather_r256": (lambda: _moe(8, 2048, 1024, 2048, "gather"), _cfg(128, 128)),
    # COMBINE: the 16-wide swapped tile up to 1 row per expert, and up to 4 at a K of at most 768 bytes; then, at such a K and
    # rows x K of at most 8192 bytes, the 256x16 CTA pair; else 128x128 1x1.
    "combine_r1": (lambda: _moe(256, 8, 7168, 2048, "combine"), _cfg(128, 16, swap=True)),
    "combine_r2_long_k": (lambda: _moe(256, 512, 7168, 2048, "combine"), _cfg(128, 128)),
    "combine_r4": (lambda: _moe(128, 512, 2048, 768, "combine"), _cfg(128, 16, swap=True)),
    "combine_r4_k1024": (lambda: _moe(256, 1024, 3072, 1024, "combine"), _cfg(128, 128)),
    "combine_r5": (lambda: _moe(128, 640, 2048, 768, "combine"), _cfg(256, 16, "2x1", 2, swap=True)),
    "combine_r16_k512": (lambda: _moe(256, 4096, 2048, 512, "combine"), _cfg(256, 16, "2x1", 2, swap=True)),
    "combine_r16_k768": (lambda: _moe(128, 2048, 2048, 768, "combine"), _cfg(128, 128)),
    "combine_r205": (lambda: _moe(160, 32768, 6144, 2560, "combine"), _cfg(128, 128)),
    # An override-shape graph runs other token counts than it declares: it keeps the non-swapped tile, and every mode but GATHER
    # pairs its CTAs (a sorted 1x1 token box of a decode-size token tensor loads slowly: NONE 1x2 / 1x1 = 0.72-1.02 at 1-8 tokens).
    "gather_override_r1": (lambda: _moe(256, 8, 4096, 7168, "gather", override=True), _cfg(128, 128)),
    "none_override_r1": (lambda: _moe(256, 8, 4096, 7168, override=True), _cfg(128, 128, "1x2")),
    "none_override_short_k": (lambda: _moe(256, 32768, 4096, 512, override=True), _cfg(128, 128, "1x2")),
    "combine_override_r1": (lambda: _moe(256, 8, 7168, 2048, "combine", override=True), _cfg(128, 128, "1x2")),
}


@requires_sm100_tree
@pytest.mark.parametrize("case", list(_MOE_PICKS))
def test_grouped_general_pick_follows_the_rows_per_expert(case, offline):
    graph, want = _MOE_PICKS[case]
    assert _pick(graph()) == want


@requires_sm100_tree
@pytest.mark.parametrize(
    "E,S,N,K", [(160, 32768, 5120, 6144), (128, 32768, 3072, 4096), (256, 131072, 4096, 7168)], ids=["coder_4096", "qwen235_4096", "dsv3_16384"]
)
@pytest.mark.parametrize("mode", ["none", "gather", "combine"])
def test_grouped_general_baseline_is_one_cta_of_128_columns(E, S, N, K, mode, offline):
    """No CTA pair (it pads every expert to 256 rows) and no 256-column accumulator (one partial TMEM stage); NONE pairs its
    CTAs along N (1x2) at these rows per expert (205-512) and K (4096-7168 bytes), GATHER and a COMBINE of fixed extents do not."""
    config = C100._baseline_config(analyze(_moe(E, S, N, K, mode)), shapes_fixed=True)
    assert (config.cta_tile_m, config.cta_tile_n, config.cta_group, config.cga_size_m, config.split_k_slices, config.swap_ab) == (128, 128, 1, 1, 1, False)
    assert config.cga_size_n == (2 if mode == "none" else 1)


@requires_sm100_tree
@pytest.mark.parametrize(
    "occupancy,cluster",
    [(_OCC68, "1x2"), ((68, 33) + _OCC68[2:], "1x1")],
    ids=["pairs_cover_68_sms", "pairs_leave_sms_idle"],
)
def test_grouped_general_baseline_pairs_ctas_only_where_the_pairs_cover_the_part(occupancy, cluster, monkeypatch):
    """A grouped launch is max_active_clusters(size) clusters with no fallback cluster: a size that leaves SMs idle loses them."""
    device = planning.DeviceProperties(100, "sm100 test part", 68, 0, occupancy=occupancy)
    monkeypatch.setattr(planning, "current_device_properties", lambda: device)
    config = C100._baseline_config(analyze(_moe(256, 32768, 4096, 7168)), shapes_fixed=True)
    assert config.name == _cfg(128, 128, cluster)


# Dense template: (M, N, K, fp8 output) -> pick on the 68-SM part; a batch, or override / dynamic shapes, in _DENSE_GRAPH_PICKS.
_DENSE_PICKS = {
    "decode_deep_k_two_slices": (1, 4096, 7168, False, _cfg(128, 128, "1x4", slices=2)),
    "two_slices_of_a_filled_grid_need_k6144": (64, 4096, 4096, False, _cfg(128, 128, "1x4")),
    "two_slices_of_a_sparse_grid_from_k3072": (8, 3072, 4096, False, _cfg(128, 128, "1x4", slices=2)),
    "two_slices_need_k3072": (1, 3072, 2048, False, _cfg(128, 128, "1x4")),
    "split_at_the_k_floor_takes_64_columns": (1, 1024, 2048, False, _cfg(128, 64, "1x4", slices=4)),
    "twelve_n_tiles_four_slices": (8, 1536, 2048, False, _cfg(128, 128, "1x4", slices=4)),
    "short_k_64_columns": (8, 2048, 512, False, _cfg(128, 64, "1x4")),
    "short_k_fp8_out_keeps_128_columns": (8, 2048, 512, True, _cfg(128, 128, "1x4")),
    "wide_n_decode_never_256_columns": (1, 12288, 4096, False, _cfg(128, 128, "1x4")),
    "m256_keeps_the_small_m_plan": (256, 4096, 7168, False, _cfg(128, 128, "1x4")),
    # Past 256 rows: the 2x1 CTA pair from 512 rows at a K of 2048 bytes or more, with 16 or more N tiles, rows that fill whole
    # pairs and more tiles than SMs; elsewhere the 1x2 cluster from a K of 1536 bytes; else the widest 1x4 / 1x2 cluster.
    "m512_cta_pair": (512, 4096, 7168, False, _cfg(128, 128, "2x1", 2)),
    "m4096_cta_pair": (4096, 7168, 2048, False, _cfg(128, 128, "2x1", 2)),
    "m8192_sixteen_n_tiles_cta_pair": (8192, 2048, 4096, False, _cfg(128, 128, "2x1", 2)),
    "m640_half_empty_pair_takes_1x2": (640, 6144, 2560, False, _cfg(128, 128, "1x2")),
    "m512_one_wave_of_tiles_takes_1x2": (512, 2048, 4096, False, _cfg(128, 128, "1x2")),
    "m4096_twelve_n_tiles_take_1x2": (4096, 1536, 2048, False, _cfg(128, 128, "1x2")),
    "m4096_k1536_takes_1x2": (4096, 4096, 1536, False, _cfg(128, 128, "1x2")),
    "m384_takes_1x2": (384, 4096, 2048, False, _cfg(128, 128, "1x2")),
    "m4096_ten_n_tiles_1x2": (4096, 1280, 2560, False, _cfg(128, 128, "1x2")),
    "m4096_k1024_keeps_1x4": (4096, 4096, 1024, False, _cfg(128, 128, "1x4")),
    "m4096_short_k_one_cta": (4096, 2048, 512, False, _cfg(128, 128, "1x4")),
    # An FP8 block-128 output runs 1-CTA tiles only, and 1x2 did not pay there.
    "m512_fp8_out_one_block_per_tile": (512, 4096, 7168, True, _cfg(128, 128, "1x4")),
    "m4096_fp8_out_keeps_1x4": (4096, 4096, 7168, True, _cfg(128, 128, "1x4")),
    "m4096_fp8_out_twelve_n_tiles_keeps_1x4": (4096, 1536, 2048, True, _cfg(128, 128, "1x4")),
}


@requires_sm100_tree
@pytest.mark.parametrize("case", list(_DENSE_PICKS))
def test_dense_general_pick(case, offline):
    M, N, K, fp8, want = _DENSE_PICKS[case]
    assert _pick(_dense(M, N, K, fp8=fp8)) == want


_DENSE_GRAPH_PICKS = {
    # 64-column tiles only while the M tiles of every batch times the N tiles fill at most half the SMs (68 here): two M tiles
    # or a batch double the grid, and past that 128-column tiles were faster (M = 192: 8.6 against 10.1 us).
    "two_m_tiles_fill_the_part": (lambda: _dense(192, 4096, 1024), _cfg(128, 128, "1x4")),
    "two_m_tiles_leave_room": (lambda: _dense(160, 2048, 512), _cfg(128, 64, "1x4")),
    "batch_fills_the_part": (lambda: _dense(8, 4096, 1024, batch=2), _cfg(128, 128, "1x4")),
    "batch_of_one_m_tile_each": (lambda: _dense(1, 512, 512, batch=16), _cfg(128, 128, "1x4")),
    # An override- or dynamic-shape graph runs other M than it declares: neither split K (its workspace grows with M) nor
    # 64-column tiles, the plan of more rows.
    "override_decode_deep_k": (lambda: _dense(1, 4096, 7168, override=True), _cfg(128, 128, "1x4")),
    "override_decode_short_k": (lambda: _dense(1, 4096, 512, override=True), _cfg(128, 128, "1x4")),
    "dynamic_decode_short_k": (lambda: _dense(1, 4096, 512, dynamic=True), _cfg(128, 128, "1x4")),
    # The CTA pair and the 1x2 cluster were measured on one GEMM of K-major operands, batch 1, at the declared rows: a gated pair
    # (1-CTA tiles only), an N-major weight, a batch and an override-shape graph (it may run at decode sizes, where the pair lost
    # 1.2-1.5x) keep the 1-CTA plan.
    "m4096_gated_pair_keeps_1x4": (lambda: _dense(4096, 2048, 4096, gated=True), _cfg(128, 128, "1x4")),
    "m4096_n_major_weight_keeps_1x4": (lambda: _dense(4096, 4096, 7168, n_major=True), _cfg(128, 128, "1x4")),
    "m1024_batch_of_two_keeps_1x4": (lambda: _dense(1024, 4096, 7168, batch=2), _cfg(128, 128, "1x4")),
    "override_m4096_keeps_1x4": (lambda: _dense(4096, 4096, 7168, override=True), _cfg(128, 128, "1x4")),
}


@requires_sm100_tree
@pytest.mark.parametrize("case", list(_DENSE_GRAPH_PICKS))
def test_dense_general_pick_of_a_batched_or_reshaped_graph(case, offline):
    graph, want = _DENSE_GRAPH_PICKS[case]
    assert _pick(graph()) == want


@requires_sm100_tree
def test_dense_general_split_fills_one_wave_of_the_part(monkeypatch):
    """The slices fill one wave of the device's SMs: 32 N tiles take 2 slices on 68 SMs and 4 on 148."""
    monkeypatch.setattr(C100, "probe_chain", lambda chain, config: None)
    for sm, want in ((68, 2), (148, 4)):
        device = planning.DeviceProperties(100, "sm100 test part", sm, 0, occupancy=_OCC68)
        monkeypatch.setattr(planning, "current_device_properties", lambda device=device: device)
        assert C100.plan_config(analyze(_dense(1, 4096, 7168)), shapes_fixed=True).split_k_slices == want


@requires_sm100_tree
def test_dense_general_dynamic_shapes_do_not_split(offline):
    assert _pick(_dense(1, 4096, 7168, dynamic=True)) == _cfg(128, 128, "1x4")


@requires_sm100_tree
def test_grouped_general_shortlist_leads_with_the_pick(offline):
    """P-08: the autotuning list holds the pick, the non-swapped baseline (here the 128x128 1x1 plan, which is also the
    fallback), the other token tiles, the CTA pairs along the weight rows and the other clusters, each once and at most
    MAX_PLAN_CONFIGS."""
    got = _shortlist(_moe(256, 8, 4096, 7168))
    assert got[:2] == [_cfg(128, 16, swap=True), _cfg(128, 128)]
    assert set(got) >= {
        _cfg(128, 32, swap=True),
        _cfg(128, 64, swap=True),
        _cfg(256, 16, "2x1", 2, swap=True),
        _cfg(256, 32, "2x1", 2, swap=True),
        _cfg(128, 128, "1x2"),
        _cfg(128, 128, "1x4"),
    }
    assert len(got) == len(set(got)) <= planning.MAX_PLAN_CONFIGS


@requires_sm100_tree
@pytest.mark.parametrize("mode,first", [("gather", "1x1"), ("none", "1x2")])
def test_grouped_general_override_shortlist_is_not_swapped(mode, first, offline):
    got = _shortlist(_moe(256, 8, 4096, 7168, mode, override=True))
    assert got == [_cfg(128, 128, first)] + [_cfg(128, 128, c) for c in ("1x1", "1x2", "1x4") if c != first]


@requires_sm100_tree
def test_dense_general_shortlist_varies_split_cluster_and_tile(offline):
    got = _shortlist(_dense(1, 4096, 7168))
    assert got[:2] == [_cfg(128, 128, "1x4", slices=2), _cfg(128, 128)]
    assert set(got) >= {_cfg(128, 128, "1x4"), _cfg(128, 128, "1x4", slices=4), _cfg(128, 128, slices=2), _cfg(128, 64)}
    assert not any("2ctamma" in name for name in got), "a CTA pair is slower at decode sizes"
    assert len(got) == len(set(got)) <= planning.MAX_PLAN_CONFIGS
    assert _cfg(128, 128, "2x1", 2) in _shortlist(_dense(4096, 4096, 7168))


@requires_sm100_tree
def test_dense_general_cta_pair_is_followed_by_its_one_cta_plan(offline, monkeypatch):
    """A CTA pair pick leads the shortlist; the 1-CTA plan the chain takes without it follows, then the 128x128 1x1 fallback,
    the other clusters and the 2x2 pair. Where the probe declines the pair, the pick is that 1-CTA plan, not the fallback."""
    got = _shortlist(_dense(4096, 4096, 7168))
    assert got[:3] == [_cfg(128, 128, "2x1", 2), _cfg(128, 128, "1x4"), _cfg(128, 128)]
    assert set(got) >= {_cfg(128, 128, "1x2"), _cfg(128, 128, "2x2", 2), _cfg(128, 64)}
    assert len(got) == len(set(got)) <= planning.MAX_PLAN_CONFIGS

    def no_pairs(chain, config):
        if config.is_cta_pair_mma:
            raise NotImplementedError(f"declined {_name(config)}")

    monkeypatch.setattr(C100, "probe_chain", no_pairs)
    assert _pick(_dense(4096, 4096, 7168)) == _cfg(128, 128, "1x4")


@requires_sm100_tree
def test_general_pick_falls_back_to_the_one_cta_128x128_plan(monkeypatch):
    """C-09: when the probe declines the pick and the baseline, the 128x128 1x1 1-CTA plan; when it declines every plan, the
    baseline's reason."""
    monkeypatch.setattr(planning, "current_device_properties", lambda: _DEVICE)
    fallback = _cfg(128, 128)

    def only_fallback(chain, config):
        if _name(config) != fallback:
            raise NotImplementedError(f"declined {config.name}")

    monkeypatch.setattr(C100, "probe_chain", only_fallback)
    assert _pick(_dense(512, 4096, 7168, fp8=True)) == fallback
    assert _pick(_moe(256, 8, 4096, 7168, "gather")) == fallback

    def none(chain, config):
        raise NotImplementedError(f"declined {_name(config)}")

    monkeypatch.setattr(C100, "probe_chain", none)
    with pytest.raises(NotImplementedError, match=f"declined {_cfg(128, 128, '1x4')}$"):
        C100.plan_config(analyze(_dense(512, 4096, 7168, fp8=True)), shapes_fixed=True)


# With the real probe (an SM100 GPU; no kernel is launched).

_REAL = {
    "gated_fp8_decode": lambda: _moe(256, 8, 2048, 7168, "gather", gated=True, fp8=True),
    "gated_fp8_r16_k7168": lambda: _moe(256, 4096, 2048, 7168, "gather", gated=True, fp8=True),
    "gated_fp8_r40": lambda: _moe(512, 20480, 512, 2048, "gather", top_k=10, gated=True, fp8=True),
    "gather_bf16_prefill": lambda: _moe(256, 32768, 4096, 7168, "gather"),
    "none_fp8_prefill": lambda: _moe(8, 2048, 1024, 1024, fp8=True),
    "none_fp8_r256_k1536": lambda: _moe(8, 2048, 1024, 1536, fp8=True),
    "combine_decode": lambda: _moe(256, 64, 7168, 2048, "combine"),
    "combine_prefill": lambda: _moe(160, 32768, 6144, 2560, "combine"),
    "dense_decode": lambda: _dense(1, 4096, 7168),
    "dense_fp8_prefill": lambda: _dense(2048, 4096, 7168, fp8=True),
    "dense_bf16_prefill": lambda: _dense(2048, 4096, 7168),
}
_REAL_PICKS = {
    "gated_fp8_r16_k7168": _cfg(128, 32, swap=True),
    "gated_fp8_r40": _cfg(128, 64, swap=True),
    "gather_bf16_prefill": _cfg(128, 128),
    "none_fp8_r256_k1536": _cfg(128, 128, "1x2"),
    "dense_bf16_prefill": _cfg(128, 128, "2x1", 2),
}


@requires_sm100
@pytest.mark.parametrize("case", list(_REAL))
def test_real_probe_accepts_the_pick_and_every_shortlist_entry(case):
    graph = _REAL[case]()
    chain = analyze(graph)
    pick = C100.plan_config(chain, shapes_fixed=True)
    shortlist = C100.plan_configs(chain, shapes_fixed=True)
    assert shortlist[0] == pick
    assert len(shortlist) >= 3
    for config in shortlist:
        C100.probe_chain(chain, config)
    if case in _REAL_PICKS:
        # The rule's pick itself, not a fallback after the probe declined it.
        assert _name(pick) == _REAL_PICKS[case]
    if case.startswith("gated_fp8"):
        # A wide quantize declines the CTA pair: the shortlist drops it.
        assert not any(c.cta_group == 2 for c in shortlist)


@requires_sm100
def test_dense_cta_pair_pick_writes_the_bytes_of_the_one_cta_plan():
    """The CTA pair a dense chain now takes from 512 rows keeps the 128-column register promotion of the 1x4 plan it had: one
    MMA order per K window and one scale product per window, so the two write the same bf16 bytes."""
    import torch
    from cudnn.gemm.frost.tile_config import by_name

    M, N, K = 1024, 4096, 2048
    kb, nb = K // _BLOCK, N // _BLOCK
    t = {}
    graph = _dense(M, N, K, t=t)
    pick = C100.plan_config(analyze(graph), shapes_fixed=True)
    assert _name(pick) == _cfg(128, 128, "2x1", 2)
    gen = torch.Generator(device="cuda").manual_seed(11)
    a = (torch.randn(1, M, K, device="cuda", generator=gen) * 0.5).to(torch.float8_e4m3fn)
    sa = torch.rand(1, M, kb, device="cuda", generator=gen) + 0.5
    w = (torch.randn(1, N, K, device="cuda", generator=gen) * 0.5).to(torch.float8_e4m3fn)
    sb = torch.rand(1, nb, kb, device="cuda", generator=gen) + 0.5
    outs = []
    for config in (pick, by_name(_cfg(128, 128, "1x4"))):
        out = torch.full((1, M, N), float("nan"), device="cuda", dtype=torch.bfloat16)
        plan = C100.jit_from_cudnn_graph(graph, config=config)
        plan({t["tok"]: a, t["sfa"]: sa, t["w0"]: w.transpose(1, 2), t["sfb0"]: sb.transpose(1, 2), t["out"]: out})
        torch.cuda.synchronize()
        outs.append(out)
    assert not torch.isnan(outs[0]).any()
    assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))


@requires_sm100
@pytest.mark.parametrize("dense", [True, False], ids=["dense", "moe_none"])
def test_fp8_block128_output_plans_through_the_public_api(dense):
    """C-09: the default geometry of a graph with an FP8 block-128 output is one the wide quantize runs. It was a CTA pair (or
    a 256-column tile) from a few hundred rows on -- 2048 rows here, 256 per expert -- and the engine proposed no plan."""
    import torch
    from cudnn.engines import is_python_engine

    E, S, N, K = 8, 2048, 1024, 1024
    t = {}
    graph = _dense(S, N, K, fp8=True, t=t) if dense else _moe(E, S, N, K, fp8=True, t=t)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    frost = [i for i in range(graph.get_execution_plan_count()) if is_python_engine(graph.get_engine_and_knobs_at_index(i)[0])]
    assert frost, "no FROST plan proposed"
    graph.select_plan(frost[0])
    graph.check_support()
    graph.build_plans()

    gen = torch.Generator(device="cuda").manual_seed(3)
    lead, kb, nb = (1 if dense else E), K // _BLOCK, N // _BLOCK
    tok = (torch.randn(1, S, K, device="cuda", generator=gen) * 0.5).to(torch.float8_e4m3fn)
    sa = torch.rand(1, S, kb, device="cuda", generator=gen) + 0.5
    w = (torch.randn(lead, N, K, device="cuda", generator=gen) * 0.5).to(torch.float8_e4m3fn)
    sb = torch.rand(lead, nb, kb, device="cuda", generator=gen) + 0.5
    tok_f = tok[0].float() * sa[0].repeat_interleave(_BLOCK, 1)
    w_f = w.float() * sb.repeat_interleave(_BLOCK, 1).repeat_interleave(_BLOCK, 2)
    q = torch.full((1, S, N), 0x7F, device="cuda", dtype=torch.uint8)  # an e4m3 NaN: an unwritten byte shows
    qs = torch.full((1, S, nb), float("nan"), device="cuda")
    pack = {t["tok"]: tok, t["sfa"]: sa, t["w0"]: w.transpose(1, 2), t["sfb0"]: sb.transpose(1, 2), t["q"]: q.view(torch.float8_e4m3fn), t["qs"]: qs}
    if dense:
        ref = tok_f @ w_f[0].T
    else:
        offsets = [i * S // E for i in range(E + 1)]
        pack[t["fto"]] = torch.tensor(offsets, dtype=torch.int32, device="cuda").view(-1, 1, 1)
        ref = torch.cat([tok_f[offsets[e] : offsets[e + 1]] @ w_f[e].T for e in range(E)])
    workspace = torch.empty(max(1, graph.get_workspace_size()), device="cuda", dtype=torch.uint8)
    graph.execute(pack, workspace)
    torch.cuda.synchronize()
    want_scale = ref.view(S, nb, _BLOCK).abs().amax(dim=2) / 448.0
    torch.testing.assert_close(qs[0], want_scale, rtol=1e-4, atol=0.0)
    got = q[0].view(torch.float8_e4m3fn).float() * qs[0].repeat_interleave(_BLOCK, dim=1)
    assert float((got - ref).norm() / ref.norm()) < 4e-2


@requires_sm100
def test_override_shape_dense_plan_runs_other_m_in_the_workspace_it_declares():
    """An override-shape graph declared at M = 1, its workspace sized once with get_workspace_size(), runs at M = 64 and
    512 through the public API. Planned for the declared M it split K, and its workspace, which grows with M, refused both."""
    import torch
    from cudnn.engines import is_python_engine

    N, K = 4096, 7168
    kb, nb = K // _BLOCK, N // _BLOCK
    t = {}
    graph = _dense(1, N, K, override=True, t=t)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    frost = [i for i in range(graph.get_execution_plan_count()) if is_python_engine(graph.get_engine_and_knobs_at_index(i)[0])]
    assert frost, "no FROST plan proposed"
    graph.select_plan(frost[0])
    graph.check_support()
    graph.build_plans()
    workspace = torch.empty(max(1, graph.get_workspace_size()), device="cuda", dtype=torch.uint8)
    gen = torch.Generator(device="cuda").manual_seed(5)
    w = (torch.randn(1, N, K, device="cuda", generator=gen) * 0.5).to(torch.float8_e4m3fn)
    sb = torch.rand(1, nb, kb, device="cuda", generator=gen) + 0.5
    w_f = w[0].float() * sb[0].repeat_interleave(_BLOCK, 0).repeat_interleave(_BLOCK, 1)
    for M in (64, 512):
        a = (torch.randn(1, M, K, device="cuda", generator=gen) * 0.5).to(torch.float8_e4m3fn)
        sa = torch.rand(1, M, kb, device="cuda", generator=gen) + 0.5
        out = torch.full((1, M, N), float("nan"), device="cuda", dtype=torch.bfloat16)
        pack = {t["tok"]: a, t["w0"]: w.transpose(1, 2), t["sfa"]: sa, t["sfb0"]: sb.transpose(1, 2), t["out"]: out}
        uids = [t["tok"].get_uid(), t["sfa"].get_uid(), t["out"].get_uid()]
        shapes = [[1, M, K], [1, M, kb], [1, M, N]]
        strides = [[M * K, K, 1], [M * kb, kb, 1], [M * N, N, 1]]
        graph.execute(pack, workspace, override_uids=uids, override_shapes=shapes, override_strides=strides)
        torch.cuda.synchronize()
        ref = (a[0].float() * sa[0].repeat_interleave(_BLOCK, 1)) @ w_f.T
        assert float((out[0].float() - ref).norm() / ref.norm()) < 2e-2, f"M = {M}"


@requires_sm100
def test_recommend_offers_the_shortlist():
    """P-08: an autotuning caller gets the pick first and the other candidates after it, each replayable from its knobs."""
    from cudnn.gemm.frost.heuristics import analyze_facts, recommend
    from cudnn.gemm.frost.knobs import GemmKnobs

    graph = _moe(256, 64, 7168, 2048, "combine")
    facts = analyze_facts(graph)
    plans = recommend("A", facts, {"frost_gemm": 7})
    assert len(plans) >= 4
    assert plans[0].knobs == GemmKnobs.from_config(C100.plan_config(facts.chain, shapes_fixed=True))
    for plan in plans:
        C100.probe_chain(facts.chain, C100.plan_config(facts.chain, knobs=plan.knobs))
