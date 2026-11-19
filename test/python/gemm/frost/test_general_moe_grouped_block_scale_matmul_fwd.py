# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""General MoE grouped block-scale matmul (sm100_general_moe_grouped_block_scale_matmul_fwd*.py).

A grouped matmul whose scale tensors are linear, e.g. DeepSeek FP8 with 1x128 token and 128x128 weight scales.
The four templates are the grouped block-scale pipelines (plain, weight-by-token, GATHER and GATHER
weight-by-token) with the plain tcgen05.mma on the raw operands and the scales applied to the accumulator.
Registry gates (no GPU), then end-to-end numerics against an fp32 dequantize-then-matmul reference."""

from __future__ import annotations

import dataclasses
import re

import cudnn
import cudnn.gemm.frost  # noqa: F401  (installs recorder)
import pytest
import torch

import warm_stress
from gemm_test_utils import Plan, ceil_div, keep_kernel_ptx, kernel_ptx, requires_sm100
from moe_block_scale_test_utils import _build_graph as _native_graph
from cudnn.gemm.frost import compiler as C
from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.gemm.frost.kernel_registry import GraphType, classify_graph_type, select_template, swap_ab_graph, template_path
from cudnn.gemm.frost.tile_config import by_name

pytestmark = pytest.mark.L0

_CUDNN = {"fp8_e4m3": cudnn.data_type.FP8_E4M3, "bf16": cudnn.data_type.BFLOAT16, "fp16": cudnn.data_type.HALF, "fp32": cudnn.data_type.FLOAT}
_TORCH = {"fp8_e4m3": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
_STEM = "sm100_general_moe_grouped_block_scale_matmul_fwd"
# (GATHER, weight-by-token) -> template
_TEMPLATES = {
    (False, False): f"{_STEM}.py",
    (False, True): f"{_STEM}_swap_ab.py",
    (True, False): f"{_STEM}_gather_mode.py",
    (True, True): f"{_STEM}_gather_mode_swap_ab.py",
}


def _config(m, n, cluster="1x1", cta=1):
    return f"CONFIG_sm100_{m}x{n}x128_{min(m, 128)}x{n}x32_cluster{cluster}_{cta}ctamma"


_TILE_128 = _config(128, 128)


def _graph(
    E,
    S,
    N,
    K,
    G,
    *,
    gather_rows=None,
    scatter=None,
    combine=None,
    relu=False,
    block_a=(1, 128),
    block_b=(128, 128),
    sfa="fp32",
    sfb="fp32",
    out="bf16",
    shared_weight_scale=False,
):
    """``out[rows of group g] = dequant(token rows) @ dequant(weight[g % E]).T`` with linear scales.

    ``S`` is the token tensor's row count; under GATHER ``gather_rows`` routed rows index into it. ``scatter`` /
    ``combine`` are the top-k of a SCATTER / COMBINE graph (the output is fp32 under COMBINE). The weight scale is
    one matrix per expert, or one that every expert shares (``shared_weight_scale``)."""
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    t = {}
    t["tok"] = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=cudnn.data_type.FP8_E4M3)
    t["w"] = g.tensor(name="weight", dim=[E, K, N], stride=[K * N, 1, K], data_type=cudnn.data_type.FP8_E4M3)
    am, ak = ceil_div(S, block_a[0]), ceil_div(K, block_a[1])
    bk, bn = ceil_div(K, block_b[0]), ceil_div(N, block_b[1])
    t["sfa"] = g.tensor(name="SFA", dim=[1, am, ak], stride=[am * ak, ak, 1], data_type=_CUDNN[sfa])
    t["sfb"] = g.tensor(name="SFB", dim=[1 if shared_weight_scale else E, bk, bn], stride=[bk * bn, 1, bk], data_type=_CUDNN[sfb])
    t["fto"] = g.tensor(name="first_token_offset", dim=[G + 1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    tok_d = g.block_scale_dequantize(input=t["tok"], descale=t["sfa"], block_size=list(block_a))
    w_d = g.block_scale_dequantize(input=t["w"], descale=t["sfb"], block_size=list(block_b))
    routing, mode = {}, cudnn.moe_grouped_matmul_mode.NONE
    if gather_rows is not None:
        mode = cudnn.moe_grouped_matmul_mode.GATHER
        t["idx"] = g.tensor(name="token_index", dim=[1, gather_rows, 1], stride=[gather_rows, 1, 1], data_type=cudnn.data_type.INT32)
        routing["token_index"] = t["idx"]
    top_k = scatter or combine
    if top_k:
        mode = cudnn.moe_grouped_matmul_mode.COMBINE if combine else cudnn.moe_grouped_matmul_mode.SCATTER
        t["idx"] = g.tensor(name="token_index", dim=[1, S, 1], stride=[S, 1, 1], data_type=cudnn.data_type.INT32)
        t["ks"] = g.tensor(name="token_ks", dim=[1, S, 1], stride=[S, 1, 1], data_type=cudnn.data_type.INT32)
        routing = {"token_index": t["idx"], "token_ks": t["ks"], "top_k": top_k}
        if combine:
            t["scores"] = g.tensor(name="scores", dim=[1, S // combine, combine], stride=[S, combine, 1], data_type=cudnn.data_type.FLOAT)
            routing["top_k_scores"] = t["scores"]
    o = g.moe_grouped_matmul(tok_d, w_d, t["fto"], mode=mode, compute_data_type=cudnn.data_type.FLOAT, name="moe", **routing)
    if relu:
        o = g.relu(input=o, name="relu")
    o.set_data_type(_CUDNN["fp32" if combine else out]).set_output(True)
    t["out"] = o
    return g, t


def _offsets(rows, groups, seed=1):
    """Ragged groups that stop short of the capacity, with one empty group."""
    used = rows - rows // 8
    cuts = sorted(torch.randint(0, used, (groups - 1,), generator=torch.Generator().manual_seed(seed)).tolist())
    offsets = [0, *cuts, used]
    if groups > 2:
        offsets[2] = offsets[1]
    return offsets


def _run(
    E=8,
    S=512,
    N=256,
    K=512,
    *,
    gather_rows=None,
    scatter=None,
    combine=None,
    relu=False,
    swap=False,
    config=_TILE_128,
    block_a=(1, 128),
    block_b=(128, 128),
    sfa="fp32",
    sfb="fp32",
    out="bf16",
    offsets=None,
    seed=0,
    tol=2e-2,
):
    """Run one grouped product through its general template; returns (plan, output, reference, used rows)."""
    gather = gather_rows is not None
    rows = gather_rows if gather else S
    offsets = _offsets(rows, E) if offsets is None else offsets
    g, t = _graph(
        E,
        S,
        N,
        K,
        len(offsets) - 1,
        gather_rows=gather_rows,
        scatter=scatter,
        combine=combine,
        relu=relu,
        block_a=block_a,
        block_b=block_b,
        sfa=sfa,
        sfb=sfb,
        out=out,
    )
    plan = Plan(g, config=by_name(config), swap_ab=swap)
    assert select_template(plan.chain, plan._compiled.config).file == _TEMPLATES[(gather, swap)]

    gen = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    tok = (torch.randn(1, S, K, device=dev, generator=gen) * 0.5).to(torch.float8_e4m3fn)
    w = (torch.randn(E, N, K, device=dev, generator=gen) * 0.5).to(torch.float8_e4m3fn)
    am, ak = ceil_div(S, block_a[0]), ceil_div(K, block_a[1])
    bk, bn = ceil_div(K, block_b[0]), ceil_div(N, block_b[1])

    def scale(shape, dt):
        s = (torch.rand(shape, device=dev, generator=gen) * 1.5 + 0.25) * (torch.randint(0, 2, shape, device=dev, generator=gen) * 2 - 1)
        return s.to(_TORCH[dt])

    sa = scale((1, am, ak), sfa)
    sb_nk = scale((E, bn, bk), sfb)  # the graph tensor is its [E, k blocks, n blocks] transpose
    tok_f = tok[0].float() * sa[0].float().repeat_interleave(block_a[0], 0)[:S].repeat_interleave(block_a[1], 1)[:, :K]
    w_f = w.float() * sb_nk.float().repeat_interleave(block_b[1], 1)[:, :N].repeat_interleave(block_b[0], 2)[:, :, :K]
    idx = torch.randint(0, S, (1, rows, 1), device=dev, generator=gen, dtype=torch.int32) if gather else None
    ref = torch.zeros(rows, N, device=dev)
    for gi in range(len(offsets) - 1):
        lo, hi = offsets[gi], offsets[gi + 1]
        if hi > lo:
            src = tok_f[idx[0, lo:hi, 0].long()] if gather else tok_f[lo:hi]
            ref[lo:hi] = src @ w_f[gi % E].T
    if relu:
        ref = ref.relu()

    result = torch.full((1, rows, N), float("nan"), device=dev, dtype=_TORCH[out])
    pack = {
        t["tok"]: tok,
        t["w"]: w.transpose(1, 2),
        t["sfa"]: sa,
        t["sfb"]: sb_nk.transpose(1, 2),
        t["fto"]: torch.tensor(offsets, dtype=torch.int32, device=dev).view(-1, 1, 1),
        t["out"]: result,
    }
    if gather:
        pack[t["idx"]] = idx
    used = offsets[-1]
    top_k = scatter or combine
    if top_k:
        # Routed row r goes to slot dest[r] % top_k of token dest[r] // top_k.
        dest = torch.randperm(S, device=dev, generator=gen).to(torch.int32)
        pack[t["idx"]], pack[t["ks"]] = (dest // top_k).view(1, S, 1), (dest % top_k).view(1, S, 1)
        if combine:
            scores = torch.rand(1, S // combine, combine, device=dev, generator=gen)
            result = torch.full((1, S // combine, N), -123.0, device=dev)
            pack[t["scores"]], pack[t["out"]] = scores, result
        plan(pack)
        torch.cuda.synchronize()
        placed = dest[:used].long()
        if combine:
            # Every token's row is the score-weighted sum of its routed slots; a token without one is zero.
            expected = torch.zeros(S // combine, N, device=dev)
            expected.index_add_(0, placed // combine, ref[:used] * scores[0, placed // combine, placed % combine, None])
            got = result[0]
        else:
            # The routed rows land at their destinations; no other row is written.
            expected, got = ref[:used], result[0, placed].float()
            untouched = torch.ones(S, dtype=torch.bool, device=dev)
            untouched[placed] = False
            assert torch.isnan(result[0, untouched].float()).all()
        err = float((got - expected).norm() / expected.norm())
        assert err < tol, f"relative error {err:.3e}"
        return plan, result, ref, used
    plan(pack)
    torch.cuda.synchronize()
    got = result[0, :used].float()
    assert torch.isfinite(got).all()
    if used:
        err = float((got - ref[:used]).norm() / ref[:used].norm())
        assert err < tol, f"relative error {err:.3e}"
    return plan, result, ref, used


# Registry (no GPU)


def test_linear_scales_classify_as_the_general_grouped_types():
    chain = analyze(_graph(8, 512, 256, 512, 8)[0])
    assert chain.block_scale.linear_scales
    assert classify_graph_type(chain) is GraphType.GENERAL_MOE_BLOCK_SCALE
    assert classify_graph_type(swap_ab_graph(chain)) is GraphType.GENERAL_MOE_BLOCK_SCALE_SWAP_AB
    gathered = analyze(_graph(8, 300, 256, 512, 8, gather_rows=512)[0])
    assert classify_graph_type(gathered) is GraphType.GENERAL_MOE_BLOCK_SCALE


def test_swizzled_scales_keep_the_block_scaled_mma_pipelines():
    assert classify_graph_type(analyze(_native_graph(8, 512, 256, 512, 8, combo="mxfp8"))) is GraphType.MOE_BLOCK_SCALE
    # GATHER reads a linear token scale itself: a native-dtype one stays with the block-scaled pipelines, beside a
    # swizzled weight scale or on its own.
    for kw in (dict(), dict(dequant_b=False)):
        gather = analyze(_native_graph(8, 512, 256, 512, 8, combo="mxfp8", gather_rows=512, **kw))
        assert classify_graph_type(gather) is GraphType.MOE_BLOCK_SCALE


@pytest.mark.parametrize("gather,swap", list(_TEMPLATES))
def test_each_grouped_pipeline_selects_its_general_template(gather, swap):
    chain = analyze(_graph(8, 300 if gather else 512, 256, 512, 8, gather_rows=512 if gather else None)[0])
    config = by_name(_TILE_128)
    if swap:
        chain, config = swap_ab_graph(chain), dataclasses.replace(config, swap_ab=True)
    assert select_template(chain, config).file == _TEMPLATES[(gather, swap)]
    assert select_template(chain, config).accepts(chain, config) is None


@pytest.mark.parametrize(
    "swap,config,share,runs,table",
    [
        # Token rows on the lanes: one weight scale per 128 columns, one run per thread.
        (False, _TILE_128, 64, 1, False),
        # Tokens on the columns: a scale per column. A thread shuffles up to 8 out of its warp's grid ahead of the MMA ...
        (True, _config(128, 16), 8, 8, False),
        # ... and reads more of them from its warpgroup's scale table (any multiple of 16 token columns).
        (True, _config(128, 32), 16, 16, True),
        (True, _config(128, 96), 48, 48, True),
        (True, _TILE_128, 64, 64, True),
    ],
)
def test_general_grouped_plan_shares_the_promotion(swap, config, share, runs, table):
    chain = analyze(_graph(8, 512, 256, 512, 8)[0])
    cfg = by_name(config)
    if swap:
        chain, cfg = swap_ab_graph(chain), dataclasses.replace(cfg, swap_ab=True)
    plan = C._general_block_scale_plan(chain, cfg)
    assert (plan["promo_groups"], plan["promo_cols"], plan["sfb_groups"]) == (2, share, runs)
    assert plan["promo_in_registers"] is True
    assert plan["promo_scale_table"] is table
    # The table's threads take a column pair's scales as they fold it; the others hold every run's ahead of the MMA.
    assert plan["promo_scales_ahead"] is not table


@pytest.mark.parametrize("swap", [False, True], ids=["gather", "gather_swap"])
@pytest.mark.parametrize("config", [_TILE_128, _config(128, 128, "2x1", 2), _config(128, 64)])
def test_general_gather_plan_stays_inside_the_register_pool(swap, config):
    """setmaxnreg waits for registers that are never freed when the roles ask for more than the CTA's entry pool."""
    chain = analyze(_graph(8, 300, 256, 512, 8, gather_rows=512)[0])
    cfg = by_name(config)
    if swap:
        chain, cfg = swap_ab_graph(chain), dataclasses.replace(cfg, swap_ab=True)
    plan = C._general_block_scale_plan(chain, cfg)
    # Per-thread counts of the control, token-copy, epilogue and promotion warps against 20 warps at the entry count.
    per_warp = 4 * C._GENERAL_BS_GATHER_CTL_REGS + 4 * plan["gather_copy_regs"] + 4 * plan["epi_regs"] + 8 * plan["promo_regs"]
    assert per_warp <= 20 * 96
    assert plan["promo_in_registers"] is True


def test_general_grouped_plan_caps_the_stages_of_long_k_tiles():
    def stages(K):
        chain = swap_ab_graph(analyze(_graph(8, 512, 256, K, 8)[0]))
        return C._general_block_scale_plan(chain, dataclasses.replace(by_name(_config(128, 16)), swap_ab=True))["ab_stages"]

    assert stages(7168) == C._GENERAL_BS_MOE_MAX_AB_STAGES
    assert stages(2048) > C._GENERAL_BS_MOE_MAX_AB_STAGES


# End-to-end numerics (GPU)


_CASES = {
    # the four pipelines on DeepSeek FP8 block-128 scales, ragged groups with an empty one
    "fwd": dict(),
    "swap": dict(swap=True),
    "gather": dict(S=300, gather_rows=512),
    "gather_swap": dict(S=300, gather_rows=512, swap=True),
    # paired-CTA MMA
    "fwd_2cta": dict(config=_config(128, 128, "2x1", 2)),
    "swap_2cta": dict(swap=True, config=_config(128, 128, "2x1", 2)),
    "gather_2cta": dict(S=300, gather_rows=512, config=_config(128, 128, "2x1", 2)),
    "gather_swap_2cta": dict(S=300, gather_rows=512, swap=True, config=_config(128, 128, "2x1", 2)),
    # narrow token tiles of the weight-by-token pipelines
    "swap_n16": dict(swap=True, config=_config(128, 16)),
    "swap_n32": dict(swap=True, config=_config(128, 32)),
    "swap_n64": dict(swap=True, config=_config(128, 64)),
    "gather_swap_n32": dict(S=2000, gather_rows=1024, swap=True, config=_config(128, 32), E=16, N=512, K=1024),
    # token tiles between the powers of two (three partial stages, 8- and 16-column promotion loads)
    "swap_n48": dict(swap=True, config=_config(128, 48)),
    "swap_n96": dict(swap=True, config=_config(128, 96)),
    "gather_swap_n80": dict(S=300, gather_rows=512, swap=True, config=_config(128, 80)),
    "gather_swap_n64": dict(S=300, gather_rows=512, swap=True, config=_config(128, 64)),
    # other tiles of the token-by-weight pipelines; a 256-column accumulator keeps its sum in TMEM
    "fwd_n64": dict(config=_config(128, 64)),
    "fwd_n256": dict(config=_config(128, 256), N=512),
    "gather_n64": dict(S=64, gather_rows=1024, E=16, N=384, K=896, config=_config(128, 64)),
    # extents that end inside a tile and inside a scale block
    "partial_tiles": dict(E=4, S=300, N=200, K=448),
    "partial_tiles_swap": dict(E=4, S=300, N=200, K=448, swap=True, config=_config(128, 32)),
    # other scale dtypes and block shapes
    "half_scales": dict(sfa="bf16", sfb="fp16"),
    "half_scales_swap": dict(sfa="bf16", sfb="fp16", swap=True, config=_config(128, 32)),
    "small_blocks": dict(block_a=(1, 64), block_b=(64, 32), K=384),
    "small_blocks_swap": dict(block_a=(1, 64), block_b=(64, 32), K=384, swap=True, config=_config(128, 32)),
    "fp32_output": dict(E=3, S=1000, N=384, K=1024, out="fp32", config=_config(128, 128, "2x1", 2), tol=1e-5),
    # more groups than experts: group g runs expert g % E
    "groups_wrap_experts": dict(E=4, offsets=[0, 40, 40, 130, 200, 260, 300, 390, 448]),
    # the routing modes and the epilogue of the plain and weight-by-token pipelines
    "scatter": dict(scatter=2),
    "scatter_swap": dict(scatter=2, swap=True, config=_config(128, 32)),
    "combine": dict(combine=2, tol=1e-5),
    "combine_swap": dict(combine=2, swap=True, config=_config(128, 32), tol=1e-5),
    "combine_swap_2cta": dict(combine=2, swap=True, config=_config(128, 128, "2x1", 2), tol=1e-5),
    "relu": dict(relu=True),
    "relu_swap": dict(relu=True, swap=True, config=_config(128, 64)),
}


@requires_sm100
@pytest.mark.parametrize("case", list(_CASES), ids=list(_CASES))
def test_general_grouped_block_scale_numerics(case):
    _run(**_CASES[case])


@requires_sm100
@pytest.mark.parametrize("kw", [dict(), dict(swap=True, config=_config(128, 32)), dict(S=300, gather_rows=512)], ids=["fwd", "swap", "gather"])
def test_general_grouped_block_scale_repeats_bitwise(kw):
    _, first, _, _ = _run(**kw)
    _, second, _, _ = _run(**kw)
    assert torch.equal(first.view(torch.int16), second.view(torch.int16))


@requires_sm100
@pytest.mark.parametrize("swap", [False, True], ids=["fwd", "swap"])
def test_general_grouped_block_scale_rows_do_not_depend_on_their_group_neighbors(swap):
    """A routed row's result is the same bits whatever else shares its group and tile."""
    config = _config(128, 32) if swap else _TILE_128
    E, S, N, K = 4, 256, 256, 512
    _, full, _, _ = _run(E=E, S=S, N=N, K=K, swap=swap, config=config, offsets=[0, 100, 100, 180, 256])
    # the same tokens, routed as one group per token run of 4 rows inside expert 0's first group
    _, alone, _, _ = _run(E=E, S=S, N=N, K=K, swap=swap, config=config, offsets=[0, 4, 4, 4, 4])
    assert torch.equal(full[0, :4].view(torch.int16), alone[0, :4].view(torch.int16))


@requires_sm100
def test_general_grouped_block_scale_rejects_a_mismatched_scale_buffer():
    g, t = _graph(8, 512, 256, 512, 8)
    plan = Plan(g, config=by_name(_TILE_128))
    pack = {
        t["tok"]: torch.zeros(1, 512, 512, device="cuda").to(torch.float8_e4m3fn),
        t["w"]: torch.zeros(8, 256, 512, device="cuda").to(torch.float8_e4m3fn).transpose(1, 2),
        t["sfa"]: torch.ones(1, 512, 3, device="cuda"),  # one K block short
        t["sfb"]: torch.ones(8, 4, 2, device="cuda"),
        t["fto"]: torch.zeros(9, 1, 1, dtype=torch.int32, device="cuda"),
        t["out"]: torch.zeros(1, 512, 256, device="cuda", dtype=torch.bfloat16),
    }
    with pytest.raises(ValueError, match=r"A scale\[0\] must have shape"):
        plan(pack)


# The count of weight scale matrices is part of the compiled kernel (one per expert, read at the tile's expert, or one
# that every expert shares, read at matrix 0): a call binds the count the graph declared.

_COUNT_CASES = {
    "fwd": dict(),
    "swap": dict(swap=True, config=_config(128, 32)),
    "gather": dict(gather=True),
    "gather_swap": dict(gather=True, swap=True, config=_config(128, 32)),
}


def _count_case(shared, *, swap=False, gather=False, config=_TILE_128):
    """Two experts over unit operands: a routed row's 128 outputs are K times the weight scale its kernel read.

    The weight scale matrices hold 1 (expert 0) and 2 (expert 1); the token scale is 1. Returns the plan, the pack
    without the weight scale, the output and the two matrices' backing tensor, [E, n blocks, k blocks]."""
    E, S, N, K = 2, 256, 128, 512
    g, t = _graph(E, S, N, K, E, gather_rows=S if gather else None, shared_weight_scale=shared)
    plan = Plan(g, config=by_name(config), swap_ab=swap)
    assert select_template(plan.chain, plan._compiled.config).file == _TEMPLATES[(gather, swap)]
    dev = "cuda"
    backing = torch.empty(E, 1, 4, device=dev)
    backing[0], backing[1] = 1.0, 2.0
    out = torch.full((1, S, N), float("nan"), device=dev, dtype=torch.bfloat16)
    pack = {
        t["tok"]: torch.ones(1, S, K, device=dev).to(torch.float8_e4m3fn),
        t["w"]: torch.ones(E, N, K, device=dev).to(torch.float8_e4m3fn).transpose(1, 2),
        t["sfa"]: torch.ones(1, S, 4, device=dev),
        t["fto"]: torch.tensor([0, 128, 256], dtype=torch.int32, device=dev).view(-1, 1, 1),
        t["out"]: out,
    }
    if gather:
        pack[t["idx"]] = torch.arange(S, dtype=torch.int32, device=dev).view(1, S, 1)
    return plan, t, pack, out, backing


# A GATHER graph declares one weight scale matrix per expert: the analyzer has no shared spelling for it.
_COUNT_PARAMS = [(case, shared) for case in _COUNT_CASES for shared in (False, True) if not (shared and _COUNT_CASES[case].get("gather"))]
_COUNT_IDS = [f"{case}-{'shared' if shared else 'per_expert'}" for case, shared in _COUNT_PARAMS]


@requires_sm100
@pytest.mark.parametrize("case,shared", _COUNT_PARAMS, ids=_COUNT_IDS)
def test_general_grouped_block_scale_reads_the_declared_scale_matrix_count(case, shared):
    """The control of the refusals below: the declared count, bound, scales each group by its own (or the one) matrix."""
    plan, t, pack, out, backing = _count_case(shared, **_COUNT_CASES[case])
    # A view of the two matrices' storage either way, so the bound tensor is what selects the matrix.
    pack[t["sfb"]] = (backing[:1] if shared else backing).transpose(1, 2)
    plan(pack)
    torch.cuda.synchronize()
    expected = torch.tensor([512.0, 512.0 if shared else 1024.0], device="cuda").repeat_interleave(128)[:, None].expand(256, 128)
    assert torch.equal(out[0].float(), expected)


@requires_sm100
@pytest.mark.parametrize("case,shared", _COUNT_PARAMS, ids=_COUNT_IDS)
def test_general_grouped_block_scale_refuses_the_other_scale_matrix_count(case, shared):
    """A kernel compiled for one scale matrix per expert reads matrix ``expert`` of what is bound, past the end of a
    single matrix; one compiled for a shared matrix reads matrix 0 and would ignore the others. Either binding is
    refused before the launch, in the kernel's axis order and in the graph's, and nothing is written.

    The single matrix is a view of the two matrices' storage: were the binding accepted, the kernel would read
    initialized memory of this test, not past an allocation."""
    plan, t, pack, out, backing = _count_case(shared, **_COUNT_CASES[case])
    other = backing if shared else backing[:1]
    wanted, compiled = (1, "shared by every matrix") if shared else (2, "per matrix")
    for bound in (other, other.transpose(1, 2)):
        pack[t["sfb"]] = bound
        with pytest.raises(ValueError, match=rf"[AB] scale\[0\] must have shape \[{wanted}, 1, 4\].*compiled for one scale matrix {compiled}"):
            plan(pack)
    torch.cuda.synchronize()
    assert torch.isnan(out.float()).all()


# Releasing what was read (python/cudnn/AGENTS.md Rule 11)
#
# A grouped template hands three shared-memory stages back to their writers: a slot of the scheduler ring (every
# consumer warp reads the tile's words and credits the slot's empty barrier), the broadcast mailbox (a scheduler warp
# reads the claimed index and acknowledges to cluster rank 0) and, where a thread has many column runs, a stage of the
# promotion scale table (released with the partial). Each release goes through cudnn.frost.tile_dsl.consume, which
# computes the address of the arrive from every word read, so no word is still unread when the stage may be rewritten.
# The GATHER templates acknowledge the mailbox with a reduction, which the helper has no form for: they store the
# index they read in front of it.


def _consumed_names(expr, body):
    """The names a release passes as ``consumed``: a name, a list, or ``slot_words`` and what the code put into it."""
    expr = expr.strip()
    if expr != "slot_words":
        return [n.strip() for n in expr.strip("[]").split(",")]
    made = re.findall(r"slot_words = \[([^\]]*)\]", body)[-1]
    return [n.strip() for n in made.split(",")] + re.findall(r"slot_words\.append\((\w+)\)", body)


@pytest.mark.parametrize("gather,swap", list(_TEMPLATES))
def test_general_grouped_templates_release_every_word_they_read(gather, swap):
    """At the source: every release of a stage a warp read goes through the consume helper with each word it loaded."""
    source = template_path(_TEMPLATES[(gather, swap)]).read_text()
    # The scheduler ring: no bare arrive on a slot's empty barrier, and a release names every word loaded since the
    # warp acquired the slot (its wait on the full barrier).
    assert "mbarrier_arrive(sched_empty_mbar_ptr" not in source
    releases = list(re.finditer(r"consume_then_arrive\(sched_empty_mbar_ptr\.subview\(\w+\), (.+), rz, pred=elect_one\)", source))
    # The loader, the MMA warp's three arms, the epilogue and the promotion warps; GATHER adds its relay and copy warps.
    assert len(releases) == (8 if gather else 6)
    for release in releases:
        body = source[source.rindex("sched_full_mbar_ptr.subview(", 0, release.start()) : release.start()]
        loaded = list(dict.fromkeys(re.findall(r"^\s*(\w+) = .*\.load\(\)\s*$", body, flags=re.M)))
        assert loaded, "a slot release without a slot read"
        assert sorted(_consumed_names(release.group(1), body)) == sorted(loaded), f"line {source.count(chr(10), 0, release.start()) + 1}"
    # The broadcast mailbox.
    assert "mbarrier_arrive(nvvm.mapa(sched_bcast_empty_mbar_ptr" not in source
    read = source.index("linear_idx = (sched_bcast_slot.subview(bcast_stage)).load()")
    if gather:
        store = source.index(".subview(0).store(cutlass.Int64(linear_idx))", read)
        assert store < source.index("_publish_bcast_ack(nvvm.mapa(bcast_ack_count.subview(bcast_stage), 0))", read)
    else:
        assert "consume_then_arrive_on_peer(sched_bcast_empty_mbar_ptr.subview(bcast_stage), 0, linear_idx, rz, pred=lane == 0)" in source[read:]
    # The promotion scale table: both reads are collected, and the partial's release passes them.
    assert len(re.findall(r"\(promo_tbl\.subview\([^)]*\)\)\.load\(\)\n\s+tbl_words\.append\(scales\[mi\]\[run\]\)", source)) == 2
    assert source.count("partial_empty_mbar_ptr.subview(partial_slot), tbl_words, rz, pred=elect_one") == 1
    assert source.count("partial_empty_mbar_ptr.subview(partial_slot), pair_leader_rank, tbl_words, rz, pred=elect_one, release_cta=False") == 1
    # The mask has the one use the helper allows, and the host passes the literal zero.
    assert source.count("release_zero_arg") == 2 and "rz = release_zero(release_zero_arg)" in source
    assert re.search(r"# @@TMA_STORE_ONLY:END@@\n\s+# [^\n]*\n\s+cutlass\.Int32\(0\),\n\s+\)\.launch\(", source)


# Per rendering: the run, and the loaded registers each release's address is computed from, sorted. A slot word is one
# 32-bit register: the MMA warp reads the validity word (once per arm it renders), the loader and the promotion warps
# six words, the epilogue seven; the relay warp of a paired GATHER plan one and the token-copy warps four. The
# mailbox is one word where an mbarrier acknowledges it. A scale table adds the partial's release (TABLE: at least one
# register; the count depends on how the loads are packed).
TABLE = "table"
_RELEASE_RENDERINGS = {
    "fwd": (dict(), [1, 1, 6, 6, 7]),
    "fwd_2cta": (dict(config=_config(128, 128, "2x1", 2)), [1, 1, 1, 6, 6, 7]),
    "swap_n32": (dict(swap=True, config=_config(128, 32)), [1, 1, 6, 6, 7, TABLE]),
    # a paired plan releases the partial (and with it the table) on its pair leader's barrier
    "swap_2cta": (dict(swap=True, config=_config(128, 128, "2x1", 2)), [1, 1, 1, 6, 6, 7, TABLE]),
    "gather": (dict(S=300, gather_rows=512), [1, 4, 6, 6, 7]),
    "gather_2cta": (dict(S=300, gather_rows=512, config=_config(128, 128, "2x1", 2)), [1, 1, 1, 4, 6, 6, 7]),
    "gather_swap_n32": (dict(S=300, gather_rows=512, swap=True, config=_config(128, 32)), [1, 4, 6, 6, 7, TABLE]),
}


def _mailbox_store_precedes_the_reduction(ptx):
    """GATHER, a cluster of several thread blocks: the index a lane read from the mailbox is stored in front of the
    reduction that acknowledges it. Returns False when the PTX has no such reduction (a cluster of one)."""
    lines = ptx.splitlines()
    reductions = [i for i, line in enumerate(lines) if "red.release.cluster.shared::cluster.add.u32" in line]
    if not reductions:
        return False
    assert len(reductions) == 1
    stores = [i for i in range(reductions[0]) if lines[i].strip().startswith("st.shared.b64")]
    assert stores, "no shared-memory store in front of the acknowledgement"
    stored = re.search(r"(%rd\d+);", lines[stores[-1]]).group(1)
    widened = [i for i in range(stores[-1]) if re.match(rf"\s*cvt\.s64\.s32\s+{stored},", lines[i])]
    assert widened, f"the stored register {stored} is not the widened index"
    index = re.search(r"(%r\d+);", lines[widened[-1]]).group(1)
    loads = [i for i in range(widened[-1]) if re.match(rf"\s*ld\.shared\.b32\s+{index},", lines[i])]
    assert loads, f"the stored value {index} is not what was read from shared memory"
    # Straight-line code from the read to the store: no label between them.
    assert not any(lines[i].rstrip().endswith(":") for i in range(loads[-1], stores[-1]))
    return True


@requires_sm100
@pytest.mark.parametrize("case", list(_RELEASE_RENDERINGS))
def test_general_grouped_release_addresses_depend_on_the_words_read(case, monkeypatch, tmp_path):
    """In the generated PTX: the address of every such arrive is computed, through the kernel's release-zero argument,
    from the registers the warp loaded. The rest of a rendering's arrives release no stage read with shared-memory loads."""
    kw, expected = _RELEASE_RENDERINGS[case]
    keep_kernel_ptx(monkeypatch, C, tmp_path)
    plan, *_ = _run(**kw)
    ptx = kernel_ptx(plan)
    sites = warm_stress.assert_release_dependence(ptx, sites=len(expected), words=1)
    found = sorted(site.loaded_registers for site in sites if site.dependent)
    words = sorted(w for w in expected if w != TABLE)
    if TABLE in expected:
        assert plan._compiled is not None and C._general_block_scale_plan(plan.chain, plan._compiled.config)["promo_scale_table"]
        # The table's release is the one the slot words do not account for.
        for w in words:
            found.remove(w)
        assert len(found) == 1 and found[0] >= 1
    else:
        assert found == words
    # Every dependent arrive is predicated (the elected lane, or the one lane that credits rank 0) ...
    assert all(site.text.startswith("@") for site in sites if site.dependent)
    # ... and the mailbox of a GATHER cluster is acknowledged behind the store of what was read.
    gather, clustered = "gather_rows" in kw, "cluster2x1" in kw.get("config", "")
    assert _mailbox_store_precedes_the_reduction(ptx) == (gather and clustered)


def _exact_case(*, swap=False, gather=False, config=_TILE_128, E=8, G=1024, rows_per_group=16, N=128, K=128):
    """``G`` groups of ``rows_per_group`` rows over ``E`` experts with an integer-valued reference of every output byte.

    Operands and scales are small integers and powers of two: every product and partial sum is exact in fp32 in any
    order, and the bf16 output is that sum rounded once. Returns the plan, its pack, the output and the reference."""
    S = G * rows_per_group
    g, t = _graph(E, S, N, K, G, gather_rows=S if gather else None)
    plan = Plan(g, config=by_name(config), swap_ab=swap)
    assert select_template(plan.chain, plan._compiled.config).file == _TEMPLATES[(gather, swap)]
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(11)
    tok = torch.randint(-3, 4, (1, S, K), device=dev, generator=gen).float()
    w = torch.randint(-3, 4, (E, N, K), device=dev, generator=gen).float()
    sa = torch.exp2(torch.randint(-1, 2, (1, S, K // 128), device=dev, generator=gen).float())
    sb_nk = torch.exp2(torch.randint(-1, 2, (E, N // 128, K // 128), device=dev, generator=gen).float())
    idx = torch.randperm(S, device=dev, generator=gen).to(torch.int32).view(1, S, 1) if gather else None
    out = torch.empty(1, S, N, device=dev, dtype=torch.bfloat16)
    pack = {
        t["tok"]: tok.to(torch.float8_e4m3fn),
        t["w"]: w.to(torch.float8_e4m3fn).transpose(1, 2),
        t["sfa"]: sa,
        t["sfb"]: sb_nk.transpose(1, 2),
        t["fto"]: (torch.arange(G + 1, dtype=torch.int32, device=dev) * rows_per_group).view(-1, 1, 1),
        t["out"]: out,
    }
    if gather:
        pack[t["idx"]] = idx
    tok_f = tok[0] * sa[0].repeat_interleave(128, 1)
    routed = tok_f[idx[0, :, 0].long()] if gather else tok_f
    expert = torch.arange(G, device=dev).repeat_interleave(rows_per_group) % E
    w_f = w * sb_nk.repeat_interleave(128, 1).repeat_interleave(128, 2)
    reference = torch.einsum("sk,snk->sn", routed.double(), w_f.double()[expert]).float().to(torch.bfloat16).view(1, S, N)
    return plan, pack, out, reference


def _stress_case(name, *, swap=False, gather=False, config=_TILE_128):
    """A launch whose thread blocks each take many tiles, with an integer-valued reference of every output byte.

    1024 groups of 16 rows over 8 experts: 1024 tiles for the launch's thread blocks, so every scheduler slot is
    rewritten several times inside one launch."""
    G, rows_per_group = 1024, 16
    plan, pack, out, reference = _exact_case(swap=swap, gather=gather, config=config, G=G, rows_per_group=rows_per_group)
    workspace = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda")
    cfg = plan._compiled.config
    blocks = C._grid_num_clusters(cfg) * cfg.cga_size_m * cfg.cga_size_n
    # The routed rows are the tile's columns once the operands are swapped, its rows otherwise.
    tiles = G * -(-rows_per_group // (cfg.cta_tile_n if swap else cfg.cta_tile_m))
    return warm_stress.StressCase(
        name=f"general grouped block scale {name}",
        launch=lambda: plan(pack, workspace=workspace),
        outputs={"out": out},
        thread_blocks=blocks,
        # A thread block's share of the tiles through its two-slot ring.
        refills_per_launch=max(tiles // blocks - 2, 0),
        reference={"out": reference},
    )


_STRESS = {
    "fwd": dict(),
    "swap": dict(swap=True, config=_config(128, 16)),
    "swap_table": dict(swap=True, config=_config(128, 32)),
    "gather": dict(gather=True),
    "gather_swap": dict(gather=True, swap=True, config=_config(128, 16)),
}


@requires_sm100
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("case", list(_STRESS))
def test_general_grouped_warm_launches_keep_writing_the_reference_bytes(case):
    """Warm launches, back to back and behind a launch that overwrites every SM's shared memory, write the bytes of an
    independent reference: the run-time detector of a slot, mailbox or table word that is read after its stage was
    rewritten. A rate bound for this shape on this GPU, and a pass only where the helper's control (the unsafe twin of
    its test ring) is seen to fail; the order itself is pinned in the generated code by the test above.

    The launch runs one thread block per SM (the kernel fixes its register count at entry that way), which the report
    flags: the control's ring does not fail in that geometry, so the bound is weaker than at two blocks per SM."""
    warm_stress.build_launchers(torch.device("cuda"))
    stress = _stress_case(case, **_STRESS[case])
    assert stress.refills_per_launch >= 4, "the shape rewrites every scheduler slot inside one launch"
    warm_stress.assert_warm_stress(stress, validation=warm_stress.stress_validation(torch.device("cuda")))


# A CUDA graph replays a launch with the arguments it was captured with, the scheduler's launch epoch among them.

_GRAPH_REPLAYS = {
    "fwd": dict(),
    # every cluster's first claim but a few is already past the tiles
    "fwd_fewer_tiles_than_clusters": dict(G=8),
    # cluster rank 0 claims for both thread blocks of a pair
    "fwd_2cta": dict(config=_config(128, 128, "2x1", 2)),
    "swap_table": dict(swap=True, config=_config(128, 32)),
    "gather": dict(gather=True),
    "gather_2cta": dict(gather=True, config=_config(128, 128, "2x1", 2)),
    "gather_swap": dict(gather=True, swap=True, config=_config(128, 16)),
}


@requires_sm100
@pytest.mark.parametrize("case", list(_GRAPH_REPLAYS))
def test_general_grouped_graph_replays_write_every_tile(case):
    """Every replay of a captured launch counts its tiles from zero, though it runs with the launch epoch it was
    captured with: back-to-back replays of one graph, a second graph of the same plan, eager launches in between and
    a replay behind a workspace another use filled with other bytes all write the reference bytes, through the same
    workspace and each into an output poisoned before it."""
    plan, pack, out, reference = _exact_case(**{"G": 256, "N": 256, **_GRAPH_REPLAYS[case]})
    workspace = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def launch():
        plan._compiled(pack, workspace=workspace, stream=torch.cuda.current_stream().cuda_stream)

    def run(step, what):
        out.fill_(float("nan"))
        step()
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), reference.view(torch.int16)), what

    graphs = [torch.cuda.CUDAGraph() for _ in range(2)]
    try:
        with torch.cuda.stream(stream):
            run(launch, "the eager launch before the captures")
            for graph in graphs:
                with torch.cuda.graph(graph, stream=stream):
                    launch()
        for i in range(3):
            run(graphs[0].replay, f"replay {i + 1} of a graph, back to back")
        with torch.cuda.stream(stream):
            run(launch, "an eager launch between the replays")
        for i in range(2):
            run(graphs[1].replay, f"replay {i + 1} of a second graph of the plan")
        for i in range(2):
            for j, graph in enumerate(graphs):
                run(graph.replay, f"round {i + 1} of alternating replays, graph {j + 1}")
        workspace.random_(0, 256)
        run(graphs[0].replay, "a replay behind a workspace filled with other bytes")
        run(graphs[0].replay, "the replay after it")
    finally:
        torch.cuda.synchronize()
        for graph in graphs:
            graph.reset()
