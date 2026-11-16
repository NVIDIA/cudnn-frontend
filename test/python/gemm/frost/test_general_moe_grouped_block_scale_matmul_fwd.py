# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""General MoE grouped block-scale matmul (sm100_general_moe_grouped_block_scale_matmul_fwd*.py).

A grouped matmul whose scale tensors are linear, e.g. DeepSeek FP8 with 1x128 token and 128x128 weight scales.
The four templates are the grouped block-scale pipelines (plain, weight-by-token, GATHER and GATHER
weight-by-token) with the plain tcgen05.mma on the raw operands and the scales applied to the accumulator.
Registry gates (no GPU), then end-to-end numerics against an fp32 dequantize-then-matmul reference."""

from __future__ import annotations

import dataclasses

import cudnn
import cudnn.gemm.frost  # noqa: F401  (installs recorder)
import pytest
import torch

from gemm_test_utils import Plan, ceil_div, requires_sm100
from moe_block_scale_test_utils import _build_graph as _native_graph
from cudnn.gemm.frost import compiler as C
from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.gemm.frost.kernel_registry import GraphType, classify_graph_type, select_template, swap_ab_graph
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
    E, S, N, K, G, *, gather_rows=None, scatter=None, combine=None, relu=False, block_a=(1, 128), block_b=(128, 128), sfa="fp32", sfb="fp32", out="bf16"
):
    """``out[rows of group g] = dequant(token rows) @ dequant(weight[g % E]).T`` with linear scales.

    ``S`` is the token tensor's row count; under GATHER ``gather_rows`` routed rows index into it. ``scatter`` /
    ``combine`` are the top-k of a SCATTER / COMBINE graph (the output is fp32 under COMBINE)."""
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    t = {}
    t["tok"] = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=cudnn.data_type.FP8_E4M3)
    t["w"] = g.tensor(name="weight", dim=[E, K, N], stride=[K * N, 1, K], data_type=cudnn.data_type.FP8_E4M3)
    am, ak = ceil_div(S, block_a[0]), ceil_div(K, block_a[1])
    bk, bn = ceil_div(K, block_b[0]), ceil_div(N, block_b[1])
    t["sfa"] = g.tensor(name="SFA", dim=[1, am, ak], stride=[am * ak, ak, 1], data_type=_CUDNN[sfa])
    t["sfb"] = g.tensor(name="SFB", dim=[E, bk, bn], stride=[bk * bn, 1, bk], data_type=_CUDNN[sfb])
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
