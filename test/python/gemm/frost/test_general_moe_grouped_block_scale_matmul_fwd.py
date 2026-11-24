# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""General MoE grouped block-scale matmul (sm100_general_moe_grouped_block_scale_matmul_fwd*.py).

A grouped matmul whose scale tensors are linear, e.g. DeepSeek FP8 with 1x128 token and 128x128 weight scales.
The four templates are the grouped block-scale pipelines (plain, weight-by-token, GATHER and GATHER
weight-by-token) with the plain tcgen05.mma on the raw operands and the scales applied to the accumulator.
Registry gates (no GPU), then end-to-end numerics against an fp32 dequantize-then-matmul reference."""

from __future__ import annotations

import ast
import concurrent.futures
import contextlib
import dataclasses
import re

import cudnn
import cudnn.gemm.frost  # noqa: F401  (installs recorder)
import pytest
import torch

import warm_stress
from gemm_test_utils import Plan, ceil_div, keep_kernel_ptx, kernel_ptx, requires_sm100, requires_sm100_tree
from moe_block_scale_test_utils import _build_graph as _native_graph
from cudnn.gemm.frost import compiler as C
from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.gemm.frost.kernel_registry import GraphType, classify_graph_type, select_template, swap_ab_graph, template_path
from cudnn.gemm.frost.tile_config import by_name

# Every test here concerns the sm100 tree's general templates and compiler (several call names only that compiler has):
# a process that runs another family's tree skips them.
pytestmark = [pytest.mark.L0, requires_sm100_tree]

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
    combine_out="fp32",
    offset_dtype="int32",
    fto_alignment=1,
):
    """``out[rows of group g] = dequant(token rows) @ dequant(weight[g % E]).T`` with linear scales.

    ``S`` is the token tensor's row count; under GATHER ``gather_rows`` routed rows index into it. ``scatter`` /
    ``combine`` are the top-k of a SCATTER / COMBINE graph (the output is ``combine_out`` under COMBINE). The weight
    scale is one matrix per expert, or one that every expert shares (``shared_weight_scale``). first_token_offset is
    ``offset_dtype``, INT32 or INT64, and declares ``fto_alignment`` as its alignment_value."""
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    t = {}
    t["tok"] = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=cudnn.data_type.FP8_E4M3)
    t["w"] = g.tensor(name="weight", dim=[E, K, N], stride=[K * N, 1, K], data_type=cudnn.data_type.FP8_E4M3)
    am, ak = ceil_div(S, block_a[0]), ceil_div(K, block_a[1])
    bk, bn = ceil_div(K, block_b[0]), ceil_div(N, block_b[1])
    t["sfa"] = g.tensor(name="SFA", dim=[1, am, ak], stride=[am * ak, ak, 1], data_type=_CUDNN[sfa])
    t["sfb"] = g.tensor(name="SFB", dim=[1 if shared_weight_scale else E, bk, bn], stride=[bk * bn, 1, bk], data_type=_CUDNN[sfb])
    fto_dtype = {"int32": cudnn.data_type.INT32, "int64": cudnn.data_type.INT64}[offset_dtype]
    t["fto"] = g.tensor(name="first_token_offset", dim=[G + 1, 1, 1], stride=[1, 1, 1], data_type=fto_dtype, alignment_value=fto_alignment)
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
    o.set_data_type(_CUDNN[combine_out if combine else out]).set_output(True)
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
    combine_out="fp32",
    place_token=None,
    offset_dtype="int32",
    fto_alignment=1,
):
    """Run one grouped product through its general template; returns (plan, output, reference, used rows).

    ``place_token`` (optional) maps the token tensor to the buffer the call binds, e.g. a copy at a chosen address."""
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
        combine_out=combine_out,
        offset_dtype=offset_dtype,
        fto_alignment=fto_alignment,
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
        t["tok"]: tok if place_token is None else place_token(tok),
        t["w"]: w.transpose(1, 2),
        t["sfa"]: sa,
        t["sfb"]: sb_nk.transpose(1, 2),
        t["fto"]: torch.tensor(offsets, dtype=getattr(torch, offset_dtype), device=dev).view(-1, 1, 1),
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
            result = torch.full((1, S // combine, N), -123.0, device=dev, dtype=_TORCH[combine_out])
            pack[t["scores"]], pack[t["out"]] = scores, result
        plan(pack)
        torch.cuda.synchronize()
        placed = dest[:used].long()
        if combine:
            # Every token's row is the score-weighted sum of its routed slots; a token without one is zero.
            expected = torch.zeros(S // combine, N, device=dev)
            expected.index_add_(0, placed // combine, ref[:used] * scores[0, placed // combine, placed % combine, None])
            got = result[0].float()
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


@pytest.mark.parametrize(
    "rows,gather,alignment,group_cols",
    [
        (1, False, 1, 1),
        # A run of 16 columns may start at any routed row: one column per scale.
        (32, False, 1, 1),
        (32, False, 8, 8),
        # Groups start on a 32-row block: a thread's 16 columns stay inside one.
        (32, False, 32, 16),
        (2, False, 8, 2),
        # Gathered columns are any token rows, whatever the offsets.
        (32, True, 32, 1),
    ],
    ids=["one_row", "rows32_unaligned", "rows32_align8", "rows32_align32", "rows2_align8", "rows32_gather_align32"],
)
def test_a_swapped_token_scale_of_several_rows_is_shared_only_within_the_offset_alignment(rows, gather, alignment, group_cols):
    """Swapped, the token columns of a tile start at its group's first routed row, so a run of columns shares a token
    scale only when every group starts on a run boundary of the scale block (alignment_value), and never under GATHER.
    A longer run gave the next block's rows the scale of its first column (relative error 0.3-1.3)."""
    chain = analyze(_graph(8, 300 if gather else 512, 256, 512, 8, gather_rows=512 if gather else None, block_a=(rows, 128), fto_alignment=alignment)[0])
    chain, cfg = swap_ab_graph(chain), dataclasses.replace(by_name(_config(128, 32)), swap_ab=True)
    plan = C._general_block_scale_plan(chain, cfg)
    assert plan["promo_n_cols"] == 16
    assert plan["sfb_group_cols"] == group_cols


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


@pytest.mark.parametrize(
    "swap,config",
    [(False, _TILE_128), (False, _config(128, 128, "2x1", 2)), (True, _config(128, 16)), (True, _config(128, 128, "2x1", 2))],
    ids=["fwd", "fwd_2cta", "swap_n16", "swap_2cta"],
)
def test_general_grouped_plan_fills_the_register_pool(swap, config):
    """The plain grouped templates' control warpgroup (MMA, TMA, scheduler) lowers itself to what the epilogue and the
    two promotion warpgroups leave of the 64K registers of 512 threads: 40 a thread, the pool exactly. More would wait in
    setmaxnreg.inc forever; at 24 the scheduler warp spilled its claim loop."""
    chain = analyze(_graph(8, 512, 256, 512, 8)[0])
    cfg = by_name(config)
    if swap:
        chain, cfg = swap_ab_graph(chain), dataclasses.replace(cfg, swap_ab=True)
    plan = C._general_block_scale_plan(chain, cfg)
    assert plan["ctl_regs"] == 40
    assert 4 * 32 * (plan["ctl_regs"] + plan["epi_regs"] + 2 * plan["promo_regs"]) == 65536
    assert "\nctl_reg_count = 40\n" in C._render_general_block_scale_tile_constants(cfg, chain, select_template(chain, cfg))


def test_general_grouped_plan_caps_the_stages_of_long_k_tiles():
    def stages(K):
        chain = swap_ab_graph(analyze(_graph(8, 512, 256, K, 8)[0]))
        return C._general_block_scale_plan(chain, dataclasses.replace(by_name(_config(128, 16)), swap_ab=True))["ab_stages"]

    assert stages(7168) == C._GENERAL_BS_MOE_MAX_AB_STAGES
    assert stages(2048) > C._GENERAL_BS_MOE_MAX_AB_STAGES


@pytest.mark.parametrize("out", ["bf16", "fp16", "fp32"])
def test_general_grouped_bulk_combine_loads_at_most_64_columns_per_lane(out):
    """The bulk COMBINE drain sizes a TMEM load at 256 bytes of output, 128 columns of a 16-bit output: a tcgen05.ld
    with 128 destination registers, more than ptxas allows any instruction of a 512-thread kernel (65536 / 512 in
    all). The general template drains at most 64 columns per load; the 256-thread block-scale template keeps the
    wide load."""
    cfg = by_name(_TILE_128)
    chain = analyze(_graph(8, 512, 256, 512, 8, combine=2, combine_out=out)[0])
    assert C._use_bulk_combine(chain, cfg) and select_template(chain, cfg).warps_per_cta == 16
    assert C._epi_n_for_chain(cfg, chain) == 64
    native = analyze(_native_graph(8, 512, 256, 512, 8, combo="mxfp8", combine_top_k=2, output_dt=_CUDNN[out]))
    assert C._use_bulk_combine(native, cfg) and select_template(native, cfg).warps_per_cta is None
    assert C._epi_n_for_chain(cfg, native) == 256 // {"bf16": 2, "fp16": 2, "fp32": 4}[out]


@pytest.mark.parametrize("out", ["bf16", "fp32"])
def test_general_grouped_bulk_combine_staging_rows_are_padded(out):
    """A bulk COMBINE stages one row per epilogue thread; rows a multiple of 128 bytes long all start in the same bank,
    so the reservation pads each row by 16 bytes and the templates lay the rows out that far apart."""
    cfg = by_name(_TILE_128)
    chain = analyze(_graph(8, 512, 256, 512, 8, combine=2, combine_out=out)[0])
    row_bytes = 128 * {"bf16": 2, "fp32": 4}[out]
    assert C._combine_smem_bytes(chain, cfg) == 128 * (row_bytes + 16) + 16
    for gather in (False, True):
        src = template_path(_TEMPLATES[(gather, False)]).read_text()
        assert "combine_row_pad = moe_combine_staging_pad_bytes * 8 // cd_dtype.width" in src
        assert "epi_stage_rows * (epi_cols_per_mma_m + combine_row_pad)" in src


def test_general_grouped_swapped_combine_drains_with_lane_loads():
    """A swapped COMBINE of one output loads its tile's routes a column per lane before the accumulator wait; each column
    step takes its route from that lane (a shuffle) and reduces under a predicate. The other outputs and the
    non-swapped COMBINE keep their epilogue."""
    from cudnn.gemm.frost.sm100 import epilogue_codegen as EC

    swapped = swap_ab_graph(analyze(_graph(8, 512, 256, 512, 8, combine=2)[0]))
    cfg = dataclasses.replace(by_name(_config(128, 32)), swap_ab=True)
    assert C._swap_combine_lanes(swapped, cfg)
    consts = C._render_general_block_scale_tile_constants(cfg, swapped, select_template(swapped, cfg))
    assert "\nswap_combine_lanes = True\n" in consts
    snippets = EC.generate(swapped, vec_bytes_epi=4, output_elem_bytes=4, swap_combine_lanes=True)
    assert "_combine_row_ok = row < M" in snippets.aux_views and "row = cutlass.min(row, M - 1)" in snippets.aux_views
    assert "cute.arch.shuffle_sync(combine_lane_tokens" in snippets.epilogue and "moe_combine_add_if(" in snippets.epilogue
    assert "token_index[" not in snippets.epilogue, "a column step loads no route"
    # The routes start at the drain's first tile column.
    src = template_path(_TEMPLATES[(False, True)]).read_text()
    drain_col = re.search(r"\n +coord_n_c = (.+)\n", src).group(1)
    route_args = [arg.strip() for arg in re.search(r"moe_combine_lane_routes\(\n(.*?)\n +\)\n", src, re.S).group(1).split("\n")]
    assert f"{drain_col}," in route_args
    # At most one routed row per expert, a tile drains one column: it keeps the per-column route loads.
    assert not C._swap_combine_lanes(swap_ab_graph(analyze(_graph(16, 16, 256, 512, 16, combine=2)[0])), cfg)
    plain = analyze(_graph(8, 512, 256, 512, 8, combine=2)[0])
    assert not C._swap_combine_lanes(plain, by_name(_TILE_128))
    with pytest.raises(ValueError, match="swap_combine_lanes"):
        EC.generate(plain, vec_bytes_epi=16, output_elem_bytes=4, swap_combine_lanes=True)
    assert not C._swap_combine_lanes(swap_ab_graph(analyze(_graph(8, 512, 256, 512, 8, scatter=2)[0])), cfg)


def test_general_grouped_combine_l2_policies_follow_the_output_size():
    """A COMBINE's weight loads carry the L2 policy its producer picks per tile (evict_first for every group while the
    output fits in L2, for none while the weights the launch can read fit too) and its bulk reductions run evict_last
    once the output takes more than half the L2; the other modes keep their loads and the source templates their
    reductions."""
    from cudnn.frost.device import l2_cache_bytes, resolve_device

    src = template_path(_TEMPLATES[(False, False)]).read_text()
    assert src.count("l2_cache_hint=b_l2_policy,") == 2
    assert "if b_first_streams & (b_first_every_group | (group_end - group_begin <= cgrp_tile_mnk[0])):" in src
    assert "moe_combine_bulk as _moe_combine_bulk," in src and "\ndef moe_combine_bulk(\n" in src
    # The weight-by-token template's weights are its A operand; one token tile of a group reads a weight tile once.
    swap_src = template_path(_TEMPLATES[(False, True)]).read_text()
    assert swap_src.count("l2_cache_hint=a_l2_policy,") == 2
    assert "if a_first_streams & (a_first_every_group | (group_end - group_begin <= cgrp_tile_mnk[1])):" in swap_src
    l2 = int(l2_cache_bytes(resolve_device(None)))
    cfg = by_name(_TILE_128)
    for kw, combine in ((dict(combine=2), True), (dict(), False)):
        chain = analyze(_graph(8, 512, 256, 512, 8, **kw)[0])
        consts = C._render_general_block_scale_tile_constants(cfg, chain, select_template(chain, cfg))
        bounds = tuple(int(re.search(rf"\n{name} = (\d+)\n", consts).group(1)) for name in C._COMBINE_L2_POLICY_CONSTANTS)
        assert bounds == ((l2, l2, l2 // 2) if combine else (0, 1 << 62, 1 << 62))
    for name in ("sm100_moe_grouped_block_scale_matmul_fwd.py", "sm100_general_moe_grouped_block_scale_matmul_fwd_gather_mode.py"):
        assert "\ndef moe_combine_bulk(" not in template_path(name).read_text()


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
    # token scale blocks of several rows. Swapped, a group's first routed row (the first column of its tiles) is any
    # row, so the ragged groups here start inside a block; aligned offsets declared as such keep longer runs. The
    # token-by-weight and gathered cases are controls: a lane reads its own row's scale.
    "token_blocks_of_32_rows": dict(block_a=(32, 128)),
    "token_blocks_of_32_rows_gather": dict(S=300, gather_rows=512, block_a=(32, 128)),
    "token_blocks_of_32_rows_swap": dict(block_a=(32, 128), swap=True, config=_config(128, 32)),
    "token_blocks_of_16_rows_swap_n16": dict(block_a=(16, 128), swap=True, config=_config(128, 16)),
    "token_blocks_of_2_rows_swap_n64": dict(block_a=(2, 128), swap=True, config=_config(128, 64)),
    "token_blocks_of_32_rows_swap_2cta": dict(block_a=(32, 128), swap=True, config=_config(128, 128, "2x1", 2)),
    "token_blocks_of_32_rows_gather_swap": dict(S=300, gather_rows=512, block_a=(32, 128), swap=True, config=_config(128, 32)),
    "token_blocks_of_32_rows_swap_aligned": dict(
        block_a=(32, 128), swap=True, config=_config(128, 32), offsets=[0, 64, 128, 192, 256, 320, 384, 448, 448], fto_alignment=32
    ),
    # scale K blocks of two sizes: a promotion window is the smaller block, and a larger block spans two windows; K=448
    # ends inside the last block and leaves a window of the last K tile past K. The token scale, the weight scale,
    # the weight scale folded into the token grid (swapped), and the token grid (gathered, swapped).
    "k_block_of_two_windows_tokens": dict(block_a=(1, 128), block_b=(64, 128), K=448),
    "k_block_of_two_windows_weights": dict(block_a=(1, 64), block_b=(128, 128), K=448),
    "k_block_of_two_windows_weights_swap": dict(block_a=(1, 64), block_b=(128, 128), K=448, swap=True, config=_config(128, 16)),
    "k_block_of_two_windows_tokens_gather": dict(S=300, gather_rows=512, block_a=(1, 128), block_b=(64, 128), K=448),
    "k_block_of_two_windows_tokens_gather_swap": dict(S=300, gather_rows=512, block_a=(1, 128), block_b=(64, 128), K=448, swap=True, config=_config(128, 32)),
    "fp32_output": dict(E=3, S=1000, N=384, K=1024, out="fp32", config=_config(128, 128, "2x1", 2), tol=1e-5),
    # more groups than experts: group g runs expert g % E
    "groups_wrap_experts": dict(E=4, offsets=[0, 40, 40, 130, 200, 260, 300, 390, 448]),
    # INT64 offsets (the scheduler's shared-memory copy holds 8-byte entries), over more than one 32-group step
    "int64_offsets": dict(offset_dtype="int64", offsets=_offsets(512, 40)),
    "int64_offsets_swap": dict(offset_dtype="int64", offsets=_offsets(512, 40), swap=True, config=_config(128, 32)),
    "int64_offsets_gather": dict(offset_dtype="int64", offsets=_offsets(512, 40), S=300, gather_rows=512),
    "int64_offsets_gather_swap": dict(offset_dtype="int64", offsets=_offsets(512, 40), S=300, gather_rows=512, swap=True, config=_config(128, 32)),
    # a K below the scale block: the 128-wide blocks are no floor on K (the K-tile and the scale block both end inside it)
    "k_below_the_block": dict(K=64),
    "k_below_the_block_gather": dict(K=64, S=300, gather_rows=512),
    # the routing modes and the epilogue of the plain and weight-by-token pipelines
    "scatter": dict(scatter=2),
    "scatter_swap": dict(scatter=2, swap=True, config=_config(128, 32)),
    "combine": dict(combine=2, tol=1e-5),
    "combine_swap": dict(combine=2, swap=True, config=_config(128, 32), tol=1e-5),
    "combine_swap_2cta": dict(combine=2, swap=True, config=_config(128, 128, "2x1", 2), tol=1e-5),
    # 16-bit COMBINE outputs on 1-CTA token-by-weight tiles: 128 columns of a row per bulk reduction, drained 64 at a time
    "combine_bf16": dict(combine=2, combine_out="bf16", tol=1e-2),
    "combine_fp16_cluster1x4": dict(combine=2, combine_out="fp16", N=512, config=_config(128, 128, "1x4"), tol=1e-2),
    # the padded bulk staging rows with a partial last N tile (its columns past N reduce atomically)
    "combine_partial_tiles": dict(E=4, S=300, N=200, K=448, combine=2, tol=1e-5),
    # lane-parallel swapped COMBINE: weight rows past M (a partial feature tile), token tiles of one to three 32-lane
    # batches, 16-bit outputs, and the CTA pair along the weight rows
    "combine_swap_partial_tiles": dict(E=4, S=300, N=200, K=448, combine=2, swap=True, config=_config(128, 32), tol=1e-5),
    "combine_swap_n16": dict(combine=2, swap=True, config=_config(128, 16), tol=1e-5),
    "combine_swap_n96_partial": dict(E=4, S=300, N=200, K=448, combine=2, swap=True, config=_config(128, 96), tol=1e-5),
    "combine_swap_bf16": dict(combine=2, combine_out="bf16", swap=True, config=_config(128, 64), tol=1e-2),
    "combine_swap_fp16_partial": dict(E=4, S=300, N=200, K=448, combine=2, combine_out="fp16", swap=True, config=_config(128, 32), tol=1e-2),
    "combine_swap_2cta_256x32": dict(E=4, S=300, N=200, K=448, combine=2, swap=True, config=_config(256, 32, "2x1", 2), tol=1e-5),
    "combine_swap_2cta_256x16": dict(E=4, S=300, N=200, K=448, combine=2, swap=True, config=_config(256, 16, "2x1", 2), tol=1e-5),
    "combine_swap_2cta_256x16_bf16": dict(E=8, S=512, N=384, K=512, combine=2, combine_out="bf16", swap=True, config=_config(256, 16, "2x1", 2), tol=1e-2),
    # one routed row per expert: the per-column route loads
    "combine_swap_one_row_per_expert": dict(E=16, S=16, N=200, K=448, combine=2, swap=True, config=_config(128, 16), tol=1e-5),
    "relu": dict(relu=True),
    "relu_swap": dict(relu=True, swap=True, config=_config(128, 64)),
}


@requires_sm100
@pytest.mark.parametrize("case", list(_CASES), ids=list(_CASES))
def test_general_grouped_block_scale_numerics(case):
    _run(**_CASES[case])


@requires_sm100
@pytest.mark.parametrize(
    "bounds",
    [(0, 0, 0), (1 << 62, 0, 1 << 62), (1 << 62, 1 << 62, 1 << 62)],
    ids=["one_tile_groups_evict_first_reductions_evict_last", "every_group_evict_first", "no_evict_first"],
)
@pytest.mark.parametrize("out,swap", [("fp32", False), ("bf16", False), ("fp32", True)], ids=["fp32", "bf16", "swap_fp32"])
def test_general_grouped_combine_l2_policies_keep_the_numerics(monkeypatch, bounds, out, swap):
    """Every L2 policy arm of a COMBINE (weight tiles evict_first per group, for all or for none; reductions evict_last
    or not), forced by the thresholds a launch's sizes are held against, on groups of one and of several M (token) tiles."""
    monkeypatch.setattr(C, "_combine_l2_policy_bytes", lambda chain: bounds)
    kw = dict(swap=True, config=_config(128, 32)) if swap else {}
    _run(E=4, S=1000, N=384, K=512, combine=2, combine_out=out, tol=1e-5 if out == "fp32" else 1e-2, **kw)


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
# six words, the epilogue seven; the relay warp of a GATHER plan (either MMA mode) one and the token-copy warps four.
# The mailbox of a cluster of several thread blocks is one word where an mbarrier acknowledges it; a thread block that
# is its own cluster claims without one. A scale table adds the partial's release (TABLE: at least one register; the
# count depends on how the loads are packed).
TABLE = "table"
_RELEASE_RENDERINGS = {
    "fwd": (dict(), [1, 6, 6, 7]),
    "fwd_2cta": (dict(config=_config(128, 128, "2x1", 2)), [1, 1, 1, 6, 6, 7]),
    "swap_n32": (dict(swap=True, config=_config(128, 32)), [1, 6, 6, 7, TABLE]),
    # a paired plan releases the partial (and with it the table) on its pair leader's barrier
    "swap_2cta": (dict(swap=True, config=_config(128, 128, "2x1", 2)), [1, 1, 1, 6, 6, 7, TABLE]),
    "gather": (dict(S=300, gather_rows=512), [1, 1, 4, 6, 6, 7]),
    "gather_2cta": (dict(S=300, gather_rows=512, config=_config(128, 128, "2x1", 2)), [1, 1, 1, 4, 6, 6, 7]),
    "gather_swap_n32": (dict(S=300, gather_rows=512, swap=True, config=_config(128, 32)), [1, 1, 4, 6, 6, 7, TABLE]),
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


def _exact_case(*, swap=False, gather=False, config=_TILE_128, E=8, G=1024, rows_per_group=16, N=128, K=128, stg=False):
    """``G`` groups of ``rows_per_group`` rows over ``E`` experts with an integer-valued reference of every output byte.

    Operands and scales are small integers and powers of two: every product and partial sum is exact in fp32 in any
    order, and the bf16 output is that sum rounded once. ``stg`` stores the output with per-thread stores instead of
    TMA. Returns the plan, its pack, the output and the reference."""
    S = G * rows_per_group
    g, t = _graph(E, S, N, K, G, gather_rows=S if gather else None)
    plan = Plan(g, config=by_name(config), swap_ab=swap, force_stg_epi=stg)
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


# The token loads of the sorted templates stay inside the token tensor
#
# fwd and fwd_swap_ab load a group's token rows through the host's map of the whole token tensor, at absolute rows:
# a box that crosses the tensor's last row is zero-filled past it. They used to rebase a per-thread-block copy of the
# map onto each group with tensormap.replace, which keeps the encoder's size class (bit 21 of qword 1: a strided span
# of at least 128 KiB) of the whole tensor: a box that crossed the end of a smaller group then read up to 128 KiB
# from the group's first row, past the end of the token tensor for the last groups, and the process lost its CUDA
# context where no memory was mapped there. These calls put the token tensor in front of address space that is
# reserved and never mapped.


class _CudaArray:
    """A raw device range for ``torch.as_tensor``."""

    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3, "strides": None}


@contextlib.contextmanager
def _before_unmapped_memory():
    """``place(t)``: a copy of ``t`` whose last byte is the last mapped byte of a CUDA VMM mapping; the two granules of
    address space after it are reserved and never mapped, so a read past its end faults. Released on exit."""
    from cuda.bindings import driver as cu

    def ok(result):
        err, *rest = result
        assert err == cu.CUresult.CUDA_SUCCESS, err
        return rest[0] if rest else None

    torch.cuda.init()
    device = torch.cuda.current_device()
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device
    granule = int(ok(cu.cuMemGetAllocationGranularity(prop, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM)))
    mappings = []

    def place(t):
        nbytes = t.numel() * t.element_size()
        size = -(-nbytes // granule) * granule
        va = int(ok(cu.cuMemAddressReserve(size + 2 * granule, granule, 0, 0)))
        handle = ok(cu.cuMemCreate(size, prop, 0))
        mappings.append((va, size, handle))
        ok(cu.cuMemMap(va, size, 0, handle, 0))
        access = cu.CUmemAccessDesc()
        access.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = device
        access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        ok(cu.cuMemSetAccess(va, size, [access], 1))
        region = torch.as_tensor(_CudaArray(va + size - nbytes, nbytes), device="cuda")
        region.copy_(t.contiguous().view(-1).view(torch.uint8))
        return region.view(t.dtype).view(t.shape)

    try:
        yield place
    finally:
        # Driver calls only: they return an error code on a context a fault left unusable, where torch would raise.
        cu.cuCtxSynchronize()
        for va, size, handle in mappings:
            cu.cuMemUnmap(va, size)
            cu.cuMemRelease(handle)
            cu.cuMemAddressFree(va, size + 2 * granule)


_TAIL_LAYOUTS = {
    # Ragged groups that end at the token tensor's last row, none a whole number of token boxes (128 rows, 32 swapped):
    # every group spans less than 128 KiB at K = 512, the whole tensor 256 KiB.
    "ragged": dict(E=8, S=512, K=512, offsets=[0, 60, 125, 190, 250, 315, 380, 445, 512]),
    # 8 KiB rows: a one-row group 128 KiB before the end, and a 15-row group (120 KiB) after it whose 128-row token box
    # crosses the tensor's end. (A rebased map with the size class set from 120 KiB on read past the tensor there.)
    "one_row_128_kib_before_the_end": dict(E=4, S=64, K=8192, offsets=[0, 20, 48, 49, 64]),
    # Decode, one token routed to 8 experts: a token tensor of 8 rows, fewer than any token box (16 columns of the
    # narrowest swapped tile, 128 rows unswapped).
    "decode_one_row_per_group": dict(E=8, S=8, K=512, offsets=list(range(9))),
}
_TAIL_CASES = {
    "fwd": dict(),
    "fwd_combine": dict(combine=2, tol=1e-5),
    "swap": dict(swap=True, config=_config(128, 32)),
    "swap_combine": dict(combine=2, swap=True, config=_config(128, 32), tol=1e-5),
    # the 16-wide swapped tile, the pick at one routed row per expert (NONE and COMBINE), and a CTA pair along the weight
    # rows, whose 128 token columns are more than the decode layout's tensor holds
    "swap_n16": dict(swap=True, config=_config(128, 16)),
    "swap_n16_combine": dict(combine=2, swap=True, config=_config(128, 16), tol=1e-5),
    "swap_2cta": dict(swap=True, config=_config(128, 128, "2x1", 2)),
}


@requires_sm100
@pytest.mark.parametrize("layout", list(_TAIL_LAYOUTS))
@pytest.mark.parametrize("case", list(_TAIL_CASES))
def test_general_grouped_token_loads_stop_at_the_end_of_the_token_tensor(case, layout):
    """A token tensor whose end is the end of mapped memory: the groups' partial boxes read nothing past it, and the
    product is the reference's."""
    with _before_unmapped_memory() as place:
        _run(N=256, place_token=place, **_TAIL_LAYOUTS[layout], **_TAIL_CASES[case])


# The scheduler counter in the workspace
#
# A grouped launch counts its tiles in one word of the caller's scratch workspace, its launch epoch above the count,
# and starts the count over when the word carries another epoch. The word follows the launch's per-thread-block
# tensor-map slots: where a plan with one slot per thread block keeps its counter, a plan with two keeps a tensor map,
# whose first word is a device address; other uses of the workspace leave other words. A launch that took such a word
# for its own count claimed no tile and wrote nothing.


def _epoch_source():
    """The host's launch-epoch draw. ``cudnn.gemm.frost.compiler`` becomes the active family's module, so the shared
    launch code's module, which draws it, is reached through that code."""
    return C._CompiledMoeBase._launch_moe.__globals__["_next_sched_epoch"]


def test_general_grouped_launch_epochs_carry_bit_30_and_never_repeat_the_last():
    """No device address's upper half (below 2**25 even for a 57-bit address) and no small integer equals an epoch, and a
    launch never draws the epoch of the launch before it; from several threads at once too."""
    draw = _epoch_source()
    epochs = [draw() for _ in range(1000)]
    assert all((1 << 30) <= e < (1 << 31) for e in epochs)
    assert all(a != b for a, b in zip(epochs, epochs[1:]))
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        drawn = list(pool.map(lambda _: draw(), range(4000)))
    assert all((1 << 30) <= e < (1 << 31) for e in drawn)


def _int64(word):
    return word - (1 << 64) if word >= 1 << 63 else word


@requires_sm100
def test_general_grouped_counter_takes_no_count_another_use_left():
    """X (bf16 TMA-store output: two tensor-map slots per thread block) and Y (the same product through per-thread
    stores: one slot) share one workspace, so Y's counter word lies among X's slots. Whatever Y finds on it -- what X
    left, a device address with a small upper half, small integers, or a word whose upper half is the epoch after the
    one drawn last -- every launch writes every tile: the bytes of an independent reference."""
    x_plan, x_pack, x_out, x_ref = _exact_case(G=256)
    y_plan, y_pack, y_out, y_ref = _exact_case(G=256, stg=True)
    x, y = x_plan._compiled, y_plan._compiled
    assert (x._desc_slots_per_cta, y._desc_slots_per_cta) == (2, 1) and x._grid_ctas == y._grid_ctas
    # X writes the second slot of each of its thread blocks, its output map (the token slot stays unused): Y's view of
    # the workspace starts one slot in when that puts Y's counter, the slot after Y's grid_ctas slots, on such a map.
    shift = 1 - y._grid_ctas % 2
    workspace = torch.zeros(max(x.workspace_bytes, 128 * shift + y.workspace_bytes), dtype=torch.uint8, device="cuda")
    y_workspace = workspace[128 * shift :]
    words = workspace.view(torch.int64)
    counter = (shift + y._grid_ctas * y._desc_slots_per_cta) * 16  # Y's counter, in int64 words: inside X's slots

    def run(plan, pack, out, reference, what, ws):
        out.fill_(float("nan"))
        plan(pack, workspace=ws)
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), reference.view(torch.int16)), what

    for i in range(3):
        run(x_plan, x_pack, x_out, x_ref, f"X, round {i + 1}", workspace)
        # What X left on Y's counter: the first word of one of its tensor maps, a device address.
        left = int(words[counter]) & ((1 << 64) - 1)
        assert 0 < left >> 32 < 1 << 25, f"X left {left:#x} on Y's counter word"
        run(y_plan, y_pack, y_out, y_ref, f"Y behind X, round {i + 1}", y_workspace)
    planted = [(f"an address with upper half {h}", (h << 32) | 0x18010000) for h in (1, 2, 3, 4, 256, 30508, 32767, 65535)]
    planted += [("the integer 5", 5), ("the int32 pair (2176, 2177)", (2177 << 32) | 2176), ("all ones", (1 << 64) - 1)]
    for what, word in planted:
        words[counter] = _int64(word)
        run(y_plan, y_pack, y_out, y_ref, f"Y on {what}", y_workspace)
    # A count of launches would reach this upper half next.
    next_epoch = _epoch_source()() + 1
    words[counter] = _int64((next_epoch << 32) | 0x18010000)
    run(y_plan, y_pack, y_out, y_ref, f"Y on a word whose upper half is {next_epoch}, one past the epoch drawn last", y_workspace)


# A programmatic predecessor
#
# The grouped templates launch as programmatic dependents: their thread blocks may start before the kernel ahead of them
# has finished, so every role waits for it (griddepcontrol.wait) before it reads what that kernel writes. A routing
# kernel may release its dependents before it has written the routing: the scheduler, which reads first_token_offset and
# opens the workspace counter, waits too.

_ROUTED = [0, 128, 256, 384, 448]
_EARLIER = [0, 64, 192, 320, 448]  # another routing of the same rows: what the offsets hold until the writer runs


@requires_sm100
@pytest.mark.parametrize("captured", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("gather,swap", list(_TEMPLATES), ids=["none", "none_swap", "gather", "gather_swap"])
def test_general_grouped_routing_is_read_behind_a_programmatic_predecessor(gather, swap, captured):
    """Behind a predecessor that releases its dependents first and writes first_token_offset after a spin (the offsets
    hold another routing until then), the grouped launch of each of the four templates, eager or replayed from a CUDA
    graph, writes the bytes of a launch with the routing in place."""
    from pdl_late_writer import late_write

    E, R, N, K = 4, 512, 256, 256
    S = 300 if gather else R
    g, t = _graph(E, S, N, K, E, gather_rows=R if gather else None)
    plan = Plan(g, config=by_name(_config(128, 32) if swap else _TILE_128), swap_ab=swap)
    assert select_template(plan.chain, plan._compiled.config).file == _TEMPLATES[(gather, swap)]
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(0)
    tok = (torch.randn(1, S, K, device=dev, generator=gen) * 0.5).to(torch.float8_e4m3fn)
    sa = torch.rand(1, S, K // 128, device=dev, generator=gen) + 0.5
    w = (torch.randn(E, N, K, device=dev, generator=gen) * 0.5).to(torch.float8_e4m3fn)
    sb = torch.rand(E, N // 128, K // 128, device=dev, generator=gen) + 0.5
    idx = torch.randint(0, S, (1, R, 1), device=dev, generator=gen, dtype=torch.int32) if gather else None
    routed = torch.tensor(_ROUTED, dtype=torch.int32, device=dev)
    earlier = torch.tensor(_EARLIER, dtype=torch.int32, device=dev)
    fto = routed.clone()
    out = torch.empty(1, R, N, device=dev, dtype=torch.bfloat16)
    pack = {t["tok"]: tok, t["w"]: w.transpose(1, 2), t["sfa"]: sa, t["sfb"]: sb.transpose(1, 2), t["fto"]: fto.view(-1, 1, 1), t["out"]: out}
    if gather:
        pack[t["idx"]] = idx
    workspace = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device=dev)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def launch(writer):
        if writer:
            late_write(routed, fto, spin_us=2000, stream=stream.cuda_stream)
        plan._compiled(pack, workspace=workspace, stream=stream.cuda_stream)

    def run(offsets, step):
        fto.copy_(offsets)
        out.fill_(float("nan"))
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            step()
        torch.cuda.synchronize()
        return out.clone()

    reference = run(routed, lambda: launch(False))
    tok_f = tok[0].float() * sa[0].repeat_interleave(128, 1)
    src = tok_f[idx[0, :, 0].long()] if gather else tok_f
    w_f = w.float() * sb.repeat_interleave(128, 1).repeat_interleave(128, 2)
    used = _ROUTED[-1]
    want = torch.cat([src[_ROUTED[e] : _ROUTED[e + 1]] @ w_f[e].T for e in range(E)])
    assert float((reference[0, :used].float() - want).norm() / want.norm()) < 2e-2
    # The detector: the other routing, read in place of this one, writes other bytes.
    assert not torch.equal(run(earlier, lambda: launch(False)).view(torch.int16), reference.view(torch.int16))
    if not captured:
        for i in range(3):
            got = run(earlier, lambda: launch(True))
            assert torch.equal(got.view(torch.int16), reference.view(torch.int16)), f"eager launch {i + 1}"
        return
    graph = torch.cuda.CUDAGraph()
    try:
        fto.copy_(routed)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            launch(True)
        for i in range(3):
            got = run(earlier, graph.replay)
            assert torch.equal(got.view(torch.int16), reference.view(torch.int16)), f"replay {i + 1}"
    finally:
        torch.cuda.synchronize()
        graph.reset()


# The COMBINE output fill
#
# A COMBINE kernel only adds into its output, so every call zeroes the output first. A general grouped plan zeroes it
# with its own programmatic launch ahead of the kernel (sm100/combine_zero.py) instead of a memset: the fill releases the
# kernel at once and waits for the stream work ahead of it before its first store, and every role of the kernel waits
# for the fill. The fill writes a region whose rows are whole 16-byte vectors, 16-byte aligned; the memset zeroes any
# other output.


def _exact_combine_case(*, swap=False, config=None, E=4, S=512, N=256, K=256, top_k=2, pad=0):
    """A COMBINE call whose every output value is exact in fp32 in any order of its reductions: small-integer operands,
    power-of-two scales and scores. The output is a view of ``backing``, a buffer with ``pad`` more columns per row, all
    of it NaN. Returns the plan, its pack, the output, ``backing`` and the reference of the output."""
    g, t = _graph(E, S, N, K, E, combine=top_k)
    plan = Plan(g, config=by_name(config or (_config(128, 32) if swap else _TILE_128)), swap_ab=swap)
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(5)
    tok = torch.randint(-3, 4, (1, S, K), device=dev, generator=gen).float()
    w = torch.randint(-3, 4, (E, N, K), device=dev, generator=gen).float()
    sa = torch.exp2(torch.randint(-1, 2, (1, S, K // 128), device=dev, generator=gen).float())
    sb_nk = torch.exp2(torch.randint(-1, 2, (E, N // 128, K // 128), device=dev, generator=gen).float())
    scores = torch.exp2(torch.randint(-1, 2, (1, S // top_k, top_k), device=dev, generator=gen).float())
    offsets = _offsets(S, E)
    dest = torch.randperm(S, device=dev, generator=gen).to(torch.int32)
    backing = torch.full((1, S // top_k, N + pad), float("nan"), device=dev)
    out = backing[:, :, :N]
    pack = {
        t["tok"]: tok.to(torch.float8_e4m3fn),
        t["w"]: w.to(torch.float8_e4m3fn).transpose(1, 2),
        t["sfa"]: sa,
        t["sfb"]: sb_nk.transpose(1, 2),
        t["fto"]: torch.tensor(offsets, dtype=torch.int32, device=dev).view(-1, 1, 1),
        t["idx"]: (dest // top_k).view(1, S, 1),
        t["ks"]: (dest % top_k).view(1, S, 1),
        t["scores"]: scores,
        t["out"]: out,
    }
    tok_f = tok[0] * sa[0].repeat_interleave(128, 1)
    w_f = w * sb_nk.repeat_interleave(128, 1).repeat_interleave(128, 2)
    used = offsets[-1]
    rows = torch.cat([tok_f[offsets[e] : offsets[e + 1]].double() @ w_f[e].double().T for e in range(E)]).float()
    placed = dest[:used].long()
    reference = torch.zeros(S // top_k, N, device=dev)
    reference.index_add_(0, placed // top_k, rows * scores[0, placed // top_k, placed % top_k, None])
    return plan, pack, out, backing, reference.view(1, S // top_k, N)


def _no_memset(*args, **kwargs):
    pytest.fail("a general COMBINE call zeroed its output with the memset")


def test_only_general_grouped_combine_plans_build_the_output_fill():
    """A general grouped graph that does not COMBINE, and a COMBINE of the block-scaled MMA pipelines (swizzled scales),
    zero nothing with a fill kernel: they build none."""
    for chain in (
        analyze(_graph(8, 512, 256, 512, 8)[0]),
        analyze(_graph(8, 300, 256, 512, 8, gather_rows=512)[0]),
        analyze(_graph(8, 512, 256, 512, 8, scatter=2)[0]),
        analyze(_native_graph(8, 512, 256, 512, 8, combo="mxfp8", combine_top_k=2)),
    ):
        assert C._combine_zero_kernel(chain) is None


@requires_sm100
def test_general_grouped_combine_fill_that_cannot_be_built_leaves_the_memset(monkeypatch, caplog):
    """A general COMBINE plan whose fill fails to build zeroes with the memset and says so; one built where compiling is
    refused (compiled_cache.refuse_compiles) and no compiled fill exists zeroes with the memset too."""
    import logging

    from cudnn.frost.compiled_cache import CompileRefused
    from cudnn.gemm.frost.sm100 import combine_zero

    chain = analyze(_graph(4, 512, 256, 256, 4, combine=2)[0])

    def fail(*args, **kwargs):
        raise RuntimeError("the fill does not compile here")

    def refuse(*args, **kwargs):
        raise CompileRefused("the fill is not in the compiled cache")

    monkeypatch.setattr(combine_zero, "build", fail)
    with caplog.at_level(logging.WARNING):
        assert C._combine_zero_kernel(chain) is None
    assert "the fill does not compile here" in caplog.text
    monkeypatch.setattr(combine_zero, "build", refuse)
    assert C._combine_zero_kernel(chain) is None


def test_combine_fill_takes_whole_aligned_vectors_only():
    """The fill writes rows of whole 16-byte vectors, 16-byte aligned at every row start; anything else is the memset's."""
    from cudnn.gemm.frost.sm100.combine_zero import CombineZero

    takes = CombineZero.takes
    assert takes(0x1000, 8192, 8192, 1) and takes(0x1000, 1040, 1024, 300) and takes(0x1010, 7, 16, 1)
    assert not takes(0x1004, 8192, 8192, 1)  # start
    assert not takes(0x1000, 1028, 1024, 2)  # pitch
    assert not takes(0x1000, 8196, 8196, 1)  # a partial vector
    assert not takes(0x1000, 0, 0, 1) and not takes(0x1000, 1024, 1024, 0)  # nothing to write


@requires_sm100
@pytest.mark.parametrize(
    "rows,width,pitch,sm_count", [(1, 2048, 2048, None), (1, 1 << 20, 1 << 20, 1), (300, 1024, 1028, 1), (5, 4, 8, 1), (70, 7168, 7168, None)]
)
def test_combine_fill_writes_exactly_its_region(rows, width, pitch, sm_count):
    """``rows`` rows of ``width`` fp32 every ``pitch`` elements are zero after the fill, and every other element of the
    buffer keeps its NaN. An SM count of 1 caps the grid at combine_zero.CTAS_PER_SM thread blocks, which then stride
    over rows and columns."""
    from cudnn.frost.device import multiprocessor_count
    from cudnn.gemm.frost.sm100 import combine_zero

    fill = combine_zero.build(C._frost_compile_options(), sm_count or multiprocessor_count(torch.cuda.current_device()))
    backing = torch.full((rows + 2, pitch), float("nan"), device="cuda")
    region = backing[1 : rows + 1, :width]
    assert fill.takes(region.data_ptr(), pitch * 4, width * 4, rows)
    fill(region.data_ptr(), pitch * 4, width * 4, rows, C._as_custream(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()
    assert torch.equal(region, torch.zeros_like(region))
    outside = torch.ones_like(backing, dtype=torch.bool)
    outside[1 : rows + 1, :width] = False
    assert torch.isnan(backing[outside]).all()


@requires_sm100
@pytest.mark.parametrize("swap", [False, True], ids=["fwd", "swap"])
def test_general_grouped_combine_zeroes_its_output_with_its_fill_kernel(swap, monkeypatch):
    """A general COMBINE plan builds the fill, and its calls zero the output with it, no memset: back-to-back calls each
    write the reference bytes into an output poisoned before them, and so do the replays of a captured call."""
    from cudnn.frost import buffers
    from cudnn.gemm.frost.sm100.combine_zero import CombineZero

    plan, pack, out, _backing, reference = _exact_combine_case(swap=swap)
    assert isinstance(plan._compiled.combine_zero, CombineZero)
    monkeypatch.setattr(buffers, "apply_zero_fill_plan", _no_memset)
    workspace = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for i in range(3):
        with torch.cuda.stream(stream):
            out.fill_(float("nan"))
            plan._compiled(pack, workspace=workspace, stream=stream.cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(out, reference), f"eager call {i + 1}"
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            plan._compiled(pack, workspace=workspace, stream=stream.cuda_stream)
        for i in range(3):
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, reference), f"replay {i + 1}"
    finally:
        torch.cuda.synchronize()
        graph.reset()


_PAIR = _config(128, 128, "2x1", 2)


@requires_sm100
@pytest.mark.parametrize("captured", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("swap,config", [(False, None), (True, None), (False, _PAIR), (True, _PAIR)], ids=["fwd", "swap", "fwd_2cta", "swap_2cta"])
def test_general_grouped_combine_output_is_zeroed_behind_a_programmatic_predecessor(swap, config, captured):
    """Behind a predecessor that releases its dependents first and writes NaN over the whole COMBINE output after a spin,
    the call writes the reference bytes, eager or replayed from a CUDA graph, on one-CTA tiles and CTA pairs: the fill
    stores its zeros only once the predecessor has finished, and the kernel adds into the output only once the fill
    has."""
    from pdl_late_writer import late_write

    plan, pack, out, _backing, reference = _exact_combine_case(swap=swap, config=config)
    workspace = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda")
    words = out.view(torch.int32)
    poison = torch.full_like(out, float("nan")).view(torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def launch():
        late_write(poison, words, spin_us=2000, stream=stream.cuda_stream)
        plan._compiled(pack, workspace=workspace, stream=stream.cuda_stream)

    def run(step, what):
        out.zero_()
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            step()
        torch.cuda.synchronize()
        assert torch.equal(out, reference), what

    if not captured:
        for i in range(3):
            run(launch, f"eager launch {i + 1}")
        return
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            launch()
        for i in range(3):
            run(graph.replay, f"replay {i + 1}")
    finally:
        torch.cuda.synchronize()
        graph.reset()


@requires_sm100
@pytest.mark.parametrize("pad,memsets", [(0, 0), (4, 0), (1, 1)], ids=["dense", "pitch_260", "pitch_257"])
@pytest.mark.parametrize("swap", [False, True], ids=["fwd", "swap"])
def test_general_grouped_combine_fill_writes_only_the_output(swap, pad, memsets, monkeypatch):
    """An output that is a view of a buffer with ``pad`` more fp32 columns per row: the call writes the reference bytes
    and every padding column keeps its NaN. The fill kernel zeroes a dense output and rows 1040 bytes apart; rows 1028
    bytes apart are not 16-byte aligned, and the memset zeroes them."""
    from cudnn.frost import buffers

    plan, pack, out, backing, reference = _exact_combine_case(swap=swap, pad=pad)
    fills = []
    memset = buffers.apply_zero_fill_plan
    monkeypatch.setattr(buffers, "apply_zero_fill_plan", lambda *args: fills.append(args) or memset(*args))
    plan(pack)
    torch.cuda.synchronize()
    assert torch.equal(out, reference)
    assert torch.isnan(backing[:, :, out.shape[-1] :]).all()
    assert len(fills) == memsets


# The scheduler's copy of the routing offsets
#
# The scheduler warp of a grouped template copies first_token_offset into shared memory once per thread block and
# searches the groups there. The copy is sized when the plan is built, for the num_groups + 1 offsets its graph
# declares: a call that routes more groups is refused before anything is written, one that routes fewer runs. A plan
# whose offsets would take more than compiler._GENERAL_BS_SCHED_FTO_SMEM_MAX_BYTES searches them in global memory.


def test_general_grouped_plan_sizes_its_copy_of_the_routing_offsets(monkeypatch):
    """24 groups: 25 offsets in shared memory, counted in the plan's shared-memory reserve; none above the byte bound."""
    chain = analyze(_graph(8, 512, 256, 512, 24)[0])
    cfg = by_name(_TILE_128)
    plan = C._general_block_scale_plan(chain, cfg)
    assert plan["sched_fto_entries"] == 25
    assert "\nsched_fto_entries = 25\n" in C._render_general_block_scale_tile_constants(cfg, chain, select_template(chain, cfg))
    monkeypatch.setattr(C, "_GENERAL_BS_SCHED_FTO_SMEM_MAX_BYTES", 99)
    assert C._general_block_scale_plan(chain, cfg)["sched_fto_entries"] == 0


@requires_sm100
@pytest.mark.parametrize("gather,swap", list(_TEMPLATES), ids=["fwd", "swap", "gather", "gather_swap"])
def test_general_grouped_call_with_more_groups_than_its_plan_is_refused(gather, swap):
    """A plan built for 8 groups over 4 experts: a call that routes 4 groups writes the reference, one that routes 12
    is refused and leaves the output as it was."""
    E, R, N, K = 4, 512, 256, 256
    S = 300 if gather else R
    g, t = _graph(E, S, N, K, 2 * E, gather_rows=R if gather else None)
    plan = Plan(g, config=by_name(_config(128, 32) if swap else _TILE_128), swap_ab=swap)
    assert select_template(plan.chain, plan._compiled.config).file == _TEMPLATES[(gather, swap)]
    assert plan._compiled.sched_groups_max == 2 * E
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(0)
    tok = (torch.randn(1, S, K, device=dev, generator=gen) * 0.5).to(torch.float8_e4m3fn)
    sa = torch.rand(1, S, K // 128, device=dev, generator=gen) + 0.5
    w = (torch.randn(E, N, K, device=dev, generator=gen) * 0.5).to(torch.float8_e4m3fn)
    sb = torch.rand(E, N // 128, K // 128, device=dev, generator=gen) + 0.5
    idx = torch.randint(0, S, (1, R, 1), device=dev, generator=gen, dtype=torch.int32) if gather else None
    out = torch.empty(1, R, N, device=dev, dtype=torch.bfloat16)
    pack = {t["tok"]: tok, t["w"]: w.transpose(1, 2), t["sfa"]: sa, t["sfb"]: sb.transpose(1, 2), t["out"]: out}
    if gather:
        pack[t["idx"]] = idx

    def call(offsets):
        out.fill_(float("nan"))
        plan({**pack, t["fto"]: torch.tensor(offsets, dtype=torch.int32, device=dev).view(-1, 1, 1)})
        torch.cuda.synchronize()

    fewer = [0, 128, 256, 384, 448]
    call(fewer)
    tok_f = tok[0].float() * sa[0].repeat_interleave(128, 1)
    src = tok_f[idx[0, :, 0].long()] if gather else tok_f
    w_f = w.float() * sb.repeat_interleave(128, 1).repeat_interleave(128, 2)
    want = torch.cat([src[fewer[e] : fewer[e + 1]] @ w_f[e].T for e in range(E)])
    assert float((out[0, : fewer[-1]].float() - want).norm() / want.norm()) < 2e-2
    more = [0, 40, 80, 128, 170, 210, 256, 300, 340, 384, 400, 420, 448]
    with pytest.raises(ValueError, match="built for at most 8 routed groups"):
        call(more)
    assert torch.isnan(out.float()).all()


@requires_sm100
@pytest.mark.parametrize("swap", [False, True], ids=["fwd", "swap"])
def test_general_grouped_combine_with_more_groups_than_its_plan_is_refused_before_its_output_fill(swap):
    """A COMBINE launch zero-fills its output before the kernel adds into it: a call refused for its group count is
    refused before that fill, and the output keeps its bytes. (Tiles whose plan keeps the copy of the offsets: the
    1-CTA 128x128 COMBINE goes without it, _smem_carveout_crossed.)"""
    E, S, N, K = 4, 512, 256, 256
    g, t = _graph(E, S, N, K, 2 * E, combine=2)
    plan = Plan(g, config=by_name(_config(128, 32) if swap else _config(128, 128, "2x1", 2)), swap_ab=swap)
    assert plan._compiled.sched_groups_max == 2 * E
    dev = "cuda"
    dest = torch.randperm(S, device=dev).to(torch.int32)
    out = torch.full((1, S // 2, N), float("nan"), device=dev)
    pack = {
        t["tok"]: torch.zeros(1, S, K, device=dev).to(torch.float8_e4m3fn),
        t["w"]: torch.zeros(E, N, K, device=dev).to(torch.float8_e4m3fn).transpose(1, 2),
        t["sfa"]: torch.ones(1, S, K // 128, device=dev),
        t["sfb"]: torch.ones(E, N // 128, K // 128, device=dev).transpose(1, 2),
        t["fto"]: (torch.arange(3 * E + 1, dtype=torch.int32, device=dev) * (S // (3 * E))).view(-1, 1, 1),
        t["idx"]: (dest // 2).view(1, S, 1),
        t["ks"]: (dest % 2).view(1, S, 1),
        t["scores"]: torch.rand(1, S // 2, 2, device=dev),
        t["out"]: out,
    }
    with pytest.raises(ValueError, match="built for at most 8 routed groups"):
        plan(pack)
    torch.cuda.synchronize()
    assert torch.isnan(out).all()


@requires_sm100
@pytest.mark.parametrize("gather,swap", list(_TEMPLATES), ids=["fwd", "swap", "gather", "gather_swap"])
def test_general_grouped_plan_without_the_copy_searches_global_memory(gather, swap, monkeypatch):
    """A byte bound below the plan's offsets: no copy, no bound on the call's groups, and 64 ragged groups (two steps of
    the 32-group search) still write the reference."""
    monkeypatch.setattr(C, "_GENERAL_BS_SCHED_FTO_SMEM_MAX_BYTES", 64)
    kw = dict(S=300, gather_rows=512) if gather else dict()
    if swap:
        kw.update(swap=True, config=_config(128, 32))
    plan, *_ = _run(E=8, offsets=_offsets(512, 64), **kw)
    assert "\nsched_fto_entries = 0\n" in open(plan.generated_path).read()
    assert plan._compiled.sched_groups_max == 0


# Rows past a group's end
#
# fwd and fwd_swap_ab load a tile's token rows at absolute rows of the whole token tensor: past its group's end, a tile
# loads the next group's rows, or for the last group the unrouted rows after the last offset (zeros past the end of the
# tensor). No drain stores such a row, and with a token scale (every graph here) the promotion scales it by zero; an
# unrouted row may hold anything.

_UNROUTED = {
    "fwd": dict(),
    "fwd_2cta": dict(config=_config(128, 128, "2x1", 2)),
    "fwd_scatter": dict(scatter=2),
    "fwd_combine": dict(combine=2, tol=1e-5),
    "swap": dict(swap=True, config=_config(128, 32)),
    "swap_combine": dict(combine=2, swap=True, config=_config(128, 16), tol=1e-5),
}


@requires_sm100
@pytest.mark.parametrize("case", list(_UNROUTED))
def test_general_grouped_unrouted_token_rows_reach_no_output(case):
    """With the token rows after the last offset as they are and holding NaN codes: every routed row (or COMBINE sum)
    is the reference's and finite, and a NONE launch writes no row past the routed ones (the output starts as NaN, and
    a stored row past them would be the zero its scale makes of it)."""
    offsets = _offsets(512, 8)
    used = offsets[-1]
    assert used < 512

    def poison(tok):
        bad = tok.clone()
        bad[0, used:].view(torch.uint8).fill_(0x7F)  # e4m3fn NaN
        return bad

    for place in (None, poison):
        _, result, _, _ = _run(offsets=offsets, place_token=place, **_UNROUTED[case])
        if "scatter" not in case and "combine" not in case:
            assert torch.isnan(result[0, used:].float()).all()


# Out-of-range COMBINE routes
#
# A COMBINE adds nothing through a route whose token index is negative or not below the output's token count, or whose
# slot is negative or not below top-k. The token-by-weight drain checks a route per row (moe_combine_stage,
# moe_combine_bulk); the lane-parallel swapped drain checks the tile's routes as it loads them, a column per lane
# (tile_helpers.moe_combine_lane_routes); a swapped graph of one routed row per expert loads each column's route.

_OUT_OF_RANGE_ROUTES = {
    "fwd": dict(),
    "fwd_bf16": dict(out="bf16"),
    "swap_n16": dict(swap=True, config=_config(128, 16)),
    "swap_n32": dict(swap=True, config=_config(128, 32)),
    "swap_n96": dict(swap=True, config=_config(128, 96)),
    "swap_2cta_256x16": dict(swap=True, config=_config(256, 16, "2x1", 2)),
    "swap_2cta_256x32": dict(swap=True, config=_config(256, 32, "2x1", 2)),
    "swap_one_row_per_expert": dict(swap=True, config=_config(128, 16), E=16, S=16),
}


@requires_sm100
@pytest.mark.parametrize("case", list(_OUT_OF_RANGE_ROUTES))
def test_general_grouped_combine_ignores_out_of_range_routes(case):
    """Some routed rows take an out-of-range route -- token -1 or T, slot -1 or top-k, in turn -- and the others a
    permutation of the slots: the output is the reference without those routes, and the guard rows around it, a finite
    sentinel, are untouched (a reduction through a token index past the output would land there)."""
    kw = dict(_OUT_OF_RANGE_ROUTES[case])
    E, S, N, K, top_k = kw.get("E", 8), kw.get("S", 512), 256, 512, 2
    swap, out = kw.get("swap", False), kw.get("out", "fp32")
    g, t = _graph(E, S, N, K, E, combine=top_k, combine_out=out)
    plan = Plan(g, config=by_name(kw.get("config", _TILE_128)), swap_ab=swap)
    src = plan.generated_path.read_text()
    # Each case runs the drain it is named for.
    lanes, bulk = "\nswap_combine_lanes = True\n" in src, "\nmoe_combine_bulk_enabled = True\n" in src
    assert (lanes, bulk) == ((swap and E < S), not swap), (lanes, bulk)
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(3)
    tok = (torch.randn(1, S, K, device=dev, generator=gen) * 0.5).to(torch.float8_e4m3fn)
    w = (torch.randn(E, N, K, device=dev, generator=gen) * 0.5).to(torch.float8_e4m3fn)
    sa = torch.rand(1, S, K // 128, device=dev, generator=gen) + 0.5
    sb = torch.rand(E, N // 128, K // 128, device=dev, generator=gen) + 0.5
    offsets = _offsets(S, E)
    used, T = offsets[-1], S // top_k
    dest = torch.randperm(S, device=dev, generator=gen).to(torch.int32)
    index, slots = (dest // top_k).clone(), (dest % top_k).clone()
    bad = torch.arange(3, used, 7 if used > 64 else 3, device=dev)
    kinds = torch.arange(bad.numel(), device=dev) % 4
    assert set(kinds.tolist()) == {0, 1, 2, 3}
    index[bad[kinds == 0]] = -1
    index[bad[kinds == 1]] = T
    slots[bad[kinds == 2]] = -1
    slots[bad[kinds == 3]] = top_k
    scores = torch.rand(1, T, top_k, device=dev, generator=gen)
    guard, sentinel = 4, -123.0
    backing = torch.full((T + 2 * guard, N), sentinel, device=dev, dtype=_TORCH[out])
    result = backing[guard : guard + T].view(1, T, N)
    plan(
        {
            t["tok"]: tok,
            t["w"]: w.transpose(1, 2),
            t["sfa"]: sa,
            t["sfb"]: sb.transpose(1, 2),
            t["fto"]: torch.tensor(offsets, dtype=torch.int32, device=dev).view(-1, 1, 1),
            t["idx"]: index.view(1, S, 1),
            t["ks"]: slots.view(1, S, 1),
            t["scores"]: scores,
            t["out"]: result,
        }
    )
    torch.cuda.synchronize()
    tok_f = tok[0].float() * sa[0].repeat_interleave(128, 1)
    w_f = w.float() * sb.repeat_interleave(128, 1).repeat_interleave(128, 2)
    expected = torch.zeros(T, N, device=dev)
    for e in range(E):
        lo, hi = offsets[e], offsets[e + 1]
        if hi > lo:
            ids, ks = index[lo:hi].long(), slots[lo:hi].long()
            valid = (ids >= 0) & (ids < T) & (ks >= 0) & (ks < top_k)
            rows = tok_f[lo:hi] @ w_f[e].T
            expected.index_add_(0, ids[valid], rows[valid] * scores[0, ids[valid], ks[valid], None])
    assert bool((backing[:guard].float() == sentinel).all()) and bool((backing[guard + T :].float() == sentinel).all()), "a reduction landed in a guard row"
    got = result[0].float()
    assert bool(torch.isfinite(got).all())
    err = float((got - expected).norm() / expected.norm())
    assert err < (1e-2 if out == "bf16" else 1e-5), f"relative error {err:.3e}"


# Calls with nothing to compute
#
# A grouped call with no routed rows (an expert-parallel rank that received none) has no output to write: it returns
# before the launch, whose host would otherwise encode a TMA descriptor with a zero extent, which the DSL answers with a
# process abort. A GATHER whose token buffer has no rows has routed rows to write and nothing to read them from: it is
# refused before anything is written.

_EMPTY = {"fwd": dict(), "swap": dict(swap=True, config=_config(128, 32))}


def _no_launch(*args, **kwargs):
    pytest.fail("a call with nothing to compute reached the launch")


@requires_sm100
@pytest.mark.parametrize("case", list(_EMPTY))
def test_general_grouped_call_without_routed_rows_launches_nothing(case, monkeypatch):
    """Zero token rows, bound as zero-row views of live buffers: the call returns, and the buffers behind the views keep
    their bytes."""
    kw = _EMPTY[case]
    E, N, K = 8, 256, 512
    g, t = _graph(E, 512, N, K, E)
    plan = Plan(g, config=by_name(kw.get("config", _TILE_128)), swap_ab=kw.get("swap", False))
    monkeypatch.setattr(type(plan._compiled), "_launch_moe", _no_launch)
    dev = "cuda"
    tokens = torch.zeros(1, 8, K, device=dev).to(torch.float8_e4m3fn)
    backing = torch.full((1, 8, N), float("nan"), device=dev, dtype=torch.bfloat16)
    pack = {
        t["tok"]: tokens[:, :0],
        t["w"]: torch.ones(E, N, K, device=dev).to(torch.float8_e4m3fn).transpose(1, 2),
        t["sfa"]: torch.ones(1, 0, K // 128, device=dev),
        t["sfb"]: torch.ones(E, N // 128, K // 128, device=dev).transpose(1, 2),
        t["fto"]: torch.zeros(E + 1, 1, 1, dtype=torch.int32, device=dev),
        t["out"]: backing[:, :0],
    }
    assert plan(pack) is None
    torch.cuda.synchronize()
    assert torch.isnan(backing.float()).all()


@requires_sm100
@pytest.mark.parametrize("swap", [False, True], ids=["gather", "gather_swap"])
def test_general_grouped_gather_from_an_empty_token_buffer_is_refused(swap, monkeypatch):
    """Routed rows that index a token buffer without rows: refused, and the output keeps its bytes."""
    E, R, N, K = 8, 512, 256, 512
    g, t = _graph(E, 300, N, K, E, gather_rows=R)
    plan = Plan(g, config=by_name(_config(128, 32) if swap else _TILE_128), swap_ab=swap)
    monkeypatch.setattr(type(plan._compiled), "_launch_moe", _no_launch)
    dev = "cuda"
    out = torch.full((1, R, N), float("nan"), device=dev, dtype=torch.bfloat16)
    pack = {
        t["tok"]: torch.zeros(1, 8, K, device=dev).to(torch.float8_e4m3fn)[:, :0],
        t["w"]: torch.ones(E, N, K, device=dev).to(torch.float8_e4m3fn).transpose(1, 2),
        t["sfa"]: torch.ones(1, 0, K // 128, device=dev),
        t["sfb"]: torch.ones(E, N // 128, K // 128, device=dev).transpose(1, 2),
        t["fto"]: (torch.arange(E + 1, dtype=torch.int32, device=dev) * (R // E)).view(-1, 1, 1),
        t["idx"]: torch.zeros(1, R, 1, dtype=torch.int32, device=dev),
        t["out"]: out,
    }
    with pytest.raises(ValueError, match="routes rows into a token buffer that has none"):
        plan(pack)
    torch.cuda.synchronize()
    assert torch.isnan(out.float()).all()


# The GATHER handoff from the token copies to the MMA, and a pair's last cross-CTA arrives (generated PTX)


def _kernel_lines(ptx):
    """The kernel's instructions and labels, stripped, without comments."""
    body = ptx[ptx.index(".entry") :]
    body = body[: body.index("\n}\n") + 2]
    return [line.strip() for line in body.splitlines() if line.strip() and not line.strip().startswith("//")]


@requires_sm100
@pytest.mark.parametrize("cta", [1, 2], ids=["1cta", "2cta"])
@pytest.mark.parametrize("swap", [False, True], ids=["gather", "gather_swap"])
def test_general_gather_relay_fences_each_copied_stage_for_the_mma(swap, cta, monkeypatch, tmp_path):
    """The token-copy warps fill a stage with cp.async, which writes through the generic proxy. tcgen05.mma reads it
    through the async proxy, and the mbarrier completion that reports the copies orders the generic proxy only: the PTX
    ISA asks for fence.proxy.async between them. In either MMA mode the copy warps report to the relay warp, which
    waits, fences and then arrives on the stage's ab_full, the barrier the MMA warp waits on (the MMA warp is
    unchanged). Per-thread output stores keep the epilogue's TMA-store fences out of the kernel: the relay's is its one."""
    keep_kernel_ptx(monkeypatch, C, tmp_path)
    config = _config(128, 128, "2x1", 2) if cta == 2 else (_config(128, 16) if swap else _TILE_128)
    plan, *_ = _exact_case(gather=True, swap=swap, config=config, stg=True)
    lines = _kernel_lines(kernel_ptx(plan))
    fences = [i for i, line in enumerate(lines) if line.startswith("fence.proxy.async")]
    assert len(fences) == 1, f"{len(fences)} proxy fences in the kernel"
    waited = next(line for line in reversed(lines[: fences[0]]) if "mbarrier." in line)
    arrived = next(line for line in lines[fences[0] + 1 :] if "mbarrier." in line)
    assert waited.startswith("mbarrier.try_wait"), waited
    assert arrived.lstrip("@!%p0123456789 ").startswith("mbarrier.arrive"), arrived


@requires_sm100
@pytest.mark.parametrize(
    "gather,swap,cta",
    [(False, False, 1), *((gather, swap, 2) for gather, swap in _TEMPLATES)],
    ids=["fwd_1cta", "fwd_2cta", "swap_2cta", "gather_2cta", "gather_swap_2cta"],
)
def test_general_grouped_pairs_end_on_a_cluster_barrier(gather, swap, cta, monkeypatch, tmp_path):
    """A pair's follower releases the leader's partial accumulators (and its epilogue the leader's accumulator stages)
    with relaxed remote arrives, which nothing else orders before the leader thread block exits and its shared memory
    goes to the next one. A paired kernel ends on barrier.cluster.arrive (release) and barrier.cluster.wait (acquire),
    which every thread reaches after its role; a 1-CTA kernel has no such barrier."""
    keep_kernel_ptx(monkeypatch, C, tmp_path)
    plan, *_ = _exact_case(gather=gather, swap=swap, config=_config(128, 128, "2x1", 2) if cta == 2 else _TILE_128)
    lines = _kernel_lines(kernel_ptx(plan))
    end = len(lines) - 1 - lines[::-1].index("ret;")
    assert (lines[end - 2 : end] == ["barrier.cluster.arrive;", "barrier.cluster.wait;"]) is (cta == 2), lines[end - 3 : end + 1]


# A thread block's tensor-map rewrites, and the epilogue's last bulk stores (generated PTX, source)
#
# A thread block rewrites its own copy of a tensor map in global memory with tensormap.replace: the output map of a
# TMA-store epilogue, on a group change. (The token loads of every template go through the host's maps.) The order is
# CUTLASS's: an acquire of the old map behind a drain of the bulk stores still reading it, the rewrite, a release, and
# an acquire of the rewritten map before the next bulk copy through it. Without that last acquire a copy can fetch the
# previous group's map: wrong rows, silently, at a rate no numerics or warm-stress cell shows (the same order on the
# token map the sorted templates rewrote before wrote wrong rows in none of about 159k launches as built, and in 70 of
# 9000 launches of fwd behind 16 queued copies).

_TENSORMAP_ACQUIRE = re.compile(r"fence\.proxy\.tensormap::generic\.acquire\.\w+ \[(%rd\d+)\], 128;")


def _bulk_copy_map(line):
    """(the tensor map register, whether it stores) of a cp.async.bulk.tensor instruction: a store names its map first,
    a load second (after its shared-memory destination)."""
    store = ".global.shared::" in line.split()[0]
    return re.findall(r"\[([^\]]*)\]", line)[0 if store else 1].split(",")[0].strip(), store


@requires_sm100
@pytest.mark.parametrize("gather,swap", list(_TEMPLATES), ids=["fwd", "swap", "gather", "gather_swap"])
def test_general_grouped_tensor_map_rewrites_are_acquired_after_their_release(gather, swap, monkeypatch, tmp_path):
    """Every release of a rewritten map is followed, before any other tensor-map or bulk tensor instruction, by an
    acquire of that map; the rewrite opens with an acquire of the same map, behind cp.async.bulk.wait_group.read 0 for
    an output map. Every template rewrites its output map and no other: fwd and fwd_swap_ab load their token rows
    through the host's map, the GATHER templates copy theirs."""
    keep_kernel_ptx(monkeypatch, C, tmp_path)
    plan, *_ = _exact_case(gather=gather, swap=swap, config=_config(128, 32) if swap else _TILE_128)
    lines = [" ".join(re.sub(r"^@!?%p\d+\s+", "", line).split()) for line in _kernel_lines(kernel_ptx(plan))]
    ops = [line for line in lines if line.startswith(("fence.proxy.tensormap", "tensormap.replace", "cp.async.bulk.tensor", "cp.async.bulk.wait_group"))]
    copies = {}
    for line in ops:
        if line.startswith("cp.async.bulk.tensor"):
            reg, store = _bulk_copy_map(line)
            copies.setdefault(reg, set()).add("store" if store else "load")
    releases = [i for i, line in enumerate(ops) if line.startswith("fence.proxy.tensormap::generic.release")]
    assert len(releases) == 1, ops
    rewritten = []
    for i in releases:
        after = _TENSORMAP_ACQUIRE.fullmatch(ops[i + 1])
        assert after is not None, f"the release is followed by {ops[i + 1]!r}"
        j = i - 1
        while ops[j].startswith("tensormap.replace"):
            j -= 1
        assert j < i - 1, "a release without a rewrite in front of it"
        before = _TENSORMAP_ACQUIRE.fullmatch(ops[j])
        assert before is not None and before.group(1) == after.group(1), f"the rewrite of {after.group(1)} opens with {ops[j]!r}"
        kinds = copies.get(after.group(1), set())
        assert len(kinds) == 1, f"{after.group(1)} is used by {kinds or 'no'} bulk copies"
        if kinds == {"store"}:
            assert ops[j - 1] == "cp.async.bulk.wait_group.read 0;", f"the output map's rewrite follows {ops[j - 1]!r}"
        rewritten.append(next(iter(kinds)))
    assert rewritten == ["store"]


def _epilogue_exit(source):
    """The statements of the epilogue warps' branch (``if warp_idx < num_epilogue_warps``) after its tile loop."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "warp_idx < num_epilogue_warps":
            loop = next(i for i, stmt in enumerate(node.body) if isinstance(stmt, ast.While) and ast.unparse(stmt.test) == "is_valid != 0")
            return node.body[loop + 1 :]
    raise AssertionError("no epilogue branch")


@pytest.mark.parametrize("gather,swap", list(_TEMPLATES), ids=["fwd", "swap", "gather", "gather_swap"])
def test_general_grouped_epilogue_waits_for_its_last_bulk_stores_to_read_before_it_exits(gather, swap):
    """The bulk stores of a TMA-store epilogue read the staging ring after they are issued. After its tile loop, warp 0
    waits until the last ones are done reading (cp.async.bulk.wait_group.read 0): the thread block does not exit, and
    hand its shared memory to the next one (a programmatic dependent's among them), while they still read it."""
    tail = _epilogue_exit(template_path(_TEMPLATES[(gather, swap)]).read_text())
    drains = [
        stmt
        for stmt in tail
        if isinstance(stmt, ast.If)
        and ast.unparse(stmt.test) == "warp_idx == 0"
        and [ast.unparse(s) for s in stmt.body] == ["nvvm.cp_async_bulk_wait_group(0, read=True)"]
    ]
    assert len(drains) == 1, [ast.unparse(stmt) for stmt in tail]
