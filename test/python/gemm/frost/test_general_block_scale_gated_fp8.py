# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gated pair (SwiGLU) and FP8 block-128 output of the five general block-scale templates.

``sm100_general_block_scale_matmul.py`` and the four ``sm100_general_moe_grouped_block_scale_matmul_fwd*.py`` run
one plain MMA per tile pass. Two GEMMs that share the token operand (a gated FC1: ``swish(tokens @ gate) * (tokens @
up)``) run as two passes of every tile, and the epilogue drains the two main accumulator stages together. An fp32
scale per 128 output columns of a row (FP8 block-128, the input format of the next layer) is a quantize block wider
than an epilogue vector: the epilogue takes the block's maximum in a first sweep of the tile and quantizes in a second.

Registry and plan gates (no GPU), then end-to-end numerics against an fp32 dequantize-then-matmul reference, replay
and batch-invariance byte comparisons."""

from __future__ import annotations

import ast
import dataclasses
import re

import cudnn
import cudnn.gemm.frost  # noqa: F401  (installs recorder)
import pytest
import torch

import warm_stress
from gemm_test_utils import Plan, keep_kernel_ptx, kernel_ptx, requires_sm100, requires_sm100_tree
from cudnn.gemm.frost import compiler as C
from cudnn.gemm.frost import epilogue_codegen as EC
from cudnn.gemm.frost.fusion_ir import BlockQuantizeSpec, FusionOp, wide_quant
from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.gemm.frost.kernel_registry import general_glu_pair, general_glu_pair_reject, select_template, swap_ab_graph, template_path
from cudnn.gemm.frost.tile_config import by_name

# Every test here concerns the sm100 tree's general templates and compiler (several call names only that compiler has):
# a process that runs another family's tree skips them.
pytestmark = [pytest.mark.L0, requires_sm100_tree]

DT = cudnn.data_type
_BLOCK = 128
_FP8_MAX = 448.0
_STEM = "sm100_general_moe_grouped_block_scale_matmul_fwd"
# (kind, weight-by-token) -> template
_TEMPLATES = {
    ("dense", False): "sm100_general_block_scale_matmul.py",
    ("moe", False): f"{_STEM}.py",
    ("moe", True): f"{_STEM}_swap_ab.py",
    ("gather", False): f"{_STEM}_gather_mode.py",
    ("gather", True): f"{_STEM}_gather_mode_swap_ab.py",
}


def _config(n, cluster="1x1", cta=1):
    return f"CONFIG_sm100_128x{n}x128_128x{n}x32_cluster{cluster}_{cta}ctamma"


_TILE_128 = _config(128)


def _graph(kind, E, S, N, K, *, rows=None, gemms=2, fp8=False, mode=None, quant_block=_BLOCK, scale_dtype=DT.FLOAT):
    """``swish(tokens @ w0) * (tokens @ w1)`` (``gemms=2``) or ``tokens @ w0`` over FP8 block-128 operands.

    ``kind`` is dense | moe | gather; ``S`` is the token tensor's row count and, under GATHER, ``rows`` routed rows
    index into it. ``fp8`` quantizes the result to FP8 e4m3 with one scale per ``quant_block`` columns of a row."""
    g = cudnn.pygraph(io_data_type=DT.BFLOAT16, intermediate_data_type=DT.FLOAT, compute_data_type=DT.FLOAT)
    lead = 1 if kind == "dense" else E
    kb, nb = K // _BLOCK, N // _BLOCK
    t = {"w": [], "sfb": []}
    t["tok"] = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=DT.FP8_E4M3)
    t["sfa"] = g.tensor(name="SFA", dim=[1, S, kb], stride=[S * kb, kb, 1], data_type=DT.FLOAT)
    tok_d = g.block_scale_dequantize(input=t["tok"], descale=t["sfa"], block_size=[1, _BLOCK])
    weights = []
    for i in range(gemms):
        w = g.tensor(name=f"w{i}", dim=[lead, K, N], stride=[K * N, 1, K], data_type=DT.FP8_E4M3)
        sf = g.tensor(name=f"SFB{i}", dim=[lead, kb, nb], stride=[kb * nb, 1, kb], data_type=DT.FLOAT)
        t["w"].append(w)
        t["sfb"].append(sf)
        weights.append(g.block_scale_dequantize(input=w, descale=sf, block_size=[_BLOCK, _BLOCK]))
    if kind == "dense":
        c = [g.matmul(A=tok_d, B=weights[i], name=f"mm{i}") for i in range(gemms)]
    else:
        t["fto"] = g.tensor(name="first_token_offset", dim=[E + 1, 1, 1], stride=[1, 1, 1], data_type=DT.INT32)
        routing, moe_mode = {}, cudnn.moe_grouped_matmul_mode.NONE
        routed = rows if kind == "gather" else S
        if kind == "gather":
            t["idx"] = g.tensor(name="token_index", dim=[1, routed, 1], stride=[routed, 1, 1], data_type=DT.INT32)
            routing, moe_mode = {"token_index": t["idx"]}, cudnn.moe_grouped_matmul_mode.GATHER
        if mode == "scatter":
            t["idx"] = g.tensor(name="token_index", dim=[1, S, 1], stride=[S, 1, 1], data_type=DT.INT32)
            t["ks"] = g.tensor(name="token_ks", dim=[1, S, 1], stride=[S, 1, 1], data_type=DT.INT32)
            routing, moe_mode = {"token_index": t["idx"], "token_ks": t["ks"], "top_k": 2}, cudnn.moe_grouped_matmul_mode.SCATTER
        c = [g.moe_grouped_matmul(tok_d, weights[i], t["fto"], mode=moe_mode, compute_data_type=DT.FLOAT, name=f"moe{i}", **routing) for i in range(gemms)]
    y = c[0] if gemms == 1 else g.mul(a=g.swish(input=c[0], name="silu"), b=c[1], name="mul")
    if fp8:
        t["q"], t["qs"] = g.block_scale_quantize(input=y, block_size=quant_block, name="q")
        t["q"].set_output(True).set_data_type(DT.FP8_E4M3)
        t["qs"].set_output(True).set_data_type(scale_dtype)
    else:
        y.set_data_type(DT.BFLOAT16).set_output(True)
        t["out"] = y
    return g, t


def _offsets(rows, groups, seed=1):
    """Ragged groups that stop short of the capacity, with one empty group."""
    used = rows - rows // 8
    cuts = sorted(torch.randint(0, used, (groups - 1,), generator=torch.Generator().manual_seed(seed)).tolist())
    offsets = [0, *cuts, used]
    if groups > 2:
        offsets[2] = offsets[1]
    return offsets


def _e4m3_step(x):
    """Spacing of e4m3 values at |x| (3 mantissa bits; subnormals below 2**-6 share the smallest step)."""
    exp = torch.floor(torch.log2(x.abs().clamp_min(2.0**-6)))
    return torch.exp2(exp - 3)


@dataclasses.dataclass
class _Result:
    data: torch.Tensor  # the output buffer: bf16 [1, rows, N] or FP8 bytes [1, rows, N]
    scale: torch.Tensor | None  # fp32 [1, rows, N / 128] under fp8
    ref: torch.Tensor  # fp32 [rows, N]
    used: int


def _run(kind="moe", E=6, S=512, N=256, K=512, *, rows=None, gemms=2, fp8=False, swap=False, config=_TILE_128, offsets=None, seed=0, tok_seed=None, stg=False):
    """Run one product through its general template and check it against the fp32 reference."""
    routed = S if kind != "gather" else rows
    lead = 1 if kind == "dense" else E
    g, t = _graph(kind, E, S, N, K, rows=rows, gemms=gemms, fp8=fp8)
    plan = Plan(g, config=by_name(config), swap_ab=swap, force_stg_epi=stg)
    assert select_template(plan.chain, plan._compiled.config).file == _TEMPLATES[(kind, swap)]

    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(seed)
    tok_gen = gen if tok_seed is None else torch.Generator(device=dev).manual_seed(tok_seed)
    kb, nb = K // _BLOCK, N // _BLOCK
    tok = (torch.randn(1, S, K, device=dev, generator=tok_gen) * 0.5).to(torch.float8_e4m3fn)
    sa = torch.rand(1, S, kb, device=dev, generator=tok_gen) * 1.5 + 0.25
    tok_f = tok[0].float() * sa[0].repeat_interleave(_BLOCK, 1)
    ws, sbs, w_f = [], [], []
    for _ in range(gemms):
        w = (torch.randn(lead, N, K, device=dev, generator=gen) * 0.5).to(torch.float8_e4m3fn)
        sb = (torch.rand(lead, nb, kb, device=dev, generator=gen) * 1.5 + 0.25) * (torch.randint(0, 2, (lead, nb, kb), device=dev, generator=gen) * 2 - 1)
        ws.append(w)
        sbs.append(sb)
        w_f.append(w.float() * sb.repeat_interleave(_BLOCK, 1).repeat_interleave(_BLOCK, 2))

    def product(src, e):
        first = src @ w_f[0][e].T
        return first if gemms == 1 else torch.nn.functional.silu(first) * (src @ w_f[1][e].T)

    pack = {t["tok"]: tok, t["sfa"]: sa}
    for i in range(gemms):
        pack[t["w"][i]] = ws[i].transpose(1, 2)
        pack[t["sfb"][i]] = sbs[i].transpose(1, 2)  # the graph tensor is the [lead, k blocks, n blocks] transpose
    if kind == "dense":
        ref, used = product(tok_f, 0), S
    else:
        offsets = _offsets(routed, E) if offsets is None else offsets
        used = offsets[-1]
        pack[t["fto"]] = torch.tensor(offsets, dtype=torch.int32, device=dev).view(-1, 1, 1)
        idx = None
        if kind == "gather":
            idx = torch.randint(0, S, (1, routed, 1), device=dev, generator=gen, dtype=torch.int32)
            pack[t["idx"]] = idx
        ref = torch.zeros(routed, N, device=dev)
        for gi in range(len(offsets) - 1):
            lo, hi = offsets[gi], offsets[gi + 1]
            if hi > lo:
                ref[lo:hi] = product(tok_f[idx[0, lo:hi, 0].long()] if idx is not None else tok_f[lo:hi], gi % E)

    if not fp8:
        out = torch.full((1, routed, N), float("nan"), device=dev, dtype=torch.bfloat16)
        pack[t["out"]] = out
        plan(pack)
        torch.cuda.synchronize()
        got = out[0, :used].float()
        assert torch.isfinite(got).all()
        assert torch.isnan(out[0, used:].float()).all(), "a row past the routed rows was written"
        err = float((got - ref[:used]).norm() / ref[:used].norm())
        assert err < 5e-3, f"relative error {err:.3e}"
        return _Result(out, None, ref, used)

    # 0x7F is an e4m3 NaN and the scale starts as NaN: an unwritten byte or scale shows.
    q = torch.full((1, routed, N), 0x7F, device=dev, dtype=torch.uint8)
    qs = torch.full((1, routed, nb), float("nan"), device=dev)
    pack[t["q"]], pack[t["qs"]] = q.view(torch.float8_e4m3fn), qs
    plan(pack)
    torch.cuda.synchronize()
    assert bool((q[0, used:] == 0x7F).all()) and bool(torch.isnan(qs[0, used:]).all()), "a row past the routed rows was written"
    scale, codes = qs[0, :used], q[0, :used].view(torch.float8_e4m3fn).float()
    assert torch.isfinite(scale).all() and torch.isfinite(codes).all()
    # The scale is the block's maximum over the largest e4m3 magnitude ...
    want_scale = ref[:used].view(used, nb, _BLOCK).abs().amax(dim=2) / _FP8_MAX
    assert torch.allclose(scale, want_scale, rtol=1e-4, atol=0.0), f"scale off by up to {float(((scale - want_scale).abs() / want_scale).max()):.2e}"
    # ... and every value is the reference over its scale, rounded to e4m3: at most one e4m3 step away, and almost
    # always the same code (the kernel multiplies by an approximate reciprocal and sums in another order).
    per_elem = scale.repeat_interleave(_BLOCK, dim=1)
    want = (ref[:used] / per_elem).to(torch.float8_e4m3fn).float()
    diff = (codes - want).abs()
    assert bool((diff <= _e4m3_step(torch.maximum(codes.abs(), want.abs()))).all()), f"a code is {float(diff.max())} away from the reference's"
    assert float((diff == 0).float().mean()) > 0.995
    err = float((codes * per_elem - ref[:used]).norm() / ref[:used].norm())
    assert err < 4e-2, f"relative error {err:.3e}"
    return _Result(q, qs, ref, used)


# Registry and plan (no GPU)


@pytest.mark.parametrize("kind,swap", list(_TEMPLATES))
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
def test_a_gated_pair_selects_each_general_template(kind, swap, fp8):
    chain = analyze(_graph(kind, 6, 300 if kind == "gather" else 512, 256, 512, rows=512, fp8=fp8)[0])
    config = by_name(_TILE_128)
    if swap:
        chain, config = swap_ab_graph(chain), dataclasses.replace(config, swap_ab=True)
    assert chain.num_gemms == 2 and general_glu_pair(chain)
    tmpl = select_template(chain, config)
    assert tmpl.file == _TEMPLATES[(kind, swap)]
    assert tmpl.accepts(chain, config) is None
    plan = C._general_block_scale_plan(chain, config)
    assert plan["glu_pair"] is True and plan["acc_stages"] == 2
    assert plan["wide_quant"] is fp8
    # A block of output columns lies along the lanes once the weights are the A operand.
    assert plan["wide_quant_lanes"] is (fp8 and swap)
    # The pipeline plan is the single GEMM's with the same output: the pair changes what a pass loads, not how the
    # MMA and the promotion run it.
    single = analyze(_graph(kind, 6, 300 if kind == "gather" else 512, 256, 512, rows=512, gemms=1, fp8=fp8)[0])
    if swap:
        single = swap_ab_graph(single)
    base = C._general_block_scale_plan(single, config)
    for key in (
        "promo_groups",
        "promo_cols",
        "promo_in_registers",
        "promo_scales_ahead",
        "promo_scale_table",
        "partial_stages",
        "ab_stages",
        "promo_regs",
        "epi_regs",
    ):
        assert plan[key] == base[key], key


def test_a_gated_pair_needs_one_token_operand_and_two_weights():
    chain = analyze(_graph("moe", 6, 512, 256, 512)[0])
    assert general_glu_pair_reject(chain) is None
    assert "share the token operand" in general_glu_pair_reject(dataclasses.replace(chain, num_a_operands=2))
    scatter = analyze(_graph("moe", 6, 512, 256, 512, mode="scatter")[0])
    assert "NONE or GATHER" in general_glu_pair_reject(scatter)


@pytest.mark.parametrize(
    "config,match",
    [
        # One main accumulator stage per GEMM: a 256-column accumulator leaves one.
        (_config(256), "need a main accumulator stage each"),
        (_config(128, "2x1", 2), "1-CTA MMA tiles"),
    ],
)
def test_a_gated_pair_rejects_tiles_without_two_accumulator_stages(config, match):
    chain = analyze(_graph("moe", 6, 512, 512, 512)[0])
    with pytest.raises(NotImplementedError, match=match):
        C._general_block_scale_plan(chain, by_name(config))


@pytest.mark.parametrize("swap,config", [(False, _config(64)), (False, _config(256))], ids=["n64", "n256"])
def test_a_wide_quantize_needs_a_tile_of_one_block(swap, config):
    chain = analyze(_graph("moe", 6, 512, 512, 512, gemms=1, fp8=True)[0])
    with pytest.raises(NotImplementedError, match="exactly one block along the quantized axis"):
        C._general_block_scale_plan(chain, by_name(config))


def test_a_wide_quantize_takes_any_token_tile_once_swapped():
    chain = swap_ab_graph(analyze(_graph("moe", 6, 512, 256, 512, fp8=True)[0]))
    for n in (16, 32, 48, 64, 128):
        plan = C._general_block_scale_plan(chain, dataclasses.replace(by_name(_config(n)), swap_ab=True))
        assert plan["wide_quant_lanes"] is True


def test_an_fp32_scale_is_the_wide_quantizer():
    chain = analyze(_graph("moe", 6, 512, 256, 512, fp8=True)[0])
    quant = wide_quant(chain)
    assert quant is not None and (quant.block_size, quant.scale_dtype, quant.scale_dim) == (_BLOCK, "fp32", (1, 512, 2))
    assert wide_quant(analyze(_graph("moe", 6, 512, 256, 512)[0])) is None
    with pytest.raises(ValueError, match="no scale reordering"):
        BlockQuantizeSpec(source_ref=0, block_size=_BLOCK, scale_dtype="fp32", scale_reorder="F8_128x4")


def test_a_row_block_of_128_is_the_same_quantizer():
    """``block_size=[1, 128]`` spells the block as block_scale_dequantize does."""
    chain = analyze(_graph("moe", 6, 512, 256, 512, fp8=True, quant_block=[1, _BLOCK])[0])
    assert wide_quant(chain).block_size == _BLOCK


def test_an_fp32_scale_covers_128_values():
    g, _ = _graph("moe", 6, 512, 256, 512, fp8=True, quant_block=32)
    with pytest.raises(NotImplementedError, match="covers 128 values"):
        C.probe_supported(g, by_name(_TILE_128))


def test_an_fp32_scale_goes_with_e4m3_data():
    g, t = _graph("moe", 6, 512, 256, 512, fp8=True)
    t["q"].set_data_type(DT.FP8_E5M2)
    with pytest.raises(NotImplementedError, match="FP8 E4M3 data"):
        C.probe_supported(g, by_name(_TILE_128))


def test_an_fp32_scale_is_rejected_outside_the_general_templates():
    """A graph without linear input scales lowers to templates that drain a tile in one sweep."""
    g = cudnn.pygraph(io_data_type=DT.BFLOAT16, intermediate_data_type=DT.FLOAT, compute_data_type=DT.FLOAT)
    a = g.tensor(name="A", dim=[1, 256, 512], stride=[256 * 512, 512, 1], data_type=DT.BFLOAT16)
    b = g.tensor(name="B", dim=[1, 512, 256], stride=[512 * 256, 1, 512], data_type=DT.BFLOAT16)
    q, qs = g.block_scale_quantize(input=g.matmul(A=a, B=b, name="mm"), block_size=_BLOCK, name="q")
    q.set_output(True).set_data_type(DT.FP8_E4M3)
    qs.set_output(True).set_data_type(DT.FLOAT)
    with pytest.raises(NotImplementedError, match="general block-scale templates only"):
        C.probe_supported(g, by_name(_TILE_128))


def test_the_routing_offsets_copy_keeps_the_kernel_in_its_carveout():
    """The scheduler's shared-memory copy of the routing offsets takes only room the A/B stages leave, and not where it
    may move the kernel to a larger shared-memory carveout: a GATHER FC1 of 512 experts (K = 2048, N = 512) in 128x128
    tiles fills six 32 KB stages to just under the 196 KB carveout (198704 bytes measured) and goes without the copy,
    in 128x16 weight-by-token tiles it takes the copy."""
    g, _ = _graph("gather", 512, 4096, 512, 2048, rows=4096, fp8=True)
    chain = analyze(g)
    plan = C._general_block_scale_plan(chain, by_name(_config(128, "1x2")))
    assert (plan["ab_stages"], plan["sched_fto_entries"]) == (6, 0)
    swap_cfg = dataclasses.replace(by_name(_config(16)), swap_ab=True)
    assert C._general_block_scale_plan(swap_ab_graph(chain), swap_cfg)["sched_fto_entries"] == 513
    # The plan's shared memory with a 4 KB header reserve: the kernel's own lies up to 3 KB below it. The 196 KB carveout
    # holds 199680 bytes of a thread block's (the system reserves 1 KB of it).
    assert C._smem_carveout_crossed(200704, 2176, 4096)  # the case above
    assert C._smem_carveout_crossed(201000, 128, 4096)  # the kernel may still be under the edge, and pass it
    assert not C._smem_carveout_crossed(199000, 128, 4096)  # under the edge even at the plan's count
    assert not C._smem_carveout_crossed(225280, 2176, 4096)  # in the 228 KB carveout either way
    assert not C._smem_carveout_crossed(150000, 2048, 4096)


# End-to-end numerics (GPU)


_GATED = {
    "dense": dict(kind="dense", S=300, N=384, K=896),
    "dense_cluster1x4": dict(kind="dense", S=777, N=512, K=1024, config=_config(128, "1x4")),
    "fwd": dict(),
    "fwd_cluster1x4": dict(E=6, S=3001, N=384, K=896, config=_config(128, "1x4")),
    "swap": dict(swap=True),
    "swap_n16": dict(E=6, S=3001, N=384, K=896, swap=True, config=_config(16)),
    "swap_n48": dict(swap=True, config=_config(48)),
    "gather": dict(kind="gather", S=300, rows=512),
    "gather_cluster1x2": dict(kind="gather", E=6, S=750, rows=3001, N=384, K=896, config=_config(128, "1x2")),
    "gather_swap": dict(kind="gather", S=300, rows=512, swap=True),
    "gather_swap_n32": dict(kind="gather", E=6, S=750, rows=3001, N=384, K=896, swap=True, config=_config(32)),
    # every group one row: most of every tile is past its group's end
    "fwd_one_row_groups": dict(E=8, S=64, offsets=[0, 1, 2, 3, 4, 5, 6, 7, 8]),
    "gather_swap_one_row_groups": dict(kind="gather", E=8, S=4, rows=64, offsets=[0, 1, 2, 3, 4, 5, 6, 7, 8], swap=True, config=_config(16)),
    # swapped, two tiles per cluster along the weight's output columns: N = 384 puts the last cluster's second tile past N
    "swap_cluster2x1": dict(E=6, S=3001, N=384, K=896, swap=True, config=_config(32, "2x1")),
    "gather_swap_cluster2x1": dict(kind="gather", E=6, S=750, rows=3001, N=384, K=896, swap=True, config=_config(32, "2x1")),
}
# A 64-column tile holds half a quantize block.
_GATED_BF16_ONLY = {"fwd_n64": dict(config=_config(64)), "gather_n64": dict(kind="gather", S=300, rows=512, config=_config(64))}


@requires_sm100
@pytest.mark.parametrize("case", [*_GATED, *_GATED_BF16_ONLY])
def test_gated_pair_numerics(case):
    _run(**{**_GATED, **_GATED_BF16_ONLY}[case])


@requires_sm100
@pytest.mark.parametrize("case", ["dense", "fwd", "swap", "gather", "gather_swap"])
def test_gated_pair_numerics_through_the_stg_store(case):
    """The bf16 output of these shapes goes out by TMA; the same drain with per-thread stores."""
    _run(stg=True, **_GATED[case])


@requires_sm100
@pytest.mark.parametrize("case", list(_GATED))
def test_gated_pair_fp8_block128_numerics(case):
    _run(fp8=True, **_GATED[case])


@requires_sm100
@pytest.mark.parametrize("case", ["dense", "fwd", "swap_n16", "swap_cluster2x1", "gather_cluster1x2", "gather_swap_n32"])
def test_one_gemm_fp8_block128_numerics(case):
    _run(fp8=True, gemms=1, **_GATED[case])


@requires_sm100
@pytest.mark.parametrize(
    "cluster,N",
    [("2x1", 384), ("2x1", 512), ("4x1", 640), ("4x1", 512)],
    ids=["2x1_partial", "2x1_whole", "4x1_partial", "4x1_whole"],
)
@pytest.mark.parametrize("kind", ["moe", "gather"])
def test_a_cluster_along_the_lanes_writes_the_bytes_of_one_tile_clusters(kind, cluster, N):
    """Swapped, a cluster of two (four) tiles along the weight's output columns covers 256 (512) of them. With N = 384
    (640) the last cluster's second (second to fourth) tile lies past N: its lanes are no output column, and it stores
    neither codes nor scales. A scale it stored would be block N / 128 or more of a token, which in the [tokens,
    N / 128] scale tensor is the next token's block 0 onward. Codes and scales are the bytes of the plan with one tile
    per cluster; N = 512 (whole clusters) is the control."""
    kw = dict(kind=kind, E=6, S=750 if kind == "gather" else 3001, rows=3001 if kind == "gather" else None, N=N, K=896, swap=True, fp8=True)
    one = _run(config=_config(32), **kw)
    many = _run(config=_config(32, cluster), **kw)
    assert torch.equal(many.data.view(torch.uint8), one.data.view(torch.uint8))
    # The rows past the routed ones are NaN in both: compared as bits.
    assert torch.equal(many.scale.view(torch.int32), one.scale.view(torch.int32))


# Replay and batch invariance: raw bytes

_REPLAY = {"fwd": dict(), "swap": dict(swap=True, config=_config(32)), "gather": dict(kind="gather", S=300, rows=512), "dense": dict(kind="dense", S=300)}


@requires_sm100
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize("case", list(_REPLAY))
def test_gated_pair_repeats_bitwise(case, fp8):
    first = _run(fp8=fp8, **_REPLAY[case])
    second = _run(fp8=fp8, **_REPLAY[case])
    assert torch.equal(first.data.view(torch.uint8), second.data.view(torch.uint8))
    if fp8:
        assert torch.equal(first.scale[0, : first.used].view(torch.int32), second.scale[0, : second.used].view(torch.int32))


@requires_sm100
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize("swap", [False, True], ids=["fwd", "swap"])
def test_gated_pair_rows_do_not_depend_on_their_group_neighbors(swap, fp8):
    """A routed row's result -- and, under FP8 output, its scales -- is the same bits whatever shares its group and tile."""
    kw = dict(E=4, S=256, N=256, K=512, swap=swap, config=_config(32) if swap else _TILE_128, fp8=fp8)
    full = _run(offsets=[0, 100, 100, 180, 256], **kw)
    # the same tokens, with only the first four routed (expert 0)
    alone = _run(offsets=[0, 4, 4, 4, 4], **kw)
    assert torch.equal(full.data[0, :4].view(torch.uint8), alone.data[0, :4].view(torch.uint8))
    if fp8:
        assert torch.equal(full.scale[0, :4].view(torch.int32), alone.scale[0, :4].view(torch.int32))


# The scheduler counter lives in the caller's scratch workspace


@requires_sm100
@pytest.mark.parametrize("swap", [False, True], ids=["gather", "gather_swap"])
def test_grouped_launches_need_no_cleared_workspace(swap):
    """A general grouped kernel starts its tile count from a launch epoch: the workspace may hold anything, the last
    launch's count and another plan's included."""
    kw = dict(kind="gather", S=300, rows=512, swap=swap, config=_config(32) if swap else _TILE_128)
    first = _run(fp8=True, **kw)
    plans = []
    for fp8 in (True, False):
        g, t = _graph("gather", 6, 300, 256, 512, rows=512, fp8=fp8)
        plans.append((Plan(g, config=by_name(kw["config"]), swap_ab=swap), t, fp8))
    workspace = torch.randint(0, 256, (max(p.workspace_bytes for p, _, _ in plans),), dtype=torch.uint8, device="cuda")
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(0)
    tok = (torch.randn(1, 300, 512, device=dev, generator=gen) * 0.5).to(torch.float8_e4m3fn)
    sa = torch.rand(1, 300, 4, device=dev, generator=gen) * 1.5 + 0.25
    ws, sbs = [], []
    for _ in range(2):
        ws.append((torch.randn(6, 256, 512, device=dev, generator=gen) * 0.5).to(torch.float8_e4m3fn))
        sbs.append((torch.rand(6, 2, 4, device=dev, generator=gen) * 1.5 + 0.25) * (torch.randint(0, 2, (6, 2, 4), device=dev, generator=gen) * 2 - 1))
    offsets = torch.tensor(_offsets(512, 6), dtype=torch.int32, device=dev).view(-1, 1, 1)
    idx = torch.randint(0, 300, (1, 512, 1), device=dev, generator=gen, dtype=torch.int32)
    for round_ in range(3):
        for plan, t, fp8 in plans:
            pack = {t["tok"]: tok, t["sfa"]: sa, t["fto"]: offsets, t["idx"]: idx}
            for i in range(2):
                pack[t["w"][i]], pack[t["sfb"][i]] = ws[i].transpose(1, 2), sbs[i].transpose(1, 2)
            if fp8:
                q = torch.full((1, 512, 256), 0x7F, device=dev, dtype=torch.uint8)
                qs = torch.full((1, 512, 2), float("nan"), device=dev)
                pack[t["q"]], pack[t["qs"]] = q.view(torch.float8_e4m3fn), qs
            else:
                pack[t["out"]] = torch.full((1, 512, 256), float("nan"), device=dev, dtype=torch.bfloat16)
            plan(pack, workspace=workspace)
            torch.cuda.synchronize()
            if fp8:
                # The same operands as `first` (same seeds): the same bytes, on every round.
                assert torch.equal(q, first.data), f"round {round_}"
                assert torch.equal(qs[0, : first.used].view(torch.int32), first.scale[0, : first.used].view(torch.int32))


@requires_sm100
@pytest.mark.parametrize("swap", [False, True], ids=["fwd", "swap"])
def test_a_gated_pair_refuses_one_scale_matrix_for_either_weight(swap):
    """Both weights of a pair are read at the tile's expert, and so are their scales: a single scale matrix bound for
    either weight is refused before the launch (the kernel is compiled for the count the graph declared).

    The single matrix is a view of a tensor that holds one per expert, in the kernel's axis order."""
    E, S, N, K = 2, 256, 128, 512
    g, t = _graph("moe", E, S, N, K)
    plan = Plan(g, config=by_name(_config(32) if swap else _TILE_128), swap_ab=swap)
    dev = "cuda"
    per_expert = [torch.ones(E, N // _BLOCK, K // _BLOCK, device=dev) for _ in range(2)]
    out = torch.full((1, S, N), float("nan"), device=dev, dtype=torch.bfloat16)
    pack = {
        t["tok"]: torch.ones(1, S, K, device=dev).to(torch.float8_e4m3fn),
        t["sfa"]: torch.ones(1, S, K // _BLOCK, device=dev),
        t["fto"]: torch.tensor([0, 128, 256], dtype=torch.int32, device=dev).view(-1, 1, 1),
        t["out"]: out,
    }
    for i in range(2):
        pack[t["w"][i]] = torch.ones(E, N, K, device=dev).to(torch.float8_e4m3fn).transpose(1, 2)
    for short in range(2):
        for i in range(2):
            pack[t["sfb"][i]] = per_expert[i][:1] if i == short else per_expert[i].transpose(1, 2)
        with pytest.raises(ValueError, match=rf"[AB] scale\[{short}\] must have shape \[2, 1, 4\].*compiled for one scale matrix per matrix"):
            plan(pack)
    torch.cuda.synchronize()
    assert torch.isnan(out.float()).all()
    # The declared count for both: the pair runs.
    for i in range(2):
        pack[t["sfb"][i]] = per_expert[i].transpose(1, 2)
    plan(pack)
    torch.cuda.synchronize()
    silu_512 = 512.0 / (1.0 + torch.exp(torch.tensor(-512.0)))
    assert torch.allclose(out[0].float(), torch.full((S, N), float(silu_512 * 512.0), device=dev), rtol=5e-3)


# Releasing what was read (python/cudnn/AGENTS.md Rule 11): the pair's slot word and the lanes' column maxima
#
# A pair's scheduler slot carries one more word (which of the two GEMMs the pass computes), read by the loader, the
# promotion warps and the epilogue: it reaches the address of the slot's release like the other words. A quantize
# block across the lanes (the weight-by-token templates) keeps each epilogue warp's column maxima in shared memory;
# the warps read each other's behind a thread-block barrier, and the buffer is rewritten two drains later, behind the
# next drain's barrier. That release has no address to compute, so the order is the one Rule 11 names for it: every
# maximum a lane reads reaches a global store of its drain (the scale, or the data it scales) in front of that barrier.

_GATED_RELEASES = {
    # the loader and the promotion warps read 7 words, the epilogue 8; "table" is the partial's release behind the
    # promotion scale table (at least one register). GATHER adds its relay warp (1) and token-copy warps (4). These
    # thread blocks are their own clusters: no mailbox acknowledgement.
    "fwd": (dict(kind="moe"), [1, 7, 7, 8]),
    "swap_fp8": (dict(kind="moe", fp8=True, swap=True, config=_config(32)), [1, 7, 7, 8, "table"]),
    "gather_fp8": (dict(kind="gather", S=300, rows=512, fp8=True), [1, 1, 4, 7, 7, 8]),
}


def _gated_plan(kind="moe", E=6, S=512, N=256, K=512, *, rows=None, fp8=False, swap=False, config=_TILE_128):
    g, t = _graph(kind, E, S, N, K, rows=rows, fp8=fp8)
    plan = Plan(g, config=by_name(config), swap_ab=swap)
    assert select_template(plan.chain, plan._compiled.config).file == _TEMPLATES[(kind, swap)]
    return plan, t


@requires_sm100
@pytest.mark.parametrize("case", list(_GATED_RELEASES))
def test_a_gated_pair_releases_its_slot_behind_every_word_it_read(case, monkeypatch, tmp_path):
    """In the generated PTX of a pair: every slot release is computed from the words read, the pass word included."""
    kw, expected = _GATED_RELEASES[case]
    keep_kernel_ptx(monkeypatch, C, tmp_path)
    plan, _ = _gated_plan(**kw)
    sites = warm_stress.assert_release_dependence(kernel_ptx(plan), sites=len(expected), words=1)
    found = sorted(site.loaded_registers for site in sites if site.dependent)
    for words in (w for w in expected if w != "table"):
        found.remove(words)
    assert found == [] if "table" not in expected else (len(found) == 1 and found[0] >= 1)


@requires_sm100
@pytest.mark.parametrize("kind", ["moe", "gather"])
def test_a_lanes_quantize_stores_what_it_read_of_the_column_maxima(kind, monkeypatch, tmp_path):
    """In the generated PTX of a quantize block across the lanes: the maxima a lane loads behind the epilogue's
    barrier reach a global store further down, before the loop comes back to that barrier and the buffer may be
    rewritten. The barrier appears once, so everything behind it up to the end of the function is that drain."""
    keep_kernel_ptx(monkeypatch, C, tmp_path)
    plan, _ = _gated_plan(kind=kind, S=300 if kind == "gather" else 512, rows=512 if kind == "gather" else None, fp8=True, swap=True, config=_config(32))
    assert "wide_quant_lanes = True" in open(plan.generated_path).read()
    lines = [line.strip() for line in kernel_ptx(plan).splitlines()]
    barriers = [i for i, line in enumerate(lines) if re.match(r"barrier\.sync\s+1, 128;", line)]
    assert len(barriers) == 1, "the epilogue's barrier of a lanes quantize is one instruction"
    register = re.compile(r"%[a-z]+\d+")
    # The loads of the four warps' maxima: the shared-memory loads between the barrier and the first shuffle of the scale.
    first_shuffle = next(i for i in range(barriers[0], len(lines)) if lines[i].startswith("shfl.sync"))
    loads = [i for i in range(barriers[0], first_shuffle) if lines[i].startswith("ld.shared")]
    assert len(loads) == 4, "one maximum per epilogue warp"
    for load in loads:
        reached = set(register.findall(lines[load].split(",")[0]))
        stored = False
        for line in lines[load + 1 :]:
            if line.startswith(("//", "$")) or line.endswith(":"):
                continue
            opcode, _, operands = line.lstrip("@!%p0123456789 ").partition(" ")
            if opcode.startswith("st.global"):
                stored = stored or bool(reached & set(register.findall(operands)))
                continue
            results, _, sources = operands.partition("}" if operands.startswith("{") else ",")
            if reached & set(register.findall(sources)):
                reached |= set(register.findall(results))
        assert stored, f"the maximum loaded at PTX line {load + 1} reaches no global store"


def _fp8_case(kind, G=512):
    """A swapped gated FP8 plan over ``G`` groups of 16 routed rows and its pack: the plan, the pack, codes, scales."""
    E, rows_per_group, N, K = 8, 16, 128, 128
    S = G * rows_per_group
    plan, t = _gated_plan(kind=kind, E=E, S=S, N=N, K=K, rows=S if kind == "gather" else None, fp8=True, swap=True, config=_config(16))
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(5)
    q = torch.empty(1, S, N, device=dev, dtype=torch.uint8)
    qs = torch.empty(1, S, N // _BLOCK, device=dev)
    pack = {
        t["tok"]: (torch.randn(1, S, K, device=dev, generator=gen) * 0.5).to(torch.float8_e4m3fn),
        t["sfa"]: torch.rand(1, S, K // _BLOCK, device=dev, generator=gen) + 0.5,
        t["fto"]: (torch.arange(E + 1, dtype=torch.int32, device=dev) * (S // E)).view(-1, 1, 1),
        t["q"]: q.view(torch.float8_e4m3fn),
        t["qs"]: qs,
    }
    for i in range(2):
        pack[t["w"][i]] = (torch.randn(E, N, K, device=dev, generator=gen) * 0.5).to(torch.float8_e4m3fn).transpose(1, 2)
        pack[t["sfb"][i]] = (torch.rand(E, N // _BLOCK, K // _BLOCK, device=dev, generator=gen) + 0.5).transpose(1, 2)
    if kind == "gather":
        pack[t["idx"]] = torch.randperm(S, device=dev, generator=gen).to(torch.int32).view(1, S, 1)
    return plan, pack, q, qs


def _fp8_stress_case(kind):
    """A swapped gated FP8 launch of many tiles per thread block; the reference is the cold launches' bytes (the
    approximate reciprocal of the scale has no independent byte model)."""
    plan, pack, q, qs = _fp8_case(kind)
    S = q.shape[1]
    workspace = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda")
    cfg = plan._compiled.config
    blocks = C._grid_num_clusters(cfg) * cfg.cga_size_m * cfg.cga_size_n
    # Every tile of 16 routed rows runs twice (the pair's two passes), each pass through a scheduler slot.
    passes = 2 * (S // cfg.cta_tile_n)
    return warm_stress.StressCase(
        name=f"general grouped gated fp8 {kind} swap",
        launch=lambda: plan(pack, workspace=workspace),
        outputs={"codes": q, "scales": qs},
        thread_blocks=blocks,
        refills_per_launch=max(passes // blocks - 2, 0),
    )


@requires_sm100
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("kind", ["moe", "gather"])
def test_gated_fp8_warm_launches_keep_writing_the_cold_bytes(kind):
    """Warm launches of a swapped gated FP8 plan, back to back and behind a launch that overwrites every SM's shared
    memory, write the bytes of its cold launches: codes and scales. The shape rewrites every scheduler slot and both
    buffers of column maxima many times inside one launch. A rate bound at one thread block per SM (see the grouped
    cells in test_general_moe_grouped_block_scale_matmul_fwd.py); the generated-code tests above pin the order."""
    warm_stress.build_launchers(torch.device("cuda"))
    stress = _fp8_stress_case(kind)
    assert stress.refills_per_launch >= 4
    warm_stress.assert_warm_stress(stress, validation=warm_stress.stress_validation(torch.device("cuda")))


@requires_sm100
@pytest.mark.parametrize("kind", ["moe", "gather"])
def test_gated_fp8_graph_replays_write_the_eager_bytes(kind):
    """A swapped gated FP8 plan captured in a CUDA graph (the launch epoch is captured with it): every replay, codes
    and scales poisoned before it, writes the bytes of the eager launch, back to back and around an eager launch."""
    plan, pack, q, qs = _fp8_case(kind, G=64)
    workspace = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def launch():
        plan._compiled(pack, workspace=workspace, stream=torch.cuda.current_stream().cuda_stream)

    def run(step):
        q.fill_(0x7F)
        qs.fill_(float("nan"))
        step()
        torch.cuda.synchronize()
        return q.clone(), qs.clone()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            codes, scales = run(launch)
            with torch.cuda.graph(graph, stream=stream):
                launch()
        assert not bool((codes == 0x7F).any()) and bool(torch.isfinite(scales).all()), "the eager launch left an output unwritten"
        for what in ("replay 1", "replay 2", "replay 3", "eager", "replay 4", "replay 5"):
            if what == "eager":
                with torch.cuda.stream(stream):
                    got = run(launch)
            else:
                got = run(graph.replay)
            assert torch.equal(got[0], codes), what
            assert torch.equal(got[1].view(torch.int32), scales.view(torch.int32)), what
    finally:
        torch.cuda.synchronize()
        graph.reset()


# A wide quantize runs at the extents it was built for
#
# The scale output of a wide quantize has the graph's shape, [1, rows, N / 128] (a constant rule), while the epilogue
# stores scale block col // 128 of a row for the call's N, through the call's strides: at a larger N it wrote scales past
# that tensor (a row's last block over the next row's first), and at an N inside a block it left that block's scale
# unwritten. A call at other extents is refused before anything is written; a call without rows has nothing to write.


def _wide_quant_pack(t, kind, E, S, rows, N, K, gemms):
    """Operands for a call at runtime ``N`` with the graph-shaped scale view: (pack, codes, the scale view's backing).

    The weights are bound in the kernel's order, [lead, N, K]; so are the grouped weight scales, [E, N / 128, K / 128],
    while the dense one is the graph's [1, K / 128, N / 128] view of them."""
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(3)
    lead = 1 if kind == "dense" else E
    kb, nb = K // _BLOCK, -(-N // _BLOCK)
    pack = {
        t["tok"]: (torch.randn(1, S, K, device=dev, generator=gen) * 0.5).to(torch.float8_e4m3fn),
        t["sfa"]: torch.rand(1, S, kb, device=dev, generator=gen) + 0.5,
    }
    for i in range(gemms):
        pack[t["w"][i]] = (torch.randn(lead, N, K, device=dev, generator=gen) * 0.5).to(torch.float8_e4m3fn)
        sb = torch.rand(lead, nb, kb, device=dev, generator=gen) + 0.5
        pack[t["sfb"][i]] = sb.transpose(1, 2) if kind == "dense" else sb
    if kind != "dense":
        pack[t["fto"]] = (torch.arange(E + 1, dtype=torch.int32, device=dev) * (rows // E)).view(-1, 1, 1)
    if kind == "gather":
        pack[t["idx"]] = torch.randint(0, S, (1, rows, 1), device=dev, generator=gen, dtype=torch.int32)
    codes = torch.full((1, rows, N), 0x7F, device=dev, dtype=torch.uint8)
    # The graph's [1, rows, 2] scale view in front of a guard: a scale stored past the view lands in the guard.
    backing = torch.full((rows * 2 + 4096,), float("nan"), device=dev)
    pack[t["q"]], pack[t["qs"]] = codes.view(torch.float8_e4m3fn), backing[: rows * 2].view(1, rows, 2)
    return pack, codes, backing


@requires_sm100
@pytest.mark.parametrize("N", [384, 320, 128], ids=["wider", "partial_block", "narrower"])
@pytest.mark.parametrize("kind,gemms", [("dense", 1), ("moe", 1), ("gather", 2)], ids=["dense", "moe", "gated_gather"])
def test_a_wide_quantize_refuses_a_call_at_another_n(kind, gemms, N):
    """Built for N = 256, called at another N with the graph-shaped scale view: refused, and neither a code nor a scale
    (in the view or past it) is written."""
    E, S, N0, K = 4, 256, 256, 512
    rows = 256
    S_tok = 192 if kind == "gather" else S
    g, t = _graph(kind, E, S_tok, N0, K, rows=rows if kind == "gather" else None, gemms=gemms, fp8=True)
    plan = Plan(g, config=by_name(_TILE_128))
    assert select_template(plan.chain, plan._compiled.config).file == _TEMPLATES[(kind, False)]
    pack, codes, backing = _wide_quant_pack(t, kind, E, S_tok, rows, N, K, gemms)
    with pytest.raises(ValueError, match=rf"runs only at the kernel-frame \(M, N\) = \({rows}, {N0}\) it was built for, not \({rows}, {N}\)"):
        plan(pack)
    torch.cuda.synchronize()
    assert bool((codes == 0x7F).all()) and bool(torch.isnan(backing).all())
    # The control: at the built extents the same buffers are written, the guard is not.
    pack, codes, backing = _wide_quant_pack(t, kind, E, S_tok, rows, N0, K, gemms)
    plan(pack)
    torch.cuda.synchronize()
    assert not bool((codes == 0x7F).any()) and bool(torch.isfinite(backing[: rows * 2]).all()) and bool(torch.isnan(backing[rows * 2 :]).all())


@requires_sm100
@pytest.mark.parametrize("kind,gemms", [("moe", 1), ("gather", 2)], ids=["moe", "gated_gather"])
def test_a_swapped_wide_quantize_refuses_a_call_at_another_token_count(kind, gemms):
    """A weight-by-token plan built for 256 routed rows, called with 128 and the graph-shaped scale tensor: refused with
    the call's extents in the kernel frame, whose M and N are the graph's N and routed rows, and neither a code nor a
    scale is written."""
    E, S, N, K = 4, 256, 256, 512
    rows, half = 256, 128
    S_tok = 192 if kind == "gather" else S
    g, t = _graph(kind, E, S_tok, N, K, rows=rows if kind == "gather" else None, gemms=gemms, fp8=True)
    plan = Plan(g, config=by_name(_config(32)), swap_ab=True)
    assert select_template(plan.chain, plan._compiled.config).file == _TEMPLATES[(kind, True)]
    pack, codes, _ = _wide_quant_pack(t, kind, E, S_tok if kind == "gather" else half, half, N, K, gemms)
    backing = torch.full((rows * 2 + 4096,), float("nan"), device="cuda")
    pack[t["qs"]] = backing[: rows * 2].view(1, rows, 2)
    with pytest.raises(
        ValueError, match=rf"runs only at the kernel-frame \(M, N\) = \({N}, {rows}\) it was built for, not \({N}, {half}\).*kernel frame of this swap-AB plan"
    ):
        plan(pack)
    torch.cuda.synchronize()
    assert bool((codes == 0x7F).all()) and bool(torch.isnan(backing).all())
    # The control: at the built extents the same buffers are written, the guard is not.
    pack, codes, backing = _wide_quant_pack(t, kind, E, S_tok, rows, N, K, gemms)
    plan(pack)
    torch.cuda.synchronize()
    assert not bool((codes == 0x7F).any()) and bool(torch.isfinite(backing[: rows * 2]).all()) and bool(torch.isnan(backing[rows * 2 :]).all())


@requires_sm100
@pytest.mark.parametrize("swap", [False, True], ids=["fwd", "swap"])
@pytest.mark.parametrize("gemms", [1, 2], ids=["one_gemm", "gated"])
def test_a_wide_quantize_call_without_rows_writes_nothing(gemms, swap, monkeypatch):
    """An expert-parallel rank with no routed rows calls the plan with zero token rows (zero-row views of live
    buffers) and the graph-shaped scale tensor: the call returns before the launch, and nothing is written."""
    E, S0, N, K = 4, 256, 256, 512
    g, t = _graph("moe", E, S0, N, K, gemms=gemms, fp8=True)
    plan = Plan(g, config=by_name(_config(32) if swap else _TILE_128), swap_ab=swap)
    assert select_template(plan.chain, plan._compiled.config).file == _TEMPLATES[("moe", swap)]

    def no_launch(*args, **kwargs):
        pytest.fail("a call without rows reached the launch")

    monkeypatch.setattr(type(plan._compiled), "_launch_moe", no_launch)
    dev = "cuda"
    codes = torch.full((1, 8, N), 0x7F, device=dev, dtype=torch.uint8)
    scales = torch.full((1, S0, N // _BLOCK), float("nan"), device=dev)
    pack = {
        t["tok"]: torch.zeros(1, 8, K, device=dev).to(torch.float8_e4m3fn)[:, :0],
        t["sfa"]: torch.ones(1, 0, K // _BLOCK, device=dev),
        t["fto"]: torch.zeros(E + 1, 1, 1, dtype=torch.int32, device=dev),
        t["q"]: codes.view(torch.float8_e4m3fn)[:, :0],
        t["qs"]: scales,
    }
    for i in range(gemms):
        pack[t["w"][i]] = torch.ones(E, N, K, device=dev).to(torch.float8_e4m3fn).transpose(1, 2)
        pack[t["sfb"][i]] = torch.ones(E, N // _BLOCK, K // _BLOCK, device=dev).transpose(1, 2)
    assert plan(pack) is None
    torch.cuda.synchronize()
    assert bool((codes == 0x7F).all()) and bool(torch.isnan(scales).all())


# Register moves and the pair's first-pass stage (source, no GPU)
#
# setmaxnreg is a warpgroup instruction: the PTX ISA leaves it undefined unless every warp of the warpgroup executes the
# same one. An mbarrier phase that completes while no thread waits on it is what compute-sanitizer synccheck reports as
# a missing wait (and it ends the launch): the epilogue waits on every acc_full phase the promotion warps complete.


def _setmaxnreg_sites(source):
    """(action, count, the run-time ``if`` tests above it) of every ``nvvm.setmaxregister`` call of a template."""
    sites = []

    def visit(node, tests):
        if isinstance(node, ast.If):
            test = ast.unparse(node.test)
            inner = tests if test.startswith("cutlass.const_expr(") else [*tests, test]
            for child in [*node.body, *node.orelse]:
                visit(child, inner)
            return
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "nvvm.setmaxregister":
            sites.append((ast.unparse(node.args[1]).rsplit(".", 1)[-1], ast.unparse(node.args[0]), tests))
        for child in ast.iter_child_nodes(node):
            visit(child, tests)

    visit(ast.parse(source), [])
    return sites


@pytest.mark.parametrize("kind,swap", list(_TEMPLATES))
def test_every_warpgroup_of_a_general_template_moves_its_registers_at_one_instruction(kind, swap):
    """No setmaxregister sits in a branch that one warp takes (``warp_idx == <role>``). The control warpgroup (warps
    4..7, four warps of four roles) lowers its count at one entry before they branch; every other call is taken by
    whole warpgroups."""
    sites = _setmaxnreg_sites(template_path(_TEMPLATES[(kind, swap)]).read_text())
    assert sites
    for action, count, tests in sites:
        assert not any(re.search(r"\bwarp_idx ==", test) for test in tests), f"setmaxregister({count}, {action}) under {tests}"
    control = [site for site in sites if site[2] == ["warp_idx >= 4 and warp_idx < 8"]]
    assert len(control) == 1 and control[0][0] == "DECREASE"


@pytest.mark.parametrize("kind,swap", list(_TEMPLATES))
def test_a_gated_pair_waits_on_the_stage_of_its_first_pass(kind, swap):
    """The epilogue drains a tile's two passes together, behind the second pass's acc_full. The first pass completes a
    phase of its own stage's acc_full too: where the epilogue skips the first pass, it waits on that phase. Otherwise
    compute-sanitizer synccheck reports every gated launch of more than one tile per thread block as a missing wait
    (about 500 reports on a 2048 x 2048 x 256 dense pair) and ends it. The source pin stands in for that tool run."""
    source = template_path(_TEMPLATES[(kind, swap)]).read_text()
    begin = source.index("# The tile's first pass (the second paired GEMM) stays in its accumulator stage")
    skip = source[begin : source.index("if epi_drain", begin)]
    assert "mbarrier_try_wait_parity(acc_full_mbar_ptr.subview(tile_iter % acc_stages), acc_full_phase_bit" in skip


# The drain width of a gathered pair's lanes quantize (no GPU)


@pytest.mark.parametrize("n", [64, 128])
def test_a_gathered_gated_pair_drains_its_lanes_quantize_32_columns_at_a_time(n):
    """A gated pair's quantize sweep holds a drain chunk of both GEMMs before the SwiGLU. Under the async GATHER
    templates' 112-register epilogue a chunk of 64 columns (2 x 64 fp32 values per lane) spilled 8 KB a thread on a
    128-column swapped tile, so the chunk is 32 columns there. A single GEMM, and the sorted template's 168-register
    epilogue, keep 64."""

    def epi_n(kind, gemms):
        chain = swap_ab_graph(analyze(_graph(kind, 6, 300 if kind == "gather" else 512, 256, 512, rows=512, gemms=gemms, fp8=True)[0]))
        cfg = dataclasses.replace(by_name(_config(n)), swap_ab=True)
        C._general_block_scale_plan(chain, cfg)
        return C._epi_n_for_chain(cfg, chain)

    assert epi_n("gather", 2) == 32
    assert epi_n("gather", 1) == 64
    assert epi_n("moe", 2) == 64


def test_a_wide_quantize_refuses_a_drain_chunk_narrower_than_its_store_vector(monkeypatch):
    """Both sweeps take a drain chunk as whole store vectors (32 values here): a 16-column chunk would compile to sweeps
    that skip their columns, so the plan is refused."""
    chain = analyze(_graph("moe", 6, 512, 256, 512, fp8=True)[0])
    cfg = by_name(_TILE_128)
    assert C._general_block_scale_plan(chain, cfg)["wide_quant"]
    monkeypatch.setattr(C, "_epi_n_for_chain", lambda cfg, chain: 16)
    with pytest.raises(NotImplementedError, match="drains whole 32-value vectors"):
        C._general_block_scale_plan(chain, cfg)


# A wide quantize's block maximum skips NaN element by element in either orientation, a nonzero scale is at least the
# smallest normal fp32 (2**-126), whose reciprocal the kernel's flush-to-zero approximation keeps, and a block of zeros
# stores scale 0 and codes 0.

_MIN_NORMAL = 2.0**-126


@requires_sm100
@pytest.mark.parametrize("swap", [False, True], ids=["fwd", "swap"])
def test_a_wide_quantize_skips_a_nan_and_floors_a_tiny_scale(swap):
    """One GEMM, FP8 block-128 output, three experts of 128 routed rows each over exact operands (small integers and
    powers of two: every output value is exact in fp32):

    - expert 0: an e4m3 NaN in output column 5 of the weight, beside column 6 whose weights are 4 times the others'. The
      block's scale is its largest finite magnitude times RN(1/448), and the NaN's code is NaN. The non-swapped drain
      took each 32-value vector's maximum NaN-propagating and folded the vectors NaN-skipping: columns 0..31, column 6
      among them, dropped out of the scale, and column 6 saturated.
    - expert 1: the values are about 1e-36, so amax / 448 is subnormal: the scale is 2**-126, and the codes are the
      values times 2**126. The flush-to-zero reciprocal of a subnormal scale is infinite, which made the codes the
      values times FLT_MAX.
    - expert 2: the token rows are zero: scale 0, codes 0."""
    E, R, N, K = 3, 128, 256, 256
    S = E * R
    g, t = _graph("moe", E, S, N, K, gemms=1, fp8=True)
    plan = Plan(g, config=by_name(_config(32) if swap else _TILE_128), swap_ab=swap)
    assert select_template(plan.chain, plan._compiled.config).file == _TEMPLATES[("moe", swap)]
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(7)
    kb, nb = K // _BLOCK, N // _BLOCK
    tok = torch.randint(-1, 2, (1, S, K), device=dev, generator=gen).float()
    tok[0, :R, 3] = 1.0  # every row of expert 0 multiplies the NaN
    tok[0, 2 * R :] = 0.0
    w = torch.randint(-1, 2, (E, N, K), device=dev, generator=gen).float()
    w[0, 6] *= 4.0
    sa = torch.ones(1, S, kb, device=dev)
    sa[0, R : 2 * R] = 2.0**-64
    sb = torch.ones(E, nb, kb, device=dev)
    sb[1] = 2.0**-61
    w8 = w.to(torch.float8_e4m3fn)
    w8.view(torch.uint8)[0, 5, 3] = 0x7F  # an e4m3 NaN
    w_ref = w.double()
    w_ref[0, 5, 3] = float("nan")
    tok_ref = tok[0].double() * sa[0].double().repeat_interleave(_BLOCK, 1)
    w_ref = w_ref * sb.double().repeat_interleave(_BLOCK, 1).repeat_interleave(_BLOCK, 2)
    ref = torch.cat([tok_ref[e * R : (e + 1) * R] @ w_ref[e].T for e in range(E)])
    q = torch.full((1, S, N), 0x55, device=dev, dtype=torch.uint8)
    qs = torch.full((1, S, nb), float("nan"), device=dev)
    pack = {
        t["tok"]: tok.to(torch.float8_e4m3fn),
        t["sfa"]: sa,
        t["w"][0]: w8.transpose(1, 2),
        t["sfb"][0]: sb.transpose(1, 2),
        t["fto"]: (torch.arange(E + 1, dtype=torch.int32, device=dev) * R).view(-1, 1, 1),
        t["q"]: q.view(torch.float8_e4m3fn),
        t["qs"]: qs,
    }
    plan(pack)
    torch.cuda.synchronize()

    nan = torch.isnan(ref)
    amax = torch.nan_to_num(ref.abs(), nan=0.0).view(S, nb, _BLOCK).amax(dim=2).float()
    want = amax * torch.tensor(1.0 / _FP8_MAX, dtype=torch.float32, device=dev)
    want = torch.where(amax > 0, want.clamp_min(_MIN_NORMAL), want)
    scale = qs[0]
    for e, what in enumerate(("a block with a NaN", "a block of about 1e-36", "a block of zeros")):
        rows = slice(e * R, (e + 1) * R)
        assert torch.equal(
            scale[rows].view(torch.int32), want[rows].view(torch.int32)
        ), f"{what}: scale {scale[rows].flatten()[:4].tolist()}, want {want[rows].flatten()[:4].tolist()}"
    codes = q[0].view(torch.float8_e4m3fn).float()
    assert bool(nan[:R, 5].all()) and int(nan.sum()) == R
    assert bool(torch.isnan(codes[nan]).all()), "a NaN's code is not NaN"
    assert bool((q[0, 2 * R :] == 0).all()), "a block of zeros stored a nonzero code"
    finite = ~nan
    finite[2 * R :] = False
    per_elem = scale.repeat_interleave(_BLOCK, dim=1)[finite]
    got = codes[finite]
    want_codes = (ref.float()[finite] / per_elem).to(torch.float8_e4m3fn).float()
    diff = (got - want_codes).abs()
    assert bool((diff <= _e4m3_step(torch.maximum(got.abs(), want_codes.abs()))).all()), f"a code is {float(diff.max())} away from the reference's"
    assert float((diff == 0).float().mean()) > 0.995


# The pair's drain: the first pass's stage back after the first sweep, and SiLU in one MUFU op
#
# A pair's first pass waits in its main accumulator stage for the second, and the MMA warp starts the next tile's first
# pass into that stage once the epilogue releases it. Of a wide quantize's two sweeps, the first is the last read of
# that stage: the second sweep reloads only the stage the first wrote its values back to. The drain's SiLU runs while
# the MMA warp waits: x * (1/2 + tanh.approx(x / 2) / 2) takes one MUFU op per value where x * rcp(1 + ex2(-x log2 e))
# takes two (ex2.approx, rcp.approx). Only these pairs evaluate it so; every other SiLU keeps the two-op form.


def _wide_quant_region(kind, swap):
    source = template_path(_TEMPLATES[(kind, swap)]).read_text()
    begin = source.index("# @@WIDE_QUANT_ONLY:BEGIN@@")
    return source[begin : source.index("# @@WIDE_QUANT_ONLY:END@@", begin)]


_ACC_RELEASE = re.compile(r"nvvm\.mbarrier_arrive\(acc_empty_mbar_ptr\.subview\(([^)]*)\)\)")


@pytest.mark.parametrize("kind,swap", list(_TEMPLATES))
def test_a_gated_wide_quantize_releases_the_first_pass_stage_after_the_first_sweep(kind, swap):
    """(source, no GPU) A pair arrives on its first pass's stage (the second paired GEMM's, one stage back) once the first
    sweep, its last reader, is done with it: behind the sweep's tcgen05.st of the values computed from every word it
    loaded and the wait for those stores, a wait for the loads and a fence (python/cudnn/AGENTS.md Rule 11, tensor
    memory: the first read of the loaded registers precedes the release), and before the second sweep. The second sweep
    arrives on the other stage only. When both arrived after the second sweep's loads, the MMA warp waited for the whole
    drain before each tile's first pass."""
    region = _wide_quant_region(kind, swap)
    first, second = region.split("# @@INJECT_WQ_SCALE@@")
    store_wait = "nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)"
    steps = [
        "_cv = nvvm.tcgen05_ld(",
        "# @@INJECT_WQ_VALUES@@",
        "nvvm.tcgen05_st(shape, cutlass.inttoptr(tmem_col_addr_gemms[0] + wq_col, 6, cutlass.Float32), wq_src)",
        store_wait,
        "if cutlass.const_expr(glu_pair):",
        "nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)",
        "nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)",
        "nvvm.mbarrier_arrive(acc_empty_mbar_ptr.subview(acc_stage - 1))",
    ]
    missing = [step for step in steps if step not in first]
    assert not missing, f"the first sweep has no {missing}"
    order = [first.index(step) for step in steps]
    assert order == sorted(order), "the first sweep releases the first pass's stage out of order"
    assert not _ACC_RELEASE.findall(first[: first.index(store_wait)]), "a release inside the first sweep's loads"
    assert _ACC_RELEASE.findall(first) == ["acc_stage - 1"]
    released = _ACC_RELEASE.findall(second)
    assert released == ["acc_stage"], f"the second sweep releases {released}"


def _silu_graph(gemms=2, S=512, N=256, K=128, out=DT.FLOAT, beta=None):
    """``swish(tokens @ w0) * (tokens @ w1)`` (``gemms=2``) or ``swish(tokens @ w0)``, dense, FP8 block-128 operands."""
    g = cudnn.pygraph(io_data_type=DT.BFLOAT16, intermediate_data_type=DT.FLOAT, compute_data_type=DT.FLOAT)
    kb, nb = K // _BLOCK, N // _BLOCK
    t = {"w": [], "sfb": []}
    t["tok"] = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=DT.FP8_E4M3)
    t["sfa"] = g.tensor(name="SFA", dim=[1, S, kb], stride=[S * kb, kb, 1], data_type=DT.FLOAT)
    tok_d = g.block_scale_dequantize(input=t["tok"], descale=t["sfa"], block_size=[1, _BLOCK])
    c = []
    for i in range(gemms):
        t["w"].append(g.tensor(name=f"w{i}", dim=[1, K, N], stride=[K * N, 1, K], data_type=DT.FP8_E4M3))
        t["sfb"].append(g.tensor(name=f"SFB{i}", dim=[1, kb, nb], stride=[kb * nb, 1, kb], data_type=DT.FLOAT))
        c.append(g.matmul(A=tok_d, B=g.block_scale_dequantize(input=t["w"][i], descale=t["sfb"][i], block_size=[_BLOCK, _BLOCK]), name=f"mm{i}"))
    y = g.swish(input=c[0], name="silu") if beta is None else g.swish(input=c[0], swish_beta=beta, name="silu")
    if gemms == 2:
        y = g.mul(a=y, b=c[1], name="mul")
    y.set_output(True).set_data_type(out)
    t["out"] = y
    return g, t


def _mxfp8_pair_graph(M=256, N=256, K=512):
    """``swish(A @ B0) * (A @ B1)`` over MXFP8 (E8M0 scales per 32 values): a pair of the source block-scale templates."""
    g = cudnn.pygraph(io_data_type=DT.BFLOAT16, intermediate_data_type=DT.FLOAT, compute_data_type=DT.FLOAT)
    rk = dict(reordering_type=cudnn.tensor_reordering.F8_128x4)
    a = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1], data_type=DT.FP8_E4M3)
    sfa = g.tensor(name="SFA", dim=[1, M, K // 32], stride=[M * K // 32, K // 32, 1], data_type=DT.FP8_E8M0, **rk)
    a_d = g.block_scale_dequantize(input=a, descale=sfa, block_size=[1, 32])
    c = []
    for i in range(2):
        b = g.tensor(name=f"B{i}", dim=[1, K, N], stride=[K * N, 1, K], data_type=DT.FP8_E4M3)
        sfb = g.tensor(name=f"SFB{i}", dim=[1, K // 32, N], stride=[K // 32 * N, 1, K // 32], data_type=DT.FP8_E8M0, **rk)
        c.append(g.matmul(A=a_d, B=g.block_scale_dequantize(input=b, descale=sfb, block_size=[32, 1]), name=f"mm{i}"))
    g.mul(a=g.swish(input=c[0]), b=c[1]).set_output(True).set_data_type(DT.BFLOAT16)
    return g


def _plain_pair_graph(M=256, N=256, K=512):
    """``swish(A @ B0) * (A @ B1)`` in bf16: a pair of the plain templates."""
    g = cudnn.pygraph(io_data_type=DT.BFLOAT16, intermediate_data_type=DT.FLOAT, compute_data_type=DT.FLOAT)
    a = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1], data_type=DT.BFLOAT16)
    c = [g.matmul(A=a, B=g.tensor(name=f"B{i}", dim=[1, K, N], stride=[K * N, 1, K], data_type=DT.BFLOAT16), name=f"mm{i}") for i in range(2)]
    g.mul(a=g.swish(input=c[0]), b=c[1]).set_output(True).set_data_type(DT.BFLOAT16)
    return g


def test_only_a_general_gated_pair_evaluates_silu_through_tanh_approx():
    """(no GPU) The five templates' pairs, bf16 or FP8 out, take the one-op SiLU; a general template's single GEMM with
    a SiLU, a pair of the source block-scale templates (MXFP8) and a plain bf16 pair keep the two-op form, and so does
    a swish whose beta is not 1."""
    for kind, swap in _TEMPLATES:
        for fp8 in (False, True):
            chain = analyze(_graph(kind, 6, 300 if kind == "gather" else 512, 256, 512, rows=512, fp8=fp8)[0])
            assert C._glu_silu_tanh(swap_ab_graph(chain) if swap else chain), (kind, swap, fp8)
    assert C._glu_silu_tanh(analyze(_silu_graph()[0]))
    single = analyze(_silu_graph(gemms=1)[0])
    assert C._is_general_block_scale(single) and not C._glu_silu_tanh(single)
    assert not C._glu_silu_tanh(analyze(_mxfp8_pair_graph()))
    assert not C._glu_silu_tanh(analyze(_plain_pair_graph()))

    def swish(silu_tanh, beta=None):
        op = FusionOp(op="swish", attrs=() if beta is None else (("swish_beta", beta),))
        return "\n".join(EC._emit_op(op, "x", 0, {}, silu_tanh=silu_tanh)[0])

    one_op = "x * (cutlass.full_like(x, cutlass.Float32(0.5)) + cutlass.full_like(x, cutlass.Float32(0.5)) * cute.math.tanh(x * cutlass.full_like(x, cutlass.Float32(0.5)), fastmath=True))"
    assert swish(True) == f"_op_0 = {one_op}" and swish(True, beta=1.0) == f"_op_0 = {one_op}"
    for form in (swish(False), swish(False, beta=1.0), swish(True, beta=2.0)):
        assert "cute.math.tanh" not in form and "cute.math.exp2(" in form and "cute.math.rcp(" in form


@requires_sm100
@pytest.mark.parametrize(
    "case",
    ["dense_bf16", "gather_fp8", "gather_swap_fp8", "dense_one_gemm"],
)
def test_a_gated_pair_takes_one_tanh_approx_per_silu(case, monkeypatch, tmp_path):
    """In the generated PTX: a pair's SiLU is tanh.approx.f32 (one MUFU op) and the kernel takes no ex2.approx (which
    the two-op form needs); a general template's single GEMM with a SiLU keeps ex2.approx and takes no tanh.approx."""
    keep_kernel_ptx(monkeypatch, C, tmp_path)
    if case == "dense_one_gemm":
        plan = Plan(_silu_graph(gemms=1, out=DT.BFLOAT16)[0], config=by_name(_TILE_128))
        counts = {op: kernel_ptx(plan).count(op) for op in ("tanh.approx.f32", "ex2.approx")}
        assert counts["tanh.approx.f32"] == 0 and counts["ex2.approx"] > 0, counts
        return
    if case == "dense_bf16":
        plan = Plan(_graph("dense", 6, 512, 256, 512)[0], config=by_name(_TILE_128))
    else:
        swap = case == "gather_swap_fp8"
        plan, _ = _gated_plan(kind="gather", S=300, rows=512, fp8=True, swap=swap, config=_config(32) if swap else _TILE_128)
    counts = {op: kernel_ptx(plan).count(op) for op in ("tanh.approx.f32", "ex2.approx")}
    assert counts["tanh.approx.f32"] > 0 and counts["ex2.approx"] == 0, counts


@requires_sm100
def test_a_gated_pair_silu_stays_within_a_tanh_approx_of_the_exact_one():
    """A pair's SiLU against x * sigmoid(x) in fp64, over exact gate values with the up GEMM 1: one-hot token rows times
    e4m3 weights and a power-of-two scale (x in [-28, 28], every value an e4m3 one over 16), output in fp32. The bound,
    2^-12 |x|, is the error of x * (1/2 + tanh(x / 2) / 2) through a tanh within a relative 2^-11 of its value; this part
    measured at most 3.4e-6 |x| on these inputs, the two-op form 8.2e-8 |x|."""
    S, N, K = 512, 256, 128
    g, t = _silu_graph(S=S, N=N, K=K)
    plan = Plan(g, config=by_name(_TILE_128))
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(11)
    tok = torch.zeros(1, S, K, device=dev)
    tok[0, torch.arange(S), torch.arange(S) % K] = 1.0
    w0 = ((torch.rand(1, N, K, device=dev, generator=gen) * 2 - 1) * 448).to(torch.float8_e4m3fn)
    out = torch.full((1, S, N), float("nan"), device=dev)
    plan(
        {
            t["tok"]: tok.to(torch.float8_e4m3fn),
            t["sfa"]: torch.ones(1, S, 1, device=dev),
            t["w"][0]: w0.transpose(1, 2),
            t["w"][1]: torch.ones(1, N, K, device=dev).to(torch.float8_e4m3fn).transpose(1, 2),
            # the graph's [1, K / 128, N / 128] view of [1, N / 128, K / 128] scales, as _run binds them
            t["sfb"][0]: torch.full((1, N // _BLOCK, 1), 1.0 / 16, device=dev).transpose(1, 2),
            t["sfb"][1]: torch.ones(1, N // _BLOCK, 1, device=dev).transpose(1, 2),
            t["out"]: out,
        }
    )
    torch.cuda.synchronize()
    x = (w0[0].double().T / 16)[torch.arange(S) % K]  # [S, N]: row r takes weight column r % K
    want = x * torch.sigmoid(x)
    got = out[0].double()
    assert bool(torch.isfinite(got).all())
    assert x.min() < -16 and x.max() > 16 and x.unique().numel() > 100
    bound = 2.0**-12 * x.abs() + 2.0**-22 * want.abs()
    assert bool(((got - want).abs() <= bound).all()), f"error up to {float(((got - want).abs() / x.abs().clamp_min(1e-30)).max()):.3e} |x|"


@requires_sm100
def test_a_gated_fp8_pair_of_several_tiles_per_thread_block_writes_the_bytes_of_every_cluster():
    """The production FC1's shape of work, GATHER into FP8 block-128 on 128x128 tiles, with several tiles per thread
    block: each thread block's next tile runs its first pass in the stage the drain gave back after its first sweep.
    Clusters of one, two and four tiles along the weight's columns schedule the tiles differently and write the same
    codes and scales; each passes the fp32 reference (``_run``). One K block makes a pass shorter than a drain: with the
    stage given back before the first sweep had read it, the next pass's store reached it under the sweep and the
    output was not finite (with K = 512 it came too late to show)."""
    kw = dict(kind="gather", E=6, S=2048, rows=8192, N=512, K=128, fp8=True)
    one = _run(config=_TILE_128, **kw)
    for cluster in ("1x2", "1x4"):
        many = _run(config=_config(128, cluster), **kw)
        assert torch.equal(many.data.view(torch.uint8), one.data.view(torch.uint8)), cluster
        assert torch.equal(many.scale[0, : one.used].view(torch.int32), one.scale[0, : one.used].view(torch.int32)), cluster
