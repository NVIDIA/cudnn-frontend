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

import dataclasses
import re

import cudnn
import cudnn.gemm.frost  # noqa: F401  (installs recorder)
import pytest
import torch

import warm_stress
from gemm_test_utils import Plan, keep_kernel_ptx, kernel_ptx, requires_sm100
from cudnn.gemm.frost import compiler as C
from cudnn.gemm.frost.fusion_ir import BlockQuantizeSpec, wide_quant
from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.gemm.frost.kernel_registry import general_glu_pair, general_glu_pair_reject, select_template, swap_ab_graph
from cudnn.gemm.frost.tile_config import by_name

pytestmark = pytest.mark.L0

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
@pytest.mark.parametrize("case", ["dense", "fwd", "swap_n16", "gather_cluster1x2", "gather_swap_n32"])
def test_one_gemm_fp8_block128_numerics(case):
    _run(fp8=True, gemms=1, **_GATED[case])


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
    # promotion scale table (at least one register)
    "fwd": (dict(kind="moe"), [1, 1, 7, 7, 8]),
    "swap_fp8": (dict(kind="moe", fp8=True, swap=True, config=_config(32)), [1, 1, 7, 7, 8, "table"]),
    "gather_fp8": (dict(kind="gather", S=300, rows=512, fp8=True), [1, 4, 7, 7, 8]),
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


def _fp8_stress_case(kind):
    """A swapped gated FP8 launch of many tiles per thread block; the reference is the cold launches' bytes (the
    approximate reciprocal of the scale has no independent byte model)."""
    E, G, rows_per_group, N, K = 8, 512, 16, 128, 128
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
    workspace = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device=dev)
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
