# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a graph declares and a call binds to the general block-scale templates, and what the plan does with it.

``sm100_general_block_scale_matmul.py`` and the four ``sm100_general_moe_grouped_block_scale_matmul_fwd*.py`` apply
linear scale tensors to the accumulator. Each case below is a graph or a binding that a plan either serves correctly
or refuses, before any launch, with a message that names the requirement:

- a grouped weight scale bound with strides other than the declared ones. The graph orders it [E, K blocks, N blocks],
  the kernel [E, N blocks, K blocks]: a binding is read in the one order its shape fits, and refused when both fit (a
  square block grid) unless it is the declared view. A grouped weight is read in the order its shape and the plan's
  contiguous axis fit;
- a grouped linear scale aligned to its elements only, which the promotion warps serve as a dense plan does;
- a gated pair whose two weight scale tensors declare different counts of scale matrices;
- cc 10.7 under a CuTe DSL without the ``sm_107a`` target (python/cudnn/AGENTS.md Rule 7);
- a one-sided graph whose unscaled operand is bf16, fp16 or int8;
- a uint8 buffer bound for an int8 operand (grouped or dense) or an int8 scale, refused by a call that hands the plan
  its buffers and read as int8 through graph.execute, whose pack describes it as the declared tensor;
- a weight-by-token call whose token count breaks the alignment ``first_token_offset`` declares.

Host checks run without a launch; the end-to-end cases compare against an fp32 dequantize-then-matmul reference and
poison the output, so a refused call shows that nothing was written."""

from __future__ import annotations

import dataclasses
import math
import types

import cudnn
import cudnn.gemm.frost  # noqa: F401  (installs recorder)
import pytest
import torch

from gemm_test_utils import Plan, requires_int8_mma, requires_sm100, requires_sm100_tree
from cudnn.frost import buffers
from cudnn.gemm.frost import compiler as C
from cudnn.gemm.frost import heuristics
from cudnn.gemm.frost.engine import FrostGemmEngine
from cudnn.gemm.frost.graph_analyzer import analyze, analyze_with_binding, swap_ab_binding
from cudnn.gemm.frost.kernel_registry import general_glu_pair_reject, select_template, swap_ab_graph
from cudnn.gemm.frost.tile_config import by_name

pytestmark = pytest.mark.L0

DT = cudnn.data_type
_CUDNN = {"fp8_e4m3": DT.FP8_E4M3, "bf16": DT.BFLOAT16, "fp16": DT.HALF, "int8": DT.INT8, "fp32": DT.FLOAT}
_TORCH = {"fp8_e4m3": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "int8": torch.int8, "fp32": torch.float32}
_BLOCK = 128
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
_SWAP_32 = _config(128, 32)


def _cdiv(a, b):
    return -(-a // b)


def _moe_graph(E, S, N, K, *, data="fp8_e4m3", sfa="fp32", sfb="fp32", gather_rows=None, gemms=1, sfb_dims=None, groups=None, alignment=1, w_major="k"):
    """``tokens @ weight[g % E]`` per group, or ``swish(tokens @ w0) * (tokens @ w1)`` with ``gemms=2``, bf16 output.

    ``sfa`` / ``sfb`` are the token / weight scale dtypes (None: that operand is not scaled), 1 x 128 and 128 x 128
    blocks. A weight scale is declared K-contiguous, [E, K / 128, N / 128] with strides (K / 128 * N / 128, 1, K / 128):
    the transpose of an [E, N / 128, K / 128] tensor. ``sfb_dims[i]`` declares other dims for weight ``i``. The weight
    [E, K, N] is K-contiguous, or N-contiguous with ``w_major="n"``."""
    groups = E if groups is None else groups
    g = cudnn.pygraph(io_data_type=DT.BFLOAT16, intermediate_data_type=DT.FLOAT, compute_data_type=DT.FLOAT)
    t = {"w": [], "sfb": []}
    kb, nb = _cdiv(K, _BLOCK), _cdiv(N, _BLOCK)
    t["tok"] = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=_CUDNN[data])
    tok = t["tok"]
    if sfa is not None:
        t["sfa"] = g.tensor(name="SFA", dim=[1, S, kb], stride=[S * kb, kb, 1], data_type=_CUDNN[sfa])
        tok = g.block_scale_dequantize(input=t["tok"], descale=t["sfa"], block_size=[1, _BLOCK])
    t["fto"] = g.tensor(name="first_token_offset", dim=[groups + 1, 1, 1], stride=[1, 1, 1], data_type=DT.INT32, alignment_value=alignment)
    routing, mode = {}, cudnn.moe_grouped_matmul_mode.NONE
    if gather_rows is not None:
        mode = cudnn.moe_grouped_matmul_mode.GATHER
        t["idx"] = g.tensor(name="token_index", dim=[1, gather_rows, 1], stride=[gather_rows, 1, 1], data_type=DT.INT32)
        routing["token_index"] = t["idx"]
    products = []
    for i in range(gemms):
        w = g.tensor(name=f"w{i}", dim=[E, K, N], stride=[K * N, 1, K] if w_major == "k" else [K * N, N, 1], data_type=_CUDNN[data])
        t["w"].append(w)
        if sfb is not None:
            dims = list(sfb_dims[i]) if sfb_dims else [E, kb, nb]
            sf = g.tensor(name=f"SFB{i}", dim=dims, stride=[dims[1] * dims[2], 1, dims[1]], data_type=_CUDNN[sfb])
            t["sfb"].append(sf)
            w = g.block_scale_dequantize(input=w, descale=sf, block_size=[_BLOCK, _BLOCK])
        products.append(g.moe_grouped_matmul(tok, w, t["fto"], mode=mode, compute_data_type=DT.FLOAT, name=f"moe{i}", **routing))
    y = products[0] if gemms == 1 else g.mul(a=g.swish(input=products[0], name="silu"), b=products[1], name="mul")
    y.set_data_type(DT.BFLOAT16).set_output(True)
    t["out"] = y
    return g, t


def _dense_gated_graph(L, M, N, K, sfb_leads):
    """``swish(A @ B0) * (A @ B1)`` over ``L`` batches with linear scales; weight ``i`` declares ``sfb_leads[i]`` scale matrices."""
    g = cudnn.pygraph(io_data_type=DT.BFLOAT16, intermediate_data_type=DT.FLOAT, compute_data_type=DT.FLOAT)
    kb, nb = K // _BLOCK, N // _BLOCK
    a = g.tensor(name="A", dim=[L, M, K], stride=[M * K, K, 1], data_type=DT.FP8_E4M3)
    sfa = g.tensor(name="SFA", dim=[L, M, kb], stride=[M * kb, kb, 1], data_type=DT.FLOAT)
    a_d = g.block_scale_dequantize(input=a, descale=sfa, block_size=[1, _BLOCK])
    products = []
    for i, lead in enumerate(sfb_leads):
        w = g.tensor(name=f"B{i}", dim=[L, K, N], stride=[K * N, 1, K], data_type=DT.FP8_E4M3)
        sf = g.tensor(name=f"SFB{i}", dim=[lead, kb, nb], stride=[kb * nb, 1, kb], data_type=DT.FLOAT)
        products.append(g.matmul(A=a_d, B=g.block_scale_dequantize(input=w, descale=sf, block_size=[_BLOCK, _BLOCK]), name=f"mm{i}"))
    g.mul(a=g.swish(input=products[0], name="silu"), b=products[1], name="mul").set_data_type(DT.BFLOAT16).set_output(True)
    return g


def _offsets(rows, groups, seed=1):
    """Ragged groups that stop short of the capacity, with one empty group."""
    used = rows - rows // 8
    cuts = sorted(torch.randint(0, used, (groups - 1,), generator=torch.Generator().manual_seed(seed)).tolist())
    offsets = [0, *cuts, used]
    if groups > 2:
        offsets[2] = offsets[1]
    return offsets


def _data(dtype, shape, gen, dev):
    if dtype == "int8":
        return torch.randint(-8, 8, shape, device=dev, generator=gen, dtype=torch.int32).to(torch.int8)
    return (torch.randn(shape, device=dev, generator=gen) * 0.5).to(_TORCH[dtype])


def _scale(dtype, shape, gen, dev):
    if dtype == "int8":
        return torch.randint(1, 4, shape, device=dev, generator=gen, dtype=torch.int32).to(torch.int8)
    s = (torch.rand(shape, device=dev, generator=gen) * 1.5 + 0.25) * (torch.randint(0, 2, shape, device=dev, generator=gen) * 2 - 1)
    return s.to(_TORCH[dtype])


@dataclasses.dataclass
class _Moe:
    """One grouped product's tensors, its pack and its fp32 reference (``ref``, [routed rows, N])."""

    t: dict
    pack: dict
    tok: torch.Tensor  # [1, S, K]
    w_nk: list  # per weight [E, N, K], the kernel's order
    sa: "torch.Tensor | None"  # [1, S, K / 128]
    sb_nk: list  # per weight [E, N / 128, K / 128], the kernel's order
    idx: "torch.Tensor | None"
    offsets: list
    out: torch.Tensor
    ref: torch.Tensor


def _moe_case(E, S, N, K, *, data="fp8_e4m3", sfa="fp32", sfb="fp32", gather_rows=None, gemms=1, dev="cuda", seed=0, w_major="k"):
    """Graph, data and reference; the pack binds every tensor as declared (the weight scale as the declared transpose)."""
    g, t = _moe_graph(E, S, N, K, data=data, sfa=sfa, sfb=sfb, gather_rows=gather_rows, gemms=gemms, w_major=w_major)
    gen = torch.Generator(device=dev).manual_seed(seed)
    rows = S if gather_rows is None else gather_rows
    kb, nb = _cdiv(K, _BLOCK), _cdiv(N, _BLOCK)
    tok = _data(data, (1, S, K), gen, dev)
    sa = _scale(sfa, (1, S, kb), gen, dev) if sfa is not None else None
    tok_f = tok[0].double()
    if sa is not None:
        tok_f = tok_f * sa[0].double().repeat_interleave(_BLOCK, 1)[:, :K]
    w_nk, sb_nk, w_f = [], [], []
    for _ in range(gemms):
        w = _data(data, (E, N, K), gen, dev)
        w_nk.append(w)
        wf = w.double()
        if sfb is not None:
            sb = _scale(sfb, (E, nb, kb), gen, dev)
            sb_nk.append(sb)
            wf = wf * sb.double().repeat_interleave(_BLOCK, 1)[:, :N].repeat_interleave(_BLOCK, 2)[:, :, :K]
        w_f.append(wf)
    offsets = _offsets(rows, E)
    idx = torch.randint(0, S, (1, rows, 1), device=dev, generator=gen, dtype=torch.int32) if gather_rows is not None else None
    ref = torch.zeros(rows, N, device=dev, dtype=torch.float64)
    for gi in range(len(offsets) - 1):
        lo, hi = offsets[gi], offsets[gi + 1]
        if hi > lo:
            src = tok_f[idx[0, lo:hi, 0].long()] if idx is not None else tok_f[lo:hi]
            first = src @ w_f[0][gi % E].T
            ref[lo:hi] = first if gemms == 1 else torch.nn.functional.silu(first) * (src @ w_f[1][gi % E].T)
    out = torch.full((1, rows, N), float("nan"), device=dev, dtype=torch.bfloat16)
    pack = {t["tok"]: tok, t["fto"]: torch.tensor(offsets, dtype=torch.int32, device=dev).view(-1, 1, 1), t["out"]: out}
    if sa is not None:
        pack[t["sfa"]] = sa
    for i in range(gemms):
        pack[t["w"][i]] = w_nk[i].transpose(1, 2) if w_major == "k" else w_nk[i].transpose(1, 2).contiguous()
        if sfb is not None:
            pack[t["sfb"][i]] = sb_nk[i].transpose(1, 2)
    if idx is not None:
        pack[t["idx"]] = idx
    return g, _Moe(t, pack, tok, w_nk, sa, sb_nk, idx, offsets, out, ref)


def _check(case: _Moe, tol=2e-2):
    used = case.offsets[-1]
    got = case.out[0, :used].double()
    assert torch.isfinite(got).all()
    assert torch.isnan(case.out[0, used:].float()).all(), "a row past the routed rows was written"
    err = float((got - case.ref[:used]).norm() / case.ref[:used].norm())
    assert err < tol, f"relative error {err:.3e}"


# Grouped weight scale and weight: the axis order of a binding (host only)
#
# The graph declares the weight [E, K, N] and its scale [E, K blocks, N blocks]; the kernel reads [E, N, K] and
# [E, N blocks, K blocks]. The declared view is read in the graph's order. Any other buffer is read in the one order
# its shape fits; a scale buffer both orders fit (N blocks == K blocks) is refused unless it is the declared view: a
# contiguous [E, X, X] buffer is the graph-order matrix or the kernel-order one, and nothing in it says which.


def _scale_bindings(sb_nk):
    """The weight scales ``sb_nk`` ([E, N blocks, K blocks]) as the buffers a caller may bind: name -> buffer."""
    E, nb, kb = sb_nk.shape
    declared = sb_nk.transpose(1, 2)  # [E, kb, nb] with the declared strides
    padded = torch.zeros(E, kb, 2 * nb, dtype=sb_nk.dtype, device=sb_nk.device)
    padded[:, :, :nb] = declared
    fused = torch.zeros(E, 2 * nb, kb, dtype=sb_nk.dtype, device=sb_nk.device)  # the up half of a fused [gate; up] weight's scales
    fused[:, nb:] = sb_nk
    return {
        "declared": declared,
        "graph_contiguous": declared.contiguous(),
        "graph_padded_pitch": padded[:, :, :nb],
        "graph_expert_stride": fused[:, nb:].transpose(1, 2),
        "kernel_contiguous": sb_nk.contiguous(),
    }


def _resolve(g, pack, swap):
    """``_resolve_moe_variant_pack`` of the plan's chain and binding, without compiling a kernel."""
    chain, binding = analyze_with_binding(g)
    if swap:
        chain, binding = swap_ab_graph(chain), swap_ab_binding(binding)
    return C._resolve_moe_variant_pack(types.SimpleNamespace(chain=chain, binding=binding, _compiler=C), pack)


def _cpu_case(E, S, N, K, *, gather=False, seed=0, w_major="k"):
    return _moe_case(E, S, N, K, gather_rows=S if gather else None, dev="cpu", seed=seed, w_major=w_major)


def _bits(x):
    """One-byte values (fp8, int8) compared as their bytes."""
    return x.view(torch.uint8) if x.element_size() == 1 else x


def _same(a, b):
    """The same values in the same logical shape, whatever the strides."""
    return tuple(a.shape) == tuple(b.shape) and torch.equal(_bits(a), _bits(b))


_AXIS_CASES = [(gather, swap) for gather in (False, True) for swap in (False, True)]
_AXIS_IDS = [f"{'gather' if gather else 'none'}{'_swap' if swap else ''}" for gather, swap in _AXIS_CASES]


@requires_sm100_tree
@pytest.mark.parametrize("gather,swap", _AXIS_CASES, ids=_AXIS_IDS)
@pytest.mark.parametrize("layout", ["declared", "graph_contiguous", "graph_padded_pitch", "graph_expert_stride", "kernel_contiguous"])
def test_grouped_weight_scale_is_read_in_the_one_order_its_shape_fits(gather, swap, layout):
    """N = 256, K = 512: a [E, 4, 2] buffer is graph order and a [E, 2, 4] one kernel order, whatever their strides."""
    g, case = _cpu_case(4, 256, 256, 512, gather=gather)
    case.pack[case.t["sfb"][0]] = _scale_bindings(case.sb_nk[0])[layout]
    _a, _b, _o, _x, _f, sfa, sfb, *_ = _resolve(g, case.pack, swap)
    weight_scale = (sfa if swap else sfb)[0]
    assert _same(weight_scale, case.sb_nk[0]), f"{layout} was read as {tuple(weight_scale.shape)} {tuple(weight_scale.stride())}"


@requires_sm100_tree
@pytest.mark.parametrize("gather,swap", _AXIS_CASES, ids=_AXIS_IDS)
@pytest.mark.parametrize("layout", ["graph_contiguous", "graph_padded_pitch", "graph_expert_stride", "kernel_contiguous"])
def test_grouped_weight_scale_both_orders_fit_is_refused(gather, swap, layout):
    """N = K = 512, a 4 x 4 block grid: any binding but the declared view fits both orders and is refused, naming both."""
    g, case = _cpu_case(4, 256, 512, 512, gather=gather)
    bindings = _scale_bindings(case.sb_nk[0])
    case.pack[case.t["sfb"][0]] = bindings[layout]
    with pytest.raises(ValueError, match=r"weight scale\[0\].*both the graph's order.*and the kernel's.*declared"):
        _resolve(g, case.pack, swap)
    # The declared view is still read in the graph's order.
    case.pack[case.t["sfb"][0]] = bindings["declared"]
    _a, _b, _o, _x, _f, sfa, sfb, *_ = _resolve(g, case.pack, swap)
    assert _same((sfa if swap else sfb)[0], case.sb_nk[0])


def _weight_bindings(w_nk, major="k"):
    """The weight ``w_nk`` ([E, N, K]) as the buffers a caller may bind: name -> buffer, each contiguous along K (or N
    with ``major="n"``): the declared view, a padded graph-order view, the graph-order half of a buffer of twice the
    expert pitch, and the kernel-order tensor."""
    E, N, K = w_nk.shape
    if major == "k":
        padded = torch.zeros(E, N, 2 * K, dtype=w_nk.dtype, device=w_nk.device)
        padded[:, :, :K] = w_nk
        fused = torch.zeros(E, 2 * N, K, dtype=w_nk.dtype, device=w_nk.device)
        fused[:, N:] = w_nk
        return {
            "declared": w_nk.transpose(1, 2),
            "graph_padded_pitch": padded[:, :, :K].transpose(1, 2),
            "graph_expert_stride": fused[:, N:].transpose(1, 2),
            "kernel_contiguous": w_nk,
        }
    w_kn = w_nk.transpose(1, 2).contiguous()
    padded = torch.zeros(E, K, 2 * N, dtype=w_nk.dtype, device=w_nk.device)
    padded[:, :, :N] = w_kn
    fused = torch.zeros(E, 2 * K, N, dtype=w_nk.dtype, device=w_nk.device)
    fused[:, K:] = w_kn
    return {
        "declared": w_kn,
        "graph_padded_pitch": padded[:, :, :N],
        "graph_expert_stride": fused[:, K:],
        "kernel_contiguous": w_kn.transpose(1, 2),
    }


@requires_sm100_tree
@pytest.mark.parametrize("swap", [False, True], ids=["none", "swap"])
@pytest.mark.parametrize("major", ["k", "n"], ids=["k_major", "n_major"])
@pytest.mark.parametrize("square", [False, True], ids=["n256_k512", "n512_k512"])
@pytest.mark.parametrize("layout", ["declared", "graph_padded_pitch", "graph_expert_stride", "kernel_contiguous"])
def test_grouped_weight_is_read_in_the_order_its_shape_and_contiguous_axis_fit(swap, major, square, layout):
    """A K-major plan loads its weight along K, an N-major one along N: of the two orders a square [E, X, X] buffer
    fits, only one has its unit stride there. (Only the graph's order does for a padded or fused-half graph view, only
    the kernel's for an [E, N, K] tensor.)"""
    g, case = _cpu_case(4, 256, 512 if square else 256, 512, w_major=major)
    case.pack[case.t["w"][0]] = _weight_bindings(case.w_nk[0], major)[layout]
    a_bufs, b_bufs, *_ = _resolve(g, case.pack, swap)
    weight = (a_bufs if swap else b_bufs)[0]
    assert _same(weight, case.w_nk[0]), f"{layout} was read as {tuple(weight.shape)} {tuple(weight.stride())}"
    assert weight.stride(2 if major == "k" else 1) == 1


@requires_sm100_tree
@pytest.mark.parametrize("swap", [False, True], ids=["none", "swap"])
def test_grouped_weight_of_another_contiguous_axis_is_refused(swap):
    """A gated pair's second weight bound N-contiguous to a K-major plan is refused, naming the axis: the multi-GEMM
    launch checks only the first weight's layout, and the kernel loads every weight along K."""
    g, case = _moe_case(4, 256, 256, 512, gemms=2, dev="cpu")
    case.pack[case.t["w"][1]] = case.w_nk[1].transpose(1, 2).contiguous().transpose(1, 2)  # [E, N, K], N-contiguous
    with pytest.raises(ValueError, match=r"weight operand\[1\] must be contiguous along K"):
        _resolve(g, case.pack, swap)


# A gated pair's weight scale counts (planning: no kernel is built; check_support reads the GPU's occupancy)
#
# A general template renders ONE count of weight scale matrices for both GEMMs of a gated pair (sfb_batch_broadcast,
# or sfa_batch_broadcast when swapped): both weight scale tensors must declare it, at plan time.


@requires_sm100
@pytest.mark.parametrize("swap", [False, True], ids=["none", "swap"])
@pytest.mark.parametrize(
    "dims", [((4, 4, 2), (1, 4, 2)), ((1, 4, 2), (4, 4, 2)), ((4, 4, 2), (4, 5, 2))], ids=["per_expert_then_shared", "shared_then_per_expert", "malformed"]
)
def test_gated_pair_weight_scales_declare_one_shape(swap, dims):
    g, _ = _moe_graph(4, 256, 256, 512, gemms=2, sfb_dims=dims)
    chain = analyze(g)
    config = by_name(_SWAP_32 if swap else _TILE_128)
    if swap:
        config = dataclasses.replace(config, swap_ab=True)
    # The kernel's chain of the plan: probe_chain swaps the graph's chain itself under a swap config.
    reason = general_glu_pair_reject(swap_ab_graph(chain) if swap else chain)
    assert reason is not None and "one shape" in reason and str(list(dims[0])) in reason and str(list(dims[1])) in reason, reason
    with pytest.raises(NotImplementedError, match="one shape"):
        C.probe_chain(chain, config)
    with pytest.raises(NotImplementedError, match="one shape"):
        FrostGemmEngine().check_support(g)


@requires_sm100
@pytest.mark.parametrize("leads", [(2, 1), (1, 2)], ids=["per_batch_then_shared", "shared_then_per_batch"])
def test_dense_gated_pair_weight_scales_declare_one_shape(leads):
    g = _dense_gated_graph(2, 256, 128, 512, leads)
    reason = general_glu_pair_reject(analyze(g))
    assert reason is not None and "one shape" in reason, reason
    with pytest.raises(NotImplementedError, match="one shape"):
        FrostGemmEngine().check_support(g)


@requires_sm100
@pytest.mark.parametrize("lead", [1, 4], ids=["shared", "per_expert"])
def test_gated_pair_weight_scales_of_one_shape_plan(lead):
    """The control: both weight scales declare the same count, which the kernel renders."""
    g, _ = _moe_graph(4, 256, 256, 512, gemms=2, sfb_dims=((lead, 4, 2), (lead, 4, 2)))
    chain = analyze(g)
    assert general_glu_pair_reject(chain) is None
    assert general_glu_pair_reject(swap_ab_graph(chain)) is None
    FrostGemmEngine().check_support(g)


# Rule 7: the DSL target on cc 10.7 (host only)


@requires_sm100_tree
def test_frost_gemm_declines_cc_10_7_under_a_dsl_without_its_target(monkeypatch):
    """The version floor admits nvidia-cutlass-dsl 4.7.x, whose Arch has no sm_107a: the support check names the
    target and the version it found instead of admitting a plan whose compile fails inside the DSL."""
    g, _ = _moe_graph(4, 256, 256, 512)
    engine = FrostGemmEngine()
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.7.0")))
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: False)
    monkeypatch.setattr(C, "_current_arch", lambda device=None: 107)
    with pytest.raises(NotImplementedError, match=r"sm_107a.*4\.7\.0"):
        C.probe_cutedsl()
    with pytest.raises(NotImplementedError, match=r"frost_gemm: .*sm_107a.*4\.7\.0"):
        engine.check_support(g)
    facts = heuristics.analyze_facts(g)
    assert facts is not None and heuristics.recommend("A", facts, {"frost_gemm": 0}) == []
    # The same DSL serves the other SM100-class parts.
    for arch in (100, 103):
        monkeypatch.setattr(C, "_current_arch", lambda device=None, _a=arch: _a)
        C.probe_cutedsl()
    # A DSL with the target serves cc 10.7.
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.8.0")))
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    monkeypatch.setattr(C, "_current_arch", lambda device=None: 107)
    C.probe_cutedsl()
    # The version floor answers first: a DSL below it is not asked for its targets.
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.6.0")))

    def unreachable():
        raise AssertionError("the target probe ran although the version floor declined")

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", unreachable)
    with pytest.raises(NotImplementedError, match=r"requires nvidia-cutlass-dsl >= 4\.7\.0; found 4\.6\.0"):
        C.probe_cutedsl()


# The token capacity of a weight-by-token plan (host only)


@requires_sm100_tree
def test_weight_by_token_capacity_refusal_names_the_offset_alignment():
    """A swapped plan stores its output in chunks of tokens, as many as the graph's token count, the epilogue vector and
    the alignment_value of first_token_offset all divide: a call with another token count is refused, and the refusal
    says what to change."""
    E, S, N, K = 4, 1024, 256, 512
    g, _ = _moe_graph(E, S, N, K, alignment=128)
    chain = swap_ab_graph(analyze(g))
    config = dataclasses.replace(by_name(_SWAP_32), swap_ab=True)
    multiple = math.gcd(C._epi_vec_bytes(chain, config) // 2, 128, S)
    assert multiple > 2, "this shape needs a chunk alignment that a token count can break"
    rows = S - multiple // 2
    holder = types.SimpleNamespace(chain=chain, config=config)
    weights = [torch.zeros(E, N, K, dtype=torch.float8_e4m3fn)]
    tokens = [torch.zeros(1, rows, K, dtype=torch.float8_e4m3fn)]
    outputs = [torch.zeros(1, N, rows, dtype=torch.bfloat16)]
    offsets = torch.zeros(E + 1, 1, 1, dtype=torch.int32)
    with pytest.raises(ValueError, match=rf"token count {rows} .*multiple of {multiple}.*alignment_value"):
        C._launch_moe_swap_ab(holder, weights, tokens, outputs, [], offsets, [], [], (N, rows, K), None, None)


# End to end (GPU): the bindings above, launched


@requires_sm100
@pytest.mark.parametrize("swap", [False, True], ids=["fwd", "fwd_swap_ab"])
@pytest.mark.parametrize("layout", ["graph_contiguous", "graph_padded_pitch", "graph_expert_stride", "kernel_contiguous"])
def test_grouped_weight_scale_bindings_of_a_rectangular_grid_run(swap, layout):
    g, case = _moe_case(4, 256, 256, 512)
    plan = Plan(g, config=by_name(_SWAP_32 if swap else _TILE_128), swap_ab=swap)
    assert select_template(plan.chain, plan._compiled.config).file == _TEMPLATES[(False, swap)]
    case.pack[case.t["sfb"][0]] = _scale_bindings(case.sb_nk[0])[layout]
    plan(case.pack)
    torch.cuda.synchronize()
    _check(case)


def _offset_view(values, elements=1):
    """``values`` copied into a buffer ``elements`` past an allocation's start: aligned to its elements, not 16 bytes."""
    storage = torch.empty(values.numel() + elements, dtype=values.dtype, device=values.device)
    view = storage[elements:].view(values.shape)
    view.copy_(values)
    return view


@requires_sm100
@pytest.mark.parametrize("swap", [False, True], ids=["fwd", "fwd_swap_ab"])
def test_grouped_linear_scales_need_only_element_alignment(swap):
    """The promotion warps load a linear scale one element at a time (the kernel assumes element alignment), as the
    dense plan's LinearScaleRule holds it: a 4-byte-aligned fp32 view serves, token scale and weight scale alike."""
    g, case = _moe_case(4, 256, 256, 512)
    plan = Plan(g, config=by_name(_SWAP_32 if swap else _TILE_128), swap_ab=swap)
    sa = _offset_view(case.sa)
    sb = _offset_view(case.sb_nk[0]).transpose(1, 2)
    assert sa.data_ptr() % 16 and sb.data_ptr() % 16
    case.pack[case.t["sfa"]], case.pack[case.t["sfb"][0]] = sa, sb
    plan(case.pack)
    torch.cuda.synchronize()
    _check(case)


@requires_sm100
def test_grouped_weight_scale_of_a_square_grid_is_refused_before_the_launch():
    g, case = _moe_case(4, 256, 512, 512)
    plan = Plan(g, config=by_name(_TILE_128))
    case.pack[case.t["sfb"][0]] = _scale_bindings(case.sb_nk[0])["graph_contiguous"]
    with pytest.raises(ValueError, match=r"both the graph's order.*and the kernel's"):
        plan(case.pack)
    torch.cuda.synchronize()
    assert torch.isnan(case.out.float()).all()
    case.pack[case.t["sfb"][0]] = case.sb_nk[0].transpose(1, 2)
    plan(case.pack)
    torch.cuda.synchronize()
    _check(case)


@requires_sm100
@pytest.mark.parametrize(
    "N,layout,refused",
    [
        (512, "declared", False),
        (512, "graph_contiguous", True),
        (512, "kernel_contiguous", True),
        (256, "graph_contiguous", False),
        (256, "kernel_contiguous", False),
    ],
)
def test_grouped_weight_scale_bindings_through_graph_execute(N, layout, refused):
    """The public pack hands frost_gemm a buffer of the declared extents with its own strides, and describes a dense
    buffer of other extents (a kernel-order [E, N blocks, K blocks] one when N != K) as the declaration: either way the
    plan reads the order that the call's shape settles, and refuses a square grid's buffer that only the declared
    strides could settle."""
    from cudnn.engines.manifest import MANIFEST
    from cudnn.gemm.frost.knobs import GemmKnobs

    g, case = _moe_case(4, 256, N, 512)
    engine_id = next(row.engine_id for row in MANIFEST if row.name == "frost_gemm")
    g.validate()
    g.build_operation_graph()
    g.create_execution_plan(engine_id, GemmKnobs.from_config(by_name(_TILE_128)).to_public())
    g.check_support()
    g.build_plans()
    case.pack[case.t["sfb"][0]] = _scale_bindings(case.sb_nk[0])[layout]
    workspace = torch.empty(max(1, g.get_workspace_size()), dtype=torch.uint8, device="cuda")
    if refused:
        with pytest.raises(ValueError, match=r"both the graph's order.*and the kernel's"):
            g.execute(case.pack, workspace)
        torch.cuda.synchronize()
        assert torch.isnan(case.out.float()).all()
        return
    g.execute(case.pack, workspace)
    torch.cuda.synchronize()
    _check(case)


# A one-sided graph whose unscaled operand is bf16, fp16 or int8 (GPU)
#
# The analyzer gives a raw FP8 operand the other's scale metadata as a unit scale (fake_dequant); a raw 16-bit or
# int8 operand has neither a scale tensor nor that metadata, and the plain grouped launch takes no scale for it.

_ONE_SIDED = [
    pytest.param(dict(data="bf16", sfb=None), id="bf16_raw_weight"),
    pytest.param(dict(data="fp16", sfa=None), id="fp16_raw_token"),
    pytest.param(dict(data="int8", sfb=None, gather_rows=256), id="int8_raw_weight_gather", marks=requires_int8_mma),
    pytest.param(dict(data="int8", sfa=None), id="int8_raw_token", marks=requires_int8_mma),
    pytest.param(dict(data="bf16", sfb=None, gemms=2), id="bf16_raw_weights_gated"),
]


@requires_sm100
@pytest.mark.parametrize("kw", _ONE_SIDED)
def test_one_sided_block_scale_with_an_unscaled_16_or_8_bit_operand(kw):
    g, case = _moe_case(4, 256, 256, 512, **kw)
    plan = Plan(g, config=by_name(_TILE_128))
    assert select_template(plan.chain, plan._compiled.config).file == _TEMPLATES[("gather_rows" in kw, False)]
    plan(case.pack)
    torch.cuda.synchronize()
    _check(case)


# A uint8 buffer for an int8 operand or scale (GPU)
#
# The kernel's argument check takes a uint8 buffer for an Int8 parameter, and the kernel reads its bytes as signed:
# a call that hands the plan its buffers refuses every buffer whose dtype is not the one the graph declares.
# graph.execute is not such a call: its pack describes a buffer of the declared extents and element width as the
# declared tensor, so the plan receives an int8 operand and reads the bytes as signed, as a backend engine does.


@requires_sm100
@requires_int8_mma
@pytest.mark.parametrize("operand", ["token", "weight"])
def test_grouped_int8_operand_refuses_a_uint8_buffer(operand):
    g, case = _moe_case(4, 256, 256, 512, data="int8")
    plan = Plan(g, config=by_name(_TILE_128))
    key = case.t["tok"] if operand == "token" else case.t["w"][0]
    declared = case.pack[key]
    case.pack[key] = declared.view(torch.uint8)
    with pytest.raises(ValueError, match=rf"{operand} operand\[0\] must be int8.*got uint8"):
        plan(case.pack)
    torch.cuda.synchronize()
    assert torch.isnan(case.out.float()).all()
    case.pack[key] = declared
    plan(case.pack)
    torch.cuda.synchronize()
    _check(case)


@requires_sm100
@requires_int8_mma
@pytest.mark.parametrize("operand", ["A", "B"])
def test_dense_int8_operand_refuses_a_uint8_buffer(operand):
    """The dense template's launch path holds an int8 operand to int8 too (the same argument check, the same MMA)."""
    M, N, K = 256, 256, 512
    g = cudnn.pygraph(io_data_type=DT.BFLOAT16, intermediate_data_type=DT.FLOAT, compute_data_type=DT.FLOAT)
    a_t = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1], data_type=DT.INT8)
    b_t = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K], data_type=DT.INT8)
    sfa_t = g.tensor(name="SFA", dim=[1, M, K // _BLOCK], stride=[M * K // _BLOCK, K // _BLOCK, 1], data_type=DT.FLOAT)
    sfb_t = g.tensor(name="SFB", dim=[1, K // _BLOCK, N // _BLOCK], stride=[K * N // _BLOCK**2, 1, K // _BLOCK], data_type=DT.FLOAT)
    lhs = g.block_scale_dequantize(input=a_t, descale=sfa_t, block_size=[1, _BLOCK])
    rhs = g.block_scale_dequantize(input=b_t, descale=sfb_t, block_size=[_BLOCK, _BLOCK])
    out_t = g.matmul(A=lhs, B=rhs, name="mm")
    out_t.set_output(True).set_data_type(DT.BFLOAT16)
    plan = Plan(g, config=by_name(_TILE_128))
    assert select_template(plan.chain, plan._compiled.config).file == "sm100_general_block_scale_matmul.py"
    gen = torch.Generator(device="cuda").manual_seed(3)
    a = _data("int8", (1, M, K), gen, "cuda")
    w_nk = _data("int8", (1, N, K), gen, "cuda")
    sa = _scale("fp32", (1, M, K // _BLOCK), gen, "cuda")
    sb_nk = _scale("fp32", (1, N // _BLOCK, K // _BLOCK), gen, "cuda")
    ref = (a[0].double() * sa[0].double().repeat_interleave(_BLOCK, 1)) @ (
        w_nk[0].double() * sb_nk[0].double().repeat_interleave(_BLOCK, 0).repeat_interleave(_BLOCK, 1)
    ).T
    out = torch.full((1, M, N), float("nan"), device="cuda", dtype=torch.bfloat16)
    pack = {a_t: a, b_t: w_nk.transpose(1, 2), sfa_t: sa, sfb_t: sb_nk.transpose(1, 2), out_t: out}
    key = a_t if operand == "A" else b_t
    declared = pack[key]
    pack[key] = declared.view(torch.uint8)
    with pytest.raises(ValueError, match=rf"{operand} operand\[0\] must be int8.*got uint8"):
        plan(pack)
    torch.cuda.synchronize()
    assert torch.isnan(out.float()).all()
    pack[key] = declared
    plan(pack)
    torch.cuda.synchronize()
    err = float((out[0].double() - ref).norm() / ref.norm())
    assert err < 1e-2, f"relative error {err:.3e}"


@requires_sm100
@requires_int8_mma
def test_graph_execute_reads_a_uint8_buffer_bound_for_an_int8_operand_as_int8():
    """The public pack takes a buffer of the declared extents and element width as the declared tensor (FlashInfer binds
    its FP8 and FP4 blocks as uint8), so frost_gemm gets an int8 token operand and reads its bytes as signed: what a
    backend engine, which takes the pointer, reads too (python/cudnn/AGENTS.md Rule 9)."""
    from cudnn.engines.manifest import MANIFEST
    from cudnn.gemm.frost.knobs import GemmKnobs

    g, case = _moe_case(4, 256, 256, 512, data="int8")
    engine_id = next(row.engine_id for row in MANIFEST if row.name == "frost_gemm")
    g.validate()
    g.build_operation_graph()
    g.create_execution_plan(engine_id, GemmKnobs.from_config(by_name(_TILE_128)).to_public())
    g.check_support()
    g.build_plans()
    case.pack[case.t["tok"]] = case.pack[case.t["tok"]].view(torch.uint8)
    workspace = torch.empty(max(1, g.get_workspace_size()), dtype=torch.uint8, device="cuda")
    g.execute(case.pack, workspace)
    torch.cuda.synchronize()
    _check(case)


@requires_sm100
@pytest.mark.parametrize("swap", [False, True], ids=["fwd", "fwd_swap_ab"])
def test_grouped_int8_scale_refuses_a_uint8_buffer(swap):
    g, case = _moe_case(4, 256, 256, 512, sfa="int8")
    plan = Plan(g, config=by_name(_SWAP_32 if swap else _TILE_128), swap_ab=swap)
    case.pack[case.t["sfa"]] = case.sa.view(torch.uint8)
    with pytest.raises(ValueError, match=r"scale\[0\].*int8.*got uint8"):
        plan(case.pack)
    torch.cuda.synchronize()
    assert torch.isnan(case.out.float()).all()
    case.pack[case.t["sfa"]] = case.sa
    plan(case.pack)
    torch.cuda.synchronize()
    _check(case)
