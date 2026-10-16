# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``cudnn.triangle_multiplicative_update`` -- the three-launch API behind cuEquivariance's signature -- against the fp64 reference, the
hand-chained kernels (bitwise), CUDA-graph capture, the execute-path contract (no allocation, no synchronisation), and its typed declines;
plus the GPU-free contract tests (lazy export in every import order, the signature pin, the weight layouts, the traffic model; the padded
contraction graph and the K2 operand views are pinned in ``test_contraction_graph.py``).  The module is named ``test_trimul_api.py`` so a
whole-tree session can hold it beside ``triangle_attention/cutedsl/test_api.py`` (two ``test_api.py`` basenames under directories without
``__init__.py`` make pytest drop whichever is collected second).

Tolerance of the end-to-end oracle (derived, stated).  The reference inserts the kernels' bf16 rounding points (``rounding="kernel"``:
``z_in``, ``a``, ``b``, ``y``, ``u`` each rounded once), so what remains is (i) the fp32 accumulation of K1 / K2 / K3 against fp64 (K1 / K3:
``<= K * 2^-24`` relative, negligible unless it flips a bf16 rounding of an operand -- ~1 % of the ``a`` / ``b`` elements move by one ulp,
2^-8 relative; K2: N products of two bf16 values summed in fp32, ``<= N * 2^-24`` relative on ``y`` -- flips the bf16 rounding of a few
percent of the ``y`` elements by one ulp), (ii) the sigmoid through ``tanh.approx`` in K1 and K3 (``<= 2^-12`` on sigma), (iii) the single
bf16 rounding of the output (half-ulp = 2^-9 relative).  The propagated ulp flips of ``a`` / ``b`` into ``y`` and of ``y`` into the output
GEMM add roughly ``2^-8 * sqrt(f * K) / sqrt(K)`` ~ 2^-8 * 0.3 relative at a 10 % flip fraction.  Together, the BULK bound: ``|diff| <= 4e-3 + |ref| / 64``
(two bf16 half-ulps of the output plus one propagated half-ulp, plus an absolute floor for (ii)).  The propagated term has a TAIL the bulk
bound does not cover: a flipped ``y`` element moves the output by ``ulp(y) * rstd * |W_o|`` ~ 2^-8 * 2 * 0.09 ~ 7e-4 per flip (absolute, not
relative to the output), several flips per row add in quadrature, and over tens of millions of cells the 4-5 sigma events reach 1-1.6e-2.
So the oracle is three-part: the bulk bound on all but a pinned fraction of the cells (``_TAIL_FRAC_MAX`` = 2e-5; measured 7e-7 .. 7.6e-6),
a cap ``2e-2 + |ref| / 64`` on EVERY cell, and the relative RMS below 3e-3 (measured 1.68e-3 = the output rounding alone).  The test prints
the measured figures.  The stage-level tolerances (``|ref| / 128``, zero cells beyond) live in the K1 / K3 modules, whose references consume
the kernels' own operands.

The kernels are rendered for Rubin (cc 10.7) and Blackwell SM100 / SM103 (cc 10.0 / 10.3); the accept tests skip elsewhere with the reason stated.  Every launch is a single process under the
harness's ``timeout`` (exit 124 = hang).
"""

from __future__ import annotations

import inspect
import json
import math
import subprocess
import sys

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Triangle multiplicative update tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

from cudnn.api_base import TensorDesc  # noqa: E402
from cudnn.triangle_multiplicative_update import api as tapi  # noqa: E402
from cudnn.triangle_multiplicative_update import reference as tref  # noqa: E402
from cudnn.triangle_multiplicative_update.kernels import contraction as k2  # noqa: E402
from cudnn.triangle_multiplicative_update.kernels import trimul_proj as k1  # noqa: E402
from cudnn.triangle_multiplicative_update.kernels import trimul_tail as k3  # noqa: E402

_SM107 = (10, 7)
#: the compute capabilities the API serves: the substrate's rendered bodies (Rubin 10.7, Blackwell 10.0 / 10.3)
_SERVED = tapi._SERVED_CCS


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


def _arch() -> str:
    from cudnn.gemm.frost import ln_prologue as _sub

    return _sub.body_arch_for_cc(_cc()) if _cc() in _sub.ARCHS_BY_CC else "sm107"


requires_served_arch = pytest.mark.skipif(
    _cc() not in _SERVED, reason=f"triangle_multiplicative_update is rendered for compute capabilities {_SERVED} only; found {_cc()}"
)

CUEQ_PARAMS = [
    "x",
    "direction",
    "mask",
    "norm_in_weight",
    "norm_in_bias",
    "p_in_weight",
    "p_in_bias",
    "g_in_weight",
    "g_in_bias",
    "norm_out_weight",
    "norm_out_bias",
    "p_out_weight",
    "p_out_bias",
    "g_out_weight",
    "g_out_bias",
    "eps",
    "precision",
]


# ---------------------------------------------------------------------------
# GPU-free: the export, the signature, the declines, the models, the padded contraction graph, the weight layouts, the reference
# ---------------------------------------------------------------------------
def _probe(code: str) -> str:
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-1500:]
    return run.stdout


@pytest.mark.parametrize("order", ["symbol_first", "package_first", "kernels_first"])
def test_lazy_export_is_callable_in_every_import_order(order):
    """The function shares its package's name: after `import cudnn.triangle_multiplicative_update...` the attribute is the PACKAGE (Python binds a
    submodule on its parent, bypassing the lazy table) -- the package module is callable, so every spelling runs the function."""
    code = f"""
import cudnn, inspect
order = {order!r}
if order == "symbol_first":
    f = cudnn.triangle_multiplicative_update
elif order == "package_first":
    import cudnn.triangle_multiplicative_update
    f = cudnn.triangle_multiplicative_update
else:
    import cudnn.triangle_multiplicative_update.kernels.contraction
    f = cudnn.triangle_multiplicative_update
assert callable(f), type(f)
from cudnn.triangle_multiplicative_update import triangle_multiplicative_update as fn
assert callable(fn)
print(type(f).__name__)
"""
    out = _probe(code)
    assert out.strip() in ("function", "_CallablePackage"), out


def test_package_import_pulls_no_torch():
    out = _probe("import sys, cudnn\nimport cudnn.triangle_multiplicative_update\nprint('torch' in sys.modules)")
    assert out.strip() == "False", out


def test_signature_mirrors_cueq_0_12_0():
    """cuEquivariance's parameter names, order and defaults, verbatim; the two additions are keyword-only and last (append-only API)."""
    sig = inspect.signature(tapi.triangle_multiplicative_update)
    names = list(sig.parameters)
    assert names[: len(CUEQ_PARAMS)] == CUEQ_PARAMS
    assert names[len(CUEQ_PARAMS) :] == ["out", "stream"]
    p = sig.parameters
    assert p["direction"].default == "outgoing" and p["eps"].default == 1e-5 and p["precision"].default is None
    assert all(p[n].default is None for n in CUEQ_PARAMS if n not in ("x", "direction", "eps", "precision"))
    assert p["out"].kind is inspect.Parameter.KEYWORD_ONLY and p["stream"].kind is inspect.Parameter.KEYWORD_ONLY
    assert tapi.CUEQ_SIGNATURE_VERSION == "0.12.0"


def _desc(N=256, B=1, cz=128, dtype=torch.bfloat16, device="cuda"):
    return TensorDesc(dtype=dtype, shape=(B, N, N, cz), stride=(N * N * cz, N * cz, cz, 1), stride_order=(3, 2, 1, 0), device=device)


@pytest.mark.parametrize(
    "kw,exc",
    [
        (dict(cz=100, c=100), ValueError),  # cuEquivariance's own admission rule: the pair width is a multiple of 32
        (dict(cz=160, c=160), NotImplementedError),  # a multiple of 32 the substrate does not serve
        (dict(cz=384, c=384), NotImplementedError),  # withdrawn: the K = 384 renderings spill through a 1.7 KB stack and have no oracle cell
        (dict(cz=256, c=128), NotImplementedError),  # c != c_z (one K for both output GEMMs)
        (dict(N=32), NotImplementedError),  # below the output stage's 64-row MN-major box
        (dict(N=217, direction="incoming"), NotImplementedError),  # the incoming direction's contiguous extent must be a multiple of 8
        (dict(dtype=torch.float16), NotImplementedError),  # bf16 only
    ],
)
def test_check_support_declines_typed(kw, exc):
    kw = dict(kw)
    direction = kw.pop("direction", "outgoing")
    c = kw.pop("c", kw.get("cz", 128))
    plan = tapi.TriangleMultiplicativeUpdateFwd(_desc(**kw), direction, c_hidden=c)
    with pytest.raises(exc):
        plan.check_support()
    with pytest.raises(ValueError):
        tapi.TriangleMultiplicativeUpdateFwd(_desc(), "sideways", c_hidden=128)


def test_check_support_declines_a_foreign_arch_typed():
    plan = tapi.TriangleMultiplicativeUpdateFwd(_desc(), "outgoing", c_hidden=128)
    assert _SERVED == ((10, 0), (10, 3), (10, 7))
    if _cc() in _SERVED:
        assert plan.check_support()
        assert plan.arch == _arch() and plan.cc == _cc()
    else:
        with pytest.raises(NotImplementedError, match="10.0, 10.3, 10.7"):
            plan.check_support()
        assert plan.arch is None  # never resolved on a part without a body


def test_check_support_declines_a_knob_set_the_arch_body_cannot_render_typed():
    """A caller's knobs that land the output stage in the sm100 body's unallocatable class (K = 256 register-form LayerNorm + a packed epilogue) are a
    typed NotImplementedError of check_support on a Blackwell device -- never a ptxas error inside compile(); the same declaration passes on Rubin
    (the body renders it, spilling) and is declined on neither for the heuristics' own pick."""
    knobs = tapi.TriMulKnobs(ln_pass_form="registers", epi_math="packed")
    plan = tapi.TriangleMultiplicativeUpdateFwd(_desc(N=256, cz=256), "outgoing", c_hidden=256, knobs=knobs)
    assert [p.ln_pass_form for p in plan._stage_params()] == ["registers", "registers"]
    if _cc() in _SERVED and _arch() == "sm100":
        with pytest.raises(NotImplementedError, match="C7600"):
            plan.check_support()
        assert tapi.TriangleMultiplicativeUpdateFwd(_desc(N=256, cz=256), "outgoing", c_hidden=256).check_support()  # the pick: a re-read form
    elif _cc() in _SERVED:
        assert plan.check_support()
    else:
        with pytest.raises(NotImplementedError, match="10.0, 10.3, 10.7"):
            plan.check_support()


def test_heuristics_sm100_column_is_the_measured_gb200_table():
    """The SM100 column (compute capability 10.0 / 10.3) = the table MEASURED on a 152-SM GB200 perf node (the sweep of record: sweep of record: a 152-SM GB200 perf node, 2026-10-14, every knob as a single-axis arm against the seeded pick, 16-20 arms per cell, graph replay >= 100 x 3, in-process control twins, the fp64 oracle on every arm):
    every (width, N bucket) pick and every refinement below is the literal record the heuristics ship; the N < 256 floors (129 / 0) carry the Rubin
    picks with the recorded contraction tile (unmeasured on SM100, stated); the Rubin table is unchanged for cc 10.7 and for ``None``; the register-split
    hint rows follow the column; a caller's overrides merge over the SM100 column exactly as over the Rubin one."""
    from dataclasses import replace

    from cudnn.triangle_multiplicative_update import heuristics as th

    assert (
        th.arch_column_for_cc(None),
        th.arch_column_for_cc((10, 0)),
        th.arch_column_for_cc((10, 3)),
        th.arch_column_for_cc((10, 7)),
        th.arch_column_for_cc((8, 0)),
    ) == (
        "sm107",
        "sm100",
        "sm100",
        "sm107",
        "sm107",
    )
    # the bucket picks: (width, N floor) -> (k2 tile, K1 order, K3 prologue, ring depth, LayerNorm form, epilogue form, work item, the item's B-ring depth
    # (None = the shipped ring; 4 on every c 256 pick since the 2026-10-14 A/B: the second resident A slot on the 227 KiB part))
    picks = {
        (128, 1024): (k2.MULTICAST_A2_TILE_CONFIG, "major", "transpose", 2, "registers", "packed", "block", None),
        (128, 768): (k2.DESIGN_TILE_CONFIG, "major", "transpose", 1, "registers", "packed", "block", None),
        (128, 512): (k2.MULTICAST_A2_TILE_CONFIG, "major", "transpose", 1, "registers", "packed", "block", None),
        (128, 384): (k2.SMALL_N_TILE_CONFIG, "major", "transpose", 2, "smem_rolled_pf", "packed", "block", None),
        (128, 256): (k2.DESIGN_TILE_CONFIG, "major", "transpose", 2, "smem_rolled", "packed", "block", None),
        (128, 129): (k2.DESIGN_TILE_CONFIG, "interleaved", "transpose", 1, "registers", "packed", "", None),
        (128, 0): (k2.SMALL_N_TILE_CONFIG, "interleaved", "transpose", 2, "registers", "packed", "", None),
        (256, 1024): (k2.MULTICAST_A2_TILE_CONFIG, "major", "mn_major", 1, "smem_rolled_pf", "packed", "block", 4),
        (256, 768): (k2.DESIGN_TILE_CONFIG, "major", "mn_major", 1, "smem_rolled_pf", "packed", "block", 4),
        (256, 512): (k2.MULTICAST_A2_TILE_CONFIG, "major", "mn_major", 1, "smem_rolled_pf", "packed", "block", 4),
        (256, 384): (k2.SMALL_N_TILE_CONFIG, "major", "mn_major", 1, "smem_rolled_pf", "packed", "block", 4),
        (256, 256): (k2.DESIGN_TILE_CONFIG, "major", "mn_major", 1, "smem_rolled_pf", "packed", "block", 4),
        (256, 129): (k2.DESIGN_TILE_CONFIG, "major", "mn_major", 1, "smem_rolled_pf", "packed", "block", 4),
        (256, 0): (k2.SMALL_N_TILE_CONFIG, "major", "mn_major", 2, "smem_rolled_pf", "packed", "block", 4),
    }
    # the refinements: (width, N floor, batch bucket, direction or None) -> the ONE axis overridden
    refinements = [
        (128, 0, 1, "incoming", {"clc_stages": 1}),
        (128, 129, 1, "outgoing", {"ln_once": "block", "k1_pass_order": "major"}),
        (128, 129, 4, None, {"ln_once": "block", "k1_pass_order": "major"}),
        (128, 0, 4, None, {"ln_once": "block", "k1_pass_order": "major"}),
        (128, 256, 1, "incoming", {"ln_pass_form": "smem_rolled_pf"}),
        (128, 256, 4, None, {"ln_pass_form": "registers"}),
        (128, 384, 4, None, {"ln_pass_form": "registers"}),
        (128, 512, 4, None, {"clc_stages": 2}),
        (256, 0, 4, None, {"clc_stages": 1}),
    ]
    axes = ("k2_tile", "k1_pass_order", "k3_prologue", "clc_stages", "ln_pass_form", "epi_math", "ln_once", "ln_once_b_stages")
    for c in (128, 256):
        for n in (33, 128, 217, 256, 384, 512, 768, 1024, 1536, 2048):
            for b in (1, 4):
                for d in k2.DIRECTIONS:
                    rubin, sm100 = th.recommend(n, c, b, d), th.recommend(n, c, b, d, cc=(10, 0))
                    assert sm100.is_complete() and th.recommend(n, c, b, d, cc=(10, 7)) == rubin and th.recommend(n, c, b, d, cc=None) == rubin
                    floor = next(f for f in (1024, 768, 512, 384, 256, 129, 0) if n >= f)
                    want = dict(zip(axes, picks[(c, floor)]))
                    for w_, f_, b_, d_, over in refinements:
                        if (w_, f_) == (c, floor) and b_ == th.batch_bucket(b) and (d_ is None or d_ == d):
                            want.update(over)
                    got = {a: (getattr(sm100, a) or "") if a == "ln_once" else getattr(sm100, a) for a in axes}
                    assert got == want, (n, c, b, d, got, want)
                    assert sm100.k_block == th.default_k_block(n), (n, c, b, d)
                    assert th.recommend(n, c, b, d, cc=(10, 3)) == sm100
    # the caller's overrides merge over the SM100 column exactly as over the Rubin one
    base = th.recommend(1024, 128, 1, "outgoing", cc=(10, 0))
    r = th.resolve(1024, 128, 1, "outgoing", knobs=th.TriMulKnobs(clc_stages=4), cc=(10, 0))
    assert r.clc_stages == 4 and replace(r, clc_stages=base.clc_stages) == base and r.is_complete()
    assert th.resolve(1024, 128, 1, "outgoing", k2_tile=k2.DESIGN_TILE_CONFIG, cc=(10, 0)).k2_tile == k2.DESIGN_TILE_CONFIG
    assert th.k2_tile_for(1024, 128, 1, "outgoing", cc=(10, 0)) == base.k2_tile
    # the hint rows per column: the Rubin rows for 10.7 / None (the per-tile item), the row measured on a GB200 for 10.0 / 10.3 (the SM100 column's own
    # register-form K1 pick under the block item with the biases), nothing else hinted on either column
    assert th.register_split_hint("k1", 128, False, "registers", "packed", "major", "") == 1
    assert th.register_split_hint("k1", 128, False, "registers", "packed", "major", "", cc=(10, 0)) == 0
    assert th.register_split_hint("k1", 128, False, "registers", "packed", "major", "", cc=(10, 7)) == 1
    assert th.register_split_hint("k1", 128, True, "registers", "packed", "major", "block", cc=(10, 0)) == 1
    assert th.register_split_hint("k1", 128, True, "registers", "packed", "major", "block", cc=(10, 3)) == 1
    assert th.register_split_hint("k1", 128, True, "registers", "packed", "major", "block") == 0
    assert th.register_split_hint("k3", 128, True, "registers", "packed", None, "", cc=(10, 0)) == 0
    assert th.register_split_hint("k1", 128, True, "registers", "packed_biasfold", "major", "block", cc=(10, 0)) == 0
    assert th.register_split_hint("k1", 128, True, "smem_rolled_pf", "packed", "major", "block", cc=(10, 0)) == 0
    assert th.register_split_hint("k1", 128, False, "registers", "packed", "major", "block", cc=(10, 0)) == 0
    assert th.register_split_hint("k1", 256, True, "smem_rolled_pf", "packed", "major", "block", cc=(10, 0)) == 0


def test_autograd_is_a_typed_decline_not_a_silent_constant():
    """No backward is registered: under grad mode a `requires_grad` input or parameter is declined with NotImplementedError BEFORE any
    device work (CPU tensors suffice here), in the drop-in and in the class API's execute path; under torch.no_grad() the same tensors pass
    the check (the drop-in then fails later on the CPU device -- a ValueError naming the device, i.e. the autograd gate did not fire)."""
    c = 128
    x = torch.randn(1, 64, 64, c).to(torch.bfloat16)
    mask = torch.ones(1, 64, 64, dtype=torch.bfloat16)
    p = _params(c, c, True, True, seed=3)
    # the drop-in: x, then a parameter
    with pytest.raises(NotImplementedError, match="no backward registered.*x requires grad"):
        _call(x.clone().requires_grad_(True), "outgoing", mask, p)
    pg = dict(p, p_in_weight=p["p_in_weight"].clone().requires_grad_(True))
    with pytest.raises(NotImplementedError, match="p_in_weight requires grad"):
        _call(x, "outgoing", mask, pg)
    with torch.no_grad():
        with pytest.raises(ValueError, match="CUDA tensor"):
            _call(x.clone().requires_grad_(True), "outgoing", mask, pg)
    # the class API: execute's argument check declines a requires_grad x / out / prepared weight the same way
    plan = tapi.TriangleMultiplicativeUpdateFwd(x, "outgoing", c_hidden=c, has_mask=True, has_in_bias=True, has_out_bias=True)
    w = tapi.prepare_weights(*[p[n] for n in CUEQ_PARAMS[3:15]])
    out = torch.empty_like(x)
    buf = torch.empty(plan.get_workspace_size() + 256, dtype=torch.uint8)  # the CPU allocator aligns to 64 B; the check wants 256
    off = (-buf.data_ptr()) % 256
    ws = buf[off : off + plan.get_workspace_size()]
    with pytest.raises(NotImplementedError, match="x requires grad"):
        plan._check_execute_args(x.clone().requires_grad_(True), mask, w, out, ws)
    with pytest.raises(NotImplementedError, match="out requires grad"):
        plan._check_execute_args(x, mask, w, out.clone().requires_grad_(True), ws)
    wg = tapi.prepare_weights(*[(pg[n] if n != "p_in_weight" else pg[n].detach()) for n in CUEQ_PARAMS[3:15]])
    object.__setattr__(wg, "w_in", wg.w_in.clone().requires_grad_(True))
    with pytest.raises(NotImplementedError, match="weights.w_in requires grad"):
        plan._check_execute_args(x, mask, wg, out, ws)
    with torch.no_grad():
        plan._check_execute_args(x.clone().requires_grad_(True), mask, wg, out, ws)  # the gate is grad-mode scoped; the rest of the check passes on CPU


def test_workspace_and_traffic_models():
    plan = tapi.TriangleMultiplicativeUpdateFwd(_desc(N=512), "outgoing", c_hidden=128)
    assert plan.get_workspace_size() == 2 * 128 * 512 * 512 * 2 + 128 * 512 * 512 * 2
    assert plan.traffic_bytes() == 18 * 512 * 512 * 128 + 2 * 512 * 512  # the three-kernel byte model + the mask
    plan_nomask = tapi.TriangleMultiplicativeUpdateFwd(_desc(N=512), "outgoing", c_hidden=128, has_mask=False)
    assert plan_nomask.traffic_bytes() == 18 * 512 * 512 * 128
    p217 = tapi.TriangleMultiplicativeUpdateFwd(_desc(N=217, B=2), "outgoing", c_hidden=128)
    assert p217.planes_shape == (2, 128, 434, 224) and p217.y_shape == (128, 434, 224)
    assert p217.get_workspace_size() == tapi._align_up(2 * 2 * 128 * 434 * 224) + tapi._align_up(k2.k2_slack_bytes(217)) + tapi._align_up(2 * 128 * 434 * 224)
    assert k2.k2_slack_bytes(217) == 7 * 224 * 2 and k2.k2_slack_bytes(256) == 0
    assert plan.flop() == k1.proj_flop(128, 128, 1, 512) + k2.contraction_flop(128, 512) + k3.tail_flop(128, 128, 1, 512)


def _params(c_z, c, with_in_bias, with_out_bias, seed, device="cpu", dtype=torch.bfloat16):
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    p = dict(
        norm_in_weight=(1.0 + 0.1 * r(c_z)),
        norm_in_bias=(0.1 * r(c_z)),
        p_in_weight=(r(2 * c, c_z) / math.sqrt(c_z)),
        p_in_bias=(0.2 * r(2 * c)) if with_in_bias else None,
        g_in_weight=(r(2 * c, c_z) / math.sqrt(c_z)),
        g_in_bias=(0.2 * r(2 * c)) if with_in_bias else None,
        norm_out_weight=(1.0 + 0.1 * r(c)),
        norm_out_bias=(0.1 * r(c)),
        p_out_weight=(r(c_z, c) / math.sqrt(c)),
        p_out_bias=(0.2 * r(c_z)) if with_out_bias else None,
        g_out_weight=(r(c_z, c_z) / math.sqrt(c_z)),
        g_out_bias=(0.2 * r(c_z)) if with_out_bias else None,
    )
    return {k: (v.to(dtype).to(device) if v is not None else None) for k, v in p.items()}


def test_prepare_weights_builds_the_kernel_layouts_once():
    p = _params(128, 128, True, True, seed=1)
    w = tapi.prepare_weights(*[p[n] for n in CUEQ_PARAMS[3:15]])
    assert w.c_z == 128 and w.c == 128 and w.has_in_bias and w.has_out_bias
    assert torch.equal(w.w_in, k1.stack_projection_weights(p["p_in_weight"], p["g_in_weight"]))
    assert torch.equal(w.b_in, k1.stack_projection_biases(p["p_in_bias"], p["g_in_bias"]))
    assert torch.equal(w.b_out, k3.stack_tail_biases(p["p_out_bias"], p["g_out_bias"]))
    # one-sided biases pair with zeros AT PREPARATION (never per call)
    w1 = tapi.prepare_weights(*[p[n] if n != "g_in_bias" else None for n in CUEQ_PARAMS[3:15]])
    assert torch.equal(w1.b_in, k1.stack_projection_biases(p["p_in_bias"], torch.zeros_like(p["p_in_bias"])))
    w0 = tapi.prepare_weights(*[p[n] if "bias" not in n or "norm" in n else None for n in CUEQ_PARAMS[3:15]])
    assert w0.b_in is None and w0.b_out is None and not w0.has_in_bias and not w0.has_out_bias
    with pytest.raises(ValueError, match="required"):
        tapi.prepare_weights(*[p[n] if n != "g_out_weight" else None for n in CUEQ_PARAMS[3:15]])
    p32 = _params(128, 128, False, False, seed=1, dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="bf16"):
        tapi.prepare_weights(*[p32[n] for n in CUEQ_PARAMS[3:15]])
    w32 = tapi.prepare_weights(*[p32[n] for n in CUEQ_PARAMS[3:15]], cast_to_bf16=True)
    assert w32.w_in.dtype == torch.bfloat16


def test_inference_mode_parameters_bypass_the_version_keyed_weight_cache(monkeypatch):
    """Parameters created under torch.inference_mode() carry no version counter (`t._version` raises RuntimeError), so they have no identity
    the drop-in's weight cache can key on: such a parameter set is prepared on every call and never cached (documented; the prepared class API's
    one-time `prepare_weights` is the remedy) -- a constant version would miss an in-place update made inside inference mode.  Ordinary
    parameters hit the cache as before, also when the CALL runs under inference mode (one preparation for two lookups), and an in-place update
    of one of them still misses once (the version counter is part of the identity).  GPU-free (CPU bf16 parameters)."""
    c = 128
    p = _params(c, c, True, True, seed=47)
    with torch.inference_mode():
        pi = _params(c, c, True, True, seed=47)
    assert all(t.is_inference() for t in pi.values() if t is not None) and not any(t.is_inference() for t in p.values() if t is not None)
    calls = []
    real = tapi.prepare_weights
    monkeypatch.setattr(tapi, "prepare_weights", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    params_i = tuple(pi[n] for n in CUEQ_PARAMS[3:15])
    n_cached = len(tapi._WEIGHTS)
    w1 = tapi._cached_weights(params_i, False)  # the head: RuntimeError 'Inference tensors do not track version counter'
    w2 = tapi._cached_weights(params_i, False)
    assert len(calls) == 2 and w1 is not w2 and len(tapi._WEIGHTS) == n_cached, "inference tensors are prepared per call, never cached"
    assert torch.equal(w1.w_in, w2.w_in) and torch.equal(w1.w_in, tapi.prepare_weights(*params_i).w_in)
    params = tuple(p[n] for n in CUEQ_PARAMS[3:15])
    calls.clear()
    w3 = tapi._cached_weights(params, False)
    with torch.inference_mode():  # inference MODE alone does not bypass: these tensors keep their version counters
        w4 = tapi._cached_weights(params, False)
    assert len(calls) == 1 and w3 is w4 and len(tapi._WEIGHTS) == n_cached + 1
    p["p_in_weight"].mul_(1.0)  # an in-place update bumps the version counter: the identity changes and the cache misses once
    assert tapi._cached_weights(params, False) is not w3 and len(calls) == 2
    # one inference tensor among ordinary ones is enough to bypass (the set has no complete identity)
    mixed = tuple(pi[n] if n == "g_out_weight" else p[n] for n in CUEQ_PARAMS[3:15])
    calls.clear()
    assert tapi._cached_weights(mixed, False) is not tapi._cached_weights(mixed, False) and len(calls) == 2


def test_parse_precision_mirrors_cueq_spellings():
    for spelling in (None, "default", "none", "DEFAULT", -1, 0):
        assert tapi._parse_precision(spelling) == "default"
    for spelling in ("bf16", "bfloat16", "BFLOAT16", 5):
        assert tapi._parse_precision(spelling) == "bfloat16"
    with pytest.raises(NotImplementedError, match="mxfp8"):
        tapi._parse_precision("mxfp8")
    with pytest.raises(NotImplementedError):
        tapi._parse_precision("ieee")
    with pytest.raises(ValueError):
        tapi._parse_precision(object())


def test_reference_without_rounding_is_the_formula():
    """``rounding="none"`` equals a literal transcription of the definition (LayerNorms, gated projections, the two einsums, output gate)."""
    torch.manual_seed(0)
    B, N, cz = 2, 9, 32
    c = cz
    x = torch.randn(B, N, N, cz, dtype=torch.float64)
    mask = (torch.rand(B, N, N) > 0.3).double()
    p = {k: (v.double() if v is not None else None) for k, v in _params(cz, c, True, True, seed=3, dtype=torch.float32).items()}
    ln = torch.nn.functional.layer_norm
    z_in = ln(x, (cz,), p["norm_in_weight"], p["norm_in_bias"], 1e-5)
    lin = torch.nn.functional.linear
    a = torch.sigmoid(lin(z_in, p["g_in_weight"][:c], p["g_in_bias"][:c])) * lin(z_in, p["p_in_weight"][:c], p["p_in_bias"][:c]) * mask[..., None]
    b = torch.sigmoid(lin(z_in, p["g_in_weight"][c:], p["g_in_bias"][c:])) * lin(z_in, p["p_in_weight"][c:], p["p_in_bias"][c:]) * mask[..., None]
    for direction, eq in (("outgoing", "bikc,bjkc->bijc"), ("incoming", "bkic,bkjc->bijc")):
        y = torch.einsum(eq, a, b)
        u = ln(y, (c,), p["norm_out_weight"], p["norm_out_bias"], 1e-5)
        want = torch.sigmoid(lin(z_in, p["g_out_weight"], p["g_out_bias"])) * lin(u, p["p_out_weight"], p["p_out_bias"])
        got = tref.triangle_multiplicative_update_reference(x, direction, mask, **{k: p[k] for k in CUEQ_PARAMS[3:15]}, rounding="none")
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError):
        tref.triangle_multiplicative_update_reference(x, "outgoing", mask, **{k: p[k] for k in CUEQ_PARAMS[3:15]}, rounding="half")
    # batchless x and mask
    got3 = tref.triangle_multiplicative_update_reference(x[0], "outgoing", mask[0], **{k: p[k] for k in CUEQ_PARAMS[3:15]}, rounding="none")
    assert got3.shape == (1, N, N, cz)


# ---------------------------------------------------------------------------
# Rubin: the oracle, the bitwise chain pin, the wrapper, the execute contract, CUDA graphs, the declines on a live device
# ---------------------------------------------------------------------------
_SENTINEL = 1.5e30
_TOL_ABS = 4e-3
_TOL_REL = 1.0 / 64.0
#: The propagated-flip TAIL (the module docstring): a few cells per million exceed the bulk bound by up to ~2x; their fraction and their
#: magnitude are both pinned.  Measured on the first green run: fraction 7e-7 .. 7.6e-6, max |diff| 0.0098 .. 0.0158 on |ref| <= 3.5.
_TAIL_FRAC_MAX = 2e-5
_TAIL_ABS_CAP = 2e-2
_REL_RMS_MAX = 3e-3


def _oracle_stats(out, ref):
    """The three-part end-to-end oracle: the bulk bound ``4e-3 + |ref| / 64`` on all but a pinned fraction of the cells, the cap
    ``2e-2 + |ref| / 64`` on EVERY cell, and the relative RMS -- returns the measured figures and the verdict (no assert; the drivers share it)."""
    diff = (out.double() - ref).abs()
    bulk = _TOL_ABS + ref.abs() * _TOL_REL
    cap = _TAIL_ABS_CAP + ref.abs() * _TOL_REL
    n = diff.numel()
    bad_bulk = int((diff > bulk).sum().item())
    bad_cap = int((diff > cap).sum().item())
    rel_rms = (diff.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    st = dict(
        cells=n,
        beyond_bulk=bad_bulk,
        beyond_bulk_frac=bad_bulk / n,
        beyond_cap=bad_cap,
        max_abs_diff=diff.max().item(),
        ref_abs_max=ref.abs().max().item(),
        rel_rms=rel_rms,
    )
    st["ok"] = bad_cap == 0 and st["beyond_bulk_frac"] <= _TAIL_FRAC_MAX and rel_rms < _REL_RMS_MAX
    return st


def _inputs(n_res, batch, c_z, c, mask_kind, with_in_bias, with_out_bias, seed=0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    bf16 = torch.bfloat16
    x = (torch.randn(batch, n_res, n_res, c_z, generator=g) * 0.7 + torch.randn(c_z, generator=g)[None, None, None, :] * 0.5).to(bf16)
    if mask_kind == "none":
        mask = None
    elif mask_kind == "all":
        mask = torch.ones(batch, n_res, n_res)
    elif mask_kind == "padded":  # a right-padded protein per batch entry
        mask = torch.zeros(batch, n_res, n_res)
        for b in range(batch):
            L = (3 * n_res) // 4 if b == 0 else max(1, n_res // 3)
            mask[b, :L, :L] = 1.0
    elif mask_kind == "random":
        mask = (torch.rand(batch, n_res, n_res, generator=g) > 0.3).float()
    else:
        raise ValueError(mask_kind)
    p = _params(c_z, c, with_in_bias, with_out_bias, seed=seed + 100, device=device)
    return x.to(device), (mask.to(bf16).to(device) if mask is not None else None), p


def _call(x, direction, mask, p, **kw):
    return tapi.triangle_multiplicative_update(x, direction, mask, *[p[n] for n in CUEQ_PARAMS[3:15]], **kw)


def _check_oracle(out, ref, label):
    st = _oracle_stats(out, ref)
    msg = (
        f"{label}: {st['beyond_bulk']} of {st['cells']} cells beyond the bulk bound (fraction {st['beyond_bulk_frac']:.2g}, budget {_TAIL_FRAC_MAX:.0e}), "
        f"{st['beyond_cap']} beyond the cap; max |diff| {st['max_abs_diff']:.4g} at |ref| max {st['ref_abs_max']:.4g}; rel RMS {st['rel_rms']:.3g}"
    )
    print(msg)
    assert st["ok"], msg
    return st["max_abs_diff"], st["rel_rms"]


@requires_served_arch
@pytest.mark.parametrize(
    "n_res,batch,direction,mask_kind,with_in_bias,with_out_bias,c_z",
    [
        (64, 1, "outgoing", "all", True, True, 128),
        (128, 2, "incoming", "padded", True, True, 128),
        (217, 1, "outgoing", "random", False, False, 128),  # a tail N: the padded planes, K = 224 in the contraction; an ODD mask length (47 089)
        (256, 1, "incoming", "none", False, True, 128),  # no mask tensor at the ABI
        (512, 1, "outgoing", "padded", True, False, 128),
        (768, 1, "outgoing", "all", True, True, 128),
        (128, 1, "outgoing", "random", True, True, 256),  # c_z = c = 256: one API oracle cell per direction for every served width
        (128, 2, "incoming", "padded", False, True, 256),
    ],
)
def test_api_matches_the_fp64_reference(n_res, batch, direction, mask_kind, with_in_bias, with_out_bias, c_z):
    x, mask, p = _inputs(n_res, batch, c_z, c_z, mask_kind, with_in_bias, with_out_bias)
    out = _call(x, direction, mask, p)
    torch.cuda.synchronize()
    assert out.shape == x.shape and out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all(), "a non-finite output"
    ref = tref.triangle_multiplicative_update_reference(x, direction, mask, **{k: p[k] for k in CUEQ_PARAMS[3:15]}, rounding="kernel")
    _check_oracle(out, ref, f"{direction} N={n_res} B={batch} mask={mask_kind} biases={with_in_bias}/{with_out_bias}")
    # the distance to the pure fp64 formula (the kernels' numerics class: bf16 at five points) -- reported, bounded loosely
    ref_none = tref.triangle_multiplicative_update_reference(x, direction, mask, **{k: p[k] for k in CUEQ_PARAMS[3:15]}, rounding="none")
    d = (out.double() - ref_none).abs()
    rel_rms_formula = (d.pow(2).mean().sqrt() / ref_none.pow(2).mean().sqrt()).item()
    print(f"  vs the unrounded fp64 formula: max |diff| {d.max().item():.4g}, rel RMS {rel_rms_formula:.3g}")
    assert rel_rms_formula < 2e-2
    # two-launch trick: a first-launch race would show here
    out2 = _call(x, direction, mask, p)
    torch.cuda.synchronize()
    assert torch.equal(out2, out)


@requires_served_arch
def test_api_equals_the_hand_chained_kernels_bitwise():
    """The API adds nothing but glue: K1 -> K2 (the engine's instance over the padded planes) -> K3 by hand, same buffers' contents, bitwise."""
    import cudnn.gemm.frost  # noqa: F401
    from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
    from cudnn.gemm.frost.tile_config import by_name

    n_res, batch, c = 217, 2, 128
    x, mask, p = _inputs(n_res, batch, c, c, "random", True, True, seed=5)
    out_api = _call(x, "outgoing", mask, p)
    torch.cuda.synchronize()
    w = tapi.prepare_weights(*[p[n] for n in CUEQ_PARAMS[3:15]])
    stream = torch.cuda.current_stream().cuda_stream
    pshape = k1.planes_shape(c, batch, n_res)
    flat = torch.full((math.prod(pshape) + k2.k2_slack_bytes(n_res) // 2,), _SENTINEL, device="cuda", dtype=torch.bfloat16)  # planes + the over-read slack
    planes = flat[: math.prod(pshape)].view(*pshape)
    k1.run_trimul_proj(
        k1.build_trimul_proj(k1.proj_params(c, c, has_bias=True, has_mask=True, want_stats=False), _arch()),
        x,
        w.w_in,
        planes,
        w.norm_in_weight,
        w.norm_in_bias,
        bias_stacked=w.b_in,
        mask=mask,
        stats=None,
        stream=stream,
    )
    y = torch.full(k3.y_shape(c, batch, n_res), _SENTINEL, device="cuda", dtype=torch.bfloat16)
    compiled = jit_from_cudnn_graph(k2.contraction_graph(cudnn, c, n_res, "outgoing", batch=batch, padded=True), config=by_name(k2.tile_config_for(n_res)))
    a_v, b_v, y_v = k2.k2_operand_views(planes[0], planes[1], y, channels=c, batch=batch, n_res=n_res, direction="outgoing")
    bd = compiled.binding
    compiled({bd.a_operands[0]: a_v, bd.b_operands[0]: b_v, bd.outputs[0]: y_v}, stream=stream)
    out = torch.empty_like(x)
    k3.run_trimul_tail(
        k3.build_trimul_tail(k3.tail_params(c, c, has_bias=True, want_stats=False), _arch()),
        y,
        x,
        w.p_out_weight,
        w.g_out_weight,
        out,
        w.norm_out_weight,
        w.norm_out_bias,
        w.norm_in_weight,
        w.norm_in_bias,
        biases=w.b_out,
        stats=None,
        residual=None,
        stream=stream,
    )
    torch.cuda.synchronize()
    assert torch.equal(out, out_api)
    # the contraction over the padded K equals torch.bmm over the valid K (the pad is zero in both planes): bitwise, as the prototype measured
    a_ref = planes[0].view(c * batch, n_res, -1)[:, :, :n_res].float()
    b_ref = planes[1].view(c * batch, n_res, -1)[:, :, :n_res].float()
    y_ref = torch.bmm(a_ref, b_ref.transpose(1, 2))
    y_got = y_v[:, :, :n_res].float()
    assert not (y_got == _SENTINEL).any()
    # the prototype measured the engine BITWISE equal to torch.bmm at K % 64 == 0; at a padded K (224 here) cuBLAS's accumulation order differs,
    # so the contraction is pinned to within one bf16 ulp of the fp32 bmm (|y| * 2^-8) and the exactly-equal fraction is reported
    tol = y_ref.abs() * 2.0**-8 + 2.0**-14
    n_bad = int(((y_got - y_ref).abs() > tol).sum().item())
    frac_equal = (y_got == y_ref.to(torch.bfloat16).float()).double().mean().item()
    print(f"K2 over the padded planes vs torch.bmm (fp32): {n_bad} cells beyond one bf16 ulp; {100 * frac_equal:.2f} % bitwise equal after rounding")
    assert n_bad == 0
    if n_res % 64 == 0:
        assert frac_equal == 1.0


@requires_served_arch
def test_wrapper_equals_the_class_api_and_caches_the_plan():
    n_res, batch, c = 128, 2, 128
    x, mask, p = _inputs(n_res, batch, c, c, "padded", False, True, seed=7)
    n_plans = len(tapi._PLANS)
    out_w = _call(x, "incoming", mask, p)
    assert len(tapi._PLANS) == n_plans + 1
    out_w2 = _call(x, "incoming", mask, p)
    assert len(tapi._PLANS) == n_plans + 1, "a second call of the same shape must hit the plan cache"
    w = tapi.prepare_weights(*[p[n] for n in CUEQ_PARAMS[3:15]])
    plan = tapi.TriangleMultiplicativeUpdateFwd(x, "incoming", c_hidden=c, has_mask=True, has_in_bias=False, has_out_bias=True)
    assert plan.check_support()
    plan.compile()
    out = torch.full_like(x, _SENTINEL)
    ws = torch.empty(plan.get_workspace_size(), dtype=torch.uint8, device="cuda")
    plan.execute(x, mask, w, out, ws)
    torch.cuda.synchronize()
    assert torch.equal(out, out_w) and torch.equal(out_w2, out_w)
    # the flag contract, both directions
    with pytest.raises(ValueError, match="has_mask=True"):
        plan.execute(x, None, w, out, ws)
    plan_nm = tapi.TriangleMultiplicativeUpdateFwd(x, "incoming", c_hidden=c, has_mask=False, has_in_bias=False, has_out_bias=True)
    plan_nm.compile()
    with pytest.raises(ValueError, match="has_mask=False"):
        plan_nm.execute(x, mask, w, out, ws)
    w_nb = tapi.prepare_weights(*[p[n] if n not in ("p_out_bias", "g_out_bias") else None for n in CUEQ_PARAMS[3:15]])
    with pytest.raises(ValueError, match="has_out_bias"):
        plan.execute(x, mask, w_nb, out, ws)
    with pytest.raises(ValueError, match="workspace"):
        plan.execute(x, mask, w, out, ws[: plan.get_workspace_size() - 256])
    with pytest.raises(ValueError, match="x must be"):
        plan.execute(x.float(), mask, w, out, ws)
    with pytest.raises(ValueError, match="alias"):
        plan.execute(x, mask, w, x, ws)
    # the no-mask plan equals the all-valid mask
    x2, mask_all, p2 = _inputs(n_res, batch, c, c, "all", False, True, seed=7)
    assert torch.equal(_call(x2, "incoming", None, p2), _call(x2, "incoming", mask_all, p2))


@requires_served_arch
def test_execute_allocates_nothing_and_never_synchronizes():
    n_res, batch, c = 256, 1, 128
    x, mask, p = _inputs(n_res, batch, c, c, "random", True, True, seed=9)
    w = tapi.prepare_weights(*[p[n] for n in CUEQ_PARAMS[3:15]])
    plan = tapi.TriangleMultiplicativeUpdateFwd(x, "outgoing", c_hidden=c, has_mask=True, has_in_bias=True, has_out_bias=True)
    plan.compile()
    out = torch.empty_like(x)
    ws = torch.empty(plan.get_workspace_size(), dtype=torch.uint8, device="cuda")
    plan.execute(x, mask, w, out, ws)
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            plan.execute(x, mask, w, out, ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before, "execute() allocated"


@requires_served_arch
def test_cuda_graph_capture_replays_bitwise_and_follows_new_data():
    """The wrapper (warm) and the class API capture into a CUDA graph; a replay equals the eager result on the same data and follows a
    change of the input's CONTENTS (not its address)."""
    n_res, batch, c = 256, 1, 128
    x, mask, p = _inputs(n_res, batch, c, c, "padded", True, False, seed=11)
    eager = _call(x, "outgoing", mask, p)  # warm: compiles + caches the plan and the weights
    torch.cuda.synchronize()
    out = torch.empty_like(x)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _call(x, "outgoing", mask, p, out=out)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=side):
        _call(x, "outgoing", mask, p, out=out)
    out.fill_(float("nan"))
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
    x_new, _, _ = _inputs(n_res, batch, c, c, "padded", True, False, seed=12)
    x.copy_(x_new)
    eager_new = _call(x_new, "outgoing", mask, p)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager_new)
    # the class API's own capture helper
    w = tapi.prepare_weights(*[p[n] for n in CUEQ_PARAMS[3:15]])
    plan = tapi.TriangleMultiplicativeUpdateFwd(x, "outgoing", c_hidden=c, has_mask=True, has_in_bias=True, has_out_bias=False)
    plan.compile()
    out2 = torch.empty_like(x)
    ws = torch.empty(plan.get_workspace_size(), dtype=torch.uint8, device="cuda")
    g2 = plan.make_cuda_graph(x, mask, w, out2, ws)
    out2.fill_(float("nan"))
    g2.replay()
    torch.cuda.synchronize()
    assert torch.equal(out2, eager_new)


@requires_served_arch
def test_masks_of_other_dtypes_and_batchless_inputs_follow_cueq():
    n_res, batch, c = 128, 1, 128
    x, mask, p = _inputs(n_res, batch, c, c, "random", True, True, seed=13)
    base = _call(x, "outgoing", mask, p)
    assert torch.equal(_call(x, "outgoing", mask.bool(), p), base)  # a bool mask is cast once, like cuEquivariance
    assert torch.equal(_call(x, "outgoing", mask.float(), p), base)
    out3 = _call(x[0], "outgoing", mask[0], p)  # batchless [N, N, c_z] in -> [N, N, c_z] out
    assert out3.shape == (n_res, n_res, c) and torch.equal(out3, base[0])
    # precision='bfloat16' casts wider inputs and parameters once per call (cuEquivariance's path); the result equals the bf16 call
    p32 = {k: (v.float() if v is not None else None) for k, v in p.items()}
    out32 = tapi.triangle_multiplicative_update(x.float(), "outgoing", mask.float(), *[p32[n] for n in CUEQ_PARAMS[3:15]], precision="bfloat16")
    assert torch.equal(out32, base)
    with pytest.raises(NotImplementedError, match="bf16"):
        tapi.triangle_multiplicative_update(x.float(), "outgoing", mask, *[p[n] for n in CUEQ_PARAMS[3:15]])


@requires_served_arch
def test_rejects_on_a_live_device_are_typed():
    n_res, batch, c = 128, 1, 128
    x, mask, p = _inputs(n_res, batch, c, c, "all", True, True, seed=15)
    with pytest.raises(NotImplementedError, match="bf16"):
        _call(x.half(), "outgoing", mask, p)
    with pytest.raises(ValueError, match="direction"):
        _call(x, "sideways", mask, p)
    with pytest.raises(ValueError, match="required"):
        _call(x, "outgoing", mask, dict(p, norm_in_weight=None))
    with pytest.raises(NotImplementedError, match="mxfp8"):
        _call(x, "outgoing", mask, p, precision="mxfp8")
    with pytest.raises(ValueError, match="mask must be"):
        _call(x, "outgoing", mask[:, :64, :64], p)
    with pytest.raises(ValueError, match="contiguous"):
        _call(x.transpose(1, 2), "outgoing", mask, p)
    x217, mask217, p217 = _inputs(217, 1, c, c, "all", False, False, seed=16)
    with pytest.raises(NotImplementedError, match="multiple of 8|% 8"):
        _call(x217, "incoming", mask217, p217)
    x100, _, p100 = _inputs(64, 1, 96, 96, "none", False, False, seed=17)  # c_z = 96 = 3 x 32: cuEquivariance admits the width, the kernels decline
    with pytest.raises(NotImplementedError, match="c_z in"):
        _call(x100, "outgoing", None, p100)
    assert json.dumps(sorted(tapi.SUPPORTED_CHANNELS)) == "[128, 256]"


def test_heuristics_table_is_total_in_domain_and_resolvable():
    """Every pick of the measured table is a member of the knob domain AND a FROST GEMM catalog name; the recommendation is total over N, c, B and
    direction and always COMPLETE; the per-cell refinements override one axis each and never overlap; the API's K1 tile-order pin equals the
    c = 128 default; a caller's PARTIAL record merges over the table (an unset axis keeps the measured pick); the vocabulary check is typed per
    axis."""
    from cudnn.gemm.frost.ln_prologue import LN_PASS_FORMS, LN_PASS_FORMS_REORDERED
    from cudnn.gemm.frost.tile_config import by_name
    from cudnn.triangle_multiplicative_update import heuristics as th

    for name in th.K2_TILE_DOMAIN:
        assert by_name(name).name == name
    for c in (128, 256, 384, 512):
        for n in (33, 64, 128, 217, 256, 384, 512, 768, 1024, 2048, 4096):
            for b in (1, 2, 4, 8):
                for d in ("outgoing", "incoming"):
                    kn = th.recommend(n, c, b, d)
                    assert kn.is_complete()
                    th.validate_knobs(kn)
                    assert kn.ln_pass_form not in LN_PASS_FORMS_REORDERED, (n, c, b, d, kn)  # a summation-ORDER-changing form is never a default
                    assert k2.tile_config_for(n, channels=c, batch=b, direction=d) == kn.k2_tile
                    assert th.recommend(n, c, th.batch_bucket(b), d) == kn  # a batch takes its measured bucket's pick (B >= 4 -> 4, else 1)
    assert th.batch_bucket(1) == th.batch_bucket(3) == 1 and th.batch_bucket(4) == th.batch_bucket(16) == 4
    # the LayerNorm pass-form vocabulary: every member (the pipelined `smem_rolled_pf` and the reordered `smem_rolled_tree` included) is admitted as a
    # caller's knob over any pick -- a typed check, no device work; the reordered one is named by the substrate
    for form in LN_PASS_FORMS:
        th.validate_knobs(th.TriMulKnobs(ln_pass_form=form).merged_over(th.recommend(512, 256, 1)))
    assert "smem_rolled_pf" in LN_PASS_FORMS and set(LN_PASS_FORMS_REORDERED) == {"smem_rolled_tree"} <= set(LN_PASS_FORMS)
    # the LayerNorm-once work item (wave 2, T27): a geometry axis every pick carries (OFF until its A/B/A lands a bucket default); the vocabulary is
    # the substrate's and typed; the output stage takes "block" at two N tiles only (c = 256) and keeps today's item elsewhere -- the API's builder
    from cudnn.gemm.frost.ln_prologue import LN_ONCE_FORMS

    assert LN_ONCE_FORMS == ("", "block", "passes")
    for form in LN_ONCE_FORMS:
        th.validate_knobs(th.TriMulKnobs(ln_once=form).merged_over(th.recommend(512, 256, 1)))
    with pytest.raises(ValueError, match="ln_once"):
        th.validate_knobs(th.TriMulKnobs(ln_once="tile"))
    assert th.resolve(512, 256, 1, "outgoing", knobs=th.TriMulKnobs(ln_once="block")).ln_once == "block"
    # the measured default (T27, the 212-SM Rubin class): every c = 256 bucket runs the LayerNorm-once BLOCK item (K1 +64..+139 %, K3 +31..+68 % standalone,
    # both bias arms, bitwise); the c = 128 buckets keep today's item until their own A/B/A lands a value
    for n in (33, 128, 217, 256, 384, 512, 768, 1024, 2048):
        for b in (1, 4):
            for d in ("outgoing", "incoming"):
                assert th.recommend(n, 256, b, d).ln_once == "block", (n, b, d)
    assert th.resolve(512, 256, 1, "outgoing", knobs=th.TriMulKnobs(ln_once="")).ln_once == ""  # a caller's "" over the pick = today's item
    # c = 128: every bucket from N = 256 runs the block item with the pass-major order (+43..+53 % standalone K1 on the major buckets; +15..+45 % on
    # the formerly interleaved 256 x 1 / 384 buckets against their interleaved pick, the major order alone -0.3..-2.6 %); the N < 256 buckets keep
    # today's item and order until measured
    for n, b, want in (
        (512, 1, "block"),
        (768, 1, "block"),
        (1024, 4, "block"),
        (2048, 1, "block"),
        (256, 4, "block"),
        (512, 4, "block"),
        (256, 1, "block"),
        (384, 1, "block"),
        (384, 4, "block"),
        (200, 1, ""),
        (128, 1, ""),
        (33, 1, ""),
    ):
        kn = th.recommend(n, 128, b, "outgoing")
        assert kn.ln_once == want, (n, b, kn)
        assert not (kn.ln_once and kn.k1_pass_order != "major"), (n, b, kn)  # the lever never meets the interleaved order in a pick
    assert (
        tapi.TriangleMultiplicativeUpdateFwd(_desc(N=512, cz=256), "outgoing", c_hidden=256, knobs=tapi.TriMulKnobs(ln_once="block"))._k3_ln_once() == "block"
    )
    assert tapi.TriangleMultiplicativeUpdateFwd(_desc(N=512, cz=256), "outgoing", c_hidden=256, knobs=tapi.TriMulKnobs(ln_once="passes"))._k3_ln_once() == ""
    assert tapi.TriangleMultiplicativeUpdateFwd(_desc(N=512), "outgoing", c_hidden=128, knobs=tapi.TriMulKnobs(ln_once="block"))._k3_ln_once() == ""
    assert tapi.TriangleMultiplicativeUpdateFwd(_desc(N=512), "outgoing", c_hidden=128)._k3_ln_once() == ""
    # the refinement table: one axis per row, the keys disjoint on that axis, every (floor, batch bucket) a real bucket of the width
    for width, rows in th._REFINEMENTS.items():
        floors = {f for f, _ in th._PICKS[width]}
        seen = set()
        for floor, bb, direction, override in rows:
            assert floor in floors and bb in th.BATCH_BUCKETS and direction in (None, "outgoing", "incoming")
            axes = [
                a
                for a in ("k2_tile", "k1_pass_order", "k3_prologue", "clc_stages", "k_block", "ln_pass_form", "epi_math", "min_blocks_per_mp", "ln_once")
                if getattr(override, a) is not None
            ]
            assert len(axes) == 1, (width, floor, bb, direction, axes)
            for d in (("outgoing", "incoming") if direction is None else (direction,)):
                assert (floor, bb, d, axes[0]) not in seen, "two refinements override one axis of one cell"
                seen.add((floor, bb, d, axes[0]))
    # the measured picks the table carries (the bucket rule and the per-cell refinements)
    assert tapi.K1_PASS_ORDER == th.DEFAULT_K1_PASS_ORDER == th.recommend(200, 128, 1).k1_pass_order == th.recommend(128, 128, 1).k1_pass_order  # below N = 256
    assert th.recommend(512, 256, 1).k1_pass_order == "major"  # the measured per-width sign of the projection stage's tile order
    assert th.recommend(512, 128, 1).k1_pass_order == "major" and th.recommend(512, 128, 1).clc_stages == 1  # the ring-depth interaction (verification pass)
    assert th.recommend(128, 128, 1).k3_prologue == "transpose" and th.recommend(128, 256, 1).k3_prologue == "mn_major"
    assert th.recommend(384, 128, 1).k2_tile == k2.SMALL_N_TILE_CONFIG and th.recommend(512, 128, 1).k2_tile == k2.MULTICAST_A2_TILE_CONFIG
    assert th.recommend(768, 128, 1).k2_tile == k2.DESIGN_TILE_CONFIG and th.recommend(2048, 128, 1).k2_tile == k2.MULTICAST_A4_TILE_CONFIG
    assert th.recommend(128, 128, 1, "incoming").clc_stages == 1 and th.recommend(128, 128, 1, "outgoing").clc_stages == 2  # the direction-keyed ring depth
    assert (
        th.recommend(256, 128, 1, "outgoing").k2_tile == th.recommend(256, 128, 1, "incoming").k2_tile == k2.DESIGN_TILE_CONFIG
    )  # the pair tile stays at B = 1
    assert th.recommend(256, 128, 4).clc_stages == 1 and th.recommend(256, 128, 1, "incoming").clc_stages == 2  # the batch-keyed ring depth
    assert (
        th.recommend(256, 128, 4).k1_pass_order == "major"
        and th.recommend(256, 128, 1).k1_pass_order == "major"
        and th.recommend(200, 128, 1).k1_pass_order == "interleaved"
    )  # the K1 order: major from N = 256 (the LayerNorm-once item needs it; measured against the interleaved pick), interleaved below
    assert th.recommend(384, 128, 1).clc_stages == 4 and th.recommend(384, 128, 4).clc_stages == 1
    assert th.recommend(128, 256, 4).clc_stages == 1 and th.recommend(128, 256, 1).clc_stages == 2
    # the row tile of the two LayerNorm-prologue stages: the substrate's rule unless a pick sets it (every member of the domain is bitwise the same)
    assert th.K_BLOCK_DOMAIN == (64, 128, 256)
    assert th.recommend(64, 128, 1).k_block == 64 and th.recommend(128, 128, 1).k_block == 128 and th.recommend(217, 128, 1).k_block == 256
    assert th.recommend(256, 128, 1).k_block == th.recommend(512, 128, 1).k_block == th.recommend(2048, 256, 4).k_block == 256
    assert th.recommend(384, 128, 1).k_block == th.default_k_block(384) == 128 and th.recommend(33, 128, 1).k_block == 64  # (33: clamped into the domain)
    assert th.recommend(384, 256, 4, "incoming").k_block == 128 and th.recommend(640, 128, 1).k_block == 128  # N mod 256 in 1..128: fewer rows at 128
    assert th.recommend(300, 128, 1).k_block == 128 and th.recommend(400, 128, 1).k_block == 256  # 400: 512 rows per bi either way -> the rule's 256
    assert th.recommend(768, 128, 1).k_block == 256 and th.recommend(217, 128, 1).k_block == 256  # multiples of 256 and N < 256 keep the rule
    # the substrate forms (bitwise the register / scalar rendering): c = 256 runs the PIPELINED rolled re-read LayerNorm form (`smem_rolled_pf`: the
    # rolled form with the next chunk's loads carried one iteration ahead; K1 +2.9..+5.6 %, K3 +20..+34 % over the rolled form on every measured
    # N >= 256 cell of both bias arms, 212-SM Rubin class) + the packed epilogue on every bucket the flip covers; c = 128 runs the packed epilogue
    # everywhere and the rolled form where it measured ahead of the packed-only pick (N 256 at B = 1, N 384, N 512 at B >= 4, N >= 1024), the
    # register form elsewhere -- the pipelined form's c = 128 sign is mixed by cell (-1.4..+2.6 %) and flips nothing there; the packed
    # epilogue's BIAS-PATH form `packed_biasfold` (the gate bias folded into the tanh argument's fma) rides on the register-form K1 cells
    for n in (33, 128, 217, 256, 384, 512, 768, 1024, 4096):
        for b in (1, 4):
            assert th.recommend(n, 256, b).ln_pass_form == "smem_rolled_pf" and th.recommend(n, 256, b).epi_math == "packed", (n, b)
            kn = th.recommend(n, 128, b)
            # the bias-path fold is picked EXACTLY where the projection stage runs the register LayerNorm form at N >= 256 (the
            # epilogue-bound rendering: +1.0..+1.7 % with the biases; a rolled K1 loses ~0.7 % under it): 768 <= N < 1024, 512 <= N < 768 at
            # B = 1, 256 <= N < 384 at B >= 4 -- bitwise the packed form, byte-identical to it without biases
            assert (kn.epi_math == "packed_biasfold") == (kn.ln_pass_form == "registers" and n >= 256), (n, b, kn)
            assert kn.epi_math in ("packed", "packed_biasfold"), (n, b)
    assert th.recommend(256, 128, 1).ln_pass_form == th.recommend(384, 128, 1).ln_pass_form == th.recommend(384, 128, 4).ln_pass_form == "smem_rolled"
    assert th.recommend(1024, 128, 1).ln_pass_form == th.recommend(1024, 128, 4).ln_pass_form == th.recommend(4096, 128, 1).ln_pass_form == "smem_rolled"
    assert th.recommend(256, 128, 4).ln_pass_form == "registers" and th.recommend(512, 128, 4).ln_pass_form == "smem_rolled"  # the batch-keyed LN forms
    assert (
        th.recommend(512, 128, 1).epi_math
        == th.recommend(768, 128, 1).epi_math
        == th.recommend(768, 128, 4).epi_math
        == th.recommend(256, 128, 4).epi_math
        == "packed_biasfold"
    )
    assert (
        th.recommend(512, 128, 4).epi_math
        == th.recommend(1024, 128, 1).epi_math
        == th.recommend(384, 128, 1).epi_math
        == th.recommend(256, 128, 1).epi_math
        == "packed"
    )
    assert (
        th.recommend(512, 128, 1, "incoming").epi_math == "packed_biasfold" and th.recommend(128, 128, 1).epi_math == "packed"
    )  # both directions; N < 256 unmeasured
    for n in (33, 128, 217, 512, 768):
        assert th.recommend(n, 128, 1).ln_pass_form == "registers", n
    assert th.recommend(768, 128, 4).ln_pass_form == th.recommend(128, 128, 4).ln_pass_form == th.recommend(217, 128, 4).ln_pass_form == "registers"
    from dataclasses import replace

    base = th.recommend(512, 128, 1)
    assert th.resolve(512, 128, 1, "outgoing", k2_tile=k2.SMALL_N_TILE_CONFIG) == replace(base, k2_tile=k2.SMALL_N_TILE_CONFIG)  # one axis over the table
    assert th.resolve(512, 128, 1, "outgoing", knobs=th.TriMulKnobs(clc_stages=4)) == replace(
        base, clc_stages=4
    )  # a PARTIAL record: the rest stays the table's
    assert th.resolve(1024, 128, 1, "outgoing", knobs=th.TriMulKnobs(clc_stages=2)) == replace(
        th.recommend(1024, 128, 1), clc_stages=2
    )  # keeps pair-A4 / major / transpose
    assert th.resolve(512, 128, 1, "outgoing", knobs=th.TriMulKnobs()) == base  # an empty record changes nothing
    assert th.resolve(512, 128, 1, "outgoing") == base
    full = th.TriMulKnobs(
        k2_tile=k2.SMALL_N_TILE_CONFIG,
        k1_pass_order="major",
        k3_prologue="mn_major",
        clc_stages=4,
        k_block=128,
        ln_pass_form="smem",
        epi_math="scalar",
        ln_once="",
    )
    assert full.is_complete() and th.resolve(512, 128, 1, "outgoing", knobs=full) == full  # a complete record replaces every axis
    assert not replace(full, ln_once=None).is_complete()  # the LayerNorm-once item is a geometry axis of the record (T27)
    four = th.TriMulKnobs(k2_tile=k2.SMALL_N_TILE_CONFIG, k1_pass_order="major", k3_prologue="mn_major", clc_stages=4)
    assert not four.is_complete() and th.resolve(512, 128, 1, "outgoing", knobs=four) == replace(
        four, k_block=256, ln_pass_form=base.ln_pass_form, epi_math=base.epi_math, ln_once=base.ln_once
    )  # the row tile from the rule, the two forms from the table
    assert th.resolve(512, 128, 1, "outgoing", knobs=th.TriMulKnobs(clc_stages=4), k2_tile=k2.SMALL_N_TILE_CONFIG) == replace(
        base, clc_stages=4, k2_tile=k2.SMALL_N_TILE_CONFIG
    )
    for bad, axis in (
        (dict(k2_tile="CONFIG_bogus"), "k2_tile"),
        (dict(k1_pass_order="zigzag"), "k1_pass_order"),
        (dict(k3_prologue="swizzle"), "k3_prologue"),
        (dict(clc_stages=3), "clc_stages"),
        (dict(k_block=96), "k_block"),
        (dict(k_block=32), "k_block"),  # the projection stage alone would admit it; the output stage's MN-major box needs >= 64
        (dict(k_block=True), "k_block"),
        (dict(ln_pass_form="rows"), "ln_pass_form"),
        (dict(epi_math="vector"), "epi_math"),
        (dict(min_blocks_per_mp=2), "min_blocks_per_mp"),  # two CTAs per SM would cap the allocation at 80 registers with the split impossible
        (dict(min_blocks_per_mp=True), "min_blocks_per_mp"),
    ):
        with pytest.raises(ValueError, match=axis):
            th.validate_knobs(th.TriMulKnobs(**bad))
    th.validate_knobs(th.TriMulKnobs())  # every axis unset = "the table decides": a valid partial record
    th.validate_knobs(th.TriMulKnobs(min_blocks_per_mp=1))
    with pytest.raises(TypeError):
        th.validate_knobs("CONFIG_bogus")


def test_register_split_hint_is_a_rendering_rule_the_knob_overrides():
    """The `.minnctapersm` hint is resolved per RENDERING (stage, width, bias arm, LayerNorm form, epilogue form), not per geometry: the table never
    sets the axis (`recommend` leaves it None, `is_complete` does not name it), `register_split_hint` returns a domain member for every rendering the
    API can build (0 = as shipped unless a measured row says 1), a bias-ON rendering is hinted only by a row naming its exact form, and a caller's `knobs=` value overrides
    the rule for both stages at compile time (pinned on the params the API hands the substrate, GPU-free)."""
    from cudnn.gemm.frost.ln_prologue import LN_PASS_FORMS, EPI_MATHS, MIN_BLOCKS_PER_MP_DOMAIN

    # the measured bias-ON rows: the c 128 register-form renderings running the bias-path epilogue fold (K1 pass-major)
    _BIAS_ON_ROWS = {("k1", 128, "registers", "packed_biasfold", "major")}
    from cudnn.triangle_multiplicative_update import heuristics as th

    # the rows are Rubin measurements: a plan declared on a Blackwell device (cc 10.0 / 10.3) reads 0 = the rendering as shipped for every row
    _k1_row = 1 if th.arch_column_for_cc(_cc()) == "sm107" else 0

    assert th.MIN_BLOCKS_PER_MP_DOMAIN == MIN_BLOCKS_PER_MP_DOMAIN == (0, 1)
    for c in (128, 256):
        for n in (33, 256, 512, 1024):
            for b in (1, 4):
                kn = th.recommend(n, c, b, "outgoing")
                assert kn.min_blocks_per_mp is None and kn.is_complete()
    from cudnn.gemm.frost.ln_prologue import PASS_ORDERS

    for stage in ("k1", "k3"):
        for c in (128, 256):
            for lnf in LN_PASS_FORMS:
                for epm in EPI_MATHS:
                    for order in PASS_ORDERS + (None,):
                        assert th.register_split_hint(stage, c, True, lnf, epm, order) == (
                            1 if (stage, c, lnf, epm, order) in _BIAS_ON_ROWS else 0
                        ), "a bias-ON rendering is hinted only by its measured row (the exact form)"
                        assert th.register_split_hint(stage, c, False, lnf, epm, order) in MIN_BLOCKS_PER_MP_DOMAIN
    for (stage, c, has_bias, lnf, epm, order, item), hint in th._REGISTER_SPLIT_HINTS.items():
        assert stage in ("k1", "k3") and c in (128, 256) and isinstance(has_bias, bool) and lnf in LN_PASS_FORMS and epm in EPI_MATHS and hint == 1, (
            stage,
            c,
            has_bias,
            lnf,
            epm,
            order,
            item,
            hint,
        )
        # a K1 row names its tile order (the interleaved renderings spill 4 / 13 under the hint where the major ones carry none); a K3 row has none
        assert (order in PASS_ORDERS) if stage == "k1" else (order is None), (stage, order)
        # every row names the PER-TILE item: under the LayerNorm-once item the honoured split spills (the c 128 register form 0 / 0 -> 4 / 13, the
        # rolled form 1 / 1 -> 9 / 15) and the whole call runs 0.5-1.3 % faster unhinted on every bias-off cell, within the noise with the biases
        assert item == "", (stage, c, has_bias, lnf, epm, order, item)
        assert th.register_split_hint(stage, c, has_bias, lnf, epm, order) == 1 and th.register_split_hint(stage, c, has_bias, lnf, epm, order, "") == 1
        for other_item in ("block", "passes"):
            assert th.register_split_hint(stage, c, has_bias, lnf, epm, order, other_item) == 0, (stage, c, has_bias, lnf, epm, order, other_item)
    # the measured rows (the A/B/A on the 212-SM Rubin perf-node class): the projection stage at c 128, bias off, pass-major tile order -- the
    # register form (+1.5..+1.8 % on the call) and the rolled form (+1.1..+1.2 %); the interleaved rolled renderings (-0.1..-0.7 %) and the
    # output stage at c 256 (its spill vanishes, its time does not move) are NOT rows
    assert th.register_split_hint("k1", 128, False, "registers", "packed", "major") == 1
    assert th.register_split_hint("k1", 128, False, "smem_rolled", "packed", "major") == 1
    # the three packed epilogue forms are ONE rendering without biases (their differences sit under has_bias): the bias-off lookup reads the
    # family's row; a bias-on rendering is never hinted whichever form it runs; "scalar" is not a member
    for epm in ("packed_hoist", "packed_biasfold"):
        assert th.register_split_hint("k1", 128, False, "registers", epm, "major") == 1, epm
        assert th.register_split_hint("k1", 128, False, "smem_rolled", epm, "major") == 1, epm
        assert th.register_split_hint("k1", 128, True, "registers", epm, "major") == (
            1 if epm == "packed_biasfold" else 0
        ), epm  # the bias-ON row names the fold form exactly
        assert th.register_split_hint("k1", 128, False, "smem_rolled", epm, "interleaved") == 0, epm
    assert th.register_split_hint("k1", 128, False, "registers", "scalar", "major") == 0
    assert th.register_split_hint("k1", 128, False, "smem_rolled", "packed", "interleaved") == 0
    assert th.register_split_hint("k3", 256, False, "smem_rolled", "packed") == 0
    assert th.register_split_hint("k1", 128, True, "registers", "packed", "major") == 0  # the packed form with biases: no row
    assert th.register_split_hint("k1", 128, True, "registers", "packed_biasfold", "major") == 1
    assert th.register_split_hint("k3", 128, True, "registers", "packed_biasfold") == 0
    assert th.register_split_hint("k1", 128, True, "smem_rolled", "packed_biasfold", "major") == 0  # a rolled K1 with biases: inside the bar
    # the API's wiring, GPU-free: a bare declaration hands the two stages the rule with THEIR bias flags, K1 its tile order AND its work item -- every
    # c 128 pick from N = 256 carries the LayerNorm-once item, so on the Rubin column a bare declaration runs UNHINTED there (no Rubin row names that
    # item: its renderings spill under the hint and the whole call runs faster without it), while the SM100 column's row hints the register-form K1
    # pick with the plain packed epilogue under the block item WITH the in-biases (the 512 and 1024 buckets below; the 256 bucket's rolled pick and
    # every bias-off plan stay unhinted, the output stage always); the per-tile item (a caller's `ln_once=""`) keeps the measured rows; the knob
    # `min_blocks_per_mp` overrides both stages
    sm100_column = th.arch_column_for_cc(_cc()) == "sm100"
    for N, hb_in, hb_out in ((512, False, False), (1024, False, True), (512, True, False), (512, True, True), (1024, True, True), (256, False, False)):
        want_k1 = 1 if (sm100_column and hb_in and N in (512, 1024)) else 0
        assert tapi.TriangleMultiplicativeUpdateFwd(_desc(N=N), "outgoing", c_hidden=128, has_in_bias=hb_in, has_out_bias=hb_out)._register_split_hints() == (
            want_k1,
            0,
        ), (N, hb_in, hb_out)
    per_tile = tapi.TriMulKnobs(ln_once="")
    assert tapi.TriangleMultiplicativeUpdateFwd(
        _desc(N=512), "outgoing", c_hidden=128, has_in_bias=False, has_out_bias=False, knobs=per_tile
    )._register_split_hints() == (_k1_row, 0)
    assert tapi.TriangleMultiplicativeUpdateFwd(
        _desc(N=1024), "outgoing", c_hidden=128, has_in_bias=False, has_out_bias=True, knobs=per_tile
    )._register_split_hints() == (_k1_row, 0)
    assert tapi.TriangleMultiplicativeUpdateFwd(
        _desc(N=512), "outgoing", c_hidden=128, has_in_bias=True, has_out_bias=False, knobs=per_tile
    )._register_split_hints() == (_k1_row, 0)
    assert tapi.TriangleMultiplicativeUpdateFwd(
        _desc(N=512), "outgoing", c_hidden=128, has_in_bias=True, has_out_bias=True, knobs=per_tile
    )._register_split_hints() == (_k1_row, 0)
    assert tapi.TriangleMultiplicativeUpdateFwd(
        _desc(N=1024), "outgoing", c_hidden=128, has_in_bias=True, has_out_bias=True, knobs=per_tile
    )._register_split_hints() == (
        0,
        0,
    )  # rolled K1 with biases, packed K3: no row either item
    assert tapi.TriangleMultiplicativeUpdateFwd(_desc(N=200), "outgoing", c_hidden=128, has_in_bias=False)._register_split_hints() == (
        0,
        0,
    )  # interleaved K1 (N < 256; the per-tile item): no row for the interleaved order
    assert tapi.TriangleMultiplicativeUpdateFwd(_desc(N=256), "outgoing", c_hidden=128, has_in_bias=False, knobs=per_tile)._register_split_hints() == (
        _k1_row,
        0,
    )  # major from N = 256 with the per-tile item: the rolled form's row
    assert tapi.TriangleMultiplicativeUpdateFwd(
        _desc(N=512, cz=256), "outgoing", c_hidden=256, has_in_bias=False, has_out_bias=False
    )._register_split_hints() == (0, 0)
    assert tapi.TriangleMultiplicativeUpdateFwd(
        _desc(N=512, cz=256), "outgoing", c_hidden=256, knobs=tapi.TriMulKnobs(min_blocks_per_mp=1)
    )._register_split_hints() == (1, 1)
    with pytest.raises(ValueError, match="stage"):
        th.register_split_hint("k2", 128, False, "smem_rolled", "packed")
    # the override reaches both stages' params; None hands each stage the rule with ITS bias flag
    assert th.resolve(512, 256, 1, "outgoing", knobs=th.TriMulKnobs(min_blocks_per_mp=1)).min_blocks_per_mp == 1
    assert th.resolve(512, 256, 1, "outgoing").min_blocks_per_mp is None
    assert k3.tail_params(256, 256, has_bias=False, min_blocks_per_mp=1).min_blocks_per_mp == 1
    assert k1.proj_params(256, 256, has_bias=False, has_mask=True, min_blocks_per_mp=1).min_blocks_per_mp == 1
    assert k3.tail_params(256, 256, has_bias=False).min_blocks_per_mp == 0 and k1.proj_params(128, 128, has_bias=True, has_mask=True).min_blocks_per_mp == 0


def test_k2_tile_knob_vocabulary_is_a_typed_decline_before_any_device_work():
    """An unknown tile-config name (or any out-of-vocabulary knob) is a ValueError from check_support -- never a bare KeyError out of
    compile() after the eligibility checks passed (GPU-free: the vocabulary check precedes the device checks)."""
    plan = tapi.TriangleMultiplicativeUpdateFwd(_desc(), "outgoing", c_hidden=128, k2_tile="CONFIG_bogus")
    assert plan.k2_tile == "CONFIG_bogus"
    with pytest.raises(ValueError, match="k2_tile"):
        plan.check_support()
    plan = tapi.TriangleMultiplicativeUpdateFwd(_desc(), "outgoing", c_hidden=128, knobs=tapi.TriMulKnobs(clc_stages=7))
    with pytest.raises(ValueError, match="clc_stages"):
        plan.check_support()
    # the geometry's measured default is what a bare declaration gets -- the column of the device the plan is declared on (the Rubin table on a
    # CPU declaration / cc 10.7, the SM100 column on cc 10.0 / 10.3; both pick the A-multicast x2 pair tile at N 512)
    from cudnn.triangle_multiplicative_update import heuristics as th

    plan = tapi.TriangleMultiplicativeUpdateFwd(_desc(N=512), "outgoing", c_hidden=128)
    assert plan.knobs == th.recommend(512, 128, 1, cc=plan.cc) and plan.k2_tile == k2.MULTICAST_A2_TILE_CONFIG
    assert th.recommend(512, 128, 1, cc=(10, 0)).epi_math == "packed" and th.recommend(512, 128, 1).epi_math == "packed_biasfold"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs one CUDA device for the tensors")
def test_execute_declines_tensors_on_another_device_than_the_plan():
    """A plan declared on cuda:1 handed cuda:0 tensors: every tensor agrees with x, but x is not on the PLAN's device -- the kernels are loaded
    on cuda:1 and would launch there with cuda:0 pointers.  A ValueError before any launch, on ONE GPU (the declaration never touches cuda:1
    until compile)."""
    n_res, c = 64, 128
    plan = tapi.TriangleMultiplicativeUpdateFwd(_desc(N=n_res, device="cuda:1"), "outgoing", c_hidden=c, has_mask=True, has_in_bias=True, has_out_bias=True)
    assert plan.device_index == 1
    x, mask, p = _inputs(n_res, 1, c, c, "all", True, True, seed=31, device="cuda:0")
    w = tapi.prepare_weights(*[p[n] for n in CUEQ_PARAMS[3:15]])
    out = torch.empty_like(x)
    ws = torch.empty(plan.get_workspace_size(), dtype=torch.uint8, device="cuda:0")
    with pytest.raises(ValueError, match="declared on cuda:1"):
        plan._check_execute_args(x, mask, w, out, ws)
    plan0 = tapi.TriangleMultiplicativeUpdateFwd(_desc(N=n_res, device="cuda:0"), "outgoing", c_hidden=c, has_mask=True, has_in_bias=True, has_out_bias=True)
    plan0._check_execute_args(x, mask, w, out, ws)  # the same tensors on the plan's device pass every host check


def _aligned_u8(nbytes: int, align: int = 256):
    """A ``nbytes`` uint8 CPU buffer whose base is ``align``-byte aligned (the CPU allocator aligns to 64 B; the workspace check wants 256), plus
    the backing allocation (kept alive by the caller) and the offset of the aligned base inside it."""
    backing = torch.empty(nbytes + align, dtype=torch.uint8)
    off = (-backing.data_ptr()) % align
    return backing[off : off + nbytes], backing, off


def test_out_and_workspace_byte_ranges_intersecting_an_operand_are_a_typed_decline_before_any_launch():
    """`out` (the output stage writes it while reading x) and the workspace (the projection stage writes the planes while reading x, the mask
    and the parameters; the contraction writes y) must be disjoint from every tensor the kernels read -- as BYTE RANGES, never by pointer
    equality: a caller-provided `out` whose range starts inside x (two views over one allocation) passed the old `data_ptr() ==` test and the
    output stage overwrote rows of x it had still to read (reproduced on hardware at N = 512: 16.7 M differing elements, max |diff| 2.56).
    The check runs on the host (CPU tensors suffice), names both operands and fires before any launch; disjoint views of the same allocation
    pass it.  Pinned per operand class: out over x (half-way and by ONE element), out over the mask, out over a parameter, out inside the
    workspace, the workspace over x."""
    n, c = 64, 128
    plan = tapi.TriangleMultiplicativeUpdateFwd(_desc(N=n, device="cpu"), "outgoing", c_hidden=c, has_mask=True, has_in_bias=True, has_out_bias=True)
    p = _params(c, c, True, True, seed=41)
    w = tapi.prepare_weights(*[p[k] for k in CUEQ_PARAMS[3:15]])
    mask = torch.ones(1, n, n, dtype=torch.bfloat16)
    numel = n * n * c
    ws_bytes = plan.get_workspace_size()
    ws, _keep, _ = _aligned_u8(ws_bytes)
    buf = torch.empty(3 * numel, dtype=torch.bfloat16)  # ONE allocation: x, an out starting half-way through x, a disjoint out after x
    x = buf[:numel].view(1, n, n, c)
    out_half = buf[numel // 2 : numel // 2 + numel].view(1, n, n, c)
    out_one = buf[numel - 1 : 2 * numel - 1].view(1, n, n, c)  # shares exactly x's last element
    out_disjoint = buf[numel : 2 * numel].view(1, n, n, c)
    out_far = buf[2 * numel :].view(1, n, n, c)
    assert out_half.data_ptr() != x.data_ptr() and out_one.data_ptr() != x.data_ptr()  # the old test saw distinct pointers
    with pytest.raises(ValueError, match=r"out must not alias x.*byte ranges intersect") as info:
        plan._check_execute_args(x, mask, w, out_half, ws)
    assert "out" in str(info.value) and " x" in str(info.value)
    with pytest.raises(ValueError, match="out must not alias x"):
        plan._check_execute_args(x, mask, w, out_one, ws)
    plan._check_execute_args(x, mask, w, out_disjoint, ws)  # distinct ranges of one allocation pass (the hardware control is bitwise the baseline)
    plan._check_execute_args(x, mask, w, out_far, ws)
    with pytest.raises(ValueError, match="out must not alias x"):  # full aliasing is the degenerate intersection, still named
        plan._check_execute_args(x, mask, w, x, ws)
    # out over the MASK (the projection stage reads it while the output stage... no: K1 reads the mask, K3 writes out -- a later launch, but the
    # two overlap under PDL and a CUDA graph replays them back to back: declined like any read buffer)
    mbuf = torch.empty(numel + n * n // 2, dtype=torch.bfloat16)
    mask_v = mbuf[: n * n].view(1, n, n).fill_(1.0)
    out_m = mbuf[n * n // 2 : n * n // 2 + numel].view(1, n, n, c)
    with pytest.raises(ValueError, match="out must not alias mask"):
        plan._check_execute_args(x, mask_v, w, out_m, ws)
    # out over a PARAMETER (the output gate's weight)
    wbuf = torch.empty(numel + c * c // 2, dtype=torch.bfloat16)
    g_out = wbuf[: c * c].view(c, c).copy_(w.g_out_weight)
    out_w = wbuf[c * c // 2 : c * c // 2 + numel].view(1, n, n, c)
    w_al = tapi.TriMulWeights(**{**{f: getattr(w, f) for f in w.__dataclass_fields__}, "g_out_weight": g_out})
    with pytest.raises(ValueError, match=r"out must not alias weights\.g_out_weight"):
        plan._check_execute_args(x, mask, w_al, out_w, ws)
    plan._check_execute_args(x, mask, w_al, out_far, ws)  # the relocated weight itself is fine
    # out INSIDE the workspace, and the workspace over x
    big, _keep2, off = _aligned_u8(ws_bytes + 512 + 2 * numel)
    ws2 = big[:ws_bytes]
    out_in_ws = big[512 : 512 + 2 * numel].view(torch.bfloat16).view(1, n, n, c)
    with pytest.raises(ValueError, match="out must not alias workspace"):
        plan._check_execute_args(x, mask, w, out_in_ws, ws2)
    x_in_ws = big[256 : 256 + 2 * numel].view(torch.bfloat16).view(1, n, n, c)
    with pytest.raises(ValueError, match="workspace must not alias x"):
        plan._check_execute_args(x_in_ws, mask, w, out_far, ws2)
    # the span helper itself: a view's range is [base, last element + itemsize), an empty tensor spans nothing, a stride-0 broadcast one element
    from cudnn.api_base import tensor_byte_span

    t = torch.empty(10, dtype=torch.bfloat16)
    assert tensor_byte_span(t[2:5]) == (t.data_ptr() + 4, t.data_ptr() + 10)
    assert tensor_byte_span(t[::3]) == (t.data_ptr(), t.data_ptr() + 2 * 9 + 2)
    lo, hi = tensor_byte_span(t[:0])
    assert lo == hi, "an empty tensor spans nothing (its base may read 0)"
    assert tensor_byte_span(t[3].expand(7)) == (t.data_ptr() + 6, t.data_ptr() + 8)


def test_drop_in_validates_out_before_use_and_returns_it_as_given(monkeypatch):
    """`out` is checked for tensor-ness and rank BEFORE anything is done with it (a non-tensor used to die with AttributeError at the
    unsqueeze, after the plan was built), and a caller's `out` is returned AS GIVEN -- the function squeezes only an output it allocated: a
    4-D out with a 3-D x comes back as the caller's 4-D object, a 3-D out with a 3-D x as that same 3-D object (execute sees its 4-D view).
    GPU-free: the plan is a stub recording what execute received (the launches are the device cells'), the stream handle a constant."""
    c, n = 128, 64
    x3 = torch.randn(n, n, c).to(torch.bfloat16)
    p = _params(c, c, False, False, seed=43)
    kw = {k: p[k] for k in CUEQ_PARAMS[3:15]}
    received = []

    class _StubPlan:
        def get_workspace_size(self):
            return 0

        def execute(self, x4, m, weights, out4, workspace, current_stream=None):
            received.append((x4, out4))

    monkeypatch.setattr(tapi, "_cached_plan", lambda *a, **k: _StubPlan())
    monkeypatch.setattr(tapi, "_stream_handle", lambda stream, device=None: 0)
    with pytest.raises(TypeError, match="out must be a torch.Tensor"):
        tapi.triangle_multiplicative_update(x3, "outgoing", None, **kw, out=[1.0])
    with pytest.raises(ValueError, match=r"out must be \[B, N, N, c_z\]"):
        tapi.triangle_multiplicative_update(x3, "outgoing", None, **kw, out=torch.empty(n * n * c, dtype=torch.bfloat16))
    assert received == [], "a malformed out is declined before the plan runs"
    out4 = torch.empty(1, n, n, c, dtype=torch.bfloat16)
    got = tapi.triangle_multiplicative_update(x3, "outgoing", None, **kw, out=out4)
    assert got is out4 and received[-1][1] is out4 and tuple(received[-1][0].shape) == (1, n, n, c)
    out3 = torch.empty(n, n, c, dtype=torch.bfloat16)
    got = tapi.triangle_multiplicative_update(x3, "outgoing", None, **kw, out=out3)
    assert got is out3 and received[-1][1].data_ptr() == out3.data_ptr() and tuple(received[-1][1].shape) == (1, n, n, c)
    got = tapi.triangle_multiplicative_update(x3.unsqueeze(0), "outgoing", None, **kw, out=out3)  # a 4-D x with a 3-D out at B = 1
    assert got is out3
    got = tapi.triangle_multiplicative_update(x3, "outgoing", None, **kw)  # allocated by the function: squeezed back to x's rank
    assert tuple(got.shape) == (n, n, c) and tuple(received[-1][1].shape) == (1, n, n, c)
    got = tapi.triangle_multiplicative_update(x3.unsqueeze(0), "outgoing", None, **kw)
    assert tuple(got.shape) == (1, n, n, c)


@requires_served_arch
def test_out_views_of_one_allocation_on_the_device_disjoint_runs_intersecting_declines_before_any_launch():
    """The review's hardware recipe (N = 512, c = 128, contiguous bf16 views over ONE allocation, `out` beginning half-way through x): the call
    is declined before any launch -- x, which the overlapping `out` would have overwritten, is intact -- while a disjoint `out` of the same
    allocation is served and equals the separately allocated baseline bitwise; a 4-D `out` with a 3-D x comes back as the caller's object."""
    n_res, c = 512, 128
    x_src, mask, p = _inputs(n_res, 1, c, c, "all", False, False, seed=45)
    numel = x_src.numel()
    buf = torch.empty(2 * numel + numel // 2, dtype=torch.bfloat16, device="cuda")
    x = buf[:numel].view(1, n_res, n_res, c)
    x.copy_(x_src)
    base = _call(x, "outgoing", mask, p)
    out_half = buf[numel // 2 : numel // 2 + numel].view(1, n_res, n_res, c)
    out_disjoint = buf[numel + numel // 2 :].view(1, n_res, n_res, c)
    torch.cuda.synchronize()
    with pytest.raises(ValueError, match="out must not alias x"):
        _call(x, "outgoing", mask, p, out=out_half)
    torch.cuda.synchronize()
    assert torch.equal(x, x_src), "the declined call launched nothing: x is intact"
    got = _call(x, "outgoing", mask, p, out=out_disjoint)
    torch.cuda.synchronize()
    assert got is out_disjoint and torch.equal(out_disjoint, base)
    out4 = torch.empty_like(x)
    got = _call(x[0], "outgoing", mask[0], p, out=out4)
    torch.cuda.synchronize()
    assert got is out4 and torch.equal(out4, base)


@requires_served_arch
def test_knobs_are_performance_only_bitwise_and_the_unfit_tile_is_a_typed_decline():
    """Review-grade accept + reject for the knob surface: a non-default contraction tile, the pass-major projection order, the transpose
    prologue and a 1-deep ring compute the SAME bits as the default plan (performance only); a catalog config of another arch family is a
    typed NotImplementedError from check_support (the eligibility probe), before any compile."""
    n_res, batch, c = 128, 1, 128
    x, mask, p = _inputs(n_res, batch, c, c, "random", True, True, seed=27)
    w = tapi.prepare_weights(*[p[n] for n in CUEQ_PARAMS[3:15]])
    outs = []
    for knobs in (
        None,
        tapi.TriMulKnobs(k2_tile=k2.SMALL_N_TILE_CONFIG, k1_pass_order="major", k3_prologue="transpose", clc_stages=1),
        tapi.TriMulKnobs(k2_tile=k2.SMALL_N_MULTICAST_B2_TILE_CONFIG, clc_stages=4),
        tapi.TriMulKnobs(k_block=64),  # the row tile: 4 bi x 64 k per pair tile instead of the rule's 2 x 128 at N = 128
        tapi.TriMulKnobs(k_block=256),  # 1 bi x 256 k: the second half of every pair tile is padding at N = 128 (legal, wasteful)
        tapi.TriMulKnobs(min_blocks_per_mp=1),  # the .minnctapersm hint: ptxas honours the two stages' register split (a launch attribute, the same body)
        tapi.TriMulKnobs(
            ln_pass_form="smem_rolled_pf"
        ),  # the pipelined rolled LayerNorm re-read: the same words, the same order (K1 here; K3 keeps the transpose form)
        tapi.TriMulKnobs(epi_math="packed_hoist"),  # the bias words loaded before the TMEM loads: the same words, the same ops
        tapi.TriMulKnobs(epi_math="packed_biasfold"),  # the gate bias staged halved, folded into the tanh argument's fma (output-bitwise, both stages)
    ):
        plan = tapi.TriangleMultiplicativeUpdateFwd(x, "outgoing", c_hidden=c, has_mask=True, has_in_bias=True, has_out_bias=True, knobs=knobs)
        assert plan.check_support()
        plan.compile()
        out = torch.full_like(x, _SENTINEL)
        ws = torch.empty(plan.get_workspace_size(), dtype=torch.uint8, device="cuda")
        plan.execute(x, mask, w, out, ws)
        torch.cuda.synchronize()
        assert not (out.float() == _SENTINEL).any() and torch.isfinite(out.float()).all()
        outs.append(out)
    assert all(torch.equal(outs[0], o) for o in outs[1:]), "a performance knob changed the output bits"
    plan = tapi.TriangleMultiplicativeUpdateFwd(x, "outgoing", c_hidden=c, k2_tile=k2.SMALL_N_TILE_CONFIG)
    assert plan.check_support()  # the single-axis override takes the same path
    other_family = "CONFIG_sm120_512x512x32_16x16x32_cluster1x1_warps2x4"  # a catalog name this arch tree cannot render
    plan = tapi.TriangleMultiplicativeUpdateFwd(x, "outgoing", c_hidden=c, knobs=tapi.TriMulKnobs(k2_tile=other_family))
    with pytest.raises(ValueError, match="k2_tile"):  # outside the MEASURED domain: the vocabulary check fires first
        plan.check_support()
    # the eligibility probe itself: a measured-domain tile the engine cannot run for THIS geometry is a NotImplementedError, never a
    # bare error -- exercised directly (every domain member serves every geometry of the sweep, so no public declaration reaches it today)
    plan = tapi.TriangleMultiplicativeUpdateFwd(x, "outgoing", c_hidden=c)
    plan.k2_tile = other_family
    with pytest.raises(NotImplementedError, match="cannot serve|sm120|arch"):
        plan._probe_contraction_tile()


@requires_served_arch
def test_tree_ln_form_knob_is_admitted_meets_the_oracle_and_its_distance_to_the_default_plan_is_measured():
    """`TriMulKnobs(ln_pass_form="smem_rolled_tree")` (the LayerNorm statistics in four partial sums -- a different fp32 ORDER) is a caller's knob the
    API admits: the output meets the SAME three-part oracle as the default plan (nothing widened) and its distance to the default plan's output is
    REPORTED, not pinned bitwise (the one form in LN_PASS_FORMS_REORDERED); the two-launch check holds on it."""
    n_res, batch, c = 128, 1, 256  # c = 256: both the projection and the output stage run the re-read forms (the c 128 output stage keeps the transpose form)
    x, mask, p = _inputs(n_res, batch, c, c, "random", True, True, seed=31)
    w = tapi.prepare_weights(*[p[n] for n in CUEQ_PARAMS[3:15]])
    outs = {}
    for name, knobs in (("default", None), ("tree", tapi.TriMulKnobs(ln_pass_form="smem_rolled_tree"))):
        plan = tapi.TriangleMultiplicativeUpdateFwd(x, "outgoing", c_hidden=c, has_mask=True, has_in_bias=True, has_out_bias=True, knobs=knobs)
        assert plan.check_support()
        plan.compile()
        out = torch.full_like(x, _SENTINEL)
        ws = torch.empty(plan.get_workspace_size(), dtype=torch.uint8, device="cuda")
        plan.execute(x, mask, w, out, ws)
        out2 = torch.full_like(x, _SENTINEL)
        plan.execute(x, mask, w, out2, ws)
        torch.cuda.synchronize()
        assert not (out.float() == _SENTINEL).any() and torch.isfinite(out.float()).all(), name
        assert torch.equal(out, out2), f"{name}: the two-launch check"
        outs[name] = out
    ref = tref.triangle_multiplicative_update_reference(x, "outgoing", mask, **{k: p[k] for k in CUEQ_PARAMS[3:15]}, rounding="kernel")
    _check_oracle(outs["default"], ref, "default plan")
    _check_oracle(outs["tree"], ref, "ln_pass_form=smem_rolled_tree")
    n_diff = int((outs["default"].view(torch.int16) != outs["tree"].view(torch.int16)).sum().item())
    d = (outs["default"].float() - outs["tree"].float()).abs()
    print(f"tree vs default plan: {n_diff} cells differ (of {d.numel()}); max |diff| {d.max().item():.4g}")


@requires_served_arch
def test_stock_caller_invocation_shape_hits_the_caches_and_matches_the_class_api():
    """Provider parity on the code path a stock caller takes: `from cudnn import triangle_multiplicative_update` (the lazy export), the
    call spelled with cuEquivariance's KEYWORDS (as the model code does: `triangle_multiplicative_update(x, direction=..., mask=...,
    norm_in_weight=..., ...)`), under torch.no_grad() (inference), twice -- the second call must hit BOTH caches (plan + prepared weights)
    and return the same bits; the result equals the class API's bitwise."""
    from cudnn import triangle_multiplicative_update as fn  # the stock import

    n_res, batch, c = 128, 1, 128
    x, mask, p = _inputs(n_res, batch, c, c, "random", True, True, seed=21)
    kw = {k: p[k] for k in CUEQ_PARAMS[3:15]}
    with torch.no_grad():
        out1 = fn(x, direction="incoming", mask=mask, eps=1e-5, precision=None, **kw)
        n_plans, n_weights = len(tapi._PLANS), len(tapi._WEIGHTS)
        out2 = fn(x, direction="incoming", mask=mask, eps=1e-5, precision=None, **kw)
    torch.cuda.synchronize()
    assert len(tapi._PLANS) == n_plans and len(tapi._WEIGHTS) == n_weights, "the second stock call must hit the plan and the weight caches"
    assert torch.equal(out1, out2)
    w = tapi.prepare_weights(**kw)
    plan = tapi.TriangleMultiplicativeUpdateFwd(x, "incoming", c_hidden=c, has_mask=True, has_in_bias=True, has_out_bias=True)
    plan.compile()
    out = torch.empty_like(x)
    ws = torch.empty(plan.get_workspace_size(), dtype=torch.uint8, device="cuda")
    plan.execute(x, mask, w, out, ws)
    torch.cuda.synchronize()
    assert torch.equal(out, out1)
    assert out1.grad_fn is None and not out1.requires_grad


@requires_served_arch
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two visible GPUs")
def test_plan_declared_on_a_non_current_device_compiles_and_runs_there():
    """The declaration names cuda:1 while torch's current device is cuda:0: compile loads the plans on the DECLARED device (the K2 cache is
    keyed on it), execute runs there, and the result equals the same computation on cuda:0 bitwise (the same kernels, the same inputs)."""
    n_res, batch, c = 128, 1, 128
    x0, mask0, p0 = _inputs(n_res, batch, c, c, "padded", True, True, seed=23, device="cuda:0")
    dev1 = torch.device("cuda:1")
    x1, mask1 = x0.to(dev1), mask0.to(dev1)
    p1 = {k: (v.to(dev1) if v is not None else None) for k, v in p0.items()}
    assert torch.cuda.current_device() == 0
    with torch.no_grad():
        out1 = _call(x1, "outgoing", mask1, p1)  # the drop-in: plan keyed on str(x.device) = cuda:1
        out0 = _call(x0, "outgoing", mask0, p0)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    assert out1.device == dev1 and torch.cuda.current_device() == 0
    assert torch.equal(out1.cpu(), out0.cpu())
    plan1 = tapi.TriangleMultiplicativeUpdateFwd(x1, "outgoing", c_hidden=c, has_mask=True, has_in_bias=True, has_out_bias=True)
    assert plan1.device_index == 1
    plan1.compile()
    w1 = tapi.prepare_weights(*[p1[n] for n in CUEQ_PARAMS[3:15]])
    out = torch.empty_like(x1)
    ws = torch.empty(plan1.get_workspace_size(), dtype=torch.uint8, device=dev1)
    plan1.execute(x1, mask1, w1, out, ws)
    torch.cuda.synchronize(1)
    assert torch.equal(out.cpu(), out0.cpu())
