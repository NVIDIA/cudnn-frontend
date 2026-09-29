# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The four backward projection GEMMs of the gated attention block, on the shipped FROST GEMM.

``nn.Linear`` orientation (``api_bwd.py``): ``dW = dY^T @ X`` (A M-major, B N-major) and
``dX = dY @ W`` (B N-major).  Nothing here writes a kernel; what is under test is the
DRIVER's claims (``kernels/proj_gemm.py``):

* the appended ``a_major`` / ``b_major`` kwargs declare the graph strides the FROST GEMM
  engine renders as its M-major-A / N-major-B template arms, and the tile the block
  FORCES for every ``n % 256 == 0`` (``_forced_tile_config``) renders and computes
  correctly on cc 10.7 under those majors -- the riskiest assumption of the backward
  track (IMPL spec R3), settled by :func:`test_forced_tile_renders_mn_major_on_cc107`;
* ``run_wgrad_gemm`` / ``run_dgrad_gemm`` bind transposed VIEWS (zero-copy) against a plan
  DECLARED with the matching majors, and refuse a view whose stride-1 axis is not the
  declared one BEFORE any launch -- the graph fallback would read the declared strides
  with no check (a silent reinterpretation, spec R12);
* ``split_k`` semantics: ``0`` the driver's pick, ``1`` a pinned JIT at one slice that
  refuses the graph fallback, ``S >= 2`` the fixed-order two-kernel split whose fp32
  partials ride ``plan.workspace_bytes``.

The accept tests need a Rubin device (the block targets SM107 only); the reject / host
tests run anywhere.
"""

import functools
import os
import sys

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

from cudnn.gated_attention_block import GatedAttentionBlockGeometry  # noqa: E402
from cudnn.gated_attention_block.kernels.proj_gemm import (  # noqa: E402
    ProjGemmPlan,
    SplitKPinRefused,
    _frost_plan_index,
    build_proj_gemm,
    run_dgrad_gemm,
    run_wgrad_gemm,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_SM107 = (10, 7)
_FORCED_TILE = "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma"


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


requires_rubin = pytest.mark.skipif(_cc() != _SM107, reason=f"the block targets SM107 only; found {_cc()}")


# ---------------------------------------------------------------------------
# Tolerance: bf16 / f16 inputs, fp32 accumulate, bf16 / f16 output, vs an fp64 oracle
# ---------------------------------------------------------------------------
#
# The oracle multiplies the SAME quantized inputs in fp64, so the only differences are
# (i) the output rounding -- half an ulp = 2^-9 relative for bf16 (8 significand bits),
# 2^-12 for f16 -- and (ii) the fp32 accumulation order over K terms, whose error is
# ABSOLUTE and about sqrt(K) * 2^-24 * |typical partial sum| (K = 8192: ~1e-5 of the
# tensor's scale, three orders below the bound).  So rtol = 2^-7 (= 2 bf16 ulps, 4x the
# half-ulp; the SAME bound for f16 is 16x its half-ulp -- deliberately one bound per
# suite, not per dtype) and atol = 2^-7 * max|ref| (tensor-scaled, test/AGENTS.md: a
# fixed absolute bound turns wrong when the magnitudes grow).  Never widened; a
# violation is reported with its magnitude.
_RTOL = 2.0**-7


def _assert_close_vs_fp64(out: torch.Tensor, ref64: torch.Tensor, what: str) -> None:
    ref_max = ref64.abs().max().item()
    atol = _RTOL * ref_max
    diff = (out.double() - ref64).abs()
    scale = atol + _RTOL * ref64.abs()
    worst = (diff / scale).max().item()
    msg = f"{what}: max|diff| = {diff.max().item():.4g} (max|ref| = {ref_max:.4g}); worst cell at {worst:.3f} of the bound (rtol = atol/max|ref| = {_RTOL})"
    print(msg)
    torch.testing.assert_close(out.double(), ref64, rtol=_RTOL, atol=atol, msg=msg)


def _wgrad_operands(rows: int, t: int, cols: int, dtype: torch.dtype, seed: int = 0):
    """``dy_like [T, rows]`` and ``x [T, cols]`` (both row-major -- what the block holds), plus ``dw [rows, cols]``."""
    torch.manual_seed(seed)
    dy = (torch.randn(t, rows, device="cuda", dtype=torch.float32) * 0.5).to(dtype)
    x = (torch.randn(t, cols, device="cuda", dtype=torch.float32) * 0.5).to(dtype)
    dw = torch.zeros(rows, cols, device="cuda", dtype=dtype)
    return dy, x, dw


def _dgrad_operands(t: int, k: int, n: int, dtype: torch.dtype, seed: int = 0):
    """``dy_like [T, K]`` and the UN-transposed row-major weight ``w [K, N]``, plus ``dx [T, N]``."""
    torch.manual_seed(seed)
    dy = (torch.randn(t, k, device="cuda", dtype=torch.float32) * 0.5).to(dtype)
    w = (torch.randn(k, n, device="cuda", dtype=torch.float32) * 0.05).to(dtype)
    dx = torch.zeros(t, n, device="cuda", dtype=dtype)
    return dy, w, dx


def _ws(plan: ProjGemmPlan) -> torch.Tensor:
    return torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda")


# The block's two geometries: the test geometry every cutedsl/ suite runs, and Qwen3.5-397B's
# full-attention layer.  `(dm, HD, N)` = (d_model, h_q * d_head, n_qkvg); every backward GEMM's
# n is one of dm / HD (both % 256 == 0 at both geometries), so all four take the FORCED tile.
_GEOMS = {
    "test": GatedAttentionBlockGeometry(d_model=512, h_q=8, h_kv=2, d_head=256, rope_dim=64),
    "397B": GatedAttentionBlockGeometry(d_model=4096, h_q=32, h_kv=2, d_head=256, rope_dim=64),
}


def _dims(geom_id: str) -> tuple[int, int, int]:
    g = _GEOMS[geom_id]
    return g.d_model, g.h_q * g.d_head, g.n_qkvg


def _stage_mkn(stage: str, geom_id: str, t: int) -> tuple[int, int, int]:
    """``(m, k, n)`` of one backward projection plan at the block's geometry (spec section 1.2 table)."""
    dm, hd, n_qkvg = _dims(geom_id)
    return {
        "B1_dw_o": (dm, t, hd),  # dW_o [dm, HD] = dY[T, dm]^T @ O_gated[T, HD]
        "B7_dw_qkvg": (n_qkvg, t, dm),  # dW_qkvg [N, dm] = dQKVG[T, N]^T @ h[T, dm]
        "B2_do_gated": (t, dm, hd),  # dO_gated [T, HD] = dY[T, dm] @ W_o[dm, HD]
        "B8_dh": (t, n_qkvg, dm),  # dh [T, dm] = dQKVG[T, N] @ W_qkvg[N, dm]
    }[stage]


_WGRAD = ("B1_dw_o", "B7_dw_qkvg")
_DGRAD = ("B2_do_gated", "B8_dh")


@functools.lru_cache(maxsize=None)
def _plan(kind: str, m: int, k: int, n: int, dtype: torch.dtype, split_k: int = 0) -> ProjGemmPlan:
    """One plan per distinct declaration for the whole process: the rendered kernel is symbolic in
    (M, N, K) and cached by source digest, but every build still asks the backend for its ranked
    list, so the tests below share plans rather than rebuild them."""
    majors = dict(a_major="m", b_major="n") if kind == "wgrad" else dict(a_major="k", b_major="n")
    return build_proj_gemm(m=m, k=k, n=n, dtype=dtype, label=f"{kind}_{m}x{k}x{n}_{str(dtype).replace('torch.', '')}_sk{split_k}", split_k=split_k, **majors)


def _park_the_default_stream(seconds: float = 0.5) -> None:
    """Enqueue a long spin on torch's CURRENT (default) stream so that anything wrongly launched
    there runs LATE -- after a side stream is long done (test_block_end_to_end.py's probe)."""
    if hasattr(torch.cuda, "_sleep"):
        torch.cuda._sleep(int(seconds * 2.0e9))  # cycles at ~2 GHz
        return
    x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    for _ in range(16):
        x = x @ x


# ---------------------------------------------------------------------------
# R3 -- THE probe: the forced tile renders and computes M-major A / N-major B on cc 10.7
# ---------------------------------------------------------------------------


@requires_rubin
def test_forced_tile_renders_mn_major_on_cc107():
    """The riskiest assumption of the backward track (spec R3): the bf16 M-major-A /
    N-major-B renderings of the tile the block FORCES (``..._cluster2x1_2ctamma``, D10)
    have never run on cc 10.7 -- the FROST GEMM suite covers the layouts on the sm100
    pipeline, and the block's own forward only ever rendered K-major operands.

    B1's geometry at the 397B column shapes: ``dW_o [dm=4096, HD=8192] = dY[T=8192,
    dm]^T @ O_gated[T, HD]``.  Three assertions, in the order a failure would show:
    the plan IS the forced JIT (a fallback to the heuristic is a FAILURE here, not a
    skip -- it changes route, config and possibly split-K), the output is not silently
    zero (the >256 KiB tcgen05 descriptor landmine's signature, ``test_proj_gemm.py::
    test_output_is_not_silently_zero``), and the numbers match the fp64 oracle."""
    m, k, n = 4096, 8192, 8192
    plan = build_proj_gemm(m=m, k=k, n=n, dtype=torch.bfloat16, label="r3_probe_dw_o", a_major="m", b_major="n")
    assert plan.tile_config_name == _FORCED_TILE, f"not the forced tile: {plan.tile_config_name!r} (route {plan.route!r})"
    assert plan.jit is not None, f"no JIT artifact -- the forced compile fell back to the graph heuristic (route {plan.route!r})"
    assert (plan.a_major, plan.b_major) == ("m", "n")
    dy, x, dw = _wgrad_operands(m, k, n, torch.bfloat16)
    run_wgrad_gemm(plan, dy, x, dw, _ws(plan))
    torch.cuda.synchronize()
    nonzero = (dw != 0).float().mean().item()
    assert nonzero > 0.99, f"only {100 * nonzero:.1f}% of dW is non-zero -- suspect the MN-major SMEM descriptor / the >256 KiB descriptor version"
    assert torch.isfinite(dw.float()).all()
    ref = dy.double().T @ x.double()
    _assert_close_vs_fp64(dw, ref, "dW_o = dY^T @ O_gated (m=4096, k=8192, n=8192, bf16)")


# ---------------------------------------------------------------------------
# Numerics vs the fp64 oracle -- the four plans at both geometries
# ---------------------------------------------------------------------------
#
# T = 4104 is not a multiple of any tile: a ragged K tail for the wgrads (K = T; the CTA K tile
# is 64 bf16 elements, 4104 = 64 * 64 + 8) and a ragged M tail for the dgrads (M = T, tile 128).
# f16 rides the small geometry only (one more rendering per driver; the same bound).

_WGRAD_CASES = [(st, g, t, torch.bfloat16) for g in ("test", "397B") for st in _WGRAD for t in (2048, 4104)] + [
    (st, "test", 2048, torch.float16) for st in _WGRAD
]
_DGRAD_CASES = [(st, g, t, torch.bfloat16) for g in ("test", "397B") for st in _DGRAD for t in (2048, 4104)] + [
    (st, "test", 2048, torch.float16) for st in _DGRAD
]


def _case_id(c) -> str:
    return f"{c[0]}-{c[1]}-T{c[2]}-{str(c[3]).replace('torch.', '')}"


@requires_rubin
@pytest.mark.parametrize("stage,geom_id,t,dtype", _WGRAD_CASES, ids=[_case_id(c) for c in _WGRAD_CASES])
def test_wgrad_matches_fp64(stage, geom_id, t, dtype):
    """``dW = dY^T @ X`` through ``run_wgrad_gemm`` on the plan each stage builds: both weight
    gradients (B1, B7) at the test geometry and at the 397B column shapes, vs ``dy.double().T @
    x.double()`` within the bf16-output bound above."""
    m, k, n = _stage_mkn(stage, geom_id, t)
    plan = _plan("wgrad", m, k, n, dtype)
    assert plan.jit is not None and plan.tile_config_name == _FORCED_TILE, (plan.tile_config_name, plan.route)
    dy, x, dw = _wgrad_operands(m, k, n, dtype)
    run_wgrad_gemm(plan, dy, x, dw, _ws(plan))
    torch.cuda.synchronize()
    assert (dw != 0).float().mean().item() > 0.99, "dW is (nearly) all zero"
    _assert_close_vs_fp64(dw, dy.double().T @ x.double(), f"{stage} @ {geom_id}, T={t}, {dtype}")


@requires_rubin
@pytest.mark.parametrize("stage,geom_id,t,dtype", _DGRAD_CASES, ids=[_case_id(c) for c in _DGRAD_CASES])
def test_dgrad_matches_fp64(stage, geom_id, t, dtype):
    """``dX = dY @ W`` through ``run_dgrad_gemm`` with the UN-transposed row-major weight: both
    input gradients (B2, B8), same geometries, same bound."""
    m, k, n = _stage_mkn(stage, geom_id, t)
    plan = _plan("dgrad", m, k, n, dtype)
    assert plan.jit is not None and plan.tile_config_name == _FORCED_TILE, (plan.tile_config_name, plan.route)
    dy, w, dx = _dgrad_operands(m, k, n, dtype)
    run_dgrad_gemm(plan, dy, w, dx, _ws(plan))
    torch.cuda.synchronize()
    assert (dx != 0).float().mean().item() > 0.99, "dX is (nearly) all zero"
    _assert_close_vs_fp64(dx, dy.double() @ w.double(), f"{stage} @ {geom_id}, T={t}, {dtype}")


# ---------------------------------------------------------------------------
# The driver's contract: zero-copy binds, the plan pins, split-K, workspace, stream
# ---------------------------------------------------------------------------


@requires_rubin
def test_bind_is_zero_copy(monkeypatch):
    """The drivers bind VIEWS: the A / B / C buffers ``run_proj_gemm`` receives share their
    ``data_ptr()`` with the caller's tensors and carry the declared rank-3 dims and strides, and
    two further executes allocate nothing (``torch.cuda.memory_allocated`` delta 0 -- Rule 1: no
    per-execute allocation; a ``reshape`` that copied would show here)."""
    import cudnn.gated_attention_block.kernels.proj_gemm as pg

    seen = {}
    orig = pg.run_proj_gemm

    def spy(plan_, a, w, out, workspace, handle=None, **kw):
        seen.update(a=a, w=w, out=out)
        return orig(plan_, a, w, out, workspace, handle, **kw)

    monkeypatch.setattr(pg, "run_proj_gemm", spy)
    m, k, n = _stage_mkn("B1_dw_o", "test", 2048)
    plan = _plan("wgrad", m, k, n, torch.bfloat16)
    dy, x, dw = _wgrad_operands(m, k, n, torch.bfloat16)
    ws = _ws(plan)
    run_wgrad_gemm(plan, dy, x, dw, ws)
    torch.cuda.synchronize()
    assert seen["a"].data_ptr() == dy.data_ptr() and seen["w"].data_ptr() == x.data_ptr() and seen["out"].data_ptr() == dw.data_ptr()
    assert tuple(seen["a"].shape) == (1, m, k) and tuple(seen["a"].stride()) == (k * m, 1, m), (seen["a"].shape, seen["a"].stride())
    assert tuple(seen["w"].shape) == (1, k, n) and tuple(seen["w"].stride()) == (k * n, n, 1)
    before = torch.cuda.memory_allocated()
    run_wgrad_gemm(plan, dy, x, dw, ws)
    run_wgrad_gemm(plan, dy, x, dw, ws)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before, "an execute allocated"
    # dgrad: the un-transposed weight is bound as the declared [1, K, N] row-major view.
    m, k, n = _stage_mkn("B2_do_gated", "test", 2048)
    plan = _plan("dgrad", m, k, n, torch.bfloat16)
    dy, w, dx = _dgrad_operands(m, k, n, torch.bfloat16)
    ws = _ws(plan)
    run_dgrad_gemm(plan, dy, w, dx, ws)
    torch.cuda.synchronize()
    assert seen["a"].data_ptr() == dy.data_ptr() and seen["w"].data_ptr() == w.data_ptr() and seen["out"].data_ptr() == dx.data_ptr()
    assert tuple(seen["w"].shape) == (1, k, n) and tuple(seen["w"].stride()) == (k * n, n, 1)
    before = torch.cuda.memory_allocated()
    run_dgrad_gemm(plan, dy, w, dx, ws)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before, "an execute allocated"


@requires_rubin
@pytest.mark.parametrize("stage", _WGRAD + _DGRAD)
def test_plan_is_frost_and_named(stage):
    """Every backward plan at the test geometry: ``frost_gemm`` is pinned in the graph's ranked
    list (the name, exact or with its knob bracket), the plan IS the forced JIT at the named tile
    (D10 -- a fallback to the heuristic is a FAILURE here, not a skip: it changes route, config and
    possibly split-K), one slice, and the majors are stamped for the perf table."""
    m, k, n = _stage_mkn(stage, "test", 2048)
    kind = "wgrad" if stage in _WGRAD else "dgrad"
    plan = _plan(kind, m, k, n, torch.bfloat16)
    names = [plan.graph.get_plan_name_at_index(i) for i in range(len(plan.graph.plans))]
    assert _frost_plan_index(names) is not None, names
    assert plan.tile_config_name == _FORCED_TILE, (plan.tile_config_name, plan.route)
    assert plan.jit is not None and plan.route == "graph+jit"
    assert plan.jit.config.split_k_slices == 1 and plan.split_k == 0
    assert (plan.a_major, plan.b_major) == (("m", "n") if kind == "wgrad" else ("k", "n"))
    assert (plan.m, plan.k, plan.n) == (m, k, n)


@requires_rubin
@pytest.mark.parametrize("split_k", [0, 2], ids=["auto", "split_k=2"])
def test_two_runs_bitwise(split_k):
    """Determinism by construction (D6): two executes of the same plan agree bit for bit -- the
    split-K reducer is a fixed-order tree, no atomics anywhere.  The ``split_k=2`` plan is a JIT
    CONFIG replace (``plan.jit.config.split_k_slices == 2``), not a graph-route knob replay, and its
    fp32 partials show in ``plan.workspace_bytes``; its result meets the same fp64 bound."""
    m, k, n = _stage_mkn("B1_dw_o", "test", 2048)
    plan = _plan("wgrad", m, k, n, torch.bfloat16, split_k)
    # The catalog spells a split config by suffix (`TileConfig.name`, `by_name` round-trips it), so the
    # stamped name says which slice count runs -- what the perf table wants to see.
    assert plan.tile_config_name == _FORCED_TILE + ("_splitK2" if split_k == 2 else "") and plan.jit is not None
    assert plan.jit.config.split_k_slices == (2 if split_k == 2 else 1) and plan.split_k == split_k
    if split_k == 2:
        assert plan.jit.workspace_bytes > 0 and plan.workspace_bytes >= plan.jit.workspace_bytes >= 2 * m * n * 4
    dy, x, dw1 = _wgrad_operands(m, k, n, torch.bfloat16)
    dw2 = torch.zeros_like(dw1)
    ws = _ws(plan)
    run_wgrad_gemm(plan, dy, x, dw1, ws)
    run_wgrad_gemm(plan, dy, x, dw2, ws)
    torch.cuda.synchronize()
    assert torch.equal(dw1, dw2), f"two executes differ: max|diff| = {(dw1.float() - dw2.float()).abs().max().item()}"
    _assert_close_vs_fp64(dw1, dy.double().T @ x.double(), f"B1 wgrad split_k={split_k}")


@requires_rubin
def test_split_k_1_pins_one_slice_and_refuses_the_fallback(monkeypatch):
    """``split_k=1`` is a PIN, not a no-op: the plan is the forced JIT at one slice; and when the
    forced compile is refused, the graph-heuristic fallback ``split_k=0`` takes silently is REFUSED
    (typed ``SplitKPinRefused`` carrying the compiler's decline) -- what the G3b recompute plans
    need for their bit-identical claim (spec R4 / R12).  The compiler is stubbed to decline so the
    fallback arm is exercised without a shape the tile cannot take."""
    import cudnn.gemm.frost.compiler as compiler

    m, k, n = _stage_mkn("B7_dw_qkvg", "test", 2048)
    plan = _plan("wgrad", m, k, n, torch.bfloat16, 1)
    assert plan.jit is not None and plan.jit.config.split_k_slices == 1 and plan.tile_config_name == _FORCED_TILE and plan.route == "graph+jit"
    assert plan.split_k == 1 and plan.jit.workspace_bytes == 0

    orig = compiler.jit_from_cudnn_graph

    def decline(graph, *args, **kwargs):
        # Decline ONLY the forced tile's compile: the backend's own frost_gemm plan (the heuristic's
        # cluster2x4 pick, built inside `_backend_pin`) goes through the same entry point and must
        # still build, or the graph fallback under test could never be reached.
        config = kwargs.get("config", args[0] if args else None)
        if config is not None and "cluster2x1_2ctamma" in config.name:
            raise NotImplementedError(f"probe: {config.name} declined")
        return orig(graph, *args, **kwargs)

    monkeypatch.setattr(compiler, "jit_from_cudnn_graph", decline)
    with pytest.raises(SplitKPinRefused, match=r"split_k=1 pins the JIT at .*cluster2x1_2ctamma.*fallback is refused") as ei:
        build_proj_gemm(m=m, k=k, n=n, dtype=torch.bfloat16, label="pinned", a_major="m", b_major="n", split_k=1)
    assert isinstance(ei.value.__cause__, NotImplementedError)
    with pytest.raises(SplitKPinRefused, match="split_k=2"):
        build_proj_gemm(m=m, k=k, n=n, dtype=torch.bfloat16, label="pinned2", a_major="m", b_major="n", split_k=2)
    # split_k=0 under the same decline: the documented fallback -- a GRAPH-route plan, the heuristic's config.
    fell = build_proj_gemm(m=m, k=k, n=n, dtype=torch.bfloat16, label="fallback", a_major="m", b_major="n", split_k=0)
    assert fell.jit is None and fell.route == "graph" and fell.tile_config_name == "heuristic (graph engine)"


@requires_rubin
def test_workspace_is_honest():
    """The block's engine scratch is ``max(plan.workspace_bytes)`` over every backward plan
    REGARDLESS of route (spec R8): a split-K plan's fp32 partials are in its number, a too-small or
    missing workspace is a typed refusal BEFORE the launch (an overflow would not fail it), and the
    max covers each plan."""
    m, k, n = _stage_mkn("B1_dw_o", "test", 2048)
    plan2 = _plan("wgrad", m, k, n, torch.bfloat16, 2)
    assert plan2.workspace_bytes >= plan2.jit.workspace_bytes >= 2 * m * n * 4
    dy, x, dw = _wgrad_operands(m, k, n, torch.bfloat16)
    with pytest.raises(ValueError, match="workspace"):
        run_wgrad_gemm(plan2, dy, x, dw, torch.empty(16, dtype=torch.uint8, device="cuda"))
    with pytest.raises(ValueError, match="workspace"):
        run_wgrad_gemm(plan2, dy, x, dw, None)
    plans = [_plan("wgrad" if st in _WGRAD else "dgrad", *_stage_mkn(st, "test", 2048), torch.bfloat16) for st in _WGRAD + _DGRAD] + [plan2]
    scratch = max(p.workspace_bytes for p in plans)
    assert scratch >= plan2.jit.workspace_bytes and all(scratch >= p.workspace_bytes for p in plans)
    ws = torch.empty(scratch, dtype=torch.uint8, device="cuda")
    run_wgrad_gemm(plan2, dy, x, dw, ws)  # the shared scratch serves the split-K plan
    torch.cuda.synchronize()
    _assert_close_vs_fp64(dw, dy.double().T @ x.double(), "B1 wgrad split_k=2 on the shared scratch")


@requires_rubin
@pytest.mark.parametrize("how", ["ambient", "explicit"])
def test_stream_threading(how):
    """The GEMM launches on the CALLER's stream -- ambient (``with torch.cuda.stream(s):``) or
    explicit (``stream=``) -- through ``run_proj_gemm``'s JIT ``stream=`` (Rule 5).  Deterministic
    probe (``test_a_caller_stream_orders_every_stage``): the default stream is parked behind a
    long spin and the INPUTS are zeroed on the side stream right after the GEMM, so a launch that
    landed on the default stream would run late and read zeros; correct threading gives a result
    BIT-IDENTICAL to the default-stream run."""
    m, k, n = _stage_mkn("B1_dw_o", "test", 2048)
    plan = _plan("wgrad", m, k, n, torch.bfloat16)
    dy, x, dw_ref = _wgrad_operands(m, k, n, torch.bfloat16)
    ws = _ws(plan)
    run_wgrad_gemm(plan, dy, x, dw_ref, ws)
    torch.cuda.synchronize()
    assert dw_ref.abs().max().item() > 0
    dw = torch.zeros_like(dw_ref)
    side = torch.cuda.Stream()
    torch.cuda.synchronize()
    _park_the_default_stream()
    if how == "ambient":
        with torch.cuda.stream(side):
            run_wgrad_gemm(plan, dy, x, dw, ws)
    else:
        run_wgrad_gemm(plan, dy, x, dw, ws, stream=side.cuda_stream)
    with torch.cuda.stream(side):
        dy.zero_()  # ordered AFTER the GEMM on the side stream; a GEMM parked on the default stream reads these zeros instead
        x.zero_()
    torch.cuda.synchronize()
    assert torch.equal(dw, dw_ref), f"the GEMM ran off the caller's stream ({how}): zeros = {(dw == 0).float().mean().item():.0%}"


# ---------------------------------------------------------------------------
# Reject / host tests -- run anywhere
# ---------------------------------------------------------------------------


def test_build_refuses_an_unknown_major_or_split_k_before_any_graph():
    """The kwargs are validated FIRST: a typo in a major or a negative / bool split_k is a
    ``ValueError`` naming the kwarg, before ``cudnn`` builds anything."""
    with pytest.raises(ValueError, match="a_major"):
        build_proj_gemm(m=256, k=256, n=256, dtype=torch.bfloat16, label="bad", a_major="n")
    with pytest.raises(ValueError, match="b_major"):
        build_proj_gemm(m=256, k=256, n=256, dtype=torch.bfloat16, label="bad", b_major="m")
    with pytest.raises(ValueError, match="split_k"):
        build_proj_gemm(m=256, k=256, n=256, dtype=torch.bfloat16, label="bad", split_k=-1)
    with pytest.raises(ValueError, match="split_k"):
        build_proj_gemm(m=256, k=256, n=256, dtype=torch.bfloat16, label="bad", split_k=True)


def test_tma_rule_is_typed():
    """B1 at ``dm = 4100``: A is M-major, so TMA reads M contiguously and needs ``M % 8 == 0``
    at bf16 (16-byte contiguous-extent rule).  A typed ``ValueError`` naming the operand
    and the rule, raised by the DRIVER -- the engine's own decline would surface only as
    'no frost_gemm plan' after the graph was built."""
    with pytest.raises(ValueError, match=r"A is m-major.*M % 8 == 0.*16-byte"):
        build_proj_gemm(m=4100, k=8192, n=8192, dtype=torch.bfloat16, label="dw_o", a_major="m", b_major="n")
    with pytest.raises(ValueError, match=r"B is n-major.*N % 8 == 0.*16-byte"):
        build_proj_gemm(m=4096, k=8192, n=8196, dtype=torch.bfloat16, label="dw_o", a_major="m", b_major="n")


def test_wrong_major_view_is_a_typed_refusal():
    """A K-major plan handed an M-major view (``dy.view(T, dm).t()``) -> ``ValueError`` naming
    the operand and BOTH strides, raised by the DRIVER before any launch.  Why the driver
    checks: the JIT route re-reads runtime strides and refuses a mismatch, but the graph
    fallback binds pointers against the DECLARED strides with no check -- a wrong view there
    is a silent reinterpretation (spec R12).  Hand-built plans, so this runs on any device
    (nothing is launched; the check precedes every route)."""
    t, dm, hd = 256, 512, 1024
    dy = torch.zeros(t, dm, device="cuda", dtype=torch.bfloat16)
    x = torch.zeros(t, hd, device="cuda", dtype=torch.bfloat16)
    dw = torch.zeros(dm, hd, device="cuda", dtype=torch.bfloat16)
    ws = torch.empty(1, dtype=torch.uint8, device="cuda")
    # (1) the plan's majors are the first gate: a wgrad through a K-major plan is refused by name.
    kmajor = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=dm, k=t, n=hd, label="kmajor", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"a_major='k'.*b_major='k'"):
        run_wgrad_gemm(kmajor, dy, x, dw, ws)
    # (2) the right plan, the wrong VIEW: dy_like arrives already transposed, so the A view's
    #     stride-1 axis is K, not the declared M -- refused naming A and both strides.
    mn = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=dm, k=t, n=hd, label="mn", dtype=torch.bfloat16, a_major="m", b_major="n")
    with pytest.raises(ValueError, match=r"A.*stride-1 axis.*declared") as ei:
        run_wgrad_gemm(mn, dy.t().contiguous().t(), x, dw, ws)  # a [T, rows] view whose storage is [rows, T]
    assert "stride" in str(ei.value)
    # (3) B through a plan that declared it N-major, handed a K-major storage: same refusal for B.
    with pytest.raises(ValueError, match=r"B.*stride-1 axis.*declared"):
        run_wgrad_gemm(mn, dy, x.t().contiguous().t(), dw, ws)
    # (4) a dgrad through a plan whose B is K-major (the FORWARD's declaration): refused by name.
    fwd = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=t, k=dm, n=hd, label="fwd", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"b_major='k'"):
        run_dgrad_gemm(fwd, dy, torch.zeros(dm, hd, device="cuda", dtype=torch.bfloat16), torch.zeros(t, hd, device="cuda", dtype=torch.bfloat16), ws)
