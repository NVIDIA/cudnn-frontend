# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The four backward projection GEMMs of the gated attention block, on the shipped FROST GEMM.

``nn.Linear`` orientation (``api_bwd.py``): ``dW = dY^T @ X`` (A M-major, B N-major) and
``dX = dY @ W`` (B N-major).  Nothing here writes a kernel; what is under test is the
DRIVER's claims (``kernels/proj_gemm.py``):

* the appended ``a_major`` / ``b_major`` kwargs declare the graph strides the FROST GEMM
  engine renders as its M-major-A / N-major-B template arms, and the tile the block
  FORCES for every ``n % 256 == 0`` (``_forced_tile_config``) renders and computes
  correctly on cc 10.7 under those majors -- the riskiest assumption behind the block's
  backward, settled by :func:`test_forced_tile_renders_mn_major_on_cc107`;
* ``run_wgrad_gemm`` / ``run_dgrad_gemm`` bind transposed VIEWS (zero-copy) against a plan
  DECLARED with the matching majors, and refuse a view whose strides are not EXACTLY the
  declared ones (a wrong major, or a column slice of a wider slab) BEFORE any launch, on
  both routes -- the graph fallback would read the declared strides with no check (a
  silent reinterpretation) and the JIT re-labels B into kernel order only on
  an exact match;
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

from gated_block_stream_probe import park_the_default_stream  # noqa: E402

_FORCED_TILE = "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma"

# The REGISTERED marker of cutedsl/conftest.py (the skip is applied at collection) -- switched from the per-module
# skipif copy when this module was next touched.
requires_rubin = pytest.mark.requires_rubin


# ---------------------------------------------------------------------------
# Tolerance: bf16 / f16 inputs, fp32 accumulate, bf16 / f16 output, vs an fp64 oracle
# ---------------------------------------------------------------------------
#
# The oracle multiplies the SAME quantized inputs in fp64, so the only differences are
# (i) the output rounding -- half an ulp = 2^-9 relative for bf16 (8 significand bits),
# 2^-12 for f16 -- and (ii) the fp32 accumulation order over K terms, whose error is
# ABSOLUTE and about sqrt(K) * 2^-24 * |typical partial sum| (K = 8192: ~1e-5 of the
# tensor's scale, three orders below either bound).  So rtol = 4x the half-ulp of the
# OUTPUT dtype -- 2^-7 for bf16, 2^-10 for f16, derived the same way;
# one bound per dtype, never one for both (a shared 2^-7 would let an f16 cell 16x its
# half-ulp through, so the f16 arms would assert less than they appear to) -- and
# atol = rtol * max|ref| (tensor-scaled, test/AGENTS.md: a fixed absolute bound turns
# wrong when the magnitudes grow).  Measured worst cells on Rubin (cc 10.7, 204 SMs), 2026-09-29: bf16
# 0.16-0.23 of its bound, f16 0.02-0.03 of the 2^-7 bound = 0.16-0.24 of 2^-10 -- the
# same 4-6x margin at both dtypes.  Never widened; a violation is reported with its magnitude.
_RTOL_BY_DTYPE = {torch.bfloat16: 2.0**-7, torch.float16: 2.0**-10}


def _assert_close_vs_fp64(out: torch.Tensor, ref64: torch.Tensor, what: str) -> None:
    rtol = _RTOL_BY_DTYPE[out.dtype]
    ref_max = ref64.abs().max().item()
    atol = rtol * ref_max
    diff = (out.double() - ref64).abs()
    scale = atol + rtol * ref64.abs()
    worst = (diff / scale).max().item()
    msg = f"{what}: max|diff| = {diff.max().item():.4g} (max|ref| = {ref_max:.4g}); worst cell at {worst:.3f} of the bound (rtol = atol/max|ref| = {rtol}, {out.dtype})"
    print(msg)
    torch.testing.assert_close(out.double(), ref64, rtol=rtol, atol=atol, msg=msg)


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
    """``(m, k, n)`` of one backward projection plan at the block's geometry."""
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


# ---------------------------------------------------------------------------
# THE probe: the forced tile renders and computes M-major A / N-major B on cc 10.7
# ---------------------------------------------------------------------------


@requires_rubin
def test_forced_tile_renders_mn_major_on_cc107():
    """The riskiest assumption behind the block's backward: the bf16 M-major-A /
    N-major-B renderings of the tile the block FORCES (``..._cluster2x1_2ctamma``)
    have never run on cc 10.7 -- the FROST GEMM suite covers the layouts on the sm100
    pipeline, and the block's own forward only ever rendered K-major operands.

    B1's geometry at the 397B column shapes: ``dW_o [dm=4096, HD=8192] = dY[T=8192,
    dm]^T @ O_gated[T, HD]``.  Three assertions, in the order a failure would show:
    the plan IS the forced JIT (a fallback to the heuristic is a FAILURE here, not a
    skip -- it changes route, config and possibly split-K), the output is not silently
    zero (the >256 KiB tcgen05 descriptor landmine's signature, ``test_proj_gemm.py::
    test_output_is_not_silently_zero``), and the numbers match the fp64 oracle."""
    m, k, n = 4096, 8192, 8192
    plan = build_proj_gemm(m=m, k=k, n=n, dtype=torch.bfloat16, label="mn_major_probe_dw_o", a_major="m", b_major="n")
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
    (a fallback to the heuristic is a FAILURE here, not a skip: it changes route, config and
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
@pytest.mark.parametrize("split_k,t", [(0, 2048), (2, 2048), (2, 4104)], ids=["auto", "split_k=2", "split_k=2-T4104"])
def test_two_runs_bitwise(split_k, t):
    """Determinism by construction: two executes of the same plan agree bit for bit -- the
    split-K reducer is a fixed-order tree, no atomics anywhere.  The ``split_k=2`` plan is a JIT
    CONFIG replace (``plan.jit.config.split_k_slices == 2``), not a graph-route knob replay, and its
    fp32 partials show in ``plan.workspace_bytes``; its result meets the same fp64 bound.  The
    block's wgrads have K = T, so ``T = 4104`` splits an UNEVEN K (65 CTA-K tiles of 64 over two
    slices, the last tile ragged) -- the slice bookkeeping the even case never exercises."""
    m, k, n = _stage_mkn("B1_dw_o", "test", t)
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
    _assert_close_vs_fp64(dw1, dy.double().T @ x.double(), f"B1 wgrad split_k={split_k}, T={t}")


@requires_rubin
def test_split_k_1_pins_one_slice_and_refuses_the_fallback(monkeypatch):
    """``split_k=1`` is a PIN, not a no-op: the plan is the forced JIT at one slice; and when the
    forced compile is refused, the graph-heuristic fallback ``split_k=0`` takes silently is REFUSED
    (typed ``SplitKPinRefused`` carrying the compiler's decline) -- what the follow-up recompute
    plans need for their bit-identical claim.  The compiler is stubbed to decline so the
    fallback arm is exercised without a shape the tile cannot take."""
    import cudnn.gemm.frost.compiler as compiler

    m, k, n = _stage_mkn("B7_dw_qkvg", "test", 2048)
    plan = _plan("wgrad", m, k, n, torch.bfloat16, 1)
    assert plan.jit is not None and plan.jit.config.split_k_slices == 1 and plan.tile_config_name == _FORCED_TILE and plan.route == "graph+jit"
    assert plan.split_k == 1 and plan.jit.workspace_bytes == 0
    # The recompute plans' premise, pinned: the pinned one-slice plan and the driver's pick run the SAME config,
    # so their outputs are bitwise equal -- a tripwire against a future catalog `split_k_slices`
    # change (or an auto-split) on the forced tile.
    dy, x, dw_pinned = _wgrad_operands(m, k, n, torch.bfloat16)
    dw_auto = torch.zeros_like(dw_pinned)
    plan0 = _plan("wgrad", m, k, n, torch.bfloat16, 0)
    run_wgrad_gemm(plan, dy, x, dw_pinned, _ws(plan))
    run_wgrad_gemm(plan0, dy, x, dw_auto, _ws(plan0))
    torch.cuda.synchronize()
    assert dw_pinned.abs().max().item() > 0 and torch.equal(
        dw_pinned, dw_auto
    ), "split_k=1 (pinned) and split_k=0 (the driver's pick) differ on the forced tile"

    orig = compiler.jit_from_cudnn_graph

    def decline(graph, *args, **kwargs):
        # Decline ONLY the compile `build_proj_gemm` issues ITSELF (its caller frame): the backend's own
        # frost_gemm plan -- the heuristic's pick, built inside `_backend_pin` -> `g.build_plans()` -> the
        # engine -> `graph_analyzer` -- goes through the same entry point with the same kwargs and must
        # still build, or the graph fallback under test could never be reached.  Keyed on the caller, not
        # on the config NAME: a heuristic that one day picks the forced tile's family would otherwise
        # turn this test into an unrelated `_backend_pin` error.
        if sys._getframe(1).f_code.co_name == "build_proj_gemm":
            config = kwargs.get("config", args[0] if args else None)
            raise NotImplementedError(f"probe: {getattr(config, 'name', config)} declined")
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
def test_split_k_workspace_must_be_a_cuda_buffer_on_the_launch_device():
    """A split-K plan carves its fp32 partials out of the CALLER's workspace.  A host (CPU) buffer of the right
    size and alignment passes the presence / size / alignment checks, so without a device pin it reached the
    launch boundary as a bogus device pointer.  ``run_proj_gemm`` binds the workspace with the launch device
    (``out.device``): a CPU buffer -- and a buffer on another CUDA device, where one is visible -- is a typed
    ``ValueError`` BEFORE any launch (the output is untouched); the CUDA twin still runs (positive control).
    Shared by both backward drivers and the forward."""
    m, k, n = _stage_mkn("B7_dw_qkvg", "test", 2048)
    plan = _plan("wgrad", m, k, n, torch.bfloat16, 2)
    assert plan.jit is not None and plan.jit.workspace_bytes > 0, "split_k=2 must carry fp32 partials"
    dy, x, dw = _wgrad_operands(m, k, n, torch.bfloat16)
    before = dw.clone()
    ws_cpu = torch.empty(plan.workspace_bytes, dtype=torch.uint8)  # host memory: right size, aligned, wrong place
    with pytest.raises(ValueError, match="workspace must be a CUDA device buffer"):
        run_wgrad_gemm(plan, dy, x, dw, ws_cpu)
    torch.cuda.synchronize()
    assert torch.equal(dw, before), "the refusal must fire before any launch"
    if torch.cuda.device_count() > 1:
        other = (dw.device.index + 1) % torch.cuda.device_count()  # an ordinal that is NOT the launch device, whichever that is
        ws_other = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device=f"cuda:{other}")
        with pytest.raises(ValueError, match=rf"workspace must be on {dw.device}, got {ws_other.device}"):
            run_wgrad_gemm(plan, dy, x, dw, ws_other)
        torch.cuda.synchronize()
        assert torch.equal(dw, before)
    run_wgrad_gemm(plan, dy, x, dw, _ws(plan))  # positive control: the CUDA workspace on the launch device
    torch.cuda.synchronize()
    assert dw.abs().max().item() > 0


@requires_rubin
def test_workspace_is_honest():
    """The block's engine scratch is ``max(plan.workspace_bytes)`` over every backward plan
    REGARDLESS of route: a split-K plan's fp32 partials are in its number, a too-small or
    missing workspace is a typed refusal BEFORE the launch (an overflow would not fail it), and the
    max covers each plan.

    The plan under test must make ``workspace_bytes = max(graph, jit)`` LOAD-BEARING: at the test
    geometry the backend's own frost_gemm plan auto-splits 3-way (``SPLIT_K_SLC=3`` in its name,
    ``graph.get_workspace_size()`` = 12 MiB, measured on Rubin cc 10.7, 2026-09-29), which already exceeds
    a JIT ``split_k=2``'s 8 MiB -- a ``workspace_bytes`` that forgot the JIT term would still pass
    there.  ``split_k=8`` needs 32 MiB, and the precondition below asserts the ordering, so the
    test tells you to move the shape if the heuristic ever splits deeper instead of going quiet."""
    m, k, n = _stage_mkn("B1_dw_o", "test", 2048)
    plan8 = _plan("wgrad", m, k, n, torch.bfloat16, 8)
    assert plan8.jit is not None and plan8.jit.config.split_k_slices == 8 and plan8.tile_config_name == _FORCED_TILE + "_splitK8"
    graph_ws, jit_ws = int(plan8.graph.get_workspace_size()), int(plan8.jit.workspace_bytes)
    assert jit_ws >= 8 * m * n * 4
    assert (
        graph_ws < jit_ws
    ), f"precondition: the graph's number ({graph_ws}) must be BELOW the JIT's ({jit_ws}) for max() to be load-bearing -- raise split_k or move the shape"
    assert plan8.workspace_bytes >= jit_ws, f"plan.workspace_bytes = {plan8.workspace_bytes} forgot the JIT's split-K partials ({jit_ws}; graph {graph_ws})"
    dy, x, dw = _wgrad_operands(m, k, n, torch.bfloat16)
    with pytest.raises(ValueError, match="workspace"):
        run_wgrad_gemm(plan8, dy, x, dw, torch.empty(16, dtype=torch.uint8, device="cuda"))
    with pytest.raises(ValueError, match="workspace"):
        run_wgrad_gemm(plan8, dy, x, dw, torch.empty(graph_ws, dtype=torch.uint8, device="cuda"))  # the graph-only size is NOT enough
    with pytest.raises(ValueError, match="workspace"):
        run_wgrad_gemm(plan8, dy, x, dw, None)
    plans = [_plan("wgrad" if st in _WGRAD else "dgrad", *_stage_mkn(st, "test", 2048), torch.bfloat16) for st in _WGRAD + _DGRAD] + [plan8]
    scratch = max(p.workspace_bytes for p in plans)
    assert scratch >= jit_ws and all(scratch >= p.workspace_bytes for p in plans)
    ws = torch.empty(scratch, dtype=torch.uint8, device="cuda")
    run_wgrad_gemm(plan8, dy, x, dw, ws)  # the shared scratch serves the split-K plan
    dw2 = torch.zeros_like(dw)
    run_wgrad_gemm(plan8, dy, x, dw2, ws)
    torch.cuda.synchronize()
    assert torch.equal(dw, dw2), "two executes of the split_k=8 plan differ"
    _assert_close_vs_fp64(dw, dy.double().T @ x.double(), "B1 wgrad split_k=8 on the shared scratch")


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
    park_the_default_stream()
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


def test_split_k_needs_pin_frost():
    """A pinned split is a JIT plan by definition; ``pin_frost=False`` deselects the FROST engine, so
    there is nothing to pin -- a typed ``ValueError`` naming both kwargs, not a silently dropped knob
    (before this, the plan came back ``route='graph'``, ``jit=None`` with ``plan.split_k`` still
    stamped with the request)."""
    for split_k in (1, 2):
        with pytest.raises(ValueError, match=rf"split_k={split_k}.*pin_frost=False"):
            build_proj_gemm(m=512, k=2048, n=2048, dtype=torch.bfloat16, label="x", a_major="m", b_major="n", split_k=split_k, pin_frost=False)


_FP8 = getattr(torch, "float8_e4m3fn", None)


@pytest.mark.skipif(_FP8 is None, reason="this torch has no float8_e4m3fn")
def test_fp8_mn_major_is_a_typed_decline():
    """The drivers serve bf16 / f16.  An fp8 (e4m3) operand with an M-major A or an N-major B is
    a typed ``NotImplementedError`` BEFORE any graph exists -- that rendering is unmeasured on
    cc 10.7 and the quantized backward lands its own fp8 GEMM drivers -- rather than an admitted
    but never-run path.  The forward's K-major fp8 plans are untouched (``test_proj_gemm.py``)."""
    with pytest.raises(NotImplementedError, match=r"fp8 \(e4m3\).*a_major='m'"):
        build_proj_gemm(m=512, k=2048, n=2048, dtype=_FP8, label="fp8_dw", a_major="m", b_major="n")
    with pytest.raises(NotImplementedError, match=r"fp8 \(e4m3\).*b_major='n'"):
        build_proj_gemm(m=512, k=2048, n=2048, dtype=_FP8, label="fp8_dx", a_major="k", b_major="n")
    with pytest.raises(NotImplementedError, match=r"fp8 \(e4m3\).*w_dtype"):
        build_proj_gemm(m=512, k=2048, n=2048, dtype=torch.bfloat16, w_dtype=_FP8, label="fp8_w", a_major="k", b_major="n")


def test_workspace_bytes_is_the_max_of_graph_and_jit():
    """``ProjGemmPlan.workspace_bytes`` on fakes, so the ``max(graph, jit)`` rule is pinned
    independently of what the backend heuristic happens to report: the JIT's split-K
    partials count whenever a JIT exists (``run_proj_gemm`` launches it), the graph's number when
    it is the larger, the JIT's alone on the JIT-only route, and never 0."""

    class _Graph:
        def __init__(self, ws):
            self.ws = ws

        def get_workspace_size(self):
            return self.ws

    class _Jit:
        def __init__(self, ws):
            self.workspace_bytes = ws

    plan = ProjGemmPlan(graph=_Graph(12 << 20), a=None, b=None, c=None, m=512, k=2048, n=2048, label="ws", dtype=torch.bfloat16)
    plan.route = "graph+jit"
    plan.jit = _Jit(32 << 20)
    assert plan.workspace_bytes == 32 << 20  # the JIT's split-K partials win
    plan.jit = _Jit(8 << 20)
    assert plan.workspace_bytes == 12 << 20  # the graph's number wins (the test-geometry case above)
    plan.jit = _Jit(0)
    assert plan.workspace_bytes == 12 << 20
    plan.jit = None
    plan.route = "graph"
    assert plan.workspace_bytes == 12 << 20
    plan.graph = _Graph(0)
    assert plan.workspace_bytes == 1  # never 0: an empty buffer would fail Workspace's presence check
    plan.route = "jit-only"
    plan.jit = _Jit(8 << 20)
    assert plan.workspace_bytes == 8 << 20  # no backend plan to ask
    plan.jit = None
    assert plan.workspace_bytes == 1


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
    is a silent reinterpretation.  Hand-built plans, so this runs on any device
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
    # (5) the right major, a PADDED view: x as a column slice of a wider slab has the declared stride-1
    #     axis but a row stride != n -- refused naming both strides, on the graph route (jit=None) ...
    x_wide = torch.zeros(t, hd + 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"B.*strides \(\d+, 1088, 1\) but the plan declared \(\d+, 1024, 1\).*exactly the declared layout"):
        run_wgrad_gemm(mn, dy, x_wide[:, :hd], dw, ws)
    # (6) ... and on the JIT route alike (the JIT re-labels B into kernel order only on an exact match,
    #     so its own refusal would be the generic 'input layout' one; the driver's typed message comes first).
    mn_jit = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=dm, k=t, n=hd, label="mn_jit", dtype=torch.bfloat16, a_major="m", b_major="n")
    mn_jit.jit = object()
    with pytest.raises(ValueError, match=r"B.*strides \(\d+, 1088, 1\) but the plan declared"):
        run_wgrad_gemm(mn_jit, dy, x_wide[:, :hd], dw, ws)
    # (7) a padded A: dy_like as a column slice of a wider slab -> the transposed view's K stride != m.
    dy_wide = torch.zeros(t, dm + 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"A.*strides \(\d+, 1, 576\) but the plan declared \(\d+, 1, 512\)"):
        run_wgrad_gemm(mn_jit, dy_wide[:, :dm], x, dw, ws)
    # (8) a dgrad's un-transposed weight as a column slice: same refusal for B.
    dg = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=t, k=dm, n=hd, label="dg", dtype=torch.bfloat16, a_major="k", b_major="n")
    w_wide = torch.zeros(dm, hd + 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"B \(w\).*strides \(\d+, 1088, 1\) but the plan declared"):
        run_dgrad_gemm(dg, dy, w_wide[:, :hd], torch.zeros(t, hd, device="cuda", dtype=torch.bfloat16), ws)


def test_wrong_output_dtype_is_a_typed_refusal():
    """The OUTPUT's dtype is checked like A's and W's: a ``dw`` / ``dx`` / ``out`` whose dtype is
    not the plan's ``out_dtype`` is a ``ValueError`` naming the plan and both dtypes, raised by
    ``run_proj_gemm`` -- so the forward and both backward drivers share the gate -- before any
    route.  Why: the graph carries C's dtype and the JIT binds a pointer, so a same-size f16
    buffer on a bf16 plan would be filled with bf16 bit patterns (wrong values, no error), and a
    NARROWER buffer (an e4m3 ``[T, N]`` on a bf16 plan, half the bytes) would have 2-byte stores
    run past its allocation.  A spy stands in for the JIT to prove nothing launched.  Hand-built
    plans, so this runs on any device; a plan with no ``out_dtype`` (the stream-routing probes)
    checks nothing, exactly as for A / W."""
    from cudnn.gated_attention_block.kernels.proj_gemm import run_proj_gemm

    class _SpyJit:
        def __init__(self):
            self.calls = []

        def __call__(self, vp, **kw):
            self.calls.append(vp)

    t, dm, hd = 256, 512, 1024
    narrow = _FP8 if _FP8 is not None else torch.uint8  # a 1-byte output buffer on a 2-byte plan
    dy = torch.zeros(t, dm, device="cuda", dtype=torch.bfloat16)
    x = torch.zeros(t, hd, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(dm, hd, device="cuda", dtype=torch.bfloat16)
    ws = torch.empty(1, dtype=torch.uint8, device="cuda")
    # (1) wgrad: a same-size f16 dw on a bf16 plan -- it passes the shape / stride gates and is refused by dtype.
    mn = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=dm, k=t, n=hd, label="mn", dtype=torch.bfloat16, out_dtype=torch.bfloat16, a_major="m", b_major="n")
    mn.jit = _SpyJit()
    with pytest.raises(ValueError, match=r"^mn: out is torch\.float16 but this plan was built for torch\.bfloat16; refusing to reinterpret the bytes$"):
        run_wgrad_gemm(mn, dy, x, torch.zeros(dm, hd, device="cuda", dtype=torch.float16), ws)
    # (2) dgrad: a NARROWER dx (1 B/elem) on the bf16 plan -- the case whose stores would overrun the buffer.
    dg = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=t, k=dm, n=hd, label="dg", dtype=torch.bfloat16, out_dtype=torch.bfloat16, a_major="k", b_major="n")
    dg.jit = _SpyJit()
    with pytest.raises(ValueError) as ei:
        run_dgrad_gemm(dg, dy, w, torch.zeros(t, hd, device="cuda", dtype=narrow), ws)
    assert str(ei.value) == f"dg: out is {narrow} but this plan was built for torch.bfloat16; refusing to reinterpret the bytes"
    # (3) the forward's entry point shares the gate: a K-major plan through run_proj_gemm directly.
    fwd = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=t, k=hd, n=dm, label="fwd", dtype=torch.bfloat16, out_dtype=torch.bfloat16)
    fwd.jit = _SpyJit()
    with pytest.raises(ValueError, match=r"^fwd: out is torch\.float16 but this plan was built for torch\.bfloat16"):
        run_proj_gemm(fwd, x, w, torch.zeros(t, dm, device="cuda", dtype=torch.float16), ws)
    assert not (mn.jit.calls or dg.jit.calls or fwd.jit.calls), "a mismatched output reached the launch"
    # (4) the declared dtype passes the gate and reaches the (spy) launch, once per call.
    run_wgrad_gemm(mn, dy, x, torch.zeros(dm, hd, device="cuda", dtype=torch.bfloat16), ws)
    run_dgrad_gemm(dg, dy, w, torch.zeros(t, hd, device="cuda", dtype=torch.bfloat16), ws)
    run_proj_gemm(fwd, x, w, torch.zeros(t, dm, device="cuda", dtype=torch.bfloat16), ws)
    assert [len(p.jit.calls) for p in (mn, dg, fwd)] == [1, 1, 1]
    # (5) a hand-built plan with no out_dtype (the stream-routing probes) checks nothing -- as for A / W.
    probe = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=t, k=hd, n=dm, label="probe")
    probe.jit = _SpyJit()
    run_proj_gemm(probe, x, w, torch.zeros(t, dm, device="cuda", dtype=torch.float16), ws)
    assert len(probe.jit.calls) == 1
