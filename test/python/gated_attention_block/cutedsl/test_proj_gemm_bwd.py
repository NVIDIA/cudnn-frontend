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

import os
import sys

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

from cudnn.gated_attention_block.kernels.proj_gemm import (  # noqa: E402
    ProjGemmPlan,
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
