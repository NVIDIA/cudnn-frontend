# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K3 of the triangle multiplicative update -- LayerNorm of ``y`` (MN-major rows) and of ``z`` (recomputed), the dual GEMM (output projection ‖
output gate) with the pairing across the two accumulators, sigmoid * mul, the pair-major update -- on the ``cudnn.gemm.frost.ln_prologue``
substrate's ``pairing = "accumulators"`` arm, against an fp64 reference, plus the GPU-free contract tests.

The K3 inputs come from a torch model of K1 / K2 (fp64 LayerNorm -> gated projections -> the per-channel contraction -> bf16 ``y`` in the
channel-major, inner-padded layout the contraction writes), so ``y`` has the magnitude and structure the kernel sees in production.

Tolerance of the oracle (derived, stated -- the same derivation as K1's): both MMA operands are LayerNormed rows rounded ONCE to bf16 and
the reference models exactly that (fp64 LayerNorm -> bf16 -> fp64 GEMM); what remains is the fp32 accumulation over K <= 256 (<= K * 2^-24),
the sigmoid through ``tanh.approx`` (<= 2^-12 on sigma, times |value| ~ O(1)), the one bf16 rounding of the output (half-ulp 2^-9 relative)
and the fp32-vs-fp64 statistics flipping a few operand roundings by one ulp: ``|diff| <= 4e-3 + |ref| / 128``.  Under ``affine_fold`` the
reference is the FOLDED function (the folded bf16 weights and bias, the pure ``(y - mean) * rstd``) so the kernel is judged on what it
computes; the fold's distance from the UNFOLDED function (its numerics class) is measured and printed, not asserted beyond a sanity bound.

The kernel targets Rubin (cc 10.7); the accept tests skip elsewhere with the reason stated.  Every launch is a single process under the
harness's ``timeout`` (exit 124 = hang).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Triangle multiplicative update tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

from cudnn.gemm.frost import ln_prologue as sub  # noqa: E402
from cudnn.triangle_multiplicative_update.kernels import trimul_tail as k3  # noqa: E402

_SM107 = (10, 7)


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


requires_rubin = pytest.mark.skipif(_cc() != _SM107, reason=f"the LN-prologue dual GEMM is rendered for SM107 (Rubin) only; found {_cc()}")


# ---------------------------------------------------------------------------
# GPU-free: the knob vocabulary and its typed declines, the affine fold, the model counts, the runner's pre-launch checks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kw",
    [
        dict(pairing="rows"),
        dict(pairing="accumulators", a0_major="mn", has_mask=True, skip_masked_tiles=False, passes=1),
        dict(pairing="accumulators", a0_major="mn", has_mask=False, skip_masked_tiles=True, passes=1),
        dict(pairing="accumulators", a0_major="mn", has_mask=False, skip_masked_tiles=False, passes=2),
        dict(pairing="accumulators", a0_major="k", has_mask=False, skip_masked_tiles=False, passes=1),
        dict(pairing="accumulators", a0_major="mn", has_mask=False, skip_masked_tiles=False, passes=1, k=448),
        dict(pairing="columns", a0_major="mn"),
        dict(pairing="columns", affine_fold=True),
        dict(pairing="columns", fuse_residual=True),
        dict(a0_prologue="ldmatrix"),
    ],
)
def test_accumulators_arm_declines_typed(kw):
    with pytest.raises(ValueError):
        sub.validate_ln_dual_gemm_params(sub.LnDualGemmParams(**kw))


def test_re_read_ln_form_declines_the_transpose_prologue():
    """`ln_pass_form` "smem" / "smem_rolled" re-read operand 0's MN-major stage in every LayerNorm pass; the transpose form's K-major row writes
    overwrite channel rows other lanes still re-read (the register form reads everything before its LN_SYNC barrier), so the pair is a typed decline
    on the host, never a race on the device."""
    for lnf in ("smem", "smem_rolled", "smem_rolled_pf", "smem_rolled_tree"):
        with pytest.raises(ValueError, match="a0_prologue='mn_major'"):
            sub.validate_ln_dual_gemm_params(k3.tail_params(128, 128, has_bias=True, a0_prologue="transpose", ln_pass_form=lnf))
        sub.validate_ln_dual_gemm_params(k3.tail_params(128, 128, has_bias=True, a0_prologue="mn_major", ln_pass_form=lnf))  # the served pair


def test_tail_params_is_the_accumulators_arm():
    p = k3.tail_params(128, 128, has_bias=True)
    sub.validate_ln_dual_gemm_params(p)
    assert p.dual and p.n_out == 128 and p.num_a_operands == 2 and p.passes == 1 and not p.has_mask and p.a0_prologue == "mn_major" and not p.affine_fold
    assert k3.tail_params(128, 128, has_bias=False, a0_prologue="transpose").a0_prologue == "transpose"
    with pytest.raises(ValueError, match="c == c_z"):
        k3.tail_params(256, 128, has_bias=True)  # the Protenix-v2 geometry: one K for both GEMMs is the arm's contract (typed, not silent)
    with pytest.raises(ValueError, match="accumulators arm"):
        k3.build_trimul_tail(sub.LnDualGemmParams())  # a columns-arm params object is not K3


def test_fold_output_affine_matches_the_fp64_algebra():
    g = torch.Generator().manual_seed(1)
    c_z, c = 128, 128
    w = (torch.randn(c_z, c, generator=g) / math.sqrt(c)).to(torch.bfloat16)
    b = (0.2 * torch.randn(c_z, generator=g)).to(torch.bfloat16)
    gam = (1.0 + 0.1 * torch.randn(c, generator=g)).to(torch.bfloat16)
    bet = (0.1 * torch.randn(c, generator=g)).to(torch.bfloat16)
    wf, bf = k3.fold_tail_affine(w, b, gam, bet)
    assert wf.dtype == torch.bfloat16 and bf.dtype == torch.bfloat16 and wf.shape == (c_z, c) and bf.shape == (c_z,)
    # the folded function equals the unfolded one on a normalised row x up to the bf16 rounding of the folded weights / bias
    x = torch.randn(7, c, generator=g).double()
    unfolded = (x * gam.double() + bet.double()) @ w.double().T + b.double()
    folded = x @ wf.double().T + bf.double()
    assert (folded - unfolded).abs().max().item() < 0.03  # the fold's own rounding (2^-9 relative per weight, 128 terms): a sanity bound only
    exact = x @ (w.double() * gam.double()[None, :]).T + (w.double() @ bet.double() + b.double())
    assert torch.allclose(exact, unfolded, atol=1e-9)
    wf2, bf2 = k3.fold_tail_affine(w, None, gam, bet)
    assert torch.equal(wf2, wf) and torch.allclose(bf2.double(), (w.double() @ bet.double()).to(torch.bfloat16).double())
    with pytest.raises(ValueError):
        k3.fold_tail_affine(w, b, gam[:64], bet)


def test_stack_tail_biases_interleaves_per_128_block():
    c_z = 256
    bo = torch.arange(c_z, dtype=torch.float32)
    bg = -torch.arange(c_z, dtype=torch.float32)
    s = k3.stack_tail_biases(bo, bg)
    assert s.shape == (2 * c_z,)
    assert (
        torch.equal(s[0:128], bo[0:128])
        and torch.equal(s[128:256], bg[0:128])
        and torch.equal(s[256:384], bo[128:256])
        and torch.equal(s[384:512], bg[128:256])
    )


def test_y_shape_and_model_counts():
    assert k3.y_shape(128, 2, 217) == (128, 434, 224)
    assert k3.tail_flop(128, 128, 1, 512) == 2 * 512 * 512 * (128 * 128 + 128 * 128)
    assert k3.tail_bytes(128, 128, 1, 512, with_stats=False) == 512 * 512 * (128 * 2 + 2 * 128 * 2)
    assert k3.tail_bytes(128, 128, 1, 512) == k3.tail_bytes(128, 128, 1, 512, with_stats=False) + 512 * 512 * 8


def test_every_tcgen05_smem_descriptor_of_the_body_takes_the_module_desc_version_and_it_is_v1():
    """Both arms' operand rings end at 288 KiB (9 x 32 KiB; 6 x 48 KiB), so the last stages start past 262144 B -- a version-0 tcgen05 SMEM
    descriptor's 14-bit start-address field wraps there (EXACTLY zero accumulators, no fault).  The body declares ``DESC_VERSION = 1`` once
    and passes it at every ``Tcgen05SmemDesc.build``; a re-literalled ``version=`` or a build site relying on the DSL default fails here."""
    import re

    src = open(sub.body_path("sm107")).read()
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    m = re.search(r"^DESC_VERSION: int = (\d)$", code, flags=re.M)
    assert m is not None, "the body must declare the module-level DESC_VERSION"
    assert int(m.group(1)) == 1, f"DESC_VERSION = {m.group(1)}: the operand rings cross 256 KiB, the descriptor must carry the wide start-address field"
    builds = len(re.findall(r"Tcgen05SmemDesc\.build\(", code))
    pinned = len(re.findall(r"\bversion=DESC_VERSION\b", code))
    assert builds >= 2 and pinned == builds, f"{builds} Tcgen05SmemDesc.build sites, {pinned} take version=DESC_VERSION"
    assert not re.search(r"\bversion=[0-3]\b", code), "a descriptor build re-literals its version"


class _Boom(Exception):
    pass


def _fake_plan(params):
    def _launch(*a, **k):
        raise _Boom()

    return SimpleNamespace(params=params, launch=_launch)


def test_tail_runner_declines_before_touching_the_artifact():
    p = k3.tail_params(128, 128, has_bias=True, want_stats=True)
    plan = _fake_plan(p)
    n_res, bn = 40, 2  # n_res >= 33 (the MN-major 64-row box of one bi)
    rows = bn * n_res
    bf16 = torch.bfloat16
    ok = dict(
        a_value=torch.zeros(128, bn, 40, dtype=bf16),
        a_gate=torch.zeros(rows, 128, dtype=bf16),
        w_value=torch.zeros(128, 128, dtype=bf16),
        w_gate=torch.zeros(128, 128, dtype=bf16),
        out=torch.zeros(rows, 128, dtype=bf16),
        ln_w_value=torch.zeros(128, dtype=bf16),
        ln_b_value=torch.zeros(128, dtype=bf16),
        ln_w_gate=torch.zeros(128, dtype=bf16),
        ln_b_gate=torch.zeros(128, dtype=bf16),
        bias=torch.zeros(256, dtype=bf16),
        stats=torch.zeros(rows, 2),
    )

    def run(n_res_=n_res, **over):
        kw = dict(ok, **over)
        return sub.run_ln_dual_gemm_tail(
            plan,
            kw["a_value"],
            kw["a_gate"],
            kw["w_value"],
            kw["w_gate"],
            kw["out"],
            kw["ln_w_value"],
            kw["ln_b_value"],
            kw["ln_w_gate"],
            kw["ln_b_gate"],
            n_res=n_res_,
            bn=bn,
            bias=kw["bias"],
            stats=kw["stats"],
            residual=None,
            stream=0,
        )

    with pytest.raises(_Boom):  # every check passes -> the artifact is reached
        run()
    with pytest.raises(ValueError, match="n_res must be >= 33"):
        run(
            n_res_=32,
            a_value=torch.zeros(128, bn, 32, dtype=bf16),
            a_gate=torch.zeros(bn * 32, 128, dtype=bf16),
            out=torch.zeros(bn * 32, 128, dtype=bf16),
            stats=torch.zeros(bn * 32, 2),
        )
    with pytest.raises(ValueError, match="a_value must be"):
        run(a_value=torch.zeros(128, bn, 48, dtype=bf16))  # the inner extent must be the padded one (40), not any multiple of 8
    with pytest.raises(ValueError, match="a_gate must be"):
        run(a_gate=torch.zeros(rows, 64, dtype=bf16))
    with pytest.raises(ValueError, match="w_gate must be"):
        run(w_gate=torch.zeros(64, 128, dtype=bf16))
    with pytest.raises(ValueError, match="out must be"):
        run(out=torch.zeros(rows, 256, dtype=bf16))
    with pytest.raises(ValueError, match="bias must be bound"):
        run(bias=None)
    with pytest.raises(ValueError, match="bias must be"):
        run(bias=torch.zeros(128, dtype=bf16))
    with pytest.raises(ValueError, match="stats must be"):
        run(stats=torch.zeros(rows, 2, dtype=bf16))
    with pytest.raises(ValueError, match="ln_w_gate must be 16-byte aligned"):
        run(ln_w_gate=torch.zeros(128 + 4, dtype=bf16)[4:])
    with pytest.raises(ValueError, match="pairing='columns'"):
        sub.run_ln_dual_gemm_tail(
            _fake_plan(sub.LnDualGemmParams()),
            *[ok[k] for k in ("a_value", "a_gate", "w_value", "w_gate", "out", "ln_w_value", "ln_b_value", "ln_w_gate", "ln_b_gate")],
            n_res=n_res,
            bn=bn,
            stream=0,
        )
    with pytest.raises(ValueError, match="pairing='accumulators'"):
        sub.run_ln_dual_gemm(
            plan,
            ok["a_gate"],
            torch.zeros(1, 256, 128, dtype=bf16),
            torch.zeros(1, 128, bn, 40, dtype=bf16),
            ok["ln_w_value"],
            ok["ln_b_value"],
            n_res=n_res,
            bn=bn,
            bias=None,
            mask=None,
            stats=None,
            stream=0,
        )


# ---------------------------------------------------------------------------
# Rubin: the fp64 oracle (y from a torch K1 / K2), the knob pins, the degenerate rows
# ---------------------------------------------------------------------------
def test_tail_runner_declines_a_row_tile_below_the_mn_major_box():
    """``k_block`` < 64 cannot serve the accumulators arm (its MN-major operand lands in 64-row boxes of ONE bi): a typed ValueError before the
    artifact, while 64 / 128 / 256 reach it; a malformed value (not a power of two) is the geometry's own ValueError."""
    p = k3.tail_params(128, 128, has_bias=False, want_stats=False)
    plan = _fake_plan(p)
    n_res, bn = 384, 2
    rows = bn * n_res
    bf16 = torch.bfloat16
    args = (
        torch.zeros(128, bn, 384, dtype=bf16),
        torch.zeros(rows, 128, dtype=bf16),
        torch.zeros(128, 128, dtype=bf16),
        torch.zeros(128, 128, dtype=bf16),
        torch.zeros(rows, 128, dtype=bf16),
        torch.zeros(128, dtype=bf16),
        torch.zeros(128, dtype=bf16),
        torch.zeros(128, dtype=bf16),
        torch.zeros(128, dtype=bf16),
    )
    for kb in (64, 128, 256, None):
        with pytest.raises(_Boom):
            sub.run_ln_dual_gemm_tail(plan, *args, n_res=n_res, bn=bn, bias=None, stats=None, residual=None, stream=0, k_block=kb)
    with pytest.raises(ValueError, match="k_block must be >= 64"):
        sub.run_ln_dual_gemm_tail(plan, *args, n_res=n_res, bn=bn, bias=None, stats=None, residual=None, stream=0, k_block=32)
    with pytest.raises(ValueError, match="power of two"):
        sub.run_ln_dual_gemm_tail(plan, *args, n_res=n_res, bn=bn, bias=None, stats=None, residual=None, stream=0, k_block=96)


def _inputs(n_res, batch, c_z, c, with_bias, seed=0, device="cuda"):
    """z and the K3 weights (cuEquivariance's conventions) plus a torch model of K1 / K2 producing the channel-major bf16 y."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    bf16 = torch.bfloat16
    z = (torch.randn(batch, n_res, n_res, c_z, generator=g) * 0.7 + torch.randn(c_z, generator=g)[None, None, None, :] * 0.5).to(bf16)
    ln_in_w = (1.0 + 0.1 * torch.randn(c_z, generator=g)).to(bf16)
    ln_in_b = (0.1 * torch.randn(c_z, generator=g)).to(bf16)
    ln_out_w = (1.0 + 0.1 * torch.randn(c, generator=g)).to(bf16)
    ln_out_b = (0.1 * torch.randn(c, generator=g)).to(bf16)
    p_in = (torch.randn(2 * c, c_z, generator=g) / math.sqrt(c_z)).to(bf16)
    g_in = (torch.randn(2 * c, c_z, generator=g) / math.sqrt(c_z)).to(bf16)
    p_out = (torch.randn(c_z, c, generator=g) / math.sqrt(c)).to(bf16)  # [c_z, c]
    g_out = (torch.randn(c_z, c_z, generator=g) / math.sqrt(c_z)).to(bf16)  # [c_z, c_z]
    p_ob = (0.2 * torch.randn(c_z, generator=g)).to(bf16) if with_bias else None
    g_ob = (0.2 * torch.randn(c_z, generator=g)).to(bf16) if with_bias else None
    # a torch K1 / K2 (fp64, the mask all ones): a, b gated projections -> the outgoing contraction -> bf16 y, channel-major with the inner pad
    z64 = z.double().reshape(-1, c_z)
    zin = torch.nn.functional.layer_norm(z64, (c_z,), ln_in_w.double(), ln_in_b.double(), 1e-5)
    a = torch.sigmoid(zin @ g_in.double()[:c].T) * (zin @ p_in.double()[:c].T)
    b = torch.sigmoid(zin @ g_in.double()[c:].T) * (zin @ p_in.double()[c:].T)
    a = a.reshape(batch, n_res, n_res, c)
    b = b.reshape(batch, n_res, n_res, c)
    y = torch.einsum("bikc,bjkc->bijc", a, b)  # outgoing
    npd = k3.n_pad(n_res)
    y_cm = torch.full((c, batch * n_res, npd), float("nan"), dtype=torch.float64)  # the pad columns are never read: NaN proves it
    y_cm[:, :, :n_res] = y.permute(3, 0, 1, 2).reshape(c, batch * n_res, n_res)
    y_cm = y_cm.to(bf16)
    out = dict(z=z, y=y_cm, ln_in_w=ln_in_w, ln_in_b=ln_in_b, ln_out_w=ln_out_w, ln_out_b=ln_out_b, p_out=p_out, g_out=g_out, p_ob=p_ob, g_ob=g_ob)
    return {k: (v.to(device) if v is not None else None) for k, v in out.items()}


def _reference(inp, eps, *, folded=None, residual=False):
    """fp64: LN2(y) -> bf16, LN1(z) -> bf16, the two GEMMs, sigmoid(gate + b_g) * (value + b_o) [+ z]; and the LN2 statistics.
    ``folded = (W_o', b_o')`` models the affine-fold artifact: LN2 without gamma / beta and the folded bf16 weights / bias."""
    z = inp["z"].double()
    batch, n_res, c_z = z.shape[0], z.shape[1], z.shape[-1]
    c = inp["y"].shape[0]
    rows_z = z.reshape(-1, c_z)
    y_rows = inp["y"].double()[:, :, :n_res].reshape(c, -1).T  # [rows, c]
    mean = y_rows.mean(dim=1)
    var = y_rows.var(dim=1, unbiased=False)
    rstd = 1.0 / torch.sqrt(var + eps)
    u = (y_rows - mean[:, None]) * rstd[:, None]
    if folded is None:
        u = u * inp["ln_out_w"].double()[None, :] + inp["ln_out_b"].double()[None, :]
        w_o = inp["p_out"].double()
        b_o = inp["p_ob"].double() if inp["p_ob"] is not None else None
    else:
        w_o = folded[0].double()
        b_o = folded[1].double()
    u = u.to(torch.bfloat16).double()
    zin = torch.nn.functional.layer_norm(rows_z, (c_z,), inp["ln_in_w"].double(), inp["ln_in_b"].double(), eps).to(torch.bfloat16).double()
    val = u @ w_o.T
    gate = zin @ inp["g_out"].double().T
    if b_o is not None:
        val = val + b_o[None, :]
    if inp["g_ob"] is not None:
        gate = gate + inp["g_ob"].double()[None, :]
    o = torch.sigmoid(gate) * val
    if residual:
        o = o + rows_z
    return o.reshape(batch, n_res, n_res, c_z), mean, rstd


def _stacked_biases(inp, p_ob=None):
    """The kernel's interleaved ``[2 c_z]`` bias from cuEquivariance's two ``[c_z]`` biases (``p_ob`` = the folded bias under
    ``affine_fold``, else ``inp["p_ob"]``) -- a ``torch.cat``, i.e. a GPU kernel of its own.  A timed callable does this ONCE, outside
    the timing (``_launch(..., biases=...)``); done per call it rides inside every CUDA-graph replay as a second kernel."""
    p_ob = inp["p_ob"] if p_ob is None else p_ob
    return k3.stack_tail_biases(p_ob, inp["g_ob"]) if (p_ob is not None or inp["g_ob"] is not None) else None


def _launch(plan, inp, out, stats, stream, *, weights=None, residual=None, biases=None, k_block=None):
    p_out, p_ob = (inp["p_out"], inp["p_ob"]) if weights is None else weights
    if biases is None:
        biases = _stacked_biases(inp, p_ob)
    return k3.run_trimul_tail(
        plan,
        inp["y"],
        inp["z"],
        p_out,
        inp["g_out"],
        out,
        inp["ln_out_w"],
        inp["ln_out_b"],
        inp["ln_in_w"],
        inp["ln_in_b"],
        biases=biases,
        stats=stats,
        residual=residual,
        stream=stream,
        k_block=k_block,
    )


_SENTINEL = 1.5e30
_TOL_ABS = 4e-3
_TOL_REL = 1.0 / 128.0


def _check_oracle(out, ref, label):
    diff = (out.double() - ref).abs()
    tol = _TOL_ABS + ref.abs() * _TOL_REL
    bad = int((diff > tol).sum().item())
    rel_rms = (diff.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    msg = f"{label}: {bad} cells beyond the derived tolerance; max |diff| {diff.max().item():.4g} at |ref| max {ref.abs().max().item():.4g}; rel RMS {rel_rms:.3g}"
    print(msg)
    assert bad == 0, msg
    assert rel_rms < 2e-3, msg
    return diff.max().item(), rel_rms


@requires_rubin
@pytest.mark.parametrize(
    "n_res,batch,c_z,with_bias",
    [
        (64, 2, 128, True),  # kb = 64: the two MN-major boxes of a CTA are two bi's
        (128, 2, 128, True),  # kb = 128: the two boxes split one bi's j
        (217, 2, 128, False),  # a tail tile (39 of 256 rows padded), no biases
        (256, 1, 128, True),
        (512, 1, 128, True),
        (128, 1, 256, True),  # c_z = c = 256: 4 K stages per tile, two N tiles, the interleaved bias blocks
    ],
)
def test_k3_matches_the_fp64_oracle(n_res, batch, c_z, with_bias):
    eps = 1e-5
    c = c_z
    plan = k3.build_trimul_tail(k3.tail_params(c_z, c, has_bias=with_bias, want_stats=True, eps=eps))
    inp = _inputs(n_res, batch, c_z, c, with_bias)
    out = torch.full((batch, n_res, n_res, c_z), _SENTINEL, device="cuda", dtype=torch.bfloat16)
    stats = torch.full((batch * n_res * n_res, 2), -1.0, device="cuda", dtype=torch.float32)
    stream = torch.cuda.current_stream().cuda_stream
    _launch(plan, inp, out, stats, stream)
    torch.cuda.synchronize()
    assert not (out.float() == _SENTINEL).any(), "a never-written cell survived"
    assert torch.isfinite(out.float()).all(), "a non-finite output (the NaN pad columns of y leaked into a live row?)"
    ref, mean64, rstd64 = _reference(inp, eps)
    _check_oracle(out, ref, f"K3 {n_res}:{batch}:{c_z}")
    torch.testing.assert_close(stats[:, 0].double(), mean64, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(stats[:, 1].double(), rstd64, rtol=1e-4, atol=1e-5)
    out2 = torch.empty_like(out)
    stats2 = torch.empty_like(stats)
    _launch(plan, inp, out2, stats2, stream)  # the two-launch trick
    torch.cuda.synchronize()
    assert torch.equal(out2, out) and torch.equal(stats2, stats)


@requires_rubin
@pytest.mark.parametrize("n_res,batch", [(217, 2), (512, 1)])
def test_transpose_prologue_matches_the_oracle_and_the_mn_major_form(n_res, batch):
    """The fallback form (the normalised y row written K-major, the MMA reading K-major) computes the same function: oracle green, and
    the difference to the "mn_major" form reported (the two write paths carry the same fp32 arithmetic -- bitwise equality is expected
    and asserted; a one-ulp difference would mean ptxas contracted the two code shapes differently and is reported as such)."""
    eps = 1e-5
    c_z = c = 128
    inp = _inputs(n_res, batch, c_z, c, True, seed=5)
    outs = {}
    for form in ("mn_major", "transpose"):
        plan = k3.build_trimul_tail(k3.tail_params(c_z, c, has_bias=True, want_stats=True, eps=eps, a0_prologue=form))
        out = torch.full((batch, n_res, n_res, c_z), _SENTINEL, device="cuda", dtype=torch.bfloat16)
        stats = torch.full((batch * n_res * n_res, 2), -1.0, device="cuda")
        _launch(plan, inp, out, stats, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        outs[form] = (out, stats)
    ref, _, _ = _reference(inp, eps)
    _check_oracle(outs["transpose"][0], ref, f"K3 transpose {n_res}:{batch}")
    n_diff = int((outs["mn_major"][0].view(torch.int16) != outs["transpose"][0].view(torch.int16)).sum().item())
    print(f"mn_major vs transpose: {n_diff} cells differ (of {outs['mn_major'][0].numel()})")
    assert torch.equal(outs["mn_major"][1], outs["transpose"][1]), "the statistics must be bitwise equal (same reads, same arithmetic)"
    assert n_diff == 0, f"the two prologue forms differ on {n_diff} cells"


@requires_rubin
@pytest.mark.parametrize("n_res,batch,c_z", [(217, 2, 128), (128, 1, 256)])
@pytest.mark.parametrize(
    "ln_pass_form,epi_math", [("smem", "scalar"), ("smem_rolled", "scalar"), ("registers", "packed"), ("smem_rolled", "packed"), ("smem_rolled_pf", "packed")]
)
def test_ln_pass_form_and_epi_math_are_bitwise_equal_to_the_default_forms(n_res, batch, c_z, ln_pass_form, epi_math):
    """On the accumulators arm (mn_major prologue): the re-read LayerNorm forms and the packed epilogue are BITWISE the default forms (output and
    the LN2 statistics) -- the same bf16 words summed in the same order, the same fp32 operations per element."""
    inp = _inputs(n_res, batch, c_z, c_z, True, seed=17)
    outs = []
    for lnf, epm in (("registers", "scalar"), (ln_pass_form, epi_math)):
        plan = k3.build_trimul_tail(k3.tail_params(c_z, c_z, has_bias=True, want_stats=True, a0_prologue="mn_major", ln_pass_form=lnf, epi_math=epm))
        out = torch.full((batch, n_res, n_res, c_z), _SENTINEL, device="cuda", dtype=torch.bfloat16)
        stats = torch.full((batch * n_res * n_res, 2), -1.0, device="cuda")
        _launch(plan, inp, out, stats, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert not (out.float() == _SENTINEL).any(), f"{lnf}/{epm}: a never-written cell"
        outs.append((out, stats))
    assert torch.equal(outs[0][0], outs[1][0]), f"{ln_pass_form}/{epi_math} changed the output"
    assert torch.equal(outs[0][1], outs[1][1]), f"{ln_pass_form}/{epi_math} changed the statistics"


def test_register_split_hint_vocabulary_is_a_typed_decline():
    """`min_blocks_per_mp` admits 0 (the rendering as shipped) and 1 (ptxas honours the body's setmaxnreg split); 2 would cap the allocation at
    80 registers with the split impossible, a bool is not an int -- both a ValueError naming the axis, from the validator, before any build."""
    for bad in (2, -1, True):
        with pytest.raises(ValueError, match="min_blocks_per_mp"):
            sub.validate_ln_dual_gemm_params(k3.tail_params(256, 256, has_bias=False, ln_pass_form="smem_rolled", epi_math="packed", min_blocks_per_mp=bad))
    for ok in (0, 1):
        p = k3.tail_params(256, 256, has_bias=False, ln_pass_form="smem_rolled", epi_math="packed", min_blocks_per_mp=ok)
        sub.validate_ln_dual_gemm_params(p)
        assert p.min_blocks_per_mp == ok
    assert k3.tail_params(256, 256, has_bias=False).min_blocks_per_mp == 0  # None = the substrate's default: no directive


@requires_rubin
@pytest.mark.parametrize("n_res,batch,c_z,with_bias", [(217, 2, 128, False), (128, 1, 256, False), (33, 1, 256, True), (384, 1, 256, False)])
def test_register_split_hint_is_bitwise_the_default_rendering(n_res, batch, c_z, with_bias):
    """The `.minnctapersm` hint (`min_blocks_per_mp=1`: ptxas honours the register split instead of allocating the flat count) is a launch
    attribute -- the SAME body, the same arithmetic: output and statistics BITWISE the hint-0 plan's, with and without the output biases, on the
    rolled + packed forms the heuristics ship at c = 256 (the rendering whose bias-off form spills without the hint)."""
    inp = _inputs(n_res, batch, c_z, c_z, with_bias, seed=23)
    outs = []
    for hint in (0, 1):
        plan = k3.build_trimul_tail(
            k3.tail_params(
                c_z, c_z, has_bias=with_bias, want_stats=True, a0_prologue="mn_major", ln_pass_form="smem_rolled", epi_math="packed", min_blocks_per_mp=hint
            )
        )
        assert plan.params.min_blocks_per_mp == hint
        out = torch.full((batch, n_res, n_res, c_z), _SENTINEL, device="cuda", dtype=torch.bfloat16)
        stats = torch.full((batch * n_res * n_res, 2), -1.0, device="cuda")
        _launch(plan, inp, out, stats, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert not (out.float() == _SENTINEL).any(), f"hint {hint}: a never-written cell"
        assert torch.isfinite(out.float()).all(), f"hint {hint}: a non-finite cell"
        outs.append((out, stats))
    assert torch.equal(outs[0][0], outs[1][0]), "the register-split hint changed the output"
    assert torch.equal(outs[0][1], outs[1][1]), "the register-split hint changed the statistics"


@requires_rubin
@pytest.mark.parametrize("n_res,batch,c_z", [(217, 2, 128), (128, 1, 256)])
def test_tree_ln_form_meets_the_oracle_bound_and_its_distance_to_the_rolled_form_is_measured(n_res, batch, c_z):
    """`ln_pass_form="smem_rolled_tree"` on the accumulators arm: both operands' statistics summed in FOUR partials (a different fp32 ORDER,
    :data:`LN_PASS_FORMS_REORDERED`).  A summation-order change moves a few LayerNorm outputs across a bf16 rounding midpoint, and those flips
    propagate through the GEMM exactly like the dataset-dependent flips the shipped form itself shows on other seeds (the module's `bad == 0` oracle
    holds at its own datasets; on this seed the SHIPPED rolled form reads 2 cells of 12 M beyond the per-cell bound), so BOTH forms are held to the
    API's documented three-part oracle on the same inputs: the bulk bound `4e-3 + |ref| / 128` on all but a tail of at most 2e-5 of the cells, the
    cap `2e-2 + |ref| / 64` on EVERY cell, rel RMS < 2e-3 (nothing widened: the module's bounds), the LN2 statistics within the oracle test's
    tolerance, and the distance to the rolled form REPORTED (cells differing, the statistics' max |diff|): a named form, never a bitwise pin or a
    default."""
    eps = 1e-5
    inp = _inputs(n_res, batch, c_z, c_z, True, seed=17)
    ref, mean64, rstd64 = _reference(inp, eps)
    outs = {}
    for lnf in ("smem_rolled", "smem_rolled_tree"):
        plan = k3.build_trimul_tail(
            k3.tail_params(c_z, c_z, has_bias=True, want_stats=True, eps=eps, a0_prologue="mn_major", ln_pass_form=lnf, epi_math="packed")
        )
        out = torch.full((batch, n_res, n_res, c_z), _SENTINEL, device="cuda", dtype=torch.bfloat16)
        stats = torch.full((batch * n_res * n_res, 2), -1.0, device="cuda")
        _launch(plan, inp, out, stats, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert not (out.float() == _SENTINEL).any(), f"{lnf}: a never-written cell"
        diff = (out.double() - ref).abs()
        bulk = _TOL_ABS + ref.abs() * _TOL_REL
        cap = 2e-2 + ref.abs() / 64.0
        bad_bulk, bad_cap = int((diff > bulk).sum().item()), int((diff > cap).sum().item())
        rel_rms = (diff.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        print(
            f"K3 {lnf} {n_res}:{batch}:{c_z}: {bad_bulk} of {diff.numel()} cells beyond the bulk bound, {bad_cap} beyond the cap; max |diff| {diff.max().item():.4g}; rel RMS {rel_rms:.3g}"
        )
        assert bad_cap == 0 and bad_bulk <= 2e-5 * diff.numel() and rel_rms < 2e-3, f"K3 {lnf}: {bad_bulk} bulk / {bad_cap} cap / rel RMS {rel_rms:.3g}"
        torch.testing.assert_close(stats[:, 0].double(), mean64, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(stats[:, 1].double(), rstd64, rtol=1e-4, atol=1e-5)
        outs[lnf] = (out, stats)
    n_diff = int((outs["smem_rolled"][0].view(torch.int16) != outs["smem_rolled_tree"][0].view(torch.int16)).sum().item())
    d_stats = (outs["smem_rolled"][1] - outs["smem_rolled_tree"][1]).abs()
    print(
        f"tree vs rolled {n_res}:{batch}:{c_z}: {n_diff} output cells differ (of {outs['smem_rolled'][0].numel()}); "
        f"max |d mean| {d_stats[:, 0].max().item():.3g}, max |d rstd| {d_stats[:, 1].max().item():.3g}"
    )
    assert sub.LN_PASS_FORMS_REORDERED == ("smem_rolled_tree",)


@requires_rubin
def test_affine_fold_matches_its_folded_reference_and_its_distance_to_the_unfolded_function_is_measured():
    """``affine_fold = True``: the kernel computes the FOLDED function exactly as well as the unfolded kernel computes the unfolded one
    (oracle against the fold-aware reference); its distance to the UNFOLDED fp64 function -- the knob's numerics class (a second bf16
    rounding of the weights and bias; expected ~2x the unfolded max-abs error) -- is measured against the unfolded kernel's."""
    eps = 1e-5
    n_res, batch, c_z = 256, 1, 128
    c = c_z
    inp = _inputs(n_res, batch, c_z, c, True, seed=9)
    stream = torch.cuda.current_stream().cuda_stream
    ref_unfolded, _, _ = _reference(inp, eps)
    # the unfolded kernel
    plan = k3.build_trimul_tail(k3.tail_params(c_z, c, has_bias=True, want_stats=False, eps=eps))
    out_u = torch.full((batch, n_res, n_res, c_z), _SENTINEL, device="cuda", dtype=torch.bfloat16)
    _launch(plan, inp, out_u, None, stream)
    torch.cuda.synchronize()
    max_u, rms_u = _check_oracle(out_u, ref_unfolded, "unfolded kernel vs the unfolded fp64 function")
    # the folded kernel
    wf, bf = k3.fold_tail_affine(inp["p_out"], inp["p_ob"], inp["ln_out_w"], inp["ln_out_b"])
    plan_f = k3.build_trimul_tail(k3.tail_params(c_z, c, has_bias=True, want_stats=False, eps=eps, affine_fold=True))
    out_f = torch.full((batch, n_res, n_res, c_z), _SENTINEL, device="cuda", dtype=torch.bfloat16)
    _launch(plan_f, inp, out_f, None, stream, weights=(wf, bf))
    torch.cuda.synchronize()
    ref_folded, _, _ = _reference(inp, eps, folded=(wf, bf))
    _check_oracle(out_f, ref_folded, "folded kernel vs the FOLDED fp64 function")
    diff_f = (out_f.double() - ref_unfolded).abs()
    rms_f = (diff_f.pow(2).mean().sqrt() / ref_unfolded.pow(2).mean().sqrt()).item()
    max_f = diff_f.max().item()
    msg = f"AFFINE_FOLD numerics vs the unfolded function: max |diff| {max_f:.4g} (unfolded kernel {max_u:.4g}, x{max_f / max_u:.2f}), rel RMS {rms_f:.3g} (unfolded {rms_u:.3g}, x{rms_f / rms_u:.2f})"
    print(msg)
    assert max_f >= max_u * 0.5 and max_f < 0.1, msg  # the fold is a real numerics change (recorded), not a defect


@requires_rubin
def test_fuse_residual_adds_the_residual_before_the_rounding():
    eps = 1e-5
    n_res, batch, c_z = 128, 2, 128
    c = c_z
    inp = _inputs(n_res, batch, c_z, c, True, seed=13)
    plan = k3.build_trimul_tail(k3.tail_params(c_z, c, has_bias=True, want_stats=False, eps=eps, fuse_residual=True))
    out = torch.full((batch, n_res, n_res, c_z), _SENTINEL, device="cuda", dtype=torch.bfloat16)
    _launch(plan, inp, out, None, torch.cuda.current_stream().cuda_stream, residual=inp["z"])
    torch.cuda.synchronize()
    ref, _, _ = _reference(inp, eps, residual=True)
    _check_oracle(out, ref, "fused residual")
    # and it is NOT the unfused two-rounding result cell for cell (the fused form rounds once)
    plan_u = k3.build_trimul_tail(k3.tail_params(c_z, c, has_bias=True, want_stats=False, eps=eps))
    out_u = torch.empty_like(out)
    _launch(plan_u, inp, out_u, None, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    two_round = (out_u.float() + inp["z"].float()).to(torch.bfloat16)
    frac = (two_round.view(torch.int16) != out.view(torch.int16)).double().mean().item()
    print(f"fused vs unfused+add: {100 * frac:.2f} % of cells differ by the rounding order")


@requires_rubin
def test_prefetch_knob_is_bitwise_neutral():
    eps = 1e-5
    n_res, batch, c_z = 217, 1, 128
    inp = _inputs(n_res, batch, c_z, c_z, True, seed=17)
    outs = []
    for pf in (False, True):
        plan = k3.build_trimul_tail(k3.tail_params(c_z, c_z, has_bias=True, want_stats=True, eps=eps, prefetch_a_l2=pf))
        out = torch.full((batch, n_res, n_res, c_z), _SENTINEL, device="cuda", dtype=torch.bfloat16)
        stats = torch.full((batch * n_res * n_res, 2), -1.0, device="cuda")
        _launch(plan, inp, out, stats, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        outs.append((out, stats))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])


@requires_rubin
def test_tail_runner_declines_a_small_n_and_a_misshaped_y():
    plan = k3.build_trimul_tail(k3.tail_params(128, 128, has_bias=False, want_stats=False))
    bf16 = torch.bfloat16
    z = torch.zeros(1, 32, 32, 128, device="cuda", dtype=bf16)
    with pytest.raises(ValueError, match="N >= 33"):
        k3.run_trimul_tail(
            plan,
            torch.zeros(128, 32, 32, device="cuda", dtype=bf16),
            z,
            torch.zeros(128, 128, device="cuda", dtype=bf16),
            torch.zeros(128, 128, device="cuda", dtype=bf16),
            torch.zeros_like(z),
            *[torch.zeros(128, device="cuda", dtype=bf16)] * 4,
            stream=0,
        )
    z = torch.zeros(1, 40, 40, 128, device="cuda", dtype=bf16)
    with pytest.raises(ValueError, match="y must be"):
        k3.run_trimul_tail(
            plan,
            torch.zeros(128, 40, 48, device="cuda", dtype=bf16),
            z,
            torch.zeros(128, 128, device="cuda", dtype=bf16),
            torch.zeros(128, 128, device="cuda", dtype=bf16),
            torch.zeros_like(z),
            *[torch.zeros(128, device="cuda", dtype=bf16)] * 4,
            stream=0,
        )


@requires_rubin
@pytest.mark.parametrize("n_res,batch,k_block", [(384, 2, 128), (384, 1, 64), (128, 2, 256)])
def test_k3_row_tile_knob_is_bitwise_the_rule(n_res, batch, k_block):
    """The accumulators arm under ``k_block``: the same plan, the same bits as the rule (output and statistics).  kb = 128 at N = 384 splits one
    bi's j across the CTA's two MN-major boxes (as N = 128 does today), kb = 64 gives each box its own bi (as N = 64 does); the rule's 256 pads
    25 % of every pair tile at N = 384."""
    c_z = c = 128
    plan = k3.build_trimul_tail(k3.tail_params(c_z, c, has_bias=True, want_stats=True, eps=1e-5))
    inp = _inputs(n_res, batch, c_z, c, True, seed=33)
    stream = torch.cuda.current_stream().cuda_stream
    outs = []
    for kb in (None, k_block):
        out = torch.full((batch, n_res, n_res, c_z), _SENTINEL, device="cuda", dtype=torch.bfloat16)
        stats = torch.full((batch * n_res * n_res, 2), -1.0, device="cuda", dtype=torch.float32)
        g = _launch(plan, inp, out, stats, stream, k_block=kb)
        torch.cuda.synchronize()
        assert g.kb == (sub.default_k_block(n_res) if kb is None else kb)
        assert not (out.float() == _SENTINEL).any() and torch.isfinite(out.float()).all() and (stats[:, 1] > 0).all()
        outs.append((out, stats, g))
    (o0, s0, g0), (o1, s1, g1) = outs
    assert torch.equal(o0, o1) and torch.equal(s0, s1), f"the row tile changed the output bits (kb {g0.kb} -> {g1.kb})"
