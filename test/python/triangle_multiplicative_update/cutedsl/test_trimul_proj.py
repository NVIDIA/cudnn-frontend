# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K1 of the triangle multiplicative update -- the LayerNorm-prologue gated dual projection on the
``cudnn.gemm.frost.ln_prologue`` substrate -- against an fp64 reference, plus the GPU-free contract tests.

Tolerance of the oracle (derived, stated): the kernel's operand is the LayerNormed row rounded ONCE to bf16 and the reference
models exactly that (fp64 LayerNorm -> bf16 -> fp64 GEMM), so the remaining differences are (i) the fp32 accumulation over
K <= 256 (<= K * 2^-24 relative -- negligible), (ii) the sigmoid through ``tanh.approx`` (<= 2^-11 absolute on the tanh, i.e.
<= 2^-12 on the sigmoid, times |value| ~ O(1)), (iii) the one bf16 rounding of the output (half-ulp = 2^-9 relative), and (iv) the
fp32-vs-fp64 LayerNorm statistics flipping the bf16 rounding of a few operands by one ulp (2^-8 * |x_j| * |W_nj| per flip).
Together: ``|diff| <= 4e-3 + |ref| / 128`` (two bf16 half-ulps relative plus an absolute floor for (ii) and (iv)); the test also
reports the measured maximum.  Masked and padded rows must be EXACTLY +0.0 (a select in the kernel, sdpa-invariants.md S 2).

The kernel targets Rubin (cc 10.7); the accept tests skip elsewhere with the reason stated.  Every launch is a single process
under the harness's ``timeout`` (exit 124 = hang).
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
from cudnn.triangle_multiplicative_update.kernels import trimul_proj as k1  # noqa: E402
from sdpa.frost import frost_test_utils as sass_utils  # noqa: E402  -- the SASS probe (trace-compile + nvdisasm opcode counts) and the spill-pin bound

_SM107 = (10, 7)


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


requires_rubin = pytest.mark.skipif(_cc() != _SM107, reason=f"the LN-prologue dual GEMM is rendered for SM107 (Rubin) only; found {_cc()}")


# ---------------------------------------------------------------------------
# GPU-free: the row-block geometry, the weight interleave, the typed declines, the model counts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n_res,bn,k_block",
    [
        (8, 1, None),
        (9, 3, None),
        (64, 128, None),
        (100, 7, None),
        (128, 256, None),
        (217, 217, None),
        (256, 1, None),
        (384, 2, None),
        (512, 512, None),
        (1000, 3, None),
        (384, 2, 128),  # the k_block knob: 3 full k blocks of 2 bi (the rule's 256 pads 25 % of every pair tile at N = 384)
        (384, 1, 64),
        (217, 3, 32),  # 7 blocks of 32 (224 rows per bi: 3 % padding instead of the rule's 15 %) -- the projection stage alone
        (128, 5, 256),  # a k_block above N: one bi per tile, half of every tile padding (legal)
        (1000, 3, 8),
    ],
)
def test_row_block_geometry_covers_every_valid_row_exactly_once(n_res, bn, k_block):
    g = sub.row_block_geometry(n_res, bn, k_block)
    assert g.kb == (sub.default_k_block(n_res) if k_block is None else k_block)
    assert g.r_i * g.kb == 256 and g.kb & (g.kb - 1) == 0 and g.kb >= 8 and g.kb == 1 << g.lkb
    assert g.kbc * g.rc == 128 and g.kbc == min(g.kb, 128)
    assert g.n_pad % 8 == 0 and n_res <= g.n_pad < n_res + 8
    assert g.m_sched == g.pair_tiles * 256 and g.rows_valid == bn * n_res
    covered = set()
    for t in range(g.pair_tiles):
        ib, kbi = divmod(t, g.n_kb)
        for r in range(256):
            bi = ib * g.r_i + (r >> g.lkb)
            k = kbi * g.kb + (r & (g.kb - 1))
            if bi < bn and k < n_res:
                assert (bi, k) not in covered
                covered.add((bi, k))
    assert len(covered) == bn * n_res


def test_row_block_geometry_k_block_is_a_typed_vocabulary():
    """The knob's domain: an int power of two in [8, 256]; the rule pads 25 % at N = 384 and k_block = 128 pads none (the lever's arithmetic)."""
    for bad in (0, 4, 96, 512, 7, -128, 128.0, "128", True):
        with pytest.raises(ValueError, match="k_block"):
            sub.row_block_geometry(384, 2, bad)
    assert sub.validate_k_block(128) == 128 and sub.default_k_block(384) == 256 and sub.default_k_block(128) == 128 and sub.default_k_block(33) == 64
    rule, knob = sub.row_block_geometry(384, 4), sub.row_block_geometry(384, 4, 128)
    assert rule.rows_valid == knob.rows_valid == 4 * 384
    assert rule.rows_computed == 4 * 2 * 256 == 4 * rule.rows_valid // 3 and knob.rows_computed == knob.rows_valid  # 25 % padding -> none
    assert (rule.pair_tiles, knob.pair_tiles) == (8, 6) and (rule.kbc, rule.rc) == (knob.kbc, knob.rc) == (128, 1)  # the SAME per-CTA TMA box
    assert sub.row_block_geometry(384, 1, 64).kbc == 64 and sub.row_block_geometry(384, 1, 64).rc == 2


def test_interleave_value_gate_puts_value_and_gate_blocks_in_one_256_tile():
    c, kk = 256, 64
    v = torch.arange(c * kk, dtype=torch.float32).reshape(c, kk)
    g = -v
    w = sub.interleave_value_gate(v, g)
    assert w.shape == (2 * c, kk)
    for j in range(c // 128):
        assert torch.equal(w[256 * j : 256 * j + 128], v[128 * j : 128 * j + 128])
        assert torch.equal(w[256 * j + 128 : 256 * j + 256], g[128 * j : 128 * j + 128])
    b = sub.interleave_value_gate_bias(torch.arange(c, dtype=torch.float32), -torch.arange(c, dtype=torch.float32))
    assert torch.equal(b[:128], torch.arange(128, dtype=torch.float32)) and torch.equal(b[128:256], -torch.arange(128, dtype=torch.float32))


def test_stack_projection_weights_follows_the_cueq_row_convention():
    c, cz = 128, 64
    p = torch.randn(2 * c, cz)
    g = torch.randn(2 * c, cz)
    w = k1.stack_projection_weights(p, g)
    assert w.shape == (2, 2 * c, cz) and w.is_contiguous()
    assert torch.equal(w[0], sub.interleave_value_gate(p[:c], g[:c]))  # pass a
    assert torch.equal(w[1], sub.interleave_value_gate(p[c:], g[c:]))  # pass b
    b = k1.stack_projection_biases(torch.arange(2 * c, dtype=torch.float32), torch.zeros(2 * c))
    assert b.shape == (2, 2 * c) and torch.equal(b[1, :128], torch.arange(c, 2 * c, dtype=torch.float32))


@pytest.mark.parametrize(
    "kw",
    [dict(k=100), dict(k=0), dict(c_out=64), dict(c_out=0), dict(passes=0), dict(eps=0.0), dict(pass_order="zigzag"), dict(pass_order="interleaved", passes=3)],
)
def test_params_validation_declines_typed(kw):
    with pytest.raises(ValueError):
        sub.validate_ln_dual_gemm_params(sub.LnDualGemmParams(**kw))
    with pytest.raises(TypeError):
        sub.validate_ln_dual_gemm_params(object())


def test_body_path_names_one_rendered_body_per_arch():
    import os

    assert os.path.isfile(sub.body_path()) and os.path.basename(sub.body_path("sm107")) == "sm107_dual_gemm.py"
    with pytest.raises(ValueError, match="rendered body"):
        sub.body_path("sm100")  # a later step's body: a typed decline until it exists, never a silent fallback to the sm107 one


def test_planes_shape_and_strides_are_the_k2_contract():
    assert k1.planes_shape(128, 2, 217) == (2, 128, 434, 224)
    assert k1.plane_strides(2, 217) == (434 * 224, 224, 1)
    assert k1.planes_shape(256, 1, 512) == (2, 256, 512, 512)


def test_model_counts():
    assert k1.proj_flop(128, 128, 1, 512) == 2 * 512 * 512 * 128 * 512
    assert k1.proj_bytes(128, 128, 1, 512, with_mask=False, with_stats=False) == 512 * 512 * (128 * 2 + 2 * 128 * 2)
    assert k1.proj_bytes(128, 128, 1, 512) == k1.proj_bytes(128, 128, 1, 512, with_mask=False, with_stats=False) + 512 * 512 * (2 + 8)


class _Boom(Exception):
    pass


def _fake_plan(params):
    def _launch(*a, **k):
        raise _Boom()

    return SimpleNamespace(params=params, launch=_launch)


def test_runner_declines_before_touching_the_artifact():
    p = sub.LnDualGemmParams(k=128, c_out=128, passes=2)
    plan = _fake_plan(p)
    n_res, bn = 16, 4
    rows = bn * n_res
    bf16 = torch.bfloat16
    ok = dict(
        z=torch.zeros(rows, 128, dtype=bf16),
        w=torch.zeros(2, 256, 128, dtype=bf16),
        out=torch.zeros(2, 128, bn, 16, dtype=bf16),
        ln_weight=torch.zeros(128, dtype=bf16),
        ln_bias=torch.zeros(128, dtype=bf16),
        bias=torch.zeros(2, 256, dtype=bf16),
        mask=torch.zeros(rows, dtype=bf16),
        stats=torch.zeros(rows, 2),
    )

    def run(**over):
        kw = dict(ok, **over)
        return sub.run_ln_dual_gemm(
            plan, kw["z"], kw["w"], kw["out"], kw["ln_weight"], kw["ln_bias"], n_res=n_res, bn=bn, bias=kw["bias"], mask=kw["mask"], stats=kw["stats"], stream=0
        )

    with pytest.raises(_Boom):  # every check passes -> the artifact is reached
        run()
    with pytest.raises(ValueError, match="z must be"):
        run(z=ok["z"].float())
    with pytest.raises(ValueError, match="w must be"):
        run(w=torch.zeros(2, 128, 128, dtype=bf16))
    with pytest.raises(ValueError, match="out must be"):
        run(out=torch.zeros(2, 128, bn, 15, dtype=bf16))
    with pytest.raises(ValueError, match="bias must be bound"):
        run(bias=None)
    with pytest.raises(ValueError, match="mask must be"):
        run(mask=torch.zeros(rows + 1, dtype=bf16))
    with pytest.raises(ValueError, match="stats must be"):
        run(stats=torch.zeros(rows, 2, dtype=bf16))
    with pytest.raises(ValueError, match="ln_weight must be"):
        run(ln_weight=torch.zeros(64, dtype=bf16))
    # alignment: a contiguous but OFFSET view has the right shape and dtype and would fault (or read the wrong half-word) in the kernel
    with pytest.raises(ValueError, match="ln_weight must be 16-byte aligned"):
        run(ln_weight=torch.zeros(128 + 4, dtype=bf16)[4:])  # 8-byte offset
    with pytest.raises(ValueError, match="bias must be 16-byte aligned"):
        run(bias=torch.zeros(2 * 256 + 4, dtype=bf16)[4:].view(2, 256))
    with pytest.raises(ValueError, match="mask must be 4-byte aligned"):
        run(mask=torch.zeros(rows + 1, dtype=bf16)[1:])  # 2-byte offset: the kernel reads aligned 32-bit words
    with pytest.raises(ValueError, match="stats must be 8-byte aligned"):
        run(stats=torch.zeros(rows * 2 + 1)[1:].view(rows, 2))  # 4-byte offset: the kernel stores 8-byte (mean, rstd) pairs
    with pytest.raises(_Boom):  # aligned offset views pass (16 / 4 / 8 bytes)
        run(ln_weight=torch.zeros(128 + 8, dtype=bf16)[8:], mask=torch.zeros(rows + 2, dtype=bf16)[2:], stats=torch.zeros(rows * 2 + 2)[2:].view(rows, 2))


# ---------------------------------------------------------------------------
# Rubin: the fp64 oracle, the skip pin, the degenerate rows
# ---------------------------------------------------------------------------
def _inputs(n_res, batch, c_z, c, mask_kind, with_bias, seed=0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    bf16 = torch.bfloat16
    # per-channel offsets so the LayerNorm mean is not ~0 (exercises the two-pass statistics), unit-ish variance
    z = (torch.randn(batch, n_res, n_res, c_z, generator=g) * 0.7 + torch.randn(c_z, generator=g)[None, None, None, :] * 0.5).to(bf16)
    ln_w = (1.0 + 0.1 * torch.randn(c_z, generator=g)).to(bf16)
    ln_b = (0.1 * torch.randn(c_z, generator=g)).to(bf16)
    p_in = (torch.randn(2 * c, c_z, generator=g) / math.sqrt(c_z)).to(bf16)  # rows [a; b]
    g_in = (torch.randn(2 * c, c_z, generator=g) / math.sqrt(c_z)).to(bf16)
    if with_bias:
        p_b = (0.2 * torch.randn(2 * c, generator=g)).to(bf16)
        g_b = (0.2 * torch.randn(2 * c, generator=g)).to(bf16)
    else:
        p_b = g_b = None
    if mask_kind == "all":
        mask = torch.ones(batch, n_res, n_res)
    elif mask_kind == "padded":  # a right-padded protein per batch: valid tokens L_b, m[i, k] = (i < L) & (k < L)
        mask = torch.zeros(batch, n_res, n_res)
        for b in range(batch):
            L = (3 * n_res) // 4 if b == 0 else max(1, n_res // 3)
            mask[b, :L, :L] = 1.0
    elif mask_kind == "random":
        mask = (torch.rand(batch, n_res, n_res, generator=g) > 0.3).float()
    elif mask_kind == "fractional":  # 70 % of the rows carry a FRACTIONAL weight in [0.25, 1), the rest 0 -- distinguishes the rounding order
        mask = (torch.rand(batch, n_res, n_res, generator=g) > 0.3).float() * (0.25 + 0.75 * torch.rand(batch, n_res, n_res, generator=g))
    else:
        raise ValueError(mask_kind)
    mask = mask.to(bf16)
    return {k: (v.to(device) if v is not None else None) for k, v in dict(z=z, ln_w=ln_w, ln_b=ln_b, p_in=p_in, g_in=g_in, p_b=p_b, g_b=g_b, mask=mask).items()}


def _reference(inp, c, eps):
    """fp64: LayerNorm -> bf16 operand -> GEMM -> sigmoid(gate + b_g) * (value + b_p) * mask, channel-major [2, c, B*N, N]; stats."""
    z = inp["z"].double()
    batch, n_res = z.shape[0], z.shape[1]
    c_z = z.shape[-1]
    rows = z.reshape(-1, c_z)
    mean = rows.mean(dim=1)
    var = rows.var(dim=1, unbiased=False)
    rstd = 1.0 / torch.sqrt(var + eps)
    x = (rows - mean[:, None]) * rstd[:, None] * inp["ln_w"].double()[None, :] + inp["ln_b"].double()[None, :]
    x = x.to(torch.bfloat16).double()  # the kernel's MMA operand rounding
    m = inp["mask"].double().reshape(-1)
    outs = []
    for pss in range(2):
        wp = inp["p_in"].double()[pss * c : (pss + 1) * c]
        wg = inp["g_in"].double()[pss * c : (pss + 1) * c]
        val = x @ wp.T
        gate = x @ wg.T
        if inp["p_b"] is not None:
            val = val + inp["p_b"].double()[pss * c : (pss + 1) * c][None, :]
            gate = gate + inp["g_b"].double()[pss * c : (pss + 1) * c][None, :]
        o = torch.sigmoid(gate) * val * m[:, None]
        o = torch.where(m[:, None] == 0, torch.zeros_like(o), o)
        outs.append(o.T.reshape(c, batch * n_res, n_res))
    return torch.stack(outs), mean, rstd


def _launch(plan, inp, c, planes, stats, stream, *, w_stack=None, b_stack=None, k_block=None):
    """``w_stack`` / ``b_stack`` = the interleaved weight / bias stacks (``torch.cat`` copies, GPU kernels of their own): a timed callable
    builds them ONCE and passes them in; per call they ride inside every CUDA-graph replay as extra kernels."""
    if w_stack is None:
        w_stack = k1.stack_projection_weights(inp["p_in"], inp["g_in"])
    if b_stack is None:
        b_stack = k1.stack_projection_biases(inp["p_b"], inp["g_b"]) if inp["p_b"] is not None else None
    return k1.run_trimul_proj(
        plan, inp["z"], w_stack, planes, inp["ln_w"], inp["ln_b"], bias_stacked=b_stack, mask=inp["mask"], stats=stats, stream=stream, k_block=k_block
    )


_SENTINEL = 1.5e30


@requires_rubin
@pytest.mark.parametrize(
    "n_res,batch,c_z,c,mask_kind,with_bias",
    [
        (64, 2, 128, 128, "all", True),
        (128, 2, 128, 128, "padded", True),
        (217, 2, 128, 128, "random", False),
        (217, 1, 128, 128, "random", True),  # an ODD mask length (47 089 rows): the last row's mask is read without touching the byte past the tensor
        (256, 1, 128, 128, "padded", True),
        (512, 1, 128, 128, "all", True),
        (128, 1, 256, 256, "padded", True),  # two K tiles per stage group (K = 256) and two N tiles (c = 256)
        (128, 1, 128, 128, "fractional", True),  # fractional mask weights (the mask multiplies in fp32 BEFORE the bf16 rounding)
    ],
)
def test_k1_matches_the_fp64_oracle(n_res, batch, c_z, c, mask_kind, with_bias):
    eps = 1e-5
    params = k1.proj_params(c_z, c, has_bias=with_bias, has_mask=True, want_stats=True, eps=eps)
    plan = k1.build_trimul_proj(params)
    inp = _inputs(n_res, batch, c_z, c, mask_kind, with_bias)
    planes = torch.full(k1.planes_shape(c, batch, n_res), _SENTINEL, device="cuda", dtype=torch.bfloat16)
    stats = torch.full((batch * n_res * n_res, 2), -1.0, device="cuda", dtype=torch.float32)
    stream = torch.cuda.current_stream().cuda_stream
    g = _launch(plan, inp, c, planes, stats, stream)
    torch.cuda.synchronize()
    npd = g.n_pad
    valid = planes[:, :, :, :n_res].float()
    assert not (valid == _SENTINEL).any(), "a never-written cell survived in the valid region"
    if npd > n_res:
        assert (planes[:, :, :, n_res:npd].view(torch.int16) == 0).all(), "the k padding must be written as exactly +0.0"
    ref, mean64, rstd64 = _reference(inp, c, eps)
    m = inp["mask"].float().reshape(-1)
    masked_rows = (m == 0).nonzero().flatten()
    if masked_rows.numel():
        got_masked = planes[:, :, :, :n_res].reshape(2, c, -1)[:, :, masked_rows]
        assert (got_masked.view(torch.int16) == 0).all(), "a masked row must be EXACTLY +0.0"
    diff = (valid.double() - ref).abs()
    tol = 4e-3 + ref.abs() / 128.0
    bad = (diff > tol).sum().item()
    assert bad == 0, f"{bad} cells beyond the derived tolerance; max |diff| {diff.max().item():.4g} at |ref| max {ref.abs().max().item():.4g}"
    rel_rms = (diff.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    assert rel_rms < 2e-3, f"relative RMS {rel_rms:.3g}"
    # the statistics: valid rows, fp32 two-pass vs fp64
    torch.testing.assert_close(stats[:, 0].double(), mean64, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(stats[:, 1].double(), rstd64, rtol=1e-4, atol=1e-5)
    # two-launch trick: a first-launch race would show here
    planes2 = torch.empty_like(planes)
    stats2 = torch.empty_like(stats)
    _launch(plan, inp, c, planes2, stats2, stream)
    torch.cuda.synchronize()
    assert torch.equal(planes2, planes) and torch.equal(stats2, stats)


# The sm_107a renderings the triangle multiplicative update SHIPS (the API's parameter sets: biases, mask, no statistics; the default tile
# order / prologue form / ring depth) and their spill counts, MEASURED on the branch's toolchain (nvdisasm of the trace-time cubin): the
# register split the body requests is dropped by ptxas (no `.maxnreg` directive -> C7508), so every warp runs flat at the entry count (REG 168)
# and the LayerNorm row held in registers between its statistics passes spills from K = 256 up (K1 664 B / K3 872 B of stack).  A pin is a
# BOUND (frost_test_utils.assert_no_new_spills): more spills = a regression; fewer = update the pin with the measurement.  USETMAXREG 0
# documents the dropped split -- when THAT assertion fails the DSL started emitting `.maxnreg`: re-measure the roles against entry x warps
# and update these pins.  (The kernel-test renderings with want_stats=True spill about twice as much at K = 256: 468 / 470 and 903 / 910.)
_SASS_SPILL_PINS = {
    ("k1", 128, "registers", "scalar"): dict(STL=1, LDL=1),
    ("k1", 256, "registers", "scalar"): dict(STL=240, LDL=241),
    ("k3", 128, "registers", "scalar"): dict(STL=1, LDL=1),
    ("k3", 256, "registers", "scalar"): dict(STL=499, LDL=504),
    # The re-read LayerNorm form with ROLLED chunk loops (`ln_pass_form="smem_rolled"`) holds no row in registers and keeps its normalise pass a
    # real loop (cutlass.range(..., unroll=1)): the K = 256 residency spills are GONE on both arms (0 / 0 measured, REG 112-114 vs the flat 168),
    # with the packed epilogue on top.  The unrolled re-read form ("smem") only halves them (K1 118 / 118, K3 235 / 235): the normalise pass's
    # unrolled broadcast gamma / beta loads are hoisted by the compiler and the row re-appears as live registers -- which is why the rolled form exists.
    ("k1", 256, "smem_rolled", "packed"): dict(STL=0, LDL=0),
    ("k3", 256, "smem_rolled", "packed"): dict(STL=0, LDL=0),
    # The PIPELINED rolled forms (`smem_rolled_pf` = the next chunk's loads carried across the loop back-edge; `smem_rolled_tree` = + four partial
    # sums per statistics pass): +4 / +20 live words per lane in the statistics / normalise passes, no spill on either kernel at either width
    # (0 / 0 measured; the bias-off K3 K = 256 rendering, REG 80 / STACK 24 / STL 5 / LDL 7 on the shipped rolled form, renders at REG 106 / STACK 0
    # under them -- the extra live words move ptxas off the 80-register target it spilled for).
    ("k1", 128, "smem_rolled_pf", "packed"): dict(STL=0, LDL=0),
    ("k1", 256, "smem_rolled_pf", "packed"): dict(STL=0, LDL=0),
    ("k1", 256, "smem_rolled_tree", "packed"): dict(STL=0, LDL=0),
    ("k3", 256, "smem_rolled_pf", "packed"): dict(STL=0, LDL=0),
    ("k3", 256, "smem_rolled_tree", "packed"): dict(STL=0, LDL=0),
}
_SASS_COUNTS = {**sass_utils.SASS_OPCODE_COUNTS, "USETMAXREG": ("USETMAXREG",)}
_SASS_PROBE_BODY = """
from cudnn.gemm.frost import ln_prologue
from cudnn.triangle_multiplicative_update.kernels import trimul_proj as k1
from cudnn.triangle_multiplicative_update.kernels import trimul_tail as k3
arm, kk = params_kw["arm"], int(params_kw["k"])
lnf, epm = params_kw.get("ln_pass_form", "registers"), params_kw.get("epi_math", "scalar")
has_bias, hint = bool(params_kw.get("has_bias", True)), int(params_kw.get("min_blocks_per_mp", 0))
order = params_kw.get("pass_order", "interleaved")  # the K1 tile order of the rendering (the heuristics ship "major" from N = 512 at c = 128)
# the API's parameter sets (api.py compile(): biases on, the mask bound, no statistics, the heuristics' default knobs) at the pinned LN / epilogue forms
if arm == "k1":
    p = k1.proj_params(kk, kk, has_bias=has_bias, has_mask=True, want_stats=False, pass_order=order, clc_stages=2, ln_pass_form=lnf, epi_math=epm, min_blocks_per_mp=hint)
else:
    p = k3.tail_params(kk, kk, has_bias=has_bias, want_stats=False, ln_pass_form=lnf, epi_math=epm, min_blocks_per_mp=hint)
plan = ln_prologue.build_ln_dual_gemm(p)
print("EXPECT_K", kk)
"""

# The `.minnctapersm` hint (`min_blocks_per_mp`; `heuristics.register_split_hint`) on the renderings the hint exists for -- the bias-OFF forms, where
# ptxas's two-CTAs-per-SM target (65536 / (2 x 384) = 80 registers) is reached with spills; the bias-ON twin of each as the identity reference.
# Keyed (arm, K, ln_pass_form, epi_math, has_bias, min_blocks_per_mp); the values are BOUNDS measured on the branch's toolchain like the table above:
#   hint 0 (the rendering as shipped): the bias-off K3 K = 256 rolled rendering at REG 80 / STACK 24 spills 5 / 7 on the EPILOGUE's hot path (three
#     swizzled staging-store addresses + two row-decode constants reloaded per tile ahead of the TMEM loads: +7-8 % on the kernel vs the bias-on
#     twin, 69.8 k local requests in ncu); the bias-off K1 K = 128 rolled rendering 1 / 1 at STACK 8; USETMAXREG 0 (the split dropped, C7508).
#   hint 1 (.minnctapersm 1 -- the SAME cubin as a .maxnreg 168 build): ptxas honours the six setmaxnreg requests, USETMAXREG 6 (2 TRY_ALLOC 232 +
#     4 DEALLOC 24), REG = the 168 entry count, STACK 0 and NO spill on the K3 renderings (bias off and on: the identity, no new spill); the K1
#     INTERLEAVED rolled renderings carry 4 STL / 13 LDL of role-prologue values under the hint (stored at kernel entry / at the INCREASE warps'
#     entry, reloaded once per role prologue) -- the per-rendering sign the rule keys on the tile order for.
_SASS_SPLIT_HINT_PINS = {
    ("k3", 256, "smem_rolled", "packed", False, 0, "interleaved"): dict(STL=5, LDL=7, USETMAXREG=0),
    ("k3", 256, "smem_rolled", "packed", False, 1, "interleaved"): dict(STL=0, LDL=0, USETMAXREG=6),
    ("k3", 256, "smem_rolled", "packed", True, 1, "interleaved"): dict(STL=0, LDL=0, USETMAXREG=6),
    ("k1", 128, "smem_rolled", "packed", False, 0, "interleaved"): dict(STL=1, LDL=1, USETMAXREG=0),
    # the K1 K = 128 INTERLEAVED rolled rendering under the hint: the split honoured (USETMAXREG 6) but 4 / 13 role-prologue spills (STACK 16; the
    # interleaved pass decode + the scheduler's tile count stored at entry and reloaded once per role prologue) -- MEASURED on the branch's toolchain
    ("k1", 128, "smem_rolled", "packed", False, 1, "interleaved"): dict(STL=4, LDL=13, USETMAXREG=6),
    # the production renderings the rule hints (heuristics._REGISTER_SPLIT_HINTS; the `major` order the heuristics ship from N = 512 at c = 128):
    # the register form's 1 / 1 at STACK 8 and the rolled form's flat REG 80 both go to 0 / 0 at the honoured split
    ("k1", 128, "registers", "packed", False, 0, "major"): dict(STL=1, LDL=1, USETMAXREG=0),
    ("k1", 128, "registers", "packed", False, 1, "major"): dict(STL=0, LDL=0, USETMAXREG=6),
    ("k1", 128, "smem_rolled", "packed", False, 0, "major"): dict(STL=0, LDL=0, USETMAXREG=0),
    ("k1", 128, "smem_rolled", "packed", False, 1, "major"): dict(STL=0, LDL=0, USETMAXREG=6),
}


@pytest.mark.parametrize("arm,k,ln_pass_form,epi_math,has_bias,hint,order", sorted(_SASS_SPLIT_HINT_PINS))
def test_sm107_register_split_hint_renderings_spill_pins(tmp_path, arm, k, ln_pass_form, epi_math, has_bias, hint, order):
    """Trace-compile the bias-off renderings (and the bias-on identity twin) with and without the `.minnctapersm` hint for sm_107a (no GPU): the
    spill counts are bounds like the table above; USETMAXREG is EXACT -- 0 without the hint (the dropped split), 6 with it (the honoured split)."""
    probe = sass_utils.run_sass_probe(
        tmp_path,
        probe_src=sass_utils.sass_probe_source(_SASS_PROBE_BODY, _SASS_COUNTS),
        arch="sm_107a",
        params=dict(arm=arm, k=k, ln_pass_form=ln_pass_form, epi_math=epi_math, has_bias=has_bias, min_blocks_per_mp=hint, pass_order=order),
        tag=f"ln_dual_gemm_{arm}_k{k}_{ln_pass_form}_{epi_math}_bias{int(has_bias)}_hint{hint}_{order}",
    )
    st = probe.stats
    pins = _SASS_SPLIT_HINT_PINS[(arm, k, ln_pass_form, epi_math, has_bias, hint, order)]
    print(
        f"{arm} K={k} {ln_pass_form} {epi_math} bias {has_bias} hint {hint}: STL {st['STL']} LDL {st['LDL']} USETMAXREG {st['USETMAXREG']} lines {st['LINES']} cubin {probe.cubin_md5}"
    )
    sass_utils.assert_no_new_spills(st, pins, tag=f"{arm} K={k} {ln_pass_form} {epi_math} bias {has_bias} hint {hint}: ")
    assert st["USETMAXREG"] == pins["USETMAXREG"], (
        f"{arm} K={k} bias {has_bias} hint {hint}: USETMAXREG {st['USETMAXREG']} != {pins['USETMAXREG']} -- the register split is "
        f"{'honoured' if st['USETMAXREG'] else 'dropped'} where the hint says {'honoured' if pins['USETMAXREG'] else 'dropped'}; re-measure the roles and the pins"
    )


@pytest.mark.parametrize("arm,k,ln_pass_form,epi_math", sorted(_SASS_SPILL_PINS))
def test_sm107_rendering_spill_pins(tmp_path, arm, k, ln_pass_form, epi_math):
    """Trace-compile the rendering for sm_107a (no GPU needed), read STL / LDL / USETMAXREG off the SASS: the served widths' spill counts are
    pinned per arm and LayerNorm / epilogue form, the dropped register split is documented by USETMAXREG == 0."""
    probe = sass_utils.run_sass_probe(
        tmp_path,
        probe_src=sass_utils.sass_probe_source(_SASS_PROBE_BODY, _SASS_COUNTS),
        arch="sm_107a",
        params=dict(arm=arm, k=k, ln_pass_form=ln_pass_form, epi_math=epi_math),
        tag=f"ln_dual_gemm_{arm}_k{k}_{ln_pass_form}_{epi_math}",
    )
    st = probe.stats
    print(f"{arm} K={k} {ln_pass_form} {epi_math}: STL {st['STL']} LDL {st['LDL']} USETMAXREG {st['USETMAXREG']} lines {st['LINES']} cubin {probe.cubin_md5}")
    sass_utils.assert_no_new_spills(st, _SASS_SPILL_PINS[(arm, k, ln_pass_form, epi_math)], tag=f"{arm} K={k} {ln_pass_form} {epi_math}: ")
    assert (
        st["USETMAXREG"] == 0
    ), f"{arm} K={k}: the rendering now carries a REAL register split (USETMAXREG {st['USETMAXREG']}) -- re-measure the roles and the spill pins"


@pytest.mark.parametrize(
    "kw,axis",
    [
        (dict(ln_pass_form="rows"), "ln_pass_form"),
        (dict(epi_math="vector"), "epi_math"),
    ],
)
def test_ln_pass_form_and_epi_math_decline_typed(kw, axis):
    """The two substrate forms are a closed vocabulary (performance only); a value outside it is a typed ValueError naming the axis."""
    with pytest.raises(ValueError, match=axis):
        sub.validate_ln_dual_gemm_params(k1.proj_params(128, 128, has_bias=True, has_mask=True, **kw))


@requires_rubin
@pytest.mark.parametrize("n_res,batch,c_z,c,mask_kind", [(217, 2, 128, 128, "random"), (128, 1, 256, 256, "padded")])
@pytest.mark.parametrize(
    "ln_pass_form,epi_math", [("smem", "scalar"), ("smem_rolled", "scalar"), ("registers", "packed"), ("smem_rolled", "packed"), ("smem_rolled_pf", "packed")]
)
def test_ln_pass_form_and_epi_math_are_bitwise_equal_to_the_default_forms(n_res, batch, c_z, c, mask_kind, ln_pass_form, epi_math):
    """The re-read LayerNorm forms sum the same bf16 words in the same order (a re-load of the unchanged stage replaces a register), and the packed
    epilogue runs the scalar form's fp32 operations in its association on fp32x2 pairs: planes AND statistics BITWISE equal to the default forms."""
    inp = _inputs(n_res, batch, c_z, c, mask_kind, True, seed=13)
    outs = []
    for lnf, epm in (("registers", "scalar"), (ln_pass_form, epi_math)):
        plan = k1.build_trimul_proj(k1.proj_params(c_z, c, has_bias=True, has_mask=True, want_stats=True, ln_pass_form=lnf, epi_math=epm))
        planes = torch.full(k1.planes_shape(c, batch, n_res), _SENTINEL, device="cuda", dtype=torch.bfloat16)
        stats = torch.full((batch * n_res * n_res, 2), -1.0, device="cuda")
        _launch(plan, inp, c, planes, stats, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert not (planes[:, :, :, :n_res] == _SENTINEL).any(), f"{lnf}/{epm}: a never-written cell in the valid region"
        outs.append((planes, stats))
    assert torch.equal(outs[0][0], outs[1][0]), f"{ln_pass_form}/{epi_math} changed the planes"
    assert torch.equal(outs[0][1], outs[1][1]), f"{ln_pass_form}/{epi_math} changed the statistics"


@requires_rubin
@pytest.mark.parametrize("n_res,batch,c_z,c,mask_kind", [(217, 2, 128, 128, "random"), (128, 1, 256, 256, "padded")])
def test_tree_ln_form_meets_the_oracle_bound_and_its_distance_to_the_rolled_form_is_measured(n_res, batch, c_z, c, mask_kind):
    """`ln_pass_form="smem_rolled_tree"` sums each statistics pass in FOUR partials (a different fp32 ORDER, :data:`LN_PASS_FORMS_REORDERED`): the
    planes and the statistics meet the SAME fp64-oracle bound as every form (nothing widened), and its distance to the bitwise forms is REPORTED
    (cells differing, the statistics' max |diff|) -- it is a measured, named form, never a bitwise pin and never a silent default."""
    eps = 1e-5
    inp = _inputs(n_res, batch, c_z, c, mask_kind, True, seed=13)
    ref, mean64, rstd64 = _reference(inp, c, eps)
    outs = {}
    for lnf in ("smem_rolled", "smem_rolled_tree"):
        plan = k1.build_trimul_proj(k1.proj_params(c_z, c, has_bias=True, has_mask=True, want_stats=True, eps=eps, ln_pass_form=lnf, epi_math="packed"))
        planes = torch.full(k1.planes_shape(c, batch, n_res), _SENTINEL, device="cuda", dtype=torch.bfloat16)
        stats = torch.full((batch * n_res * n_res, 2), -1.0, device="cuda")
        _launch(plan, inp, c, planes, stats, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        valid = planes[:, :, :, :n_res].float()
        assert not (valid == _SENTINEL).any(), f"{lnf}: a never-written cell in the valid region"
        diff = (valid.double() - ref).abs()
        tol = 4e-3 + ref.abs() / 128.0  # the module's derived bound (test_k1_matches_the_fp64_oracle), unchanged
        bad = int((diff > tol).sum().item())
        rel_rms = (diff.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        print(f"{lnf} {n_res}:{batch}:{c_z}: {bad} cells beyond the derived tolerance; max |diff| {diff.max().item():.4g}; rel RMS {rel_rms:.3g}")
        assert bad == 0 and rel_rms < 2e-3, f"{lnf}: {bad} cells beyond the derived tolerance, rel RMS {rel_rms:.3g}"
        torch.testing.assert_close(stats[:, 0].double(), mean64, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(stats[:, 1].double(), rstd64, rtol=1e-4, atol=1e-5)
        outs[lnf] = (planes, stats)
    n_diff = int((outs["smem_rolled"][0].view(torch.int16) != outs["smem_rolled_tree"][0].view(torch.int16)).sum().item())
    d_stats = (outs["smem_rolled"][1] - outs["smem_rolled_tree"][1]).abs()
    print(
        f"tree vs rolled {n_res}:{batch}:{c_z}: {n_diff} plane cells differ (of {outs['smem_rolled'][0].numel()}); "
        f"max |d mean| {d_stats[:, 0].max().item():.3g}, max |d rstd| {d_stats[:, 1].max().item():.3g}"
    )
    assert sub.LN_PASS_FORMS_REORDERED == ("smem_rolled_tree",)  # the one form outside the bitwise pins, by name


@requires_rubin
def test_skip_vote_is_bitwise_equal_to_the_unskipped_arm():
    """A padded protein masks whole 256-row tiles (every pair row with i >= L): the skipped-MMA arm and the control arm that
    computes everything must agree BITWISE on the planes and the statistics."""
    n_res, batch, c_z, c = 256, 2, 128, 128
    inp = _inputs(n_res, batch, c_z, c, "padded", True, seed=3)
    outs = []
    for skip in (True, False):
        plan = k1.build_trimul_proj(k1.proj_params(c_z, c, has_bias=True, has_mask=True, want_stats=True, skip_masked_tiles=skip))
        planes = torch.full(k1.planes_shape(c, batch, n_res), _SENTINEL, device="cuda", dtype=torch.bfloat16)
        stats = torch.full((batch * n_res * n_res, 2), -1.0, device="cuda")
        _launch(plan, inp, c, planes, stats, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        outs.append((planes, stats))
    assert torch.equal(outs[0][0], outs[1][0]), "the warp-vote skip changed the output"
    assert torch.equal(outs[0][1], outs[1][1]), "the warp-vote skip changed the statistics"
    m = inp["mask"].reshape(-1)
    assert (m == 0).sum() > 0 and (outs[0][0][:, :, :, :n_res].reshape(2, c, -1)[:, :, m == 0].view(torch.int16) == 0).all()


@requires_rubin
def test_mask_multiplies_in_fp32_before_the_bf16_rounding():
    """The pin that separates the kernel's mask semantics (``bf16(sigmoid(g) * v * m)``, one rounding) from cuEquivariance's
    after-rounding form (``bf16(bf16(sigmoid(g) * v) * m)``, two roundings): with FRACTIONAL mask weights the two fp64 models
    differ by one bf16 ulp on a sizeable fraction of the cells, and the kernel must side with the before-rounding model there.
    (``{0, 1}`` and power-of-two masks cannot tell the two apart -- which is why every other cell passes under both.)"""
    n_res, batch, c_z, c = 128, 1, 128, 128
    eps = 1e-5
    inp = _inputs(n_res, batch, c_z, c, "fractional", True, seed=11)
    plan = k1.build_trimul_proj(k1.proj_params(c_z, c, has_bias=True, has_mask=True, want_stats=False, eps=eps))
    planes = torch.full(k1.planes_shape(c, batch, n_res), _SENTINEL, device="cuda", dtype=torch.bfloat16)
    _launch(plan, inp, c, planes, None, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    ref, _, _ = _reference(inp, c, eps)  # fp64 sigmoid(g) * v * m, unrounded
    m = inp["mask"].double().reshape(-1)
    live = m != 0
    ref_rows = ref.reshape(2, c, -1)[:, :, live]  # [2, c, live rows]: fp64 sigmoid(g) * v * m
    sv = ref_rows / m[live][None, None, :]  # fp64 sigmoid(g) * v on the live rows
    pre = ref_rows.to(torch.bfloat16).view(torch.int16)  # ONE rounding after the mask
    post = (sv.to(torch.bfloat16).double() * m[live][None, None, :]).to(torch.bfloat16).view(torch.int16)  # round, mask, round
    got = planes[:, :, :, :n_res].reshape(2, c, -1)[:, :, live].view(torch.int16)
    distinguishing = pre != post
    frac_dist = distinguishing.double().mean().item()
    match_pre = (got == pre)[distinguishing].double().mean().item()
    match_post = (got == post)[distinguishing].double().mean().item()
    msg = f"distinguishing cells {100 * frac_dist:.1f} %: kernel == before-rounding model on {100 * match_pre:.1f} %, == after-rounding model on {100 * match_post:.1f} %"
    print(msg)  # the measured fractions (pytest -rA shows them for a passing test)
    assert frac_dist > 0.05, msg  # the fractional mask must actually separate the two models
    # measured on the dev node 2026-10-07 (N = 128, seed 11): 25.8 % distinguishing cells, before-rounding 99.9 %, after-rounding 0.1 %
    assert match_pre > 0.9 and match_post < 0.1, msg


@requires_rubin
@pytest.mark.parametrize("n_res,batch,c_z,c,mask_kind", [(217, 2, 128, 128, "random"), (128, 1, 256, 256, "padded")])
def test_interleaved_pass_order_is_bitwise_equal_to_pass_major(n_res, batch, c_z, c, mask_kind):
    """``pass_order="interleaved"`` (the pass folded into the scheduler's M axis so both passes of a row block are adjacent -- the
    shared-z tile order) computes the same tiles with the same arithmetic: planes and statistics BITWISE equal to ``"major"``."""
    inp = _inputs(n_res, batch, c_z, c, mask_kind, True, seed=7)
    outs = []
    for order in ("major", "interleaved"):
        plan = k1.build_trimul_proj(k1.proj_params(c_z, c, has_bias=True, has_mask=True, want_stats=True, pass_order=order))
        planes = torch.full(k1.planes_shape(c, batch, n_res), _SENTINEL, device="cuda", dtype=torch.bfloat16)
        stats = torch.full((batch * n_res * n_res, 2), -1.0, device="cuda")
        _launch(plan, inp, c, planes, stats, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert not (planes[:, :, :, :n_res] == _SENTINEL).any(), f"{order}: a never-written cell in the valid region"
        outs.append((planes, stats))
    assert torch.equal(outs[0][0], outs[1][0]), "the interleaved tile order changed the planes"
    assert torch.equal(outs[0][1], outs[1][1]), "the interleaved tile order changed the statistics"


@requires_rubin
def test_no_mask_artifact_matches_the_all_valid_mask():
    """``has_mask=False`` (no mask tensor at the ABI) equals the all-valid mask run, cell for cell."""
    n_res, batch, c_z, c = 128, 1, 128, 128
    inp = _inputs(n_res, batch, c_z, c, "all", True, seed=5)
    stream = torch.cuda.current_stream().cuda_stream
    plan_m = k1.build_trimul_proj(k1.proj_params(c_z, c, has_bias=True, has_mask=True, want_stats=False))
    plan_n = k1.build_trimul_proj(k1.proj_params(c_z, c, has_bias=True, has_mask=False, want_stats=False))
    out_m = torch.full(k1.planes_shape(c, batch, n_res), _SENTINEL, device="cuda", dtype=torch.bfloat16)
    out_n = torch.empty_like(out_m)
    w_stack = k1.stack_projection_weights(inp["p_in"], inp["g_in"])
    b_stack = k1.stack_projection_biases(inp["p_b"], inp["g_b"])
    k1.run_trimul_proj(plan_m, inp["z"], w_stack, out_m, inp["ln_w"], inp["ln_b"], bias_stacked=b_stack, mask=inp["mask"], stats=None, stream=stream)
    k1.run_trimul_proj(plan_n, inp["z"], w_stack, out_n, inp["ln_w"], inp["ln_b"], bias_stacked=b_stack, mask=None, stats=None, stream=stream)
    torch.cuda.synchronize()
    assert torch.equal(out_m, out_n)


@requires_rubin
def test_runner_declines_a_plan_for_another_k_and_a_misshaped_mask():
    plan = k1.build_trimul_proj(k1.proj_params(128, 128, has_bias=False, has_mask=True, want_stats=False))
    z = torch.zeros(1, 16, 16, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(2, 256, 256, device="cuda", dtype=torch.bfloat16)
    planes = torch.zeros(k1.planes_shape(128, 1, 16), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="compiled for K=128"):
        k1.run_trimul_proj(
            plan,
            z,
            w,
            planes,
            torch.zeros(256, device="cuda", dtype=torch.bfloat16),
            torch.zeros(256, device="cuda", dtype=torch.bfloat16),
            mask=torch.ones(1, 16, 16, device="cuda", dtype=torch.bfloat16),
            stream=0,
        )
    z = torch.zeros(1, 16, 16, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="mask must be"):
        k1.run_trimul_proj(
            plan,
            z,
            w,
            planes,
            torch.zeros(128, device="cuda", dtype=torch.bfloat16),
            torch.zeros(128, device="cuda", dtype=torch.bfloat16),
            mask=torch.ones(1, 16, 15, device="cuda", dtype=torch.bfloat16),
            stream=0,
        )


@requires_rubin
@pytest.mark.parametrize("n_res,batch,k_block", [(384, 2, 128), (384, 1, 64), (217, 1, 32), (128, 2, 256)])
def test_k1_row_tile_knob_is_bitwise_the_rule(n_res, batch, k_block):
    """The row tile (k's per 256-row pair tile) is a RUNTIME operand of the compiled kernel: the same plan launched with ``k_block`` computes the
    SAME bits as the rule (planes AND statistics) -- a row's LayerNorm, MMA K order and epilogue do not depend on the tile it sits in.  At N = 384
    the rule pads 25 % of every pair tile (kb = 256 over 384 k); kb = 128 / 64 pad none (fewer scheduler tiles, the same per-CTA TMA box at 128)."""
    c_z = c = 128
    params = k1.proj_params(c_z, c, has_bias=True, has_mask=True, want_stats=True, eps=1e-5)
    plan = k1.build_trimul_proj(params)
    inp = _inputs(n_res, batch, c_z, c, "random", True, seed=31)
    stream = torch.cuda.current_stream().cuda_stream
    outs = []
    for kb in (None, k_block):
        planes = torch.full(k1.planes_shape(c, batch, n_res), _SENTINEL, device="cuda", dtype=torch.bfloat16)
        stats = torch.full((batch * n_res * n_res, 2), -1.0, device="cuda", dtype=torch.float32)
        g = _launch(plan, inp, c, planes, stats, stream, k_block=kb)
        torch.cuda.synchronize()
        assert g.kb == (sub.default_k_block(n_res) if kb is None else kb) and g.rows_valid == batch * n_res * n_res
        assert not (planes[:, :, :, :n_res].float() == _SENTINEL).any() and (stats[:, 1] > 0).all()
        outs.append((planes, stats, g))
    (p0, s0, g0), (p1, s1, g1) = outs
    assert torch.equal(p0, p1) and torch.equal(s0, s1), f"the row tile changed the output bits (kb {g0.kb} -> {g1.kb})"
    if n_res == 384:
        assert g0.rows_computed == 4 * g0.rows_valid // 3 and g1.rows_computed == g1.rows_valid  # the 25 % the lever removes
