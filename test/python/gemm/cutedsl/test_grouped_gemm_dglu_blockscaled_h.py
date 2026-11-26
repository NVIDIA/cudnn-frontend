# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The MXFP8 dGLU kernel (SM100) writing the forward activation h.

With ``h_tensor`` / ``sfh_col_tensor`` the dSwiGLU epilogue also writes h = prob * silu(beta * gate) * (beta * up),
the FC2 input it recomputes from the saved pre-activation C, column-quantized for the FC2 weight gradient: E4M3 codes
[rows, N] and one E8M0 scale per 32-row block of a column, per expert in the block-scaled MMA scale-factor layout of
an N x rows operand.

- dH (row and column E4M3 with both SFD planes) is byte-equal with h on and off; the deterministic dprob (an FP32 sum)
  agrees to FP32 rounding.
- h is byte-equal to a reference quantization of the activation where the activation is exact in float32 (gates of 0
  or >= 17 after beta: the sigmoid is exactly 1/2 or 1), and within quantization error elsewhere.
- On the dynamic scheduler (2-CTA and 1-CTA tiles, scalar epilogue math) and the static scheduler.
"""

import pytest
import torch

from gemm.cutedsl.test_grouped_gemm_glu_blockscaled_optional_c import ROUTINGS, launch, levels, make_kernel, valid_rows
from gemm.cutedsl.test_grouped_gemm_glu_blockscaled_optional_c import make_problem as make_mxfp8_problem
from gemm.cutedsl.test_grouped_gemm_glu_blockscaled_optional_c import require_sm100  # noqa: F401  (autouse fixture)

#: Poison for every output byte: an output the kernel should write but does not stays 0xA5.
POISON = 0xA5
#: dGLU N (W2 rows, the width of h) and K.
N, K = 512, 512


def make_problem(routing, seed, exact):
    """The dGLU problem; ``exact``: gates of 0 or 40..60 (beta >= 0.5), so silu(beta * gate) is exact in float32."""
    problem = make_mxfp8_problem(ROUTINGS[routing], n=N, k=K, seed=seed, dglu=True)
    if exact:
        g = torch.Generator(device="cuda").manual_seed(seed + 7)
        c = problem["c"].view(problem["rows"], N // 32, 2, 32)
        gate = torch.rand(c[:, :, 0].shape, generator=g, device="cuda") * 20 + 40
        gate[torch.rand(gate.shape, generator=g, device="cuda") < 0.25] = 0.0
        c[:, :, 0] = gate.to(torch.bfloat16)
    return problem


def run_dglu(problem, h, **options):
    """One launch with E4M3 dH and both SFD planes (and h when ``h``); returns the outputs by name over all rows."""
    import cutlass
    from cutlass.cute.nvgpu import OperandMajorMode
    from cutlass.cute.runtime import from_dlpack

    one_cta = options.pop("one_cta", False)
    kernel = make_kernel("dglu", len(problem["offsets"]), one_cta=one_cta, discrete_col_sfd=True, **options)
    rows, n = problem["rows"], problem["n"]

    def fill(count):
        return torch.full((count,), POISON, device="cuda", dtype=torch.uint8)

    out = dict(d=fill(rows * 2 * n), d_col=fill(rows * 2 * n), sfd_row=fill(rows * 2 * n // 32), sfd_col=fill(rows * 2 * n // 32))
    if h:
        out.update(h=fill(rows * n), sfh=fill(rows * n // 32))
    out["dprob"] = torch.zeros(rows, n // 256, device="cuda")
    workspace = torch.zeros(max(256, kernel.get_workspace_bytes()), device="cuda", dtype=torch.uint8)

    def t(x):
        return from_dlpack(x, assumed_align=16)

    def e8(x):
        return t(x.view(torch.float8_e8m0fnu))

    def codes(x, cols):
        return t(x.view(rows, cols).view(torch.float8_e4m3fn))

    args = [
        t(problem["a"]),
        t(problem["b"].permute(1, 2, 0)),
        e8(problem["sfb"]),
        cutlass.Int32(n),
        cutlass.Int32(problem["k"]),
        cutlass.Int64(problem["k"]),
        OperandMajorMode.K,
        from_dlpack(workspace, assumed_align=128).iterator,
        # (c, d, d_col, sfa, sfd_row, sfd_col, amax, norm_const, padded_offsets, alpha, beta, prob, dprob, dbias)
        t(problem["c"]),
        codes(out["d"], 2 * n),
        codes(out["d_col"], 2 * n),
        e8(problem["sfa"]),
        e8(out["sfd_row"]),
        e8(out["sfd_col"]),
        None,
        t(torch.ones(1, device="cuda")),
        from_dlpack(problem["offsets"], assumed_align=4),
        t(problem["alpha"]),
        t(problem["beta"]),
        t(problem["prob"]),
        t(out["dprob"].unsqueeze(-1)),
        None,
    ]
    kwargs = dict(h_tensor=codes(out["h"], n), sfh_col_tensor=e8(out["sfh"])) if h else {}
    launch(kernel, args, (kernel.test_compile_key, h, rows, n), **kwargs)
    return out


def h_reference(problem):
    """float32 h = prob * ((beta * gate) * sigmoid(beta * gate)) * (beta * up) with the kernel's multiplication order."""
    rows, n = problem["rows"], problem["n"]
    c = problem["c"].view(rows, n // 32, 2, 32).float()
    expert = torch.searchsorted(problem["offsets"].long(), torch.arange(rows, device="cuda"), right=True).clamp(max=len(problem["beta"]) - 1)
    beta = problem["beta"][expert].view(rows, 1, 1)
    gate, up = c[:, :, 0] * beta, c[:, :, 1] * beta
    swish = gate * torch.sigmoid(gate)
    return ((up * swish) * problem["prob"].view(rows, 1, 1)).reshape(rows, n)


def quantize_h(h, offsets):
    """The reference column quantization: per 32-row block of a column, scale = 2^ceil(log2(amax * (1/448)))
    (E8M0, round up), codes = E4M3(h / scale); scales in the per-expert MMA scale-factor layout."""
    rows, n = h.shape
    blocks = h.view(rows // 32, 32, n)
    amax = blocks.abs().amax(dim=1)
    v = amax * torch.tensor(1 / 448.0, dtype=torch.float32, device="cuda")
    mant, exp = torch.frexp(v)
    # v = mant * 2^exp with mant in [0.5, 1): round up to a power of two unless v is one.
    e = torch.where(mant == 0.5, exp - 1, exp).clamp(-127, 127)
    codes = (blocks * torch.pow(2.0, -e.float()).view(rows // 32, 1, n)).view(rows, n).to(torch.float8_e4m3fn)
    scale_bytes = (e + 127).to(torch.uint8)  # [rows / 32, n]
    sf = torch.empty(rows * n // 32, dtype=torch.uint8, device="cuda")
    begin = 0
    for end in offsets.tolist():
        if end > begin:
            count = end - begin
            rb = torch.arange(count // 32, device="cuda").view(-1, 1)
            col = torch.arange(n, device="cuda").view(1, -1)
            index = begin * n // 32 + (col // 128) * 4 * count + (rb // 4) * 512 + (col % 32) * 16 + ((col // 32) % 4) * 4 + rb % 4
            sf[index.reshape(-1)] = scale_bytes[begin // 32 : end // 32].reshape(-1)
        begin = end
    return codes.view(torch.uint8), sf


def dequantize_h(codes, sf, offsets, n):
    """Inverse of quantize_h's layout: float32 h from the kernel's codes and scales."""
    rows = codes.shape[0]
    scale = torch.empty(rows // 32, n, device="cuda")
    begin = 0
    for end in offsets.tolist():
        if end > begin:
            count = end - begin
            rb = torch.arange(count // 32, device="cuda").view(-1, 1)
            col = torch.arange(n, device="cuda").view(1, -1)
            index = begin * n // 32 + (col // 128) * 4 * count + (rb // 4) * 512 + (col % 32) * 16 + ((col // 32) % 4) * 4 + rb % 4
            scale[begin // 32 : end // 32] = torch.pow(2.0, sf[index].float() - 127)
        begin = end
    return codes.view(torch.float8_e4m3fn).float() * scale.repeat_interleave(32, dim=0)


#: Scheduler variants: the dynamic scheduler (2-CTA, 1-CTA, scalar epilogue math) and the static scheduler.
VARIANTS = {
    "dynamic": dict(use_dynamic_sched=True),
    "dynamic_1cta": dict(use_dynamic_sched=True, one_cta=True),
    "dynamic_scalar": dict(use_dynamic_sched=True, vectorized_f32=False),
    "static": dict(use_dynamic_sched=False),
}
CASES = [(variant, routing) for variant in VARIANTS for routing in sorted(ROUTINGS)]
L0 = {
    ("dynamic", "uneven"),
    ("dynamic", "small_experts_64"),
    ("dynamic_1cta", "uneven"),
    ("dynamic_scalar", "big_expert"),
    ("static", "many_experts_64"),
}


@pytest.mark.parametrize("variant, routing", levels(CASES, L0, lambda variant, routing: f"{variant}-{routing}"))
def test_dglu_h_matches_reference_and_leaves_dh_unchanged(variant, routing):
    problem = make_problem(routing, seed=11, exact=True)
    want = run_dglu(problem, h=False, **VARIANTS[variant])
    got = run_dglu(problem, h=True, **VARIANTS[variant])
    rows = valid_rows(problem)
    # ptxas fuses some of the epilogue's packed FP32 multiply-adds differently once the h code is live, so dprob
    # (an FP32 sum) may move in its last bits; it is not an h-dependent value.
    dprob_got, dprob_want = got.pop("dprob"), want.pop("dprob")
    torch.testing.assert_close(dprob_got, dprob_want, rtol=2.0**-20, atol=2.0**-20 * dprob_want.abs().max().item())
    for name in want:
        a, b = got[name].view(torch.uint8), want[name].view(torch.uint8)
        if name in ("d", "d_col"):
            a, b = a.view(problem["rows"], -1)[:rows], b.view(problem["rows"], -1)[:rows]
        assert torch.equal(a, b), f"{variant} {routing}: {name} changes with h on"
    codes, sf = quantize_h(h_reference(problem), problem["offsets"])
    h_codes = got["h"].view(problem["rows"], N)
    assert torch.equal(h_codes[:rows], codes[:rows]), f"{variant} {routing}: h codes differ from the reference"
    assert torch.equal(got["sfh"][: rows * N // 32], sf[: rows * N // 32]), f"{variant} {routing}: h scales differ from the reference"


@pytest.mark.L0
@pytest.mark.parametrize("variant", ["dynamic", "static"])
def test_dglu_h_general_inputs_within_quantization_error(variant):
    """Random gates (sigmoid by the kernel's approximate exp2 / reciprocal): every h scale and code is written, and the
    dequantized h is within E4M3 rounding of the float reference."""
    problem = make_problem("uneven", seed=13, exact=False)
    got = run_dglu(problem, h=True, **VARIANTS[variant])
    rows = valid_rows(problem)
    codes = got["h"].view(problem["rows"], N)[:rows]
    deq = dequantize_h(codes, got["sfh"], problem["offsets"], N)
    ref = h_reference(problem)[:rows]
    _, want_sf = quantize_h(h_reference(problem), problem["offsets"])
    sf_ok = (got["sfh"][: rows * N // 32] == want_sf[: rows * N // 32]).float().mean().item()
    assert sf_ok > 0.99, f"{variant}: only {sf_ok:.4f} of the h scales match the reference"
    scale = deq.abs().view(rows // 32, 32, N).amax(dim=1).repeat_interleave(32, dim=0)
    assert ((deq - ref).abs() <= 2.0**-3 * ref.abs() + scale * 2.0**-9).all(), f"{variant}: h beyond E4M3 rounding of the reference"


# ---- GroupedGemmDgluSm100 and grouped_gemm_dglu_wrapper_sm100 ----
#
# The API with ``sample_h`` / ``sample_sfh_col`` and the wrapper with ``generate_h=True`` must write the bytes the
# direct kernel call above writes (h, its scales, dH and both SFD planes); without them, the bytes of the direct call
# without h, the seven default wrapper results and the plan cache key of before.

#: The direct runs' configuration (run_dglu on the dynamic scheduler) in API terms.
API_CONFIG = dict(
    acc_dtype=torch.float32,
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=(2, 1),
    sf_vec_size=32,
    vector_f32=True,
    discrete_col_sfd=True,
    use_dynamic_sched=True,
)
#: Executions of one compiled plan, alternating between two problems of the same shape.
API_LAUNCHES = 4
CHECKED = ("d", "d_col", "sfd_row", "sfd_col")


def api_operands(problem, h):
    """The problem's operands in their canonical forms (2-D A, C, D; (experts, n, k) B; flat scale buffers), outputs
    poisoned and a zeroed dprob slot buffer; h and its scales when ``h``."""
    rows, n = problem["rows"], problem["n"]

    def fill(shape, dtype):
        return torch.full(shape, POISON, device="cuda", dtype=torch.uint8).view(dtype)

    e8m0 = torch.float8_e8m0fnu
    ops = dict(
        a=problem["a"],
        b=problem["b"],
        c=problem["c"],
        sfa=problem["sfa"].view(e8m0),
        sfb=problem["sfb"].view(e8m0),
        offsets=problem["offsets"].clone(),
        alpha=problem["alpha"],
        beta=problem["beta"],
        prob=problem["prob"],
        norm_const=torch.ones(1, device="cuda"),
        d=fill((rows, 2 * n), torch.float8_e4m3fn),
        d_col=fill((rows, 2 * n), torch.float8_e4m3fn),
        sfd_row=fill((rows * 2 * n // 32,), e8m0),
        sfd_col=fill((rows * 2 * n // 32,), e8m0),
        dprob=torch.zeros(rows, n // 256, 1, device="cuda"),
    )
    if h:
        ops.update(h=fill((rows, n), torch.float8_e4m3fn), sfh=fill((rows * n // 32,), e8m0))
    experts = problem["b"].shape[0]
    ops["b_ptrs"] = torch.tensor([problem["b"][e].data_ptr() for e in range(experts)], dtype=torch.int64, device="cuda")
    ops["sfb_ptrs"] = torch.tensor([x.data_ptr() for x in ops["sfb"].view(experts, -1)], dtype=torch.int64, device="cuda")
    return ops


def make_api(ops, discrete=False, **overrides):
    from cudnn import GroupedGemmDgluSm100

    kwargs = dict(
        sample_a=ops["a"],
        sample_c=ops["c"],
        sample_d_row=ops["d"],
        sample_d_col=ops["d_col"],
        sample_sfa=ops["sfa"],
        sample_padded_offsets=ops["offsets"],
        sample_alpha=ops["alpha"],
        sample_beta=ops["beta"],
        sample_prob=ops["prob"],
        sample_dprob=ops["dprob"],
        sample_sfd_row=ops["sfd_row"],
        sample_sfd_col=ops["sfd_col"],
        sample_norm_const=ops["norm_const"],
        sample_h=ops.get("h"),
        sample_sfh_col=ops.get("sfh"),
        _deterministic=True,
        **API_CONFIG,
    )
    if discrete:
        experts, n, k = ops["b"].shape
        kwargs.update(num_experts=experts, b_shape=(n, k), b_dtype=ops["b"].dtype)
    else:
        kwargs.update(sample_b=ops["b"], sample_sfb=ops["sfb"])
    kwargs.update(overrides)
    return GroupedGemmDgluSm100(**kwargs)


def execute(api, ops, workspace, discrete=False, **overrides):
    weights = dict(b_ptrs=ops["b_ptrs"], sfb_ptrs=ops["sfb_ptrs"]) if discrete else dict(b_tensor=ops["b"], sfb_tensor=ops["sfb"])
    kwargs = dict(
        a_tensor=ops["a"],
        c_tensor=ops["c"],
        d_row_tensor=ops["d"],
        d_col_tensor=ops["d_col"],
        sfa_tensor=ops["sfa"],
        padded_offsets=ops["offsets"],
        alpha_tensor=ops["alpha"],
        beta_tensor=ops["beta"],
        prob_tensor=ops["prob"],
        dprob_tensor=ops["dprob"],
        sfd_row_tensor=ops["sfd_row"],
        sfd_col_tensor=ops["sfd_col"],
        norm_const_tensor=ops["norm_const"],
        workspace=workspace,
        h_tensor=ops.get("h"),
        sfh_col_tensor=ops.get("sfh"),
        **weights,
    )
    kwargs.update(overrides)
    api.execute(**kwargs)
    torch.cuda.synchronize()


def assert_same_as_kernel(got, want, names, label):
    for name in names:
        differing = (got[name].view(torch.uint8).reshape(-1) != want[name].view(torch.uint8).reshape(-1)).sum().item()
        assert differing == 0, f"{label}: {differing} bytes of {name} differ from the direct kernel call"


def assert_dprob_close(got, want_slots):
    want = want_slots.sum(dim=1) if got.numel() != want_slots.numel() else want_slots
    torch.testing.assert_close(got.view(want.shape), want, rtol=2.0**-20, atol=2.0**-20 * want.abs().max().item())


@pytest.mark.L0
@pytest.mark.parametrize("mode", ["dense", "dense_static", "discrete"])
def test_dglu_api_h_matches_kernel(mode, require_sm100, monkeypatch):
    """GroupedGemmDgluSm100 with and without h, each plan executed API_LAUNCHES times over two problems: the direct
    kernel call's bytes every time. ``dense_static``: dense weights without the dynamic N/K/L compile."""
    if mode == "dense_static":
        monkeypatch.setenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "0")
    discrete = mode == "discrete"
    problems = [make_problem("uneven", seed=seed, exact=False) for seed in (21, 22)]
    for h in (True, False):
        wants = [run_dglu(problem, h=h, use_dynamic_sched=True) for problem in problems]
        names = CHECKED + (("h", "sfh") if h else ())
        api = make_api(api_operands(problems[0], h), discrete=discrete)
        assert api.check_support()
        api.compile()
        workspace = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")
        for launch in range(API_LAUNCHES):
            ops = api_operands(problems[launch % 2], h)
            execute(api, ops, workspace, discrete=discrete)
            assert_same_as_kernel(ops, wants[launch % 2], names, f"{mode} h={h} launch {launch}")
            assert_dprob_close(ops["dprob"], wants[launch % 2]["dprob"])


DEFAULT_KEYS = ["d_row_tensor", "d_col_tensor", "dprob_tensor", "dbias_tensor", "amax_tensor", "sfd_row_tensor", "sfd_col_tensor"]


@pytest.mark.L0
def test_dglu_wrapper_generate_h(require_sm100):
    """generate_h=False: the seven default results with the direct call's bytes; generate_h=True: h_tensor and
    sfh_col_tensor appended, the direct call's bytes, a plan keyed as the default plan plus "h", reused by a call with
    other data."""
    from cudnn import grouped_gemm_dglu_wrapper_sm100
    from cudnn.gemm.cutedsl.grouped.dglu import api as dglu_api

    def call(problem, generate_h):
        ops = api_operands(problem, h=False)
        dprob = torch.zeros(problem["rows"], 1, 1, device="cuda")
        out = grouped_gemm_dglu_wrapper_sm100(
            ops["a"],
            ops["c"],
            ops["sfa"],
            ops["offsets"],
            ops["alpha"],
            ops["beta"],
            ops["prob"],
            dprob,
            b_tensor=ops["b"],
            sfb_tensor=ops["sfb"],
            norm_const_tensor=ops["norm_const"],
            d_dtype=torch.float8_e4m3fn,
            deterministic=True,
            **API_CONFIG,
            **({"generate_h": True} if generate_h else {}),
        )
        torch.cuda.synchronize()
        return out

    def renamed(out):
        got = dict(d=out["d_row_tensor"], d_col=out["d_col_tensor"], sfd_row=out["sfd_row_tensor"], sfd_col=out["sfd_col_tensor"])
        if "h_tensor" in out:
            got.update(h=out["h_tensor"], sfh=out["sfh_col_tensor"])
        return got

    problems = [make_problem("uneven", seed=seed, exact=False) for seed in (23, 24)]
    cache = dglu_api._cache_of_GroupedGemmDgluSm100Objects
    before = set(cache)
    out = call(problems[0], generate_h=False)
    assert list(out.keys()) == DEFAULT_KEYS
    want = run_dglu(problems[0], h=False, use_dynamic_sched=True)
    assert_same_as_kernel(renamed(out), want, CHECKED, "generate_h=False")
    assert_dprob_close(out["dprob_tensor"], want["dprob"])
    # The default plan: new, or already compiled by an earlier caller of the same shapes.
    default_keys = set(cache) - before or {key for key in cache if key[-1] != "h"}

    for launch, problem in enumerate(problems):
        out = call(problem, generate_h=True)
        assert list(out.keys()) == DEFAULT_KEYS + ["h_tensor", "sfh_col_tensor"]
        rows, n = problem["rows"], problem["n"]
        assert out["h_tensor"].shape == (rows, n) and out["h_tensor"].dtype == torch.float8_e4m3fn
        assert out["sfh_col_tensor"].shape == (rows * n // 32,) and out["sfh_col_tensor"].dtype == torch.float8_e8m0fnu
        want = run_dglu(problem, h=True, use_dynamic_sched=True)
        assert_same_as_kernel(renamed(out), want, CHECKED + ("h", "sfh"), f"generate_h=True call {launch}")
        assert_dprob_close(out["dprob_tensor"], want["dprob"])
        if launch == 0:
            h_keys = set(cache) - before - default_keys
            assert len(h_keys) == 1, "generate_h=True compiles one plan"
            (h_key,) = h_keys
            assert h_key[-1] == "h" and h_key[:-1] in default_keys, "the h plan's key is the default plan's key plus 'h'"
            after_first = set(cache)
        else:
            assert set(cache) == after_first, "the second generate_h=True call reuses the plan"


@pytest.mark.L0
def test_dglu_api_h_rejections(require_sm100, monkeypatch):
    """Invalid h requests raise ValueError at check_support (or at execute for a presence mismatch), before any
    compile."""
    from cudnn import GroupedGemmDgluSm100, grouped_gemm_dglu_wrapper_sm100

    problem = make_problem("uneven", seed=25, exact=False)
    ops = api_operands(problem, h=True)
    rows, n = problem["rows"], problem["n"]

    def rejected(match, **overrides):
        with pytest.raises(ValueError, match=match):
            make_api(ops, **overrides).check_support()

    rejected("together", sample_sfh_col=None)
    rejected("together", sample_h=None)
    rejected("dswiglu", act_func="dgeglu")
    rejected("interleaved_32", glu_layout="gate_then_up")
    rejected("h dtype", sample_h=ops["h"].view(torch.float8_e5m2))
    rejected("h", sample_h=ops["h"][:, : n // 2])
    rejected("SFH_col", sample_sfh_col=ops["sfh"][: rows * n // 64])
    rejected("SFH_col dtype", sample_sfh_col=ops["sfh"].view(torch.uint8))

    # The BF16 backend has no h output.
    bf16 = {name: ops[name].to(torch.bfloat16) for name in ("a", "b")}
    with pytest.raises(ValueError, match="sample_h"):
        GroupedGemmDgluSm100(
            sample_a=bf16["a"],
            sample_c=ops["c"],
            sample_d_row=ops["c"],
            sample_d_col=None,
            sample_sfa=None,
            sample_padded_offsets=ops["offsets"],
            sample_alpha=ops["alpha"],
            sample_beta=ops["beta"],
            sample_prob=ops["prob"],
            sample_dprob=torch.zeros(rows, 1, 1, device="cuda"),
            sample_b=bf16["b"],
            sample_h=ops["h"],
            sample_sfh_col=ops["sfh"],
        ).check_support()

    # The h tensors at execute must match the plan.
    api = make_api(ops)
    assert api.check_support()
    with pytest.raises(ValueError, match="presence"):
        execute(api, ops, None, h_tensor=None, sfh_col_tensor=None)
    api = make_api(ops, sample_h=None, sample_sfh_col=None)
    assert api.check_support()
    with pytest.raises(ValueError, match="presence"):
        execute(api, ops, None)

    # The wrapper: an h request the kernel cannot serve.
    with pytest.raises(ValueError, match="dswiglu"):
        grouped_gemm_dglu_wrapper_sm100(
            ops["a"],
            ops["c"],
            ops["sfa"],
            ops["offsets"],
            ops["alpha"],
            ops["beta"],
            ops["prob"],
            torch.zeros(rows, 1, 1, device="cuda"),
            b_tensor=ops["b"],
            sfb_tensor=ops["sfb"],
            norm_const_tensor=ops["norm_const"],
            d_dtype=torch.float8_e4m3fn,
            sf_vec_size=32,
            act_func="dgeglu",
            generate_h=True,
        )
    with pytest.raises(ValueError, match="generate_h"):
        grouped_gemm_dglu_wrapper_sm100(
            bf16["a"],
            ops["c"],
            None,
            ops["offsets"],
            ops["alpha"],
            ops["beta"],
            ops["prob"],
            torch.zeros(rows, 1, 1, device="cuda"),
            b_tensor=bf16["b"],
            generate_h=True,
        )

    # SM107 (Rubin): the h output is SM100/103 only.
    import cudnn.api_base

    monkeypatch.setattr(cudnn.api_base, "get_device_type", lambda: "rubin")
    rejected("SM100/103")
