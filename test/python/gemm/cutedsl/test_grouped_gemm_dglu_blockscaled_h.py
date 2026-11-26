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
- On the stock dynamic scheduler (2-CTA and 1-CTA tiles) and the input signal scheduler (TMA SFA atoms and compact
  SFA, every M tile ready).
"""

import pytest
import torch

from gemm.cutedsl import _streaming_signal_blockscaled as bs
from gemm.cutedsl._streaming_signal import signal_ready
from gemm.cutedsl._streaming_signal_blockscaled import require_sm10x  # noqa: F401  (autouse fixture)

#: Poison for every output byte: an output the kernel should write but does not stays 0xA5.
POISON = 0xA5
#: dGLU N (W2 rows, the width of h) and K.
N, K = 512, 512


@pytest.fixture
def require_sm100():
    if bs.is_rubin():
        pytest.skip("the SM100 kernels")


def make_problem(routing, seed, exact):
    """The dGLU problem; ``exact``: gates of 0 or 40..60 (beta >= 0.5), so silu(beta * gate) is exact in float32."""
    problem = bs.make_problem("dglu", bs.ROUTINGS[routing], n=N, k=K, seed=seed)
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
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import OperandMajorMode
    from cutlass.cute.runtime import from_dlpack
    from cutlass.utils import HardwareInfo

    one_cta = options.pop("one_cta", False)
    kernel = bs.make_kernel("dglu", len(problem["layout"]) - 1, one_cta=one_cta, discrete_col_sfd=True, **options)
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

    sfa = e8(problem["a_exp"] if kernel.test_compact else problem["sfa"])
    args = [
        t(problem["a"]),
        t(problem["b"].permute(1, 2, 0)),
        e8(problem["sfb"]),
        cutlass.Int32(n),
        cutlass.Int32(problem["k"]),
        cutlass.Int64(problem["k"]),
        OperandMajorMode.K,
        from_dlpack(workspace, assumed_align=128).iterator,
        t(problem["c"]),
        codes(out["d"], 2 * n),
        codes(out["d_col"], 2 * n),
        sfa,
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
    clusters = HardwareInfo().get_max_active_clusters(kernel.test_cluster_size)
    key = (kernel.test_compile_key, h, rows, n)
    if key not in COMPILED:
        COMPILED[key] = cute.compile(kernel, *args, clusters, bs.cu_stream(), **kwargs)
    if kernel.test_signal:
        signal_ready(kernel)(workspace)
    COMPILED[key](*[x for i, x in enumerate(args) if i != bs.b_major_arg(kernel)], bs.cu_stream(), **kwargs)
    torch.cuda.synchronize()
    return out


COMPILED = {}


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


def valid(problem):
    return bs.valid_rows(problem)


#: (scheduler variant, routing): the stock dynamic scheduler (2-CTA, 1-CTA, scalar epilogue math), the signal
#: scheduler with TMA SFA atoms and with compact SFA.
VARIANTS = {
    "dynamic": {},
    "dynamic_1cta": dict(one_cta=True),
    "dynamic_scalar": dict(vectorized_f32=False),
    "signal": None,
    "compact_signal": None,
}
CASES = [(variant, routing) for variant in VARIANTS for routing in sorted(bs.ROUTINGS)]
L0 = {
    ("dynamic", "uneven"),
    ("dynamic_1cta", "uneven"),
    ("dynamic_scalar", "big_expert"),
    ("signal", "many_experts_64"),
    ("compact_signal", "many_experts_64"),
    ("compact_signal", "small_experts_64"),
}


def options_for(variant, problem):
    if VARIANTS[variant] is not None:
        return dict(use_dynamic_sched=True, **VARIANTS[variant])
    return bs.variant_options("dglu", problem, variant)


@pytest.mark.parametrize("variant, routing", bs.levels(CASES, L0, lambda variant, routing: f"{variant}-{routing}"))
def test_dglu_h_matches_reference_and_leaves_dh_unchanged(variant, routing, require_sm100):
    problem = make_problem(routing, seed=11, exact=True)
    options = options_for(variant, problem)
    want = run_dglu(problem, h=False, **dict(options))
    got = run_dglu(problem, h=True, **dict(options))
    rows = valid(problem)
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
@pytest.mark.parametrize("variant", ["dynamic", "compact_signal"])
def test_dglu_h_general_inputs_within_quantization_error(variant, require_sm100):
    """Random gates (sigmoid by the kernel's approximate exp2 / reciprocal): every h scale and code is written, and the
    dequantized h is within E4M3 rounding of the float reference."""
    problem = make_problem("uneven", seed=13, exact=False)
    got = run_dglu(problem, h=True, **options_for(variant, problem))
    rows = valid(problem)
    codes = got["h"].view(problem["rows"], N)[:rows]
    deq = dequantize_h(codes, got["sfh"], problem["offsets"], N)
    ref = h_reference(problem)[:rows]
    _, want_sf = quantize_h(h_reference(problem), problem["offsets"])
    sf_ok = (got["sfh"][: rows * N // 32] == want_sf[: rows * N // 32]).float().mean().item()
    assert sf_ok > 0.99, f"{variant}: only {sf_ok:.4f} of the h scales match the reference"
    scale = deq.abs().view(rows // 32, 32, N).amax(dim=1).repeat_interleave(32, dim=0)
    assert ((deq - ref).abs() <= 2.0**-3 * ref.abs() + scale * 2.0**-9).all(), f"{variant}: h beyond E4M3 rounding of the reference"
