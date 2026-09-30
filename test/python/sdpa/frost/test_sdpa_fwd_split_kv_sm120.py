# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV split on SM120, driven through the adapter rather than the template.

KNOB ROUTE: the adapter never decides the split itself — the heuristic's
chooser picks a value and the test passes it explicitly as the ``split_kv``
constructor knob, exactly as ``lower_dsl_prefill`` forwards a plan's knobs.
The chooser and the two-launch execute are shared with SM100; what differs here
is the geometry (one CTA per tile, no cluster) and that the config bars a split
under the flattened LPT schedulers.
"""

import math
from typing import NamedTuple, Optional

import pytest

import torch

from cudnn.frost.compiled_cache import positional_entry

from frost_test_utils import requires_blackwell_geforce, requires_dsl

pytestmark = [requires_dsl, pytest.mark.L0]


class _ApiCaseResult(NamedTuple):
    split: int
    output: torch.Tensor
    reference: torch.Tensor
    workspace_bytes: int
    expected_split: int
    stats: Optional[torch.Tensor]


def _expected_split(api):
    """What the chooser asks for on THIS device, from the adapter's OWN tile
    geometry. SM120 runs one CTA per tile (no cluster), and the adapter may pick
    either q_tile, so reading them off `api` keeps the expectation tied to the
    launch that actually happens rather than to one part's SM count."""
    from cudnn._device import device_info
    from cudnn.sdpa.fwd.heuristics import choose_split_kv

    return choose_split_kv(
        q_tiles=-(-api.s_q_max // api.q_tile),
        heads_q=api.h_q,
        batch=api.batch_size,
        kv_tiles=-(-api.s_k_max // api.kv_tile),
        sm_count=device_info(torch.cuda.current_device()).sm_count,
        ctas_per_tile=1,
        # The combine's grid is (S_q, H, B) — see choose_split_kv.
        combine_rows=api.batch_size * api.h_q * api.s_q_max,
    )


def _sm120_reference(q, k, v, *, causal=False, stats_log2=False):
    """Keep the FP32 oracle's score matrix bounded to one query head."""
    b, h_q, s_q, d = q.shape
    h_kv, s_kv = k.shape[1:3]
    output = torch.empty((b, h_q, s_q, v.shape[-1]), device=q.device, dtype=torch.float32)
    stats = torch.empty((b, h_q, s_q), device=q.device, dtype=torch.float32)
    mask = None
    if causal:
        i = torch.arange(s_q, device=q.device).view(s_q, 1)
        j = torch.arange(s_kv, device=q.device).view(1, s_kv)
        mask = j > i
    for head in range(h_q):
        kv_head = head // (h_q // h_kv)
        qb = q[:, head : head + 1].float()
        kb = k[:, kv_head : kv_head + 1].float()
        vb = v[:, kv_head : kv_head + 1].float()
        scores = torch.matmul(qb, kb.transpose(-1, -2)) / math.sqrt(d)
        if mask is not None:
            scores.masked_fill_(mask, float("-inf"))
        p = torch.softmax(scores, dim=-1)
        output[:, head : head + 1] = torch.matmul(p, vb)
        stats[:, head : head + 1] = torch.logsumexp(scores, dim=-1) * (math.log2(math.e) if stats_log2 else 1.0)
    return output, stats


@pytest.mark.parametrize("h_kv", [1, 2, 4])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("stats_log2", [False, True])
def test_sm120_split_reference_matches_dense(h_kv, causal, stats_log2):
    """Head chunking preserves GQA mapping, strided inputs, masks and Stats."""
    torch.manual_seed(0)
    q = torch.randn(2, 17, 4, 16, device="cuda", dtype=torch.float16).transpose(1, 2)
    k = torch.randn(2, 23, h_kv, 16, device="cuda", dtype=torch.float16).transpose(1, 2)
    v = torch.randn(2, 23, h_kv, 32, device="cuda", dtype=torch.float16).transpose(1, 2)
    output, stats = _sm120_reference(q, k, v, causal=causal, stats_log2=stats_log2)
    kb, vb = (x.float().repeat_interleave(4 // h_kv, dim=1) for x in (k, v))
    scores = torch.matmul(q.float(), kb.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if causal:
        scores.masked_fill_(torch.arange(23, device="cuda")[None, :] > torch.arange(17, device="cuda")[:, None], float("-inf"))
    torch.testing.assert_close(output, torch.matmul(torch.softmax(scores, dim=-1), vb), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(stats, torch.logsumexp(scores, dim=-1) * (math.log2(math.e) if stats_log2 else 1.0), rtol=1e-5, atol=1e-6)


def _sm120_case(
    h_q, h_kv, s_q, s_kv, *, d=128, with_lse=False, workspace=True, causal=False, lse_layout="contiguous", split_kv=None, stats_log2=False, zero_copy=False
):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120 part required")
    b, dev = 1, "cuda"
    torch.manual_seed(0)

    def operand(h, s):
        if zero_copy:
            return torch.randn(b, s, h, d, device=dev, dtype=torch.float16).transpose(1, 2)
        return torch.randn(b, h, s, d, device=dev, dtype=torch.float16)

    q, k, v = operand(h_q, s_q), operand(h_kv, s_kv), operand(h_kv, s_kv)
    o = torch.zeros_like(q)
    lse_storage = None
    if not with_lse:
        lse = None
    elif lse_layout == "contiguous":
        lse = torch.zeros(b, h_q, s_q, device=dev, dtype=torch.float32)
    elif lse_layout == "strided":
        lse_storage = torch.full((s_q + 7, h_q + 2, b), -12345.0, device=dev, dtype=torch.float32)
        lse = lse_storage.permute(2, 1, 0)[:, :h_q, :s_q]
    else:
        raise ValueError(f"unknown LSE layout {lse_layout!r}")

    kw = dict(is_causal=causal, stats_log2=stats_log2)
    # Probe pass: the chooser reads the adapter's own tile geometry.
    probe = SdpaFwdDslSm120(sample_q=q, sample_k=k, sample_v=v, sample_o=o, sample_lse=lse, **kw)
    assert probe.check_support()
    expected = _expected_split(probe) if split_kv is None else split_kv
    # Knob route: the chosen split arrives as an explicit constructor knob
    # (split sets ride SCHED_NATURAL — the config bars a split under the LPT
    # remaps a causal graph would otherwise derive).
    if expected > 1:
        kw.update(split_kv=expected, sched_policy=0)
    api = SdpaFwdDslSm120(sample_q=q, sample_k=k, sample_v=v, sample_o=o, sample_lse=lse, **kw)
    assert api.check_support()
    split = api.split_kv
    ws_bytes = api.scratch_workspace_bytes()
    api.compile()
    if zero_copy:
        assert api._dense_spec is not None
    ws = torch.empty(ws_bytes, dtype=torch.uint8, device=dev) if (workspace and ws_bytes) else None
    api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=lse, workspace=ws)
    torch.cuda.synchronize()

    reference, reference_stats = _sm120_reference(q, k, v, causal=causal, stats_log2=stats_log2)
    if lse is not None:
        torch.testing.assert_close(lse, reference_stats, rtol=3e-2, atol=5e-2)
    if lse_storage is not None:
        gaps = torch.ones_like(lse_storage, dtype=torch.bool)
        gaps[:s_q, :h_q, :] = False
        assert torch.all(lse_storage[gaps] == -12345.0), "the combine LSE store touched padding outside its declared view"
    return _ApiCaseResult(split, o.float(), reference, ws_bytes, expected, None if lse is None else lse.clone())


def test_sm120_splits_a_decode_shape():
    """A decode shape the chooser wants split -- and the adapter must honor it."""
    result = _sm120_case(8, 1, 128, 32768)
    if result.expected_split == 1:
        pytest.skip("this part is small enough that the shape already fills it")
    assert result.split == result.expected_split > 1
    assert result.workspace_bytes > 0
    assert (result.output - result.reference).abs().max().item() <= 2e-2


@pytest.mark.parametrize("zero_copy", [False, True])
def test_sm120_does_not_split_a_full_part(zero_copy):
    """A launch that fills the part is left alone. Whether 1024x64 fills it
    depends on the SM count, so the expectation comes from the chooser rather
    than a fixed 1 that only holds on one device."""
    result = _sm120_case(64, 8, 1024, 16384, zero_copy=zero_copy)
    assert result.split == result.expected_split
    conversion_bytes = 0 if zero_copy else 2 * (64 * 1024 + 8 * 16384) * 128 * 2
    assert result.workspace_bytes >= conversion_bytes
    assert (result.workspace_bytes > conversion_bytes) == (result.split > 1), "only splits add partial slabs beyond the conversion buffers"
    assert (result.output - result.reference).abs().max().item() <= 2e-2


@pytest.mark.parametrize("workspace", [True, False], ids=["carved", "missing"])
def test_sm120_split_with_and_without_workspace(workspace):
    if not workspace:
        with pytest.raises(ValueError, match="requires a .* workspace"):
            _sm120_case(8, 1, 128, 32768, workspace=False, split_kv=2, zero_copy=True)
        return
    result = _sm120_case(8, 1, 128, 32768, workspace=True, split_kv=2, zero_copy=True)
    assert result.split == result.expected_split
    assert (result.output - result.reference).abs().max().item() <= 2e-2


def test_sm120_split_writes_the_recombined_lse():
    result = _sm120_case(8, 1, 128, 32768, with_lse=True)
    assert result.split == result.expected_split
    assert (result.output - result.reference).abs().max().item() <= 2e-2


def test_sm120_d512_split_recombines_o_and_lse():
    """The d512 flavor under a KV split: its fp32 partial O (512 wide) and LSE
    recombine through the shared combine pass."""
    result = _sm120_case(8, 1, 64, 8192, d=512, with_lse=True, split_kv=4)
    assert result.split == 4
    assert (result.output - result.reference).abs().max().item() <= 2e-2


def test_sm120_split_writes_strided_recombined_lse():
    contiguous = _sm120_case(8, 1, 128, 1024, with_lse=True, split_kv=2)
    strided = _sm120_case(8, 1, 128, 1024, with_lse=True, lse_layout="strided", split_kv=2)
    assert strided.split == strided.expected_split == 2
    assert (strided.output - strided.reference).abs().max().item() <= 2e-2
    torch.testing.assert_close(strided.stats, contiguous.stats, atol=0, rtol=0)


def test_sm120_causal_split_requires_the_natural_scheduler():
    """The config bars a split under the LPT remaps a causal graph derives.
    Knob-route contract, both directions: a split WITHOUT an explicit
    scheduler lets the adapter derive LPT and must fail loudly at compile
    (honored-or-error, never silently degraded), while split + explicit
    NATURAL — what the heuristic actually emits — compiles and matches."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120 part required")
    b, d, dev = 1, 128, "cuda"
    torch.manual_seed(0)
    q = torch.randn(b, 8, 512, d, device=dev, dtype=torch.float16)
    k = torch.randn(b, 1, 8192, d, device=dev, dtype=torch.float16)
    v = torch.randn(b, 1, 8192, d, device=dev, dtype=torch.float16)
    o = torch.zeros_like(q)

    bad = SdpaFwdDslSm120(sample_q=q, sample_k=k, sample_v=v, sample_o=o, is_causal=True, split_kv=2)
    assert bad.check_support()
    with pytest.raises(ValueError, match="split_kv"):
        bad.compile()  # derived LPT + split: the config backstop rejects

    result = _sm120_case(8, 1, 512, 8192, causal=True)
    if result.expected_split == 1:
        pytest.skip("this part is small enough that the causal shape already fills it")
    assert result.split == result.expected_split > 1, "the causal arm must actually exercise the split"
    assert (result.output - result.reference).abs().max().item() <= 2e-2


# --- FP8, including a QUANTIZED O ------------------------------------------
#
# The SM120 FP8 adapter shares the split plumbing with SM100: under a split the
# kernel writes half partials and the combine performs the only cast down to
# the O dtype, applying scale_o there. Nothing above exercises the FP8 arm.


def _sm120_fp8_case(h_q, h_kv, s_q, s_kv, *, out_dtype, split_kv, scale_o=1.0, zero_copy=False):
    """FP8 through the SM120 adapter; returns (split, O, amax)."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120 part required")
    b, d, dev = 1, 128, "cuda"
    torch.manual_seed(0)

    def mk(*sh):
        return (torch.randn(*sh, device=dev) * 0.5).to(torch.float8_e4m3fn)

    q, k, v = mk(b, h_q, s_q, d), mk(b, h_kv, s_kv, d), mk(b, h_kv, s_kv, d)
    o = torch.zeros(b, h_q, s_q, d, device=dev, dtype=out_dtype)
    if zero_copy:
        q, k, v, o = (t.transpose(1, 2).contiguous().transpose(1, 2) for t in (q, k, v, o))
    amax = torch.zeros(1, dtype=torch.float32, device=dev)

    def one():
        return torch.ones(1, dtype=torch.float32, device=dev)

    kw = dict(pertensor_fp8=True, dtype_o=out_dtype)
    if split_kv > 1:
        kw.update(split_kv=split_kv, sched_policy=0)
    api = SdpaFwdDslSm120(sample_q=q, sample_k=k, sample_v=v, sample_o=o, **kw)
    assert api.check_support()
    ws_bytes = api.scratch_workspace_bytes()
    api.compile()
    ws = torch.empty(ws_bytes, dtype=torch.uint8, device=dev) if ws_bytes else None
    api.execute(
        q_tensor=q,
        k_tensor=k,
        v_tensor=v,
        o_tensor=o,
        amax_o=amax,
        workspace=ws,
        descale_q=one(),
        descale_k=one(),
        descale_v=one(),
        scale_o=one() * scale_o,
    )
    torch.cuda.synchronize()

    # Independent fp32 oracle. Comparing a split run against an unsplit one only
    # proves they agree -- they share the inputs, the scales and the attention
    # math, so a defect there moves both. descale_* are 1.0, so the dequantized
    # inputs are just the fp8 values.
    qf, kf, vf = q.float(), k.float(), v.float()
    if h_q != h_kv:
        rep = h_q // h_kv
        kf, vf = kf.repeat_interleave(rep, dim=1), vf.repeat_interleave(rep, dim=1)
    reference = (torch.softmax(qf @ kf.transpose(-1, -2) / math.sqrt(d), dim=-1) @ vf) * scale_o
    atol = 5e-2
    if out_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # A quantized O lands on its own lattice; allow the store's own step.
        atol = max(atol, 3.0 * (reference - reference.to(out_dtype).float()).abs().max().item())
    diff = (o.float() - reference).abs().max().item()
    assert diff <= atol, f"SM120 split={api.split_kv} max|O-ref|={diff:.4f} > {atol:.4f}"
    return api.split_kv, o.float().clone(), amax.item(), ws_bytes


@pytest.mark.parametrize("out_dtype", [torch.float16, torch.float8_e4m3fn, torch.float8_e5m2], ids=["f16_out", "e4m3_out", "e5m2_out"])
def test_sm120_fp8_split_matches_unsplit(out_dtype):
    """A split must not move the result, quantized O included."""
    _, unsplit, amax_one, _ = _sm120_fp8_case(8, 1, 128, 8192, out_dtype=out_dtype, split_kv=1)
    split, got, amax_split, ws = _sm120_fp8_case(8, 1, 128, 8192, out_dtype=out_dtype, split_kv=4)
    assert split == 4 and ws > 0
    step = (unsplit.abs().max() * torch.finfo(out_dtype).eps).item()
    assert (got - unsplit).abs().max().item() <= 4 * step + 5e-3
    assert abs(amax_split - amax_one) <= 0.03, "amax must describe the recombined output at either split"


@pytest.mark.parametrize("zero_copy", [False, True])
def test_sm120_quantized_split_reduces_in_half(zero_copy):
    """Partials stay half; only the staged final-O buffer follows O's dtype."""
    from cudnn.sdpa.fwd.api_dsl import ws_align

    _, _, _, ws_fp8 = _sm120_fp8_case(8, 1, 128, 8192, out_dtype=torch.float8_e4m3fn, split_kv=4, zero_copy=zero_copy)
    _, _, _, ws_half = _sm120_fp8_case(8, 1, 128, 8192, out_dtype=torch.float16, split_kv=4, zero_copy=zero_copy)
    partials = ws_align(4 * 8 * 128 * 128 * 2) + ws_align(4 * 8 * 128 * 4)
    core = partials + ws_align(8)  # Amax scratch and the identity scalar
    # Q needs a compact copy on the staged path; the one-head K/V already
    # have compact physical storage. The final O copy has its own dtype.
    q_bytes = 0 if zero_copy else ws_align(8 * 128 * 128)
    o_elements = 0 if zero_copy else 8 * 128 * 128
    assert ws_fp8 == core + q_bytes + ws_align(o_elements)
    assert ws_half == core + q_bytes + ws_align(o_elements * 2)


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_sm120_quantized_split_applies_scale_o_once(scale):
    """scale_o is applied at the combine's cast, not in the split epilogue."""
    _, one, amax_one, _ = _sm120_fp8_case(8, 1, 128, 8192, out_dtype=torch.float8_e4m3fn, split_kv=4)
    _, scaled, amax_scaled, _ = _sm120_fp8_case(8, 1, 128, 8192, out_dtype=torch.float8_e4m3fn, split_kv=4, scale_o=scale)
    want = one * scale
    rel = (scaled - want).abs().max().item() / max(want.abs().max().item(), 1e-6)
    assert rel <= 0.15, f"stored O is not scale_o x the unscaled O (rel {rel:.3f})"
    assert abs(amax_scaled - amax_one) <= 0.03, "amax describes the PRE-quant output, so scale_o cannot move it"


def test_sm120_split_keeps_half_partials():
    """SM120 partials stay half: sO aliases sKV here, so there is no room to
    widen the O tile the way the SM100 kernels do, and these producers are
    compiled without the fp32 partial-tensor slot.

    The predicate that selects fp32 lives on the shared base, so if it ever
    stops excluding this arch the adapter would hand an SM120 kernel an
    argument it cannot take -- an ABI mismatch at launch, not a wrong number."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120 part required")
    b, h, s_q, s_kv, d, dev = 1, 8, 128, 8192, 128, "cuda"
    torch.manual_seed(0)
    q = torch.randn(b, h, s_q, d, device=dev, dtype=torch.float16)
    k = torch.randn(b, 1, s_kv, d, device=dev, dtype=torch.float16)
    v = torch.randn(b, 1, s_kv, d, device=dev, dtype=torch.float16)
    o = torch.zeros_like(q)
    api = SdpaFwdDslSm120(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=o,
        split_kv=4,
        sched_policy=0,
    )
    assert api.check_support()
    assert not api._fp32_partial_split()
    assert api._partial_dtype_tag() == "f16"


@pytest.mark.parametrize("stats_log2", [False, True], ids=["ln", "log2"])
@pytest.mark.parametrize("d", [128, 256, 512])
def test_sm120_split_stats_base(d, stats_log2):
    result = _sm120_case(4, 2, 128, 1024, d=d, with_lse=True, lse_layout="strided", split_kv=4, stats_log2=stats_log2)
    assert result.split == 4
    torch.testing.assert_close(result.output, result.reference, atol=3e-2, rtol=3e-2)


@requires_blackwell_geforce
@pytest.mark.parametrize("fp8", [False, True], ids=["f16", "fp8"])
@pytest.mark.parametrize("splits", [1, 4], ids=["unsplit", "split4"])
@pytest.mark.parametrize("stats_log2", [False, True], ids=["ln", "log2"])
def test_sm120_direct_template_stats_base(fp8, splits, stats_log2):
    """Direct callers may enable log2 even on a partial-producing template.

    The adapter clears that flag before compiling partials, so going through
    it cannot detect a missing kernel-entry guard. Check the actual partial
    LSE units here, then feed them to the real natural-log combiner.
    """
    import cutlass
    import cuda.bindings.driver as cuda

    from cudnn.frost.compiled_cache import positional_entry
    from cudnn.sdpa.fwd.api_dsl import _load_sm120_kernel_module
    from cudnn.sdpa.fwd.config_sm120 import DTYPE_E4M3, DTYPE_FP16, TemplateParams
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine

    b, h, sq, skv, d = 1, 2, 64, 512, 128
    dtype = torch.float8_e4m3fn if fp8 else torch.float16
    q = torch.full((b, sq, h, d), 0.5, device="cuda", dtype=dtype)
    # Different nonzero maxima per 128-key chunk make both a wrong log base
    # and a second conversion observable in the partials and the final O.
    chunk = torch.arange(skv, device="cuda") // 128 + 1
    k = (chunk.view(1, skv, 1, 1) * 0.5).expand(b, skv, 1, d).to(dtype).contiguous()
    v = (chunk.view(1, skv, 1, 1) * 0.125).expand(b, skv, 1, d).to(dtype).contiguous()
    partial_o = torch.full((splits * b, sq, h, d), float("nan"), device="cuda", dtype=torch.float16)
    partial_lse = torch.full((splits * b, h, sq), float("nan"), device="cuda")
    seq_q = torch.full((b,), sq, device="cuda", dtype=torch.int32)
    seq_kv = torch.full((b,), skv, device="cuda", dtype=torch.int32)
    params = TemplateParams(
        dtype_qkv=DTYPE_E4M3 if fp8 else DTYPE_FP16,
        dtype_o=DTYPE_FP16,
        q_tile=64,
        kv_tile=128,
        split_kv=splits,
        stats_log2=stats_log2,
    )
    module = _load_sm120_kernel_module(None, params, fp8=fp8)
    kernel = module.compile(torch.cuda.get_device_capability(), b=b, qh=h, kh=1, sq=sq, skv=skv, d_qk=d, d_v=d)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    # Compile PARAMS.stats_log2 directly, including split > 1. The adapter
    # clears that bit and would hide the kernel-entry regression this probes.
    args = [
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        partial_o.data_ptr(),
        partial_lse.data_ptr(),
        0,
        seq_kv.data_ptr(),
        0,
        (b, h, 1, sq, skv, 0),
        tuple(q.stride()[:3]),
        tuple(k.stride()[:3]),
        tuple(v.stride()[:3]),
        tuple(partial_o.stride()[:3]),
        tuple(partial_lse.stride()),
        0,
        math.log2(math.e) / math.sqrt(d),
        seq_q.data_ptr(),
        None,
        None,
        None,
        0,
    ]
    if fp8:
        amax = torch.zeros(1, device="cuda", dtype=torch.int32)
        unit_scale = torch.ones(1, device="cuda")
        args += [unit_scale.data_ptr()] * 4 + [amax.data_ptr(), None]
    raw = positional_entry(kernel)
    assert raw is not None
    raw(*args, stream)
    torch.cuda.synchronize()

    scores = torch.einsum("bqhd,bknd->bhqk", q.double(), k.double()) / math.sqrt(d)
    values = v.double().transpose(1, 2).expand(b, h, skv, d)
    for split in range(splits):
        lo, hi = split * (skv // splits), (split + 1) * (skv // splits)
        split_scores = scores[..., lo:hi]
        expected_lse = split_scores.logsumexp(-1)
        if stats_log2 and splits == 1:
            expected_lse *= math.log2(math.e)
        torch.testing.assert_close(partial_lse[split * b : (split + 1) * b].double(), expected_lse, atol=2e-4, rtol=2e-5)
        expected_o = (split_scores.softmax(-1) @ values[..., lo:hi, :]).transpose(1, 2)
        torch.testing.assert_close(partial_o[split * b : (split + 1) * b].double(), expected_o, atol=3e-3, rtol=3e-3)

    if splits > 1:
        output = torch.full((b, sq, h, d), float("nan"), device="cuda", dtype=torch.float16)
        lse = torch.full((b, h, sq), float("nan"), device="cuda")
        combine = positional_entry(split_combine.compile_ptr(has_lse=True, stats_log2=stats_log2, dtype_partial="f16"))
        combine(
            partial_o.data_ptr(), partial_lse.data_ptr(), output.data_ptr(), lse.data_ptr(), (b, h, sq, d), splits, output.stride(), lse.stride(), int(stream)
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(output.double(), (scores.softmax(-1) @ values).transpose(1, 2), atol=3e-3, rtol=3e-3)
        torch.testing.assert_close(lse.double(), scores.logsumexp(-1) * (math.log2(math.e) if stats_log2 else 1.0), atol=2e-4, rtol=2e-5)
