"""Tests for YARN RoPE helper and end-to-end RoPE+SDPA fusion."""

import math
import pytest
import torch

import cudnn
from cudnn.yarn import (
    compute_yarn_freqs,
    compute_yarn_inv_freq,
    yarn_get_mscale,
)

# RoPE backend op was added in cuDNN 9.24. Use this for tests that build a graph
# (the helper-only math tests above don't need it).
_skip_if_no_rope_backend = pytest.mark.skipif(
    cudnn.backend_version() < 92400,
    reason="RoPE backend op (CUDNN_BACKEND_OPERATION_ROPE_FWD_DESCRIPTOR) requires cuDNN >= 9.24",
)


# Megatron-Core reference (replicated locally; no mcore dependency)
def _mcore_yarn_inv_freq(head_dim, base, scaling_factor, original_max_position, beta_fast, beta_slow, device):
    inv_extra = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    if scaling_factor <= 1.0:
        return inv_extra
    inv_inter = inv_extra / scaling_factor
    low = math.floor((head_dim * math.log(original_max_position / (beta_fast * 2 * math.pi))) / (2 * math.log(base)))
    high = math.ceil((head_dim * math.log(original_max_position / (beta_slow * 2 * math.pi))) / (2 * math.log(base)))
    low = max(low, 0)
    high = min(high, head_dim - 1)
    if low == high:
        high = high + 0.001
    idx = torch.arange(head_dim // 2, device=device, dtype=torch.float32)
    ramp = torch.clamp((idx - low) / (high - low), 0.0, 1.0)
    mask = 1.0 - ramp
    return inv_inter * (1.0 - mask) + inv_extra * mask


@pytest.mark.L0
def test_yarn_inv_freq_matches_mcore_reference():
    """compute_yarn_inv_freq must match the Megatron-Core formulation bit-for-bit."""
    cases = [
        # (head_dim, factor, orig, beta_fast, beta_slow)
        (64, 40.0, 4096, 32, 1),  # DSv3
        (128, 4.0, 32768, 32, 1),  # Qwen2.5
        (128, 1.0, 4096, 32, 1),  # YARN inactive (factor=1)
        (192, 8.0, 8192, 32, 1),  # Hypothetical large
    ]
    for d, s, L, bf, bs in cases:
        ours = compute_yarn_inv_freq(
            d,
            base=10000.0,
            scaling_factor=s,
            original_max_position=L,
            beta_fast=bf,
            beta_slow=bs,
            device="cuda",
        )
        ref = _mcore_yarn_inv_freq(d, 10000.0, s, L, bf, bs, "cuda")
        torch.testing.assert_close(ours, ref, rtol=0, atol=0)


@pytest.mark.L0
def test_yarn_mscale_dsv3():
    """DSv3 config: factor=40, mscale_factor=1.0 -> 0.1*log(40)+1 ~= 1.3689."""
    m = yarn_get_mscale(40.0, 1.0)
    assert abs(m - (0.1 * math.log(40) + 1.0)) < 1e-12


@pytest.mark.L0
def test_yarn_mscale_inactive_when_factor_le_1():
    assert yarn_get_mscale(1.0) == 1.0
    assert yarn_get_mscale(0.5) == 1.0


@pytest.mark.L0
def test_yarn_freqs_factor1_matches_standard_rope():
    """factor=1 should produce identical freqs to standard RoPE (no YARN modification)."""
    head_dim = 128
    max_s = 256
    freqs_yarn, mscale = compute_yarn_freqs(
        max_s,
        head_dim,
        scaling_factor=1.0,
        original_max_position=4096,
    )
    # Standard RoPE: angle[s, i] = s * (1 / 10000^(2i/d))
    inv_std = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device="cuda", dtype=torch.float32) / head_dim))
    pos = torch.arange(max_s, device="cuda", dtype=torch.float32)
    angles_std = torch.outer(pos, inv_std)
    expected = torch.zeros(max_s, 1, 1, head_dim, device="cuda", dtype=torch.float32)
    expected[:, 0, 0, : head_dim // 2] = angles_std

    torch.testing.assert_close(freqs_yarn, expected, rtol=0, atol=0)
    assert mscale == 1.0


@pytest.mark.L0
def test_yarn_freqs_shape_and_zero_pad():
    head_dim = 64
    max_s = 100
    freqs, _ = compute_yarn_freqs(max_s, head_dim, scaling_factor=40.0, original_max_position=4096)
    assert freqs.shape == (max_s, 1, 1, head_dim)
    # Last D/2 columns must be zeros (kernel only reads first D/2)
    half = head_dim // 2
    assert torch.all(freqs[:, 0, 0, half:] == 0)


@pytest.mark.L0
def test_yarn_freqs_low_freq_dims_scaled_high_freq_passthrough():
    """For factor>1, low-freq dims (high index) get divided by factor; dim 0 is high-freq pass-through."""
    head_dim = 64
    factor = 40.0
    inv_freq = compute_yarn_inv_freq(
        head_dim,
        scaling_factor=factor,
        original_max_position=4096,
        beta_fast=32,
        beta_slow=1,
    )
    inv_extra = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device="cuda", dtype=torch.float32) / head_dim))
    # Dim 0: high-freq -> extrapolation (unchanged)
    assert torch.isclose(inv_freq[0], inv_extra[0], rtol=1e-6)
    # Last dim: low-freq -> interpolation (divided by factor)
    expected_last = inv_extra[-1] / factor
    assert torch.isclose(inv_freq[-1], expected_last, rtol=1e-6)


# ---------- End-to-end: YARN + cuDNN RoPE+SDPA ----------


def _build_and_run(b, h_q, h_k, s_q, s_kv, d, freqs, output_scale, attn_scale, dtype=torch.bfloat16):
    torch.manual_seed(0)
    q = torch.randn(b, h_q, s_q, d, device="cuda", dtype=dtype)
    k = torch.randn(b, h_k, s_kv, d, device="cuda", dtype=dtype)
    v = torch.randn(b, h_k, s_kv, d, device="cuda", dtype=dtype)

    cdt = cudnn.data_type.BFLOAT16 if dtype == torch.bfloat16 else cudnn.data_type.HALF
    g = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    Q = g.tensor(name="Q", dim=list(q.shape), stride=list(q.stride()), data_type=cdt)
    K = g.tensor(name="K", dim=list(k.shape), stride=list(k.stride()), data_type=cdt)
    V = g.tensor(name="V", dim=list(v.shape), stride=list(v.stride()), data_type=cdt)
    F = g.tensor(name="freqs", dim=list(freqs.shape), stride=list(freqs.stride()), data_type=cudnn.data_type.FLOAT)

    Qr = g.rope(input=Q, freqs=F, output_scale=output_scale, name="RoPE_Q")
    Qr.set_data_type(cdt).set_dim(list(q.shape)).set_stride(list(q.stride()))
    Kr = g.rope(input=K, freqs=F, output_scale=output_scale, name="RoPE_K")
    Kr.set_data_type(cdt).set_dim(list(k.shape)).set_stride(list(k.stride()))

    O, _ = g.sdpa(q=Qr, k=Kr, v=V, is_inference=True, attn_scale=attn_scale, name="SDPA")
    O.set_output(True).set_data_type(cdt).set_dim([b, h_q, s_q, d]).set_stride(list(q.stride()))
    g.build([cudnn.heur_mode.A])

    handle = cudnn.create_handle()
    o = torch.empty(b, h_q, s_q, d, device="cuda", dtype=dtype)
    q_rot = torch.empty_like(q)
    k_rot = torch.empty_like(k)
    ws = torch.empty(max(int(g.get_workspace_size()), 1), device="cuda", dtype=torch.uint8)
    g.execute(
        {
            Q.get_uid(): q.data_ptr(),
            K.get_uid(): k.data_ptr(),
            V.get_uid(): v.data_ptr(),
            F.get_uid(): freqs.data_ptr(),
            Qr.get_uid(): q_rot.data_ptr(),
            Kr.get_uid(): k_rot.data_ptr(),
            O.get_uid(): o.data_ptr(),
        },
        ws.data_ptr(),
        handle,
    )
    return o, q, k, v


def _reference_sdpa_with_yarn(q, k, v, freqs, mscale, attn_scale):
    """f32 reference: apply RoPE with mscale-folded cos/sin, then standard SDPA."""
    d = q.shape[-1]
    half = d // 2
    s_q, s_kv = q.shape[2], k.shape[2]

    angles_q = freqs[:s_q, 0, 0, :half].float()
    cos_q = (torch.cos(angles_q) * mscale).unsqueeze(0).unsqueeze(0)
    sin_q = (torch.sin(angles_q) * mscale).unsqueeze(0).unsqueeze(0)
    qf = q.float()
    q_rot = torch.cat([qf[..., :half] * cos_q - qf[..., half:] * sin_q, qf[..., half:] * cos_q + qf[..., :half] * sin_q], dim=-1)
    # Quantize back to input dtype to match GPU path (RoPE writes bf16/f16 workspace)
    q_rot = q_rot.to(q.dtype).float()

    angles_k = freqs[:s_kv, 0, 0, :half].float()
    cos_k = (torch.cos(angles_k) * mscale).unsqueeze(0).unsqueeze(0)
    sin_k = (torch.sin(angles_k) * mscale).unsqueeze(0).unsqueeze(0)
    kf = k.float()
    k_rot = torch.cat([kf[..., :half] * cos_k - kf[..., half:] * sin_k, kf[..., half:] * cos_k + kf[..., :half] * sin_k], dim=-1)
    k_rot = k_rot.to(k.dtype).float()

    scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) * attn_scale
    return torch.matmul(torch.softmax(scores, dim=-1), v.float())


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
@_skip_if_no_rope_backend
def test_yarn_e2e_dsv3_config():
    """End-to-end YARN + RoPE + SDPA matches a Python reference for the DSv3 config."""
    b, h, s, d = 1, 4, 256, 64
    factor = 40.0
    base_attn = 1.0 / math.sqrt(d)

    freqs, mscale = compute_yarn_freqs(
        s,
        d,
        scaling_factor=factor,
        original_max_position=4096,
        beta_fast=32,
        beta_slow=1,
        mscale_factor=1.0,
    )
    assert mscale > 1.0  # YARN active

    o_gpu, q, k, v = _build_and_run(b, h, h, s, s, d, freqs, output_scale=mscale, attn_scale=base_attn)
    o_ref = _reference_sdpa_with_yarn(q, k, v, freqs, mscale, base_attn)

    diff = (o_gpu.float() - o_ref).abs()
    assert diff.max().item() < 2e-2, f"YARN e2e diverged: max diff {diff.max().item()}"


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
@_skip_if_no_rope_backend
def test_yarn_e2e_factor1_no_op():
    """factor=1 should give identical result to plain RoPE+SDPA (no scaling, no mscale)."""
    b, h, s, d = 1, 4, 128, 64
    base_attn = 1.0 / math.sqrt(d)

    freqs, mscale = compute_yarn_freqs(s, d, scaling_factor=1.0)
    assert mscale == 1.0

    # Reference: plain RoPE
    o_gpu, q, k, v = _build_and_run(b, h, h, s, s, d, freqs, output_scale=mscale, attn_scale=base_attn)
    o_ref = _reference_sdpa_with_yarn(q, k, v, freqs, mscale, base_attn)

    diff = (o_gpu.float() - o_ref).abs()
    assert diff.max().item() < 2e-2


# ---------- Partial-dim RoPE (DSv3 MLA-style: rotate last K of head_dim) ----------


def _build_and_run_partial(b, h_q, h_k, s_q, s_kv, head_dim, rope_dim, freqs, output_scale, attn_scale, dtype=torch.bfloat16):
    """Run RoPE+SDPA where head_dim > rope_dim. Last rope_dim of head_dim gets rotated;
    first (head_dim - rope_dim) is scaled-pass-through."""
    torch.manual_seed(0)
    q = torch.randn(b, h_q, s_q, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(b, h_k, s_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(b, h_k, s_kv, head_dim, device="cuda", dtype=dtype)

    cdt = cudnn.data_type.BFLOAT16 if dtype == torch.bfloat16 else cudnn.data_type.HALF
    g = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    Q = g.tensor(name="Q", dim=list(q.shape), stride=list(q.stride()), data_type=cdt)
    K = g.tensor(name="K", dim=list(k.shape), stride=list(k.stride()), data_type=cdt)
    V = g.tensor(name="V", dim=list(v.shape), stride=list(v.stride()), data_type=cdt)
    F = g.tensor(name="freqs", dim=list(freqs.shape), stride=list(freqs.stride()), data_type=cudnn.data_type.FLOAT)

    Qr = g.rope(input=Q, freqs=F, output_scale=output_scale, rope_dim=rope_dim, name="RoPE_Q")
    Qr.set_data_type(cdt).set_dim(list(q.shape)).set_stride(list(q.stride()))
    Kr = g.rope(input=K, freqs=F, output_scale=output_scale, rope_dim=rope_dim, name="RoPE_K")
    Kr.set_data_type(cdt).set_dim(list(k.shape)).set_stride(list(k.stride()))

    O, _ = g.sdpa(q=Qr, k=Kr, v=V, is_inference=True, attn_scale=attn_scale, name="SDPA")
    O.set_output(True).set_data_type(cdt).set_dim([b, h_q, s_q, head_dim]).set_stride(list(q.stride()))
    g.build([cudnn.heur_mode.A])

    handle = cudnn.create_handle()
    o = torch.empty(b, h_q, s_q, head_dim, device="cuda", dtype=dtype)
    q_rot = torch.empty_like(q)
    k_rot = torch.empty_like(k)
    ws = torch.empty(max(int(g.get_workspace_size()), 1), device="cuda", dtype=torch.uint8)
    g.execute(
        {
            Q.get_uid(): q.data_ptr(),
            K.get_uid(): k.data_ptr(),
            V.get_uid(): v.data_ptr(),
            F.get_uid(): freqs.data_ptr(),
            Qr.get_uid(): q_rot.data_ptr(),
            Kr.get_uid(): k_rot.data_ptr(),
            O.get_uid(): o.data_ptr(),
        },
        ws.data_ptr(),
        handle,
    )
    return o, q, k, v


def _ref_partial_rope_sdpa(q, k, v, freqs, mscale, attn_scale, rope_dim):
    """Reference: rotate last rope_dim of each Q/K head, scale entire output by mscale,
    then standard SDPA. Matches the GPU kernel's "scaled passthrough + scaled rotation"."""
    head_dim = q.shape[-1]
    nope_dim = head_dim - rope_dim
    half = rope_dim // 2
    s_q, s_kv = q.shape[2], k.shape[2]

    qf = q.float()
    angles_q = freqs[:s_q, 0, 0, :half].float()
    cos_q = (torch.cos(angles_q) * mscale).unsqueeze(0).unsqueeze(0)
    sin_q = (torch.sin(angles_q) * mscale).unsqueeze(0).unsqueeze(0)
    q_nope = qf[..., :nope_dim] * mscale
    q_rope_lo = qf[..., nope_dim : nope_dim + half]
    q_rope_hi = qf[..., nope_dim + half :]
    q_rope = torch.cat([q_rope_lo * cos_q - q_rope_hi * sin_q, q_rope_hi * cos_q + q_rope_lo * sin_q], dim=-1)
    q_full = torch.cat([q_nope, q_rope], dim=-1).to(q.dtype).float()

    kf = k.float()
    angles_k = freqs[:s_kv, 0, 0, :half].float()
    cos_k = (torch.cos(angles_k) * mscale).unsqueeze(0).unsqueeze(0)
    sin_k = (torch.sin(angles_k) * mscale).unsqueeze(0).unsqueeze(0)
    k_nope = kf[..., :nope_dim] * mscale
    k_rope_lo = kf[..., nope_dim : nope_dim + half]
    k_rope_hi = kf[..., nope_dim + half :]
    k_rope = torch.cat([k_rope_lo * cos_k - k_rope_hi * sin_k, k_rope_hi * cos_k + k_rope_lo * sin_k], dim=-1)
    k_full = torch.cat([k_nope, k_rope], dim=-1).to(k.dtype).float()

    scores = torch.matmul(q_full, k_full.transpose(-2, -1)) * attn_scale
    return torch.matmul(torch.softmax(scores, dim=-1), v.float())


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
@_skip_if_no_rope_backend
def test_partial_rope_dsv3_layout():
    """DSv3-style partial RoPE: head_dim=192 = nope=128 + rope=64, full YARN config."""
    b, h, s = 1, 4, 256
    head_dim = 192
    rope_dim = 64
    base_attn = 1.0 / math.sqrt(head_dim)

    # YARN freqs sized to rope_dim (not head_dim)
    freqs, mscale = compute_yarn_freqs(
        s,
        rope_dim,
        scaling_factor=40.0,
        original_max_position=4096,
        beta_fast=32,
        beta_slow=1,
        mscale_factor=1.0,
    )
    assert freqs.shape == (s, 1, 1, rope_dim)

    o_gpu, q, k, v = _build_and_run_partial(b, h, h, s, s, head_dim, rope_dim, freqs, output_scale=mscale, attn_scale=base_attn)
    o_ref = _ref_partial_rope_sdpa(q, k, v, freqs, mscale, base_attn, rope_dim)

    diff = (o_gpu.float() - o_ref).abs()
    assert diff.max().item() < 2e-2, f"DSv3 partial RoPE diverged: max diff {diff.max().item()}"


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
@_skip_if_no_rope_backend
def test_partial_rope_zero_nope_matches_full_rotation():
    """rope_dim == head_dim must match the full-rotation path (no behavior change)."""
    b, h, s, d = 1, 4, 128, 64
    base_attn = 1.0 / math.sqrt(d)
    freqs, mscale = compute_yarn_freqs(s, d, scaling_factor=1.0)  # plain RoPE

    o_full, *_ = _build_and_run(b, h, h, s, s, d, freqs, output_scale=mscale, attn_scale=base_attn)
    o_part, *_ = _build_and_run_partial(b, h, h, s, s, d, d, freqs, output_scale=mscale, attn_scale=base_attn)
    diff = (o_full.float() - o_part.float()).abs()
    assert diff.max().item() < 1e-3


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
@_skip_if_no_rope_backend
def test_yarn_mscale_fold_via_attn_scale():
    """YARN's mscale fold-in equivalence:
    (output_scale=mscale on Q&K, attn_scale=base) == (output_scale=1, attn_scale=base*mscale^2)
    """
    b, h, s, d = 1, 4, 128, 64
    base_attn = 1.0 / math.sqrt(d)
    freqs, mscale = compute_yarn_freqs(s, d, scaling_factor=40.0, original_max_position=4096)

    o_a, *_ = _build_and_run(b, h, h, s, s, d, freqs, output_scale=mscale, attn_scale=base_attn)
    o_b, *_ = _build_and_run(b, h, h, s, s, d, freqs, output_scale=1.0, attn_scale=base_attn * mscale * mscale)

    diff = (o_a.float() - o_b.float()).abs()
    assert diff.max().item() < 2e-2
