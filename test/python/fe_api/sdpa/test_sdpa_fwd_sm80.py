# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SM80 (A100) SDPA forward API.

Skipped automatically on non-SM80 devices or when the optional CuTe-DSL
dependency (``nvidia-cutlass-dsl``) is missing.  Compares the kernel's
output against a fp32 torch reference at a per-element fp16 tolerance.
"""

import math
import pytest
import torch

from test_utils import torch_fork_set_rng


def _is_sm80() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(0)
    return (major, minor) == (8, 0)


def _dsl_available() -> bool:
    # The kernels need the CuTe DSL *with* cutlass.experimental (cutlass-dsl
    # >= 4.7). The package imports lazily (PEP 562), so a missing/old DSL
    # only surfaces at kernel-load time — probe it here so the suite SKIPS
    # instead of erroring mid-test.
    try:
        import cutlass.experimental  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not (_is_sm80() and _dsl_available()),
    reason="SM80 SDPA API requires an SM80 (A100) device and nvidia-cutlass-dsl >= 4.7.",
)


def _bshd_randn(b, h, s, d, **kw):
    """BHSD-logical tensor with the BSHD-physical stride order the SM80
    adapters require."""
    return torch.randn((b, s, h, d), **kw).permute(0, 2, 1, 3)


def _ref_sdpa(q, k, v, *, is_causal, window_size, scale):
    """Reference SDPA in fp32 with the same masking semantics the kernel
    promises (causal / SWA window described by (left, right))."""
    _b, h_q, s_q, _d_qk = q.shape
    _, h_kv, s_kv, _ = k.shape
    _, _, _, _d_v = v.shape
    g = h_q // h_kv
    # Expand K/V across GQA groups for ref.
    k_ref = k.repeat_interleave(g, dim=1).to(torch.float32)
    v_ref = v.repeat_interleave(g, dim=1).to(torch.float32)
    q_ref = q.to(torch.float32)

    scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) * scale  # [B,H,Sq,Skv]

    if is_causal:
        # Causal aligns the right edge of Q to the right edge of K.
        i = torch.arange(s_q, device=q.device).view(s_q, 1)
        j = torch.arange(s_kv, device=q.device).view(1, s_kv)
        keep = j <= (i + (s_kv - s_q))
        if window_size[0] >= 0:
            keep &= ((i + (s_kv - s_q)) - j) <= window_size[0]
        scores = scores.masked_fill(~keep, float("-inf"))
    elif window_size[0] >= 0:
        i = torch.arange(s_q, device=q.device).view(s_q, 1)
        j = torch.arange(s_kv, device=q.device).view(1, s_kv)
        keep = (i - j).abs() <= window_size[0]
        scores = scores.masked_fill(~keep, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    o = torch.matmul(probs, v_ref).to(q.dtype)
    return o


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sdpa_fwd_sm80_smoke():
    """One representative case at L0 (llama flavor, fp16, causal, MHA);
    the full flavor x mask x GQA x dtype sweep runs at L2."""
    test_sdpa_fwd_sm80_wrapper(torch.float16, 128, 128, "causal", (8, 8))


@pytest.mark.L2
@pytest.mark.parametrize("d_qk,d_v", [(64, 64), (128, 128), (192, 128), (256, 256)], ids=["gptoss", "llama", "dsv3", "qwen"])
@pytest.mark.parametrize("mask", ["none", "causal", "swa"])
@pytest.mark.parametrize("gqa", [(8, 8), (16, 4)], ids=["mha", "gqa4x"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@torch_fork_set_rng(seed=0)
def test_sdpa_fwd_sm80_wrapper(dtype, d_qk, d_v, mask, gqa):
    """End-to-end check against torch reference SDPA (full sweep, L2)."""
    try:
        from cudnn.sdpa.fwd import sdpa_fwd_wrapper_sm80
    except ImportError as e:
        pytest.skip(f"SM80 SDPA API not available: {e}")

    b, s_q, s_kv = 2, 1024, 1024
    h_q, h_kv = gqa
    device = "cuda"

    q = _bshd_randn(b, h_q, s_q, d_qk, dtype=dtype, device=device)
    k = _bshd_randn(b, h_kv, s_kv, d_qk, dtype=dtype, device=device)
    v = _bshd_randn(b, h_kv, s_kv, d_v, dtype=dtype, device=device)
    scale = 1.0 / math.sqrt(d_qk)

    is_causal = mask == "causal"
    window = (128, 0) if mask == "swa" else (-1, -1)
    if mask == "swa":
        is_causal = True

    try:
        out = sdpa_fwd_wrapper_sm80(
            q_tensor=q,
            k_tensor=k,
            v_tensor=v,
            is_causal=is_causal,
            window_size=window,
            scale_softmax=scale,
        )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    o = out["o_tensor"]
    lse = out["lse_tensor"]

    o_ref = _ref_sdpa(q, k, v, is_causal=is_causal, window_size=window, scale=scale)

    # ~4x FP16 ULP — matches the cudnn-FE upstream SDPA test tolerance.
    torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=4e-3)
    assert lse.shape == (b, h_q, s_q)
    assert lse.dtype == torch.float32


@pytest.mark.L0
@pytest.mark.parametrize("d", [64, 256], ids=["generic-kernel", "d256-kernel"])
@pytest.mark.parametrize("s_q", [128, 100], ids=["tile-aligned", "unaligned"])
def test_sdpa_fwd_sm80_padded_row_lse_trim(d, s_q):
    """Padded query rows (q >= seq_len_q[b]) must read LSE = -inf in BOTH
    store paths: the is_even_mn fast path (tile-aligned S_q) used to skip the
    trim, leaving finite LSE on padded rows — caught by
    test_mhas_v2::test_sdpa_random_bwd_L0 once a random config combined a
    tile-aligned S_q with bottom-right padding."""
    try:
        from cudnn.sdpa.fwd import sdpa_fwd_wrapper_sm80
    except ImportError as e:
        pytest.skip(f"SM80 SDPA API not available: {e}")

    b, h = 4, 2
    torch.manual_seed(0)
    seq_len_q = torch.tensor([s_q, 3 * s_q // 4, s_q // 2, 5], dtype=torch.int32, device="cuda")
    seq_kv_lens = torch.tensor([s_q // 4, s_q, s_q // 5, 3 * s_q // 5], dtype=torch.int32, device="cuda")

    out = sdpa_fwd_wrapper_sm80(
        _bshd_randn(b, h, s_q, d, dtype=torch.float16, device="cuda"),
        _bshd_randn(b, h, s_q, d, dtype=torch.float16, device="cuda"),
        _bshd_randn(b, h, s_q, d, dtype=torch.float16, device="cuda"),
        scale_softmax=1.0 / math.sqrt(d),
        is_causal=True,
        causal_bottom_right=True,
        seq_kv_lens=seq_kv_lens,
        seq_len_q=seq_len_q,
    )
    lse = out["lse_tensor"]
    for bi in range(b):
        tail = lse[bi, :, seq_len_q[bi] :]
        assert torch.isinf(tail).all() and (tail < 0).all(), f"batch {bi}: padded rows lack the -inf LSE trim"
        head = lse[bi, :, : seq_len_q[bi]]
        assert torch.isfinite(head).any(), f"batch {bi}: valid rows unexpectedly all -inf"


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sdpa_fwd_sm80_check_support_rejections():
    """check_support rejects out-of-envelope heads and bad layouts eagerly."""
    try:
        from cudnn.sdpa.fwd import SdpaFwdDslSm80
    except ImportError as e:
        pytest.skip(f"SM80 SDPA API not available: {e}")

    b, h, s = 1, 4, 128
    q = torch.zeros((b, s, h, 512), dtype=torch.float16, device="cuda").permute(0, 2, 1, 3)
    lse = torch.zeros((b, h, s), dtype=torch.float32, device="cuda")
    api = SdpaFwdDslSm80(sample_q=q, sample_k=q, sample_v=q, sample_o=q, sample_lse=lse)
    with pytest.raises(ValueError, match="exceeds"):
        api.check_support()

    q64 = torch.zeros((b, s, h, 64), dtype=torch.float32, device="cuda").permute(0, 2, 1, 3)
    lse64 = torch.zeros((b, h, s), dtype=torch.float32, device="cuda")
    api = SdpaFwdDslSm80(sample_q=q64, sample_k=q64, sample_v=q64, sample_o=q64, sample_lse=lse64)
    with pytest.raises(ValueError, match="dtype"):
        api.check_support()
