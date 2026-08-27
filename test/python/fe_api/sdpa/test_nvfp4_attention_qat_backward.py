# SPDX-License-Identifier: Apache-2.0

"""Numerical and API coverage for Triton NVFP4 QAT attention backward."""

import math

import pytest
import torch
import torch.nn.functional as F

from test_utils import torch_fork_set_rng


def _environment_supported() -> bool:
    """Return whether the current process can execute the Blackwell kernels."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in {(10, 0), (10, 3), (12, 0), (12, 1)}:
        return False
    try:
        import cutlass  # noqa: F401
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _environment_supported(),
    reason="NVFP4 attention QAT backward requires Blackwell, CuTe DSL, and Triton",
)


def _fake_quantize_nvfp4_reference(tensor: torch.Tensor) -> torch.Tensor:
    """Reference E2M1 data with one E4M3 scale per 16 values."""
    logical_cols = tensor.shape[-1]
    padded_cols = (logical_cols + 15) // 16 * 16
    values = tensor.float()
    if padded_cols != logical_cols:
        values = F.pad(values, (0, padded_cols - logical_cols))
    blocks = values.reshape(*values.shape[:-1], padded_cols // 16, 16)
    block_amax = blocks.abs().amax(dim=-1, keepdim=True)
    decode_scale = (block_amax / 6.0).clamp_min(2.0**-9).to(torch.float8_e4m3fn).float()
    scaled = blocks / decode_scale

    positive_levels = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=tensor.device)
    distances = (scaled.abs().unsqueeze(-1) - positive_levels).abs()
    minimum_distance = distances.amin(dim=-1, keepdim=True)
    tied = distances == minimum_distance
    even_codes = torch.arange(positive_levels.numel(), device=tensor.device) % 2 == 0
    even_tie = tied & even_codes
    nearest_code = torch.where(tied.sum(dim=-1) > 1, even_tie.to(torch.int8).argmax(dim=-1), distances.argmin(dim=-1))
    quantized = positive_levels[nearest_code] * scaled.sign()
    dequantized = (quantized * decode_scale).reshape(*values.shape)
    return dequantized[..., :logical_cols].to(tensor.dtype)


def _reference_case(seqlen_q: int, seqlen_kv: int, *, is_causal: bool):
    """Construct one input case and its PyTorch gradient reference."""
    batch, heads, head_dim = 1, 2, 128
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn((batch, heads, seqlen_q, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, seqlen_kv, head_dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    do = torch.randn_like(q)

    fake_q = _fake_quantize_nvfp4_reference(q)
    fake_k = _fake_quantize_nvfp4_reference(k)
    fake_v = _fake_quantize_nvfp4_reference(v)
    scores = torch.matmul(fake_q.float(), fake_k.float().transpose(-1, -2)) * scale
    if is_causal:
        row = torch.arange(seqlen_q, device="cuda")[:, None]
        col = torch.arange(seqlen_kv, device="cuda")[None, :]
        scores = scores.masked_fill(col > row, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probability = torch.softmax(scores, dim=-1)
    high_precision_o = torch.matmul(probability, fake_v.float()).to(torch.bfloat16)

    delta = (high_precision_o.float() * do.float()).sum(dim=-1)
    grad_probability = torch.matmul(do.float(), fake_v.float().transpose(-1, -2))
    grad_score = probability * (grad_probability - delta.unsqueeze(-1))
    grad_score_bf16 = grad_score.to(torch.bfloat16).float()
    dq = torch.matmul(grad_score_bf16, fake_k.float()) * scale
    dk = torch.matmul(grad_score_bf16.transpose(-1, -2), fake_q.float()) * scale
    fake_probability = _fake_quantize_nvfp4_reference(probability).to(torch.bfloat16).float()
    dv = torch.matmul(fake_probability.transpose(-1, -2), do.float())
    return (q, k, v, high_precision_o, do, lse, scale), (dq, dk, dv)


@pytest.mark.L0
@torch_fork_set_rng(seed=31)
def test_nvfp4_attention_qat_backward_wrapper_matches_reference():
    """Match the wrapper outputs against the NVFP4 PyTorch reference."""
    from cudnn import nvfp4_attention_qat_backward

    inputs, expected = _reference_case(64, 64, is_causal=False)
    q, k, v, high_precision_o, do, lse, scale = inputs
    result = nvfp4_attention_qat_backward(
        do,
        q,
        k,
        v,
        high_precision_o,
        lse,
        softmax_scale=scale,
    )

    for actual, reference in zip(result, expected):
        torch.testing.assert_close(actual.float(), reference, rtol=3.0e-2, atol=3.0e-2)


@pytest.mark.L1
@pytest.mark.parametrize(("seqlen_q", "seqlen_kv", "is_causal"), [(37, 45, False), (63, 63, True)])
@torch_fork_set_rng(seed=41)
def test_nvfp4_attention_qat_backward_tails_and_causal(seqlen_q, seqlen_kv, is_causal):
    """Cover non-tile-aligned cross attention and causal attention."""
    from cudnn import nvfp4_attention_qat_backward

    inputs, expected = _reference_case(seqlen_q, seqlen_kv, is_causal=is_causal)
    q, k, v, high_precision_o, do, lse, scale = inputs
    result = nvfp4_attention_qat_backward(
        do,
        q,
        k,
        v,
        high_precision_o,
        lse,
        softmax_scale=scale,
        is_causal=is_causal,
    )

    for actual, reference in zip(result, expected):
        torch.testing.assert_close(actual.float(), reference, rtol=4.0e-2, atol=4.0e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=53)
def test_nvfp4_attention_qat_backward_class_uses_caller_buffers():
    """Write into caller buffers and reject an undersized workspace."""
    from cudnn import Nvfp4AttentionQatBackward

    inputs, _ = _reference_case(32, 48, is_causal=False)
    q, k, v, high_precision_o, do, lse, scale = inputs
    op = Nvfp4AttentionQatBackward(q, k, v, high_precision_o, do, lse, softmax_scale=scale)
    assert op.check_support()
    op.compile()

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    workspace = torch.empty(op.scratch_workspace_bytes(), dtype=torch.uint8, device=q.device)
    op.execute(q, k, v, high_precision_o, do, lse, dq, dk, dv, workspace)
    assert torch.isfinite(dq).all()
    assert torch.isfinite(dk).all()
    assert torch.isfinite(dv).all()

    with pytest.raises(ValueError, match="workspace must contain at least"):
        op.execute(q, k, v, high_precision_o, do, lse, dq, dk, dv, workspace[:-1])


@pytest.mark.L0
@torch_fork_set_rng(seed=59)
def test_nvfp4_attention_qat_backward_zero_and_masked_blocks_are_finite():
    """Keep all-zero and fully masked quantization blocks finite."""
    from cudnn import nvfp4_attention_qat_backward

    shape = (1, 1, 32, 128)
    q = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    high_precision_o = torch.zeros_like(q)
    do = torch.randn_like(q)
    lse = torch.arange(1, shape[2] + 1, device="cuda", dtype=torch.float32).log().view(1, 1, -1)
    result = nvfp4_attention_qat_backward(do, q, k, v, high_precision_o, lse, is_causal=True)

    assert torch.isfinite(result[0]).all()
    assert torch.isfinite(result[1]).all()
    assert torch.isfinite(result[2]).all()
    torch.testing.assert_close(result[0], torch.zeros_like(q))
    torch.testing.assert_close(result[1], torch.zeros_like(k))


@pytest.mark.L0
def test_nvfp4_attention_qat_backward_rejects_unsupported_contracts():
    """Reject unsupported activation dtypes and causal cross attention."""
    from cudnn import Nvfp4AttentionQatBackward

    q = torch.empty((1, 1, 16, 128), device="cuda", dtype=torch.float16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    o = torch.empty_like(q)
    do = torch.empty_like(q)
    lse = torch.empty((1, 1, 16), device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="bfloat16"):
        Nvfp4AttentionQatBackward(q, k, v, o, do, lse).check_support()

    q = q.to(torch.bfloat16)
    k = torch.empty((1, 1, 15, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.empty_like(k)
    o = torch.empty_like(q)
    do = torch.empty_like(q)
    with pytest.raises(ValueError, match="causal QAT backward requires equal"):
        Nvfp4AttentionQatBackward(q, k, v, o, do, lse, is_causal=True).check_support()
