"""
Tests for the cuDNN SDPA PyTorch custom operator (cudnn.experimental.ops.sdpa).

Each test runs both forward and backward, verifying output shapes, dtypes,
numerical correctness against a PyTorch reference for both O and dQ/dK/dV.

All tensors use BHSD layout (batch, num_heads, seq_len, head_dim).
"""

import pytest
import torch
import math

import cudnn
from cudnn.experimental.ops.sdpa import scaled_dot_product_attention, _fprop_cache, _bprop_cache

# ---------------------------------------------------------------------------
# PyTorch reference implementation (differentiable)
# ---------------------------------------------------------------------------


def sdpa_reference_fwd_bwd(
    q,
    k,
    v,
    attn_scale=None,
    is_causal=False,
    diagonal_alignment=0,
    left_bound=-1,
    right_bound=-1,
    seq_len_q=None,
    seq_len_kv=None,
):
    """
    Pure-PyTorch differentiable SDPA reference in BHSD layout.

    Runs forward and backward (via .sum().backward()) and returns (o, dq, dk, dv).
    All computation in float32 for numerical stability.

    Args:
        q: (B, H_q, S_q, D_qk)  — will be cloned with requires_grad
        k: (B, H_k, S_kv, D_qk)
        v: (B, H_v, S_kv, D_v)
    Returns:
        (o, dq, dk, dv) all in the original dtype
    """
    if attn_scale is None:
        attn_scale = 1.0 / math.sqrt(q.shape[-1])

    # Clone to float32 with grad tracking
    q_ref = q.detach().float().requires_grad_(True)
    k_ref = k.detach().float().requires_grad_(True)
    v_ref = v.detach().float().requires_grad_(True)

    B, H_q, S_q, D_qk = q_ref.shape
    _, H_k, S_kv, _ = k_ref.shape
    _, H_v, _, D_v = v_ref.shape

    q_t = q_ref
    k_t = k_ref
    v_t = v_ref

    # Expand for GQA/MQA
    if H_q != H_k:
        assert H_q % H_k == 0
        k_t = k_t.unsqueeze(2).expand(-1, -1, H_q // H_k, -1, -1).reshape(B, H_q, S_kv, D_qk)
    if H_q != H_v:
        assert H_q % H_v == 0
        v_t = v_t.unsqueeze(2).expand(-1, -1, H_q // H_v, -1, -1).reshape(B, H_q, S_kv, D_v)

    # Attention scores
    s = torch.einsum("bhqd,bhkd->bhqk", q_t, k_t) * attn_scale

    # Causal / sliding window mask
    rb = right_bound if right_bound >= 0 else None
    lb = left_bound if left_bound >= 0 else None
    if is_causal and rb is None:
        rb = 0

    if rb is not None:
        if diagonal_alignment == 0:  # TOP_LEFT
            causal_mask = torch.ones(S_q, S_kv, dtype=torch.bool, device=q.device)
            causal_mask.triu_(diagonal=1 + rb)
        else:  # BOTTOM_RIGHT
            if seq_len_q is not None and seq_len_kv is not None:
                causal_mask = torch.ones(B, 1, S_q, S_kv, dtype=torch.bool, device=q.device)
                sl_q = seq_len_q.flatten()
                sl_kv = seq_len_kv.flatten()
                for i in range(B):
                    causal_mask[i, :, :, :].triu_(diagonal=int(sl_kv[i]) - int(sl_q[i]) + 1 + rb)
            else:
                causal_mask = torch.ones(S_q, S_kv, dtype=torch.bool, device=q.device)
                causal_mask.triu_(diagonal=S_kv - S_q + 1 + rb)
        s = s.masked_fill(causal_mask, float("-inf"))

    if lb is not None:
        if diagonal_alignment == 0:  # TOP_LEFT
            swa_mask = torch.ones(S_q, S_kv, dtype=torch.bool, device=q.device)
            swa_mask.tril_(diagonal=-1 * lb)
        else:  # BOTTOM_RIGHT
            if seq_len_q is not None and seq_len_kv is not None:
                swa_mask = torch.ones(B, 1, S_q, S_kv, dtype=torch.bool, device=q.device)
                sl_q = seq_len_q.flatten()
                sl_kv = seq_len_kv.flatten()
                for i in range(B):
                    swa_mask[i, :, :, :].tril_(diagonal=int(sl_kv[i]) - int(sl_q[i]) - lb)
            else:
                swa_mask = torch.ones(S_q, S_kv, dtype=torch.bool, device=q.device)
                swa_mask.tril_(diagonal=-1 * lb + (S_kv - S_q))
        s = s.masked_fill(swa_mask, float("-inf"))

    # Padding mask on scores
    if seq_len_kv is not None:
        sl_kv = seq_len_kv.flatten()
        s_mask = torch.zeros(B, 1, S_q, S_kv, dtype=torch.bool, device=q.device)
        for i in range(B):
            s_mask[i, :, :, sl_kv[i] :] = True
        s = s.masked_fill(s_mask, float("-inf"))

    p = torch.softmax(s, dim=-1)

    # Padding mask on probabilities
    if seq_len_q is not None:
        sl_q = seq_len_q.flatten()
        p_mask = torch.zeros(B, 1, S_q, S_kv, dtype=torch.bool, device=q.device)
        for i in range(B):
            p_mask[i, :, sl_q[i] :, :] = True
        p = p.masked_fill(p_mask, 0.0)

    o = torch.einsum("bhqk,bhkd->bhqd", p, v_t)

    # Backward
    o.sum().backward()

    return (
        o.detach().to(q.dtype),
        q_ref.grad.detach().to(q.dtype),
        k_ref.grad.detach().to(k.dtype),
        v_ref.grad.detach().to(v.dtype),
    )


def _skip_if_unsupported_d256(D):
    if D != 256:
        return
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip("d=256 backward path requires SM100+")
    try:
        import cudnn.sdpa  # noqa: F401
    except ImportError:
        pytest.skip("d=256 OSS SDPA path requires optional cutedsl dependencies in this environment")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCudnnSdpa:
    """Combined forward + backward tests with numerical gradient verification."""

    @pytest.mark.L0
    @pytest.mark.parametrize("D", [128, 256])
    def test_basic(self, D):
        """Basic forward + backward, no masking."""
        _skip_if_unsupported_d256(D)
        B, H, S = 2, 8, 128

        q = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)

        o = scaled_dot_product_attention(q, k, v)
        assert o.shape == (B, H, S, D)
        assert o.dtype == torch.float16

        loss = o.sum()
        loss.backward()

        # Reference
        o_ref, dq_ref, dk_ref, dv_ref = sdpa_reference_fwd_bwd(q, k, v)

        torch.testing.assert_close(o.float(), o_ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(q.grad.float(), dq_ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(k.grad.float(), dk_ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(v.grad.float(), dv_ref.float(), atol=2e-2, rtol=2e-2)

    @pytest.mark.L0
    def test_d256_uses_oss_forward_path(self):
        _skip_if_unsupported_d256(256)
        try:
            import cudnn.sdpa  # noqa: F401
        except ImportError:
            pytest.skip("OSS SDPA d=256 optional dependencies are not installed in this environment")

        B, H, S, D = 2, 8, 128, 256
        _fprop_cache.clear()

        q = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)

        o = scaled_dot_product_attention(q, k, v)
        loss = o.sum()
        loss.backward()

        o_ref, dq_ref, dk_ref, dv_ref = sdpa_reference_fwd_bwd(q, k, v)

        assert len(_fprop_cache) == 0, "D=256 OSS forward path should bypass the cuDNN graph cache"
        torch.testing.assert_close(o.float(), o_ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(q.grad.float(), dq_ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(k.grad.float(), dk_ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(v.grad.float(), dv_ref.float(), atol=2e-2, rtol=2e-2)

    @pytest.mark.L0
    @pytest.mark.parametrize("D", [128])
    def test_basic_bf16(self, D):
        """BFloat16 forward + backward."""
        B, H, S = 2, 4, 64

        q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        k = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        v = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)

        o = scaled_dot_product_attention(q, k, v)
        assert o.shape == (B, H, S, D)
        assert o.dtype == torch.bfloat16

        loss = o.sum()
        loss.backward()

        o_ref, dq_ref, dk_ref, dv_ref = sdpa_reference_fwd_bwd(q, k, v)

        torch.testing.assert_close(o.float(), o_ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(q.grad.float(), dq_ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(k.grad.float(), dk_ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(v.grad.float(), dv_ref.float(), atol=2e-2, rtol=2e-2)

    @pytest.mark.L0
    @pytest.mark.parametrize("D", [128])
    def test_causal_top_left(self, D):
        """Causal mask with TOP_LEFT alignment, forward + backward."""
        B, H, S = 2, 4, 128

        q = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)

        o = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        assert o.shape == (B, H, S, D)

        loss = o.sum()
        loss.backward()

        o_ref, dq_ref, dk_ref, dv_ref = sdpa_reference_fwd_bwd(q, k, v, is_causal=True, diagonal_alignment=0)

        torch.testing.assert_close(o.float(), o_ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(q.grad.float(), dq_ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(k.grad.float(), dk_ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(v.grad.float(), dv_ref.float(), atol=2e-2, rtol=2e-2)

    @pytest.mark.L0
    @pytest.mark.parametrize("D", [128])
    def test_causal_bottom_right_with_padding(self, D):
        """BOTTOM_RIGHT causal with variable sequence lengths, forward + backward."""
        B, H, S = 2, 4, 128

        q = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)

        seq_len_q = torch.tensor([[64], [96]], dtype=torch.int32, device="cuda").reshape(B, 1, 1, 1)
        seq_len_kv = torch.tensor([[80], [128]], dtype=torch.int32, device="cuda").reshape(B, 1, 1, 1)

        o = scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            enable_gqa=True,
            diagonal_alignment=1,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
        )
        assert o.shape == (B, H, S, D)

        loss = o.sum()
        loss.backward()

        o_ref, dq_ref, dk_ref, dv_ref = sdpa_reference_fwd_bwd(
            q,
            k,
            v,
            is_causal=True,
            diagonal_alignment=1,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
        )

        # Zero out padded regions for comparison (seq dim is index 2 in BHSD)
        o_cmp = o.detach().clone()
        dq_cmp = q.grad.detach().clone()
        for i in range(B):
            m = seq_len_q[i].item()
            o_cmp[i, :, m:, :] = 0
            o_ref[i, :, m:, :] = 0
            dq_cmp[i, :, m:, :] = 0
            dq_ref[i, :, m:, :] = 0

        torch.testing.assert_close(o_cmp.float(), o_ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(dq_cmp.float(), dq_ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(k.grad.float(), dk_ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(v.grad.float(), dv_ref.float(), atol=2e-2, rtol=2e-2)

    @pytest.mark.L0
    @pytest.mark.parametrize("D", [128])
    def test_sliding_window(self, D):
        """Sliding window attention, forward + backward."""
        B, H, S = 2, 4, 256

        q = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)

        o = scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            enable_gqa=True,
            diagonal_alignment=0,
            left_bound=32,
            right_bound=0,
        )
        assert o.shape == (B, H, S, D)

        loss = o.sum()
        loss.backward()

        o_ref, dq_ref, dk_ref, dv_ref = sdpa_reference_fwd_bwd(
            q,
            k,
            v,
            diagonal_alignment=0,
            left_bound=32,
            right_bound=0,
        )

        torch.testing.assert_close(o.float(), o_ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(q.grad.float(), dq_ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(k.grad.float(), dk_ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(v.grad.float(), dv_ref.float(), atol=2e-2, rtol=2e-2)

    @pytest.mark.L0
    @pytest.mark.parametrize("D", [128])
    def test_variable_sequence_lengths(self, D):
        """Padding mask with actual sequence lengths, forward + backward."""
        B, H, S = 2, 4, 128

        q = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)

        seq_len_q = torch.tensor([[64], [100]], dtype=torch.int32, device="cuda").reshape(B, 1, 1, 1)
        seq_len_kv = torch.tensor([[80], [128]], dtype=torch.int32, device="cuda").reshape(B, 1, 1, 1)

        o = scaled_dot_product_attention(
            q,
            k,
            v,
            enable_gqa=True,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
        )
        assert o.shape == (B, H, S, D)

        loss = o.sum()
        loss.backward()

        o_ref, dq_ref, dk_ref, dv_ref = sdpa_reference_fwd_bwd(
            q,
            k,
            v,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
        )

        # Zero out padded regions for comparison (seq dim is index 2 in BHSD)
        o_cmp = o.detach().clone()
        dq_cmp = q.grad.detach().clone()
        for i in range(B):
            m = seq_len_q[i].item()
            o_cmp[i, :, m:, :] = 0
            o_ref[i, :, m:, :] = 0
            dq_cmp[i, :, m:, :] = 0
            dq_ref[i, :, m:, :] = 0

        torch.testing.assert_close(o_cmp.float(), o_ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(dq_cmp.float(), dq_ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(k.grad.float(), dk_ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(v.grad.float(), dv_ref.float(), atol=2e-2, rtol=2e-2)


class TestCudnnSdpaGQA:
    """Grouped Query Attention tests."""

    @pytest.mark.L0
    @pytest.mark.parametrize("D", [128])
    def test_gqa(self, D):
        """GQA forward + backward with H_q > H_k = H_v."""
        B, S = 2, 64
        H_q, H_k, H_v = 8, 2, 2

        q = torch.randn(B, H_q, S, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, H_k, S, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, H_v, S, D, dtype=torch.float16, device="cuda", requires_grad=True)

        o = scaled_dot_product_attention(q, k, v, enable_gqa=True)
        assert o.shape == (B, H_q, S, D)

        loss = o.sum()
        loss.backward()

        o_ref, dq_ref, dk_ref, dv_ref = sdpa_reference_fwd_bwd(q, k, v)

        torch.testing.assert_close(o.float(), o_ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(q.grad.float(), dq_ref.float(), atol=2e-2, rtol=2e-2)
        # GQA: dK/dV reference grads are summed across the GQA groups by autograd,
        # so shapes match k/v directly
        torch.testing.assert_close(k.grad.float(), dk_ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(v.grad.float(), dv_ref.float(), atol=2e-2, rtol=2e-2)


class TestCudnnSdpaCaching:
    """Graph caching tests."""

    @pytest.mark.L0
    def test_fprop_cache_reuse(self):
        """Same config should reuse the cached graph."""
        B, H, S, D = 2, 4, 64, 64

        q = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")
        k = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")
        v = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")

        initial_cache_size = len(_fprop_cache)

        scaled_dot_product_attention(q, k, v)
        after_first = len(_fprop_cache)
        assert after_first == initial_cache_size + 1

        q2 = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")
        k2 = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")
        v2 = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")
        scaled_dot_product_attention(q2, k2, v2)
        after_second = len(_fprop_cache)
        assert after_second == after_first, "Cache should be reused for same config"

    @pytest.mark.L0
    def test_different_shapes_create_new_entry(self):
        """Different shapes should create a new cache entry."""
        D = 64

        q1 = torch.randn(1, 4, 32, D, dtype=torch.float16, device="cuda")
        k1 = torch.randn(1, 4, 32, D, dtype=torch.float16, device="cuda")
        v1 = torch.randn(1, 4, 32, D, dtype=torch.float16, device="cuda")

        q2 = torch.randn(1, 4, 64, D, dtype=torch.float16, device="cuda")
        k2 = torch.randn(1, 4, 64, D, dtype=torch.float16, device="cuda")
        v2 = torch.randn(1, 4, 64, D, dtype=torch.float16, device="cuda")

        initial = len(_fprop_cache)
        scaled_dot_product_attention(q1, k1, v1)
        scaled_dot_product_attention(q2, k2, v2)
        assert len(_fprop_cache) == initial + 2, "Different shapes should create exactly 2 cache entries"


class TestCudnnSdpaAPIValidation:
    """API validation tests."""

    @pytest.mark.L0
    def test_attn_mask_not_supported(self):
        """attn_mask should raise NotImplementedError."""
        q = torch.randn(1, 4, 16, 64, dtype=torch.float16, device="cuda")
        k = torch.randn(1, 4, 16, 64, dtype=torch.float16, device="cuda")
        v = torch.randn(1, 4, 16, 64, dtype=torch.float16, device="cuda")
        mask = torch.ones(1, 1, 16, 16, dtype=torch.float16, device="cuda")

        with pytest.raises(NotImplementedError, match="attn_mask"):
            scaled_dot_product_attention(q, k, v, attn_mask=mask)

    @pytest.mark.L0
    def test_dropout_not_supported(self):
        """dropout_p > 0 should raise NotImplementedError."""
        q = torch.randn(1, 4, 16, 64, dtype=torch.float16, device="cuda")
        k = torch.randn(1, 4, 16, 64, dtype=torch.float16, device="cuda")
        v = torch.randn(1, 4, 16, 64, dtype=torch.float16, device="cuda")

        with pytest.raises(NotImplementedError, match="dropout"):
            scaled_dot_product_attention(q, k, v, dropout_p=0.1)

    @pytest.mark.L0
    def test_enable_gqa_validation(self):
        """enable_gqa=False with mismatched heads should raise ValueError."""
        q = torch.randn(1, 8, 16, 64, dtype=torch.float16, device="cuda")
        k = torch.randn(1, 2, 16, 64, dtype=torch.float16, device="cuda")
        v = torch.randn(1, 2, 16, 64, dtype=torch.float16, device="cuda")

        with pytest.raises(ValueError, match="enable_gqa"):
            scaled_dot_product_attention(q, k, v, enable_gqa=False)

        # Should work with enable_gqa=True
        o = scaled_dot_product_attention(q, k, v, enable_gqa=True)
        assert o.shape == (1, 8, 16, 64)

    @pytest.mark.L0
    def test_default_scale(self):
        """Default scale should be 1/sqrt(D)."""
        B, H, S, D = 1, 2, 32, 64

        q = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")
        k = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")
        v = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")

        # Default scale
        o1 = scaled_dot_product_attention(q, k, v)
        # Explicit scale = 1/sqrt(D)
        o2 = scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(D))

        torch.testing.assert_close(o1, o2)


class TestCudnnSdpaTorchCompile:
    """Tests for torch.compile compatibility."""

    @pytest.mark.L0
    def test_torch_compile_forward(self):
        """torch.compile should work for forward pass."""
        B, H, S, D = 2, 4, 64, 128

        compiled_sdpa = torch.compile(scaled_dot_product_attention, fullgraph=True)

        q = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")
        k = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")
        v = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")

        # Eager
        o_eager = scaled_dot_product_attention(q, k, v)

        # Compiled
        o_compiled = compiled_sdpa(q, k, v)

        torch.testing.assert_close(o_eager, o_compiled)

    @pytest.mark.L0
    @pytest.mark.parametrize("D", [128, 256])
    def test_torch_compile_backward(self, D):
        """torch.compile should work for forward + backward pass."""
        _skip_if_unsupported_d256(D)
        B, H, S = 2, 4, 64

        compiled_sdpa = torch.compile(scaled_dot_product_attention, fullgraph=True)

        q = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)

        # Eager forward + backward
        o_eager = scaled_dot_product_attention(q, k, v)
        o_eager.sum().backward()
        dq_eager = q.grad.clone()
        dk_eager = k.grad.clone()
        dv_eager = v.grad.clone()

        q.grad, k.grad, v.grad = None, None, None

        # Compiled forward + backward
        o_compiled = compiled_sdpa(q, k, v)
        o_compiled.sum().backward()

        torch.testing.assert_close(o_eager, o_compiled)
        torch.testing.assert_close(dq_eager, q.grad)
        torch.testing.assert_close(dk_eager, k.grad)
        torch.testing.assert_close(dv_eager, v.grad)

    @pytest.mark.L0
    def test_torch_compile_causal(self):
        """torch.compile with causal masking."""
        B, H, S, D = 2, 4, 128, 128

        compiled_sdpa = torch.compile(scaled_dot_product_attention, fullgraph=True)

        q = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)

        # Eager
        o_eager = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        o_eager.sum().backward()
        dq_eager = q.grad.clone()

        q.grad, k.grad, v.grad = None, None, None

        # Compiled
        o_compiled = compiled_sdpa(q, k, v, is_causal=True, enable_gqa=True)
        o_compiled.sum().backward()

        torch.testing.assert_close(o_eager, o_compiled)
        torch.testing.assert_close(dq_eager, q.grad)
