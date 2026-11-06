# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX NSA forward ops: parity with the PyTorch wrappers, which run the same kernels."""

import numpy as np
import pytest
import torch

from native_sparse_attention.cutedsl.nsa_utils import generate_block_indices

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


def jax_nsa(capabilities):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in capabilities:
        pytest.skip(f"requires compute capability in {capabilities}")
    from cudnn.frost.buffers import cutedsl_requirement_error

    requirement_error = cutedsl_requirement_error("JAX NSA tests")
    if requirement_error:
        pytest.skip(requirement_error)
    jax = pytest.importorskip("jax", minversion="0.9.1")
    from cudnn.ops import nsa

    return jax, nsa


def to_jax(t):
    import jax.numpy as jnp

    if t.is_floating_point():
        return jnp.asarray(t.float().cpu().numpy(), getattr(jnp, str(t.dtype).split(".")[-1]))
    return jnp.asarray(t.cpu().numpy())


def offsets(lens):
    return torch.tensor(np.concatenate([[0], np.cumsum(lens)]), dtype=torch.int32, device="cuda")


def assert_same(jax_array, torch_tensor):
    np.testing.assert_array_equal(np.asarray(jax_array, np.float32), torch_tensor.float().cpu().numpy())


@pytest.mark.parametrize(
    "dtype,is_causal,heads,head_dim,block_size",
    [(torch.bfloat16, False, (4, 1), 128, 64), (torch.float16, True, (16, 2), 64, 32), (torch.bfloat16, False, (2, 2), 192, 64)],
)
def test_selection_attention_jax_matches_torch(dtype, is_causal, heads, head_dim, block_size):
    jax, nsa = jax_nsa(((9, 0), (10, 0), (10, 3)))
    from cudnn import NSA

    torch.manual_seed(0)
    lens = [384, 256]
    h_q, h_kv = heads
    cu = offsets(lens)
    q = torch.randn(sum(lens), h_q, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(sum(lens), h_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(sum(lens), h_kv, 128, device="cuda", dtype=dtype)
    counts, indices = generate_block_indices(lens, h_kv, [4, 4], block_size)
    expected = NSA.selection_attention_wrapper(
        q_tensor=q,
        k_tensor=k,
        v_tensor=v,
        block_indices_tensor=indices,
        block_counts_tensor=counts,
        cum_seqlen_q_tensor=cu,
        cum_seqlen_k_tensor=cu,
        block_size=block_size,
        is_causal=is_causal,
    )
    run = jax.jit(lambda *a: nsa.selection_attention_forward_jax(*a, max(lens), block_size, is_causal=is_causal))
    result = run(*(to_jax(t) for t in (q, k, v, indices, counts, cu)))
    for name, tensor in zip(("o_tensor", "l_tensor", "m_tensor"), expected):
        assert_same(result[name], tensor)


@pytest.mark.parametrize("dtype,heads,head_dim,lens", [(torch.bfloat16, (4, 1), 128, [1024, 1024]), (torch.float16, (8, 2), 64, [1024, 512])])
def test_compression_attention_jax_matches_torch(dtype, heads, head_dim, lens):
    jax, nsa = jax_nsa(((10, 0), (10, 3)))
    from cudnn import NSA

    torch.manual_seed(0)
    h_q, h_kv = heads
    cu = offsets(lens)
    q = torch.randn(sum(lens), h_q, head_dim, device="cuda", dtype=dtype)
    k, v = (torch.randn(sum(lens), h_kv, head_dim, device="cuda", dtype=dtype) for _ in range(2))
    o, lse = NSA.compression_attention_wrapper(q_tensor=q, k_tensor=k, v_tensor=v, cum_seqlen_q_tensor=cu, cum_seqlen_k_tensor=cu, enable_lse=True)
    result = jax.jit(nsa.compression_attention_forward_jax)(*(to_jax(t) for t in (q, k, v, cu, cu)))
    assert_same(result["o_tensor"], o)
    assert_same(result["lse_tensor"], lse)


@pytest.mark.parametrize("dtype,is_causal,lens,k_lens", [(torch.bfloat16, True, [4096, 4096], [128, 128]), (torch.float16, False, [4096, 2048], [128, 64])])
def test_top_k_reduction_jax_matches_torch(dtype, is_causal, lens, k_lens):
    jax, nsa = jax_nsa(((10, 0), (10, 3)))
    from cudnn import NSA

    torch.manual_seed(0)
    h_q, h_kv, head_dim = 4, 1, 128
    cu_q, cu_k = offsets(lens), offsets(k_lens)
    q = torch.randn(sum(lens), h_q, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(sum(k_lens), h_kv, head_dim, device="cuda", dtype=dtype)
    # (T, H_q, 1) stored head-major, as the compression wrapper returns it.
    lse = (-torch.randn(1, h_q, sum(lens), device="cuda")).permute(2, 1, 0)
    scores, indices = NSA.topk_reduction_wrapper(
        q_tensor=q, k_tensor=k, lse_tensor=lse, cum_seqlen_q_tensor=cu_q, cum_seqlen_k_tensor=cu_k, max_s_q=max(lens), max_s_k=max(k_lens), is_causal=is_causal
    )
    run = jax.jit(lambda *a: nsa.top_k_reduction_forward_jax(*a, max(lens), max(k_lens), is_causal=is_causal))
    result = run(*(to_jax(t) for t in (q, k, lse.contiguous(), cu_q, cu_k)))
    assert_same(result["topk_scores_tensor"], scores)
    assert_same(result["topk_indices_tensor"], indices)


def test_nsa_jax_rejects_unsupported_inputs():
    jax, nsa = jax_nsa(((10, 0), (10, 3)))
    import jax.numpy as jnp

    q = jnp.zeros((256, 4, 128), jnp.bfloat16)
    kv = jnp.zeros((256, 1, 128), jnp.bfloat16)
    cu = jnp.asarray([0, 256], jnp.int32)
    with pytest.raises(TypeError, match="must be a JAX array"):
        nsa.compression_attention_forward_jax(torch.zeros(1), kv, kv, cu, cu)
    with pytest.raises(ValueError, match="head dimension"):
        wide = jnp.zeros((256, 1, 256), jnp.bfloat16)
        nsa.compression_attention_forward_jax(jnp.zeros((256, 4, 256), jnp.bfloat16), wide, wide, cu, cu)
    with pytest.raises(ValueError, match="block_size"):
        nsa.selection_attention_forward_jax(q, kv, kv, jnp.zeros((256, 1, 4), jnp.int32), jnp.zeros((256, 1), jnp.int32), cu, 256, block_size=48)
