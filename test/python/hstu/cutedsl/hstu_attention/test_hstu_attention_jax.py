# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX HSTU attention: parity with the PyTorch API, which runs the same kernels."""

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


def jax_ops():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("JAX HSTU attention requires SM100 or SM103")
    from cudnn.frost.buffers import cutedsl_requirement_error

    requirement_error = cutedsl_requirement_error("JAX HSTU attention tests")
    if requirement_error:
        pytest.skip(requirement_error)
    jax = pytest.importorskip("jax", minversion="0.9.1")
    import cudnn.ops

    return jax, cudnn.ops


def packed_inputs(lens, heads, head_dim, seed=0):
    rng = np.random.default_rng(seed)
    cu_seqlens = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    q, k, v, do = (rng.standard_normal((int(cu_seqlens[-1]), heads, head_dim)).astype(np.float32) for _ in range(4))
    return q, k, v, do, cu_seqlens


@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize("window_size", [(-1, 0), (-1, -1), (16, 0)], ids=["causal", "full", "local"])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_hstu_attention_jax_matches_torch(head_dim, window_size, dtype):
    jax, ops = jax_ops()
    import jax.numpy as jnp

    lens = (100, 200, 37)
    q, k, v, do, cu = packed_inputs(lens, 2, head_dim)
    alpha, max_seqlen = head_dim**-0.5, max(lens)
    tq, tk, tv, tdo = (torch.tensor(x, device="cuda", dtype=getattr(torch, dtype)) for x in (q, k, v, do))
    tcu = torch.tensor(cu, device="cuda")
    from cudnn.hstu.hstu_attention import hstu_attention_backward, hstu_attention_forward

    expected_o = hstu_attention_forward(tq, tk, tv, tcu, tcu, max_seqlen, max_seqlen, window_size, alpha)["o_tensor"]
    expected = hstu_attention_backward(tdo, tq, tk, tv, tcu, tcu, max_seqlen, max_seqlen, window_size, alpha)

    jq, jk, jv, jdo = (jnp.asarray(x, getattr(jnp, dtype)) for x in (q, k, v, do))
    jcu = jnp.asarray(cu)
    forward = jax.jit(lambda *a: ops.hstu_attention_forward_jax(*a, max_seqlen, max_seqlen, window_size, alpha))
    backward = jax.jit(lambda *a: ops.hstu_attention_backward_jax(*a, max_seqlen, max_seqlen, window_size, alpha))
    o = forward(jq, jk, jv, jcu, jcu)["o_tensor"]
    grads = backward(jdo, jq, jk, jv, jcu, jcu)

    np.testing.assert_array_equal(np.asarray(o, np.float32), expected_o.float().cpu().numpy())
    for name in ("dq_tensor", "dk_tensor", "dv_tensor"):
        np.testing.assert_allclose(np.asarray(grads[name], np.float32), expected[name].float().cpu().numpy(), atol=1e-2, rtol=1e-2)


def test_hstu_attention_jax_rejects_unsupported_inputs():
    jax, ops = jax_ops()
    import jax.numpy as jnp

    q, k, v, _, cu = packed_inputs((64, 64), 2, 64)
    jq, jk, jv = (jnp.asarray(x, jnp.bfloat16) for x in (q, k, v))
    jcu = jnp.asarray(cu)
    with pytest.raises(TypeError, match="must be a JAX array"):
        ops.hstu_attention_forward_jax(torch.zeros(1), jk, jv, jcu, jcu, 64, 64)
    with pytest.raises(ValueError, match="max_seqlen_q == 1"):
        ops.hstu_attention_forward_jax(jq, jk, jv, jcu, jcu, 1, 64)
    with pytest.raises(ValueError, match="head_dim"):
        wide = jnp.zeros((128, 2, 256), jnp.bfloat16)
        ops.hstu_attention_forward_jax(wide, wide, wide, jcu, jcu, 64, 64)
    with pytest.raises(TypeError, match="bfloat16 or float16"):
        ops.hstu_attention_forward_jax(jq.astype(jnp.float32), jk, jv, jcu, jcu, 64, 64)
