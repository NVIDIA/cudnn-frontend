# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp


def _require_sm100():
    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("JAX has no CUDA device")
    from cudnn.tensor_adapter import get_compute_capability
    if get_compute_capability()[0] < 10:
        pytest.skip("CSA CuTeDSL kernels require compute capability >= 10")


@pytest.mark.L0
@pytest.mark.parametrize("head_dim", [128, 512])
def test_csa_compressor_jax_matches_overlap_reference(head_dim):
    _require_sm100()
    from cudnn import csa_compressor_backward_jax_sm100, csa_compressor_forward_jax_sm100

    key = jax.random.key(head_dim)
    kv = jax.random.normal(key, (16, 2 * head_dim), jnp.bfloat16)
    score = jax.random.normal(jax.random.fold_in(key, 1), kv.shape, jnp.bfloat16)
    ape = jax.random.normal(jax.random.fold_in(key, 2), (4, 2 * head_dim), jnp.float32)
    cu = jnp.array([0, 16], jnp.int32)
    cu_comp = jnp.array([0, 4], jnp.int32)

    got = jax.jit(
        lambda x, s, a: csa_compressor_forward_jax_sm100(x, s, a, cu, cu_comp, total_comp=4)
    )(kv, score, ape)
    def reference(x, scores, bias):
        a_kv, b_kv = jnp.split(x.reshape(4, 4, 2 * head_dim), 2, axis=-1)
        a_score, b_score = jnp.split(scores.reshape(4, 4, 2 * head_dim) + bias, 2, axis=-1)
        shifted_kv = jnp.concatenate([jnp.zeros_like(a_kv[:1]), a_kv[:-1]], axis=0)
        shifted_score = jnp.concatenate([jnp.full_like(a_score[:1], -jnp.inf), a_score[:-1]], axis=0)
        values = jnp.concatenate([shifted_kv, b_kv], axis=1)
        logits = jnp.concatenate([shifted_score, b_score], axis=1)
        return jnp.sum(values * jax.nn.softmax(logits.astype(jnp.float32), axis=1).astype(values.dtype), axis=1)

    ref = reference(kv, score, ape)
    np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(ref, np.float32), atol=8e-3, rtol=8e-3)

    grad_kv, grad_score, grad_ape = csa_compressor_backward_jax_sm100(
        kv, score, ape, cu, cu_comp, jnp.ones_like(got)
    )
    ref_grad_kv, ref_grad_score, ref_grad_ape = jax.grad(
        lambda x, s, a: reference(x, s, a).astype(jnp.float32).sum(), argnums=(0, 1, 2)
    )(kv, score, ape)
    np.testing.assert_allclose(np.asarray(grad_kv, np.float32), np.asarray(ref_grad_kv, np.float32), atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(grad_score, np.float32), np.asarray(ref_grad_score, np.float32), atol=1e-7, rtol=1e-6
    )
    np.testing.assert_allclose(np.asarray(grad_ape), np.asarray(ref_grad_ape), atol=1e-6, rtol=1e-6)
