# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX shape and GPU integration tests for NSA wrappers."""

from __future__ import annotations

import pytest


def _jax_dependencies():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("cutlass.jax")
    return jax, jnp


def _gpu_device(jax, minimum_compute_capability):
    for device in jax.local_devices():
        if device.platform != "gpu":
            continue
        capability = getattr(device, "compute_capability", None)
        if capability is None:
            continue
        if isinstance(capability, (tuple, list)):
            major, minor = (int(value) for value in capability[:2])
            capability_number = major * 10 + minor
        elif "." in str(capability):
            major, minor = (int(value) for value in str(capability).split(".", 1))
            capability_number = major * 10 + minor
        else:
            capability_number = int(capability)
            if capability_number < 10:
                capability_number *= 10
        if capability_number >= minimum_compute_capability:
            return device
    pytest.skip(f"JAX SM{minimum_compute_capability}+ device is not available")


@pytest.mark.L0
def test_jax_sliding_window_attention_abstract_shape():
    jax, jnp = _jax_dependencies()

    from cudnn.jax import TupleDict, sliding_window_attention_wrapper

    q = jax.ShapeDtypeStruct((2, 4, 128, 64), jnp.bfloat16)
    k = jax.ShapeDtypeStruct((2, 2, 128, 64), jnp.bfloat16)
    v = jax.ShapeDtypeStruct((2, 2, 128, 64), jnp.bfloat16)
    result = jax.eval_shape(
        lambda q, k, v: sliding_window_attention_wrapper(
            q,
            k,
            v,
            left_bound=16,
            is_infer=True,
        ),
        q,
        k,
        v,
    )
    assert isinstance(result, TupleDict)
    assert result["o_tensor"].shape == q.shape
    assert result["o_tensor"].dtype == jnp.bfloat16
    assert result["stats_tensor"] is None


@pytest.mark.L0
def test_jax_sliding_window_attention_jit():
    jax, jnp = _jax_dependencies()
    device = _gpu_device(jax, 80)

    from cudnn.jax import sliding_window_attention_wrapper

    q_shape = (1, 4, 128, 64)
    kv_shape = (1, 2, 128, 64)
    q = jax.device_put(
        jax.random.normal(jax.random.key(0), q_shape, dtype=jnp.bfloat16),
        device,
    )
    k = jax.device_put(
        jax.random.normal(jax.random.key(1), kv_shape, dtype=jnp.bfloat16),
        device,
    )
    v = jax.device_put(
        jax.random.normal(jax.random.key(2), kv_shape, dtype=jnp.bfloat16),
        device,
    )
    left_bound = 16
    scale = 0.125

    @jax.jit
    def run(q, k, v):
        return sliding_window_attention_wrapper(
            q,
            k,
            v,
            left_bound=left_bound,
            right_bound=0,
            is_infer=True,
            attn_scale=scale,
        )

    lowered = run.lower(q, k, v)
    stablehlo = lowered.as_text("stablehlo")
    assert "stablehlo.custom_call" in stablehlo
    assert "__cudnn$fmha" in stablehlo

    result = lowered.compile()(q, k, v)
    result["o_tensor"].block_until_ready()
    assert result["stats_tensor"] is None

    k_expanded = jnp.repeat(
        k.astype(jnp.float32),
        q.shape[1] // k.shape[1],
        axis=1,
    )
    v_expanded = jnp.repeat(
        v.astype(jnp.float32),
        q.shape[1] // v.shape[1],
        axis=1,
    )
    logits = (
        jnp.einsum(
            "bhqd,bhkd->bhqk",
            q.astype(jnp.float32),
            k_expanded,
        )
        * scale
    )
    q_position = jnp.arange(q.shape[2])[:, None]
    k_position = jnp.arange(k.shape[2])[None, :]
    valid = (k_position > q_position - left_bound) & (k_position <= q_position)
    probability = jax.nn.softmax(
        jnp.where(valid[None, None, :, :], logits, -jnp.inf),
        axis=-1,
    )
    reference = jnp.einsum("bhqk,bhkd->bhqd", probability, v_expanded)
    max_error = jnp.max(
        jnp.abs(result["o_tensor"].astype(jnp.float32) - reference.astype(jnp.float32))
    )
    assert jnp.allclose(
        result["o_tensor"],
        reference,
        atol=3e-2,
        rtol=3e-2,
    ), f"maximum absolute error: {max_error}"
