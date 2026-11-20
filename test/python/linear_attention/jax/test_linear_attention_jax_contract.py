# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

pytest.importorskip("jax", minversion="0.9.1")
import jax.numpy as jnp

from cudnn import ops
from cudnn.linear_attention import _jax_execution

pytestmark = pytest.mark.L0


@pytest.fixture
def forward_residual(monkeypatch):
    q = jnp.zeros((1, 1, 64), jnp.bfloat16)
    g = jnp.zeros((1, 1), jnp.float32)
    cu = jnp.arange(2, dtype=jnp.int32)
    state = jnp.zeros((1, 1, 64, 64), jnp.float32)

    def execute(primals, config, **cotangents):
        assert not cotangents, "invalid inputs reached backward execution"
        return {"O": q, "final_state": state}

    monkeypatch.setattr(_jax_execution, "_execute", execute)

    def prepare(name):
        if name == "gated_delta_net_v2":
            args = (q, q, q, jnp.broadcast_to(g[..., None], q.shape), q, q, cu)
        else:
            args = (q, q, q, g, g, cu)
        forward = getattr(ops, name + "_forward_jax")
        return forward(*args, output_final_state=True)

    return prepare


@pytest.mark.parametrize(
    "backward_name,forward_name",
    [
        ("gated_delta_net", "gated_delta_net_v2"),
        ("gated_delta_net_v2", "gated_delta_net"),
        ("gated_delta_product", "gated_delta_net"),
        ("mamba2", "gated_delta_net"),
    ],
)
def test_backward_rejects_another_operations_residual(forward_residual, backward_name, forward_name):
    output, _, residual = forward_residual(forward_name)
    backward = getattr(ops, backward_name + "_backward_jax")
    with pytest.raises(ValueError, match="requires a matching forward residual"):
        backward(residual, output)


@pytest.mark.parametrize("port", ["doutput", "d_final_state"])
def test_backward_rejects_numpy_cotangents(forward_residual, port):
    output, state, residual = forward_residual("gated_delta_net")
    cotangents = {"doutput": output, "d_final_state": state}
    cotangents[port] = np.zeros(cotangents[port].shape, dtype=cotangents[port].dtype)
    with pytest.raises(TypeError, match=f"{port} must be a JAX array"):
        ops.gated_delta_net_backward_jax(residual, **cotangents)
