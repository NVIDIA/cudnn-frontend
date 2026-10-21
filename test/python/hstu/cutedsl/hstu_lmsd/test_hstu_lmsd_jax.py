# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX HSTU LMSD: parity with the PyTorch API, which runs the same kernels."""

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


def jax_ops():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("JAX HSTU LMSD requires an SM10x GPU")
    from cudnn.frost.buffers import cutedsl_requirement_error

    requirement_error = cutedsl_requirement_error("JAX HSTU LMSD tests")
    if requirement_error:
        pytest.skip(requirement_error)
    jax = pytest.importorskip("jax", minversion="0.9.1")
    import cudnn.ops

    return jax, cudnn.ops


@pytest.mark.parametrize(
    "rows,hidden,options,compute_dweight",
    [
        (1000, 256, {}, True),
        (777, 960, {}, True),
        (33, 64, {"concat_u": False}, True),
        (500, 512, {"concat_x": False, "apply_u_silu": False}, True),
        (300, 128, {"concat_u": False, "concat_x": False}, False),
    ],
)
def test_hstu_lmsd_jax_matches_torch(rows, hidden, options, compute_dweight):
    jax, ops = jax_ops()
    import jax.numpy as jnp
    from cudnn.hstu.hstu_lmsd import hstu_lmsd_backward, hstu_lmsd_forward

    options = {"apply_u_silu": True, "concat_u": True, "concat_x": True, **options}
    segments = 1 + options["concat_u"] + options["concat_x"]
    rng = np.random.default_rng(1)
    x, u, dy = rng.standard_normal((rows, hidden)), rng.standard_normal((rows, hidden)), rng.standard_normal((rows, segments * hidden))
    weight, bias = rng.standard_normal(hidden), rng.standard_normal(hidden)
    tt = lambda a: torch.tensor(a, device="cuda", dtype=torch.bfloat16)
    jj = lambda a: jnp.asarray(a, jnp.bfloat16)

    forward = hstu_lmsd_forward(tt(x), tt(u), tt(weight), tt(bias), dropout_ratio=0.0, **options)
    backward = hstu_lmsd_backward(
        tt(dy),
        tt(x),
        tt(u),
        tt(weight),
        tt(bias),
        forward["mean_tensor"],
        forward["rstd_tensor"],
        None,
        dropout_ratio=0.0,
        compute_dweight=compute_dweight,
        **options,
    )
    jf = jax.jit(lambda *a: ops.hstu_lmsd_forward_jax(*a, **options))(jj(x), jj(u), jj(weight), jj(bias))
    jb = jax.jit(lambda *a: ops.hstu_lmsd_backward_jax(*a, compute_dweight=compute_dweight, **options))(
        jj(dy), jj(x), jj(u), jj(weight), jj(bias), jf["mean_tensor"], jf["rstd_tensor"]
    )

    assert jf["mask_tensor"] is None
    for name in ("y_tensor", "mean_tensor", "rstd_tensor"):
        np.testing.assert_array_equal(np.asarray(jf[name], np.float32), forward[name].float().cpu().numpy())
    for name in ("dx_tensor", "du_tensor", "dweight_tensor", "dbias_tensor"):
        if backward[name] is None:
            assert jb[name] is None
        else:
            np.testing.assert_array_equal(np.asarray(jb[name], np.float32), backward[name].float().cpu().numpy())


def test_hstu_lmsd_jax_rejects_dropout():
    jax, ops = jax_ops()
    import jax.numpy as jnp

    x = jnp.zeros((8, 64), jnp.bfloat16)
    w = jnp.zeros((64,), jnp.bfloat16)
    with pytest.raises(NotImplementedError, match="dropout"):
        ops.hstu_lmsd_forward_jax(x, x, w, w, dropout_ratio=0.1)
    with pytest.raises(TypeError, match="bfloat16"):
        ops.hstu_lmsd_forward_jax(x.astype(jnp.float16), x, w, w)
