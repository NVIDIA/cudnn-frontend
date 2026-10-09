# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared MXFP8 dGLU JAX ABI, independent dprob reference and captured replay."""

from functools import partial

import numpy as np
import pytest

jax = pytest.importorskip("jax")
ml_dtypes = pytest.importorskip("ml_dtypes")
import jax.numpy as jnp

from cudnn.frost.buffers import cutedsl_requirement_error

requirement = cutedsl_requirement_error("JAX grouped dGLU tests")
if requirement:
    pytest.skip(requirement, allow_module_level=True)

from gemm.jax.test_gemm_amax_jax import skip_unless_sm100

pytestmark = pytest.mark.L0


def problem(backward, experts, flat_sf, bf16_prob, n=512):
    rng = np.random.default_rng(796)
    m, k = experts * 256, 256
    return dict(
        a_tensor=rng.integers(-2, 3, (m, k)).astype(ml_dtypes.float8_e4m3fn),
        b_tensor=rng.integers(-2, 3, (experts, n, k)).astype(ml_dtypes.float8_e4m3fn),
        c_tensor=rng.uniform(-2, 2, (m, 2 * n)).astype(ml_dtypes.bfloat16),
        sfa_tensor=np.full(m * k // 32, 127, np.uint8),
        sfb_tensor=np.full(experts * n * k // 32, 127, np.uint8),
        padded_offsets=np.arange(256, m + 1, 256, dtype=np.int32),
        alpha_tensor=rng.uniform(0.5, 1.5, experts).astype(np.float32),
        beta_tensor=np.ones(experts, np.float32),
        prob_tensor=rng.integers(1, 3, m).astype(ml_dtypes.bfloat16 if bf16_prob else np.float32),
        norm_const_tensor=np.array([0.01], np.float32),
    )


def check_result(result, arrays, act_func):
    jax.block_until_ready(result)
    expected = []
    n = arrays["b_tensor"].shape[1]
    for expert, b in enumerate(arrays["b_tensor"]):
        rows = slice(expert * 256, (expert + 1) * 256)
        accumulator = arrays["a_tensor"][rows].astype(np.float32) @ b.astype(np.float32).T
        pair = arrays["c_tensor"][rows].astype(np.float32).reshape(256, n // 32, 2, 32)
        gate, up = pair[:, :, 0].reshape(256, n), pair[:, :, 1].reshape(256, n)
        if act_func == "dswiglu":
            gate *= arrays["beta_tensor"][expert]
            up *= arrays["beta_tensor"][expert]
            activation = up * gate / (1 + np.exp(-gate))
        else:
            gate = np.minimum(gate, 7)
            up = np.clip(up, -7, 7)
            activation = (up + 1) * gate / (1 + np.exp(-1.702 * gate))
        expected.append(np.sum(accumulator * arrays["alpha_tensor"][expert] ** 2 * activation, axis=1))
    np.testing.assert_allclose(np.asarray(result["dprob_tensor"]), np.concatenate(expected), rtol=2e-4, atol=2e-3)
    for name in ("d_row_tensor", "d_col_tensor"):
        output = np.asarray(result[name]).astype(np.float32)
        assert output.shape == arrays["c_tensor"].shape
        assert np.isfinite(output).all() and np.any(output != 0)


@pytest.mark.parametrize("act_func", ["dswiglu", "dgeglu"])
@pytest.mark.parametrize("experts", [1, 4])
@pytest.mark.parametrize("shared_wrapper", [False, True], ids=["jax_entry", "wrapper"])
def test_mxfp8_dglu_matches_reference_and_replays(act_func, experts, shared_wrapper):
    skip_unless_sm100()
    import cudnn

    arrays = problem(True, experts, True, True, n=512)
    arrays["beta_tensor"] = np.linspace(0.5, 1.5, experts, dtype=np.float32)
    options = dict(act_func=act_func, discrete_col_sfd=True, d_dtype=ml_dtypes.float8_e4m3fn)
    bridge = partial(cudnn.grouped_gemm_dglu_wrapper_sm100, sf_vec_size=32, dprob_tensor=None) if shared_wrapper else cudnn.grouped_gemm_dglu_jax_sm100
    bridge = partial(bridge, **options)
    inputs = {name: jnp.asarray(value) for name, value in arrays.items()}
    check_result(bridge(**inputs), arrays, act_func)
    compiled = jax.jit(bridge, compiler_options={"xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL"})
    check_result(compiled(**inputs), arrays, act_func)
    arrays["alpha_tensor"] *= 0.5
    arrays["prob_tensor"] *= 0.5
    inputs.update(alpha_tensor=jnp.asarray(arrays["alpha_tensor"]), prob_tensor=jnp.asarray(arrays["prob_tensor"]))
    check_result(compiled(**inputs), arrays, act_func)


@pytest.mark.parametrize(
    "option,value", [("sf_vec_size", 16), ("dprob_tensor", 1), ("generate_dbias", True), ("use_dynamic_sched", True), ("deterministic", True), ("n", 512)]
)
def test_mxfp8_dglu_rejects_unsupported_options(option, value):
    skip_unless_sm100()
    import cudnn

    inputs = {name: jnp.asarray(value) for name, value in problem(True, 1, True, False).items()}
    options = dict(sf_vec_size=32, d_dtype=ml_dtypes.float8_e4m3fn, dprob_tensor=None)
    options[option] = value
    with pytest.raises(ValueError, match=option):
        cudnn.grouped_gemm_dglu_wrapper_sm100(**inputs, **options)


@pytest.mark.parametrize("family,name", [("rubin", "BlockScaledMoEGroupedGemmDgluKernel"), ("blackwell", "BlockScaledMoEGroupedGemmDgluDbiasKernel")])
def test_dglu_jax_architecture_dispatch(monkeypatch, family, name):
    from cudnn.gemm.cutedsl.grouped.dglu import jax_blockscaled_api

    monkeypatch.setattr(jax_blockscaled_api, "get_device_type", lambda: family)
    assert jax_blockscaled_api.dglu_kernel_type().__name__ == name
