# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contiguous MXFP8 JAX parity with the established torch kernel entry points."""

from functools import partial

import numpy as np
import pytest

jax = pytest.importorskip("jax")
ml_dtypes = pytest.importorskip("ml_dtypes")
torch = pytest.importorskip("torch")
import jax.numpy as jnp

from fe_api.gemm.test_gemm_amax_jax import skip_unless_sm100

pytestmark = pytest.mark.L0


def problem(backward, experts, flat_sf, bf16_prob):
    rng = np.random.default_rng(796)
    m, n, k = experts * 256, 256, 256
    arrays = dict(
        a_tensor=rng.integers(-2, 3, (m, k)).astype(ml_dtypes.float8_e4m3fn),
        b_tensor=rng.integers(-2, 3, (experts, n, k)).astype(ml_dtypes.float8_e4m3fn),
        sfa_tensor=rng.integers(125, 129, (1, m // 128, k // 128, 32, 4, 4), dtype=np.uint8),
        sfb_tensor=rng.integers(125, 129, (experts, n // 128, k // 128, 32, 4, 4), dtype=np.uint8),
        padded_offsets=np.arange(256, m + 1, 256, dtype=np.int32),
        alpha_tensor=rng.uniform(0.5, 1.5, experts).astype(np.float32),
        prob_tensor=rng.integers(1, 3, m).astype(ml_dtypes.bfloat16 if bf16_prob else np.float32),
        norm_const_tensor=np.array([0.01], np.float32),
    )
    if backward:
        arrays.update(c_tensor=rng.uniform(-2, 2, (m, 2 * n)).astype(ml_dtypes.bfloat16), beta_tensor=np.ones(experts, np.float32))
    if flat_sf:
        for name in ("sfa_tensor", "sfb_tensor"):
            arrays[name] = arrays[name].reshape(-1)
    return arrays


def torch_inputs(arrays):
    result = {}
    for name, array in arrays.items():
        if name.startswith("sf"):
            tensor = torch.from_numpy(array).cuda().view(torch.float8_e8m0fnu)
        elif array.dtype == ml_dtypes.float8_e4m3fn:
            tensor = torch.from_numpy(array.view(np.uint8)).cuda().view(torch.float8_e4m3fn)
        elif array.dtype == ml_dtypes.bfloat16:
            tensor = torch.from_numpy(array.view(np.uint16)).cuda().view(torch.bfloat16)
        else:
            tensor = torch.from_numpy(array).cuda()
        result[name] = tensor
    return result


def assert_outputs(result, reference):
    from cudnn.api_base import TupleDict

    assert isinstance(result, TupleDict)
    assert tuple(result.keys()) == tuple(reference.keys())
    assert all(value is result[i] for i, value in enumerate(result))
    jax.block_until_ready(result)
    for name, expected in reference.items():
        if expected is None:
            assert result[name] is None
        elif name == "dprob_tensor":
            np.testing.assert_allclose(np.asarray(result[name]), expected.cpu().numpy(), rtol=1e-4, atol=1e-4)
        else:
            actual = np.asarray(result[name])
            assert actual.shape == tuple(expected.shape)
            np.testing.assert_array_equal(actual.view(np.uint8), expected.contiguous().view(torch.uint8).cpu().numpy())


@pytest.mark.parametrize("backward", [False, True], ids=["swiglu", "dswiglu"])
@pytest.mark.parametrize("experts", [1, 4])
@pytest.mark.parametrize("flat_sf,bf16_prob", [(False, False), (True, True)])
def test_canonical_jax_parity(backward, experts, flat_sf, bf16_prob, monkeypatch):
    skip_unless_sm100()
    import cudnn

    if backward:
        import cudnn.gemm.cutedsl.grouped.dswiglu.api as eager_api

        monkeypatch.setattr(eager_api, "_cache_of_GroupedGemmDswigluSm100Objects", {})
    arrays = problem(backward, experts, flat_sf, bf16_prob)
    name = "dswiglu" if backward else "swiglu"
    eager = getattr(cudnn, f"grouped_gemm_{name}_wrapper_sm100")
    bridge = partial(getattr(cudnn, f"grouped_gemm_{name}_wrapper_sm100"), d_dtype=ml_dtypes.float8_e4m3fn, sf_vec_size=32)
    reference = eager(**torch_inputs(arrays), d_dtype=torch.float8_e4m3fn, sf_vec_size=32)
    inputs = {name: jnp.asarray(array) for name, array in arrays.items()}
    assert_outputs(bridge(**inputs), reference)
    compiled = jax.jit(bridge, compiler_options={"xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL"})
    assert_outputs(compiled(**inputs), reference)
    arrays["alpha_tensor"] *= 0.5
    inputs["alpha_tensor"] = jnp.asarray(arrays["alpha_tensor"])
    reference = eager(**torch_inputs(arrays), d_dtype=torch.float8_e4m3fn, sf_vec_size=32)
    assert_outputs(compiled(**inputs), reference)


@pytest.mark.parametrize("backward", [False, True])
def test_canonical_jax_rejects_invalid_sf(backward):
    skip_unless_sm100()
    import cudnn

    inputs = {name: jnp.asarray(array) for name, array in problem(backward, 1, True, False).items()}
    inputs["sfa_tensor"] = inputs["sfa_tensor"][:-1]
    bridge = partial(getattr(cudnn, f"grouped_gemm_{'dswiglu' if backward else 'swiglu'}_wrapper_sm100"), d_dtype=ml_dtypes.float8_e4m3fn, sf_vec_size=32)
    with pytest.raises(ValueError, match="SFA"):
        jax.jit(bridge)(**inputs)


def test_canonical_jax_forward_backward_chain(monkeypatch):
    skip_unless_sm100()
    import cudnn
    import cudnn.gemm.cutedsl.grouped.dswiglu.api as eager_api

    monkeypatch.setattr(eager_api, "_cache_of_GroupedGemmDswigluSm100Objects", {})
    forward = problem(False, 4, False, False)
    forward["b_tensor"] = np.tile(forward["b_tensor"], (1, 2, 1))
    forward["sfb_tensor"] = np.tile(forward["sfb_tensor"], (1, 2, 1, 1, 1, 1))
    backward = problem(True, 4, False, False)
    forward_ref = cudnn.grouped_gemm_swiglu_wrapper_sm100(**torch_inputs(forward), d_dtype=torch.float8_e4m3fn, sf_vec_size=32)
    backward_torch = torch_inputs(backward)
    backward_torch["c_tensor"] = forward_ref["c_tensor"]
    backward_ref = cudnn.grouped_gemm_dswiglu_wrapper_sm100(**backward_torch, d_dtype=torch.float8_e4m3fn, sf_vec_size=32)
    backward.pop("c_tensor")

    def step(fwd, bwd):
        fwd_result = cudnn.grouped_gemm_swiglu_wrapper_sm100(**fwd, d_dtype=ml_dtypes.float8_e4m3fn, sf_vec_size=32)
        bwd_result = cudnn.grouped_gemm_dswiglu_wrapper_sm100(**bwd, c_tensor=fwd_result["c_tensor"], d_dtype=ml_dtypes.float8_e4m3fn, sf_vec_size=32)
        return fwd_result, bwd_result

    inputs = tuple({name: jnp.asarray(array) for name, array in values.items()} for values in (forward, backward))
    compiled = jax.jit(step, compiler_options={"xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL"})
    for _ in range(2):
        fwd_result, bwd_result = compiled(*inputs)
        assert_outputs(fwd_result, forward_ref)
        assert_outputs(bwd_result, backward_ref)


def test_canonical_jax_without_torch():
    skip_unless_sm100()
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import sys
        sys.modules['torch'] = None
        import jax
        import jax.numpy as jnp
        import ml_dtypes
        import numpy as np
        from functools import partial
        from cudnn import grouped_gemm_swiglu_wrapper_sm100, grouped_gemm_dswiglu_wrapper_sm100
        fp8 = ml_dtypes.float8_e5m2
        a = jnp.ones((256, 256), fp8)
        b = jnp.ones((1, 256, 256), fp8)
        sf = jnp.full((1, 2, 2, 32, 4, 4), 127, jnp.uint8)
        offsets = jnp.array([256], jnp.int32)
        alpha = jnp.ones((1,), jnp.float32)
        prob = jnp.ones((256,), jnp.float32)
        norm = jnp.array([0.01], jnp.float32)
        fwd = partial(grouped_gemm_swiglu_wrapper_sm100, d_dtype=fp8, sf_vec_size=32)
        out = jax.jit(fwd)(a, b, sf, sf, offsets, alpha, prob_tensor=prob, norm_const_tensor=norm)
        np.testing.assert_array_equal(np.asarray(out['c_tensor']).astype(np.float32), 256)
        c = jnp.ones((256, 512), jnp.bfloat16)
        bwd = partial(grouped_gemm_dswiglu_wrapper_sm100, d_dtype=ml_dtypes.float8_e4m3fn, sf_vec_size=32)
        out = jax.jit(bwd)(a, b, c, sf, sf, offsets, alpha, alpha, prob, norm)
        assert np.isfinite(np.asarray(out['dprob_tensor'])).all()
        assert sys.modules['torch'] is None
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], env={**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}, capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backward", [False, True])
def test_canonical_jax_runtime_offsets_and_padding(backward, monkeypatch):
    skip_unless_sm100()
    import cudnn
    import cudnn.gemm.cutedsl.grouped.dswiglu.api as eager_api

    monkeypatch.setattr(eager_api, "_cache_of_GroupedGemmDswigluSm100Objects", {})
    arrays = problem(backward, 4, False, False)
    name = "dswiglu" if backward else "swiglu"
    bridge = partial(getattr(cudnn, f"grouped_gemm_{name}_wrapper_sm100"), d_dtype=ml_dtypes.float8_e4m3fn, sf_vec_size=32)
    compiled = jax.jit(bridge, compiler_options={"xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL"})
    inputs = {name: jnp.asarray(array) for name, array in arrays.items()}
    jax.block_until_ready(compiled(**inputs))
    arrays["padded_offsets"] = np.array([0, 256, 256, 768], np.int32)
    inputs["padded_offsets"] = jnp.asarray(arrays["padded_offsets"])
    result = jax.block_until_ready(compiled(**inputs))
    reference = getattr(cudnn, f"grouped_gemm_{name}_wrapper_sm100")(**torch_inputs(arrays), d_dtype=torch.float8_e4m3fn, sf_vec_size=32)
    keys = ("d_row_tensor", "d_col_tensor", "dprob_tensor") if backward else ("c_tensor", "d_tensor", "d_col_tensor")
    for key in keys:
        actual = np.asarray(result[key]).astype(np.float32)
        expected = reference[key].float().cpu().numpy()
        np.testing.assert_allclose(actual[:768], expected[:768], rtol=1e-4 if key == "dprob_tensor" else 0, atol=1e-4 if key == "dprob_tensor" else 0)
        np.testing.assert_array_equal(actual[768:], 0)


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize(
    "option,value",
    [
        ("current_stream", 0),
        ("sf_vec_size", 16),
        ("acc_dtype", jnp.float16),
        ("cd_major", "m"),
        ("vector_f32", True),
        ("m_aligned", 128),
        ("discrete_col_sfd", True),
    ],
)
def test_shared_wrapper_rejects_unsupported_jax_options(backward, option, value):
    skip_unless_sm100()
    import cudnn

    inputs = {name: jnp.asarray(array) for name, array in problem(backward, 1, True, False).items()}
    wrapper = getattr(cudnn, f"grouped_gemm_{'dswiglu' if backward else 'swiglu'}_wrapper_sm100")
    options = dict(d_dtype=ml_dtypes.float8_e4m3fn, sf_vec_size=32)
    options[option] = value
    with pytest.raises(ValueError, match=option):
        jax.jit(partial(wrapper, **options))(**inputs)


@pytest.mark.parametrize("option", ["dprob_tensor_buf", "amax_tensor_buf", "epilogue_op"])
def test_shared_backward_rejects_jax_output_buffers_and_epilogues(option):
    skip_unless_sm100()
    from cudnn import grouped_gemm_dswiglu_wrapper_sm100

    inputs = {name: jnp.asarray(array) for name, array in problem(True, 1, True, False).items()}
    options = {option: "relu" if option == "epilogue_op" else jnp.empty((256,))}
    with pytest.raises(ValueError, match=option):
        grouped_gemm_dswiglu_wrapper_sm100(**inputs, **options, d_dtype=ml_dtypes.float8_e4m3fn, sf_vec_size=32)


@pytest.mark.parametrize("backward", [False, True])
def test_shared_wrapper_rejects_mixed_frameworks_and_missing_alpha(backward):
    skip_unless_sm100()
    import cudnn

    arrays = problem(backward, 1, True, False)
    inputs = {name: jnp.asarray(array) for name, array in arrays.items()}
    wrapper = partial(getattr(cudnn, f"grouped_gemm_{'dswiglu' if backward else 'swiglu'}_wrapper_sm100"), d_dtype=ml_dtypes.float8_e4m3fn, sf_vec_size=32)
    alpha = inputs["alpha_tensor"]
    inputs["alpha_tensor"] = None
    with pytest.raises(ValueError, match="alpha_tensor is required"):
        wrapper(**inputs)
    inputs["alpha_tensor"] = alpha
    inputs["b_tensor"] = None
    with pytest.raises(ValueError, match="b_tensor is required"):
        wrapper(**inputs)
    inputs["b_tensor"] = torch_inputs(arrays)["b_tensor"]
    with pytest.raises(ValueError, match="b_tensor must be a JAX"):
        wrapper(**inputs)


def test_no_separate_public_jax_entry_points():
    import cudnn

    assert not hasattr(cudnn, "grouped_gemm_swiglu_jax_sm100")
    assert not hasattr(cudnn, "grouped_gemm_dswiglu_jax_sm100")


def test_shared_backward_rejects_e5m2_output():
    skip_unless_sm100()
    from cudnn import grouped_gemm_dswiglu_wrapper_sm100

    inputs = {name: jnp.asarray(array) for name, array in problem(True, 1, True, False).items()}
    with pytest.raises(ValueError, match="d_dtype must be e4m3"):
        grouped_gemm_dswiglu_wrapper_sm100(**inputs, d_dtype=ml_dtypes.float8_e5m2, sf_vec_size=32)
