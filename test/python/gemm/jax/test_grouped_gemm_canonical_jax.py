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
from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("JAX grouped GEMM tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.gemm.cutedsl.grouped.dswiglu.jax_api import grouped_gemm_dswiglu
from cudnn.gemm.cutedsl.grouped.swiglu.jax_api import grouped_gemm_swiglu

from gemm.jax.test_gemm_amax_jax import skip_unless_sm100

pytestmark = pytest.mark.L0

JAX_ENTRY = {"swiglu": grouped_gemm_swiglu, "dswiglu": grouped_gemm_dswiglu}


def problem(backward, experts, flat_sf, bf16_prob, n=256):
    rng = np.random.default_rng(796)
    m, k = experts * 256, 256
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


def block_scaled_problem(data_format, experts, n=512, backward=False):
    """Operands for "mxfp8", "nvfp4" (E4M3 scales per 16) or "mxfp4" (E8M0 scales per 32); FP4 is packed uint8."""
    arrays = problem(backward, experts, True, False, n=n)
    del arrays["norm_const_tensor"]
    if data_format == "mxfp8":
        return arrays, 32
    rng = np.random.default_rng(797)
    m, k = arrays["a_tensor"].shape
    sf_vec_size, sf_low, sf_dtype = (16, 0x30, ml_dtypes.float8_e4m3fn) if data_format == "nvfp4" else (32, 125, ml_dtypes.float8_e8m0fnu)
    rest_k = k // (4 * sf_vec_size)
    arrays.update(
        a_tensor=rng.integers(0, 256, (m, k // 2), dtype=np.uint8),
        b_tensor=rng.integers(0, 256, (experts, n, k // 2), dtype=np.uint8),
        sfa_tensor=rng.integers(sf_low, sf_low + 4, (1, m // 128, rest_k, 32, 4, 4), dtype=np.uint8).view(sf_dtype),
        sfb_tensor=rng.integers(sf_low, sf_low + 4, (experts, n // 128, rest_k, 32, 4, 4), dtype=np.uint8).view(sf_dtype),
    )
    return arrays, sf_vec_size


def poison_allocator():
    """Fill freed JAX device memory so an output the kernel never writes cannot read back as zeros."""
    for _ in range(8):
        jax.block_until_ready(jnp.full((1 << 22,), 0x55, jnp.uint8))


def torch_inputs(arrays):
    result = {}
    for name, array in arrays.items():
        if name.startswith("sf"):
            sf_dtype = torch.float8_e4m3fn if array.dtype == ml_dtypes.float8_e4m3fn else torch.float8_e8m0fnu
            tensor = torch.from_numpy(array.view(np.uint8)).cuda().view(sf_dtype)
        elif array.dtype == np.uint8:
            tensor = torch.from_numpy(array).cuda().view(torch.float4_e2m1fn_x2)
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
@pytest.mark.parametrize("flat_sf,bf16_prob,discrete_col_sfd", [(False, False, False), (True, True, True)])
@pytest.mark.parametrize("shared_wrapper", [False, True], ids=["namespace", "wrapper"])
def test_canonical_jax_parity(backward, experts, flat_sf, bf16_prob, discrete_col_sfd, shared_wrapper, monkeypatch):
    skip_unless_sm100()
    import cudnn

    if backward:
        import cudnn.gemm.cutedsl.grouped.dswiglu.api as eager_api

        monkeypatch.setattr(eager_api, "_cache_of_GroupedGemmDswigluSm100Objects", {})
    # Expert-packed column scales differ from the regular layout only across several 128-row atom blocks.
    arrays = problem(backward, experts, flat_sf, bf16_prob, n=512 if discrete_col_sfd else 256)
    name = "dswiglu" if backward else "swiglu"
    eager = getattr(cudnn, f"grouped_gemm_{name}_wrapper_sm100")
    bridge = partial(eager, d_dtype=ml_dtypes.float8_e4m3fn, sf_vec_size=32) if shared_wrapper else JAX_ENTRY[name]
    bridge = partial(bridge, discrete_col_sfd=discrete_col_sfd)
    eager = partial(eager, d_dtype=torch.float8_e4m3fn, sf_vec_size=32, discrete_col_sfd=discrete_col_sfd)
    reference = eager(**torch_inputs(arrays))
    inputs = {name: jnp.asarray(array) for name, array in arrays.items()}
    assert_outputs(bridge(**inputs), reference)
    with jax.no_tracing(True):
        assert_outputs(bridge(**inputs), reference)
    compiled = jax.jit(bridge, compiler_options={"xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL"})
    assert_outputs(compiled(**inputs), reference)
    arrays["alpha_tensor"] *= 0.5
    inputs["alpha_tensor"] = jnp.asarray(arrays["alpha_tensor"])
    reference = eager(**torch_inputs(arrays))
    assert_outputs(compiled(**inputs), reference)


@pytest.mark.parametrize("act_func", ["swiglu", "geglu"])
@pytest.mark.parametrize("experts", [1, 4])
@pytest.mark.parametrize("shared_wrapper", [False, True], ids=["jax_entry", "wrapper"])
def test_canonical_jax_glu_parity(act_func, experts, shared_wrapper):
    skip_unless_sm100()
    import cudnn

    arrays = problem(False, experts, True, True, n=512)
    eager = cudnn.grouped_gemm_glu_wrapper_sm100
    options = dict(act_func=act_func, discrete_col_sfd=True, generate_c=True, d_dtype=ml_dtypes.float8_e4m3fn)
    reference = eager(**torch_inputs(arrays), sf_vec_size=32, **{**options, "d_dtype": torch.float8_e4m3fn})
    bridge = partial(eager, sf_vec_size=32) if shared_wrapper else cudnn.grouped_gemm_glu_jax_sm100
    bridge = partial(bridge, **options)
    inputs = {name: jnp.asarray(array) for name, array in arrays.items()}
    assert_outputs(bridge(**inputs), reference)
    assert_outputs(jax.jit(bridge)(**inputs), reference)
    assert bridge(**inputs, generate_c=False)["c_tensor"] is None


def kernel_facing_torch(arrays):
    """Kernel-facing views of the canonical buffers, for the torch SReLU, dSReLU and GLU + Hadamard + quant wrappers,
    which take only those layouts."""
    tensors = torch_inputs(arrays)
    for name in {"a_tensor", "c_tensor"} & tensors.keys():
        tensor = tensors[name]
        tensors[name] = tensor.as_strided((*tensor.shape, 1), (*tensor.stride(), tensor.numel()))
    m = tensors["prob_tensor"].numel()
    tensors["prob_tensor"] = tensors["prob_tensor"].as_strided((m, 1, 1), (1, m, m))
    tensors["b_tensor"] = tensors["b_tensor"].permute(1, 2, 0)
    for name in {"sfa_tensor", "sfb_tensor"} & tensors.keys():
        tensors[name] = tensors[name].permute(3, 4, 1, 5, 2, 0)
    return tensors


def canonical_reference(result):
    """A kernel-facing torch result in the canonical JAX layouts: unit L mode dropped, physical 6-D scale buffers."""
    for name, tensor in result.items():
        if tensor is not None and tensor.ndim == 6:
            result[name] = tensor.permute(5, 2, 4, 0, 1, 3)
        elif tensor is not None and name != "amax_tensor" and tensor.ndim == 3:
            result[name] = tensor.reshape(tensor.shape[:-1])
    return result


def dglu_reference(arrays, **options):
    import cudnn

    dprob = torch.zeros((arrays["a_tensor"].shape[0],), device="cuda")
    return cudnn.grouped_gemm_dglu_wrapper_sm100(**torch_inputs(arrays), dprob_tensor=dprob, d_dtype=torch.float8_e4m3fn, sf_vec_size=32, **options)


@pytest.mark.parametrize("act_func", ["dswiglu", "dgeglu"])
@pytest.mark.parametrize("experts", [1, 4])
@pytest.mark.parametrize("shared_wrapper", [False, True], ids=["jax_entry", "wrapper"])
def test_canonical_jax_dglu_parity(act_func, experts, shared_wrapper):
    skip_unless_sm100()
    import cudnn

    arrays = problem(True, experts, False, False, n=512)
    arrays["beta_tensor"] = np.linspace(0.5, 1.5, experts, dtype=np.float32)
    options = dict(act_func=act_func, discrete_col_sfd=True)
    bridge = partial(cudnn.grouped_gemm_dglu_wrapper_sm100, sf_vec_size=32, dprob_tensor=None) if shared_wrapper else cudnn.grouped_gemm_dglu_jax_sm100
    bridge = partial(bridge, d_dtype=ml_dtypes.float8_e4m3fn, **options)
    inputs = {name: jnp.asarray(array) for name, array in arrays.items()}
    reference = dglu_reference(arrays, **options)
    assert_outputs(bridge(**inputs), reference)
    compiled = jax.jit(bridge, compiler_options={"xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL"})
    assert_outputs(compiled(**inputs), reference)
    arrays["alpha_tensor"] *= 0.5
    inputs["alpha_tensor"] = jnp.asarray(arrays["alpha_tensor"])
    assert_outputs(compiled(**inputs), dglu_reference(arrays, **options))


@pytest.mark.parametrize(
    "option,value", [("sf_vec_size", 16), ("dprob_tensor", 1), ("generate_dbias", True), ("use_dynamic_sched", True), ("deterministic", True), ("n", 512)]
)
def test_jax_dglu_rejects_unsupported_options(option, value):
    skip_unless_sm100()
    import cudnn

    inputs = {name: jnp.asarray(array) for name, array in problem(True, 1, True, False).items()}
    options = dict(sf_vec_size=32, d_dtype=ml_dtypes.float8_e4m3fn, dprob_tensor=None)
    options[option] = value
    with pytest.raises(ValueError, match=option):
        cudnn.grouped_gemm_dglu_wrapper_sm100(**inputs, **options)


def wgrad_problem(splits):
    rng = np.random.default_rng(796)
    m, n, tokens = 384, 640, sum(splits)
    return dict(
        a_tensor=rng.integers(-2, 3, (m, tokens)).astype(ml_dtypes.float8_e4m3fn),
        b_tensor=rng.integers(-2, 3, (tokens, n)).astype(ml_dtypes.float8_e4m3fn),
        sfa_tensor=rng.integers(125, 129, (m, tokens // 32), dtype=np.uint8),
        sfb_tensor=rng.integers(125, 129, (n, tokens // 32), dtype=np.uint8),
        offsets_tensor=np.cumsum(splits).astype(np.int32),
    )


# One-expert splits span exactly one K tile (128 tokens on SM100, 256 on Rubin), the case a static
# scale-factor column count broke.
@pytest.mark.parametrize("splits", [(128,), (256,), (256, 384, 0, 128)], ids=["128tokens", "256tokens", "4experts"])
@pytest.mark.parametrize("shared_wrapper", [False, True], ids=["jax_entry", "wrapper"])
def test_canonical_jax_wgrad_parity(splits, shared_wrapper):
    skip_unless_sm100()
    import cudnn

    arrays = wgrad_problem(splits)
    eager = partial(cudnn.grouped_gemm_wgrad_wrapper_sm100, sf_vec_size=32)
    bridge = eager if shared_wrapper else cudnn.grouped_gemm_wgrad_jax_sm100
    inputs = {name: jnp.asarray(array) for name, array in arrays.items()}
    inputs["sfa_tensor"] = inputs["sfa_tensor"].reshape(-1)
    # An empty expert's output must be written, not left over from the allocator.
    poison_allocator()
    assert_outputs(bridge(**inputs), eager(**torch_inputs(arrays)))
    compiled = jax.jit(bridge, compiler_options={"xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL"})
    assert_outputs(compiled(**inputs), eager(**torch_inputs(arrays)))
    arrays["offsets_tensor"] = np.cumsum((0,) * (len(splits) - 1) + (sum(splits),)).astype(np.int32)
    inputs["offsets_tensor"] = jnp.asarray(arrays["offsets_tensor"])
    assert_outputs(compiled(**inputs), eager(**torch_inputs(arrays)))


@pytest.mark.parametrize("data_format", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize("shared_wrapper", [False, True], ids=["jax_entry", "wrapper"])
def test_canonical_jax_wgrad_fp4(data_format, shared_wrapper):
    skip_unless_sm100()
    import cudnn
    from cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad import _NEEDS_FP4_LAYOUT_WORKAROUND

    if _NEEDS_FP4_LAYOUT_WORKAROUND:
        pytest.skip("FP4 wgrad from JAX requires nvidia-cutlass-dsl >= 4.8")
    rng = np.random.default_rng(798)
    m, n, splits = 384, 640, (256, 512, 0, 256)
    tokens = sum(splits)
    sf_vec_size, sf_low, sf_dtype = (16, 0x30, ml_dtypes.float8_e4m3fn) if data_format == "nvfp4" else (32, 125, ml_dtypes.float8_e8m0fnu)
    arrays = dict(
        a_tensor=rng.integers(0, 256, (m, tokens // 2), dtype=np.uint8),
        # Packed FP4 B is token-innermost: (n, tokens / 2).
        b_tensor=rng.integers(0, 256, (n, tokens // 2), dtype=np.uint8),
        sfa_tensor=rng.integers(sf_low, sf_low + 4, (m, tokens // sf_vec_size), dtype=np.uint8).view(sf_dtype),
        sfb_tensor=rng.integers(sf_low, sf_low + 4, (n, tokens // sf_vec_size), dtype=np.uint8).view(sf_dtype),
        offsets_tensor=np.cumsum(splits).astype(np.int32),
    )
    if data_format == "nvfp4":
        arrays.update(global_scale_a=rng.uniform(0.5, 2, len(splits)).astype(np.float32), global_scale_b=rng.uniform(0.5, 2, len(splits)).astype(np.float32))
    eager = partial(cudnn.grouped_gemm_wgrad_wrapper_sm100, sf_vec_size=sf_vec_size)
    reference_inputs = torch_inputs(arrays)
    reference_inputs["b_tensor"] = reference_inputs["b_tensor"].T
    reference = eager(**reference_inputs)
    bridge = eager if shared_wrapper else partial(cudnn.grouped_gemm_wgrad_jax_sm100, sf_vec_size=sf_vec_size)
    inputs = {name: jnp.asarray(array) for name, array in arrays.items()}
    poison_allocator()
    assert_outputs(bridge(**inputs), reference)
    compiled = jax.jit(bridge, compiler_options={"xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL"})
    assert_outputs(compiled(**inputs), reference)


@pytest.mark.parametrize(
    "shared_wrapper,option,value",
    [
        (True, "sf_vec_size", 16),
        (True, "output_mode", "discrete"),
        (True, "global_scale_a", 1),
        (True, "sf_fp8_dtype_override", "e5m3"),
        (True, "accumulate_on_output", True),
        (False, "input_order", "tensor_ragged"),
        (False, "acc_dtype", jnp.float16),
        (False, "wgrad_ptrs", 1),
    ],
)
def test_jax_wgrad_rejects_unsupported_options(shared_wrapper, option, value):
    skip_unless_sm100()
    import cudnn

    inputs = {name: jnp.asarray(array) for name, array in wgrad_problem((256,)).items()}
    fn = partial(cudnn.grouped_gemm_wgrad_wrapper_sm100, sf_vec_size=32) if shared_wrapper else cudnn.grouped_gemm_wgrad_jax_sm100
    with pytest.raises(ValueError, match=option):
        fn(**inputs, **{option: value})


def test_jax_wgrad_rejects_invalid_sf():
    skip_unless_sm100()
    import cudnn

    inputs = {name: jnp.asarray(array) for name, array in wgrad_problem((256,)).items()}
    inputs["sfb_tensor"] = inputs["sfb_tensor"][:-1]
    with pytest.raises(ValueError, match="SFB"):
        jax.jit(cudnn.grouped_gemm_wgrad_jax_sm100)(**inputs)


@pytest.mark.parametrize("b_major,fp8_output,with_prob", [("k", False, True), ("n", False, True), ("k", True, False), ("n", True, True)])
@pytest.mark.parametrize("experts", [1, 4])
@pytest.mark.parametrize("shared_wrapper", [False, True], ids=["jax_entry", "wrapper"])
def test_canonical_jax_quant_parity(b_major, fp8_output, with_prob, experts, shared_wrapper):
    skip_unless_sm100()
    import cudnn

    arrays = problem(False, experts, True, False, n=512)
    if b_major == "n":
        arrays["b_tensor"] = np.ascontiguousarray(arrays["b_tensor"].transpose(0, 2, 1))
    if not with_prob:
        del arrays["prob_tensor"]
    if not fp8_output:
        del arrays["norm_const_tensor"]
    options = dict(b_major=b_major, discrete_col_sfd=fp8_output)
    # BF16 output also returns the per-expert amax, the eager wrapper's default.
    eager = partial(cudnn.grouped_gemm_quant_wrapper_sm100, sf_vec_size=32, **options)
    reference = partial(eager, d_dtype=torch.float8_e4m3fn if fp8_output else torch.bfloat16)
    bridge = eager if shared_wrapper else partial(cudnn.grouped_gemm_quant_jax_forward, generate_amax=True, **options)
    bridge = partial(bridge, d_dtype=ml_dtypes.float8_e4m3fn if fp8_output else ml_dtypes.bfloat16)
    inputs = {name: jnp.asarray(array) for name, array in arrays.items()}
    assert_outputs(bridge(**inputs), reference(**torch_inputs(arrays)))
    compiled = jax.jit(bridge, compiler_options={"xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL"})
    assert_outputs(compiled(**inputs), reference(**torch_inputs(arrays)))
    arrays["alpha_tensor"] *= 0.5
    inputs["alpha_tensor"] = jnp.asarray(arrays["alpha_tensor"])
    assert_outputs(compiled(**inputs), reference(**torch_inputs(arrays)))


@pytest.mark.parametrize("data_format", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize("shared_wrapper", [False, True], ids=["jax_entry", "wrapper"])
def test_canonical_jax_quant_fp4(data_format, shared_wrapper):
    skip_unless_sm100()
    import cudnn

    arrays, sf_vec_size = block_scaled_problem(data_format, 4)
    eager = partial(cudnn.grouped_gemm_quant_wrapper_sm100, sf_vec_size=sf_vec_size)
    reference = eager(**torch_inputs(arrays), d_dtype=torch.bfloat16)
    assert reference["amax_tensor"] is not None
    bridge = eager if shared_wrapper else partial(cudnn.grouped_gemm_quant_jax_forward, sf_vec_size=sf_vec_size, generate_amax=True)
    bridge = partial(bridge, d_dtype=ml_dtypes.bfloat16)
    inputs = {name: jnp.asarray(array) for name, array in arrays.items()}
    poison_allocator()
    assert_outputs(bridge(**inputs), reference)
    compiled = jax.jit(bridge, compiler_options={"xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL"})
    assert_outputs(compiled(**inputs), reference)
    assert_outputs(compiled(**inputs), reference)


def bf16_problem(experts, backward=False, n=512):
    rng = np.random.default_rng(800)
    m, k = experts * 256, 256
    scaled = lambda *shape: (rng.standard_normal(shape, dtype=np.float32) * 0.25).astype(ml_dtypes.bfloat16)
    arrays = dict(
        a_tensor=scaled(m, k),
        b_tensor=scaled(experts, n, k),
        padded_offsets=np.arange(256, m + 1, 256, dtype=np.int32),
        alpha_tensor=rng.uniform(0.5, 1.5, experts).astype(np.float32),
        prob_tensor=rng.uniform(0.5, 1.5, m).astype(np.float32),
    )
    if backward:
        arrays.update(c_tensor=scaled(m, 2 * n), beta_tensor=rng.uniform(0.5, 1.5, experts).astype(np.float32))
    return arrays


@pytest.mark.parametrize("act_func", ["swiglu", "geglu"])
@pytest.mark.parametrize("shared_wrapper", [False, True], ids=["jax_entry", "wrapper"])
def test_canonical_jax_glu_bf16(act_func, shared_wrapper):
    skip_unless_sm100()
    import cudnn

    arrays = bf16_problem(4)
    eager = partial(cudnn.grouped_gemm_glu_wrapper_sm100, sfa_tensor=None, act_func=act_func, generate_c=True)
    reference = eager(**torch_inputs(arrays))
    bridge = eager if shared_wrapper else partial(cudnn.grouped_gemm_glu_jax_sm100, act_func=act_func, generate_c=True)
    inputs = {name: jnp.asarray(array) for name, array in arrays.items()}
    poison_allocator()
    assert_outputs(bridge(**inputs), reference)
    compiled = jax.jit(bridge, compiler_options={"xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL"})
    assert_outputs(compiled(**inputs), reference)


@pytest.mark.parametrize("act_func", ["dswiglu", "dgeglu"])
@pytest.mark.parametrize("shared_wrapper", [False, True], ids=["jax_entry", "wrapper"])
def test_canonical_jax_dglu_bf16(act_func, shared_wrapper):
    skip_unless_sm100()
    import cudnn

    arrays = bf16_problem(4, backward=True)
    m = arrays["a_tensor"].shape[0]
    eager = partial(cudnn.grouped_gemm_dglu_wrapper_sm100, sfa_tensor=None, act_func=act_func)
    reference = eager(**torch_inputs(arrays), dprob_tensor=torch.zeros((m,), device="cuda"))
    bridge = partial(eager, dprob_tensor=None) if shared_wrapper else partial(cudnn.grouped_gemm_dglu_jax_sm100, act_func=act_func)
    inputs = {name: jnp.asarray(array) for name, array in arrays.items()}
    poison_allocator()
    assert_outputs(bridge(**inputs), reference)
    compiled = jax.jit(bridge, compiler_options={"xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL"})
    assert_outputs(compiled(**inputs), reference)
    # The dprob accumulator must restart from zero on every call.
    assert_outputs(compiled(**inputs), reference)


def test_canonical_jax_wgrad_bf16_dense():
    skip_unless_sm100()
    import cudnn

    rng = np.random.default_rng(801)
    m, n, splits = 384, 640, (256, 512, 0, 256)
    tokens = sum(splits)
    scaled = lambda *shape: (rng.standard_normal(shape, dtype=np.float32) * 0.25).astype(ml_dtypes.bfloat16)
    arrays = dict(a_tensor=scaled(m, tokens), b_tensor=scaled(tokens, n), offsets_tensor=np.cumsum(splits).astype(np.int32))
    reference = cudnn.grouped_gemm_wgrad_wrapper_sm100(**torch_inputs(arrays), sfa_tensor=None, sfb_tensor=None)
    inputs = {name: jnp.asarray(array) for name, array in arrays.items()}
    poison_allocator()
    assert_outputs(cudnn.grouped_gemm_wgrad_jax_sm100(**inputs), reference)
    compiled = jax.jit(cudnn.grouped_gemm_wgrad_jax_sm100, compiler_options={"xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL"})
    assert_outputs(compiled(**inputs), reference)


@pytest.mark.parametrize("data_format,act_func", [("nvfp4", "swiglu"), ("nvfp4", "geglu"), ("nvfp4", "srelu"), ("mxfp4", "swiglu")])
@pytest.mark.parametrize("shared_wrapper", [False, True], ids=["jax_entry", "wrapper"])
def test_canonical_jax_glu_hadamard_parity(data_format, act_func, shared_wrapper):
    skip_unless_sm100()
    import cudnn

    arrays, sf_vec_size = block_scaled_problem(data_format, 4)
    eager = partial(cudnn.grouped_gemm_glu_hadamard_wrapper_sm100, sf_vec_size=sf_vec_size, act_func=act_func)
    reference = eager(**torch_inputs(arrays))
    bridge = eager if shared_wrapper else partial(cudnn.grouped_gemm_glu_hadamard_jax_forward, sf_vec_size=sf_vec_size, act_func=act_func)
    inputs = {name: jnp.asarray(array) for name, array in arrays.items()}
    poison_allocator()
    assert_outputs(bridge(**inputs), reference)
    compiled = jax.jit(bridge, compiler_options={"xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL"})
    assert_outputs(compiled(**inputs), reference)
    # The amax accumulators must restart on every call.
    assert_outputs(compiled(**inputs), reference)


@pytest.mark.parametrize(
    "d_quant,rht",
    [(False, None), (True, None), (False, "rowwise_bf16"), (False, "rowwise"), (True, "rowwise"), (False, "colwise"), (True, "colwise")],
)
@pytest.mark.parametrize("shared_wrapper", [False, True], ids=["jax_entry", "wrapper"])
def test_canonical_jax_glu_hadamard_quant_parity(d_quant, rht, shared_wrapper):
    skip_unless_sm100()
    import cudnn
    import cutlass

    if not hasattr(cutlass, "FloatNV8E5M3FNU"):
        pytest.skip("GLU + Hadamard + quant requires nvidia-cutlass-dsl >= 4.8")
    arrays, sf_vec_size = block_scaled_problem("nvfp4", 4)
    m, (experts, n) = arrays["a_tensor"].shape[0], arrays["b_tensor"].shape[:2]
    arrays["sfa_tensor"] = arrays["sfa_tensor"].reshape(1, m // 128, -1, 32, 4, 4)
    arrays["sfb_tensor"] = arrays["sfb_tensor"].reshape(experts, n // 128, -1, 32, 4, 4)
    # Packed NVFP4 outputs are requested with the cutlass dtype from JAX and the packed torch dtype from torch.
    jax_options = dict(sf_vec_size=sf_vec_size, norm_const=0.5, rht_norm_const=0.25)
    torch_options = dict(jax_options)
    if d_quant:
        jax_options["d_dtype"], torch_options["d_dtype"] = cutlass.Float4E2M1FN, torch.float4_e2m1fn_x2
    if rht == "rowwise_bf16":
        jax_options["rht_rowwise_dtype"], torch_options["rht_rowwise_dtype"] = ml_dtypes.bfloat16, torch.bfloat16
    elif rht is not None:
        jax_options[f"rht_{rht}_dtype"], torch_options[f"rht_{rht}_dtype"] = cutlass.Float4E2M1FN, torch.float4_e2m1fn_x2
    reference = canonical_reference(cudnn.grouped_gemm_glu_hadamard_quant_wrapper_sm100(**kernel_facing_torch(arrays), **torch_options))
    bridge = cudnn.grouped_gemm_glu_hadamard_quant_wrapper_sm100 if shared_wrapper else cudnn.grouped_gemm_glu_hadamard_quant_jax_forward
    bridge = partial(bridge, **jax_options)
    inputs = {name: jnp.asarray(array) for name, array in arrays.items()}
    poison_allocator()
    assert_outputs(bridge(**inputs), reference)
    compiled = jax.jit(bridge, compiler_options={"xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL"})
    assert_outputs(compiled(**inputs), reference)


@pytest.mark.parametrize("data_format,fp8_output", [("mxfp8", True), ("mxfp8", False), ("nvfp4", False), ("mxfp4", False)])
@pytest.mark.parametrize("shared_wrapper", [False, True], ids=["jax_entry", "wrapper"])
def test_canonical_jax_srelu_parity(data_format, fp8_output, shared_wrapper):
    skip_unless_sm100()
    import cudnn

    arrays, sf_vec_size = block_scaled_problem(data_format, 4)
    for name in ("sfa_tensor", "sfb_tensor"):
        rows, experts = (arrays["a_tensor"].shape[0], 1) if name == "sfa_tensor" else (arrays["b_tensor"].shape[1], arrays["b_tensor"].shape[0])
        arrays[name] = arrays[name].reshape(experts, rows // 128, -1, 32, 4, 4)
    if fp8_output:
        arrays["norm_const_tensor"] = np.array([0.01], np.float32)
    options = dict(sf_vec_size=sf_vec_size, discrete_col_sfd=fp8_output)
    eager = partial(cudnn.grouped_gemm_srelu_wrapper_sm100, **options)
    reference = canonical_reference(eager(**kernel_facing_torch(arrays), d_dtype=torch.float8_e4m3fn if fp8_output else torch.bfloat16))
    bridge = eager if shared_wrapper else partial(cudnn.grouped_gemm_srelu_jax_forward, **options)
    bridge = partial(bridge, d_dtype=ml_dtypes.float8_e4m3fn if fp8_output else ml_dtypes.bfloat16)
    inputs = {name: jnp.asarray(array) for name, array in arrays.items()}
    inputs["sfa_tensor"] = inputs["sfa_tensor"].reshape(-1)
    poison_allocator()
    assert_outputs(bridge(**inputs), reference)
    compiled = jax.jit(bridge, compiler_options={"xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL"})
    assert_outputs(compiled(**inputs), reference)
    assert_outputs(compiled(**inputs), reference)


@pytest.mark.parametrize("data_format", ["mxfp8", "nvfp4", "mxfp4"])
@pytest.mark.parametrize("shared_wrapper", [False, True], ids=["jax_entry", "wrapper"])
def test_canonical_jax_dsrelu_parity(data_format, shared_wrapper):
    skip_unless_sm100()
    import cudnn

    arrays, sf_vec_size = block_scaled_problem(data_format, 4)
    rng = np.random.default_rng(799)
    m, (experts, n) = arrays["a_tensor"].shape[0], arrays["b_tensor"].shape[:2]
    arrays["c_tensor"] = rng.uniform(-2, 2, (m, n)).astype(ml_dtypes.bfloat16)
    arrays["sfa_tensor"] = arrays["sfa_tensor"].reshape(1, m // 128, -1, 32, 4, 4)
    arrays["sfb_tensor"] = arrays["sfb_tensor"].reshape(experts, n // 128, -1, 32, 4, 4)
    fp8_output = data_format == "mxfp8"
    if fp8_output:
        arrays["norm_const_tensor"] = np.array([0.01], np.float32)
    options = dict(sf_vec_size=sf_vec_size, discrete_col_sfd=fp8_output)
    eager = partial(cudnn.grouped_gemm_dsrelu_wrapper_sm100, **options)

    def reference():
        result = eager(
            **kernel_facing_torch(arrays), dprob_tensor=torch.zeros((m, 1, 1), device="cuda"), d_dtype=torch.float8_e4m3fn if fp8_output else torch.bfloat16
        )
        result = canonical_reference(result)
        result["dprob_tensor"] = result["dprob_tensor"].reshape(m)
        if not fp8_output:
            # Torch allocates D_col but the kernel does not write it without quantization.
            result["d_col_tensor"] = None
        return result

    bridge = eager if shared_wrapper else partial(cudnn.grouped_gemm_dsrelu_jax_sm100, **options)
    bridge = partial(bridge, d_dtype=ml_dtypes.float8_e4m3fn if fp8_output else ml_dtypes.bfloat16)
    inputs = {name: jnp.asarray(array) for name, array in arrays.items()}
    inputs["sfa_tensor"] = inputs["sfa_tensor"].reshape(-1)
    poison_allocator()
    assert_outputs(bridge(**inputs), reference())
    compiled = jax.jit(bridge, compiler_options={"xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL"})
    assert_outputs(compiled(**inputs), reference())
    assert_outputs(compiled(**inputs), reference())


@pytest.mark.parametrize(
    "shared_wrapper,option,value",
    [
        (True, "sf_vec_size", 16),
        (True, "use_dynamic_sched", True),
        (True, "row_scale_tensor", 1),
        (True, "d_tensor", 1),
        (True, "bias_tensor", 1),
        (False, "b_major", "m"),
        (False, "norm_const_tensor", None),
    ],
)
def test_jax_quant_rejects_unsupported_options(shared_wrapper, option, value):
    skip_unless_sm100()
    import cudnn

    arrays = problem(False, 1, True, False)
    del arrays["norm_const_tensor"]
    inputs = {name: jnp.asarray(array) for name, array in arrays.items()}
    options = dict(sf_vec_size=32, generate_amax=False) if shared_wrapper else {}
    # FP8 output requires norm_const.
    options.update({"d_dtype": ml_dtypes.float8_e4m3fn} if option == "norm_const_tensor" else {}, **{option: value})
    fn = cudnn.grouped_gemm_quant_wrapper_sm100 if shared_wrapper else cudnn.grouped_gemm_quant_jax_forward
    with pytest.raises(ValueError, match=option):
        fn(**inputs, **options)


@pytest.mark.parametrize(
    "operation,rubin,blackwell",
    [
        ("glu", "moe_blockscaled_grouped_gemm_glu_rubin", "moe_blockscaled_grouped_gemm_glu_bias"),
        ("dglu", "moe_blockscaled_grouped_gemm_dglu_rubin", "moe_blockscaled_grouped_gemm_dglu_dbias"),
        ("wgrad", "moe_blockscaled_grouped_gemm_wgrad_rubin", "moe_blockscaled_grouped_gemm_wgrad"),
        ("quant", "moe_blockscaled_grouped_gemm_quant_rubin", "grouped_gemm_quant"),
        ("glu_hadamard_quant", "moe_blockscaled_grouped_gemm_glu_hadamard_quant_rubin", "moe_blockscaled_grouped_gemm_glu_hadamard_quant"),
    ],
)
def test_jax_kernel_dispatch(operation, rubin, blackwell, monkeypatch):
    import importlib

    api = importlib.import_module(f"cudnn.gemm.cutedsl.grouped.{operation}.jax_api")
    # CuTeDSL 4.7 runs the SM100 paths but cannot import the Rubin kernels.
    families = [("blackwell", blackwell)] + ([] if cutedsl_arch_requirement_error((10, 7)) else [("rubin", rubin)])
    for family, name in families:
        monkeypatch.setattr(api, "get_device_type", lambda: family)
        assert getattr(api, f"{operation}_kernel_type")().__module__.rsplit(".", 1)[1] == name


def test_jax_wgrad_plan_caches_kernel_per_architecture(monkeypatch):
    skip_unless_sm100()
    requirement = cutedsl_arch_requirement_error((10, 7))
    if requirement:
        pytest.skip(requirement)
    from cudnn.gemm.cutedsl.grouped.wgrad import jax_api

    arrays = wgrad_problem((256,))
    inputs = {name.removesuffix("_tensor"): jnp.asarray(array) for name, array in arrays.items()}
    inputs.update(sfa=inputs["sfa"].view(ml_dtypes.float8_e8m0fnu), sfb=inputs["sfb"].view(ml_dtypes.float8_e8m0fnu))
    wgrad = jax.ShapeDtypeStruct((1, 384, 640), jnp.bfloat16)
    monkeypatch.setattr(jax_api, "kernel_cache", {})
    kernels = []
    for family in ("blackwell", "rubin", "blackwell"):
        monkeypatch.setattr(jax_api, "get_device_type", lambda: family)
        kernels.append(jax_api.wgrad_plan(inputs, wgrad, (256, 256), None, 32)[0])
    assert [type(kernel).__name__ for kernel in kernels] == [
        "BlockScaledMoEGroupedGemmWgradKernel",
        "BlockScaledMoEGroupedGemmWgradRubinKernel",
        "BlockScaledMoEGroupedGemmWgradKernel",
    ]
    assert kernels[0] is kernels[2]


def test_canonical_jax_wrappers_without_torch():
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
        from cudnn import grouped_gemm_dglu_wrapper_sm100, grouped_gemm_quant_wrapper_sm100, grouped_gemm_wgrad_wrapper_sm100
        fp8 = ml_dtypes.float8_e4m3fn
        a = jnp.ones((256, 256), fp8)
        b = jnp.ones((1, 256, 256), fp8)
        c = jnp.ones((256, 512), jnp.bfloat16)
        sf = jnp.full((1, 2, 2, 32, 4, 4), 127, jnp.uint8)
        offsets = jnp.array([256], jnp.int32)
        ones = jnp.ones((1,), jnp.float32)
        prob = jnp.ones((256,), jnp.float32)
        dglu = partial(grouped_gemm_dglu_wrapper_sm100, sfb_tensor=sf, norm_const_tensor=ones, dprob_tensor=None, d_dtype=fp8, sf_vec_size=32)
        out = jax.jit(dglu)(a, c, sf, offsets, ones, ones, prob, b_tensor=b)
        assert np.isfinite(np.asarray(out['dprob_tensor'])).all()
        wgrad = partial(grouped_gemm_wgrad_wrapper_sm100, sf_vec_size=32)
        sf_t = jnp.full((256, 8), 127, jnp.uint8)
        out = jax.jit(wgrad)(a, a, sf_t, sf_t, offsets)
        np.testing.assert_array_equal(np.asarray(out['wgrad_tensor']).astype(np.float32), 256)
        quant = partial(grouped_gemm_quant_wrapper_sm100, sf_vec_size=32, generate_amax=False)
        out = jax.jit(quant)(a, sf, offsets, ones, b_tensor=b, sfb_tensor=sf)
        np.testing.assert_array_equal(np.asarray(out['d_tensor']).astype(np.float32), 256)
        import cudnn
        srelu = partial(cudnn.grouped_gemm_srelu_wrapper_sm100, sf_vec_size=32)
        out = jax.jit(srelu)(a, b, sf, sf, offsets, ones, prob_tensor=prob)
        np.testing.assert_array_equal(np.asarray(out['c_tensor']).astype(np.float32), 256)
        dsrelu = partial(cudnn.grouped_gemm_dsrelu_wrapper_sm100, sf_vec_size=32, d_dtype=fp8, norm_const_tensor=ones)
        out = jax.jit(dsrelu)(a, b, jnp.ones((256, 256), jnp.bfloat16), sf, sf, offsets, ones, prob)
        assert np.isfinite(np.asarray(out['dprob_tensor'])).all()
        # Packed FP4 ones (0x22), E4M3 unit scales (0x38), scales per 16.
        a4 = jnp.full((256, 128), 0x22, jnp.uint8)
        b4 = jnp.full((1, 256, 128), 0x22, jnp.uint8)
        sf4 = jnp.full((1, 2, 4, 32, 4, 4), 0x38, jnp.uint8).view(ml_dtypes.float8_e4m3fn)
        import cutlass
        hadamards = [cudnn.grouped_gemm_glu_hadamard_wrapper_sm100]
        if hasattr(cutlass, 'FloatNV8E5M3FNU'):
            hadamards.append(cudnn.grouped_gemm_glu_hadamard_quant_wrapper_sm100)
        for hadamard in hadamards:
            out = jax.jit(hadamard)(a4, sf4, offsets, ones, prob, b_tensor=b4, sfb_tensor=sf4)
            np.testing.assert_array_equal(np.asarray(out['c_tensor']).astype(np.float32), 256)
        glu = partial(cudnn.grouped_gemm_glu_wrapper_sm100, generate_c=True)
        bf = jnp.ones((256, 256), jnp.bfloat16)
        out = jax.jit(glu)(bf, None, offsets, ones, b_tensor=jnp.ones((1, 256, 256), jnp.bfloat16), prob_tensor=prob)
        np.testing.assert_array_equal(np.asarray(out['c_tensor']).astype(np.float32), 256)
        assert sys.modules['torch'] is None
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], env={**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}, capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("data_format", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize("act_func", ["swiglu", "geglu"])
@pytest.mark.parametrize("shared_wrapper", [False, True], ids=["jax_entry", "wrapper"])
def test_canonical_jax_glu_fp4(data_format, act_func, shared_wrapper):
    skip_unless_sm100()
    import cudnn

    arrays, sf_vec_size = block_scaled_problem(data_format, 4)
    eager = partial(cudnn.grouped_gemm_glu_wrapper_sm100, sf_vec_size=sf_vec_size, act_func=act_func, generate_c=True)
    reference = eager(**torch_inputs(arrays), d_dtype=torch.bfloat16)
    assert reference["amax_tensor"] is not None and reference["sfd_row_tensor"] is None
    # Torch allocates D_col but the kernel does not write it without quantization.
    reference["d_col_tensor"] = None
    bridge = eager if shared_wrapper else partial(cudnn.grouped_gemm_glu_jax_sm100, sf_vec_size=sf_vec_size, act_func=act_func, generate_c=True)
    bridge = partial(bridge, d_dtype=ml_dtypes.bfloat16)
    inputs = {name: jnp.asarray(array) for name, array in arrays.items()}
    poison_allocator()
    assert_outputs(bridge(**inputs), reference)
    compiled = jax.jit(bridge, compiler_options={"xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL"})
    assert_outputs(compiled(**inputs), reference)
    # The amax accumulator must restart from -inf on every call.
    assert_outputs(compiled(**inputs), reference)


@pytest.mark.parametrize("data_format", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize("act_func", ["dswiglu", "dgeglu"])
@pytest.mark.parametrize("shared_wrapper", [False, True], ids=["jax_entry", "wrapper"])
def test_canonical_jax_dglu_fp4(data_format, act_func, shared_wrapper):
    skip_unless_sm100()
    import cudnn

    arrays, sf_vec_size = block_scaled_problem(data_format, 4, backward=True)
    m = arrays["a_tensor"].shape[0]
    eager = partial(cudnn.grouped_gemm_dglu_wrapper_sm100, sf_vec_size=sf_vec_size, act_func=act_func)
    reference = eager(**torch_inputs(arrays), dprob_tensor=torch.zeros((m,), device="cuda"), d_dtype=torch.bfloat16)
    assert reference["amax_tensor"] is not None and reference["sfd_row_tensor"] is None
    # Torch allocates D_col but the kernel does not write it without quantization.
    reference["d_col_tensor"] = None
    bridge = partial(eager, dprob_tensor=None) if shared_wrapper else partial(cudnn.grouped_gemm_dglu_jax_sm100, sf_vec_size=sf_vec_size, act_func=act_func)
    bridge = partial(bridge, d_dtype=ml_dtypes.bfloat16)
    inputs = {name: jnp.asarray(array) for name, array in arrays.items()}
    poison_allocator()
    assert_outputs(bridge(**inputs), reference)
    compiled = jax.jit(bridge, compiler_options={"xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL"})
    assert_outputs(compiled(**inputs), reference)
    # The dprob and amax accumulators must restart on every call.
    assert_outputs(compiled(**inputs), reference)


@pytest.mark.parametrize(
    "shared_wrapper,option,value",
    [
        (True, "sf_vec_size", 16),
        (True, "b_major", "n"),
        (True, "sf_fp8_dtype_override", "e5m3"),
        (True, "use_dynamic_sched", True),
        (True, "act_func", "situglu"),
        (False, "n", 512),
        (False, "vector_f32", True),
        (False, "use_dynamic_sched", True),
        (False, "act_func", "situglu"),
    ],
)
def test_jax_glu_rejects_unsupported_options(shared_wrapper, option, value):
    skip_unless_sm100()
    import cudnn

    inputs = {name: jnp.asarray(array) for name, array in problem(False, 1, True, False).items()}
    fn = partial(cudnn.grouped_gemm_glu_wrapper_sm100, sf_vec_size=32) if shared_wrapper else cudnn.grouped_gemm_glu_jax_sm100
    options = {"d_dtype": ml_dtypes.float8_e4m3fn, option: value}
    with pytest.raises(ValueError, match=option):
        fn(**inputs, **options)


@pytest.mark.parametrize(
    "backward,name,shape",
    [(False, "alpha", (1,)), (False, "norm_const", (0,)), (True, "alpha", (1,)), (True, "beta", (1,)), (True, "norm_const", (0,))],
)
def test_canonical_jax_rejects_malformed_scalar_buffers(backward, name, shape):
    skip_unless_sm100()
    import cudnn

    inputs = {key: jnp.asarray(array) for key, array in problem(backward, 4, True, False).items()}
    inputs[f"{name}_tensor"] = jnp.ones(shape, jnp.float32)
    bridge = grouped_gemm_dswiglu if backward else partial(cudnn.grouped_gemm_glu_jax_sm100, d_dtype=ml_dtypes.float8_e4m3fn)
    # Lowering first keeps a missing check from launching the kernel on malformed buffers.
    with pytest.raises(ValueError, match=name):
        jax.jit(bridge).lower(**inputs)
    with pytest.raises(ValueError, match=name):
        bridge(**inputs)


@pytest.mark.parametrize("backward", [False, True])
def test_canonical_jax_rejects_invalid_sf(backward):
    skip_unless_sm100()
    import cudnn

    inputs = {name: jnp.asarray(array) for name, array in problem(backward, 1, True, False).items()}
    inputs["sfa_tensor"] = inputs["sfa_tensor"][:-1]
    bridge = JAX_ENTRY["dswiglu" if backward else "swiglu"]
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
        fwd_result = grouped_gemm_swiglu(**fwd)
        bwd_result = grouped_gemm_dswiglu(**bwd, c_tensor=fwd_result["c_tensor"])
        return fwd_result, bwd_result

    inputs = tuple({name: jnp.asarray(array) for name, array in values.items()} for values in (forward, backward))
    compiled = jax.jit(step, compiler_options={"xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL"})
    for _ in range(2):
        fwd_result, bwd_result = compiled(*inputs)
        assert_outputs(fwd_result, forward_ref)
        assert_outputs(bwd_result, backward_ref)


@pytest.mark.parametrize("shared_wrapper", [False, True])
def test_canonical_jax_without_torch(shared_wrapper):
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
        import cudnn.jax
        assert "cudnn.gemm.cutedsl.grouped.swiglu.jax_api" not in sys.modules
        assert "cudnn.gemm.cutedsl.grouped.glu.jax_api" not in sys.modules
        from cudnn.gemm.cutedsl.grouped.swiglu.jax_api import grouped_gemm_swiglu
        from cudnn.gemm.cutedsl.grouped.dswiglu.jax_api import grouped_gemm_dswiglu
        fp8 = ml_dtypes.float8_e5m2
        a = jnp.ones((256, 256), fp8)
        b = jnp.ones((1, 256, 256), fp8)
        sf = jnp.full((1, 2, 2, 32, 4, 4), 127, jnp.uint8)
        offsets = jnp.array([256], jnp.int32)
        alpha = jnp.ones((1,), jnp.float32)
        prob = jnp.ones((256,), jnp.float32)
        norm = jnp.array([0.01], jnp.float32)
        fwd = partial(grouped_gemm_swiglu, d_dtype=fp8)
        out = jax.jit(fwd)(a, b, sf, sf, offsets, alpha, prob_tensor=prob, norm_const_tensor=norm)
        np.testing.assert_array_equal(np.asarray(out['c_tensor']).astype(np.float32), 256)
        c = jnp.ones((256, 512), jnp.bfloat16)
        bwd = grouped_gemm_dswiglu
        out = jax.jit(bwd)(a, b, c, sf, sf, offsets, alpha, alpha, prob, norm)
        assert np.isfinite(np.asarray(out['dprob_tensor'])).all()
        assert sys.modules['torch'] is None
    """)
    if shared_wrapper:
        script = script.replace(
            "from cudnn.gemm.cutedsl.grouped.swiglu.jax_api import grouped_gemm_swiglu\n"
            "from cudnn.gemm.cutedsl.grouped.dswiglu.jax_api import grouped_gemm_dswiglu",
            "from cudnn import grouped_gemm_swiglu_wrapper_sm100, grouped_gemm_dswiglu_wrapper_sm100\n"
            "grouped_gemm_swiglu = partial(grouped_gemm_swiglu_wrapper_sm100, sf_vec_size=32)\n"
            "grouped_gemm_dswiglu = partial(grouped_gemm_dswiglu_wrapper_sm100, sf_vec_size=32, d_dtype=ml_dtypes.float8_e4m3fn)",
        )
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
    bridge = JAX_ENTRY[name]
    compiled = jax.jit(bridge, compiler_options={"xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL"})
    inputs = {name: jnp.asarray(array) for name, array in arrays.items()}
    jax.block_until_ready(compiled(**inputs))
    arrays["padded_offsets"] = np.array([0, 256, 256, 768], np.int32)
    inputs["padded_offsets"] = jnp.asarray(arrays["padded_offsets"])
    poison_allocator()
    result = jax.block_until_ready(compiled(**inputs))
    reference = getattr(cudnn, f"grouped_gemm_{name}_wrapper_sm100")(**torch_inputs(arrays), d_dtype=torch.float8_e4m3fn, sf_vec_size=32)
    keys = ("d_row_tensor", "d_col_tensor", "dprob_tensor") if backward else ("c_tensor", "d_tensor", "d_col_tensor")
    for key in keys:
        actual = np.asarray(result[key]).astype(np.float32)
        expected = reference[key].float().cpu().numpy()
        np.testing.assert_allclose(actual[:768], expected[:768], rtol=1e-4 if key == "dprob_tensor" else 0, atol=1e-4 if key == "dprob_tensor" else 0)
        if key == "dprob_tensor":
            np.testing.assert_array_equal(actual[768:], 0)
    # Rows below 768 are the first six 128-row scale atoms.
    for key, active in (("sfd_row_tensor", np.s_[:, :6]), ("sfd_col_tensor", np.s_[:, :, :6])):
        expected = reference[key].contiguous().view(torch.uint8).cpu().numpy()
        np.testing.assert_array_equal(np.asarray(result[key]).view(np.uint8)[active], expected[active])


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
    ],
)
@pytest.mark.parametrize("shared_wrapper", [False, True])
def test_jax_rejects_unsupported_options(backward, option, value, shared_wrapper):
    skip_unless_sm100()
    import cudnn

    inputs = {name: jnp.asarray(array) for name, array in problem(backward, 1, True, False).items()}
    wrapper = JAX_ENTRY["dswiglu" if backward else "swiglu"]
    options = {}
    if shared_wrapper:
        wrapper = getattr(cudnn, f"grouped_gemm_{'dswiglu' if backward else 'swiglu'}_wrapper_sm100")
        options.update(d_dtype=ml_dtypes.float8_e4m3fn, sf_vec_size=32)
    options[option] = value
    with pytest.raises(ValueError if shared_wrapper else TypeError, match=option):
        jax.jit(partial(wrapper, **options))(**inputs)


@pytest.mark.parametrize("option", ["dprob_tensor_buf", "amax_tensor_buf", "epilogue_op"])
@pytest.mark.parametrize("shared_wrapper", [False, True])
def test_jax_backward_rejects_output_buffers_and_epilogues(option, shared_wrapper):
    skip_unless_sm100()

    inputs = {name: jnp.asarray(array) for name, array in problem(True, 1, True, False).items()}
    options = {option: "relu" if option == "epilogue_op" else jnp.empty((256,))}
    entry = grouped_gemm_dswiglu
    if shared_wrapper:
        from cudnn import grouped_gemm_dswiglu_wrapper_sm100

        entry = partial(grouped_gemm_dswiglu_wrapper_sm100, d_dtype=ml_dtypes.float8_e4m3fn, sf_vec_size=32)
    with pytest.raises(ValueError if shared_wrapper else TypeError, match=option):
        entry(**inputs, **options)


@pytest.mark.parametrize("backward", [False, True])
def test_jax_namespace_rejects_mixed_frameworks_and_missing_alpha(backward):
    skip_unless_sm100()
    import cudnn

    arrays = problem(backward, 1, True, False)
    inputs = {name: jnp.asarray(array) for name, array in arrays.items()}
    wrapper = JAX_ENTRY["dswiglu" if backward else "swiglu"]
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


def test_jax_backward_rejects_e5m2_output():
    skip_unless_sm100()

    inputs = {name: jnp.asarray(array) for name, array in problem(True, 1, True, False).items()}
    with pytest.raises(ValueError, match="d_dtype must be e4m3"):
        grouped_gemm_dswiglu(**inputs, d_dtype=ml_dtypes.float8_e5m2)


@pytest.mark.parametrize("operation", ["swiglu", "dswiglu"])
def test_torch_wrapper_without_jax(operation):
    import subprocess
    import sys

    script = f"""
import sys
sys.modules['jax'] = None
import cudnn
assert callable(cudnn.grouped_gemm_{operation}_wrapper_sm100)
assert sys.modules['jax'] is None
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backward", [False, True])
def test_shared_wrapper_keeps_existing_defaults(backward):
    skip_unless_sm100()
    import cudnn

    wrapper = getattr(cudnn, f"grouped_gemm_{'dswiglu' if backward else 'swiglu'}_wrapper_sm100")
    inputs = {name: jnp.asarray(array) for name, array in problem(backward, 1, True, False).items()}
    with pytest.raises(ValueError, match="sf_vec_size"):
        wrapper(**inputs)
    with pytest.raises(ValueError, match="FP8 output dtype"):
        wrapper(**inputs, sf_vec_size=32)
