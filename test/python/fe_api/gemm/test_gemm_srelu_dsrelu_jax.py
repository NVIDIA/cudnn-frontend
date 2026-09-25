# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
JAX tests for the type-erased GemmSreluSm100 / GemmDsreluSm100 APIs.

Strategy: run each wrapper once with torch tensors and once with JAX arrays built
from the same bytes, and require bit-identical outputs — the two framework paths
share one compiled kernel, so any divergence is a metadata/layout bug.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
ml_dtypes = pytest.importorskip("ml_dtypes")
torch = pytest.importorskip("torch")
import jax.numpy as jnp

from fe_api.gemm.test_gemm_amax_jax import device_sync, skip_unless_sm100

from cudnn.api_base import ceil_div


def make_inputs(m, n, k, l, sf_vec_size, rng):
    rest_k = ceil_div(ceil_div(k, sf_vec_size), 4)
    a_np = rng.standard_normal((m, k, l), dtype=np.float32).astype(ml_dtypes.float8_e4m3fn)
    b_np = rng.standard_normal((n, k, l), dtype=np.float32).astype(ml_dtypes.float8_e4m3fn)
    # e8m0 value 1.0 == byte 127; physical C-contiguous atom shape (L, MN', K', 32, 4, 4)
    sfa_np = np.full((l, ceil_div(m, 128), rest_k, 32, 4, 4), 127, dtype=np.uint8).view(ml_dtypes.float8_e8m0fnu)
    sfb_np = np.full((l, ceil_div(n, 128), rest_k, 32, 4, 4), 127, dtype=np.uint8).view(ml_dtypes.float8_e8m0fnu)
    prob_np = rng.random((m, 1, l), dtype=np.float32)
    return a_np, b_np, sfa_np, sfb_np, prob_np


def to_torch(x, dtype):
    return torch.from_numpy(np.ascontiguousarray(x).view(np.uint8)).view(dtype).reshape(x.shape).cuda()


@pytest.mark.L0
def test_gemm_srelu_jax_matches_torch():
    skip_unless_sm100()
    import cudnn

    m, n, k, l = 256, 256, 512, 1
    sf_vec_size = 32
    rng = np.random.default_rng(0)
    a_np, b_np, sfa_np, sfb_np, prob_np = make_inputs(m, n, k, l, sf_vec_size, rng)

    kwargs = dict(c_dtype="bfloat16", d_dtype="bfloat16", sf_vec_size=sf_vec_size)
    res_t = cudnn.gemm_srelu_wrapper_sm100(
        a_tensor=to_torch(a_np, torch.float8_e4m3fn),
        b_tensor=to_torch(b_np, torch.float8_e4m3fn),
        sfa_tensor=to_torch(sfa_np, torch.float8_e8m0fnu),
        sfb_tensor=to_torch(sfb_np, torch.float8_e8m0fnu),
        prob_tensor=torch.from_numpy(prob_np).cuda(),
        **kwargs,
    )
    torch.cuda.synchronize()

    a_j, b_j, sfa_j, sfb_j, prob_j = (jnp.asarray(x) for x in (a_np, b_np, sfa_np, sfb_np, prob_np))
    jax.block_until_ready((a_j, b_j, sfa_j, sfb_j, prob_j))
    res_j = cudnn.gemm_srelu_wrapper_sm100(
        a_tensor=a_j,
        b_tensor=b_j,
        sfa_tensor=sfa_j,
        sfb_tensor=sfb_j,
        prob_tensor=prob_j,
        **kwargs,
    )
    device_sync()  # eager JAX path runs on the CUDA legacy default stream

    for key in ("c_tensor", "d_tensor"):
        np.testing.assert_array_equal(
            np.asarray(res_j[key]).astype(np.float32),
            res_t[key].float().cpu().numpy(),
            err_msg=f"srelu {key}: JAX output differs from torch output on identical input bytes",
        )


@pytest.mark.L0
def test_gemm_dsrelu_jax_matches_torch():
    skip_unless_sm100()
    import cudnn

    m, n, k, l = 256, 256, 512, 1
    sf_vec_size = 32
    rng = np.random.default_rng(1)
    a_np, b_np, sfa_np, sfb_np, prob_np = make_inputs(m, n, k, l, sf_vec_size, rng)
    c_in_np = rng.standard_normal((m, n, l), dtype=np.float32).astype(ml_dtypes.bfloat16)

    kwargs = dict(d_dtype="bfloat16", sf_vec_size=sf_vec_size)
    res_t = cudnn.gemm_dsrelu_wrapper_sm100(
        a_tensor=to_torch(a_np, torch.float8_e4m3fn),
        b_tensor=to_torch(b_np, torch.float8_e4m3fn),
        c_tensor=to_torch(c_in_np, torch.bfloat16),
        sfa_tensor=to_torch(sfa_np, torch.float8_e8m0fnu),
        sfb_tensor=to_torch(sfb_np, torch.float8_e8m0fnu),
        prob_tensor=torch.from_numpy(prob_np).cuda(),
        **kwargs,
    )
    torch.cuda.synchronize()

    a_j, b_j, c_j, sfa_j, sfb_j, prob_j = (jnp.asarray(x) for x in (a_np, b_np, c_in_np, sfa_np, sfb_np, prob_np))
    jax.block_until_ready((a_j, b_j, c_j, sfa_j, sfb_j, prob_j))
    res_j = cudnn.gemm_dsrelu_wrapper_sm100(
        a_tensor=a_j,
        b_tensor=b_j,
        c_tensor=c_j,
        sfa_tensor=sfa_j,
        sfb_tensor=sfb_j,
        prob_tensor=prob_j,
        **kwargs,
    )
    device_sync()

    for key in ("d_tensor", "dprob_tensor"):
        np.testing.assert_array_equal(
            np.asarray(res_j[key]).astype(np.float32),
            res_t[key].float().cpu().numpy(),
            err_msg=f"dsrelu {key}: JAX output differs from torch output on identical input bytes",
        )


@pytest.mark.L0
def test_gemm_srelu_dsrelu_jax_errors():
    skip_unless_sm100()
    import cudnn

    m, n, k, l = 256, 256, 512, 1
    rng = np.random.default_rng(2)
    a_np, b_np, sfa_np, sfb_np, prob_np = make_inputs(m, n, k, l, 32, rng)
    a_j, b_j, sfa_j, sfb_j, prob_j = (jnp.asarray(x) for x in (a_np, b_np, sfa_np, sfb_np, prob_np))

    with pytest.raises(ValueError, match="row-major"):
        cudnn.gemm_srelu_wrapper_sm100(a_j, b_j, sfa_j, sfb_j, prob_j, c_major="m", sf_vec_size=32)

    with pytest.raises(ValueError, match="Unsupported tensor framework"):
        cudnn.gemm_srelu_wrapper_sm100(a_np, b_np, sfa_np, sfb_np, prob_np, sf_vec_size=32)


@pytest.mark.L0
def test_gemm_srelu_dsrelu_jax_jit_matches_eager():
    """The XLA custom-call entry points must agree bit-for-bit with the eager wrappers."""
    skip_unless_sm100()
    import cudnn
    from cudnn import gemm_dsrelu_jax_sm100, gemm_srelu_jax_sm100

    m, n, k, l = 256, 256, 512, 1
    sf_vec_size = 32
    rng = np.random.default_rng(3)
    a_np, b_np, sfa_np, sfb_np, prob_np = make_inputs(m, n, k, l, sf_vec_size, rng)

    a_j, b_j, sfa_j, sfb_j, prob_j = (jnp.asarray(x) for x in (a_np, b_np, sfa_np, sfb_np, prob_np))
    jax.block_until_ready((a_j, b_j, sfa_j, sfb_j, prob_j))

    eager = cudnn.gemm_srelu_wrapper_sm100(
        a_tensor=a_j, b_tensor=b_j, sfa_tensor=sfa_j, sfb_tensor=sfb_j, prob_tensor=prob_j, c_dtype="bfloat16", d_dtype="bfloat16", sf_vec_size=sf_vec_size
    )
    device_sync()

    jitted = jax.jit(lambda a, b, sfa, sfb, prob: gemm_srelu_jax_sm100(a, b, sfa, sfb, prob, c_dtype="bfloat16", d_dtype="bfloat16", sf_vec_size=sf_vec_size))
    for _ in range(2):  # repeat: donation safety
        c_jit, d_jit = jitted(a_j, b_j, sfa_j, sfb_j, prob_j)
        jax.block_until_ready((c_jit, d_jit))  # XLA-stream ordered; no manual sync needed
        np.testing.assert_array_equal(np.asarray(c_jit).view(np.uint8), np.asarray(eager["c_tensor"]).view(np.uint8))
        np.testing.assert_array_equal(np.asarray(d_jit).view(np.uint8), np.asarray(eager["d_tensor"]).view(np.uint8))

    # backward
    c_in_np = rng.standard_normal((m, n, l), dtype=np.float32).astype(ml_dtypes.bfloat16)
    c_in_j = jnp.asarray(c_in_np)
    jax.block_until_ready(c_in_j)
    eager_b = cudnn.gemm_dsrelu_wrapper_sm100(
        a_tensor=a_j, b_tensor=b_j, c_tensor=c_in_j, sfa_tensor=sfa_j, sfb_tensor=sfb_j, prob_tensor=prob_j, d_dtype="bfloat16", sf_vec_size=sf_vec_size
    )
    device_sync()

    jitted_b = jax.jit(lambda a, b, c, sfa, sfb, prob: gemm_dsrelu_jax_sm100(a, b, c, sfa, sfb, prob, d_dtype="bfloat16", sf_vec_size=sf_vec_size))
    d_jit, dprob_jit = jitted_b(a_j, b_j, c_in_j, sfa_j, sfb_j, prob_j)
    jax.block_until_ready((d_jit, dprob_jit))
    np.testing.assert_array_equal(np.asarray(d_jit).view(np.uint8), np.asarray(eager_b["d_tensor"]).view(np.uint8))
    # dprob accumulates via FP32 atomic adds (ordering-nondeterministic): tight tolerance
    np.testing.assert_allclose(np.asarray(dprob_jit), np.asarray(eager_b["dprob_tensor"]), rtol=1e-5, atol=1e-5)
