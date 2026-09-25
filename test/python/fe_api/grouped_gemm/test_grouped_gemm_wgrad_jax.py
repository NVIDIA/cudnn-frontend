# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
JAX coverage for the SM100 grouped GEMM wgrad wrapper.

JAX contract: BF16 backend only, with A K-major and B N-major (both plain C-contiguous
JAX arrays); dense outputs are C-contiguous (expert, M, N) arrays, and discrete outputs
take a packed-uint8 wgrad_ptrs array (8 bytes per pointer, since JAX truncates int64
without x64 mode). Outputs are checked bit-identical against the torch wrapper run on
identical input bytes. The block-scaled backend is rejected (its B operand is K-major,
i.e. column-major, and fp4 operands are K-packed -- neither is expressible as a
row-major JAX array).
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
ml_dtypes = pytest.importorskip("ml_dtypes")
torch = pytest.importorskip("torch")
import jax.numpy as jnp

from fe_api.gemm.test_gemm_amax_jax import device_sync, skip_unless_sm100


def make_problem(m=128, n=128, group_k_list=(256, 256)):
    rng = np.random.default_rng(20260809)
    tokens = sum(group_k_list)
    a_np = (rng.standard_normal((m, tokens), dtype=np.float32) * 0.125).astype(ml_dtypes.bfloat16)  # K-major
    b_np = (rng.standard_normal((tokens, n), dtype=np.float32) * 0.125).astype(ml_dtypes.bfloat16)  # N-major
    offsets_np = np.array([sum(group_k_list[: i + 1]) for i in range(len(group_k_list))], dtype=np.int32)
    return a_np, b_np, offsets_np


def to_torch_bf16(x_np):
    return torch.from_numpy(x_np.view(np.uint8)).view(torch.bfloat16).reshape(x_np.shape).cuda()


def reference_wgrad(a_np, b_np, offsets_np):
    a32 = a_np.astype(np.float32)
    b32 = b_np.astype(np.float32)
    result = []
    begin = 0
    for end in offsets_np.tolist():
        result.append(a32[:, begin:end] @ b32[begin:end, :])
        begin = end
    return np.stack(result)


@pytest.mark.L0
def test_grouped_gemm_wgrad_jax_dense_matches_torch():
    skip_unless_sm100()
    from cudnn import grouped_gemm_wgrad_wrapper_sm100

    a_np, b_np, offsets_np = make_problem()

    kwargs = dict(
        sfa_tensor=None,
        sfb_tensor=None,
        output_mode="dense",
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
    )

    result_t = grouped_gemm_wgrad_wrapper_sm100(
        a_tensor=to_torch_bf16(a_np),
        b_tensor=to_torch_bf16(b_np),
        offsets_tensor=torch.from_numpy(offsets_np).cuda(),
        **kwargs,
    )
    torch.cuda.synchronize()

    a_j, b_j, offsets_j = (jnp.asarray(x) for x in (a_np, b_np, offsets_np))
    jax.block_until_ready((a_j, b_j, offsets_j))
    result_j = grouped_gemm_wgrad_wrapper_sm100(
        a_tensor=a_j,
        b_tensor=b_j,
        offsets_tensor=offsets_j,
        **kwargs,
    )
    device_sync()  # eager JAX path runs on the CUDA legacy default stream

    wgrad_j = np.asarray(result_j["wgrad_tensor"]).astype(np.float32)
    wgrad_t = result_t["wgrad_tensor"].float().cpu().numpy()
    np.testing.assert_array_equal(wgrad_j, wgrad_t, err_msg="wgrad dense: JAX output differs from torch output on identical input bytes")
    # Value sanity against the fp32 grouped-mm oracle (upstream tolerance).
    np.testing.assert_allclose(wgrad_j, reference_wgrad(a_np, b_np, offsets_np), rtol=3e-2, atol=8e-2)


@pytest.mark.L0
def test_grouped_gemm_wgrad_jax_discrete_matches_torch():
    skip_unless_sm100()
    from cudnn import grouped_gemm_wgrad_wrapper_sm100

    a_np, b_np, offsets_np = make_problem()
    m, n, experts = a_np.shape[0], b_np.shape[1], len(offsets_np)

    kwargs = dict(
        sfa_tensor=None,
        sfb_tensor=None,
        output_mode="discrete",
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
    )

    # ---- torch run: discrete mode with auto-generated pointers ----
    result_t = grouped_gemm_wgrad_wrapper_sm100(
        a_tensor=to_torch_bf16(a_np),
        b_tensor=to_torch_bf16(b_np),
        offsets_tensor=torch.from_numpy(offsets_np).cuda(),
        **kwargs,
    )
    torch.cuda.synchronize()
    wgrad_t = result_t["wgrad_tensor"].float().cpu().numpy()

    a_j, b_j, offsets_j = (jnp.asarray(x) for x in (a_np, b_np, offsets_np))
    jax.block_until_ready((a_j, b_j, offsets_j))

    # ---- jax run 1: auto-allocated wgrad tensor and auto-generated pointers ----
    result_j = grouped_gemm_wgrad_wrapper_sm100(
        a_tensor=a_j,
        b_tensor=b_j,
        offsets_tensor=offsets_j,
        **kwargs,
    )
    device_sync()
    np.testing.assert_array_equal(
        np.asarray(result_j["wgrad_tensor"]).astype(np.float32),
        wgrad_t,
        err_msg="wgrad discrete: JAX output differs from torch output on identical input bytes",
    )

    # ---- jax run 2: explicit packed-uint8 wgrad_ptrs into per-expert buffers ----
    expert_outputs = [jax.block_until_ready(jnp.zeros((m, n), dtype=ml_dtypes.bfloat16)) for _ in range(experts)]
    ptr_values = np.array([buf.unsafe_buffer_pointer() for buf in expert_outputs], dtype=np.int64)
    wgrad_ptrs_j = jax.block_until_ready(jnp.asarray(ptr_values.view(np.uint8)))
    grouped_gemm_wgrad_wrapper_sm100(
        a_tensor=a_j,
        b_tensor=b_j,
        offsets_tensor=offsets_j,
        wgrad_ptrs=wgrad_ptrs_j,
        **kwargs,
    )
    device_sync()
    for expert in range(experts):
        np.testing.assert_array_equal(
            np.asarray(expert_outputs[expert]).astype(np.float32),
            wgrad_t[expert],
            err_msg=f"wgrad discrete expert {expert}: explicit-ptrs JAX output differs from torch output",
        )


@pytest.mark.L0
def test_grouped_gemm_wgrad_jax_jit_matches_eager():
    """XLA custom-call entry point (cudnn.jax.call): bit-identical to the eager JAX wrapper.

    The jit entry is discrete-mode only: the per-expert outputs are caller-owned
    external buffers reached through the wgrad_ptrs input array (not XLA outputs), so
    each run gets fresh zero-filled buffers and the comparison reads them after
    blocking on the returned token plus a device sync.
    """
    import cutlass.jax

    if not cutlass.jax.is_available():
        pytest.skip("CuTeDSL JAX extensions unavailable (jax >= 0.5 required)")
    skip_unless_sm100()
    from cudnn import grouped_gemm_wgrad_jax_sm100, grouped_gemm_wgrad_wrapper_sm100

    a_np, b_np, offsets_np = make_problem()
    m, n, experts = a_np.shape[0], b_np.shape[1], len(offsets_np)

    a_j, b_j, offsets_j = (jnp.asarray(x) for x in (a_np, b_np, offsets_np))
    jax.block_until_ready((a_j, b_j, offsets_j))

    # Eager JAX wrapper baseline (dense output) on the same bytes and kernel config.
    result_eager = grouped_gemm_wgrad_wrapper_sm100(
        a_tensor=a_j,
        b_tensor=b_j,
        sfa_tensor=None,
        sfb_tensor=None,
        offsets_tensor=offsets_j,
        output_mode="dense",
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
    )
    device_sync()  # eager JAX path runs on the CUDA legacy default stream
    expected = np.asarray(result_eager["wgrad_tensor"]).view(np.uint8)  # (experts, m, 2n) bytes

    def run_and_check(fn, label):
        # Fresh zero-filled external buffers per run so each check observes that
        # run's writes (the kernel fully overwrites; buffers are immutable JAX
        # arrays mutated behind XLA's back through their raw addresses).
        expert_outputs = [jax.block_until_ready(jnp.zeros((m, n), dtype=ml_dtypes.bfloat16)) for _ in range(experts)]
        ptr_values = np.array([buf.unsafe_buffer_pointer() for buf in expert_outputs], dtype=np.int64)
        wgrad_ptrs_j = jax.block_until_ready(jnp.asarray(ptr_values.view(np.uint8)))
        token = fn(wgrad_ptrs_j)
        assert token.shape == (m, n)
        jax.block_until_ready(token)
        device_sync()  # external-buffer writes are outside XLA's dataflow
        for expert in range(experts):
            np.testing.assert_array_equal(
                np.asarray(expert_outputs[expert]).view(np.uint8),
                expected[expert],
                err_msg=f"wgrad jit expert {expert} ({label}): output differs from eager wrapper output on identical input bytes",
            )

    kwargs = dict(mma_tiler_mn=(128, 128), cluster_shape_mn=(1, 1))

    # Eager custom call
    run_and_check(lambda ptrs: grouped_gemm_wgrad_jax_sm100(a_j, b_j, offsets_j, ptrs, **kwargs), "eager custom call")

    # Under jax.jit, twice (compiled-kernel / registration cache).
    jitted = jax.jit(lambda a, b, offsets, ptrs: grouped_gemm_wgrad_jax_sm100(a, b, offsets, ptrs, **kwargs))
    run_and_check(lambda ptrs: jitted(a_j, b_j, offsets_j, ptrs), "jit call 1")
    run_and_check(lambda ptrs: jitted(a_j, b_j, offsets_j, ptrs), "jit call 2")


@pytest.mark.L0
def test_grouped_gemm_wgrad_jax_block_scaled_rejected():
    skip_unless_sm100()
    import cudnn
    from cudnn import grouped_gemm_wgrad_wrapper_sm100

    rng = np.random.default_rng(0)
    m, n, tokens = 128, 128, 512
    a_j = jnp.asarray(rng.integers(0, 100, (m, tokens), dtype=np.uint8).view(ml_dtypes.float8_e4m3fn))
    b_j = jnp.asarray(rng.integers(0, 100, (tokens, n), dtype=np.uint8).view(ml_dtypes.float8_e4m3fn))
    sfa_j = jnp.asarray(np.full((128, 32), 127, dtype=np.uint8).view(ml_dtypes.float8_e8m0fnu))
    sfb_j = jnp.asarray(np.full((128, 32), 127, dtype=np.uint8).view(ml_dtypes.float8_e8m0fnu))
    offsets_j = jnp.asarray(np.array([256, 512], dtype=np.int32))

    with pytest.raises(ValueError, match="not expressible as JAX arrays"):
        grouped_gemm_wgrad_wrapper_sm100(
            a_tensor=a_j,
            b_tensor=b_j,
            sfa_tensor=sfa_j,
            sfb_tensor=sfb_j,
            offsets_tensor=offsets_j,
            output_mode="dense",
            sf_vec_size=32,
        )

    # Class API path rejects too.
    op = cudnn.GroupedGemmWgradSm100(
        sample_a=a_j,
        sample_b=b_j,
        sample_sfa=sfa_j,
        sample_sfb=sfb_j,
        sample_offsets=offsets_j,
        sample_wgrad=jnp.zeros((2, m, n), dtype=ml_dtypes.bfloat16),
        sf_vec_size=32,
    )
    with pytest.raises(ValueError, match="not expressible as JAX arrays"):
        op.check_support()
