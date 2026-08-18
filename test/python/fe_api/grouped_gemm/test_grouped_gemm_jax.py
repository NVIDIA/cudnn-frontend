# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
JAX coverage for the unfused SM100 grouped GEMM wrapper.

JAX contract: discrete weight mode only (dense mode's expert-outermost strided B and the
column-major bias layout are not expressible as row-major JAX arrays), with b_ptrs built
from per-expert weight pointers — as a packed uint8 array (8 bytes per pointer) since JAX
truncates int64 without x64 mode. Outputs are checked bit-identical against the torch
wrapper run on identical input bytes (both paths share one compiled kernel).
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
ml_dtypes = pytest.importorskip("ml_dtypes")
torch = pytest.importorskip("torch")
import jax.numpy as jnp

from fe_api.gemm.test_gemm_amax_jax import device_sync, skip_unless_sm100


def make_problem(m=512, n=256, k=128, experts=2):
    rng = np.random.default_rng(20260716)
    a_np = (rng.standard_normal((m, k, 1), dtype=np.float32) * 0.125).astype(ml_dtypes.bfloat16)
    b_storage_np = (rng.standard_normal((experts, n, k), dtype=np.float32) * 0.125).astype(ml_dtypes.bfloat16)
    group_m = m // experts
    offsets_np = np.arange(group_m, m + 1, group_m, dtype=np.int32)
    alpha_np = np.array([0.75, -1.25][:experts], dtype=np.float32)
    prob_np = np.linspace(0.25, 0.875, m, dtype=np.float32).reshape(m, 1, 1)
    return a_np, b_storage_np, offsets_np, alpha_np, prob_np


@pytest.mark.L0
def test_grouped_gemm_jax_discrete_matches_torch():
    skip_unless_sm100()
    from cudnn import grouped_gemm_wrapper_sm100

    m, n, k, experts = 512, 256, 128, 2
    a_np, b_storage_np, offsets_np, alpha_np, prob_np = make_problem(m, n, k, experts)

    # ---- torch run (discrete mode, no bias) ----
    a_t = torch.from_numpy(a_np.view(np.uint8)).view(torch.bfloat16).reshape(m, k, 1).cuda()
    b_storage_t = torch.from_numpy(b_storage_np.view(np.uint8)).view(torch.bfloat16).reshape(experts, n, k).cuda()
    b_ptrs_t = torch.tensor([b_storage_t[i].data_ptr() for i in range(experts)], dtype=torch.int64, device="cuda")
    result_t = grouped_gemm_wrapper_sm100(
        a_tensor=a_t,
        padded_offsets=torch.from_numpy(offsets_np).cuda(),
        alpha_tensor=torch.from_numpy(alpha_np).cuda(),
        prob_tensor=torch.from_numpy(prob_np).cuda(),
        b_ptrs=b_ptrs_t,
        n=n,
        b_dtype=torch.bfloat16,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        generate_c=True,
    )
    torch.cuda.synchronize()

    # ---- jax run on identical bytes ----
    a_j = jnp.asarray(a_np)
    b_experts_j = [jnp.asarray(b_storage_np[i]) for i in range(experts)]  # per-expert (n, k) k-major
    offsets_j, alpha_j, prob_j = (jnp.asarray(x) for x in (offsets_np, alpha_np, prob_np))
    jax.block_until_ready((a_j, offsets_j, alpha_j, prob_j, *b_experts_j))

    # Packed uint8 pointer array (8 little-endian bytes per pointer): JAX truncates
    # int64 without x64 mode. The weight arrays must stay alive while the kernel runs.
    ptr_values = np.array([w.unsafe_buffer_pointer() for w in b_experts_j], dtype=np.int64)
    b_ptrs_j = jax.block_until_ready(jnp.asarray(ptr_values.view(np.uint8)))

    result_j = grouped_gemm_wrapper_sm100(
        a_tensor=a_j,
        padded_offsets=offsets_j,
        alpha_tensor=alpha_j,
        prob_tensor=prob_j,
        b_ptrs=b_ptrs_j,
        n=n,
        b_dtype="bfloat16",
        c_dtype="bfloat16",
        d_dtype="bfloat16",
        generate_c=True,
    )
    device_sync()  # eager JAX path runs on the CUDA legacy default stream

    for key in ("c_tensor", "d_tensor"):
        np.testing.assert_array_equal(
            np.asarray(result_j[key]).astype(np.float32),
            result_t[key].float().cpu().numpy(),
            err_msg=f"unfused grouped {key}: JAX output differs from torch output on identical input bytes",
        )


@pytest.mark.L0
def test_grouped_gemm_jax_jit_matches_eager():
    """XLA custom-call entry point (cudnn.jax.call): bit-identical to the eager JAX wrapper."""
    import cutlass.jax

    if not cutlass.jax.is_available():
        pytest.skip("CuTeDSL JAX extensions unavailable (jax >= 0.5 required)")
    skip_unless_sm100()
    from cudnn import grouped_gemm_jax_sm100, grouped_gemm_wrapper_sm100

    m, n, k, experts = 512, 256, 128, 2
    a_np, b_storage_np, offsets_np, alpha_np, prob_np = make_problem(m, n, k, experts)

    a_j = jnp.asarray(a_np)
    b_experts_j = [jnp.asarray(b_storage_np[i]) for i in range(experts)]  # per-expert (n, k) k-major
    offsets_j, alpha_j, prob_j = (jnp.asarray(x) for x in (offsets_np, alpha_np, prob_np))
    jax.block_until_ready((a_j, offsets_j, alpha_j, prob_j, *b_experts_j))
    ptr_values = np.array([w.unsafe_buffer_pointer() for w in b_experts_j], dtype=np.int64)
    b_ptrs_j = jax.block_until_ready(jnp.asarray(ptr_values.view(np.uint8)))

    # Eager wrapper baseline on the same bytes; the last padded offset covers every
    # row, so full-buffer bitwise comparison against the jit path is well-defined.
    result_eager = grouped_gemm_wrapper_sm100(
        a_tensor=a_j,
        padded_offsets=offsets_j,
        alpha_tensor=alpha_j,
        prob_tensor=prob_j,
        b_ptrs=b_ptrs_j,
        n=n,
        b_dtype="bfloat16",
        generate_c=True,
    )
    device_sync()  # eager JAX path runs on the CUDA legacy default stream
    expected = {key: np.asarray(result_eager[key]).view(np.uint8) for key in ("d_tensor", "c_tensor")}

    def check(d_tensor, c_tensor):
        jax.block_until_ready((d_tensor, c_tensor))
        for got, key in ((d_tensor, "d_tensor"), (c_tensor, "c_tensor")):
            np.testing.assert_array_equal(
                np.asarray(got).view(np.uint8),
                expected[key],
                err_msg=f"unfused grouped {key}: jit output differs from eager wrapper output on identical input bytes",
            )

    # Eager custom call
    check(*grouped_gemm_jax_sm100(a_j, offsets_j, alpha_j, b_ptrs_j, n, prob_j, generate_c=True))

    # Under jax.jit, twice (compiled-kernel / registration cache). n stays static.
    jitted = jax.jit(
        lambda a, offsets, alpha, ptrs, prob: grouped_gemm_jax_sm100(a, offsets, alpha, ptrs, n, prob, generate_c=True),
    )
    check(*jitted(a_j, offsets_j, alpha_j, b_ptrs_j, prob_j))
    check(*jitted(a_j, offsets_j, alpha_j, b_ptrs_j, prob_j))

    # generate_c=False returns (d, None)
    d_only, c_none = grouped_gemm_jax_sm100(a_j, offsets_j, alpha_j, b_ptrs_j, n, prob_j)
    jax.block_until_ready(d_only)
    assert c_none is None
    np.testing.assert_array_equal(np.asarray(d_only).view(np.uint8), expected["d_tensor"])


@pytest.mark.L0
def test_grouped_gemm_jax_errors():
    skip_unless_sm100()
    from cudnn import grouped_gemm_wrapper_sm100

    m, n, k, experts = 512, 256, 128, 2
    a_np, b_storage_np, offsets_np, alpha_np, prob_np = make_problem(m, n, k, experts)
    a_j = jnp.asarray(a_np)
    offsets_j, alpha_j, prob_j = (jnp.asarray(x) for x in (offsets_np, alpha_np, prob_np))
    b_dense_j = jnp.asarray(b_storage_np)  # (experts, n, k): not the dense-mode layout

    with pytest.raises(ValueError, match="not expressible as JAX arrays"):
        grouped_gemm_wrapper_sm100(
            a_tensor=a_j,
            padded_offsets=offsets_j,
            alpha_tensor=alpha_j,
            prob_tensor=prob_j,
            b_tensor=b_dense_j,
        )

    b_experts_j = [jnp.asarray(b_storage_np[i]) for i in range(experts)]
    jax.block_until_ready(b_experts_j)
    ptr_values = np.array([w.unsafe_buffer_pointer() for w in b_experts_j], dtype=np.int64)
    b_ptrs_j = jnp.asarray(ptr_values.view(np.uint8))
    bias_j = jnp.zeros((n, experts), dtype=ml_dtypes.bfloat16)
    with pytest.raises(ValueError, match="bias_tensor is not expressible"):
        grouped_gemm_wrapper_sm100(
            a_tensor=a_j,
            padded_offsets=offsets_j,
            alpha_tensor=alpha_j,
            prob_tensor=prob_j,
            b_ptrs=b_ptrs_j,
            n=n,
            b_dtype="bfloat16",
            bias_tensor=bias_j,
        )


@pytest.mark.L0
def test_grouped_gemm_jax_addressed_rows_survive_a_dirty_allocator():
    """Rows before ``padded_offsets[-1]`` are correct whatever the output buffer held.

    The outputs are no longer zero-initialized, so the rows the kernel does not
    address carry whatever XLA last left in the buffer. A test that runs on a clean
    allocator passes either way -- fresh device memory reads as zeros -- so this one
    dirties the allocator first and checks only the contract that still holds.
    """
    import cutlass.jax

    if not cutlass.jax.is_available():
        pytest.skip("CuTeDSL JAX extensions unavailable (jax >= 0.5 required)")
    skip_unless_sm100()
    from cudnn import grouped_gemm_jax_sm100, grouped_gemm_wrapper_sm100

    m, n, k, experts = 1024, 256, 128, 2
    a_np, b_storage_np, _, alpha_np, prob_np = make_problem(m, n, k, experts)
    # 256-aligned, and the last offset stops short of m, so the tail is never addressed
    offsets_np = np.array([256, 512], dtype=np.int32)

    to_torch = lambda x: torch.from_numpy(x.view(np.uint8)).view(torch.bfloat16).cuda()
    a_t = to_torch(a_np).reshape(m, k, 1)
    b_t = to_torch(b_storage_np).reshape(experts, n, k)
    b_ptrs_t = torch.tensor([b_t[i].data_ptr() for i in range(experts)], dtype=torch.int64, device="cuda")
    expected = grouped_gemm_wrapper_sm100(
        a_tensor=a_t,
        padded_offsets=torch.from_numpy(offsets_np).cuda(),
        alpha_tensor=torch.from_numpy(alpha_np).cuda(),
        prob_tensor=torch.from_numpy(prob_np).cuda(),
        b_ptrs=b_ptrs_t,
        n=n,
        b_dtype=torch.bfloat16,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
    )["d_tensor"]
    device_sync()

    a_j = jnp.asarray(a_np)
    weights = [jnp.asarray(b_storage_np[i]) for i in range(experts)]
    offsets_j, alpha_j, prob_j = (jnp.asarray(x) for x in (offsets_np, alpha_np, prob_np))
    jax.block_until_ready((a_j, offsets_j, alpha_j, prob_j, *weights))
    ptr_values = np.array([w.unsafe_buffer_pointer() for w in weights], dtype=np.int64)
    b_ptrs_j = jax.block_until_ready(jnp.asarray(ptr_values.view(np.uint8)))
    jitted = jax.jit(lambda a, off, alpha, ptrs, prob: grouped_gemm_jax_sm100(a, off, alpha, ptrs, n, prob)[0])

    # Poison the allocator so a reused output buffer cannot read back as zeros.
    for _ in range(8):
        junk = jax.block_until_ready(jnp.full((m, n, 1), -7.5, dtype=jnp.bfloat16))
        del junk

    actual = jax.block_until_ready(jitted(a_j, offsets_j, alpha_j, b_ptrs_j, prob_j))
    valid = int(offsets_np[-1])
    np.testing.assert_array_equal(
        np.asarray(actual)[:valid],
        expected[:valid].float().cpu().numpy().astype(np.asarray(actual).dtype),
    )
