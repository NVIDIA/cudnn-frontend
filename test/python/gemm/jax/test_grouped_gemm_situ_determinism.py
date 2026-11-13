# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SiTU JAX byte replay and expert isolation; atomic dprob has no determinism guarantee."""

from functools import partial

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

ml_dtypes = pytest.importorskip("ml_dtypes")
torch = pytest.importorskip("torch")

from determinism_utils import assert_bitwise, assert_replays, assert_golden
from gemm.jax.test_grouped_gemm_canonical_jax import assert_situ_math_reference, problem, skip_unless_situ_supported

pytestmark = pytest.mark.L0


def active_outputs(result, rows, start=0):
    outputs = {}
    for name, value in result.items():
        if value is None:
            continue
        array = np.asarray(value)
        if name == "sfd_row_tensor":
            array = array[:, start // 128 : (start + rows) // 128]
        elif name == "sfd_col_tensor":
            # Expert-packed column scales occupy a contiguous prefix of the capacity.
            per_tile = array.shape[1] * 512
            array = array.reshape(-1)[start // 128 * per_tile : (start + rows) // 128 * per_tile]
        else:
            array = array[start : start + rows]
        # dprob uses atomic reductions; it is checked numerically by the parity suite.
        if name != "dprob_tensor":
            assert np.isfinite(array.astype(np.float32)).all(), name
            outputs[name] = torch.from_numpy(np.array(array, copy=True).view(np.uint8))
    return outputs


@pytest.mark.parametrize("backward", [False, True], ids=["forward", "backward"])
@pytest.mark.parametrize("rows", [256, 1024, 32768])
def test_situ_jax_replay_and_capacity_isolation(backward, rows):
    skip_unless_situ_supported()
    import cudnn

    # A single active expert plus an empty expert and an unused allocation tail.
    arrays = problem(backward, 2, False, False, n=512)
    rng = np.random.default_rng(710)
    capacity = rows + 256
    arrays["a_tensor"] = rng.integers(-2, 3, (capacity, 256)).astype(ml_dtypes.float8_e4m3fn)
    arrays["sfa_tensor"] = np.full((1, capacity // 128, 2, 32, 4, 4), 127, np.uint8)
    arrays["sfb_tensor"] = np.full_like(arrays["sfb_tensor"], 127)
    arrays["prob_tensor"] = np.ones(capacity, np.float32)
    arrays["padded_offsets"] = np.array([rows, rows], np.int32)
    if backward:
        arrays["c_tensor"] = rng.uniform(-2, 2, (capacity, 1024)).astype(ml_dtypes.bfloat16)
    entry = cudnn.grouped_gemm_dglu_jax_sm100 if backward else cudnn.grouped_gemm_glu_jax_sm100
    options = dict(
        act_func="dsituglu" if backward else "situglu",
        discrete_col_sfd=True,
        situ_beta1=4.0,
        situ_beta2=25.0,
        d_dtype=ml_dtypes.float8_e4m3fn,
    )
    if not backward:
        options["generate_c"] = True
    compiled = jax.jit(partial(entry, **options))

    def run(check_reference=False):
        inputs = {name: jnp.asarray(np.array(array, copy=True)) for name, array in arrays.items()}
        concurrent = jax.jit(lambda x: x @ x)(jnp.ones((256, 256), jnp.float32))
        result = jax.block_until_ready(compiled(**inputs))
        concurrent.block_until_ready()
        if check_reference:
            assert_situ_math_reference(arrays, result, backward, (4.0, 25.0))
        return active_outputs(result, rows)

    baseline = assert_replays(run, repeats=4)
    arrays["a_tensor"][rows:] = np.nan
    arrays["prob_tensor"][rows:] = np.nan
    arrays["b_tensor"][1] *= -1
    if backward:
        arrays["c_tensor"][rows:] = np.nan
    # Unused capacity and an empty expert must not affect any active output byte.
    assert_bitwise(baseline, run(), "unused capacity and empty expert")
    if rows == 256:
        assert_bitwise(baseline, run(check_reference=True), "independently checked golden outputs")
        fixed_inputs = {name: torch.from_numpy(np.array(array, copy=True).view(np.uint8)) for name, array in arrays.items()}
        assert_golden(
            "jax_grouped_dsitu_mxfp8" if backward else "jax_grouped_situ_mxfp8",
            fixed_inputs,
            baseline,
        )


@pytest.mark.parametrize("backward", [False, True], ids=["forward", "backward"])
def test_situ_jax_expert_batch_invariance(backward):
    skip_unless_situ_supported()
    import cudnn

    base = problem(backward, 2, False, False, n=512)
    entry = cudnn.grouped_gemm_dglu_jax_sm100 if backward else cudnn.grouped_gemm_glu_jax_sm100
    options = dict(
        act_func="dsituglu" if backward else "situglu",
        discrete_col_sfd=True,
        situ_beta1=4.0,
        situ_beta2=25.0,
        d_dtype=ml_dtypes.float8_e4m3fn,
    )
    if not backward:
        options["generate_c"] = True
    compiled = jax.jit(partial(entry, **options))

    def run(order, target):
        experts = len(order)
        arrays = problem(backward, experts, False, False, n=512)
        arrays["a_tensor"] = np.concatenate([base["a_tensor"][i * 256 : (i + 1) * 256] for i in order])
        arrays["b_tensor"] = base["b_tensor"][list(order)]
        arrays["alpha_tensor"] = base["alpha_tensor"][list(order)]
        arrays["prob_tensor"] = np.concatenate([base["prob_tensor"][i * 256 : (i + 1) * 256] for i in order])
        arrays["sfa_tensor"].fill(127)
        arrays["sfb_tensor"].fill(127)
        if backward:
            arrays["c_tensor"] = np.concatenate([base["c_tensor"][i * 256 : (i + 1) * 256] for i in order])
            arrays["beta_tensor"] = base["beta_tensor"][list(order)]
        inputs = {name: jnp.asarray(array) for name, array in arrays.items()}
        result = jax.block_until_ready(compiled(**inputs))
        for name, value in inputs.items():
            np.testing.assert_array_equal(np.asarray(value).view(np.uint8), arrays[name].view(np.uint8))
        return active_outputs(result, 256, target * 256)

    single = run((0,), 0)
    assert_bitwise(single, run((0, 1), 0), "packed with neighbor")
    assert_bitwise(single, run((1, 0), 1), "expert reordered")
    assert_bitwise(single, run((0, 0), 1), "expert duplicated")
    base["a_tensor"][256:] *= -1
    base["b_tensor"][1] *= -1
    assert_bitwise(single, run((0, 1), 0), "neighbor values changed")
