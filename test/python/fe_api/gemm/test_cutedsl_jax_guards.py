# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
The grouped/discrete-grouped/proj_rope CuTeDSL APIs are torch-only for now: they must
reject JAX arrays with a clear error at the public entry points (instead of failing
deep inside pointer-array/workspace machinery), and their modules must import without
torch. Real JAX support for these APIs is future work.
"""

import inspect

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

TORCH_ONLY_WRAPPERS = [
    "grouped_gemm_wrapper_sm100",
    "grouped_gemm_swiglu_wrapper_sm100",
    "grouped_gemm_dswiglu_wrapper_sm100",
    "grouped_gemm_srelu_wrapper_sm100",
    "grouped_gemm_dsrelu_wrapper_sm100",
    "grouped_gemm_glu_wrapper_sm100",
    "grouped_gemm_dglu_wrapper_sm100",
    "grouped_gemm_glu_hadamard_wrapper_sm100",
    "grouped_gemm_quant_wrapper_sm100",
    "grouped_gemm_wgrad_wrapper_sm100",
    "discrete_grouped_gemm_swiglu_wrapper_sm100",
    "discrete_grouped_gemm_dswiglu_wrapper_sm100",
    "gemm_proj_rope_mxfp8_wrapper_sm100",
]


@pytest.mark.L0
@pytest.mark.parametrize("wrapper_name", TORCH_ONLY_WRAPPERS)
def test_torch_only_wrapper_rejects_jax(wrapper_name):
    import cudnn

    try:
        wrapper = getattr(cudnn, wrapper_name)
    except (AttributeError, ImportError):
        pytest.skip(f"{wrapper_name} is not exported in this build")

    dummy = jnp.asarray(np.zeros((16, 16, 1), dtype=np.float32))
    # Fill every required parameter with the dummy JAX array; the framework guard
    # runs on the first tensor argument before any validation.
    kwargs = {}
    for name, param in inspect.signature(wrapper).parameters.items():
        if param.default is inspect.Parameter.empty and param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            kwargs[name] = dummy

    with pytest.raises(ValueError, match="torch tensors only"):
        wrapper(**kwargs)
