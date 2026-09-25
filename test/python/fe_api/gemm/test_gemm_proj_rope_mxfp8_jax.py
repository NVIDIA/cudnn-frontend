# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
JAX coverage for the fused projection GEMM + RoPE + MXFP8 quantize wrapper.

JAX contract: w_out_in=True only (the [in, out] weight layout reaches the kernel
through a transposed strided view, which has no row-major JAX equivalent). Outputs
are checked bit-identical against the torch wrapper run on identical input bytes
(both paths share one compiled kernel).
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
ml_dtypes = pytest.importorskip("ml_dtypes")
torch = pytest.importorskip("torch")
import jax.numpy as jnp

from fe_api.gemm.test_gemm_amax_jax import device_sync, skip_unless_sm100
from fe_api.gemm.test_gemm_proj_rope_mxfp8_utils import BLOCK, HEAD_DIM, NUM_HEADS, Q_LORA, Q_OUT, QK_ROPE


def to_torch(x, dtype):
    return torch.from_numpy(np.ascontiguousarray(x).view(np.uint8)).view(dtype).reshape(x.shape).cuda()


def run_both(kwargs_np, jax_dtypes, torch_dtypes):
    """Run the wrapper with torch tensors and JAX arrays built from the same bytes."""
    import cudnn

    torch_kwargs = {name: to_torch(arr, torch_dtypes[name]) for name, arr in kwargs_np.items()}
    result_t = cudnn.gemm_proj_rope_mxfp8_wrapper_sm100(**torch_kwargs, w_out_in=True)
    torch.cuda.synchronize()

    jax_kwargs = {name: jnp.asarray(arr.view(jax_dtypes[name])) for name, arr in kwargs_np.items()}
    jax.block_until_ready(tuple(jax_kwargs.values()))
    result_j = cudnn.gemm_proj_rope_mxfp8_wrapper_sm100(**jax_kwargs, w_out_in=True)
    device_sync()  # eager JAX path runs on the CUDA legacy default stream

    for key in ("out_fp8_row", "out_scales_row", "out_fp8_col", "out_scales_col"):
        np.testing.assert_array_equal(
            np.asarray(result_j[key]).view(np.uint8),
            result_t[key].view(torch.uint8).cpu().numpy(),
            err_msg=f"proj_rope {key}: JAX output differs from torch output on identical input bytes",
        )


@pytest.mark.L0
def test_gemm_proj_rope_mxfp8_bf16in_jax_matches_torch():
    skip_unless_sm100()
    tokens = 256
    rng = np.random.default_rng(0)
    kwargs_np = {
        "x": (rng.standard_normal((tokens, Q_LORA), dtype=np.float32) * 0.5).astype(ml_dtypes.bfloat16),
        "w": (rng.standard_normal((Q_OUT, Q_LORA), dtype=np.float32) * 0.02).astype(ml_dtypes.bfloat16),
        "cos": rng.standard_normal((tokens, QK_ROPE), dtype=np.float32).astype(ml_dtypes.bfloat16),
        "sin": rng.standard_normal((tokens, QK_ROPE), dtype=np.float32).astype(ml_dtypes.bfloat16),
    }
    dtypes_j = {"x": ml_dtypes.bfloat16, "w": ml_dtypes.bfloat16, "cos": ml_dtypes.bfloat16, "sin": ml_dtypes.bfloat16}
    dtypes_t = {"x": torch.bfloat16, "w": torch.bfloat16, "cos": torch.bfloat16, "sin": torch.bfloat16}
    run_both(kwargs_np, dtypes_j, dtypes_t)


@pytest.mark.L0
def test_gemm_proj_rope_mxfp8_mxfp8in_jax_matches_torch():
    skip_unless_sm100()
    tokens = 256
    rng = np.random.default_rng(1)
    kwargs_np = {
        "x": (rng.standard_normal((tokens, Q_LORA), dtype=np.float32) * 0.5).astype(ml_dtypes.float8_e4m3fn),
        "w": (rng.standard_normal((Q_OUT, Q_LORA), dtype=np.float32) * 0.02).astype(ml_dtypes.float8_e4m3fn),
        "cos": rng.standard_normal((tokens, QK_ROPE), dtype=np.float32).astype(ml_dtypes.bfloat16),
        "sin": rng.standard_normal((tokens, QK_ROPE), dtype=np.float32).astype(ml_dtypes.bfloat16),
        # E8M0 biased exponents around 1.0 (byte 127)
        "x_scale": rng.integers(125, 130, size=(tokens, Q_LORA // BLOCK)).astype(np.uint8),
        "w_scale": rng.integers(125, 130, size=(Q_OUT, Q_LORA // BLOCK)).astype(np.uint8),
    }
    dtypes_j = {
        "x": ml_dtypes.float8_e4m3fn,
        "w": ml_dtypes.float8_e4m3fn,
        "cos": ml_dtypes.bfloat16,
        "sin": ml_dtypes.bfloat16,
        "x_scale": np.uint8,
        "w_scale": np.uint8,
    }
    dtypes_t = {
        "x": torch.float8_e4m3fn,
        "w": torch.float8_e4m3fn,
        "cos": torch.bfloat16,
        "sin": torch.bfloat16,
        "x_scale": torch.uint8,
        "w_scale": torch.uint8,
    }
    run_both(kwargs_np, dtypes_j, dtypes_t)


@pytest.mark.L0
@pytest.mark.parametrize("input_path", ["bf16in", "mxfp8in"])
def test_gemm_proj_rope_mxfp8_jax_jit_matches_eager(input_path):
    """XLA custom-call entry point (cudnn.jax.call): bit-identical to the eager JAX wrapper."""
    import cutlass.jax

    if not cutlass.jax.is_available():
        pytest.skip("CuTeDSL JAX extensions unavailable (jax >= 0.5 required)")
    skip_unless_sm100()
    import cudnn
    from cudnn import gemm_proj_rope_mxfp8_jax_sm100

    tokens = 256
    rng = np.random.default_rng(4)
    cos = jnp.asarray(rng.standard_normal((tokens, QK_ROPE), dtype=np.float32).astype(ml_dtypes.bfloat16))
    sin = jnp.asarray(rng.standard_normal((tokens, QK_ROPE), dtype=np.float32).astype(ml_dtypes.bfloat16))
    if input_path == "bf16in":
        x = jnp.asarray((rng.standard_normal((tokens, Q_LORA), dtype=np.float32) * 0.5).astype(ml_dtypes.bfloat16))
        w = jnp.asarray((rng.standard_normal((Q_OUT, Q_LORA), dtype=np.float32) * 0.02).astype(ml_dtypes.bfloat16))
        scales = {}
    else:
        x = jnp.asarray((rng.standard_normal((tokens, Q_LORA), dtype=np.float32) * 0.5).astype(ml_dtypes.float8_e4m3fn))
        w = jnp.asarray((rng.standard_normal((Q_OUT, Q_LORA), dtype=np.float32) * 0.02).astype(ml_dtypes.float8_e4m3fn))
        scales = {
            "x_scale": jnp.asarray(rng.integers(125, 130, size=(tokens, Q_LORA // BLOCK)).astype(np.uint8)),
            "w_scale": jnp.asarray(rng.integers(125, 130, size=(Q_OUT, Q_LORA // BLOCK)).astype(np.uint8)),
        }
    jax.block_until_ready((x, w, cos, sin, *scales.values()))

    result_eager = cudnn.gemm_proj_rope_mxfp8_wrapper_sm100(x, w, cos, sin, **scales, w_out_in=True)
    device_sync()  # eager JAX path runs on the CUDA legacy default stream
    expected = [np.asarray(result_eager[key]).view(np.uint8) for key in ("out_fp8_row", "out_scales_row", "out_fp8_col", "out_scales_col")]

    def check(outputs):
        jax.block_until_ready(outputs)
        for got, want, key in zip(outputs, expected, ("out_fp8_row", "out_scales_row", "out_fp8_col", "out_scales_col")):
            np.testing.assert_array_equal(
                np.asarray(got).view(np.uint8),
                want,
                err_msg=f"proj_rope {key}: jit output differs from eager wrapper output on identical input bytes",
            )

    # Eager custom call
    check(gemm_proj_rope_mxfp8_jax_sm100(x, w, cos, sin, **scales))

    # Under jax.jit, twice (compiled-kernel / registration cache)
    jitted = jax.jit(lambda *args: gemm_proj_rope_mxfp8_jax_sm100(*args))
    check(jitted(x, w, cos, sin, *scales.values()))
    check(jitted(x, w, cos, sin, *scales.values()))


@pytest.mark.L0
def test_gemm_proj_rope_mxfp8_jax_errors():
    skip_unless_sm100()
    import cudnn

    tokens = 256
    rng = np.random.default_rng(2)
    x = jnp.asarray((rng.standard_normal((tokens, Q_LORA), dtype=np.float32) * 0.5).astype(ml_dtypes.bfloat16))
    w_in_out = jnp.asarray((rng.standard_normal((Q_LORA, Q_OUT), dtype=np.float32) * 0.02).astype(ml_dtypes.bfloat16))
    cos = jnp.asarray(rng.standard_normal((tokens, QK_ROPE), dtype=np.float32).astype(ml_dtypes.bfloat16))
    sin = jnp.asarray(rng.standard_normal((tokens, QK_ROPE), dtype=np.float32).astype(ml_dtypes.bfloat16))

    with pytest.raises(ValueError, match="not expressible as JAX arrays"):
        cudnn.gemm_proj_rope_mxfp8_wrapper_sm100(x, w_in_out, cos, sin, w_out_in=False)

    with pytest.raises(ValueError, match="Unsupported tensor framework"):
        cudnn.gemm_proj_rope_mxfp8_wrapper_sm100(np.asarray(x), np.asarray(w_in_out), np.asarray(cos), np.asarray(sin))
