# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Stream-ordering tests for the FROST SM100 DSL SDPA-forward engines.

The DSL kernels are compiled with TVM-FFI. A kernel compiled without a
stream parameter launches on the CUDA default stream regardless of
``torch.cuda.current_stream()``, so under ``with torch.cuda.stream(s):`` the
launch races the torch-side ops (input copies, ``zero_()`` resets) that were
enqueued on ``s``. The kernels therefore take an env-stream parameter
(``make_fake_stream(use_tvm_ffi_env_stream=True)``): TVM-FFI syncs it to
torch's current stream before every call.

These tests enqueue a long spin kernel on a side stream, mutate the inputs
*behind* it, and immediately run SDPA on the same stream. If the kernel
ignores the side stream it reads the pre-mutation garbage and the output is
wrong; correct env-stream routing makes it wait for the mutation.
"""

import math

import pytest
import torch

from test_utils import torch_fork_set_rng
from frost_test_utils import requires_pre_rubin_blackwell, requires_dsl

pytestmark = [requires_pre_rubin_blackwell, requires_dsl]

# ~0.5-1 s of spin: long enough that an unordered kernel launch on the
# default stream reliably overtakes the side-stream mutation.
_SLEEP_CYCLES = 1_000_000_000


def _ref_sdpa(q, k, v, *, scale):
    q_ref, k_ref, v_ref = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
    scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v_ref).to(q.dtype)


@pytest.mark.L0
def test_get_default_stream_follows_torch_current_stream():
    """APIBase._get_default_stream(None) must resolve to torch's *current*
    stream (not legacy stream 0), so wrapper default paths stay ordered."""
    from types import SimpleNamespace
    import logging

    from cudnn.api_base import APIBase

    dummy = SimpleNamespace(_logger=logging.getLogger("test"))
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        resolved = APIBase._get_default_stream(dummy, None)
        assert int(resolved) == s.cuda_stream
    resolved_default = APIBase._get_default_stream(dummy, None)
    assert int(resolved_default) == torch.cuda.current_stream().cuda_stream


@pytest.mark.L0
@pytest.mark.parametrize("d", [512, 256, 128], ids=["d512", "d256", "d128"])
@torch_fork_set_rng(seed=0)
def test_sdpa_fwd_dsl_side_stream_ordering(d):
    """SDPA on a non-default stream must observe torch ops enqueued before it
    on that stream (env-stream routing of the TVM-FFI kernel launch)."""
    try:
        import cutlass  # noqa: F401
        from cudnn.sdpa.fwd.api_dsl import sdpa_fwd_wrapper_dsl_sm100 as sdpa_fwd_wrapper_dsl
    except ImportError as e:
        pytest.skip(f"DSL dependencies unavailable: {e}")

    dev = torch.device("cuda")
    dtype = torch.bfloat16
    b, h, s_q, s_kv = 1, 4, 256, 256
    scale = 1.0 / math.sqrt(d)

    q_real = torch.randn(b, h, s_q, d, dtype=dtype, device=dev)
    k = torch.randn(b, h, s_kv, d, dtype=dtype, device=dev)
    v = torch.randn(b, h, s_kv, d, dtype=dtype, device=dev)
    ref = _ref_sdpa(q_real, k, v, scale=scale)

    # Warmup on the default stream: triggers compile + JIT and pins the
    # object cache, so the timed trial below measures only stream ordering.
    q = q_real.clone()
    try:
        out_warm = sdpa_fwd_wrapper_dsl(q, k, v, scale_softmax=scale)["o_tensor"]
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"DSL SDPA unsupported for d={d} on this build: {e}")
    torch.cuda.synchronize()
    torch.testing.assert_close(out_warm, ref, atol=2e-2, rtol=2e-2)

    # Race trial: poison Q, then on a side stream enqueue [sleep, restore Q,
    # SDPA]. An unordered (default-stream) kernel launch overtakes the
    # restore and reads the poisoned Q.
    q.fill_(float("nan"))
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        torch.cuda._sleep(_SLEEP_CYCLES)
        q.copy_(q_real)
        out = sdpa_fwd_wrapper_dsl(q, k, v, scale_softmax=scale)["o_tensor"]
    torch.cuda.synchronize()

    assert not torch.isnan(out).any(), "SDPA kernel read the poisoned Q: launch was not ordered after the " "side-stream copy (env-stream routing failed)"
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.L0
@pytest.mark.parametrize("mxfp8", [False, True], ids=["fp8", "mxfp8"])
@torch_fork_set_rng(seed=0)
def test_fp8_dense_postprocessing_follows_explicit_stream(mxfp8):
    """Dense output copy-back and amax postprocessing use the launch stream."""
    from cuda.bindings import driver as cuda_driver

    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    b, h, s, d = 1, 4, 128, 128
    dev = torch.device("cuda")

    if mxfp8:
        from sdpa.mxfp8_quant import quantize_to_mxfp8

        def quantize(tensor, *, columnwise=False):
            row, _, row_sf, col, _, col_sf = quantize_to_mxfp8(tensor, b, h, s, d, fp8_dtype=torch.float8_e4m3fn)
            return (col, col_sf) if columnwise else (row, row_sf)

        q, sf_q = quantize(torch.randn(b, h, s, d, device=dev) * 0.5)
        k, sf_k = quantize(torch.randn(b, h, s, d, device=dev) * 0.5)
        v, sf_v = quantize(torch.randn(b, h, s, d, device=dev) * 0.5, columnwise=True)
        execute_args = dict(sf_q=sf_q, sf_k=sf_k, sf_v=sf_v)
        api_args = {}
    else:

        def make_fp8():
            return (torch.randn(b, h, s, d, device=dev) * 0.5).to(torch.float8_e4m3fn)

        q, k, v = make_fp8(), make_fp8(), make_fp8()
        one = torch.ones(1, dtype=torch.float32, device=dev)
        scale_o = torch.full((1,), 2.0, dtype=torch.float32, device=dev)
        execute_args = dict(descale_q=one, descale_k=one, descale_v=one, scale_o=scale_o)
        api_args = dict(pertensor_fp8=True)

    # Preserve D-contiguity while forcing the adapter's O scratch/copy-back path.
    output_storage = torch.empty(b, h, s + 1, d, dtype=torch.float16, device=dev)
    output = output_storage[:, :, :s, :]
    assert not output.transpose(1, 2).is_contiguous()
    amax = torch.empty(1, dtype=torch.float32, device=dev)

    api = SdpaFwdDslSm100(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=output,
        dtype_o=torch.float16,
        scale_softmax=1.0 / math.sqrt(d),
        **api_args,
    )
    assert api.check_support()
    api.compile()

    api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=output, amax_o=amax, **execute_args)
    torch.cuda.synchronize()
    expected_output, expected_amax = output.clone(), amax.clone()

    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        torch.cuda._sleep(_SLEEP_CYCLES)
        output.fill_(float("nan"))
        amax.fill_(float("nan"))
    api.execute(
        q_tensor=q,
        k_tensor=k,
        v_tensor=v,
        o_tensor=output,
        amax_o=amax,
        current_stream=cuda_driver.CUstream(side.cuda_stream),
        **execute_args,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected_output, atol=0, rtol=0)
    torch.testing.assert_close(amax, expected_amax, atol=0, rtol=0)
