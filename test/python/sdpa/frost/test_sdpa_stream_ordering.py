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

pytestmark = requires_pre_rubin_blackwell

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

    assert not torch.isnan(out).any(), "SDPA kernel read the poisoned Q: launch was not ordered after the " "side-stream copy (env-stream routing broken)"
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
