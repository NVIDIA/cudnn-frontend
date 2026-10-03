# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed Stats head strides remain Int64 through the prepared host and stores."""

import ctypes

import pytest
import torch

from frost_test_utils import requires_dsl
from test_utils import torch_fork_set_rng

pytestmark = [requires_dsl]


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("precision", ["half", "fp8"])
@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256), (512, 512)])
@pytest.mark.parametrize("product", [False, True], ids=["wide-stride", "wide-product"])
@torch_fork_set_rng(seed=723)
def test_prepared_thd_stats_wide_head_stride(precision, d, dv, product):
    """Step physical head slabs and poison addresses reached by signed-32-bit math."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90, SdpaFwdDslSm100, SdpaFwdDslSm120

    cc = torch.cuda.get_device_capability()
    if cc not in ((9, 0), (10, 0), (10, 3), (10, 7), (12, 0), (12, 1)):
        pytest.skip("requires a prepared THD forward architecture")
    if cc == (9, 0) and precision != "half":
        pytest.skip("SM90 THD only serves half inputs")
    api_type = SdpaFwdDslSm90 if cc == (9, 0) else SdpaFwdDslSm120 if cc[0] == 12 else SdpaFwdDslSm100
    h, sq, skv = (5 if product else 2), 2, 3
    dtype = torch.bfloat16 if precision == "half" else torch.float8_e4m3fn
    q, k, v = ((torch.randn(1, seq, h, dim, device="cuda") * 0.25).to(dtype).transpose(1, 2) for seq, dim in ((sq, d), (skv, d), (skv, dv)))
    o = torch.empty((1, sq, h, dv), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    head_stride = sq + (2**30 if product else 2**32)
    origin = 2**31 if product else 0
    span = h * head_stride
    torch.cuda.empty_cache()
    if (origin + span) * 4 + 1024**3 > torch.cuda.mem_get_info()[0]:
        pytest.skip("physical Stats probe needs room for complete padded head slabs")
    try:
        storage = torch.empty(origin + span, device="cuda", dtype=torch.float32)
    except torch.OutOfMemoryError:
        pytest.skip("another allocation consumed the guarded Stats capacity")
    carrier = storage.narrow(0, origin, span).view(h, head_stride)
    stats = carrier[:, :sq].unsqueeze(0)
    guards = []
    for head in range(1, h):
        offset = head * head_stride
        narrowed = ctypes.c_int32(offset).value
        if offset != narrowed:
            guard = storage.narrow(0, origin + narrowed, sq)
            guard.fill_(float("nan"))
            guards.append(guard)
    api = api_type(q, k, v, o, sample_lse=stats, thd=True)
    api.compile()
    assert api._thd_spec is not None
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    lengths = dict(seq_q_lens=torch.tensor([sq], device="cuda", dtype=torch.int32), seq_kv_lens=torch.tensor([skv], device="cuda", dtype=torch.int32))

    def run():
        api.execute(q, k, v, o, lse_tensor=carrier, workspace=workspace, **lengths)

    def check():
        logits = (q.double() @ k.double().transpose(-1, -2)) * d**-0.5
        expected_o = (logits.softmax(-1) @ v.double()).to(o.dtype)
        torch.testing.assert_close(o, expected_o, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(stats, logits.logsumexp(-1).float(), atol=2e-2, rtol=2e-2)
        assert guards and all(torch.isnan(guard).all().item() for guard in guards)

    stats.fill_(float("nan"))
    run()
    check()
    captured = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(captured):
            run()
        q.copy_((q.float() * 0.5).to(dtype))
        stats.fill_(float("nan"))
        o.fill_(float("nan"))
        captured.replay()
        check()
    finally:
        captured.reset()
