# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared copies for existing large-head backward conversion layouts."""

import pytest
import torch

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
from sdpa.frost.test_sdpa_bwd_staged_sm100 import _case, _check
from sdpa.frost.test_sdpa_bwd_dsl_sm100 import _check_prepared

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_staged_execute_uses_prepared_copies(dtype, monkeypatch):
    import cutlass.cute as cute

    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    case = _case(dtype=dtype, d=264, roles=("q", "k", "v", "o", "do", "dq", "dk", "dv"))
    case.graph.execute(case.pack, case.workspace)
    _check(case)

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared SM100 backward rebuilt tensor staging at execute")

    with monkeypatch.context() as guards:
        for name in ("view", "reshape", "as_strided", "permute", "transpose", "copy_", "zero_"):
            guards.setattr(torch.Tensor, name, forbidden)
        for name in ("empty", "empty_like", "empty_strided", "zeros", "zeros_like"):
            guards.setattr(torch, name, forbidden)
        guards.setattr(cute, "compile", forbidden)
        for name in ("make_fake_tensor", "make_fake_compact_tensor", "make_ptr", "from_dlpack"):
            guards.setattr(cute.runtime, name, forbidden)
        torch.cuda.set_sync_debug_mode("error")
        try:
            case.graph.execute(case.pack, case.workspace)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    _check(case)


@pytest.mark.parametrize("role", ["q", "stats", "dq", "dv"])
def test_invalid_bindings_precede_prepared_gather(role, monkeypatch):
    from cudnn.sdpa.bwd import api_dsl, prepared_sm100

    case = _case(d=264, roles=("q", "dq"))
    api = api_dsl.SdpaBwdDslSm100(**{"sample_" + n: t for n, t in case.tensors.items()}, is_causal=True)
    api.compile()
    workspace = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")
    args = {n + "_tensor": t for n, t in case.tensors.items()}
    args[role + "_tensor"] = case.tensors[role].to(torch.float32 if role != "stats" else torch.bfloat16)
    monkeypatch.setattr(prepared_sm100, "_copy", lambda *a: pytest.fail("invalid binding reached the first prepared gather"))
    with pytest.raises(ValueError, match=role):
        api.execute(**args, workspace=workspace)


@pytest.mark.parametrize("overlap", [False, True])
def test_gradient_copy_back_retains_order(overlap, monkeypatch):
    from cudnn.sdpa.bwd import api_dsl, prepared_sm100

    case = _case(d=264, hkv=4, roles=("q", "dq", "dk", "dv"))
    b, h, _, d = case.tensors["dq"].shape
    capacity = max(case.tensors[role].shape[2] for role in ("dq", "dk", "dv"))
    owner = torch.full((b, capacity, h, 1 if overlap else 3, d + 3), 17, device="cuda", dtype=case.tensors["dq"].dtype)
    for i, role in enumerate(("dq", "dk", "dv")):
        seq = case.tensors[role].shape[2]
        case.tensors[role] = owner[:, :seq, :, 0 if overlap else i, 1 : d + 1].transpose(1, 2)
    api = api_dsl.SdpaBwdDslSm100(**{"sample_" + n: t for n, t in case.tensors.items()}, is_causal=True, scale_softmax=264**-0.5)
    api.compile()
    workspace = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")
    args = {n + "_tensor": t for n, t in case.tensors.items()}
    calls, original = [], prepared_sm100._copy

    def record(entry, frame, stream):
        calls.append(len(frame[0]))
        return original(entry, frame, stream)

    monkeypatch.setattr(prepared_sm100, "_copy", record)
    api.execute(**args, workspace=workspace)
    assert calls[-3:] == [1, 1, 1]
    if overlap:
        torch.testing.assert_close(case.tensors["dv"].float(), case.expected[2].float(), atol=0.003, rtol=0.03)
    else:
        _check_prepared(case)
        assert torch.all(owner[..., :1] == 17)
        assert torch.all(owner[..., -2:] == 17)


@pytest.mark.parametrize("explicit_stream", [False, True])
def test_staged_execution_preserves_other_current_device(explicit_stream):
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm100

    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    with torch.cuda.device(0):
        case = _case(d=264)
        api = SdpaBwdDslSm100(**{"sample_" + n: t for n, t in case.tensors.items()}, is_causal=True, scale_softmax=264**-0.5)
        api.compile()
        workspace = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")
        args = {n + "_tensor": t for n, t in case.tensors.items()}
        launch = torch.cuda.Stream()
        launch.wait_stream(torch.cuda.current_stream())
        with torch.cuda.device(1):
            api.execute(**args, workspace=workspace, current_stream=launch.cuda_stream if explicit_stream else None)
            assert torch.cuda.current_device() == 1, "staged execution leaked Q's device into the caller"
        launch.synchronize()
        _check(case)
