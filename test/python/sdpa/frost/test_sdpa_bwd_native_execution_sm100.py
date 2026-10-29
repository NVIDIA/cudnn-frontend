# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native large-head backward execution has no tensor copies or gather launches."""

import pytest
import torch

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
from sdpa.frost.test_sdpa_bwd_native_layouts_sm100 import _case, _check
from sdpa.frost.test_sdpa_bwd_dsl_sm100 import _check_prepared

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_execute_uses_current_pointers(dtype, monkeypatch):
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
def test_invalid_bindings_precede_native_launch(role, monkeypatch):
    from cudnn.sdpa.bwd import api_dsl, prepared_sm100

    case = _case(d=264, roles=("q", "dq"))
    api = api_dsl.SdpaBwdDslSm100(**{"sample_" + n: t for n, t in case.tensors.items()}, is_causal=True)
    api.compile()
    workspace = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")
    args = {n + "_tensor": t for n, t in case.tensors.items()}
    args[role + "_tensor"] = case.tensors[role].to(torch.float32 if role != "stats" else torch.bfloat16)
    from dataclasses import replace

    api._prepared = replace(api._prepared, fn=lambda *a: pytest.fail("invalid binding reached the kernel"), native_binding=True)
    with pytest.raises(ValueError, match=role):
        api.execute(**args, workspace=workspace)


@pytest.mark.parametrize("explicit_stream", [False, True])
def test_native_execution_preserves_other_current_device(explicit_stream):
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
            assert torch.cuda.current_device() == 1, "native execution leaked Q's device into the caller"
        launch.synchronize()
        _check(case)


@pytest.mark.parametrize("overlap", [False, True])
def test_interleaved_gradients_and_overlap_rejection(overlap):
    from dataclasses import replace
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm100

    case = _case(d=264, hkv=4, roles=("q", "dq", "dk", "dv"))
    b, h, _, d = case.tensors["dq"].shape
    capacity = max(case.tensors[role].shape[2] for role in ("dq", "dk", "dv"))
    owner = torch.full((b, capacity, h, 1 if overlap else 3, d + 24), 17, device="cuda", dtype=case.tensors["dq"].dtype)
    for i, role in enumerate(("dq", "dk", "dv")):
        seq = case.tensors[role].shape[2]
        case.tensors[role] = owner[:, :seq, :, 0 if overlap else i, 8 : d + 8].transpose(1, 2)
    api = SdpaBwdDslSm100(**{"sample_" + n: t for n, t in case.tensors.items()}, is_causal=True, scale_softmax=d**-0.5)
    api.compile()
    workspace = torch.full((api.scratch_workspace_bytes(),), 0xA7, dtype=torch.uint8, device="cuda")
    args = {n + "_tensor": t for n, t in case.tensors.items()}
    if overlap:
        api._prepared = replace(api._prepared, fn=lambda *a: pytest.fail("overlapping gradients reached the kernel"), native_binding=True)
        with pytest.raises(ValueError, match="overlaps"):
            api.execute(**args, workspace=workspace)
        assert torch.all(owner == 17) and torch.all(workspace == 0xA7)
    else:
        api.execute(**args, workspace=workspace)
        _check_prepared(case)
        assert torch.all(owner[..., :8] == 17)
        assert torch.all(owner[..., -16:] == 17)
