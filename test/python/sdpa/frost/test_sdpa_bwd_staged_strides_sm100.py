# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physical wide-address inputs and outputs through the retained staging leg."""

import ctypes
from types import SimpleNamespace

import pytest
import torch

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
from sdpa.frost.test_sdpa_bwd_dsl_sm100 import _reference, _check_prepared

pytestmark = [pytest.mark.L1, pytest.mark.gpu_exclusive, requires_dsl, requires_pre_rubin_blackwell]


@pytest.mark.parametrize("role", ["q", "dq", "dv"])
@pytest.mark.parametrize("product", [False, True])
def test_staged_physical_batch_stride(role, product):
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm100

    torch.manual_seed(724)
    b, h, hk, s, d = (5 if product else 2), 4, 2, 128, 512
    tensors = {name: torch.randn(b, heads, s, d, dtype=torch.bfloat16, device="cuda") * 0.1 for name, heads in (("q", h), ("k", hk), ("v", hk), ("do", h))}
    o, stats, _, *expected = _reference(*(tensors[name] for name in ("q", "k", "v", "do")), group=h // hk)
    tensors.update(o=o.to(torch.bfloat16), stats=stats.unsqueeze(-1))
    for out, source in (("dq", "q"), ("dk", "k"), ("dv", "v")):
        tensors[out] = torch.empty_like(tensors[source]).fill_(float("nan"))
    source = tensors[role]
    # +3 forces the existing dense staging route. D remains contiguous.
    strides = ((2**30 if product else 2**32) + source.stride(0) + 3, *source.stride()[1:])
    origin = 2**31 if product else 0
    elements = origin + 1 + sum((n - 1) * st for n, st in zip(source.shape, strides))
    if elements * 2 + (1 << 30) > torch.cuda.mem_get_info()[0]:
        pytest.skip("physical staging probe needs one guarded wide allocation")
    owner = torch.empty(elements, device="cuda", dtype=source.dtype)
    decoys = []
    for index in range(1, b):
        wrapped = origin + ctypes.c_int32(index * strides[0]).value
        if wrapped == origin + index * strides[0]:
            continue  # This batch still fits Int32 and is a live row, not a guard.
        decoy = owner.as_strided((1, *source.shape[1:]), source.stride(), wrapped)
        decoy.fill_(float("nan"))
        decoys.append(decoy)
    tensors[role] = owner.as_strided(source.shape, strides, origin).copy_(source)
    api = SdpaBwdDslSm100(**{"sample_" + name: value for name, value in tensors.items()}, scale_softmax=d**-0.5)
    api.compile()
    assert api._prepared is None and api._staged_prepared is not None
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    args = {name + "_tensor": value for name, value in tensors.items()}
    case = SimpleNamespace(tensors=tensors, expected=expected)
    api.execute(**args, workspace=workspace)
    _check_prepared(case)
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            api.execute(**args, workspace=workspace)
        for name in ("dq", "dk", "dv"):
            tensors[name].fill_(float("nan"))
        workspace.fill_(0xBD)
        capture.replay()
        _check_prepared(case)
        assert all(torch.isnan(decoy).all() for decoy in decoys)
    finally:
        capture.reset()
