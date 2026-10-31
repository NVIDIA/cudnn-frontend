# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Route assertions shared by the native binder migration tests."""

import pytest


def forbid_python_forward_binding(monkeypatch):
    from cudnn.sdpa.fwd import prepared

    def forbidden(*args, **kwargs):
        pytest.fail("native template entered the Python core binder")

    for name in ("bind_dense", "bind_dense_split", "execute_quantized", "_bind_block_output", "_bind_mxfp8_scales"):
        monkeypatch.setattr(prepared, name, forbidden, raising=False)


def intercept_native_forward(monkeypatch, spec, inspect_frame):
    """Observe the actual native launch, including its current pointers/stream."""
    original = spec.fn

    def launch(*frame):
        inspect_frame(dict(zip(spec.order, frame, strict=True)))
        return original(*frame)

    monkeypatch.setattr(spec, "fn", launch)
    # Native binders retain the callable at construction.
    monkeypatch.setattr(spec, "native", type(spec.native)(spec))


def forbid_native_forward_launch(monkeypatch, spec):
    intercept_native_forward(monkeypatch, spec, lambda frame: pytest.fail("invalid binding reached attention"))


def assert_rejected_without_writes(run, outputs, workspace, *, match=None):
    """A binding error must precede even Amax/scratch initialization."""
    import torch

    tensors = [t for t in (*outputs, workspace) if t is not None and t.is_cuda]
    before = [t.contiguous().view(torch.uint8).clone() for t in tensors]
    with pytest.raises(ValueError, match=match):
        run()
    for tensor, saved in zip(tensors, before, strict=True):
        assert torch.equal(tensor.contiguous().view(torch.uint8), saved)


def compact_forward_kernel_count(api, buffers, execute):
    """Baseline algorithm launches for the same dense plan without layout padding.

    Quantized hosts may reset and unscale Amax in addition to attention. Compare
    actual captures so those required operations are not mistaken for layout copies.
    """
    from copy import copy
    from dataclasses import replace
    import torch
    from test_sdpa_bwd_staged_sm80 import _kernel_count

    compact = copy(api)
    compact._compiled_kernel = None
    compact._is_supported = False
    operands = dict(buffers)
    for role in ("q", "k", "v", "o"):
        tensor = buffers[role].permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
        operands[role] = tensor
        desc = getattr(api, role + "_desc")
        setattr(compact, role + "_desc", replace(desc, stride=tuple(tensor.stride()), stride_order=(3, 1, 2, 0)))
    assert compact.check_support()
    compact.compile()
    assert compact._dense_spec.native is not None and not hasattr(compact, "_staged_spec")
    workspace = torch.empty(compact.scratch_workspace_bytes(), dtype=torch.uint8, device=operands["q"].device)
    capture = torch.cuda.CUDAGraph(keep_graph=True)
    try:
        with torch.cuda.graph(capture):
            execute(compact, operands, workspace)
        return _kernel_count(capture)
    finally:
        capture.reset()
