# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-suite conftest for the FROST DSL SDPA engines."""

import pytest


@pytest.fixture(autouse=True)
def _frost_opt_in(monkeypatch):
    """The FROST manifest rows are opt-in; this suite exercises them.

    Per test rather than at import, so the flag cannot leak into the rest of a
    full ``pytest test/python`` session and silently opt everything in."""
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")


@pytest.fixture(autouse=True)
def _sdpa_adapter_workspace(request, monkeypatch):
    """An ``SdpaFwdDsl`` adapter owns no workspace (R2): executed without one on
    a plan whose ``scratch_workspace_bytes()`` is non-zero it raises the contract
    error. This suite drives the adapters directly in ~80 places, so the HARNESS
    supplies the buffer here -- per test, through monkeypatch, so the adapters
    themselves stay strict. Opt out with ``@pytest.mark.no_workspace_shim`` to
    test the contract error itself."""
    if request.node.get_closest_marker("no_workspace_shim"):
        yield
        return
    try:
        from cudnn.sdpa.fwd import api_dsl
    except ImportError:  # GPU-free tests in this directory never reach the adapters
        yield
        return
    import functools

    import torch
    from cudnn._torch_stream import stream_context

    for cls in (api_dsl.SdpaFwdDslSm100, api_dsl.SdpaFwdDslSm120, api_dsl.SdpaFwdDslSm80):
        real = cls.execute

        @functools.wraps(real)
        def shim(self, *args, workspace=None, _real=real, **kwargs):
            if workspace is None and (n := self.scratch_workspace_bytes()):
                q = kwargs["q_tensor"] if "q_tensor" in kwargs else args[0]
                # Allocated on the launch stream (R1) so the allocator's reuse ordering covers the launch.
                with stream_context(kwargs.get("current_stream"), q.device):
                    workspace = torch.empty(n, dtype=torch.uint8, device=q.device)
            return _real(self, *args, workspace=workspace, **kwargs)

        monkeypatch.setattr(cls, "execute", shim)
    yield
