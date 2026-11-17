# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real dispatcher registrations across legacy and mixed varlen schemas.

CUDA kernel registrations can be called with CPU tensors through call_boxed:
the tiny recorder bodies below need no GPU and isolate backend ownership from
kernel eligibility, architecture and numerical differences (#2639).
"""

import itertools

import pytest
import torch

from torch.nn.attention import SDPBackend

from cudnn.torch import sdpa_provider as provider

pytestmark = pytest.mark.L0
_IDS = itertools.count()
_OPS = ("_varlen_attn", "_varlen_attn_out", "_varlen_attn_backward")
_FLASH = SDPBackend.FLASH_ATTENTION.value
_CUDNN = SDPBackend.CUDNN_ATTENTION.value


@pytest.fixture
def dispatcher(monkeypatch):
    # Define Torch's real varlen ops before replacing only this test's namespace.
    pytest.importorskip("torch.nn.attention.varlen")
    factory = torch.library.Library
    namespace = f"fe_provider_dispatch_{next(_IDS)}"
    definitions = factory(namespace, "DEF")
    original = factory(namespace, "IMPL")
    events = []
    handles = []

    def configure(generation):
        for name in ("_scaled_dot_product_cudnn_attention", "_scaled_dot_product_cudnn_attention_backward"):
            definitions.define(f"{name}(Tensor query) -> Tensor")
        overrides = []
        for name in _OPS:
            backend = generation == "selected" or (generation == "mixed" and name != "_varlen_attn_out")
            tail = f", int backend={_FLASH}" if backend else ""
            definitions.define(f"{name}(Tensor query, bool supported=True{tail}) -> Tensor")

            def native(query, supported=True, backend=_FLASH, *, _name=name):
                events.append(("native", _name))
                return query + 10

            def serve(query, supported=True, *, _name=name):
                events.append(("provider", _name))
                return query + 1

            original.impl(name, native, "CUDA")
            overrides.append((name, serve, lambda call: call.get("supported", True), "fwd_torch"))

        ops = getattr(torch.ops, namespace)
        monkeypatch.setattr(torch.ops, "torch_attn", ops)
        monkeypatch.setattr(provider, "_VARLEN_OVERRIDES", tuple(overrides))
        monkeypatch.setattr(provider, "calls", dict.fromkeys(provider.calls, 0))
        monkeypatch.setattr(torch.library, "Library", lambda ns, *args, **kwargs: factory(namespace if ns in ("aten", "torch_attn") else ns, *args, **kwargs))
        handle = provider.install_sdpa_overrides()
        handles.append(handle)
        return ops, events, handle

    yield configure
    for handle in handles:
        handle.remove()
    original._destroy()
    definitions._destroy()


_CASES = [
    pytest.param(generation, op, selection, supported, id=f"{generation}-{op}-{selection}-supported{supported}")
    for generation in ("legacy", "mixed", "selected")
    for op in _OPS
    for selection in (("default", "flash", "cudnn") if generation == "selected" or (generation == "mixed" and op != "_varlen_attn_out") else ("default",))
    for supported in ((False, True) if selection == "cudnn" else (True,))
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("generation,name,selection,supported", _CASES)
def test_installed_varlen_backend_ownership(dispatcher, generation, name, selection, supported, dtype):
    ops, events, handle = dispatcher(generation)
    op = getattr(ops, name).default
    keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
    query = torch.ones(2, 3, dtype=dtype)
    kwargs = {} if selection == "default" else {"backend": _CUDNN if selection == "cudnn" else _FLASH}
    if not supported:
        kwargs["supported"] = False

    actual = torch.library.get_kernel(op, "CUDA").call_boxed(keyset, query, **kwargs)
    served = generation == "legacy" or (selection == "cudnn" and supported)
    assert events == [("provider" if served else "native", name)]
    torch.testing.assert_close(actual, query + (1 if served else 10), rtol=0, atol=0)

    # Removing one installation must restore the original body for all schemas.
    handle.remove()
    events.clear()
    actual = torch.library.get_kernel(op, "CUDA").call_boxed(keyset, query, **kwargs)
    assert events == [("native", name)]
    torch.testing.assert_close(actual, query + 10, rtol=0, atol=0)
