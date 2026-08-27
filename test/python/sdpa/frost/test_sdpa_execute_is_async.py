# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rule 3 regression: a FROST SDPA ``execute()`` reads no device memory to the host.

A D2H read makes execute synchronous, so its cost becomes the depth of the queue
rather than its own work, and the path stops being CUDA-graph capturable. Two
angles here, because each catches what the other misses:

* the cause -- the KNOWN COMMON D2H and blocking entry points are made to raise
  for the duration of one execute, so a regression names the offender rather
  than surfacing later as a capture failure. Needs no capture support. This is a
  blacklist and cannot be exhaustive: .to(some_cpu_tensor), a CPU-target copy_,
  and driver-level synchronization all get through. It catches the shapes that
  actually keep appearing, and the capture test below is the backstop.
* the symptom, on the path that has no other coverage -- graph.execute() with NO
  handle leaves ExecutionContext.stream None and the adapter resolves the stream
  itself. test_sdpa_stream_respect.py captures the handle-carrying path; this
  file captures the one where the fallback picks the stream.

Scoped to the dense f16 rows, which are clean. The THD and per-tensor FP8 rows
still read device memory back; each needs a kernel-side change and is listed
under Rule 3 in python/cudnn/AGENTS.md. Widen the parametrization as they land.
"""

from __future__ import annotations

import pytest
import torch

import cudnn
from cudnn.engines import is_python_engine
from frost_test_utils import requires_pre_rubin_blackwell, requires_dsl

pytestmark = [pytest.mark.L0]

_B, _H, _S = 2, 8, 256
_HALF, _F32 = cudnn.data_type.HALF, cudnn.data_type.FLOAT

# The Tensor methods that keep showing up. `to` is here because
# .to("cpu") is a D2H read wearing a dtype-cast's clothes; the guard lets a
# device-to-device .to() through and only trips on a host target.
_D2H_METHODS = ("item", "tolist", "cpu", "numpy", "__float__", "__int__", "__bool__", "__index__", "to")
# Module-level entry points that read or block. torch.is_nonzero reaches the
# value without going through Tensor.__bool__, and a synchronize does not read
# anything but still makes execute() synchronous, which is what Rule 3 is about.
_D2H_FUNCTIONS = (
    (torch, "is_nonzero"),
    (torch, "equal"),
    (torch.cuda, "synchronize"),
    (torch.cuda.Stream, "synchronize"),
    (torch.cuda.Event, "synchronize"),
)


def _targets_host(args, kwargs) -> bool:
    """True when a Tensor.to(...) call names a non-CUDA destination."""
    for candidate in (*args, kwargs.get("device")):
        if isinstance(candidate, str) and not candidate.startswith("cuda"):
            return True
        if isinstance(candidate, torch.device) and candidate.type != "cuda":
            return True
    return False


def _build(d):
    dims = (_B, _H, _S, d)
    strides = (_S * _H * d, d, _H * d, 1)
    g = cudnn.pygraph(io_data_type=_HALF, intermediate_data_type=_F32, compute_data_type=_F32)
    q = g.tensor(dim=dims, stride=strides, data_type=_HALF, name="q")
    k = g.tensor(dim=dims, stride=strides, data_type=_HALF, name="k")
    v = g.tensor(dim=dims, stride=strides, data_type=_HALF, name="v")
    o, _ = g.sdpa(name="sdpa", q=q, k=k, v=v, attn_scale=1.0 / (d**0.5), is_inference=True, use_causal_mask=True)
    o.set_output(True).set_dim(dims).set_stride(strides)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    python = [i for i, p in enumerate(g.plans) if is_python_engine(p.engine_id)]
    if not python:
        pytest.skip(f"no FROST SDPA engine claimed the graph for d={d}")
    g.select_plan(python[0])
    g.check_support()
    g.build_plans()
    mk = lambda: torch.randn(_B, _S, _H, d, device="cuda", dtype=torch.float16).transpose(1, 2)  # noqa: E731
    o_buf = torch.empty(_B, _S, _H, d, device="cuda", dtype=torch.float16).transpose(1, 2)
    vp = {q: mk(), k: mk(), v: mk(), o: o_buf}
    ws = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8) if g.get_workspace_size() else None
    return g, vp, ws, o_buf


@requires_pre_rubin_blackwell
@requires_dsl
@pytest.mark.parametrize("d", [256, 512])
def test_execute_reads_no_device_memory_to_the_host(monkeypatch, d):
    g, vp, ws, _out = _build(d)
    g.execute(vp, ws)  # compile/warm outside the ban: JIT may legitimately sync
    torch.cuda.synchronize()

    caught = []

    def _forbid_method(name, original):
        def guard(self, *a, **kw):
            if getattr(self, "is_cuda", False) and (name != "to" or _targets_host(a, kw)):
                caught.append(f"Tensor.{name}")
                raise AssertionError(f"execute() called Tensor.{name}() on a CUDA tensor — see Rule 3 in python/cudnn/AGENTS.md")
            return original(self, *a, **kw)

        return guard

    def _forbid_function(label, original):
        def guard(*a, **kw):
            caught.append(label)
            raise AssertionError(f"execute() called {label}() — see Rule 3 in python/cudnn/AGENTS.md")

        return guard

    for name in _D2H_METHODS:
        monkeypatch.setattr(torch.Tensor, name, _forbid_method(name, getattr(torch.Tensor, name)))
    for owner, name in _D2H_FUNCTIONS:
        label = f"{getattr(owner, '__name__', owner.__class__.__name__)}.{name}"
        monkeypatch.setattr(owner, name, _forbid_function(label, getattr(owner, name)))

    g.execute(vp, ws)
    monkeypatch.undo()  # the sync below is the test's own, not execute()'s
    torch.cuda.synchronize()
    assert not caught


@requires_pre_rubin_blackwell
@requires_dsl
def test_execute_without_a_handle_is_cuda_graph_capturable():
    """The no-handle path, which is the one that resolves the stream itself.

    graph.execute() with no handle leaves ExecutionContext.stream None, so the
    adapter falls back to torch's current stream -- the branch this PR rewrote.
    Capture is the check that the fallback lands on the CAPTURING stream: a
    hardcoded stream 0 records an empty graph, and a replay into a zeroed
    output then reproduces nothing.
    """
    g, vp, ws, out = _build(256)
    g.execute(vp, ws)
    torch.cuda.synchronize()
    expected = out.clone()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        g.execute(vp, ws)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    captured = torch.cuda.CUDAGraph()
    with torch.cuda.graph(captured):
        g.execute(vp, ws)
    out.zero_()
    torch.cuda.synchronize()
    captured.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expected), "replay did not reproduce the eager result — the launch did not land on the capturing stream"
