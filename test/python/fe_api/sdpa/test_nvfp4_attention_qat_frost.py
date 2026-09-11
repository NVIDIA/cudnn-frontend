# SPDX-License-Identifier: Apache-2.0

"""SM100 FROST QAT backend: numerical, workspace and launch contracts."""

import importlib.util
import math

import pytest
import torch

from test_utils import torch_fork_set_rng
from fe_api.sdpa.test_nvfp4_attention_qat_backward import _reference_case


def _available():
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

    installed, version = cutedsl_state()
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (10, 0)
        and installed
        and not cutedsl_too_old(version)
        and importlib.util.find_spec("triton") is not None
    )


pytestmark = [pytest.mark.L0, pytest.mark.skipif(not _available(), reason="Requires SM100, Triton, and CuTe DSL >= 4.7.0")]


def _prepare(inputs, **options):
    from cudnn import Nvfp4AttentionQatBackward

    options.setdefault("backend", "triton")
    op = Nvfp4AttentionQatBackward(*inputs[:6], softmax_scale=inputs[6], **options)
    op.check_support()
    op.compile()
    outputs = tuple(torch.empty_like(t) for t in inputs[:3])
    workspace = torch.empty(op.scratch_workspace_bytes(), dtype=torch.uint8, device=inputs[0].device)
    return op, outputs, workspace


def _close(actual, expected):
    for a, b in zip(actual, expected):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        torch.testing.assert_close(a, b, atol=0.005, rtol=0.005)
        assert (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30) < 0.01


def test_frost_backend_name():
    """FROST identifies the backend, not its implementation language."""
    from cudnn import Nvfp4AttentionQatBackward, nvfp4_attention_qat_backward

    q = torch.empty((1, 2, 256, 128), device="cuda", dtype=torch.bfloat16)
    lse = torch.empty((1, 2, 256), device="cuda", dtype=torch.float32)
    inputs = (q, q, q, q, q, lse)
    assert Nvfp4AttentionQatBackward(*inputs).backend == "auto"
    op = Nvfp4AttentionQatBackward(*inputs, backend="frost")
    assert op.backend == "frost" and op.check_support()
    # The old draft-only spelling is not a second public backend or alias.
    with pytest.raises(ValueError, match="backend must be 'auto', 'triton', or 'frost'"):
        Nvfp4AttentionQatBackward(*inputs, backend="cutedsl").check_support()
    with pytest.raises(ValueError, match="backend must be 'auto', 'triton', or 'frost'"):
        nvfp4_attention_qat_backward(*inputs, backend="cutedsl")


@pytest.mark.parametrize("sequence,chunk", [(256, 0), (512, 1)])
@pytest.mark.parametrize("backend", ["auto", "frost"])
@torch_fork_set_rng(seed=71)
def test_frost_matches_triton_and_reference(sequence, chunk, backend):
    inputs, expected = _reference_case(sequence, sequence, is_causal=False)
    reference, ref, ref_ws = _prepare(inputs)
    candidate, got, ws = _prepare(inputs, backend=backend, head_chunk=chunk)
    assert candidate.selected_backend == "frost" and reference.selected_backend == "triton"
    assert candidate._compiled_kernel.__class__.__module__ == "cudnn.sdpa.bwd.qat._frost"
    reference.execute(*inputs[:6], *ref, ref_ws)
    candidate.execute(*inputs[:6], *got, ws)
    _close(got, ref)
    for actual, golden in zip(got, expected):
        torch.testing.assert_close(actual.float(), golden, atol=0.03, rtol=0.03)
    h, s = inputs[0].shape[1:3]
    assert ws.numel() == 3 * h * s * 128 * 2 + h * s * 4 + (chunk or h) * s * s * 2


@torch_fork_set_rng(seed=73)
def test_frost_precompiled_no_allocations_no_sync_and_graph_replay(monkeypatch):
    import cutlass.cute as cute
    from cudnn.sdpa.bwd.qat import _nvfp4
    from torch.utils._python_dispatch import TorchDispatchMode

    inputs, _ = _reference_case(256, 256, is_causal=False)
    reference, ref, ref_ws = _prepare(inputs)
    candidate, got, ws = _prepare(inputs, backend="auto", head_chunk=1)
    reference.execute(*inputs[:6], *ref, ref_ws)

    def forbidden(*args, **kwargs):
        raise AssertionError("execute attempted compilation/JIT dispatch")

    class NoCopiesOrAllocations(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            assert func.overloadpacket not in {
                torch.ops.aten.empty,
                torch.ops.aten.empty_like,
                torch.ops.aten.empty_strided,
                torch.ops.aten.clone,
                torch.ops.aten._to_copy,
                torch.ops.aten.copy_,
            }, f"unexpected allocation or copy: {func}"
            return func(*args, **(kwargs or {}))

    def run():
        candidate.execute(*inputs[:6], *got, ws)

    # Exercise the detector itself before trusting the negative assertion.
    with pytest.raises(AssertionError, match="unexpected allocation"):
        with NoCopiesOrAllocations():
            torch.empty_like(inputs[0])
    with monkeypatch.context() as patch:
        patch.setattr(cute, "compile", forbidden)
        patch.setattr(_nvfp4.fake_quantize_q, "run", forbidden)
        patch.setattr(_nvfp4.fake_quantize_kv, "run", forbidden)
        old_sync_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            with NoCopiesOrAllocations():
                run()
        finally:
            torch.cuda.set_sync_debug_mode(old_sync_mode)
        _close(got, ref)
        for _ in range(2):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        inputs[4].normal_()
        ws.fill_(255)
        for output in got:
            output.fill_(float("nan"))
        graph.replay()
    reference.execute(*inputs[:6], *ref, ref_ws)
    _close(got, ref)


@torch_fork_set_rng(seed=79)
@pytest.mark.parametrize("backend", ["auto", "frost"])
def test_frost_explicit_stream_and_runtime_scale(monkeypatch, backend):
    import cuda.bindings.driver as cuda

    inputs, _ = _reference_case(256, 256, is_causal=False)
    reference, ref, ref_ws = _prepare(inputs)
    candidate, got, ws = _prepare(inputs, backend=backend)
    producer = torch.cuda.current_stream()
    launch = torch.cuda.Stream()
    launch.wait_stream(producer)
    # Both implementations must honor an execute-time scale different from
    # their default specialization; rebuild matching forward auxiliaries.
    from fe_api.sdpa.test_nvfp4_attention_qat_backward import _fake_quantize_nvfp4_reference

    q, k, v, o, do, lse = inputs[:6]
    scale = 0.0625
    with torch.cuda.stream(launch):
        fq, fk, fv = (_fake_quantize_nvfp4_reference(t).float() for t in (q, k, v))
        scores = fq @ fk.transpose(-1, -2) * scale
        lse.copy_(scores.logsumexp(-1))
        o.copy_(scores.softmax(-1) @ fv)
        do.normal_()
    bmm = torch.bmm
    observed_streams = []

    def checked_bmm(*args, **kwargs):
        observed_streams.append(torch.cuda.current_stream().cuda_stream)
        return bmm(*args, **kwargs)

    monkeypatch.setattr(torch, "bmm", checked_bmm)
    # Intentionally call outside the launch stream's torch context.
    candidate.execute(q, k, v, o, do, lse, *got, ws, softmax_scale=scale, current_stream=cuda.CUstream(launch.cuda_stream))
    with torch.cuda.stream(launch):
        reference.execute(q, k, v, o, do, lse, *ref, ref_ws, softmax_scale=scale)
    producer.wait_stream(launch)
    # dK is produced in-kernel; only the dQ GEMM goes through torch.bmm.
    assert observed_streams == [launch.cuda_stream]
    _close(got, ref)


@torch_fork_set_rng(seed=83)
def test_frost_declines_tails_causal_and_bad_chunk():
    from cudnn import Nvfp4AttentionQatBackward

    for sq, sk, options, match in (
        (257, 257, {}, "divisible by 256"),
        (256, 512, {}, "equal sequence"),
        (256, 256, {"is_causal": True}, "noncausal"),
        (256, 256, {"head_chunk": 3}, "divide the head count"),
    ):
        inputs, _ = _reference_case(sq, sk, is_causal=False)
        op = Nvfp4AttentionQatBackward(*inputs[:6], backend="frost", **options)
        with pytest.raises((ValueError, NotImplementedError), match=match):
            op.check_support()


@torch_fork_set_rng(seed=89)
def test_frost_wrapper_and_zero_inputs():
    from cudnn import nvfp4_attention_qat_backward

    q = torch.zeros((1, 2, 256, 128), device="cuda", dtype=torch.bfloat16)
    lse = torch.full((1, 2, 256), math.log(256), device="cuda", dtype=torch.float32)
    result = nvfp4_attention_qat_backward(q, q, q, q, q, lse, backend="frost", head_chunk=1)
    assert list(result.keys()) == ["dq_tensor", "dk_tensor", "dv_tensor"]
    for tensor in result:
        torch.testing.assert_close(tensor, torch.zeros_like(q), atol=0, rtol=0)


@torch_fork_set_rng(seed=97)
def test_frost_old_dsl_declines_before_kernel_import(monkeypatch):
    from cudnn import Nvfp4AttentionQatBackward
    from cudnn.frost import buffers

    inputs, _ = _reference_case(256, 256, is_causal=False)
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.6.2")))
    with pytest.raises(NotImplementedError, match=r"requires nvidia-cutlass-dsl >= 4.7.0; found 4.6.2"):
        Nvfp4AttentionQatBackward(*inputs[:6], backend="frost").check_support()


@torch_fork_set_rng(seed=101)
def test_frost_runtime_buffers_and_device_contract():
    inputs, _ = _reference_case(256, 256, is_causal=False)
    op, outputs, ws = _prepare(inputs, backend="frost")
    with pytest.raises(ValueError, match="at least"):
        op.execute(*inputs[:6], *outputs, ws[:-16])
    with pytest.raises(ValueError, match="16-byte aligned"):
        unaligned = torch.empty(ws.numel() + 1, device=ws.device, dtype=ws.dtype)[1:]
        op.execute(*inputs[:6], *outputs, unaligned)
    with pytest.raises(ValueError, match="must be on"):
        op.execute(inputs[0].cpu(), *inputs[1:6], *outputs, ws)
    # Explicit backward must not accidentally build a second autograd graph.
    inputs[0].requires_grad_(True)
    op.execute(*inputs[:6], *outputs, ws)
    assert all(not output.requires_grad for output in outputs)
