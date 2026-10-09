# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Compact GQA backward contract and numerical checks."""

import pytest
import torch
from itertools import accumulate

from cudnn.api_base import TensorDesc


def api_class():
    """Load the API when the GPU and CuTe DSL are supported."""
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("SM107 required")
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        pytest.skip("A CuTe DSL build with SM107 support is required")
    from cudnn.sdpa.bwd.compact_gqa import CompactGqaBackward

    return CompactGqaBackward


def desc(tokens, heads, dtype=torch.bfloat16):
    """Describe a contiguous THD tensor on the current CUDA device."""
    return TensorDesc(
        dtype=dtype, shape=(tokens, heads, 256), stride=(heads * 256, 256, 1), stride_order=(2, 1, 0), device=torch.device("cuda", torch.cuda.current_device())
    )


@pytest.mark.L0
def test_support_contract():
    """Check supported metadata and bounded workspace sizing."""
    cls = api_class()
    q, k = desc(65536, 8), desc(65536, 1)
    lse = TensorDesc(dtype=torch.float32, shape=(65536, 8, 1), stride=(8, 1, 1), stride_order=(2, 1, 0), device=q.device)
    plan = cls(q, k, k, q, q, lse, query_rows=32768)
    assert plan.check_support()
    assert 8 * 1024**3 < plan.scratch_workspace_bytes() < 9 * 1024**3
    assert plan._layout["ds"][1] <= 8 * 1024**3
    assert 8 * 1024**3 < plan.scratch_workspace_bytes(32768) < 9 * 1024**3
    with pytest.raises(NotImplementedError):
        cls(desc(1024, 8), desc(1024, 1), desc(1024, 1), q, q, lse).check_support()
    with pytest.raises(NotImplementedError):
        cls(desc(65536, 8, torch.float16), k, k, q, q, lse).check_support()


def fixture(return_call=False, lengths=(128, 384), spans=None, native_forward=True):
    """Build packed inputs and a matching native backward call."""
    te = pytest.importorskip("transformer_engine.pytorch.cpp_extensions.fused_attn")
    spans = lengths if spans is None else spans
    total, maximum = sum(spans), max(lengths)
    q = torch.randn((total, 8, 256), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((total, 1, 256), device="cuda", dtype=torch.bfloat16)
    v, do = torch.randn_like(k), torch.randn_like(q)
    if maximum == 0:
        assert not return_call
        inputs = (q, k, v, torch.zeros_like(q), do, torch.zeros((total, 8, 1), device="cuda"))
        return inputs, lambda: tuple(torch.zeros_like(t) for t in (q, k, v))
    cu = torch.tensor(list(accumulate(lengths, initial=0)), device="cuda", dtype=torch.int32)
    cup = torch.tensor(list(accumulate(spans, initial=0)), device="cuda", dtype=torch.int32)
    opts = dict(
        cu_seqlens_q_padded=cup,
        cu_seqlens_kv_padded=cup,
        attn_scale=1 / 16,
        dropout=0.0,
        qkv_layout="thd_thd_thd",
        o_format="thd",
        attn_mask_type="padding_causal",
    )
    backend = te.FusedAttnBackend["F16_arbitrary_seqlen"]
    if native_forward:
        o, aux = te.fused_attn_fwd(True, maximum, maximum, cu, cu, q, k, v, torch.bfloat16, backend, **opts)
    else:
        # Adapter-only tests need valid O/LSE, but do not exercise TE's native kernels.
        assert return_call
        o, lse = fp64_forward(q, k, v, lengths, spans)
        aux = [lse]
    args = (maximum, maximum, cu, cu, q, k, v, o, do, torch.bfloat16, aux, backend)
    kw = dict(opts, do_format="thd", dqkv_layout="thd_thd_thd")
    if return_call:
        return args, kw
    return (q, k, v, o, do, aux[0]), lambda: te.fused_attn_bwd(*args, **kw)


def check_gradients(actual, expected):
    """Check finite gradients and relative error against the reference."""
    errors = []
    for actual_tensor, reference in zip(actual, expected[:3]):
        a, b = actual_tensor.float(), reference.float()
        assert torch.isfinite(a).all()
        relative = float((a - b).norm() / b.norm().clamp_min(1e-20))
        near_zero = float(b.abs().max()) < 1e-5 and float((a - b).abs().max()) < 1e-6
        assert relative < 0.01 or near_zero
        assert float((a - b).abs().max()) < 0.03 * max(float(b.abs().max()), 1.0)
        errors.append(relative)
    return errors


@pytest.mark.L2
def test_full_packed_stream_pointers_and_workspace():
    """Check 64K gradients, stream ordering, and workspace ownership."""
    from functools import partial

    cls = api_class()
    torch.manual_seed(1909)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        inputs, native = fixture(lengths=(32768, 32768))
        plan = cls(*inputs, query_rows=32768)
        plan.compile()
        workspace = torch.empty(plan.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
        plan.initialize_workspace(workspace)
        execute = partial(plan.execute, sequence_offsets=(0, 32768, 65536))
        outputs = [torch.empty_like(t) for t in inputs[:3]]
        reference = native()
        execute(*inputs, *outputs, workspace)
    stream.synchronize()
    first = check_gradients(outputs, reference)
    snapshots = [t.clone() for t in outputs]
    with torch.cuda.stream(stream):
        inputs2, native2 = fixture(lengths=(32768, 32768))
        outputs2 = [torch.empty_like(t) for t in inputs2[:3]]
        reference2 = native2()
    # Launch explicitly while torch's current stream is different.
    old_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        before = torch.cuda.memory_allocated()
        execute(*inputs2, *outputs2, workspace, current_stream=stream.cuda_stream)
        assert torch.cuda.memory_allocated() == before
    finally:
        torch.cuda.set_sync_debug_mode(old_mode)
    stream.synchronize()
    second = check_gradients(outputs2, reference2)
    assert all(torch.equal(a, b) for a, b in zip(outputs, snapshots))
    replacement = torch.empty_like(workspace)
    assert replacement.data_ptr() != workspace.data_ptr()
    with pytest.raises(ValueError, match="Initialize"):
        execute(*inputs2, *outputs2, replacement, current_stream=stream.cuda_stream)
    plan.initialize_workspace(replacement, current_stream=stream.cuda_stream)
    execute(*inputs2, *outputs2, replacement, current_stream=stream.cuda_stream)
    stream.synchronize()
    check_gradients(outputs2, reference2)
    with pytest.raises(ValueError, match="overlap"):
        execute(*inputs2, inputs2[0], outputs2[1], outputs2[2], replacement, current_stream=stream.cuda_stream)
    with pytest.raises(ValueError, match="stream"):
        execute(*inputs2, *outputs2, replacement, current_stream=torch.cuda.current_stream().cuda_stream)
    plan.close()
    print({"relative_l2": [first, second], "scratch_bytes": plan.scratch_workspace_bytes()})


@pytest.mark.L2
def test_changing_packs_reuse_compiled_kernel():
    """Check changing packs and padding with one compiled plan."""
    cls = api_class()
    torch.manual_seed(2300)
    plan = None
    cases = [
        ((128, 384), None),
        ((1, 127, 129, 513), None),
        ((41, 173), (128, 256)),
        ((0, 1, 127, 4097), (128, 1, 256, 4224)),
        ((0, 0), (128, 256)),
        ((1,) * 129, None),
        ((4097, 8191), None),
        ((16384, 49152), None),
        ((65536,), None),
        ((32768, 32768), None),
        ((2049, 127, 511, 4097, 33), None),
    ]
    try:
        for lengths, spans in cases:
            inputs, native = fixture(lengths=lengths, spans=spans)
            if plan is None:
                plan = cls(*inputs, max_seqlen=65536, query_rows=32768)
                plan.compile()
                compiled = plan._compiled_kernel
                workspace = torch.empty(plan.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
                plan.initialize_workspace(workspace)
            outputs = [torch.empty_like(t) for t in inputs[:3]]
            reference = native()
            offsets = tuple(accumulate(lengths if spans is None else spans, initial=0))
            old = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                before = torch.cuda.memory_allocated()
                plan.execute(*inputs, *outputs, workspace, sequence_offsets=offsets, sequence_lengths=lengths)
                assert torch.cuda.memory_allocated() == before
                assert plan._compiled_kernel is compiled
            finally:
                torch.cuda.set_sync_debug_mode(old)
            torch.cuda.synchronize()
            for gradient, (output, expected) in enumerate(zip(outputs, reference[:3])):
                for start, length, span in zip(offsets, lengths, lengths if spans is None else spans):
                    if length == 1 and gradient < 2:
                        # Single-token dQ/dK are analytically zero.
                        for tensor in (output, expected):
                            assert float(tensor[start : start + length].abs().max()) < 1e-5
                    elif length:
                        check_gradients((output[start : start + length],), (expected[start : start + length],))
                    assert torch.count_nonzero(output[start + length : start + span]) == 0
    finally:
        if plan is not None:
            plan.close()


@pytest.mark.L2
def test_te_certification_and_fallback(monkeypatch):
    """Check certified packing, per-step dispatch, and native fallback."""
    api_class()
    from cudnn.sdpa.bwd.compact_gqa import te as adapter
    from transformer_engine.pytorch.attention.dot_product_attention import backends
    import inspect

    args, kw = fixture(return_call=True)
    original = backends.fused_attn_bwd
    monkeypatch.setattr(backends, "fused_attn_bwd", original)
    bound = inspect.signature(original).bind(*args, **kw)
    bound.apply_defaults()
    x = bound.arguments
    reference = original(*args, **kw)
    assert not adapter.eligible(x)
    cu = x["cu_seqlens_q"]
    adapter.register_packing(cu, (0, 128, 512))
    adapter.register_packing(x["cu_seqlens_q_padded"], (0, 128, 512))
    old = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        assert adapter.eligible(x)
    finally:
        torch.cuda.set_sync_debug_mode(old)
    altered = dict(x, dropout=0.1)
    assert not adapter.eligible(altered)
    altered = dict(x, deterministic=True)
    assert not adapter.eligible(altered)
    unregistered = cu.clone()
    assert not adapter.eligible(dict(x, cu_seqlens_q=unregistered))
    monkeypatch.setattr(adapter, "_step", 0)
    monkeypatch.setattr(adapter, "_native_only", False)
    for step in (1, 2, 3):
        adapter.begin_step()
        assert adapter._step == step
        result = backends.fused_attn_bwd(*args, **kw)
        check_gradients(result[:3], reference)
        assert adapter._counts == {"candidate": 1}
    cu.add_(0)
    assert not adapter.eligible(x)
    backends.fused_attn_bwd(*args, **kw)
    assert adapter._counts["native_fallback"] == 1
    adapter.set_enabled(False)
    assert adapter._workspace is None
    for plan in adapter._plans.values():
        plan.close()
    adapter._plans.clear()
    adapter._prefixes.clear()
    adapter._installed = False
    adapter._step = 0
    adapter._counts.clear()


@pytest.fixture
def isolated_adapter(monkeypatch):
    """Restore TE dispatch and adapter caches after each test."""
    api_class()
    from cudnn.sdpa.bwd.compact_gqa import te as adapter
    from transformer_engine.pytorch.attention.dot_product_attention import backends

    monkeypatch.setattr(backends, "fused_attn_bwd", backends.fused_attn_bwd)
    for name, value in {
        "_prefixes": {},
        "_plans": {},
        "_workspace": None,
        "_workspace_key": None,
        "_workspace_owner": None,
        "_workspace_capacity": 0,
        "_installed": False,
        "_enabled": False,
        "_step": 0,
        "_counts": {},
        "_packing_work": {},
        "_unmatched_packing_calls": 0,
        "_native_only": False,
        "_trace_packs": False,
    }.items():
        monkeypatch.setattr(adapter, name, value, raising=False)
    yield adapter, backends
    adapter.set_enabled(False)
    for plan in adapter._plans.values():
        plan.close()


@pytest.mark.L0
@pytest.mark.parametrize("capacity", [128, 129, 1024, 4097, 65536])
@pytest.mark.parametrize("extra_bytes", [-1, 0, 1])
def test_ds_budget_capacity_boundary(capacity, extra_bytes):
    """Reject budgets that cannot hold the final causal tile."""
    cls = api_class()
    q, k = desc(capacity, 8), desc(capacity, 1)
    lse = TensorDesc(dtype=torch.float32, shape=(capacity, 8), stride=(8, 1), stride_order=(1, 0), device=q.device)
    minimum = 8 * 128 * ((capacity + 127) // 128 * 128) * 2
    if extra_bytes < 0:
        with pytest.raises(ValueError, match="dS budget"):
            cls(q, k, k, q, q, lse, ds_budget_bytes=minimum + extra_bytes)
    else:
        plan = cls(q, k, k, q, q, lse, ds_budget_bytes=minimum + extra_bytes)
        start = (capacity + 127) // 128 * 128 - 128
        assert plan._band_geometry(capacity, plan._layout["ds"][1] // 2, start)[0] == 128


@pytest.mark.L0
def test_ds_budget_resize_is_atomic(monkeypatch):
    """Validate resized capacity before compilation or plan mutation."""
    cls = api_class()
    q, k = desc(128, 8), desc(128, 1)
    lse = TensorDesc(dtype=torch.float32, shape=(128, 8), stride=(8, 1), stride_order=(1, 0), device=q.device)
    plan = cls(q, k, k, q, q, lse, ds_budget_bytes=8 * 128 * 128 * 2)
    before = (plan._capacity, plan._layout, plan._scratch_bytes, plan._workspace_ref)
    with pytest.raises(ValueError, match="dS budget"):
        plan.scratch_workspace_bytes(129)

    def unexpected_compile(*args, **kwargs):
        """Fail if invalid capacity reaches compilation."""
        raise AssertionError("Invalid capacity reached compilation")

    monkeypatch.setattr(plan, "compile", unexpected_compile)
    with pytest.raises(ValueError, match="dS budget"):
        plan.initialize_workspace(None, max_seqlen=129)
    assert (plan._capacity, plan._layout, plan._scratch_bytes, plan._workspace_ref) == before


@pytest.mark.L0
def test_default_stream_context(monkeypatch):
    """Validate stream interop without compiling an architecture-specific kernel."""
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        pytest.skip("CompactGqaBackward requires CuTe DSL >=4.7.0")
    from cudnn.sdpa.bwd.compact_gqa import CompactGqaBackward as cls

    plan = cls.__new__(cls)
    plan.device, plan._stream = torch.device("cuda", torch.cuda.current_device()), None
    default = torch.cuda.default_stream(plan.device)

    def unexpected_external(*args, **kwargs):
        """Fail if a default stream reaches ExternalStream."""
        raise AssertionError("Default stream reached ExternalStream")

    monkeypatch.setattr(torch.cuda, "ExternalStream", unexpected_external)
    with torch.cuda.stream(torch.cuda.Stream()):
        for handle in (0, 1, default.cuda_stream):
            with plan._context(handle):
                assert torch.cuda.current_stream(plan.device) == default
        with pytest.raises(ValueError, match="cudaStreamPerThread"):
            with plan._context(2):
                raise AssertionError("unsupported per-thread context was entered")


@pytest.mark.L0
@pytest.mark.parametrize("trace", [False, True])
def test_te_older_signature_falls_back(isolated_adapter, monkeypatch, trace):
    """Unsupported TE signatures must preserve native dispatch."""
    adapter, backends = isolated_adapter
    calls = []

    def native(q):
        """Record native dispatch for the older signature."""
        calls.append(q)
        return "native"

    monkeypatch.setattr(backends, "fused_attn_bwd", native)
    monkeypatch.setattr(adapter, "_trace_packs", trace)
    adapter.install()
    adapter.set_enabled(True)
    q = torch.empty((1, 8, 256), device="cuda", dtype=torch.bfloat16)
    assert backends.fused_attn_bwd(q) == "native"
    assert len(calls) == 1
    assert adapter._counts == {"native_fallback": 1}
    assert adapter._unmatched_packing_calls == int(trace)


@pytest.mark.L0
def test_te_none_window_falls_back(isolated_adapter, monkeypatch):
    """A missing window tuple must not fail in eligibility checks."""
    import inspect

    adapter, backends = isolated_adapter
    torch.manual_seed(2300)
    args, kwargs = fixture(return_call=True, native_forward=False)
    signature = inspect.signature(backends.fused_attn_bwd)

    def native(*args, **kwargs):
        """Return the native fallback sentinel."""
        return "native"

    native.__signature__ = signature
    monkeypatch.setattr(backends, "fused_attn_bwd", native)
    adapter.install()
    adapter.set_enabled(True)
    assert backends.fused_attn_bwd(*args, **dict(kwargs, window_size=None)) == "native"
    assert adapter._counts == {"native_fallback": 1}


@pytest.mark.L0
def test_te_kernel_error_propagates(isolated_adapter, monkeypatch):
    """Do not convert candidate execution errors into native fallback."""
    import inspect
    from cudnn.sdpa.bwd.compact_gqa import api

    adapter, backends = isolated_adapter
    torch.manual_seed(2300)
    args, kwargs = fixture(return_call=True, native_forward=False)
    bound = inspect.signature(backends.fused_attn_bwd).bind(*args, **kwargs)
    bound.apply_defaults()
    for name in ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded"):
        adapter.register_packing(bound.arguments[name], (0, 128, 512))

    def failed_compile(*args, **kwargs):
        """Simulate a candidate compilation failure."""
        raise TypeError("candidate compilation failed")

    monkeypatch.setattr(api.CompactGqaBackward, "compile", failed_compile)
    adapter.install()
    adapter.set_enabled(True)
    with pytest.raises(TypeError, match="candidate compilation failed"):
        backends.fused_attn_bwd(*args, **kwargs)


def fp64_forward(q, k, v, lengths, spans):
    """Construct packed O/LSE independently of the native TE forward backend."""
    output = torch.zeros_like(q)
    lse = torch.zeros((*q.shape[:2], 1), device=q.device, dtype=torch.float32)
    start = 0
    for length, span in zip(lengths, spans):
        if length:
            qs, ks, vs = (t[start : start + length].double() for t in (q, k, v))
            scores = torch.einsum("qhd,khd->hqk", qs, ks.expand(-1, q.shape[1], -1)) / 16
            mask = torch.ones((length, length), device=q.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~mask, -float("inf"))
            output[start : start + length] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vs.expand(-1, q.shape[1], -1))
            lse[start : start + length, :, 0] = scores.logsumexp(-1).T
        start += span
    return output, lse


def fp64_gradients(inputs, lengths):
    """Compute independent causal attention gradients in FP64."""
    result = [torch.zeros_like(t, dtype=torch.float64) for t in inputs[:3]]
    start = 0
    for length in lengths:
        q, k, v = [t[start : start + length].double().detach().requires_grad_() for t in inputs[:3]]
        scores = torch.einsum("qhd,khd->hqk", q, k.expand(-1, 8, -1)) / 16
        mask = torch.ones((length, length), device=q.device, dtype=torch.bool).tril()
        p = scores.masked_fill(~mask, -float("inf")).softmax(-1)
        output = torch.einsum("hqk,khd->qhd", p, v.expand(-1, 8, -1))
        grads = torch.autograd.grad(output, (q, k, v), inputs[4][start : start + length].double())
        for target, grad in zip(result, grads):
            target[start : start + length].copy_(grad)
        start += length
    return result


@pytest.mark.L0
def test_te_workspace_plan_switch(isolated_adapter):
    """Reusing one scratch allocation across A-B-A plans must be safe."""
    import inspect

    adapter, backends = isolated_adapter
    torch.manual_seed(2300)
    signature = inspect.signature(backends.fused_attn_bwd)

    def call(lengths, lse_rank):
        """Build a certified TE call with the requested LSE rank."""
        args, kwargs = fixture(return_call=True, lengths=lengths, native_forward=False)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        x = bound.arguments
        aux = list(x["aux_ctx_tensors"])
        shape = (sum(lengths), 8) if lse_rank == 2 else (sum(lengths), 8, 1)
        aux[0] = aux[0].view(shape)
        x["aux_ctx_tensors"] = aux
        for name in ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded"):
            adapter.register_packing(x[name], tuple(accumulate(lengths, initial=0)))
        return bound

    large_a, small_a, large_b = call((1024,), 2), call((128, 384), 2), call((1024,), 3)
    x = small_a.arguments
    reference = fp64_gradients(tuple(x[n] for n in ("q", "k", "v", "o", "d_o")), (128, 384))
    adapter.install()
    adapter.set_enabled(True)
    backends.fused_attn_bwd(*large_a.args, **large_a.kwargs)
    first = backends.fused_attn_bwd(*small_a.args, **small_a.kwargs)
    check_gradients(first[:3], reference)
    workspace = adapter._workspace
    backends.fused_attn_bwd(*large_b.args, **large_b.kwargs)
    last = backends.fused_attn_bwd(*small_a.args, **small_a.kwargs)
    check_gradients(last[:3], reference)
    assert adapter._workspace is workspace
    assert len(adapter._plans) == 2
    assert adapter._counts == {"candidate": 4}
    for before, after in zip(first[:3], last[:3]):
        torch.testing.assert_close(after, before, rtol=0, atol=0)
