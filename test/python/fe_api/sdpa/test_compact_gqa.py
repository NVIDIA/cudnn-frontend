# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Compact GQA backward contract and numerical checks."""

import pytest
import torch
from itertools import accumulate

from cudnn.api_base import TensorDesc


def api_class():
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("SM107 required")
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        pytest.skip("A CuTe DSL build with SM107 support is required")
    from cudnn.sdpa.bwd.compact_gqa import CompactGqaBackward

    return CompactGqaBackward


def desc(tokens, heads, dtype=torch.bfloat16):
    return TensorDesc(
        dtype=dtype, shape=(tokens, heads, 256), stride=(heads * 256, 256, 1), stride_order=(2, 1, 0), device=torch.device("cuda", torch.cuda.current_device())
    )


@pytest.mark.L0
def test_support_contract():
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


def fixture(return_call=False, lengths=(128, 384), spans=None):
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
    o, aux = te.fused_attn_fwd(True, maximum, maximum, cu, cu, q, k, v, torch.bfloat16, backend, **opts)
    args = (maximum, maximum, cu, cu, q, k, v, o, do, torch.bfloat16, aux, backend)
    kw = dict(opts, do_format="thd", dqkv_layout="thd_thd_thd")
    if return_call:
        return args, kw
    return (q, k, v, o, do, aux[0]), lambda: te.fused_attn_bwd(*args, **kw)


def check_gradients(actual, expected):
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
