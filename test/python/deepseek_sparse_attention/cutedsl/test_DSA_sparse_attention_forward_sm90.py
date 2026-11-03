# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM90 coverage for DSA sparse-prefill forward."""

import math

import pytest
import torch

from deepseek_sparse_attention.cutedsl.dsa_reference import ref_sparse_attention_forward

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


def _require_sm90():
    """Skip focused device tests unless the active CUDA device is SM90."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 GPU required")


@pytest.mark.L0
@pytest.mark.parametrize("num_heads", [16, 32, 64, 96, 128])
@pytest.mark.parametrize("head_dim", [512, 576])
def test_DSA_sparse_attention_forward_sm90_support_matrix(num_heads, head_dim):
    """Admit every documented SM90 head-count and QK-width combination."""
    _require_sm90()
    from cudnn import DSA

    q = torch.empty((1, num_heads, head_dim), dtype=torch.bfloat16, device="cuda")
    kv = torch.empty((1, head_dim), dtype=torch.bfloat16, device="cuda")
    indices = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    assert DSA.SparseAttentionForward(q, kv, indices).check_support()


@pytest.mark.L1
@pytest.mark.parametrize("num_heads", [16, 32, 64, 96, 128])
@pytest.mark.parametrize("head_dim", [512, 576])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_DSA_sparse_attention_forward_sm90_matches_reference(num_heads, head_dim, dtype):
    """Match the independent oracle across dtypes, heads, tails, and empty rows."""
    _require_sm90()
    from cudnn import DSA

    torch.manual_seed(9100 + num_heads + head_dim)
    total_s_q, total_s_kv, logical_topk = 3, 97, 131
    q = (torch.randn(total_s_q, num_heads, head_dim, device="cuda") / math.sqrt(head_dim)).to(dtype)
    kv = (torch.randn(total_s_kv, head_dim, device="cuda") / math.sqrt(head_dim)).to(dtype)
    indices = torch.randint(0, total_s_kv, (total_s_q, logical_topk), dtype=torch.int32, device="cuda")
    indices[:, 1] = indices[:, 0]
    indices[0, -2:] = torch.tensor([-1, total_s_kv], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([logical_topk, 65, 0], dtype=torch.int32, device="cuda")
    sinks = torch.linspace(-0.5, 0.5, num_heads, dtype=torch.float32, device="cuda")
    scale = 0.75 / math.sqrt(head_dim)

    result = DSA.sparse_attention_forward_wrapper(
        q,
        kv,
        indices,
        attn_sink=sinks,
        topk_length=lengths,
        softmax_scale=scale,
    )
    reference = ref_sparse_attention_forward(
        q,
        kv,
        sinks,
        indices,
        topk_length=lengths,
        softmax_scale=scale,
        return_full=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(result["out"].float(), reference[0].float(), atol=8e-3, rtol=3.01 / 128)
    torch.testing.assert_close(result["max_logits"], reference[1], atol=2e-5, rtol=2.01 / 65536)
    torch.testing.assert_close(result["lse"], reference[2], atol=2e-5, rtol=2.01 / 65536)
    assert result["lse_indexer"] is None


@pytest.mark.L1
def test_DSA_sparse_attention_forward_sm90_compile_is_symbolic_and_execute_does_not_jit(monkeypatch):
    """Reuse one symbolic artifact across runtime sequence and head extents."""
    _require_sm90()
    from cudnn import DSA
    from cudnn.deepseek_sparse_attention.sparse_attention_forward import _interface_sm90 as interface_sm90

    q = torch.randn(2, 64, 576, dtype=torch.bfloat16, device="cuda") * 0.05
    kv = torch.randn(89, 576, dtype=torch.bfloat16, device="cuda") * 0.05
    indices = torch.arange(67, dtype=torch.int32, device="cuda").expand(2, -1).contiguous()
    op = DSA.SparseAttentionForward(q, kv, indices)
    assert op.check_support()
    op.compile()
    compiled = op._compiled_kernel
    device_index = q.device.index if q.device.index is not None else torch.cuda.current_device()
    assert compiled is interface_sm90.compile_sparse_attention_forward_sm90(device_index, q.dtype, 576, False, False, 0)

    monkeypatch.setattr(interface_sm90.cute, "compile", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("execute retraced")))
    out = torch.empty((2, 64, 512), dtype=q.dtype, device="cuda")
    stats = torch.empty((2, 64), dtype=torch.float32, device="cuda")
    op.execute(q, kv, indices, out=out, max_logits=stats, lse=torch.empty_like(stats))

    q_short = q[:1]
    indices_short = indices[:1, :65].contiguous()
    op_short = DSA.SparseAttentionForward(q_short, kv[:73], indices_short)
    assert op_short.check_support()
    op_short.compile()
    assert op_short._compiled_kernel is compiled

    q_wide = torch.randn(1, 128, 576, dtype=q.dtype, device=q.device) * 0.05
    op_wide = DSA.SparseAttentionForward(q_wide, kv, indices[:1])
    assert op_wide.check_support()
    op_wide.compile()
    assert op_wide._compiled_kernel is compiled


@pytest.mark.L0
def test_DSA_sparse_attention_forward_sm90_gates_cutedsl_before_kernel_import(monkeypatch):
    """Report the CuTeDSL requirement before importing or tracing the kernel."""
    _require_sm90()
    from cudnn import DSA
    from cudnn.deepseek_sparse_attention.sparse_attention_forward import api as api_module

    error = "SM90 DSA forward requires nvidia-cutlass-dsl >= 4.8.0; found 4.7.0"
    monkeypatch.setattr(api_module, "sm90_dsa_cutedsl_requirement_error", lambda: error)
    q = torch.empty((1, 16, 512), dtype=torch.bfloat16, device="cuda")
    kv = torch.empty((1, 512), dtype=torch.bfloat16, device="cuda")
    indices = torch.empty((1, 0), dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match=r"SM90 DSA forward requires nvidia-cutlass-dsl >= 4\.8\.0; found 4\.7\.0"):
        DSA.sparse_attention_forward_wrapper(q, kv, indices)
    with pytest.raises(RuntimeError, match=r"SM90 DSA forward requires nvidia-cutlass-dsl >= 4\.8\.0; found 4\.7\.0"):
        DSA.SparseAttentionForward(q, kv, indices).check_support()

    from cudnn.deepseek_sparse_attention.sparse_attention_forward import _interface_sm90 as interface_sm90

    interface_sm90.compile_sparse_attention_forward_sm90.cache_clear()
    monkeypatch.setattr(
        interface_sm90,
        "sm90_dsa_cutedsl_requirement_error",
        lambda: error,
    )
    with pytest.raises(RuntimeError, match=r"SM90 DSA forward requires nvidia-cutlass-dsl >= 4\.8\.0; found 4\.7\.0"):
        interface_sm90.compile_sparse_attention_forward_sm90(0, torch.bfloat16, 512, False, False, 0)
    interface_sm90.compile_sparse_attention_forward_sm90.cache_clear()


@pytest.mark.L1
@pytest.mark.parametrize("head_dim", [512, 576])
def test_DSA_sparse_attention_forward_sm90_indexer_prefix_lse(head_dim):
    """Match prefix statistics and preserve the short-row source sentinel."""
    _require_sm90()
    from cudnn import DSA

    torch.manual_seed(9512 + head_dim)
    total_s_q, total_s_kv, num_heads = 4, 631, 32
    logical_topk, indexer_topk = 579, 512
    q = torch.randn(total_s_q, num_heads, head_dim, dtype=torch.bfloat16, device="cuda") * 0.05
    kv = torch.randn(total_s_kv, head_dim, dtype=torch.bfloat16, device="cuda") * 0.05
    indices = torch.randint(0, total_s_kv, (total_s_q, logical_topk), dtype=torch.int32, device="cuda")
    lengths = torch.tensor([logical_topk, 544, 300, 0], dtype=torch.int32, device="cuda")

    result = DSA.sparse_attention_forward_wrapper(
        q,
        kv,
        indices,
        topk_length=lengths,
        indexer_topk=indexer_topk,
    )
    reference = ref_sparse_attention_forward(
        q,
        kv,
        None,
        indices,
        topk_length=lengths,
        indexer_topk=indexer_topk,
        return_full=True,
    )
    torch.testing.assert_close(result["out"].float(), reference[0].float(), atol=8e-3, rtol=3.01 / 128)
    torch.testing.assert_close(result["max_logits"], reference[1], atol=2e-5, rtol=2.01 / 65536)
    torch.testing.assert_close(result["lse"], reference[2], atol=2e-5, rtol=2.01 / 65536)
    torch.testing.assert_close(result["lse_indexer"], reference[3], atol=2e-5, rtol=2.01 / 65536)
    assert torch.isposinf(result["lse_indexer"][2:]).all()


@pytest.mark.L1
@pytest.mark.parametrize("logical_topk,total_s_kv", [(0, 7), (17, 0)])
def test_DSA_sparse_attention_forward_sm90_empty_sparse_domain(logical_topk, total_s_kv):
    """Return defined zero/inf outputs when no sparse KV token is live."""
    _require_sm90()
    from cudnn import DSA

    q = torch.zeros(2, 16, 512, dtype=torch.bfloat16, device="cuda")
    kv = torch.empty(total_s_kv, 512, dtype=torch.bfloat16, device="cuda")
    indices = torch.zeros(2, logical_topk, dtype=torch.int32, device="cuda")
    op = DSA.SparseAttentionForward(q, kv, indices)
    assert op.check_support()
    op.compile()
    out = torch.full((2, 16, 512), float("nan"), dtype=q.dtype, device="cuda")
    max_logits = torch.full((2, 16), float("nan"), dtype=torch.float32, device="cuda")
    lse = torch.full_like(max_logits, float("nan"))

    op.execute(q, kv, indices, out=out, max_logits=max_logits, lse=lse)

    torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
    assert torch.isneginf(max_logits).all()
    assert torch.isposinf(lse).all()


@pytest.mark.L1
def test_DSA_sparse_attention_forward_sm90_tile_and_length_boundaries():
    """Cover negative, empty, partial, paired, and clamped top-k lengths."""
    _require_sm90()
    from cudnn import DSA

    torch.manual_seed(9640)
    lengths_host = [-7, 0, 1, 63, 64, 65, 128, 999]
    total_s_q, total_s_kv, num_heads, head_dim, logical_topk = len(lengths_host), 73, 16, 576, 129
    q = torch.randn(total_s_q, num_heads, head_dim, dtype=torch.bfloat16, device="cuda") * 0.04
    kv = torch.randn(total_s_kv, head_dim, dtype=torch.bfloat16, device="cuda") * 0.04
    # Row zero is deliberately nonfinite.  Slots outside topk_length point at
    # it, proving that a predicated zero-byte gather cannot contaminate the
    # QK/PV reductions with 0 * NaN at 64- and 128-slot boundaries.
    kv[0].fill_(float("nan"))
    indices = torch.randint(1, total_s_kv, (total_s_q, logical_topk), dtype=torch.int32, device="cuda")
    for row, length in enumerate(lengths_host):
        first_masked = max(0, min(logical_topk, length))
        indices[row, first_masked:] = 0
    indices[-1, -2:] = torch.tensor([-1, total_s_kv], dtype=torch.int32, device="cuda")
    lengths = torch.tensor(lengths_host, dtype=torch.int32, device="cuda")
    sink = torch.linspace(-0.25, 0.25, num_heads, dtype=torch.float32, device="cuda")

    result = DSA.sparse_attention_forward_wrapper(q, kv, indices, attn_sink=sink, topk_length=lengths)
    reference = ref_sparse_attention_forward(q, kv, sink, indices, topk_length=lengths, return_full=True)
    torch.testing.assert_close(result["out"].float(), reference[0].float(), atol=8e-3, rtol=3.01 / 128)
    torch.testing.assert_close(result["max_logits"], reference[1], atol=2e-5, rtol=2.01 / 65536)
    torch.testing.assert_close(result["lse"], reference[2], atol=2e-5, rtol=2.01 / 65536)
    assert not torch.isnan(result["out"]).any()
    assert not torch.isnan(result["max_logits"]).any()
    assert not torch.isnan(result["lse"]).any()


@pytest.mark.L1
def test_DSA_sparse_attention_forward_sm90_wrapper_normalizes_strided_inputs():
    """Normalize wrapper-only strided inputs without weakening prepared execution."""
    _require_sm90()
    from cudnn import DSA

    torch.manual_seed(9067)
    total_s_q, total_s_kv, num_heads, head_dim, logical_topk = 2, 83, 16, 512, 67
    q = (torch.randn(total_s_q, num_heads, head_dim * 2, dtype=torch.bfloat16, device="cuda") * 0.03)[..., ::2]
    kv = (torch.randn(total_s_kv, head_dim * 2, dtype=torch.bfloat16, device="cuda") * 0.03)[:, ::2]
    indices = torch.randint(0, total_s_kv, (total_s_q, logical_topk * 2), dtype=torch.int32, device="cuda")[:, ::2]
    sink = torch.linspace(-0.2, 0.2, num_heads * 2, dtype=torch.float32, device="cuda")[::2]
    lengths = torch.tensor([logical_topk, 0, logical_topk - 1, 0], dtype=torch.int32, device="cuda")[::2]
    assert not q.is_contiguous() and not kv.is_contiguous() and not indices.is_contiguous()

    result = DSA.sparse_attention_forward_wrapper(q, kv, indices, attn_sink=sink, topk_length=lengths)
    reference = ref_sparse_attention_forward(q, kv, sink, indices, topk_length=lengths, return_full=True)
    torch.testing.assert_close(result["out"].float(), reference[0].float(), atol=8e-3, rtol=3.01 / 128)
    torch.testing.assert_close(result["max_logits"], reference[1], atol=2e-5, rtol=2.01 / 65536)
    torch.testing.assert_close(result["lse"], reference[2], atol=2e-5, rtol=2.01 / 65536)


@pytest.mark.L1
def test_DSA_sparse_attention_forward_sm90_prepared_rejects_outputs_before_launch():
    """Reject malformed or aliased prepared bindings before the first launch."""
    _require_sm90()
    from cudnn import DSA

    q = torch.zeros(2, 16, 512, dtype=torch.bfloat16, device="cuda")
    kv = torch.zeros(7, 512, dtype=q.dtype, device="cuda")
    indices = torch.zeros(2, 1, dtype=torch.int32, device="cuda")
    op = DSA.SparseAttentionForward(q, kv, indices)
    assert op.check_support()
    op.compile()
    launched = False

    def forbidden_launch(*_args, **_kwargs):
        """Record any launch that incorrectly escapes pre-launch validation."""
        nonlocal launched
        launched = True
        raise AssertionError("invalid prepared output reached the kernel")

    op._compiled_kernel = forbidden_launch
    stats = torch.empty(2, 16, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="out must have shape"):
        op.execute(q, kv, indices, out=torch.empty(2, 16, 511, dtype=q.dtype, device="cuda"), max_logits=stats, lse=torch.empty_like(stats))
    assert not launched
    with pytest.raises(ValueError, match="out must not alias Q"):
        op.execute(q, kv, indices, out=q, max_logits=stats, lse=torch.empty_like(stats))
    assert not launched

    strided_q = torch.zeros(2, 16, 1024, dtype=q.dtype, device=q.device)[..., ::2]
    with pytest.raises(ValueError, match="Q must be contiguous"):
        op.execute(
            strided_q,
            kv,
            indices,
            out=torch.empty(2, 16, 512, dtype=q.dtype, device=q.device),
            max_logits=stats,
            lse=torch.empty_like(stats),
        )
    assert not launched

    misaligned_stats_storage = torch.empty(stats.numel() + 1, dtype=stats.dtype, device=stats.device)
    misaligned_stats = misaligned_stats_storage[1:].view_as(stats)
    with pytest.raises(ValueError, match="max_logits base pointer must be 16-byte aligned"):
        op.execute(
            q,
            kv,
            indices,
            out=torch.empty(2, 16, 512, dtype=q.dtype, device=q.device),
            max_logits=misaligned_stats,
            lse=torch.empty_like(stats),
        )
    assert not launched

    shared_stats = torch.empty(stats.numel() + 4, dtype=stats.dtype, device=stats.device)
    overlapping_max = shared_stats[: stats.numel()].view_as(stats)
    overlapping_lse = shared_stats[4:].view_as(stats)
    assert overlapping_max.data_ptr() % 16 == 0 and overlapping_lse.data_ptr() % 16 == 0
    with pytest.raises(ValueError, match="max_logits must not alias lse"):
        op.execute(
            q,
            kv,
            indices,
            out=torch.empty(2, 16, 512, dtype=q.dtype, device=q.device),
            max_logits=overlapping_max,
            lse=overlapping_lse,
        )
    assert not launched


@pytest.mark.L1
def test_DSA_sparse_attention_forward_sm90_prepared_execute_graph_replay():
    """Replay changed inputs through an allocation-free prepared CUDA Graph."""
    _require_sm90()
    from cudnn import DSA

    torch.manual_seed(9017)
    q = torch.randn(2, 96, 512, dtype=torch.bfloat16, device="cuda") * 0.05
    kv = torch.randn(73, 512, dtype=torch.bfloat16, device="cuda") * 0.05
    indices = torch.arange(65, dtype=torch.int32, device="cuda").expand(2, -1).contiguous()
    sink = torch.zeros(96, dtype=torch.float32, device="cuda")
    lengths = torch.tensor([65, 64], dtype=torch.int32, device="cuda")
    op = DSA.SparseAttentionForward(q, kv, indices, sample_attn_sink=sink, sample_topk_length=lengths)
    assert op.check_support()
    op.compile()
    out = torch.empty((2, 96, 512), dtype=q.dtype, device="cuda")
    max_logits = torch.empty((2, 96), dtype=torch.float32, device="cuda")
    lse = torch.empty_like(max_logits)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op.execute(q, kv, indices, attn_sink=sink, topk_length=lengths, out=out, max_logits=max_logits, lse=lse)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    try:
        previous_sync_debug_mode = torch.cuda.get_sync_debug_mode()
        with torch.cuda.graph(graph, stream=stream):
            torch.cuda.set_sync_debug_mode("error")
            try:
                op.execute(q, kv, indices, attn_sink=sink, topk_length=lengths, out=out, max_logits=max_logits, lse=lse)
            finally:
                torch.cuda.set_sync_debug_mode(previous_sync_debug_mode)

        kv.mul_(2)
        out.fill_(float("nan"))
        max_logits.fill_(float("nan"))
        lse.fill_(float("nan"))
        graph.replay()
        reference = ref_sparse_attention_forward(q, kv, sink, indices, topk_length=lengths, return_full=True)
        torch.testing.assert_close(out.float(), reference[0].float(), atol=8e-3, rtol=3.01 / 128)
        torch.testing.assert_close(max_logits, reference[1], atol=2e-5, rtol=2.01 / 65536)
        torch.testing.assert_close(lse, reference[2], atol=2e-5, rtol=2.01 / 65536)
    finally:
        graph.reset()
