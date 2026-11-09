# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys

import cutlass
import pytest
import torch

from cudnn.ops import simplified_sparse_attention, simplified_sparse_attention_reserve
from ssa_reference import check_reference, pack, run, sample

pytestmark = pytest.mark.L0


@pytest.fixture
def supported_gpu():
    from cudnn.frost.buffers import cutedsl_requirement_error

    error = cutedsl_requirement_error("simplified_sparse_attention", floor=(4, 8, 0))
    if error:
        pytest.skip(error)
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Requires SM100 or SM103")


@pytest.mark.usefixtures("supported_gpu")
@pytest.mark.parametrize("heads", [16, 32, 96], ids=lambda n: f"bf16-h{n}")
@pytest.mark.parametrize("topk", [512, 1024, 2048], ids=lambda n: f"k{n}")
def test_public_forward_backward_reference(heads, topk):
    inputs = pack([sample(65, heads), sample(17, heads, seed=31)])
    before = [t.clone() for t in (*inputs[0], inputs[2])]
    result = run(inputs, backward=True, topk=topk)
    check_reference(inputs, result, backward=True, topk=topk)
    for actual, expected in zip((*inputs[0], inputs[2]), before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_lazy_imports():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import cudnn.ops
import cudnn.simplified_sparse_attention
assert "torch" not in sys.modules
assert "cutlass" not in sys.modules
assert "jax" not in sys.modules
assert "simplified_sparse_attention" in cudnn.ops.__all__
assert "simplified_sparse_attention_jax" in cudnn.ops.__all__
from cudnn.simplified_sparse_attention import (
    SimplifiedSparseAttentionMetadata,
    simplified_sparse_attention,
    simplified_sparse_attention_reserve,
    simplified_sparse_attention_jax,
)
assert SimplifiedSparseAttentionMetadata is cudnn.ops.SimplifiedSparseAttentionMetadata
assert simplified_sparse_attention is cudnn.ops.simplified_sparse_attention
assert simplified_sparse_attention_reserve is cudnn.ops.simplified_sparse_attention_reserve
assert simplified_sparse_attention_jax is cudnn.ops.simplified_sparse_attention_jax
""",
        ],
        check=True,
    )


@pytest.mark.parametrize("name", ["a_major", "b_major", "c_major"])
def test_dense_tensor_layout_rejected_before_allocation(monkeypatch, name):
    from cudnn.simplified_sparse_attention.cutedsl.kernels.dense_gemm_sm100 import prepare_tensors

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("Allocated tensors before checking their layouts")

    monkeypatch.setattr(torch, "empty", unexpected_allocation)
    layouts = dict(a_major="k", b_major="n", c_major="n")
    layouts[name] = "invalid"
    with pytest.raises(ValueError, match=name):
        prepare_tensors((16, 32, 64, 1), cutlass.BFloat16, cutlass.BFloat16, **layouts)


@pytest.mark.parametrize("unroll,threads,blocks,expected", [(0, 512, 4, 2), (0, 1024, 2, 2), (0, 256, 1, 4), (1, 512, 4, 1), (8, 512, 4, 8)])
@pytest.mark.parametrize("local_indices", [False, True])
def test_prefill_constructor_settings(unroll, threads, blocks, expected, local_indices):
    from cudnn.simplified_sparse_attention.cutedsl.kernels.cute_dsl_filter_topk.filtered_top_k_prefill_varlen import FilteredTopKKernelVarlenPrefill

    kernel = FilteredTopKKernelVarlenPrefill(
        cutlass.Float32,
        131072,
        512,
        unroll_factor=unroll,
        num_threads=threads,
        target_blocks_per_sm=blocks,
        subtract_row_start=local_indices,
    )
    assert kernel.unroll_factor == expected
    assert kernel.num_threads_per_cta == threads
    assert kernel.subtract_row_start_on_output is local_indices


def test_old_dsl_fails_before_kernel_import(monkeypatch):
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.7.0")))
    with pytest.raises(ImportError, match="found 4.7.0"):
        simplified_sparse_attention_reserve(128, 256, 512)


@pytest.mark.usefixtures("supported_gpu")
def test_capture_rejected_before_allocation():
    inputs, metadata, _ = pack([sample(8)])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    probe = torch.zeros(1, device="cuda")
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        with pytest.raises(NotImplementedError, match="external CUDA graph capture"):
            simplified_sparse_attention(*inputs, metadata, 512)
        with pytest.raises(NotImplementedError, match="external CUDA graph capture"):
            simplified_sparse_attention_reserve(128, 131072, 512)
        probe.add_(1)
    graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    assert probe.item() == 1


@pytest.mark.usefixtures("supported_gpu")
@pytest.mark.parametrize("change", ["dtype", "heads", "topk", "empty"])
def test_rejects_unsupported_inputs(change):
    inputs, metadata, _ = pack([sample(8)])
    topk = 512
    if change == "dtype":
        inputs[0] = inputs[0].float()
    elif change == "heads":
        inputs[0] = inputs[0][:, :8]
    elif change == "topk":
        topk = 32
    else:
        inputs[0] = inputs[0][:0]
    with pytest.raises(NotImplementedError):
        simplified_sparse_attention(*inputs, metadata, topk)


@pytest.mark.usefixtures("supported_gpu")
def test_selected_support_reference():
    tensors, upstream = sample(1025)
    tensors[0], tensors[3] = tensors[0][:65], tensors[3][:65]
    _, metadata, _ = pack([sample(65)])
    metadata.key_ends.fill_(1025)
    inputs = tensors, metadata, upstream[:65]
    actual = run(inputs, backward=True)
    check_reference(inputs, actual, backward=True)


def test_filtered_runtime_declines_before_allocation(monkeypatch):
    from cudnn.simplified_sparse_attention.cutedsl.kernels import indexer_forward_topk_fused_tcgen05 as fused

    monkeypatch.setitem(sys.modules, "torch._higher_order_ops.cudagraph_conditional_nodes", None)

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("Allocated filtered state before checking the runtime")

    monkeypatch.setattr(torch, "empty", unexpected_allocation)
    with pytest.raises(NotImplementedError, match="PyTorch CUDA conditional graph support"):
        fused._filtered_state(torch.device("cuda", 0), torch.bfloat16, 512, 128, 131072)
