# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Correctness tests for MoE Grouped Matmul.

Uses Mixtral/DeepSeek-inspired shapes.
"""

import pytest
import torch
import cudnn


def get_cc():
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def ref_moe_grouped_matmul(token, weight, first_token_offset, num_experts):
    """Reference: loop over experts, matmul with column-major weight."""
    _, M, K = token.shape
    _, _, N = weight.shape
    output = torch.zeros(1, M, N, dtype=token.dtype, device=token.device)
    for b_e in range(num_experts):
        start = first_token_offset[b_e, 0, 0].item()
        end = first_token_offset[b_e + 1, 0, 0].item() if b_e + 1 < first_token_offset.shape[0] else M
        if start < end:
            output[0, start:end] = (token[0, start:end].float() @ weight[b_e].float()).to(token.dtype)
    return output


class TestMoEGroupedMatmul:

    @pytest.mark.L0
    @pytest.mark.skipif(cudnn.backend_version() < 91800, reason="moe_grouped_matmul requires cuDNN >= 9.18.0")
    @pytest.mark.parametrize(
        "num_experts,tokens,K,N",
        [
            (4, 512, 256, 256),  # small
            (8, 1024, 512, 512),  # Mixtral-like
            (4, 256, 256, 256),  # minimal
        ],
        ids=["4exp_512tok", "8exp_1024tok", "4exp_256tok"],
    )
    @pytest.mark.parametrize("dtype", [torch.float16], ids=["fp16"])
    def test_moe_forward(self, cudnn_handle, num_experts, tokens, K, N, dtype):
        torch.manual_seed(42)

        tpe = tokens // num_experts
        actual_tokens = tpe * num_experts

        token = torch.randn(1, actual_tokens, K, dtype=dtype, device="cuda")
        weight_raw = torch.randn(num_experts, N, K, dtype=dtype, device="cuda")
        weight = weight_raw.transpose(1, 2)  # col-major inner
        offset_count = num_experts + int(cudnn.backend_version() >= 92800)
        fto_vals = torch.arange(offset_count, dtype=torch.int32, device="cuda") * tpe
        fto = fto_vals.reshape(-1, 1, 1)

        stream = torch.cuda.current_stream().cuda_stream
        cudnn.set_stream(handle=cudnn_handle, stream=stream)

        from cudnn.gemm import moe_grouped_matmul

        result = moe_grouped_matmul(token, weight, fto, mode="none", top_k=1)

        Y_ref = ref_moe_grouped_matmul(token, weight, fto, num_experts)

        torch.testing.assert_close(Y_ref, result, atol=0.125, rtol=0.125)


def _fresh_handle_cache(monkeypatch, op):
    if hasattr(op, "_handles"):
        monkeypatch.setattr(op, "_handles", type(op._handles)())
    monkeypatch.setattr(op, "_tls", __import__("threading").local(), raising=False)
    monkeypatch.setattr(op, "_cudnn_handles", {}, raising=False)


@pytest.mark.L0
def test_handle_cache_is_thread_local_and_pinned_to_the_token_device(monkeypatch):
    """Each host thread gets its own handle, created on the token's GPU and bound to that thread's stream."""
    import contextlib
    import importlib
    import threading
    from types import SimpleNamespace

    op = importlib.import_module("cudnn.gemm.ops.moe_grouped_matmul")

    _fresh_handle_cache(monkeypatch, op)
    local = threading.local()
    active = threading.local()
    lock = threading.Lock()
    created, bindings, results = [], [], []
    barrier = threading.Barrier(2)

    @contextlib.contextmanager
    def device(index):
        active.index = index
        try:
            yield
        finally:
            active.index = None

    def create_handle():
        with lock:
            handle = (len(created) + 1, getattr(active, "index", None))
            created.append(handle)
            return handle

    def set_stream(*, handle, stream):
        with lock:
            bindings.append((handle, stream))

    monkeypatch.setattr(op.torch.cuda, "device", device)
    monkeypatch.setattr(op.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(op.torch.cuda, "current_stream", lambda unused=None: local.stream)
    monkeypatch.setattr(op.cudnn, "create_handle", create_handle)
    monkeypatch.setattr(op.cudnn, "set_stream", set_stream)

    def worker(stream):
        local.stream = SimpleNamespace(cuda_stream=stream)
        barrier.wait()
        first = op._get_handle(torch.device("cuda", 1))
        second = op._get_handle(torch.device("cuda", 1))
        results.append((stream, first, second))
        barrier.wait()  # both threads are alive while either takes its handle

    threads = [threading.Thread(target=worker, args=(stream,)) for stream in (101, 202)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 2 and all(first is second for _, first, second in results)
    assert len({first for _, first, _ in results}) == 2, "threads share one re-streamed handle"
    assert all(index == 1 for _, index in created), f"handles were not created on the token's device: {created}"
    for stream, handle, _ in results:
        assert {s for h, s in bindings if h == handle} == {stream}


@pytest.mark.L0
def test_handle_cache_reuses_the_handles_of_exited_threads(monkeypatch):
    """Short-lived workers must not leave one native handle each behind: a finished thread's handle
    serves the next thread, and none is destroyed while a cached graph may still hold it."""
    import contextlib
    import importlib
    import threading
    from types import SimpleNamespace

    op = importlib.import_module("cudnn.gemm.ops.moe_grouped_matmul")
    _fresh_handle_cache(monkeypatch, op)
    created, destroyed = [], []
    monkeypatch.setattr(op.torch.cuda, "device", lambda index: contextlib.nullcontext())
    monkeypatch.setattr(op.torch.cuda, "current_stream", lambda unused=None: SimpleNamespace(cuda_stream=7))
    monkeypatch.setattr(op.cudnn, "create_handle", lambda: created.append(object()) or created[-1])
    monkeypatch.setattr(op.cudnn, "destroy_handle", destroyed.append)
    monkeypatch.setattr(op.cudnn, "set_stream", lambda *, handle, stream: None)

    for _ in range(4):
        worker = threading.Thread(target=op._get_handle, args=(torch.device("cuda", 0),))
        worker.start()
        worker.join()
    assert len(created) == 1, f"{len(created)} native handles for 4 sequential threads"
    assert destroyed == []
