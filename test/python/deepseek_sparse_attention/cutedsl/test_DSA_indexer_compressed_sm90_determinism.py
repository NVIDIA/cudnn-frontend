# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM90 compact IndexerForward + Top-K correctness, determinism and graph reuse."""

from __future__ import annotations

from collections import Counter

import cuda.bindings.driver as cuda
import pytest
import torch

from cudnn import DSA
from determinism_utils import assert_bitwise, assert_replays, snapshot
from deepseek_sparse_attention.cutedsl.dsa_utils import quantize_fp8_1x128

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]
SM_SCALE = 128**-0.5
ALLOWED_INF = {"logits": -torch.inf, "lse": -torch.inf}


def _require_sm90():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 GPU required")


def _sample(q_len: int, k_len: int, precision: str, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q_bf16 = (torch.randn(q_len, 64, 128, generator=generator, device="cuda") * 0.25).bfloat16()
    k_bf16 = (torch.randn(k_len, 1, 128, generator=generator, device="cuda") * 0.25).bfloat16()
    w_bf16 = (torch.rand(q_len, 64, generator=generator, device="cuda") + 0.1).bfloat16()
    w = w_bf16.float() if precision == "bf16" else w_bf16
    if precision == "bf16":
        q, k, q_scale, k_scale = q_bf16, k_bf16, None, None
        q_ref, k_ref = q.float(), k.float()
    else:
        q, q_scale = quantize_fp8_1x128(q_bf16)
        k, k_scale = quantize_fp8_1x128(k_bf16)
        q_ref = q.float() * q_scale.unsqueeze(-1)
        k_ref = k.float() * k_scale.unsqueeze(-1)
    return dict(q=q, k=k, w=w, q_scale=q_scale, k_scale=k_scale, q_ref=q_ref, k_ref=k_ref)


def _pack(samples, q_causal_offsets=None):
    q_lengths = [sample["q"].shape[0] for sample in samples]
    k_lengths = [sample["k"].shape[0] for sample in samples]
    packed = {
        "q": torch.cat([sample["q"] for sample in samples]),
        "k": torch.cat([sample["k"] for sample in samples]),
        "w": torch.cat([sample["w"] for sample in samples]),
        "cu_q": torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda"),
        "cu_k": torch.tensor([0, *torch.tensor(k_lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda"),
        "max_q": max(q_lengths, default=0),
        "max_k": max(k_lengths, default=0),
    }
    if q_causal_offsets is not None:
        packed["q_causal_offsets"] = torch.tensor(q_causal_offsets, dtype=torch.int32, device="cuda")
    if samples[0]["q_scale"] is not None:
        packed["q_scale"] = torch.cat([sample["q_scale"] for sample in samples])
        packed["k_scale"] = torch.cat([sample["k_scale"] for sample in samples])
    else:
        packed["q_scale"] = packed["k_scale"] = None
    return packed


def _invoke(packed, precision: str, preallocated, *, topk: int = 8, return_lse: bool = True, stream=None):
    return DSA.indexer_forward_top_k_wrapper(
        packed["q"],
        packed["k"],
        packed["w"],
        top_k=topk,
        ratio=4,
        qhead_per_kv_head=64,
        cu_seqlens_q=packed["cu_q"],
        cu_seqlens_k=packed["cu_k"],
        max_seqlen_q=packed["max_q"],
        max_seqlen_k=packed["max_k"],
        precision=precision,
        q_scale=packed["q_scale"],
        k_scale=packed["k_scale"],
        sm_scale=SM_SCALE,
        q_causal_offsets=packed.get("q_causal_offsets"),
        topk_indices_global=False,
        deterministic=True,
        return_lse=return_lse,
        stream=stream,
        **preallocated,
    )


def _run(samples, precision: str, *, topk: int = 8, preallocated=None, q_causal_offsets=None, stream=None):
    packed = _pack(samples, q_causal_offsets)
    total_q = packed["q"].shape[0]
    if preallocated is None:
        offsets, count = DSA.compress_topk_cand_buffer_size_thd(packed["cu_q"], packed["cu_k"], 4, packed.get("q_causal_offsets"))
        preallocated = {
            "cand_batch_offsets": offsets,
            "cand_buffer": torch.empty(count, dtype=torch.float32, device="cuda"),
            "out_indices": torch.empty(total_q, topk, dtype=torch.int32, device="cuda"),
            "out_logits": torch.empty(total_q, topk, dtype=torch.float32, device="cuda"),
            "softmax_out": torch.empty(total_q, topk, dtype=torch.float32, device="cuda"),
            "lse_out": torch.empty(total_q, dtype=torch.float32, device="cuda"),
        }
    result = _invoke(packed, precision, preallocated, topk=topk, stream=stream)
    return packed, preallocated, result


def _reference(sample, q_causal_offset=0):
    scores = torch.einsum("qhd,kd->qkh", sample["q_ref"].float(), sample["k_ref"][:, 0].float())
    scores = (torch.relu(scores) * sample["w"].float()[:, None, :]).sum(-1) * SM_SCALE
    for row in range(scores.shape[0]):
        scores[row, (q_causal_offset + row + 1) // 4 :] = -torch.inf
    return scores


def _check_reference(samples, output, topk=8, q_causal_offsets=None):
    q_causal_offsets = q_causal_offsets or [0] * len(samples)
    q_base = 0
    for sample, q_causal_offset in zip(samples, q_causal_offsets):
        scores = _reference(sample, q_causal_offset)
        q_len = scores.shape[0]
        indices = output["indices"][q_base : q_base + q_len]
        logits = output["logits"][q_base : q_base + q_len]
        softmax = output["softmax"][q_base : q_base + q_len]
        lse = output["lse"][q_base : q_base + q_len]
        for row in range(q_len):
            visible = min(scores.shape[1], (q_causal_offset + row + 1) // 4)
            valid = indices[row] >= 0
            assert int(valid.sum()) == min(topk, visible)
            assert bool((indices[row, ~valid] == -1).all())
            assert bool(torch.isneginf(logits[row, ~valid]).all())
            assert bool((softmax[row, ~valid] == 0).all())
            if visible:
                expected = torch.topk(scores[row, :visible], min(topk, visible))
                assert set(indices[row, valid].tolist()) == set(expected.indices.tolist())
                gathered = scores[row, indices[row, valid].long()]
                torch.testing.assert_close(logits[row, valid], gathered, atol=3e-3, rtol=3e-3)
                torch.testing.assert_close(softmax[row, valid], torch.softmax(logits[row, valid], dim=0), atol=1e-5, rtol=1e-4)
                torch.testing.assert_close(lse[row], torch.logsumexp(scores[row, :visible], dim=0), atol=3e-3, rtol=3e-3)
            else:
                assert torch.isneginf(lse[row])
        q_base += q_len


@pytest.mark.parametrize("precision", ["bf16", "fp8"])
def test_sm90_compressed_indexer_ragged_replay_batch_invariance_and_reference(precision):
    _require_sm90()
    samples = [_sample(0, 0, precision, 7), _sample(3, 1, precision, 11), _sample(7, 2, precision, 17)]

    def run(chosen):
        _, _, result = _run(chosen, precision)
        return result

    actual = assert_replays(lambda: run(samples), repeats=4, allowed_inf=ALLOWED_INF)
    _check_reference(samples, actual)
    individual = [snapshot(run([sample]), allowed_inf=ALLOWED_INF) for sample in samples]
    for order in ((0, 1, 2), (2, 0, 1), (1, 1)):
        chosen = [samples[index] for index in order]
        packed = snapshot(run(chosen), allowed_inf=ALLOWED_INF)
        q_base = 0
        for index, sample in zip(order, chosen):
            q_len = sample["q"].shape[0]
            assert_bitwise({name: value[q_base : q_base + q_len] for name, value in packed.items()}, individual[index], f"batch order {order}")
            q_base += q_len


def test_sm90_compressed_indexer_deterministic_ties_choose_smallest_indices():
    _require_sm90()
    sample = _sample(32, 8, "bf16", 23)
    sample["q"].zero_()
    sample["k"].zero_()
    sample["q_ref"].zero_()
    sample["k_ref"].zero_()
    _, _, result = _run([sample], "bf16", topk=4)
    for row in range(32):
        visible = min(8, (row + 1) // 4)
        valid = min(4, visible)
        selected = result["indices"][row]
        selected = selected[selected >= 0].sort().values
        assert torch.equal(selected, torch.arange(valid, device="cuda", dtype=torch.int32))


def test_sm90_compressed_indexer_caller_lse_buffer_enables_lse():
    _require_sm90()
    samples = [_sample(7, 2, "bf16", 29)]
    packed, buffers, expected = _run(samples, "bf16")
    expected_lse = expected["lse"].clone()
    buffers["lse_out"].fill_(torch.nan)
    actual = _invoke(packed, "bf16", buffers, return_lse=False)
    torch.testing.assert_close(actual["lse"], expected_lse, atol=0, rtol=0)


def test_sm90_compressed_indexer_nonzero_causal_offsets():
    _require_sm90()
    samples = [_sample(7, 4, "bf16", 29), _sample(9, 5, "bf16", 31)]
    q_causal_offsets = [5, 2]
    _, _, result = _run(samples, "bf16", q_causal_offsets=q_causal_offsets)
    _check_reference(samples, result, q_causal_offsets=q_causal_offsets)


def test_sm90_compressed_indexer_internal_sizing_uses_explicit_launch_stream():
    _require_sm90()
    samples = [_sample(11, 3, "bf16", 30)]
    packed = _pack(samples)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    raw_stream = cuda.CUstream(side.cuda_stream)
    result = _invoke(packed, "bf16", {}, stream=raw_stream)
    side.synchronize()
    _check_reference(samples, snapshot(dict(result), allowed_inf=ALLOWED_INF))


def test_sm90_compressed_indexer_rejects_unvalidated_configurations():
    _require_sm90()
    sample = _sample(7, 2, "bf16", 30)
    packed = _pack([sample])
    common = dict(
        top_k=4,
        qhead_per_kv_head=64,
        cu_seqlens_q=packed["cu_q"],
        cu_seqlens_k=packed["cu_k"],
        max_seqlen_q=packed["max_q"],
        max_seqlen_k=packed["max_k"],
        precision="bf16",
    )
    with pytest.raises(NotImplementedError, match="ratio=4"):
        DSA.indexer_forward_top_k_wrapper(packed["q"], packed["k"], packed["w"], ratio=2, **common)
    with pytest.raises(NotImplementedError, match="packed THD"):
        DSA.indexer_forward_top_k_wrapper(sample["q"].unsqueeze(0), sample["k"].unsqueeze(0), sample["w"].unsqueeze(0), top_k=4, ratio=4)

    alias_sample = _sample(32, 8, "bf16", 31)
    alias_packed = _pack([alias_sample])
    offsets, count = DSA.compress_topk_cand_buffer_size_thd(alias_packed["cu_q"], alias_packed["cu_k"], 4)
    cand = torch.empty(count, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="cand_buffer overlaps out_logits"):
        DSA.indexer_forward_top_k_wrapper(
            alias_packed["q"],
            alias_packed["k"],
            alias_packed["w"],
            top_k=1,
            ratio=4,
            qhead_per_kv_head=64,
            cu_seqlens_q=alias_packed["cu_q"],
            cu_seqlens_k=alias_packed["cu_k"],
            max_seqlen_q=alias_packed["max_q"],
            max_seqlen_k=alias_packed["max_k"],
            cand_batch_offsets=offsets,
            cand_buffer=cand,
            out_indices=torch.empty((32, 1), dtype=torch.int32, device="cuda"),
            out_logits=cand[:32].view(32, 1),
            return_softmax=False,
        )


def test_sm90_compressed_indexer_compile_count_ignores_runtime_thd_geometry(monkeypatch):
    _require_sm90()
    from cudnn.deepseek_sparse_attention.indexer_forward import _interface_sm90
    from cudnn.deepseek_sparse_attention.indexer_top_k import compress_top_k_sm100

    _interface_sm90._compile_cache.clear()
    compress_top_k_sm100._compile_cache.clear()
    original_compile = _interface_sm90.cute.compile
    counts = Counter()

    def counted_compile(kernel, *args, **kwargs):
        counts[type(kernel).__name__] += 1
        return original_compile(kernel, *args, **kwargs)

    monkeypatch.setattr(_interface_sm90.cute, "compile", counted_compile)
    for lengths, offsets, seed in (
        ([(3, 1), (7, 2)], [0, 5], 31),
        ([(0, 0), (3, 1), (513, 129)], [11, 2, 7], 41),
    ):
        samples = [_sample(q_len, k_len, "bf16", seed + i) for i, (q_len, k_len) in enumerate(lengths)]
        _run(samples, "bf16", q_causal_offsets=offsets)
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        samples = [_sample(5, 2, "bf16", 51), _sample(9, 3, "bf16", 52)]
        _run(samples, "bf16", q_causal_offsets=[3, 9], stream=cuda.CUstream(side.cuda_stream))
    side.synchronize()
    assert counts["IndexerForwardSm90"] == 1
    assert counts["CompressTopkStage2"] == 1


def test_sm90_compressed_indexer_cuda_graph_replay_changed_inputs(monkeypatch):
    _require_sm90()
    samples = [_sample(3, 1, "bf16", 51), _sample(11, 3, "bf16", 53)]
    packed = _pack(samples)
    offsets, count = DSA.compress_topk_cand_buffer_size_thd(packed["cu_q"], packed["cu_k"], 4)
    buffers = {
        "cand_batch_offsets": offsets,
        "cand_buffer": torch.empty(count, dtype=torch.float32, device="cuda"),
        "out_indices": torch.empty(packed["q"].shape[0], 8, dtype=torch.int32, device="cuda"),
        "out_logits": torch.empty(packed["q"].shape[0], 8, dtype=torch.float32, device="cuda"),
        "softmax_out": torch.empty(packed["q"].shape[0], 8, dtype=torch.float32, device="cuda"),
        "lse_out": torch.empty(packed["q"].shape[0], dtype=torch.float32, device="cuda"),
    }
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    raw_stream = cuda.CUstream(stream.cuda_stream)
    with torch.cuda.stream(stream):
        _invoke(packed, "bf16", buffers, stream=raw_stream)  # compile and warm up
    stream.synchronize()

    from cudnn.deepseek_sparse_attention.indexer_forward import _compressed_top_k_sm90, _interface_sm90
    from cudnn.deepseek_sparse_attention.indexer_top_k import compress_top_k_sm100

    def forbidden(*args, **kwargs):
        raise AssertionError("capture path attempted JIT, allocation, or candidate-offset construction")

    monkeypatch.setattr(_interface_sm90.cute, "compile", forbidden)
    monkeypatch.setattr(compress_top_k_sm100.cute, "compile", forbidden)
    monkeypatch.setattr(_compressed_top_k_sm90, "_compress_cand_batch_offsets", forbidden)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            with monkeypatch.context() as capture_patch:
                for name in ("empty", "empty_like", "zeros", "zeros_like"):
                    capture_patch.setattr(torch, name, forbidden)
                capture_patch.setattr(torch.Tensor, "item", forbidden)
                _invoke(packed, "bf16", buffers, stream=raw_stream)
        with torch.cuda.stream(stream):
            graph.replay()
        torch.cuda.synchronize()
        expected = snapshot(
            {"indices": buffers["out_indices"], "logits": buffers["out_logits"], "softmax": buffers["softmax_out"], "lse": buffers["lse_out"]},
            allowed_inf=ALLOWED_INF,
        )

        with torch.cuda.stream(stream):
            packed["q"].add_(torch.tensor(0.125, dtype=torch.bfloat16, device="cuda"))
            buffers["cand_buffer"].fill_(torch.nan)
            buffers["out_indices"].fill_(123456)
            buffers["out_logits"].fill_(torch.nan)
            buffers["softmax_out"].fill_(torch.nan)
            buffers["lse_out"].fill_(torch.nan)
            graph.replay()
        torch.cuda.synchronize()
        changed = snapshot(
            {"indices": buffers["out_indices"], "logits": buffers["out_logits"], "softmax": buffers["softmax_out"], "lse": buffers["lse_out"]},
            allowed_inf=ALLOWED_INF,
        )
        assert any(torch.count_nonzero(changed[name].view(torch.uint8) != expected[name].view(torch.uint8)).item() for name in changed)

        eager_buffers = {
            "cand_batch_offsets": offsets,
            "cand_buffer": torch.empty_like(buffers["cand_buffer"]),
            "out_indices": torch.empty_like(buffers["out_indices"]),
            "out_logits": torch.empty_like(buffers["out_logits"]),
            "softmax_out": torch.empty_like(buffers["softmax_out"]),
            "lse_out": torch.empty_like(buffers["lse_out"]),
        }
        with torch.cuda.stream(stream):
            eager = _invoke(packed, "bf16", eager_buffers, stream=raw_stream)
        stream.synchronize()
        assert_bitwise(snapshot(dict(eager), allowed_inf=ALLOWED_INF), changed, "captured versus eager changed inputs")

        # Replay again from the same changed stable-address operands: all bytes must repeat.
        with torch.cuda.stream(stream):
            for name, tensor in buffers.items():
                if name == "cand_batch_offsets":
                    continue
                tensor.fill_(torch.nan if tensor.dtype.is_floating_point else 123456)
            graph.replay()
        torch.cuda.synchronize()
        replayed = snapshot(
            {"indices": buffers["out_indices"], "logits": buffers["out_logits"], "softmax": buffers["softmax_out"], "lse": buffers["lse_out"]},
            allowed_inf=ALLOWED_INF,
        )
        assert_bitwise(replayed, changed, "CUDA Graph replay")
    finally:
        graph.reset()
