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


def _prepare_plan(packed, offsets, count, *, topk=8, precision="bf16", deterministic=True, return_softmax=True, return_lse=True):
    """Build the public plan from one packed sample contract."""
    return DSA.prepare_indexer_forward_top_k(
        packed["q"],
        packed["k"],
        packed["w"],
        packed["cu_q"],
        packed["cu_k"],
        offsets,
        top_k=topk,
        max_seqlen_q=packed.get("max_q", packed["q"].shape[0]),
        max_seqlen_k=packed.get("max_k", packed["k"].shape[0]),
        max_candidate_count=count,
        ratio=4,
        sm_scale=SM_SCALE,
        precision=precision,
        sample_q_scale=packed["q_scale"],
        sample_k_scale=packed["k_scale"],
        sample_q_causal_offsets=packed["q_causal_offsets"],
        return_softmax=return_softmax,
        return_lse=return_lse,
        deterministic=deterministic,
    )


def _plan_outputs(total_q, topk):
    """Allocate every optional plan output before execution or capture."""
    return {
        "out_indices": torch.empty(total_q, topk, dtype=torch.int32, device="cuda"),
        "out_logits": torch.empty(total_q, topk, dtype=torch.float32, device="cuda"),
        "softmax_out": torch.empty(total_q, topk, dtype=torch.float32, device="cuda"),
        "lse_out": torch.empty(total_q, dtype=torch.float32, device="cuda"),
    }


def _execute_plan(plan, packed, offsets, outputs, workspace, stream=None):
    """Bind one packed payload to a prepared plan without adapting storage."""
    plan.execute(
        packed["q"],
        packed["k"],
        packed["w"],
        packed["cu_q"],
        packed["cu_k"],
        offsets,
        outputs["out_indices"],
        outputs["out_logits"],
        workspace=workspace,
        q_scale=packed["q_scale"],
        k_scale=packed["k_scale"],
        q_causal_offsets=packed["q_causal_offsets"],
        softmax_out=outputs["softmax_out"],
        lse_out=outputs["lse_out"],
        current_stream=stream,
    )


def _run_prepared_plan(samples, precision, weight_dtype):
    """Pack, prepare, and execute one weight-dtype specialization."""
    packed = _pack(samples, [0] * len(samples))
    packed["w"] = packed["w"].to(weight_dtype)
    offsets, count = DSA.compress_topk_cand_buffer_size_thd(packed["cu_q"], packed["cu_k"], 4, packed["q_causal_offsets"])
    plan = _prepare_plan(packed, offsets, count, precision=precision)
    outputs = _plan_outputs(packed["q"].shape[0], 8)
    workspace = torch.empty(plan.get_workspace_size(), dtype=torch.uint8, device="cuda")
    _execute_plan(plan, packed, offsets, outputs, workspace)
    return {
        "indices": outputs["out_indices"],
        "logits": outputs["out_logits"],
        "softmax": outputs["softmax_out"],
        "lse": outputs["lse_out"],
    }


@pytest.mark.parametrize(
    ("precision", "weight_dtype"),
    (("bf16", torch.float32), ("bf16", torch.bfloat16), ("fp8", torch.bfloat16)),
    ids=("bf16-w-fp32", "bf16-w-bf16", "fp8-w-bf16"),
)
def test_sm90_prepared_plan_ragged_replay_batch_invariance_and_reference(precision, weight_dtype):
    """Check replay, ragged batch isolation, and the independent reference."""
    _require_sm90()
    samples = [_sample(0, 0, precision, 107), _sample(3, 1, precision, 109), _sample(7, 2, precision, 113)]
    actual = assert_replays(lambda: _run_prepared_plan(samples, precision, weight_dtype), repeats=4, allowed_inf=ALLOWED_INF)
    _check_reference(samples, actual)
    individual = [snapshot(_run_prepared_plan([sample], precision, weight_dtype), allowed_inf=ALLOWED_INF) for sample in samples]
    for order in ((0, 1, 2), (2, 0, 1), (1, 1)):
        chosen = [samples[index] for index in order]
        packed = snapshot(_run_prepared_plan(chosen, precision, weight_dtype), allowed_inf=ALLOWED_INF)
        q_base = 0
        for index, sample in zip(order, chosen):
            q_len = sample["q"].shape[0]
            assert_bitwise({name: value[q_base : q_base + q_len] for name, value in packed.items()}, individual[index], f"prepared batch order {order}")
            q_base += q_len


@pytest.mark.parametrize(
    ("precision", "weight_dtype"),
    (("bf16", torch.float32), ("bf16", torch.bfloat16), ("fp8", torch.bfloat16)),
    ids=("bf16-w-fp32", "bf16-w-bf16", "fp8-w-bf16"),
)
def test_sm90_prepared_plan_matches_eager_wrapper(precision, weight_dtype):
    """Match each prepared precision/weight variant to the eager wrapper."""
    _require_sm90()
    samples = [_sample(0, 0, precision, 61), _sample(3, 1, precision, 63), _sample(7, 2, precision, 65)]
    q_causal_offsets = [0, 1, 0]
    packed = _pack(samples, q_causal_offsets)
    packed["w"] = packed["w"].to(weight_dtype)
    offsets, count = DSA.compress_topk_cand_buffer_size_thd(packed["cu_q"], packed["cu_k"], 4, packed["q_causal_offsets"])
    plan = _prepare_plan(packed, offsets, count, precision=precision)
    workspace = torch.empty(plan.get_workspace_size(), dtype=torch.uint8, device="cuda")
    outputs = _plan_outputs(packed["q"].shape[0], 8)
    _execute_plan(plan, packed, offsets, outputs, workspace)

    eager_buffers = {
        "cand_batch_offsets": offsets,
        "cand_buffer": torch.empty(count, dtype=torch.float32, device="cuda"),
        **_plan_outputs(packed["q"].shape[0], 8),
    }
    expected = snapshot(dict(_invoke(packed, precision, eager_buffers)), allowed_inf=ALLOWED_INF)
    actual = snapshot(
        {
            "indices": outputs["out_indices"],
            "logits": outputs["out_logits"],
            "softmax": outputs["softmax_out"],
            "lse": outputs["lse_out"],
        },
        allowed_inf=ALLOWED_INF,
    )
    assert_bitwise(actual, expected, f"prepared {precision} versus eager")
    _check_reference(samples, actual, q_causal_offsets=q_causal_offsets)


def test_sm90_prepared_plan_default_optional_outputs():
    """Keep disabled LSE absent while serving the default softmax output."""
    _require_sm90()
    packed = _pack([_sample(7, 2, "bf16", 66)], [1])
    offsets, count = DSA.compress_topk_cand_buffer_size_thd(packed["cu_q"], packed["cu_k"], 4, packed["q_causal_offsets"])
    plan = _prepare_plan(packed, offsets, count, return_softmax=True, return_lse=False)
    outputs = _plan_outputs(7, 8)
    workspace = torch.empty(plan.get_workspace_size(), dtype=torch.uint8, device="cuda")
    plan.execute(
        packed["q"],
        packed["k"],
        packed["w"],
        packed["cu_q"],
        packed["cu_k"],
        offsets,
        outputs["out_indices"],
        outputs["out_logits"],
        workspace=workspace,
        q_causal_offsets=packed["q_causal_offsets"],
        softmax_out=outputs["softmax_out"],
    )
    eager_buffers = {
        "cand_batch_offsets": offsets,
        "cand_buffer": torch.empty(count, dtype=torch.float32, device="cuda"),
        "out_indices": torch.empty_like(outputs["out_indices"]),
        "out_logits": torch.empty_like(outputs["out_logits"]),
        "softmax_out": torch.empty_like(outputs["softmax_out"]),
    }
    expected = snapshot(dict(_invoke(packed, "bf16", eager_buffers, return_lse=False)), allowed_inf=ALLOWED_INF)
    actual = snapshot({"indices": outputs["out_indices"], "logits": outputs["out_logits"], "softmax": outputs["softmax_out"]}, allowed_inf=ALLOWED_INF)
    assert_bitwise(actual, expected, "prepared default optional outputs")


def test_sm90_prepared_plan_zero_candidate_workspace_short_sequence():
    """Verify short sequences need no candidate bytes and emit sentinels."""
    _require_sm90()
    samples = [_sample(3, 1, "bf16", 67)]
    packed = _pack(samples, [0])
    offsets, count = DSA.compress_topk_cand_buffer_size_thd(packed["cu_q"], packed["cu_k"], 4, packed["q_causal_offsets"])
    assert count == 0
    plan = _prepare_plan(packed, offsets, count)
    assert plan.get_workspace_size() == 0
    outputs = _plan_outputs(3, 8)
    _execute_plan(plan, packed, offsets, outputs, torch.empty(0, dtype=torch.uint8, device="cuda"))
    assert bool((outputs["out_indices"] == -1).all())
    assert bool(torch.isneginf(outputs["out_logits"]).all())
    assert bool((outputs["softmax_out"] == 0).all())
    assert bool(torch.isneginf(outputs["lse_out"]).all())


def test_sm90_prepared_plan_rejects_unvalidated_specialization_before_compile(monkeypatch):
    """Reject unsupported static axes before reaching the compiler."""
    _require_sm90()
    packed = _pack([_sample(7, 2, "bf16", 68)], [0])
    offsets, count = DSA.compress_topk_cand_buffer_size_thd(packed["cu_q"], packed["cu_k"], 4, packed["q_causal_offsets"])
    from cudnn.deepseek_sparse_attention.indexer_forward import _interface_sm90

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported plan reached compilation")

    monkeypatch.setattr(_interface_sm90.cute, "compile", forbidden)
    plan = DSA.IndexerForwardTopKPlan(
        packed["q"],
        packed["k"],
        packed["w"],
        packed["cu_q"],
        packed["cu_k"],
        offsets,
        top_k=8,
        max_seqlen_q=packed["max_q"],
        max_seqlen_k=packed["max_k"],
        max_candidate_count=count,
        ratio=2,
        sample_q_causal_offsets=packed["q_causal_offsets"],
    )
    with pytest.raises(NotImplementedError, match="ratio=4"):
        plan.check_support()


def test_sm90_prepared_plan_rejects_workspace_failures_before_launch(monkeypatch):
    """Reject undersized or aliased workspace without mutating outputs."""
    _require_sm90()
    packed = _pack([_sample(7, 2, "bf16", 69)], [1])
    offsets, count = DSA.compress_topk_cand_buffer_size_thd(packed["cu_q"], packed["cu_k"], 4, packed["q_causal_offsets"])
    plan = _prepare_plan(packed, offsets, count)
    outputs = _plan_outputs(7, 8)
    for tensor in outputs.values():
        tensor.fill_(123456 if tensor.dtype == torch.int32 else torch.nan)
    sentinels = {name: tensor.clone() for name, tensor in outputs.items()}

    from cudnn.deepseek_sparse_attention.indexer_forward import _compressed_top_k_sm90, prepared

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid workspace reached a kernel launch")

    with monkeypatch.context() as patch:
        patch.setattr(prepared, "indexer_fwd_compress_topk_sm90", forbidden)
        with pytest.raises(ValueError, match="requires .* workspace"):
            _execute_plan(
                plan,
                packed,
                offsets,
                outputs,
                torch.empty(plan.get_workspace_size() - 1, dtype=torch.uint8, device="cuda"),
            )

    with monkeypatch.context() as patch:
        patch.setattr(_compressed_top_k_sm90, "indexer_fwd_sm90", forbidden)
        with pytest.raises(ValueError, match="cand_buffer overlaps q"):
            _execute_plan(plan, packed, offsets, outputs, packed["q"].view(torch.uint8))

    for name, tensor in outputs.items():
        assert_bitwise({name: tensor}, {name: sentinels[name]}, f"workspace rejection preserved {name}")


def test_sm90_prepared_plan_compile_cache_ignores_capacities(monkeypatch):
    """Bound compilation to codegen axes across changing capacities."""
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
    last_plan = last_packed = last_offsets = None
    for qk_lengths, candidate_slack, seed in (
        ([(3, 2), (11, 6)], 17, 71),
        ([(0, 0), (7, 4), (9, 4)], 113, 81),
        ([(1, 1), (127, 63), (130, 65)], 257, 86),
    ):
        packed = _pack([_sample(q_len, k_len, "bf16", seed + i) for i, (q_len, k_len) in enumerate(qk_lengths)], [1] * len(qk_lengths))
        offsets, count = DSA.compress_topk_cand_buffer_size_thd(packed["cu_q"], packed["cu_k"], 4, packed["q_causal_offsets"])
        last_plan = _prepare_plan(packed, offsets, count + candidate_slack)
        last_packed, last_offsets = packed, offsets

    # Candidate capacity, packed totals, segment count and prefix contents are
    # plan/runtime facts, not generated-code axes.
    assert counts["IndexerForwardSm90"] == 1
    assert counts["CompressTopkStage2"] == 1

    # Retained artifacts accept different addresses and a different launch
    # stream without consulting the process-global compile caches.
    cloned = {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in last_packed.items()}
    cloned_offsets = last_offsets.clone()
    outputs = _plan_outputs(cloned["q"].shape[0], 8)
    workspace = torch.empty(last_plan.get_workspace_size(), dtype=torch.uint8, device="cuda")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _execute_plan(last_plan, cloned, cloned_offsets, outputs, workspace, cuda.CUstream(side.cuda_stream))
    side.synchronize()
    assert counts["IndexerForwardSm90"] == 1
    assert counts["CompressTopkStage2"] == 1

    packed = _pack([_sample(14, 8, "bf16", 91)], [1])
    offsets, count = DSA.compress_topk_cand_buffer_size_thd(packed["cu_q"], packed["cu_k"], 4, packed["q_causal_offsets"])
    _prepare_plan(packed, offsets, count, deterministic=False)
    assert counts["IndexerForwardSm90"] == 1
    assert counts["CompressTopkStage2"] == 2


def test_sm90_prepared_plan_cuda_graph_replay_updates_packed_schedule(monkeypatch):
    """Replay changed device metadata without JIT, allocation, or host sync."""
    _require_sm90()
    # Both schedules keep Q/K storage and metadata addresses fixed. Only the
    # prefix/causal/candidate-offset contents change between graph replays.
    sample = _sample(14, 8, "bf16", 101)
    packed = {
        "q": sample["q"],
        "k": sample["k"],
        "w": sample["w"],
        "q_scale": None,
        "k_scale": None,
        "cu_q": torch.tensor([0, 0, 3, 14], dtype=torch.int32, device="cuda"),
        "cu_k": torch.tensor([0, 0, 2, 8], dtype=torch.int32, device="cuda"),
        "q_causal_offsets": torch.tensor([0, 1, 0], dtype=torch.int32, device="cuda"),
        "max_q": 14,
        "max_k": 8,
    }
    next_cu_q = torch.tensor([0, 7, 7, 14], dtype=torch.int32, device="cuda")
    next_cu_k = torch.tensor([0, 4, 4, 8], dtype=torch.int32, device="cuda")
    next_q_offsets = torch.tensor([0, 0, 1], dtype=torch.int32, device="cuda")
    offsets, count = DSA.compress_topk_cand_buffer_size_thd(packed["cu_q"], packed["cu_k"], 4, packed["q_causal_offsets"])
    next_offsets, next_count = DSA.compress_topk_cand_buffer_size_thd(next_cu_q, next_cu_k, 4, next_q_offsets)
    capacity = max(count, next_count)
    plan = _prepare_plan(packed, offsets, capacity)
    workspace = torch.empty(plan.get_workspace_size(), dtype=torch.uint8, device="cuda")
    outputs = _plan_outputs(14, 8)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    raw_stream = cuda.CUstream(stream.cuda_stream)

    from cudnn.deepseek_sparse_attention.indexer_forward import _compressed_top_k_sm90, _interface_sm90
    from cudnn.deepseek_sparse_attention.indexer_top_k import compress_top_k_sm100

    # A prepared plan owns both artifacts; execution must not depend on the
    # process-global compile caches remaining populated.
    _interface_sm90._compile_cache.clear()
    compress_top_k_sm100._compile_cache.clear()

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared capture attempted JIT, allocation, D2H read, or candidate-offset construction")

    monkeypatch.setattr(_interface_sm90.cute, "compile", forbidden)
    monkeypatch.setattr(compress_top_k_sm100.cute, "compile", forbidden)
    monkeypatch.setattr(_compressed_top_k_sm90, "_compress_cand_batch_offsets", forbidden)

    with torch.cuda.stream(stream):
        _execute_plan(plan, packed, offsets, outputs, workspace, raw_stream)
    stream.synchronize()
    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
    sync_debug_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.cuda.stream(stream):
            for _ in range(3):
                _execute_plan(plan, packed, offsets, outputs, workspace, raw_stream)
    finally:
        torch.cuda.set_sync_debug_mode(sync_debug_mode)
    stream.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            with monkeypatch.context() as capture_patch:
                for name in ("empty", "empty_like", "zeros", "zeros_like"):
                    capture_patch.setattr(torch, name, forbidden)
                capture_patch.setattr(torch.Tensor, "item", forbidden)
                sync_debug_mode = torch.cuda.get_sync_debug_mode()
                torch.cuda.set_sync_debug_mode("error")
                try:
                    _execute_plan(plan, packed, offsets, outputs, workspace, raw_stream)
                finally:
                    torch.cuda.set_sync_debug_mode(sync_debug_mode)

        with torch.cuda.stream(stream):
            packed["cu_q"].copy_(next_cu_q)
            packed["cu_k"].copy_(next_cu_k)
            packed["q_causal_offsets"].copy_(next_q_offsets)
            offsets.copy_(next_offsets)
            workspace.fill_(0xA5)
            outputs["out_indices"].fill_(123456)
            outputs["out_logits"].fill_(torch.nan)
            outputs["softmax_out"].fill_(torch.nan)
            outputs["lse_out"].fill_(torch.nan)
            graph.replay()
        stream.synchronize()

        # Drop the capture tripwires only after capture/replay has completed;
        # the independent public wrapper may populate its ordinary cache now.
        monkeypatch.undo()
        eager_buffers = {
            "cand_batch_offsets": offsets,
            "cand_buffer": torch.empty(capacity, dtype=torch.float32, device="cuda"),
            **_plan_outputs(14, 8),
        }
        with torch.cuda.stream(stream):
            eager = _invoke(packed, "bf16", eager_buffers, stream=raw_stream)
        stream.synchronize()
        actual = snapshot(
            {
                "indices": outputs["out_indices"],
                "logits": outputs["out_logits"],
                "softmax": outputs["softmax_out"],
                "lse": outputs["lse_out"],
            },
            allowed_inf=ALLOWED_INF,
        )
        expected = snapshot(
            dict(eager),
            allowed_inf=ALLOWED_INF,
        )
        assert_bitwise(actual, expected, "prepared plan changed schedule")

        with torch.cuda.stream(stream):
            workspace.fill_(0x5A)
            outputs["out_indices"].fill_(123456)
            outputs["out_logits"].fill_(torch.nan)
            outputs["softmax_out"].fill_(torch.nan)
            outputs["lse_out"].fill_(torch.nan)
            graph.replay()
        stream.synchronize()
        replayed = snapshot(
            {
                "indices": outputs["out_indices"],
                "logits": outputs["out_logits"],
                "softmax": outputs["softmax_out"],
                "lse": outputs["lse_out"],
            },
            allowed_inf=ALLOWED_INF,
        )
        assert_bitwise(replayed, expected, "prepared plan deterministic replay")
    finally:
        graph.reset()


def test_sm90_prepared_plan_megatron_query_window_graph_integration():
    """A Megatron-like caller owns Q windowing, stable buffers and graph buckets.

    One plan is captured twice with different Q/output/metadata pointers. Both
    calls reuse the full K tensor and the same candidate workspace in stream
    order. The tail window absorbs unused Q capacity in an empty-K segment, so
    no K copy or cuDNN-FE scheduler abstraction is needed.
    """
    _require_sm90()
    topk, window_q, total_q, total_k = 8, 8, 14, 8

    def make_payload(seed):
        """Create stable-capacity query windows over one full K payload."""
        full = _sample(total_q, total_k, "bf16", seed)
        q0 = full["q"][:window_q].clone()
        q1 = torch.cat(
            (
                full["q"][window_q:],
                torch.zeros((window_q - (total_q - window_q), 64, 128), dtype=full["q"].dtype, device="cuda"),
            )
        )
        w0 = full["w"][:window_q].clone()
        w1 = torch.cat(
            (
                full["w"][window_q:],
                torch.zeros((window_q - (total_q - window_q), 64), dtype=full["w"].dtype, device="cuda"),
            )
        )
        return full, q0, q1, w0, w1

    initial, q0, q1, w0, w1 = make_payload(127)
    k = initial["k"].clone()
    cu_q = (
        torch.tensor([0, 8, 8], dtype=torch.int32, device="cuda"),
        torch.tensor([0, 6, 8], dtype=torch.int32, device="cuda"),
    )
    cu_k = tuple(torch.tensor([0, 8, 8], dtype=torch.int32, device="cuda") for _ in range(2))
    q_offsets = (
        torch.tensor([0, 0], dtype=torch.int32, device="cuda"),
        torch.tensor([8, 0], dtype=torch.int32, device="cuda"),
    )
    cand = tuple(DSA.compress_topk_cand_buffer_size_thd(cu_q[i], cu_k[i], 4, q_offsets[i]) for i in range(2))
    capacity = max(count for _, count in cand)

    prepared = {
        "q": q0,
        "k": k,
        "w": w0,
        "q_scale": None,
        "k_scale": None,
        "cu_q": cu_q[0],
        "cu_k": cu_k[0],
        "q_causal_offsets": q_offsets[0],
        "max_q": window_q,
        "max_k": total_k,
    }
    plan = _prepare_plan(prepared, cand[0][0], capacity, topk=topk)
    workspace = torch.empty(plan.get_workspace_size(), dtype=torch.uint8, device="cuda")
    outputs = (_plan_outputs(window_q, topk), _plan_outputs(window_q, topk))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    raw_stream = cuda.CUstream(stream.cuda_stream)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            for i, (query, weight) in enumerate(((q0, w0), (q1, w1))):
                binding = {
                    "q": query,
                    "k": k,
                    "w": weight,
                    "q_scale": None,
                    "k_scale": None,
                    "cu_q": cu_q[i],
                    "cu_k": cu_k[i],
                    "q_causal_offsets": q_offsets[i],
                }
                # Megatron enters the graph stream context and need not expose
                # a driver stream object to the plan API.
                _execute_plan(plan, binding, cand[i][0], outputs[i], workspace)

        def replay_and_compare(full):
            """Replay both windows and compare their concatenation to eager."""
            with torch.cuda.stream(stream):
                workspace.fill_(0xA5)
                for result in outputs:
                    result["out_indices"].fill_(123456)
                    result["out_logits"].fill_(torch.nan)
                    result["softmax_out"].fill_(torch.nan)
                    result["lse_out"].fill_(torch.nan)
                graph.replay()
            stream.synchronize()

            combined = {
                "indices": torch.cat((outputs[0]["out_indices"], outputs[1]["out_indices"][:6])),
                "logits": torch.cat((outputs[0]["out_logits"], outputs[1]["out_logits"][:6])),
                "softmax": torch.cat((outputs[0]["softmax_out"], outputs[1]["softmax_out"][:6])),
                "lse": torch.cat((outputs[0]["lse_out"], outputs[1]["lse_out"][:6])),
            }
            packed = _pack([full], [0])
            eager_offsets, eager_count = DSA.compress_topk_cand_buffer_size_thd(packed["cu_q"], packed["cu_k"], 4, packed["q_causal_offsets"])
            eager_buffers = {
                "cand_batch_offsets": eager_offsets,
                "cand_buffer": torch.empty(eager_count, dtype=torch.float32, device="cuda"),
                **_plan_outputs(total_q, topk),
            }
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                eager = _invoke(packed, "bf16", eager_buffers, stream=raw_stream)
            stream.synchronize()
            expected = snapshot(dict(eager), allowed_inf=ALLOWED_INF)
            assert_bitwise(snapshot(combined, allowed_inf=ALLOWED_INF), expected, "Megatron Q-window composition")

            # The two unused rows are a Q-capacity segment with no K rows.
            assert bool((outputs[1]["out_indices"][6:] == -1).all())
            assert bool(torch.isneginf(outputs[1]["out_logits"][6:]).all())
            assert bool((outputs[1]["softmax_out"][6:] == 0).all())
            assert bool(torch.isneginf(outputs[1]["lse_out"][6:]).all())

        replay_and_compare(initial)

        changed, next_q0, next_q1, next_w0, next_w1 = make_payload(131)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            q0.copy_(next_q0)
            q1.copy_(next_q1)
            w0.copy_(next_w0)
            w1.copy_(next_w1)
            k.copy_(changed["k"])
        replay_and_compare(changed)
    finally:
        graph.reset()
