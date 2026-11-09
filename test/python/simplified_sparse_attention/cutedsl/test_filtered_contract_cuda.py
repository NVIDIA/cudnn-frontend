# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Cold filtered repair and changed-input replay on SM100/SM103.

Zero indexer queries force exact repair of 65K-way ties. The public API runs
without warmup; selected output/dQ rows use an independent reference.
"""

import json

import pytest

import torch

pytestmark = [
    pytest.mark.L1,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
        reason="The packaged CuTe kernels require SM100 or SM103",
    ),
]


def test_cold_filtered_repair_and_changed_input_replays(monkeypatch):
    from cudnn.ops import simplified_sparse_attention
    from cudnn.simplified_sparse_attention.cutedsl import backend
    from cudnn.simplified_sparse_attention.cutedsl.kernels import _cuda_preload
    from cudnn.simplified_sparse_attention.cutedsl.kernels import indexer_forward_topk_fused_tcgen05 as fused

    assert fused._preload_tvm_ffi_cuda is _cuda_preload.preload_tvm_ffi_cuda
    heads, topk = 16, 512
    # First three calls share storage despite crossing the old exact-shape
    # and 128/256 tile keys. Fourth grows capacity; fifth shrinks again.
    shapes = ((128, 131072), (193, 131329), (129, 131079), (257, 147713), (193, 131329))
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(20261002)
    stream = torch.cuda.Stream(device=device)
    previous_keys = set(fused._state_filtered)
    saved_forward = backend._run_sparse_forward
    observed = {}

    def capture_selected(*args, **kwargs):
        result = saved_forward(*args, **kwargs)
        observed["indices"] = result[5].detach().clone()
        observed["lengths"] = result[6].detach().clone()
        observed["predict"] = result[8].detach().clone()
        return result

    monkeypatch.setattr(backend, "_run_sparse_forward", capture_selected)
    # Keep prior caller allocations alive so later calls cannot recycle them.
    # The graph must reuse its persistent buffers while caller addresses change.
    caller_allocations = []
    graph = state = compiled_bundle = None
    previous_stats = (0, 0)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.cuda.stream(stream):
            for iteration, (tq, tk) in enumerate(shapes):
                row = torch.arange(tq, device=device, dtype=torch.int32)
                print(json.dumps(dict(event="filtered_before_forward", iteration=iteration)), flush=True)

                def random(shape):
                    return (torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator) * 0.2).requires_grad_()

                inputs = (
                    random((tq, heads, 256)),
                    random((tk, 1, 256)),
                    random((tk, 1, 256)),
                    torch.zeros((tq, 128), device=device, dtype=torch.bfloat16, requires_grad=True),
                    random((tk, 128)),
                )
                if iteration == 0:
                    starts, ends = torch.zeros_like(row), torch.full_like(row, tk // 2)
                    valid = torch.ones(tq, device=device, dtype=torch.bool)
                elif iteration == 1:
                    starts, ends = torch.full_like(row, tk // 2), torch.full_like(row, tk)
                    valid = row % 5 != 0
                else:
                    starts, ends = tk // 4 + (row % 4) * 256, 3 * tk // 4 - (row % 3) * 128
                    valid = row % 7 != 3
                if iteration == 2:
                    # Mix ordinary rows, short documents, and empty padding
                    # after a larger shape used these same persistent buffers.
                    ends = torch.where(row % 11 == 0, starts + 17, ends)
                    valid = valid & (row % 13 != 0)
                    # Short rows collect every key without repair; nonzero
                    # indexer queries also exercise runtime-shape score loads
                    # and the native compact predict path, not only ties.
                    with torch.no_grad():
                        short = row % 11 == 0
                        inputs[3][short] = torch.randn((int(short.sum().item()), 128), device=device, dtype=torch.bfloat16, generator=generator) * 0.2
                metadata = backend.RowMetadata(
                    starts,
                    ends,
                    valid,
                    valid.sum().float().reshape(1),
                    triangular_scores=False,
                    mask_invalid_rows=True,
                )
                caller_allocations.append((inputs, metadata))
                upstream = torch.randn(inputs[0].shape, device=device, dtype=torch.bfloat16, generator=generator) * 0.2
                frozen_upstream = upstream.clone()
                output, loss = simplified_sparse_attention(*inputs, metadata, topk, loss_coeff=0.1)
                torch.cuda.synchronize(device)

                new_keys = set(fused._state_filtered) - previous_keys
                assert len(new_keys) == 1, "test must create exactly one cold filtered state"
                current_state = fused._state_filtered[next(iter(new_keys))]
                assert len(current_state["fallback_graphs"]) == 1
                current_graph = next(iter(current_state["fallback_graphs"].values()))
                # k1 is compiled per split factor; these few-block states all
                # split 8 ways, so growth reuses it too.
                current_compiled = tuple(current_state[name] for name in ("k1", "ps", "thrk", "pf", "p3", "fb_gather", "fb_scatter", "fb_score_compiled"))
                if state is None:
                    compiled_bundle = current_compiled
                else:
                    assert all(a is b for a, b in zip(current_compiled, compiled_bundle)), "capacity growth must reuse all compiled filtered kernels"
                    if iteration == 3:
                        assert current_state is not state and current_graph is not graph
                    else:
                        assert current_state is state and current_graph is graph
                state, graph = current_state, current_graph
                assert state["query_capacity"] >= tq and state["key_capacity"] >= tk
                torch.testing.assert_close(state["starts"][:tq], starts, rtol=0, atol=0)
                torch.testing.assert_close(state["ends"][:tq], torch.where(valid, ends, starts), rtol=0, atol=0)
                assert torch.count_nonzero(state["starts"][tq:]).item() == 0
                assert torch.count_nonzero(state["ends"][tq:]).item() == 0
                assert torch.count_nonzero(state["q"][tq : ((tq + 127) // 128) * 128]).item() == 0
                assert torch.count_nonzero(state["k"][tk : ((tk + 255) // 256) * 256]).item() == 0
                positions = torch.tensor([0, tk // 2, tk - 1], device=device)
                torch.testing.assert_close(state["k"][positions], inputs[4][positions], rtol=0, atol=0)

                repaired = int(state["flags"][0].item())
                repair_rows = state["flags"][1 : repaired + 1].long()
                assert 0 < repaired <= tq
                assert bool(((repair_rows >= 0) & (repair_rows < tq)).all().item())
                assert repair_rows.unique().numel() == repaired
                repair_mask = torch.zeros_like(valid)
                repair_mask[repair_rows] = True
                wide = valid & (ends - starts > state["tcap_buf"])
                assert bool(repair_mask[wide].all().item()), "wide tied rows must run exact repair"
                assert not bool(repair_mask[~valid].any().item())
                if iteration == 1:
                    assert repaired > 128, "replay must repair beyond the initial query extent"
                stats = tuple(int(value) for value in state["fallback_stats"].tolist())
                assert tuple(a - b for a, b in zip(stats, previous_stats)) == (1, repaired)
                previous_stats = stats

                modules = (state["fb_gather"], state["fb_score_compiled"], graph["fb_topk_launch"].launcher, state["fb_scatter"])
                for compiled in modules:
                    owner = compiled._ssa_cuda_preload
                    assert len(owner.libraries) >= 1 and all(owner.libraries)
                    assert device.index in owner.devices

                indices = observed["indices"]
                assert bool(((indices >= 0) & (indices < tk)).all().item())
                lengths = torch.where(valid, (ends - starts).clamp_max(topk), 0)
                torch.testing.assert_close(observed["lengths"], lengths, rtol=0, atol=0)
                slots = torch.arange(topk, device=device)[None, :]
                selected = slots < lengths[:, None]
                adjacent = slots[:, 1:] < lengths[:, None]
                assert bool((indices[:, 1:] > indices[:, :-1])[adjacent].all().item())
                assert bool(((indices >= starts[:, None]) & (indices < ends[:, None]))[selected].all().item())
                index_scores = (inputs[3].detach().float()[:, None, :] * inputs[4].detach()[indices.long()].float()).sum(-1) * fused.SCALE
                weights = (index_scores - index_scores.max(-1, keepdim=True).values).exp() * selected
                expected_predict = weights / weights.sum(-1, keepdim=True).clamp_min(1)
                torch.testing.assert_close(observed["predict"], expected_predict, rtol=2e-6, atol=1e-7)

                # Build every reference before backward, using a private upstream
                # copy. A mutation of the caller gradient cannot alter the oracle.
                references = []
                for query_row in (0, tq // 2, tq - 1):
                    if not bool(valid[query_row].item()):
                        continue
                    selected = indices[query_row, : int(lengths[query_row].item())].long()
                    q = inputs[0][query_row].detach().float()
                    k = inputs[1][selected, 0].detach().float()
                    v = inputs[2][selected, 0].detach().float()
                    probability = ((q @ k.t()) / 16.0).softmax(dim=-1)
                    expected_output = probability @ v
                    dprob = frozen_upstream[query_row].float() @ v.t()
                    dscore = probability * (dprob - (probability * dprob).sum(-1, keepdim=True))
                    expected_dq = (dscore @ k) / 16.0
                    references.append((query_row, expected_output, expected_dq))

                print(json.dumps(dict(event="filtered_before_backward", iteration=iteration, repaired_rows=repaired)), flush=True)
                torch.autograd.backward((output, loss), (upstream, torch.ones_like(loss)))
                torch.cuda.synchronize(device)
                torch.testing.assert_close(upstream, frozen_upstream, rtol=0, atol=0)
                for name, tensor in [("output", output), ("loss", loss)] + [(f"gradient{index}", tensor.grad) for index, tensor in enumerate(inputs)]:
                    assert tensor is not None and bool(torch.isfinite(tensor).all().item()), name
                assert torch.count_nonzero(output[~valid]).item() == 0
                assert torch.count_nonzero(inputs[0].grad[~valid]).item() == 0
                assert torch.count_nonzero(inputs[3].grad[~valid]).item() == 0
                errors = []
                for query_row, expected_output, expected_dq in references:
                    for name, actual, expected in (
                        ("output", output[query_row], expected_output),
                        ("dq", inputs[0].grad[query_row], expected_dq),
                    ):
                        norm = float(expected.norm().item())
                        error = float((actual.float() - expected).norm().item())
                        assert norm > 0 and error <= 2**-7 * norm, (name, query_row, error, norm)
                        errors.append(dict(row=query_row, quantity=name, relative_l2=error / norm))
                print(json.dumps(dict(event="filtered_passed", iteration=iteration, repaired_rows=repaired, probes=errors)), flush=True)
            # Enqueue a replay and then force capacity growth on the same
            # nondefault stream without a host synchronize between calls.
            # Keep both outputs alive: replacing the state cannot release
            # graph-owned memory while the first replay still uses it.
            queued = []
            for rows in (129, 513):
                keys = 131329
                qi = torch.zeros((rows, 128), dtype=torch.bfloat16, device=device)
                ki = torch.zeros((keys, 128), dtype=torch.bfloat16, device=device)
                begin = torch.full((rows,), 17, dtype=torch.int32, device=device)
                end = torch.full_like(begin, keys - 3)
                ids = torch.empty((rows, topk), dtype=torch.int32, device=device)
                prob = torch.empty((rows, topk), dtype=torch.float32, device=device)
                fused.run_fused_filtered(qi, ki, ids, prob, begin, end)
                queued.append((qi, ki, begin, end, ids, prob))
            assert any(old_state is state for _, old_state in fused._retired_filtered)
            torch.cuda.synchronize(device)
            for _, _, begin, end, ids, prob in queued:
                assert bool(((ids >= begin[:, None]) & (ids < end[:, None])).all().item())
                assert bool((ids[:, 1:] > ids[:, :-1]).all().item())
                torch.testing.assert_close(prob, torch.full_like(prob, 1 / topk), rtol=1e-6, atol=0)
            fused._release_retired_filtered()
            assert not any(old_state is state for _, old_state in fused._retired_filtered)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def test_reserved_state_at_its_maxima():
    """reserve_indexer storage ends exactly at the padded query tile and key chunk.

    Calls at the declared maxima reuse the reserved state and repair graph,
    including forced exact repair; run under memcheck for the slack-free tails.
    """
    from cudnn.ops import simplified_sparse_attention_reserve as reserve_indexer
    from cudnn.simplified_sparse_attention.cutedsl.kernels import indexer_forward_topk_fused_tcgen05 as fused

    tq, tk, topk = 200, 131072, 512
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(20261029)
    row = torch.arange(tq, device=device, dtype=torch.int32)
    starts, ends = (row % 3) * 1024, tk - (row % 5) * 2048
    k = torch.randn((tk, 128), device=device, dtype=torch.bfloat16, generator=generator) * 0.2
    q = torch.randn((tq, 128), device=device, dtype=torch.bfloat16, generator=generator) * 0.2
    reserved, plain = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)

    def select(q_idx, stream):
        ids = torch.empty((tq, topk), dtype=torch.int32, device=device)
        prob = torch.empty((tq, topk), dtype=torch.float32, device=device)
        with torch.cuda.stream(stream):
            fused.run_fused_filtered(q_idx, k, ids, prob, starts, ends)
        torch.cuda.synchronize(device)
        return ids, prob

    with torch.cuda.stream(reserved):
        reserve_indexer(tq, tk, topk)
    key = next(key for key in fused._state_filtered if key[-1] == reserved.cuda_stream)
    state = fused._state_filtered[key]
    graph = state["fallback_graphs"]["compact"]
    assert (state["query_capacity"], state["key_capacity"]) == (256, tk)
    # Distinct scores: slack-free storage selects what headroom storage selects.
    torch.testing.assert_close(select(q, reserved), select(q, plain), rtol=0, atol=0)
    # Zero indexer queries tie every score, so every row takes exact repair.
    ids, _ = select(torch.zeros_like(q), reserved)
    assert int(state["flags"][0].item()) == tq
    assert fused._state_filtered[key] is state and state["fallback_graphs"]["compact"] is graph
    assert bool(((ids >= starts[:, None]) & (ids < ends[:, None])).all().item())
    assert bool((ids[:, 1:] > ids[:, :-1]).all().item())
