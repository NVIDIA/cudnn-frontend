# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for MiniMax Lightning Indexer exact decode."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from test_utils import torch_fork_set_rng

BLOCK_SIZE = 128
TOP_K = 16
HEADS = 4
HEAD_DIM = 128


def _require_cuda():
    try:
        from cudnn import NSA
    except ImportError:
        pytest.skip("cudnn CuTe DSL optional dependencies are not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Lightning Indexer requires SM80+")
    return NSA


def _inputs(
    batch: int,
    k_capacity: int,
    positions: list[int],
    seed: int = 0,
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        (batch, 1, HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k = torch.randn(
        (batch, k_capacity, 1, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    position_ids = torch.tensor(positions, dtype=torch.int64, device="cuda").view(batch, 1)
    return q, k, position_ids


def _outputs(batch: int):
    indices = torch.empty(
        (batch, 1, HEADS, TOP_K),
        dtype=torch.int32,
        device="cuda",
    ).transpose(1, 2)
    counts = torch.empty(
        (batch, 1, HEADS),
        dtype=torch.int32,
        device="cuda",
    ).transpose(1, 2)
    return indices, counts


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
):
    batch = q.shape[0]
    k_len = k.shape[1]
    num_blocks = (k_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    scores = torch.matmul(
        q.transpose(1, 2).float(),
        k.transpose(1, 2).float().transpose(-1, -2),
    )
    key_positions = torch.arange(k_len, device=q.device)
    scores.masked_fill_(
        key_positions[None, None, None, :] > position_ids[:, None, :, None],
        float("-inf"),
    )
    pad = num_blocks * BLOCK_SIZE - k_len
    if pad:
        scores = torch.nn.functional.pad(scores, (0, pad), value=float("-inf"))
    block_scores = scores.view(batch, HEADS, 1, num_blocks, BLOCK_SIZE).amax(-1)
    current = (position_ids // BLOCK_SIZE)[:, None, :, None].expand(-1, HEADS, -1, -1)
    block_scores.scatter_(-1, current, float("inf"))

    # Define exact-score ties toward lower block ids. Output slot order is not
    # public, but a deterministic tie rule makes the selected set testable.
    order = torch.argsort(block_scores, dim=-1, descending=True, stable=True)
    selected = min(TOP_K, num_blocks)
    chosen = order[..., :selected]
    chosen_scores = torch.gather(block_scores, -1, chosen)
    result = torch.full(
        (batch, HEADS, 1, TOP_K),
        -1,
        dtype=torch.int32,
        device=q.device,
    )
    result[..., :selected] = chosen.masked_fill(chosen_scores == float("-inf"), -1).to(torch.int32)
    counts = (result >= 0).sum(-1, dtype=torch.int32)
    return result, counts


def _canonical_sets(indices: torch.Tensor) -> torch.Tensor:
    sentinel = torch.iinfo(torch.int32).max
    return torch.sort(indices.masked_fill(indices < 0, sentinel), dim=-1).values


def _check(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
) -> None:
    expected_indices, expected_counts = _reference(q, k, position_ids)
    torch.testing.assert_close(counts, expected_counts, rtol=0, atol=0)
    torch.testing.assert_close(
        _canonical_sets(indices),
        _canonical_sets(expected_indices),
        rtol=0,
        atol=0,
    )
    slots = torch.arange(TOP_K, device=indices.device)
    valid = slots < counts[..., None]
    assert bool(torch.all(indices.masked_select(valid) >= 0))
    assert bool(torch.all(indices.masked_select(~valid) == -1))


@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize(
    "k_capacity,position",
    [
        pytest.param(1, 0, marks=pytest.mark.L1),
        pytest.param(127, 126, marks=pytest.mark.L1),
        pytest.param(128, 127, marks=pytest.mark.L1),
        pytest.param(129, 128, marks=pytest.mark.L0),
        pytest.param(2047, 2046, marks=pytest.mark.L1),
        pytest.param(2048, 255, marks=pytest.mark.L1),
        pytest.param(2049, 255, marks=pytest.mark.L1),
        pytest.param(2177, 2176, marks=pytest.mark.L1),
        pytest.param(4097, 4096, marks=pytest.mark.L0),
    ],
)
def test_lightning_indexer_decode_reference(k_capacity, position):
    NSA = _require_cuda()
    q, k, position_ids = _inputs(1, k_capacity, [position], seed=position)
    result = NSA.lightning_indexer(q, k, position_ids)
    _check(
        q,
        k,
        position_ids,
        result["block_indices"],
        result["block_counts"],
    )


@pytest.mark.L1
@torch_fork_set_rng(seed=11)
def test_lightning_indexer_static_cache_batch_positions():
    NSA = _require_cuda()
    q, k, position_ids = _inputs(3, 4097, [255, 2048, 4096], seed=11)
    # Make the unfilled/static-cache tail attractive. Explicit positions must
    # still exclude it rather than inferring the current block from capacity.
    k[0, 256:] = 32
    k[1, 2049:] = 32
    result = NSA.lightning_indexer(q, k, position_ids)
    _check(
        q,
        k,
        position_ids,
        result["block_indices"],
        result["block_counts"],
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=19)
def test_lightning_indexer_accepts_official_transpose_views():
    NSA = _require_cuda()
    # MiniMax produces BHSD after norm/RoPE. Transposing to the public BSHD
    # convention must remain a zero-copy view, including size-one odd strides.
    q_bhsd = torch.randn(
        (2, HEADS, 1, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    k_bhsd = torch.randn(
        (2, 1, 4097, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    q = q_bhsd.transpose(1, 2)
    k = k_bhsd.transpose(1, 2)
    assert q.untyped_storage().data_ptr() == q_bhsd.untyped_storage().data_ptr()
    assert k.untyped_storage().data_ptr() == k_bhsd.untyped_storage().data_ptr()
    position_ids = torch.tensor([[4096]], dtype=torch.int64, device="cuda").expand(2, -1)
    assert position_ids.stride() == (0, 1)
    result = NSA.lightning_indexer(q, k, position_ids)
    _check(
        q,
        k,
        position_ids,
        result["block_indices"],
        result["block_counts"],
    )


@pytest.mark.L1
def test_lightning_indexer_exact_ties_choose_low_blocks():
    NSA = _require_cuda()
    q = torch.ones(
        (1, 1, HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    k = torch.zeros(
        (1, 2177, 1, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    # Every completed block has the same exact maximum.
    position_ids = torch.tensor([[2176]], dtype=torch.int64, device="cuda")
    result = NSA.lightning_indexer(q, k, position_ids)
    expected = torch.tensor(
        [17, *range(15)],
        dtype=torch.int32,
        device="cuda",
    )
    torch.testing.assert_close(result["block_indices"][0, 0, 0], expected, rtol=0, atol=0)
    assert int(result["block_counts"][0, 0, 0]) == TOP_K


@pytest.mark.L1
def test_lightning_indexer_all_negative_block_scores():
    NSA = _require_cuda()
    q = torch.ones(
        (1, 1, HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    k = torch.empty(
        (1, 2177, 1, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    for block in range(17):
        k[:, block * BLOCK_SIZE : (block + 1) * BLOCK_SIZE] = -(block + 1)
    k[:, 2176:] = 64
    position_ids = torch.tensor([[2176]], dtype=torch.int64, device="cuda")
    result = NSA.lightning_indexer(q, k, position_ids)
    expected = torch.tensor(
        [17, *range(15)],
        dtype=torch.int32,
        device="cuda",
    )
    torch.testing.assert_close(result["block_indices"][0, 0, 0], expected, rtol=0, atol=0)


@pytest.mark.L1
@torch_fork_set_rng(seed=29)
def test_lightning_indexer_prepared_streams_and_graph():
    NSA = _require_cuda()
    q0, k0, p0 = _inputs(1, 4097, [4096], seed=29)
    q1, k1, p1 = _inputs(1, 4097, [2048], seed=31)
    out0, count0 = _outputs(1)
    out1, count1 = _outputs(1)
    plan = NSA.LightningIndexer(q0, k0, p0, out0, count0)
    assert plan.check_support()
    plan.compile()
    # Finish input production first, then deliberately initialize workspaces on
    # the default stream and consume them on two fresh streams without an
    # explicit wait. The workspace's readiness event supplies that dependency.
    torch.cuda.synchronize()
    default = torch.cuda.current_stream()
    producer = torch.cuda.Stream()
    initialization_gate = torch.cuda.Event()
    with torch.cuda.stream(producer):
        torch.cuda._sleep(20_000_000)
        initialization_gate.record()
    default.wait_event(initialization_gate)
    workspace0 = plan.make_workspace()
    workspace1 = plan.make_workspace()
    assert workspace0 is not None and workspace1 is not None

    stream0 = torch.cuda.Stream()
    stream1 = torch.cuda.Stream()
    with torch.cuda.stream(stream0):
        plan.execute(
            q0,
            k0,
            p0,
            out0,
            count0,
            workspace0,
            current_stream=stream0.cuda_stream,
        )
    with torch.cuda.stream(stream1):
        plan.execute(
            q1,
            k1,
            p1,
            out1,
            count1,
            workspace1,
            current_stream=stream1.cuda_stream,
        )
    default.wait_stream(stream0)
    default.wait_stream(stream1)
    _check(q0, k0, p0, out0, count0)
    _check(q1, k1, p1, out1, count1)

    graph = torch.cuda.CUDAGraph()
    plan.execute(q0, k0, p0, out0, count0, workspace0)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        plan.execute(q0, k0, p0, out0, count0, workspace0)
    q0.normal_()
    k0.normal_()
    graph.replay()
    torch.cuda.synchronize()
    _check(q0, k0, p0, out0, count0)


@pytest.mark.L1
@torch_fork_set_rng(seed=37)
def test_lightning_indexer_initializes_caller_workspace():
    NSA = _require_cuda()
    q, k, position_ids = _inputs(2, 4097, [2048, 4096], seed=37)
    indices, counts = _outputs(2)
    plan = NSA.LightningIndexer(q, k, position_ids, indices, counts)
    assert plan.check_support()
    plan.compile()
    workspace = torch.full(
        (plan.workspace_size // 4,),
        17,
        dtype=torch.int32,
        device="cuda",
    )
    plan.initialize_workspace(workspace)
    plan.execute(q, k, position_ids, indices, counts, workspace)
    _check(q, k, position_ids, indices, counts)


@pytest.mark.L1
@torch_fork_set_rng(seed=39)
def test_lightning_indexer_wrapper_accepts_raw_workspace_on_stream():
    NSA = _require_cuda()
    q, k, position_ids = _inputs(2, 4097, [2048, 4096], seed=39)
    indices, counts = _outputs(2)
    sizing_plan = NSA.LightningIndexer(q, k, position_ids, indices, counts)
    assert sizing_plan.check_support()
    workspace = torch.empty(
        sizing_plan.workspace_size // 4,
        dtype=torch.int32,
        device="cuda",
    )
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    result = NSA.lightning_indexer(
        q,
        k,
        position_ids,
        workspace=workspace,
        stream=stream.cuda_stream,
    )
    torch.cuda.current_stream().wait_stream(stream)
    _check(
        q,
        k,
        position_ids,
        result["block_indices"],
        result["block_counts"],
    )
    repeated = NSA.lightning_indexer(
        q,
        k,
        position_ids,
        workspace=workspace,
        stream=stream.cuda_stream,
    )
    torch.cuda.current_stream().wait_stream(stream)
    _check(
        q,
        k,
        position_ids,
        repeated["block_indices"],
        repeated["block_counts"],
    )


@pytest.mark.L1
@pytest.mark.parametrize(
    "k_capacity,position",
    [(129, -1), (129, 129), (4097, -1), (4097, 4097)],
)
def test_lightning_indexer_invalid_position_returns_empty(k_capacity, position):
    NSA = _require_cuda()
    q, k, position_ids = _inputs(1, k_capacity, [position], seed=41)
    result = NSA.lightning_indexer(q, k, position_ids)
    assert int(result["block_counts"].max()) == 0
    assert bool(torch.all(result["block_indices"] == -1))


@pytest.mark.L0
def test_lightning_indexer_metadata_support_validation():
    NSA = _require_cuda()
    q, k, position_ids = _inputs(1, 4097, [4096])
    indices, counts = _outputs(1)
    tensor_plan = NSA.LightningIndexer(q, k, position_ids, indices, counts)
    metadata = (
        tensor_plan.q_desc,
        tensor_plan.k_desc,
        tensor_plan.position_desc,
        tensor_plan.indices_desc,
        tensor_plan.counts_desc,
    )
    assert NSA.LightningIndexer(*metadata).check_support()

    with pytest.raises(ValueError, match="q dtype mismatch"):
        NSA.LightningIndexer(replace(tensor_plan.q_desc, dtype=torch.float16), *metadata[1:]).check_support()
    with pytest.raises(ValueError, match="batch size must be positive"):
        NSA.LightningIndexer(
            replace(tensor_plan.q_desc, shape=(0, 1, HEADS, HEAD_DIM)),
            *metadata[1:],
        ).check_support()
    with pytest.raises(ValueError, match="batch size must be <= 65535"):
        NSA.LightningIndexer(
            replace(tensor_plan.q_desc, shape=(65536, 1, HEADS, HEAD_DIM)),
            *metadata[1:],
        ).check_support()
    with pytest.raises(ValueError, match="S_k must be <= 32768"):
        bad_k = replace(
            tensor_plan.k_desc,
            shape=(1, 32769, 1, HEAD_DIM),
            stride=(32769 * HEAD_DIM, HEAD_DIM, HEAD_DIM, 1),
        )
        NSA.LightningIndexer(metadata[0], bad_k, *metadata[2:]).check_support()
    with pytest.raises(ValueError, match="batch-broadcast"):
        bad_position = replace(
            tensor_plan.position_desc,
            stride=(2, 1),
            stride_order=(1, 0),
        )
        NSA.LightningIndexer(metadata[0], metadata[1], bad_position, *metadata[3:]).check_support()
    with pytest.raises(ValueError, match="must share a device"):
        other_device_k = replace(tensor_plan.k_desc, device=torch.device("cuda:1"))
        NSA.LightningIndexer(metadata[0], other_device_k, *metadata[2:]).check_support()
    with pytest.raises(ValueError, match="requires CUDA tensors"):
        cpu_metadata = tuple(replace(desc, device=torch.device("cpu")) for desc in metadata)
        NSA.LightningIndexer(*cpu_metadata).check_support()


@pytest.mark.L0
def test_lightning_indexer_rejects_misaligned_q_at_execute():
    NSA = _require_cuda()
    sample_q = torch.empty(
        (1, 1, HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    storage = torch.empty(
        1 + HEADS * HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    q = storage[1:].view(1, 1, HEADS, HEAD_DIM)
    k = torch.empty(
        (1, 2048, 1, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    position_ids = torch.tensor([[2047]], dtype=torch.int64, device="cuda")
    indices, counts = _outputs(1)
    plan = NSA.LightningIndexer(sample_q, k, position_ids, indices, counts)
    assert plan.check_support()
    plan.compile()
    with pytest.raises(ValueError, match="q must be 16-byte aligned"):
        plan.execute(q, k, position_ids, indices, counts)
