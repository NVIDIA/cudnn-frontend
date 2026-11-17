# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
The unified SDPA forward must serve dropout together with generate_stats=True.

That combination is the training forward: the backward needs Stats, and it regenerates
the dropout mask from the same seed and offset. So the forward's mask has to be the one
any backward engine will regenerate, whichever forward engine produced O and Stats.
cuDNN guarantees that for the unified forward from 9.28.0; on older versions the frontend
declines the combination.

The test therefore checks the unified forward three ways:
  * against a torch reference that applies the mask the engine dumped;
  * against the composite forward with the same seed and offset (same mask, same O and Stats);
  * against itself without Stats (generating Stats must not change the mask or O).
"""

import math

import cudnn
import pytest
import torch

from sdpa.fp16_ref import compute_ref

DROPOUT_PROB = 0.1
UNIFIED_DROPOUT_WITH_STATS_MIN_CUDNN_VER = 92800
# (s_q, s_kv, d): a regular shape, the decode shape, and sequence lengths that end inside a tile. The last two are
# shapes where older cuDNN versions drew a mask that differs from the composite forward's.
SHAPES = {"regular": (192, 256, 64), "decode": (1, 256, 64), "ragged_tiles": (33, 77, 128)}
SEED, OFFSET = 123456, 7


def _cudnn_dtype(torch_dtype):
    return {torch.float16: cudnn.data_type.HALF, torch.bfloat16: cudnn.data_type.BFLOAT16}[torch_dtype]


def _run_sdpa(cudnn_handle, q, k, v, attn_scale, right_bound, implementation, generate_stats):
    b, h_q, s_q, _ = q.shape
    _, _, s_kv, d_v = v.shape
    dtype = _cudnn_dtype(q.dtype)

    graph = cudnn.pygraph(
        io_data_type=dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )
    q_t = graph.tensor_like(q)
    k_t = graph.tensor_like(k)
    v_t = graph.tensor_like(v)
    seed_t = graph.tensor(dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
    offset_t = graph.tensor(dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
    mask_t = graph.tensor(dim=(b, h_q, s_q, s_kv), stride=(h_q * s_q * s_kv, s_q * s_kv, s_kv, 1), data_type=cudnn.data_type.FLOAT)

    o_t, stats_t = graph.sdpa(
        name="sdpa_dropout_stats",
        q=q_t,
        k=k_t,
        v=v_t,
        generate_stats=generate_stats,
        attn_scale=attn_scale,
        diagonal_band_right_bound=right_bound,
        dropout=(DROPOUT_PROB, seed_t, offset_t),
        rng_dump=mask_t,
        implementation=implementation,
    )
    o = torch.empty(b, h_q, s_q, d_v, dtype=q.dtype, device="cuda")
    stats = torch.empty(b, h_q, s_q, 1, dtype=torch.float32, device="cuda")
    mask = torch.zeros(b, h_q, s_q, s_kv, dtype=torch.float32, device="cuda")
    seed = torch.full((1, 1, 1, 1), SEED, dtype=torch.int64, device="cuda")
    offset = torch.full((1, 1, 1, 1), OFFSET, dtype=torch.int64, device="cuda")
    o_t.set_output(True).set_dim(o.shape).set_stride(o.stride())
    variant_pack = {q_t: q, k_t: k, v_t: v, o_t: o, seed_t: seed, offset_t: offset, mask_t: mask}
    if generate_stats:
        stats_t.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim(stats.shape).set_stride(stats.stride())
        variant_pack[stats_t] = stats

    try:
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()
    except cudnn.cudnnGraphNotSupportedError:
        return None

    workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda")
    graph.execute(variant_pack, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()
    return o, (stats if generate_stats else None), mask


@pytest.mark.L0
@pytest.mark.parametrize("right_bound", [None, 0], ids=["no_mask", "causal"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", list(SHAPES), ids=list(SHAPES))
def test_sdpa_unified_dropout_with_stats(cudnn_handle, shape, dtype, right_bound):
    b, h_q, h_kv = 2, 4, 2
    s_q, s_kv, d = SHAPES[shape]
    attn_scale = 1.0 / math.sqrt(d)
    rng = torch.Generator(device="cuda").manual_seed(0xD0)
    q = torch.randn(b, h_q, s_q, d, dtype=dtype, device="cuda", generator=rng)
    k = torch.randn(b, h_kv, s_kv, d, dtype=dtype, device="cuda", generator=rng)
    v = torch.randn(b, h_kv, s_kv, d, dtype=dtype, device="cuda", generator=rng)

    unified = cudnn.attention_implementation.UNIFIED
    # Dropout without Stats is the baseline the unified forward has served since cuDNN 9.22; where even that is
    # declined (older cuDNN, or a device whose unified forward has no dropout), there is nothing to compare.
    no_stats = _run_sdpa(cudnn_handle, q, k, v, attn_scale, right_bound, unified, generate_stats=False)
    if no_stats is None:
        pytest.skip("unified SDPA forward does not serve dropout on this device / cuDNN version")

    with_stats = _run_sdpa(cudnn_handle, q, k, v, attn_scale, right_bound, unified, generate_stats=True)
    if cudnn.backend_version() < UNIFIED_DROPOUT_WITH_STATS_MIN_CUDNN_VER:
        assert with_stats is None, "dropout with generate_stats=True on the unified forward needs cuDNN 9.28.0"
        pytest.skip("dropout with generate_stats=True on the unified forward requires cuDNN 9.28.0")
    assert with_stats is not None, "the unified SDPA forward serves dropout but declined dropout with generate_stats=True"
    o, stats, mask = with_stats

    # The dump is the 0/1 keep mask. Count kept entries where the score is not masked out by the band.
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    allowed = torch.ones(s_q, s_kv, dtype=torch.bool, device="cuda")
    if right_bound is not None:
        allowed = torch.tril(allowed, diagonal=right_bound)
    kept = mask[:, :, allowed]
    keep_fraction = kept.mean().item()
    # 0.02 for the large shapes; four standard deviations where the allowed region is small (decode, causal).
    tolerance = max(0.02, 4.0 * math.sqrt(DROPOUT_PROB * (1.0 - DROPOUT_PROB) / kept.numel()))
    assert abs(keep_fraction - (1.0 - DROPOUT_PROB)) < tolerance, f"keep fraction {keep_fraction:.4f}"

    # Reference with the engine's own mask. Stats are the softmax statistics, which dropout does not touch.
    o_ref, stats_ref, _, _ = compute_ref(
        q.float(),
        k.float(),
        v.float(),
        attn_scale=attn_scale,
        right_bound=right_bound,
        dropout_prob=DROPOUT_PROB,
        dropout_mask=mask,
        torch_type=dtype,
    )
    torch.testing.assert_close(o.float(), o_ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(stats, stats_ref, rtol=2e-2, atol=2e-2)

    # Generating Stats must not change the mask or the output.
    o_no_stats, _, mask_no_stats = no_stats
    assert torch.equal(mask, mask_no_stats), "the dropout mask depends on generate_stats"
    torch.testing.assert_close(o, o_no_stats, rtol=1e-2, atol=1e-2)

    # Same seed and offset on the composite forward: the mask must be identical, since a backward on either engine
    # regenerates it from seed and offset alone.
    composite = _run_sdpa(cudnn_handle, q, k, v, attn_scale, right_bound, cudnn.attention_implementation.COMPOSITE, generate_stats=True)
    if composite is not None:
        o_comp, stats_comp, mask_comp = composite
        assert torch.equal(mask, mask_comp), "unified and composite forwards generated different dropout masks"
        torch.testing.assert_close(o, o_comp, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(stats, stats_comp, rtol=1e-3, atol=1e-3)
