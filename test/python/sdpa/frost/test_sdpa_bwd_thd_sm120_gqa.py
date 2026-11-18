# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native SM120 THD GQA through D256; deterministic backward remains unsupported."""

import pytest
import torch

from d128_thd_test_utils import GRADS, check_case, copy_case_storage, make_adapter, make_case, make_workspace, poison, variant_pack
from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from frost_test_utils import _SM, requires_dsl
from test_sdpa_bwd_thd_sm120 import _adapter_arguments, _graph_case

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM not in (120, 121), reason="requires SM120 or SM121")]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("stats_layout", ["TH1", "HN"])
@pytest.mark.parametrize("dims", [(64, 64), (128, 128), (192, 128), (256, 256), (248, 184), (8, 8), (120, 88)])
def test_wide_gqa_origins(dtype, stats_layout, dims):
    case = make_case(
        (65, 0, 33, 3),
        (17, 129, 0, 97),
        heads=6,
        kv_heads=2,
        d_qk=dims[0],
        d_v=dims[1],
        dtype=dtype,
        stats_layout=stats_layout,
        causal=True,
        bottom_right=True,
    )
    graph, _, pack, workspace = _graph_case(case)
    for _ in range(2):
        poison(case, workspace)
        graph.execute(pack, workspace.tensor)
        check_case(case, workspace)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dims,tiles", [((128, 128), (64, 32)), ((192, 128), (64, 32)), ((128, 128), (32, 32)), ((64, 64), (32, 32))])
def test_explicit_native_tiles(dtype, dims, tiles):
    from cudnn.sdpa.bwd.api_dsl_sm120_thd import SdpaBwdDslSm120Thd

    case = make_case((65, 0, 3), (17, 129, 97), heads=6, kv_heads=2, d_qk=dims[0], d_v=dims[1], dtype=dtype, causal=True)
    api = make_adapter(case, api_type=SdpaBwdDslSm120Thd, tile_m=tiles[0], tile_n=tiles[1])
    assert (api._params.tile_q, api._params.tile_kv) == tiles
    workspace = make_workspace(api.scratch_workspace_bytes())
    poison(case, workspace)
    api.execute(**_adapter_arguments(case), workspace=workspace.tensor)
    check_case(case, workspace)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_wide_gqa_rebind_and_capture(dtype):
    options = dict(
        heads=6, kv_heads=2, d_qk=256, d_v=256, dtype=dtype, stats_layout="HN", causal=True, bottom_right=True, envelope=(257, 257), capacities=(1024, 1024)
    )
    old = make_case((65, 0, 33), (17, 129, 0), **options)
    fresh = make_case((1, 129, 0), (97, 0, 65), seed=151, layout_shift=7, **options)
    changed = make_case((0, 193, 1), (17, 0, 129), seed=153, layout_shift=13, **options)
    graph, refs, pack, workspace = _graph_case(old)
    fresh_workspace = make_workspace(graph.get_workspace_size())
    graph.execute(pack, workspace.tensor)
    check_case(old, workspace)
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        graph.execute(pack, workspace.tensor)
    poison(fresh, fresh_workspace)
    graph.execute(variant_pack(fresh, refs), fresh_workspace.tensor)
    check_case(fresh, fresh_workspace)
    untouched = snapshot({r: fresh.tensors[r] for r in GRADS})
    copy_case_storage(old, changed)
    for _ in range(3):
        poison(old, workspace)
        capture.replay()
        check_case(old, workspace)
    assert_bitwise({r: fresh.tensors[r] for r in GRADS}, untouched, "retained replay changed fresh bindings")
    capture.reset()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_single_kv_tile_replay_and_batch(dtype):
    """Bounded byte coverage; multiple-KV-tile dQ atomics have no guarantee."""
    options = dict(heads=6, kv_heads=2, d_qk=256, d_v=256, dtype=dtype, causal=True, bottom_right=True)
    target = make_case((17,), (19,), **options).sequences[0]
    neighbor = make_case((17,), (19,), seed=151, **options).sequences[0]
    changed_neighbor = make_case((17,), (19,), seed=153, **options).sequences[0]
    expected = None
    for order in ([0], [0, 1, 0], [1, 0, 1], [0] * 129):
        sequences = [target if i == 0 else neighbor for i in order]
        case = make_case((17,) * len(order), (19,) * len(order), sequences=sequences, **options)
        graph, _, pack, workspace = _graph_case(case)
        target_index = order.index(0)

        def outputs(index):
            return {r: case.tensors[r][0, case.origins[r][index] : case.origins[r][index] + (17 if r == "dq" else 19)] for r in GRADS}

        def run():
            poison(case, workspace)
            graph.execute(pack, workspace.tensor)
            check_case(case, workspace)
            return outputs(target_index)

        got = assert_replays(run, repeats=3)
        if expected is None:
            expected = got
        for i, source in enumerate(order):
            if source == 0:
                assert_bitwise(outputs(i), expected, f"batch size {len(order)}, sequence {i}")
        if order == [0, 1, 0]:
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture):
                graph.execute(pack, workspace.tensor)
            changed = make_case((17,) * 3, (19,) * 3, sequences=[target, changed_neighbor, target], **options)
            copy_case_storage(case, changed)
            poison(case, workspace)
            capture.replay()
            check_case(case, workspace)
            assert_bitwise(outputs(0), expected, "changed neighbor replay")
            assert_bitwise(outputs(2), expected, "duplicate after changed neighbor replay")
            capture.reset()
    assert_golden(f"sdpa-sm120-thd-bwd-d256-gqa-single-kv-{dtype}", {r: target[r] for r in ("q", "k", "v", "o", "do", "stats")}, expected)
