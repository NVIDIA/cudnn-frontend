# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native SM80 packed-head attention preserves the ordinary graph contract."""

import pytest
import torch
import cudnn

from frost_test_utils import _SM, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 80, reason="requires native SM80")]


@pytest.mark.parametrize("dims", [(128, 128), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("stats_log2", [False, True])
def test_dense_pack_split_partial_initialization(dims, dtype, stats_log2):
    from test_sdpa_fwd_split_sm80 import test_dense_split_partial_initialization

    test_dense_split_partial_initialization(dims, dtype, stats_log2, pack_gqa=True)


@pytest.mark.parametrize(
    "d,dv,g,m,n",
    [
        (48, 32, 3, 16, 16),
        (64, 64, 8, 16, 32),
        (128, 128, 12, 32, 32),
        (192, 128, 3, 32, 32),
        (256, 256, 8, 16, 32),
        (512, 512, 3, 32, 16),
        (512, 512, 128, 32, 16),
        (576, 512, 8, 32, 16),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("graph_mode", [False, True])
@pytest.mark.parametrize("splits", [1, 3])
def test_paged_pack_rebind(d, dv, g, m, n, dtype, graph_mode, splits, cudnn_handle, monkeypatch, single_q=False):
    from cudnn.sdpa.fwd import api_dsl
    from test_sdpa_fwd_paged_sm80 import test_native_paged_adapter_rebind_replay

    seen = []
    original = api_dsl._sm80_load_kernel_module

    def record(flavor, params):
        seen.append(params.pack_g)
        return original(flavor, params)

    monkeypatch.setattr(api_dsl, "_sm80_load_kernel_module", record)
    knobs = {
        cudnn.knob_type.PACK_GQA: 1,
        cudnn.knob_type.TILE_M: m,
        cudnn.knob_type.TILE_N: n,
        cudnn.knob_type.SPLIT_KV: splits,
        cudnn.knob_type.SCHED_POLICY: 0,
    }
    test_native_paged_adapter_rebind_replay(
        (d, dv, 16, "HND", "NHD", "br_swa"),
        dtype,
        False,
        graph_mode,
        cudnn_handle,
        monkeypatch,
        case_index=53,
        graph_knobs=knobs,
        has_sink=True,
        sink_endpoints=True,
        split_kv=splits,
        sink_stride=1,
        explicit_tiles=(m, n),
        pack_gqa=True,
        heads=(g * 2, 2),
        single_q=single_q,
    )
    assert seen and all(group == g for group in seen), seen


@pytest.mark.parametrize("graph_mode", [False, True])
@pytest.mark.parametrize("splits", [1, 3])
def test_large_head_group_single_query(graph_mode, splits, cudnn_handle, monkeypatch):
    test_paged_pack_rebind(512, 512, 128, 32, 16, torch.float16, graph_mode, splits, cudnn_handle, monkeypatch, single_q=True)


@pytest.mark.parametrize("d,dv", [(192, 128), (256, 256), (512, 512)])
@pytest.mark.parametrize("splits", [1, 3])
def test_dense_pack_bias_replay(d, dv, splits):
    from test_sdpa_prepared_sm80 import _case, _check

    knobs = {
        cudnn.knob_type.PACK_GQA: 1,
        cudnn.knob_type.TILE_M: 32,
        cudnn.knob_type.TILE_N: 16,
        cudnn.knob_type.SPLIT_KV: splits,
        cudnn.knob_type.SCHED_POLICY: 1,
    }
    case = _case(d, dv, layout="gapped", features=True, causal=True, heads=(6, 2), graph_knobs=knobs)
    assert case.graph.get_engine_and_knobs_at_index(case.graph._plan_index)[1] == knobs
    case.graph.execute(case.pack, case.workspace)
    _check(case)
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            case.graph.execute(case.pack, case.workspace)
        case.bufs["bias"].mul_(-0.5)
        case.bufs["sink"].add_(2.0)
        case.bufs["v"].mul_(-0.7)
        case.bufs["o"].fill_(float("nan"))
        capture.replay()
        _check(case)
    finally:
        capture.reset()


@pytest.mark.parametrize("d,splits,g,m,n", [(64, 1, 3, 16, 16), (128, 3, 12, 32, 32), (128, 3, 128, 16, 16), (256, 1, 8, 16, 32), (512, 17, 3, 32, 16)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=lambda dtype: str(dtype).split(".")[-1])
def test_pack_forward_determinism(d, splits, g, m, n, dtype):
    from test_sdpa_fwd_split_sm80_determinism import test_split_forward_determinism

    test_split_forward_determinism(d, splits, dtype, tile_m=m, tile_n=n, golden_suffix=f"-pack{g}-m{m}-n{n}", pack_gqa=True, heads=(g * 2, 2))


def test_standalone_thd_pack_declines(cudnn_handle, monkeypatch):
    from test_sdpa_fwd_paged_sm80 import test_native_paged_adapter_rebind_replay

    with pytest.raises(NotImplementedError, match="PackGQA requires dense Q"):
        test_native_paged_adapter_rebind_replay(
            (128, 128, 16, "HND", "NHD", "br_swa"),
            torch.bfloat16,
            True,
            False,
            cudnn_handle,
            monkeypatch,
            case_index=61,
            pack_gqa=True,
        )
