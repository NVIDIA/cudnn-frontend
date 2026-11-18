# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small native query/KV tiles preserve paging, packed queries and split semantics."""

import cudnn
import pytest
import torch

from frost_test_utils import _SM, requires_dsl
from test_sdpa_fwd_paged_sm80 import test_native_paged_adapter_rebind_replay as _paged_probe

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 80, reason="requires native SM80")]


@pytest.mark.parametrize(
    "d,dv,m,n",
    [
        (48, 32, 16, 16),
        (64, 64, 64, 16),
        (128, 128, 16, 32),
        (128, 128, 32, 16),
        (192, 128, 32, 32),
        (192, 128, 128, 16),
        (256, 256, 16, 16),
        (248, 200, 32, 16),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_small_tiles_paged_rebind(d, dv, m, n, dtype, thd, graph_mode, cudnn_handle, monkeypatch):
    splits = 1 if thd else 3
    knobs = {cudnn.knob_type.TILE_M: m, cudnn.knob_type.TILE_N: n, cudnn.knob_type.SPLIT_KV: splits, cudnn.knob_type.SCHED_POLICY: 0}
    _paged_probe(
        (d, dv, 16, "HND", "NHD", "br_swa"),
        dtype,
        thd,
        graph_mode,
        cudnn_handle,
        monkeypatch,
        case_index=47,
        graph_knobs=knobs,
        has_sink=True,
        sink_endpoints=True,
        split_kv=splits,
        sink_stride=1,
        explicit_tiles=(m, n),
    )


@pytest.mark.parametrize("d,m,n", [(64, 16, 16), (128, 16, 32), (128, 32, 16), (256, 16, 16)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_small_tiles_determinism(d, m, n, dtype):
    from test_sdpa_fwd_split_sm80_determinism import test_split_forward_determinism

    test_split_forward_determinism(d, 3, dtype, tile_m=m, tile_n=n, golden_suffix=f"-m{m}-n{n}")


@pytest.mark.parametrize("m", [16, 128])
def test_dense_d192_copy_waves(m, cudnn_handle, monkeypatch):
    from test_sdpa_sm80_graph_tiles import test_public_graph_geometry_rebind_replay

    test_public_graph_geometry_rebind_replay(192, 128, m, 16, cudnn_handle, monkeypatch)
