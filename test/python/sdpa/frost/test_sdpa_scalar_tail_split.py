# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native half split plans mask scalar KV tails without length buffers."""

import pytest

pytestmark = pytest.mark.L0

_SHAPES = [(64, 64), (56, 48), (96, 80), (192, 128), (184, 120), (256, 256), (248, 224), (512, 512), (504, 496)]


@pytest.mark.parametrize("device_cc", [(10, 0), (10, 3), (10, 7)])
@pytest.mark.parametrize("d_shape", _SHAPES)
def test_half_scalar_tail_explicit_candidate_admission(device_cc, d_shape):
    import cudnn
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS, SdpaFwdKnobs, mismatch

    name = "sdpa_fwd_prefill_sm107" if device_cc == (10, 7) else "sdpa_fwd_prefill_sm100"
    caps = next(spec.capabilities for spec in ENGINE_SPECS if spec.name == name)
    # Rubin D256 dense split is the existing decode tile; scalar-tail
    # admission does not add a split implementation to its prefill body.
    sq = 1 if device_cc == (10, 7) and d_shape in ((256, 256), (248, 224)) else 64
    facts = ga.SdpaGraphFacts(b=2, h_q=4, h_kv=2, s_q=sq, s_kv=257, d_qk=d_shape[0], d_v=d_shape[1], dtype=cudnn.data_type.BFLOAT16, device_cc=device_cc)
    assert mismatch(caps, facts, SdpaFwdKnobs(split_kv=4, tile_m=128, tile_n=128, cga=2, pack_gqa=False)) is None


@pytest.mark.parametrize("d_shape", _SHAPES)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("splits", [4, 16])
def test_half_scalar_tail_graph_overrides_and_replay(d_shape, dtype, splits, monkeypatch):
    import cudnn
    import torch
    from test_sdpa_fwd_dsl_sm100 import test_dsl_sm100_kv_tail_mask_takes_the_native_prepared_launch as run_tail

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("requires native SM100/SM103/SM107")
    dtype = getattr(torch, dtype)
    run_tail(
        dtype,
        dtype,
        monkeypatch,
        splits,
        2,
        d_shape=d_shape,
        stats_output=splits == 4,
        knob_overrides={cudnn.knob_type.TILE_M: 128, cudnn.knob_type.TILE_N: 128, cudnn.knob_type.PACK_GQA: 0},
        query_length=1 if torch.cuda.get_device_capability() == (10, 7) and d_shape in ((256, 256), (248, 224)) else 64,
    )
