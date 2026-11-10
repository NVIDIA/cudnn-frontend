# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native D64 page loaders for dense and ragged Q on SM107."""

import pytest
import torch


@pytest.mark.parametrize(
    "cga,packed,split,tile_n",
    [
        (1, False, 1, 128),
        (2, False, 1, 128),
        (1, True, 1, 128),
        (2, True, 1, 128),
        (1, False, 4, 128),
        (2, False, 4, 128),
        (1, True, 4, 128),
        (2, True, 4, 128),
        (1, False, 1, 64),
        (2, False, 1, 64),
        (1, True, 1, 64),
        (2, True, 1, 64),
        (1, False, 4, 64),
        (2, False, 4, 64),
        (1, True, 4, 64),
        (2, True, 4, 64),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dense(cga, packed, split, tile_n, dtype, monkeypatch):
    from native_d128_dense_paged_sm107_checks import test_dense_replay

    test_dense_replay(cga, packed, split, tile_n, dtype, monkeypatch, head_dim=64)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    "cga,packed,layout,page,tile_n,policy",
    [
        (1, False, "mixed", 8, 128, 0),
        (2, True, "mixed", 16, 128, 0),
        (1, True, "hnd", 64, 64, 1),
        (2, False, "nhd", 256, 64, 1),
    ],
)
def test_thd(dtype_name, cga, packed, layout, page, tile_n, policy, monkeypatch):
    from sm107_half_test_utils import _rubin_only
    from test_sdpa_fwd_paged_thd_origins import test_paged_thd_origins_rebind_and_capture
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    _rubin_only()
    original = SdpaFwdDslSm100.compile

    def compile_native(api):
        original(api)
        assert api.flavor == (64, 64)
        assert api._k_mod.CFG.TILE_K == api._k_mod.CFG.TILE_O == 64
        assert api._k_mod.CFG.PAGED_KV and api._k_mod.CFG.THD_VARLEN
        assert api._k_mod.__file__.endswith("sm107/prefill_d128_f16.py")

    monkeypatch.setattr(SdpaFwdDslSm100, "compile", compile_native)
    test_paged_thd_origins_rebind_and_capture(
        64,
        dtype_name,
        layout,
        "head" if packed else "token",
        cga,
        1,
        packed,
        page_size=page,
        window_left=73,
        sink=-120.0,
        tile_n=tile_n,
        sched_policy=policy,
        cpu_reference=True,
    )


@pytest.mark.parametrize(
    "dtype_name,cga,split,tile_n",
    [
        ("bfloat16", 1, 1, 64),
        ("float16", 2, 1, 128),
        ("bfloat16", 1, 4, 128),
        ("float16", 2, 4, 64),
    ],
)
def test_dense_determinism(dtype_name, cga, split, tile_n):
    from native_d128_dense_paged_determinism import test_replay_batch_and_golden

    test_replay_batch_and_golden(dtype_name, dtype_name, cga, split, tile_n, head_dim=64)


@pytest.mark.parametrize(
    "dims,page,sq,cga,split,packed,policy,empty",
    [
        ((64, 64), 8, 1, 1, 8, True, 0, False),
        ((64, 64), 256, 129, 2, 1, False, 1, False),
        ((48, 32), 32, 2, 1, 4, True, 0, False),
        ((64, 48), 128, 257, 2, 1, False, 0, False),
        ((64, 64), 8, 1, 2, 4, True, 0, True),
        ((64, 64), 1024, 2, 1, 1, False, 1, False),
        ((32, 32), 16, 1, 1, 1, False, 0, False),
    ],
)
def test_boundaries(dims, page, sq, cga, split, packed, policy, empty, monkeypatch):
    from native_d128_dense_paged_sm107_checks import test_page_boundaries

    test_page_boundaries(dims, page, sq, cga, split, packed, policy, empty, monkeypatch)


def test_admission():
    from test_sdpa_fwd_dsl_sm107 import _caps, _f16_facts
    from cudnn.sdpa.fwd import engines

    caps = _caps("sdpa_fwd_prefill_sm107")
    assert (64, 64) in caps.paged_d_shapes
    for thd in (False, True):
        facts = _f16_facts(d_qk=64, d_v=64, thd=thd, padded=True, has_paged_kv=True, page_size=16)
        for cga in (1, 2):
            for packed in (False, True):
                for n in (64, 128):
                    for split in (1, 4):
                        knobs = engines.SdpaFwdKnobs(cga=cga, pack_gqa=packed, tile_n=n, split_kv=split)
                        assert engines._selected_d_shape(caps, facts, knobs) == (64, 64)
                        reason = engines.mismatch(caps, facts, knobs)
                        if thd and split > 1:
                            assert reason is not None, (thd, knobs)
                        else:
                            assert reason is None, (thd, knobs, reason)
