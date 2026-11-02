# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native SM107 D128 PackGQA replay matrix, kept outside the L0 smoke tier."""

import math

import pytest

from frost_test_utils import requires_dsl
from sm107_half_test_utils import _dense_mask, _half_oracle, _half_prefold_api, _rubin_only

pytestmark = [pytest.mark.L1, requires_dsl]


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
@pytest.mark.parametrize("sched_policy", [0, 1, 2])
@pytest.mark.parametrize("with_stats", [False, True])
@pytest.mark.parametrize(
    "geometry",
    [
        "mha",
        "split",
        "keyless",
        "keyless_no_sink",
        "window",
        "envelope",
        "asymmetric",
        "partial6",
        "partial12",
        "partial_keyless",
        "partial_window",
        "partial_envelope",
        "partial_asymmetric",
        "partial192",
    ],
)
def test_native_d128_packing_dense_replay(dtype_name, sched_policy, with_stats, geometry):
    """Native PackGQA covers every scheduler, per-head sinks, keyless rows and FP32 partials."""
    _check_native_d128_packing_dense_replay(dtype_name, sched_policy, with_stats, geometry)


def _check_native_d128_packing_dense_replay(dtype_name, sched_policy, with_stats, geometry, *, tile_n=None, head_dims=None, cga=None):
    import torch
    from cuda.bindings import driver as cuda

    _rubin_only()
    group, sq, sk, split, causal, br, window, sink = {
        "mha": (1, 33, 128, 1, False, False, None, False),
        "split": (4, 513, 768, 2, False, False, None, False),
        "keyless": (8, 257, 129, 1, True, True, None, True),
        "keyless_no_sink": (8, 257, 129, 1, True, True, None, False),
        "window": (16, 513, 769, 1, True, True, 127, True),
        "envelope": (4, 513, 768, 2, False, False, None, False),
        "asymmetric": (4, 513, 769, 1, True, True, None, True),
        "partial6": (6, 129, 257, 1, False, False, None, False),
        "partial12": (12, 257, 513, 1, True, True, None, True),
        "partial_keyless": (6, 257, 65, 1, True, True, None, False),
        "partial_window": (12, 129, 257, 1, True, True, 47, True),
        "partial_envelope": (24, 129, 385, 2, True, True, None, False),
        "partial_asymmetric": (10, 129, 257, 1, True, True, None, True),
        "partial192": (192, 9, 65, 1, False, False, None, False),
    }[geometry]
    d_qk, d_v = {"envelope": (64, 64), "asymmetric": (96, 80), "partial_envelope": (64, 64), "partial_asymmetric": (96, 80)}.get(geometry, (128, 128))
    if head_dims is not None:
        d_qk, d_v = head_dims
    api, t = _half_prefold_api(
        d_qk,
        d_v,
        getattr(torch, dtype_name),
        prefolded=False,
        with_stats=with_stats,
        split_kv=split,
        b=2,
        hq=2 * group,
        hkv=2,
        s_q=sq,
        s_kv=sk,
        causal=causal,
        bottom_right=br,
        window_left=window,
        sink=sink,
        pack_gqa=True,
        tile_n=tile_n,
    )
    api.sched_policy = sched_policy
    if cga is not None:
        api.cga = cga
    assert api.check_support()
    api.compile()
    assert api._k_mod.__file__.endswith("sm107/prefill_d128_f16.py")
    cfg = api._k_mod.CFG
    if head_dims is not None:
        assert api.flavor == (64, 64) and cfg.TILE_K == cfg.TILE_O == 64
        assert cfg.CTA_MMA == cga
    if tile_n is not None:
        assert (cfg.TILE_N, cfg.S_SLOTS) == (tile_n, 2 if tile_n == 64 else 1)
        assert (api._k_mod.PARAMS.tile_n, api._k_mod.PARAMS.s_slots) == (cfg.TILE_N, cfg.S_SLOTS)
    assert (cfg.PACK_G, cfg.STAGES_KV, cfg.SOFTMAX_WARPGROUPS) == (math.gcd(group, 128), 4, 2)
    ws = torch.empty(max(api.scratch_workspace_bytes(), 1), dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def execute():
        api.execute(
            q_tensor=t["q"],
            k_tensor=t["k"],
            v_tensor=t["v"],
            o_tensor=t["out"],
            lse_tensor=t["lse"],
            sinks=t["sinks"],
            workspace=ws,
            current_stream=cuda.CUstream(stream.cuda_stream),
        )

    with torch.cuda.stream(stream):
        execute()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            execute()
        for _ in range(2):
            t["v"].neg_()
            t["out"].fill_(torch.nan)
            if t["lse"] is not None:
                t["lse"].fill_(torch.nan)
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            mask = _dense_mask(sq, sk, causal=causal, bottom_right=br, window_left=window)
            ref_o, ref_lse = _half_oracle(t["q"], t["k"], t["v"], logit_scale=t["attn_scale"], masked=mask, sinks=t["sinks"])
            torch.testing.assert_close(t["out"].double(), ref_o, atol=4e-3, rtol=4e-3)
            if t["lse"] is not None:
                torch.testing.assert_close(t["lse"].double(), ref_lse, atol=4e-4, rtol=0)
    finally:
        graph.reset()

    if geometry == "mha":
        # PACK_G=1 must be the unpacked function, including exact rounding.
        plain, u = _half_prefold_api(
            d_qk, d_v, getattr(torch, dtype_name), prefolded=False, with_stats=with_stats, split_kv=1, b=2, hq=2, hkv=2, s_q=sq, s_kv=sk, tile_n=tile_n
        )
        plain.sched_policy = sched_policy
        if cga is not None:
            plain.cga = cga
        assert plain.check_support()
        plain.compile()
        plain_ws = torch.empty(max(plain.scratch_workspace_bytes(), 1), dtype=torch.uint8, device="cuda")
        plain.execute(q_tensor=u["q"], k_tensor=u["k"], v_tensor=u["v"], o_tensor=u["out"], lse_tensor=u["lse"], workspace=plain_ws)
        torch.cuda.synchronize()
        assert torch.equal(t["out"], u["out"])
        if with_stats:
            assert torch.equal(t["lse"], u["lse"])
