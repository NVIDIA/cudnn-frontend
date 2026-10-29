# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native D64 packed attention retains physical origins and captured bindings."""

import pytest
import torch

from frost_test_utils import requires_dsl, requires_rubin

pytestmark = [pytest.mark.L0, requires_dsl, requires_rubin]


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("cga", [1, 2])
@pytest.mark.parametrize("packed", [False, True])
def test_native_d64_independent_origins(dtype_name, cga, packed, monkeypatch):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100
    from test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture

    compile_plan = SdpaFwdDslSm100.compile
    seen = []

    def compile_native(api):
        compile_plan(api)
        assert api.flavor == (64, 64)
        assert api._k_mod.CFG.TILE_K == api._k_mod.CFG.TILE_O == 64
        assert api._k_mod.CFG.CTA_MMA == cga
        assert api._k_mod.CFG.THD_VARLEN and bool(api._k_mod.CFG.PACK_GQA) == packed
        seen.append(api.flavor)

    monkeypatch.setattr(SdpaFwdDslSm100, "compile", compile_native)
    test_independent_thd_origins_rebind_and_capture(
        64, dtype_name, "head" if packed else "token", True, packed, 1, engine_name="sdpa_fwd_prefill_sm107", cga=cga
    )
    assert seen


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cga", [1, 2])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("stats_layout,tile_n,scheduler", [("token", 128, 0), ("head", 64, 1), ("padded", 128, 1)])
def test_native_d64_ragged_replay(dtype, cga, packed, stats_layout, tile_n, scheduler):
    from test_sdpa_prepared_thd import _check_rubin_ragged_pack_rebind

    _check_rubin_ragged_pack_rebind(
        dtype,
        stats_layout,
        stats_layout != "token",
        "partial12",
        cga=cga,
        packed=packed,
        sched_policy=scheduler,
        tile_n=tile_n,
        head_dim=64,
    )


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("cga", [1, 2])
@pytest.mark.parametrize(
    "head_dims,geometry,tile_n,scheduler,with_stats",
    [
        ((64, 64), "partial12", 128, 0, True),
        ((48, 32), "partial_window", 64, 1, True),
        ((24, 16), "keyless_no_sink", 128, 1, False),
        ((64, 64), "mha", 64, 0, True),
        pytest.param((64, 64), "mha", 64, 1, True, id="mha_lpt"),
    ],
)
def test_native_d64_dense_pack_replay(dtype_name, cga, head_dims, geometry, tile_n, scheduler, with_stats):
    from test_sdpa_native_pack_sm107 import _check_native_d128_packing_dense_replay

    _check_native_d128_packing_dense_replay(dtype_name, scheduler, with_stats, geometry, tile_n=tile_n, head_dims=head_dims, cga=cga)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cga", [1, 2])
@pytest.mark.parametrize("packed", [False, True])
def test_native_d64_standalone_compact_thd(dtype, cga, packed):
    from itertools import accumulate

    from sm107_half_test_utils import _dense_mask, _half_oracle
    from test_sdpa_prepared_thd import _buffers, _compact_thd_api, _execute_compact_thd

    b, ql, kl, hq, hk, d = 3, 257, 1025, 24, 2, 64
    a = _buffers(b, ql, kl, hq, hk, d, seed=721, dtype=dtype)
    fresh = _buffers(b, ql, kl, hq, hk, d, seed=722, dtype=dtype)
    api = _compact_thd_api(a, b, ql, kl, hq, hk, d, cga=cga, pack_gqa=packed)
    assert api.flavor == (64, 64) and api._k_mod.CFG.TILE_K == api._k_mod.CFG.TILE_O == 64
    ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device="cuda", dtype=torch.uint8)

    def lengths(bufs, qs, ks):
        cq, ck = [0, *accumulate(qs)], [0, *accumulate(ks)]
        bufs["cu_q"].copy_(torch.tensor(cq, device="cuda", dtype=torch.int32))
        bufs["cu_kv"].copy_(torch.tensor(ck, device="cuda", dtype=torch.int32))
        bufs["q"][cq[-1] :].fill_(torch.nan)
        bufs["k"][ck[-1] :].fill_(torch.nan)
        bufs["v"][ck[-1] :].fill_(torch.nan)
        bufs["o"].fill_(19)
        bufs["lse"].fill_(19)
        return cq, ck

    def check(bufs, cq, ck):
        torch.cuda.synchronize()
        for i in range(b):
            lo, hi, klo, khi = cq[i], cq[i + 1], ck[i], ck[i + 1]
            if lo == hi:
                continue
            q = bufs["q"][lo:hi].transpose(0, 1).unsqueeze(0)
            k = bufs["k"][klo:khi].transpose(0, 1).unsqueeze(0)
            v = bufs["v"][klo:khi].transpose(0, 1).unsqueeze(0)
            out, lse = _half_oracle(q, k, v, logit_scale=d**-0.5, masked=_dense_mask(hi - lo, khi - klo, causal=True))
            torch.testing.assert_close(bufs["o"][lo:hi].double(), out[0].transpose(0, 1), atol=4e-3, rtol=4e-3)
            torch.testing.assert_close(bufs["lse"][lo:hi].double(), lse[0].T, atol=4e-4, rtol=0)
        assert torch.all(bufs["o"][cq[-1] :] == 19)
        assert torch.all(bufs["lse"][cq[-1] :] == 19)

    qa, ka = lengths(a, [257, 0, 73], [1025, 0, 117])
    qf, kf = lengths(fresh, [17, 129, 0], [65, 513, 0])
    _execute_compact_thd(api, a, ws)
    check(a, qa, ka)
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            _execute_compact_thd(api, a, ws)
        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            _execute_compact_thd(api, fresh, ws)
        finally:
            torch.cuda.set_sync_debug_mode(previous)
        check(fresh, qf, kf)
        a["q"].mul_(0.5)
        a["v"].mul_(0.75)
        qa, ka = lengths(a, [31, 7, 0], [40, 19, 0])
        ws.fill_(255)
        capture.replay()
        check(a, qa, ka)
        check(fresh, qf, kf)
    finally:
        capture.reset()
