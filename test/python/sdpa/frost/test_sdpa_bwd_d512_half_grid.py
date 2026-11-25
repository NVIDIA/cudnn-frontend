# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D512 half admission and chunking respect the final GEMM's batch/head grid."""

import pytest
import torch

import cudnn
from cudnn.api_base import TensorDesc
from cudnn.sdpa.bwd import engines
from cudnn.sdpa.bwd.heuristics import recommend
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

pytestmark = pytest.mark.L0

_DTYPES = [pytest.param(torch.float16, id="fp16"), pytest.param(torch.bfloat16, id="bf16")]
_OWNERS = [
    pytest.param((10, 0), False, id="sm100-role-split"),
    pytest.param((10, 0), True, id="sm100-2x2"),
    pytest.param((10, 3), False, id="sm103-role-split"),
    pytest.param((10, 3), True, id="sm103-2x2"),
    pytest.param((10, 7), True, id="sm107-2x2"),
]


def _declarations(monkeypatch, cc, twin, dtype, *, b=1, h=1, hk=None, d=512, thd=False):
    from cudnn.frost import buffers, device
    from cudnn.sdpa.bwd.kernels.sm100.d512_f16_role_split.host import SdpaBwdDslSm100D512RoleSplit
    from cudnn.sdpa.bwd.kernels.sm100.d512_f16_2x2.host import SdpaBwdDslSm100D512TwoByTwo
    from cudnn.sdpa.bwd.kernels.sm107.d512_f16_2x2.host import SdpaBwdDslSm107D512

    # Only model declaration-time architecture gates. These tests neither
    # compile nor execute an SM103/SM107 kernel on a different device.
    for module in (engines, buffers):
        monkeypatch.setattr(module, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.8.0")))
        monkeypatch.setattr(module, "cutedsl_arch_requirement_error", lambda _: None)
    monkeypatch.setattr(device, "compute_capability", lambda _: cc)
    hk = h if hk is None else hk
    samples, ports = {}, []
    for role in ("q", "k", "v", "o", "do", "dq", "dk", "dv", "stats"):
        kv = role in ("k", "v", "dk", "dv")
        length, heads = (3, hk) if kv else (2, h)
        shape = (b, heads, length, 1 if role == "stats" else d)
        stride = (h * 2, 2, 1, 1) if role == "stats" else (length * heads * d, d, heads * d, 1)
        samples["sample_" + role] = TensorDesc(
            torch.float32 if role == "stats" else dtype, shape, stride, TensorDesc._compute_stride_order(shape, stride), torch.device("cuda", 0)
        )
        if role != "stats":
            ports.append((role, shape, stride))
    facts = SdpaGraphFacts(
        is_backward=True,
        b=b,
        h_q=h,
        h_kv=hk,
        s_q=2,
        s_kv=3,
        d_qk=d,
        d_v=d,
        dtype=cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16,
        device_cc=cc,
        thd=thd,
        padded=thd,
        max_total_seq_len_q=2 if thd else None,
        max_total_seq_len_kv=3 if thd else None,
        port_layouts=tuple(ports),
    )
    if cc == (10, 7):
        api_type, engine_id = SdpaBwdDslSm107D512, 20608
    else:
        api_type, engine_id = (SdpaBwdDslSm100D512TwoByTwo, 20627) if twin else (SdpaBwdDslSm100D512RoleSplit, 20626)
    api = api_type(**samples, thd=thd, max_total_seq_len_q=facts.max_total_seq_len_q, max_total_seq_len_kv=facts.max_total_seq_len_kv)
    spec = next(s for s in engines.ENGINE_SPECS if s.name == api._NAME)
    return facts, api, spec, engine_id


@pytest.mark.parametrize("cc,twin", _OWNERS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("d", [264, 384, 512])
@pytest.mark.parametrize("b", [65535, 65536])
@pytest.mark.parametrize("entry", ["graph", "standalone"])
def test_d512_half_batch_boundary(monkeypatch, cc, twin, dtype, d, b, entry):
    facts, api, spec, engine_id = _declarations(monkeypatch, cc, twin, dtype, b=b, d=d)
    if entry == "standalone":
        if b <= 65535:
            assert api.check_support()
        else:
            with pytest.raises(ValueError, match="grid.*65535"):
                api.check_support()
    else:
        reason = engines.mismatch(spec.capabilities, facts)
        assert reason is None if b <= 65535 else reason is not None and "grid" in reason and "65535" in reason
        for mode in ("A", "FALLBACK"):
            assert [p.engine_id for p in recommend(mode, facts, {spec.name: engine_id})] == ([engine_id] if b <= 65535 else [])


@pytest.mark.parametrize("cc,twin", _OWNERS[:4])
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("b", [65535, 65536])
def test_thd_still_uses_batched_final_gemms(monkeypatch, cc, twin, dtype, b):
    facts, api, spec, engine_id = _declarations(monkeypatch, cc, twin, dtype, b=b, thd=True)
    reason = engines.mismatch(spec.capabilities, facts)
    if b <= 65535:
        assert reason is None, reason
        assert api.check_support()
    else:
        assert reason is not None and "grid" in reason and "65535" in reason, reason
        with pytest.raises(ValueError, match="grid.*65535"):
            api.check_support()
    assert [p.engine_id for p in recommend("A", facts, {spec.name: engine_id})] == ([engine_id] if b <= 65535 else [])


@pytest.mark.parametrize("cc,twin", _OWNERS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("hk", [1, 65536], ids=["mqa", "mha"])
def test_head_chunking_preserves_large_head_declarations(monkeypatch, cc, twin, dtype, hk):
    facts, api, spec, _ = _declarations(monkeypatch, cc, twin, dtype, h=65536, hk=hk)
    assert api.check_support()
    assert engines.mismatch(spec.capabilities, facts) is None
    assert 0 < api._qh_chunk < 65536
    assert api.h_q % api._qh_chunk == 0
    assert api.batch_size * api._qh_chunk <= 65535


@pytest.mark.parametrize("cc,twin", _OWNERS[:4])
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("b,h,hk", [(255, 256, 256), (256, 256, 256), (256, 256, 1), (512, 128, 128)])
def test_thd_chunk_bounds_include_batch_product(monkeypatch, cc, twin, dtype, b, h, hk):
    facts, api, spec, _ = _declarations(monkeypatch, cc, twin, dtype, b=b, h=h, hk=hk, thd=True)
    assert engines.mismatch(spec.capabilities, facts) is None
    assert api.check_support()
    assert api.h_q % api._qh_chunk == 0
    assert api.batch_size * api._qh_chunk <= 65535
    # Immediately below the grid limit, the already-valid chunk is unchanged.
    assert api._qh_chunk == (256 if b == 255 else h // 2)


@requires_dsl
@requires_pre_rubin_blackwell
@pytest.mark.parametrize("twin", [False, True], ids=["role-split", "2x2"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("hkv", [256, 1], ids=["mha", "mqa-cross-chunk"])
def test_thd_grid_product_numerics_and_replay(monkeypatch, twin, dtype, hkv):
    import test_sdpa_bwd_thd_sm100 as helper
    from determinism_utils import assert_bitwise, snapshot

    name = "sdpa_bwd_sm100_d512_2x2" if twin else "sdpa_bwd_sm100_d512_role_split"
    monkeypatch.setattr(helper, "_ENGINE", name)
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    # The old budget-only rule picks 256 heads: the final GEMM then launches
    # grid.z = 256 * 256. Few live sequences keep input/reference memory small;
    # all 256 metadata entries still participate in the actual launch grid.
    lengths = [2, 0, 3] + [0] * 252 + [2]
    case, dq, dk, dv = helper._run_graph(lengths, lengths, h=256, hkv=hkv, d=264, dtype=dtype, poison=True)
    outputs = dict(zip(("dq", "dk", "dv"), (dq, dk, dv)))
    expected = snapshot(outputs)
    # Reuse the prepared plan, poisoned outputs and workspace under capture.
    for tensor in (dq, dk, dv):
        tensor.fill_(float("nan"))
    case.workspace.fill_(0xBD)
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        case.graph.execute(case.vp, case.workspace)
    capture.replay()
    torch.cuda.synchronize()
    helper._check(case, dq, dk, dv)
    assert_bitwise(snapshot(outputs), expected)
    # Changed upstream gradients must be consumed by the retained capture.
    case.do.neg_()
    for tensor in (dq, dk, dv):
        tensor.fill_(float("nan"))
    capture.replay()
    torch.cuda.synchronize()
    helper._check(case, dq, dk, dv)
    for name, actual in snapshot(outputs).items():
        original = expected[name]
        assert not torch.equal(actual.view(torch.int16), original.view(torch.int16))
    # Removing empty neighbors changes the chunk from 128 to 256. Preserve
    # packed data and the original upstream gradients, then compare all outputs.
    compact, cq, ck, cv = helper._run_graph([2, 3, 2], [2, 3, 2], h=256, hkv=hkv, d=264, dtype=dtype, poison=True)
    for name in ("q", "k", "v", "o", "lse"):
        assert torch.equal(getattr(compact, name), getattr(case, name))
    assert torch.equal(compact.do, -case.do)
    assert_bitwise(snapshot(dict(zip(("dq", "dk", "dv"), (cq, ck, cv)))), expected, "empty-neighbor rebatch")
