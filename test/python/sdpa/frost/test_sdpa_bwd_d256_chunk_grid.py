# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workspace-sized backward chunks must also fit the final batched GEMM grid."""

import pytest
import torch

import cudnn
from cudnn.sdpa.bwd import api_dsl_sm107 as shared, engines
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell, select_engine

pytestmark = pytest.mark.L0
_ROUTES = [
    pytest.param(kind, dtype, id=f"{kind}-{str(dtype).split('.')[-1]}")
    for kind, dtypes in (
        ("sm100-half", (torch.float16, torch.bfloat16)),
        ("sm103-half", (torch.float16, torch.bfloat16)),
        ("sm107-half", (torch.float16, torch.bfloat16)),
        ("fp8", (torch.float16, torch.bfloat16, torch.float8_e4m3fn)),
        ("mx", (torch.bfloat16,)),
    )
    for dtype in dtypes
]


@pytest.mark.parametrize(
    "b,h,group,bpe,batch_chunking,expected",
    [
        pytest.param(256, 256, 1, 2, True, (256, 128), id="half-head-chunk"),
        pytest.param(256, 256, 256, 2, True, (128, 256), id="half-whole-mqa-group"),
        pytest.param(65536, 1, 1, 2, True, (32768, 1), id="half-large-batch"),
        pytest.param(1, 65536, 1, 2, True, (1, 32768), id="half-large-heads"),
        pytest.param(256, 256, 1, 1, False, (256, 128), id="fp8-complete-batch"),
        pytest.param(256, 256, 1, 2.0625, False, (256, 128), id="mxfp8-complete-batch"),
        pytest.param(255, 256, 1, 2, True, (255, 256), id="legal-product"),
        pytest.param(2, 8, 4, 2, True, (2, 8), id="legal-gqa"),
    ],
)
def test_budget_chunks_also_fit_launch(b, h, group, bpe, batch_chunking, expected):
    bc, hc = shared._sm107_chunks(b, h, group, 128, 256, bpe, batch_chunking=batch_chunking)
    assert b % bc == 0 and h % hc == 0 and hc % group == 0
    assert bc * hc <= 65535
    assert (bc, hc) == expected


def _api(monkeypatch, kind, *, b, h, hk, thd=False, dtype=torch.bfloat16, d=256, dv=256):
    from cudnn.frost import buffers, device
    from test_sdpa_bwd_dsl_sm107 import _adapter, _adapter_desc
    from test_sdpa_bwd_mxfp8_sm107 import _mxfp8_adapter

    cc = (10, 0) if kind == "sm100-half" else (10, 3) if kind == "sm103-half" else (10, 7)
    for module in (engines, buffers):
        monkeypatch.setattr(module, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.8.0")))
        monkeypatch.setattr(module, "cutedsl_arch_requirement_error", lambda _: None)
    monkeypatch.setattr(device, "compute_capability", lambda _: cc)
    kwargs = dict(b=b, hq=h, hkv=hk, sq=2, skv=3, thd=thd, max_total_seq_len_q=2 if thd else None, max_total_seq_len_kv=3 if thd else None)
    if kind == "mx":
        api = _mxfp8_adapter(**kwargs)
    else:
        cls = (
            shared.SdpaBwdDslSm107Fp8
            if kind == "fp8"
            else engines._adapter(
                engines._SM100_D256
                if kind in ("sm100-half", "sm103-half")
                else engines._SM107_D192_2X2 if d == 192 else engines._SM107_D256_2X2 if kind == "sm107-twin" else engines._SM107_D256_NATIVE
            )
        )
        input_dtype = torch.float8_e4m3fn if kind == "fp8" else dtype
        if d == dv:
            api = _adapter(cls, dt=input_dtype, grad_dt=dtype, **kwargs)
        else:
            samples = {}
            for name in ("q", "k", "v", "o", "do", "dq", "dk", "dv", "stats"):
                kv = name in ("k", "v", "dk", "dv")
                dim = 1 if name == "stats" else d if name in ("q", "k", "dq", "dk") else dv
                dt = torch.float32 if name == "stats" else dtype if name in ("dq", "dk", "dv") else input_dtype
                samples["sample_" + name] = _adapter_desc((b, hk if kv else h, 3 if kv else 2, dim), dt, name)
            api = cls(**samples, thd=thd, max_total_seq_len_q=2 if thd else None, max_total_seq_len_kv=3 if thd else None)
    facts = SdpaGraphFacts(
        is_backward=True,
        b=b,
        h_q=h,
        h_kv=hk,
        s_q=2,
        s_kv=3,
        d_qk=d,
        d_v=dv,
        dtype=cudnn.data_type.FP8_E4M3 if kind in ("fp8", "mx") else cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16,
        dtype_o={torch.float16: cudnn.data_type.HALF, torch.bfloat16: cudnn.data_type.BFLOAT16, torch.float8_e4m3fn: cudnn.data_type.FP8_E4M3}[dtype],
        is_fp8=kind == "fp8",
        is_mxfp8=kind == "mx",
        device_cc=cc,
        thd=thd,
        padded=thd,
        max_total_seq_len_q=2 if thd else None,
        max_total_seq_len_kv=3 if thd else None,
    )
    spec = next(s for s in engines.ENGINE_SPECS if s.name == api._NAME)
    return api, facts, spec


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("b,h,hk", [(65536, 1, 1), (1, 65536, 1), (256, 256, 256), (256, 256, 1)])
def test_sm107_dense_twin_preserves_shared_chunk_bounds(monkeypatch, dtype, b, h, hk):
    api, facts, spec = _api(monkeypatch, "sm107-twin", b=b, h=h, hk=hk, dtype=dtype)
    reason = engines.mismatch(spec.capabilities, facts)
    if h // hk > 65535:
        assert reason is not None and "grid" in reason, reason
        with pytest.raises(ValueError, match="grid.*65535"):
            api.check_support()
    else:
        assert reason is None, reason
        assert api.check_support()
        assert api._b_chunk * api._qh_chunk <= 65535


@pytest.mark.parametrize("kind,dtype", [route for route in _ROUTES if route.values[0] != "mx"])
@pytest.mark.parametrize("b", [32767, 32768])
@pytest.mark.parametrize("entry", ["graph", "standalone"])
def test_d192_sibling_checks_final_gemm_grid(monkeypatch, kind, dtype, b, entry):
    api, facts, spec = _api(monkeypatch, kind, b=b, h=2, hk=1, thd=True, dtype=dtype, d=192, dv=128)
    if entry == "graph":
        reason = engines.mismatch(spec.capabilities, facts)
        assert reason is None if b == 32767 else reason is not None and "grid" in reason
    elif b == 32767:
        assert api.check_support()
    else:
        with pytest.raises(ValueError, match="grid.*65535"):
            api.check_support()


@pytest.mark.parametrize("kind,dtype", _ROUTES)
@pytest.mark.parametrize("b,h,hk,accepted", [(65535, 1, 1, True), (65536, 1, 1, False), (32767, 2, 1, True), (32768, 2, 1, False)])
@pytest.mark.parametrize("entry", ["graph", "standalone"])
def test_thd_actual_batch_and_indivisible_group(monkeypatch, kind, dtype, b, h, hk, accepted, entry):
    api, facts, spec = _api(monkeypatch, kind, b=b, h=h, hk=hk, thd=True, dtype=dtype)
    if entry == "graph":
        reason = engines.mismatch(spec.capabilities, facts)
        assert reason is None if accepted else reason is not None and "grid" in reason and "65535" in reason
    elif accepted:
        assert api.check_support()
    else:
        with pytest.raises(ValueError, match="grid.*65535"):
            api.check_support()


@pytest.mark.parametrize("kind,dtype", _ROUTES)
def test_dense_large_batch_depends_on_batch_chunking(monkeypatch, kind, dtype):
    api, facts, spec = _api(monkeypatch, kind, b=65536, h=1, hk=1, dtype=dtype)
    reason = engines.mismatch(spec.capabilities, facts)
    if kind.endswith("half"):
        assert reason is None, reason
        assert api.check_support()
        assert api._b_chunk * api._qh_chunk <= 65535
    else:
        assert reason is not None and "grid" in reason, reason
        with pytest.raises(ValueError, match="grid.*65535"):
            api.check_support()


@pytest.mark.parametrize("kind,dtype", _ROUTES)
def test_thd_chunk_uses_sequence_count_not_storage_batch(monkeypatch, kind, dtype):
    api, facts, spec = _api(monkeypatch, kind, b=256, h=256, hk=256, thd=True, dtype=dtype)
    assert api.check_support()
    assert engines.mismatch(spec.capabilities, facts) is None
    assert api._b_chunk == 1
    assert api.batch_size * api._qh_chunk <= 65535
    assert api._qh_chunk == 128


@pytest.mark.parametrize("kind,dtype", _ROUTES)
@pytest.mark.parametrize("single_dq", [True, False])
def test_dense_group_preserves_qualified_subgroup_path(monkeypatch, kind, dtype, single_dq):
    monkeypatch.setattr(shared, "DQ_SINGLE_LAUNCH", single_dq)
    api, facts, spec = _api(monkeypatch, kind, b=1, h=65536, hk=1, dtype=dtype)
    accepted = kind in ("sm100-half", "sm103-half") and dtype == torch.bfloat16 and single_dq
    reason = engines.mismatch(spec.capabilities, facts)
    if accepted:
        assert reason is None, reason
        assert api.check_support()
        assert api._b_chunk * api._qh_chunk <= 65535
    else:
        assert reason is not None and "grid" in reason, reason
        with pytest.raises(ValueError, match="grid.*65535"):
            api.check_support()


@requires_dsl
@requires_pre_rubin_blackwell
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("hkv", [256, 128], ids=["mha", "gqa"])
def test_sm100_thd_full_chain_grid_reference(monkeypatch, dtype, hkv):
    import test_sdpa_bwd_thd_sm107 as helper
    from determinism_utils import assert_bitwise, snapshot

    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")

    def run(lengths):
        case = helper._thd_case(lengths, lengths, 256, 256, dtype, hkv=hkv, poison=True)
        graph, vp, ports = helper._build_thd_bwd_graph(case, hkv=hkv)
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A])
        select_engine(graph, "sdpa_bwd_sm100_d256")
        graph.check_support()
        graph.build_plans()
        outputs = tuple(torch.full_like(x, float("nan")) for x in (case.q, case.k, case.v))
        vp.update(zip(ports, outputs))
        workspace = torch.full((max(graph.get_workspace_size(), 1),), 0xFF, device="cuda", dtype=torch.uint8)
        graph.execute(vp, workspace)
        torch.cuda.synchronize()
        helper._check(case, *outputs)
        return case, outputs, graph, vp, workspace

    # Storage has one batch; the final GEMM still launches all 256 sequences.
    lengths = [2, 0, 3] + [0] * 252 + [2]
    case, outputs, graph, vp, workspace = run(lengths)
    names = ("dq", "dk", "dv")
    expected = snapshot(dict(zip(names, outputs)))
    for tensor in outputs:
        tensor.fill_(float("nan"))
    workspace.fill_(0xBD)
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        graph.execute(vp, workspace)
    capture.replay()
    torch.cuda.synchronize()
    helper._check(case, *outputs)
    assert_bitwise(snapshot(dict(zip(names, outputs))), expected)
    case.do.neg_()
    for tensor in outputs:
        tensor.fill_(float("nan"))
    capture.replay()
    torch.cuda.synchronize()
    helper._check(case, *outputs)
    for name, actual in snapshot(dict(zip(names, outputs))).items():
        assert not torch.equal(actual.view(torch.int16), expected[name].view(torch.int16))
    # Same packed bytes, but removing empty neighbors changes the head chunk.
    compact, compact_outputs, *_ = run([2, 3, 2])
    for name in ("q", "k", "v", "o", "lse"):
        assert torch.equal(getattr(compact, name), getattr(case, name))
    assert torch.equal(compact.do, -case.do)
    assert_bitwise(snapshot(dict(zip(names, compact_outputs))), expected, "empty-neighbor rebatch")


@requires_dsl
@requires_pre_rubin_blackwell
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_sm100_dense_batch_chunks_preserve_whole_group(monkeypatch, dtype):
    import test_sdpa_bwd_d256_sm100 as helper
    from cudnn.sdpa.bwd import chunk_grid
    from test_sdpa_bwd_thd_sm107 import _grad_failure
    from determinism_utils import assert_bitwise, snapshot

    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    # Dense staging scales with padded B*H*S*D. The host cases above exercise
    # the actual 65535 boundary; lower this limit to exercise the same native
    # whole-group batch split without a multi-GiB padded operand per port.
    monkeypatch.setattr(chunk_grid, "GRID_Z_MAX", 15)
    selected = []
    original = shared._sm107_chunks

    def record(*args, **kwargs):
        chunk = original(*args, **kwargs)
        selected.append(chunk)
        return chunk

    monkeypatch.setattr(shared, "_sm107_chunks", record)
    generator = torch.Generator(device="cpu").manual_seed(17)

    def draw(s, h):
        return torch.randn(1, s, h, 256, generator=generator).to(device="cuda", dtype=dtype).permute(0, 2, 1, 3)

    q, do, k, v = draw(2, 4), draw(2, 4), draw(3, 1), draw(3, 1)
    o, lse, _, *refs = helper._reference64(q, k, v, do, group=4)

    def repeat(x):
        return x.permute(0, 2, 1, 3).repeat(4, 1, 1, 1).permute(0, 2, 1, 3)

    data = {name: repeat(x) for name, x in zip(("q", "k", "v", "do", "o"), (q, k, v, do, o.to(dtype)))}
    data["stats"] = lse.float().unsqueeze(-1).repeat(4, 1, 1, 1).contiguous()
    graph, ports, grad_ports = helper._build_graph(b=4, hq=4, hkv=1, sq=2, skv=3, dt=dtype)
    select_engine(graph, "sdpa_bwd_sm100_d256")
    graph.check_support()
    graph.build_plans()
    assert selected and all(chunk == (2, 4) for chunk in selected)
    outputs = tuple(torch.full_like(data[name], float("nan")) for name in ("q", "k", "v"))
    vp = {ports[name]: tensor for name, tensor in data.items()}
    vp.update(zip(grad_ports, outputs))
    workspace = torch.full((max(graph.get_workspace_size(), 1),), 0xFF, dtype=torch.uint8, device="cuda")

    def check(sign=1):
        for output, ref in zip(outputs, refs):
            # Bound oracle temporaries while checking every batch and element.
            for start in range(0, 4, 2):
                got = output[start : start + 2]
                error = _grad_failure("dense chunk", got, (ref * sign).expand_as(got), dtype)
                assert error is None, error

    graph.execute(vp, workspace)
    torch.cuda.synchronize()
    check()
    expected = snapshot({str(i): x.cpu() for i, x in enumerate(outputs)})
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        graph.execute(vp, workspace)
    for x in outputs:
        x.fill_(float("nan"))
    workspace.fill_(0xBD)
    capture.replay()
    torch.cuda.synchronize()
    check()
    assert_bitwise(snapshot({str(i): x.cpu() for i, x in enumerate(outputs)}), expected)
    data["do"].neg_()
    capture.replay()
    torch.cuda.synchronize()
    check(-1)
