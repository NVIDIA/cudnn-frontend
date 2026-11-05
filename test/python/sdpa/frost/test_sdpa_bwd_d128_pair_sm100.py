# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense d64 / d128 deterministic backward: the dK/dV + dQ kernel pair, its selection, workspace, numerics and replay."""

import pytest
import torch

from types import SimpleNamespace

from frost_test_utils import cuda_launch_names, requires_dsl, requires_pre_rubin_blackwell, select_engine
from test_sdpa_bwd_d128_sm100 import _ENGINE, _case, _check_case
from test_sdpa_bwd_dsl_sm107 import _bshd_empty, _build_graph, _causal_keep, _reference64

pytestmark = [pytest.mark.L0, requires_dsl]


def _require_pair():
    from cudnn.frost.buffers import cutedsl_state
    from cudnn.sdpa.bwd.d128_contract import ACCUMULATION, select_implementation

    if select_implementation(torch.cuda.get_device_capability(), cutedsl_state()[1], 1, 1, 128, 128, 0.125) != ACCUMULATION:
        pytest.skip("the deterministic d128 pair requires SM100 / SM103 and public CuTeDSL >= 4.8")


def test_config_geometry():
    from cudnn.sdpa.bwd.config_sm100_d128_dq import CFG, HEAD_DIMS, make_bar_layout, make_config, validate_config

    for d in HEAD_DIMS:
        cfg = make_config(d)
        validate_config(cfg, make_bar_layout(cfg))
        assert cfg.OPERAND_BYTES == 128 * d * 2 and cfg.TMEM_DQ + d <= cfg.TMEM_COLS
    assert make_config(128) == CFG and make_config(64).SMEM_BYTES < CFG.SMEM_BYTES
    with pytest.raises(ValueError):
        make_config(96)


def test_pair_workspace_layout():
    from cudnn.sdpa.bwd.d128_contract import pair_workspace_layout, workspace_layout

    regions, total = pair_workspace_layout(2, 3, 300)
    assert regions == ((0, (2, 3, 384), (3 * 384, 384, 1)),) and total == 4 * 2 * 3 * 384 + 240
    assert total < workspace_layout(2, 3, 300, 64)[1]
    with pytest.raises(ValueError):
        pair_workspace_layout(0, 3, 300)


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("left", [None, 128])
def test_deterministic_request_selects_the_pair(monkeypatch, d, left):
    from types import SimpleNamespace

    from cudnn import _pybind_module
    from cudnn.frost import buffers, device
    from cudnn.sdpa.bwd import api_dsl_sm100_d128
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128
    from cudnn.sdpa.bwd.d128_contract import PAIR, RECOMPUTE
    from test_sdpa_bwd_dsl_sm107 import _adapter

    monkeypatch.setattr(device, "compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(_pybind_module, "_SdpaBwdBinder", SimpleNamespace(supports_zero_workspace=True))
    monkeypatch.setattr(api_dsl_sm100_d128, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.8.1")), raising=False)
    monkeypatch.setattr(buffers, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.8.1")))
    monkeypatch.setattr(buffers, "cutedsl_too_old", lambda version: False)
    monkeypatch.setattr(buffers, "cutedsl_arch_requirement_error", lambda cc: None)
    kw = dict(d=d, deterministic=True, is_causal=True, **({"window_size_left": left} if left is not None else {}))
    api = _adapter(SdpaBwdDslSm100D128, **kw)
    assert api.check_support() and api._implementation == PAIR
    assert api.scratch_workspace_bytes() == 4 * 2 * 2 * 512 + 240
    negative = _adapter(SdpaBwdDslSm100D128, scale_softmax=-0.125, **kw)
    assert negative.check_support() and negative._implementation == RECOMPUTE and negative.scratch_workspace_bytes() == 0
    plain = _adapter(SdpaBwdDslSm100D128, **{**kw, "deterministic": False})
    assert plain.check_support() and plain._implementation == ("recompute" if (d == 128 and left is not None) else "accumulation")


def test_dkdv_rendering_is_a_template_parameter():
    from cudnn.frost.template_loader import load_template
    from cudnn.sdpa.bwd.kernels.sm100 import bprop_d128_f16 as shipped

    pytest.importorskip("cutlass")
    assert shipped.DQ_OUT is True and shipped.HEAD_DIM == 128
    dkdv = load_template(shipped.__file__, (128, False), tag="test_dkdv_only")
    assert dkdv.DQ_OUT is False and dkdv.HEAD_DIM == 128 and dkdv.CFG_ACCUM == shipped.CFG_ACCUM
    assert load_template(shipped.__file__, 64, tag="test_d64_shipped").DQ_OUT is True


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize(
    "d,causal,left,sq,sk",
    [
        (128, False, None, 128, 128),
        (128, True, None, 128, 256),
        (128, False, None, 65, 129),
        (128, True, None, 257, 193),
        (128, True, None, 333, 333),
        (128, False, None, 1, 65),
        (128, False, None, 65, 3),
        (128, True, None, 65, 1),
        (128, True, 128, 257, 385),
        (128, False, 257, 193, 257),
        (64, False, None, 512, 512),
        (64, True, None, 257, 193),
        (64, True, 128, 2048, 2048),
        (64, True, 1, 257, 385),
        (64, False, 65, 193, 257),
        (64, True, 2, 1, 2),
    ],
    ids=[
        "d128-dense",
        "d128-causal",
        "d128-dense-tail",
        "d128-causal-tail",
        "d128-causal-333",
        "d128-one-query",
        "d128-three-keys",
        "d128-causal-one-key",
        "d128-w128",
        "d128-dense-w257",
        "d64-dense",
        "d64-causal",
        "d64-w128-2k",
        "d64-w1",
        "d64-dense-w65",
        "d64-w2-1x2",
    ],
)
def test_pair_matches_fp64(dt, d, causal, left, sq, sk):
    _require_pair()
    _case(dt=dt, d=d, causal=causal, left=left, b=2, h=3, sq=sq, sk=sk, deterministic=True, accumulation=None, allow_zero=True)


def _tensors(dt, d, causal, left, b, hq, hkv, sq, sk, seed=729):
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def draw(s, h):
        return torch.randn(b, s, h, d, generator=gen).to(device="cuda", dtype=dt).permute(0, 2, 1, 3)

    q, k, v, do = draw(sq, hq), draw(sk, hkv), draw(sk, hkv), draw(sq, hq)
    keep = _causal_keep(sq, sk, left=left, causal=causal) if causal or left is not None else None
    o64, lse, _, *expected = _reference64(q, k, v, do, keep, group=hq // hkv)
    tensors = dict(q=q, k=k, v=v, do=do, o=_bshd_empty(b, sq, hq, d, dt).copy_(o64), stats=lse.float().unsqueeze(-1).contiguous())
    tensors.update(
        dq=_bshd_empty(b, sq, hq, d, dt, fill=float("nan")),
        dk=_bshd_empty(b, sk, hkv, d, dt, fill=float("nan")),
        dv=_bshd_empty(b, sk, hkv, d, dt, fill=float("nan")),
    )
    return tensors, expected, keep


def _grouped_case(dt, d, causal, left, b, hq, hkv, sq, sk, seed=729):
    """A dense GQA / MQA graph under a deterministic request: the pair, its delta-only workspace, the fp64 gate."""
    tensors, expected, keep = _tensors(dt, d, causal, left, b, hq, hkv, sq, sk, seed)
    graph, refs, outputs = _build_graph(
        b=b,
        hq=hq,
        hkv=hkv,
        sq=sq,
        skv=sk,
        d=d,
        dt=dt,
        use_causal_mask=causal,
        scale="default",
        use_deterministic_algorithm=True,
        **({"diagonal_band_left_bound": left} if left is not None else {}),
    )
    select_engine(graph, _ENGINE)
    graph.check_support()
    graph.build_plans()
    prepared = graph._compiled_plans[graph._plan_index]._prepared
    assert prepared.spec.artifact.implementation == "pair"
    rounded = (sq + 127) // 128 * 128
    required = 4 * b * hq * rounded + 240
    assert graph.get_workspace_size() == required == prepared.spec.workspace_bytes
    refs.update(zip(("dq", "dk", "dv"), outputs))
    pack = {refs[n]: value for n, value in tensors.items()}
    workspace = torch.full((required,), 0xA5, dtype=torch.uint8, device="cuda")
    graph.execute(pack, workspace)
    case = SimpleNamespace(graph=graph, tensors=tensors, refs=refs, pack=pack, expected=expected, keep=keep, dt=dt, workspace=workspace)
    _check_case(case, allow_zero=True)
    return case


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize(
    "d,causal,left,hq,hkv,sq,sk",
    [
        (128, False, None, 4, 2, 128, 128),
        (128, True, None, 6, 2, 257, 193),
        (128, True, None, 8, 1, 333, 333),
        (128, False, None, 3, 1, 65, 129),
        (128, True, 128, 4, 2, 257, 385),
        (64, True, None, 8, 1, 512, 512),
        (64, True, 128, 4, 2, 257, 385),
        (64, False, 65, 2, 1, 193, 257),
    ],
    ids=[
        "d128-4of2",
        "d128-6of2-causal-tail",
        "d128-mqa8-causal",
        "d128-mqa3-tail",
        "d128-4of2-w128",
        "d64-mqa8-causal",
        "d64-4of2-w128",
        "d64-2of1-dense-w65",
    ],
)
def test_pair_grouped_matches_fp64(dt, d, causal, left, hq, hkv, sq, sk):
    _require_pair()
    _grouped_case(dt, d, causal, left, 2, hq, hkv, sq, sk)


@requires_pre_rubin_blackwell
def test_pair_grouped_bitwise_and_replay():
    _require_pair()
    case = _grouped_case(torch.bfloat16, 128, True, None, 2, 8, 2, 1024, 1153)
    expected_bits = {name: case.tensors[name].clone() for name in ("dq", "dk", "dv")}
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            case.graph.execute(case.pack, case.workspace)
    for replay in (False, True, True):
        case.workspace.fill_(0x5A)
        for name in expected_bits:
            case.tensors[name].fill_(float("nan"))
        if replay:
            graph.replay()
        else:
            case.graph.execute(case.pack, case.workspace)
        torch.cuda.synchronize()
        for name, expected in expected_bits.items():
            assert torch.equal(case.tensors[name], expected), f"{name} changed across identical deterministic executions"


@requires_pre_rubin_blackwell
def test_pair_standalone_window_past_the_keys_zeroes_dq():
    """Query tiles whose window lies entirely past the keys own no key tile: their dQ rows must still be written, as zeros.
    The graph refuses a window with S_q > S_kv, so this drives the standalone surface (radius = graph left bound - 1)."""
    _require_pair()
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128

    dt, left, sq, sk = torch.bfloat16, 49, 1000, 100
    tensors, expected, _ = _tensors(dt, 64, True, left, 1, 2, 2, sq, sk)
    api = SdpaBwdDslSm100D128(**{"sample_" + name: value for name, value in tensors.items()}, is_causal=True, window_size_left=left - 1, deterministic=True)
    api.check_support()
    api.compile()
    assert api._prepared.artifact.implementation == "pair"
    workspace = torch.full((api.scratch_workspace_bytes(),), 0xA5, dtype=torch.uint8, device="cuda")
    api.execute(**{name + "_tensor": value for name, value in tensors.items()}, workspace=workspace)
    torch.cuda.synchronize()
    case = SimpleNamespace(tensors=tensors, expected=expected, dt=dt)
    _check_case(case, allow_zero=True)
    assert bool((tensors["dq"][:, :, sk + left :].float() == 0).all()), "query rows past every key must hold exact zeros"


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_pair_tail_keys_with_negative_lse(dt):
    """A tail KV tile whose zero-filled key rows meet a very negative LSE: the tail columns are masked before the exponent."""
    _require_pair()
    from frost_test_utils import select_engine
    from test_sdpa_bwd_d128_sm100 import _ENGINE
    from test_sdpa_bwd_dsl_sm107 import _bshd_empty, _build_graph, _check

    b, h, sq, sk = 1, 1, 65, 129
    gen = torch.Generator(device="cpu").manual_seed(731)
    q = torch.full((b, sq, h, 128), 3.0).to(device="cuda", dtype=dt).permute(0, 2, 1, 3)
    k = (torch.full((b, sk, h, 128), -4.0) + torch.randn(b, sk, h, 128, generator=gen)).to(device="cuda", dtype=dt).permute(0, 2, 1, 3)
    v = torch.randn(b, sk, h, 128, generator=gen).to(device="cuda", dtype=dt).permute(0, 2, 1, 3)
    do = torch.randn(b, sq, h, 128, generator=gen).to(device="cuda", dtype=dt).permute(0, 2, 1, 3)
    o64, lse, _, *expected = _reference64(q, k, v, do)
    assert lse.max().item() < -100
    tensors = dict(q=q, k=k, v=v, do=do, o=_bshd_empty(b, sq, h, 128, dt).copy_(o64), stats=lse.float().unsqueeze(-1).contiguous())
    tensors.update(
        dq=_bshd_empty(b, sq, h, 128, dt, fill=float("nan")),
        dk=_bshd_empty(b, sk, h, 128, dt, fill=float("nan")),
        dv=_bshd_empty(b, sk, h, 128, dt, fill=float("nan")),
    )
    graph, refs, outputs = _build_graph(b=b, hq=h, sq=sq, skv=sk, d=128, dt=dt, use_causal_mask=False, scale="default", use_deterministic_algorithm=True)
    select_engine(graph, _ENGINE)
    graph.check_support()
    graph.build_plans()
    assert graph._compiled_plans[graph._plan_index]._prepared.spec.artifact.implementation == "pair"
    refs.update(zip(("dq", "dk", "dv"), outputs))
    workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda")
    graph.execute({refs[name]: value for name, value in tensors.items()}, workspace)
    for name, want in zip(("dq", "dk", "dv"), expected):
        _check(name, tensors[name], want, dt)


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("d,causal", [(128, True), (64, False)], ids=["d128-causal", "d64-dense"])
def test_pair_three_launches_without_allocation_or_sync(d, causal):
    _require_pair()
    case = _case(dt=torch.bfloat16, d=d, causal=causal, b=2, h=3, sq=257, sk=385, deterministic=True, accumulation=None, check_build=True)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.set_sync_debug_mode("error")
    try:
        case.graph.execute(case.pack, case.workspace)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.cuda.memory_allocated() == before
    names = cuda_launch_names(lambda: case.graph.execute(case.pack, case.workspace))
    if names is None:
        pytest.skip("the profiler records no CUDA activity on this device")
    assert len(names[0]) == 3, names[0]
    for name, expected in zip(names[0], ("dot_do_o_kernel", "bprop_d128_accum_kernel", "bprop_d128_dq_kernel")):
        assert expected in name, names[0]
    _check_case(case)


@requires_pre_rubin_blackwell
@pytest.mark.parametrize(
    "dt,d,causal,left,sq,sk", [(torch.bfloat16, 128, True, None, 1024, 1153), (torch.float16, 64, True, 128, 640, 640)], ids=["d128-causal", "d64-w128"]
)
def test_pair_bitwise_across_runs_and_poisoned_workspace(dt, d, causal, left, sq, sk):
    """Repeated eager execution over a poisoned workspace and poisoned outputs returns the same raw bits."""
    _require_pair()
    case = _case(dt=dt, d=d, causal=causal, left=left, b=2, h=3, sq=sq, sk=sk, deterministic=True, accumulation=None)
    expected_bits = {name: case.tensors[name].clone() for name in ("dq", "dk", "dv")}
    for fill in (0xA5, 0x00, 0xFF):
        case.workspace.fill_(fill)
        for name in expected_bits:
            case.tensors[name].fill_(float("nan"))
        case.graph.execute(case.pack, case.workspace)
        torch.cuda.synchronize()
        for name, expected in expected_bits.items():
            assert torch.equal(case.tensors[name], expected), f"{name} changed across identical deterministic executions"


@requires_pre_rubin_blackwell
def test_pair_rebind_and_replay():
    """Fresh operands through the same plan, then a CUDA Graph capture replayed with changed inputs."""
    _require_pair()
    case = _case(dt=torch.bfloat16, d=128, causal=True, b=2, h=3, sq=640, sk=640, deterministic=True, accumulation=None)
    for _ in range(2):
        fresh = {name: torch.empty_like(tensor) for name, tensor in case.tensors.items()}
        for name in ("q", "k", "v", "do"):
            fresh[name].normal_()
        o, lse, _, *expected = _reference64(*(fresh[name] for name in ("q", "k", "v", "do")), case.keep)
        fresh["o"].copy_(o)
        fresh["stats"].copy_(lse.unsqueeze(-1))
        for name in ("dq", "dk", "dv"):
            fresh[name].fill_(float("nan"))
        case.tensors, case.expected = fresh, expected
        case.pack = {case.refs[name]: tensor for name, tensor in fresh.items()}
        case.graph.execute(case.pack, case.workspace)
        _check_case(case)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            case.graph.execute(case.pack, case.workspace)
    for name in ("q", "k", "v", "do"):
        case.tensors[name].normal_()
    o, lse, _, *expected = _reference64(*(case.tensors[name] for name in ("q", "k", "v", "do")), case.keep)
    case.tensors["o"].copy_(o)
    case.tensors["stats"].copy_(lse.unsqueeze(-1))
    for name in ("dq", "dk", "dv"):
        case.tensors[name].fill_(float("nan"))
    case.expected = expected
    graph.replay()
    torch.cuda.synchronize()
    _check_case(case)
    bits = {name: case.tensors[name].clone() for name in ("dq", "dk", "dv")}
    for name in bits:
        case.tensors[name].fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    for name, expected in bits.items():
        assert torch.equal(case.tensors[name], expected), f"{name} changed across identical replays"
