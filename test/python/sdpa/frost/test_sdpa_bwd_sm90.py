# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hopper dense d64 / d128 backward: the key-owner kernel with the ordered dQ relay, its selection, workspace, numerics and replay."""

from types import SimpleNamespace

import pytest
import torch

from frost_test_utils import cuda_launch_names, requires_dsl, select_engine
from test_sdpa_bwd_dsl_sm107 import _bshd_empty, _build_graph, _causal_keep, _check, _reference64

pytestmark = [pytest.mark.L0, requires_dsl]
_ENGINE = "sdpa_bwd_sm90"
requires_hopper = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="requires SM90 (cc 9.0)")


def test_geometry():
    from cudnn.sdpa.bwd.kernels.sm90 import bprop_f16 as kernel

    assert (kernel.HEAD_DIM, kernel.RATIO, kernel.STAGES, kernel.THREADS) == (128, 1, 3, 384)
    assert kernel.SMEM_BYTES <= 227 * 1024 and kernel.SMEM_BYTES % 1024 == 0
    assert (
        kernel.BARS
        == 2 * kernel.RESIDENT_BYTES + kernel.STAGES * 2 * kernel.STEP_BYTES + 2 * kernel.DS_BYTES + kernel.DQS_BYTES + kernel.STAGES * kernel.STATS_BYTES
    )
    assert kernel.PRODUCER_REGS * 128 + kernel.CONSUMER_REGS * 256 <= 65536
    with pytest.raises(ValueError):
        from cudnn.frost.template_loader import load_template

        load_template(kernel.__file__, (96, 1), tag="test_sm90_bad_head_dim")


def test_engine_row_and_placement(monkeypatch):
    from cudnn.engines.manifest import MANIFEST
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS
    from cudnn.sdpa.bwd.heuristics import TRAIL_ONLY_ROWS

    spec = next(s for s in ENGINE_SPECS if s.name == _ENGINE)
    caps = spec.capabilities
    assert (
        (caps.sm_lo, caps.sm_hi) == (90, 90)
        and caps.d == frozenset({64, 128})
        and caps.deterministic
        and caps.gqa
        and caps.causal
        and not caps.swa
        and not caps.thd
    )
    assert _ENGINE not in TRAIL_ONLY_ROWS
    family = next(f for f in MANIFEST if f.name == "frost_sdpa_bwd")
    assert family.slots[_ENGINE].opt_in
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    assert _ENGINE not in family.offered_ids()
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    assert _ENGINE in family.offered_ids()


@pytest.mark.parametrize("b,hq,hkv", [(65536, 1, 1), (1, 65536, 65536)], ids=["batch", "heads"])
def test_declaration_declines_grid_limits(b, hq, hkv):
    import cudnn
    from cudnn.sdpa.bwd.engines import _sm90_declaration_mismatch
    from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

    def facts(**over):
        base = dict(is_backward=True, b=b, h_q=hq, h_kv=hkv, s_q=2, s_kv=2, d_qk=64, d_v=64, dtype=cudnn.data_type.BFLOAT16, causal=True, device_cc=(9, 0))
        return SdpaGraphFacts(**{**base, **over})

    version = ("nvidia-cutlass-dsl", "4.8.0")
    assert "65535" in _sm90_declaration_mismatch(facts(), version)
    assert _sm90_declaration_mismatch(facts(b=min(b, 65535), h_q=min(hq, 65535), h_kv=min(hkv, 65535)), version) is None


def _tensors(dt, d, causal, b, hq, hkv, sq, sk, seed=729):
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def draw(s, h):
        return torch.randn(b, s, h, d, generator=gen).to(device="cuda", dtype=dt).permute(0, 2, 1, 3)

    q, k, v, do = draw(sq, hq), draw(sk, hkv), draw(sk, hkv), draw(sq, hq)
    keep = _causal_keep(sq, sk, causal=True) if causal else None
    o64, lse, _, *expected = _reference64(q, k, v, do, keep, group=hq // hkv)
    tensors = dict(q=q, k=k, v=v, do=do, o=_bshd_empty(b, sq, hq, d, dt).copy_(o64), stats=lse.float().unsqueeze(-1).contiguous())
    tensors.update(
        dq=_bshd_empty(b, sq, hq, d, dt, fill=float("nan")),
        dk=_bshd_empty(b, sk, hkv, d, dt, fill=float("nan")),
        dv=_bshd_empty(b, sk, hkv, d, dt, fill=float("nan")),
    )
    return tensors, expected, keep


def _case(dt, d, causal, b, hq, hkv, sq, sk, deterministic=True, seed=729):
    """A dense graph pinned to the Hopper chain: its three launches, its workspace, the fp64 gate."""
    from cudnn.sdpa.bwd.d128_contract import relay_workspace_layout

    tensors, expected, keep = _tensors(dt, d, causal, b, hq, hkv, sq, sk, seed)
    graph, refs, outputs = _build_graph(
        b=b, hq=hq, hkv=hkv, sq=sq, skv=sk, d=d, dt=dt, use_causal_mask=causal, scale="default", use_deterministic_algorithm=deterministic
    )
    select_engine(graph, _ENGINE)
    graph.check_support()
    graph.build_plans()
    prepared = graph._compiled_plans[graph._plan_index]._prepared
    assert prepared.spec.artifact.implementation == "relay"
    required = relay_workspace_layout(b, hq, sq, d)[1]
    assert graph.get_workspace_size() == required == prepared.spec.workspace_bytes
    refs.update(zip(("dq", "dk", "dv"), outputs))
    pack = {refs[n]: value for n, value in tensors.items()}
    workspace = torch.full((required,), 0xA5, dtype=torch.uint8, device="cuda")
    graph.execute(pack, workspace)
    torch.cuda.synchronize()
    case = SimpleNamespace(graph=graph, tensors=tensors, refs=refs, pack=pack, expected=expected, keep=keep, dt=dt, workspace=workspace)
    _check_case(case)
    return case


def _check_case(case):
    for name, want in zip(("dq", "dk", "dv"), case.expected):
        got = case.tensors[name]
        if bool((want == 0).all()):
            # an identically-zero gradient (one query over one key): the separately rounded delta leaves a residual of a few ulp
            assert bool(torch.isfinite(got.float()).all()) and got.float().abs().max().item() <= 1e-5, f"{name}: expected (near) zeros"
        else:
            _check(name, got, want, case.dt)


@requires_hopper
@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize(
    "d,causal,hq,hkv,sq,sk",
    [
        (128, False, 3, 3, 512, 512),
        (128, True, 3, 3, 512, 512),
        (128, False, 3, 3, 257, 193),
        (128, True, 3, 3, 257, 385),
        (128, True, 2, 2, 1, 2),
        (128, False, 4, 2, 384, 256),
        (128, True, 8, 1, 333, 333),
        (64, False, 3, 3, 512, 512),
        (64, True, 3, 3, 1024, 1024),
        (64, True, 4, 2, 257, 385),
        (64, False, 6, 2, 193, 257),
    ],
    ids=[
        "d128-dense",
        "d128-causal",
        "d128-dense-tail",
        "d128-causal-tail",
        "d128-one-query",
        "d128-4of2",
        "d128-mqa8-causal",
        "d64-dense",
        "d64-causal",
        "d64-4of2-causal",
        "d64-6of2-tail",
    ],
)
def test_chain_matches_fp64(dt, d, causal, hq, hkv, sq, sk):
    _case(dt, d, causal, 2, hq, hkv, sq, sk)


@requires_hopper
@pytest.mark.parametrize("deterministic", [True, False])
def test_both_requests_take_the_chain_bitwise(deterministic):
    case = _case(torch.bfloat16, 128, True, 2, 4, 2, 640, 640, deterministic=deterministic)
    expected_bits = {name: case.tensors[name].clone() for name in ("dq", "dk", "dv")}
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            case.graph.execute(case.pack, case.workspace)
    for fill, replay in ((0x5A, False), (0x00, True), (0xFF, True)):
        case.workspace.fill_(fill)
        for name in expected_bits:
            case.tensors[name].fill_(float("nan"))
        if replay:
            graph.replay()
        else:
            case.graph.execute(case.pack, case.workspace)
        torch.cuda.synchronize()
        for name, expected in expected_bits.items():
            got = case.tensors[name].contiguous().view(torch.uint8)
            assert torch.equal(got, expected.contiguous().view(torch.uint8)), f"{name} changed across identical executions"
    # the captured graph reads its buffers, not values fixed at capture: a changed dO reproduces a fresh fp64 reference
    generator = torch.Generator(device="cuda").manual_seed(731)
    case.tensors["do"].copy_((torch.randn(case.tensors["do"].shape, device="cuda", generator=generator) * 0.5).to(case.dt))
    graph.replay()
    torch.cuda.synchronize()
    _, _, _, *fresh = _reference64(case.tensors["q"], case.tensors["k"], case.tensors["v"], case.tensors["do"], case.keep, group=2)
    for name, ref in zip(("dq", "dk", "dv"), fresh):
        _check(name, case.tensors[name], ref, case.dt)


@requires_hopper
@pytest.mark.parametrize(
    "dt,d,causal,b,hq,hkv,sq,sk",
    [(torch.bfloat16, 128, False, 2, 4, 4, 512, 4096), (torch.float16, 64, True, 3, 8, 4, 1536, 1536)],
    ids=["d128-dense-256cta", "d64-causal-144cta"],
)
def test_multiwave_grid_matches_fp64_and_repeats(dt, d, causal, b, hq, hkv, sq, sk):
    assert (sk + 127) // 128 * hkv * b > torch.cuda.get_device_properties(0).multi_processor_count
    case = _case(dt, d, causal, b, hq, hkv, sq, sk)
    expected = {name: case.tensors[name].clone() for name in ("dq", "dk", "dv")}
    case.workspace.fill_(0x5A)
    for name in expected:
        case.tensors[name].fill_(float("nan"))
    case.graph.execute(case.pack, case.workspace)
    torch.cuda.synchronize()
    for name, bits in expected.items():
        assert torch.equal(
            case.tensors[name].contiguous().view(torch.uint8), bits.contiguous().view(torch.uint8)
        ), f"{name} changed across identical executions"


@requires_hopper
@pytest.mark.parametrize("dt,d,causal,hq,hkv", [(torch.bfloat16, 128, True, 4, 2), (torch.float16, 64, False, 4, 4)], ids=["d128-causal-gqa", "d64-dense-mha"])
def test_batch_invariance(dt, d, causal, hq, hkv):
    b, sq, sk = 3, 320, 257
    case = _case(dt, d, causal, b, hq, hkv, sq, sk)
    inputs = {name: case.tensors[name].clone() for name in ("q", "k", "v", "o", "do", "stats")}
    grads = {name: case.tensors[name].clone() for name in ("dq", "dk", "dv")}

    def bits(tensor):
        return tensor.contiguous().view(torch.uint8)

    for order in ([2, 0, 1], [0, 0, 2], [1, 1, 1]):
        for name, source in inputs.items():
            case.tensors[name].copy_(source[order])
        for name in grads:
            case.tensors[name].fill_(float("nan"))
        case.graph.execute(case.pack, case.workspace)
        torch.cuda.synchronize()
        for name, expected in grads.items():
            assert torch.equal(bits(case.tensors[name]), bits(expected[order])), f"{name} depends on the batch order {order}"
    # the same sequence declared alone
    graph, refs, outputs = _build_graph(
        b=1, hq=hq, hkv=hkv, sq=sq, skv=sk, d=d, dt=dt, use_causal_mask=causal, scale="default", use_deterministic_algorithm=True
    )
    select_engine(graph, _ENGINE)
    graph.check_support()
    graph.build_plans()
    refs.update(zip(("dq", "dk", "dv"), outputs))
    single = {name: _bshd_empty(1, grads[name].shape[2], grads[name].shape[1], d, dt, fill=float("nan")) for name in grads}
    pack = {refs[name]: inputs[name][1:2].clone() for name in inputs}
    pack.update({refs[name]: single[name] for name in single})
    graph.execute(pack, torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda"))
    torch.cuda.synchronize()
    for name, expected in grads.items():
        assert torch.equal(bits(single[name]), bits(expected[1:2])), f"{name} depends on the batch size"


@requires_hopper
def test_three_launches_without_allocation_or_sync():
    case = _case(torch.bfloat16, 64, True, 2, 3, 3, 257, 385)
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
    for name, expected in zip(names[0], ("dot_do_o_kernel", "bprop_sm90_kernel", "cast_dq_d128_kernel")):
        assert expected in name, names[0]


def _adapter(b, hq, hkv, s, d, **kw):
    from cudnn.sdpa.bwd.api_dsl_sm90 import SdpaBwdDslSm90

    dt = torch.bfloat16

    def empty(h):
        return _bshd_empty(b, s, h, d, dt)

    tensors = dict(
        q=empty(hq),
        k=empty(hkv),
        v=empty(hkv),
        o=empty(hq),
        do=empty(hq),
        stats=torch.zeros(b, hq, s, 1, device="cuda"),
        dq=empty(hq),
        dk=empty(hkv),
        dv=empty(hkv),
    )
    return SdpaBwdDslSm90(**{"sample_" + name: value for name, value in tensors.items()}, deterministic=True, **kw)


@requires_hopper
@pytest.mark.parametrize(
    "kw,token",
    [
        (dict(window_size_left=128), "sliding"),
        (dict(causal_bottom_right=True, is_causal=True), "bottom-right"),
        (dict(hkv=3), "integral head ratio"),
        (dict(d=256), "64 or 128"),
        (dict(scale_softmax=-0.125), "positive"),
        (dict(b=65536, hq=1, hkv=1, s=2, d=64), "65535"),
        (dict(hq=65536, hkv=65536, s=2, d=64), "65535"),
    ],
)
def test_adapter_declines_by_name(kw, token):
    kw = dict(kw)
    api = _adapter(kw.pop("b", 1), kw.pop("hq", 4), kw.pop("hkv", 4), kw.pop("s", 128), kw.pop("d", 128), **kw)
    with pytest.raises(NotImplementedError, match=token):
        api.check_support()


@requires_hopper
def test_adapter_accepts_the_grid_limit():
    _adapter(65535, 1, 1, 2, 64).check_support()
