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


def test_engine_row_and_placement():
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
    assert _ENGINE in TRAIL_ONLY_ROWS
    family = next(f for f in MANIFEST if f.name == "frost_sdpa_bwd")
    assert _ENGINE in family.slots


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
            assert torch.equal(case.tensors[name], expected), f"{name} changed across identical executions"


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


@requires_hopper
@pytest.mark.parametrize(
    "kw,token",
    [
        (dict(window_size_left=128), "sliding"),
        (dict(causal_bottom_right=True, is_causal=True), "bottom-right"),
        (dict(hkv=3), "integral head ratio"),
        (dict(d=256), "64 or 128"),
        (dict(scale_softmax=-0.125), "positive"),
    ],
)
def test_adapter_declines_by_name(kw, token):
    from cudnn.sdpa.bwd.api_dsl_sm90 import SdpaBwdDslSm90

    d = kw.pop("d", 128)
    hkv = kw.pop("hkv", 4)
    dt, b, hq, s = torch.bfloat16, 1, 4, 128

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
    api = SdpaBwdDslSm90(**{"sample_" + name: value for name, value in tensors.items()}, deterministic=True, **kw)
    with pytest.raises(NotImplementedError, match=token):
        api.check_support()
