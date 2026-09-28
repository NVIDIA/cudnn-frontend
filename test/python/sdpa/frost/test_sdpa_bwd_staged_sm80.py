# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The retained SM80 staging paths execute prepared pointer chains."""

import pytest
import torch

from frost_test_utils import requires_dsl
from sdpa.frost.test_sdpa_bwd_prepared_sm80 import _case, _check, _reference
from sdpa.frost.test_sdpa_bwd_thd_sm80 import _run_graph, _check as _check_thd, _thd_case

pytestmark = [
    requires_dsl,
    pytest.mark.L0,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0), reason="requires SM80"),
]


@pytest.fixture(autouse=True)
def _pointer_only(monkeypatch):
    import cutlass.cute as cute

    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")

    def forbidden(*args, **kwargs):
        pytest.fail("staged backward reached tensor-fake construction")

    monkeypatch.setattr(cute.runtime, "make_fake_tensor", forbidden)
    monkeypatch.setattr(cute.runtime, "make_fake_compact_tensor", forbidden)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "d,dv,hk,features,deterministic", [(48, 32, 4, False, False), (96, 80, 2, True, False), (160, 112, 1, False, True), (224, 208, 2, False, False)]
)
def test_dense_staged_rebind_and_replay(dtype, d, dv, hk, features, deterministic):
    case = _case(d, dv, dtype=dtype, hk=hk, features=features, deterministic=deterministic)
    _check(case)
    bufs = {name: torch.empty_strided(buf.shape, buf.stride(), device=buf.device, dtype=buf.dtype).copy_(buf) for name, buf in case.bufs.items()}
    bufs["q"].mul_(0.7)
    bufs["do"].mul_(1.25)
    if features:
        bufs["bias"].mul_(0.5)
        bufs["sink"].add_(0.2)
    o, stats, dq, dk, dv, aux = _reference(bufs, case.scale, case.causal, case.padding)
    bufs["o"].copy_(o)
    bufs["stats"].copy_(stats)
    expected = dict(dq=dq, dk=dk, dv=dv)
    if features:
        expected.update(dbias=aux.dbias, dsink=aux.dsink)
    pack = {case.refs[name]: buf for name, buf in bufs.items()}
    workspace = torch.empty_like(case.workspace)
    case.graph.execute(pack, workspace)
    _check(case, bufs, expected)
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            case.graph.execute(pack, workspace)
        for name in expected:
            bufs[name].fill_(float("nan"))
        workspace.fill_(0xBD)
        capture.replay()
        _check(case, bufs, expected)
    finally:
        capture.reset()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("d,dv,hkv", [(96, 80, 4), (160, 112, 2), (224, 208, 1)])
@pytest.mark.parametrize("stats_layout", ["head_major", "token_major"])
def test_packed_staged_preserves_output_tail(dtype, d, dv, hkv, stats_layout):
    case, graph, pack, workspace, outputs = _run_graph(
        (64, 0, 96),
        (80, 0, 112),
        h=4,
        hkv=hkv,
        d=d,
        d_v=dv,
        dtype=dtype,
        stats_layout=stats_layout,
        poison=True,
        pad_cap=128,
        use_causal_mask=True,
        use_deterministic_algorithm=True,
    )
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            graph.execute(pack, workspace)
        changed = _thd_case((48, 32, 64), (64, 48, 80), 4, d, dtype, hkv=hkv, d_v=dv, cap_q=case.cap_q, cap_kv=case.cap_kv, poison=True, seed=23, causal=True)
        for port, tensor in pack.items():
            name = port.get_name()
            if name in ("q", "k", "v", "o", "do"):
                tensor.copy_(getattr(changed, name))
            elif name == "seq_len_q":
                tensor.copy_(torch.tensor(changed.lens_q, device="cuda", dtype=torch.int32).view_as(tensor))
            elif name == "seq_len_kv":
                tensor.copy_(torch.tensor(changed.lens_kv, device="cuda", dtype=torch.int32).view_as(tensor))
            elif name == "stats":
                if stats_layout == "token_major":
                    tensor.view(changed.cap_q, changed.h).copy_(changed.lse[0].transpose(0, 1))
                else:
                    tensor.view(1, changed.h, -1)[..., : changed.cap_q].copy_(changed.lse)
        for output in outputs:
            output.fill_(float("nan"))
        workspace.fill_(0xBD)
        capture.replay()
        _check_thd(changed, *outputs)
        for output, total in zip(outputs, (changed.t_q, changed.t_kv, changed.t_kv)):
            assert torch.isfinite(output[0, :total]).all()
            assert torch.isnan(output[0, total:]).all(), "packed capacity tail was overwritten"
    finally:
        capture.reset()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("d", [64, 128])
def test_rope_staged_replay(dtype, d):
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm80

    torch.manual_seed(417)
    q = torch.randn(2, 4, 128, d, device="cuda", dtype=dtype) * 0.2
    k, v = (torch.randn(2, 2, 128, d, device="cuda", dtype=dtype) * 0.2 for _ in range(2))
    do = torch.randn_like(q) * 0.2
    freqs = torch.arange(128, device="cuda")[:, None] * torch.linspace(0.001, 0.02, d // 2, device="cuda")[None, :]

    def reference():
        qr, kr, vr = (x.detach().double().requires_grad_() for x in (q, k, v))

        def rotate(x):
            a, b = x.chunk(2, -1)
            c, s = freqs.double().cos()[None, None], freqs.double().sin()[None, None]
            return torch.cat((a * c - b * s, b * c + a * s), -1).to(dtype).double()

        logits = rotate(qr) @ rotate(kr).repeat_interleave(2, 1).transpose(-1, -2) * d**-0.5
        logits.masked_fill_(torch.arange(128, device="cuda")[None, :] > torch.arange(128, device="cuda")[:, None], -torch.inf)
        out = logits.softmax(-1) @ vr.repeat_interleave(2, 1)
        out.backward(do.double())
        return out.detach().to(dtype), logits.detach().logsumexp(-1).float(), (qr.grad, kr.grad, vr.grad)

    o, stats, expected = reference()
    outputs = tuple(torch.empty_like(x) for x in (q, k, v))
    api = SdpaBwdDslSm80(q, k, v, o, do, stats, *outputs, has_rope=True, rope_max_s=128, is_causal=True)
    api.compile()
    assert api._staged_prepared is not None
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)

    def run():
        api.execute(q, k, v, o, do, stats, *outputs, workspace=workspace, rope_freqs=freqs)

    def check():
        for got, want in zip(outputs, expected):
            torch.testing.assert_close(got.double(), want, atol=0.006, rtol=0.04)

    run()
    check()
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            run()
        freqs.mul_(0.5)
        q.mul_(0.75)
        new_o, new_stats, expected = reference()
        o.copy_(new_o)
        stats.copy_(new_stats)
        for output in outputs:
            output.fill_(float("nan"))
        workspace.fill_(0xBD)
        capture.replay()
        check()
    finally:
        capture.reset()


def test_rope_eager_accepts_cpu_angles(monkeypatch):
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm80

    original = SdpaBwdDslSm80.execute
    observed = []

    def execute(self, *args, **kwargs):
        if not torch.cuda.is_current_stream_capturing():
            kwargs["rope_freqs"] = kwargs["rope_freqs"].cpu()
            observed.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SdpaBwdDslSm80, "execute", execute)
    test_rope_staged_replay(torch.float16, 128)
    assert observed


def test_staged_singleton_strides_keep_existing_copy_set():
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm80

    q = torch.empty(1, 1, 128, 128, device="cuda", dtype=torch.float16).as_strided((1, 1, 128, 128), (7, 3, 128, 1))
    stats = torch.empty(1, 1, 128, device="cuda", dtype=torch.float32)
    api = SdpaBwdDslSm80(q, q, q, q, q, stats, q, q, q, has_rope=True, rope_max_s=128)
    api.check_support()
    assert q.transpose(1, 2).is_contiguous()
    assert [role for role, _, _ in api._staged_layout.copies] == ["dq"]


def test_sm80_direct_adapter_declines_old_dsl(monkeypatch):
    from cudnn.frost import buffers
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm80

    q = torch.empty(1, 1, 128, 96, device="cuda", dtype=torch.float16)
    stats = torch.empty(1, 1, 128, device="cuda", dtype=torch.float32)
    api = SdpaBwdDslSm80(q, q, q, q, q, stats, q, q, q)
    monkeypatch.setattr(buffers, "cutedsl_requirement_error", lambda name: f"{name} requires nvidia-cutlass-dsl >= 4.7.0; found 4.6.2")
    with pytest.raises(NotImplementedError, match=r"SdpaBwdDslSm80.*4\.7\.0.*4\.6\.2"):
        api.check_support()


@pytest.mark.parametrize("features", [False, True])
def test_staged_workspace_slice_with_odd_extra_byte(features):
    case = _case(96, 80, hk=2, features=features)
    size = case.workspace.numel()
    owner = torch.full((size + 513,), 0xD7, device="cuda", dtype=torch.uint8)
    workspace = owner[256 : 256 + size + 1]

    def check():
        _check(case)
        assert (owner[:256] == 0xD7).all()
        assert (owner[256 + size :] == 0xD7).all()

    case.graph.execute(case.pack, workspace)
    check()
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            case.graph.execute(case.pack, workspace)
        workspace[:size].fill_(0xBD)
        for role in ("dq", "dk", "dv"):
            case.bufs[role].fill_(float("nan"))
        capture.replay()
        check()
    finally:
        capture.reset()


@pytest.mark.parametrize("role", ["dbias", "dsink"])
@pytest.mark.parametrize("problem", ["size", "shape", "unexpected", "missing", "dtype"])
def test_staged_auxiliary_outputs_validate_before_writes(role, problem, monkeypatch):
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm80

    q = torch.zeros(2, 4, 128, 96, device="cuda", dtype=torch.float16)
    v = torch.zeros(2, 4, 128, 80, device="cuda", dtype=torch.float16)
    stats = torch.zeros(2, 4, 128, device="cuda", dtype=torch.float32)
    outputs = (torch.empty_like(q), torch.empty_like(q), torch.empty_like(v))
    enabled = problem != "unexpected"
    bias = torch.zeros(1, 4, 128, 128, device="cuda") if enabled and role == "dbias" else None
    sink = torch.zeros(4, device="cuda") if enabled and role == "dsink" else None
    aux = torch.empty_like(bias if role == "dbias" else sink) if enabled else torch.empty(4, device="cuda")
    api = SdpaBwdDslSm80(
        q,
        q,
        v,
        v,
        v,
        stats,
        *outputs,
        sample_bias=bias,
        sample_sink=sink,
        has_bias=bias is not None,
        bias_is_fp32=True,
        **{"sample_" + role: aux if enabled else None},
    )
    api.compile()
    assert api._staged_prepared is not None
    workspace = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")
    if problem == "size":
        aux = torch.empty(aux.numel() + 1, device="cuda", dtype=aux.dtype)
    elif problem == "shape":
        aux = aux.transpose(1, 2) if role == "dbias" else aux.view(2, 2)
    elif problem == "missing":
        aux = None
    elif problem == "dtype":
        aux = aux.to(torch.int32)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid auxiliary output reached a staging write or launch")

    with monkeypatch.context() as guards:
        guards.setattr(torch.Tensor, "copy_", forbidden)
        guards.setattr(torch.Tensor, "zero_", forbidden)
        from cudnn.sdpa.bwd import staged_sm80

        guards.setattr(staged_sm80, "_copy", forbidden)
        from dataclasses import replace

        guards.setattr(api, "_staged_prepared", replace(api._staged_prepared, fn=forbidden))
        with pytest.raises(ValueError, match=role):
            api.execute(q, q, v, v, v, stats, *outputs, workspace=workspace, bias_tensor=bias, sink_tensor=sink, **{role + "_tensor": aux})
