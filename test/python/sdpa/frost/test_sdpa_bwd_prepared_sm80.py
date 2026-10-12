# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM80 prepared backward contracts, native layouts and physical wide addresses."""

import ctypes
from types import SimpleNamespace

import cudnn
import pytest
import torch

from frost_test_utils import requires_dsl, select_engine
from sdpa.frost.test_sdpa_bwd_dsl_sm120 import _ref_bwd

pytestmark = [requires_dsl, pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0), reason="requires SM80")]


@pytest.fixture(autouse=True)
def _enable_frost(monkeypatch):
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")


def _view(b, h, s, d, dtype, layout):
    if layout == "bhsd":
        return torch.randn(b, h, s, d, dtype=dtype, device="cuda") * 0.3
    if layout == "sbhd":
        return (torch.randn(s, b, h, d, dtype=dtype, device="cuda") * 0.3).permute(1, 2, 0, 3)
    return (torch.randn(b, s, h, d + (8 if layout == "gapped" else 0), dtype=dtype, device="cuda") * 0.3)[..., :d].transpose(1, 2)


def _reference(bufs, scale, causal, padding):
    # The canonical helper returns a batch-reduced dBias. Run its independent
    # references per batch when the declaration requests separate dBias planes.
    if bufs.get("bias") is not None and bufs["bias"].shape[0] > 1:
        parts = []
        for batch in range(bufs["q"].shape[0]):
            inputs = {name: bufs[name][batch : batch + 1] for name in ("q", "k", "v", "do", "bias")}
            if "sink" in bufs:
                inputs["sink"] = bufs["sink"]
            lengths = tuple(values[batch : batch + 1] for values in padding) if padding is not None else None
            parts.append(_reference(inputs, scale, causal, lengths))
        return (
            *(torch.cat([part[i] for part in parts]) for i in range(5)),
            SimpleNamespace(dbias=torch.cat([part[5].dbias for part in parts]), dsink=sum(part[5].dsink for part in parts)),
        )
    return _ref_bwd(
        *(bufs[n] for n in ("q", "k", "v", "do")), scale=scale, is_causal=causal, bias=bufs.get("bias"), sink_token=bufs.get("sink"), padding=padding
    )


def _wide_buffer(src, *, product, axis):
    strides = list(src.stride())
    strides[axis] += 2**30 if product else 2**32
    origin = 2**31 if product else 0
    elements = 1 + sum((n - 1) * st for n, st in zip(src.shape, strides))
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    if (elements + origin) * src.element_size() + 512 * 2**20 > free:
        pytest.skip("physical Int64 probe needs one guarded wide allocation")
    try:
        backing = torch.empty(elements + origin, device="cuda", dtype=src.dtype)
    except torch.OutOfMemoryError:
        pytest.skip("another GPU worker consumed the guarded allocation capacity")
    if product:
        shape = list(src.shape)
        shape[axis] = 1
        for i in range(src.shape[axis]):
            backing.as_strided(shape, src.stride(), origin + ctypes.c_int32(i * strides[axis]).value).fill_(float("nan"))
    else:
        backing.as_strided(src.shape, src.stride()).fill_(float("nan"))
    return backing.as_strided(src.shape, strides, origin).copy_(src), backing


def _case(
    d=128,
    dv=None,
    *,
    hk=2,
    dtype=torch.bfloat16,
    layout="gapped",
    causal=True,
    features=False,
    deterministic=False,
    wide=None,
    product=False,
    axis=0,
    sq=128,
    skv=128,
    padding=None,
    bias_batch=1,
    scale=None,
):
    dv = d if dv is None else dv
    scale = d**-0.5 if scale is None else scale
    b, h = (5 if product else 2), 4
    if wide is not None and axis == 2:
        b, sq, skv = 1, (5 if product else 2), (5 if product else 2)
    torch.manual_seed(9128)
    bufs = {n: _view(b, heads, seq, dim, dtype, layout) for n, heads, seq, dim in (("q", h, sq, d), ("k", hk, skv, d), ("v", hk, skv, dv), ("do", h, sq, dv))}
    padding = (([sq - 7, sq // 2], [skv - 3, skv // 2]) if padding is None else padding) if features else None
    if features:
        bufs.update(bias=torch.randn(bias_batch, h, sq, skv, device="cuda") * 0.1, sink=torch.randn(1, h, 1, 1, device="cuda"))
        bufs.update({role: torch.tensor(values, dtype=torch.int32, device="cuda").reshape(b, 1, 1, 1) for role, values in zip(("seq_q", "seq_kv"), padding)})
    o, stats, dq, dk, dv_ref, aux = _reference(bufs, scale, causal, padding)
    bufs["o"] = _view(b, h, sq, dv, dtype, "sbhd").copy_(o)
    bufs["stats"] = (
        stats
        if d == 64 and hk == h and not causal and not features and not deterministic
        else torch.empty(b, h, sq * 2, 1, device="cuda")[:, :, ::2].copy_(stats)
    )
    expected = dict(dq=dq, dk=dk, dv=dv_ref)
    for name, src in (("dq", "q"), ("dk", "k"), ("dv", "v")):
        bufs[name] = _view(*bufs[src].shape, dtype, "bhsd").fill_(float("nan"))
    if features:
        bufs.update(dbias=torch.empty_like(bufs["bias"], dtype=dtype), dsink=torch.empty_like(bufs["sink"]))
        expected.update(dbias=aux.dbias, dsink=aux.dsink)
    backing = None
    if wide is not None:
        bufs[wide], backing = _wide_buffer(bufs[wide], product=product, axis=axis)
    io = cudnn.data_type.BFLOAT16 if dtype == torch.bfloat16 else cudnn.data_type.HALF
    graph = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    refs = {name: graph.tensor_like(buf, name=name) for name, buf in bufs.items() if name not in ("dq", "dk", "dv")}
    kwargs = dict(
        q=refs["q"],
        k=refs["k"],
        v=refs["v"],
        o=refs["o"],
        dO=refs["do"],
        stats=refs["stats"],
        attn_scale=scale,
        use_causal_mask=causal,
        use_deterministic_algorithm=deterministic,
    )
    if features:
        kwargs.update(
            bias=refs["bias"],
            dBias=refs["dbias"],
            sink_token=refs["sink"],
            dSink_token=refs["dsink"],
            use_padding_mask=True,
            seq_len_q=refs["seq_q"],
            seq_len_kv=refs["seq_kv"],
        )
    for name, ref in zip(("dq", "dk", "dv"), graph.sdpa_backward(**kwargs)):
        ref.set_output(True).set_dim(bufs[name].shape).set_stride(bufs[name].stride()).set_data_type(io)
        refs[name] = ref
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    select_engine(graph, "sdpa_bwd_sm80")
    graph.check_support()
    graph.build_plans()
    workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda").fill_(0xBD)
    pack = {refs[name]: buf for name, buf in bufs.items()}
    graph.execute(pack, workspace)
    return SimpleNamespace(
        graph=graph,
        workspace=workspace,
        pack=pack,
        refs=refs,
        bufs=bufs,
        expected=expected,
        backing=backing,
        dtype=dtype,
        scale=scale,
        causal=causal,
        features=features,
        padding=padding,
    )


def _check(case, bufs=None, expected=None):
    bufs = case.bufs if bufs is None else bufs
    for name, ref in (case.expected if expected is None else expected).items():
        torch.testing.assert_close(bufs[name].float(), ref.float(), atol=0.03, rtol=0.03)
        error = torch.linalg.vector_norm(bufs[name].float() - ref.float())
        bound = 0.03 * torch.linalg.vector_norm(ref.float()) + 1e-7
        assert error <= bound, f"{name}: relative gradient error exceeds 3 percent"


@pytest.mark.L0
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("d,dv,hk,causal", [(64, 64, 4, False), (128, 128, 2, True), (192, 128, 1, True), (256, 256, 2, False)])
@pytest.mark.parametrize("layout", ["bshd", "bhsd", "gapped", "sbhd"])
def test_native_layouts(dtype, d, dv, hk, causal, layout):
    _check(_case(d, dv, hk=hk, causal=causal, dtype=dtype, layout=layout))


@pytest.mark.L0
@pytest.mark.parametrize(
    "d,hk,causal,features,deterministic",
    [(64, 4, False, False, False), (128, 2, True, True, False), (192, 2, True, False, True), (256, 4, False, False, False)],
)
def test_rebind_stream_capture(d, hk, causal, features, deterministic):
    case = _case(d, 128 if d == 192 else d, hk=hk, causal=causal, features=features, deterministic=deterministic)
    _check(case)
    bufs = {name: torch.empty_strided(buf.shape, buf.stride(), device=buf.device, dtype=buf.dtype).copy_(buf) for name, buf in case.bufs.items()}
    bufs["q"].mul_(0.75)
    bufs["do"].mul_(1.25)
    if features:
        bufs["sink"].add_(0.25)
        bufs["bias"].mul_(0.5)
    o, stats, dq, dk, dv, aux = _reference(bufs, case.scale, causal, case.padding)
    bufs["o"].copy_(o)
    bufs["stats"].copy_(stats)
    expected = dict(dq=dq, dk=dk, dv=dv)
    if features:
        expected.update(dbias=aux.dbias, dsink=aux.dsink)
    pack = {case.refs[name]: buf for name, buf in bufs.items()}
    workspace = torch.empty_like(case.workspace).fill_(0xBD)
    stream, other = torch.cuda.Stream(), torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    other.wait_stream(torch.cuda.current_stream())
    handle = cudnn.create_handle()
    cudnn.set_stream(handle, stream.cuda_stream)
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(other):
            case.graph.execute(pack, workspace, handle=handle)
        torch.cuda.current_stream().wait_stream(stream)
        _check(case, bufs, expected)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(capture, stream=stream):
            with torch.cuda.stream(other):
                case.graph.execute(pack, workspace, handle=handle)
        for name in expected:
            bufs[name].fill_(float("nan"))
        workspace.fill_(0xBD)
        capture.replay()
        _check(case, bufs, expected)
    finally:
        capture.reset()
        cudnn.destroy_handle(handle)


@pytest.mark.L0
@pytest.mark.parametrize("d,hk,causal,features", [(64, 4, False, False), (128, 2, True, True)])
def test_no_tensor_plumbing_or_warm_jit(d, hk, causal, features, monkeypatch):
    import cutlass.cute as cute
    from cudnn.sdpa.fwd.api_dsl import WorkspaceCarver

    case = _case(d, hk=hk, causal=causal, features=features)

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared backward rebuilt tensors, allocated or compiled")

    with monkeypatch.context() as p:
        for name in ("view", "reshape", "as_strided", "transpose", "contiguous", "copy_", "zero_", "repeat_interleave"):
            p.setattr(torch.Tensor, name, forbidden)
        for name in ("empty", "empty_like", "zeros", "zeros_like"):
            p.setattr(torch, name, forbidden)
        p.setattr(cute, "compile", forbidden)
        p.setattr(WorkspaceCarver, "__init__", forbidden)
        torch.cuda.set_sync_debug_mode("error")
        try:
            case.graph.execute(case.pack, case.workspace)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    _check(case)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("role", ["q", "k", "v", "o", "do", "stats", "dq", "dk", "dv"])
@pytest.mark.parametrize("d,hk,causal", [(64, 4, False), (128, 2, True)])
@pytest.mark.parametrize("product", [False, True])
def test_physical_batch_stride(role, d, hk, causal, product):
    case = _case(d, hk=hk, causal=causal, wide=role, product=product)
    assert (case.bufs[role].shape[0] - 1) * case.bufs[role].stride(0) > 2**32
    _check(case)
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            case.graph.execute(case.pack, case.workspace)
        for name in case.expected:
            case.bufs[name].fill_(float("nan"))
        case.workspace.fill_(0xBD)
        capture.replay()
        _check(case)
    finally:
        capture.reset()


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("role", ["q", "k", "v", "o", "do", "stats", "dq", "dk", "dv"])
@pytest.mark.parametrize("product", [False, True])
def test_physical_row_stride(role, product):
    _check(_case(128, hk=2, wide=role, product=product, axis=2))


def _adapter(case):
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm80

    api = SdpaBwdDslSm80(
        **{"sample_" + name: buf for name, buf in case.bufs.items() if name not in ("seq_q", "seq_kv")},
        is_causal=case.causal,
        scale_softmax=case.scale,
        has_bias=case.features,
        bias_is_fp32=True,
        bias_batch=1,
        seq_q_lens_present=case.features,
        seq_kv_lens_present=case.features,
    )
    api.check_support()
    api.compile()
    assert api._prepared is not None
    return api


def _execute_adapter(api, bufs, workspace):
    names = {"seq_q": "seq_q_lens", "seq_kv": "seq_kv_lens"}
    api.execute(**{names.get(name, name + "_tensor"): buf for name, buf in bufs.items()}, workspace=workspace)


@pytest.mark.L0
@pytest.mark.parametrize("d,hk,causal,features", [(64, 4, False, False), (128, 2, True, True)])
def test_standalone_first_execute_is_prepared(d, hk, causal, features, monkeypatch):
    import cutlass.cute as cute

    case = _case(d, hk=hk, causal=causal, features=features)
    api = _adapter(case)
    assert api._use_d64 == (d == 64)
    assert api.scratch_workspace_bytes() == case.workspace.numel()

    def forbidden(*args, **kwargs):
        raise AssertionError("first standalone execute attempted compilation or a tensor view")

    with monkeypatch.context() as p:
        p.setattr(cute, "compile", forbidden)
        for name in ("view", "reshape", "transpose", "as_strided", "contiguous"):
            p.setattr(torch.Tensor, name, forbidden)
        _execute_adapter(api, case.bufs, case.workspace)
    _check(case)


@pytest.mark.L0
@pytest.mark.parametrize(
    "kind",
    [
        "missing_workspace",
        "short_workspace",
        "workspace_overlap",
        "missing_length",
        "extra_sink",
        "wrong_dtype",
        "cpu_length",
        "changed_layout",
        "short_storage",
        "misaligned_q",
    ],
)
def test_invalid_binding_fails_before_launch(kind):
    from dataclasses import replace
    from cudnn.sdpa.bwd.prepared import ROLES, execute
    from cudnn.sdpa.fwd.prepared import facts_of_tensor

    case = _case(features=True)
    api = _adapter(case)
    facts = {name: facts_of_tensor(case.bufs.get(name)) for name in ROLES}
    called = []
    spec = replace(api._prepared, fn=lambda *args: called.append(args))
    workspace_ptr = case.workspace.data_ptr()
    if kind == "missing_workspace":
        workspace_ptr = 0
    elif kind == "workspace_overlap":
        workspace_ptr = facts["q"].ptr
    elif kind == "missing_length":
        facts["seq_q"] = None
    elif kind == "extra_sink":
        ops = list(spec.operands)
        ops[11] = None
        spec = replace(spec, operands=tuple(ops))
    elif kind == "wrong_dtype":
        facts["q"] = facts["q"]._replace(dtype="float32")
    elif kind == "cpu_length":
        facts["seq_q"] = facts["seq_q"]._replace(device=(1, 0))
    elif kind == "short_storage":
        facts["q"] = facts["q"]._replace(span=1)
    elif kind == "misaligned_q":
        facts["q"] = facts["q"]._replace(ptr=facts["q"].ptr + 2)
    elif kind == "changed_layout":
        shape = facts["q"].shape
        strides = list(facts["q"].strides)
        strides[1], strides[2] = strides[2], strides[1]
        facts["q"] = facts["q"]._replace(strides=tuple(strides))
    if kind == "short_workspace":
        api._prepared = spec
        with pytest.raises(ValueError, match="workspace"):
            _execute_adapter(api, case.bufs, case.workspace[:1])
    else:
        geometry = tuple((op.shape, op.strides) if op is not None else None for op in spec.operands)
        with pytest.raises(ValueError):
            execute(spec, facts, workspace_ptr, torch.cuda.current_stream().cuda_stream, geometry=geometry)
    assert not called


@pytest.mark.L0
@pytest.mark.parametrize("role", ["q", "do", "stats", "dq", "dk", "dv"])
@pytest.mark.parametrize("ordered", [False, True])
def test_raw_storage_and_explicit_override(role, ordered, monkeypatch):
    from dataclasses import replace

    case = _case(layout="bshd")
    tensor = case.bufs[role]
    span = 1 + sum((n - 1) * st for n, st in zip(tensor.shape, tensor.stride()))
    backing = torch.empty(span + 2, dtype=tensor.dtype, device="cuda")
    declared = backing.as_strided(tensor.shape, tensor.stride()).copy_(tensor)
    case.pack[case.refs[role]] = backing[::2]
    case.bufs[role] = declared
    pack, kwargs = case.pack, {}
    if ordered:
        items = list(reversed(list(pack.items())))
        kwargs["tensor_uids"] = [ref.get_uid() for ref, _ in items]
        pack = [buf for _, buf in items]
    case.graph.execute(pack, case.workspace, **kwargs)
    _check(case)
    ref = case.refs[role]
    kwargs.update(override_uids=[ref.get_uid()], override_shapes=[list(ref.get_dim())], override_strides=[list(ref.get_stride())])
    case.graph.execute(pack, case.workspace, **kwargs)
    _check(case)
    plan = case.graph._compiled_plans[case.graph._plan_index]
    launches = []
    monkeypatch.setattr(plan._prepared, "spec", replace(plan._prepared.spec, fn=lambda *args: launches.append(args)))
    kwargs["override_shapes"][0][2] //= 2
    with pytest.raises(ValueError, match="runtime geometry"):
        case.graph.execute(pack, case.workspace, **kwargs)
    assert not launches


@pytest.mark.L0
@pytest.mark.parametrize("role", ["o", "do"])
def test_wrapper_cache_tracks_current_port_geometry(role, monkeypatch):
    from cuda.bindings import driver
    from cudnn.sdpa.bwd import api_dsl

    case = _case(features=True)
    monkeypatch.setattr(api_dsl, "_sm80_bwd_cache", {})
    stream, other = torch.cuda.Stream(), torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    other.wait_stream(torch.cuda.current_stream())
    for change in (False, True):
        if change:
            case.bufs[role] = case.bufs[role].contiguous()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(other):
            result = api_dsl.sdpa_bwd_wrapper_sm80(
                *(case.bufs[n] for n in ("q", "k", "v", "o", "do", "stats")),
                is_causal=case.causal,
                scale_softmax=case.scale,
                current_stream=driver.CUstream(stream.cuda_stream),
                seq_len_q=case.bufs["seq_q"],
                seq_kv_lens=case.bufs["seq_kv"],
                bias_tensor=case.bufs["bias"],
                sinks=case.bufs["sink"],
            )
        torch.cuda.current_stream().wait_stream(stream)
        _check(case, {name: result[name + "_tensor"].reshape(ref.shape) for name, ref in case.expected.items()})
        assert len(api_dsl._sm80_bwd_cache) == (2 if change else 1)
        assert all(api._prepared is not None for api in api_dsl._sm80_bwd_cache.values())


@pytest.mark.L0
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_zero_scale_with_bias(dtype, causal):
    # Bias enters after the scale, so attn_scale = 0 gives uniform P over the bias: dQ = dK = 0, finite dV/dBias.
    _check(_case(128, dtype=dtype, causal=causal, features=True, scale=0.0))


@pytest.mark.L0
@pytest.mark.parametrize("d", [64, 256])
@pytest.mark.parametrize("bias_batch", [1, 2])
def test_partial_tiles_empty_padding_and_bias_batch(d, bias_batch):
    _check(_case(d, sq=33, skv=65, hk=2, features=True, padding=([0, 27], [59, 0]), bias_batch=bias_batch))
    if bias_batch == 2:
        _check(_case(d, sq=33, skv=65, hk=2, features=True, bias_batch=bias_batch))


@pytest.mark.L0
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "d,dv,bias_batch,staged", [(48, 32, 1, True), (96, 80, 2, True), (64, 64, 1, False), (128, 128, 2, False), (192, 128, 1, False), (256, 256, 2, False)]
)
def test_wrapper_aux_outputs_need_no_torch_clear(dtype, d, dv, bias_batch, staged, monkeypatch):
    from cudnn.sdpa.bwd import api_dsl

    case = _case(d, dv, dtype=dtype, sq=33, skv=65, features=True, padding=([27, 0], [59, 31]), bias_batch=bias_batch)
    monkeypatch.setattr(api_dsl, "_sm80_bwd_cache", {})

    def run():
        return api_dsl.sdpa_bwd_wrapper_sm80(
            *(case.bufs[name] for name in ("q", "k", "v", "o", "do", "stats")),
            is_causal=case.causal,
            scale_softmax=case.scale,
            seq_len_q=case.bufs["seq_q"],
            seq_kv_lens=case.bufs["seq_kv"],
            bias_tensor=case.bufs["bias"],
            sinks=case.bufs["sink"],
        )

    def check(result):
        _check(case, {name: result[name + "_tensor"].reshape(ref.shape) for name, ref in case.expected.items()})

    def refresh_reference():
        o, stats, dq, dk, dv_ref, aux = _reference(case.bufs, case.scale, case.causal, case.padding)
        case.bufs["o"].copy_(o)
        case.bufs["stats"].copy_(stats)
        case.expected = dict(dq=dq, dk=dk, dv=dv_ref, dbias=aux.dbias, dsink=aux.dsink)

    check(run())
    plan = next(iter(api_dsl._sm80_bwd_cache.values()))
    assert (plan._staged_prepared is not None) == staged
    assert (plan._prepared is not None) != staged
    case.bufs = {name: torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device).copy_(t) for name, t in case.bufs.items()}
    case.bufs["q"].mul_(0.75)
    case.bufs["do"].mul_(1.25)
    case.bufs["bias"].mul_(0.5)
    case.bufs["sink"].add_(0.25)
    refresh_reference()
    empty, empty_like = torch.empty, torch.empty_like
    zeros, zeros_like = torch.zeros, torch.zeros_like
    poisoned = set()

    def poison_sink(*args, **kwargs):
        result = empty(*args, **kwargs)
        if result.dtype == torch.float32 and result.shape == (case.bufs["q"].shape[1],):
            result.fill_(float("nan"))
            poisoned.add("sink")
        return result

    def poison_bias(tensor, **kwargs):
        result = empty_like(tensor, **kwargs)
        if tensor is case.bufs["bias"]:
            result.fill_(float("nan"))
            poisoned.add("bias")
        return result

    monkeypatch.setattr(torch, "empty", poison_sink)
    monkeypatch.setattr(torch, "empty_like", poison_bias)
    for name in ("zeros", "zeros_like"):
        monkeypatch.setattr(torch, name, lambda *a, **k: pytest.fail("wrapper redundantly cleared auxiliary outputs"))
    check(run())
    assert poisoned == {"bias", "sink"}
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = run()
        # Previously active rows become entirely masked. The prepared chain
        # must overwrite every auxiliary output, including the partial tiles.
        case.padding = ([0, 27], [59, 0])
        for name, values in zip(("seq_q", "seq_kv"), case.padding):
            case.bufs[name].copy_(torch.tensor(values, dtype=torch.int32, device="cuda").reshape_as(case.bufs[name]))
        # The reference may legitimately allocate zeros; only the wrapper's
        # allocating call is forbidden from rebuilding those clears.
        with monkeypatch.context() as guard:
            guard.setattr(torch, "zeros", zeros)
            guard.setattr(torch, "zeros_like", zeros_like)
            refresh_reference()
        graph.replay()
        check(captured)
    finally:
        graph.reset()
