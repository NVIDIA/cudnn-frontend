# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recomputed primal output for BF16 dGeGLU; diagnostic geometry, no timing."""

import pytest
import torch

from fe_api.grouped_gemm._workspace import ws

import cudnn
from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

pytestmark = pytest.mark.L0


def mkl(tensor):
    m, n = tensor.shape
    return tensor.as_strided((m, n, 1), (n, 1, m * n))


def problem(discrete, n=320):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10 or cutedsl_too_old(cutedsl_state()[1]):
        pytest.skip("requires Blackwell and CuTe DSL>=4.7.0")
    torch.manual_seed(461)
    m, k, e = 1024, 128, 3
    a = mkl(torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.125)
    c = mkl(torch.randn(m, 2 * n, device="cuda", dtype=torch.bfloat16) * 8)
    weights = torch.randn(e, n, k, device="cuda", dtype=torch.bfloat16) * 0.125
    b = weights.permute(1, 2, 0)
    ptrs = torch.tensor([w.data_ptr() for w in weights], device="cuda", dtype=torch.int64)
    d, baseline_d = torch.empty_like(c), torch.empty_like(c)
    activation = mkl(torch.empty(m, n, device="cuda", dtype=torch.bfloat16))
    prob = torch.rand(m, 1, 1, device="cuda")
    offsets = torch.tensor([256, 768, 768], device="cuda", dtype=torch.int32)
    alpha = torch.tensor([1.0, 0.5, 1.5], device="cuda")
    beta = torch.tensor([0.5, 1.0, 1.5], device="cuda")
    dp, baseline_dp = torch.empty_like(prob), torch.empty_like(prob)
    decl = dict(num_experts=e, b_shape=(n, k), b_dtype=torch.bfloat16) if discrete else dict(sample_b=b)
    bind = dict(b_ptrs=ptrs) if discrete else dict(b_tensor=b)
    return dict(
        a=a,
        c=c,
        weights=weights,
        b=b,
        ptrs=ptrs,
        d=d,
        baseline_d=baseline_d,
        activation=activation,
        prob=prob,
        offsets=offsets,
        alpha=alpha,
        beta=beta,
        dp=dp,
        baseline_dp=baseline_dp,
        decl=decl,
        bind=bind,
    )


def make_plan(p, vector_f32, activation, *, d=None, dp=None, act_func="dgeglu", geglu_alpha=1.0, linear_offset=0.0):
    return cudnn.GroupedGemmDgluSm100(
        p["a"],
        p["c"],
        p["d"] if d is None else d,
        None,
        None,
        p["offsets"],
        p["alpha"],
        p["beta"],
        p["prob"],
        p["dp"] if dp is None else dp,
        **p["decl"],
        act_func=act_func,
        vector_f32=vector_f32,
        geglu_alpha=geglu_alpha,
        glu_clamp_min=-10.0,
        glu_clamp_max=10.0,
        linear_offset=linear_offset,
        round_dgrad_to_input_dtype=True,
        sample_activation=activation,
    )


def reference(p, ends, *, geglu_alpha=1.0, linear_offset=0.0):
    expected = torch.empty_like(p["activation"])
    n = p["activation"].shape[1]
    begin = 0
    for expert, end in enumerate(ends):
        if end > begin:
            pair = (p["c"][begin:end, :, 0].float() * p["beta"][expert]).reshape(end - begin, n // 32, 2, 32)
            gate = pair[:, :, 0].reshape(end - begin, n).clamp(max=10)
            up = pair[:, :, 1].reshape(end - begin, n).clamp(-10, 10)
            expected[begin:end, :, 0] = (((gate * (gate * geglu_alpha).sigmoid()) * (up + linear_offset)) * p["prob"][begin:end, 0, 0, None]).bfloat16()
        begin = end
    return expected


def check_activation(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=2e-5)


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("vector_f32", [False, True])
@pytest.mark.parametrize("n", [288, 320])
def test_dglu_recomputed_activation_changed_input_graph(discrete, vector_f32, n):
    p = problem(discrete, n=n)
    plan = make_plan(p, vector_f32, p["activation"])
    baseline = make_plan(p, vector_f32, None, d=p["baseline_d"], dp=p["baseline_dp"])
    plan.compile()
    baseline.compile()
    workspaces = {plan: ws(plan), baseline: ws(baseline)}

    def run():
        for op, d, dp, aux in ((plan, p["d"], p["dp"], p["activation"]), (baseline, p["baseline_d"], p["baseline_dp"], None)):
            dp.zero_()
            op.execute(
                p["a"], p["c"], d, None, None, p["offsets"], p["alpha"], p["beta"], p["prob"], dp, **p["bind"], activation_tensor=aux, workspace=workspaces[op]
            )

    run()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    stale = None
    for step, ends in enumerate(([256, 768, 768], [0, 256, 768], [256, 512, 1024])):
        p["offsets"].copy_(torch.tensor(ends, device="cuda", dtype=torch.int32))
        p["a"].normal_(std=0.125)
        p["c"].normal_(std=8)
        p["prob"].mul_(0.75)
        p["prob"][::17] = 0
        for key in ("d", "baseline_d", "dp", "baseline_dp", "activation"):
            p[key].fill_(float("nan"))
        graph.replay()
        expected = reference(p, ends)
        actual = p["activation"][: ends[-1]]
        check_activation(actual, expected[: ends[-1]])
        assert torch.equal(actual[::17], torch.zeros_like(actual[::17]))
        torch.testing.assert_close(p["d"][: ends[-1]], p["baseline_d"][: ends[-1]], rtol=0, atol=0)
        torch.testing.assert_close(p["dp"][: ends[-1]], p["baseline_dp"][: ends[-1]], rtol=2e-5, atol=2e-5)
        assert torch.isnan(p["activation"][ends[-1] :]).all()
        if step == 1:
            for wrong in (stale, torch.zeros_like(actual), torch.full_like(actual, float("nan"))):
                with pytest.raises(AssertionError):
                    check_activation(wrong, expected[: ends[-1]])
        stale = actual.clone()


@pytest.mark.parametrize("vector_f32", [False, True])
def test_dglu_activation_survives_zero_gradient_and_alpha(vector_f32):
    p = problem(False)
    plan = make_plan(p, vector_f32, p["activation"], geglu_alpha=1.702, linear_offset=1.0)
    plan.compile()
    workspace = ws(plan)
    # The primal must be recomputed from C/prob even when derivative operands
    # cannot carry information. Also cover a nonzero linear offset.
    p["a"].zero_()
    p["alpha"].zero_()
    p["activation"].fill_(float("nan"))
    p["d"].fill_(float("nan"))
    p["dp"].zero_()
    plan.execute(
        p["a"],
        p["c"],
        p["d"],
        None,
        None,
        p["offsets"],
        p["alpha"],
        p["beta"],
        p["prob"],
        p["dp"],
        **p["bind"],
        activation_tensor=p["activation"],
        workspace=workspace,
    )
    expected = reference(p, [256, 768, 768], geglu_alpha=1.702, linear_offset=1.0)
    check_activation(p["activation"][:768], expected[:768])
    assert p["activation"][:768].abs().max() > 0
    assert torch.equal(p["d"][:768], torch.zeros_like(p["d"][:768]))
    assert torch.equal(p["dp"][:768], torch.zeros_like(p["dp"][:768]))
    assert torch.isnan(p["activation"][768:]).all()


def wrapper_call(p, activation):
    p["dp"].zero_()
    return cudnn.grouped_gemm_dglu_wrapper_sm100(
        p["a"],
        p["c"],
        None,
        p["offsets"],
        p["alpha"],
        p["beta"],
        p["prob"],
        p["dp"],
        b_tensor=p["b"],
        act_func="dgeglu",
        vector_f32=True,
        geglu_alpha=1.0,
        glu_clamp_min=-10.0,
        glu_clamp_max=10.0,
        linear_offset=0.0,
        round_dgrad_to_input_dtype=True,
        activation_tensor=activation,
    )


def test_dglu_activation_wrapper_cache_and_dynamic_m(monkeypatch):
    import cudnn.gemm.cutedsl.grouped.dglu.api as api

    monkeypatch.setattr(api, "_dglu_wrapper_memo", {})
    monkeypatch.setattr(api, "_cache_of_GroupedGemmDgluSm100Objects", {})
    p = problem(False)
    other = torch.empty_like(p["activation"])
    ends = [256, 768, 768]
    for index, aux in enumerate((None, p["activation"], other, None, p["activation"])):
        p["c"].normal_(std=8)
        if aux is not None:
            aux.fill_(float("nan"))
        if index == 2:
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
        try:
            out = wrapper_call(p, aux)
        finally:
            if index == 2:
                torch.cuda.set_sync_debug_mode("default")
        assert len(out) == (7 if aux is None else 8)
        if aux is not None:
            assert out["activation_tensor"] is aux
            check_activation(aux[:768], reference(p, ends)[:768])
            assert torch.isnan(aux[768:]).all()
        else:
            assert "activation_tensor" not in out
    assert len(api._cache_of_GroupedGemmDgluSm100Objects) == 2
    assert len(api._dglu_wrapper_memo) == 2

    # A different M needs new live metadata, while both compiled tensor shapes
    # specialize their leading extent dynamically.
    for key in ("a", "c"):
        p[key] = mkl(p[key][:512, :, 0])
    for key in ("prob", "dp"):
        p[key] = p[key][:512]
    p["activation"] = mkl(torch.empty(512, 320, device="cuda", dtype=torch.bfloat16))
    p["activation"].fill_(float("nan"))
    ends = [0, 256, 512]
    p["offsets"].copy_(torch.tensor(ends, device="cuda", dtype=torch.int32))
    out = wrapper_call(p, p["activation"])
    check_activation(out["activation_tensor"], reference(p, ends))
    assert len(api._cache_of_GroupedGemmDgluSm100Objects) == 2


@pytest.mark.parametrize("case", ["shape", "stride", "dtype", "alignment", "c_alias", "d_alias", "activation", "c_dtype", "d_dtype"])
def test_dglu_activation_rejects_invalid_declaration(case):
    p = problem(False)
    aux, d, act_func = p["activation"], p["d"], "dgeglu"
    if case == "shape":
        aux = torch.empty(1024, 352, 1, device="cuda", dtype=torch.bfloat16)
    elif case == "stride":
        aux = torch.empty(1024, 640, device="cuda", dtype=torch.bfloat16)[:, ::2].unsqueeze(-1)
    elif case == "dtype":
        aux = aux.float()
    elif case == "alignment":
        storage = torch.empty(1024 * 320 + 16, device="cuda", dtype=torch.bfloat16)
        aux = mkl(storage[8 : 8 + 1024 * 320].view(1024, 320))
    elif case in ("c_alias", "d_alias"):
        aux = mkl(p["c" if case == "c_alias" else "d"].reshape(-1)[: 1024 * 320].view(1024, 320))
    elif case == "activation":
        act_func = "dswiglu"
    elif case == "c_dtype":
        p["c"] = p["c"].float()
    elif case == "d_dtype":
        d = d.float()
    with pytest.raises((ValueError, TypeError), match="activation"):
        make_plan(p, True, aux, d=d, act_func=act_func).check_support()


def test_dglu_activation_execute_declaration_and_alias():
    p = problem(False)
    for declared in (True, False):
        op = make_plan(p, True, p["activation"] if declared else None)
        op.compile()
        workspace = ws(op)
        with pytest.raises(ValueError, match="activation_tensor.*compiled"):
            op.execute(
                p["a"],
                p["c"],
                p["d"],
                None,
                None,
                p["offsets"],
                p["alpha"],
                p["beta"],
                p["prob"],
                p["dp"],
                **p["bind"],
                activation_tensor=None if declared else p["activation"],
                workspace=workspace,
            )
        if declared:
            alias = mkl(p["c"].reshape(-1)[: 1024 * 320].view(1024, 320))
            with pytest.raises(ValueError, match="activation output must not overlap"):
                op.execute(
                    p["a"],
                    p["c"],
                    p["d"],
                    None,
                    None,
                    p["offsets"],
                    p["alpha"],
                    p["beta"],
                    p["prob"],
                    p["dp"],
                    **p["bind"],
                    activation_tensor=alias,
                    workspace=workspace,
                )


@pytest.mark.parametrize("wrapper", [False, True])
def test_dglu_activation_rejects_blockscaled(wrapper):
    p = problem(False)
    a, b = p["a"].to(torch.float8_e4m3fn), p["b"].to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="activation output is supported only by the BF16 kernel"):
        if wrapper:
            cudnn.grouped_gemm_dglu_wrapper_sm100(
                a,
                p["c"],
                None,
                p["offsets"],
                p["alpha"],
                p["beta"],
                p["prob"],
                p["dp"],
                b_tensor=b,
                act_func="dgeglu",
                activation_tensor=p["activation"],
            )
        else:
            cudnn.GroupedGemmDgluSm100(
                a,
                p["c"],
                p["d"],
                None,
                None,
                p["offsets"],
                p["alpha"],
                p["beta"],
                p["prob"],
                p["dp"],
                sample_b=b,
                act_func="dgeglu",
                sample_activation=p["activation"],
            ).check_support()


@pytest.mark.parametrize("discrete", [False, True])
def test_dglu_activation_caller_workspace_and_first_capture(discrete, compile_allocates_nothing):
    from cudnn.api_base import TensorDesc
    from cudnn.frost.buffers import DeviceView

    p = problem(discrete)
    auxiliary = p["activation"]
    declaration = TensorDesc(
        dtype=auxiliary.dtype,
        shape=tuple(auxiliary.shape),
        stride=tuple(auxiliary.stride()),
        stride_order=(1, 0, 2),
        device=auxiliary.device,
        name="sample_activation",
    )
    plan = make_plan(p, True, declaration)
    assert plan.check_support()
    compile_allocates_nothing(plan)
    scratch = ws(plan)
    # The caller retains storage while the API accepts the framework-neutral view.
    view = DeviceView(scratch.data_ptr(), (scratch.numel(),), "uint8", auxiliary.device.index)
    p["offsets"] = p["offsets"].clone()
    if discrete:
        p["ptrs"] = p["ptrs"].clone()
        p["bind"] = dict(b_ptrs=p["ptrs"])

    def run(workspace):
        plan.execute(
            p["a"],
            p["c"],
            p["d"],
            None,
            None,
            p["offsets"],
            p["alpha"],
            p["beta"],
            p["prob"],
            p["dp"],
            **p["bind"],
            activation_tensor=auxiliary,
            workspace=workspace,
        )

    for invalid in (None, scratch[:1]):
        with pytest.raises(ValueError, match="workspace"):
            run(invalid)
    overlapping = auxiliary.view(-1).view(torch.uint8)[: scratch.numel()]
    with pytest.raises(ValueError, match="workspace must not overlap activation_tensor"):
        run(overlapping)
    auxiliary.fill_(float("nan"))
    graph = torch.cuda.CUDAGraph()
    # No successful execute precedes capture, and the pointer/offset tensors are new.
    with torch.cuda.graph(graph):
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        p["dp"].zero_()
        run(view)
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    for scale in (0.75, 0.5):
        p["prob"].mul_(scale)
        auxiliary.fill_(float("nan"))
        graph.replay()
        check_activation(auxiliary[:768], reference(p, [256, 768, 768])[:768])
        assert torch.isnan(auxiliary[768:]).all()


@pytest.fixture
def compile_allocates_nothing():
    """Recipe R11 detector: ``compile_allocates_nothing(api)`` runs ``api.compile()`` and
    asserts the torch caching allocator saw no new allocation (compile-time stand-ins
    must be fake cute tensors, never ``torch.empty``)."""

    def run(api):
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        api.compile()
        torch.cuda.synchronize()
        after = torch.cuda.memory_stats()["allocation.all.allocated"]
        assert after == before, f"{type(api).__name__}.compile() made {after - before} torch allocation(s) (Rule 8, recipe R11)"

    return run


@pytest.mark.parametrize("kind", ["offsets", "pointers"])
@pytest.mark.parametrize("valid", [True, False])
def test_debug_device_values_use_requested_stream(monkeypatch, kind, valid):
    from cudnn.gemm.cutedsl.grouped import backend_utils as utils

    caller, producer = torch.cuda.Stream(), torch.cuda.Stream()
    values = [256, 512] if valid else [256, 255]
    if kind == "pointers":
        values = [16, 32] if valid else [16, 0]
    with torch.cuda.stream(producer):
        tensor = torch.tensor(values, device="cuda", dtype=torch.int64)
    producer.synchronize()
    helper = "_host_int_values" if kind == "offsets" else "_host_pointer_values"
    original = getattr(utils, helper)
    calls = []

    def read(tensor):
        # Check the actual D2H boundary directly: no scheduling race or sleep
        # is needed for the old implementation to fail this stream contract.
        handle = torch.cuda.current_stream(tensor.device).cuda_stream
        assert handle == producer.cuda_stream
        calls.append(handle)
        return original(tensor)

    monkeypatch.setattr(utils, "DEBUG_VALIDATE_DEVICE_VALUES", True)
    monkeypatch.setattr(utils, helper, read)

    def validate():
        if kind == "offsets":
            utils.debug_validate_offsets(tensor, expert_cnt=2, limit=512, mode="padded", stream=producer.cuda_stream)
        else:
            utils.debug_validate_pointer_values(tensor, "b_ptrs", stream=producer.cuda_stream)

    with torch.cuda.stream(caller):
        if valid:
            validate()
        else:
            with pytest.raises(ValueError, match="non-decreasing|non-null"):
                validate()
        assert torch.cuda.current_stream().cuda_stream == caller.cuda_stream
    assert calls == [producer.cuda_stream]


@pytest.mark.parametrize("kind", ["offsets", "pointers"])
def test_debug_device_values_disabled_never_reads_host(monkeypatch, kind):
    from cudnn.gemm.cutedsl.grouped import backend_utils as utils

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled validation touched stream or device data")

    monkeypatch.setattr(utils, "DEBUG_VALIDATE_DEVICE_VALUES", False)
    for name in ("_check_not_capturing", "_host_int_values", "_host_pointer_values"):
        monkeypatch.setattr(utils, name, forbidden)
    if kind == "offsets":
        utils.debug_validate_offsets(None, expert_cnt=2, limit=512, mode="padded", stream=None)
    else:
        utils.debug_validate_pointer_values(None, "b_ptrs", stream=None)
