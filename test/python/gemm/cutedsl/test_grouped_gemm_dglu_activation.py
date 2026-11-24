# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recomputed primal output for BF16 dGeGLU and dSwiGLU; diagnostic geometry, no timing."""

import numpy as np
import pytest
import torch

from determinism_utils import assert_bitwise, assert_replays
from gemm.cutedsl._workspace import ws

import cudnn
from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

pytestmark = pytest.mark.L0
ACT_FUNCS = ["dgeglu", "dswiglu"]


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


def reference(p, ends, *, act_func="dgeglu", geglu_alpha=1.0, linear_offset=0.0):
    expected = torch.empty_like(p["activation"])
    n = p["activation"].shape[1]
    begin = 0
    for expert, end in enumerate(ends):
        if end > begin:
            pair = (p["c"][begin:end, :, 0].float() * p["beta"][expert]).reshape(end - begin, n // 32, 2, 32)
            gate, up = pair[:, :, 0].reshape(end - begin, n), pair[:, :, 1].reshape(end - begin, n)
            if act_func == "dswiglu":
                # The SwiGLU forward has no clamp or linear offset.
                primal = torch.nn.functional.silu(gate) * up
            else:
                gate, up = gate.clamp(max=10), up.clamp(-10, 10)
                primal = (gate * (gate * geglu_alpha).sigmoid()) * (up + linear_offset)
            expected[begin:end, :, 0] = (primal * p["prob"][begin:end, 0, 0, None]).bfloat16()
        begin = end
    return expected


def check_activation(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=2e-5)


@pytest.mark.parametrize("act_func", ACT_FUNCS)
@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("vector_f32", [False, True])
@pytest.mark.parametrize("n", [288, 320])
def test_dglu_recomputed_activation_changed_input_graph(discrete, vector_f32, n, act_func):
    p = problem(discrete, n=n)
    plan = make_plan(p, vector_f32, p["activation"], act_func=act_func)
    baseline = make_plan(p, vector_f32, None, d=p["baseline_d"], dp=p["baseline_dp"], act_func=act_func)
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
    try:
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
            expected = reference(p, ends, act_func=act_func)
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
    finally:
        graph.reset()


@pytest.mark.parametrize("act_func", ACT_FUNCS)
@pytest.mark.parametrize("vector_f32", [False, True])
def test_dglu_activation_survives_zero_gradient_and_alpha(vector_f32, act_func):
    p = problem(False)
    plan = make_plan(p, vector_f32, p["activation"], act_func=act_func, geglu_alpha=1.702, linear_offset=1.0)
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
    expected = reference(p, [256, 768, 768], act_func=act_func, geglu_alpha=1.702, linear_offset=1.0)
    check_activation(p["activation"][:768], expected[:768])
    assert p["activation"][:768].abs().max() > 0
    assert torch.equal(p["d"][:768], torch.zeros_like(p["d"][:768]))
    assert torch.equal(p["dp"][:768], torch.zeros_like(p["dp"][:768]))
    assert torch.isnan(p["activation"][768:]).all()


def wrapper_call(p, activation, act_func="dgeglu"):
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
        act_func=act_func,
        vector_f32=True,
        geglu_alpha=1.0,
        glu_clamp_min=-10.0,
        glu_clamp_max=10.0,
        linear_offset=0.0,
        round_dgrad_to_input_dtype=True,
        activation_tensor=activation,
    )


@pytest.mark.parametrize("act_func", ACT_FUNCS)
def test_dglu_activation_wrapper_cache_and_dynamic_m(monkeypatch, act_func):
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
            out = wrapper_call(p, aux, act_func)
        finally:
            if index == 2:
                torch.cuda.set_sync_debug_mode("default")
        assert len(out) == (7 if aux is None else 8)
        if aux is not None:
            assert out["activation_tensor"] is aux
            check_activation(aux[:768], reference(p, ends, act_func=act_func)[:768])
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
    out = wrapper_call(p, p["activation"], act_func)
    check_activation(out["activation_tensor"], reference(p, ends, act_func=act_func))
    assert len(api._cache_of_GroupedGemmDgluSm100Objects) == 2


@pytest.mark.parametrize("act_func", ACT_FUNCS)
@pytest.mark.parametrize("case", ["shape", "stride", "dtype", "alignment", "c_alias", "d_alias", "c_dtype", "d_dtype"])
def test_dglu_activation_rejects_invalid_declaration(case, act_func):
    p = problem(False)
    aux, d = p["activation"], p["d"]
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
    elif case == "c_dtype":
        p["c"] = p["c"].float()
    elif case == "d_dtype":
        d = d.float()
    with pytest.raises((ValueError, TypeError), match="activation"):
        make_plan(p, True, aux, d=d, act_func=act_func).check_support()


@pytest.mark.parametrize("act_func", ACT_FUNCS)
def test_dglu_activation_execute_declaration_and_alias(act_func):
    p = problem(False)
    for declared in (True, False):
        op = make_plan(p, True, p["activation"] if declared else None, act_func=act_func)
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


@pytest.mark.parametrize("act_func", ACT_FUNCS)
@pytest.mark.parametrize("wrapper", [False, True])
def test_dglu_activation_rejects_blockscaled(wrapper, act_func):
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
                act_func=act_func,
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
                act_func=act_func,
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
    try:
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
    finally:
        graph.reset()


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


@pytest.mark.parametrize("act_func", ACT_FUNCS)
@pytest.mark.parametrize("entry", ["class", "wrapper_dense", "wrapper_discrete"])
def test_dglu_activation_rejects_jax(entry, act_func):
    jnp = pytest.importorskip("jax.numpy")
    m, k, n = 256, 128, 64
    a, c = jnp.zeros((m, k, 1), jnp.bfloat16), jnp.zeros((m, 2 * n, 1), jnp.bfloat16)
    offsets, scale = jnp.full((1,), m, jnp.int32), jnp.ones((1,), jnp.float32)
    prob, activation = jnp.ones((m, 1, 1), jnp.float32), jnp.zeros((m, n, 1), jnp.bfloat16)
    message = "activation_tensor is unsupported for the JAX" if entry == "wrapper_dense" else "activation output currently requires torch tensors"
    with pytest.raises(ValueError, match=message):
        if entry == "class":
            cudnn.GroupedGemmDgluSm100(
                a,
                c,
                jnp.zeros((m, 2 * n, 1), jnp.bfloat16),
                None,
                None,
                offsets,
                scale,
                scale,
                prob,
                prob,
                num_experts=1,
                b_shape=(n, k),
                b_dtype=jnp.bfloat16,
                act_func=act_func,
                sample_activation=activation,
            ).check_support()
        else:
            weights = dict(b_tensor=jnp.zeros((n, k, 1), jnp.bfloat16))
            if entry == "wrapper_discrete":
                # The packed uint8 pointer table works without JAX x64 mode.
                weights = dict(b_ptrs=jnp.asarray(np.array([256], np.int64).view(np.uint8)), n=n, b_dtype=jnp.bfloat16)
            cudnn.grouped_gemm_dglu_wrapper_sm100(
                a,
                c,
                None,
                offsets,
                scale,
                scale,
                prob,
                None if entry == "wrapper_dense" else prob,
                act_func=act_func,
                activation_tensor=activation,
                **weights,
            )


def exact_glu_inputs(m, k, two_n, experts):
    # One power-of-two entry per A row and power-of-two alpha keep every forward FP32
    # pre-activation a BF16 value, so the saved C is exactly the forward GLU input.
    rows = torch.arange(m, device="cuda")
    a = torch.zeros(m, k, device="cuda", dtype=torch.bfloat16)
    values = (2.0 ** torch.randint(-1, 2, (m,), device="cuda")) * (1 - 2 * torch.randint(2, (m,), device="cuda"))
    a[rows, torch.randint(k, (m,), device="cuda")] = values.bfloat16()
    w1 = (torch.randn(experts, two_n, k, device="cuda") * 4).bfloat16()
    return mkl(a), w1


@pytest.mark.parametrize(
    "discrete,vector_f32,glu_layout,n",
    [
        (False, False, "interleaved_32", 288),
        (False, True, "interleaved_32", 288),
        (True, False, "interleaved_32", 320),
        (True, True, "interleaved_32", 320),
        (False, False, "gate_then_up", 256),
        (False, True, "gate_then_up", 256),
    ],
)
def test_dswiglu_activation_is_bitwise_forward_glu_output(discrete, vector_f32, glu_layout, n):
    """h equals the forward GLU D byte for byte when C holds the forward pre-activation."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10 or cutedsl_too_old(cutedsl_state()[1]):
        pytest.skip("requires Blackwell and CuTe DSL>=4.7.0")
    if glu_layout == "gate_then_up" and torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("gate_then_up requires SM100/103")
    torch.manual_seed(467)
    m, k, k2 = 1280, 128, 256
    ends = [256, 256, 768, 1024]  # an empty expert and capacity past the final offset
    experts = len(ends)
    a1, w1 = exact_glu_inputs(m, k, 2 * n, experts)
    offsets = torch.tensor(ends, device="cuda", dtype=torch.int32)
    alpha1 = torch.tensor([1.0, 0.5, 2.0, 0.5], device="cuda")
    beta = torch.tensor([1.0, 4.0, 2.0, 0.5], device="cuda")
    prob = torch.rand(m, 1, 1, device="cuda")
    prob[::13] = 0
    forward = cudnn.grouped_gemm_glu_wrapper_sm100(
        a1,
        None,
        offsets,
        alpha1,
        b_tensor=w1.permute(1, 2, 0),
        prob_tensor=prob,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        vector_f32=vector_f32,
        act_func="swiglu",
        generate_c=True,
        glu_layout=glu_layout,
    )
    saved = forward["c_tensor"][:, :, 0].float()
    # Save C pre-divided by beta; powers of two keep beta * C equal to the forward value.
    row_beta = torch.ones(m, 1, device="cuda")
    for expert, (begin, end) in enumerate(zip([0] + ends[:-1], ends)):
        row_beta[begin:end] = beta[expert]
    c = mkl((saved / row_beta).bfloat16())
    assert torch.equal((c[:, :, 0].float() * row_beta)[: ends[-1]], saved[: ends[-1]])

    a2 = mkl(torch.randn(m, k2, device="cuda", dtype=torch.bfloat16) * 0.125)
    w2 = torch.randn(experts, n, k2, device="cuda", dtype=torch.bfloat16) * 0.125
    weights = dict(b_tensor=w2.permute(1, 2, 0))
    if discrete:
        weights = dict(b_ptrs=torch.tensor([w.data_ptr() for w in w2], device="cuda", dtype=torch.int64), n=n, b_dtype=torch.bfloat16)
    activation = mkl(torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16))
    dprob = torch.zeros_like(prob)
    out = cudnn.grouped_gemm_dglu_wrapper_sm100(
        a2,
        c,
        None,
        offsets,
        torch.tensor([1.0, 0.5, 1.5, 1.0], device="cuda"),
        beta,
        prob,
        dprob,
        act_func="dswiglu",
        vector_f32=vector_f32,
        glu_layout=glu_layout,
        activation_tensor=activation,
        use_dynamic_sched=discrete,
        **weights,
    )
    torch.cuda.synchronize()  # Retain the discrete weight owners until the launch completes.
    assert out["activation_tensor"] is activation
    valid = ends[-1]
    actual, expected = activation[:valid, :, 0], forward["d_tensor"][:valid].reshape(valid, n)
    assert_bitwise({"activation": actual}, {"activation": expected})
    assert torch.isnan(activation[valid:]).all()
    # Independent reference on the interleaved view of the saved pre-activation.
    interleaved = c
    if glu_layout == "gate_then_up":
        gate, up = c[:, :n, 0], c[:, n:, 0]
        interleaved = mkl(torch.stack((gate.reshape(m, n // 32, 32), up.reshape(m, n // 32, 32)), dim=2).reshape(m, 2 * n))
    ref = reference(dict(activation=activation, c=interleaved, beta=beta, prob=prob), ends, act_func="dswiglu")
    check_activation(actual, ref[:valid, :, 0])
    assert actual.abs().max() > 0 and torch.equal(actual[::13], torch.zeros_like(actual[::13]))


@pytest.mark.parametrize("act_func", ACT_FUNCS)
def test_dglu_activation_replay_and_expert_batch_invariance(act_func):
    """Raw-byte replay under concurrent work, and expert reorder/duplicate/neighbor isolation."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10 or cutedsl_too_old(cutedsl_state()[1]):
        pytest.skip("requires Blackwell and CuTe DSL>=4.7.0")
    torch.manual_seed(479)
    group, k, n, experts = 1024, 256, 2048, 4  # 128 tiles cross the persistent schedule
    m = group * experts
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.125
    c = torch.randn(m, 2 * n, device="cuda", dtype=torch.bfloat16) * 4
    w = torch.randn(experts, n, k, device="cuda", dtype=torch.bfloat16) * 0.125
    alpha = torch.tensor([1.0, 0.5, 1.5, -0.75], device="cuda")
    beta = torch.tensor([0.5, 1.0, 1.5, 2.0], device="cuda")
    prob = torch.rand(m, 1, 1, device="cuda")
    inputs = dict(a=a, c=c, w=w, alpha=alpha, beta=beta, prob=prob)
    saved_inputs = {name: tensor.clone() for name, tensor in inputs.items()}

    def invoke(a, c, w, alpha, beta, prob):
        count = w.shape[0]
        rows = a.shape[0]
        activation = mkl(torch.empty(rows, n, device="cuda", dtype=torch.bfloat16))
        out = cudnn.grouped_gemm_dglu_wrapper_sm100(
            mkl(a),
            mkl(c),
            None,
            torch.arange(1, count + 1, device="cuda", dtype=torch.int32) * group,
            alpha,
            beta,
            prob,
            torch.zeros_like(prob),
            b_tensor=w.permute(1, 2, 0),
            act_func=act_func,
            vector_f32=True,
            geglu_alpha=1.0,
            glu_clamp_min=-10.0,
            glu_clamp_max=10.0,
            linear_offset=0.0,
            use_dynamic_sched=True,
            activation_tensor=activation,
        )
        # dprob accumulates across N tiles with float atomics and is not byte-compared.
        return dict(activation=out["activation_tensor"][:, :, 0], d_row=out["d_row_tensor"][:, :, 0])

    packed = assert_replays(lambda: invoke(**inputs), repeats=8)
    ends = [group * (i + 1) for i in range(experts)]
    expected = reference(dict(activation=mkl(packed["activation"]), c=mkl(c), beta=beta, prob=prob), ends, act_func=act_func)
    check_activation(packed["activation"], expected[:, :, 0])
    for order in ((1,), (2, 0, 2)):
        rows = [row for i in order for row in range(i * group, (i + 1) * group)]
        picked = list(order)
        result = invoke(a[rows], c[rows], w[picked].contiguous(), alpha[picked], beta[picked], prob[rows].contiguous())
        for name, tensor in packed.items():
            assert_bitwise({name: result[name]}, {name: tensor[rows]}, f"order {order}")
    neighbor = dict(inputs, a=a.clone(), c=c.clone())
    neighbor["a"][group : 2 * group].zero_()
    neighbor["c"][group : 2 * group].fill_(0.25)
    changed = invoke(**neighbor)
    for name, tensor in packed.items():
        assert_bitwise({name: changed[name][:group]}, {name: tensor[:group]}, "neighbor")
        assert_bitwise({name: changed[name][2 * group :]}, {name: tensor[2 * group :]}, "neighbor")
    for name, tensor in saved_inputs.items():
        assert_bitwise({name: inputs[name]}, {name: tensor}, "input mutation")
