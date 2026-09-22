# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rule 3 / Rule 8 detectors for the BF16 grouped-GEMM APIs (unfused, GLU, dGLU, wgrad) and the
block-scaled wgrad API: execute() never blocks the host and never allocates, scratch is the
caller's ``workspace=`` (recipe R2), the wgrad pointer table is caller-layer work (R4), and the
first execute of a shape is CUDA-graph capturable. The opt-in device-value validation refuses to
run under stream capture (R6)."""

from __future__ import annotations

import pytest
import torch

import cudnn
from cudnn.api_base import TensorDesc
from fe_api.grouped_gemm._workspace import ws
from fe_api.grouped_gemm.test_grouped_gemm_wgrad_utils import (
    allocate_grouped_gemm_wgrad_output,
    allocate_grouped_gemm_wgrad_tensors,
    grouped_gemm_wgrad_init,
)
from test_grouped_gemm_bf16_utils import (
    assert_grouped_gemm_close,
    grouped_gemm_bf16_reference,
    make_grouped_gemm_bf16_problem,
)
from test_grouped_gemm_dglu_bf16_utils import make_grouped_gemm_dglu_bf16_problem
from test_grouped_gemm_glu_bf16_utils import make_grouped_gemm_glu_bf16_problem
from test_grouped_gemm_wgrad_bf16_utils import make_grouped_gemm_wgrad_bf16_problem


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        pytest.skip("SM100 is required")


def _nmajor(m, n, dtype=torch.bfloat16):
    # (M, N, 1) with stride (N, 1, M*N): the kernels' N-major output layout.
    return torch.empty((1, m, n), dtype=dtype, device="cuda").permute(1, 2, 0)


def _offsets_desc(offsets):
    return TensorDesc(dtype=offsets.dtype, shape=tuple(offsets.shape), stride=(1,), stride_order=(0,), device=offsets.device, name="sample_offsets")


# --------------------------------------------------------------------------------------------
# One case per API family: build the (public) API on a bf16 problem, then hand out execute
# kwargs over FRESH output buffers and FRESH offsets / pointer tables per call.
# --------------------------------------------------------------------------------------------


class _Case:
    exact_outputs: tuple = ()  # output keys compared bit-exactly across executes / replays
    accumulated_outputs: tuple = ()  # atomically accumulated (order-dependent): tolerance only

    def build(self, *, offsets_desc=False):
        raise NotImplementedError

    def kwargs(self, workspace, *, fresh_tables=True):
        raise NotImplementedError

    def reset_outputs(self, outputs):
        for key in self.exact_outputs:
            outputs[key].fill_(float("nan"))
        for key in self.accumulated_outputs:
            outputs[key].zero_()


class _Unfused(_Case):
    exact_outputs = ("c", "d")

    def __init__(self, discrete):
        self.discrete = discrete
        self.p = make_grouped_gemm_bf16_problem(discrete=discrete)
        self.m, self.n = self.p["a"].shape[0], self.p["n"]

    def build(self, *, offsets_desc=False):
        p = self.p
        kw = dict(
            sample_a=p["a"],
            sample_c=_nmajor(self.m, self.n),
            sample_d=_nmajor(self.m, self.n),
            sample_padded_offsets=_offsets_desc(p["offsets"]) if offsets_desc else p["offsets"],
            sample_alpha=p["alpha"],
            sample_prob=p["prob"],
            generate_c=True,
        )
        if self.discrete:
            kw.update(num_experts=p["experts"], b_shape=(self.n, p["k"]), b_dtype=torch.bfloat16)
        else:
            kw.update(sample_b=p["b"])
        self.api = cudnn.GroupedGemmSm100(**kw)
        assert self.api.check_support() is True
        return self.api

    def kwargs(self, workspace, *, fresh_tables=True):
        p = self.p
        outputs = {"c": _nmajor(self.m, self.n), "d": _nmajor(self.m, self.n)}
        kw = dict(
            a_tensor=p["a"],
            c_tensor=outputs["c"],
            d_tensor=outputs["d"],
            padded_offsets=p["offsets"].clone() if fresh_tables else p["offsets"],
            alpha_tensor=p["alpha"],
            prob_tensor=p["prob"],
            workspace=workspace,
        )
        if self.discrete:
            kw["b_ptrs"] = p["b_ptrs"].clone() if fresh_tables else p["b_ptrs"]
        else:
            kw["b_tensor"] = p["b"]
        return kw, outputs


class _Glu(_Case):
    exact_outputs = ("c", "d")

    def __init__(self, discrete):
        self.discrete = discrete
        self.p = make_grouped_gemm_glu_bf16_problem(discrete=discrete)
        self.m, self.n = self.p["a"].shape[0], self.p["n"]

    def build(self, *, offsets_desc=False):
        p = self.p
        kw = dict(
            sample_a=p["a"],
            sample_c=_nmajor(self.m, self.n),
            sample_d=_nmajor(self.m, self.n // 2),
            sample_sfa=None,
            sample_padded_offsets=_offsets_desc(p["offsets"]) if offsets_desc else p["offsets"],
            sample_alpha=p["alpha"],
            sample_d_col=None,
            sample_prob=p["prob"],
            act_func="swiglu",
            generate_c=True,
        )
        if self.discrete:
            kw.update(num_experts=p["experts"], b_shape=(self.n, p["k"]), b_dtype=torch.bfloat16)
        else:
            kw.update(sample_b=p["b"])
        self.api = cudnn.GroupedGemmGluSm100(**kw)
        assert self.api.check_support() is True
        return self.api

    def kwargs(self, workspace, *, fresh_tables=True):
        p = self.p
        outputs = {"c": _nmajor(self.m, self.n), "d": _nmajor(self.m, self.n // 2)}
        kw = dict(
            a_tensor=p["a"],
            c_tensor=outputs["c"],
            d_tensor=outputs["d"],
            sfa_tensor=None,
            padded_offsets=p["offsets"].clone() if fresh_tables else p["offsets"],
            alpha_tensor=p["alpha"],
            prob_tensor=p["prob"],
            workspace=workspace,
        )
        if self.discrete:
            kw["b_ptrs"] = p["b_ptrs"].clone() if fresh_tables else p["b_ptrs"]
        else:
            kw["b_tensor"] = p["b"]
        return kw, outputs


class _Dglu(_Case):
    exact_outputs = ("d_row",)
    accumulated_outputs = ("dprob",)

    def __init__(self, discrete):
        self.discrete = discrete
        self.p = make_grouped_gemm_dglu_bf16_problem(discrete=discrete)
        self.m, self.n = self.p["a"].shape[0], self.p["n"]

    def build(self, *, offsets_desc=False):
        p = self.p
        kw = dict(
            sample_a=p["a"],
            sample_c=p["c"],
            sample_d_row=_nmajor(self.m, 2 * self.n),
            sample_d_col=None,
            sample_sfa=None,
            sample_padded_offsets=_offsets_desc(p["offsets"]) if offsets_desc else p["offsets"],
            sample_alpha=p["alpha"],
            sample_beta=p["beta"],
            sample_prob=p["prob"],
            sample_dprob=p["dprob"],
            act_func="dswiglu",
        )
        if self.discrete:
            kw.update(num_experts=p["experts"], b_shape=(self.n, p["k"]), b_dtype=torch.bfloat16)
        else:
            kw.update(sample_b=p["b"])
        self.api = cudnn.GroupedGemmDgluSm100(**kw)
        assert self.api.check_support() is True
        return self.api

    def kwargs(self, workspace, *, fresh_tables=True):
        p = self.p
        outputs = {"d_row": _nmajor(self.m, 2 * self.n), "dprob": torch.zeros_like(p["dprob"])}
        kw = dict(
            a_tensor=p["a"],
            c_tensor=p["c"],
            d_row_tensor=outputs["d_row"],
            d_col_tensor=None,
            sfa_tensor=None,
            padded_offsets=p["offsets"].clone() if fresh_tables else p["offsets"],
            alpha_tensor=p["alpha"],
            beta_tensor=p["beta"],
            prob_tensor=p["prob"],
            dprob_tensor=outputs["dprob"],
            workspace=workspace,
        )
        if self.discrete:
            kw["b_ptrs"] = p["b_ptrs"].clone() if fresh_tables else p["b_ptrs"]
        else:
            kw["b_tensor"] = p["b"]
        return kw, outputs


class _Wgrad(_Case):
    exact_outputs = ("wgrad",)

    def __init__(self, discrete):
        self.discrete = discrete
        self.p = make_grouped_gemm_wgrad_bf16_problem(discrete=discrete)

    def build(self, *, offsets_desc=False):
        p = self.p
        kw = dict(
            sample_a=p["a"],
            sample_b=p["b"],
            sample_sfa=None,
            sample_sfb=None,
            sample_offsets=_offsets_desc(p["offsets"]) if offsets_desc else p["offsets"],
            acc_dtype=torch.float32,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
            input_order=p["input_order"],
        )
        if self.discrete:
            kw.update(sample_wgrad_expert=p["output"][0], num_experts=p["experts"], wgrad_shape=(p["m"], p["n"]), wgrad_dtype=p["output_dtype"])
        else:
            kw.update(sample_wgrad=p["output"])
        self.api = cudnn.GroupedGemmWgradSm100(**kw)
        assert self.api.check_support() is True
        return self.api

    def kwargs(self, workspace, *, fresh_tables=True):
        p = self.p
        outputs = {"wgrad": torch.empty_like(p["output"])}
        kw = dict(
            a_tensor=p["a"],
            b_tensor=p["b"],
            sfa_tensor=None,
            sfb_tensor=None,
            offsets_tensor=p["offsets"].clone() if fresh_tables else p["offsets"],
            workspace=workspace,
        )
        if self.discrete:
            kw["wgrad_ptrs"] = cudnn.wgrad_expert_ptrs(outputs["wgrad"])  # a fresh table for a fresh output
        else:
            kw["wgrad_tensor"] = outputs["wgrad"]
        return kw, outputs


_FAMILIES = {"unfused": _Unfused, "glu": _Glu, "dglu": _Dglu, "wgrad": _Wgrad}


def _block_scaled_wgrad(discrete):
    """The FP8/e8m0 block-scaled wgrad API (torch-only), dense or discrete."""
    cfg = grouped_gemm_wgrad_init(
        ab_dtype=torch.float8_e4m3fn,
        wgrad_dtype=torch.bfloat16,
        acc_dtype=torch.float32,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
    )
    inputs = allocate_grouped_gemm_wgrad_tensors(cfg)
    wgrad = allocate_grouped_gemm_wgrad_output(cfg)
    kw = dict(
        sample_a=inputs["a_tensor"],
        sample_b=inputs["b_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_sfb=inputs["sfb_tensor"],
        sample_offsets=inputs["offsets_tensor"],
        acc_dtype=cfg["acc_dtype"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
    )
    if discrete:
        kw.update(num_experts=cfg["l"], wgrad_shape=(cfg["m"], cfg["n"]), wgrad_dtype=cfg["wgrad_dtype"])
    else:
        kw.update(sample_wgrad=wgrad)
    op = cudnn.GroupedGemmWgradSm100(**kw)
    assert op.check_support() is True
    execute_kw = dict(
        a_tensor=inputs["a_tensor"],
        b_tensor=inputs["b_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        sfb_tensor=inputs["sfb_tensor"],
        offsets_tensor=inputs["offsets_tensor"],
        wgrad_tensor=wgrad,
    )
    return op, execute_kw


def _assert_outputs_match(case, actual, expected):
    for key in case.exact_outputs:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    for key in case.accumulated_outputs:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-3, atol=1e-3)


# --------------------------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_execute_never_synchronizes(discrete):
    case = _Unfused(discrete)
    api = case.build()
    api.compile()
    workspace = ws(api)
    kw, ref = case.kwargs(workspace, fresh_tables=False)
    api.execute(**kw)
    torch.cuda.synchronize()
    expected_c, expected_d = grouped_gemm_bf16_reference(case.p)
    assert_grouped_gemm_close(ref["c"], expected_c)
    assert_grouped_gemm_close(ref["d"], expected_d)

    # Fresh offsets / pointer tables per execute: a tensor the plan has never seen is
    # exactly what an identity-keyed validation memo misses on, which is how the old
    # per-tensor D2H read hid from every warm test.
    runs = [case.kwargs(workspace) for _ in range(3)]
    torch.cuda.synchronize()

    torch.cuda.set_sync_debug_mode("error")
    try:
        for kw, _ in runs:
            api.execute(**kw)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    for _, outputs in runs:
        _assert_outputs_match(case, outputs, ref)


@pytest.mark.L0
@pytest.mark.parametrize("family", sorted(_FAMILIES))
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_execute_allocates_nothing_and_never_synchronizes(family, discrete):
    """Recipe R9: after a warm execute, three executes over fresh outputs and fresh offsets /
    pointer tables make no caching-allocator allocation and no host sync, and reproduce the warm
    result. The wgrad discrete table comes from ``wgrad_expert_ptrs`` (one device fill, R4)."""
    case = _FAMILIES[family](discrete)
    api = case.build()
    api.compile()
    workspace = ws(api)
    kw, ref = case.kwargs(workspace)
    api.execute(**kw)
    runs = [case.kwargs(workspace) for _ in range(3)]
    torch.cuda.synchronize()

    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    torch.cuda.set_sync_debug_mode("error")
    try:
        for kw, _ in runs:
            api.execute(**kw)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_stats()["allocation.all.allocated"] - before
    assert allocated == 0, f"{type(api).__name__}.execute() made {allocated} torch allocation(s) over 3 executes (Rule 8)"
    for _, outputs in runs:
        _assert_outputs_match(case, outputs, ref)


@pytest.mark.L0
@pytest.mark.parametrize("family", [*sorted(_FAMILIES), "wgrad_blockscaled"])
def test_execute_requires_workspace(family):
    """Recipe R2, APIBase flavour: the workspace is always required (>= 128 bytes, a multiple of
    128); None, an undersized buffer and a misaligned view raise before launch."""
    if family == "wgrad_blockscaled":
        api, execute_kw = _block_scaled_wgrad(False)

        def with_workspace(workspace):
            return dict(execute_kw, workspace=workspace)

    else:
        case = _FAMILIES[family](False)
        api = case.build()

        def with_workspace(workspace):
            return case.kwargs(workspace, fresh_tables=False)[0]

    api.compile()
    nbytes = api.scratch_workspace_bytes()
    assert nbytes >= 128 and nbytes % 128 == 0

    with pytest.raises(ValueError, match=r"requires a \d+-byte workspace but execute\(\) received none"):
        api.execute(**with_workspace(None))
    with pytest.raises(ValueError, match=r"needs a \d+-byte workspace, got \d+ bytes"):
        api.execute(**with_workspace(torch.empty(nbytes - 64, dtype=torch.uint8, device="cuda")))
    with pytest.raises(ValueError, match="must be 128-byte aligned"):
        api.execute(**with_workspace(torch.empty(nbytes + 128, dtype=torch.uint8, device="cuda")[64 : 64 + nbytes]))
    api.execute(**with_workspace(torch.empty(nbytes, dtype=torch.uint8, device="cuda")))
    torch.cuda.synchronize()


@pytest.mark.L0
@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_first_execute_under_capture(family):
    """Rule 8's capture detector: compile eagerly, then the FIRST execute of the plan -- over an
    offsets tensor and a pointer table it has never seen -- runs inside ``torch.cuda.graph`` with a
    torch workspace and makes no allocation inside the window (read INSIDE the window: on torch 2.13
    ``capture_begin`` itself accounts for two); the replay is bit-identical to an eager execute
    and stable across replays."""
    case = _FAMILIES[family](True)
    api = case.build()
    api.compile()
    workspace = ws(api)
    kw, captured = case.kwargs(workspace)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        api.execute(**kw)
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"] - before
    assert allocated == 0, f"{allocated} allocation(s) inside the capture"

    case.reset_outputs(captured)
    graph.replay()
    torch.cuda.synchronize()
    replayed = {key: value.clone() for key, value in captured.items()}
    for key in case.exact_outputs:
        assert torch.isfinite(replayed[key].float()).all(), f"the captured first execute replayed a non-finite {key}"

    eager_kw, eager = case.kwargs(ws(api))
    api.execute(**eager_kw)
    torch.cuda.synchronize()
    _assert_outputs_match(case, replayed, eager)

    case.reset_outputs(captured)
    graph.replay()
    torch.cuda.synchronize()
    _assert_outputs_match(case, captured, replayed)


@pytest.mark.L0
def test_retain_workspace_guards_device_and_keeps_eager_buffers_until_their_launch_completes():
    """R2 lifetime for the caller's workspace: a CPU or other-device buffer is refused before any
    pointer is taken; a torch buffer is record_stream'ed; an immutable-framework buffer (stand-in:
    a non-torch object) is held by the API until an event recorded on its launch stream after the
    launch has completed -- overlapping calls keep every buffer, not just the latest one."""
    from cudnn.gemm.cutedsl.grouped.backend_utils import retain_workspace

    class _Desc:
        device = torch.device("cuda", torch.cuda.current_device())

    class _Api:
        a_desc = _Desc()

    class _EagerBuffer:  # what a JAX array looks like to retain_workspace: a device and nothing torch
        def __init__(self, device):
            self.device = device

    api = _Api()
    stream = torch.cuda.current_stream().cuda_stream
    with pytest.raises(ValueError, match="CUDA buffer"):
        retain_workspace(api, torch.empty(64, dtype=torch.uint8), stream)
    with pytest.raises(ValueError, match="plan's device"):  # metadata-only: the index need not exist
        retain_workspace(api, _EagerBuffer(torch.device("cuda", torch.cuda.current_device() + 1)), stream)

    ws = torch.empty(64, dtype=torch.uint8, device="cuda")
    retain_workspace(api, ws, stream)  # torch: record_stream, nothing retained
    assert not hasattr(api, "_live_workspaces")

    buffers = [_EagerBuffer(_Desc.device) for _ in range(3)]
    torch.cuda._sleep(2_000_000_000)  # a long kernel in flight on the launch stream: nothing recorded behind it completes yet
    for b in buffers:
        retain_workspace(api, b, stream)
    assert [e.buffer for e in api._live_workspaces] == buffers, "every in-flight buffer is held, not only the latest"
    assert all(e.event is not None for e in api._live_workspaces[:-1]) and api._live_workspaces[-1].event is None
    sentinel = _EagerBuffer(_Desc.device)
    retain_workspace(api, sentinel, stream)  # fences the third buffer (its launch is enqueued by now)
    torch.cuda.synchronize()
    late = _EagerBuffer(_Desc.device)
    retain_workspace(api, late, stream)  # every fenced launch has completed: those buffers are released
    held = [e.buffer for e in api._live_workspaces]
    assert not any(b in held for b in buffers) and late in held


@pytest.mark.L0
def test_wgrad_expert_ptrs_rejects_overlapping_or_strided_slices():
    """The discrete kernels read each expert's slice through one contiguous (hidden, intermediate)
    descriptor: a view whose slices are strided or overlap is refused before a pointer table is built."""
    e, h, i = 4, 64, 32
    stacked = torch.empty(e, h, i, dtype=torch.bfloat16, device="cuda")
    assert tuple(cudnn.wgrad_expert_ptrs(stacked).shape) == (e,)
    with pytest.raises(ValueError, match="contiguous"):
        cudnn.wgrad_expert_ptrs(stacked.transpose(1, 2))
    with pytest.raises(ValueError, match="overlap"):
        cudnn.wgrad_expert_ptrs(stacked.as_strided((e, h, i), (h * i // 2, i, 1)))


@pytest.mark.L0
def test_wgrad_discrete_requires_wgrad_ptrs():
    """Design D2 / recipe R4: the discrete wgrad APIs consume a caller-provided pointer table and
    never derive one per execute; ``wgrad_expert_ptrs`` is the caller-layer derivation the wrapper
    uses, and it equals the explicit table."""
    bf16 = _Wgrad(True)
    api = bf16.build()
    api.compile()
    kw, _ = bf16.kwargs(ws(api))
    kw["wgrad_tensor"] = bf16.p["output"]
    kw.pop("wgrad_ptrs")
    with pytest.raises(ValueError, match="wgrad_expert_ptrs"):
        api.execute(**kw)

    block_scaled, execute_kw = _block_scaled_wgrad(True)
    block_scaled.compile()
    with pytest.raises(ValueError, match="wgrad_expert_ptrs"):
        block_scaled.execute(**execute_kw, workspace=ws(block_scaled))

    # The helper's table is exactly the per-expert base addresses, for any expert count.
    for experts in (3, 1, 0):
        stacked = torch.empty((experts, 128, 128), dtype=torch.bfloat16, device="cuda")
        table = cudnn.wgrad_expert_ptrs(stacked)
        assert table.dtype is torch.int64 and table.shape == (experts,) and table.is_cuda
        assert table.tolist() == [stacked[index].data_ptr() for index in range(experts)]

    # Wrapper: a derived table and an explicit one produce the same result.
    problem = make_grouped_gemm_wgrad_bf16_problem(discrete=True)
    common = dict(
        a_tensor=problem["a"],
        b_tensor=problem["b"],
        sfa_tensor=None,
        sfb_tensor=None,
        offsets_tensor=problem["offsets"],
        output_mode="discrete",
        acc_dtype=torch.float32,
        wgrad_dtype=problem["output_dtype"],
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        input_order=problem["input_order"],
    )
    derived = cudnn.grouped_gemm_wgrad_wrapper_sm100(**common, wgrad_tensor=problem["output"], wgrad_ptrs=None)["wgrad_tensor"]
    explicit_output = torch.empty_like(problem["output"])
    explicit = cudnn.grouped_gemm_wgrad_wrapper_sm100(**common, wgrad_tensor=explicit_output, wgrad_ptrs=cudnn.wgrad_expert_ptrs(explicit_output))
    torch.cuda.synchronize()
    assert derived is problem["output"]
    assert explicit["wgrad_tensor"] is explicit_output
    torch.testing.assert_close(explicit_output, derived, rtol=0, atol=0)


@pytest.mark.L0
@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_sample_offsets_may_be_tensordesc(family):
    """Rule 3: construction, check_support() and compile() read no offset values, so the sample
    offsets can be a metadata-only TensorDesc; the first execute binds the live tensor."""
    case = _FAMILIES[family](False)
    api = case.build(offsets_desc=True)
    api.compile()
    kw, outputs = case.kwargs(ws(api))
    api.execute(**kw)
    torch.cuda.synchronize()
    for key in case.exact_outputs:
        assert torch.isfinite(outputs[key].float()).all()


@pytest.mark.L0
@pytest.mark.allow_host_sync  # the debug gate blocks by design (R6); the R9 detector must not veto it
def test_debug_validation_env_var(monkeypatch):
    from cudnn.gemm.cutedsl.grouped import backend_utils

    case = _Unfused(True)
    api = case.build()
    api.compile()
    workspace = ws(api)
    m = case.p["a"].shape[0]
    good = case.p["offsets"]

    def run(offsets, b_ptrs=None):
        kw, _ = case.kwargs(workspace, fresh_tables=False)
        kw["padded_offsets"] = offsets
        if b_ptrs is not None:
            kw["b_ptrs"] = b_ptrs
        api.execute(**kw)

    monkeypatch.setattr(backend_utils, "DEBUG_VALIDATE_DEVICE_VALUES", True)
    run(good)  # well-formed values pass the blocking checks

    decreasing = good.flip(0)
    with pytest.raises(ValueError, match="non-decreasing"):
        run(decreasing)
    unaligned = good.clone()
    unaligned[0] += 8
    with pytest.raises(ValueError, match="256-aligned"):
        run(unaligned)
    too_long = good.clone()
    too_long[-1] = m + 256
    with pytest.raises(ValueError, match="last value"):
        run(too_long)
    null_entry = case.p["b_ptrs"].clone()
    null_entry[0] = 0
    with pytest.raises(ValueError, match="non-null and 16-byte aligned"):
        run(good, b_ptrs=null_entry)

    # R6: with the flag on, a captured stream is refused BEFORE any D2H read -- our
    # RuntimeError naming the env var, not torch's "operation not permitted when
    # stream is capturing".
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match=backend_utils.DEBUG_VALIDATE_DEVICE_VALUES_ENV):
        with torch.cuda.graph(graph):
            run(good)
