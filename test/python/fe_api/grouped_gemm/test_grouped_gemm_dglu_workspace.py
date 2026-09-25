# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workspace device and asynchronous lifetime regressions; no timing claims."""

import pytest
import torch

from cuda.bindings import driver as cuda
from cudnn.frost.buffers import DeviceView
from cudnn.frost.workspace import Workspace
from fe_api.grouped_gemm.test_grouped_gemm_dglu_activation import make_plan, problem


@pytest.mark.L0
@pytest.mark.parametrize("kind", ["cpu", "wrong_device"])
def test_dglu_workspace_device_rejected_before_consumer(kind, monkeypatch):
    p = problem(True)
    op = make_plan(p, True, None)
    op.check_support()
    nbytes = op.scratch_workspace_bytes()
    if kind == "cpu":
        storage = torch.empty(nbytes + 128, dtype=torch.uint8)
        offset = -storage.data_ptr() % 128
        workspace = storage[offset : offset + nbytes]
    else:
        storage = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        # Intercept before any launch: no operation is run on this deliberately
        # mismatched device declaration, even on a single-GPU CI worker.
        workspace = DeviceView(storage.data_ptr(), (nbytes,), "uint8", storage.device.index + 1)
    seen = []
    monkeypatch.setattr(op._implementation, "_compiled_kernel", lambda *args: seen.append(args))
    with pytest.raises(ValueError, match="workspace must be"):
        op.execute(p["a"], p["c"], p["d"], None, None, p["offsets"], p["alpha"], p["beta"], p["prob"], p["dp"], **p["bind"], workspace=workspace)
    assert not seen


@pytest.mark.L0
def test_workspace_cuda_array_interface_device():
    storage = torch.empty(256, dtype=torch.uint8, device="cuda")

    class CudaArray:
        __cuda_array_interface__ = storage.__cuda_array_interface__

    # CAI-only producers need not implement __dlpack_device__ or .device.
    assert Workspace(CudaArray(), 256, "test", device=storage.device.index).take(256, "uint8").data_ptr() == storage.data_ptr()
    with pytest.raises(ValueError, match="workspace must be on"):
        Workspace(CudaArray(), 256, "test", device=storage.device.index + 1)


@pytest.mark.L1
def test_dglu_jax_workspace_overlapping_consumers(monkeypatch):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import cudnn.gemm.cutedsl.grouped.dglu.api as wrapper
    from fe_api.grouped_gemm.test_grouped_gemm_dglu_jax import _packed_jax_ptrs, make_problem, skip_unless_sm100

    skip_unless_sm100()
    monkeypatch.setattr(wrapper, "_dglu_wrapper_memo", {})
    monkeypatch.setattr(wrapper, "_cache_of_GroupedGemmDgluSm100Objects", {})
    a, c, b, offsets, alpha, beta, prob = make_problem()
    aj, cj, oj, alj, bej, pj = (jnp.asarray(x) for x in (a, c, offsets, alpha, beta, prob))
    weights = [jnp.asarray(x) for x in b]
    dp = jnp.zeros(prob.shape, dtype=jnp.float32)
    jax.block_until_ready((aj, cj, oj, alj, bej, pj, weights, dp))
    ptrs = _packed_jax_ptrs(weights)
    kwargs = dict(
        a_tensor=aj,
        c_tensor=cj,
        sfa_tensor=None,
        padded_offsets=oj,
        alpha_tensor=alj,
        beta_tensor=bej,
        prob_tensor=pj,
        dprob_tensor=dp,
        b_ptrs=ptrs,
        n=128,
        b_dtype="bfloat16",
        d_dtype="bfloat16",
        generate_dbias=True,
    )
    # Compile and warm actual eager JAX once. Only the subsequent scratch consumer
    # is intercepted: intentionally recycled scratch never reaches a GEMM.
    result = wrapper.grouped_gemm_dglu_wrapper_sm100(**kwargs)
    torch.cuda.synchronize()
    memo = wrapper._dglu_wrapper_memo.copy()
    plan = next(value[0]._implementation for value in memo.values() if value[1] == "jax" and value[2:4] == (512, 256))
    nbytes = plan.scratch_workspace_bytes()
    monkeypatch.setattr(wrapper, "_dglu_wrapper_memo", {})  # full path, then memo hit
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    initialized = [torch.cuda.Event(), torch.cuda.Event()]
    done = [torch.cuda.Event(), torch.cuda.Event()]
    observed = [torch.empty(nbytes, dtype=torch.uint8, device="cuda") for _ in streams]
    pointers, outputs, replacements = [], [result], []

    def delayed_consumer(*args):
        i = len(pointers)
        ptr, launch_stream = args[-4:-2]
        assert int(launch_stream) == streams[i].cuda_stream
        pointers.append(ptr)
        with torch.cuda.stream(streams[i]):
            assert cuda.cuMemsetD8Async(ptr, 17 + i, nbytes, launch_stream)[0] == cuda.CUresult.CUDA_SUCCESS
            initialized[i].record()
            # Keep the delay bounded. Waiting on an unrecorded CUDA event is
            # a no-op, so it cannot serve as a host-released gate. The pending
            # assertion below proves overlap; an expired window fails the setup.
            torch.cuda._sleep(600_000_000)
            assert cuda.cuMemcpyDtoDAsync(observed[i].data_ptr(), ptr, nbytes, launch_stream)[0] == cuda.CUresult.CUDA_SUCCESS
            done[i].record()

    monkeypatch.setattr(plan, "_compiled_kernel", delayed_consumer)
    try:
        for stream in streams:
            outputs.append(wrapper.grouped_gemm_dglu_wrapper_sm100(**kwargs, current_stream=cuda.CUstream(stream.cuda_stream)))
        producer = torch.cuda.current_stream()
        for event in initialized:
            producer.wait_event(event)
        # Churn the JAX pool while both foreign consumers remain in flight. With
        # the old temporary JAX workspace the pool recycles one of these pointers.
        reused = set()
        for _ in range(32):
            value = jax.block_until_ready(jnp.empty((nbytes,), dtype=jnp.uint8))
            replacements.append(value)
            ptr = value.unsafe_buffer_pointer()
            if ptr in pointers:
                reused.add(ptr)
            assert cuda.cuMemsetD8Async(ptr, 165, nbytes, cuda.CUstream(producer.cuda_stream))[0] == cuda.CUresult.CUDA_SUCCESS
        producer.synchronize()
        assert all(not event.query() for event in done), "probe must observe both consumers pending"
        for event in done:
            event.synchronize()
        print("scratch allocations recycled by JAX:", len(reused))
        for i, value in enumerate(observed):
            torch.testing.assert_close(value, torch.full_like(value, 17 + i), rtol=0, atol=0)
    finally:
        # Own every allocation/operand until the bounded byte consumers complete,
        # including on an assertion failure in the RED run.
        torch.cuda.synchronize()


@pytest.mark.L1
def test_jax_wrapper_scratch_stream_capture():
    jax = pytest.importorskip("jax")
    from cudnn.gemm.cutedsl.grouped.backend_utils import wrapper_workspace

    device = jax.devices("gpu")[0]
    stream = torch.cuda.Stream()
    launch = cuda.CUstream(stream.cuda_stream)
    observed = torch.empty(256, dtype=torch.uint8, device="cuda")

    def run():
        with wrapper_workspace("jax", 256, device, launch) as workspace:
            assert cuda.cuMemsetD8Async(workspace.data_ptr(), 37, 256, launch)[0] == cuda.CUresult.CUDA_SUCCESS
            assert cuda.cuMemcpyDtoDAsync(observed.data_ptr(), workspace.data_ptr(), 256, launch)[0] == cuda.CUresult.CUDA_SUCCESS

    run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    for _ in range(3):
        observed.zero_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(observed, torch.full_like(observed, 37), rtol=0, atol=0)


@pytest.mark.L0
@pytest.mark.parametrize(
    "discrete,operand",
    [(False, x) for x in ("a_tensor", "c_tensor", "d_row_tensor", "padded_offsets", "alpha_tensor", "beta_tensor", "prob_tensor", "dprob_tensor", "b_tensor")]
    + [(True, "b_ptrs")],
)
def test_dglu_bf16_workspace_alias_rejected_before_launch(discrete, operand, monkeypatch):
    p = problem(discrete)
    op = make_plan(p, True, None)
    op.check_support()
    nbytes = op.scratch_workspace_bytes()
    args = dict(
        a_tensor=p["a"],
        c_tensor=p["c"],
        d_row_tensor=p["d"],
        d_col_tensor=None,
        sfa_tensor=None,
        padded_offsets=p["offsets"],
        alpha_tensor=p["alpha"],
        beta_tensor=p["beta"],
        prob_tensor=p["prob"],
        dprob_tensor=p["dp"],
        **p["bind"],
    )
    tensor = args[operand]
    elements = 1 + sum((extent - 1) * stride for extent, stride in zip(tensor.shape, tensor.stride()))
    storage = torch.empty(nbytes + elements * tensor.element_size() + 128, dtype=torch.uint8, device=tensor.device)

    # Legal adjacent slices share an allocation and must not be rejected.
    def view(offset):
        return storage[offset : offset + elements * tensor.element_size()].view(tensor.dtype).as_strided(tensor.shape, tensor.stride())

    seen = []
    monkeypatch.setattr(op._implementation, "_compiled_kernel", lambda *a, **k: seen.append((a, k)))
    args[operand] = view(nbytes)
    op.execute(**args, workspace=storage[:nbytes])
    assert len(seen) == 1
    seen.clear()
    args[operand] = view(0)
    # Never let deliberately aliased scratch reach a real kernel.
    with pytest.raises(ValueError, match="workspace must not overlap " + operand):
        op.execute(**args, workspace=storage[:nbytes])
    assert not seen


@pytest.mark.L0
@pytest.mark.parametrize(
    "discrete,operand",
    [
        (False, x)
        for x in (
            "a_tensor",
            "c_tensor",
            "d_row_tensor",
            "d_col_tensor",
            "sfa_tensor",
            "padded_offsets",
            "alpha_tensor",
            "beta_tensor",
            "prob_tensor",
            "dprob_tensor",
            "b_tensor",
            "sfb_tensor",
            "dbias_tensor",
            "sfd_row_tensor",
            "sfd_col_tensor",
            "amax_tensor",
            "norm_const_tensor",
        )
    ]
    + [(True, "b_ptrs"), (True, "sfb_ptrs")],
)
def test_dglu_blockscaled_workspace_alias_rejected_before_launch(discrete, operand, monkeypatch):
    from unittest.mock import Mock
    from cudnn.gemm.cutedsl.grouped.dglu._blockscaled_api import GroupedGemmDgluBlockScaledAPI
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode

    # Exercise the real execute gate, intercepting only the compiled consumer.
    op = object.__new__(GroupedGemmDgluBlockScaledAPI)
    op._logger = Mock()
    op._get_default_stream = lambda stream: stream
    op._has_dbias = True
    op.a_desc = Mock(device=torch.device("cuda", torch.cuda.current_device()))
    op.weight_mode = MoEWeightMode.DISCRETE if discrete else MoEWeightMode.DENSE
    op._compiled_kernel = Mock()
    op.scratch_workspace_bytes = lambda: 128
    monkeypatch.setattr(op, "_record_pointer_stream", lambda *a: None)
    names = [
        "a_tensor",
        "c_tensor",
        "d_row_tensor",
        "d_col_tensor",
        "sfa_tensor",
        "padded_offsets",
        "alpha_tensor",
        "beta_tensor",
        "prob_tensor",
        "dprob_tensor",
        "dbias_tensor",
        "sfd_row_tensor",
        "sfd_col_tensor",
        "amax_tensor",
        "norm_const_tensor",
    ]
    names += ["b_ptrs", "sfb_ptrs"] if discrete else ["b_tensor", "sfb_tensor"]
    # Packed FP4 A is a raw uint8 tensor. FP8 scale buffers must work too.
    args = {
        name: torch.empty(
            128, dtype=torch.uint8 if name == "a_tensor" else torch.float8_e4m3fn if name in ("sfa_tensor", "sfb_tensor") else torch.float32, device="cuda"
        )
        for name in names
    }
    storage = torch.empty(1024, dtype=torch.uint8, device="cuda")
    dtype = args[operand].dtype
    args[operand] = storage[128:640].view(dtype)
    op.execute(**args, workspace=storage[:128])
    assert op._compiled_kernel.call_count == 1
    op._compiled_kernel.reset_mock()
    # Begin inside the operand's strided byte span, rather than at its pointer.
    args[operand] = storage[:512].view(dtype)[::2]
    with pytest.raises(ValueError, match="workspace must not overlap " + operand):
        op.execute(**args, workspace=storage[128:256])
    op._compiled_kernel.assert_not_called()
