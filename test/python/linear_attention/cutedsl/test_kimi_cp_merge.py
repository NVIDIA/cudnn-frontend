# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge-only tests: the same FP32 H/M are shared across every precision."""

import json

import pytest
import torch

from cudnn.linear_attention.merge import KimiCPMerge, kimi_cp_merge
from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Kimi CP merge tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def supported_device():
    if not torch.cuda.is_available():
        pytest.skip("Kimi CP merge needs CUDA")
    from cudnn.api_base import TensorDesc

    sample = TensorDesc(torch.float32, (1, 1, 64, 64), (4096, 4096, 64, 1), (3, 2, 1, 0), torch.device("cuda", torch.cuda.current_device()))
    try:
        KimiCPMerge(sample, sample, sample, sample).check_support()
    except NotImplementedError as exc:
        pytest.skip(str(exc))


def inputs(p=8, h=2, v=64, k=64):
    # CPU draws make the corpus independent of device SM counts.
    rng = torch.Generator().manual_seed(918)
    hm = torch.randn(p, h, v, k, generator=rng) * 0.1
    mm = torch.randn(p, h, k, k, generator=rng) * 0.008 + torch.eye(k) * 0.94
    seed = torch.randn(1, h, v, k, generator=rng) * 0.1
    return tuple(x.cuda() for x in (hm, mm, seed))


def oracle(h, m, seed, reverse=False):
    # IEEE oracle, independent of ambient torch TF32 configuration.
    state = seed[0].double()
    product = torch.eye(m.shape[-1], device=m.device, dtype=torch.float64).expand(m.shape[1], -1, -1).clone()
    states = torch.empty_like(h, dtype=torch.float64)
    order = range(len(h) - 1, -1, -1) if reverse else range(len(h))
    for j in order:
        states[j] = state
        matrix = m[j].double().transpose(-1, -2) if reverse else m[j].double()
        state = state @ matrix + h[j].double()
        product = product @ matrix
    return states, state[None], product[None]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("precision,tol", [("ieee", 2e-6), ("bf16", 0.02), ("tf32", 0.003), ("tf32x3", 5e-6)])
@pytest.mark.parametrize("v,k", [(64, 64), (128, 64), (64, 128), (128, 128), (256, 128)])
def test_merge_precision(reverse, precision, tol, v, k):
    h, m, seed = inputs(v=v, k=k)
    expected = oracle(h, m, seed, reverse)
    actual = kimi_cp_merge(h, m, seed, reverse=reverse, precision=precision, output_transition=True)
    for got, want in zip(actual, expected):
        error = (got.double() - want).norm() / want.norm().clamp_min(1e-12)
        assert error < tol, (precision, error.item())
    # The first incoming state is copied, not quantized or multiplied by I.
    first = len(h) - 1 if reverse else 0
    assert torch.equal(actual["initial_states"][first], seed[0])


@pytest.mark.parametrize("reverse", [False, True])
def test_tf32x3_recovers_fp32_master_residual(reverse):
    h, m, seed = inputs(p=16, v=128, k=128)
    expected = oracle(h, m, seed, reverse)[1]
    errors = {}
    for precision in ("ieee", "bf16", "tf32", "tf32x3"):
        actual = kimi_cp_merge(h, m, seed, reverse=reverse, precision=precision)["final_state"]
        errors[precision] = ((actual.double() - expected).norm() / expected.norm()).item()
    assert errors["tf32x3"] < errors["tf32"] * 0.02
    assert errors["tf32x3"] < errors["bf16"] * 0.02
    print("MERGE_RELATIVE_L2 " + json.dumps(dict(reverse=reverse, pieces=16, heads=2, v=128, k=128, errors=errors)), flush=True)


def test_additive_h_is_not_quantized():
    h, m, _ = inputs(p=1)
    m.zero_()
    for precision in ("ieee", "bf16", "tf32", "tf32x3"):
        actual = kimi_cp_merge(h, m, precision=precision)
        assert torch.count_nonzero(actual["initial_states"]) == 0
        assert torch.equal(actual["final_state"][0], h[0])


@pytest.mark.parametrize("precision", ["ieee", "bf16", "tf32", "tf32x3"])
def test_prepared_merge_rebind_capture_no_alloc_or_sync(precision):
    h, m, seed = inputs()
    x, tail = torch.empty_like(h), torch.empty_like(seed)
    plan = KimiCPMerge(h, m, x, tail, seed, precision=precision).compile()
    assert plan.scratch_workspace_bytes() == 0
    plan.execute(h, m, x, tail, seed)
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        count = torch.cuda.memory_stats()["allocation.all.allocated"]
        for _ in range(3):
            plan.execute(h, m, x, tail, seed)
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == count
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    h2, m2, seed2 = h.clone(), m.clone(), seed.clone()
    x2, tail2 = torch.empty_like(x), torch.empty_like(tail)
    ambient = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(ambient)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            plan.execute(h2, m2, x2, tail2, seed2, current_stream=stream)
        ambient.wait_stream(stream)
        h2.mul_(1.5)
        graph.replay()
        torch.cuda.synchronize()
        expected = kimi_cp_merge(h2, m2, seed2, precision=precision)
        torch.testing.assert_close(tail2, expected["final_state"], rtol=0, atol=0)
        torch.testing.assert_close(x2, expected["initial_states"], rtol=0, atol=0)
    finally:
        graph.reset()


def test_merge_rejects_invalid_bindings_before_launch(monkeypatch):
    h, m, seed = inputs()
    x, tail = torch.empty_like(h), torch.empty_like(seed)
    plan = KimiCPMerge(h, m, x, tail, seed, precision="tf32").compile()
    import cudnn.linear_attention.frost.common.piece_chain as module

    monkeypatch.setattr(module, "run_state_chain", lambda *a, **kw: pytest.fail("invalid binding reached launch"))
    for kwargs in (dict(h=h.half()), dict(m=m[:1]), dict(initial_states=h), dict(initial_state=None), dict(transition=tail)):
        tensors = dict(h=h, m=m, initial_states=x, final_state=tail, initial_state=seed)
        tensors.update(kwargs)
        with pytest.raises(ValueError):
            plan.execute(**tensors)


def test_prepared_merge_rejects_autograd_declaration_before_compile(monkeypatch):
    h, m, seed = inputs()
    h.requires_grad_(True)
    plan = KimiCPMerge(h, m, torch.empty_like(h), torch.empty_like(seed), seed)
    import cudnn.linear_attention.frost.common.piece_chain as module

    monkeypatch.setattr(module, "build_state_chain", lambda **kwargs: pytest.fail("autograd declaration reached compiler"))
    with pytest.raises(ValueError, match="no autograd"):
        plan.compile()


def test_requires_grad_input_is_allowed_with_grad_disabled():
    h, m, seed = inputs()
    h.requires_grad_(True)
    with torch.no_grad():
        actual = kimi_cp_merge(h, m, seed, precision="tf32x3")
        expected = kimi_cp_merge(h.detach(), m, seed, precision="tf32x3")
    torch.testing.assert_close(actual["initial_states"], expected["initial_states"], rtol=0, atol=0)
    torch.testing.assert_close(actual["final_state"], expected["final_state"], rtol=0, atol=0)


def test_merge_cache_modes_do_not_alias():
    from cudnn.linear_attention.frost.common.piece_chain import build_state_chain

    args = dict(
        heads_out=2,
        dim_v=64,
        dim_k=64,
        pieces=8,
        transpose=False,
        has_seed=True,
        has_tail=True,
        emit_summary=False,
        device=torch.cuda.current_device(),
        opt_level=3,
    )
    compiled = [build_state_chain(**args, merge_precision=p).compiled for p in ("ieee", "bf16", "tf32", "tf32x3")]
    assert len({id(c) for c in compiled}) == 4
    assert build_state_chain(**args, merge_precision="tf32").compiled is compiled[2]
    with pytest.raises(ValueError, match="merge_precision"):
        build_state_chain(**args, merge_precision="fp16")


@pytest.mark.parametrize("precision", ["ieee", "bf16", "tf32", "tf32x3"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("product_only", [False, True])
def test_ragged_piece_chain_and_seed_indices(precision, reverse, product_only):
    from cudnn.linear_attention.frost.common.piece_chain import build_state_chain, run_state_chain

    h, m, _ = inputs(p=6)
    seed = inputs(p=3)[0][:, :, :64, :64].clone()
    x, tail = torch.empty_like(h), torch.empty_like(seed)
    product = torch.empty((3, 2, 64, 64), dtype=torch.float32, device="cuda")
    row_table = torch.tensor([0, 2, 6, 12], dtype=torch.int32, device="cuda")  # 1,2,3 pieces * 2 heads
    seed_indices = torch.tensor([2, 0, 1], dtype=torch.int32, device="cuda")
    plan = build_state_chain(
        heads_out=2,
        dim_v=64,
        dim_k=64,
        pieces=4,
        transpose=reverse,
        has_seed=not product_only,
        has_tail=not product_only,
        emit_summary=True,
        filled_only=True,
        has_seed_indices=not product_only,
        device=torch.cuda.current_device(),
        opt_level=3,
        merge_precision=precision,
    )
    run_state_chain(
        plan,
        3,
        h if not product_only else None,
        m,
        x if not product_only else None,
        seed if not product_only else None,
        tail if not product_only else None,
        product,
        torch.cuda.current_stream().cuda_stream,
        main_rows=row_table,
        seed_indices=seed_indices if not product_only else None,
    )
    tol = dict(ieee=2e-6, bf16=0.02, tf32=0.003, tf32x3=5e-6)[precision]
    for seq, (start, end) in enumerate(((0, 1), (1, 3), (3, 6))):
        seed_row = (2, 0, 1)[seq]
        expected = oracle(h[start:end], m[start:end], seed[seed_row : seed_row + 1], reverse)
        actual = ((x[start:end], expected[0]), (tail[seq : seq + 1], expected[1]), (product[seq : seq + 1], expected[2]))
        if product_only:
            actual = actual[-1:]
        for got, want in actual:
            assert ((got.double() - want).norm() / want.norm().clamp_min(1e-12)) < tol


@pytest.mark.parametrize("precision", ["bf16", "tf32", "tf32x3"])
def test_single_piece_copy_only_does_not_read_summary(precision):
    from cudnn.linear_attention.frost.common.piece_chain import build_state_chain, run_state_chain

    h, m, seed = inputs(p=1)
    h.fill_(float("nan"))
    m.fill_(float("nan"))
    x = torch.empty_like(h)
    rows = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    plan = build_state_chain(
        heads_out=2,
        dim_v=64,
        dim_k=64,
        pieces=4,
        transpose=False,
        has_seed=True,
        has_tail=False,
        emit_summary=False,
        filled_only=True,
        device=torch.cuda.current_device(),
        opt_level=3,
        merge_precision=precision,
    )
    run_state_chain(plan, 1, h, m, x, seed, None, None, torch.cuda.current_stream().cuda_stream, main_rows=rows)
    assert torch.equal(x, seed)


@pytest.mark.parametrize("bad", ["dtype", "stride", "shape", "autograd", "precision", "reverse", "alias"])
def test_wrapper_rejects_before_compilation(monkeypatch, bad):
    import cudnn.linear_attention.merge as module

    h, m, seed = inputs()
    kwargs = dict(initial_state=seed)
    if bad == "dtype":
        h = h.half()
    elif bad == "stride":
        m = m.transpose(-1, -2)
    elif bad == "shape":
        m = m[:1]
    elif bad == "autograd":
        h.requires_grad_(True)
    elif bad == "precision":
        kwargs["precision"] = "fp16"
    elif bad == "alias":
        m = h
    else:
        kwargs["reverse"] = "yes"
    monkeypatch.setattr(module, "_plan", lambda *a, **kw: pytest.fail("invalid input reached compiler"))
    with pytest.raises(ValueError):
        module.kimi_cp_merge(h, m, **kwargs)


def test_old_dsl_declines_before_kernel_import(monkeypatch):
    import cudnn.linear_attention.merge as module

    h, m, seed = inputs()
    monkeypatch.setattr(module, "cutedsl_requirement_error", lambda name: "requires nvidia-cutlass-dsl >= 4.8.0; found 4.7.2")
    with pytest.raises(NotImplementedError, match="4.7.2"):
        KimiCPMerge(h, m, torch.empty_like(h), torch.empty_like(seed)).check_support()


def test_internal_precision_gate_precedes_kernel_import(monkeypatch):
    import sys
    import cudnn.linear_attention.frost.common.piece_chain as module

    kernel_module = "cudnn.linear_attention.frost.common.state_merge_mma"
    sys.modules.pop(kernel_module, None)
    monkeypatch.setattr(module, "cutedsl_requirement_error", lambda name: "requires nvidia-cutlass-dsl >= 4.8.0; found 4.7.2")
    with pytest.raises(NotImplementedError, match="4.7.2"):
        module.build_state_chain(
            heads_out=1,
            dim_v=64,
            dim_k=64,
            pieces=1,
            transpose=False,
            has_seed=False,
            has_tail=True,
            emit_summary=False,
            device=torch.cuda.current_device(),
            opt_level=97,
            merge_precision="bf16",
        )
    assert kernel_module not in sys.modules


def test_legacy_state_chain_compile_signature():
    """Existing compiled KDA hosts omit the new constexpr precision argument."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cudnn.linear_attention.frost.common.piece_chain import build_state_chain, launch_state_chain, run_state_chain

    h, m, seed = inputs(p=2)
    x, tail = torch.empty_like(h), torch.empty_like(seed)
    args = dict(
        heads_out=2,
        dim_v=64,
        dim_k=64,
        pieces=2,
        transpose=False,
        has_seed=True,
        has_tail=True,
        emit_summary=False,
        device=torch.cuda.current_device(),
        opt_level=3,
    )
    facts = build_state_chain(**args)
    tensors = [from_dlpack(t, assumed_align=16) for t in (h, m, x, seed, tail)]
    compiled = cute.compile(
        launch_state_chain,
        cutlass.Int32(2),
        64,
        64,
        8,
        cutlass.Int32(2),
        False,
        True,
        True,
        False,
        cutlass.Int32(1),
        *tensors,
        None,
        None,
        None,
        cuda.CUstream(0),
        options="--enable-tvm-ffi --opt-level 3",
    )
    run_state_chain(facts._replace(compiled=compiled), 1, h, m, x, seed, tail, None, torch.cuda.current_stream().cuda_stream)
    expected = kimi_cp_merge(h, m, seed, precision="ieee")
    torch.testing.assert_close(x, expected["initial_states"], rtol=0, atol=0)
    torch.testing.assert_close(tail, expected["final_state"], rtol=0, atol=0)
