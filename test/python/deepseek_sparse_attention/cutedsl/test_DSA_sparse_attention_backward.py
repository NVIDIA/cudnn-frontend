# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SparseAttentionBackward.

Gradients are generated via autograd on the gather-based forward reference.
"""

import ast
import inspect
import math
import textwrap

import pytest
import torch

from test_utils import assert_bitwise_runs, bitwise_bits, torch_fork_set_rng

from deepseek_sparse_attention.cutedsl.dsa_utils import (
    _require_exact_sm100,
    _require_sm90,
    _require_sm100,
    dsa_init,
    with_dsa_sparse_attention_backward_params,
)
from deepseek_sparse_attention.cutedsl.dsa_reference import (
    check_ref_dsa_sparse_attention_backward,
    ref_sparse_attention_forward_chunked,
)

# SM100-class compute capabilities routed to the H96/H128 D576 two-CTA backend.
_TWO_CTA_CAPABILITIES = ((10, 0), (10, 3))


def _import_or_skip(module_name):
    """Import an optional CuTeDSL module or skip when the environment lacks it."""
    return pytest.importorskip(module_name, reason="Environment not supported: cudnn[cutedsl] not installed")


@pytest.mark.L0
@pytest.mark.parametrize("sm_count,expected", [(132, 66), (148, 74), (152, 76), (160, 76)])
def test_DSA_sparse_attention_backward_d576_deterministic_compact_slots(sm_count, expected):
    """One deterministic wave uses every complete two-SM cluster up to the compact workspace cap."""
    interface = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100_d576")

    assert interface._deterministic_compact_slots(sm_count) == expected


def _assert_qcluster_dq_close(actual, expected, msg):
    """Reject erased dQ while allowing q-cluster accumulation-order noise."""
    actual_float = actual.float()
    expected_float = expected.float()
    assert torch.count_nonzero(expected_float).item() > 0, f"{msg}: reference dq is zero"
    assert torch.count_nonzero(actual_float).item() > 0, f"{msg}: computed dq is zero"
    relative_l2 = (actual_float - expected_float).norm() / expected_float.norm()
    assert relative_l2 < 2e-2, f"{msg}: relative L2 error {relative_l2.item():.3e}"


def _spy_d576_two_cta_execute(monkeypatch):
    """Count calls into the D576 two-CTA launch path; the selector alone does not prove the route ran."""
    from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm100_d576

    calls = []
    real_execute = _interface_sm100_d576._execute_d576_2cta

    def counted_execute(*args, **kwargs):
        calls.append(1)
        return real_execute(*args, **kwargs)

    monkeypatch.setattr(_interface_sm100_d576, "_execute_d576_2cta", counted_execute)
    return calls


def _spy_prepared_sm100_execute(monkeypatch):
    """Record the backend and launchable of every freshly prepared SM100 plan."""
    from cudnn.deepseek_sparse_attention.sparse_attention_backward import api

    # Do not let a plan built by an earlier test hide the compile() path. Keep
    # the lower artifact memo intact so this still exercises normal reuse.
    monkeypatch.setattr(api, "_cache_of_SparseAttentionBackwardObjects", {})
    calls = []
    real_execute = api.SparseAttentionBackward.execute

    def recorded_execute(plan, *args, **kwargs):
        calls.append((plan._backend, plan._compiled_kernel))
        return real_execute(plan, *args, **kwargs)

    monkeypatch.setattr(api.SparseAttentionBackward, "execute", recorded_execute)
    return calls


def _assert_prepared_sm100_backend(calls, expected_backend):
    """Require actual prepared execution through the expected compiled route."""
    assert calls, "the wrapper did not execute a prepared SM100 plan"
    assert all(backend == expected_backend for backend, _ in calls), f"expected only {expected_backend} prepared calls, got {calls}"
    assert all(callable(compiled_kernel) for _, compiled_kernel in calls), "prepared execute did not receive a compiled launchable"


def _allocate(cfg, has_topk_length: bool):
    total_s_q = cfg["s_q"]
    total_s_kv = cfg["s_kv"]
    h = cfg["h_q"] if cfg.get("h_q") else 64
    d = cfg["head_dim"]
    topk = cfg["topk"]
    device = "cuda"

    q = torch.randn(total_s_q, h, d, dtype=torch.bfloat16, device=device)
    kv = torch.randn(total_s_kv, d, dtype=torch.bfloat16, device=device)
    attn_sink = torch.randn(h, dtype=torch.float32, device=device)

    topk_k = min(topk, total_s_kv)
    topk_idxs = torch.stack([torch.randperm(total_s_kv, device=device)[:topk_k] for _ in range(total_s_q)]).to(torch.int32)
    if topk_k < topk:
        pad = torch.full((total_s_q, topk - topk_k), -1, dtype=torch.int32, device=device)
        topk_idxs = torch.cat([topk_idxs, pad], dim=-1)

    topk_length = None
    if has_topk_length:
        topk_length = torch.randint(1, topk_k + 1, (total_s_q,), dtype=torch.int32, device=device)

    return q, kv, attn_sink, topk_idxs, topk_length


def _plan_case(
    num_heads,
    head_dim,
    topk,
    *,
    s_q=8,
    s_kv=256,
    dtype=torch.bfloat16,
    has_topk_length=False,
    deterministic=False,
    q_cluster_mode="off",
    compile_plan=True,
):
    """Inputs, a checked plan, caller-owned outputs, and scratch for one route."""
    from cudnn import DSA

    device = torch.device("cuda")
    softmax_scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(s_q, num_heads, head_dim, dtype=dtype, device=device) / 10
    kv = torch.randn(s_kv, head_dim, dtype=dtype, device=device) / 10
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    topk_length = torch.randint(1, topk + 1, (s_q,), dtype=torch.int32, device=device) if has_topk_length else None
    out, lse = ref_sparse_attention_forward_chunked(q, kv, attn_sink, topk_idxs, topk_length=topk_length, softmax_scale=softmax_scale)
    dout = torch.randn_like(out)
    plan = DSA.SparseAttentionBackward(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        sample_topk_length=topk_length,
        softmax_scale=softmax_scale,
        deterministic=deterministic,
        q_cluster_mode=q_cluster_mode,
    )
    assert plan.check_support()
    if compile_plan:
        plan.compile()
    workspace_bytes = plan.scratch_workspace_bytes()
    assert workspace_bytes > 0
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
    inputs = (q, kv, out, dout, lse, attn_sink, topk_idxs)
    outputs = (torch.empty_like(q), torch.empty_like(kv), torch.empty_like(attn_sink))
    return plan, inputs, outputs, topk_length, softmax_scale, workspace


def _assert_execute_allocates_nothing_and_never_synchronizes(plan, inputs, outputs, topk_length, softmax_scale, workspace):
    """Recipe R9: three warm executes with caller outputs and scratch make no torch allocation and never block."""
    from cuda.bindings import driver as cuda

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def execute():
        dq, dkv, d_sink = outputs
        return plan.execute(*inputs, dq, dkv, topk_length=topk_length, softmax_scale=softmax_scale, workspace=workspace, current_stream=stream, d_sink=d_sink)

    execute()  # warm the legacy SM90 route and establish steady-state allocator counters
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            result = execute()
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()["allocation.all.allocated"]
    assert after == before, f"execute() made {after - before} torch allocation(s) (Rule 8, recipe R9)"
    assert all(actual is expected for actual, expected in zip(result, outputs))
    q, kv, out, dout, lse, attn_sink, topk_idxs = inputs
    check_ref_dsa_sparse_attention_backward(
        q, kv, attn_sink, topk_idxs, out, dout, lse, *result, softmax_scale=softmax_scale, topk_length=topk_length, atol=5e-2, rtol=5e-2
    )


@pytest.mark.parametrize(
    "num_heads,head_dim,expected_backend,expected_block_tile",
    [
        (16, 576, "h16_m128", 128),
        (16, 512, "generic_m64", 64),
        (32, 576, "h32_m64", 64),
        (64, 576, "generic_m64", 64),
        (96, 576, "h96_h64_h32", 64),
        (128, 576, "generic_m64", 64),
    ],
)
@pytest.mark.L0
def test_DSA_sparse_attention_backward_sm100_auto_dispatch(
    num_heads,
    head_dim,
    expected_backend,
    expected_block_tile,
):
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100 import _select_sm100_backend
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    assert _select_sm100_backend(num_heads, head_dim) == (expected_backend, expected_block_tile)


@pytest.mark.L0
def test_DSA_sparse_attention_backward_sm100_h96_dispatch():
    """Select native H96 only inside the validated two-CTA envelope."""
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100 import _select_sm100_backend
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    assert _select_sm100_backend(96, 576, max_topk=512) == ("h96_h64_h32", 64)
    assert _select_sm100_backend(96, 576, max_topk=1024) == ("h96_h64_h32", 64)
    assert _select_sm100_backend(96, 576, max_topk=2048, deterministic=True) == ("h96_h64_h32", 64)
    for max_topk in range(2048, 2052):
        assert _select_sm100_backend(
            96,
            576,
            head_dim_v=512,
            dtype=torch.bfloat16,
            max_topk=max_topk,
            device_capability=(10, 0),
            deterministic=False,
        ) == ("h96_d576_2cta_m64", 64)
        assert _select_sm100_backend(
            96,
            576,
            head_dim_v=512,
            dtype=torch.bfloat16,
            max_topk=max_topk,
            device_capability=(10, 0),
            deterministic=True,
        ) == ("h96_d576_2cta_m64", 64)
    assert _select_sm100_backend(
        96,
        576,
        head_dim_v=512,
        dtype=torch.bfloat16,
        max_topk=2048,
        device_capability=(10, 7),
        deterministic=True,
    ) == ("h96_h64_h32", 64)


@pytest.mark.L0
def test_DSA_sparse_attention_backward_h96_ordinary_skips_padded_dkv_generation():
    """Ordinary H96 produces 12 useful round operands; deterministic H96 retains its fixed 16."""
    module = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_h128_d576_2cta")
    common = (module.BFloat16, 576, 512, 64, 128)

    ordinary_h96 = module.FlashAttentionDSABackwardSm100H128D576TwoCTA(*common, num_heads=96)
    ordinary_h128 = module.FlashAttentionDSABackwardSm100H128D576TwoCTA(*common, num_heads=128)
    deterministic_h96 = module.FlashAttentionDSABackwardSm100H128D576TwoCTADeterministic(*common, num_heads=96)

    assert (ordinary_h96.round_gens_per_group, ordinary_h96.round_gens_per_tile) == (3, 12)
    assert (ordinary_h128.round_gens_per_group, ordinary_h128.round_gens_per_tile) == (4, 16)
    assert (deterministic_h96.round_gens_per_group, deterministic_h96.round_gens_per_tile) == (4, 16)


@pytest.mark.L0
def test_DSA_sparse_attention_backward_sm100_h96_deterministic_composition():
    """The H96 stages must share one fixed-owner reduction policy."""
    interface = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100")
    h96 = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_h96")

    kernel_cls = interface._get_sm100_kernel_class("h96_h64_h32", deterministic=True)
    assert kernel_cls is h96.FlashAttentionDSABackwardSm100H96Deterministic
    assert kernel_cls.num_dkv_shards == 128
    assert h96.FlashAttentionDSABackwardSm100H96H64Deterministic.num_dkv_shards == 128
    assert h96.FlashAttentionDSABackwardSm100H96H64Deterministic.q_wave_ctas == 128
    assert not h96.FlashAttentionDSABackwardSm100H96H64Deterministic.finalize_dkv
    assert h96.FlashAttentionDSABackwardSm100H96H32Deterministic.num_dkv_shards == 128
    assert h96.FlashAttentionDSABackwardSm100H96H32Deterministic.q_wave_ctas == 128
    assert not h96.FlashAttentionDSABackwardSm100H96H32Deterministic.initialize_dkv
    assert h96.FlashAttentionDSABackwardSm100H96H32Deterministic.finalize_dkv

    adaptive_bytes = interface.flash_attn_bwd_sm100_workspace_size(7, 1025, 576, 96, deterministic=True, backend="generic_m64")
    h96_bytes = interface.flash_attn_bwd_sm100_workspace_size(7, 1025, 576, 96, deterministic=True, backend="h96_h64_h32")
    assert h96_bytes > adaptive_bytes


@pytest.mark.L0
@pytest.mark.parametrize("max_topk", [512, 1024])
@pytest.mark.parametrize(
    "num_heads,head_dim,expected_backend,static_stride",
    [
        (16, 512, "generic_m64", False),
        (16, 576, "h16_m128", True),
        (32, 512, "generic_m64", False),
        (32, 576, "h32_m64", True),
        (64, 512, "generic_m64", False),
        (64, 576, "generic_m64", False),
        (96, 512, "generic_m64", False),
        (96, 576, "h96_d576_2cta_m64", True),
        (128, 512, "h128_2cta_m64", False),
    ],
)
def test_DSA_sparse_attention_backward_sm100_topk_stride_specialization(max_topk, num_heads, head_dim, expected_backend, static_stride, monkeypatch):
    """Specialize only the measured backends, without making Q a compile key."""
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm100 as interface
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    backend, _ = interface._select_sm100_backend(
        num_heads,
        head_dim,
        head_dim_v=512,
        dtype=torch.bfloat16,
        max_topk=max_topk,
        device_capability=(10, 0),
    )
    assert backend == expected_backend

    symbolic_q = object()
    static_tensor, dynamic_tensor = object(), object()
    fake_calls = []
    dynamic_calls = []
    monkeypatch.setattr(interface.cute, "sym_int", lambda: symbolic_q)

    def fake_compact(dtype, shape, **kwargs):
        """Record the compile-time tensor layout requested by the backend."""
        fake_calls.append((dtype, shape, kwargs))
        return static_tensor

    def dynamic(tensor):
        """Record the fallback conversion of the runtime tensor."""
        dynamic_calls.append(tensor)
        return dynamic_tensor

    monkeypatch.setattr(interface.cute.runtime, "make_fake_compact_tensor", fake_compact)
    monkeypatch.setattr(interface, "to_cute_tensor", dynamic)
    indices = torch.empty((3, max_topk), dtype=torch.int32)
    result = interface._topk_indices_compile_tensor(indices, max_topk, backend)
    if static_stride:
        assert result is static_tensor
        assert fake_calls == [(interface.cutlass.Int32, (symbolic_q, max_topk), {"stride_order": (1, 0), "assumed_align": 16})]
        assert not dynamic_calls
    else:
        assert result is dynamic_tensor
        assert dynamic_calls == [indices]
        assert not fake_calls


@pytest.mark.L0
def test_DSA_sparse_attention_backward_qcluster_public_contract():
    """Keep adaptive pairing opt-in without exposing a shared-count knob."""
    from cudnn import DSA

    wrapper = inspect.signature(DSA.sparse_attention_backward_wrapper).parameters
    plan = inspect.signature(DSA.SparseAttentionBackward).parameters
    assert wrapper["q_cluster_mode"].default == "off"
    assert plan["q_cluster_mode"].default == "off"
    assert wrapper["deterministic_relay_cluster_size"].default is None
    assert plan["deterministic_relay_cluster_size"].default is None
    assert "q_cluster_shared_topk" not in wrapper

    dummy = torch.empty(1)
    with pytest.raises(ValueError, match="q_cluster_mode must be 'off' or 'adaptive_pair'"):
        DSA.sparse_attention_backward_wrapper(dummy, dummy, dummy, dummy, dummy, dummy, dummy, q_cluster_mode="fixed")


@pytest.mark.L0
@pytest.mark.parametrize("value", [True, 0, 2, 128, 4.0, "4"])
def test_DSA_sparse_attention_backward_relay_cluster_public_validation(value):
    """Reject coercible and unsupported values before staging any tensors."""
    from cudnn import DSA

    dummy = torch.empty(1)
    with pytest.raises(ValueError, match=r"must be one of \{4, 8, 16, 32, 64\} or None"):
        DSA.sparse_attention_backward_wrapper(
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            deterministic=True,
            deterministic_relay_cluster_size=value,
        )


@pytest.mark.L0
def test_DSA_sparse_attention_backward_relay_cluster_requires_deterministic():
    """Never silently ignore a relay override on the ordinary path."""
    from cudnn import DSA

    dummy = torch.empty(1)
    with pytest.raises(ValueError, match="requires deterministic=True"):
        DSA.sparse_attention_backward_wrapper(
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            deterministic_relay_cluster_size=4,
        )


@pytest.mark.L0
@pytest.mark.parametrize(
    "heads,topk,expected_tile,expected_divisor",
    [(16, 257, 128, 8), (16, 2047, 128, 8), (16, 2048, 128, 8), (32, 129, 64, 6), (32, 2047, 64, 6), (32, 2048, 64, 6)],
)
def test_DSA_sparse_attention_backward_qcluster_workspace_is_topk_bounded(heads, topk, expected_tile, expected_divisor):
    """Metadata storage scales with top-k and Q, never with the KV extent."""
    from cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100 import _align_workspace_bytes, flash_attn_bwd_sm100_workspace_size
    from cudnn.deepseek_sparse_attention.sparse_attention_backward._qcluster_sm100 import _layout, workspace_size

    q, dim = 7, 576
    tile, shared, stride, threshold_divisor = _layout(topk, heads)
    assert tile == expected_tile
    assert shared == ((topk - 1) // tile) * tile
    assert threshold_divisor == expected_divisor
    extra = _align_workspace_bytes(q * stride * 4)
    extra += _align_workspace_bytes((q + (q + 1) // 2) * 4)
    extra += _align_workspace_bytes(2 * heads * 4)

    for kv in (4096, 262144):
        base = flash_attn_bwd_sm100_workspace_size(q, kv, dim, heads)
        assert workspace_size(q, kv, dim, heads, topk) == base + extra


@pytest.mark.L1
def test_DSA_sparse_attention_backward_qcluster_compile_key_is_shape_dynamic():
    """Reuse one compiled artifact across different Q and KV extents."""
    _require_exact_sm100()
    from cudnn.deepseek_sparse_attention.sparse_attention_backward._qcluster_sm100 import _compile_artifacts, compile_plan

    capability = torch.cuda.get_device_capability()
    _compile_artifacts.cache_clear()
    first = compile_plan(capability, 5, 1024, 16, 576, 512, False)
    first_info = _compile_artifacts.cache_info()
    second = compile_plan(capability, 9, 1536, 16, 576, 512, False)
    second_info = _compile_artifacts.cache_info()

    assert first_info.misses == second_info.misses == 1
    assert second_info.hits == first_info.hits + 1
    assert first.transform is second.transform
    assert first.shared is second.shared


@pytest.mark.L0
def test_DSA_sparse_attention_backward_deterministic_compile_cache_is_device_scoped(monkeypatch):
    """Same-arch devices must not share context-owned CuTe launchables."""
    from contextlib import contextmanager

    from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm100 as interface

    compiled = []
    current_device = [None]

    @contextmanager
    def device_context(device):
        """Emulate a restoring CUDA device guard for the host-only cache test."""
        previous, current_device[0] = current_device[0], device
        try:
            yield
        finally:
            current_device[0] = previous

    class FakeKernel:
        """Accept production kernel-constructor arguments without codegen."""

        def __init__(self, **kwargs):
            """Construct a storage-free stand-in for a deterministic kernel."""
            pass

    def fake_compile(*args, **kwargs):
        """Record the guarded device and return a unique launchable stand-in."""
        artifact = object()
        compiled.append((current_device[0], kwargs["options"], artifact))
        return artifact

    compile_artifact = interface._compile_deterministic_sm100
    compile_artifact.cache_clear()
    monkeypatch.setattr(interface, "_deterministic_kernel_class_for_config", lambda shards, relay_size: FakeKernel)
    monkeypatch.setattr(interface, "_fake_compact_tensor", lambda *args, **kwargs: object())
    monkeypatch.setattr(interface.cute, "sym_int", lambda: object())
    monkeypatch.setattr(interface.cute.runtime, "make_fake_stream", lambda **kwargs: object())
    monkeypatch.setattr(interface.cute, "compile", fake_compile)
    monkeypatch.setattr(torch.cuda, "device", device_context)

    def build(device, capability=(10, 0), relay_cluster_size=1):
        """Compile one representative signature for a device and capability."""
        return compile_artifact(device, capability, "generic_m64", torch.bfloat16, 512, 512, 64, 64, True, 32, relay_cluster_size)

    try:
        device0 = build(0)
        device0_cluster4 = build(0, relay_cluster_size=4)
        device1 = build(1)
        device1_again = build(1)
        sm103 = build(2, (10, 3))
        sm107 = build(3, (10, 7))
    finally:
        compile_artifact.cache_clear()

    assert device0_cluster4 is not device0
    assert device0 is not device1
    assert device1_again is device1
    assert sm103 is not device1
    assert sm107 is not sm103
    assert [(device, options) for device, options, _ in compiled] == [
        (0, "--enable-tvm-ffi --gpu-arch sm_100a"),
        (0, "--enable-tvm-ffi --gpu-arch sm_100a"),
        (1, "--enable-tvm-ffi --gpu-arch sm_100a"),
        (2, "--enable-tvm-ffi --gpu-arch sm_103a"),
        (3, "--enable-tvm-ffi --gpu-arch sm_107a"),
    ]
    assert current_device[0] is None


@pytest.mark.L1
@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("heads", [16, 32])
@pytest.mark.parametrize("with_lengths", [False, True], ids=["full", "lengths"])
def test_DSA_sparse_attention_backward_qcluster_matches_baseline_and_graph(heads, with_lengths, topk):
    """Shared, fallback, and odd-tail rows remain correct on graph replay."""
    _require_exact_sm100()
    from cudnn import DSA

    torch.manual_seed(31)
    q_len, kv_len, dim, overlap = 5, 2 * topk, 576, 3 * topk // 4
    q = torch.randn(q_len, heads, dim, dtype=torch.bfloat16, device="cuda") / 10
    kv = torch.randn(kv_len, dim, dtype=torch.bfloat16, device="cuda") / 10
    sink = torch.randn(heads, dtype=torch.float32, device="cuda")
    first = torch.randperm(kv_len, device="cuda")[:topk]
    remaining = torch.arange(kv_len, device="cuda")[~torch.isin(torch.arange(kv_len, device="cuda"), first)]
    second = torch.cat((first[:overlap], remaining[: topk - overlap]))
    third = torch.randperm(kv_len, device="cuda")[:topk]
    fourth = torch.arange(kv_len, device="cuda")[~torch.isin(torch.arange(kv_len, device="cuda"), third)][:topk]
    fourth[0] = fourth[1]
    fourth[-1] = -1
    fifth = torch.randperm(kv_len, device="cuda")[:topk]
    topk_idxs = torch.stack((first, second, third, fourth, fifth)).to(torch.int32)
    lengths = torch.tensor([topk // 4, topk // 4, topk, topk, topk], dtype=torch.int32, device="cuda") if with_lengths else None
    scale = dim**-0.5
    out, lse = ref_sparse_attention_forward_chunked(q, kv, sink, topk_idxs, topk_length=lengths, softmax_scale=scale)
    dout = torch.randn_like(out)

    baseline = DSA.sparse_attention_backward_wrapper(q, kv, out, dout, lse, sink, topk_idxs, softmax_scale=scale, topk_length=lengths)
    plan = DSA.SparseAttentionBackward(
        q,
        kv,
        out,
        dout,
        lse,
        sink,
        topk_idxs,
        sample_topk_length=lengths,
        softmax_scale=scale,
        q_cluster_mode="adaptive_pair",
    )
    assert plan.check_support()
    plan.compile()
    workspace = torch.empty(plan.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")
    dq, dkv = torch.empty_like(q), torch.empty_like(kv)

    def run():
        return DSA.sparse_attention_backward_wrapper(
            q,
            kv,
            out,
            dout,
            lse,
            sink,
            topk_idxs,
            softmax_scale=scale,
            topk_length=lengths,
            dq=dq,
            dkv=dkv,
            workspace=workspace,
            q_cluster_mode="adaptive_pair",
        )

    result = run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_result = run()
    graph.replay()
    torch.cuda.synchronize()
    try:
        _assert_qcluster_dq_close(result["dq"], baseline["dq"], "dq")
        _assert_qcluster_dq_close(graph_result["dq"], baseline["dq"], "graph dq")
        for name in ("dkv", "d_sink"):
            torch.testing.assert_close(result[name], baseline[name], rtol=5e-2, atol=5e-2, msg=name)
            torch.testing.assert_close(graph_result[name], baseline[name], rtol=5e-2, atol=5e-2, msg=f"graph {name}")
    finally:
        graph.reset()

    workspace_bytes = plan.scratch_workspace_bytes()
    dq_bytes, dkv_bytes, sink_bytes = q.numel() * q.element_size(), kv.numel() * kv.element_size(), sink.numel() * sink.element_size()
    carrier = torch.empty(workspace_bytes + dq_bytes + dkv_bytes + sink_bytes, dtype=torch.uint8, device="cuda")
    direct_workspace = carrier[:workspace_bytes]
    offset = workspace_bytes
    direct_dq = carrier[offset : offset + dq_bytes].view(torch.bfloat16).view_as(q)
    offset += dq_bytes
    direct_dkv = carrier[offset : offset + dkv_bytes].view(torch.bfloat16).view_as(kv)
    offset += dkv_bytes
    direct_d_sink = carrier[offset : offset + sink_bytes].view(torch.float32).view_as(sink)
    direct = plan.execute(q, kv, out, dout, lse, sink, topk_idxs, direct_dq, direct_dkv, lengths, scale, workspace=direct_workspace, d_sink=direct_d_sink)
    _assert_qcluster_dq_close(direct[0], baseline["dq"], "direct dq")
    for name, actual in zip(("dkv", "d_sink"), direct[1:]):
        torch.testing.assert_close(actual, baseline[name], rtol=5e-2, atol=5e-2, msg=f"direct {name}")

    aliased_dq = direct_workspace[:dq_bytes].view(torch.bfloat16).view_as(q)
    with pytest.raises(ValueError, match="must not overlap"):
        plan.execute(q, kv, out, dout, lse, sink, topk_idxs, aliased_dq, direct_dkv, lengths, scale, workspace=direct_workspace, d_sink=direct_d_sink)


@pytest.mark.L1
@pytest.mark.parametrize("heads", [16, 32])
def test_DSA_sparse_attention_backward_qcluster_graph_replay_switches_overlap_path(heads):
    """One captured graph follows in-place high/zero/high overlap changes."""
    _require_exact_sm100()
    from cudnn import DSA

    torch.manual_seed(32)
    q_len, topk, kv_len, dim = 4, 512, 2048, 576
    overlap = 3 * topk // 4
    rows = torch.arange(kv_len, dtype=torch.int32, device="cuda").view(q_len, topk)
    high_overlap = rows.clone()
    high_overlap[1] = torch.cat((rows[0, :overlap], rows[1, : topk - overlap]))
    high_overlap[3] = torch.cat((rows[2, :overlap], rows[3, : topk - overlap]))
    zero_overlap = rows.clone()

    q = torch.randn(q_len, heads, dim, dtype=torch.bfloat16, device="cuda") / 10
    kv = torch.randn(kv_len, dim, dtype=torch.bfloat16, device="cuda") / 10
    sink = torch.randn(heads, dtype=torch.float32, device="cuda")
    dout = torch.randn(q_len, heads, 512, dtype=torch.bfloat16, device="cuda") / 10
    scale = dim**-0.5
    high_out, high_lse = ref_sparse_attention_forward_chunked(q, kv, sink, high_overlap, softmax_scale=scale)
    zero_out, zero_lse = ref_sparse_attention_forward_chunked(q, kv, sink, zero_overlap, softmax_scale=scale)
    baselines = {
        "high": DSA.sparse_attention_backward_wrapper(q, kv, high_out, dout, high_lse, sink, high_overlap, softmax_scale=scale),
        "zero": DSA.sparse_attention_backward_wrapper(q, kv, zero_out, dout, zero_lse, sink, zero_overlap, softmax_scale=scale),
    }

    runtime_indices = high_overlap.clone()
    runtime_out = high_out.clone()
    runtime_lse = high_lse.clone()
    plan = DSA.SparseAttentionBackward(
        q,
        kv,
        runtime_out,
        dout,
        runtime_lse,
        sink,
        runtime_indices,
        softmax_scale=scale,
        q_cluster_mode="adaptive_pair",
    )
    assert plan.check_support()
    plan.compile()
    assert plan._qcluster_plan is not None
    workspace = torch.empty(plan.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")
    dq, dkv, d_sink = torch.empty_like(q), torch.empty_like(kv), torch.empty_like(sink)

    def run():
        return plan.execute(
            q,
            kv,
            runtime_out,
            dout,
            runtime_lse,
            sink,
            runtime_indices,
            dq,
            dkv,
            softmax_scale=scale,
            workspace=workspace,
            d_sink=d_sink,
        )

    def assert_state(actual, expected, expected_shared):
        _assert_qcluster_dq_close(actual[0], baselines[expected]["dq"], f"{expected} dq")
        for name, tensor in zip(("dkv", "d_sink"), actual[1:]):
            torch.testing.assert_close(tensor, baselines[expected][name], rtol=5e-2, atol=5e-2, msg=f"{expected} {name}")
        shared_lengths = plan._qcluster_plan._buffers(workspace)[4]
        assert torch.equal(shared_lengths, torch.full_like(shared_lengths, expected_shared))

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            run_outputs = run()

        graph.replay()
        torch.cuda.synchronize()
        assert_state(run_outputs, "high", overlap)

        runtime_indices.copy_(zero_overlap)
        runtime_out.copy_(zero_out)
        runtime_lse.copy_(zero_lse)
        graph.replay()
        torch.cuda.synchronize()
        assert_state(run_outputs, "zero", 0)

        runtime_indices.copy_(high_overlap)
        runtime_out.copy_(high_out)
        runtime_lse.copy_(high_lse)
        graph.replay()
        torch.cuda.synchronize()
        assert_state(run_outputs, "high", overlap)
    finally:
        graph.reset()


@pytest.mark.L0
def test_DSA_sparse_attention_backward_deterministic_policy_is_independent():
    """Keep deterministic scheduling policy separate from ordinary tuning."""
    try:
        import cutlass

        from cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100 import FlashAttentionDSABackwardSm100
        from cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_deterministic import (
            FlashAttentionDSABackwardSm100Deterministic,
            FlashAttentionDSABackwardSm100Deterministic32,
            FlashAttentionDSABackwardSm100Deterministic64,
            FlashAttentionDSABackwardSm100DeterministicCluster16,
            FlashAttentionDSABackwardSm100DeterministicCluster32,
            FlashAttentionDSABackwardSm100DeterministicCluster64,
            FlashAttentionDSABackwardSm100DeterministicCluster4,
            FlashAttentionDSABackwardSm100DeterministicCluster8,
            _RELAY_WORKSPACE_BUDGET_BYTES,
            _deterministic_kernel_class_for_config,
            _relay_workspace_bytes,
            deterministic_dkv_relay_cluster_size,
            deterministic_dkv_shards,
            deterministic_kernel_class,
            use_cluster_dkv_relay,
        )
        from cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100_common import flash_attn_bwd_sm100_workspace_size
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    assert issubclass(FlashAttentionDSABackwardSm100Deterministic, FlashAttentionDSABackwardSm100)
    assert FlashAttentionDSABackwardSm100Deterministic.num_dkv_shards == 128
    assert FlashAttentionDSABackwardSm100Deterministic.q_wave_ctas == FlashAttentionDSABackwardSm100Deterministic.num_dkv_shards
    assert FlashAttentionDSABackwardSm100Deterministic.dkv_fold_group_size == 8
    assert FlashAttentionDSABackwardSm100Deterministic.serialize_head_blocks
    assert deterministic_dkv_shards(1) == 32
    assert deterministic_dkv_shards(128) == 32
    assert deterministic_dkv_shards(129) == 64
    assert deterministic_dkv_shards(256) == 64
    assert deterministic_dkv_shards(257) == 128
    assert deterministic_dkv_shards(1024) == 128
    assert deterministic_kernel_class(128) is FlashAttentionDSABackwardSm100Deterministic32
    assert deterministic_kernel_class(256) is FlashAttentionDSABackwardSm100Deterministic64
    assert deterministic_kernel_class(257) is FlashAttentionDSABackwardSm100Deterministic
    assert deterministic_kernel_class(16384, 278528, 576) is FlashAttentionDSABackwardSm100DeterministicCluster4
    assert FlashAttentionDSABackwardSm100DeterministicCluster4.num_dkv_shards == 32
    assert FlashAttentionDSABackwardSm100DeterministicCluster4.q_wave_ctas == 128
    assert FlashAttentionDSABackwardSm100DeterministicCluster4.dkv_cluster_shape == (4, 1, 1)
    assert FlashAttentionDSABackwardSm100DeterministicCluster4.dkv_mask_rows_per_word == 8
    assert FlashAttentionDSABackwardSm100DeterministicCluster8.num_dkv_shards == 16
    assert FlashAttentionDSABackwardSm100DeterministicCluster8.q_wave_ctas == 128
    assert FlashAttentionDSABackwardSm100DeterministicCluster8.dkv_cluster_shape == (8, 1, 1)
    assert FlashAttentionDSABackwardSm100DeterministicCluster8.dkv_mask_rows_per_word == 4
    assert FlashAttentionDSABackwardSm100DeterministicCluster16.num_dkv_shards == 8
    assert FlashAttentionDSABackwardSm100DeterministicCluster16.q_wave_ctas == 128
    assert FlashAttentionDSABackwardSm100DeterministicCluster16.dkv_cluster_shape == (16, 1, 1)
    assert FlashAttentionDSABackwardSm100DeterministicCluster16.dkv_mask_rows_per_word == 2
    assert FlashAttentionDSABackwardSm100DeterministicCluster32.num_dkv_shards == 4
    assert FlashAttentionDSABackwardSm100DeterministicCluster32.q_wave_ctas == 128
    assert FlashAttentionDSABackwardSm100DeterministicCluster32.dkv_cluster_shape == (16, 1, 1)
    assert FlashAttentionDSABackwardSm100DeterministicCluster32.dkv_relay_phases == 2
    assert FlashAttentionDSABackwardSm100DeterministicCluster32.dkv_mask_rows_per_word == 1
    assert FlashAttentionDSABackwardSm100DeterministicCluster64.num_dkv_shards == 2
    assert FlashAttentionDSABackwardSm100DeterministicCluster64.q_wave_ctas == 128
    assert FlashAttentionDSABackwardSm100DeterministicCluster64.dkv_cluster_shape == (16, 1, 1)
    assert FlashAttentionDSABackwardSm100DeterministicCluster64.dkv_relay_phases == 4
    assert FlashAttentionDSABackwardSm100DeterministicCluster64.dkv_mask_rows_per_word == 1
    assert FlashAttentionDSABackwardSm100DeterministicCluster64.dkv_mask_word_bits == 64
    assert use_cluster_dkv_relay(16384, 278528, 576)
    assert use_cluster_dkv_relay(16384, 131072, 576)
    assert not use_cluster_dkv_relay(128, 131072, 576)
    assert deterministic_dkv_relay_cluster_size(8191, 65537, 512) == 8
    assert deterministic_dkv_relay_cluster_size(8192, 65537, 512) == 4
    assert deterministic_dkv_relay_cluster_size(16384, 278528, 576) == 4
    assert deterministic_dkv_relay_cluster_size(16384, 524288, 576) == 16
    assert deterministic_dkv_relay_cluster_size(16384, 1048576, 576) == 16
    assert deterministic_dkv_relay_cluster_size(16384, 1048576, 512) == 16
    assert _relay_workspace_bytes(16384, 453432, 576, 64, 4) <= _RELAY_WORKSPACE_BUDGET_BYTES
    assert _relay_workspace_bytes(16384, 453433, 576, 64, 4) > _RELAY_WORKSPACE_BUDGET_BYTES
    assert deterministic_dkv_relay_cluster_size(16384, 453432, 576) == 4
    assert deterministic_dkv_relay_cluster_size(16384, 453433, 576) == 16
    for kernel_cls in (
        FlashAttentionDSABackwardSm100DeterministicCluster4,
        FlashAttentionDSABackwardSm100DeterministicCluster8,
        FlashAttentionDSABackwardSm100DeterministicCluster16,
        FlashAttentionDSABackwardSm100DeterministicCluster32,
        FlashAttentionDSABackwardSm100DeterministicCluster64,
    ):
        dkv_shape = kernel_cls._get_workspace_size_dKV_for_q(16384, 524288, 576, 64, 1, cutlass.Float32)
        assert math.prod(dkv_shape) == _relay_workspace_bytes(16384, 524288, 576, 64, kernel_cls.dkv_relay_cluster_size)
        assert not kernel_cls.run_dkv_prefold
        assert kernel_cls.num_dkv_final_terms == kernel_cls.num_dkv_shards
        assert kernel_cls.dkv_final_stride == 1
    assert deterministic_kernel_class(8191, 65537, 512) is FlashAttentionDSABackwardSm100DeterministicCluster8
    assert deterministic_kernel_class(8192, 65537, 512) is FlashAttentionDSABackwardSm100DeterministicCluster4
    assert deterministic_kernel_class(16384, 524288, 576) is FlashAttentionDSABackwardSm100DeterministicCluster16
    assert deterministic_kernel_class(16384, 1048576, 576) is FlashAttentionDSABackwardSm100DeterministicCluster16
    assert deterministic_kernel_class(16384, 524288, 576, deterministic_relay_cluster_size=4) is FlashAttentionDSABackwardSm100DeterministicCluster4
    assert deterministic_kernel_class(16384, 524288, 576, deterministic_relay_cluster_size=8) is FlashAttentionDSABackwardSm100DeterministicCluster8
    assert deterministic_kernel_class(16384, 524288, 576, deterministic_relay_cluster_size=16) is FlashAttentionDSABackwardSm100DeterministicCluster16
    assert deterministic_kernel_class(16384, 524288, 576, deterministic_relay_cluster_size=32) is FlashAttentionDSABackwardSm100DeterministicCluster32
    assert deterministic_kernel_class(16384, 524288, 576, deterministic_relay_cluster_size=64) is FlashAttentionDSABackwardSm100DeterministicCluster64
    forced_workspace = [
        flash_attn_bwd_sm100_workspace_size(
            16384,
            524288,
            576,
            64,
            deterministic=True,
            backend="generic_m64",
            deterministic_relay_cluster_size=cluster_size,
        )
        for cluster_size in (4, 8, 16, 32, 64)
    ]
    assert forced_workspace[0] > forced_workspace[1] > forced_workspace[2] > forced_workspace[3]
    with pytest.raises(ValueError, match="requires deterministic=True"):
        flash_attn_bwd_sm100_workspace_size(16384, 524288, 576, 64, deterministic_relay_cluster_size=4)
    with pytest.raises(ValueError, match="supported only by the deterministic generic_m64 backend"):
        flash_attn_bwd_sm100_workspace_size(16384, 524288, 576, 64, deterministic=True, deterministic_relay_cluster_size=4)
    assert _deterministic_kernel_class_for_config(16, 8) is FlashAttentionDSABackwardSm100DeterministicCluster8
    assert _deterministic_kernel_class_for_config(32, 4) is FlashAttentionDSABackwardSm100DeterministicCluster4
    assert _deterministic_kernel_class_for_config(8, 16) is FlashAttentionDSABackwardSm100DeterministicCluster16
    assert _deterministic_kernel_class_for_config(4, 32) is FlashAttentionDSABackwardSm100DeterministicCluster32
    assert _deterministic_kernel_class_for_config(2, 64) is FlashAttentionDSABackwardSm100DeterministicCluster64
    # The cap is inclusive: switch only when the selected ordinary layout is
    # strictly larger than 16 GiB. Exercise every adaptive shard boundary.
    for s_q, boundary_k in ((128, 262144), (129, 131072), (257, 65536)):
        assert not use_cluster_dkv_relay(s_q, boundary_k, 512)
        assert use_cluster_dkv_relay(s_q, boundary_k + 1, 512)
        assert deterministic_kernel_class(s_q, boundary_k, 512) is not FlashAttentionDSABackwardSm100DeterministicCluster8
        assert deterministic_kernel_class(s_q, boundary_k + 1, 512) is FlashAttentionDSABackwardSm100DeterministicCluster8
    assert not FlashAttentionDSABackwardSm100Deterministic32.run_dkv_prefold
    assert FlashAttentionDSABackwardSm100Deterministic32.num_dkv_fold_groups == 4
    assert FlashAttentionDSABackwardSm100Deterministic32.num_dkv_final_terms == 32
    assert FlashAttentionDSABackwardSm100Deterministic64.run_dkv_prefold
    assert FlashAttentionDSABackwardSm100Deterministic64.num_dkv_fold_groups == 8
    assert FlashAttentionDSABackwardSm100Deterministic64.num_dkv_final_terms == 8
    with pytest.raises(ValueError, match="at least one query row"):
        deterministic_dkv_shards(0)

    with pytest.raises(ValueError, match="at least one query row"):
        deterministic_dkv_relay_cluster_size(0, 256, 576)
    with pytest.raises(ValueError, match="positive KV, head-dimension, and head-count extents"):
        deterministic_dkv_relay_cluster_size(128, 0, 576)
    with pytest.raises(ValueError, match="inconsistent deterministic dKV relay configuration"):
        _deterministic_kernel_class_for_config(16, 4)
    with pytest.raises(ValueError, match="unsupported deterministic dKV relay cluster size"):
        _deterministic_kernel_class_for_config(32, 2)
    with pytest.raises(ValueError, match="positive KV and head extents"):
        use_cluster_dkv_relay(128, 0, 576)
    with pytest.raises(ValueError, match="positive KV and head extents"):
        use_cluster_dkv_relay(128, 256, 0)
    with pytest.raises(ValueError, match="must be provided together"):
        deterministic_kernel_class(128, total_seqlen_kv=256)
    with pytest.raises(ValueError, match="must be provided together"):
        deterministic_kernel_class(128, head_dim=576)
    with pytest.raises(ValueError, match="valid only when the deterministic large-context relay is selected"):
        deterministic_kernel_class(257, 65536, 512, deterministic_relay_cluster_size=4)
    for value in (True, 4.0, "4", 2, 128):
        with pytest.raises(ValueError, match=r"must be one of \{4, 8, 16, 32, 64\}"):
            deterministic_kernel_class(257, 65537, 512, deterministic_relay_cluster_size=value)
    assert FlashAttentionDSABackwardSm100.q_wave_ctas == 0
    assert not FlashAttentionDSABackwardSm100.serialize_head_blocks


@pytest.mark.L0
@pytest.mark.parametrize("helper_name", ["_wait_lower_rank_mask", "_wait_lower_rank_mask64"])
def test_DSA_sparse_attention_backward_relay_acquire_load_controls_exit(helper_name):
    """Keep the GPU acquire load live by feeding its value to loop control."""
    deterministic = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_deterministic")
    tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(deterministic, helper_name))))

    acquire_values = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if not isinstance(node.value.func, ast.Attribute) or node.value.func.attr != "load":
            continue
        if not any(keyword.arg == "sem" and isinstance(keyword.value, ast.Constant) and keyword.value.value == "acquire" for keyword in node.value.keywords):
            continue
        acquire_values.update(target.id for target in node.targets if isinstance(target, ast.Name))

    controlled_values = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "while_generate" or not node.args:
            continue
        if isinstance(node.args[0], ast.List):
            controlled_values.update(value.id for value in node.args[0].elts if isinstance(value, ast.Name))

    assert acquire_values & controlled_values, "an unused acquire load is eliminated from generated PTX"


@pytest.mark.L0
@pytest.mark.parametrize("num_heads", [16, 32, 64, 96, 128])
def test_DSA_sparse_attention_backward_sm100_d576_workspace_uses_compact_rows(num_heads):
    """All head counts retain compact dKV rows, including deterministic shards."""
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100 import flash_attn_bwd_sm100_workspace_size
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    s_q, s_kv, head_dim = 7, 1025, 576
    padded_s_q = math.ceil(s_q / 8) * 8
    padded_s_kv = math.ceil(s_kv / 8) * 8
    lse_odo_bytes = num_heads * padded_s_q * 2 * torch.float32.itemsize
    expected = lse_odo_bytes + padded_s_kv * head_dim * torch.float32.itemsize
    assert flash_attn_bwd_sm100_workspace_size(s_q, s_kv, head_dim, num_heads) == expected

    deterministic_dkv_bytes = 32 * padded_s_kv * head_dim * torch.float32.itemsize
    assert (
        flash_attn_bwd_sm100_workspace_size(s_q, s_kv, head_dim, num_heads, deterministic=True, backend="generic_m64")
        == lse_odo_bytes + deterministic_dkv_bytes
    )


@pytest.mark.L0
def test_DSA_sparse_attention_backward_sm100_deterministic_workspace_respects_backend():
    """D576 two-CTA scratch exposes its compact 96-plane capacity."""
    interface = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100")

    s_q, s_kv = 7, 1025
    fixed_d512 = interface.flash_attn_bwd_sm100_workspace_size(s_q, s_kv, 512, 128, deterministic=True)
    two_cta_d512 = interface.flash_attn_bwd_sm100_workspace_size(s_q, s_kv, 512, 128, deterministic=True, backend="h128_2cta_m64")
    assert two_cta_d512 == fixed_d512

    padded_s_q = math.ceil(s_q / 8) * 8
    padded_s_kv = math.ceil(s_kv / 8) * 8
    for num_heads, backend in ((96, "h96_d576_2cta_m64"), (128, "h128_d576_2cta_m64")):
        stats_bytes = num_heads * padded_s_q * 2 * torch.float32.itemsize
        compact_dkv_bytes = 96 * padded_s_kv * 576 * torch.float32.itemsize
        compact = interface.flash_attn_bwd_sm100_workspace_size(s_q, s_kv, 576, num_heads, deterministic=True, backend=backend)
        fixed = interface.flash_attn_bwd_sm100_workspace_size(s_q, s_kv, 576, num_heads, deterministic=True)
        adaptive = interface.flash_attn_bwd_sm100_workspace_size(s_q, s_kv, 576, num_heads, deterministic=True, backend="generic_m64")
        assert compact == stats_bytes + compact_dkv_bytes
        assert adaptive < compact < fixed


@pytest.mark.L0
@pytest.mark.parametrize(
    "s_q,s_kv,num_heads",
    [
        pytest.param(257, 65537, 64, id="partial-wave-one-head-block"),
        pytest.param(385, 65543, 96, id="partial-wave-two-head-blocks"),
        pytest.param(8191, 65537, 64, id="cluster8-policy-boundary"),
        pytest.param(8192, 65537, 64, id="cluster4-policy-boundary"),
        pytest.param(16384, 278528, 64, id="one-million-cp64"),
        pytest.param(16384, 1048576, 64, id="memory-first-cluster16"),
    ],
)
def test_DSA_sparse_attention_backward_sm100_large_context_workspace_uses_cluster_masks(s_q, s_kv, num_heads):
    """Large generic plans budget the selected relay planes and row masks."""
    interface = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100")
    deterministic = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_deterministic")

    head_dim = 576
    kernel_cls = deterministic.deterministic_kernel_class(s_q, s_kv, head_dim, num_heads)
    padded_q = math.ceil(s_q / 8) * 8
    padded_kv = math.ceil(s_kv / 8) * 8
    num_waves = math.ceil(s_q / kernel_cls.q_wave_ctas)
    num_head_blocks = math.ceil(num_heads / 64)
    mask_rows_per_word = 32 // kernel_cls.dkv_relay_cluster_size
    mask_words_per_plane = num_waves * num_head_blocks * (padded_kv // mask_rows_per_word)
    mask_rows_per_plane = math.ceil(mask_words_per_plane / head_dim)
    lse_odo_bytes = num_heads * padded_q * 2 * torch.float32.itemsize
    dkv_bytes = kernel_cls.num_dkv_shards * (padded_kv + mask_rows_per_plane) * head_dim * torch.float32.itemsize
    assert interface.flash_attn_bwd_sm100_workspace_size(s_q, s_kv, head_dim, num_heads, deterministic=True, backend="generic_m64") == lse_odo_bytes + dkv_bytes


def _exercise_deterministic_sm100_case(
    num_heads,
    head_dim,
    s_q,
    s_kv,
    repeats,
    check_short_workspace=False,
    topk=None,
    shared_topk=False,
    strided_topk=False,
    sorted_topk=False,
    check_graph=False,
    has_topk_length=True,
    expected_relay_cluster_size=None,
    deterministic_relay_cluster_size=None,
    dtype=torch.bfloat16,
):
    """Run one deterministic case against bitwise and numerical contracts."""
    import cutlass
    from cudnn import DSA
    from cuda.bindings import driver as cuda
    from cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100 import (
        _get_sm100_kernel_class,
        _select_sm100_backend,
        flash_attn_bwd_sm100_workspace_size,
    )

    _require_sm100()
    device = torch.device("cuda")
    topk = min(64 if topk is None else topk, s_kv)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(s_q, num_heads, head_dim, dtype=dtype, device=device) / 10
    kv = torch.randn(s_kv, head_dim, dtype=dtype, device=device) / 10
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    assert not (shared_topk and strided_topk)
    if shared_topk:
        topk_idxs = torch.randperm(s_kv, device=device)[:topk].repeat(s_q, 1).to(torch.int32)
    elif strided_topk:
        # Adjacent relay ranks overlap half their rows while waves spread
        # traffic across the full KV extent. Every row remains duplicate-free.
        slots = torch.arange(topk, device=device, dtype=torch.int64)
        row_offsets = torch.arange(s_q, device=device, dtype=torch.int64) * max(topk // 2, 1)
        topk_idxs = ((row_offsets[:, None] + slots[None, :]) % s_kv).to(torch.int32)
    else:
        topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    if sorted_topk:
        topk_idxs = topk_idxs.sort(dim=-1).values
    topk_length = torch.randint(1, topk + 1, (s_q,), dtype=torch.int32, device=device) if has_topk_length else None
    if topk_length is not None and s_q > 1:
        topk_length[s_q // 2] = 0
    out, lse = ref_sparse_attention_forward_chunked(
        q,
        kv,
        attn_sink,
        topk_idxs,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
    )
    dout = torch.randn_like(out)
    backend, _ = _select_sm100_backend(
        num_heads,
        head_dim,
        head_dim_v=512,
        dtype=dtype,
        max_topk=topk,
        device_capability=torch.cuda.get_device_capability(),
        deterministic=True,
    )
    workspace_bytes = flash_attn_bwd_sm100_workspace_size(
        s_q,
        s_kv,
        head_dim,
        num_heads,
        deterministic=True,
        backend=backend,
        deterministic_relay_cluster_size=deterministic_relay_cluster_size,
    )
    api = DSA.SparseAttentionBackward(
        sample_q=q,
        sample_kv=kv,
        sample_out=out,
        sample_dout=dout,
        sample_lse=lse,
        sample_attn_sink=attn_sink,
        sample_topk_idxs=topk_idxs,
        sample_topk_length=topk_length,
        softmax_scale=softmax_scale,
        deterministic=True,
        deterministic_relay_cluster_size=deterministic_relay_cluster_size,
    )
    assert api.check_support()
    api.compile()
    assert api.scratch_workspace_bytes() == workspace_bytes
    expected_lse_odo_bytes = num_heads * math.ceil(s_q / 8) * 8 * 2 * torch.float32.itemsize
    kernel_cls = _get_sm100_kernel_class(
        backend,
        True,
        s_q,
        s_kv,
        head_dim,
        num_heads,
        deterministic_relay_cluster_size,
    )
    if expected_relay_cluster_size is not None:
        assert kernel_cls.dkv_relay_cluster_size == expected_relay_cluster_size
    if hasattr(kernel_cls, "_get_workspace_size_dKV_for_q"):
        dkv_shape = kernel_cls._get_workspace_size_dKV_for_q(s_q, s_kv, head_dim, num_heads, 1, cutlass.Float32)
    else:
        dkv_shape = kernel_cls._get_workspace_size_dKV(s_kv, head_dim, 1, cutlass.Float32)
    expected_dkv_bytes = math.prod(dkv_shape)
    assert workspace_bytes == expected_lse_odo_bytes + expected_dkv_bytes
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
    dq, dkv, d_sink = torch.empty_like(q), torch.empty_like(kv), torch.empty_like(attn_sink)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def run():
        """Execute after poisoning caller scratch AND outputs: the launch sequence owns every reset."""
        # The compiled deterministic kernels, never an execute-side reset,
        # must establish accumulator state and overwrite every output.
        for buffer in (workspace, dq, dkv, d_sink):
            buffer.view(torch.uint8).fill_(0xA5)
        result = api.execute(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            dq,
            dkv,
            topk_length=topk_length,
            softmax_scale=softmax_scale,
            workspace=workspace,
            current_stream=stream,
            d_sink=d_sink,
        )
        torch.cuda.synchronize()
        assert result[0] is dq and result[1] is dkv and result[2] is d_sink
        return tuple(tensor.clone() for tensor in result)

    reference = run()
    if check_short_workspace:
        with pytest.raises(ValueError, match=rf"requires a {workspace_bytes}-byte workspace"):
            DSA.sparse_attention_backward_wrapper(
                q,
                kv,
                out,
                dout,
                lse,
                attn_sink,
                topk_idxs,
                softmax_scale=softmax_scale,
                topk_length=topk_length,
                deterministic=True,
                deterministic_relay_cluster_size=deterministic_relay_cluster_size,
                workspace=torch.empty(workspace_bytes - 1, dtype=torch.uint8, device=device),
            )

    output_names = ("dQ", "dKV", "dSink")
    for repeat in range(1, repeats + 1):
        actual = run()
        for name, actual_tensor, reference_tensor in zip(output_names, actual, reference):
            assert torch.equal(bitwise_bits(actual_tensor), bitwise_bits(reference_tensor)), f"{name} differs from the first run at repetition {repeat}"
    if check_graph:
        graph_stream = torch.cuda.Stream()
        graph_stream.wait_stream(torch.cuda.current_stream())
        graph_stream_handle = cuda.CUstream(graph_stream.cuda_stream)

        def graph_execute():
            return api.execute(
                q,
                kv,
                out,
                dout,
                lse,
                attn_sink,
                topk_idxs,
                dq,
                dkv,
                topk_length=topk_length,
                softmax_scale=softmax_scale,
                workspace=workspace,
                current_stream=graph_stream_handle,
                d_sink=d_sink,
            )

        with torch.cuda.stream(graph_stream):
            graph_execute()
        torch.cuda.current_stream().wait_stream(graph_stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=graph_stream):
                graph_execute()
            for repeat in range(3):
                with torch.cuda.stream(graph_stream):
                    for buffer in (workspace, dq, dkv, d_sink):
                        buffer.view(torch.uint8).fill_(0xA5)
                    graph.replay()
                torch.cuda.synchronize()
                for name, actual_tensor, reference_tensor in zip(output_names, (dq, dkv, d_sink), reference):
                    assert torch.equal(bitwise_bits(actual_tensor), bitwise_bits(reference_tensor)), f"{name} graph replay differs at repetition {repeat}"

            # Captured setup must consume runtime lengths rather than baking
            # the warm-up mask into the graph. Empty every sparse row, obtain
            # an eager reference, and replay the same captured graph.
            if topk_length is not None:
                saved_topk_length = topk_length.clone()
                topk_length.zero_()
                empty_reference = run()
                graph_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(graph_stream):
                    for buffer in (workspace, dq, dkv, d_sink):
                        buffer.view(torch.uint8).fill_(0xA5)
                    graph.replay()
                torch.cuda.synchronize()
                for name, actual_tensor, reference_tensor in zip(output_names, (dq, dkv, d_sink), empty_reference):
                    assert torch.equal(bitwise_bits(actual_tensor), bitwise_bits(reference_tensor)), f"{name} graph replay ignores runtime topk_length"
                topk_length.copy_(saved_topk_length)
        finally:
            graph.reset()

    check_ref_dsa_sparse_attention_backward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        out,
        dout,
        lse,
        *reference,
        softmax_scale=softmax_scale,
        topk_length=topk_length,
        atol=5e-2,
        rtol=5e-2,
    )


@torch_fork_set_rng(seed=419)
@pytest.mark.parametrize(
    "num_heads,head_dim,repeats",
    [
        pytest.param(16, 576, 3, marks=pytest.mark.L0, id="H16-D576-tail-smoke"),
        pytest.param(16, 576, 1000, marks=pytest.mark.L2, id="H16-D576-repeat1000"),
        pytest.param(32, 512, 1000, marks=pytest.mark.L2, id="H32-D512-repeat1000"),
        pytest.param(64, 512, 1000, marks=pytest.mark.L2, id="H64-D512-repeat1000"),
        pytest.param(64, 576, 1000, marks=pytest.mark.L2, id="H64-D576-repeat1000"),
        pytest.param(96, 512, 1000, marks=pytest.mark.L2, id="H96-D512-repeat1000"),
        pytest.param(96, 576, 1000, marks=pytest.mark.L2, id="H96-D576-repeat1000"),
        pytest.param(128, 576, 1000, marks=pytest.mark.L2, id="H128-D576-repeat1000"),
    ],
)
def test_DSA_sparse_attention_backward_sm100_deterministic_bounded_waves(num_heads, head_dim, repeats):
    """Masked and multi-block heads must be bitwise reproducible."""
    try:
        _exercise_deterministic_sm100_case(num_heads, head_dim, 257, 256, repeats)
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")


@pytest.mark.L2
@torch_fork_set_rng(seed=20261019)
@pytest.mark.parametrize("num_heads", [96, 128], ids=["H96", "H128"])
def test_DSA_sparse_attention_backward_sm100_d576_two_cta_repeat1000(num_heads):
    """The deterministic D576 two-CTA routes remain bitwise stable across persistent waves."""
    try:
        _exercise_deterministic_sm100_case(num_heads, 576, 257, 256, repeats=1000, topk=128)
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")


@pytest.mark.L2
def test_DSA_sparse_attention_backward_sm100_h96_d576_uses_i64_query_offsets():
    """The first Q extent past INT32 element addressing stays finite and exact."""
    _require_sm100()
    if torch.cuda.get_device_capability() not in _TWO_CTA_CAPABILITIES:
        pytest.skip("H96/D576 two-CTA requires compute capability 10.0 or 10.3")
    try:
        from cuda.bindings import driver as cuda
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
        return

    num_heads, head_dim, head_dim_v, topk = 96, 576, 512, 2048
    i32_max = torch.iinfo(torch.int32).max
    s_q = i32_max // (num_heads * head_dim) + 1
    assert (s_q - 1) * num_heads * head_dim <= i32_max < s_q * num_heads * head_dim
    device = torch.device("cuda")
    try:
        q = torch.zeros((s_q, num_heads, head_dim), dtype=torch.bfloat16, device=device)
        kv = torch.zeros((topk, head_dim), dtype=torch.bfloat16, device=device)
        out = torch.zeros((s_q, num_heads, head_dim_v), dtype=torch.bfloat16, device=device)
        dout = torch.full_like(out, 1.0 / s_q)
        q[-1, :, -1] = 2.0
        kv[0, 0] = 1.0
        out[:, :, 0] = 0.5
        # DSA forward LSE covers sparse keys only; backward folds in the sink.
        lse = torch.zeros((s_q, num_heads), dtype=torch.float32, device=device)
        attn_sink = torch.zeros((num_heads,), dtype=torch.float32, device=device)
        topk_idxs = torch.zeros((s_q, topk), dtype=torch.int32, device=device)
        topk_length = torch.ones((s_q,), dtype=torch.int32, device=device)
    except torch.OutOfMemoryError:
        pytest.skip("requires enough GPU memory to cross the INT32 query-offset boundary")
        return
    scale = head_dim**-0.5

    plan = DSA.SparseAttentionBackward(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        sample_topk_length=topk_length,
        softmax_scale=scale,
        deterministic=True,
    )
    assert plan.check_support()
    assert plan._backend == "h96_d576_2cta_m64"
    assert plan._wide_addressing
    plan.compile()
    try:
        workspace = torch.empty(plan.scratch_workspace_bytes(), dtype=torch.uint8, device=device)
        dq, dkv, d_sink = torch.empty_like(q), torch.empty_like(kv), torch.empty_like(attn_sink)
    except torch.OutOfMemoryError:
        pytest.skip("requires enough GPU memory to cross the INT32 query-offset boundary")
        return
    dq.fill_(math.nan)
    dkv.fill_(math.nan)
    d_sink.fill_(math.nan)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    actual = plan.execute(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        dq,
        dkv,
        topk_length=topk_length,
        softmax_scale=scale,
        current_stream=stream,
        workspace=workspace,
        d_sink=d_sink,
    )
    assert actual == (dq, dkv, d_sink)
    dout_value = float(dout[0, 0, 0])
    expected_ds = 0.25 * dout_value
    expected_dq = torch.tensor(expected_ds * scale, dtype=torch.bfloat16, device=device)
    assert torch.count_nonzero(dq[:, :, 0] != expected_dq).item() == 0
    assert torch.count_nonzero(dq[:, :, 1:]).item() == 0
    expected_dv = torch.tensor(0.5 * dout_value * s_q * num_heads, dtype=torch.bfloat16, device=device)
    assert torch.count_nonzero(dkv[0, :head_dim_v] != expected_dv).item() == 0
    assert torch.count_nonzero(dkv[0, head_dim_v:-1]).item() == 0
    expected_tail_dk = torch.tensor(expected_ds * 2.0 * scale * num_heads, dtype=torch.bfloat16, device=device)
    assert dkv[0, -1] == expected_tail_dk
    assert torch.count_nonzero(dkv[1:]).item() == 0
    expected_d_sink = torch.full_like(d_sink, -0.25 * dout_value * s_q)
    torch.testing.assert_close(d_sink, expected_d_sink, rtol=2e-5, atol=2e-5)


@pytest.mark.L1
@torch_fork_set_rng(seed=20261021)
def test_DSA_sparse_attention_backward_sm100_h96_d576_topk2048_repeat():
    """Top-K 2048 sorted rows stay correct and bitwise stable with poisoned scratch reuse."""
    try:
        _exercise_deterministic_sm100_case(
            96,
            576,
            9,
            2065,
            repeats=2,
            topk=2048,
            sorted_topk=True,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("topk", [2048, 2049, 2050, 2051])
@pytest.mark.parametrize("deterministic", [False, True], ids=["ordinary", "deterministic"])
@torch_fork_set_rng(seed=20261029)
def test_DSA_sparse_attention_backward_sm100_h96_d576_unaligned_topk_widths(topk, deterministic):
    """Top-K 2048--2051 remains on the native H96 route with correct gradients."""
    _require_sm100()
    if torch.cuda.get_device_capability() not in _TWO_CTA_CAPABILITIES:
        pytest.skip("H96/D576 two-CTA requires compute capability 10.0 or 10.3")
    try:
        from cuda.bindings import driver as cuda
        from cudnn import DSA
    except ImportError:
        return pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    s_q, s_kv = 4, topk + 17
    num_heads, head_dim = 96, 576
    softmax_scale = 1.0 / math.sqrt(head_dim)
    device = torch.device("cuda")
    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device) / 10
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device) / 10
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    out, lse = ref_sparse_attention_forward_chunked(q, kv, attn_sink, topk_idxs, softmax_scale=softmax_scale)
    dout = torch.randn_like(out)

    plan = DSA.SparseAttentionBackward(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale=softmax_scale,
        deterministic=deterministic,
    )
    assert plan.check_support()
    assert plan._backend == "h96_d576_2cta_m64"
    plan.compile()
    workspace = torch.empty(plan.scratch_workspace_bytes(), dtype=torch.uint8, device=device)
    dq, dkv, d_sink = torch.empty_like(q), torch.empty_like(kv), torch.empty_like(attn_sink)
    result = plan.execute(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        dq,
        dkv,
        softmax_scale=softmax_scale,
        workspace=workspace,
        current_stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        d_sink=d_sink,
    )
    assert all(actual is expected for actual, expected in zip(result, (dq, dkv, d_sink), strict=True))
    if deterministic:
        first = tuple(tensor.clone() for tensor in result)
        for buffer in (workspace, dq, dkv, d_sink):
            buffer.view(torch.uint8).fill_(0xA5)
        plan.execute(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            dq,
            dkv,
            softmax_scale=softmax_scale,
            workspace=workspace,
            current_stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            d_sink=d_sink,
        )
        for actual, expected in zip((dq, dkv, d_sink), first):
            assert torch.equal(bitwise_bits(actual), bitwise_bits(expected))
    check_ref_dsa_sparse_attention_backward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        out,
        dout,
        lse,
        dq,
        dkv,
        d_sink,
        softmax_scale=softmax_scale,
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.L1
@torch_fork_set_rng(seed=420)
@pytest.mark.parametrize("s_q", [128, 129], ids=["shards32", "shards64"])
def test_DSA_sparse_attention_backward_sm100_deterministic_adaptive_shards_repeat100(s_q):
    """Every adaptive specialization remains bitwise stable across reuse."""
    try:
        _exercise_deterministic_sm100_case(64, 512, s_q, 130, repeats=100)
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")


@pytest.mark.L2
@torch_fork_set_rng(seed=421)
def test_DSA_sparse_attention_backward_sm100_cluster8_repeat100():
    """Partial waves stay exact under high row overlap and graph replay."""
    try:
        _exercise_deterministic_sm100_case(
            64,
            576,
            257,
            65537,
            repeats=100,
            topk=511,
            shared_topk=True,
            check_graph=True,
            check_short_workspace=True,
            expected_relay_cluster_size=8,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")


@pytest.mark.L2
@torch_fork_set_rng(seed=20261030)
def test_DSA_sparse_attention_backward_sm100_cluster8_topk2048():
    """The optimized production-width relay remains exact and repeatable."""
    try:
        _exercise_deterministic_sm100_case(
            64,
            576,
            257,
            65537,
            repeats=3,
            topk=2048,
            has_topk_length=True,
            check_graph=True,
            expected_relay_cluster_size=8,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")


@pytest.mark.L2
@torch_fork_set_rng(seed=423)
@pytest.mark.parametrize(
    "shared_topk,has_topk_length",
    [
        pytest.param(False, True, id="disjoint-variable-lengths"),
        pytest.param(True, False, id="overlap-full-topk"),
    ],
)
def test_DSA_sparse_attention_backward_sm100_cluster8_two_head_blocks(shared_topk, has_topk_length):
    """Two serialized head blocks cover both sparse and contended relay masks."""
    try:
        _exercise_deterministic_sm100_case(
            96,
            512,
            257,
            65537,
            repeats=3,
            topk=129,
            shared_topk=shared_topk,
            has_topk_length=has_topk_length,
            expected_relay_cluster_size=8,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")


@pytest.mark.L2
@torch_fork_set_rng(seed=424)
def test_DSA_sparse_attention_backward_sm100_cluster8_d512():
    """The D512 reduction path obeys the same row-relay ordering contract."""
    try:
        _exercise_deterministic_sm100_case(
            64,
            512,
            257,
            65537,
            repeats=3,
            topk=129,
            shared_topk=True,
            expected_relay_cluster_size=8,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")


@pytest.mark.L2
@torch_fork_set_rng(seed=20261031)
@pytest.mark.parametrize("relay_cluster_size", [4, 8, 16, 32, 64], ids=["cluster4", "cluster8", "cluster16", "cluster32", "cluster64"])
def test_DSA_sparse_attention_backward_sm100_explicit_relay_cluster_size(relay_cluster_size):
    """Every caller-selected relay remains exact, repeatable, and graph-safe."""
    try:
        _exercise_deterministic_sm100_case(
            64,
            512,
            257,
            65537,
            repeats=3,
            topk=129,
            shared_topk=True,
            check_graph=True,
            check_short_workspace=True,
            expected_relay_cluster_size=relay_cluster_size,
            deterministic_relay_cluster_size=relay_cluster_size,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")


@pytest.mark.L2
@torch_fork_set_rng(seed=20261101)
def test_DSA_sparse_attention_backward_sm100_cluster32_repeat100():
    """Two physical relay phases remain exact under sustained row overlap."""
    try:
        _exercise_deterministic_sm100_case(
            64,
            512,
            257,
            65537,
            repeats=100,
            topk=511,
            shared_topk=True,
            check_graph=True,
            check_short_workspace=True,
            expected_relay_cluster_size=32,
            deterministic_relay_cluster_size=32,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")


@pytest.mark.L2
@torch_fork_set_rng(seed=20261102)
def test_DSA_sparse_attention_backward_sm100_cluster64_repeat100():
    """Four physical relay phases remain exact under sustained row overlap."""
    try:
        _exercise_deterministic_sm100_case(
            64,
            512,
            257,
            65537,
            repeats=100,
            topk=511,
            shared_topk=True,
            check_graph=True,
            check_short_workspace=True,
            expected_relay_cluster_size=64,
            deterministic_relay_cluster_size=64,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")


@pytest.mark.L2
@torch_fork_set_rng(seed=425)
def test_DSA_sparse_attention_backward_sm100_cluster4_repeat100():
    """Long-query four-CTA relays remain exact under graph replay."""
    try:
        _exercise_deterministic_sm100_case(
            64,
            512,
            8192,
            65537,
            repeats=100,
            topk=64,
            strided_topk=True,
            check_graph=True,
            check_short_workspace=True,
            expected_relay_cluster_size=4,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")


@pytest.mark.L2
@torch_fork_set_rng(seed=427)
def test_DSA_sparse_attention_backward_sm100_cluster16_repeat100():
    """Memory-first sixteen-CTA relays remain exact under graph replay."""
    try:
        _exercise_deterministic_sm100_case(
            64,
            512,
            8192,
            1048576,
            repeats=100,
            topk=64,
            strided_topk=True,
            check_graph=True,
            check_short_workspace=True,
            expected_relay_cluster_size=16,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")


@pytest.mark.L2
@torch_fork_set_rng(seed=428)
@pytest.mark.parametrize(
    "num_heads,head_dim,s_kv,topk,shared_topk,has_topk_length",
    [
        pytest.param(64, 576, 524288, 129, False, True, id="d576-tail-variable-lengths"),
        pytest.param(96, 512, 524288, 64, True, False, id="two-head-blocks-high-overlap"),
    ],
)
def test_DSA_sparse_attention_backward_sm100_cluster16_boundaries(num_heads, head_dim, s_kv, topk, shared_topk, has_topk_length):
    """Memory-first relays cover the D576 tail and serialized head blocks."""
    try:
        _exercise_deterministic_sm100_case(
            num_heads,
            head_dim,
            8192,
            s_kv,
            repeats=3,
            topk=topk,
            shared_topk=shared_topk,
            strided_topk=not shared_topk,
            has_topk_length=has_topk_length,
            expected_relay_cluster_size=16,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")


@pytest.mark.L2
@torch_fork_set_rng(seed=426)
def test_DSA_sparse_attention_backward_sm100_cluster4_fp16():
    """The four-CTA relay remains correct for the supported FP16 path."""
    try:
        _exercise_deterministic_sm100_case(
            64,
            512,
            8192,
            65537,
            repeats=3,
            topk=64,
            strided_topk=True,
            expected_relay_cluster_size=4,
            dtype=torch.float16,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")


@pytest.mark.L1
@torch_fork_set_rng(seed=422)
@pytest.mark.parametrize(
    "num_heads,head_dim,topk,dtype,deterministic,expected_backend",
    [
        pytest.param(16, 576, 64, torch.bfloat16, False, "h16_m128", id="h16"),
        pytest.param(32, 576, 64, torch.bfloat16, False, "h32_m64", id="h32"),
        pytest.param(64, 512, 64, torch.bfloat16, False, "generic_m64", id="generic"),
        pytest.param(96, 576, 64, torch.bfloat16, False, "h96_h64_h32", id="h96"),
        pytest.param(128, 512, 128, torch.bfloat16, False, "h128_2cta_m64", id="h128-two-cta"),
        pytest.param(64, 512, 64, torch.bfloat16, True, "generic_m64", id="deterministic-generic-bf16"),
        pytest.param(64, 576, 129, torch.bfloat16, True, "generic_m64", id="deterministic-generic-d576-bf16"),
        pytest.param(64, 512, 64, torch.float16, True, "generic_m64", id="deterministic-generic-fp16"),
        pytest.param(96, 576, 128, torch.bfloat16, True, "h96_d576_2cta_m64", id="deterministic-h96-d576-two-cta"),
        pytest.param(128, 512, 128, torch.bfloat16, True, "h128_2cta_m64", id="deterministic-h128-two-cta"),
    ],
)
def test_DSA_sparse_attention_backward_sm100_plan_is_cold_capture_safe(
    monkeypatch,
    num_heads,
    head_dim,
    topk,
    dtype,
    deterministic,
    expected_backend,
):
    """Every selected SM100 plan compiles before its first captured execute."""
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm100 as interface
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm100()
    if expected_backend == "h128_2cta_m64":
        _require_exact_sm100()
    plan, inputs, outputs, topk_length, softmax_scale, workspace = _plan_case(
        num_heads,
        head_dim,
        topk,
        s_q=8,
        s_kv=max(256, 2 * topk),
        dtype=dtype,
        has_topk_length=True,
        deterministic=deterministic,
    )
    assert plan._backend == expected_backend
    assert callable(plan._compiled_kernel)

    def forbidden_compile(*args, **kwargs):
        """Fail if the first post-plan execute attempts CuTe compilation."""
        raise AssertionError("execute() attempted CuTe compilation")

    monkeypatch.setattr(interface.cute, "compile", forbidden_compile)
    dq, dkv, d_sink = outputs

    def run():
        """Launch the prepared plan into its caller-owned outputs and scratch."""
        return plan.execute(
            *inputs,
            dq,
            dkv,
            topk_length=topk_length,
            softmax_scale=softmax_scale,
            workspace=workspace,
            d_sink=d_sink,
        )

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = run()
        graph.replay()
        torch.cuda.synchronize()
        reference = tuple(tensor.clone() for tensor in captured)
        graph.replay()
        torch.cuda.synchronize()
        for actual, expected in zip(captured, reference):
            if deterministic:
                assert torch.equal(actual, expected)
            else:
                torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)
        q, kv, out, dout, lse, attn_sink, topk_idxs = inputs
        check_ref_dsa_sparse_attention_backward(
            q,
            kv,
            attn_sink,
            topk_idxs,
            out,
            dout,
            lse,
            *captured,
            softmax_scale=softmax_scale,
            topk_length=topk_length,
            atol=5e-2,
            rtol=5e-2,
        )
    finally:
        graph.reset()


@pytest.mark.L0
@torch_fork_set_rng(seed=426)
def test_DSA_sparse_attention_backward_sm100_execute_requires_compile(monkeypatch):
    """An SM100 plan validates support but never compiles implicitly in execute()."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm100()
    plan, inputs, outputs, topk_length, softmax_scale, workspace = _plan_case(
        64,
        512,
        64,
        s_q=8,
        s_kv=256,
        has_topk_length=True,
        compile_plan=False,
    )
    assert plan._compiled_kernel is None
    dq, dkv, d_sink = outputs
    execute_args = dict(
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        workspace=workspace,
        d_sink=d_sink,
    )
    with pytest.raises(RuntimeError, match=r"call compile\(\) before execute\(\)"):
        plan.execute(*inputs, dq, dkv, **execute_args)

    q, kv, out, dout, lse, attn_sink, topk_idxs = inputs
    unchecked_plan = DSA.SparseAttentionBackward(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        sample_topk_length=topk_length,
        softmax_scale=softmax_scale,
    )
    assert unchecked_plan._device_major is None
    with pytest.raises(RuntimeError, match=r"call compile\(\) before execute\(\)"):
        unchecked_plan.execute(*inputs, dq, dkv, **execute_args)

    unchecked_plan._is_supported = False

    def rejected_support():
        """Prove execute revalidates a stale support state before dispatch."""
        raise NotImplementedError("support state must be revalidated")

    monkeypatch.setattr(unchecked_plan, "check_support", rejected_support)
    with pytest.raises(NotImplementedError, match="support state must be revalidated"):
        unchecked_plan.execute(*inputs, dq, dkv, **execute_args)


@pytest.mark.L1
@torch_fork_set_rng(seed=428)
def test_DSA_sparse_attention_backward_sm100_plan_revalidates_runtime_layout():
    """Prepared plans reject runtime layouts that the low-level fast path skips."""
    _require_sm100()
    plan, inputs, outputs, topk_length, softmax_scale, workspace = _plan_case(
        64,
        512,
        64,
        s_q=8,
        s_kv=256,
        has_topk_length=True,
    )
    dq, dkv, d_sink = outputs
    execute_args = dict(
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        workspace=workspace,
        d_sink=d_sink,
    )

    q = inputs[0]
    q_storage = torch.empty((*q.shape[:-1], 2 * q.shape[-1]), dtype=q.dtype, device=q.device)
    strided_q = q_storage[..., ::2]
    assert strided_q.shape == q.shape and not strided_q.is_contiguous()
    with pytest.raises(ValueError, match="q must be contiguous"):
        plan.execute(strided_q, *inputs[1:], dq, dkv, **execute_args)

    lse = inputs[4]
    lse_storage = torch.empty(lse.numel() + 1, dtype=lse.dtype, device=lse.device)
    misaligned_lse = lse_storage[1:].view_as(lse)
    assert misaligned_lse.is_contiguous() and misaligned_lse.data_ptr() % 8 == 4
    misaligned_inputs = (*inputs[:4], misaligned_lse, *inputs[5:])
    with pytest.raises(ValueError, match="lse must be 8-byte aligned"):
        plan.execute(*misaligned_inputs, dq, dkv, **execute_args)


@pytest.mark.L1
@torch_fork_set_rng(seed=429)
def test_DSA_sparse_attention_backward_sm100_d576_deterministic_workspace_requires_32_byte_alignment():
    """Reject a 16-byte-only workspace before the Top-K 2048 kernel launches."""
    _require_sm100()
    plan, inputs, outputs, topk_length, softmax_scale, workspace = _plan_case(
        96,
        576,
        2048,
        s_q=1,
        s_kv=2048,
        has_topk_length=True,
        deterministic=True,
        compile_plan=False,
    )

    def unexpected_launch(*args, **kwargs):
        pytest.fail("the kernel launched before workspace alignment validation")

    plan._compiled_kernel = unexpected_launch
    workspace_bytes = workspace.numel()
    del workspace
    storage = torch.empty(workspace_bytes + 32, dtype=torch.uint8, device="cuda")
    offset = (16 - storage.data_ptr() % 32) % 32
    misaligned_workspace = storage[offset : offset + workspace_bytes]
    assert misaligned_workspace.is_contiguous()
    assert misaligned_workspace.data_ptr() % 16 == 0
    assert misaligned_workspace.data_ptr() % 32 == 16

    dq, dkv, d_sink = outputs
    with pytest.raises(ValueError, match="workspace must be 32-byte aligned"):
        plan.execute(
            *inputs,
            dq,
            dkv,
            topk_length=topk_length,
            softmax_scale=softmax_scale,
            workspace=misaligned_workspace,
            d_sink=d_sink,
        )


@pytest.mark.L0
@pytest.mark.parametrize("deterministic", [False, True], ids=["ordinary", "deterministic"])
def test_DSA_sparse_attention_backward_sm100_compile_cache_is_device_and_address_scoped(monkeypatch, deterministic):
    """Launchables are isolated by CUDA context and address-width specialization."""
    from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm100 as interface

    compile_artifact = interface._compile_deterministic_sm100 if deterministic else interface._compile_nondeterministic_sm100
    compiled = []

    def fake_compile(device_index, *args, **kwargs):
        """Return a unique launchable and record its device/address class."""
        compiled.append((device_index, kwargs["wide_addressing"]))
        return object()

    compile_artifact.cache_clear()
    monkeypatch.setattr(interface, "_compile_sm100_artifact", fake_compile)
    args = ((10, 0), "generic_m64", torch.bfloat16, 512, 512, 64, 64, True)
    args += (32, 1) if deterministic else (64,)
    try:
        device0 = compile_artifact(0, *args)
        assert compile_artifact(0, *args) is device0
        device0_wide = compile_artifact(0, *args, True)
        device1 = compile_artifact(1, *args)
    finally:
        compile_artifact.cache_clear()

    assert device1 is not device0
    assert device0_wide is not device0
    assert compiled == [(0, False), (0, True), (1, False)]


@pytest.mark.L0
def test_DSA_sparse_attention_backward_sm100_d576_compile_cache_separates_address_width(monkeypatch):
    """The shared H96/H128 D576 compiler never reuses compact code for a wide plan."""
    from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm100_d576 as interface

    compile_artifact = interface._compile_d576_2cta
    compiled = []

    def fake_compile(*args, **kwargs):
        compiled.append(object())
        return compiled[-1]

    compile_artifact.cache_clear()
    monkeypatch.setattr(interface.cute, "compile", fake_compile)
    monkeypatch.setattr(interface, "compile_options", lambda: "")
    args = ((10, 3), 2048, True, False, False, True, 96, 76)
    try:
        narrow = compile_artifact(*args)
        assert compile_artifact(*args) is narrow
        wide = compile_artifact(*args, True)
    finally:
        compile_artifact.cache_clear()

    assert wide is not narrow
    assert len(compiled) == 2


@pytest.mark.L0
def test_DSA_sparse_attention_backward_sm100_address_class_covers_data_and_workspace():
    """Choose Int64 only when a compact data/workspace offset can overflow."""
    from cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100_common import _requires_wide_data_addressing

    limit = 1 << 31
    heads, dim, dv = 96, 576, 512
    safe_q = limit // (heads * dim)
    common = (2048, dim, dv, heads, 2048, True, "h96_d576_2cta_m64")
    assert not _requires_wide_data_addressing(safe_q, *common, include_workspace=False)
    assert _requires_wide_data_addressing(safe_q + 1, *common, include_workspace=False)

    # Top-K indices can be the first tensor to cross the boundary for a small
    # head count, so their row span participates in the same address class.
    safe_topk_q = limit // 2048
    topk_limited = (1, 512, 512, 1, 2048, False, "generic_m64")
    assert not _requires_wide_data_addressing(safe_topk_q, *topk_limited, include_workspace=False)
    assert _requires_wide_data_addressing(safe_topk_q + 1, *topk_limited, include_workspace=False)

    # Generic deterministic H64 uses 32 byte-shaped dKV planes at this Q.
    # KV=32768 occupies exactly 2^31 bytes and remains addressable; one more
    # row requires the dynamic Int64 workspace descriptor.
    generic = (1, 512, 512, 64, 2048, True, "generic_m64")
    assert not _requires_wide_data_addressing(generic[0], 32768, *generic[1:], include_workspace=True)
    assert _requires_wide_data_addressing(generic[0], 32769, *generic[1:], include_workspace=True)


@pytest.mark.L0
def test_DSA_sparse_attention_backward_sm100_compile_targets_cover_supported_family():
    """Plan-time compilation preserves each supported SM100-family target."""
    from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm100 as interface

    assert interface._sm100_gpu_arch_flag((10, 0)) == "sm_100a"
    assert interface._sm100_gpu_arch_flag((10, 3)) == "sm_103a"
    assert interface._sm100_gpu_arch_flag((10, 7)) == "sm_107a"
    with pytest.raises(ValueError, match="does not support compute capability"):
        interface._sm100_gpu_arch_flag((10, 1))


@pytest.mark.L1
@torch_fork_set_rng(seed=423)
@pytest.mark.parametrize("deterministic", [False, True], ids=["ordinary", "deterministic"])
def test_DSA_sparse_attention_backward_sm100_artifact_is_shape_dynamic(monkeypatch, deterministic):
    """Compatible plans reuse one compiled artifact across Q and KV extents."""
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm100 as interface
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm100()
    compile_artifact = interface._compile_deterministic_sm100 if deterministic else interface._compile_nondeterministic_sm100
    compile_artifact.cache_clear()
    first, *_ = _plan_case(64, 512, 64, s_q=8, s_kv=256, has_topk_length=True, deterministic=deterministic)
    first_info = compile_artifact.cache_info()
    assert not first._wide_addressing

    def forbidden_compile(*args, **kwargs):
        """Fail when a dynamic-extent cache hit unexpectedly recompiles."""
        raise AssertionError("shape-dynamic plan unexpectedly recompiled")

    monkeypatch.setattr(interface.cute, "compile", forbidden_compile)
    second, inputs, outputs, topk_length, softmax_scale, workspace = _plan_case(
        64,
        512,
        64,
        s_q=16,
        s_kv=320,
        has_topk_length=True,
        deterministic=deterministic,
    )
    second_info = compile_artifact.cache_info()

    assert not second._wide_addressing
    assert second._compiled_kernel is first._compiled_kernel
    assert first_info.misses == second_info.misses == 1
    assert second_info.hits == first_info.hits + 1
    _assert_execute_allocates_nothing_and_never_synchronizes(
        second,
        inputs,
        outputs,
        topk_length,
        softmax_scale,
        workspace,
    )


@pytest.mark.L1
@torch_fork_set_rng(seed=427)
def test_DSA_sparse_attention_backward_sm100_direct_interface_stays_lazy(monkeypatch):
    """Direct low-level callers compile once on first use and reuse the lazy cache."""
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm100 as interface
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm100()
    _, inputs, outputs, topk_length, softmax_scale, workspace = _plan_case(
        64,
        512,
        64,
        s_q=8,
        s_kv=256,
        has_topk_length=True,
        compile_plan=False,
    )
    compile_cache = interface.flash_attn_bwd_sm100.compile_cache
    saved_cache = compile_cache.copy()
    compile_cache.clear()
    compile_calls = 0
    original_compile = interface.cute.compile

    def tracked_compile(*args, **kwargs):
        """Count actual CuTe compilation while forwarding to the compiler."""
        nonlocal compile_calls
        compile_calls += 1
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(interface.cute, "compile", tracked_compile)
    dq, dkv, d_sink = outputs
    try:
        for _ in range(2):
            result = interface.flash_attn_bwd_sm100(
                *inputs,
                softmax_scale=softmax_scale,
                topk_length=topk_length,
                dq=dq,
                dkv=dkv,
                d_sink=d_sink,
                workspace=workspace,
            )
            assert all(actual is expected for actual, expected in zip(result, outputs))
    finally:
        compile_cache.clear()
        compile_cache.update(saved_cache)

    assert compile_calls == 1


@pytest.mark.L1
@torch_fork_set_rng(seed=421)
@pytest.mark.parametrize(
    "num_heads,head_dim,s_q,s_kv",
    [
        pytest.param(64, 512, 1, 1, id="H64-q1-kv1"),
        pytest.param(16, 576, 8, 65, id="H16-q8-kv65"),
        pytest.param(32, 512, 9, 127, id="H32-q9-kv127"),
        pytest.param(64, 576, 127, 128, id="H64-q127-kv128"),
        pytest.param(96, 512, 128, 129, id="H96-q128-kv129"),
        pytest.param(96, 576, 129, 130, id="H96-D576-q129-kv130"),
        pytest.param(128, 576, 129, 130, id="H128-q129-kv130"),
    ],
)
def test_DSA_sparse_attention_backward_sm100_deterministic_boundaries(num_heads, head_dim, s_q, s_kv):
    """Exercise head tails, vector padding, wave boundaries, and exact scratch."""
    try:
        max_topk = 128 if (num_heads, head_dim) == (96, 576) else None
        _exercise_deterministic_sm100_case(num_heads, head_dim, s_q, s_kv, repeats=1, check_short_workspace=True, topk=max_topk)
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")


@pytest.mark.L0
def test_DSA_sparse_attention_backward_sm100_h128_two_cta_dispatch_is_fail_closed():
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100 import _select_sm100_backend
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    supported = {
        "head_dim_v": 512,
        "dtype": torch.bfloat16,
        "max_topk": 512,
    }
    for device_capability in ((10, 0), (10, 3)):
        for max_topk in (128, 512, 1024, 1152, 2048):
            kwargs = {**supported, "device_capability": device_capability, "max_topk": max_topk}
            for deterministic in (False, True):
                assert _select_sm100_backend(128, 512, **kwargs, deterministic=deterministic) == ("h128_2cta_m64", 64)

    supported = {**supported, "device_capability": (10, 0)}

    assert _select_sm100_backend(128, 512) == ("generic_m64", 64)
    assert _select_sm100_backend(64, 512, **supported) == ("generic_m64", 64)
    # The BF16 H128/D576 envelope has its own two-CTA route (covered below).
    assert _select_sm100_backend(128, 576, **supported) == ("h128_d576_2cta_m64", 64)
    # Deterministic execution preserves the supported two-CTA envelope;
    # other head counts and unsupported layouts keep their existing fallback.
    assert _select_sm100_backend(128, 512, **supported, deterministic=True) == ("h128_2cta_m64", 64)
    assert _select_sm100_backend(128, 512, **supported, deterministic=True, is_contiguous=False) == ("generic_m64", 64)
    assert _select_sm100_backend(96, 576, **supported, deterministic=True) == ("h96_d576_2cta_m64", 64)
    assert _select_sm100_backend(96, 576, **supported, deterministic=False) == ("h96_d576_2cta_m64", 64)
    assert _select_sm100_backend(96, 576, **supported, deterministic=True, is_contiguous=False) == ("h96_h64_h32", 64)
    assert _select_sm100_backend(16, 576, deterministic=True) == ("generic_m64", 64)

    fallback_cases = [
        {**supported, "head_dim_v": 256},
        {**supported, "dtype": torch.float16},
        {**supported, "max_topk": 0},
        {**supported, "max_topk": 64},
        {**supported, "max_topk": 513},
        {**supported, "max_topk": 1151},
        {**supported, "max_topk": 1153},
        {**supported, "max_topk": 2112},
        {**supported, "device_capability": (10, 7)},
    ]
    for kwargs in fallback_cases:
        for deterministic in (False, True):
            assert _select_sm100_backend(128, 512, **kwargs, deterministic=deterministic) == ("generic_m64", 64)


@pytest.mark.L0
@pytest.mark.parametrize("capability", list(_TWO_CTA_CAPABILITIES))
@pytest.mark.parametrize("topk", [128, 512, 1024, 1152, 2048])
@pytest.mark.parametrize("deterministic", [False, True], ids=["ordinary", "deterministic"])
def test_DSA_sparse_attention_backward_d576_2cta_dispatch(topk, capability, deterministic):
    """Contiguous BF16 H128/D576 with a supported top-k selects the two-CTA backend on CC 10.0 and 10.3."""
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100 import _select_sm100_backend
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    assert _select_sm100_backend(128, 576, head_dim_v=512, dtype=torch.bfloat16, max_topk=topk, device_capability=capability, deterministic=deterministic) == (
        "h128_d576_2cta_m64",
        64,
    )


@pytest.mark.L0
@pytest.mark.parametrize(
    "head_dim_v,dtype,topk,capability,is_contiguous",
    [
        (512, torch.float16, 128, (10, 0), True),
        (576, torch.bfloat16, 128, (10, 0), True),
        (512, torch.bfloat16, 64, (10, 0), True),
        (512, torch.bfloat16, 129, (10, 0), True),
        (512, torch.bfloat16, 128, (10, 7), True),
        (512, torch.bfloat16, 128, (9, 0), True),
        (512, torch.bfloat16, 128, (10, 0), False),
    ],
)
@pytest.mark.parametrize("deterministic", [False, True], ids=["ordinary", "deterministic"])
def test_DSA_sparse_attention_backward_d576_2cta_dispatch_fallback(head_dim_v, dtype, topk, capability, is_contiguous, deterministic):
    """Unsupported dtype, D_v, top-k width, capability, or layout keeps the generic path in either mode."""
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100 import _select_sm100_backend
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    assert _select_sm100_backend(
        128,
        576,
        head_dim_v=head_dim_v,
        dtype=dtype,
        max_topk=topk,
        device_capability=capability,
        is_contiguous=is_contiguous,
        deterministic=deterministic,
    ) == ("generic_m64", 64)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("topk", [128, 512, 1024, 1152, 2048])
@pytest.mark.parametrize("has_topk_length", [False, True], ids=["full-topk", "lengths"])
@pytest.mark.parametrize("head_dim", [512, 576], ids=["d512", "d576"])
@torch_fork_set_rng(seed=20260829)
def test_DSA_sparse_attention_backward_sm100_h128_two_cta_masks_active_positive_oob_indices(head_dim, has_topk_length, topk, monkeypatch):
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    if head_dim == 512:
        _require_exact_sm100()
    else:
        _require_sm100()
        if torch.cuda.get_device_capability() not in _TWO_CTA_CAPABILITIES:
            pytest.skip("D576 2-CTA requires an SM100-class GPU (compute capability 10.0 or 10.3)")

    s_q, s_kv, num_heads = 4, topk + 64, 128
    head_dim_v = 512
    device = "cuda"
    dtype = torch.bfloat16
    # Match the production-scale inputs used by the DSA benchmark and the
    # other backward numerical tests while keeping the stricter 1e-2 oracle
    # tolerance below. This test targets active-OOB masking, not stress-scale
    # BF16 rounding.
    q = torch.randn(s_q, num_heads, head_dim, dtype=dtype, device=device) / 10
    kv = torch.randn(s_kv, head_dim, dtype=dtype, device=device) / 10
    dout = torch.randn(s_q, num_heads, head_dim_v, dtype=dtype, device=device) / 10
    attn_sink = torch.linspace(-1.5, 1.5, num_heads, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)

    # Positive out-of-range entries are inactive under the public reference
    # contract, exactly like -1 sentinels.  Keep at least one valid active
    # entry in every row so this exercises the normal pipeline rather than the
    # empty-row fast path.
    topk_idxs[0, 1] = s_kv
    topk_idxs[1, topk // 2 - 1] = torch.iinfo(torch.int32).max
    topk_idxs[2, topk - 2] = s_kv + 17
    topk_idxs[3, topk - 1] = s_kv
    topk_length = None
    if has_topk_length:
        topk_length = torch.tensor([2, topk // 2, topk - 1, topk], dtype=torch.int32, device=device)

    softmax_scale = head_dim**-0.5
    out, lse = ref_sparse_attention_forward_chunked(
        q,
        kv,
        attn_sink,
        topk_idxs,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
    )
    dq = torch.full_like(q, float("nan"))
    dkv = torch.full_like(kv, float("nan"))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    two_cta_calls = _spy_d576_two_cta_execute(monkeypatch) if head_dim == 576 else None
    prepared_calls = _spy_prepared_sm100_execute(monkeypatch) if head_dim == 512 else None
    result = DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale=softmax_scale,
        topk_length=topk_length,
        dq=dq,
        dkv=dkv,
        stream=stream,
    )

    if head_dim == 512:
        _assert_prepared_sm100_backend(prepared_calls, "h128_2cta_m64")
    else:
        # The D576 route compiles through its own plan cache; assert that its launch entry ran.
        assert two_cta_calls, "H128/D576 call did not execute the two-CTA backend"

    assert not torch.isnan(result["dq"]).any()
    assert not torch.isnan(result["dkv"]).any()
    assert not torch.isnan(result["d_sink"]).any()
    check_ref_dsa_sparse_attention_backward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        out,
        dout,
        lse,
        result["dq"],
        result["dkv"],
        result["d_sink"],
        softmax_scale=softmax_scale,
        topk_length=topk_length,
    )


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@torch_fork_set_rng(seed=20260830)
def test_DSA_sparse_attention_backward_sm100_h128_dsink_reduction_covers_tail():
    """Every query block, including a one-row tail, contributes to dSink."""
    _require_exact_sm100()
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    s_q, s_kv, num_heads, head_dim, topk = 257, 256, 128, 512, 128
    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device="cuda") / 10
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device="cuda") / 10
    dout = torch.randn_like(q) / 10
    attn_sink = torch.linspace(-2.0, 2.0, num_heads, dtype=torch.float32, device="cuda")
    topk_idxs = torch.stack([torch.randperm(s_kv, device="cuda")[:topk] for _ in range(s_q)]).to(torch.int32)
    pattern = torch.tensor([0, 1, 63, 64, 65, 127, 128, -3], dtype=torch.int32, device="cuda")
    topk_length = pattern.repeat((s_q + pattern.numel() - 1) // pattern.numel())[:s_q]
    topk_length[-1] = topk
    softmax_scale = head_dim**-0.5
    out, lse = ref_sparse_attention_forward_chunked(q, kv, attn_sink, topk_idxs, topk_length=topk_length, softmax_scale=softmax_scale)
    dq = torch.empty_like(q)
    dkv = torch.empty_like(kv)
    d_sink_runs = []
    final_result = None
    for _ in range(8):
        final_result = DSA.sparse_attention_backward_wrapper(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            softmax_scale=softmax_scale,
            topk_length=topk_length,
            dq=dq,
            dkv=dkv,
        )
        d_sink_runs.append(final_result["d_sink"])
    torch.cuda.synchronize()
    assert final_result is not None
    delta = (out.double() * dout.double()).sum(dim=-1)
    denominator = torch.logaddexp(lse.double(), attn_sink.double().unsqueeze(0))
    tail_contribution = -torch.exp(attn_sink.double() - denominator[-1]) * delta[-1]
    assert tail_contribution.abs().max() > 1e-8
    d_sink_ref = (-torch.exp(attn_sink.double().unsqueeze(0) - denominator) * delta).sum(dim=0)
    for d_sink in d_sink_runs:
        torch.testing.assert_close(
            d_sink.double(),
            d_sink_ref,
            atol=2e-6,
            rtol=2e-6,
        )
    assert torch.isfinite(final_result["dq"]).all()
    assert torch.isfinite(final_result["dkv"]).all()


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@torch_fork_set_rng(seed=20260831)
def test_DSA_sparse_attention_backward_sm100_h128_two_cta_cuda_graph(monkeypatch):
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_exact_sm100()

    s_q, s_kv, num_heads, head_dim, topk = 4, 192, 128, 512, 128
    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device="cuda") / 10
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device="cuda") / 10
    dout = torch.randn_like(q) / 10
    attn_sink = torch.linspace(-1.5, 1.5, num_heads, dtype=torch.float32, device="cuda")
    topk_idxs = torch.stack([torch.randperm(s_kv, device="cuda")[:topk] for _ in range(s_q)]).to(torch.int32)
    topk_length = torch.full((s_q,), topk, dtype=torch.int32, device="cuda")
    softmax_scale = head_dim**-0.5
    out, lse = ref_sparse_attention_forward_chunked(
        q,
        kv,
        attn_sink,
        topk_idxs,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
    )
    dq = torch.empty_like(q)
    dkv = torch.empty_like(kv)
    prepared_calls = _spy_prepared_sm100_execute(monkeypatch)

    def run():
        return DSA.sparse_attention_backward_wrapper(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            softmax_scale=softmax_scale,
            topk_length=topk_length,
            dq=dq,
            dkv=dkv,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    dq.fill_(float("nan"))
    dkv.fill_(float("nan"))
    captured["d_sink"].fill_(float("nan"))
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    _assert_prepared_sm100_backend(prepared_calls, "h128_2cta_m64")
    assert torch.isfinite(captured["dq"]).all()
    assert torch.isfinite(captured["dkv"]).all()
    assert torch.isfinite(captured["d_sink"]).all()
    check_ref_dsa_sparse_attention_backward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        out,
        dout,
        lse,
        captured["dq"],
        captured["dkv"],
        captured["d_sink"],
        softmax_scale=softmax_scale,
        topk_length=topk_length,
    )


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("head_dim", [512, 576], ids=["d512", "d576"])
@torch_fork_set_rng(seed=20261015)
def test_DSA_sparse_attention_backward_sm100_h128_two_cta_deterministic_reuse_and_graph(head_dim):
    """Reuse caller scratch and outputs across eager calls and two graph replays."""
    try:
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cuda-python not installed")

    _require_sm100()
    if torch.cuda.get_device_capability() not in _TWO_CTA_CAPABILITIES:
        pytest.skip("H128 two-CTA requires compute capability 10.0 or 10.3")

    plan, inputs, outputs, topk_length, softmax_scale, workspace = _plan_case(128, head_dim, 128, s_q=4, s_kv=192, has_topk_length=True, deterministic=True)
    expected_backend = "h128_2cta_m64" if head_dim == 512 else "h128_d576_2cta_m64"
    assert plan._backend == expected_backend
    launch_stream = torch.cuda.Stream()
    launch_stream.wait_stream(torch.cuda.current_stream())
    stream = cuda.CUstream(launch_stream.cuda_stream)

    def poison():
        for buffer in (workspace, *outputs):
            bitwise_bits(buffer).fill_(0xA5)

    def execute():
        dq, dkv, d_sink = outputs
        return plan.execute(
            *inputs,
            dq,
            dkv,
            topk_length=topk_length,
            softmax_scale=softmax_scale,
            workspace=workspace,
            current_stream=stream,
            d_sink=d_sink,
        )

    def run():
        with torch.cuda.stream(launch_stream):
            poison()
            result = execute()
            return tuple(tensor.clone() for tensor in result)

    assert_bitwise_runs(run, repeats=10, label=f"H128/D{head_dim} two-CTA")
    with torch.cuda.stream(launch_stream):
        eager = tuple(tensor.clone() for tensor in outputs)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=launch_stream):
            execute()
        for repeat in range(2):
            with torch.cuda.stream(launch_stream):
                poison()
                graph.replay()
            torch.cuda.synchronize()
            for name, actual, expected in zip(("dQ", "dKV", "dSink"), outputs, eager):
                assert torch.equal(bitwise_bits(actual), bitwise_bits(expected)), f"{name} differs from eager on graph replay {repeat}"
    finally:
        graph.reset()

    q, kv, out, dout, lse, attn_sink, topk_idxs = inputs
    check_ref_dsa_sparse_attention_backward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        out,
        dout,
        lse,
        *outputs,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        atol=5e-2,
        rtol=5e-2,
    )


def _run_DSA_sparse_attention_backward_wrapper(
    *,
    dtype,
    acc_dtype,
    head_dim,
    head_dim_v,
    num_heads,
    topk,
    has_topk_length,
    request,
    allow_unsupported=True,
    s_q_default=1024,
    deterministic=False,
):
    """Run wrapper backward against reference gradients, optionally skipping unsupported cases."""
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    cfg = dsa_init(
        request=request,
        dtype=dtype,
        acc_dtype=acc_dtype,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        num_heads=num_heads,
        topk=topk,
        has_topk_length=has_topk_length,
        min_compute_capability=90,
        s_q_default=s_q_default,
        s_kv_default=4096,
    )
    cfg["h_q"] = num_heads

    q, kv, attn_sink, topk_idxs, topk_length = _allocate(cfg, has_topk_length)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    # Run reference forward to get out and the KV-only lse for backward.
    out, lse = ref_sparse_attention_forward_chunked(
        q,
        kv,
        attn_sink,
        topk_idxs,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
    )
    dout = torch.randn_like(out)

    try:
        result = DSA.sparse_attention_backward_wrapper(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            softmax_scale=softmax_scale,
            topk_length=topk_length,
            stream=stream,
            deterministic=deterministic,
        )
    except (ValueError, NotImplementedError, RuntimeError) as e:
        if not allow_unsupported:
            raise
        pytest.skip(f"Unsupported testcase: {e}")

    dq, dkv, d_sink = result["dq"], result["dkv"], result["d_sink"]

    if not cfg["skip_ref"]:
        # The DKV accumulation is known to have precision limitations
        # vs. the FP32 autograd reference; use generous tolerances.
        check_ref_dsa_sparse_attention_backward(
            q,
            kv,
            attn_sink,
            topk_idxs,
            out,
            dout,
            lse,
            dq,
            dkv,
            d_sink,
            softmax_scale=softmax_scale,
            topk_length=topk_length,
            atol=5e-2,
            rtol=5e-2,
        )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_sparse_attention_backward_params
def test_DSA_sparse_attention_backward_wrapper(
    dtype,
    acc_dtype,
    head_dim,
    head_dim_v,
    num_heads,
    topk,
    has_topk_length,
    request,
):
    """Check wrapper gradients against the reference over the shared DSA parameter set."""
    _run_DSA_sparse_attention_backward_wrapper(
        dtype=dtype,
        acc_dtype=acc_dtype,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        num_heads=num_heads,
        topk=topk,
        has_topk_length=has_topk_length,
        request=request,
    )


@pytest.mark.L0
@pytest.mark.parametrize("has_topk_length", [False, True], ids=["full-topk", "lengths"])
@torch_fork_set_rng(seed=422)
def test_DSA_sparse_attention_backward_wrapper_h64_d576_topk2048(has_topk_length, request):
    """Cover the primary H64/D576 training shape beyond one sparse tile."""
    _require_sm100()
    _run_DSA_sparse_attention_backward_wrapper(
        dtype=torch.bfloat16,
        acc_dtype=torch.float32,
        head_dim=576,
        head_dim_v=512,
        num_heads=64,
        topk=2048,
        has_topk_length=has_topk_length,
        request=request,
        allow_unsupported=False,
        s_q_default=8,
    )


@pytest.mark.L0
@pytest.mark.parametrize("num_heads", [96, 128])
@torch_fork_set_rng(seed=423)
def test_DSA_sparse_attention_backward_wrapper_generic_d576_head_blocks(num_heads, request):
    """Guard the other head counts that share the generic D576 pipeline."""
    _require_sm100()
    _run_DSA_sparse_attention_backward_wrapper(
        dtype=torch.bfloat16,
        acc_dtype=torch.float32,
        head_dim=576,
        head_dim_v=512,
        num_heads=num_heads,
        topk=128,
        has_topk_length=True,
        request=request,
        allow_unsupported=False,
        s_q_default=3,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=424)
def test_DSA_sparse_attention_backward_wrapper_generic_d512_reducer_generations(request):
    """Exercise three sparse tiles through the generic D512 reducer."""
    _require_sm100()
    _run_DSA_sparse_attention_backward_wrapper(
        dtype=torch.bfloat16,
        acc_dtype=torch.float32,
        head_dim=512,
        head_dim_v=512,
        num_heads=64,
        topk=129,
        has_topk_length=False,
        request=request,
        allow_unsupported=False,
        s_q_default=3,
    )


@pytest.mark.L0
@pytest.mark.parametrize("deterministic", [False, True], ids=["ordinary", "deterministic"])
@torch_fork_set_rng(seed=425)
def test_DSA_sparse_attention_backward_wrapper_generic_d576_reducer_generations(deterministic, request):
    """Exercise three sparse tiles through every generic D576 reducer mode."""
    _require_sm100()
    _run_DSA_sparse_attention_backward_wrapper(
        dtype=torch.bfloat16,
        acc_dtype=torch.float32,
        head_dim=576,
        head_dim_v=512,
        num_heads=64,
        topk=129,
        has_topk_length=False,
        request=request,
        allow_unsupported=False,
        s_q_default=3,
        deterministic=deterministic,
    )


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("topk", [128, 512, 1024, 1152, 2048])
@pytest.mark.parametrize("has_topk_length", [False, True], ids=["full-topk", "lengths"])
@pytest.mark.parametrize("deterministic", [False, True], ids=["ordinary", "deterministic"])
@torch_fork_set_rng(seed=0)
def test_DSA_sparse_attention_backward_wrapper_h128(topk, has_topk_length, deterministic, request):
    if deterministic:
        _require_sm100()
    _run_DSA_sparse_attention_backward_wrapper(
        dtype=torch.bfloat16,
        acc_dtype=torch.float32,
        head_dim=512,
        head_dim_v=512,
        num_heads=128,
        topk=topk,
        has_topk_length=has_topk_length,
        request=request,
        deterministic=deterministic,
        allow_unsupported=not deterministic,
    )


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("has_topk_length", [False, True], ids=["full-topk", "lengths"])
@torch_fork_set_rng(seed=96)
def test_DSA_sparse_attention_backward_wrapper_h96_d576(has_topk_length, request):
    """Validate the native H96 route, or its device fallback, against the FP32 reference."""
    _run_DSA_sparse_attention_backward_wrapper(
        dtype=torch.bfloat16,
        acc_dtype=torch.float32,
        head_dim=576,
        head_dim_v=512,
        num_heads=96,
        topk=512,
        has_topk_length=has_topk_length,
        request=request,
    )


@pytest.mark.L0
@pytest.mark.parametrize("has_topk_length", [False, True], ids=["sentinel-padding", "lengths"])
@torch_fork_set_rng(seed=960)
def test_DSA_sparse_attention_backward_sm100_h96_native_boundaries(has_topk_length, monkeypatch):
    """Cover empty, partial, and multi-tile rows in the native H96 backend."""
    _require_sm100()
    if torch.cuda.get_device_capability() not in _TWO_CTA_CAPABILITIES:
        pytest.skip("H96 two-CTA requires compute capability 10.0 or 10.3")
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    s_q, s_kv, topk_width = 4, 129, 512
    num_heads, head_dim = 96, 576
    softmax_scale = 1.0 / math.sqrt(head_dim)
    lengths = torch.tensor([0, 1, 65, 129], dtype=torch.int32, device=device)

    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device) / 10
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device) / 10
    attn_sink = torch.linspace(-2.0, 2.0, num_heads, dtype=torch.float32, device=device)
    topk_idxs = torch.full((s_q, topk_width), -1, dtype=torch.int32, device=device)
    for row, length in enumerate(lengths.tolist()):
        if length:
            topk_idxs[row, :length] = torch.randperm(s_kv, device=device)[:length].to(torch.int32)
    topk_length = lengths if has_topk_length else None

    out, lse = ref_sparse_attention_forward_chunked(
        q,
        kv,
        attn_sink,
        topk_idxs,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
    )
    dout = torch.randn_like(out)
    prepared_calls = _spy_prepared_sm100_execute(monkeypatch)
    result = DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale=softmax_scale,
        topk_length=topk_length,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(result["dq"]).all()
    assert torch.isfinite(result["dkv"]).all()
    assert torch.isfinite(result["d_sink"]).all()
    check_ref_dsa_sparse_attention_backward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        out,
        dout,
        lse,
        result["dq"],
        result["dkv"],
        result["d_sink"],
        softmax_scale=softmax_scale,
        topk_length=topk_length,
        atol=5e-2,
        rtol=5e-2,
    )
    _assert_prepared_sm100_backend(prepared_calls, "h96_d576_2cta_m64")


@pytest.mark.parametrize(
    "s_q,topk",
    [
        # Short query batches spread each query over split clusters (one dQ
        # owner plus dV/dK workers).  s_q=129 exceeds the two-SM cluster
        # count of SM100 parts, so those rows take the single-cluster
        # schedule that long sequences use in production.
        pytest.param(1, 128, marks=pytest.mark.L0),
        pytest.param(9, 512, marks=pytest.mark.L0),
        pytest.param(9, 1024, marks=pytest.mark.L1),
        pytest.param(9, 2048, marks=pytest.mark.L1),
        pytest.param(129, 512, marks=pytest.mark.L1),
        pytest.param(129, 1152, marks=pytest.mark.L1),
        pytest.param(129, 2048, marks=pytest.mark.L1),
    ],
)
@pytest.mark.parametrize("has_topk_length", [False, True], ids=["full-topk", "lengths"])
@pytest.mark.parametrize("deterministic", [False, True], ids=["ordinary", "deterministic"])
@torch_fork_set_rng(seed=576)
def test_DSA_sparse_attention_backward_d576_2cta_numerics(s_q, topk, has_topk_length, deterministic, monkeypatch):
    """Compare the two-CTA gradients with the PyTorch reference in both cluster regimes."""
    _require_sm100()
    if torch.cuda.get_device_capability() not in _TWO_CTA_CAPABILITIES:
        pytest.skip("D576 2-CTA requires an SM100-class GPU (compute capability 10.0 or 10.3)")
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    cfg = {"s_q": s_q, "s_kv": topk + 17, "h_q": 128, "head_dim": 576, "topk": topk}
    q, kv, attn_sink, topk_idxs, topk_length = _allocate(cfg, has_topk_length)
    softmax_scale = 1.0 / math.sqrt(576)
    out, lse = ref_sparse_attention_forward_chunked(q, kv, attn_sink, topk_idxs, topk_length=topk_length, softmax_scale=softmax_scale)
    dout = torch.randn_like(out)

    two_cta_calls = _spy_d576_two_cta_execute(monkeypatch)
    result = DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        deterministic=deterministic,
    )
    assert two_cta_calls, "the wrapper did not execute the H128/D576 two-CTA backend"
    check_ref_dsa_sparse_attention_backward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        out,
        dout,
        lse,
        result["dq"],
        result["dkv"],
        result["d_sink"],
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.L0
@pytest.mark.parametrize("duplicate", [False, True], ids=["unique", "duplicate"])
@torch_fork_set_rng(seed=20261020)
def test_DSA_sparse_attention_backward_d576_2cta_sorted_topk_numerics(duplicate, monkeypatch):
    """Sorted Top-K rows use the ordered fast path and preserve duplicate accumulation."""
    _require_sm100()
    if torch.cuda.get_device_capability() not in _TWO_CTA_CAPABILITIES:
        pytest.skip("D576 2-CTA requires an SM100-class GPU (compute capability 10.0 or 10.3)")
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    cfg = {"s_q": 9, "s_kv": 145, "h_q": 96, "head_dim": 576, "topk": 128}
    q, kv, attn_sink, topk_idxs, topk_length = _allocate(cfg, not duplicate)
    topk_idxs = topk_idxs.sort(dim=-1).values
    if duplicate:
        topk_idxs[:, 1] = topk_idxs[:, 0]
    softmax_scale = 1.0 / math.sqrt(576)
    out, lse = ref_sparse_attention_forward_chunked(
        q,
        kv,
        attn_sink,
        topk_idxs,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
    )
    dout = torch.randn_like(out)

    two_cta_calls = _spy_d576_two_cta_execute(monkeypatch)
    result = DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        deterministic=True,
    )
    assert two_cta_calls, "the wrapper did not execute the H96/D576 two-CTA backend"
    check_ref_dsa_sparse_attention_backward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        out,
        dout,
        lse,
        result["dq"],
        result["dkv"],
        result["d_sink"],
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.L0
@pytest.mark.parametrize("use_plan", [False, True], ids=["wrapper", "plan"])
@torch_fork_set_rng(seed=577)
def test_DSA_sparse_attention_backward_d576_2cta_ignored_nan_and_workspace_reuse(use_plan, monkeypatch):
    """Ignored sparse slots cannot contaminate gradients or reused storage."""
    _require_sm100()
    if torch.cuda.get_device_capability() not in _TWO_CTA_CAPABILITIES:
        pytest.skip("D576 2-CTA requires an SM100-class GPU (compute capability 10.0 or 10.3)")
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
        import cutlass.cute as cute
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    cfg = {"s_q": 9, "s_kv": 529, "h_q": 128, "head_dim": 576, "topk": 128}
    softmax_scale = 1.0 / math.sqrt(576)
    lengths = torch.tensor([-3, 0, 1, 63, 64, 65, 127, 128, 129], dtype=torch.int32, device="cuda")
    ignored = torch.arange(cfg["topk"], device="cuda").unsqueeze(0) >= lengths.unsqueeze(1)
    side_stream = torch.cuda.Stream()
    plan = None

    def unexpected_hotpath_call(*args, **kwargs):
        """Fail the test if execute() reaches a compile or an allocation."""
        pytest.fail("execute() must not compile kernels or allocate tensors")

    two_cta_calls = _spy_d576_two_cta_execute(monkeypatch)
    for _ in range(2):
        q, kv, attn_sink, _, _ = _allocate(cfg, False)
        # Reserve the final, legally allocated KV row for ignored suffixes.
        # It is never selected by any query's active sparse prefix.
        topk_idxs = torch.stack([torch.randperm(cfg["s_kv"] - 1, device="cuda")[: cfg["topk"]] for _ in range(cfg["s_q"])]).to(torch.int32)
        topk_idxs[ignored] = cfg["s_kv"] - 1
        out, lse = ref_sparse_attention_forward_chunked(q, kv, attn_sink, topk_idxs, topk_length=lengths, softmax_scale=softmax_scale)
        dout = torch.randn_like(out)
        poisoned_kv = kv.clone()
        poisoned_kv[-1] = float("nan")

        if plan is None:
            plan = DSA.SparseAttentionBackward(q, poisoned_kv, out, dout, lse, attn_sink, topk_idxs, sample_topk_length=lengths, softmax_scale=softmax_scale)
            assert plan.check_support()
            plan.compile()
            workspace_bytes = plan.scratch_workspace_bytes()
            assert workspace_bytes > 0
            workspace_storage = torch.empty(workspace_bytes + 128, dtype=torch.uint8, device="cuda")
            workspace = workspace_storage[:workspace_bytes]
            dq_buffer, dkv_buffer = torch.empty_like(q), torch.empty_like(kv)
            d_sink_buffer = torch.empty_like(attn_sink)

        dq_buffer.fill_(float("nan"))
        dkv_buffer.fill_(float("nan"))
        d_sink_buffer.fill_(float("nan"))
        workspace_storage.fill_(0xA5)
        side_stream.wait_stream(torch.cuda.current_stream())
        if use_plan:
            with monkeypatch.context() as patch:
                patch.setattr(cute, "compile", unexpected_hotpath_call)
                for name in ("empty", "empty_like", "zeros", "zeros_like"):
                    patch.setattr(torch, name, unexpected_hotpath_call)
                outputs = plan.execute(
                    q,
                    poisoned_kv,
                    out,
                    dout,
                    lse,
                    attn_sink,
                    topk_idxs,
                    topk_length=lengths,
                    dq=dq_buffer,
                    dkv=dkv_buffer,
                    d_sink=d_sink_buffer,
                    workspace=workspace,
                    current_stream=cuda.CUstream(side_stream.cuda_stream),
                )
            result = dict(zip(("dq", "dkv", "d_sink"), outputs, strict=True))
            assert result["d_sink"] is d_sink_buffer
        else:
            # Omitting stream must honor the current PyTorch stream on a
            # cache hit as well as the first invocation.
            with torch.cuda.stream(side_stream):
                result = DSA.sparse_attention_backward_wrapper(
                    q,
                    poisoned_kv,
                    out,
                    dout,
                    lse,
                    attn_sink,
                    topk_idxs,
                    softmax_scale=softmax_scale,
                    topk_length=lengths,
                    dq=dq_buffer,
                    dkv=dkv_buffer,
                    workspace=workspace,
                )
        side_stream.synchronize()

        assert len(two_cta_calls) == 1, "each call must run exactly one H128/D576 two-CTA launch sequence"
        two_cta_calls.clear()
        assert plan._backend == "h128_d576_2cta_m64"
        assert result["dq"] is dq_buffer
        assert result["dkv"] is dkv_buffer
        assert all(torch.isfinite(result[name]).all() for name in ("dq", "dkv", "d_sink"))
        assert torch.equal(result["dkv"][-1], torch.zeros_like(result["dkv"][-1]))
        assert (workspace_storage[workspace_bytes:] == 0xA5).all()
        # The dense reference would itself multiply zero weights by NaN.
        # Use the finite KV copy, which differs only in an unselected row.
        check_ref_dsa_sparse_attention_backward(
            q,
            kv,
            attn_sink,
            topk_idxs,
            out,
            dout,
            lse,
            result["dq"],
            result["dkv"],
            result["d_sink"],
            topk_length=lengths,
            softmax_scale=softmax_scale,
            atol=5e-2,
            rtol=5e-2,
        )


@pytest.mark.L0
@pytest.mark.parametrize("num_heads", [32, 128])
@pytest.mark.parametrize("has_topk_length", [False, True], ids=["full-topk", "lengths"])
@torch_fork_set_rng(seed=418)
def test_DSA_sparse_attention_backward_sm100_576_includes_sink_in_normalization(num_heads, has_topk_length, monkeypatch):
    """The 576/512 specialization must include sink mass in every gradient."""
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        pytest.skip("sink-normalization regression test targets the SM100 kernel")
    if num_heads == 128 and (major, minor) not in _TWO_CTA_CAPABILITIES:
        # The H128 case covers the D576 two-CTA route; elsewhere H128/D576 runs
        # the generic M64 kernel, which this PR does not change.
        pytest.skip("the H128/D576 sink-normalization case targets the two-CTA route (SM100-class GPUs)")

    device = torch.device("cuda")
    s_q = 5
    s_kv = topk = 512
    head_dim = 576
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device)
    # Keep the sink probability significant even for the full-top-k row so
    # both compiled variants fail clearly if they normalize by KV-only LSE.
    attn_sink = torch.full((num_heads,), math.log(topk), dtype=torch.float32, device=device)
    topk_idxs = torch.arange(topk, dtype=torch.int32, device=device).expand(s_q, -1).contiguous()
    lengths = torch.tensor([1, 63, 64, 65, 512], dtype=torch.int32, device=device)
    topk_length = lengths if has_topk_length else None

    out, lse = ref_sparse_attention_forward_chunked(
        q,
        kv,
        attn_sink,
        topk_idxs,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
    )
    # O.dO is positive for every head, so a zero d_sink cannot pass by
    # landing inside the absolute tolerance.
    dout = out.clone()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    two_cta_calls = _spy_d576_two_cta_execute(monkeypatch) if num_heads == 128 else None
    result = DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale=softmax_scale,
        topk_length=topk_length,
        stream=stream,
    )

    if num_heads == 128:
        assert two_cta_calls, "the wrapper did not execute the H128/D576 two-CTA backend"
    check_ref_dsa_sparse_attention_backward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        out,
        dout,
        lse,
        result["dq"],
        result["dkv"],
        result["d_sink"],
        softmax_scale=softmax_scale,
        topk_length=topk_length,
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=433)
@pytest.mark.parametrize(
    "head_dim,num_heads,topk_length_values",
    [
        pytest.param(512, 64, (-3, 0, 1, 63, 64, 65, 128), id="d512-mixed"),
        pytest.param(512, 128, (-3, 0, 1, 63, 64, 65, 128), id="d512-h128-two-cta"),
        pytest.param(576, 32, (0, 1, 64, 65, 128, 0), id="d576-mixed"),
        pytest.param(576, 16, (0, 1, 127, 128, 129, 511, 512, 513), id="d576-h16-m128-boundaries"),
        pytest.param(576, 32, (0, 1, 63, 64, 65, 127, 128), id="d576-h32-m64-boundaries"),
        pytest.param(576, 64, (0, 1, 63, 64, 65, 127, 128), id="d576-h64-m64-boundaries"),
        # Cross the standalone OdO/LSE preprocessor's 41-query block boundary.
        pytest.param(576, 64, (0, 1, 128) * 14, id="d576-h64-odo-query-tail"),
        pytest.param(
            576,
            64,
            (0, 1, 63, 64, 65, 127, 128, 129, 511, 512, 513, 1023, 1024, 1025, 2047, 2048),
            id="d576-h64-topk2048-boundaries",
        ),
        pytest.param(576, 128, (-3, 0, 1, 63, 64, 65, 127, 128, 129), id="d576-h128-two-cta-length-boundaries"),
    ],
)
def test_DSA_sparse_attention_backward_zero_topk_length(head_dim, num_heads, topk_length_values):
    """A nonpositive top-k length must be treated as an empty attention row."""
    if not torch.cuda.is_available():
        pytest.skip("SM90+ GPU required")
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 90:
        pytest.skip("sparse attention backward requires SM90+")
    if num_heads == 128 and head_dim == 512:
        _require_exact_sm100()
    if num_heads == 128 and head_dim == 576 and (major, minor) not in _TWO_CTA_CAPABILITIES:
        # Lengths above max_topk are part of the two-CTA route's contract; the
        # generic M64 pipeline that serves H128/D576 elsewhere does not clamp them.
        pytest.skip("the H128/D576 length-boundary case targets the two-CTA route (SM100-class GPUs)")

    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    s_q = len(topk_length_values) if topk_length_values is not None else 2
    max_topk_length = max(topk_length_values)
    if head_dim == 576 and num_heads == 16:
        s_kv, topk = 640, 513
    elif head_dim == 576 and num_heads == 64 and max_topk_length > 128:
        s_kv, topk = 2112, 2048
    else:
        s_kv, topk = 256, 128
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device) / 10
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device) / 10
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    base_topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)

    # D512 covers defensive negative handling around the 64-row tile boundary.
    # D576/H16 covers both sides of the dedicated kernel's 128-row boundary and
    # its final feature/top-k tails. D576/H32 and H64 cover the corresponding
    # M64 boundaries; the extended H64 case crosses every power-of-two boundary
    # through topk=2048. Every case also exercises the all-empty fast path.
    # D576/H128 covers the two-CTA route, including lengths above max_topk.
    topk_length_cases = [torch.zeros(s_q, dtype=torch.int32, device=device)]
    if topk_length_values is not None:
        topk_length_cases.insert(0, torch.tensor(topk_length_values, dtype=torch.int32, device=device))
    positions = torch.arange(topk, device=device).unsqueeze(0)

    for topk_length in topk_length_cases:
        topk_idxs = base_topk_idxs.clone()
        topk_idxs[positions >= topk_length.unsqueeze(1)] = -1
        empty_rows = topk_length <= 0
        # Invalid values in ignored slots exercise that the empty fast path
        # does not consult topk_idxs before exiting.
        topk_idxs[empty_rows] = torch.iinfo(torch.int32).max

        out, lse = ref_sparse_attention_forward_chunked(
            q,
            kv,
            attn_sink,
            topk_idxs,
            topk_length=topk_length,
            softmax_scale=softmax_scale,
        )
        assert torch.equal(out[empty_rows], torch.zeros_like(out[empty_rows]))
        assert torch.isposinf(lse[empty_rows]).all()
        dout = torch.randn_like(out)

        # In particular, the empty-row fast path must overwrite caller-owned
        # storage; torch.empty_like() could otherwise hide a missing dQ store.
        dq_buffer = torch.full_like(q, float("nan"))
        dkv_buffer = torch.full_like(kv, float("nan"))
        result = DSA.sparse_attention_backward_wrapper(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            softmax_scale=softmax_scale,
            topk_length=topk_length,
            dq=dq_buffer,
            dkv=dkv_buffer,
        )
        # Reaching the synchronization is also the regression check for the
        # empty-row failures: an SM100 barrier deadlock, and on SM90 both a
        # one-sided named-barrier wait and an out-of-range top-k index read
        # from the unconditional first n_block.
        torch.cuda.synchronize()

        dq, dkv, d_sink = result["dq"], result["dkv"], result["d_sink"]
        assert dq.data_ptr() == dq_buffer.data_ptr()
        assert dkv.data_ptr() == dkv_buffer.data_ptr()
        assert torch.isfinite(dq).all()
        assert torch.isfinite(dkv).all()
        assert torch.isfinite(d_sink).all()
        assert torch.equal(dq[empty_rows], torch.zeros_like(dq[empty_rows]))

        check_ref_dsa_sparse_attention_backward(
            q,
            kv,
            attn_sink,
            topk_idxs,
            out,
            dout,
            lse,
            dq,
            dkv,
            d_sink,
            softmax_scale=softmax_scale,
            topk_length=topk_length,
            atol=5e-2,
            rtol=5e-2,
        )

        if torch.all(empty_rows):
            assert torch.equal(dkv, torch.zeros_like(dkv))
            assert torch.equal(d_sink, torch.zeros_like(d_sink))


@pytest.mark.L0
@torch_fork_set_rng(seed=676)
@pytest.mark.parametrize(
    "invalid_value,topk_length_value,topk_width",
    [
        pytest.param(-1, None, 64, id="negative-padding"),
        pytest.param(-1, None, 128, id="multi-tile-negative-padding"),
        pytest.param(torch.iinfo(torch.int32).max, 1, 64, id="past-length"),
        pytest.param(-1, 64, 64, id="active-negative"),
        pytest.param(4097, None, 64, id="active-positive-oob"),
    ],
)
def test_DSA_sparse_attention_backward_sm100_masks_invalid_topk_rows(invalid_value, topk_length_value, topk_width):
    """Invalid sparse rows must contribute neither probability nor an OOB KV access."""
    _require_sm100()

    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    num_heads, head_dim, head_dim_v = 64, 576, 512
    q = torch.full((1, num_heads, head_dim), 8.0, dtype=torch.bfloat16, device=device)
    kv = torch.full((1, head_dim), -8.0, dtype=torch.bfloat16, device=device)
    dout = torch.randn(1, num_heads, head_dim_v, dtype=torch.bfloat16, device=device)
    attn_sink = torch.full((num_heads,), -math.inf, dtype=torch.float32, device=device)
    topk_idxs = torch.full((1, topk_width), invalid_value, dtype=torch.int32, device=device)
    topk_idxs[:, 0] = 0
    topk_length = None if topk_length_value is None else torch.full((1,), topk_length_value, dtype=torch.int32, device=device)

    out = kv[0, :head_dim_v].view(1, 1, head_dim_v).expand(1, num_heads, head_dim_v).contiguous()
    lse = ((q.float() * kv[0].float()).sum(dim=-1) * (192**-0.5)).contiguous()
    result = DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale=192**-0.5,
        topk_length=topk_length,
    )

    expected_dkv = torch.zeros_like(kv)
    expected_dkv[0, :head_dim_v] = dout.float().sum(dim=(0, 1)).to(torch.bfloat16)
    assert torch.isfinite(result["dq"]).all()
    assert torch.isfinite(result["dkv"]).all()
    assert torch.isfinite(result["d_sink"]).all()
    torch.testing.assert_close(result["dq"].float(), torch.zeros_like(result["dq"], dtype=torch.float32), atol=3e-5, rtol=0)
    torch.testing.assert_close(result["dkv"], expected_dkv, atol=5e-2, rtol=1e-2)
    assert torch.equal(result["d_sink"], torch.zeros_like(result["d_sink"]))


@pytest.mark.L0
@torch_fork_set_rng(seed=677)
@pytest.mark.parametrize("deterministic", [False, True], ids=["ordinary", "deterministic"])
@pytest.mark.parametrize("sink_value,empty_row", [(-math.inf, True), (math.inf, False)], ids=["empty-no-mass", "positive-infinite-sink"])
def test_DSA_sparse_attention_backward_sm100_handles_infinite_sink_limits(sink_value, empty_row, deterministic):
    """Sink/LSE infinities must use their limiting probabilities without inf-inf."""
    _require_sm100()
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    num_heads, head_dim, head_dim_v = 64, 576, 512
    q = torch.randn(1, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    kv = torch.randn(1, head_dim, dtype=torch.bfloat16, device=device)
    out = torch.zeros(1, num_heads, head_dim_v, dtype=torch.bfloat16, device=device)
    dout = torch.randn_like(out)
    attn_sink = torch.full((num_heads,), sink_value, dtype=torch.float32, device=device)
    topk_idxs = torch.full((1, 64), torch.iinfo(torch.int32).max if empty_row else -1, dtype=torch.int32, device=device)
    topk_length = torch.zeros(1, dtype=torch.int32, device=device) if empty_row else None
    if empty_row:
        lse = torch.full((1, num_heads), -math.inf, dtype=torch.float32, device=device)
    else:
        topk_idxs[:, 0] = 0
        lse = ((q.float() * kv[0].float()).sum(dim=-1) * (192**-0.5)).contiguous()

    result = DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale=192**-0.5,
        topk_length=topk_length,
        deterministic=deterministic,
    )

    assert torch.equal(result["dq"], torch.zeros_like(result["dq"]))
    assert torch.equal(result["dkv"], torch.zeros_like(result["dkv"]))
    assert torch.equal(result["d_sink"], torch.zeros_like(result["d_sink"]))


@pytest.mark.L0
@torch_fork_set_rng(seed=785)
@pytest.mark.parametrize(
    "num_heads,expected_backend",
    [
        pytest.param(16, "h16_m128", id="h16-m128"),
        pytest.param(32, "h32_m64", id="h32-m64"),
        pytest.param(96, "h96_h64_h32", id="h96-h64-h32"),
    ],
)
def test_DSA_sparse_attention_backward_sm100_specialized_masks_invalid_topk_rows(num_heads, expected_backend, monkeypatch):
    """Specialized kernels must mask invalid rows before probability use and KV access."""
    _require_sm100()

    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    head_dim, head_dim_v = 576, 512
    topk_width = 256
    q = torch.full((1, num_heads, head_dim), 8.0, dtype=torch.bfloat16, device=device)
    kv = torch.full((1, head_dim), -8.0, dtype=torch.bfloat16, device=device)
    dout = torch.randn(1, num_heads, head_dim_v, dtype=torch.bfloat16, device=device)
    attn_sink = torch.full((num_heads,), -math.inf, dtype=torch.float32, device=device)
    out = kv[0, :head_dim_v].view(1, 1, head_dim_v).expand(1, num_heads, head_dim_v).contiguous()
    one_key_lse = ((q.float() * kv[0].float()).sum(dim=-1) * (192**-0.5)).contiguous()
    prepared_calls = _spy_prepared_sm100_execute(monkeypatch)

    def run_and_check(topk_idxs, topk_length, lse):
        result = DSA.sparse_attention_backward_wrapper(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            softmax_scale=192**-0.5,
            topk_length=topk_length,
        )
        torch.cuda.synchronize()

        expected_dkv = torch.zeros_like(kv)
        expected_dkv[0, :head_dim_v] = dout.float().sum(dim=(0, 1)).to(torch.bfloat16)
        assert torch.isfinite(result["dq"]).all()
        assert torch.isfinite(result["dkv"]).all()
        assert torch.isfinite(result["d_sink"]).all()
        torch.testing.assert_close(result["dq"].float(), torch.zeros_like(result["dq"], dtype=torch.float32), atol=3e-5, rtol=0)
        torch.testing.assert_close(result["dkv"], expected_dkv, atol=5e-2, rtol=1e-2)
        assert torch.equal(result["d_sink"], torch.zeros_like(result["d_sink"]))

    # Keep the width fixed so both has_topk_length specializations compile
    # only once.  The cases cover padding, ignored entries past the logical
    # length, active negative/positive OOB entries, and a malformed length
    # larger than the physical index tensor.
    cases = (
        (-1, None),
        (torch.iinfo(torch.int32).max, 1),
        (-1, topk_width),
        (4097, None),
        (-1, topk_width + 1),
    )
    for invalid_value, topk_length_value in cases:
        topk_idxs = torch.full((1, topk_width), invalid_value, dtype=torch.int32, device=device)
        topk_idxs[:, 0] = 0
        topk_length = None if topk_length_value is None else torch.full((1,), topk_length_value, dtype=torch.int32, device=device)
        run_and_check(topk_idxs, topk_length, one_key_lse)

    # Exercise every packed-mask byte/subgroup, both H16 load passes, and
    # multiple tiles with valid rows on both sides of each invalid hole.
    topk_idxs = torch.zeros((1, topk_width), dtype=torch.int32, device=device)
    invalid_positions = torch.tensor(
        [1, 2, 3, 4, 8, 12, 28, 32, 60, 63, 65, 68, 72, 92, 96, 124, 127, 129, 132, 159, 191, 193, 220, 255],
        dtype=torch.int64,
        device=device,
    )
    topk_idxs[:, invalid_positions[::2]] = -1
    topk_idxs[:, invalid_positions[1::2]] = 4097
    valid_count = topk_width - invalid_positions.numel()
    repeated_key_lse = one_key_lse + math.log(valid_count)
    run_and_check(topk_idxs, None, repeated_key_lse)

    _assert_prepared_sm100_backend(prepared_calls, expected_backend)


@pytest.mark.L0
@torch_fork_set_rng(seed=786)
@pytest.mark.parametrize(
    "num_heads,expected_backend",
    [
        pytest.param(16, "h16_m128", id="h16-m128"),
        pytest.param(32, "h32_m64", id="h32-m64"),
        pytest.param(96, "h96_h64_h32", id="h96-h64-h32"),
    ],
)
def test_DSA_sparse_attention_backward_sm100_specialized_handles_sink_limits(num_heads, expected_backend, monkeypatch):
    """Specialized sink paths must implement empty and saturating softmax limits."""
    _require_sm100()

    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    head_dim, head_dim_v = 576, 512
    q = torch.randn(1, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    kv = torch.randn(1, head_dim, dtype=torch.bfloat16, device=device)
    out = torch.zeros(1, num_heads, head_dim_v, dtype=torch.bfloat16, device=device)
    dout = torch.randn_like(out)
    prepared_calls = _spy_prepared_sm100_execute(monkeypatch)

    cases = [(-math.inf, True), (math.inf, False)]
    # The finite-overflow case targets behavior implemented by the H16/H32
    # kernels themselves; the H96 composition intentionally retains the
    # existing generic H64 behavior for its first 64 heads.
    if num_heads != 96:
        cases.append((2.4e38, False))
    for sink_value, empty_row in cases:
        attn_sink = torch.full((num_heads,), sink_value, dtype=torch.float32, device=device)
        topk_width = 64
        topk_idxs = torch.full((1, topk_width), torch.iinfo(torch.int32).max if empty_row else -1, dtype=torch.int32, device=device)
        topk_length = torch.zeros(1, dtype=torch.int32, device=device) if empty_row else None
        if empty_row:
            lse = torch.full((1, num_heads), -math.inf, dtype=torch.float32, device=device)
        else:
            topk_idxs[:, 0] = 0
            lse = ((q.float() * kv[0].float()).sum(dim=-1) * (192**-0.5)).contiguous()

        result = DSA.sparse_attention_backward_wrapper(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            softmax_scale=192**-0.5,
            topk_length=topk_length,
        )
        torch.cuda.synchronize()

        assert torch.equal(result["dq"], torch.zeros_like(result["dq"]))
        assert torch.equal(result["dkv"], torch.zeros_like(result["dkv"]))
        assert torch.equal(result["d_sink"], torch.zeros_like(result["d_sink"]))

    _assert_prepared_sm100_backend(prepared_calls, expected_backend)


@pytest.mark.L0
@torch_fork_set_rng(seed=786)
@pytest.mark.parametrize(
    "s_q,topk",
    [
        # D576 uses its single-query dSink path for s_q=1. Ordinary D576
        # splits s_q=9 across clusters and uses one pair per query at s_q=129;
        # deterministic execution always uses one pair per query.
        pytest.param(1, 128, id="q1-k128"),
        pytest.param(9, 512, id="q9-k512"),
        pytest.param(129, 128, id="q129-k128"),
    ],
)
@pytest.mark.parametrize("has_topk_length", [False, True], ids=["sentinel-padding", "lengths"])
@pytest.mark.parametrize("head_dim,expected_backend", [(512, "h128_2cta_m64"), (576, "h128_d576_2cta_m64")], ids=["d512", "d576"])
@pytest.mark.parametrize("deterministic", [False, True], ids=["ordinary", "deterministic"])
def test_DSA_sparse_attention_backward_h128_two_cta_handles_sink_limits(head_dim, expected_backend, s_q, topk, has_topk_length, deterministic, monkeypatch):
    """Both H128 two-CTA routes must implement the empty and saturating softmax limits.

    A finite sink such as 2.4e38 overflows to +inf when rescaled by log2(e);
    the folded-LSE guard has to test the rescaled values (see #926 for the
    H16/H32 kernels), otherwise the fold evaluates inf - inf and every dq/dkv
    element becomes NaN.  The dSink reduction must likewise treat an infinite
    sink as owning all (or none) of the probability mass."""
    _require_sm100()
    if head_dim == 512:
        _require_exact_sm100()
    elif torch.cuda.get_device_capability() not in _TWO_CTA_CAPABILITIES:
        pytest.skip("D576 2-CTA requires an SM100-class GPU (compute capability 10.0 or 10.3)")
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    num_heads, head_dim_v = 128, 512
    softmax_scale = 192**-0.5
    two_cta_calls = _spy_d576_two_cta_execute(monkeypatch) if head_dim == 576 else None
    prepared_calls = _spy_prepared_sm100_execute(monkeypatch) if head_dim == 512 else None

    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    kv = torch.randn(1, head_dim, dtype=torch.bfloat16, device=device)
    # With all probability mass on the sink the forward output is zero, so
    # every gradient must be exactly zero as well.
    out = torch.zeros(s_q, num_heads, head_dim_v, dtype=torch.bfloat16, device=device)
    dout = torch.randn_like(out)

    # (sink value, empty row): 2e38 rescales to a finite value, 2.4e38 and +inf
    # saturate the denominator, and -inf with an empty row carries no mass.
    for sink_value, empty_row in ((2e38, False), (2.4e38, False), (math.inf, False), (-math.inf, True)):
        attn_sink = torch.full((num_heads,), sink_value, dtype=torch.float32, device=device)
        if empty_row:
            topk_idxs = torch.full((s_q, topk), torch.iinfo(torch.int32).max, dtype=torch.int32, device=device)
            topk_length = torch.zeros(s_q, dtype=torch.int32, device=device)
            lse = torch.full((s_q, num_heads), -math.inf, dtype=torch.float32, device=device)
        else:
            topk_idxs = torch.full((s_q, topk), -1, dtype=torch.int32, device=device)
            topk_idxs[:, 0] = 0
            topk_length = torch.ones(s_q, dtype=torch.int32, device=device) if has_topk_length else None
            lse = ((q.float() * kv[0].float()).sum(dim=-1) * softmax_scale).contiguous()

        result = DSA.sparse_attention_backward_wrapper(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            softmax_scale=softmax_scale,
            topk_length=topk_length,
            deterministic=deterministic,
        )
        torch.cuda.synchronize()

        for name in ("dq", "dkv", "d_sink"):
            assert torch.equal(result[name], torch.zeros_like(result[name])), f"{name} must be zero for sink={sink_value}"

    # Prove the route that produced these results, not just the selector.
    if head_dim == 576:
        assert two_cta_calls, "the wrapper did not execute the H128/D576 two-CTA backend"
    else:
        _assert_prepared_sm100_backend(prepared_calls, expected_backend)


@pytest.mark.L0
@pytest.mark.parametrize("head_dim", [512, 576], ids=["d512", "d576"])
@pytest.mark.parametrize("deterministic", [False, True], ids=["ordinary", "deterministic"])
@pytest.mark.parametrize(
    "dout_value,softmax_scale",
    [
        pytest.param(1.0, None, id="unit-dout"),
        pytest.param(float(2**60), None, id="large-dout"),
        pytest.param(float(2**118), 4.0, id="large-dout-positive-scale"),
    ],
)
def test_DSA_sparse_attention_backward_h128_two_cta_one_key_padded_gradients(head_dim, deterministic, dout_value, softmax_scale, monkeypatch):
    """Padded slots must not poison an exactly solvable one-key backward pass.

    With no sink and one selected key, P=1 and O=V, so dS, dQ, and dSink
    are zero. All 128 heads and four queries contribute dO to the same dV.
    The custom-scale case keeps the correct dV=2**127 finite in BF16, while
    an unmasked padded slot's dS can overflow before its BF16 conversion.
    """
    _require_sm100()
    if torch.cuda.get_device_capability() not in _TWO_CTA_CAPABILITIES:
        pytest.skip("H128 two-CTA requires compute capability 10.0 or 10.3")
    try:
        from cuda.bindings import driver as cuda
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    s_q, num_heads, head_dim_v, topk = 4, 128, 512, 128
    device = torch.device("cuda")
    scale = head_dim**-0.5 if softmax_scale is None else softmax_scale
    q = torch.full((s_q, num_heads, head_dim), 16.0, dtype=torch.bfloat16, device=device)
    kv = torch.full((1, head_dim), -1.0, dtype=torch.bfloat16, device=device)
    out = torch.full((s_q, num_heads, head_dim_v), -1.0, dtype=torch.bfloat16, device=device)
    dout = torch.full_like(out, dout_value)
    lse = torch.full((s_q, num_heads), -16.0 * head_dim * scale, dtype=torch.float32, device=device)
    attn_sink = torch.full((num_heads,), -math.inf, dtype=torch.float32, device=device)
    topk_idxs = torch.full((s_q, topk), -1, dtype=torch.int32, device=device)
    topk_idxs[:, 0] = 0
    expected = (torch.zeros_like(q), torch.zeros_like(kv), torch.zeros_like(attn_sink))
    expected[1][:, :head_dim_v] = num_heads * s_q * dout_value
    assert torch.isfinite(expected[1]).all(), "the analytic dKV must be representable in BF16"

    plan = DSA.SparseAttentionBackward(q, kv, out, dout, lse, attn_sink, topk_idxs, softmax_scale=scale, deterministic=deterministic)
    assert plan.check_support()
    expected_backend = "h128_2cta_m64" if head_dim == 512 else "h128_d576_2cta_m64"
    assert plan._backend == expected_backend
    plan.compile()
    workspace = torch.empty(plan.scratch_workspace_bytes(), dtype=torch.uint8, device=device)
    outputs = (torch.empty_like(q), torch.empty_like(kv), torch.empty_like(attn_sink))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    two_cta_calls = _spy_d576_two_cta_execute(monkeypatch) if head_dim == 576 else None
    prepared_calls = _spy_prepared_sm100_execute(monkeypatch) if head_dim == 512 else None

    def check_gradients(actual, path):
        assert len(actual) == 3
        for name, value, reference in zip(("dq", "dkv", "d_sink"), actual, expected, strict=True):
            torch.testing.assert_close(value, reference, atol=0, rtol=0, msg=f"{path} {name}: exact one-key gradient")

    workspace.fill_(0xA5)
    for output in outputs:
        output.fill_(math.nan)
    result = DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale=scale,
        deterministic=deterministic,
        dq=outputs[0],
        dkv=outputs[1],
        stream=stream,
        workspace=workspace,
    )
    assert len(result) == 3
    check_gradients(tuple(result[name] for name in ("dq", "dkv", "d_sink")), "wrapper")

    workspace.fill_(0xA5)
    for output in outputs:
        output.fill_(math.nan)
    result = plan.execute(q, kv, out, dout, lse, attn_sink, topk_idxs, outputs[0], outputs[1], current_stream=stream, workspace=workspace, d_sink=outputs[2])
    assert all(actual is original for actual, original in zip(result, outputs, strict=True))
    check_gradients(result, "plan")
    if head_dim == 576:
        assert len(two_cta_calls) == 2, "both wrapper and plan must execute the H128/D576 two-CTA backend"
    else:
        _assert_prepared_sm100_backend(prepared_calls, expected_backend)


@pytest.mark.L0
@torch_fork_set_rng(seed=676)
@pytest.mark.parametrize("head_dim,expected_backend", [(512, "h128_2cta_m64"), (576, "h128_d576_2cta_m64")], ids=["d512", "d576"])
@pytest.mark.parametrize("deterministic", [False, True], ids=["ordinary", "deterministic"])
def test_DSA_sparse_attention_backward_h128_two_cta_padded_slots_under_very_negative_lse(head_dim, expected_backend, deterministic, monkeypatch):
    """Padded slots must not overflow when every valid logit and the sink are far below zero.

    The two-CTA kernels zero-fill K and V for invalid slots, so those scores are
    exactly zero and exp2(0 - LSE) overflows once the folded LSE is below -128 in
    log2 units; the resulting infinite dS would poison dQ through inf * 0.  The
    generic kernel masks these slots (#676); both two-CTA routes must stay finite
    and must not touch unselected KV rows."""
    _require_sm100()
    if head_dim == 512:
        _require_exact_sm100()
    elif torch.cuda.get_device_capability() not in _TWO_CTA_CAPABILITIES:
        pytest.skip("D576 2-CTA requires an SM100-class GPU (compute capability 10.0 or 10.3)")
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    s_q, s_kv, topk, num_heads, valid_slots = 4, 300, 128, 128, 3
    softmax_scale = head_dim**-0.5
    two_cta_calls = _spy_d576_two_cta_execute(monkeypatch) if head_dim == 576 else None
    prepared_calls = _spy_prepared_sm100_execute(monkeypatch) if head_dim == 512 else None

    # Every logit is about -10 * head_dim * softmax_scale (below -200 nats) and the sink is disabled.
    q = torch.full((s_q, num_heads, head_dim), 10.0, dtype=torch.bfloat16, device=device)
    kv = -torch.ones(s_kv, head_dim, dtype=torch.bfloat16, device=device) + torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device) / 100
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    topk_idxs[:, valid_slots:] = -1
    attn_sink = torch.full((num_heads,), -math.inf, dtype=torch.float32, device=device)
    out, lse = ref_sparse_attention_forward_chunked(q, kv, attn_sink, topk_idxs, softmax_scale=softmax_scale)
    assert lse.max().item() < -88.0, "the test relies on exp2(-lse * log2(e)) overflowing FP32"
    dout = torch.randn_like(out)

    result = DSA.sparse_attention_backward_wrapper(q, kv, out, dout, lse, attn_sink, topk_idxs, softmax_scale=softmax_scale, deterministic=deterministic)
    torch.cuda.synchronize()

    for name in ("dq", "dkv", "d_sink"):
        assert torch.isfinite(result[name]).all(), f"{name} must stay finite with padded slots under a very negative LSE"
    unselected = torch.ones(s_kv, dtype=torch.bool, device=device)
    unselected[topk_idxs[:, :valid_slots].flatten().long()] = False
    assert torch.equal(result["dkv"][unselected], torch.zeros_like(result["dkv"][unselected]))
    assert (result["dq"] != 0).any()
    if head_dim == 576:
        assert two_cta_calls, "the wrapper did not execute the H128/D576 two-CTA backend"
    else:
        _assert_prepared_sm100_backend(prepared_calls, expected_backend)


@pytest.mark.L0
@torch_fork_set_rng(seed=678)
@pytest.mark.parametrize("num_heads,expected_backend", [(16, "h16_m128"), (32, "h32_m64"), (64, "generic_m64")])
def test_DSA_sparse_attention_backward_sm100_accumulates_odo_in_fp32(num_heads: int, expected_backend: str, monkeypatch) -> None:
    """dSink uses the saved BF16 O but must multiply O*dO in FP32."""
    _require_sm100()
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    s_q, s_kv, head_dim = 8, 64, 576
    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device)
    attn_sink = torch.linspace(2.0, 5.0, num_heads, dtype=torch.float32, device=device)
    topk_idxs = torch.arange(s_kv, dtype=torch.int32, device=device).expand(s_q, -1).contiguous()
    out, lse = ref_sparse_attention_forward_chunked(q, kv, attn_sink, topk_idxs, softmax_scale=192**-0.5)
    dout = torch.randn_like(out)
    prepared_calls = _spy_prepared_sm100_execute(monkeypatch)

    result = DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale=192**-0.5,
    )
    p_sink = torch.exp(attn_sink.view(1, -1) - torch.logaddexp(lse, attn_sink.view(1, -1)))
    expected_d_sink = (-p_sink * (out.float() * dout.float()).sum(dim=-1)).sum(dim=0)
    torch.testing.assert_close(result["d_sink"], expected_d_sink, atol=2e-5, rtol=2e-3)
    _assert_prepared_sm100_backend(prepared_calls, expected_backend)


@pytest.mark.L0
@pytest.mark.parametrize(
    "dtype,kv_value,dout_value",
    [
        pytest.param(torch.bfloat16, 14.8125, 0.2255859375, id="bf16-rounding"),
        pytest.param(torch.float16, 10.0, 1000.0, id="fp16-intermediate-overflow"),
    ],
)
def test_DSA_sparse_attention_backward_sm90_scales_ds_before_conversion(dtype, kv_value, dout_value):
    """Scale dS in FP32 before narrowing it for the Hopper MMA operands."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm90()
    device = torch.device("cuda")
    s_q, s_kv, num_heads, head_dim, head_dim_v = 1, 64, 32, 576, 512
    softmax_scale = 1.0 / math.sqrt(head_dim)

    # Q is nonzero only where KV is zero, so scores remain zero and P is
    # exactly uniform.  The Q tail makes dKV[:, head_dim_v:] a pure dK term.
    q = torch.zeros(s_q, num_heads, head_dim, dtype=dtype, device=device)
    q[..., head_dim_v:] = 0.25
    kv = torch.zeros(s_kv, head_dim, dtype=dtype, device=device)
    kv[0, :head_dim_v] = kv_value
    attn_sink = torch.full((num_heads,), -math.inf, dtype=torch.float32, device=device)
    topk_idxs = torch.arange(s_kv, dtype=torch.int32, device=device).view(1, -1)
    out, lse = ref_sparse_attention_forward_chunked(q, kv, attn_sink, topk_idxs, softmax_scale=softmax_scale)
    dout = torch.full_like(out, dout_value)

    result = DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale=softmax_scale,
    )
    torch.cuda.synchronize()

    # Mirror the intended cast graph using scalar values that are exact in
    # FP32.  The BF16 case moves by one output ULP if dS is narrowed before
    # scaling; the FP16 case overflows that premature conversion to infinity.
    kv_scalar = torch.tensor(kv_value, dtype=dtype)
    dout_scalar = torch.tensor(dout_value, dtype=dtype)
    out_scalar = (kv_scalar.float() / s_kv).to(dtype)
    dp = head_dim_v * dout_scalar.float() * kv_scalar.float()
    delta = head_dim_v * out_scalar.float() * dout_scalar.float()
    ds_unscaled = (dp - delta) / s_kv
    ds_scaled = (ds_unscaled * softmax_scale).to(dtype)
    expected_dq_value = (ds_scaled.float() * kv_scalar.float()).to(dtype).item()
    prematurely_narrowed_ds = (ds_unscaled.to(dtype).float() * softmax_scale).to(dtype)
    old_dq_value = (prematurely_narrowed_ds.float() * kv_scalar.float()).to(dtype).item()
    assert expected_dq_value != old_dq_value

    q_tail_scalar = torch.tensor(0.25, dtype=dtype)
    expected_dkv_tail_value = (ds_scaled.float() * q_tail_scalar.float() * num_heads).to(dtype).item()
    old_dkv_tail_value = (prematurely_narrowed_ds.float() * q_tail_scalar.float() * num_heads).to(dtype).item()
    assert expected_dkv_tail_value != old_dkv_tail_value

    expected_dq = torch.zeros_like(q)
    expected_dq[:, :, :head_dim_v] = expected_dq_value
    assert torch.equal(result["dkv"][0, head_dim_v:], torch.full_like(result["dkv"][0, head_dim_v:], expected_dkv_tail_value))
    assert torch.equal(result["dq"], expected_dq)
    assert torch.isfinite(result["dkv"]).all()
    assert torch.isfinite(result["d_sink"]).all()


@pytest.mark.L0
@torch_fork_set_rng(seed=878)
def test_DSA_sparse_attention_backward_sm90_rejects_a_workspace_on_another_device(monkeypatch):
    """R2: a CPU (or other-device) caller workspace is rejected before its first consumer."""
    try:
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm90
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm90()
    device = torch.device("cuda")
    head_dim, num_heads, s_q, s_kv, topk = 576, 32, 4, 256, 128
    softmax_scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(s_q, num_heads, head_dim, device=device).to(torch.bfloat16)
    kv = torch.randn(s_kv, head_dim, device=device).to(torch.bfloat16)
    attn_sink = torch.zeros((num_heads,), dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    out, lse = ref_sparse_attention_forward_chunked(q, kv, attn_sink, topk_idxs, softmax_scale=softmax_scale)
    dout = torch.randn_like(out)
    nbytes = _interface_sm90.flash_attn_bwd_sm90_workspace_size(s_q, s_kv, head_dim, num_heads)
    monkeypatch.setattr(_interface_sm90, "memset_zero_async", lambda *args: pytest.fail("a CPU workspace reached its first consumer"))
    with pytest.raises(ValueError, match="workspace must be on q's device"):
        DSA.sparse_attention_backward_wrapper(
            q, kv, out, dout, lse, attn_sink, topk_idxs, softmax_scale=softmax_scale, workspace=torch.empty(nbytes, dtype=torch.uint8)
        )


@pytest.mark.L0
@torch_fork_set_rng(seed=436)
@pytest.mark.parametrize("compact", [True, False], ids=["compact", "non-compact"])
def test_DSA_sparse_attention_backward_sm90_padded_topk_columns_contribute_zero(compact):
    """Padded top-k columns must carry exactly zero probability."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm90()
    device = torch.device("cuda")
    head_dim, num_heads = 576, 32
    s_q, s_kv, topk = 4, 256, 128
    n_valid = 100  # not a multiple of the 64-row KV tile
    softmax_scale = 1.0 / math.sqrt(head_dim)

    # Opposing Q/K signs drive the KV-only LSE below the FP32 exp2 threshold.
    q = (2.2 + torch.randn(s_q, num_heads, head_dim, device=device) * 0.01).to(torch.bfloat16)
    kv = (-2.2 + torch.randn(s_kv, head_dim, device=device) * 0.01).to(torch.bfloat16)
    attn_sink = torch.full((num_heads,), -400.0, dtype=torch.float32, device=device)

    max_topk = topk if compact else n_valid + 18
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:max_topk] for _ in range(s_q)]).to(torch.int32)
    topk_length = None
    if compact:
        topk_length = torch.full((s_q,), n_valid, dtype=torch.int32, device=device)
    else:
        topk_idxs[:, n_valid:] = -1
        topk_idxs[:, 3] = -1
        topk_idxs[:, 70] = -1

    out, lse = ref_sparse_attention_forward_chunked(
        q,
        kv,
        attn_sink,
        topk_idxs,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
    )
    # Ensure a zero-filled padded lane would overflow exp2(-LSE_log2)
    # without the probability mask.
    assert (-lse.max() * math.log2(math.e)).item() > 132
    dout = torch.randn_like(out)

    result = DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale=softmax_scale,
        topk_length=topk_length,
    )
    torch.cuda.synchronize()

    dq, dkv, d_sink = result["dq"], result["dkv"], result["d_sink"]
    assert torch.isfinite(dq).all()

    check_ref_dsa_sparse_attention_backward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        out,
        dout,
        lse,
        dq,
        dkv,
        d_sink,
        softmax_scale=softmax_scale,
        topk_length=topk_length,
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=877)
@pytest.mark.parametrize(
    "invalid_value,topk_length_value",
    [
        pytest.param(-1, None, id="noncompact-negative"),
        pytest.param(1, None, id="noncompact-positive-oob"),
        pytest.param(torch.iinfo(torch.int32).max, 1, id="garbage-past-length"),
        pytest.param(-1, 64, id="compact-negative"),
        pytest.param(1, 64, id="compact-positive-oob"),
        pytest.param(1, 65, id="oversized-length"),
        pytest.param(None, None, id="noncompact-mixed-two-tile"),
        pytest.param(None, 128, id="compact-mixed-two-tile"),
    ],
)
def test_DSA_sparse_attention_backward_sm90_masks_invalid_topk_rows(invalid_value, topk_length_value):
    """SM90 must reject invalid sparse rows before KV access, probability use, and dKV scatter."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm90()
    device = torch.device("cuda")
    num_heads, head_dim, head_dim_v = 32, 576, 512
    topk_width = 128 if invalid_value is None else 64
    q = torch.full((1, num_heads, head_dim), 8.0, dtype=torch.bfloat16, device=device)

    # Put readable guard rows on both sides of the logical one-row KV view.
    # An unfixed gather can then fail numerically without poisoning the CUDA
    # context through a wild pointer. dKV accumulation is rounded to 64 rows,
    # so the positive guard index is safe there as well.
    kv_storage = torch.zeros(3, head_dim, dtype=torch.bfloat16, device=device)
    kv_storage[1].fill_(-8.0)
    kv = kv_storage[1:2]

    # Likewise retain one full guard tile behind the logical top-k view. The
    # oversized-length case reaches only its first row in the peeled tile.
    topk_storage = torch.full((1, 2 * topk_width), -1 if invalid_value is None else invalid_value, dtype=torch.int32, device=device)
    topk_storage[:, 0] = 0
    topk_idxs = topk_storage[:, :topk_width]
    topk_length = None if topk_length_value is None else torch.full((1,), topk_length_value, dtype=torch.int32, device=device)

    attn_sink = torch.full((num_heads,), -math.inf, dtype=torch.float32, device=device)
    dout = torch.randn(1, num_heads, head_dim_v, dtype=torch.bfloat16, device=device)
    out = kv[0, :head_dim_v].view(1, 1, head_dim_v).expand(1, num_heads, head_dim_v).contiguous()
    lse = ((q.float() * kv[0].float()).sum(dim=-1) * (192**-0.5)).contiguous()
    if invalid_value is None:
        topk_idxs.zero_()
        invalid_positions = torch.tensor([idx for idx in range(topk_width) if idx % 3 != 0], dtype=torch.int64, device=device)
        topk_idxs[:, invalid_positions[::2]] = -1
        topk_idxs[:, invalid_positions[1::2]] = 1
        lse += math.log(topk_width - invalid_positions.numel())

    result = DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale=192**-0.5,
        topk_length=topk_length,
    )
    torch.cuda.synchronize()

    expected_dkv = torch.zeros_like(kv)
    expected_dkv[0, :head_dim_v] = dout.float().sum(dim=(0, 1)).to(torch.bfloat16)
    assert torch.isfinite(result["dq"]).all()
    assert torch.isfinite(result["dkv"]).all()
    assert torch.isfinite(result["d_sink"]).all()
    torch.testing.assert_close(result["dq"].float(), torch.zeros_like(result["dq"], dtype=torch.float32), atol=3e-5, rtol=0)
    torch.testing.assert_close(result["dkv"], expected_dkv, atol=5e-2, rtol=1e-2)
    assert torch.equal(result["d_sink"], torch.zeros_like(result["d_sink"]))


@pytest.mark.L0
@torch_fork_set_rng(seed=437)
@pytest.mark.parametrize(
    "sink_value,empty_row",
    [
        pytest.param(-math.inf, True, id="empty-no-mass"),
        pytest.param(2.4e38, False, id="finite-but-rescale-overflows"),
        pytest.param(float("inf"), False, id="pos-inf"),
    ],
)
def test_DSA_sparse_attention_backward_sm90_saturating_attn_sink(sink_value, empty_row):
    """A saturating attention sink must not turn the gradients into NaN."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm90()
    device = torch.device("cuda")
    head_dim, num_heads, s_q, s_kv, topk = 576, 32, 4, 256, 128
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = (torch.randn(s_q, num_heads, head_dim, device=device) / 10).to(torch.bfloat16)
    kv = (torch.randn(s_kv, head_dim, device=device) / 10).to(torch.bfloat16)
    attn_sink = torch.full((num_heads,), sink_value, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    topk_length = torch.zeros(s_q, dtype=torch.int32, device=device) if empty_row else torch.full((s_q,), topk, dtype=torch.int32, device=device)
    if empty_row:
        topk_idxs.fill_(torch.iinfo(torch.int32).max)
        out = torch.zeros(s_q, num_heads, 512, dtype=torch.bfloat16, device=device)
        lse = torch.full((s_q, num_heads), -math.inf, dtype=torch.float32, device=device)
    else:
        out, lse = ref_sparse_attention_forward_chunked(
            q,
            kv,
            attn_sink,
            topk_idxs,
            topk_length=topk_length,
            softmax_scale=softmax_scale,
        )
    dout = torch.randn_like(out)

    result = DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale=softmax_scale,
        topk_length=topk_length,
    )
    torch.cuda.synchronize()

    dq, dkv, d_sink = result["dq"], result["dkv"], result["d_sink"]
    assert torch.equal(d_sink, torch.zeros_like(d_sink))
    assert torch.equal(out, torch.zeros_like(out))
    assert torch.equal(dq, torch.zeros_like(dq))
    assert torch.equal(dkv, torch.zeros_like(dkv))


@pytest.mark.L0
@torch_fork_set_rng(seed=434)
@pytest.mark.parametrize(
    "num_heads,s_kv,topk",
    [
        pytest.param(16, 640, 513, id="h16-m128"),
        pytest.param(32, 4096, 2047, id="h32-m64"),
    ],
)
def test_DSA_sparse_attention_backward_sm100_partial_max_topk(num_heads, s_kv, topk):
    """Dedicated SM100 kernels handle a maximum top-k ending in a partial tile."""
    if not torch.cuda.is_available():
        pytest.skip("SM100 GPU required")
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        pytest.skip("dedicated H16/H32 regression test targets the SM100 kernel")

    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    s_q = 2
    head_dim = 576
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device) / 10
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device) / 10
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)

    out, lse = ref_sparse_attention_forward_chunked(
        q,
        kv,
        attn_sink,
        topk_idxs,
        softmax_scale=softmax_scale,
    )
    dout = torch.randn_like(out)
    result = DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale=softmax_scale,
    )
    torch.cuda.synchronize()

    check_ref_dsa_sparse_attention_backward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        out,
        dout,
        lse,
        result["dq"],
        result["dkv"],
        result["d_sink"],
        softmax_scale=softmax_scale,
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=385)
def test_DSA_sparse_attention_backward_check_support_declines_strided_inputs():
    """A strided input is declined in check_support with the tensor named (R5); execute never repacks it."""
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (9, 10):
        pytest.skip("SM90 or SM100 GPU required")
    device = torch.device("cuda")
    s_q, s_kv, num_heads = 2, 128, 32
    head_dim, head_dim_v, topk = 576, 512, 64
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device)
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    topk_length = torch.full((s_q,), topk, dtype=torch.int32, device=device)
    out, lse = ref_sparse_attention_forward_chunked(q, kv, attn_sink, topk_idxs, softmax_scale=softmax_scale)

    # The head-major backing of the original #385 reproduction: the eager wrapper
    # normalises it on the launch stream (develop's behaviour) and matches the
    # contiguous call; only the plan class declines it (below).
    out_head_major = out.transpose(0, 1).contiguous().transpose(0, 1)
    dout_head_major = torch.randn(num_heads, s_q, head_dim_v, dtype=torch.bfloat16, device=device).transpose(0, 1)
    assert out_head_major.stride() == dout_head_major.stride() == (head_dim_v, s_q * head_dim_v, 1)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    strided_res = DSA.sparse_attention_backward_wrapper(
        q, kv, out_head_major, dout_head_major, lse, attn_sink, topk_idxs, softmax_scale=softmax_scale, stream=stream
    )
    dense_res = DSA.sparse_attention_backward_wrapper(
        q, kv, out_head_major.contiguous(), dout_head_major.contiguous(), lse, attn_sink, topk_idxs, softmax_scale=softmax_scale, stream=stream
    )
    torch.cuda.synchronize()
    for name in ("dq", "dkv", "d_sink"):
        torch.testing.assert_close(strided_res[name], dense_res[name], atol=1e-2, rtol=1e-2)

    def strided_like(tensor):
        base = torch.zeros(*tensor.shape[:-1], 2 * tensor.shape[-1], dtype=tensor.dtype, device=tensor.device)
        base[..., ::2] = tensor
        assert not base[..., ::2].is_contiguous()
        return base[..., ::2]

    good = dict(
        sample_q=q,
        sample_kv=kv,
        sample_out=out,
        sample_dout=torch.randn_like(out),
        sample_lse=lse,
        sample_attn_sink=attn_sink,
        sample_topk_idxs=topk_idxs,
        sample_topk_length=topk_length,
        softmax_scale=softmax_scale,
    )
    assert DSA.SparseAttentionBackward(**good).check_support()
    for name in ("q", "kv", "out", "dout", "lse", "attn_sink", "topk_idxs", "topk_length"):
        kwargs = dict(good)
        kwargs[f"sample_{name}"] = strided_like(good[f"sample_{name}"])
        with pytest.raises(NotImplementedError, match=rf"contiguous {name}; got shape .* strides"):
            DSA.SparseAttentionBackward(**kwargs).check_support()


@pytest.mark.L0
@torch_fork_set_rng(seed=385)
def test_DSA_sparse_attention_backward_qh32_uses_per_query_topk_without_padding(monkeypatch):
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm90
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm90()
    device = torch.device("cuda")
    s_q, s_kv, num_heads = 2, 128, 32
    head_dim, head_dim_v, topk = 576, 512, 64
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device)
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    # The old packed-M mapping used row 0 for both queries in one CTA. Make
    # the two rows disjoint so sharing a top-k row is unambiguously incorrect.
    topk_idxs = torch.stack(
        (
            torch.arange(0, topk, dtype=torch.int32, device=device),
            torch.arange(topk, 2 * topk, dtype=torch.int32, device=device),
        )
    )

    out, lse = ref_sparse_attention_forward_chunked(
        q,
        kv,
        attn_sink,
        topk_idxs,
        softmax_scale=softmax_scale,
    )
    dout = torch.randn_like(out)

    def fail_on_pad(*args, **kwargs):
        pytest.fail("qh32 must be handled by the SM90 main kernel without torch padding")

    monkeypatch.setattr(torch.nn.functional, "pad", fail_on_pad)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    result = DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale=softmax_scale,
        stream=stream,
    )
    torch.cuda.synchronize()

    # qhpkv controls the query-head tiling and must remain part of the main
    # compile key, independently of the runtime tensor shape.
    assert any(key[1:4] == (head_dim, head_dim_v, num_heads) for key in _interface_sm90.flash_attn_bwd_sm90.compile_cache)

    dq, dkv, d_sink = result["dq"], result["dkv"], result["d_sink"]
    assert torch.isfinite(dq).all()
    assert torch.isfinite(dkv).all()
    check_ref_dsa_sparse_attention_backward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        out,
        dout,
        lse,
        dq,
        dkv,
        d_sink,
        softmax_scale=softmax_scale,
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_DSA_sparse_attention_backward_staged_store():
    """Regression test for the staged P/dS store paths of the SM100 kernel.

    The compute->MMA smem handoff for P and dS must stay correct when
    compute_mma_P_stage / compute_mma_dS_stage are raised above 1: the
    store-side layouts and the per-iteration store slot have to follow the
    producer pipeline state. This test deepens both pipelines to 2 and
    requires dq to match the stage-1 baseline bitwise (dq is written by TMA
    with a single writer per element, hence deterministic; dkv/d_sink are
    fp32 atomic reductions, so they are only gated on NaN and relative
    parity).
    """
    try:
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import (
            api as _dsa_bwd_api,
            _interface_sm100 as _dsa_bwd_iface,
            dsa_bwd_sm100 as _dsa_bwd_kmod,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        pytest.skip("staged-store regression test targets the SM100 kernel")

    s_q = s_kv = 512
    h, d, topk = 64, 512, 256  # topk/64 = 4 tiles -> the pipelines really cycle
    device = "cuda"
    q = torch.randn(s_q, h, d, dtype=torch.bfloat16, device=device) / 10
    kv = torch.randn(s_kv, d, dtype=torch.bfloat16, device=device) / 10
    attn_sink = torch.randn(h, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    topk_length = torch.full((s_q,), topk, dtype=torch.int32, device=device)
    softmax_scale = 1.0 / math.sqrt(d)

    out, lse = ref_sparse_attention_forward_chunked(q, kv, attn_sink, topk_idxs, topk_length=topk_length, softmax_scale=softmax_scale)
    dout = torch.randn_like(out)

    orig_setup = _dsa_bwd_kmod.FlashAttentionDSABackwardSm100._setup_attributes

    def run(stage_overrides):
        def patched(self):
            orig_setup(self)
            for name, value in stage_overrides.items():
                setattr(self, name, value)

        _dsa_bwd_kmod.FlashAttentionDSABackwardSm100._setup_attributes = patched
        _dsa_bwd_iface._compile_nondeterministic_sm100.cache_clear()
        _dsa_bwd_api._cache_of_SparseAttentionBackwardObjects.clear()
        try:
            result = DSA.sparse_attention_backward_wrapper(
                q,
                kv,
                out,
                dout,
                lse,
                attn_sink,
                topk_idxs,
                softmax_scale=softmax_scale,
                topk_length=topk_length,
            )
            torch.cuda.synchronize()
            return result["dq"], result["dkv"], result["d_sink"]
        finally:
            _dsa_bwd_kmod.FlashAttentionDSABackwardSm100._setup_attributes = orig_setup
            _dsa_bwd_iface._compile_nondeterministic_sm100.cache_clear()
            _dsa_bwd_api._cache_of_SparseAttentionBackwardObjects.clear()

    dq_ref, dkv_ref, d_sink_ref = run({})
    assert not torch.isnan(dq_ref).any(), "stage-1 baseline produced NaN"

    dq, dkv, d_sink = run({"compute_mma_P_stage": 2, "compute_mma_dS_stage": 2})

    assert not torch.isnan(dq).any() and not torch.isnan(dkv).any() and not torch.isnan(d_sink).any(), "staged store paths produced NaN gradients"
    assert torch.equal(dq, dq_ref), "dq must be bitwise-identical between stage 1 and stage 2"

    def rel_l2(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()

    assert rel_l2(dkv, dkv_ref) < 1e-4, "dkv parity vs stage-1 baseline"
    assert rel_l2(d_sink, d_sink_ref) < 1e-4, "d_sink parity vs stage-1 baseline"


@pytest.mark.L0
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize(
    "num_heads,head_dim,topk,q_cluster_mode",
    [
        (64, 512, 64, "off"),
        (96, 576, 1024, "off"),
        (128, 512, 128, "off"),
        (128, 576, 128, "off"),
        (16, 576, 512, "adaptive_pair"),
        (32, 576, 512, "adaptive_pair"),
    ],
    ids=["generic", "h96-composite", "h128-two-cta", "h128-d576-two-cta", "qcluster-h16", "qcluster-h32"],
)
@pytest.mark.parametrize("has_topk_length", [False, True], ids=["full-topk", "lengths"])
@torch_fork_set_rng(seed=7)
def test_DSA_sparse_attention_backward_nondefault_stream_zero_init_ordering(num_heads, head_dim, topk, q_cluster_mode, has_topk_length):
    """The SM100 interface must establish zero state on the launch stream.

    The generic and H96 composite paths clear the FP32 dKV accumulator
    in-kernel and zero ``d_sink`` with one stream-ordered memset (R4), while the
    H128 two-CTA paths (D512 and D576) launch their compiled ``zero_init``
    kernels. Q-cluster clears ``d_sink`` in its transform and initializes dKV
    in its compiled prelude. All initialization must precede backward on the
    caller-provided ``current_stream``.

    The ambient default stream is parked on ``torch.cuda._sleep`` so the
    unordered interleaving is reached reliably (the zero-fills cannot start
    until the sleep retires, while the side-stream kernel is free to run);
    the test needs the GPU to itself for that reason."""
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm100()
    if num_heads == 128 and head_dim == 512:
        _require_exact_sm100()
    if num_heads == 128 and head_dim == 576 and torch.cuda.get_device_capability() not in _TWO_CTA_CAPABILITIES:
        pytest.skip("D576 2-CTA requires an SM100-class GPU (compute capability 10.0 or 10.3)")
    if q_cluster_mode == "adaptive_pair" and torch.cuda.get_device_capability() not in _TWO_CTA_CAPABILITIES:
        pytest.skip("q_cluster_mode='adaptive_pair' requires an SM100-class GPU (compute capability 10.0 or 10.3)")
    device = torch.device("cuda")
    s_q, s_kv = 256, 1024
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device) / 10
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device) / 10
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    topk_length = torch.full((s_q,), topk, dtype=torch.int32, device=device) if has_topk_length else None

    out, lse = ref_sparse_attention_forward_chunked(
        q,
        kv,
        attn_sink,
        topk_idxs,
        softmax_scale=softmax_scale,
    )
    dout = torch.randn_like(out)

    def run(stream):
        result = DSA.sparse_attention_backward_wrapper(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            softmax_scale=softmax_scale,
            topk_length=topk_length,
            stream=stream,
            q_cluster_mode=q_cluster_mode,
        )
        torch.cuda.synchronize()
        return result["dq"], result["dkv"], result["d_sink"]

    # Control on the ambient (default) stream; also primes the compile cache
    # so the raced call below is a pure execute.
    default_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    dq_ref, dkv_ref, d_sink_ref = run(default_stream)
    assert (dkv_ref != 0).any(), "control dkv must have nonzero content"
    assert (d_sink_ref != 0).any(), "control d_sink must have nonzero content"

    # Park the ambient default stream, then launch on a side stream. The
    # interface's zero-fills must be ordered with the side-stream kernel, not
    # queued behind the sleep on the default stream.
    side_stream = torch.cuda.Stream()
    torch.cuda._sleep(2_000_000_000)
    if topk_length is not None:
        with torch.cuda.stream(side_stream):
            topk_length.fill_(0)
            topk_length.fill_(topk)
    dq, dkv, d_sink = run(cuda.CUstream(side_stream.cuda_stream))

    assert (dkv != 0).any(), "dkv accumulation was wiped by a zero-init racing on another stream"
    assert (d_sink != 0).any(), "d_sink accumulation was wiped by a zero-init racing on another stream"

    def rel_l2(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()

    if q_cluster_mode == "adaptive_pair":
        # The transform may select a different tile-aligned subset of common
        # indices, changing FP32 accumulation order without changing the
        # result. This opt-in path is explicitly non-deterministic.
        assert rel_l2(dq, dq_ref) < 5e-3, "dq parity vs default-stream control"
    else:
        assert torch.equal(dq, dq_ref), "dq must not depend on the launch stream"
    assert rel_l2(dkv, dkv_ref) < 1e-4, "dkv parity vs default-stream control"
    assert rel_l2(d_sink, d_sink_ref) < 1e-4, "d_sink parity vs default-stream control"


@pytest.mark.L0
@pytest.mark.parametrize("head_dim,num_heads", [(512, 64), (576, 16), (576, 32), (576, 96)])
@torch_fork_set_rng(seed=16)
def test_DSA_sparse_attention_backward_fp16_sm100_numerics(head_dim, num_heads):
    """SM100 must compile FP16 inputs with FP16 MMA/storage semantics."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm100()
    device = torch.device("cuda")
    s_q, s_kv, topk = 4, 128, 64
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.float16, device=device)
    kv = torch.randn(s_kv, head_dim, dtype=torch.float16, device=device)
    attn_sink = torch.linspace(-2.0, 2.0, num_heads, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    topk_length = torch.tensor([16, 32, 48, 64], dtype=torch.int32, device=device)
    out, lse = ref_sparse_attention_forward_chunked(
        q,
        kv,
        attn_sink,
        topk_idxs,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
    )
    dout = torch.randn_like(out)
    assert q.dtype == kv.dtype == out.dtype == dout.dtype == torch.float16
    assert lse.dtype == torch.float32

    result = DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale=softmax_scale,
        topk_length=topk_length,
    )
    torch.cuda.synchronize()
    dq, dkv, d_sink = result["dq"], result["dkv"], result["d_sink"]

    assert dq.dtype == dkv.dtype == torch.float16
    assert d_sink.dtype == torch.float32
    assert torch.isfinite(dq).all()
    assert torch.isfinite(dkv).all()
    assert torch.isfinite(d_sink).all()
    check_ref_dsa_sparse_attention_backward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        out,
        dout,
        lse,
        dq,
        dkv,
        d_sink,
        softmax_scale=softmax_scale,
        topk_length=topk_length,
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=23)
def test_DSA_sparse_attention_backward_interface_rejects_noncontiguous_inputs():
    """The interface re-validates contiguity for every input (Rule 1) and raises
    instead of repacking (Rule 2): a strided tensor used to be normalized with
    ``.contiguous()`` (q/kv/out/dout/lse) or escape to the DSL layer as a
    low-level stride error (attn_sink/topk_idxs/topk_length). Neither path may
    reach a compile."""
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm100
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm100()
    device = torch.device("cuda")
    s_q, s_kv, num_heads = 4, 128, 64
    head_dim, head_dim_v, topk = 576, 512, 64

    good = dict(
        q=torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device),
        kv=torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device),
        out=torch.randn(s_q, num_heads, head_dim_v, dtype=torch.bfloat16, device=device),
        dout=torch.randn(s_q, num_heads, head_dim_v, dtype=torch.bfloat16, device=device),
        lse=torch.randn(s_q, num_heads, dtype=torch.float32, device=device),
        attn_sink=torch.randn(num_heads, dtype=torch.float32, device=device),
        topk_idxs=torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32),
        topk_length=torch.randint(1, topk + 1, (s_q,), dtype=torch.int32, device=device),
    )
    outputs = dict(
        dq=torch.empty_like(good["q"]),
        dkv=torch.empty_like(good["kv"]),
        d_sink=torch.empty_like(good["attn_sink"]),
        workspace=torch.empty(_interface_sm100.flash_attn_bwd_sm100_workspace_size(s_q, s_kv, head_dim, num_heads), dtype=torch.uint8, device=device),
    )

    def strided_like(tensor):
        base = torch.zeros(*tensor.shape[:-1], 2 * tensor.shape[-1], dtype=tensor.dtype, device=tensor.device)
        base[..., ::2] = tensor
        assert not base[..., ::2].is_contiguous()
        return base[..., ::2]

    cached_keys = set(_interface_sm100.flash_attn_bwd_sm100.compile_cache)
    for name in good:
        args = dict(good)
        args[name] = strided_like(good[name])
        with pytest.raises(ValueError, match=rf"{name} must be contiguous; got shape .* strides"):
            _interface_sm100.flash_attn_bwd_sm100(
                args["q"],
                args["kv"],
                args["out"],
                args["dout"],
                args["lse"],
                args["attn_sink"],
                args["topk_idxs"],
                softmax_scale=1.0 / math.sqrt(head_dim),
                topk_length=args["topk_length"],
                **outputs,
            )
    assert set(_interface_sm100.flash_attn_bwd_sm100.compile_cache) == cached_keys, "a rejected input must not reach a compile"


@pytest.mark.L0
@torch_fork_set_rng(seed=44)
def test_DSA_sparse_attention_backward_wrapper_accepts_strided_inputs_and_keeps_output_identity():
    """The plan declines strided tensors (R5), the wrapper does not: it normalises every strided input on
    the launch stream (cold and warm plan cache alike), stages a strided caller-provided dq/dkv and copies
    back so the caller's tensors keep their identity, and the results match the contiguous call."""
    try:
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import api as sbwd_api
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm100()
    device = torch.device("cuda")
    s_q, s_kv, num_heads = 4, 128, 64
    head_dim, head_dim_v, topk = 576, 512, 64
    softmax_scale = 1.0 / math.sqrt(head_dim)

    good = dict(
        q=torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device),
        kv=torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device),
        out=torch.randn(s_q, num_heads, head_dim_v, dtype=torch.bfloat16, device=device),
        dout=torch.randn(s_q, num_heads, head_dim_v, dtype=torch.bfloat16, device=device),
        lse=torch.randn(s_q, num_heads, dtype=torch.float32, device=device),
        attn_sink=torch.randn(num_heads, dtype=torch.float32, device=device),
        topk_idxs=torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32),
        topk_length=torch.randint(1, topk + 1, (s_q,), dtype=torch.int32, device=device),
    )

    def strided_like(tensor):
        base = torch.zeros(*tensor.shape[:-1], 2 * tensor.shape[-1], dtype=tensor.dtype, device=tensor.device)
        base[..., ::2] = tensor
        assert not base[..., ::2].is_contiguous()
        return base[..., ::2]

    def run(args, **kw):
        return DSA.sparse_attention_backward_wrapper(
            args["q"],
            args["kv"],
            args["out"],
            args["dout"],
            args["lse"],
            args["attn_sink"],
            args["topk_idxs"],
            softmax_scale=softmax_scale,
            topk_length=args["topk_length"],
            deterministic=True,
            **kw,
        )

    reference = run(good)
    torch.cuda.synchronize()
    plans_before = len(sbwd_api._cache_of_SparseAttentionBackwardObjects)
    strided = {name: strided_like(t) for name, t in good.items()}
    for _ in range(2):  # warm plan cache on the second pass
        dq_user = torch.empty(s_q, num_heads, 2 * head_dim, dtype=torch.bfloat16, device=device)[..., ::2]
        dkv_user = torch.empty(s_kv, 2 * head_dim, dtype=torch.bfloat16, device=device)[:, ::2]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())  # the strided inputs were written on the current stream
        with torch.cuda.stream(side):
            got = run(strided, dq=dq_user, dkv=dkv_user, stream=side.cuda_stream)
        torch.cuda.synchronize()
        assert got["dq"] is dq_user and got["dkv"] is dkv_user, "caller-provided outputs keep their identity"
        for name in ("dq", "dkv", "d_sink"):
            torch.testing.assert_close(got[name].float(), reference[name].float(), atol=2e-2, rtol=2e-2, msg=name)
    assert len(sbwd_api._cache_of_SparseAttentionBackwardObjects) == plans_before, "a strided call reuses the contiguous plan"


@pytest.mark.L0
@torch_fork_set_rng(seed=42)
def test_DSA_sparse_attention_backward_cross_shape_validation():
    """The compiled kernel takes every dimension as a dynamic value derived
    from q, so mis-shaped companion tensors do not fail: a transposed dout or
    lse runs without any error and returns silently corrupted gradients
    (measured rel-L2 vs the correct result: ~1.1 and ~45 respectively).
    The interface must validate the cross-tensor contract up front. The
    caller-provided dq/dkv/d_sink must additionally be contiguous: the compile
    cache is keyed without output strides, so a strided out-parameter would be
    written through the wrong layout."""
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm100
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm100()
    device = torch.device("cuda")
    s_q, s_kv, num_heads = 4, 128, 64
    head_dim, head_dim_v, topk = 576, 512, 64
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device)
    out = torch.randn(s_q, num_heads, head_dim_v, dtype=torch.bfloat16, device=device)
    dout = torch.randn_like(out)
    lse = torch.randn(s_q, num_heads, dtype=torch.float32, device=device)
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    topk_length = torch.full((s_q,), topk, dtype=torch.int32, device=device)

    good = dict(
        q=q,
        kv=kv,
        out=out,
        dout=dout,
        lse=lse,
        attn_sink=attn_sink,
        topk_idxs=topk_idxs,
        topk_length=topk_length,
        dq=torch.empty_like(q),
        dkv=torch.empty_like(kv),
        d_sink=torch.empty_like(attn_sink),
        workspace=torch.empty(
            _interface_sm100.flash_attn_bwd_sm100_workspace_size(s_q, s_kv, head_dim, num_heads),
            dtype=torch.uint8,
            device=device,
        ),
    )

    def call(args):
        """Invoke the interface with one mutated tensor-contract case."""
        _interface_sm100.flash_attn_bwd_sm100(
            args["q"],
            args["kv"],
            args["out"],
            args["dout"],
            args["lse"],
            args["attn_sink"],
            args["topk_idxs"],
            softmax_scale=softmax_scale,
            topk_length=args["topk_length"],
            dq=args["dq"],
            dkv=args["dkv"],
            d_sink=args["d_sink"],
            workspace=args["workspace"],
        )

    shape_cases = {
        "kv": kv[:, : head_dim - 64].contiguous(),
        "out": out.transpose(1, 2).contiguous(),
        "dout": dout.transpose(1, 2).contiguous(),
        "lse": lse.transpose(0, 1).contiguous(),
        "attn_sink": attn_sink[: num_heads // 2].contiguous(),
        "topk_idxs": topk_idxs[: s_q - 1].contiguous(),
        "topk_length": topk_length[: s_q - 1].contiguous(),
    }
    for name, bad_tensor in shape_cases.items():
        args = dict(good)
        args[name] = bad_tensor
        with pytest.raises(AssertionError, match=f"{name} shape mismatch"):
            call(args)

    args = dict(good)
    args["topk_length"] = topk_length.to(torch.int64)
    with pytest.raises(AssertionError, match="topk_length dtype mismatch"):
        call(args)

    # A contiguous view may still begin at a four-byte storage offset. The
    # SM100 loader moves FP32 pairs and must reject that pointer rather than
    # promising an unverified eight-byte alignment to the compiler.
    lse_storage = torch.empty(lse.numel() + 1, dtype=lse.dtype, device=device)
    lse_misaligned = lse_storage[1:].view_as(lse)
    lse_misaligned.copy_(lse)
    assert lse_misaligned.is_contiguous() and lse_misaligned.data_ptr() % 8 == 4
    args = dict(good)
    args["lse"] = lse_misaligned
    with pytest.raises(ValueError, match="lse must be 8-byte aligned"):
        call(args)

    # Caller-provided out-params must be contiguous (they are not copied).
    dq_strided = torch.empty(s_q, num_heads, 2 * head_dim, dtype=q.dtype, device=device)[..., ::2]
    args = dict(good)
    args["dq"] = dq_strided
    with pytest.raises(AssertionError, match="dq must be contiguous"):
        call(args)
    dkv_strided = torch.empty(s_kv, 2 * head_dim, dtype=kv.dtype, device=device)[:, ::2]
    args = dict(good)
    args["dkv"] = dkv_strided
    with pytest.raises(AssertionError, match="dkv must be contiguous"):
        call(args)
    args = dict(good)
    args["d_sink"] = torch.empty(2 * num_heads, dtype=torch.float32, device=device)[::2]
    with pytest.raises(AssertionError, match="d_sink must be contiguous"):
        call(args)


@pytest.mark.L0
@torch_fork_set_rng(seed=51)
def test_DSA_sparse_attention_backward_check_support_validates_contract(monkeypatch):
    """check_support works on metadata-only descriptors and is the advertised
    support gate, so the cross-tensor contract must be enforced there as well,
    not only by the runtime asserts in the execution interface."""
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import SparseAttentionBackward
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm100()
    device = torch.device("cuda")
    s_q, s_kv, num_heads = 4, 128, 64
    head_dim, head_dim_v, topk = 576, 512, 64

    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device)
    out = torch.randn(s_q, num_heads, head_dim_v, dtype=torch.bfloat16, device=device)
    dout = torch.randn_like(out)
    lse = torch.randn(s_q, num_heads, dtype=torch.float32, device=device)
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    topk_length = torch.full((s_q,), topk, dtype=torch.int32, device=device)

    good = dict(
        sample_q=q,
        sample_kv=kv,
        sample_out=out,
        sample_dout=dout,
        sample_lse=lse,
        sample_attn_sink=attn_sink,
        sample_topk_idxs=topk_idxs,
        sample_topk_length=topk_length,
    )
    assert SparseAttentionBackward(**good).check_support()
    assert SparseAttentionBackward(**good, deterministic=True).check_support()

    from cudnn.frost import buffers

    def metadata_case(case_s_q, case_s_kv, case_heads, case_head_dim, case_topk):
        """Build value-free CUDA tensors for support-only route coverage."""
        case_head_dim_v = 512 if case_head_dim == 576 else case_head_dim
        return dict(
            sample_q=torch.empty(case_s_q, case_heads, case_head_dim, dtype=torch.bfloat16, device=device),
            sample_kv=torch.empty(case_s_kv, case_head_dim, dtype=torch.bfloat16, device=device),
            sample_out=torch.empty(case_s_q, case_heads, case_head_dim_v, dtype=torch.bfloat16, device=device),
            sample_dout=torch.empty(case_s_q, case_heads, case_head_dim_v, dtype=torch.bfloat16, device=device),
            sample_lse=torch.empty(case_s_q, case_heads, dtype=torch.float32, device=device),
            sample_attn_sink=torch.empty(case_heads, dtype=torch.float32, device=device),
            sample_topk_idxs=torch.empty(case_s_q, case_topk, dtype=torch.int32, device=device),
            sample_topk_length=torch.empty(case_s_q, dtype=torch.int32, device=device),
        )

    with monkeypatch.context() as old_dsl:
        old_dsl.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.6.2")))
        # CuTe DSL 4.7 is a relay-only requirement. Preserve the repository's
        # 4.6.2 floor for the established short-context and two-CTA paths.
        assert SparseAttentionBackward(**good, deterministic=True).check_support()
        for route in (
            metadata_case(4, 128, 96, 576, 64),  # H96 composite
            metadata_case(4, 128, 128, 512, 128),  # H128 D512 two-CTA
            metadata_case(4, 128, 96, 576, 128),  # H96 D576 two-CTA
            metadata_case(4, 128, 128, 576, 128),  # H128 D576 two-CTA
        ):
            assert SparseAttentionBackward(**route, deterministic=True).check_support()

        relay = metadata_case(257, 65537, 16, 512, 64)
        with pytest.raises(RuntimeError, match=r"deterministic DSA backward cluster relay requires nvidia-cutlass-dsl >= 4\.7\.0; found 4\.6\.2"):
            SparseAttentionBackward(**relay, deterministic=True).check_support()

        # Neither the relay implementation nor its version floor changes the
        # established non-deterministic backend.
        assert SparseAttentionBackward(**good).check_support()

    with monkeypatch.context() as supported_dsl:
        supported_dsl.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.7.0")))
        assert SparseAttentionBackward(**relay, deterministic=True).check_support()

    deterministic_h32 = dict(good)
    deterministic_h32["sample_q"] = q[:, :32].contiguous()
    deterministic_h32["sample_out"] = out[:, :32].contiguous()
    deterministic_h32["sample_dout"] = dout[:, :32].contiguous()
    deterministic_h32["sample_lse"] = lse[:, :32].contiguous()
    deterministic_h32["sample_attn_sink"] = attn_sink[:32].contiguous()
    assert SparseAttentionBackward(**deterministic_h32, deterministic=True).check_support()

    deterministic_h48 = dict(good)
    deterministic_h48["sample_q"] = q[:, :48].contiguous()
    deterministic_h48["sample_out"] = out[:, :48].contiguous()
    deterministic_h48["sample_dout"] = dout[:, :48].contiguous()
    deterministic_h48["sample_lse"] = lse[:, :48].contiguous()
    deterministic_h48["sample_attn_sink"] = attn_sink[:48].contiguous()
    with pytest.raises(ValueError, match="heads in"):
        SparseAttentionBackward(**deterministic_h48, deterministic=True).check_support()

    fp16_good = dict(good)
    for name in ("sample_q", "sample_kv", "sample_out", "sample_dout"):
        fp16_good[name] = fp16_good[name].to(torch.float16)
    assert SparseAttentionBackward(**fp16_good).check_support()

    bad_cases = {
        "sample_kv": kv[:, : head_dim - 64].contiguous(),
        "sample_out": out.transpose(1, 2).contiguous(),
        "sample_dout": dout.transpose(1, 2).contiguous(),
        "sample_lse": lse.transpose(0, 1).contiguous(),
        "sample_attn_sink": attn_sink[: num_heads // 2].contiguous(),
        "sample_topk_idxs": topk_idxs[: s_q - 1].contiguous(),
        "sample_topk_length": topk_length[: s_q - 1].contiguous(),
    }
    for name, bad_tensor in bad_cases.items():
        kwargs = dict(good)
        kwargs[name] = bad_tensor
        with pytest.raises(ValueError):
            SparseAttentionBackward(**kwargs).check_support()

    # Device placement: check_support must reject CPU inputs that the
    # SM90/SM100 runtime would otherwise reject at launch time.
    cpu_kwargs = {name: tensor.to("cpu") for name, tensor in good.items()}
    with pytest.raises(ValueError, match="Q must live on CUDA"):
        SparseAttentionBackward(**cpu_kwargs).check_support()

    # Cross-device: Q stays on CUDA, KV moved to CPU (no new CUDA allocation;
    # reuses the good tensors) -> device-consistency failure.
    cross_kwargs = dict(good)
    cross_kwargs["sample_kv"] = good["sample_kv"].to("cpu")
    with pytest.raises(ValueError, match="must share Q's device"):
        SparseAttentionBackward(**cross_kwargs).check_support()

    # head_dim must be one of {512, 576}: the kernel is tiled only for those.
    bad_head_dim = dict(good)
    bad_head_dim["sample_q"] = torch.randn(s_q, num_heads, 128, dtype=torch.bfloat16, device=device)
    bad_head_dim["sample_kv"] = torch.randn(s_kv, 128, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="head_dim must be 512 or 576"):
        SparseAttentionBackward(**bad_head_dim).check_support()


@pytest.mark.L0
@torch_fork_set_rng(seed=7)
def test_DSA_sparse_attention_backward_rejects_unsupported_head_dim_runtime():
    """The SM100 kernel is tiled only for head_dim in {512, 576}; any other
    head_dim indexes shared memory out of bounds and crashes inside the kernel.
    The interface must reject it before any compile/launch."""
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm100
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm100()
    device = torch.device("cuda")
    s_q, s_kv, num_heads = 4, 128, 64
    head_dim, head_dim_v, topk = 128, 128, 64
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device)
    out = torch.randn(s_q, num_heads, head_dim_v, dtype=torch.bfloat16, device=device)
    dout = torch.randn_like(out)
    lse = torch.randn(s_q, num_heads, dtype=torch.float32, device=device)
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)

    with pytest.raises(AssertionError, match="head_dim must be 512 or 576"):
        _interface_sm100.flash_attn_bwd_sm100(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            softmax_scale=softmax_scale,
        )


@pytest.mark.L0
@torch_fork_set_rng(seed=90)
def test_DSA_sparse_attention_backward_sm90_scratch_workspace_bytes_formula():
    """SM90 reports its FP32 dPsum / log2-LSE / dKV-accumulator scratch (R2) instead of allocating it per call."""
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm90 import flash_attn_bwd_sm90_workspace_size
        from cudnn.api_base import ws_align
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    s_q, s_kv, head_dim, num_heads = 7, 1025, 576, 32
    sq_r, sk_r, hd_r = 64, 1088, 576  # 64-row Q/KV tiles, 32-wide head-dim padding
    expected = 2 * ws_align(sq_r * num_heads * 4) + ws_align(sk_r * hd_r * 4)
    assert flash_attn_bwd_sm90_workspace_size(s_q, s_kv, head_dim, num_heads) == expected
    assert flash_attn_bwd_sm90_workspace_size(64, 64, 512, 16) == 2 * ws_align(64 * 16 * 4) + ws_align(64 * 512 * 4)

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        return
    from cudnn import DSA

    device = torch.device("cuda")
    plan = DSA.SparseAttentionBackward(
        sample_q=torch.empty(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device),
        sample_kv=torch.empty(s_kv, head_dim, dtype=torch.bfloat16, device=device),
        sample_out=torch.empty(s_q, num_heads, 512, dtype=torch.bfloat16, device=device),
        sample_dout=torch.empty(s_q, num_heads, 512, dtype=torch.bfloat16, device=device),
        sample_lse=torch.empty(s_q, num_heads, dtype=torch.float32, device=device),
        sample_attn_sink=torch.empty(num_heads, dtype=torch.float32, device=device),
        sample_topk_idxs=torch.empty(s_q, 64, dtype=torch.int32, device=device),
    )
    assert plan.check_support()
    assert plan.scratch_workspace_bytes() == expected


@pytest.mark.L0
@torch_fork_set_rng(seed=91)
def test_DSA_sparse_attention_backward_sm90_execute_allocates_nothing_and_never_synchronizes():
    """Recipe R9 on the Hopper route: scratch is carved from the caller workspace, outputs are the caller's."""
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm90 import flash_attn_bwd_sm90_workspace_size
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm90()
    plan, inputs, outputs, topk_length, softmax_scale, workspace = _plan_case(32, 576, 64, has_topk_length=True)
    assert plan.scratch_workspace_bytes() == flash_attn_bwd_sm90_workspace_size(8, 256, 576, 32)
    _assert_execute_allocates_nothing_and_never_synchronizes(plan, inputs, outputs, topk_length, softmax_scale, workspace)


@pytest.mark.L0
@torch_fork_set_rng(seed=100)
@pytest.mark.parametrize(
    "num_heads,head_dim,topk,deterministic,q_cluster_mode,expected_backend",
    [
        pytest.param(16, 576, 64, False, "off", "h16_m128", id="h16"),
        pytest.param(32, 576, 64, False, "off", "h32_m64", id="h32"),
        pytest.param(64, 576, 64, False, "off", "generic_m64", id="h64"),
        pytest.param(96, 576, 64, False, "off", "h96_h64_h32", id="h96"),
        pytest.param(96, 576, 128, False, "off", "h96_d576_2cta_m64", id="h96-d576-2cta"),
        pytest.param(128, 512, 64, False, "off", "generic_m64", id="h128-generic"),
        pytest.param(128, 512, 128, False, "off", "h128_2cta_m64", id="h128-2cta"),
        pytest.param(128, 576, 128, False, "off", "h128_d576_2cta_m64", id="h128-d576-2cta"),
        pytest.param(64, 576, 64, True, "off", "generic_m64", id="deterministic"),
        pytest.param(96, 576, 128, True, "off", "h96_d576_2cta_m64", id="deterministic-h96-d576-2cta"),
        pytest.param(16, 576, 512, False, "adaptive_pair", "h16_m128", id="qcluster-h16"),
        pytest.param(32, 576, 512, False, "adaptive_pair", "h32_m64", id="qcluster-h32"),
    ],
)
def test_DSA_sparse_attention_backward_sm100_execute_allocates_nothing_and_never_synchronizes(
    num_heads, head_dim, topk, deterministic, q_cluster_mode, expected_backend
):
    """Recipe R9 on every SM100 route: no torch allocation and no host sync across warm executes."""
    try:
        from cudnn import DSA  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm100()
    if expected_backend == "h128_2cta_m64":
        _require_exact_sm100()
    device_capability = torch.cuda.get_device_capability()
    if expected_backend in ("h96_d576_2cta_m64", "h128_d576_2cta_m64") and device_capability not in _TWO_CTA_CAPABILITIES:
        if expected_backend == "h96_d576_2cta_m64":
            expected_backend = "h96_h64_h32"
        else:
            pytest.skip("D576 2-CTA requires an SM100-class GPU (compute capability 10.0 or 10.3)")
    if q_cluster_mode == "adaptive_pair" and device_capability not in _TWO_CTA_CAPABILITIES:
        pytest.skip("q_cluster_mode='adaptive_pair' requires an SM100-class GPU (compute capability 10.0 or 10.3)")
    plan, inputs, outputs, topk_length, softmax_scale, workspace = _plan_case(
        num_heads,
        head_dim,
        topk,
        s_kv=max(256, 2 * topk),
        has_topk_length=True,
        deterministic=deterministic,
        q_cluster_mode=q_cluster_mode,
    )
    assert plan._backend == expected_backend
    assert (plan._qcluster_plan is not None) == (q_cluster_mode == "adaptive_pair")
    _assert_execute_allocates_nothing_and_never_synchronizes(plan, inputs, outputs, topk_length, softmax_scale, workspace)


@pytest.mark.L0
@torch_fork_set_rng(seed=101)
@pytest.mark.parametrize(
    "num_heads,head_dim,topk,q_cluster_mode",
    [
        pytest.param(32, 576, 64, "off", id="sm90"),
        pytest.param(64, 576, 64, "off", id="sm100-generic"),
        pytest.param(128, 576, 128, "off", id="sm100-d576-2cta"),
        pytest.param(16, 576, 512, "adaptive_pair", id="qcluster-h16"),
        pytest.param(32, 576, 512, "adaptive_pair", id="qcluster-h32"),
    ],
)
def test_DSA_sparse_attention_backward_execute_requires_preallocated_outputs(num_heads, head_dim, topk, q_cluster_mode, request):
    """execute() never allocates an output on any backend: a missing dq/dkv/d_sink raises before any launch."""
    try:
        from cudnn import DSA  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    route = request.node.callspec.id
    if route == "sm90":
        _require_sm90()
    else:
        _require_sm100()
        if route == "sm100-d576-2cta" and torch.cuda.get_device_capability() not in _TWO_CTA_CAPABILITIES:
            pytest.skip("D576 2-CTA requires an SM100-class GPU (compute capability 10.0 or 10.3)")
    if q_cluster_mode == "adaptive_pair" and torch.cuda.get_device_capability() not in _TWO_CTA_CAPABILITIES:
        pytest.skip("q_cluster_mode='adaptive_pair' requires an SM100-class GPU (compute capability 10.0 or 10.3)")
    plan, inputs, outputs, topk_length, softmax_scale, workspace = _plan_case(num_heads, head_dim, topk, s_kv=max(256, 2 * topk), q_cluster_mode=q_cluster_mode)
    dq, dkv, d_sink = outputs

    for missing in ("dq", "dkv", "d_sink"):
        provided = dict(dq=dq, dkv=dkv, d_sink=d_sink)
        provided[missing] = None
        with pytest.raises(ValueError, match=rf"{missing} must be preallocated"):
            plan.execute(*inputs, provided["dq"], provided["dkv"], softmax_scale=softmax_scale, workspace=workspace, d_sink=provided["d_sink"])
    # dq/dkv are positional-required and d_sink keyword-only required: omitting any is a signature
    # error, not a silent allocation -- and the pre-existing positional order (topk_length after dkv) holds.
    with pytest.raises(TypeError):
        plan.execute(*inputs, softmax_scale=softmax_scale, workspace=workspace)
    with pytest.raises(TypeError, match="d_sink"):
        plan.execute(*inputs, dq, dkv, softmax_scale=softmax_scale, workspace=workspace)
