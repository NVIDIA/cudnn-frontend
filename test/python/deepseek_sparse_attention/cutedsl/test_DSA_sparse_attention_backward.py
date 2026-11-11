# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SparseAttentionBackward.

Gradients are generated via autograd on the gather-based forward reference.
"""

import ast
import inspect
import itertools
import math
import re
import textwrap
import types

import pytest
import torch

from test_utils import assert_bitwise_runs, bitwise_bits, torch_fork_set_rng

from deepseek_sparse_attention.cutedsl.dsa_utils import (
    _require_sm100_family,
    _require_sm90,
    _require_sm100,
    dsa_init,
    with_dsa_sparse_attention_backward_params,
)
from deepseek_sparse_attention.cutedsl.dsa_reference import (
    check_ref_dsa_sparse_attention_backward,
    ref_dsa_sparse_attention_backward,
    ref_sparse_attention_forward_chunked,
)

# SM100-class compute capabilities routed to the H96/H128 D576 two-CTA backend; conflict-rank dKV storage is
# provided on the same devices.
_TWO_CTA_CAPABILITIES = ((10, 0), (10, 3))


def _require_conflict_rank_device():
    """Skip unless the GPU provides conflict-rank dKV storage (compute capability 10.0 or 10.3)."""
    _require_sm100()
    if torch.cuda.get_device_capability() not in _TWO_CTA_CAPABILITIES:
        pytest.skip("conflict-rank dKV storage requires compute capability 10.0 or 10.3")


def _import_or_skip(module_name, reason="Environment not supported: cudnn[cutedsl] not installed"):
    """Import an optional CuTeDSL module or skip when the environment lacks it."""
    return pytest.importorskip(module_name, reason=reason)


def _require_sm90_relay_dependencies():
    """Skip before the relay package import can load optional dependencies."""
    _import_or_skip("cutlass")
    _import_or_skip("cuda.bindings.driver", reason="Environment not supported: cuda-python not installed")


def _require_sm90_relay_dsl():
    """Skip feature tests when the deterministic SM90 relay cannot compile."""
    _require_sm90_relay_dependencies()
    from cudnn.deepseek_sparse_attention.sparse_attention_backward._sm90_relay import sm90_deterministic_relay_requirement_error

    requirement_error = sm90_deterministic_relay_requirement_error()
    if requirement_error is not None:
        pytest.skip(requirement_error)


def _require_sm90_deterministic_relay():
    _require_sm90()
    _require_sm90_relay_dsl()


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
    deterministic_relay_cluster_size=None,
    compile_plan=True,
    deterministic_dkv_mode="records",
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
        deterministic_relay_cluster_size=deterministic_relay_cluster_size,
        deterministic_dkv_mode=deterministic_dkv_mode,
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
    # Outside the two-CTA envelope, deterministic H96 D576 runs the generic M64 kernel only where the plan keeps
    # dKV in conflict-rank storage; the shard layout, an explicit relay and larger records keep the composition.
    shape = dict(max_topk=64, deterministic=True, total_s_q=300, total_s_kv=4096)
    assert _select_sm100_backend(96, 576, **shape) == ("generic_m64", 64)
    assert _select_sm100_backend(96, 576, **shape, dkv_mode="chain") == ("generic_m64", 64)
    assert _select_sm100_backend(96, 576, **shape, dkv_mode="shards") == ("h96_h64_h32", 64)
    assert _select_sm100_backend(96, 576, max_topk=64, deterministic=True, total_s_q=8, total_s_kv=256) == ("h96_h64_h32", 64)
    assert _select_sm100_backend(96, 576, max_topk=64, deterministic=True, total_s_q=4096, total_s_kv=65536, deterministic_relay_cluster_size=8) == (
        "h96_h64_h32",
        64,
    )


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
    assert "deterministic_dkv_relay_cluster_size" not in wrapper
    assert "deterministic_dkv_relay_cluster_size" not in plan
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
    _require_sm100_family()
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
    monkeypatch.setattr(interface, "_deterministic_kernel_class_for_config", lambda shards, relay_size, layout=None: FakeKernel)
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
    _require_sm100_family()
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
    _require_sm100_family()
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


def _rounding_step(values, dtype):
    """Spacing of ``dtype`` at each ``|value|`` (one rounding step of the output format)."""
    _, exponent = torch.frexp(values.float().abs())
    return torch.finfo(dtype).eps * torch.pow(2.0, (exponent - 1).float())


def _nondeterministic_outputs(q, kv, out, dout, lse, attn_sink, topk_idxs, topk_length, softmax_scale):
    """Run the non-deterministic plan of the same inputs and return its ``(dq, dkv, d_sink)``."""
    from cudnn import DSA
    from cuda.bindings import driver as cuda

    plan = DSA.SparseAttentionBackward(q, kv, out, dout, lse, attn_sink, topk_idxs, sample_topk_length=topk_length, softmax_scale=softmax_scale)
    assert plan.check_support()
    plan.compile()
    workspace_bytes = plan.scratch_workspace_bytes()
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=q.device) if workspace_bytes else None
    dq, dkv, d_sink = torch.empty_like(q), torch.empty_like(kv), torch.empty_like(attn_sink)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
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
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        workspace=workspace,
        current_stream=stream,
        d_sink=d_sink,
    )
    torch.cuda.synchronize()
    return dq, dkv, d_sink


def _check_ref_high_fan_in(q, kv, out, dout, lse, attn_sink, topk_idxs, outputs, *, softmax_scale, topk_length, atol=5e-2, rtol=5e-2):
    """Reference check of deterministic outputs for inputs with high-fan-in KV rows (shared or repeated rows).

    Every dKV element adds one product of BF16-rounded operands (P, dS) per (query, head, slot) that selects
    its row. On rows with hundreds of valid slots that rounding alone can exceed ``atol + rtol * |ref|``
    against the FP32 reference, for every dKV layout and for the non-deterministic path alike (the same
    elements, by the same amounts, as an FP64 reference shows). dKV is therefore bounded by the error of
    the non-deterministic plan on the same inputs: per element ``|det - ref| <= |nondet - ref| + step(|ref|)
    + atol`` and in max-abs ``max|det - ref| <= max|nondet - ref| + step(largest |value|)``, with ``step`` one
    rounding step of the output dtype. dQ and dSink keep the plain ``atol``/``rtol`` check.
    """
    dq, dkv, d_sink = outputs
    dq_ref, dkv_ref, d_sink_ref = ref_dsa_sparse_attention_backward(q, kv, attn_sink, topk_idxs, dout, softmax_scale=softmax_scale, topk_length=topk_length)
    torch.testing.assert_close(dq.float(), dq_ref, atol=atol, rtol=rtol, msg=lambda text: f"dQ: {text}")
    torch.testing.assert_close(d_sink.float(), d_sink_ref, atol=atol, rtol=rtol, msg=lambda text: f"dSink: {text}")
    nondeterministic_dkv = _nondeterministic_outputs(q, kv, out, dout, lse, attn_sink, topk_idxs, topk_length, softmax_scale)[1]
    error = (dkv.float() - dkv_ref).abs()
    nondeterministic_error = (nondeterministic_dkv.float() - dkv_ref).abs()
    excess = error - (nondeterministic_error + _rounding_step(dkv_ref, dkv.dtype) + atol)
    worst = int(excess.nan_to_num(float("inf")).argmax())
    row, column = divmod(worst, dkv.shape[-1])
    assert bool((excess <= 0).all()), (
        f"dKV: {int((~(excess <= 0)).sum())} elements exceed the non-deterministic error + one rounding step + atol; worst at "
        f"({row}, {column}): deterministic {dkv[row, column].item()}, non-deterministic {nondeterministic_dkv[row, column].item()}, "
        f"reference {dkv_ref[row, column].item()}"
    )
    largest = torch.stack((dkv_ref.abs().max(), dkv.float().abs().max(), nondeterministic_dkv.float().abs().max())).max()
    bound = nondeterministic_error.max() + _rounding_step(largest, dkv.dtype)
    assert error.max() <= bound, f"dKV: max abs error {error.max().item()} exceeds the non-deterministic {nondeterministic_error.max().item()} + one step"


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
    deterministic_dkv_mode="records",
    expected_dkv_route=None,
    duplicate_topk=False,
    invalid_topk=False,
    expected_backend=None,
    high_fan_in_reference=False,
):
    """Run one deterministic case against bitwise and numerical contracts.

    ``expected_dkv_route`` checks the plan's dKV reduction (``'records'``, ``'chain'`` or ``'shards'``) and
    ``expected_backend`` its SM100 backend. ``high_fan_in_reference`` bounds dKV by the non-deterministic
    plan's reference error (``_check_ref_high_fan_in``) instead of the plain tolerance.
    ``duplicate_topk`` repeats a KV row inside every other query's top-k (its gradients accumulate) and
    ``invalid_topk`` adds negative and past-the-end indices (inactive, as in the reference).
    """
    import cutlass
    from cudnn import DSA
    from cuda.bindings import driver as cuda
    from cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100 import (
        _get_sm100_kernel_class,
        _select_sm100_backend,
        effective_deterministic_dkv_mode,
        flash_attn_bwd_sm100_workspace_size,
    )

    _require_sm100()
    if deterministic_dkv_mode == "chain" or expected_dkv_route in ("records", "chain"):
        _require_conflict_rank_device()
    device_capability = torch.cuda.get_device_capability()
    # The plan runs 'records' as 'shards' on devices without conflict-rank storage.
    dkv_mode = effective_deterministic_dkv_mode(deterministic_dkv_mode, True, device_capability)
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
    if duplicate_topk and topk > 1:
        topk_idxs[::2, 1] = topk_idxs[::2, 0]
    if invalid_topk and topk > 2:
        topk_idxs[1::3, -1] = -1
        topk_idxs[2::3, -2] = s_kv + 3
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
        device_capability=device_capability,
        deterministic=True,
        total_s_q=s_q,
        total_s_kv=s_kv,
        deterministic_relay_cluster_size=deterministic_relay_cluster_size,
        dkv_mode=deterministic_dkv_mode,
    )
    # The expected sizes below are those of the class-sized layouts. H128 two-CTA conflict-rank plans size
    # their storage per variant and, for D576, per device wave: _check_h128_deterministic_case covers them.
    if backend in ("h128_2cta_m64", "h128_d576_2cta_m64"):
        assert not _h128_conflict_rank_selected(head_dim, s_kv, topk, dkv_mode), "use _check_h128_deterministic_case for H128 conflict-rank plans"
    workspace_bytes = flash_attn_bwd_sm100_workspace_size(
        s_q,
        s_kv,
        head_dim,
        num_heads,
        deterministic=True,
        backend=backend,
        deterministic_relay_cluster_size=deterministic_relay_cluster_size,
        max_topk=topk,
        dkv_mode=deterministic_dkv_mode,
        device_capability=device_capability,
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
        deterministic_dkv_mode=deterministic_dkv_mode,
    )
    assert api.check_support()
    api.compile()
    assert api.scratch_workspace_bytes() == workspace_bytes
    assert api._backend == backend
    if expected_backend is not None:
        assert backend == expected_backend
    if expected_dkv_route is not None:
        assert api._deterministic_dkv_route() == expected_dkv_route
    expected_lse_odo_bytes = num_heads * math.ceil(s_q / 8) * 8 * 2 * torch.float32.itemsize
    kernel_cls = _get_sm100_kernel_class(
        backend,
        True,
        s_q,
        s_kv,
        head_dim,
        num_heads,
        deterministic_relay_cluster_size,
        max_topk=topk,
        dkv_mode=dkv_mode,
    )
    if expected_relay_cluster_size is not None:
        assert kernel_cls.dkv_relay_cluster_size == expected_relay_cluster_size
    if hasattr(kernel_cls, "_get_workspace_size_dKV_for_rank"):
        dkv_shape = kernel_cls._get_workspace_size_dKV_for_rank(s_kv, head_dim, topk, 1, cutlass.Float32, num_heads, s_q)
    elif hasattr(kernel_cls, "_get_workspace_size_dKV_for_q"):
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
                deterministic_dkv_mode=deterministic_dkv_mode,
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

    if high_fan_in_reference:
        _check_ref_high_fan_in(q, kv, out, dout, lse, attn_sink, topk_idxs, reference, softmax_scale=softmax_scale, topk_length=topk_length)
        return
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
    "num_heads,head_dim,repeats,deterministic_dkv_mode",
    [
        pytest.param(16, 576, 3, "records", marks=pytest.mark.L0, id="H16-D576-tail-smoke"),
        pytest.param(16, 576, 3, "shards", marks=pytest.mark.L0, id="H16-D576-shards-tail-smoke"),
        pytest.param(16, 576, 1000, "records", marks=pytest.mark.L2, id="H16-D576-repeat1000"),
        pytest.param(32, 512, 1000, "records", marks=pytest.mark.L2, id="H32-D512-repeat1000"),
        pytest.param(64, 512, 1000, "records", marks=pytest.mark.L2, id="H64-D512-repeat1000"),
        pytest.param(64, 576, 1000, "records", marks=pytest.mark.L2, id="H64-D576-repeat1000"),
        pytest.param(96, 512, 1000, "records", marks=pytest.mark.L2, id="H96-D512-repeat1000"),
        pytest.param(96, 576, 1000, "records", marks=pytest.mark.L2, id="H96-D576-repeat1000"),
        pytest.param(128, 576, 1000, "records", marks=pytest.mark.L2, id="H128-D576-repeat1000"),
        pytest.param(64, 576, 1000, "shards", marks=pytest.mark.L2, id="H64-D576-shards-repeat1000"),
        pytest.param(96, 576, 1000, "shards", marks=pytest.mark.L2, id="H96-D576-shards-repeat1000"),
    ],
)
def test_DSA_sparse_attention_backward_sm100_deterministic_bounded_waves(num_heads, head_dim, repeats, deterministic_dkv_mode):
    """Masked and multi-block heads must be bitwise reproducible.

    At these shapes the default plans keep dKV in conflict-rank records; ``'shards'`` plans run the
    128-shard layout, and deterministic H96 D576 the H64 + H32 composition.
    """
    h96_d576_shards = deterministic_dkv_mode == "shards" and (num_heads, head_dim) == (96, 576)
    try:
        _exercise_deterministic_sm100_case(
            num_heads,
            head_dim,
            257,
            256,
            repeats,
            deterministic_dkv_mode=deterministic_dkv_mode,
            expected_dkv_route=None if h96_d576_shards else deterministic_dkv_mode,
            expected_backend="h96_h64_h32" if h96_d576_shards else "generic_m64",
        )
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


# Deterministic H128 two-CTA KV extents per head dim and top-k width that select conflict-rank dKV
# storage, and extents that keep the private-plane layout (checked by ``_h128_conflict_rank_selected``).
_H128_CONFLICT_RANK_S_KV = {
    512: {
        "conflict_rank": {128: 4096, 512: 4096, 1024: 4096, 1152: 4096, 2048: 4096},
        "private_plane": {128: 64, 512: 256, 1024: 512, 1152: 576, 2048: 1024},
    },
    576: {
        "conflict_rank": {128: 4096, 512: 4096, 1024: 4096, 1152: 4096, 2048: 4096},
        "private_plane": {128: 256, 512: 2048, 1024: 1024, 1152: 2048, 2048: 2048},
    },
}
# Scratch bytes past ``scratch_workspace_bytes()`` that the H128 deterministic tests allocate with a
# sentinel value and check afterwards: a launch that writes beyond the reported size changes them.
_H128_SCRATCH_TAIL_BYTES = 1 << 20
_H128_SCRATCH_TAIL_VALUE = 0x5A


def _h128_conflict_rank_selected(head_dim, s_kv, topk, dkv_mode="records"):
    """Whether the deterministic H128 two-CTA plan for this KV extent, top-k width and dKV mode uses conflict-rank dKV storage."""
    if head_dim == 512:
        interface = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100")
        kernel_cls = interface._get_sm100_kernel_class("h128_2cta_m64", True, 1, s_kv, 512, 128, None, max_topk=topk, dkv_mode=dkv_mode)
        return bool(getattr(kernel_cls, "rank_shard", False))
    interface = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100_d576")
    return interface._rank_shard_rounds(s_kv, topk, True, 128, dkv_mode) > 0


def _h128_private_plane_workspace_bytes(s_q, s_kv, head_dim):
    """Caller scratch of the private-plane deterministic H128 layout (``max_topk`` omitted)."""
    interface = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100")
    backend = "h128_2cta_m64" if head_dim == 512 else "h128_d576_2cta_m64"
    return interface.flash_attn_bwd_sm100_workspace_size(s_q, s_kv, head_dim, 128, True, backend=backend)


def _h128_conflict_rank_inputs(head_dim, s_q, s_kv, topk, *, has_topk_length, invalid_entries=True):
    """Deterministic H128 inputs with duplicate, ``-1`` and out-of-range top-k entries and edge lengths."""
    device = "cuda"
    q = torch.randn(s_q, 128, head_dim, dtype=torch.bfloat16, device=device) / 10
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device) / 10
    attn_sink = torch.randn(128, dtype=torch.float32, device=device)
    if s_kv >= topk:
        topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    else:
        # Fewer KV rows than top-k slots: every query selects rows more than once.
        topk_idxs = torch.randint(0, s_kv, (s_q, topk), dtype=torch.int32, device=device)
    if invalid_entries:
        # Every query repeats one row and carries a -1 sentinel and an out-of-range row (both inactive).
        topk_idxs[:, 1] = topk_idxs[:, 0]
        topk_idxs[:, 2] = -1
        topk_idxs[:, topk - 1] = s_kv
    topk_length = None
    if has_topk_length:
        topk_length = torch.randint(1, topk + 1, (s_q,), dtype=torch.int32, device=device)
        topk_length[0] = 0
        topk_length[1 % s_q] = topk
        topk_length[2 % s_q] = 1
    softmax_scale = head_dim**-0.5
    out, lse = ref_sparse_attention_forward_chunked(q, kv, attn_sink, topk_idxs, topk_length=topk_length, softmax_scale=softmax_scale)
    dout = torch.randn_like(out) / 10
    return (q, kv, out, dout, lse, attn_sink, topk_idxs), topk_length, softmax_scale


def _h128_deterministic_plan(head_dim, inputs, topk_length, softmax_scale, deterministic_dkv_mode="records"):
    """Compile a deterministic H128 two-CTA plan and allocate its outputs and its caller scratch.

    Returns ``(plan, workspace, outputs, tail)``: ``workspace`` is exactly ``scratch_workspace_bytes()``
    long and is followed in the same allocation by ``tail``, sentinel bytes that no launch may write
    (``_assert_h128_scratch_tail_untouched``).
    """
    from cudnn import DSA

    q, kv, out, dout, lse, attn_sink, topk_idxs = inputs
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
        deterministic=True,
        deterministic_dkv_mode=deterministic_dkv_mode,
    )
    assert plan.check_support()
    plan.compile()
    assert plan._backend == ("h128_2cta_m64" if head_dim == 512 else "h128_d576_2cta_m64")
    required = plan.scratch_workspace_bytes()
    scratch = torch.full((required + _H128_SCRATCH_TAIL_BYTES,), _H128_SCRATCH_TAIL_VALUE, dtype=torch.uint8, device=q.device)
    outputs = (torch.empty_like(q), torch.empty_like(kv), torch.empty_like(attn_sink))
    return plan, scratch[:required], outputs, scratch[required:]


def _assert_h128_scratch_tail_untouched(tail):
    """The sentinel bytes after the reported scratch size are intact: no launch wrote past it."""
    torch.cuda.synchronize()
    assert bool((tail == _H128_SCRATCH_TAIL_VALUE).all()), "a launch wrote past scratch_workspace_bytes()"


def _h128_execute(plan, inputs, outputs, topk_length, softmax_scale, workspace, stream):
    dq, dkv, d_sink = outputs
    return plan.execute(*inputs, dq, dkv, topk_length=topk_length, softmax_scale=softmax_scale, workspace=workspace, current_stream=stream, d_sink=d_sink)


def _require_h128_two_cta():
    _require_sm100()
    if torch.cuda.get_device_capability() not in _TWO_CTA_CAPABILITIES:
        pytest.skip("H128 two-CTA requires compute capability 10.0 or 10.3")


def _check_h128_deterministic_case(head_dim, s_q, s_kv, topk, *, has_topk_length, layout, repeats, label, deterministic_dkv_mode="records"):
    """Run one deterministic H128 two-CTA case on poisoned scratch and outputs.

    Checks the dKV layout (``layout``) that the plan's ``deterministic_dkv_mode`` selects and its scratch
    bound, the reference gradients, bitwise reruns and the sentinel tail after the scratch.
    """
    from cuda.bindings import driver as cuda

    assert _h128_conflict_rank_selected(head_dim, s_kv, topk, deterministic_dkv_mode) == (layout == "conflict_rank")
    inputs, topk_length, softmax_scale = _h128_conflict_rank_inputs(head_dim, s_q, s_kv, topk, has_topk_length=has_topk_length)
    plan, workspace, outputs, tail = _h128_deterministic_plan(head_dim, inputs, topk_length, softmax_scale, deterministic_dkv_mode)
    private_plane = _h128_private_plane_workspace_bytes(s_q, s_kv, head_dim)
    if layout == "conflict_rank":
        assert workspace.numel() <= private_plane
    else:
        assert workspace.numel() == private_plane
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def run():
        for buffer in (workspace, *outputs):
            bitwise_bits(buffer).fill_(0xA5)
        result = _h128_execute(plan, inputs, outputs, topk_length, softmax_scale, workspace, stream)
        return tuple(tensor.clone() for tensor in result)

    first = run()
    torch.cuda.synchronize()
    q, kv, out, dout, lse, attn_sink, topk_idxs = inputs
    check_ref_dsa_sparse_attention_backward(
        q, kv, attn_sink, topk_idxs, out, dout, lse, *first, softmax_scale=softmax_scale, topk_length=topk_length, atol=5e-2, rtol=5e-2
    )
    assert_bitwise_runs(run, repeats=repeats, label=label)
    _assert_h128_scratch_tail_untouched(tail)


@pytest.mark.L0
@pytest.mark.parametrize("head_dim", [512, 576], ids=["d512", "d576"])
@pytest.mark.parametrize("topk", [128, 512, 1024, 1152, 2048])
def test_DSA_sparse_attention_backward_h128_conflict_rank_workspace_bound(head_dim, topk):
    """Conflict-rank plans are selected only where their scratch fits the private-plane layout."""
    interface = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100")
    backend = "h128_2cta_m64" if head_dim == 512 else "h128_d576_2cta_m64"
    pair_options = (None,) if head_dim == 512 else (None, 66, 74, 76)
    selected = 0
    for s_kv in (64, 256, 1024, 2048, 2097, 4096, 8192, 32768, 131072, 1 << 20):
        for s_q in (1, 75, 4096):
            private_plane = _h128_private_plane_workspace_bytes(s_q, s_kv, head_dim)
            for launch_pairs in pair_options:
                extra = {} if launch_pairs is None else {"launch_pairs": launch_pairs}
                size = interface.flash_attn_bwd_sm100_workspace_size(s_q, s_kv, head_dim, 128, True, backend=backend, max_topk=topk, **extra)
                assert size <= private_plane, f"S_kv={s_kv} S_q={s_q} pairs={launch_pairs}: {size} > {private_plane}"
                if _h128_conflict_rank_selected(head_dim, s_kv, topk):
                    selected += 1
                else:
                    assert size == private_plane
    assert selected > 0
    if topk in _H128_CONFLICT_RANK_S_KV[head_dim]["conflict_rank"]:
        assert _h128_conflict_rank_selected(head_dim, _H128_CONFLICT_RANK_S_KV[head_dim]["conflict_rank"][topk], topk)
        assert not _h128_conflict_rank_selected(head_dim, _H128_CONFLICT_RANK_S_KV[head_dim]["private_plane"][topk], topk)


@pytest.mark.L0
@pytest.mark.parametrize("head_dim", [512, 576], ids=["d512", "d576"])
@pytest.mark.parametrize("topk", [128, 512, 1024, 1152, 2048])
def test_DSA_sparse_attention_backward_h128_shards_mode_keeps_private_plane(head_dim, topk):
    """``deterministic_dkv_mode='shards'`` H128 two-CTA plans keep the private-plane class and workspace on every shape.

    The KV extents select conflict-rank storage for the default mode, so ``'shards'`` changes the layout there.
    """
    interface = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100")
    d576 = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100_d576")
    backend = "h128_2cta_m64" if head_dim == 512 else "h128_d576_2cta_m64"
    pair_options = (None,) if head_dim == 512 else (None, 66, 74, 76)
    for s_kv in (_H128_CONFLICT_RANK_S_KV[head_dim]["conflict_rank"][topk], 32768, 1 << 20):
        assert _h128_conflict_rank_selected(head_dim, s_kv, topk)
        assert not _h128_conflict_rank_selected(head_dim, s_kv, topk, "shards")
        for s_q in (1, 75, 4096):
            private_plane = _h128_private_plane_workspace_bytes(s_q, s_kv, head_dim)
            for launch_pairs in pair_options:
                size = interface.flash_attn_bwd_sm100_workspace_size(
                    s_q, s_kv, head_dim, 128, True, backend=backend, max_topk=topk, launch_pairs=launch_pairs, dkv_mode="shards"
                )
                assert size == private_plane, f"S_kv={s_kv} S_q={s_q} pairs={launch_pairs}: {size} != {private_plane}"
            if head_dim == 512:
                private_cls = interface._get_sm100_kernel_class(backend, True, s_q, s_kv, 512, 128, None)
                assert not getattr(private_cls, "rank_shard", False)
                assert interface._get_sm100_kernel_class(backend, True, s_q, s_kv, 512, 128, None, max_topk=topk, dkv_mode="shards") is private_cls
            else:
                assert d576._rank_shard_rounds(s_kv, topk, True, 128, "shards") == 0
                assert not d576._odo_precompute(s_kv, topk, True, 128, "shards")
                for launch_pairs in pair_options:
                    assert d576._workspace_bytes(s_q, s_kv, True, 128, topk, launch_pairs, "shards") == private_plane
    if head_dim == 512:
        # The private-plane workspace of these extents exceeds Int32 offsets; the default plan's does not.
        wide = interface._requires_wide_data_addressing
        assert wide(160, 32768, 512, 512, 128, topk, True, backend, include_workspace=True, dkv_mode="shards")
        assert not wide(160, 32768, 512, 512, 128, topk, True, backend, include_workspace=True)


@pytest.mark.L0
@pytest.mark.parametrize("sm_count", [132, 148, 152, 160, 264])
def test_DSA_sparse_attention_backward_d576_conflict_rank_rounds_fit_ordinal_words(sm_count):
    """Multi-round D576 conflict-rank waves keep their ordinal bits within the setup kernel's words."""
    interface = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100_d576")
    rank_shard = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_h128_d576_2cta_rank_shard")

    unclamped = interface._deterministic_compact_slots(sm_count)
    assert interface._deterministic_compact_slots(sm_count, 0) == unclamped
    for rounds in (1, 2, 3):
        pairs = interface._deterministic_compact_slots(sm_count, rounds)
        assert rank_shard.rank_ordinal_words(rounds * pairs) <= rank_shard.RANK_MAX_ORDINAL_WORDS
        assert pairs == min(unclamped, rank_shard.rank_max_round_pairs(rounds))
    if sm_count == 148:
        # B200 launches 74 pairs per round whatever the round count.
        assert all(interface._deterministic_compact_slots(sm_count, rounds) == 74 for rounds in (0, 1, 2, 3))
    if sm_count >= 152:
        assert interface._deterministic_compact_slots(sm_count, 3) == 74


def _h128_kernel_geometry(kernel, *names):
    """Evaluate a conflict-rank kernel's layout methods (``rank_geometry``, ``rank_offsets``, ...) in Python integers.

    The methods run at trace time on DSL integers built from trace-time attributes, so executing their
    source with Python ints (``cute.round_up`` and ``cutlass.const_expr`` taken by their integer
    meaning) yields the plane count and region offsets the kernels use. Floor division differs from
    the DSL's only for a negative plane budget, which both clamp to the same plane count. Returns an
    object whose ``names`` are those methods bound to ``kernel``; other attributes come from ``kernel``.
    """
    shims = {
        "Int32": int,
        "Int64": int,
        "cute": types.SimpleNamespace(round_up=lambda value, multiple: -(-value // multiple) * multiple),
        "cutlass": types.SimpleNamespace(const_expr=lambda value: value),
    }
    functions = {}
    for name in names:
        owner = next(cls for cls in type(kernel).__mro__ if name in vars(cls))
        class_node = ast.parse(textwrap.dedent(inspect.getsource(owner))).body[0]
        method = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == name)
        method.decorator_list = []
        namespace = {**vars(inspect.getmodule(owner)), **shims}
        exec(compile(ast.Module(body=[method], type_ignores=[]), f"<{owner.__name__}.{name}>", "exec"), namespace)
        functions[name] = namespace[name]

    class Geometry:
        def __getattr__(self, attribute):
            if attribute in functions:
                return types.MethodType(functions[attribute], self)
            return getattr(kernel, attribute)

    return Geometry()


# KV extents for the host-versus-kernel layout checks: about 1% apart from 1 to 4M rows, so plane-count
# boundaries of both formulas fall between neighbouring extents.
_H128_GEOMETRY_S_KV = sorted({int(1.01**step) for step in range(1530)} | {2097, 4097, (3 << 20) + 5})


@pytest.mark.L0
def test_DSA_sparse_attention_backward_d512_conflict_rank_host_layout_matches_kernel():
    """The host D512 conflict-rank sizes equal the plane count and regions the kernels address."""
    cutlass = _import_or_skip("cutlass")
    interface = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100")
    common = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100_common")
    rank = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_2cta_rank_shard")
    geometries = {}
    checked = 0
    for topk in (128, 512, 1024, 1152, 2048):
        for s_kv in _H128_GEOMETRY_S_KV:
            kernel_cls = interface._get_sm100_kernel_class("h128_2cta_m64", True, 1, s_kv, 512, 128, None, max_topk=topk)
            if not getattr(kernel_cls, "rank_shard", False):
                continue
            if (kernel_cls, topk) not in geometries:
                geometries[kernel_cls, topk] = _h128_kernel_geometry(kernel_cls(cutlass.BFloat16, 512, 512, 64, topk), "rank_geometry", "rank_offsets")
            geometry = kernel = geometries[kernel_cls, topk]
            wave = kernel.max_query_pairs_per_wave
            pitch, planes = geometry.rank_geometry(s_kv)
            counters = geometry.rank_offsets(pitch, planes)[-1]
            fused = rank.rank_merged_fused(s_kv, topk, kernel.rank_record, wave)
            case = f"K={topk} S_kv={s_kv}"
            assert kernel.rank_fused == fused, case
            assert planes == rank.rank_merged_planes(s_kv, topk, kernel.rank_record, wave, fused), case
            device_words = counters + rank.RANK_MERGED_COUNTER_WORDS
            assert device_words == rank.rank_merged_workspace_words(s_kv, topk, kernel.rank_record, wave), case
            _, dkv_shape = common._workspace_shapes_sm100(1, s_kv, 512, 128, True, "h128_2cta_m64", max_topk=topk)
            assert math.prod(dkv_shape) >= 4 * device_words, case
            checked += 1
    assert checked > 0


@pytest.mark.L0
@pytest.mark.parametrize("sm_count", [132, 148, 152, 160])
def test_DSA_sparse_attention_backward_d576_conflict_rank_host_layout_covers_kernel(sm_count):
    """The host D576 conflict-rank sizes cover the plane count and regions the kernels address on a device."""
    cutlass = _import_or_skip("cutlass")
    interface = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100_d576")
    common = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100_common")
    rank = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_h128_d576_2cta_rank_shard")
    kernels = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_h128_d576_2cta")
    geometries = {}
    checked = 0
    for topk in (128, 512, 1024, 1152, 2048):
        for s_kv in _H128_GEOMETRY_S_KV:
            rounds = interface._rank_shard_rounds(s_kv, topk, True, 128)
            if rounds == 0:
                continue
            pairs = interface._deterministic_compact_slots(sm_count, rounds)
            if (topk, rounds) not in geometries:
                kernel = kernels.FlashAttentionDSABackwardSm100H128D576TwoCTADeterministicRankShard(
                    element_dtype=cutlass.BFloat16,
                    head_dim=576,
                    head_dim_v=512,
                    block_tile=64,
                    max_topk=topk,
                    num_heads=128,
                    compact_slots=pairs,
                    rank_rounds=rounds,
                )
                geometries[topk, rounds] = _h128_kernel_geometry(kernel, "rank_geometry", "rank_offsets", "rank_sorted_base")
            geometry = kernel = geometries[topk, rounds]
            pitch, planes = geometry.rank_geometry(s_kv)
            device_words = geometry.rank_offsets(pitch, planes)[6] + rank.RANK_COUNTER_REGION_WORDS
            if kernel.rank_sorts():
                # End of the last sorted-row parity.
                device_words = geometry.rank_sorted_base(pitch, planes, rank.RANK_SORT_PARITIES - 1) + kernel.max_query_pairs_per_wave * topk
            host_wave = interface._MAX_DETERMINISTIC_COMPACT_SLOTS
            case = f"K={topk} S_kv={s_kv} SMs={sm_count}"
            assert planes == min(rank.conflict_rank_planes(s_kv, topk, kernel.rank_record, host_wave), kernel.max_query_pairs_per_wave), case
            assert device_words <= rank.conflict_rank_words(s_kv, topk, kernel.rank_record, host_wave, launch_pairs=pairs), case
            _, dkv_shape = common._workspace_shapes_sm100(1, s_kv, 576, 128, True, "h128_d576_2cta_m64", max_topk=topk, launch_pairs=pairs)
            assert math.prod(dkv_shape) >= 4 * device_words, case
            checked += 1
    assert checked > 0


@pytest.mark.L0
def test_DSA_sparse_attention_backward_d576_conflict_rank_rejects_oversized_wave():
    """A conflict-rank class refuses a wave whose ordinal bits exceed the setup kernel's words."""
    cutlass = _import_or_skip("cutlass")
    kernels = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_h128_d576_2cta")
    arguments = dict(element_dtype=cutlass.BFloat16, head_dim=576, head_dim_v=512, block_tile=64, max_topk=512, num_heads=128)
    kernels.FlashAttentionDSABackwardSm100H128D576TwoCTADeterministicRankShard(**arguments, compact_slots=74, rank_rounds=3)
    with pytest.raises(ValueError, match="cluster pairs per round"):
        kernels.FlashAttentionDSABackwardSm100H128D576TwoCTADeterministicRankShard(**arguments, compact_slots=76, rank_rounds=3)


def _h128_conflict_rank_cases():
    cases = []
    for head_dim in (512, 576):
        for layout in ("conflict_rank", "private_plane"):
            for topk in (128, 512, 1024, 1152, 2048):
                for has_topk_length in (True, False):
                    level = pytest.mark.L1 if has_topk_length else pytest.mark.L2
                    case_id = f"d{head_dim}-{layout}-k{topk}-{'lengths' if has_topk_length else 'full'}"
                    cases.append(pytest.param(head_dim, layout, topk, has_topk_length, marks=level, id=case_id))
    return cases


@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("head_dim,layout,topk,has_topk_length", _h128_conflict_rank_cases())
@torch_fork_set_rng(seed=20261104)
def test_DSA_sparse_attention_backward_h128_conflict_rank_deterministic(head_dim, layout, topk, has_topk_length):
    """Conflict-rank and private-plane H128 plans: reference gradients, bounded scratch, bitwise reruns.

    Several waves (more queries than one wave's cluster pairs), duplicate rows, -1 and out-of-range
    entries, and zero, one and full ``topk_length`` values; workspace and outputs are poisoned before
    every launch, and the bytes after the scratch keep their sentinel.
    """
    try:
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        return pytest.skip("Environment not supported: cuda-python not installed")
    _require_h128_two_cta()

    s_kv = _H128_CONFLICT_RANK_S_KV[head_dim][layout][topk]
    s_q = 240 if (head_dim == 576 and topk == 512) else 160
    _check_h128_deterministic_case(
        head_dim, s_q, s_kv, topk, has_topk_length=has_topk_length, layout=layout, repeats=4, label=f"H128/D{head_dim} {layout} K={topk}"
    )


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("head_dim", [512, 576], ids=["d512", "d576"])
@torch_fork_set_rng(seed=20261106)
def test_DSA_sparse_attention_backward_h128_shards_mode_runs_private_plane(head_dim):
    """A ``deterministic_dkv_mode='shards'`` H128 plan runs the private-plane layout where the default plan stores dKV
    in conflict-rank storage: exact private-plane scratch, reference gradients, bitwise reruns, untouched scratch tail."""
    try:
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        return pytest.skip("Environment not supported: cuda-python not installed")
    _require_h128_two_cta()

    topk = 128
    s_kv = _H128_CONFLICT_RANK_S_KV[head_dim]["conflict_rank"][topk]
    assert _h128_conflict_rank_selected(head_dim, s_kv, topk)
    _check_h128_deterministic_case(
        head_dim,
        160,
        s_kv,
        topk,
        has_topk_length=True,
        layout="private_plane",
        repeats=4,
        label=f"H128/D{head_dim} shards K={topk}",
        deterministic_dkv_mode="shards",
    )


@pytest.mark.L0
@pytest.mark.parametrize(
    "head_dim,topk,s_q,s_kv",
    [
        pytest.param(512, 128, 160, 4096, id="d512-k128-fused"),
        pytest.param(576, 1024, 10, 2097, id="d576-k1024-sorted-rows"),
    ],
)
@torch_fork_set_rng(seed=20261108)
def test_DSA_sparse_attention_backward_h128_conflict_rank_smoke(head_dim, topk, s_q, s_kv):
    """One conflict-rank plan per head dim at the default level.

    D512: fused waves (each main kernel runs the previous wave's tail and prepares the next wave) over
    three waves. D576: one-round waves with sorted rows, whose first main kernel reads the rows the
    leading setup launch sorted.
    """
    try:
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        return pytest.skip("Environment not supported: cuda-python not installed")
    _require_h128_two_cta()
    if head_dim == 512:
        interface = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100")
        assert interface._get_sm100_kernel_class("h128_2cta_m64", True, 1, s_kv, 512, 128, None, max_topk=topk).rank_fused
    else:
        interface = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100_d576")
        assert interface._rank_shard_rounds(s_kv, topk, True, 128) == 1
    _check_h128_deterministic_case(head_dim, s_q, s_kv, topk, has_topk_length=True, layout="conflict_rank", repeats=2, label=f"H128/D{head_dim} smoke K={topk}")


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize(
    "topk,s_q,s_kv,rounds",
    [
        pytest.param(512, 240, 16384, 3, id="k512-three-rounds"),
        pytest.param(1024, 10, 65536, 1, id="k1024-one-round"),
    ],
)
@torch_fork_set_rng(seed=20261109)
def test_DSA_sparse_attention_backward_d576_conflict_rank_in_kernel_odo(topk, s_q, s_kv, rounds):
    """D576 conflict-rank plans past the O*dO precompute budget reduce O.dO inside the main kernel."""
    try:
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        return pytest.skip("Environment not supported: cuda-python not installed")
    _require_h128_two_cta()
    interface = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100_d576")
    assert interface._rank_shard_rounds(s_kv, topk, True, 128) == rounds
    assert not interface._odo_precompute(s_kv, topk, True, 128)
    _check_h128_deterministic_case(576, s_q, s_kv, topk, has_topk_length=True, layout="conflict_rank", repeats=3, label=f"H128/D576 in-kernel O*dO K={topk}")


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize(
    "head_dim,topk,s_q,s_kv,rounds",
    [
        pytest.param(512, 512, 96, 4096, None, id="d512-k512"),
        pytest.param(512, 2048, 10, 4096, None, id="d512-k2048"),
        # One wave: 96 queries of a 222-query three-round wave (the deterministic test runs two waves).
        pytest.param(576, 512, 96, 4096, 3, id="d576-k512-three-rounds"),
        pytest.param(576, 1024, 10, 2097, 1, id="d576-k1024-sorted-rows"),
        pytest.param(576, 1152, 10, 2097, 1, id="d576-k1152-sorted-rows"),
    ],
)
@pytest.mark.parametrize("has_topk_length", [True, False], ids=["lengths", "full"])
@torch_fork_set_rng(seed=20261105)
def test_DSA_sparse_attention_backward_h128_conflict_rank_back_to_back_inputs(head_dim, topk, s_q, s_kv, rounds, has_topk_length):
    """Back-to-back calls with different inputs on one workspace each match their own reference.

    Repeating identical inputs would mask a launch that reads the previous call's rows or records,
    so every call takes new indices, and the last one is rerun on poisoned scratch for a bitwise check.
    ``rounds`` is the D576 wave round count the shapes select.
    """
    try:
        from cuda.bindings import driver as cuda
    except ImportError:
        return pytest.skip("Environment not supported: cuda-python not installed")
    _require_h128_two_cta()
    assert _h128_conflict_rank_selected(head_dim, s_kv, topk)
    if rounds is not None:
        interface = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100_d576")
        assert interface._rank_shard_rounds(s_kv, topk, True, 128) == rounds

    calls = [_h128_conflict_rank_inputs(head_dim, s_q, s_kv, topk, has_topk_length=has_topk_length) for _ in range(3)]
    plan, workspace, outputs, tail = _h128_deterministic_plan(head_dim, *calls[0])
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    bitwise_bits(workspace).fill_(0xA5)
    results = []
    for inputs, topk_length, softmax_scale in calls:
        results.append(tuple(tensor.clone() for tensor in _h128_execute(plan, inputs, outputs, topk_length, softmax_scale, workspace, stream)))
    torch.cuda.synchronize()
    for call, ((inputs, topk_length, softmax_scale), result) in enumerate(zip(calls, results)):
        q, kv, out, dout, lse, attn_sink, topk_idxs = inputs
        try:
            check_ref_dsa_sparse_attention_backward(
                q, kv, attn_sink, topk_idxs, out, dout, lse, *result, softmax_scale=softmax_scale, topk_length=topk_length, atol=5e-2, rtol=5e-2
            )
        except AssertionError as error:
            raise AssertionError(f"back-to-back call {call} differs from its reference") from error
    inputs, topk_length, softmax_scale = calls[-1]
    bitwise_bits(workspace).fill_(0xA5)
    rerun = _h128_execute(plan, inputs, outputs, topk_length, softmax_scale, workspace, stream)
    torch.cuda.synchronize()
    for name, actual, expected in zip(("dQ", "dKV", "dSink"), rerun, results[-1]):
        assert torch.equal(bitwise_bits(actual), bitwise_bits(expected)), f"{name} depends on the previous call's scratch"
    _assert_h128_scratch_tail_untouched(tail)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("head_dim,topk", [(512, 512), (512, 1024), (576, 512), (576, 1024)], ids=["d512-k512", "d512-k1024", "d576-k512", "d576-k1024"])
@torch_fork_set_rng(seed=20261106)
def test_DSA_sparse_attention_backward_h128_conflict_rank_cuda_graph_replay(head_dim, topk):
    """A captured conflict-rank plan replays new inputs written into the captured buffers bitwise like eager calls."""
    try:
        from cuda.bindings import driver as cuda
    except ImportError:
        return pytest.skip("Environment not supported: cuda-python not installed")
    _require_h128_two_cta()

    s_q, s_kv = 96, 4096
    assert _h128_conflict_rank_selected(head_dim, s_kv, topk)
    inputs, topk_length, softmax_scale = _h128_conflict_rank_inputs(head_dim, s_q, s_kv, topk, has_topk_length=True)
    plan, workspace, outputs, tail = _h128_deterministic_plan(head_dim, inputs, topk_length, softmax_scale)
    launch_stream = torch.cuda.Stream()
    launch_stream.wait_stream(torch.cuda.current_stream())
    stream = cuda.CUstream(launch_stream.cuda_stream)

    with torch.cuda.stream(launch_stream):
        _h128_execute(plan, inputs, outputs, topk_length, softmax_scale, workspace, stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=launch_stream):
            _h128_execute(plan, inputs, outputs, topk_length, softmax_scale, workspace, stream)
        for replay in range(2):
            new_inputs, new_topk_length, _ = _h128_conflict_rank_inputs(head_dim, s_q, s_kv, topk, has_topk_length=True)
            # The new inputs were produced on the current stream; the captured buffers are rewritten and
            # replayed on the launch stream after them.
            launch_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(launch_stream):
                for buffer, value in zip(inputs, new_inputs):
                    buffer.copy_(value)
                topk_length.copy_(new_topk_length)
                for buffer in (workspace, *outputs):
                    bitwise_bits(buffer).fill_(0xA5)
                graph.replay()
            torch.cuda.synchronize()
            replayed = tuple(tensor.clone() for tensor in outputs)
            with torch.cuda.stream(launch_stream):
                for buffer in (workspace, *outputs):
                    bitwise_bits(buffer).fill_(0xA5)
                eager = _h128_execute(plan, inputs, outputs, topk_length, softmax_scale, workspace, stream)
            torch.cuda.synchronize()
            for name, actual, expected in zip(("dQ", "dKV", "dSink"), replayed, eager):
                assert torch.equal(bitwise_bits(actual), bitwise_bits(expected)), f"{name} graph replay {replay} differs from eager"
            q, kv, out, dout, lse, attn_sink, topk_idxs = inputs
            check_ref_dsa_sparse_attention_backward(
                q, kv, attn_sink, topk_idxs, out, dout, lse, *replayed, softmax_scale=softmax_scale, topk_length=topk_length, atol=5e-2, rtol=5e-2
            )
        _assert_h128_scratch_tail_untouched(tail)
    finally:
        graph.reset()


@pytest.mark.L2
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("head_dim,topk,s_kv", [(512, 1024, 131072), (576, 128, 131072)], ids=["d512-few-planes", "d576-two-rounds"])
@torch_fork_set_rng(seed=20261107)
def test_DSA_sparse_attention_backward_h128_conflict_rank_long_kv(head_dim, topk, s_kv):
    """Long KV extents: the few-plane D512 plan and the two-round top-k 128 D576 plan stay exact and reproducible."""
    try:
        from cuda.bindings import driver as cuda  # noqa: F401
    except ImportError:
        return pytest.skip("Environment not supported: cuda-python not installed")
    _require_h128_two_cta()
    if head_dim == 512:
        interface = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100")
        assert getattr(interface._get_sm100_kernel_class("h128_2cta_m64", True, 1, s_kv, 512, 128, None, max_topk=topk), "rank_few_planes", False)
    else:
        interface = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100_d576")
        assert interface._rank_shard_rounds(s_kv, topk, True, 128) == 2

    _check_h128_deterministic_case(
        head_dim, 160, s_kv, topk, has_topk_length=True, layout="conflict_rank", repeats=3, label=f"H128/D{head_dim} long KV K={topk}"
    )


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


# Conflict-rank records are the default deterministic dKV layout at these shapes. The relay tests below select
# the shard and cluster-relay layout with deterministic_dkv_mode="shards", whose automatic relay width they check;
# the explicit-relay tests pass deterministic_relay_cluster_size, which selects the relay in every mode but 'chain'.
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
            deterministic_dkv_mode="shards",
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
            deterministic_dkv_mode="shards",
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
            deterministic_dkv_mode="shards",
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
            deterministic_dkv_mode="shards",
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
            deterministic_dkv_mode="shards",
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
            deterministic_dkv_mode="shards",
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
            deterministic_dkv_mode="shards",
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
            deterministic_dkv_mode="shards",
            dtype=torch.float16,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")


def _deterministic_case_inputs(num_heads, head_dim, s_q, s_kv, topk, *, dtype=torch.bfloat16, shared_topk=False, duplicate_topk=False, invalid_topk=False):
    """One deterministic input set and its forward reference; ``topk_length`` holds a full and an empty row."""
    device = torch.device("cuda")
    softmax_scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(s_q, num_heads, head_dim, dtype=dtype, device=device) / 10
    kv = torch.randn(s_kv, head_dim, dtype=dtype, device=device) / 10
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    if shared_topk:
        topk_idxs = torch.randperm(s_kv, device=device)[:topk].repeat(s_q, 1).to(torch.int32)
    else:
        topk_idxs = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    if duplicate_topk and topk > 1:
        topk_idxs[::2, 1] = topk_idxs[::2, 0]
    if invalid_topk and topk > 2:
        topk_idxs[1::3, -1] = -1
        topk_idxs[2::3, -2] = s_kv + 3
    topk_length = torch.randint(1, topk + 1, (s_q,), dtype=torch.int32, device=device)
    topk_length[0] = topk
    if s_q > 1:
        topk_length[s_q // 2] = 0
    out, lse = ref_sparse_attention_forward_chunked(q, kv, attn_sink, topk_idxs, topk_length=topk_length, softmax_scale=softmax_scale)
    dout = torch.randn_like(out)
    return (q, kv, out, dout, lse, attn_sink, topk_idxs), topk_length, softmax_scale


@pytest.mark.L0
def test_DSA_sparse_attention_backward_deterministic_dkv_mode_public_contract():
    """Records stay the default deterministic dKV mode of the plan and the wrapper."""
    from cudnn import DSA

    wrapper = inspect.signature(DSA.sparse_attention_backward_wrapper).parameters
    plan = inspect.signature(DSA.SparseAttentionBackward).parameters
    assert wrapper["deterministic_dkv_mode"].default == "records"
    assert plan["deterministic_dkv_mode"].default == "records"
    # The new option follows every positional parameter of the existing signatures.
    assert list(wrapper)[-1] == "deterministic_dkv_mode"
    assert list(plan)[-1] == "deterministic_dkv_mode"


@pytest.mark.L0
def test_DSA_sparse_attention_backward_sm100_deterministic_dkv_mode_selection():
    """Records are the default generic M64 layout where they fit; chain mode applies where the chain relay fits."""
    deterministic = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_deterministic")
    interface = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100")
    select = deterministic.deterministic_kernel_class
    records = deterministic.FlashAttentionDSABackwardSm100DeterministicRankShard
    chain = deterministic.FlashAttentionDSABackwardSm100DeterministicRankChain

    # Records whenever their workspace fits the shard or relay layout; without a top-k width the layout is unchanged.
    assert select(300, 4096, 512, 64, max_topk=64) is records
    assert select(300, 4096, 512, 64) is deterministic.FlashAttentionDSABackwardSm100Deterministic
    assert select(257, 65537, 576, 64, max_topk=511) is records
    assert select(257, 65537, 576, 64) is deterministic.FlashAttentionDSABackwardSm100DeterministicCluster8
    # Plans whose records would be larger keep the shard layout.
    assert select(8, 256, 512, 64, max_topk=64) is deterministic.FlashAttentionDSABackwardSm100Deterministic32
    assert select(300, 4096, 512, 64, max_topk=2048) is deterministic.FlashAttentionDSABackwardSm100Deterministic
    # Chain mode: the chain relay where it applies, the default selection elsewhere.
    assert select(300, 4096, 512, 64, max_topk=64, dkv_mode="chain") is chain
    assert select(300, 4096, 576, 96, max_topk=64, dkv_mode="chain") is chain
    assert select(148, 278528, 512, 64, max_topk=64, dkv_mode="chain") is records
    assert select(8, 256, 512, 64, max_topk=64, dkv_mode="chain") is deterministic.FlashAttentionDSABackwardSm100Deterministic32
    # An explicit relay selects that relay in records mode and is rejected in chain mode.
    assert select(257, 65537, 512, 64, deterministic_relay_cluster_size=4, max_topk=129) is deterministic.FlashAttentionDSABackwardSm100DeterministicCluster4
    with pytest.raises(ValueError, match="cannot be combined with an explicit deterministic_relay_cluster_size"):
        select(257, 65537, 512, 64, deterministic_relay_cluster_size=4, max_topk=129, dkv_mode="chain")
    with pytest.raises(ValueError, match="cannot be combined with an explicit deterministic_relay_cluster_size"):
        interface.flash_attn_bwd_sm100_workspace_size(
            257, 65537, 512, 64, deterministic=True, backend="generic_m64", deterministic_relay_cluster_size=4, max_topk=129, dkv_mode="chain"
        )
    # Cache-key fields: the plane count, relay width and dKV layout reconstruct each class.
    assert (records.num_dkv_shards, records.dkv_relay_cluster_size, records.dkv_layout) == (1, 1, "records")
    assert (chain.num_dkv_shards, chain.dkv_relay_cluster_size, chain.dkv_layout) == (1, 1, "chain")
    assert deterministic._deterministic_kernel_class_for_config(1, 1, "records") is records
    assert deterministic._deterministic_kernel_class_for_config(1, 1, "chain") is chain
    assert deterministic._deterministic_kernel_class_for_config(32, 1) is deterministic.FlashAttentionDSABackwardSm100Deterministic32
    assert deterministic._deterministic_kernel_class_for_config(16, 8, "relay") is deterministic.FlashAttentionDSABackwardSm100DeterministicCluster8
    for config in ((1, 1, "planes"), (32, 1, "records"), (32, 1, "relay"), (16, 8, "shards")):
        with pytest.raises(ValueError):
            deterministic._deterministic_kernel_class_for_config(*config)
    # Deterministic H96 D576 outside the two-CTA envelope runs the generic kernel where conflict-rank storage applies.
    assert interface._select_sm100_backend(96, 576, max_topk=64, deterministic=True, total_s_q=300, total_s_kv=4096) == ("generic_m64", 64)
    assert interface._get_sm100_kernel_class("generic_m64", True, 300, 4096, 576, 96, max_topk=64) is records
    assert interface._get_sm100_kernel_class("generic_m64", True, 300, 4096, 576, 96, max_topk=64, dkv_mode="chain") is chain


@pytest.mark.L0
@pytest.mark.parametrize("deterministic_dkv_mode", ["records", "chain"])
def test_DSA_sparse_attention_backward_sm100_rank_workspace_within_shard_or_relay_layout(deterministic_dkv_mode):
    """Neither dKV mode needs more scratch than the plan's ``'shards'`` layout.

    That is the generic shard or relay layout; for chain-mode H96 D576 it is the fixed 128-plane layout of the
    H64 + H32 composition, which runs deterministic H96 D576 outside the two-CTA envelope in ``'shards'`` mode.
    """
    interface = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100")
    size = interface.flash_attn_bwd_sm100_workspace_size
    for num_heads in (16, 64, 96, 128):
        for head_dim in (512, 576):
            for s_q in (1, 147, 149, 300, 1037, 4096):
                for s_kv in (100, 4096, 65537, 278528):
                    for topk in (1, 64, 511, 2048):
                        common = (s_q, s_kv, head_dim, num_heads)
                        layout = size(*common, deterministic=True, backend="generic_m64")
                        if deterministic_dkv_mode == "chain" and (num_heads, head_dim) == (96, 576):
                            layout = size(*common, deterministic=True, backend="h96_h64_h32")
                        actual = size(*common, deterministic=True, backend="generic_m64", max_topk=topk, dkv_mode=deterministic_dkv_mode)
                        assert actual <= layout, f"H{num_heads} D{head_dim} S_q={s_q} S_kv={s_kv} K={topk}: {actual} > {layout}"


def _integer_device_methods(module, class_name, names):
    """Compile the named methods of a CuTe DSL class as plain Python over integers.

    The conflict-rank geometry methods use only integer arithmetic, ``min``/``max``, ``cute.round_up`` and
    ``const_expr``: with the DSL scalar types replaced by ``int`` they compute on the host what the kernels
    compute for a runtime KV extent.
    """
    tree = ast.parse(inspect.getsource(module))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    device = ast.parse("class IntegerDevice:\n    pass").body[0]
    device.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert sorted(node.name for node in device.body) == sorted(names)
    for node in device.body:
        node.decorator_list = []
    scope = dict(
        vars(module),
        Int32=int,
        Int64=int,
        cute=types.SimpleNamespace(round_up=lambda value, multiple: -(-value // multiple) * multiple),
        cutlass=types.SimpleNamespace(const_expr=bool),
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=[device], type_ignores=[])), f"<{class_name} on integers>", "exec"), scope)
    return scope["IntegerDevice"]


@pytest.mark.L0
def test_DSA_sparse_attention_backward_sm100_rank_device_geometry_matches_host_sizing():
    """The kernels' plane geometry and workspace regions agree with the host functions that size the workspace.

    ``rank_geometry`` repeats the host plane-count rule for the runtime KV extent (records: ``rank_m64_planes``;
    chain: ``rank_chain_planes(..., records=False)``), and the device region offsets end where the host workspace
    size does (records: before the split pairs' dQ partials). The device methods run here on integers.
    """
    shard = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_rank_shard_m64")
    chain = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_rank_chain_m64")
    records_device = _integer_device_methods(shard, "RankShardM64Reduction", ("rank_geometry", "rank_slot_stride", "rank_offsets", "rank_high", "rank_flags"))
    chain_device = _integer_device_methods(
        chain,
        "RankChainM64Reduction",
        (
            "rank_geometry",
            "rank_wave",
            "rank_state_words",
            "rank_offsets",
            "rank_high",
            "rank_chain_offsets",
            "rank_tile_flags",
            "rank_hot_depths",
            "rank_hot_list",
            "rank_hot_rows",
            "rank_hot_slots",
        ),
    )
    shapes = itertools.product((16, 64, 96, 128), (512, 576), (1, 3, 64, 127, 512, 2047, 2048), (1, 148, 149, 4096), (1, 100, 4096, 65537, 524288))
    for num_heads, head_dim, topk, s_q, s_kv in shapes:
        if not 0 < shard.RANK_M64_WAVE * topk < 1 << shard.RANK_M64_FIELD_BITS:
            continue
        case = f"H{num_heads} D{head_dim} K={topk} S_q={s_q} S_kv={s_kv}"
        wave = shard.rank_m64_wave(num_heads, head_dim, topk, s_q, s_kv)
        split = shard.rank_m64_split(num_heads, head_dim, topk, s_q, s_kv)
        records = records_device()
        records.head_dim, records.max_topk, records.q_wave_ctas = head_dim, topk, wave
        pitch, planes = records.rank_geometry(s_kv)
        assert (pitch, planes) == (shard.rank_m64_pitch(s_kv), shard.rank_m64_planes(s_kv, head_dim, topk, wave)), case
        assert all(offset % 8 == 0 for offset in records.rank_offsets(pitch, planes)), case
        end = records.rank_flags(pitch, planes) + shard.RANK_M64_FLAG_WORDS
        if split:
            end = (end + 3) // 4 * 4 + wave * 64 * shard.rank_m64_record(head_dim)
        words = shard.rank_m64_workspace_words(s_kv, head_dim, topk, wave, split)
        assert end <= words if split else end == words, case
        relay = chain_device()
        relay.head_dim, relay.max_topk = head_dim, topk
        pitch, planes = relay.rank_geometry(s_kv)
        assert (pitch, planes) == (shard.rank_m64_pitch(s_kv), chain.rank_chain_planes(s_kv, head_dim, topk, records=False)), case
        end = relay.rank_hot_list(pitch, planes) + 2 * relay.rank_hot_rows()
        assert end == chain.rank_chain_workspace_words(s_kv, head_dim, topk, records=False), case


@pytest.mark.L0
def test_DSA_sparse_attention_backward_sm100_records_plan_ignores_query_extent():
    """A records plan's compile-time wave configuration does not follow the query count.

    The wave size and the split rule depend on the head count, head dimension, top-k width and KV extent; only
    the strided wave order switches on, once a plan has ``RANK_M64_STRIDED_MIN_WAVES`` waves.
    """
    shard = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_rank_shard_m64")
    shapes = ((64, 512, 64, 4096), (64, 576, 2048, 65537), (96, 576, 64, 4096), (128, 512, 1536, 8192), (16, 576, 128, 278528), (64, 576, 2048, 278528))
    assert shard.rank_m64_split(64, 576, 2048, 1, 278528)  # a split plan: half waves at every query count
    for num_heads, head_dim, topk, s_kv in shapes:
        wave = shard.rank_m64_wave(num_heads, head_dim, topk, 1, s_kv)
        assert wave in (shard.RANK_M64_WAVE, shard.RANK_M64_WAVE // 2)
        last_unstrided = (shard.RANK_M64_STRIDED_MIN_WAVES - 1) * wave
        for extents in ((1, 2, wave - 1, wave, wave + 1, last_unstrided), (last_unstrided + 1, 4 * last_unstrided, 65536)):
            keys = {
                (shard.rank_m64_wave(num_heads, head_dim, topk, s_q, s_kv), shard.rank_m64_mode(num_heads, head_dim, topk, s_q, s_kv, wave)) for s_q in extents
            }
            assert len(keys) == 1, f"H{num_heads} D{head_dim} K={topk} S_kv={s_kv}: {sorted(keys)} over S_q {extents}"


@pytest.mark.L0
def test_DSA_sparse_attention_backward_sm100_deterministic_dkv_mode_validation():
    """Every layer that takes the dKV mode checks it, without a GPU.

    Unknown names are rejected, the non-default modes need ``deterministic=True``, ``'chain'`` is rejected on the
    backends that never run the generic M64 kernel and together with an explicit relay width, and ``'shards'``
    selects exactly the class a plan without a top-k width selects (the shard or cluster-relay layout).
    """
    common = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100_common")
    interface = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100")
    deterministic = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_deterministic")
    error = common.deterministic_dkv_mode_error
    for mode in ("records", "chain", "shards"):
        assert error(mode, True) is None
    assert error("records", False) is None
    for mode in ("chain", "shards"):
        assert error(mode, False) == f"deterministic_dkv_mode={mode!r} requires deterministic=True"
    for value in ("planes", "Records", None, 0):
        assert error(value, True).startswith("deterministic_dkv_mode must be one of")
    for backend in ("generic_m64", "h96_h64_h32"):
        assert error("chain", True, backend) is None
    for backend in ("h16_m128", "h32_m64", "h128_2cta_m64", "h96_d576_2cta_m64", "h128_d576_2cta_m64"):
        assert error("chain", True, backend) == "deterministic_dkv_mode='chain' is supported only by the deterministic SM100 generic_m64 backend"
        assert error("shards", True, backend) is None
        assert error("records", True, backend) is None

    size = interface.flash_attn_bwd_sm100_workspace_size
    with pytest.raises(ValueError, match="deterministic_dkv_mode must be one of"):
        size(300, 4096, 512, 64, deterministic=True, backend="generic_m64", max_topk=64, dkv_mode="planes")
    with pytest.raises(ValueError, match="requires deterministic=True"):
        size(300, 4096, 512, 64, deterministic=False, backend="generic_m64", max_topk=64, dkv_mode="shards")
    for backend, num_heads in (("h128_2cta_m64", 128), ("h96_d576_2cta_m64", 96), ("h128_d576_2cta_m64", 128)):
        with pytest.raises(ValueError, match="supported only by the deterministic SM100 generic_m64 backend"):
            size(257, 256, 576, num_heads, deterministic=True, backend=backend, max_topk=128, dkv_mode="chain")
        with pytest.raises(ValueError, match="supported only by the deterministic SM100 generic_m64 backend"):
            interface._get_sm100_kernel_class(backend, True, 257, 256, 576, num_heads, max_topk=128, dkv_mode="chain")
    with pytest.raises(ValueError, match="deterministic_dkv_mode must be one of"):
        interface._get_sm100_kernel_class("generic_m64", True, 300, 4096, 512, 64, max_topk=64, dkv_mode="planes")
    with pytest.raises(ValueError, match="cannot be combined with an explicit deterministic_relay_cluster_size"):
        interface._get_sm100_kernel_class("generic_m64", True, 257, 65537, 512, 64, 8, max_topk=129, dkv_mode="chain")
    with pytest.raises(ValueError, match="deterministic_dkv_mode must be one of"):
        deterministic.deterministic_kernel_class(300, 4096, 512, 64, max_topk=64, dkv_mode="planes")
    with pytest.raises(ValueError, match="deterministic_dkv_mode must be one of"):
        deterministic.rank_dkv_route(300, 4096, 512, 64, 64, dkv_mode="planes")

    select = deterministic.deterministic_kernel_class
    for shape in ((8, 256, 512, 64), (300, 4096, 512, 64), (300, 4096, 576, 96), (257, 65537, 576, 64), (8192, 1048576, 512, 64)):
        for topk in (64, 511, 2048):
            assert deterministic.rank_dkv_route(*shape, topk, dkv_mode="shards") is None
            assert select(*shape, max_topk=topk, dkv_mode="shards") is select(*shape)
    # An explicit relay selects that relay in 'records' and 'shards' mode.
    for mode in ("records", "shards"):
        assert select(257, 65537, 512, 64, 16, max_topk=129, dkv_mode=mode) is deterministic.FlashAttentionDSABackwardSm100DeterministicCluster16
    # Deterministic H96 D576 outside the two-CTA envelope: 'shards' keeps the H64 + H32 composition.
    for mode in ("records", "shards"):
        expected = ("generic_m64", 64) if mode == "records" else ("h96_h64_h32", 64)
        assert interface._select_sm100_backend(96, 576, max_topk=64, deterministic=True, total_s_q=300, total_s_kv=4096, dkv_mode=mode) == expected


@pytest.mark.L0
def test_DSA_sparse_attention_backward_dkv_mode_rejects_before_staging(monkeypatch):
    """Invalid dKV mode requests fail in the wrapper before it stages any input (no GPU needed)."""
    api = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward.api")

    def forbidden_staging(*args, **kwargs):
        raise AssertionError("an invalid dKV mode reached input staging")

    monkeypatch.setattr(api, "contiguous_on_stream", forbidden_staging)
    args = (torch.empty(0), None, None, None, None, None, None)
    with pytest.raises(ValueError, match="deterministic_dkv_mode must be one of"):
        api.sparse_attention_backward_wrapper(*args, deterministic=True, deterministic_dkv_mode="planes")
    for mode in ("chain", "shards"):
        with pytest.raises(ValueError, match=f"deterministic_dkv_mode='{mode}' requires deterministic=True"):
            api.sparse_attention_backward_wrapper(*args, deterministic=False, deterministic_dkv_mode=mode)
    with pytest.raises(ValueError, match="cannot be combined with an explicit deterministic_relay_cluster_size"):
        api.sparse_attention_backward_wrapper(*args, deterministic=True, deterministic_relay_cluster_size=8, deterministic_dkv_mode="chain")


@pytest.mark.L0
def test_DSA_sparse_attention_backward_wrapper_caches_one_plan_per_dkv_mode(monkeypatch):
    """The wrapper's plan cache keeps one plan per dKV mode; the default mode's key carries no mode entry."""
    if not torch.cuda.is_available():
        pytest.skip("Environment not supported: CUDA device required")
    api = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward.api")
    built = []

    class FakePlan:
        """Stands in for SparseAttentionBackward: records the requested mode, compiles nothing."""

        def __init__(self, **kwargs):
            self.mode = kwargs["deterministic_dkv_mode"]
            built.append(self.mode)

        def check_support(self):
            return True

        def compile(self):
            pass

        def scratch_workspace_bytes(self):
            return 0

        def execute(self, q, kv, out, dout, lse, attn_sink, topk_idxs, dq, dkv, **kwargs):
            return dq, dkv, kwargs["d_sink"]

    monkeypatch.setattr(api, "SparseAttentionBackward", FakePlan)
    monkeypatch.setattr(api, "_cache_of_SparseAttentionBackwardObjects", {})
    device = torch.device("cuda")
    q = torch.zeros(2, 64, 512, dtype=torch.bfloat16, device=device)
    inputs = (
        q,
        torch.zeros(64, 512, dtype=torch.bfloat16, device=device),
        torch.zeros_like(q),
        torch.zeros_like(q),
        torch.zeros(2, 64, dtype=torch.float32, device=device),
        torch.zeros(64, dtype=torch.float32, device=device),
        torch.zeros(2, 64, dtype=torch.int32, device=device),
    )
    for mode in ("records", "chain", "shards") * 2:
        api.sparse_attention_backward_wrapper(*inputs, softmax_scale=0.05, deterministic=True, deterministic_dkv_mode=mode)
    assert built == ["records", "chain", "shards"]
    keys = {plan.mode: key for key, plan in api._cache_of_SparseAttentionBackwardObjects.items()}
    assert len(keys) == len(api._cache_of_SparseAttentionBackwardObjects) == 3
    assert keys["chain"] == keys["records"] + ("chain",)
    assert keys["shards"] == keys["records"] + ("shards",)


@pytest.mark.L0
def test_DSA_sparse_attention_backward_sm90_rejects_chain_dkv_mode():
    """SM90 has no generic M64 kernel: a chain request fails check_support instead of being ignored."""
    _require_sm90()
    from cudnn import DSA

    device = torch.device("cuda")
    q = torch.randn(2, 64, 512, dtype=torch.bfloat16, device=device)
    kv = torch.randn(64, 512, dtype=torch.bfloat16, device=device)
    out = torch.randn(2, 64, 512, dtype=torch.bfloat16, device=device)
    lse = torch.zeros(2, 64, dtype=torch.float32, device=device)
    attn_sink = torch.zeros(64, dtype=torch.float32, device=device)
    topk_idxs = torch.zeros(2, 64, dtype=torch.int32, device=device)
    plan = DSA.SparseAttentionBackward(q, kv, out, out, lse, attn_sink, topk_idxs, softmax_scale=0.05, deterministic=True, deterministic_dkv_mode="chain")
    with pytest.raises(ValueError, match="supported only by the deterministic SM100 generic_m64 backend"):
        plan.check_support()


@pytest.mark.L0
def test_DSA_sparse_attention_backward_sm100_chain_mode_rejects_invalid_requests():
    """Non-default modes are deterministic-only, chain mode never drops a requested relay width, and the mode name is checked."""
    from cudnn import DSA

    _require_sm100()
    device = torch.device("cuda")
    q = torch.randn(2, 64, 512, dtype=torch.bfloat16, device=device)
    kv = torch.randn(64, 512, dtype=torch.bfloat16, device=device)
    out = torch.randn(2, 64, 512, dtype=torch.bfloat16, device=device)
    lse = torch.zeros(2, 64, dtype=torch.float32, device=device)
    attn_sink = torch.zeros(64, dtype=torch.float32, device=device)
    topk_idxs = torch.zeros(2, 64, dtype=torch.int32, device=device)
    cases = (
        (dict(deterministic=True, deterministic_relay_cluster_size=4, deterministic_dkv_mode="chain"), "cannot be combined with an explicit"),
        (dict(deterministic=False, deterministic_dkv_mode="chain"), "requires deterministic=True"),
        (dict(deterministic=False, deterministic_dkv_mode="shards"), "requires deterministic=True"),
        (dict(deterministic=True, deterministic_dkv_mode="planes"), "deterministic_dkv_mode must be one of"),
    )
    for options, message in cases:
        plan = DSA.SparseAttentionBackward(q, kv, out, out, lse, attn_sink, topk_idxs, softmax_scale=0.05, **options)
        with pytest.raises(ValueError, match=message):
            plan.check_support()


@pytest.mark.L0
def test_DSA_sparse_attention_backward_conflict_rank_capabilities():
    """Conflict-rank dKV storage is provided only on compute capability 10.0 and 10.3 (no GPU needed).

    On 10.7 a deterministic ``'records'`` request resolves to ``'shards'`` (the backend, kernel class and workspace
    of the ``'shards'`` plan) and ``'chain'`` is rejected by name of the compute capability; ``None`` stands for an
    SM100/SM103 device, where every request is unchanged.
    """
    common = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100_common")
    interface = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100")
    effective, error = common.effective_deterministic_dkv_mode, common.deterministic_dkv_mode_error
    for capability in (None, (10, 0), (10, 3)):
        for mode in ("records", "chain", "shards"):
            assert effective(mode, True, capability) == mode
            assert error(mode, True, device_capability=capability) is None
    assert effective("records", True, (10, 7)) == effective("shards", True, (10, 7)) == "shards"
    assert effective("records", False, (10, 7)) == "records"
    for mode in ("records", "shards"):
        assert error(mode, True, device_capability=(10, 7)) is None
    assert error("chain", True, device_capability=(10, 7)).endswith("found compute capability 10.7")

    size = interface.flash_attn_bwd_sm100_workspace_size
    select = interface._get_sm100_kernel_class
    for s_q, s_kv, head_dim, num_heads, topk in ((300, 4096, 512, 64, 64), (257, 65537, 576, 64, 511), (148, 278528, 512, 64, 64)):
        shape, options = (s_q, s_kv, head_dim, num_heads, True), dict(backend="generic_m64", max_topk=topk)
        records, shards = size(*shape, **options), size(*shape, **options, dkv_mode="shards")
        # These shapes store dKV in conflict-rank records on SM100/SM103.
        assert select("generic_m64", True, s_q, s_kv, head_dim, num_heads, max_topk=topk).dkv_layout == "records"
        assert records < shards
        for capability in ((10, 0), (10, 3)):
            assert size(*shape, **options, device_capability=capability) == records
        try:
            shards_cls = select("generic_m64", True, s_q, s_kv, head_dim, num_heads, max_topk=topk, dkv_mode="shards")
        except RuntimeError as relay_floor:
            # The cluster relay that some of these shapes select keeps its own CuTe DSL floor.
            assert "cluster relay requires nvidia-cutlass-dsl" in str(relay_floor)
            shards_cls = None
        for mode in ("records", "shards"):
            assert size(*shape, **options, dkv_mode=mode, device_capability=(10, 7)) == shards
            if shards_cls is not None:
                assert select("generic_m64", True, s_q, s_kv, head_dim, num_heads, max_topk=topk, dkv_mode=effective(mode, True, (10, 7))) is shards_cls
        with pytest.raises(ValueError, match=r"found compute capability 10\.7"):
            size(*shape, **options, dkv_mode="chain", device_capability=(10, 7))
    # Deterministic H96 D576 outside the two-CTA envelope keeps the H64 + H32 composition on 10.7.
    h96 = dict(head_dim_v=512, dtype=torch.bfloat16, max_topk=64, deterministic=True, total_s_q=300, total_s_kv=4096)
    for capability in ((10, 0), (10, 3)):
        assert interface._select_sm100_backend(96, 576, **h96, device_capability=capability) == ("generic_m64", 64)
    assert interface._select_sm100_backend(96, 576, **h96, device_capability=(10, 7)) == ("h96_h64_h32", 64)
    assert interface._select_sm100_backend(96, 576, **h96, device_capability=(10, 7), dkv_mode="shards") == ("h96_h64_h32", 64)


@pytest.mark.L0
def test_DSA_sparse_attention_backward_sm107_plans_keep_shard_layouts(monkeypatch):
    """Plans on compute capability 10.7 (mocked; nothing compiles) run ``'records'`` as ``'shards'`` and reject ``'chain'``.

    The default plan's backend, dKV route, address width, scratch size and compiled configuration equal those of the
    ``'shards'`` plan, and ``'chain'`` fails ``check_support()`` and the direct ``flash_attn_bwd_sm100`` entry with
    an error that names the compute capability, before any compilation.
    """
    if not torch.cuda.is_available():
        pytest.skip("Environment not supported: CUDA device required")
    api = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward.api")
    interface = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100")
    compiled = []

    def record_compile(*args, **kwargs):
        """Record the deterministic compile signature instead of compiling."""
        compiled.append((args, kwargs))
        return object()

    def forbidden_compile(*args, **kwargs):
        raise AssertionError("a 10.7 plan reached cute.compile")

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 7))
    monkeypatch.setattr(api, "device_capability", lambda device: (10, 7))
    monkeypatch.setattr(interface, "_device_capability", lambda device: (10, 7))
    monkeypatch.setattr(interface, "_compile_deterministic_sm100", record_compile)
    monkeypatch.setattr(interface.cute, "compile", forbidden_compile)
    device = torch.device("cuda")
    capability_error = r"found compute capability 10\.7"

    for num_heads, head_dim, backend in ((64, 512, "generic_m64"), (96, 576, "h96_h64_h32")):
        s_q, s_kv, topk, head_dim_v = 300, 4096, 64, 512
        inputs = (
            torch.zeros(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device),
            torch.zeros(s_kv, head_dim, dtype=torch.bfloat16, device=device),
            torch.zeros(s_q, num_heads, head_dim_v, dtype=torch.bfloat16, device=device),
            torch.zeros(s_q, num_heads, head_dim_v, dtype=torch.bfloat16, device=device),
            torch.zeros(s_q, num_heads, dtype=torch.float32, device=device),
            torch.zeros(num_heads, dtype=torch.float32, device=device),
            torch.zeros(s_q, topk, dtype=torch.int32, device=device),
        )
        plans = {}
        for mode in ("records", "shards"):
            plan = api.SparseAttentionBackward(*inputs, softmax_scale=0.05, deterministic=True, deterministic_dkv_mode=mode)
            assert plan.check_support()
            plan.compile()
            plans[mode] = (plan, compiled[-1])
        (records, records_compile), (shards, shards_compile) = plans["records"], plans["shards"]
        assert records._backend == shards._backend == backend
        assert records._deterministic_dkv_route() == shards._deterministic_dkv_route() == ("shards" if backend == "generic_m64" else None)
        assert records._wide_addressing == shards._wide_addressing
        assert records.scratch_workspace_bytes() == shards.scratch_workspace_bytes()
        assert records_compile == shards_compile
        # On SM100/SM103 the default plan of this shape stores dKV in conflict-rank records, which need less scratch.
        sm100_records = interface.flash_attn_bwd_sm100_workspace_size(s_q, s_kv, head_dim, num_heads, True, backend="generic_m64", max_topk=topk)
        assert sm100_records < records.scratch_workspace_bytes()
        chain = api.SparseAttentionBackward(*inputs, softmax_scale=0.05, deterministic=True, deterministic_dkv_mode="chain")
        with pytest.raises(ValueError, match=capability_error):
            chain.check_support()
        q, kv, out, dout, lse, attn_sink, topk_idxs = inputs
        outputs = dict(dq=torch.empty_like(q), dkv=torch.empty_like(kv), d_sink=torch.empty_like(attn_sink))
        with pytest.raises(ValueError, match=capability_error):
            interface.flash_attn_bwd_sm100(q, kv, out, dout, lse, attn_sink, topk_idxs, deterministic=True, dkv_mode="chain", **outputs)


@torch_fork_set_rng(seed=20261105)
@pytest.mark.parametrize(
    "num_heads,head_dim,s_q,s_kv,topk,options",
    [
        pytest.param(64, 512, 300, 4096, 64, {}, marks=pytest.mark.L1, id="d512-partial-wave"),
        pytest.param(16, 576, 300, 4096, 64, {"check_short_workspace": True}, marks=pytest.mark.L1, id="h16-d576-padded-heads"),
        pytest.param(64, 576, 300, 4096, 64, {"duplicate_topk": True, "invalid_topk": True}, marks=pytest.mark.L1, id="d576-duplicate-invalid"),
        pytest.param(64, 576, 1037, 4096, 64, {"check_graph": True}, marks=pytest.mark.L1, id="d576-strided-waves-graph"),
        pytest.param(96, 576, 300, 4096, 64, {"check_graph": True, "duplicate_topk": True}, marks=pytest.mark.L1, id="h96-d576-head-pair-graph"),
        pytest.param(128, 512, 300, 4096, 64, {"shared_topk": True, "invalid_topk": True}, marks=pytest.mark.L1, id="h128-d512-shared-rows"),
        pytest.param(64, 512, 300, 4096, 64, {"repeats": 100, "check_graph": True}, marks=pytest.mark.L2, id="d512-repeat100"),
        pytest.param(64, 512, 148, 278528, 64, {"check_graph": True, "check_short_workspace": True}, marks=pytest.mark.L2, id="large-kv-records"),
        pytest.param(64, 512, 147, 278528, 2047, {"invalid_topk": True}, marks=pytest.mark.L2, id="split-pairs"),
    ],
)
def test_DSA_sparse_attention_backward_sm100_records_bitwise_and_graph(num_heads, head_dim, s_q, s_kv, topk, options):
    """Default (records) plans: bitwise run-to-run, graph replay, duplicate and invalid indices, and runtime lengths."""
    options = dict(options)
    repeats = options.pop("repeats", 3)
    try:
        _exercise_deterministic_sm100_case(
            num_heads,
            head_dim,
            s_q,
            s_kv,
            repeats,
            topk=topk,
            expected_dkv_route="records",
            high_fan_in_reference=bool(options.get("shared_topk") or options.get("duplicate_topk")),
            **options,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")


@torch_fork_set_rng(seed=20261106)
@pytest.mark.parametrize(
    "num_heads,head_dim,s_q,s_kv,topk,expected_route",
    [
        pytest.param(64, 512, 300, 4096, 64, "chain", marks=pytest.mark.L1, id="chain"),
        pytest.param(96, 576, 300, 4096, 64, "chain", marks=pytest.mark.L1, id="chain-h96-d576"),
        pytest.param(64, 512, 128, 130, 64, "shards", marks=pytest.mark.L1, id="fallback-shards"),
        pytest.param(64, 512, 148, 278528, 64, "records", marks=pytest.mark.L2, id="fallback-records"),
    ],
)
def test_DSA_sparse_attention_backward_sm100_chain_mode_bitwise_and_fallback(num_heads, head_dim, s_q, s_kv, topk, expected_route):
    """``deterministic_dkv_mode='chain'`` runs the chain relay where it applies and the default layout elsewhere."""
    try:
        _exercise_deterministic_sm100_case(
            num_heads,
            head_dim,
            s_q,
            s_kv,
            repeats=3,
            topk=topk,
            check_graph=True,
            check_short_workspace=True,
            duplicate_topk=True,
            deterministic_dkv_mode="chain",
            expected_dkv_route=expected_route,
            high_fan_in_reference=True,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")


def _assert_same_partial_sums(actual, expected, msg):
    """Require two outputs that add the same partial products in different FP32 orders to agree.

    Reordering FP32 additions moves a sum far less than its final rounding, so the outputs differ by at most
    a rounding step of the output format. The bound is ``2**-7 * max(|row|.max(), 1)`` per row of the last
    dimension (two BF16 rounding steps at the row's largest magnitude): it holds for any fan-in, while a lost
    or repeated contribution of the row's magnitude exceeds it.
    """
    actual = actual.float()
    expected = expected.float()
    bound = 2.0**-7 * expected.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    excess = (actual - expected).abs() - bound
    worst = int(excess.argmax())
    assert excess.max() <= 0, (
        f"{msg}: {int((excess > 0).sum())} of {excess.numel()} elements exceed the reordering bound; worst at flat index "
        f"{worst}: {actual.flatten()[worst].item()} vs {expected.flatten()[worst].item()}"
    )


@pytest.mark.L1
@torch_fork_set_rng(seed=20261107)
@pytest.mark.parametrize(
    "num_heads,head_dim,deterministic_dkv_mode,expected_route",
    [
        pytest.param(64, 512, "records", "records", id="records-d512"),
        pytest.param(96, 576, "records", "records", id="records-h96-d576"),
        pytest.param(64, 512, "chain", "chain", id="chain-d512"),
        pytest.param(96, 576, "chain", "chain", id="chain-h96-d576"),
        pytest.param(64, 512, "shards", "shards", id="shards-d512"),
        pytest.param(96, 576, "shards", None, id="shards-h96-d576"),
    ],
)
def test_DSA_sparse_attention_backward_sm100_deterministic_back_to_back_inputs(num_heads, head_dim, deterministic_dkv_mode, expected_route):
    """Different inputs run back to back on one plan and one workspace; every call matches its own reference.

    The executes of an input sequence are queued on one stream with no host synchronization in between (the
    sync debug mode rejects any), eagerly in two orders and as one CUDA graph of two executes; every output is
    bitwise independent of the call before it. Conflict-rank plans are also compared with the ``'shards'`` plan
    of the same inputs, which adds the same partial products in another FP32 order. ``'shards'`` runs the
    128-shard layout here, and the H64 + H32 composition for H96 D576.
    """
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm100()
    if expected_route in ("records", "chain"):
        _require_conflict_rank_device()
    s_q, s_kv, topk = 300, 4096, 64
    cases = [
        _deterministic_case_inputs(num_heads, head_dim, s_q, s_kv, topk),
        _deterministic_case_inputs(num_heads, head_dim, s_q, s_kv, topk, shared_topk=True, duplicate_topk=True),
        _deterministic_case_inputs(num_heads, head_dim, s_q, s_kv, topk, invalid_topk=True),
    ]
    output_names = ("dQ", "dKV", "dSink")
    sample, sample_length, softmax_scale = cases[0]
    device = sample[0].device

    def make_plan(mode):
        """Build, check and compile the plan of one dKV mode with its own poisoned workspace."""
        plan = DSA.SparseAttentionBackward(
            *sample,
            sample_topk_length=sample_length,
            softmax_scale=softmax_scale,
            deterministic=True,
            deterministic_dkv_mode=mode,
        )
        assert plan.check_support()
        plan.compile()
        workspace = torch.full((plan.scratch_workspace_bytes(),), 0xA5, dtype=torch.uint8, device=device)
        return plan, workspace

    def poisoned_outputs(case):
        """Fresh caller outputs of one input set, filled with a NaN pattern."""
        (q, kv, _, _, _, attn_sink, _), _, _ = case
        buffers = (torch.empty_like(q), torch.empty_like(kv), torch.empty_like(attn_sink))
        for buffer in buffers:
            buffer.view(torch.uint8).fill_(0xA5)
        return buffers

    def launch(plan, workspace, case, outputs, stream):
        """Enqueue one execute; nothing here waits for the device."""
        (q, kv, out, dout, lse, attn_sink, topk_idxs), topk_length, scale = case
        dq, dkv, d_sink = outputs
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
            topk_length=topk_length,
            softmax_scale=scale,
            workspace=workspace,
            current_stream=stream,
            d_sink=d_sink,
        )

    def run_back_to_back(plan, workspace, order):
        """Execute the input sets in ``order`` back to back on one stream; the workspace is never reset."""
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        outputs = {index: poisoned_outputs(cases[index]) for index in order}
        torch.cuda.synchronize()
        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            for index in order:
                launch(plan, workspace, cases[index], outputs[index], stream)
        finally:
            torch.cuda.set_sync_debug_mode(previous)
        torch.cuda.synchronize()
        return [outputs[index] for index in range(len(cases))]

    plan, workspace = make_plan(deterministic_dkv_mode)
    assert plan._deterministic_dkv_route() == expected_route
    assert plan._backend == ("h96_h64_h32" if expected_route is None else "generic_m64")
    first = run_back_to_back(plan, workspace, (0, 1, 2))
    second = run_back_to_back(plan, workspace, (2, 1, 0))
    for index, (a, b) in enumerate(zip(first, second)):
        for name, x, y in zip(output_names, a, b):
            assert torch.equal(bitwise_bits(x), bitwise_bits(y)), f"input set {index}: {name} depends on the previous call"

    # Two executes of different inputs captured back to back in one graph replay the eager results.
    graph_stream = torch.cuda.Stream()
    graph_handle = cuda.CUstream(graph_stream.cuda_stream)
    graph_outputs = [poisoned_outputs(cases[index]) for index in (0, 1)]
    graph_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(graph_stream):
        for index in (0, 1):
            launch(plan, workspace, cases[index], graph_outputs[index], graph_handle)
    torch.cuda.current_stream().wait_stream(graph_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=graph_stream):
            for index in (0, 1):
                launch(plan, workspace, cases[index], graph_outputs[index], graph_handle)
        for repeat in range(2):
            with torch.cuda.stream(graph_stream):
                for buffer in (workspace, *graph_outputs[0], *graph_outputs[1]):
                    buffer.view(torch.uint8).fill_(0xA5)
                graph.replay()
            torch.cuda.synchronize()
            for index, buffers in enumerate(graph_outputs):
                for name, x, y in zip(output_names, buffers, first[index]):
                    assert torch.equal(bitwise_bits(x), bitwise_bits(y)), f"graph replay {repeat}, input set {index}: {name} differs from the eager execute"
    finally:
        graph.reset()

    if expected_route in ("records", "chain"):
        shards_plan, shards_workspace = make_plan("shards")
        shards = run_back_to_back(shards_plan, shards_workspace, (0, 1, 2))
        for index, (a, b) in enumerate(zip(first, shards)):
            for name, x, y in zip(output_names, a, b):
                _assert_same_partial_sums(x, y, f"input set {index}: {name} vs the 'shards' plan")
    # Input set 1 (shared and repeated rows) has high-fan-in rows: its dKV is bounded by the non-deterministic
    # plan's reference error; the other sets keep the plain tolerance.
    for index, (case, result) in enumerate(zip(cases, first)):
        (q, kv, out, dout, lse, attn_sink, topk_idxs), topk_length, scale = case
        if index == 1:
            _check_ref_high_fan_in(q, kv, out, dout, lse, attn_sink, topk_idxs, result, softmax_scale=scale, topk_length=topk_length)
            continue
        check_ref_dsa_sparse_attention_backward(
            q, kv, attn_sink, topk_idxs, out, dout, lse, *result, softmax_scale=scale, topk_length=topk_length, atol=5e-2, rtol=5e-2
        )


@pytest.mark.L1
@torch_fork_set_rng(seed=20261108)
@pytest.mark.parametrize(
    "deterministic_dkv_mode,expected_route,first_s_q,second_s_q",
    [
        pytest.param("records", "records", 148, 300, id="records"),
        pytest.param("chain", "chain", 300, 600, id="chain"),
    ],
)
def test_DSA_sparse_attention_backward_sm100_rank_artifact_is_shape_dynamic(monkeypatch, deterministic_dkv_mode, expected_route, first_s_q, second_s_q):
    """Records and chain plans reuse one compiled artifact across query and KV extents."""
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm100 as interface
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_conflict_rank_device()
    compile_artifact = interface._compile_deterministic_sm100
    compile_artifact.cache_clear()
    first, *_ = _plan_case(64, 512, 64, s_q=first_s_q, s_kv=4096, has_topk_length=True, deterministic=True, deterministic_dkv_mode=deterministic_dkv_mode)
    assert first._deterministic_dkv_route() == expected_route
    first_info = compile_artifact.cache_info()

    def forbidden_compile(*args, **kwargs):
        """Fail when a dynamic-extent cache hit unexpectedly recompiles."""
        raise AssertionError("shape-dynamic plan unexpectedly recompiled")

    monkeypatch.setattr(interface.cute, "compile", forbidden_compile)
    second, inputs, outputs, topk_length, softmax_scale, workspace = _plan_case(
        64, 512, 64, s_q=second_s_q, s_kv=4352, has_topk_length=True, deterministic=True, deterministic_dkv_mode=deterministic_dkv_mode
    )
    second_info = compile_artifact.cache_info()
    assert second._deterministic_dkv_route() == expected_route
    assert second._compiled_kernel is first._compiled_kernel
    assert first_info.misses == second_info.misses == 1
    assert second_info.hits == first_info.hits + 1
    _assert_execute_allocates_nothing_and_never_synchronizes(second, inputs, outputs, topk_length, softmax_scale, workspace)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@torch_fork_set_rng(seed=20261109)
@pytest.mark.parametrize("head_dim", [512, 576], ids=["d512", "d576"])
def test_DSA_sparse_attention_backward_h128_conflict_rank_artifact_is_shape_dynamic(monkeypatch, head_dim):
    """H128 two-CTA conflict-rank plans reuse one compiled artifact across query and KV extents that keep their variant.

    At top-k 128, KV extents 4096 and 4352 select the same conflict-rank variant (D512: one kernel class; D576: the
    same wave rounds and O*dO choice), so the second plan must hit the compile cache instead of compiling.
    """
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm100 as interface
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm100_d576 as d576
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_h128_two_cta()
    topk, extents = 128, ((148, 4096), (300, 4352))
    if head_dim == 512:
        variants = {interface._get_sm100_kernel_class("h128_2cta_m64", True, s_q, s_kv, 512, 128, None, max_topk=topk) for s_q, s_kv in extents}
        assert len(variants) == 1 and getattr(next(iter(variants)), "rank_shard", False)
        compile_artifact = interface._compile_deterministic_sm100
    else:
        variants = {(d576._rank_shard_rounds(s_kv, topk, True, 128), d576._odo_precompute(s_kv, topk, True, 128)) for _, s_kv in extents}
        assert len(variants) == 1 and next(iter(variants))[0] > 0
        compile_artifact = d576._compile_d576_2cta
    compile_artifact.cache_clear()
    first, *_ = _plan_case(128, head_dim, topk, s_q=extents[0][0], s_kv=extents[0][1], has_topk_length=True, deterministic=True)
    first_info = compile_artifact.cache_info()

    def forbidden_compile(*args, **kwargs):
        """Fail when a dynamic-extent cache hit unexpectedly recompiles."""
        raise AssertionError("shape-dynamic plan unexpectedly recompiled")

    monkeypatch.setattr(interface.cute, "compile", forbidden_compile)
    second, inputs, outputs, topk_length, softmax_scale, workspace = _plan_case(
        128, head_dim, topk, s_q=extents[1][0], s_kv=extents[1][1], has_topk_length=True, deterministic=True
    )
    second_info = compile_artifact.cache_info()
    assert first._backend == second._backend == ("h128_2cta_m64" if head_dim == 512 else "h128_d576_2cta_m64")
    assert second._compiled_kernel is first._compiled_kernel
    assert first_info.misses == second_info.misses == 1
    assert second_info.hits == first_info.hits + 1
    _assert_execute_allocates_nothing_and_never_synchronizes(second, inputs, outputs, topk_length, softmax_scale, workspace)


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
        _require_sm100_family()
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

    # Generic deterministic H64 uses 32 byte-shaped dKV planes at this Q when
    # conflict-rank records do not apply (a top-k width beyond their slot
    # field). KV=32768 occupies exactly 2^31 bytes and remains addressable;
    # one more row requires the dynamic Int64 workspace descriptor.
    generic = (1, 512, 512, 64, 16384, True, "generic_m64")
    assert not _requires_wide_data_addressing(generic[0], 32768, *generic[1:], include_workspace=True)
    assert _requires_wide_data_addressing(generic[0], 32769, *generic[1:], include_workspace=True)
    # With top-k 2048 the default records layout is smaller and keeps KV=32769 addressable.
    records = (1, 512, 512, 64, 2048, True, "generic_m64")
    assert not _requires_wide_data_addressing(records[0], 32769, *records[1:], include_workspace=True)


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
        _require_sm100_family()
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
    _require_sm100_family()
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

    _require_sm100_family()

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
    from cudnn.tensor_adapter import has_tcgen05

    if not has_tcgen05(major):
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
    from cudnn.tensor_adapter import has_tcgen05

    if major != 9 and not has_tcgen05(major):
        pytest.skip("sparse attention backward requires SM90+")
    if num_heads == 128 and head_dim == 512:
        _require_sm100_family()
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
        _require_sm100_family()
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
        _require_sm100_family()
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
    from cudnn.tensor_adapter import has_tcgen05

    if not has_tcgen05(major):
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
    DSA = _import_or_skip("cudnn").DSA
    cuda = _import_or_skip("cuda.bindings.driver", reason="Environment not supported: cuda-python not installed")
    _interface_sm90 = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm90")

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
    matching_keys = [key for key in _interface_sm90.flash_attn_bwd_sm90.compile_cache if key[1:4] == (head_dim, head_dim_v, num_heads) and key[-5] is False]
    assert matching_keys
    assert all(key[-1] == 0 for key in matching_keys), "non-deterministic kernels must not specialize on rounded head extents"

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
    from cudnn.tensor_adapter import has_tcgen05

    if not has_tcgen05(major):
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
        _require_sm100_family()
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

        # Above 16 GiB of shards this shape runs the cluster relay under 'shards' or an explicit relay size.
        relay = metadata_case(257, 65537, 16, 512, 64)
        relay_floor = r"deterministic DSA backward cluster relay requires nvidia-cutlass-dsl >= 4\.7\.0; found 4\.6\.2"
        with pytest.raises(RuntimeError, match=relay_floor):
            SparseAttentionBackward(**relay, deterministic=True, deterministic_dkv_mode="shards").check_support()
        with pytest.raises(RuntimeError, match=relay_floor):
            SparseAttentionBackward(**relay, deterministic=True, deterministic_relay_cluster_size=8).check_support()

        # Neither the relay implementation nor its version floor changes the
        # established non-deterministic backend.
        assert SparseAttentionBackward(**good).check_support()

    with monkeypatch.context() as supported_dsl:
        supported_dsl.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.7.0")))
        assert SparseAttentionBackward(**relay, deterministic=True, deterministic_dkv_mode="shards").check_support()
        assert SparseAttentionBackward(**relay, deterministic=True, deterministic_relay_cluster_size=8).check_support()

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
    """SM90 reports ordinary or deterministic FP32 scratch (R2)."""
    _require_sm90_relay_dsl()
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
    # Packed ownership masks follow every logical relay plane.
    expected_cluster8 = 2 * ws_align(sq_r * num_heads * 4) + ws_align(16 * (sk_r + 1) * hd_r * 4)
    expected_cluster4 = 2 * ws_align(sq_r * num_heads * 4) + ws_align(32 * (sk_r + 1) * hd_r * 4)
    expected_cluster16 = 2 * ws_align(sq_r * num_heads * 4) + ws_align(8 * (sk_r + 1) * hd_r * 4)
    expected_cluster32 = 2 * ws_align(sq_r * num_heads * 4) + ws_align(4 * (sk_r + 2) * hd_r * 4)
    expected_cluster64 = 2 * ws_align(sq_r * num_heads * 4) + ws_align(2 * (sk_r + 4) * hd_r * 4)
    assert flash_attn_bwd_sm90_workspace_size(s_q, s_kv, head_dim, num_heads, deterministic=True) == expected_cluster16
    assert (
        flash_attn_bwd_sm90_workspace_size(
            s_q,
            s_kv,
            head_dim,
            num_heads,
            deterministic=True,
            deterministic_relay_cluster_size=4,
        )
        == expected_cluster4
    )
    assert (
        flash_attn_bwd_sm90_workspace_size(
            s_q,
            s_kv,
            head_dim,
            num_heads,
            deterministic=True,
            deterministic_relay_cluster_size=8,
        )
        == expected_cluster8
    )
    assert (
        flash_attn_bwd_sm90_workspace_size(
            s_q,
            s_kv,
            head_dim,
            num_heads,
            deterministic=True,
            deterministic_relay_cluster_size=16,
        )
        == expected_cluster16
    )
    assert (
        flash_attn_bwd_sm90_workspace_size(
            s_q,
            s_kv,
            head_dim,
            num_heads,
            deterministic=True,
            deterministic_relay_cluster_size=32,
        )
        == expected_cluster32
    )
    assert (
        flash_attn_bwd_sm90_workspace_size(
            s_q,
            s_kv,
            head_dim,
            num_heads,
            deterministic=True,
            deterministic_relay_cluster_size=64,
        )
        == expected_cluster64
    )
    with pytest.raises(ValueError, match="must be 4, 8, 16, 32, or 64"):
        flash_attn_bwd_sm90_workspace_size(
            s_q,
            s_kv,
            head_dim,
            num_heads,
            deterministic=True,
            deterministic_relay_cluster_size=2,
        )
    large_stats = ws_align(1088 * 128 * 4)
    expected_large_cluster8 = 2 * large_stats + ws_align(16 * (1088 + 9) * 576 * 4)
    expected_large_cluster4 = 2 * large_stats + ws_align(32 * (1088 + 5) * 576 * 4)
    expected_large_cluster16 = 2 * large_stats + ws_align(8 * (1088 + 17) * 576 * 4)
    expected_large_cluster32 = 2 * large_stats + ws_align(4 * (1088 + 34) * 576 * 4)
    expected_large_cluster64 = 2 * large_stats + ws_align(2 * (1088 + 68) * 576 * 4)
    assert flash_attn_bwd_sm90_workspace_size(1025, 1025, 576, 128, deterministic=True) == expected_large_cluster16
    assert (
        flash_attn_bwd_sm90_workspace_size(
            1025,
            1025,
            576,
            128,
            deterministic=True,
            deterministic_relay_cluster_size=4,
        )
        == expected_large_cluster4
    )
    assert (
        flash_attn_bwd_sm90_workspace_size(
            1025,
            1025,
            576,
            128,
            deterministic=True,
            deterministic_relay_cluster_size=8,
        )
        == expected_large_cluster8
    )
    assert (
        flash_attn_bwd_sm90_workspace_size(
            1025,
            1025,
            576,
            128,
            deterministic=True,
            deterministic_relay_cluster_size=16,
        )
        == expected_large_cluster16
    )
    assert (
        flash_attn_bwd_sm90_workspace_size(
            1025,
            1025,
            576,
            128,
            deterministic=True,
            deterministic_relay_cluster_size=32,
        )
        == expected_large_cluster32
    )
    assert (
        flash_attn_bwd_sm90_workspace_size(
            1025,
            1025,
            576,
            128,
            deterministic=True,
            deterministic_relay_cluster_size=64,
        )
        == expected_large_cluster64
    )

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

    cluster4_plan = DSA.SparseAttentionBackward(
        sample_q=torch.empty(s_q, num_heads, head_dim, dtype=torch.bfloat16, device=device),
        sample_kv=torch.empty(s_kv, head_dim, dtype=torch.bfloat16, device=device),
        sample_out=torch.empty(s_q, num_heads, 512, dtype=torch.bfloat16, device=device),
        sample_dout=torch.empty(s_q, num_heads, 512, dtype=torch.bfloat16, device=device),
        sample_lse=torch.empty(s_q, num_heads, dtype=torch.float32, device=device),
        sample_attn_sink=torch.empty(num_heads, dtype=torch.float32, device=device),
        sample_topk_idxs=torch.empty(s_q, 64, dtype=torch.int32, device=device),
        deterministic=True,
        deterministic_relay_cluster_size=4,
    )
    assert cluster4_plan.check_support()
    assert cluster4_plan.scratch_workspace_bytes() == expected_cluster4


@pytest.mark.L0
def test_DSA_sparse_attention_backward_sm90_relay_cluster_size_rejects_before_staging(monkeypatch):
    """Invalid public relay attributes fail before the wrapper stages inputs."""
    _require_sm90()
    from cudnn.deepseek_sparse_attention.sparse_attention_backward import api

    def forbidden_staging(*args, **kwargs):
        raise AssertionError("invalid relay configuration reached input staging")

    monkeypatch.setattr(api, "contiguous_on_stream", forbidden_staging)
    q = torch.empty(0, device="cuda")
    args = (q, None, None, None, None, None, None)
    with pytest.raises(ValueError, match="requires deterministic=True"):
        api.sparse_attention_backward_wrapper(*args, deterministic=False, deterministic_relay_cluster_size=4)
    with pytest.raises(ValueError, match=r"must be one of \{4, 8, 16, 32, 64\}"):
        api.sparse_attention_backward_wrapper(*args, deterministic=True, deterministic_relay_cluster_size=2)
    with pytest.raises(ValueError, match=r"must be one of \{4, 8, 16, 32, 64\}"):
        api.sparse_attention_backward_wrapper(*args, deterministic=True, deterministic_relay_cluster_size=4.5)


@pytest.mark.L0
@pytest.mark.parametrize(
    "version,expected",
    [
        ("4.7.1", "deterministic SM90 DSA backward requires nvidia-cutlass-dsl >= 4.8.0; found 4.7.1"),
        ("4.8.0", None),
        ("4.9.0", None),
    ],
)
def test_DSA_sparse_attention_backward_sm90_relay_cutedsl_version_gate(monkeypatch, version, expected):
    """The SM90 relay names its 4.8 DSL floor without importing a kernel."""
    _require_sm90_relay_dependencies()
    from cudnn.deepseek_sparse_attention.sparse_attention_backward import _sm90_relay
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", version)))
    assert _sm90_relay.sm90_deterministic_relay_requirement_error() == expected


@pytest.mark.L0
def test_DSA_sparse_attention_backward_sm90_relay_feature_tests_skip_old_cutedsl(monkeypatch):
    """Feature tests skip, while the dedicated version-refusal test remains active."""
    _require_sm90_relay_dependencies()
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.7.1")))
    with pytest.raises(pytest.skip.Exception, match=r"requires nvidia-cutlass-dsl >= 4\.8\.0; found 4\.7\.1"):
        _require_sm90_relay_dsl()


@pytest.mark.L0
@pytest.mark.parametrize("missing_dependency", ["cutlass", "cuda.bindings.driver"])
def test_DSA_sparse_attention_backward_sm90_relay_dependencies_skip_before_package_import(monkeypatch, missing_dependency):
    """A missing optional dependency skips before importing the relay package."""
    attempted_imports = []

    def import_or_skip(module_name, reason=None):
        attempted_imports.append(module_name)
        if module_name == missing_dependency:
            pytest.skip(f"missing {module_name}")

    monkeypatch.setitem(globals(), "_import_or_skip", import_or_skip)
    with pytest.raises(pytest.skip.Exception, match=rf"missing {re.escape(missing_dependency)}"):
        _require_sm90_relay_dsl()

    expected_imports = ["cutlass"] if missing_dependency == "cutlass" else ["cutlass", "cuda.bindings.driver"]
    assert attempted_imports == expected_imports


@pytest.mark.L0
def test_DSA_sparse_attention_backward_sm90_relay_int32_element_boundary():
    """The largest valid signed-Int32 element offset is accepted exactly."""
    _require_sm90_relay_dependencies()
    from cudnn.deepseek_sparse_attention.sparse_attention_backward import _sm90_relay

    limit = 1 << 31
    assert _sm90_relay.sm90_deterministic_relay_indexing_error(limit) is None
    assert _sm90_relay.sm90_deterministic_relay_indexing_error(limit + 1) == (
        "deterministic SM90 DSA backward dKV accumulator has 2147483649 elements, " "which exceeds the signed Int32 element offset range"
    )


@pytest.mark.L0
@pytest.mark.parametrize("relay_cluster_size", [4, 8, 16, 32, 64])
def test_DSA_sparse_attention_backward_sm90_relay_host_span_matches_workspace_layout(relay_cluster_size):
    """The support-check span calculation matches the execution workspace."""
    _require_sm90_relay_dsl()
    from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm90, _sm90_relay

    planes, physical_plane_elems, *_ = _interface_sm90._deterministic_relay_layout(257, 1001, 576, 96, relay_cluster_size)
    assert _sm90_relay.sm90_deterministic_relay_accumulator_elements(257, 1001, 576, 96, relay_cluster_size) == planes * physical_plane_elems


@pytest.mark.L0
def test_DSA_sparse_attention_backward_sm90_relay_rejects_oversized_accumulator_during_support_check(monkeypatch):
    """The prepared plan rejects unsafe dynamic offsets before compilation."""
    _require_sm90_deterministic_relay()
    from cudnn import DSA
    from cudnn.deepseek_sparse_attention.sparse_attention_backward import api

    monkeypatch.setattr(api, "sm90_deterministic_relay_accumulator_elements", lambda *args: (1 << 31) + 1)
    device = torch.device("cuda")
    plan = DSA.SparseAttentionBackward(
        sample_q=torch.empty(1, 16, 512, dtype=torch.bfloat16, device=device),
        sample_kv=torch.empty(64, 512, dtype=torch.bfloat16, device=device),
        sample_out=torch.empty(1, 16, 512, dtype=torch.bfloat16, device=device),
        sample_dout=torch.empty(1, 16, 512, dtype=torch.bfloat16, device=device),
        sample_lse=torch.empty(1, 16, dtype=torch.float32, device=device),
        sample_attn_sink=torch.empty(16, dtype=torch.float32, device=device),
        sample_topk_idxs=torch.empty(1, 16, dtype=torch.int32, device=device),
        deterministic=True,
    )
    with pytest.raises(ValueError, match=r"2147483649 elements.*signed Int32 element offset range"):
        plan.check_support()


@pytest.mark.L0
def test_DSA_sparse_attention_backward_sm90_relay_rejects_old_cutedsl_before_compile(monkeypatch):
    """Prepared SM90 deterministic plans enforce the relay-specific DSL floor."""
    _require_sm90()
    from cudnn import DSA
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.7.1")))
    device = torch.device("cuda")
    plan = DSA.SparseAttentionBackward(
        sample_q=torch.empty(1, 16, 512, dtype=torch.bfloat16, device=device),
        sample_kv=torch.empty(64, 512, dtype=torch.bfloat16, device=device),
        sample_out=torch.empty(1, 16, 512, dtype=torch.bfloat16, device=device),
        sample_dout=torch.empty(1, 16, 512, dtype=torch.bfloat16, device=device),
        sample_lse=torch.empty(1, 16, dtype=torch.float32, device=device),
        sample_attn_sink=torch.empty(16, dtype=torch.float32, device=device),
        sample_topk_idxs=torch.empty(1, 16, dtype=torch.int32, device=device),
        deterministic=True,
    )
    with pytest.raises(RuntimeError, match=r"requires nvidia-cutlass-dsl >= 4\.8\.0; found 4\.7\.1"):
        plan.check_support()


@pytest.mark.L0
def test_DSA_sparse_attention_backward_sm90_kernels_have_public_symbol_prefix():
    """Every CuTeDSL kernel in the SM90 backward module uses the public prefix."""
    _require_sm90_relay_dsl()
    module = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm90")
    source = inspect.getsource(module)

    assert source.count("@cute.kernel") == 5
    assert source.count('.set_name_prefix("cudnn", remove_cutlass_symbol=True)') == 5


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
@torch_fork_set_rng(seed=92)
@pytest.mark.parametrize("has_topk_length", [False, True], ids=["compact", "lengths"])
@pytest.mark.parametrize(
    "relay_cluster_size",
    [4, 8, 16, 32, 64],
    ids=["cluster4", "cluster8", "cluster16", "cluster32", "cluster64"],
)
def test_DSA_sparse_attention_backward_sm90_deterministic_relay(has_topk_length, relay_cluster_size):
    """Hopper relay ranks, head phases, and query waves stay bitwise stable."""
    cuda = _import_or_skip("cuda.bindings.driver", reason="Environment not supported: cuda-python not installed")

    _require_sm90_deterministic_relay()
    plan, inputs, outputs, topk_length, softmax_scale, workspace = _plan_case(
        96,
        576,
        128,
        s_q=129,
        s_kv=256,
        has_topk_length=has_topk_length,
        deterministic=True,
        deterministic_relay_cluster_size=relay_cluster_size,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def run():
        dq, dkv, d_sink = outputs
        for tensor in (*outputs, workspace):
            tensor.view(torch.uint8).fill_(0xA5)
        result = plan.execute(
            *inputs,
            dq,
            dkv,
            topk_length=topk_length,
            softmax_scale=softmax_scale,
            current_stream=stream,
            workspace=workspace,
            d_sink=d_sink,
        )
        torch.cuda.synchronize()
        return tuple(tensor.clone() for tensor in result)

    reference = run()
    for repeat in range(1, 4):
        actual = run()
        for name, actual_tensor, reference_tensor in zip(("dQ", "dKV", "dSink"), actual, reference):
            assert torch.equal(bitwise_bits(actual_tensor), bitwise_bits(reference_tensor)), f"{name} differs at repetition {repeat}"

    q, kv, out, dout, lse, attn_sink, topk_idxs = inputs
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


@pytest.mark.L0
@torch_fork_set_rng(seed=94)
def test_DSA_sparse_attention_backward_sm90_deterministic_compile_cache_reuses_runtime_extents(monkeypatch):
    """Packed Q/KV extents stay runtime values for all deterministic artifacts."""
    _require_sm90_deterministic_relay()
    cuda = _import_or_skip("cuda.bindings.driver", reason="Environment not supported: cuda-python not installed")
    _interface_sm90 = _import_or_skip("cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm90")

    launch = _interface_sm90.flash_attn_bwd_sm90
    monkeypatch.setattr(launch, "compile_cache_relay_masks", {})
    monkeypatch.setattr(launch, "compile_cache", {})
    monkeypatch.setattr(launch, "compile_cache_post", {})
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def execute(s_q, s_kv):
        plan, inputs, outputs, topk_length, softmax_scale, workspace = _plan_case(
            64,
            512,
            64,
            s_q=s_q,
            s_kv=s_kv,
            has_topk_length=True,
            deterministic=True,
            deterministic_relay_cluster_size=16,
        )
        dq, dkv, d_sink = outputs
        result = plan.execute(
            *inputs,
            dq,
            dkv,
            topk_length=topk_length,
            softmax_scale=softmax_scale,
            current_stream=stream,
            workspace=workspace,
            d_sink=d_sink,
        )
        torch.cuda.synchronize()
        assert all(torch.isfinite(tensor).all() for tensor in result)

    execute(65, 192)
    compiled = {name: next(iter(getattr(launch, name).values())) for name in ("compile_cache_relay_masks", "compile_cache", "compile_cache_post")}
    execute(129, 256)

    for name, first_callable in compiled.items():
        cache = getattr(launch, name)
        assert len(cache) == 1, f"{name} specialized on a runtime packed extent: {tuple(cache)}"
        assert next(iter(cache.values())) is first_callable


@pytest.mark.L2
@torch_fork_set_rng(seed=20261102)
@pytest.mark.parametrize("relay_cluster_size", [32, 64], ids=["cluster32", "cluster64"])
def test_DSA_sparse_attention_backward_sm90_multiphase_relay_repeat100(relay_cluster_size):
    """Logical relays remain bitwise exact under 100 high-overlap repetitions."""
    cuda = _import_or_skip("cuda.bindings.driver", reason="Environment not supported: cuda-python not installed")

    _require_sm90_deterministic_relay()
    plan, inputs, outputs, topk_length, softmax_scale, workspace = _plan_case(
        64,
        512,
        511,
        s_q=257,
        s_kv=1024,
        deterministic=True,
        deterministic_relay_cluster_size=relay_cluster_size,
    )
    q, kv, out, dout, lse, attn_sink, topk_idxs = inputs
    topk_idxs.copy_(topk_idxs[0].clone().expand_as(topk_idxs))
    shared_out, shared_lse = ref_sparse_attention_forward_chunked(
        q,
        kv,
        attn_sink,
        topk_idxs,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
    )
    out.copy_(shared_out)
    lse.copy_(shared_lse)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def run():
        dq, dkv, d_sink = outputs
        for tensor in (*outputs, workspace):
            tensor.view(torch.uint8).fill_(0xA5)
        result = plan.execute(
            *inputs,
            dq,
            dkv,
            topk_length=topk_length,
            softmax_scale=softmax_scale,
            current_stream=stream,
            workspace=workspace,
            d_sink=d_sink,
        )
        torch.cuda.synchronize()
        return tuple(tensor.clone() for tensor in result)

    reference = run()
    for repeat in range(1, 100):
        actual = run()
        for name, actual_tensor, reference_tensor in zip(("dQ", "dKV", "dSink"), actual, reference):
            assert torch.equal(bitwise_bits(actual_tensor), bitwise_bits(reference_tensor)), f"{name} differs at repetition {repeat}"


@pytest.mark.L0
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@torch_fork_set_rng(seed=93)
@pytest.mark.parametrize(
    "relay_cluster_size",
    [None, 32, 64],
    ids=["default-cluster16", "cluster32", "cluster64"],
)
def test_DSA_sparse_attention_backward_sm90_relay_cuda_graph(relay_cluster_size):
    """C16/C32/C64 relays replay with poisoned storage and changed sparse inputs."""
    cuda = _import_or_skip("cuda.bindings.driver", reason="Environment not supported: cuda-python not installed")

    _require_sm90_deterministic_relay()
    plan, inputs, outputs, topk_length, softmax_scale, workspace = _plan_case(
        96,
        576,
        128,
        s_q=129,
        s_kv=256,
        has_topk_length=True,
        deterministic=True,
        deterministic_relay_cluster_size=relay_cluster_size,
    )
    assert plan.deterministic_relay_cluster_size == relay_cluster_size
    assert plan._sm90_deterministic_relay_cluster_size == (16 if relay_cluster_size is None else relay_cluster_size)
    q, kv, out, dout, lse, attn_sink, topk_idxs = inputs
    saved_topk_idxs = topk_idxs.clone()
    saved_topk_length = topk_length.clone()
    changed_topk_idxs = topk_idxs.roll(1, dims=1)
    launch_stream = torch.cuda.Stream()
    launch_stream.wait_stream(torch.cuda.current_stream())
    stream = cuda.CUstream(launch_stream.cuda_stream)

    def poison():
        for buffer in (workspace, *outputs):
            buffer.view(torch.uint8).fill_(0xA5)

    def execute(active_workspace=workspace):
        dq, dkv, d_sink = outputs
        return plan.execute(
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
            current_stream=stream,
            workspace=active_workspace,
            d_sink=d_sink,
        )

    with pytest.raises(ValueError, match="workspace"):
        execute(workspace[:-1])

    with torch.cuda.stream(launch_stream):
        poison()
        execute()  # Populate the legacy SM90 concrete-tensor compile caches.
        eager = tuple(tensor.clone() for tensor in outputs)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=launch_stream):
            execute()
        for repeat in range(3):
            with torch.cuda.stream(launch_stream):
                poison()
                graph.replay()
            torch.cuda.synchronize()
            for name, actual, expected in zip(("dQ", "dKV", "dSink"), outputs, eager):
                assert torch.equal(bitwise_bits(actual), bitwise_bits(expected)), f"{name} differs on graph replay {repeat}"

        with torch.cuda.stream(launch_stream):
            topk_idxs.copy_(changed_topk_idxs)
            topk_length.fill_(64)
            poison()
            execute()
            changed_eager = tuple(tensor.clone() for tensor in outputs)
        torch.cuda.synchronize()
        with torch.cuda.stream(launch_stream):
            poison()
            graph.replay()
        torch.cuda.synchronize()
        for name, actual, expected in zip(("dQ", "dKV", "dSink"), outputs, changed_eager):
            assert torch.equal(bitwise_bits(actual), bitwise_bits(expected)), f"{name} graph replay ignores changed sparse inputs"
    finally:
        with torch.cuda.stream(launch_stream):
            topk_idxs.copy_(saved_topk_idxs)
            topk_length.copy_(saved_topk_length)
        launch_stream.synchronize()
        graph.reset()


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
        pytest.param(128, 512, 128, True, "off", "h128_2cta_m64", id="deterministic-h128-2cta"),
        pytest.param(128, 576, 1024, True, "off", "h128_d576_2cta_m64", id="deterministic-h128-d576-2cta"),
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
        _require_sm100_family()
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
    if deterministic and num_heads == 128:
        # The deterministic H128 two-CTA entries run conflict-rank dKV storage (setup launches each wave).
        assert _h128_conflict_rank_selected(head_dim, max(256, 2 * topk), topk)
    _assert_execute_allocates_nothing_and_never_synchronizes(plan, inputs, outputs, topk_length, softmax_scale, workspace)


@pytest.mark.L0
@torch_fork_set_rng(seed=102)
@pytest.mark.parametrize("deterministic_dkv_mode", ["records", "chain"])
def test_DSA_sparse_attention_backward_sm100_conflict_rank_execute_allocates_nothing_and_never_synchronizes(deterministic_dkv_mode):
    """Recipe R9 on the conflict-rank dKV routes of the deterministic generic M64 plan (records and chain)."""
    try:
        from cudnn import DSA  # noqa: F401
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_conflict_rank_device()
    plan, inputs, outputs, topk_length, softmax_scale, workspace = _plan_case(
        64, 512, 64, s_q=300, s_kv=4096, has_topk_length=True, deterministic=True, deterministic_dkv_mode=deterministic_dkv_mode
    )
    assert plan._backend == "generic_m64"
    assert plan._deterministic_dkv_route() == deterministic_dkv_mode
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
