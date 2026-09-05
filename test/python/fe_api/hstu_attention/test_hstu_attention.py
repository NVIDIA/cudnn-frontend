# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and support tests for Blackwell HSTU attention."""

from __future__ import annotations

from collections import OrderedDict
import inspect
import math

import pytest
import torch
import torch.nn.functional as F
from cuda.bindings import driver as cuda

try:
    import cutlass  # noqa: F401
except (ImportError, OSError) as exc:
    pytest.skip(f"CuTe DSL is unavailable: {exc}", allow_module_level=True)

from cudnn.hstu_attention import (
    HSTUBwdSm100,
    HSTUFwdSm100,
    hstu_attention_backward,
    hstu_attention_forward,
)
from cudnn.hstu_attention import _interface, api as _api

pytestmark = [
    pytest.mark.gpu_exclusive,
    pytest.mark.xdist_group(name="gpu_exclusive"),
]

_HAS_CUDA = torch.cuda.is_available()
_IS_SM10X = _HAS_CUDA and torch.cuda.get_device_capability()[0] == 10
_IS_Q1_SPLIT_TARGET = _HAS_CUDA and torch.cuda.get_device_capability() in ((10, 0), (10, 3), (10, 7))
_IS_SM107 = _HAS_CUDA and torch.cuda.get_device_capability() == (10, 7)


def _inputs(
    *,
    batch: int = 1,
    heads: int = 2,
    seqlen: int = 128,
    head_dim: int = 64,
    dtype: torch.dtype = torch.bfloat16,
):
    torch.manual_seed(123)
    shape = (batch * seqlen, heads, head_dim)
    q = torch.randn(shape, dtype=dtype, device="cuda") * 0.2
    k = torch.randn(shape, dtype=dtype, device="cuda") * 0.2
    v = torch.randn(shape, dtype=dtype, device="cuda") * 0.2
    do = torch.randn(shape, dtype=dtype, device="cuda") * 0.2
    cu = torch.arange(
        0,
        (batch + 1) * seqlen,
        seqlen,
        dtype=torch.int32,
        device="cuda",
    )
    return q, k, v, do, cu


def _reference_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    *,
    alpha: float,
    scaling_seqlen: float,
    causal: bool,
    window_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    outputs = []
    cu_q_cpu = cu_q.cpu()
    cu_k_cpu = cu_k.cpu()
    for batch_idx in range(cu_q.numel() - 1):
        q_start, q_end = map(int, cu_q_cpu[batch_idx : batch_idx + 2])
        k_start, k_end = map(int, cu_k_cpu[batch_idx : batch_idx + 2])
        q_i = q[q_start:q_end].float()
        k_i = k[k_start:k_end].float()
        v_i = v[k_start:k_end].float()
        scores = alpha * torch.einsum("qhd,khd->hqk", q_i, k_i)
        weights = F.silu(scores)
        if window_size is not None:
            window_left = k_i.shape[0] if window_size[0] < 0 else min(window_size[0], k_i.shape[0])
            window_right = k_i.shape[0] if window_size[1] < 0 else min(window_size[1], k_i.shape[0])
            q_idx = torch.arange(q_i.shape[0], device=q.device)[:, None]
            k_idx = torch.arange(k_i.shape[0], device=q.device)[None, :]
            diagonal_offset = k_i.shape[0] - q_i.shape[0]
            mask = (k_idx >= q_idx + diagonal_offset - window_left) & (k_idx <= q_idx + diagonal_offset + window_right)
            weights = torch.where(mask.unsqueeze(0), weights, torch.zeros_like(weights))
        elif causal:
            q_idx = torch.arange(q_i.shape[0], device=q.device)[:, None]
            k_idx = torch.arange(k_i.shape[0], device=q.device)[None, :]
            diagonal_offset = k_i.shape[0] - q_i.shape[0]
            weights = torch.where(
                (k_idx <= q_idx + diagonal_offset).unsqueeze(0),
                weights,
                torch.zeros_like(weights),
            )
        outputs.append(torch.einsum("hqk,khd->qhd", weights, v_i) / scaling_seqlen)
    return torch.cat(outputs, dim=0)


def _forward_api(q, k, v, cu, *, head_dim=64, scaling_seqlen=None):
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    return HSTUFwdSm100(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=out,
        sample_cu_seqlens_q=cu,
        sample_cu_seqlens_k=cu,
        max_seqlen_q=128,
        max_seqlen_k=128,
        window_size=(-1, 0),
        scaling_seqlen=scaling_seqlen,
    )


def _int32_alias(tensor: torch.Tensor, shape) -> torch.Tensor:
    numel = 1
    for size in shape:
        numel *= size
    return tensor.flatten().view(torch.int32)[:numel].view(shape)


@pytest.mark.L0
def test_rejects_torch_stream_from_another_device(monkeypatch):
    class _FakeStream:
        device = torch.device("cuda:1")

    monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
    with pytest.raises(ValueError, match=r"stream must be on cuda:0, got cuda:1"):
        _api._as_torch_stream(_FakeStream(), torch.device("cuda:0"))


@pytest.mark.L0
def test_bwd_direct_grad_layout_rejects_static_zero_stride():
    storage = torch.empty((4, 64), dtype=torch.bfloat16)
    grad = storage.as_strided((4, 1, 64), (64, 0, 1))

    assert _interface._supports_bwd_original_qkv_layout(grad)
    assert not _interface._supports_bwd_direct_grad_layout(grad)


def _assert_public_signature(callable_obj, names, defaults) -> None:
    parameters = inspect.signature(callable_obj).parameters
    assert tuple(parameters) == tuple(names)
    for name, parameter in parameters.items():
        assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert parameter.default == defaults.get(name, inspect.Parameter.empty)


@pytest.mark.L0
def test_public_functional_and_class_signatures_are_stable():
    common_defaults = {
        "window_size": (-1, -1),
        "alpha": 1.0,
        "scaling_seqlen": None,
    }
    _assert_public_signature(
        hstu_attention_forward,
        (
            "q_tensor",
            "k_tensor",
            "v_tensor",
            "cu_seqlens_q_tensor",
            "cu_seqlens_k_tensor",
            "max_seqlen_q",
            "max_seqlen_k",
            "window_size",
            "alpha",
            "scaling_seqlen",
            "func_tensor",
            "paged_kv_tensor",
            "page_ids_tensor",
            "page_indptrs_tensor",
            "stream",
        ),
        {
            **common_defaults,
            "func_tensor": None,
            "paged_kv_tensor": None,
            "page_ids_tensor": None,
            "page_indptrs_tensor": None,
            "stream": None,
        },
    )
    _assert_public_signature(
        hstu_attention_backward,
        (
            "do_tensor",
            "q_tensor",
            "k_tensor",
            "v_tensor",
            "cu_seqlens_q_tensor",
            "cu_seqlens_k_tensor",
            "max_seqlen_q",
            "max_seqlen_k",
            "window_size",
            "alpha",
            "scaling_seqlen",
            "func_tensor",
            "deterministic",
            "stream",
            "dq_tensor",
            "dk_tensor",
            "dv_tensor",
        ),
        {
            **common_defaults,
            "func_tensor": None,
            "deterministic": False,
            "stream": None,
            "dq_tensor": None,
            "dk_tensor": None,
            "dv_tensor": None,
        },
    )
    _assert_public_signature(
        HSTUFwdSm100,
        (
            "sample_q",
            "sample_k",
            "sample_v",
            "sample_o",
            "sample_cu_seqlens_q",
            "sample_cu_seqlens_k",
            "max_seqlen_q",
            "max_seqlen_k",
            "window_size",
            "alpha",
            "scaling_seqlen",
            "sample_func",
            "sample_paged_kv",
            "sample_page_ids",
            "sample_page_indptrs",
        ),
        {
            **common_defaults,
            "sample_func": None,
            "sample_paged_kv": None,
            "sample_page_ids": None,
            "sample_page_indptrs": None,
        },
    )
    _assert_public_signature(
        HSTUBwdSm100,
        (
            "sample_do",
            "sample_q",
            "sample_k",
            "sample_v",
            "sample_dq",
            "sample_dk",
            "sample_dv",
            "sample_cu_seqlens_q",
            "sample_cu_seqlens_k",
            "max_seqlen_q",
            "max_seqlen_k",
            "window_size",
            "alpha",
            "scaling_seqlen",
            "sample_func",
            "deterministic",
        ),
        {
            **common_defaults,
            "sample_func": None,
            "deterministic": False,
        },
    )
    _assert_public_signature(
        HSTUFwdSm100.execute,
        (
            "self",
            "q_tensor",
            "k_tensor",
            "v_tensor",
            "o_tensor",
            "cu_seqlens_q_tensor",
            "cu_seqlens_k_tensor",
            "func_tensor",
            "paged_kv_tensor",
            "page_ids_tensor",
            "page_indptrs_tensor",
            "current_stream",
        ),
        {
            "func_tensor": None,
            "paged_kv_tensor": None,
            "page_ids_tensor": None,
            "page_indptrs_tensor": None,
            "current_stream": None,
        },
    )
    _assert_public_signature(
        HSTUBwdSm100.execute,
        (
            "self",
            "do_tensor",
            "q_tensor",
            "k_tensor",
            "v_tensor",
            "dq_tensor",
            "dk_tensor",
            "dv_tensor",
            "cu_seqlens_q_tensor",
            "cu_seqlens_k_tensor",
            "func_tensor",
            "current_stream",
        ),
        {
            "func_tensor": None,
            "current_stream": None,
        },
    )


@pytest.mark.L0
@pytest.mark.skipif(not _HAS_CUDA, reason="requires CUDA")
def test_support_validation_and_scaling_default(monkeypatch):
    q, k, v, _, cu = _inputs()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 0))
    api = _forward_api(q, k, v, cu)
    assert api.check_support()
    assert api.scaling_seqlen == 128.0


@pytest.mark.L0
@pytest.mark.skipif(not _HAS_CUDA, reason="requires CUDA")
def test_support_rejects_unsupported_combinations(monkeypatch):
    q, k, v, do, cu = _inputs(head_dim=192)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 0))

    with pytest.raises(ValueError, match="head_dim"):
        HSTUBwdSm100(
            sample_do=do,
            sample_q=q,
            sample_k=k,
            sample_v=v,
            sample_dq=torch.empty_like(q),
            sample_dk=torch.empty_like(k),
            sample_dv=torch.empty_like(v),
            sample_cu_seqlens_q=cu,
            sample_cu_seqlens_k=cu,
            max_seqlen_q=128,
            max_seqlen_k=128,
        ).check_support()

    q, k, v, do, cu = _inputs()
    with pytest.raises(NotImplementedError, match="deterministic"):
        HSTUBwdSm100(
            sample_do=do,
            sample_q=q,
            sample_k=k,
            sample_v=v,
            sample_dq=torch.empty_like(q),
            sample_dk=torch.empty_like(k),
            sample_dv=torch.empty_like(v),
            sample_cu_seqlens_q=cu,
            sample_cu_seqlens_k=cu,
            max_seqlen_q=128,
            max_seqlen_k=128,
            deterministic=True,
        ).check_support()

    with pytest.raises(ValueError, match="scaling_seqlen"):
        _forward_api(q, k, v, cu, scaling_seqlen=0).check_support()

    even_func = torch.empty((1, 2, q.shape[0] + 256), dtype=torch.int32, device=q.device)
    with pytest.raises(ValueError, match="positive and odd"):
        HSTUFwdSm100(
            sample_q=q,
            sample_k=k,
            sample_v=v,
            sample_o=torch.empty_like(q),
            sample_cu_seqlens_q=cu,
            sample_cu_seqlens_k=cu,
            max_seqlen_q=128,
            max_seqlen_k=128,
            sample_func=even_func,
        ).check_support()

    odd_func = torch.empty((1, 1, q.shape[0] + 256), dtype=torch.int32, device=q.device)
    q_fp16, k_fp16, v_fp16, do_fp16, cu_fp16 = _inputs(dtype=torch.float16)
    assert HSTUFwdSm100(
        sample_q=q_fp16,
        sample_k=k_fp16,
        sample_v=v_fp16,
        sample_o=torch.empty_like(q_fp16),
        sample_cu_seqlens_q=cu_fp16,
        sample_cu_seqlens_k=cu_fp16,
        max_seqlen_q=128,
        max_seqlen_k=65537,
        sample_func=odd_func,
    ).check_support()

    assert HSTUBwdSm100(
        sample_do=do_fp16,
        sample_q=q_fp16,
        sample_k=k_fp16,
        sample_v=v_fp16,
        sample_dq=torch.empty_like(q_fp16),
        sample_dk=torch.empty_like(k_fp16),
        sample_dv=torch.empty_like(v_fp16),
        sample_cu_seqlens_q=cu_fp16,
        sample_cu_seqlens_k=cu_fp16,
        max_seqlen_q=32769,
        max_seqlen_k=32769,
        sample_func=odd_func,
    ).check_support()

    with pytest.raises(ValueError, match="o_tensor storage must not overlap"):
        HSTUFwdSm100(
            sample_q=q,
            sample_k=k,
            sample_v=v,
            sample_o=q,
            sample_cu_seqlens_q=cu,
            sample_cu_seqlens_k=cu,
            max_seqlen_q=128,
            max_seqlen_k=128,
        ).check_support()

    shared_grad = torch.empty_like(q)
    with pytest.raises(ValueError, match="storage must not overlap"):
        HSTUBwdSm100(
            sample_do=do,
            sample_q=q,
            sample_k=k,
            sample_v=v,
            sample_dq=shared_grad,
            sample_dk=shared_grad,
            sample_dv=torch.empty_like(v),
            sample_cu_seqlens_q=cu,
            sample_cu_seqlens_k=cu,
            max_seqlen_q=128,
            max_seqlen_k=128,
        ).check_support()


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
def test_raw_default_stream_maps_to_pytorch_default(monkeypatch):
    q, k, v, _, cu = _inputs()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 0))
    api = _forward_api(q, k, v, cu)
    api.check_support()
    api.compile()
    assert api.check_support()

    def reject_external_stream(*args, **kwargs):
        raise AssertionError("CUstream(0) must not construct an ExternalStream")

    monkeypatch.setattr(torch.cuda, "ExternalStream", reject_external_stream)
    api.execute(q, k, v, torch.empty_like(q), cu, cu, current_stream=cuda.CUstream(0))


@pytest.mark.L0
@pytest.mark.skipif(not _HAS_CUDA, reason="requires CUDA")
def test_wrapper_allocations_follow_explicit_stream(monkeypatch):
    q, k, v, do, cu_q = _inputs()
    cu_k = cu_q.clone()
    side_stream = torch.cuda.Stream()

    class _FakeApi:
        def execute(self, **_kwargs):
            return None

    monkeypatch.setattr(_api, "_cache_get", lambda *_args: _FakeApi())

    forward_allocation_streams = []
    original_empty = torch.empty

    def tracked_empty(*args, **kwargs):
        forward_allocation_streams.append(torch.cuda.current_stream(q.device).cuda_stream)
        return original_empty(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", tracked_empty)
        result = hstu_attention_forward(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q=128,
            max_seqlen_k=128,
            stream=side_stream,
        )
    assert forward_allocation_streams == [side_stream.cuda_stream]
    assert result["o_tensor"].device == q.device

    backward_allocation_streams = []
    original_empty_grad_like = _api._empty_grad_like

    def tracked_empty_grad_like(tensor):
        backward_allocation_streams.append(torch.cuda.current_stream(q.device).cuda_stream)
        return original_empty_grad_like(tensor)

    with monkeypatch.context() as patch:
        patch.setattr(_api, "_empty_grad_like", tracked_empty_grad_like)
        result = hstu_attention_backward(
            do,
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q=128,
            max_seqlen_k=128,
            stream=side_stream,
        )
    assert backward_allocation_streams == [side_stream.cuda_stream] * 3
    assert tuple(result.keys()) == ("dq_tensor", "dk_tensor", "dv_tensor")


@pytest.mark.L0
@pytest.mark.skipif(not _HAS_CUDA, reason="requires CUDA")
def test_explicit_stream_execute_records_all_operands(monkeypatch):
    q, k, v, do, cu_q = _inputs()
    cu_k = cu_q.clone()
    func = torch.zeros((1, 1, q.shape[0] + 256), dtype=torch.int32, device=q.device)
    side_stream = torch.cuda.Stream()
    recorded = []

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_args: (10, 0))

    def record_stream(tensor, stream):
        recorded.append((id(tensor), stream.cuda_stream))

    monkeypatch.setattr(torch.Tensor, "record_stream", record_stream)
    monkeypatch.setattr(_interface, "hstu_varlen_fwd_100", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_interface, "hstu_varlen_bwd_100", lambda *_args, **_kwargs: None)

    def assert_recorded(expected):
        assert recorded == [(id(tensor), side_stream.cuda_stream) for tensor in expected]
        recorded.clear()

    o = torch.empty_like(q)
    fwd = HSTUFwdSm100(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=o,
        sample_cu_seqlens_q=cu_q,
        sample_cu_seqlens_k=cu_k,
        max_seqlen_q=128,
        max_seqlen_k=128,
        sample_func=func,
    )
    assert fwd.check_support()
    fwd._compiled_kernel = object()
    fwd.execute(q, k, v, o, cu_q, cu_k, func_tensor=func, current_stream=side_stream)
    assert_recorded((q, k, v, o, cu_q, cu_k, func))

    paged_kv = torch.empty((1, 2, 128, q.shape[1], q.shape[2]), dtype=q.dtype, device=q.device)
    page_ids = torch.zeros(1, dtype=torch.int32, device=q.device)
    page_indptrs = torch.tensor((0, 1), dtype=torch.int32, device=q.device)
    paged_o = torch.empty_like(q)
    paged_fwd = HSTUFwdSm100(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=paged_o,
        sample_cu_seqlens_q=cu_q,
        sample_cu_seqlens_k=cu_k,
        max_seqlen_q=128,
        max_seqlen_k=128,
        window_size=(-1, 0),
        sample_paged_kv=paged_kv,
        sample_page_ids=page_ids,
        sample_page_indptrs=page_indptrs,
    )
    assert paged_fwd.check_support()
    paged_fwd._compiled_kernel = object()
    paged_fwd.execute(
        q,
        k,
        v,
        paged_o,
        cu_q,
        cu_k,
        paged_kv_tensor=paged_kv,
        page_ids_tensor=page_ids,
        page_indptrs_tensor=page_indptrs,
        current_stream=side_stream,
    )
    assert_recorded((q, k, v, paged_o, cu_q, cu_k, paged_kv, page_ids, page_indptrs))

    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    bwd = HSTUBwdSm100(
        sample_do=do,
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_dq=dq,
        sample_dk=dk,
        sample_dv=dv,
        sample_cu_seqlens_q=cu_q,
        sample_cu_seqlens_k=cu_k,
        max_seqlen_q=128,
        max_seqlen_k=128,
        sample_func=func,
    )
    assert bwd.check_support()
    bwd._compiled_kernel = object()
    bwd.execute(do, q, k, v, dq, dk, dv, cu_q, cu_k, func_tensor=func, current_stream=side_stream)
    assert_recorded((do, q, k, v, dq, dk, dv, cu_q, cu_k, func))


@pytest.mark.L0
@pytest.mark.skipif(not _HAS_CUDA, reason="requires CUDA")
def test_metadata_aliases_are_rejected_by_support_and_runtime(monkeypatch):
    q, k, v, do, cu = _inputs()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_args: (10, 0))
    monkeypatch.setattr(_interface, "hstu_varlen_fwd_100", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_interface, "hstu_varlen_bwd_100", lambda *_args, **_kwargs: None)

    def forward_case(metadata_name, alias):
        o = torch.empty_like(q)
        cu_q = cu.clone()
        cu_k = cu.clone()
        func = None
        paged_kv = None
        page_ids = None
        page_indptrs = None
        window_size = (-1, 0)
        if metadata_name == "func_tensor":
            func = torch.empty((1, 1, q.shape[0] + 256), dtype=torch.int32, device=q.device)
            window_size = (-1, -1)
        elif metadata_name in ("page_ids_tensor", "page_indptrs_tensor"):
            paged_kv = torch.empty((1, 2, 128, q.shape[1], q.shape[2]), dtype=q.dtype, device=q.device)
            page_ids = torch.zeros(1, dtype=torch.int32, device=q.device)
            page_indptrs = torch.tensor((0, 1), dtype=torch.int32, device=q.device)

        if alias:
            aliases = {
                "cu_seqlens_q_tensor": lambda: _int32_alias(o, cu_q.shape),
                "cu_seqlens_k_tensor": lambda: _int32_alias(o, cu_k.shape),
                "func_tensor": lambda: _int32_alias(o, func.shape),
                "page_ids_tensor": lambda: _int32_alias(o, page_ids.shape),
                "page_indptrs_tensor": lambda: _int32_alias(o, page_indptrs.shape),
            }
            alias_tensor = aliases[metadata_name]()
            if metadata_name == "cu_seqlens_q_tensor":
                cu_q = alias_tensor
            elif metadata_name == "cu_seqlens_k_tensor":
                cu_k = alias_tensor
            elif metadata_name == "func_tensor":
                func = alias_tensor
            elif metadata_name == "page_ids_tensor":
                page_ids = alias_tensor
            else:
                page_indptrs = alias_tensor

        api = HSTUFwdSm100(
            sample_q=q,
            sample_k=k,
            sample_v=v,
            sample_o=o,
            sample_cu_seqlens_q=cu_q,
            sample_cu_seqlens_k=cu_k,
            max_seqlen_q=128,
            max_seqlen_k=128,
            window_size=window_size,
            sample_func=func,
            sample_paged_kv=paged_kv,
            sample_page_ids=page_ids,
            sample_page_indptrs=page_indptrs,
        )
        runtime_kwargs = {
            "q_tensor": q,
            "k_tensor": k,
            "v_tensor": v,
            "o_tensor": o,
            "cu_seqlens_q_tensor": cu_q,
            "cu_seqlens_k_tensor": cu_k,
            "func_tensor": func,
            "paged_kv_tensor": paged_kv,
            "page_ids_tensor": page_ids,
            "page_indptrs_tensor": page_indptrs,
        }
        return api, runtime_kwargs

    for metadata_name in (
        "cu_seqlens_q_tensor",
        "cu_seqlens_k_tensor",
        "func_tensor",
        "page_ids_tensor",
        "page_indptrs_tensor",
    ):
        alias_api, _ = forward_case(metadata_name, alias=True)
        with pytest.raises(ValueError, match=rf"o_tensor storage must not overlap {metadata_name} storage"):
            alias_api.check_support()

        api, _ = forward_case(metadata_name, alias=False)
        assert api.check_support()
        api._compiled_kernel = object()
        _, runtime_kwargs = forward_case(metadata_name, alias=True)
        with pytest.raises(ValueError, match=rf"o_tensor storage must not overlap {metadata_name} storage"):
            api.execute(**runtime_kwargs)

    def backward_case(metadata_name, alias):
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        cu_q = cu.clone()
        cu_k = cu.clone()
        func = None
        if metadata_name == "func_tensor":
            func = torch.empty((1, 1, q.shape[0] + 256), dtype=torch.int32, device=q.device)
        if alias:
            if metadata_name == "cu_seqlens_q_tensor":
                cu_q = _int32_alias(dq, cu_q.shape)
            elif metadata_name == "cu_seqlens_k_tensor":
                cu_k = _int32_alias(dq, cu_k.shape)
            else:
                func = _int32_alias(dq, func.shape)
        api = HSTUBwdSm100(
            sample_do=do,
            sample_q=q,
            sample_k=k,
            sample_v=v,
            sample_dq=dq,
            sample_dk=dk,
            sample_dv=dv,
            sample_cu_seqlens_q=cu_q,
            sample_cu_seqlens_k=cu_k,
            max_seqlen_q=128,
            max_seqlen_k=128,
            sample_func=func,
        )
        runtime_kwargs = {
            "do_tensor": do,
            "q_tensor": q,
            "k_tensor": k,
            "v_tensor": v,
            "dq_tensor": dq,
            "dk_tensor": dk,
            "dv_tensor": dv,
            "cu_seqlens_q_tensor": cu_q,
            "cu_seqlens_k_tensor": cu_k,
            "func_tensor": func,
        }
        return api, runtime_kwargs

    for metadata_name in ("cu_seqlens_q_tensor", "cu_seqlens_k_tensor", "func_tensor"):
        alias_api, _ = backward_case(metadata_name, alias=True)
        with pytest.raises(ValueError, match=rf"dq_tensor storage must not overlap {metadata_name} storage"):
            alias_api.check_support()

        api, _ = backward_case(metadata_name, alias=False)
        assert api.check_support()
        api._compiled_kernel = object()
        _, runtime_kwargs = backward_case(metadata_name, alias=True)
        with pytest.raises(ValueError, match=rf"dq_tensor storage must not overlap {metadata_name} storage"):
            api.execute(**runtime_kwargs)


@pytest.mark.L0
@pytest.mark.skipif(not _HAS_CUDA, reason="requires CUDA")
def test_rejects_unsafe_storage_metadata(monkeypatch):
    q, k, v, _, cu = _inputs()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 0))

    overlapping_q = q[:1].expand_as(q)
    with pytest.raises(ValueError, match="non-overlapping strides"):
        _forward_api(overlapping_q, k, v, cu).check_support()

    unaligned_q = torch.empty(
        (q.shape[0], q.shape[1], q.shape[2] + 1),
        dtype=q.dtype,
        device=q.device,
    )[..., 1:]
    with pytest.raises(ValueError, match="16-byte aligned"):
        _forward_api(unaligned_q, k, v, cu).check_support()

    unaligned_cu = torch.arange(
        0,
        3 * 128,
        128,
        dtype=torch.int32,
        device=q.device,
    )[1:]
    with pytest.raises(ValueError, match="16-byte aligned"):
        _forward_api(q, k, v, unaligned_cu).check_support()

    paged_storage = torch.empty(
        (1, 2, 128, q.shape[1], q.shape[2] * 2),
        dtype=q.dtype,
        device=q.device,
    )
    noncontiguous_paged_kv = paged_storage[..., ::2]
    with pytest.raises(ValueError, match="paged_kv_tensor must be contiguous"):
        HSTUFwdSm100(
            sample_q=q,
            sample_k=k,
            sample_v=v,
            sample_o=torch.empty_like(q),
            sample_cu_seqlens_q=cu,
            sample_cu_seqlens_k=cu,
            max_seqlen_q=128,
            max_seqlen_k=128,
            window_size=(-1, 0),
            sample_paged_kv=noncontiguous_paged_kv,
            sample_page_ids=torch.tensor([0], dtype=torch.int32, device=q.device),
            sample_page_indptrs=torch.tensor([0, 1], dtype=torch.int32, device=q.device),
        ).check_support()

    with pytest.raises(ValueError, match="num_pages > 0"):
        HSTUFwdSm100(
            sample_q=q,
            sample_k=k,
            sample_v=v,
            sample_o=torch.empty_like(q),
            sample_cu_seqlens_q=cu,
            sample_cu_seqlens_k=cu,
            max_seqlen_q=128,
            max_seqlen_k=128,
            window_size=(-1, 0),
            sample_paged_kv=torch.empty(
                (0, 2, 128, q.shape[1], q.shape[2]),
                dtype=q.dtype,
                device=q.device,
            ),
            sample_page_ids=torch.tensor([0], dtype=torch.int32, device=q.device),
            sample_page_indptrs=torch.tensor([0, 1], dtype=torch.int32, device=q.device),
        ).check_support()

    qkv = torch.randn(
        (q.shape[0], 3, q.shape[1], q.shape[2]),
        dtype=q.dtype,
        device=q.device,
    )
    dqkv = torch.empty_like(qkv)
    packed_api = HSTUBwdSm100(
        sample_do=torch.empty_like(q),
        sample_q=qkv[:, 0],
        sample_k=qkv[:, 1],
        sample_v=qkv[:, 2],
        sample_dq=dqkv[:, 0],
        sample_dk=dqkv[:, 1],
        sample_dv=dqkv[:, 2],
        sample_cu_seqlens_q=cu,
        sample_cu_seqlens_k=cu,
        max_seqlen_q=128,
        max_seqlen_k=128,
    )
    assert packed_api.check_support()

    overlapping_grad_storage = torch.empty(
        (q.shape[0], q.shape[1], q.shape[2] + 8),
        dtype=q.dtype,
        device=q.device,
    )
    with pytest.raises(ValueError, match="storage must not overlap"):
        HSTUBwdSm100(
            sample_do=torch.empty_like(q),
            sample_q=q,
            sample_k=k,
            sample_v=v,
            sample_dq=overlapping_grad_storage[..., : q.shape[2]],
            sample_dk=overlapping_grad_storage[..., 8 : 8 + q.shape[2]],
            sample_dv=torch.empty_like(v),
            sample_cu_seqlens_q=cu,
            sample_cu_seqlens_k=cu,
            max_seqlen_q=128,
            max_seqlen_k=128,
        ).check_support()


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
def test_explicit_api_rejects_runtime_stride_change():
    q, k, v, _, cu = _inputs()
    api = _forward_api(q, k, v, cu)
    api.check_support()
    api.compile()
    q_head_major = q.permute(1, 0, 2).contiguous().permute(1, 0, 2)
    with pytest.raises(ValueError, match="dtype/device/stride"):
        api.execute(
            q_head_major,
            k,
            v,
            torch.empty_like(q),
            cu,
            cu,
        )

    misaligned_q = torch.empty(q.numel() + 1, dtype=q.dtype, device=q.device)[1:].view_as(q)
    with pytest.raises(ValueError, match="16-byte aligned"):
        api.execute(
            misaligned_q,
            k,
            v,
            torch.empty_like(q),
            cu,
            cu,
        )


@pytest.mark.L1
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [32, 64, 128, 256])
def test_forward_matches_pytorch(dtype, head_dim):
    q, k, v, _, cu = _inputs(dtype=dtype, head_dim=head_dim)
    alpha = 0.7
    scaling_seqlen = 64.0
    result = hstu_attention_forward(
        q,
        k,
        v,
        cu,
        cu,
        max_seqlen_q=128,
        max_seqlen_k=128,
        window_size=(-1, 0),
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
    )
    assert tuple(result.keys()) == ("o_tensor",)
    actual = result["o_tensor"]
    expected = _reference_forward(
        q,
        k,
        v,
        cu,
        cu,
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
        causal=True,
    )
    torch.testing.assert_close(
        actual.float(),
        expected,
        rtol=3e-2,
        atol=3e-2,
    )


@pytest.mark.L1
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [32, 64, 128, 256])
def test_backward_matches_pytorch(dtype, head_dim):
    q, k, v, do, cu = _inputs(dtype=dtype, head_dim=head_dim)
    alpha = 0.7
    scaling_seqlen = 64.0

    q_ref = q.float().detach().requires_grad_(True)
    k_ref = k.float().detach().requires_grad_(True)
    v_ref = v.float().detach().requires_grad_(True)
    out_ref = _reference_forward(
        q_ref,
        k_ref,
        v_ref,
        cu,
        cu,
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
        causal=True,
    )
    expected = torch.autograd.grad(
        out_ref,
        (q_ref, k_ref, v_ref),
        do.float(),
    )

    actual = hstu_attention_backward(
        do,
        q,
        k,
        v,
        cu,
        cu,
        max_seqlen_q=128,
        max_seqlen_k=128,
        window_size=(-1, 0),
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
    )
    assert tuple(actual.keys()) == ("dq_tensor", "dk_tensor", "dv_tensor")
    for name, expected_grad in zip(("dq_tensor", "dk_tensor", "dv_tensor"), expected):
        torch.testing.assert_close(
            actual[name].float(),
            expected_grad,
            rtol=6e-2,
            atol=6e-2,
        )


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
@pytest.mark.parametrize("head_dim", [32, 64, 256])
def test_backward_supports_optional_gradient_outputs(head_dim, monkeypatch):
    cache = OrderedDict()
    monkeypatch.setattr(_api, "_BWD_CACHE", cache)
    q, k, v, do, cu = _inputs(head_dim=head_dim)
    kwargs = {
        "max_seqlen_q": 128,
        "max_seqlen_k": 128,
        "window_size": (-1, 0),
        "alpha": 0.7,
        "scaling_seqlen": 64.0,
    }

    expected = hstu_attention_backward(do, q, k, v, cu, cu, **kwargs)
    assert len(cache) == 1

    padded_outputs = [
        torch.full(
            (reference.shape[0], reference.shape[1] + 2, reference.shape[2]),
            float("nan"),
            dtype=reference.dtype,
            device=reference.device,
        )
        for reference in (q, k, v)
    ]
    dq, dk, dv = (storage[:, : reference.shape[1]] for storage, reference in zip(padded_outputs, (q, k, v)))
    actual = hstu_attention_backward(
        do,
        q,
        k,
        v,
        cu,
        cu,
        **kwargs,
        dq_tensor=dq,
        dk_tensor=dk,
        dv_tensor=dv,
    )

    assert len(cache) == 2
    for name, output in zip(("dq_tensor", "dk_tensor", "dv_tensor"), (dq, dk, dv)):
        assert actual[name] is output
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-2, atol=1e-2)

    partial_dq_storage = torch.full_like(padded_outputs[0], float("nan"))
    partial_dq = partial_dq_storage[:, : q.shape[1]]
    partial = hstu_attention_backward(
        do,
        q,
        k,
        v,
        cu,
        cu,
        **kwargs,
        dq_tensor=partial_dq,
    )

    assert len(cache) == 3
    assert partial["dq_tensor"] is partial_dq
    for name in ("dq_tensor", "dk_tensor", "dv_tensor"):
        torch.testing.assert_close(partial[name], expected[name], rtol=1e-2, atol=1e-2)

    with pytest.raises(ValueError, match="dq_tensor storage must not overlap q_tensor storage"):
        hstu_attention_backward(
            do,
            q,
            k,
            v,
            cu,
            cu,
            **kwargs,
            dq_tensor=q,
        )
    assert len(cache) == 3


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
def test_d128_explicit_api_supports_padded_gradient_outputs(monkeypatch):
    _interface.hstu_varlen_bwd_100.compile_cache.clear()
    torch.manual_seed(456)
    seqlen, heads, padded_heads, head_dim = 128, 8, 16, 128
    alpha = 0.7
    scaling_seqlen = 64.0

    input_storage = [
        torch.randn(
            (seqlen, padded_heads, head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.2
        for _ in range(4)
    ]
    q, k, v, do = (storage[:, :heads] for storage in input_storage)
    cu = torch.tensor((0, seqlen), dtype=torch.int32, device="cuda")

    padding_sentinel = 7.0
    grad_storage = [
        torch.full(
            (seqlen, padded_heads, head_dim),
            padding_sentinel,
            dtype=torch.bfloat16,
            device="cuda",
        )
        for _ in range(3)
    ]
    dq, dk, dv = (storage[:, :heads] for storage in grad_storage)
    for grad in (dq, dk, dv):
        assert grad.stride() == (2048, 128, 1)
        assert not grad.is_contiguous()

    api = HSTUBwdSm100(
        sample_do=do,
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_dq=dq,
        sample_dk=dk,
        sample_dv=dv,
        sample_cu_seqlens_q=cu,
        sample_cu_seqlens_k=cu,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        window_size=(-1, 0),
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
    )
    assert api.check_support()
    api.compile()
    torch.cuda.synchronize()
    for storage in grad_storage:
        torch.testing.assert_close(
            storage,
            torch.full_like(storage, padding_sentinel),
            rtol=0,
            atol=0,
        )

    def fail_copy(*_args, **_kwargs):
        raise AssertionError("padded gradient outputs must be written directly")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "copy_", fail_copy)
        api.execute(do, q, k, v, dq, dk, dv, cu, cu)

    q_ref = q.float().detach().requires_grad_(True)
    k_ref = k.float().detach().requires_grad_(True)
    v_ref = v.float().detach().requires_grad_(True)
    out_ref = _reference_forward(
        q_ref,
        k_ref,
        v_ref,
        cu,
        cu,
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
        causal=True,
    )
    expected = torch.autograd.grad(
        out_ref,
        (q_ref, k_ref, v_ref),
        do.float(),
    )
    for actual, expected_grad, storage in zip((dq, dk, dv), expected, grad_storage):
        torch.testing.assert_close(
            actual.float(),
            expected_grad,
            rtol=8e-2,
            atol=8e-2,
        )
        torch.testing.assert_close(
            storage[:, heads:],
            torch.full_like(storage[:, heads:], padding_sentinel),
            rtol=0,
            atol=0,
        )


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
def test_explicit_api_reuses_direct_grad_kernel_for_aligned_strides(monkeypatch):
    _interface.hstu_varlen_bwd_100.compile_cache.clear()
    batch, seqlen, heads, head_dim = 2, 37, 2, 64
    torch.manual_seed(123)
    input_storage = [
        torch.randn(
            (batch * seqlen, heads, head_dim + 1),
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.2
        for _ in range(4)
    ]
    q, k, v, do = (storage[..., :head_dim] for storage in input_storage)
    cu = torch.arange(
        0,
        (batch + 1) * seqlen,
        seqlen,
        dtype=torch.int32,
        device="cuda",
    )
    assert all(not _interface._supports_bwd_original_qkv_layout(tensor) for tensor in (q, k, v, do))
    alpha = 0.7
    scaling_seqlen = 32.0
    padding_sentinel = 7.0

    q_ref = q.float().detach().requires_grad_(True)
    k_ref = k.float().detach().requires_grad_(True)
    v_ref = v.float().detach().requires_grad_(True)
    out_ref = _reference_forward(
        q_ref,
        k_ref,
        v_ref,
        cu,
        cu,
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
        causal=True,
    )
    expected = torch.autograd.grad(
        out_ref,
        (q_ref, k_ref, v_ref),
        do.float(),
    )

    def fail_copy(*_args, **_kwargs):
        raise AssertionError("aligned strided gradients must be written directly")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "copy_", fail_copy)
        for feature_paddings in ((8, 16, 24), (40, 48, 56)):
            grad_storage = [
                torch.full(
                    (batch * seqlen, heads, head_dim + padding),
                    padding_sentinel,
                    dtype=q.dtype,
                    device=q.device,
                )
                for padding in feature_paddings
            ]
            dq, dk, dv = (storage[..., :head_dim] for storage in grad_storage)
            for grad in (dq, dk, dv):
                assert grad.stride(0) % 8 == 0
                assert grad.stride(1) % 8 == 0
                assert grad.stride(0) % 64 != 0

            api = HSTUBwdSm100(
                sample_do=do,
                sample_q=q,
                sample_k=k,
                sample_v=v,
                sample_dq=dq,
                sample_dk=dk,
                sample_dv=dv,
                sample_cu_seqlens_q=cu,
                sample_cu_seqlens_k=cu,
                max_seqlen_q=seqlen,
                max_seqlen_k=seqlen,
                window_size=(-1, 0),
                alpha=alpha,
                scaling_seqlen=scaling_seqlen,
            )
            assert api.check_support()
            api.compile()
            assert len(_interface.hstu_varlen_bwd_100.compile_cache) == 1
            api.execute(do, q, k, v, dq, dk, dv, cu, cu)

            for actual, expected_grad, storage in zip((dq, dk, dv), expected, grad_storage):
                torch.testing.assert_close(
                    actual.float(),
                    expected_grad,
                    rtol=8e-2,
                    atol=8e-2,
                )
                torch.testing.assert_close(
                    storage[..., head_dim:],
                    torch.full_like(
                        storage[..., head_dim:],
                        padding_sentinel,
                    ),
                    rtol=0,
                    atol=0,
                )


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
def test_d256_explicit_compile_and_packed_gradient_outputs():
    from cudnn.hstu_attention._kernels.hstu_bwd_256_cute import (
        hstu_varlen_bwd_256_cute,
    )

    hstu_varlen_bwd_256_cute.compile_cache.clear()
    torch.manual_seed(456)
    seqlen, heads, head_dim = 65, 2, 256
    alpha = 0.7
    scaling_seqlen = 37.0
    qkv = (
        torch.randn(
            (seqlen, 3, heads, head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.2
    )
    q, k, v = qkv.unbind(1)
    do = (
        torch.randn(
            (seqlen, head_dim, heads),
            dtype=torch.bfloat16,
            device="cuda",
        )
        .transpose(1, 2)
        .mul_(0.2)
    )
    cu = torch.tensor((0, seqlen), dtype=torch.int32, device="cuda")
    dqkv = torch.full_like(qkv, float("nan"))

    api = HSTUBwdSm100(
        sample_do=do,
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_dq=dqkv[:, 0],
        sample_dk=dqkv[:, 1],
        sample_dv=dqkv[:, 2],
        sample_cu_seqlens_q=cu,
        sample_cu_seqlens_k=cu,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        window_size=(-1, 0),
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
    )
    api.check_support()
    api.compile()
    torch.cuda.synchronize()
    assert torch.isnan(dqkv.float()).all()

    api.execute(
        do,
        q,
        k,
        v,
        dqkv[:, 0],
        dqkv[:, 1],
        dqkv[:, 2],
        cu,
        cu,
    )
    assert torch.isfinite(dqkv.float()).all()

    q_ref = q.float().detach().requires_grad_(True)
    k_ref = k.float().detach().requires_grad_(True)
    v_ref = v.float().detach().requires_grad_(True)
    out_ref = _reference_forward(
        q_ref,
        k_ref,
        v_ref,
        cu,
        cu,
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
        causal=True,
    )
    expected = torch.autograd.grad(
        out_ref,
        (q_ref, k_ref, v_ref),
        do.float(),
    )
    for actual, expected_grad in zip(dqkv.unbind(1), expected):
        torch.testing.assert_close(
            actual.float(),
            expected_grad,
            rtol=8e-2,
            atol=8e-2,
        )


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
def test_d256_explicit_api_is_cuda_graph_capturable():
    q, k, v, do, cu = _inputs(seqlen=64, head_dim=256)
    dq, dk, dv = (torch.empty_like(q), torch.empty_like(k), torch.empty_like(v))
    api = HSTUBwdSm100(
        sample_do=do,
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_dq=dq,
        sample_dk=dk,
        sample_dv=dv,
        sample_cu_seqlens_q=cu,
        sample_cu_seqlens_k=cu,
        max_seqlen_q=64,
        max_seqlen_k=64,
        window_size=(-1, 0),
        alpha=0.7,
        scaling_seqlen=32.0,
    )
    api.check_support()
    api.compile()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        api.execute(do, q, k, v, dq, dk, dv, cu, cu)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    assert all(torch.isfinite(grad.float()).all() for grad in (dq, dk, dv))


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
@pytest.mark.parametrize("head_dim", [32, 64, 256])
def test_varlen_tail_and_asymmetric_lengths_match_pytorch(head_dim):
    torch.manual_seed(321)
    q_lengths = (37, 129)
    k_lengths = (50, 173)
    heads = 1
    q = (
        torch.randn(
            (sum(q_lengths), heads, head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.2
    )
    k = (
        torch.randn(
            (sum(k_lengths), heads, head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.2
    )
    v = torch.randn_like(k) * 0.2
    do = torch.randn_like(q) * 0.2
    cu_q = torch.tensor([0, q_lengths[0], sum(q_lengths)], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, k_lengths[0], sum(k_lengths)], dtype=torch.int32, device="cuda")
    alpha = 0.7
    scaling_seqlen = 96.0

    expected_out = _reference_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
        causal=True,
    )
    actual_out = hstu_attention_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        window_size=(-1, 0),
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
    )["o_tensor"]
    torch.testing.assert_close(actual_out.float(), expected_out, rtol=4e-2, atol=4e-2)

    q_ref = q.float().detach().requires_grad_(True)
    k_ref = k.float().detach().requires_grad_(True)
    v_ref = v.float().detach().requires_grad_(True)
    ref_for_grad = _reference_forward(
        q_ref,
        k_ref,
        v_ref,
        cu_q,
        cu_k,
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
        causal=True,
    )
    expected_grads = torch.autograd.grad(
        ref_for_grad,
        (q_ref, k_ref, v_ref),
        do.float(),
    )
    actual_grads = hstu_attention_backward(
        do,
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        window_size=(-1, 0),
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
    )
    for name, expected_grad in zip(("dq_tensor", "dk_tensor", "dv_tensor"), expected_grads):
        torch.testing.assert_close(actual_grads[name].float(), expected_grad, rtol=8e-2, atol=8e-2)


@pytest.mark.L0
@pytest.mark.parametrize(
    ("capability", "supported", "head_dim", "is_local", "window_size_left", "expected"),
    (
        ((10, 0), True, 64, False, None, _interface._Q1_FWD_D64_D128_CONFIG),
        ((10, 3), True, 128, True, 255, _interface._Q1_FWD_D64_D128_CONFIG),
        ((10, 7), True, 256, False, None, _interface._Q1_FWD_D256_CONFIG),
        ((10, 0), True, 256, True, 255, _interface._Q1_FWD_DEFAULT_CONFIG),
        ((10, 3), True, 256, True, 255, _interface._Q1_FWD_DEFAULT_CONFIG),
        ((10, 7), True, 256, True, 255, _interface._Q1_FWD_DEFAULT_CONFIG),
        ((10, 3), True, 256, True, 256, _interface._Q1_FWD_D256_CONFIG),
        ((10, 7), True, 256, True, 511, _interface._Q1_FWD_D256_CONFIG),
        ((10, 7), False, 32, False, None, _interface._Q1_FWD_DEFAULT_CONFIG),
        ((9, 0), True, 128, False, None, _interface._Q1_FWD_DEFAULT_CONFIG),
    ),
)
def test_single_query_forward_config_selector(capability, supported, head_dim, is_local, window_size_left, expected):
    assert _interface._select_q1_fwd_config(capability, supported, head_dim, is_local, window_size_left) == expected


@pytest.mark.L0
@pytest.mark.parametrize(
    ("capability", "supported", "batch_size", "heads", "average_kv", "head_dim", "expected"),
    (
        # Missing workload metadata preserves the architecture default.
        ((10, 0), True, None, None, None, None, 8),
        ((10, 3), True, None, None, None, None, 8),
        ((10, 7), True, None, None, None, None, 13),
        # Short D64/D128 work uses the unsplit kernel on every target.
        ((10, 0), True, 64, 4, 255, 128, 1),
        ((10, 0), True, 64, 4, 256, 128, 8),
        ((10, 3), True, 64, 4, 255, 128, 1),
        ((10, 3), True, 64, 4, 256, 128, 8),
        ((10, 7), True, 64, 4, 255, 128, 1),
        ((10, 7), True, 64, 4, 256, 128, 13),
        # D256 retains split-KV at a medium grid size.
        ((10, 0), True, 64, 4, 127, 256, 1),
        ((10, 0), True, 64, 8, 127, 256, 8),
        ((10, 3), True, 64, 4, 127, 256, 1),
        ((10, 3), True, 64, 8, 127, 256, 8),
        ((10, 7), True, 64, 4, 127, 256, 1),
        ((10, 7), True, 64, 8, 127, 256, 13),
        # An already-saturated grid extends the unsplit range to 768.
        ((10, 0), True, 1024, 4, 767, 128, 1),
        ((10, 0), True, 1024, 4, 768, 128, 8),
        ((10, 3), True, 1024, 4, 767, 128, 1),
        ((10, 3), True, 1024, 4, 768, 128, 8),
        ((10, 7), True, 1024, 4, 767, 128, 1),
        ((10, 7), True, 1024, 4, 768, 128, 13),
        ((10, 7), False, 64, 4, 2048, 128, 1),
    ),
)
def test_single_query_backward_split_selector(capability, supported, batch_size, heads, average_kv, head_dim, expected):
    total_kv = None if batch_size is None or average_kv is None else batch_size * average_kv
    assert (
        _interface._select_q1_bwd_split_kv(
            "auto",
            capability,
            supported,
            batch_size=batch_size,
            num_heads=heads,
            total_kv=total_kv,
            head_dim=head_dim,
        )
        == expected
    )


@pytest.mark.L0
@pytest.mark.parametrize(
    ("capability", "batch_size", "heads", "split_kv", "head_dim", "expected"),
    (
        ((10, 0), 64, 4, 1, 128, 512),
        ((10, 0), 512, 4, 1, 128, 128),
        ((10, 0), 512, 4, 1, 256, 256),
        ((10, 0), 64, 4, 8, 256, 128),
        ((10, 3), 64, 4, 1, 128, 512),
        ((10, 3), 512, 4, 1, 128, 128),
        ((10, 3), 512, 4, 1, 256, 256),
        ((10, 3), 64, 4, 8, 256, 128),
        ((10, 7), 64, 4, 1, 128, 512),
        ((10, 7), 512, 4, 1, 128, 128),
        ((10, 7), 512, 4, 1, 256, 256),
        ((10, 7), 64, 4, 13, 256, 128),
        ((9, 0), 512, 4, 1, 256, 256),
    ),
)
def test_single_query_backward_thread_selector(capability, batch_size, heads, split_kv, head_dim, expected):
    assert _interface._select_q1_bwd_num_threads(capability, batch_size, heads, split_kv, head_dim) == expected


@pytest.mark.L0
@pytest.mark.parametrize("algorithm", ("tc", "tc-small", "legacy"))
def test_single_query_backward_rejects_removed_algorithms(algorithm):
    shape = (1, 1, 64)
    q = torch.empty(shape, dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    do = torch.empty_like(q)
    cu = torch.tensor((0, 1), dtype=torch.int32)

    with pytest.raises(ValueError, match="Unsupported qlen=1 backward algorithm"):
        _interface.hstu_varlen_bwd_100(
            do,
            q,
            k,
            v,
            cu,
            cu,
            1,
            1,
            None,
            None,
            None,
            -1,
            0,
            1.0,
            None,
            False,
            _compile_only=True,
            _q1_bwd_algorithm=algorithm,
        )


@pytest.mark.L1
@pytest.mark.skipif(not _IS_Q1_SPLIT_TARGET, reason="requires an SM100, SM103, or SM107 GPU")
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16), ids=("bf16", "fp16"))
@pytest.mark.parametrize("head_dim", (64, 128, 256))
def test_single_query_auto_forward_uses_optimized_schedule_and_matches_pytorch(head_dim, dtype):
    _interface.hstu_varlen_fwd_100.compile_cache.clear()
    torch.manual_seed(2027)
    batch, heads = 64, 1
    k_lengths = (1024, 2048, 2560, 3072) * (batch // 4)
    q = torch.randn((batch, heads, head_dim), dtype=dtype, device="cuda") * 0.2
    k = torch.randn((sum(k_lengths), heads, head_dim), dtype=dtype, device="cuda") * 0.2
    v = torch.randn_like(k) * 0.2
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    cu_k = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
    cu_k[1:] = torch.tensor(k_lengths, dtype=torch.int32, device="cuda").cumsum(0)
    alpha = 0.7
    scaling_seqlen = 2048.0

    actual = hstu_attention_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=1,
        max_seqlen_k=max(k_lengths),
        window_size=(-1, 0),
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
    )["o_tensor"]
    expected = _reference_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
        causal=True,
    )
    torch.cuda.synchronize()

    compile_key = next(iter(_interface.hstu_varlen_fwd_100.compile_cache))
    assert compile_key[3] == 64
    assert compile_key[13] == 1
    assert compile_key[-1] == (0 if head_dim == 256 else 5)
    torch.testing.assert_close(actual.float(), expected, rtol=4e-2, atol=4e-2)


@pytest.mark.L1
@pytest.mark.skipif(not _IS_Q1_SPLIT_TARGET, reason="requires an SM100, SM103, or SM107 GPU")
def test_single_query_persistent_forward_is_repeatable():
    """A one-stage TMEM output accumulator must not be reused before epilogue release."""
    _interface.hstu_varlen_fwd_100.compile_cache.clear()
    batch, heads, head_dim = 64, 4, 128
    k_pattern = (512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536)
    k_lengths = tuple(k_pattern[index % len(k_pattern)] for index in range(batch))
    q = torch.full((batch, heads, head_dim), 0.125, dtype=torch.float16, device="cuda")
    k = torch.full((sum(k_lengths), heads, head_dim), 0.125, dtype=torch.float16, device="cuda")
    v = torch.full_like(k, 0.125)
    out = torch.empty_like(q)
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    cu_k = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
    cu_k[1:] = torch.tensor(k_lengths, dtype=torch.int32, device="cuda").cumsum(0)
    alpha = 0.7
    scaling_seqlen = 2048.0
    window_size = (-1, 0)

    score = alpha * head_dim * 0.125 * 0.125
    weight = score / (1.0 + math.exp(-score))
    expected = (torch.tensor(k_lengths, dtype=torch.float32, device="cuda")[:, None, None] * (weight * 0.125 / scaling_seqlen)).expand(
        -1,
        heads,
        head_dim,
    )

    api = HSTUFwdSm100(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=out,
        sample_cu_seqlens_q=cu_q,
        sample_cu_seqlens_k=cu_k,
        max_seqlen_q=1,
        max_seqlen_k=max(k_lengths),
        window_size=window_size,
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
    )
    api.check_support()
    api.compile()

    first = None
    for repeat in range(32):
        out.fill_(float("nan") if repeat % 2 == 0 else -13.0)
        api.execute(q, k, v, out, cu_q, cu_k)
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-4)
        if first is None:
            first = out.detach().clone()
        else:
            assert torch.equal(out, first)


@pytest.mark.L1
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16), ids=("bf16", "fp16"))
@pytest.mark.parametrize("head_dim", (64, 128, 256))
def test_single_query_auto_backward_matches_pytorch(head_dim, dtype):
    _interface._hstu_varlen_bwd_q1_direct.compile_cache.clear()
    _interface.hstu_varlen_bwd_100.compile_cache.clear()
    torch.manual_seed(2026)
    batch, heads = 256, 1
    k_lengths = (1, 127, 128, 257) * (batch // 4)
    q = torch.randn((batch, heads, head_dim), dtype=dtype, device="cuda") * 0.2
    k = torch.randn((sum(k_lengths), heads, head_dim), dtype=dtype, device="cuda") * 0.2
    v = torch.randn_like(k) * 0.2
    do = torch.randn_like(q) * 0.2
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    cu_k = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
    cu_k[1:] = torch.tensor(k_lengths, dtype=torch.int32, device="cuda").cumsum(0)
    alpha = 0.7
    scaling_seqlen = 256.0

    actual = hstu_attention_backward(
        do,
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=1,
        max_seqlen_k=max(k_lengths),
        window_size=(-1, 0),
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
    )
    torch.cuda.synchronize()

    q_ref = q.cpu().float().requires_grad_(True)
    k_ref = k.cpu().float().requires_grad_(True)
    v_ref = v.cpu().float().requires_grad_(True)
    out_ref = _reference_forward(
        q_ref,
        k_ref,
        v_ref,
        cu_q,
        cu_k,
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
        causal=True,
    )
    expected = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), do.cpu().float())

    assert len(_interface._hstu_varlen_bwd_q1_direct.compile_cache) == 1
    split_kv = _interface._select_q1_bwd_split_kv(
        "auto",
        torch.cuda.get_device_capability(),
        dtype == torch.bfloat16,
        batch_size=batch,
        num_heads=heads,
        total_kv=sum(k_lengths),
        head_dim=head_dim,
    )
    expected_schedule = (split_kv, 256 // head_dim)
    assert next(iter(_interface._hstu_varlen_bwd_q1_direct.compile_cache))[-2:] == expected_schedule
    for name, expected_grad in zip(("dq_tensor", "dk_tensor", "dv_tensor"), expected):
        torch.testing.assert_close(actual[name].cpu().float(), expected_grad, rtol=8e-2, atol=8e-2)


@pytest.mark.L1
@pytest.mark.skipif(not _IS_SM107, reason="requires an SM107 GPU")
def test_single_query_sm107_auto_backward_uses_direct_kernel_and_matches_pytorch():
    _interface._hstu_varlen_bwd_q1_direct.compile_cache.clear()
    _interface.hstu_varlen_bwd_100.compile_cache.clear()
    torch.manual_seed(2030)
    batch, heads, head_dim = 8, 4, 128
    k_lengths = (3072,) * batch
    q = torch.randn((batch, heads, head_dim), dtype=torch.float16, device="cuda") * 0.2
    k = torch.randn((sum(k_lengths), heads, head_dim), dtype=torch.float16, device="cuda") * 0.2
    v = torch.randn_like(k) * 0.2
    do = torch.randn_like(q) * 0.2
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    cu_k = torch.arange(0, sum(k_lengths) + 1, k_lengths[0], dtype=torch.int32, device="cuda")
    alpha = 0.7
    scaling_seqlen = 2048.0

    actual = hstu_attention_backward(
        do,
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=1,
        max_seqlen_k=max(k_lengths),
        window_size=(-1, 0),
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
    )
    torch.cuda.synchronize()

    q_ref = q.cpu().float().requires_grad_(True)
    k_ref = k.cpu().float().requires_grad_(True)
    v_ref = v.cpu().float().requires_grad_(True)
    out_ref = _reference_forward(
        q_ref,
        k_ref,
        v_ref,
        cu_q,
        cu_k,
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
        causal=True,
    )
    expected = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), do.cpu().float())

    assert len(_interface._hstu_varlen_bwd_q1_direct.compile_cache) == 1
    assert not _interface.hstu_varlen_bwd_100.compile_cache
    for name, expected_grad in zip(("dq_tensor", "dk_tensor", "dv_tensor"), expected):
        torch.testing.assert_close(actual[name].cpu().float(), expected_grad, rtol=8e-2, atol=8e-2)


@pytest.mark.L1
@pytest.mark.skipif(not _IS_Q1_SPLIT_TARGET, reason="requires an SM100, SM103, or SM107 GPU")
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16), ids=("bf16", "fp16"))
@pytest.mark.parametrize("head_dim", (64, 128, 256))
def test_single_query_local_window_uses_optimized_paths_and_matches_pytorch(head_dim, dtype):
    _interface.hstu_varlen_fwd_100.compile_cache.clear()
    _interface._hstu_varlen_bwd_q1_direct.compile_cache.clear()
    _interface.hstu_varlen_bwd_100.compile_cache.clear()
    torch.manual_seed(2028)
    k_lengths = (1, 63, 64, 127, 128, 129, 257)
    batch, heads = len(k_lengths), 1
    q = torch.randn((batch, heads, head_dim), dtype=dtype, device="cuda") * 0.2
    k = torch.randn((sum(k_lengths), heads, head_dim), dtype=dtype, device="cuda") * 0.2
    v = torch.randn_like(k) * 0.2
    do = torch.randn_like(q) * 0.2
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    cu_k = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
    cu_k[1:] = torch.tensor(k_lengths, dtype=torch.int32, device="cuda").cumsum(0)
    alpha = 0.7
    scaling_seqlen = 256.0

    for window_size in ((0, 0), (1, 0), (63, 0), (64, 8), (128, 0), (255, 0)):
        q_ref = q.cpu().float().requires_grad_(True)
        k_ref = k.cpu().float().requires_grad_(True)
        v_ref = v.cpu().float().requires_grad_(True)
        expected_out = _reference_forward(
            q_ref,
            k_ref,
            v_ref,
            cu_q,
            cu_k,
            alpha=alpha,
            scaling_seqlen=scaling_seqlen,
            causal=False,
            window_size=window_size,
        )
        expected_grads = torch.autograd.grad(expected_out, (q_ref, k_ref, v_ref), do.cpu().float())

        actual_out = hstu_attention_forward(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q=1,
            max_seqlen_k=max(k_lengths),
            window_size=window_size,
            alpha=alpha,
            scaling_seqlen=scaling_seqlen,
        )["o_tensor"]
        actual_grads = hstu_attention_backward(
            do,
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q=1,
            max_seqlen_k=max(k_lengths),
            window_size=window_size,
            alpha=alpha,
            scaling_seqlen=scaling_seqlen,
        )
        torch.testing.assert_close(actual_out.cpu().float(), expected_out, rtol=4e-2, atol=4e-2)
        for name, expected_grad in zip(("dq_tensor", "dk_tensor", "dv_tensor"), expected_grads):
            torch.testing.assert_close(actual_grads[name].cpu().float(), expected_grad, rtol=8e-2, atol=8e-2)

    assert len(_interface.hstu_varlen_fwd_100.compile_cache) == 1
    fwd_compile_key = next(iter(_interface.hstu_varlen_fwd_100.compile_cache))
    assert fwd_compile_key[3] == (128 if head_dim == 256 else 64)
    assert fwd_compile_key[6]
    assert fwd_compile_key[-1] == (0 if head_dim == 256 else 5)
    assert not _interface.hstu_varlen_bwd_100.compile_cache
    assert len(_interface._hstu_varlen_bwd_q1_direct.compile_cache) == 1
    bwd_compile_key = next(iter(_interface._hstu_varlen_bwd_q1_direct.compile_cache))
    assert bwd_compile_key[-3]
    split_kv = _interface._select_q1_bwd_split_kv(
        "auto",
        torch.cuda.get_device_capability(),
        dtype == torch.bfloat16,
        batch_size=batch,
        num_heads=heads,
        total_kv=sum(k_lengths),
        head_dim=head_dim,
    )
    expected_schedule = (split_kv, 256 // head_dim)
    assert bwd_compile_key[-2:] == expected_schedule


@pytest.mark.L0
@pytest.mark.skipif(not _IS_Q1_SPLIT_TARGET, reason="requires an SM100, SM103, or SM107 GPU")
@pytest.mark.parametrize("head_dim", (32, 64, 128, 256))
def test_single_query_forward_cache_reuses_runtime_shapes(head_dim):
    """Packed token and batch extents must re-bind one compiled artifact."""
    _interface.hstu_varlen_fwd_100.compile_cache.clear()

    for batch_size, kv_len in ((2, 128), (3, 192)):
        q = torch.full((batch_size, 1, head_dim), 0.125, dtype=torch.bfloat16, device="cuda")
        k = torch.full((batch_size * kv_len, 1, head_dim), 0.125, dtype=torch.bfloat16, device="cuda")
        v = torch.full_like(k, 0.125)
        cu_q = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda")
        cu_k = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda") * kv_len
        out = torch.empty_like(q)
        actual, _ = _interface.hstu_varlen_fwd_100(
            q,
            k,
            v,
            cu_q,
            cu_k,
            1,
            kv_len,
            63,
            0,
            0.7,
            None,
            scaling_seqlen=256.0,
            out=out,
        )
        expected = _reference_forward(
            q.cpu().float(),
            k.cpu().float(),
            v.cpu().float(),
            cu_q,
            cu_k,
            alpha=0.7,
            scaling_seqlen=256.0,
            causal=False,
            window_size=(63, 0),
        )
        torch.testing.assert_close(actual.cpu().float(), expected, rtol=4e-2, atol=4e-2)

    assert len(_interface.hstu_varlen_fwd_100.compile_cache) == 1
    key = next(iter(_interface.hstu_varlen_fwd_100.compile_cache))
    assert key[13] == 1
    assert key[14] == 1


@pytest.mark.L0
@pytest.mark.skipif(not _IS_Q1_SPLIT_TARGET, reason="requires an SM100, SM103, or SM107 GPU")
@pytest.mark.parametrize("head_dim", (64, 128, 256))
def test_single_query_backward_cache_reuses_runtime_shapes(head_dim):
    """Packed Q/K totals and batch size must re-bind one backward artifact."""
    _interface._hstu_varlen_bwd_q1_direct.compile_cache.clear()

    for batch_size, kv_len in ((2, 128), (3, 192)):
        q = torch.full((batch_size, 1, head_dim), 0.125, dtype=torch.bfloat16, device="cuda")
        k = torch.full((batch_size * kv_len, 1, head_dim), 0.125, dtype=torch.bfloat16, device="cuda")
        v = torch.full_like(k, 0.125)
        do = torch.full_like(q, 0.125)
        cu_q = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda")
        cu_k = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda") * kv_len
        actual = hstu_attention_backward(
            do,
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q=1,
            max_seqlen_k=kv_len,
            window_size=(63, 0),
            alpha=0.7,
            scaling_seqlen=256.0,
        )

        q_ref = q.cpu().float().requires_grad_(True)
        k_ref = k.cpu().float().requires_grad_(True)
        v_ref = v.cpu().float().requires_grad_(True)
        out_ref = _reference_forward(
            q_ref,
            k_ref,
            v_ref,
            cu_q,
            cu_k,
            alpha=0.7,
            scaling_seqlen=256.0,
            causal=False,
            window_size=(63, 0),
        )
        expected = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), do.cpu().float())
        for name, expected_grad in zip(("dq_tensor", "dk_tensor", "dv_tensor"), expected):
            torch.testing.assert_close(actual[name].cpu().float(), expected_grad, rtol=8e-2, atol=8e-2)

    assert len(_interface._hstu_varlen_bwd_q1_direct.compile_cache) == 1


@pytest.mark.L1
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
@pytest.mark.parametrize("mask_mode", ["full", "local", "arbitrary"])
@pytest.mark.parametrize("head_dim", [32, 64, 128, 256])
def test_mask_modes_match_pytorch(mask_mode, head_dim):
    q, k, v, do, cu = _inputs(heads=1, head_dim=head_dim)
    seqlen = q.shape[0]
    alpha = 0.7
    scaling_seqlen = 64.0
    row = torch.arange(seqlen, device=q.device)[:, None]
    col = torch.arange(seqlen, device=q.device)[None, :]

    window_size = (-1, -1)
    func = None
    mask = torch.ones((seqlen, seqlen), dtype=torch.bool, device=q.device)
    if mask_mode == "local":
        window_size = (16, 8)
        mask = (col >= row - 16) & (col <= row + 8)
    elif mask_mode == "arbitrary":
        func = torch.full(
            (1, 3, seqlen + 256),
            seqlen,
            dtype=torch.int32,
            device=q.device,
        )
        masked_start = (row[:, 0] * 3) % 32
        masked_end = masked_start + 7
        valid_upper = 96 + row[:, 0] % 32
        func[0, 0, :seqlen] = masked_start
        func[0, 1, :seqlen] = masked_end
        func[0, 2, :seqlen] = valid_upper
        mask = (col < masked_start[:, None]) | ((col >= masked_end[:, None]) & (col < valid_upper[:, None]))

    q_ref = q.float().detach().requires_grad_(True)
    k_ref = k.float().detach().requires_grad_(True)
    v_ref = v.float().detach().requires_grad_(True)
    scores = alpha * torch.einsum("qhd,khd->hqk", q_ref, k_ref)
    weights = torch.where(
        mask.unsqueeze(0),
        F.silu(scores),
        torch.zeros_like(scores),
    )
    out_ref = torch.einsum("hqk,khd->qhd", weights, v_ref) / scaling_seqlen

    out = hstu_attention_forward(
        q,
        k,
        v,
        cu,
        cu,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        window_size=window_size,
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
        func_tensor=func,
    )["o_tensor"]
    torch.testing.assert_close(out.float(), out_ref, rtol=4e-2, atol=4e-2)

    expected_grads = torch.autograd.grad(
        out_ref,
        (q_ref, k_ref, v_ref),
        do.float(),
    )
    actual_grads = hstu_attention_backward(
        do,
        q,
        k,
        v,
        cu,
        cu,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        window_size=window_size,
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
        func_tensor=func,
    )
    for name, expected_grad in zip(("dq_tensor", "dk_tensor", "dv_tensor"), expected_grads):
        torch.testing.assert_close(
            actual_grads[name].float(),
            expected_grad,
            rtol=8e-2,
            atol=8e-2,
        )


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
def test_d256_full_tile_arbitrary_mask_is_not_skipped():
    torch.manual_seed(654)
    seqlen, heads, head_dim = 128, 1, 256
    alpha = 0.7
    scaling_seqlen = 128.0
    q, k, v, do = [
        torch.empty(
            (seqlen, heads, head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        ).uniform_(-1.0, 1.0)
        for _ in range(4)
    ]
    cu = torch.tensor((0, seqlen), dtype=torch.int32, device="cuda")
    func = torch.empty(
        (1, 3, seqlen + 256),
        dtype=torch.int32,
        device="cuda",
    )
    func[:, 0, :] = 0
    func[:, 1, :] = 96
    func[:, 2, :] = 112

    q_ref = q.float().detach().requires_grad_(True)
    k_ref = k.float().detach().requires_grad_(True)
    v_ref = v.float().detach().requires_grad_(True)
    scores = alpha * torch.einsum("qhd,khd->hqk", q_ref, k_ref)
    columns = torch.arange(seqlen, device=q.device)[None, :]
    mask = ((columns >= 96) & (columns < 112)).unsqueeze(0)
    out_ref = (
        torch.einsum(
            "hqk,khd->qhd",
            torch.where(mask, F.silu(scores), torch.zeros_like(scores)),
            v_ref,
        )
        / scaling_seqlen
    )
    expected = torch.autograd.grad(
        out_ref,
        (q_ref, k_ref, v_ref),
        do.float(),
    )
    actual = hstu_attention_backward(
        do,
        q,
        k,
        v,
        cu,
        cu,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        window_size=(-1, -1),
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
        func_tensor=func,
    )

    for name, expected_grad in zip(("dq_tensor", "dk_tensor", "dv_tensor"), expected):
        max_error = (actual[name].float() - expected_grad).abs().max()
        relative_error = max_error / (expected_grad.abs().max() + 1.0e-12)
        assert relative_error < 3.0e-2, f"{name} relative max error is {relative_error.item():.4e}"


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
@pytest.mark.parametrize("head_dim", [32, 64, 256])
def test_paged_kv_forward_matches_pytorch(head_dim):
    q, k, v, _, cu = _inputs(heads=1, seqlen=256, head_dim=head_dim)
    seqlen = q.shape[0]
    alpha = 0.7
    scaling_seqlen = 64.0
    paged_kv = (
        torch.randn(
            (3, 2, 128, q.shape[1], q.shape[2]),
            dtype=q.dtype,
            device=q.device,
        )
        * 0.2
    )
    page_ids = torch.tensor([2, 0], dtype=torch.int32, device=q.device)
    page_indptrs = torch.tensor([0, 2], dtype=torch.int32, device=q.device)
    expected_k = torch.cat((paged_kv[2, 0], paged_kv[0, 0]), dim=0)
    expected_v = torch.cat((paged_kv[2, 1], paged_kv[0, 1]), dim=0)

    out = hstu_attention_forward(
        q,
        k,
        v,
        cu,
        cu,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        window_size=(-1, 0),
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
        paged_kv_tensor=paged_kv,
        page_ids_tensor=page_ids,
        page_indptrs_tensor=page_indptrs,
    )["o_tensor"]

    scores = alpha * torch.einsum("qhd,khd->hqk", q.float(), expected_k.float())
    row = torch.arange(seqlen, device=q.device)[:, None]
    col = torch.arange(seqlen, device=q.device)[None, :]
    weights = torch.where(
        (col <= row).unsqueeze(0),
        F.silu(scores),
        torch.zeros_like(scores),
    )
    expected = torch.einsum("hqk,khd->qhd", weights, expected_v.float()) / scaling_seqlen
    torch.testing.assert_close(out.float(), expected, rtol=4e-2, atol=4e-2)


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
def test_cache_hit_does_not_inspect_cuda_metadata_values(monkeypatch):
    q, k, v, do, cu = _inputs()
    kwargs = {
        "max_seqlen_q": 128,
        "max_seqlen_k": 128,
        "window_size": (-1, 0),
        "scaling_seqlen": 64.0,
    }
    hstu_attention_forward(q, k, v, cu, cu, **kwargs)
    hstu_attention_backward(do, q, k, v, cu, cu, **kwargs)
    torch.cuda.synchronize()

    def fail_d2h(*_args, **_kwargs):
        raise AssertionError("unexpected CUDA metadata value inspection")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", fail_d2h)
        patch.setattr(torch.Tensor, "item", fail_d2h)
        patch.setattr(torch.Tensor, "tolist", fail_d2h)
        out = hstu_attention_forward(q, k, v, cu, cu, **kwargs)["o_tensor"]
        grads = hstu_attention_backward(do, q, k, v, cu, cu, **kwargs)

    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    assert all(torch.isfinite(grad).all() for grad in grads)


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
@pytest.mark.parametrize("head_dim", [32, 64, 256])
def test_runtime_alpha_and_scaling_are_not_compile_time_constants(head_dim):
    q, k, v, do, cu = _inputs(heads=1, head_dim=head_dim)
    scalar_configs = ((0.35, 32.0), (1.1, 96.0))

    _interface.hstu_varlen_fwd_100.compile_cache.clear()
    for alpha, scaling_seqlen in scalar_configs:
        actual = hstu_attention_forward(
            q,
            k,
            v,
            cu,
            cu,
            max_seqlen_q=128,
            max_seqlen_k=128,
            window_size=(-1, 0),
            alpha=alpha,
            scaling_seqlen=scaling_seqlen,
        )["o_tensor"]
        expected = _reference_forward(
            q,
            k,
            v,
            cu,
            cu,
            alpha=alpha,
            scaling_seqlen=scaling_seqlen,
            causal=True,
        )
        torch.testing.assert_close(actual.float(), expected, rtol=4e-2, atol=4e-2)
        assert len(_interface.hstu_varlen_fwd_100.compile_cache) == 1

    if head_dim == 256:
        from cudnn.hstu_attention._kernels.hstu_bwd_256_cute import (
            hstu_varlen_bwd_256_cute,
        )

        bwd_compile_cache = hstu_varlen_bwd_256_cute.compile_cache
    else:
        bwd_compile_cache = _interface.hstu_varlen_bwd_100.compile_cache
    bwd_compile_cache.clear()
    for alpha, scaling_seqlen in scalar_configs:
        q_ref = q.float().detach().requires_grad_(True)
        k_ref = k.float().detach().requires_grad_(True)
        v_ref = v.float().detach().requires_grad_(True)
        expected_out = _reference_forward(
            q_ref,
            k_ref,
            v_ref,
            cu,
            cu,
            alpha=alpha,
            scaling_seqlen=scaling_seqlen,
            causal=True,
        )
        expected_grads = torch.autograd.grad(
            expected_out,
            (q_ref, k_ref, v_ref),
            do.float(),
        )
        actual_grads = hstu_attention_backward(
            do,
            q,
            k,
            v,
            cu,
            cu,
            max_seqlen_q=128,
            max_seqlen_k=128,
            window_size=(-1, 0),
            alpha=alpha,
            scaling_seqlen=scaling_seqlen,
        )
        for name, expected_grad in zip(("dq_tensor", "dk_tensor", "dv_tensor"), expected_grads):
            torch.testing.assert_close(actual_grads[name].float(), expected_grad, rtol=8e-2, atol=8e-2)
        assert len(bwd_compile_cache) == 1


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
def test_current_stream_and_compile_cache():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        q, k, v, _, cu = _inputs()
        first = hstu_attention_forward(
            q,
            k,
            v,
            cu,
            cu,
            max_seqlen_q=128,
            max_seqlen_k=128,
            window_size=(-1, 0),
            stream=stream,
        )["o_tensor"]
        second = hstu_attention_forward(
            q,
            k,
            v,
            cu,
            cu,
            max_seqlen_q=128,
            max_seqlen_k=128,
            window_size=(-1, 0),
            stream=stream,
        )["o_tensor"]
    stream.synchronize()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
