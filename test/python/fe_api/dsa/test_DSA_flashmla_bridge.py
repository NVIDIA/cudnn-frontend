# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract and B200 integration tests for the optional FlashMLA bridge."""

from __future__ import annotations

import math
import types
from importlib import metadata as importlib_metadata
from pathlib import Path

import pytest
import torch

from cudnn import DSA
from cudnn.deepseek_sparse_attention import flashmla_bridge as bridge
from fe_api.dsa.dsa_reference import (
    check_ref_dsa_sparse_attention_backward,
    check_ref_sparse_score_recompute,
    ref_sparse_attention_forward,
)


def _fake_flashmla_distribution(root: Path, version: str = "1.0.0+15f13e5"):
    files = [
        Path("flash_mla/__init__.py"),
        Path("flash_mla/flash_mla_interface.py"),
        Path("flash_mla/cuda.cpython-312-x86_64-linux-gnu.so"),
    ]
    return types.SimpleNamespace(version=version, files=files, locate_file=lambda relative_path: root / relative_path)


def _bind_fake_flashmla_extension(monkeypatch, sparse_fwd, path: Path):
    extension = types.SimpleNamespace(__file__=str(path), sparse_prefill_fwd=lambda *_args, **_kwargs: None)
    monkeypatch.setitem(sparse_fwd.__globals__, "flash_mla_cuda", extension)


@pytest.mark.L0
def test_sparse_attention_public_namespace_is_provider_neutral():
    assert callable(DSA.sparse_attention_forward)
    assert callable(DSA.sparse_attention)
    assert callable(DSA.sparse_attention_score_recompute)
    assert issubclass(DSA.SparseAttentionBackendUnavailableError, RuntimeError)

    for provider_branded_name in (
        "flashmla_sparse_forward",
        "flashmla_sparse_attention",
        "flashmla_sparse_score_recompute",
    ):
        with pytest.raises(AttributeError):
            getattr(DSA, provider_branded_name)


@pytest.mark.L0
@pytest.mark.parametrize(
    "heads,head_dim,topk,error",
    [
        (8, 512, 64, "num_heads"),
        (64, 256, 64, "head_dim"),
        (64, 512, 0, "topk"),
    ],
)
def test_sparse_attention_forward_rejects_unsupported_semantic_contract(monkeypatch, heads, head_dim, topk, error):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: (10, 0))
    monkeypatch.setattr(
        bridge,
        "_resolve_flashmla_sparse_fwd",
        lambda: pytest.fail("unsupported semantic inputs must be rejected before resolving a provider"),
    )
    device = torch.device("cuda")
    q = torch.empty((1, heads, head_dim), dtype=torch.bfloat16, device=device)
    kv = torch.empty((1, head_dim), dtype=torch.bfloat16, device=device)
    indices = torch.empty((1, topk), dtype=torch.int32, device=device)

    with pytest.raises((TypeError, ValueError), match=error):
        DSA.sparse_attention_forward(q, kv, indices)


@pytest.mark.L0
def test_sparse_attention_forward_rejects_device_softmax_scale_without_sync(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: (10, 0))
    monkeypatch.setattr(
        bridge,
        "_resolve_flashmla_sparse_fwd",
        lambda: pytest.fail("an invalid host scalar must be rejected before resolving a provider"),
    )
    device = torch.device("cuda")
    q = torch.empty((1, 64, 512), dtype=torch.bfloat16, device=device)
    kv = torch.empty((1, 512), dtype=torch.bfloat16, device=device)
    indices = torch.empty((1, 64), dtype=torch.int32, device=device)
    device_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    torch.cuda.set_sync_debug_mode("error")
    try:
        with pytest.raises(TypeError, match="host real scalar"):
            DSA.sparse_attention_forward(q, kv, indices, softmax_scale=device_scale)
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.L0
@pytest.mark.parametrize("heads,head_dim,topk", [(32, 576, 65), (64, 512, 64), (128, 512, 129)])
def test_sparse_attention_forward_preserves_semantic_output_contract(monkeypatch, heads, head_dim, topk):
    observed = []

    def provider(q, kv, indices, *, sm_scale, d_v, attn_sink, topk_length):
        observed.append((tuple(q.shape), tuple(kv.shape), tuple(indices.shape), sm_scale, d_v, attn_sink, topk_length))
        aux_shape = q.shape[:2]
        return (
            q.new_zeros((*aux_shape, d_v)),
            torch.zeros(aux_shape, dtype=torch.float32, device=q.device),
            torch.zeros(aux_shape, dtype=torch.float32, device=q.device),
        )

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: (10, 0))
    monkeypatch.setattr(bridge, "_validated_flashmla_sparse_fwd", None)
    monkeypatch.setattr(bridge, "import_module", lambda _name: types.SimpleNamespace(flash_mla_sparse_fwd=provider))
    monkeypatch.setattr(bridge, "_probe_flashmla_provider_identity", lambda *_args: None)

    device = torch.device("cuda")
    q = torch.zeros((2, heads, head_dim), dtype=torch.bfloat16, device=device)
    kv = torch.zeros((7, head_dim), dtype=torch.bfloat16, device=device)
    indices = torch.zeros((2, topk), dtype=torch.int32, device=device)
    sink = torch.zeros((heads,), dtype=torch.float32, device=device)
    lengths = torch.tensor([topk, max(topk - 1, 0)], dtype=torch.int32, device=device)

    result = DSA.sparse_attention_forward(q, kv, indices, attn_sink=sink, topk_length=lengths)

    assert len(observed) == 1
    assert result.keys() == {"output": None, "max_logits": None, "lse": None}.keys()
    assert result["output"].shape == (2, heads, 512)
    assert result["max_logits"].shape == result["lse"].shape == (2, heads)
    assert result["output"].dtype == torch.bfloat16
    assert result["max_logits"].dtype == result["lse"].dtype == torch.float32


@pytest.mark.L0
def test_sparse_attention_forward_bounds_lengths_before_provider(monkeypatch):
    observed_lengths = []

    def provider(q, kv, indices, *, sm_scale, d_v, attn_sink, topk_length):
        observed_lengths.append(topk_length)
        aux_shape = q.shape[:2]
        return (
            q.new_zeros((*aux_shape, d_v)),
            torch.zeros(aux_shape, dtype=torch.float32, device=q.device),
            torch.zeros(aux_shape, dtype=torch.float32, device=q.device),
        )

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: (10, 0))
    monkeypatch.setattr(bridge, "_validated_flashmla_sparse_fwd", None)
    monkeypatch.setattr(bridge, "import_module", lambda _name: types.SimpleNamespace(flash_mla_sparse_fwd=provider))
    monkeypatch.setattr(bridge, "_probe_flashmla_provider_identity", lambda *_args: None)

    device = torch.device("cuda")
    topk = 64
    q = torch.zeros((2, 64, 512), dtype=torch.bfloat16, device=device)
    kv = torch.zeros((7, 512), dtype=torch.bfloat16, device=device)
    indices = torch.zeros((2, topk), dtype=torch.int32, device=device)
    lengths = torch.tensor([-5, topk + 7], dtype=torch.int32, device=device)

    DSA.sparse_attention_forward(q, kv, indices, topk_length=lengths)

    assert len(observed_lengths) == 1
    torch.testing.assert_close(observed_lengths[0], torch.tensor([0, topk], dtype=torch.int32, device=device))


@pytest.mark.L0
def test_flashmla_dependency_is_lazy_and_missing_dependency_fails_closed(monkeypatch):
    calls = []

    def unavailable(name):
        calls.append(name)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(bridge, "import_module", unavailable)
    assert callable(DSA.sparse_attention_forward)
    assert calls == []
    with pytest.raises(bridge.SparseAttentionBackendUnavailableError, match="official deepseek-ai/FlashMLA"):
        bridge._resolve_flashmla_sparse_fwd()
    assert calls == ["flash_mla"]


@pytest.mark.L0
def test_flashmla_dependency_without_sparse_entrypoint_fails_closed(monkeypatch):
    monkeypatch.setattr(bridge, "import_module", lambda _name: types.SimpleNamespace())
    with pytest.raises(bridge.SparseAttentionBackendUnavailableError, match="flash_mla_sparse_fwd"):
        bridge._resolve_flashmla_sparse_fwd()


@pytest.mark.L0
def test_flashmla_dependency_accepts_pinned_distribution_identity(monkeypatch, tmp_path):
    def compatible(q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None):
        pytest.fail("compatibility probing must not execute the external callable")

    package_path = tmp_path / "flash_mla/__init__.py"
    callable_path = tmp_path / "flash_mla/flash_mla_interface.py"
    distributions = []
    dependency = types.SimpleNamespace(__version__="1.0.0", __file__=str(package_path), flash_mla_sparse_fwd=compatible)
    _bind_fake_flashmla_extension(monkeypatch, compatible, tmp_path / "flash_mla/cuda.cpython-312-x86_64-linux-gnu.so")
    monkeypatch.setattr(bridge, "import_module", lambda _name: dependency)
    monkeypatch.setattr(
        importlib_metadata,
        "distribution",
        lambda distribution: distributions.append(distribution) or _fake_flashmla_distribution(tmp_path),
    )
    monkeypatch.setattr(bridge, "getsourcefile", lambda _callable: str(callable_path))

    assert bridge._resolve_flashmla_sparse_fwd() is compatible
    assert distributions == ["flash_mla"]


@pytest.mark.L0
@pytest.mark.parametrize("shadowed_path", ["package", "callable", "extension"])
def test_flashmla_dependency_rejects_code_not_owned_by_pinned_distribution(monkeypatch, tmp_path, shadowed_path):
    def compatible(q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None):
        pytest.fail("a shadowed provider must not execute")

    package_path = tmp_path / "flash_mla/__init__.py"
    callable_path = tmp_path / "flash_mla/flash_mla_interface.py"
    extension_path = tmp_path / "flash_mla/cuda.cpython-312-x86_64-linux-gnu.so"
    if shadowed_path == "package":
        package_path = tmp_path / "shadow/flash_mla/__init__.py"
    elif shadowed_path == "callable":
        callable_path = tmp_path / "shadow/flash_mla/flash_mla_interface.py"
    else:
        extension_path = tmp_path / "shadow/flash_mla/cuda.cpython-312-x86_64-linux-gnu.so"
    dependency = types.SimpleNamespace(__file__=str(package_path), flash_mla_sparse_fwd=compatible)
    _bind_fake_flashmla_extension(monkeypatch, compatible, extension_path)
    monkeypatch.setattr(bridge, "import_module", lambda _name: dependency)
    monkeypatch.setattr(importlib_metadata, "distribution", lambda _name: _fake_flashmla_distribution(tmp_path))
    monkeypatch.setattr(bridge, "getsourcefile", lambda _callable: str(callable_path))

    with pytest.raises(bridge.SparseAttentionBackendUnavailableError, match=r"not owned by the pinned"):
        bridge._resolve_flashmla_sparse_fwd()


@pytest.mark.L0
@pytest.mark.parametrize("installed_version", ["1.0.0", "1.0.0+deadbee"])
def test_flashmla_dependency_rejects_unverified_distribution_identity(monkeypatch, installed_version):
    def compatible(q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None):
        pytest.fail("an unverified provider must not execute")

    dependency = types.SimpleNamespace(flash_mla_sparse_fwd=compatible)
    monkeypatch.setattr(bridge, "import_module", lambda _name: dependency)
    monkeypatch.setattr(importlib_metadata, "distribution", lambda _distribution: _fake_flashmla_distribution(Path("/unused"), installed_version))

    with pytest.raises(
        bridge.SparseAttentionBackendUnavailableError,
        match=r"distribution version .* is incompatible.*1\.0\.0\+15f13e5",
    ):
        bridge._resolve_flashmla_sparse_fwd()


@pytest.mark.L0
def test_flashmla_dependency_without_distribution_metadata_fails_closed(monkeypatch):
    def compatible(q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None):
        pytest.fail("an unidentified provider must not execute")

    dependency = types.SimpleNamespace(flash_mla_sparse_fwd=compatible)
    monkeypatch.setattr(bridge, "import_module", lambda _name: dependency)

    def missing_distribution(_distribution):
        raise importlib_metadata.PackageNotFoundError("flash_mla")

    monkeypatch.setattr(importlib_metadata, "distribution", missing_distribution)

    with pytest.raises(bridge.SparseAttentionBackendUnavailableError, match=r"no installed distribution metadata"):
        bridge._resolve_flashmla_sparse_fwd()


@pytest.mark.L0
@pytest.mark.parametrize(
    "incompatible",
    [
        pytest.param(
            lambda q, kv, indices, sm_scale, d_v=512, attn_sink=None: None,
            id="missing-topk-length",
        ),
        pytest.param(
            lambda q, kv, indices, workspace, sm_scale, d_v=512, attn_sink=None, topk_length=None: None,
            id="new-required-parameter",
        ),
    ],
)
def test_flashmla_dependency_with_incompatible_call_signature_fails_closed(monkeypatch, incompatible):
    dependency = types.SimpleNamespace(flash_mla_sparse_fwd=incompatible)
    monkeypatch.setattr(bridge, "import_module", lambda _name: dependency)
    monkeypatch.setattr(bridge, "_probe_flashmla_provider_identity", lambda *_args: None)

    with pytest.raises(bridge.SparseAttentionBackendUnavailableError, match=r"incompatible signature.*topk_length"):
        bridge._resolve_flashmla_sparse_fwd()


@pytest.mark.L0
def test_flashmla_dependency_with_opaque_call_signature_fails_closed(monkeypatch):
    class OpaqueCallable:
        __signature__ = "not-an-inspect-signature"

        def __call__(self, *args, **kwargs):
            return args, kwargs

    dependency = types.SimpleNamespace(flash_mla_sparse_fwd=OpaqueCallable())
    monkeypatch.setattr(bridge, "import_module", lambda _name: dependency)
    monkeypatch.setattr(bridge, "_probe_flashmla_provider_identity", lambda *_args: None)

    with pytest.raises(bridge.SparseAttentionBackendUnavailableError, match=r"no inspectable Python signature"):
        bridge._resolve_flashmla_sparse_fwd()


@pytest.mark.L0
def test_flashmla_training_bridge_requires_sink_before_dependency_resolution(
    monkeypatch,
):
    monkeypatch.setattr(
        bridge,
        "_resolve_flashmla_sparse_fwd",
        lambda: pytest.fail("dependency must not be loaded"),
    )
    with pytest.raises(ValueError, match="attn_sink is required"):
        bridge.sparse_attention(None, None, None, None)


@pytest.mark.L0
def test_flashmla_training_metadata_normalizes_all_invalid_sentinels_without_lengths():
    indices = torch.tensor([[0, -7, 4, 5, 99]], dtype=torch.int32)
    safe_indices, safe_length = bridge._normalize_cudnn_sparse_metadata(indices, None, s_kv=5)
    assert torch.equal(safe_indices, torch.tensor([[0, -1, 4, -1, -1]], dtype=torch.int32))
    assert safe_length is None


@pytest.mark.L0
def test_flashmla_training_metadata_masks_compacts_and_bounds_lengths():
    indices = torch.tensor(
        [
            [0, 99, -7, 4, 3],
            [4, 3, 2, 1, 0],
            [4, 3, 2, 1, 0],
        ],
        dtype=torch.int32,
    )
    lengths = torch.tensor([9, 2, -3], dtype=torch.int32)

    def compactify(normalized):
        packed = torch.full_like(normalized, -1)
        packed_lengths = torch.empty(normalized.shape[0], dtype=torch.int32)
        for row, values in enumerate(normalized):
            valid = values[values >= 0]
            packed[row, : valid.numel()] = valid
            packed_lengths[row] = valid.numel()
        return {"indices": packed, "topk_length": packed_lengths}

    safe_indices, safe_length = bridge._normalize_cudnn_sparse_metadata(
        indices,
        lengths,
        s_kv=5,
        _compactify=compactify,
    )
    assert torch.equal(
        safe_indices,
        torch.tensor(
            [[0, 4, 3, -1, -1], [4, 3, -1, -1, -1], [-1, -1, -1, -1, -1]],
            dtype=torch.int32,
        ),
    )
    assert torch.equal(safe_length, torch.tensor([3, 2, 0], dtype=torch.int32))


@pytest.mark.L0
def test_flashmla_training_trusted_compact_metadata_is_identity_fast_path():
    indices = torch.tensor([[4, 3, 2, 1, 0]], dtype=torch.int32)
    lengths = torch.tensor([5], dtype=torch.int32)

    def must_not_compact(_normalized):
        pytest.fail("trusted compact metadata must not launch compactification")

    safe_indices, safe_length = bridge._normalize_cudnn_sparse_metadata(
        indices,
        lengths,
        s_kv=5,
        trusted_compact_metadata=True,
        _compactify=must_not_compact,
    )
    assert safe_indices is indices
    assert safe_length is lengths


@pytest.mark.L0
def test_flashmla_training_trusted_compact_metadata_requires_bool():
    with pytest.raises(TypeError, match="trusted_compact_metadata must be a bool"):
        bridge.sparse_attention(
            None,
            None,
            None,
            object(),
            trusted_compact_metadata=1,
        )


def _require_exact_b200():
    if not torch.cuda.is_available():
        pytest.skip("exact NVIDIA B200 required")
    if torch.cuda.get_device_capability() != (10, 0) or torch.cuda.get_device_name() != "NVIDIA B200":
        pytest.skip("exact NVIDIA B200 required")


def _require_b200_flashmla():
    _require_exact_b200()
    try:
        return bridge._resolve_flashmla_sparse_fwd()
    except bridge.SparseAttentionBackendUnavailableError as exc:
        pytest.skip(str(exc))


@pytest.mark.L0
@pytest.mark.parametrize("topk", [1, 4, 128, 129, 1152])
def test_sparse_attention_score_recompute_deepseek_v4_topk_contract(monkeypatch, topk):
    """Preserve semantic outputs across the DeepSeek-V4 Top-K envelope."""

    _require_exact_b200()
    monkeypatch.setattr(bridge, "import_module", lambda _name: pytest.fail("score recompute must not resolve or launch a forward provider"))

    device = torch.device("cuda")
    s_q, s_kv, heads, head_dim = 1, 1152, 128, 512
    q = torch.zeros((s_q, heads, head_dim), dtype=torch.bfloat16, device=device)
    kv = torch.zeros((s_kv, head_dim), dtype=torch.bfloat16, device=device)
    lse = torch.zeros((s_q, heads), dtype=torch.float32, device=device)
    indices = torch.arange(topk, dtype=torch.int32, device=device).unsqueeze(0)

    result = DSA.sparse_attention_score_recompute(q, kv, lse, indices)
    torch.cuda.synchronize()

    assert torch.equal(result["indices"], indices)
    assert result["target"].shape == indices.shape
    torch.testing.assert_close(
        result["target"],
        torch.full_like(result["target"], 1.0 / topk),
        atol=1e-6,
        rtol=1e-4,
    )


@pytest.mark.L1
@pytest.mark.parametrize(
    "heads,head_dim,topk,s_kv",
    [
        pytest.param(64, 512, 64, 96, id="native-h64-d512"),
        pytest.param(32, 576, 65, 96, id="padded-h32-d576-k65"),
        pytest.param(128, 512, 1152, 1280, id="h128-d512-small-topk"),
        pytest.param(128, 512, 1281, 1408, id="h128-d512-regular-topk"),
    ],
)
def test_flashmla_bridge_forward_matches_reference(heads, head_dim, topk, s_kv):
    _require_b200_flashmla()
    torch.manual_seed(410)
    device = torch.device("cuda")
    s_q = 4
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(s_q, heads, head_dim, dtype=torch.bfloat16, device=device)
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device)
    sink = torch.randn(heads, dtype=torch.float32, device=device)
    indices = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    indices[0, 0] = s_kv + 17
    indices[1, 1] = -9
    lengths = torch.tensor([topk, topk - 1, topk // 2, 1], dtype=torch.int32, device=device)

    actual = DSA.sparse_attention_forward(
        q,
        kv,
        indices,
        softmax_scale=scale,
        attn_sink=sink,
        topk_length=lengths,
    )
    out_ref, lse_ref = ref_sparse_attention_forward(q, kv, sink, indices, topk_length=lengths, softmax_scale=scale)

    assert actual["output"].shape == (s_q, heads, 512)
    assert actual["max_logits"].shape == actual["lse"].shape == (s_q, heads)
    torch.testing.assert_close(actual["output"], out_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(actual["lse"], lse_ref, atol=2e-4, rtol=2e-4)


@pytest.mark.L1
def test_flashmla_bridge_masks_inactive_valid_index_before_provider():
    """An ignored valid index must not pull a NaN KV row into the output."""

    _require_b200_flashmla()
    torch.manual_seed(413)
    device = torch.device("cuda")
    s_q, s_kv, heads, head_dim, topk = 1, 65, 64, 512, 64
    q = torch.randn(s_q, heads, head_dim, dtype=torch.bfloat16, device=device)
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device)
    kv_with_nan_suffix = kv.clone()
    kv_with_nan_suffix[-1].fill_(float("nan"))
    sink = torch.randn(heads, dtype=torch.float32, device=device)
    indices = torch.zeros((s_q, topk), dtype=torch.int32, device=device)
    indices[0, 1] = s_kv - 1
    lengths = torch.ones((s_q,), dtype=torch.int32, device=device)

    actual = DSA.sparse_attention_forward(q, kv_with_nan_suffix, indices, attn_sink=sink, topk_length=lengths)
    expected = DSA.sparse_attention_forward(q, kv, indices, attn_sink=sink, topk_length=lengths)

    assert bool(torch.isfinite(actual["output"]).all())
    torch.testing.assert_close(actual["output"], expected["output"], atol=0.0, rtol=0.0)
    torch.testing.assert_close(actual["lse"], expected["lse"], atol=0.0, rtol=0.0)


@pytest.mark.L1
def test_flashmla_cudnn_training_and_score_recompute_match_references():
    _require_b200_flashmla()
    torch.manual_seed(411)
    device = torch.device("cuda")
    s_q, s_kv, heads, head_dim, topk = 4, 96, 32, 576, 65
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(s_q, heads, head_dim, dtype=torch.bfloat16, device=device, requires_grad=True)
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device, requires_grad=True)
    sink = torch.randn(heads, dtype=torch.float32, device=device, requires_grad=True)
    indices = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    # Both forms are valid FlashMLA sentinels.  The bridge must prevent the
    # wider official contract from reaching cuDNN backward as an OOB index.
    indices[0, 0] = s_kv + 17
    indices[1, 1] = -9
    lengths = torch.tensor([topk + 11, topk - 1, topk // 2, 1], dtype=torch.int32, device=device)

    result = DSA.sparse_attention(
        q,
        kv,
        indices,
        sink,
        softmax_scale=scale,
        topk_length=lengths,
    )
    dout = torch.randn_like(result["output"])
    result["output"].backward(dout)

    check_ref_dsa_sparse_attention_backward(
        q,
        kv,
        sink,
        indices,
        result["output"],
        dout,
        result["lse"],
        q.grad,
        kv.grad,
        sink.grad,
        softmax_scale=scale,
        topk_length=lengths,
        atol=5e-2,
        rtol=5e-2,
    )

    score = DSA.sparse_attention_score_recompute(
        q.detach(),
        kv.detach(),
        result["lse"],
        indices,
        softmax_scale=scale,
        topk_length=lengths,
    )
    target, safe_indices = score["target"], score["indices"]
    bounded_lengths = lengths.clamp(min=0, max=topk)
    positions = torch.arange(topk, device=device).unsqueeze(0)
    invalid = (indices < 0) | (indices >= s_kv) | (positions >= bounded_lengths.unsqueeze(1))
    assert target.shape == indices.shape
    assert target.is_contiguous()
    assert bool((target[invalid] == 0).all())
    check_ref_sparse_score_recompute(
        "attention",
        q.detach().unsqueeze(0),
        result["lse"].unsqueeze(0),
        safe_indices.unsqueeze(0),
        target.unsqueeze(0),
        aux=kv.detach().unsqueeze(0),
        softmax_scale=scale,
        topk_length=None,
        atol=1e-3,
        rtol=1e-3,
    )


@pytest.mark.L2
def test_flashmla_cudnn_deepseek_v32_h128_d576_k2048_contract():
    """Exercise the production DeepSeek V3.2 H/D/Top-K contract on B200."""

    _require_b200_flashmla()
    torch.manual_seed(412)
    device = torch.device("cuda")
    s_q, s_kv, heads, head_dim, topk = 1, 2304, 128, 576, 2048
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(s_q, heads, head_dim, dtype=torch.bfloat16, device=device, requires_grad=True)
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device, requires_grad=True)
    sink = torch.randn(heads, dtype=torch.float32, device=device, requires_grad=True)
    indices = torch.randperm(s_kv, device=device)[:topk].reshape(s_q, topk).to(torch.int32)
    lengths = torch.full((s_q,), topk, dtype=torch.int32, device=device)

    result = DSA.sparse_attention(
        q,
        kv,
        indices,
        sink,
        softmax_scale=scale,
        topk_length=lengths,
        trusted_compact_metadata=True,
    )
    dout = torch.randn_like(result["output"])
    result["output"].backward(dout)

    assert result["output"].shape == (s_q, heads, 512)
    assert result["lse"].shape == (s_q, heads)
    check_ref_dsa_sparse_attention_backward(
        q,
        kv,
        sink,
        indices,
        result["output"],
        dout,
        result["lse"],
        q.grad,
        kv.grad,
        sink.grad,
        softmax_scale=scale,
        topk_length=lengths,
        atol=5e-2,
        rtol=5e-2,
    )
