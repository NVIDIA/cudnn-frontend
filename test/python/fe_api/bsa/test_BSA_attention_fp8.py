# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import importlib
import math
import sys
from types import ModuleType

import pytest
import torch

from cudnn.api_base import TupleDict
from fe_api.bsa.bsa_reference import attention_reference, block_sparse_mask
from test_utils import torch_fork_set_rng

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


def _import_bsa():
    try:
        from cudnn import BSA

        importlib.import_module("cudnn.block_sparse_attention._interface")
        return BSA
    except (ImportError, OSError) as error:
        pytest.skip(f"block sparse attention optional dependencies are unavailable: {error}")


def _require_sm100_fp8():
    if not torch.cuda.is_available():
        pytest.skip("Sage FP8 block sparse attention requires CUDA")
    major, _ = torch.cuda.get_device_capability()
    if major not in {10, 11}:
        pytest.skip("this numerical Sage FP8 test requires SM100/SM110")
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    if interface._cutlass_dsl_version() < (4, 6, 1):
        pytest.skip("Sage FP8 requires nvidia-cutlass-dsl>=4.6.1")
    return _import_bsa()


def _make_block_index(heads: int, seqlen_q: int, seqlen_k: int, topk: int) -> torch.Tensor:
    num_q_blocks = math.ceil(seqlen_q / 64)
    num_kv_blocks = math.ceil(seqlen_k / 64)
    result = torch.empty((1, heads, num_q_blocks, topk), device="cuda", dtype=torch.int32)
    block_ids = torch.arange(num_kv_blocks, device="cuda")
    for head in range(heads):
        for q_block in range(num_q_blocks):
            selected = torch.roll(block_ids, shifts=head + q_block)[:topk].sort().values
            result[0, head, q_block] = selected.to(torch.int32)
    return result


@pytest.mark.L0
def test_bsa_fp8_public_exports_do_not_eagerly_import_quantizer():
    private_module = "cudnn.block_sparse_attention._fp8_quant"
    sys.modules.pop(private_module, None)

    import cudnn

    package = importlib.import_module("cudnn.block_sparse_attention")
    assert cudnn.block_sparse_attention_fp8_forward is package.block_sparse_attention_fp8_forward
    assert cudnn.BSA.block_sparse_attention_fp8_forward is package.block_sparse_attention_fp8_forward
    assert not hasattr(cudnn, "quantize_sage_bhsd")
    assert not hasattr(cudnn.BSA, "quantize_sage_bhsd")
    assert not hasattr(package, "quantize_sage_bhsd")
    assert private_module not in sys.modules


@pytest.mark.L0
@pytest.mark.parametrize(
    ("version", "expected"),
    (
        ("4.6.1", (4, 6, 1)),
        ("4.6.1+cu13", (4, 6, 1)),
        ("4.7.0.dev0", (4, 7, 0)),
    ),
)
def test_bsa_fp8_cutedsl_version_parser(monkeypatch, version, expected):
    _import_bsa()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    monkeypatch.setattr(interface.cutlass, "__version__", version)
    assert interface._cutlass_dsl_version() == expected


@pytest.mark.L0
@pytest.mark.parametrize("version", ("4.5.0", "4.6.0"))
def test_bsa_fp8_runtime_gate_precedes_quantizer_import(monkeypatch, version):
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    private_module = "cudnn.block_sparse_attention._fp8_quant"
    sys.modules.pop(private_module, None)
    monkeypatch.setattr(interface.cutlass, "__version__", version)

    with pytest.raises(RuntimeError, match=r"nvidia-cutlass-dsl>=4\.6\.1"):
        interface.bsa_fp8_blk64_fwd(object(), object(), object(), object(), 1)
    assert private_module not in sys.modules


@pytest.mark.L0
def test_bsa_fp8_interface_quantizes_before_private_launch(monkeypatch):
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    monkeypatch.setattr(interface, "_require_sage_fp8_cutedsl", lambda: None)

    q, k, v = object(), object(), object()
    q2k, block_sizes, q2k_nums = object(), object(), object()
    quantized = tuple(object() for _ in range(6))
    expected = object()

    fake_quantizer = ModuleType("cudnn.block_sparse_attention._fp8_quant")

    def fake_quantize(actual_q, actual_k, actual_v):
        assert (actual_q, actual_k, actual_v) == (q, k, v)
        return quantized

    fake_quantizer._quantize_sage_bhsd = fake_quantize
    monkeypatch.setitem(sys.modules, fake_quantizer.__name__, fake_quantizer)

    def fake_launch(*args, **kwargs):
        assert args[:6] == quantized
        assert args[6:] == (q2k, 7, 0.125)
        assert kwargs == {"block_sizes": block_sizes, "q2k_block_nums": q2k_nums}
        return expected

    monkeypatch.setattr(interface, "_bsa_fp8_blk64_fwd_quantized", fake_launch)
    result = interface.bsa_fp8_blk64_fwd(
        q,
        k,
        v,
        q2k,
        7,
        0.125,
        block_sizes=block_sizes,
        q2k_block_nums=q2k_nums,
    )
    assert result is expected


@pytest.mark.L0
def test_bsa_fp8_forward_accepts_bf16_and_returns_documented_tupledict(monkeypatch):
    BSA = _import_bsa()
    api = importlib.import_module("cudnn.block_sparse_attention.api")
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    monkeypatch.setattr(api, "_device_arch", lambda tensor: 100)

    shape = (1, 4, 64, 128)
    q = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    q2k = torch.zeros((1, 4, 1, 1), device="cuda", dtype=torch.int32)
    expected = torch.empty(shape, device="cuda", dtype=torch.bfloat16)

    def fake_forward(*args, **kwargs):
        expected_args = (q, k, v, q2k)
        assert all(actual is expected_arg for actual, expected_arg in zip(args[:4], expected_args))
        assert args[4] == 1
        assert args[5] is None
        assert kwargs == {"block_sizes": None, "q2k_block_nums": None}
        return expected

    monkeypatch.setattr(interface, "bsa_fp8_blk64_fwd", fake_forward)
    result = BSA.block_sparse_attention_fp8_forward(
        q,
        k,
        v,
        q2k,
        block_sparse_num=1,
    )

    assert isinstance(result, TupleDict)
    assert list(result.keys()) == ["o_tensor"]
    assert result["o_tensor"] is expected
    assert result[0] is expected


@pytest.mark.L0
@torch_fork_set_rng(seed=20260709)
def test_bsa_fp8_private_cutedsl_quantizer_matches_recipe():
    _require_sm100_fp8()
    quantizer = importlib.import_module("cudnn.block_sparse_attention._fp8_quant")

    batch, heads, seqlen_q, seqlen_k, head_dim = 1, 4, 128, 192, 128
    q = torch.randn((batch, heads, seqlen_q, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, seqlen_k, head_dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    actual = quantizer._quantize_sage_bhsd(q, k, v)
    torch.cuda.synchronize()

    q_scale = q.float().abs().amax(dim=-1).clamp_min(1.0e-3) / 448.0
    q_reciprocal = q_scale.to(torch.bfloat16).reciprocal().to(torch.bfloat16)
    q_fp8 = (q * q_reciprocal.unsqueeze(-1)).to(torch.bfloat16).to(torch.float8_e4m3fn)

    k_mean = k.float().mean(dim=2, keepdim=True)
    k_centered = k.float() - k_mean
    k_blocks = k_centered.reshape(batch, heads, seqlen_k // 16, 16, head_dim)
    k_scale = k_blocks.abs().amax(dim=(-1, -2)).clamp_min(1.0e-3) / 448.0
    k_fp8 = (k_blocks / k_scale[..., None, None]).to(torch.float8_e4m3fn).reshape_as(k)

    v_scale = v.abs().float().amax(dim=(0, 2)).clamp_min(1.0e-3) / 448.0
    v_reciprocal = v_scale.reciprocal().to(torch.bfloat16)
    v_fp8 = (v * v_reciprocal[None, :, None, :]).to(torch.float8_e4m3fn)

    expected = (q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale)
    assert all(tensor.is_contiguous() for tensor in actual)
    assert torch.equal(actual[0].float(), expected[0].float())
    assert torch.allclose(actual[1].float(), expected[1].float(), atol=4.0, rtol=0.0)
    assert torch.equal(actual[2].float(), expected[2].float())
    assert torch.equal(actual[3], expected[3])
    assert torch.allclose(actual[4], expected[4], atol=1.0e-6, rtol=0.0)
    assert torch.equal(actual[5], expected[5])


@pytest.mark.L0
@pytest.mark.parametrize(
    ("heads", "seqlen_q", "topk", "expected_splits"),
    (
        (4, 64, 127, 1),
        (4, 64, 128, 16),
        (4, 64, 224, 8),
        (4, 64, 400, 16),
        (8, 64, 128, 4),
        (8, 64, 256, 8),
        (8, 64, 500, 16),
        (4, 8192, 899, 1),
        (4, 8192, 900, 4),
    ),
)
def test_bsa_fp8_sm100_sm110_auto_kv_splits(heads, seqlen_q, topk, expected_splits):
    _import_bsa()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    assert interface._sm100_blk64_auto_fp8_kv_splits(topk, heads, seqlen_q) == expected_splits


@pytest.mark.L0
def test_bsa_fp8_sm100_auto_split_uses_workspace_fallback(monkeypatch):
    _require_sm100_fp8()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    monkeypatch.setattr(interface, "_get_device_arch", lambda: 100)

    batch, heads, seqlen_q, seqlen_k, head_dim, topk = 1, 4, 64, 64, 128, 128
    q = torch.empty((batch, heads, seqlen_q, head_dim), device="cuda", dtype=torch.float8_e4m3fn)
    k = torch.empty((batch, heads, seqlen_k, head_dim), device="cuda", dtype=torch.float8_e4m3fn)
    v = torch.empty_like(k)
    q_scale = torch.empty((batch, heads, seqlen_q), device="cuda", dtype=torch.float32)
    k_scale = torch.empty((batch, heads, seqlen_k // 16), device="cuda", dtype=torch.float32)
    v_scale = torch.empty((heads, head_dim), device="cuda", dtype=torch.float32)
    q2k = torch.zeros((batch, heads, 1, topk), device="cuda", dtype=torch.int32)

    class WorkspaceFallbackCalled(Exception):
        pass

    def workspace_fallback(q_arg, value_dim, kv_splits, allow_fallback, output_dtype=None):
        assert q_arg is q
        assert value_dim == head_dim
        assert kv_splits == 16
        assert allow_fallback is True
        assert output_dtype is torch.bfloat16
        raise WorkspaceFallbackCalled

    monkeypatch.setattr(interface, "_resolve_blk64_split_workspace", workspace_fallback)
    with pytest.raises(WorkspaceFallbackCalled):
        interface._bsa_fp8_blk64_fwd_quantized(
            q,
            k,
            v,
            q_scale,
            k_scale,
            v_scale,
            q2k,
            topk,
        )


@pytest.mark.L0
def test_bsa_fp8_quantized_inputs_require_fully_contiguous_bhsd():
    _require_sm100_fp8()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")

    batch, heads, seqlen, head_dim = 1, 4, 64, 128

    def make_padded_fp8():
        storage = torch.empty((batch, heads, seqlen, head_dim + 1), device="cuda", dtype=torch.float8_e4m3fn)
        return storage[..., :head_dim]

    q = make_padded_fp8()
    k = make_padded_fp8()
    v = make_padded_fp8()
    assert all(tensor.stride(-1) == 1 and not tensor.is_contiguous() for tensor in (q, k, v))

    q_scale = torch.empty((batch, heads, seqlen), device="cuda", dtype=torch.float32)
    k_scale = torch.empty((batch, heads, seqlen // 16), device="cuda", dtype=torch.float32)
    v_scale = torch.empty((heads, head_dim), device="cuda", dtype=torch.float32)
    q2k = torch.zeros((batch, heads, 1, 1), device="cuda", dtype=torch.int32)

    with pytest.raises(AssertionError, match="fully contiguous BHSD"):
        interface._bsa_fp8_blk64_fwd_quantized(
            q,
            k,
            v,
            q_scale,
            k_scale,
            v_scale,
            q2k,
            1,
        )


@pytest.mark.L0
def test_bsa_fp8_split_workspace_estimate_uses_bf16_output_size():
    _import_bsa()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")

    batch, heads, seqlen_q, value_dim, kv_splits = 1, 4, 64, 128, 2
    q = torch.empty((batch, heads, seqlen_q, value_dim), dtype=torch.float8_e4m3fn)
    rows = batch * heads * seqlen_q
    num_q_blocks = math.ceil(seqlen_q / 64)
    expected_bytes = (
        kv_splits * rows * (value_dim + 1) * 4 + rows * (value_dim * torch.bfloat16.itemsize + 4) + batch * heads * num_q_blocks * (kv_splits + 1) * 4
    )

    assert interface._blk64_split_workspace_bytes(q, value_dim, kv_splits, output_dtype=torch.bfloat16) == expected_bytes


@pytest.mark.L0
@torch_fork_set_rng(seed=2026)
def test_bsa_fp8_sm100_bf16_forward_matches_reference():
    BSA = _require_sm100_fp8()
    batch, heads, seqlen_q, seqlen_k, head_dim, topk = 1, 4, 128, 640, 128, 5
    q = torch.randn((batch, heads, seqlen_q, head_dim), device="cuda", dtype=torch.bfloat16) * 0.5
    k = torch.randn((batch, heads, seqlen_k, head_dim), device="cuda", dtype=torch.bfloat16) * 0.5
    v = torch.randn_like(k) * 0.5
    q2k = _make_block_index(heads, seqlen_q, seqlen_k, topk)

    block_sizes = torch.full((seqlen_k // 64,), 64, device="cuda", dtype=torch.int32)
    mask = block_sparse_mask(q2k, topk, block_sizes, seqlen_q, seqlen_k, 64)
    softmax_scale = head_dim**-0.5
    reference, _ = attention_reference(q, k, v, mask, softmax_scale)

    result = BSA.block_sparse_attention_fp8_forward(
        q,
        k,
        v,
        q2k,
        block_sparse_num=topk,
        softmax_scale=softmax_scale,
    )
    torch.cuda.synchronize()

    assert list(result.keys()) == ["o_tensor"]
    actual = result["o_tensor"]
    assert actual.dtype == torch.bfloat16
    assert actual.shape == q.shape
    difference = (actual.float() - reference).abs()
    assert difference.max().item() < 0.2
    assert (difference.mean() / reference.abs().mean().clamp_min(1e-8)).item() < 0.04
