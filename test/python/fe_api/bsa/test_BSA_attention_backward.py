# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
import torch

from test_utils import torch_fork_set_rng
from fe_api.bsa.bsa_reference import attention_backward_reference, block_sparse_mask
from fe_api.bsa.bsa_utils import make_fixed_metadata, supported_block_size

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


def _import_bsa():
    try:
        from cudnn import BSA

        importlib.import_module("cudnn.block_sparse_attention._interface")

        return BSA
    except (ImportError, OSError) as error:
        pytest.skip(f"block sparse attention optional dependencies are unavailable: {error}")


@pytest.mark.L0
@torch_fork_set_rng(seed=2)
def test_bsa_attention_backward_fixed_blocks():
    BSA = _import_bsa()
    block_size = supported_block_size(backward=True)
    batch, heads, seqlen_q, seqlen_k, dim = 1, 2, 2 * block_size, 4 * block_size, 128
    q = torch.randn((batch, heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    do = torch.randn_like(q)
    q2k, block_sparse_num, block_sizes = make_fixed_metadata(batch, heads, seqlen_q, seqlen_k, block_size)
    mask = block_sparse_mask(q2k, block_sparse_num, block_sizes, seqlen_q, seqlen_k, block_size)
    _, _, dq_ref, dk_ref, dv_ref = attention_backward_reference(q, k, v, do, mask)

    forward = BSA.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        block_sparse_num,
        block_sizes,
        sparse_block_size=block_size,
    )
    backward = BSA.block_sparse_attention_backward(
        do,
        q,
        k,
        v,
        forward["o_tensor"],
        forward["lse_tensor"],
        q2k,
        block_sparse_num,
        None,
        sparse_block_size=block_size,
    )
    torch.testing.assert_close(backward["dq_tensor"].float(), dq_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(backward["dk_tensor"].float(), dk_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(backward["dv_tensor"].float(), dv_ref, atol=3e-2, rtol=3e-2)

    q_bshd, k_bshd, v_bshd = (tensor.transpose(1, 2) for tensor in (q, k, v))
    do_bshd = do.transpose(1, 2)
    forward_bshd = BSA.block_sparse_attention_forward(
        q_bshd,
        k_bshd,
        v_bshd,
        q2k,
        block_sparse_num,
        block_sizes,
        sparse_block_size=block_size,
        layout="bshd",
    )
    dq_bshd = torch.empty_like(q_bshd)
    dk_bshd = torch.empty_like(k_bshd)
    dv_bshd = torch.empty_like(v_bshd)
    backward_bshd = BSA.block_sparse_attention_backward(
        do_bshd,
        q_bshd,
        k_bshd,
        v_bshd,
        forward_bshd["o_tensor"],
        forward_bshd["lse_tensor"],
        q2k,
        block_sparse_num,
        None,
        dq_tensor=dq_bshd,
        dk_tensor=dk_bshd,
        dv_tensor=dv_bshd,
        sparse_block_size=block_size,
        layout="bshd",
    )
    assert backward_bshd["dq_tensor"].data_ptr() == dq_bshd.data_ptr()
    assert backward_bshd["dk_tensor"].data_ptr() == dk_bshd.data_ptr()
    assert backward_bshd["dv_tensor"].data_ptr() == dv_bshd.data_ptr()
    torch.testing.assert_close(backward_bshd["dq_tensor"].transpose(1, 2).float(), dq_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(backward_bshd["dk_tensor"].transpose(1, 2).float(), dk_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(backward_bshd["dv_tensor"].transpose(1, 2).float(), dv_ref, atol=3e-2, rtol=3e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=5)
@pytest.mark.parametrize(
    ("bucket_size_blocks", "seqlen_k", "softmax_scale"),
    [
        # Two Q blocks fit in the default 1088-block bucket: direct dK/dV.
        (None, 4 * 64, 0.17),
        # One Q block per bucket forces two groups: FP32 atomic fallback.
        (1, 4 * 64, 0.17),
        # A physical K tail keeps the existing predicated fallback.
        (None, 4 * 64 - 13, None),
    ],
)
def test_bsa_attention_backward_sm100_blk64(
    bucket_size_blocks,
    seqlen_k,
    softmax_scale,
    monkeypatch,
):
    if not torch.cuda.is_available():
        pytest.skip("block sparse attention tests require CUDA")
    major, _ = torch.cuda.get_device_capability()
    if major not in {10, 11}:
        pytest.skip("blk64 backward test is specific to SM100/SM110")

    BSA = _import_bsa()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    workspace_modes = []
    allocate_workspace = interface._empty_bwd_workspace_with_zeroed_accum

    def track_workspace_mode(**kwargs):
        workspace_modes.append(kwargs.get("include_dkv_accum", True))
        return allocate_workspace(**kwargs)

    monkeypatch.setattr(
        interface,
        "_empty_bwd_workspace_with_zeroed_accum",
        track_workspace_mode,
    )
    block_size = 64
    batch, heads, seqlen_q, dim = 1, 2, 2 * block_size, 128
    q = torch.randn((batch, heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    do = torch.randn_like(q)
    q2k, block_sparse_num, block_sizes = make_fixed_metadata(batch, heads, seqlen_q, seqlen_k, block_size)
    # Metadata selects KV blocks 0 and 2, leaving 1 and 3 empty.
    block_sizes[2] = block_size // 2
    mask = block_sparse_mask(q2k, block_sparse_num, block_sizes, seqlen_q, seqlen_k, block_size)
    _, _, dq_ref, dk_ref, dv_ref = attention_backward_reference(
        q,
        k,
        v,
        do,
        mask,
        softmax_scale,
    )

    dq = torch.full_like(q, torch.nan)
    dk = torch.full_like(k, torch.nan)
    dv = torch.full_like(v, torch.nan)

    forward = BSA.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        block_sparse_num,
        block_sizes,
        sparse_block_size=64,
        softmax_scale=softmax_scale,
        use_clc=False,
    )
    backward = BSA.block_sparse_attention_backward(
        do,
        q,
        k,
        v,
        forward["o_tensor"],
        forward["lse_tensor"],
        q2k,
        block_sparse_num,
        block_sizes,
        dq_tensor=dq,
        dk_tensor=dk,
        dv_tensor=dv,
        bucket_size_blocks=bucket_size_blocks,
        sparse_block_size=64,
        softmax_scale=softmax_scale,
    )
    assert backward["dq_tensor"].data_ptr() == dq.data_ptr()
    assert backward["dk_tensor"].data_ptr() == dk.data_ptr()
    assert backward["dv_tensor"].data_ptr() == dv.data_ptr()
    expect_dkv_workspace = bucket_size_blocks == 1 or seqlen_k % block_size != 0
    assert workspace_modes == [expect_dkv_workspace]
    torch.testing.assert_close(backward["dq_tensor"].float(), dq_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(backward["dk_tensor"].float(), dk_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(backward["dv_tensor"].float(), dv_ref, atol=3e-2, rtol=3e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=8)
@pytest.mark.parametrize("num_q_blocks", [1, 4])
def test_bsa_attention_backward_blk128_dk_zero_init_accumulate_transition(num_q_blocks):
    """dK zero-init on the first Q block and accumulate on later Q blocks.

    num_q_blocks=1 exercises pure zero-initialization; num_q_blocks=4 makes
    several Q blocks write the same K blocks, exercising the runtime
    initialize-to-accumulate transition of the dK MMA predicate.
    """
    if not torch.cuda.is_available():
        pytest.skip("block sparse attention tests require CUDA")
    major, _ = torch.cuda.get_device_capability()
    if major not in {10, 11}:
        pytest.skip("blk128 backward is specific to SM100/SM110")

    BSA = _import_bsa()
    block_size = 128
    batch, heads, dim = 1, 1, 128
    seqlen_q = num_q_blocks * block_size
    seqlen_k = 4 * block_size
    q = torch.randn((batch, heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    do = torch.randn_like(q)

    # Every Q block attends to the same two K blocks.
    num_kv_blocks = seqlen_k // block_size
    selected = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    q2k = selected.view(1, 1, 1, 2).expand(batch, heads, num_q_blocks, 2).contiguous()
    block_sparse_num = 2
    full_block_sizes = torch.full((num_kv_blocks,), block_size, dtype=torch.int32, device="cuda")
    mask = block_sparse_mask(q2k, block_sparse_num, full_block_sizes, seqlen_q, seqlen_k, block_size)
    _, _, dq_ref, dk_ref, dv_ref = attention_backward_reference(q, k, v, do, mask)

    forward = BSA.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        block_sparse_num,
        None,
        sparse_block_size=block_size,
    )
    backward = BSA.block_sparse_attention_backward(
        do,
        q,
        k,
        v,
        forward["o_tensor"],
        forward["lse_tensor"],
        q2k,
        block_sparse_num,
        None,
        sparse_block_size=block_size,
    )
    torch.testing.assert_close(backward["dk_tensor"].float(), dk_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(backward["dq_tensor"].float(), dq_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(backward["dv_tensor"].float(), dv_ref, atol=3e-2, rtol=3e-2)
