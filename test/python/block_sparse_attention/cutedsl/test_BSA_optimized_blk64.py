# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical and layout regressions for the SM100/SM103 blk64 path."""

import pytest
import torch

from block_sparse_attention.cutedsl.bsa_reference import attention_reference, block_sparse_mask

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


def _bsa():
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("requires SM100/SM103")
    from cudnn import BSA

    return BSA


@pytest.mark.parametrize("num_q_blocks,expected", [(2999, None), (3000, 4096), (4096, 4096), (4097, 4096), (8192, 4096)])
def test_large_bucket_policy(num_q_blocks, expected):
    from cudnn.block_sparse_attention.csrc.bwd.sm100_blk64.bsa_bwd_sm100 import sm100_bwd_auto_bucketed_k2q_size_blocks

    assert sm100_bwd_auto_bucketed_k2q_size_blocks(num_q_blocks) == expected


@pytest.mark.parametrize("misalignment", ["stride", "address", "aligned_stride"])
@pytest.mark.parametrize("tensor_index", [0, 1, 2], ids=["q", "k", "v"])
@pytest.mark.parametrize("splits", [1, 2])
def test_bshd_sliced_inputs(misalignment, tensor_index, splits):
    bsa = _bsa()
    torch.manual_seed(20260911)
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = [torch.randn((2, 64, 3, 128), device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    padding = 1 if misalignment == "stride" else 8
    start = 1 if misalignment == "address" else 0
    storage = torch.empty((2, 64, 3, 128 + padding), device="cuda", dtype=torch.bfloat16)
    sliced = storage[..., start : start + 128]
    sliced.copy_(inputs[tensor_index])
    inputs[tensor_index] = sliced
    indices = torch.zeros((2, 3, 1, 1), device="cuda", dtype=torch.int32)
    sizes = torch.full((1,), 64, device="cuda", dtype=torch.int32)
    output, lse = bsa.block_sparse_attention_forward(*inputs, indices, 1, sizes, sparse_block_size=64, layout="bshd", kv_splits=splits)
    canonical = [t.transpose(1, 2) for t in inputs]
    mask = block_sparse_mask(indices, 1, sizes, 64, 64, 64)
    ref_output, ref_lse = attention_reference(*canonical, mask)
    assert output.is_contiguous()
    torch.testing.assert_close(output.transpose(1, 2).float(), ref_output, atol=0.03, rtol=0.03)
    torch.testing.assert_close(lse, ref_lse, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("clc,splits", [(False, 1), (True, 1), (True, 2)])
@pytest.mark.parametrize("seqlen", [193, 1089])
def test_forward_empty_rows_and_partial_tiles(clc, splits, seqlen):
    bsa = _bsa()
    torch.manual_seed(19000 + seqlen)
    torch.backends.cuda.matmul.allow_tf32 = False
    q, k, v = [torch.randn((1, 2, seqlen, 128), device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    blocks = (seqlen + 63) // 64
    indices = torch.arange(blocks - 1, -1, -1, device="cuda", dtype=torch.int32).view(1, 1, 1, -1).expand(1, 2, blocks, blocks).contiguous()
    counts = (torch.arange(2 * blocks, device="cuda", dtype=torch.int32) % (blocks + 1)).view(1, 2, blocks)
    sizes = torch.full((blocks,), 64, device="cuda", dtype=torch.int32)
    sizes[-1] = seqlen - (blocks - 1) * 64
    mask = block_sparse_mask(indices, 0, sizes, seqlen, seqlen, 64, counts)
    expected, expected_lse = attention_reference(q, k, v, mask)
    output, lse = bsa.block_sparse_attention_forward(
        q,
        k,
        v,
        indices,
        block_sizes=sizes,
        q2k_block_nums=counts,
        sparse_block_size=64,
        use_clc=clc,
        kv_splits=splits,
        allow_empty_block_nums=True,
    )
    torch.testing.assert_close(output.float(), expected, atol=0.03, rtol=0.03)
    torch.testing.assert_close(lse, expected_lse, atol=0.03, rtol=0.03)


@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
@pytest.mark.parametrize("sq,sk", [(193, 129), (320, 1088), (513, 1025)])
@pytest.mark.parametrize("variable", [False, True])
def test_backward_partial_tiles_and_buckets(layout, sq, sk, variable):
    bsa = _bsa()
    torch.manual_seed(27000 + sq)
    torch.backends.cuda.matmul.allow_tf32 = False
    q = torch.randn((2, 3, sq, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((2, 3, sk, 128), device="cuda", dtype=torch.bfloat16)
    v, do = torch.randn_like(k), torch.randn_like(q)
    nq, nk = (sq + 63) // 64, (sk + 63) // 64
    width = min(4, nk)
    indices = torch.rand((2, 3, nq, nk), device="cuda").argsort(-1)[..., :width].int().contiguous()
    counts = torch.randint(0, width + 1, (2, 3, nq), device="cuda", dtype=torch.int32) if variable else None
    sizes = torch.randint(1, 65, (nk,), device="cuda", dtype=torch.int32)
    sizes[-1].clamp_(max=sk - (nk - 1) * 64)
    mask = block_sparse_mask(indices, width, sizes, sq, sk, 64, counts)
    qr, kr, vr = [t.float().detach().requires_grad_() for t in (q, k, v)]
    scores = qr @ kr.transpose(-1, -2) / 128**0.5 + mask
    nonempty = torch.isfinite(mask).any(-1, keepdim=True)
    probabilities = torch.where(nonempty, scores, 0).softmax(-1).masked_fill(~torch.isfinite(mask), 0)
    expected = probabilities @ vr
    expected.backward(do.float())
    tensors = (q, k, v, do)
    if layout == "bshd":
        tensors = tuple(t.transpose(1, 2).contiguous() for t in tensors)
    q, k, v, do = tensors
    kwargs = dict(block_sparse_num=width, block_sizes=sizes, q2k_block_nums=counts, sparse_block_size=64, layout=layout)
    output, lse = bsa.block_sparse_attention_forward(q, k, v, indices, **kwargs, allow_empty_block_nums=True)
    canonical_output = output.transpose(1, 2) if layout == "bshd" else output
    torch.testing.assert_close(canonical_output.float(), expected, atol=0.03, rtol=0.03)
    buffers = tuple(torch.empty_like(t) for t in (q, k, v))
    for bucket_size in (1, 3, nq + 1):
        for buffer in buffers:
            buffer.fill_(float("nan"))
        gradients = bsa.block_sparse_attention_backward(
            do,
            q,
            k,
            v,
            output,
            lse,
            indices,
            **kwargs,
            bucket_size_blocks=bucket_size,
            dq_tensor=buffers[0],
            dk_tensor=buffers[1],
            dv_tensor=buffers[2],
        )
        for got, buffer, ref in zip(gradients, buffers, (qr.grad, kr.grad, vr.grad)):
            assert got.data_ptr() == buffer.data_ptr()
            got = got.transpose(1, 2) if layout == "bshd" else got
            torch.testing.assert_close(got.float(), ref, atol=0.03, rtol=0.03)
