# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
import torch

from test_utils import torch_fork_set_rng
from block_sparse_attention.cutedsl.bsa_reference import attention_backward_reference, block_sparse_mask
from block_sparse_attention.cutedsl.bsa_utils import make_fixed_metadata, supported_block_size

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


def _import_bsa():
    try:
        from cudnn import BSA

        importlib.import_module("cudnn.block_sparse_attention._interface")

        return BSA
    except (ImportError, OSError) as error:
        pytest.skip(f"block sparse attention optional dependencies are unavailable: {error}")


@pytest.mark.L0
@torch_fork_set_rng(seed=7)
@pytest.mark.parametrize("seqlen", [33, 64])
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
def test_bsa_sm100_blk64_sum_odo_fp32_product(seqlen, layout):
    """Casting after O*dO has already lost BF16 product bits needed by dS."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in {10, 11}:
        pytest.skip("blk64 backward is specific to SM100/SM110")

    from cudnn.frost.buffers import cutedsl_requirement_error

    requirement_error = cutedsl_requirement_error("blk64 backward preprocessing test")
    if requirement_error is not None:
        pytest.skip(requirement_error)

    _import_bsa()
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda
    from cudnn.block_sparse_attention.csrc.bwd.sm100_blk64.bsa_bwd_sm100 import BlockSparseAttnBackwardSm100Blk64
    from cudnn.block_sparse_attention.csrc.utils.cute_dsl_utils import to_cute_tensor

    @cute.jit
    def launch_sum_odo(op: cutlass.Constexpr, out, dout, delta, lse, scaled_lse, stream):
        n, d, h, b = out.shape
        op.sum_OdO(
            cute.group_modes(out, 2, 4),
            cute.group_modes(dout, 2, 4),
            cute.group_modes(delta, 1, 3),
            cute.group_modes(lse, 1, 3),
            cute.group_modes(scaled_lse, 1, 3),
            cutlass.Float32(-1.0),
            cutlass.Float32(-1.4426950408889634),
            (n, n, d, (h, b)),
        ).launch(
            grid=(cute.ceil_div(n, op.sum_OdO_block_q), h, b),
            block=(op.sum_OdO_num_threads_d, op.sum_OdO_num_threads_q, 1),
            stream=stream,
        )

    batch, heads, dim = 2, 3, 128
    out = torch.randn((batch, heads, seqlen, dim), device="cuda", dtype=torch.bfloat16)
    dout = torch.randn_like(out)
    # 1.0625 is exactly BF16, but its square 1.12890625 is not.
    out[:, :, 0, :] = 1.0625
    dout[:, :, 0, :] = 1.0625
    if layout == "bshd":
        out, dout = (tensor.transpose(1, 2).contiguous().transpose(1, 2) for tensor in (out, dout))
    lse = torch.randn((batch, heads, seqlen), device="cuda", dtype=torch.float32)
    delta = torch.full_like(lse, torch.nan)
    scaled_lse = torch.full_like(lse, torch.nan)
    operands = [out.permute(2, 3, 1, 0), dout.permute(2, 3, 1, 0)]
    operands += [tensor.permute(2, 1, 0) for tensor in (delta, lse, scaled_lse)]
    compiled = cute.compile(
        launch_sum_odo,
        BlockSparseAttnBackwardSm100Blk64(64, False),
        *[to_cute_tensor(tensor, leading_dim=1 if i < 2 else 0) for i, tensor in enumerate(operands)],
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
    )
    compiled(*operands, cuda.CUstream(torch.cuda.current_stream().cuda_stream))

    reference = -(out.double() * dout.double()).sum(dim=-1)
    torch.testing.assert_close(delta.double(), reference, atol=2e-5, rtol=1e-6)
    torch.testing.assert_close(scaled_lse, lse * -1.4426950408889634, atol=0, rtol=0)


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
@pytest.mark.parametrize("variable_counts", [False, True])
def test_bsa_attention_backward_sm100_blk64(
    bucket_size_blocks,
    seqlen_k,
    softmax_scale,
    variable_counts,
    monkeypatch,
):
    if not torch.cuda.is_available():
        pytest.skip("block sparse attention tests require CUDA")
    major, _ = torch.cuda.get_device_capability()
    if major not in {10, 11}:
        pytest.skip("blk64 backward test is specific to SM100/SM110")

    BSA = _import_bsa()
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    if variable_counts:
        # Keep this small case on the paired-KV path, whose shared dK/dV needs the FP32 workspace.
        monkeypatch.setattr(interface, "_SM100_BLK64_PAIRED_KV_MIN_KV_TASKS", 0)
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
    counts = torch.full(q2k.shape[:-1], block_sparse_num, dtype=torch.int32, device=q.device) if variable_counts else None
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
        q2k_block_nums=counts,
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
        q2k_block_nums=counts,
    )
    assert backward["dq_tensor"].data_ptr() == dq.data_ptr()
    assert backward["dk_tensor"].data_ptr() == dk.data_ptr()
    assert backward["dv_tensor"].data_ptr() == dv.data_ptr()
    # Paired and residual tasks share FP32 accumulation even with a single Q group.
    expect_dkv_workspace = variable_counts or bucket_size_blocks == 1 or seqlen_k % block_size != 0
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
