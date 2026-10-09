# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the lifetime of shared V across the two SM90 MMA warpgroups."""

import importlib
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from block_sparse_attention.cutedsl.bsa_reference import attention_backward_reference

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


_DELAY_HELPER = """
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def _test_delay_before_v_read(*, loc=None, ip=None):
    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="nanosleep.u32 1000000;",
        constraints="",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


"""


def _load_delayed_kernel(kernel, tmp_path, monkeypatch):
    """Delay the first shared-V read in a temporary copy of the current kernel.

    This changes scheduling only: no input poisoning, math changes, production
    debug switch, or checked-in duplicate of the kernel. Exact anchors make a
    source refactor fail visibly instead of silently dropping the intervention.
    """
    source = Path(kernel.__file__).read_text()
    helper_anchor = "SM90_BWD_SPARSE_BLOCK_SIZE = 64\n"
    read_anchor = (
        "                    if const_expr(self.V_in_regs):\n"
        "                        if pds_iter == 0:\n"
        "                            cute.copy(smem_thr_copy_V, tdPsV, tdPrV_copy_view)\n"
    )
    assert source.count(helper_anchor) == 1
    assert source.count(read_anchor) == 1
    source = source.replace(helper_anchor, _DELAY_HELPER + helper_anchor)
    delayed_read = read_anchor.replace(
        "                            cute.copy",
        "                            for _delay_iteration in cutlass.range(8, unroll=1):\n"
        "                                _test_delay_before_v_read()\n"
        "                            cute.copy",
    )
    source = source.replace(read_anchor, delayed_read)
    path = tmp_path / "sm90_backward_with_test_delay.py"
    path.write_text(source)
    name = f"{kernel.__name__}._sync_regression"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses and CuTe source inspection need a real file and module entry.
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module.BlockSparseAttnBackwardSm90Blk64


@pytest.fixture(params=[False, True], ids=["normal", "first_read_delay"])
def sm90_bsa(request, tmp_path, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("shared-V lifetime regression is specific to SM90")
    try:
        from cudnn import BSA

        interface = importlib.import_module("cudnn.block_sparse_attention._interface")
        kernel = importlib.import_module("cudnn.block_sparse_attention.csrc.bwd.sm90_blk64.bsa_bwd_sm90")
    except (ImportError, OSError) as error:
        pytest.skip(f"block sparse attention optional dependencies are unavailable: {error}")

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    # The public wrapper's key does not distinguish the test-only kernel class.
    # Restore both the original cache and class binding after each test.
    monkeypatch.setattr(interface._bsa_attn_bwd_bucketed_k2q_csr, "compile_cache", {})
    if request.param:
        monkeypatch.setattr(interface, "BlockSparseAttnBackwardSm90Blk64", _load_delayed_kernel(kernel, tmp_path, monkeypatch))
    return BSA


def _one_edge_reference(q, k, v, do, q2k, block_sizes):
    """Use the existing FP32 reference once per Q block, batched together.

    Each Q block has exactly one selected KV block. Its softmax is independent
    of the other Q blocks, so dK/dV contributions can be summed by KV index.
    This avoids a large dense mask in the 385-Q-block bucket-boundary case.
    All tensors passed here use BHSD layout.
    """
    batch, heads, seqlen_q, dim = q.shape
    num_q_blocks, num_kv_blocks = seqlen_q // 64, k.shape[2] // 64
    selected = q2k[..., 0].long()
    kv_index = (selected.reshape(batch * heads, num_q_blocks) + torch.arange(batch * heads, device=q.device)[:, None] * num_kv_blocks).flatten()
    q_blocks = q.reshape(-1, 1, 64, dim)
    do_blocks = do.reshape_as(q_blocks)
    k_blocks = k.reshape(-1, 64, dim).index_select(0, kv_index).unsqueeze(1)
    v_blocks = v.reshape(-1, 64, dim).index_select(0, kv_index).unsqueeze(1)
    valid_sizes = block_sizes[selected].reshape(-1, 1, 1, 1)
    valid = torch.arange(64, device=q.device).reshape(1, 1, 1, 64) < valid_sizes
    mask = torch.zeros(valid.shape, dtype=torch.float32, device=q.device).masked_fill(~valid, float("-inf"))
    o, lse, dq, dk_per_edge, dv_per_edge = attention_backward_reference(q_blocks, k_blocks, v_blocks, do_blocks, mask)
    dk = torch.zeros((batch * heads * num_kv_blocks, 64, dim), dtype=torch.float32, device=q.device)
    dv = torch.zeros_like(dk)
    dk.index_add_(0, kv_index, dk_per_edge.squeeze(1))
    dv.index_add_(0, kv_index, dv_per_edge.squeeze(1))
    return o.reshape_as(q), lse.reshape(batch, heads, seqlen_q), dq.reshape_as(q), dk.reshape_as(k), dv.reshape_as(v)


@pytest.mark.L0
@pytest.mark.parametrize(
    "batch,heads,kv_for_q,valid_sizes",
    [
        pytest.param(1, 3, (0,), (64,), id="single_incoming"),
        pytest.param(1, 3, (0, 0), (64,), id="two_incoming"),
        pytest.param(2, 2, (0, 0, 1), (17, 40, 64), id="multibatch_partial_blocks"),
        # Q0 and Q384 share KV0, with one incoming edge in each real 384-Q
        # bucket. Other KV blocks have one edge; no high-fan-in accumulation.
        pytest.param(1, 1, (*range(384), 0), (64,) * 384, id="bucket384_tail"),
    ],
)
def test_bsa_sm90_backward_shared_v_lifetime(sm90_bsa, batch, heads, kv_for_q, valid_sizes):
    num_q_blocks, num_kv_blocks = len(kv_for_q), len(valid_sizes)
    # The smallest case matches the legal-input diagnostic reproducer. A local
    # CPU generator keeps normal/delayed inputs identical without global RNG
    # changes. Physical Q/KV lengths stay padded; block_sizes masks KV only.
    generator = torch.Generator(device="cpu").manual_seed(20261003)
    tensors = []
    for blocks in (num_q_blocks, num_kv_blocks, num_kv_blocks, num_q_blocks):
        tensor = torch.randn((batch, heads, blocks * 64, 128), generator=generator, dtype=torch.bfloat16)
        assert torch.isfinite(tensor).all()
        tensors.append(tensor.transpose(1, 2).contiguous().cuda())
    q, k, v, do = tensors
    q2k = torch.tensor(kv_for_q, device="cuda", dtype=torch.int32).view(1, 1, num_q_blocks, 1).expand(batch, heads, -1, -1).contiguous()
    counts = torch.ones((batch, heads, num_q_blocks), device="cuda", dtype=torch.int32)
    block_sizes = torch.tensor(valid_sizes, device="cuda", dtype=torch.int32)
    reference = _one_edge_reference(*(tensor.transpose(1, 2) for tensor in tensors), q2k, block_sizes)

    forward = sm90_bsa.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        block_sparse_num=1,
        block_sizes=block_sizes,
        q2k_block_nums=counts,
        sparse_block_size=64,
        layout="bshd",
    )
    backward = sm90_bsa.block_sparse_attention_backward(
        do,
        q,
        k,
        v,
        forward["o_tensor"],
        forward["lse_tensor"],
        q2k,
        block_sparse_num=1,
        block_sizes=block_sizes,
        q2k_block_nums=counts,
        bucket_size_blocks=384,
        sparse_block_size=64,
        layout="bshd",
    )
    actual = (
        forward["o_tensor"].transpose(1, 2),
        forward["lse_tensor"],
        backward["dq_tensor"].transpose(1, 2),
        backward["dk_tensor"].transpose(1, 2),
        backward["dv_tensor"].transpose(1, 2),
    )
    for name, result, expected in zip(("O", "LSE", "dQ", "dK", "dV"), actual, reference):
        assert torch.isfinite(result).all(), f"{name} contains nonfinite values"
        tolerance = 1e-5 if name == "LSE" else 3e-2
        torch.testing.assert_close(result.float(), expected, atol=tolerance, rtol=tolerance, msg=lambda message: f"{name}: {message}")
    # A tolerance comparison alone could hide small stale values in KV blocks
    # with no incoming edges, or in the masked tail of a partial block.
    for kv_block, valid_size in enumerate(valid_sizes):
        start = kv_block * 64 + (valid_size if kv_block in kv_for_q else 0)
        end = (kv_block + 1) * 64
        for gradient in actual[3:]:
            assert torch.count_nonzero(gradient[:, :, start:end]).item() == 0
