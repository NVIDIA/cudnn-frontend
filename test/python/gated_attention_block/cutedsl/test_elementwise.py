# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage (5), the sigmoid gate — and its ``has_gate=False`` twin, V compaction.

Arch-agnostic (vectorized LDG/STG, no tcgen05, no shuffle), so these run
anywhere CuTe DSL does.
"""

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.gated_attention_block.kernels.elementwise import (
    compile_elementwise_gate,
    moved_bytes,
    run_elementwise_gate,
    validate_shape,
)

pytestmark = pytest.mark.L0

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _stream():
    return torch.cuda.current_stream().cuda_stream


def _run(src, gate, dst, *, h, d, rows_per_group=2):
    r = compile_elementwise_gate(dtype=src.dtype, h=h, d=d, has_gate=gate is not None, rows_per_group=rows_per_group)
    run_elementwise_gate(r, src, gate, dst, stream=_stream())
    torch.cuda.synchronize()


def test_moved_bytes_counts_the_gate_operand():
    t, h, d = 8192, 32, 256
    assert moved_bytes(t, h, d, has_gate=True) == 3 * t * h * d * 2
    assert moved_bytes(t, h, d, has_gate=False) == 2 * t * h * d * 2


def test_validate_shape_rejects():
    with pytest.raises(ValueError, match="multiple of 8"):
        validate_shape(250, 128)
    with pytest.raises(ValueError, match="multiple of the 32 lanes"):
        validate_shape(256, 100)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("t, h, d", [(64, 8, 256), (37, 4, 128), (1, 2, 64)])
def test_sigmoid_gate_matches_torch(dtype, t, h, d):
    torch.manual_seed(0)
    src = (torch.randn(t, h, d, device="cuda", dtype=torch.float32)).to(dtype)
    gate = (torch.randn(t, h, d, device="cuda", dtype=torch.float32) * 3).to(dtype)
    dst = torch.empty_like(src)
    _run(src, gate, dst, h=h, d=d)
    ref = (src.float() * torch.sigmoid(gate.float())).to(dtype)
    # Kernel and reference both compute in fp32 and round ONCE, so the budget is
    # ~1 ulp of the OUTPUT magnitude -- which is relative, not absolute. An atol
    # alone is the wrong shape here: |src*sigmoid(gate)| reaches ~4, where an
    # fp16 ulp is already 0.004. rtol = 2^-10 (fp16) / 2^-7 (bf16), plus a small
    # atol to cover values near zero.
    rtol, atol = (2.0**-7, 1e-3) if dtype is torch.bfloat16 else (2.0**-10, 1e-4)
    torch.testing.assert_close(dst.float(), ref.float(), rtol=rtol, atol=atol)


@requires_cuda
def test_gate_saturation_is_exact_at_the_ends():
    """sigmoid via the tanh identity must still give 0 and 1 at the tails --
    a gate that saturates wrong is finite, plausible, and silently wrong."""
    t, h, d = 8, 2, 256
    src = torch.full((t, h, d), 2.0, device="cuda", dtype=torch.bfloat16)
    gate = torch.full((t, h, d), -30.0, device="cuda", dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    _run(src, gate, dst, h=h, d=d)
    assert dst.abs().max().item() == 0.0, "sigmoid(-30) should gate the output to zero"
    gate.fill_(30.0)
    _run(src, gate, dst, h=h, d=d)
    torch.testing.assert_close(dst.float(), src.float(), rtol=0, atol=1e-2)


@requires_cuda
def test_in_place_matches_out_of_place():
    t, h, d = 96, 8, 256
    torch.manual_seed(1)
    src = torch.randn(t, h, d, device="cuda", dtype=torch.bfloat16)
    gate = (torch.randn(t, h, d, device="cuda", dtype=torch.float32) * 2).to(torch.bfloat16)
    out = torch.empty_like(src)
    _run(src, gate, out, h=h, d=d)
    ip = src.clone()
    _run(ip, gate, ip, h=h, d=d)
    torch.testing.assert_close(ip, out)


@requires_cuda
def test_strided_gate_from_a_fused_projection_buffer():
    """The block's real shape: O is compact, GATE is a column slice of the fused
    [T, N] projection output. If strided operands were not addressed natively the
    block would need a repack — which is the whole reason this kernel takes three
    independent strides."""
    t, h_q, d = 64, 8, 256
    n = 4 * h_q * d  # [Q | GATE | K | V]-ish width
    torch.manual_seed(2)
    proj = torch.randn(t, n, device="cuda", dtype=torch.bfloat16)
    gate_view = proj[:, h_q * d : 2 * h_q * d].view(t, h_q, d)
    assert gate_view.stride() == (n, d, 1)  # padded token stride: NOT compact
    src = torch.randn(t, h_q, d, device="cuda", dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    _run(src, gate_view, dst, h=h_q, d=d)
    ref = src.float() * torch.sigmoid(gate_view.float())
    torch.testing.assert_close(dst.float(), ref, rtol=0, atol=8e-3)


@requires_cuda
def test_compaction_copies_a_strided_source_exactly():
    """``has_gate=False``: the V path. A strided read, a compact write, and it
    must be BIT-exact — it is a copy, not an arithmetic op."""
    t, h_kv, d = 64, 2, 256
    n = 4 * h_kv * d
    torch.manual_seed(3)
    proj = torch.randn(t, n, device="cuda", dtype=torch.bfloat16)
    v_view = proj[:, 3 * h_kv * d :].view(t, h_kv, d)
    dst = torch.empty(t, h_kv, d, device="cuda", dtype=torch.bfloat16)
    _run(v_view, None, dst, h=h_kv, d=d)
    torch.testing.assert_close(dst, v_view.contiguous(), rtol=0, atol=0)


@requires_cuda
def test_gate_operand_presence_is_enforced_both_ways():
    """Init-time flags are compile-time specializations; execute must match in
    BOTH directions (Rule 1) — never a zeros fallback, never a silent ignore."""
    t, h, d = 8, 2, 256
    x = torch.randn(t, h, d, device="cuda", dtype=torch.bfloat16)
    with_gate = compile_elementwise_gate(dtype=torch.bfloat16, h=h, d=d, has_gate=True)
    without = compile_elementwise_gate(dtype=torch.bfloat16, h=h, d=d, has_gate=False)
    with pytest.raises(ValueError, match="must be bound"):
        run_elementwise_gate(with_gate, x, None, x, stream=_stream())
    with pytest.raises(ValueError, match="silently ignore"):
        run_elementwise_gate(without, x, x, x, stream=_stream())


@requires_cuda
@pytest.mark.parametrize("h", [1, 2, 3, 8, 32])
def test_const_head_count_matches_the_runtime_arm_bit_for_bit(h):
    """`const_head_count` bakes H into the address math. It must not narrow what
    the block serves, and it must not change a single bit.

    H is fixed per compiled ARTIFACT either way -- it has always been part of
    the compile-cache key -- so a caller with a different H gets a different
    artifact, exactly as before. The knob only decides whether the kernel
    divides by a register or by a constant. H=3 is in the list on purpose: a
    NON-power-of-two still has to be correct, it just strength-reduces to a
    multiply-shift instead of a mask.
    """
    import torch

    from cudnn.gated_attention_block.kernels.elementwise import compile_elementwise_gate, run_elementwise_gate

    d, t = 256, 64
    dt, dev = torch.bfloat16, torch.device("cuda")
    src = torch.randn(t, h, d, dtype=dt, device=dev)
    gate = torch.randn(t, h, d, dtype=dt, device=dev)
    out = {}
    for const in (True, False):
        dst = torch.zeros_like(src)
        r = compile_elementwise_gate(dtype=dt, h=h, d=d, has_gate=True, const_head_count=const)
        run_elementwise_gate(r, src, gate, dst, stream=torch.cuda.current_stream(dev).cuda_stream)
        torch.cuda.synchronize()
        out[const] = dst
    assert torch.equal(out[True], out[False]), f"H={h}: const_head_count changed the result"
    ref = src.float() * torch.sigmoid(gate.float())
    assert (out[True].float() - ref).abs().max().item() < 3e-2


@requires_cuda
def test_an_artifact_refuses_a_tensor_whose_head_count_it_was_not_built_for():
    """H is baked per artifact, so a mismatched bind must RAISE.

    This closes a pre-existing hole rather than one the knob opened: `n_rows`
    was already derived from the recipe's H, so before this check a mismatched
    tensor addressed the wrong rows and returned quietly.
    """
    import torch

    from cudnn.gated_attention_block.kernels.elementwise import compile_elementwise_gate, run_elementwise_gate

    d, t = 256, 32
    dt, dev = torch.bfloat16, torch.device("cuda")
    r = compile_elementwise_gate(dtype=dt, h=8, d=d, has_gate=True)
    src = torch.randn(t, 4, d, dtype=dt, device=dev)  # H=4, artifact is H=8
    gate = torch.randn(t, 4, d, dtype=dt, device=dev)
    dst = torch.empty_like(src)
    with pytest.raises(ValueError, match="H is fixed per artifact"):
        run_elementwise_gate(r, src, gate, dst, stream=torch.cuda.current_stream(dev).cuda_stream)


@requires_cuda
def test_an_artifact_refuses_a_tensor_whose_dtype_it_was_not_built_for():
    """The dtype is baked per artifact too (it is in the compile-cache key), so a
    mismatched bind must RAISE with the OPERAND named -- for src, gate and dst
    alike, not just src.

    The tvm-ffi boundary also rejects this on cutlass-dsl >= 4.8 (``Mismatched
    Tensor on argument #N, expected dtype=bfloat16``), but that is the DSL's
    check, indexed by argument position and tied to the DSL version; this one is
    ours, mirrors ``run_quantize``'s ``dtype_in`` guard, and the match below is
    on OUR message, so it proves the host guard fired first.
    """
    t, h, d = 8, 2, 256
    r = compile_elementwise_gate(dtype=torch.bfloat16, h=h, d=d, has_gate=True)
    assert r.dtype is torch.bfloat16
    bf = torch.randn(t, h, d, device="cuda", dtype=torch.bfloat16)
    f16 = bf.to(torch.float16)
    for name, args in (("src", (f16, bf, bf.clone())), ("gate", (bf, f16, bf.clone())), ("dst", (bf, bf, f16.clone()))):
        with pytest.raises(ValueError, match=f"{name} is torch.float16 but this artifact was compiled for torch.bfloat16"):
            run_elementwise_gate(r, *args, stream=_stream())
