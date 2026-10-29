# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Miles-compatible packed QSA forward and backward."""

from __future__ import annotations

import importlib
from functools import lru_cache

import torch
from torch import Tensor

from cudnn._device import device_info

from ._contract import check_inputs, check_tensor


def _implementation(q, direction):
    from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error, cutedsl_state, cutedsl_too_old

    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        raise ImportError(cutedsl_requirement_error("QSA") or "QSA requires nvidia-cutlass-dsl")
    cc = device_info(q.device.index).compute_capability
    if cc not in ((10, 0), (10, 3), (10, 7)):
        raise NotImplementedError(f"QSA requires SM100, SM103 or SM107, got SM{cc[0]}{cc[1]}")
    reason = cutedsl_arch_requirement_error(cc)
    if reason:
        raise NotImplementedError(reason)
    arch = cc[0] * 10 + cc[1]
    family = 107 if arch == 107 else 100
    if direction == "bprop" and q.shape[1] == 12:
        family = 100
    return importlib.import_module(f".kernels.{direction}_sm{family}", __package__), arch


def _compact(sel_blocks, lo, hi, block_first, block_last, arch, stream):
    from ._metadata import compile_kernel

    ids = torch.empty(sel_blocks.shape, dtype=torch.int32, device=sel_blocks.device)
    counts = torch.empty((sel_blocks.shape[0],), dtype=torch.int32, device=sel_blocks.device)
    compile_kernel(arch)(sel_blocks, lo, hi, block_first, block_last, ids, counts, stream)
    return ids, counts


def _compact_backward(sel_blocks, lo, hi, block_first, block_last, arch, stream, kv_heads):
    if kv_heads == 1:
        from ._metadata import compile_group_kernel
        from .kernels.bprop_sm100 import NQ, BLK_BITS

        total, blocks = sel_blocks.shape
        if blocks > 1 << BLK_BITS:
            raise NotImplementedError(f"Paired QSA requires NB <= {1 << BLK_BITS} for block-owner encoding")
        groups = (total + NQ - 1) // NQ
        ids = torch.empty((groups, blocks), dtype=torch.int32, device=sel_blocks.device)
        counts = torch.empty(groups, dtype=torch.int32, device=sel_blocks.device)
        compile_group_kernel(arch, NQ, BLK_BITS)(sel_blocks, lo, hi, block_first, block_last, ids, counts, total, stream)
        return ids, counts
    return _compact(sel_blocks, lo, hi, block_first, block_last, arch, stream)


@lru_cache(maxsize=1)
def _forward_runtime():
    """Resolve forward imports after the DSL gate."""
    from cuda.bindings import driver as cuda
    from cudnn.api_base import TupleDict, _raw_stream
    from . import _forward

    return cuda, TupleDict, _raw_stream, _forward


def qsa_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sel_blocks: Tensor,
    lo: Tensor,
    hi: Tensor,
    blk_base: Tensor,
    tok_base: Tensor,
    block_first: Tensor,
    block_last: Tensor,
    scale: float,
    block_size: int = 4,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(out, out_fp32, lse)`` for packed THD QSA.

    Selection and inclusive token bounds follow Miles. Block spans and bases must
    describe the same packed sequences. See ``docs/operations/sparse_attention/qsa.md``.
    """
    total, hq, hkv, _ = check_inputs(q, k, v, sel_blocks, lo, hi, blk_base, tok_base, block_first, block_last, scale, block_size)
    with torch.cuda.device(q.device):
        impl, arch = _implementation(q, "fprop")
        cuda, TupleDict, raw_stream, _forward = _forward_runtime()
        stream = cuda.CUstream(raw_stream(torch)(q.device.index))
        out, out_fp32 = torch.empty_like(q), torch.empty_like(q, dtype=torch.float32)
        lse = torch.empty((hq, total), dtype=torch.float32, device=q.device)
        _forward.launch(impl, arch, q, k, v, sel_blocks, lo, hi, blk_base, tok_base, block_first, block_last, out, out_fp32, lse, float(scale), stream)
        return TupleDict(out=out, out_fp32=out_fp32, lse=lse)


def qsa_backward(
    grad_out: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out_fp32: Tensor,
    lse: Tensor,
    sel_blocks: Tensor,
    lo: Tensor,
    hi: Tensor,
    blk_base: Tensor,
    tok_base: Tensor,
    block_first: Tensor,
    block_last: Tensor,
    scale: float,
    block_size: int = 4,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(dq, dk, dv)`` using forward's FP32 output and natural-log LSE."""
    total, hq, hkv, _ = check_inputs(q, k, v, sel_blocks, lo, hi, blk_base, tok_base, block_first, block_last, scale, block_size)
    check_tensor("grad_out", grad_out, q.shape, q.dtype, q.device, 16)
    check_tensor("out_fp32", out_fp32, q.shape, torch.float32, q.device, 16)
    check_tensor("lse", lse, (hq, total), torch.float32, q.device)
    with torch.cuda.device(q.device):
        impl, arch = _implementation(q, "bprop")
        import cutlass
        from cuda.bindings import driver as cuda
        from cudnn.api_base import TupleDict

        stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
        compiled = impl.compile_kernel(cutlass.BFloat16, hq, hkv, arch)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        scratch = {
            "delta": torch.empty((total, hq), dtype=torch.float32, device=q.device),
            "dq_acc": torch.empty_like(q, dtype=torch.float32),
            "dk_acc": torch.empty_like(k, dtype=torch.float32),
            "dv_acc": torch.empty_like(v, dtype=torch.float32),
        }
        ids, counts = _compact_backward(sel_blocks, lo, hi, block_first, block_last, arch, stream, hkv)
        num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        impl.launch(
            compiled, grad_out, q, k, v, out_fp32, lse, ids, counts, lo, hi, block_first, block_last, dq, dk, dv, scratch, float(scale), stream, num_sms
        )
        return TupleDict(dq=dq, dk=dk, dv=dv)
