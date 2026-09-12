# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the subchannel-scaled (second-level SFA2) grouped GEMM wgrad tests.

Input generation and the byte-exact torch reference of the kernel's op, per expert ``e``
with token range ``[k0, k1)`` split into ``sgk``-token scale blocks ``b`` (ascending)::

    dW[e] = partial_b * sfa2[:, b_idx] + dW[e]     with partial_b = A[:, b] @ B[b, :]   (f32)
    dW[e] = (dW[e] * gsa[e] * gsb[e]).to(bf16)

``A = a_fp4 * sfa`` / ``B = b_fp4 * sfb`` are the first-level NVFP4 dequantized operands. The
SFA2 grid is random and independent of the data: kernel and reference replay the identical
``partial * sfa2 + acc`` f32 ops (the kernel uses a non-contractible ``mul.rn`` + add), so any
values verify exactly. Ported from the bs_ggemm_harness ``tensors/wgrad.py`` generator and
``references/wgrad.py::wgrad_gemm_2nd_level`` / ``wgrad_gemm_baseline``.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from fe_api.grouped_gemm.test_grouped_gemm_unfused_subchannel_scaled_utils import (
    _pack_fp4x2,
    _quantize_nvfp4_first_level,
)
from fe_api.grouped_gemm.test_grouped_gemm_wgrad_utils import _wgrad_assemble_scales_2d2d


def _group_ranges(token_counts: Sequence[int]) -> List[Tuple[int, int]]:
    ranges, start = [], 0
    for k in token_counts:
        ranges.append((start, start + k))
        start += k
    return ranges


def _assemble_sf(scales_f32: torch.Tensor, token_counts: Sequence[int]) -> torch.Tensor:
    """Decoded first-level scales f32 (mn, tokens_sum/16) -> the kernel's assembled
    per-expert 128x4-atom e4m3 layout (round_up(mn, 128), round_up(tokens_sum/16, 4))."""
    mn = scales_f32.shape[0]
    parts = []
    for k0, k1 in _group_ranges(token_counts):
        if k1 > k0:
            parts.append(scales_f32[:, k0 // 16 : k1 // 16].to(torch.float8_e4m3fn))
    return _wgrad_assemble_scales_2d2d(parts, mn)


def make_wgrad_subchannel_problem(
    hidden: int,
    intermediate: int,
    token_counts: Sequence[int],
    sgk: int,
    global_scales: bool = True,
    seed: Optional[int] = 0,
) -> Dict:
    """Random wgrad inputs: ``token_counts[e]`` tokens per expert (each sgk-aligned; 0 =
    empty expert), concatenated along the token (K) axis.

    Returns the kernel-facing tensors (``a_tensor`` (hidden, tokens_sum) fp4x2 K-major,
    ``b_tensor`` (tokens_sum, intermediate) fp4x2 K-major, assembled ``sfa_tensor`` /
    ``sfb_tensor``, ``sfa2_tensor`` (hidden, tokens_sum/sgk) f32 hidden-contiguous,
    ``offsets_tensor`` int32 cumulative end offsets, optional {1, 2}-valued global scales)
    plus the decoded operands the reference consumes."""
    token_counts = tuple(int(k) for k in token_counts)
    for k in token_counts:
        assert k >= 0 and k % sgk == 0, f"per-expert tokens ({k}) must be sgk-aligned ({sgk})"
    tokens_sum = sum(token_counts)
    assert tokens_sum > 0, "at least one expert must have tokens"
    if seed is not None:
        torch.manual_seed(seed)
    l = len(token_counts)

    a_master = torch.randn((hidden, tokens_sum), dtype=torch.float32, device="cuda")
    b_master = torch.randn((intermediate, tokens_sum), dtype=torch.float32, device="cuda")
    a_vals, a_scales = _quantize_nvfp4_first_level(a_master)
    b_vals, b_scales = _quantize_nvfp4_first_level(b_master)

    a_deq = a_vals * a_scales.repeat_interleave(16, dim=1)
    b_deq = b_vals * b_scales.repeat_interleave(16, dim=1)

    a_tensor = _pack_fp4x2(a_vals).contiguous()  # (hidden, tokens_sum/2), token-innermost
    b_tensor = _pack_fp4x2(b_vals).contiguous().t()  # (tokens_sum/2, intermediate), strides (1, tokens_sum/2)

    # Feature-contiguous (strides (1, hidden)) like the live rowwise-quant sf2 view.
    sfa2_tensor = (torch.rand((tokens_sum // sgk, hidden), dtype=torch.float32, device="cuda") + 0.5).T

    offsets = torch.tensor([k1 for _, k1 in _group_ranges(token_counts)], dtype=torch.int32, device="cuda")
    gsa = gsb = None
    if global_scales:
        gsa = torch.randint(1, 3, (l,), dtype=torch.float32, device="cuda")
        gsb = torch.randint(1, 3, (l,), dtype=torch.float32, device="cuda")

    return dict(
        hidden=hidden,
        intermediate=intermediate,
        token_counts=token_counts,
        tokens_sum=tokens_sum,
        sgk=sgk,
        a_tensor=a_tensor,
        b_tensor=b_tensor,
        sfa_tensor=_assemble_sf(a_scales, token_counts),
        sfb_tensor=_assemble_sf(b_scales, token_counts),
        sfa2_tensor=sfa2_tensor,
        offsets_tensor=offsets,
        global_scale_a=gsa,
        global_scale_b=gsb,
        a_deq=a_deq,
        b_deq=b_deq,
    )


def _finish(dw: torch.Tensor, problem: Dict, out_init: Optional[torch.Tensor], accumulate: bool) -> torch.Tensor:
    l = len(problem["token_counts"])
    if problem["global_scale_a"] is not None:
        alpha = (problem["global_scale_a"].float() * problem["global_scale_b"].float()).reshape(l, 1, 1)
        dw = dw * alpha
    dw = dw.to(torch.bfloat16)
    if accumulate:
        # The kernel TMA-reduce-adds the bf16 tile into gmem: one f32-exact add of two
        # bf16 values, rounded RN back to bf16.
        assert out_init is not None
        dw = (out_init.float() + dw.float()).to(torch.bfloat16)
    return dw


def wgrad_subchannel_reference(
    problem: Dict,
    sfa2: Optional[torch.Tensor] = None,
    out_init: Optional[torch.Tensor] = None,
    accumulate: bool = False,
) -> torch.Tensor:
    """Port of ``references/wgrad.py::wgrad_gemm_2nd_level`` on the decoded operands:
    partial products per sgk-token block in ascending order, folded as
    ``dw = partial * sfa2 + dw`` (each op RN f32); alpha applied to the fully accumulated
    f32 tile BEFORE the bf16 cast, like the epilogue."""
    a, b = problem["a_deq"], problem["b_deq"]
    sfa2 = problem["sfa2_tensor"] if sfa2 is None else sfa2
    sgk = problem["sgk"]
    l = len(problem["token_counts"])
    dw = torch.zeros((l, problem["hidden"], problem["intermediate"]), dtype=torch.float32, device="cuda")
    for e, (k0, k1) in enumerate(_group_ranges(problem["token_counts"])):
        for b0 in range(k0, k1, sgk):
            blk = b0 // sgk
            partial = a[:, b0 : b0 + sgk] @ b[:, b0 : b0 + sgk].T
            dw[e] = partial * sfa2[:, blk : blk + 1] + dw[e]
    return _finish(dw, problem, out_init, accumulate)


def wgrad_baseline_reference(problem: Dict) -> torch.Tensor:
    """Port of ``references/wgrad.py::wgrad_gemm_baseline`` (single-level: one GEMM per
    expert, no block splitting) -- the ``sfa2 == 1`` equivalence target."""
    a, b = problem["a_deq"], problem["b_deq"]
    l = len(problem["token_counts"])
    dw = torch.zeros((l, problem["hidden"], problem["intermediate"]), dtype=torch.float32, device="cuda")
    for e, (k0, k1) in enumerate(_group_ranges(problem["token_counts"])):
        if k1 > k0:
            dw[e] = a[:, k0:k1] @ b[:, k0:k1].T
    return _finish(dw, problem, None, False)


def assert_bytes_equal(out: torch.Tensor, ref: torch.Tensor) -> None:
    torch.cuda.synchronize()
    assert out.shape == ref.shape and out.dtype == ref.dtype, (out.shape, out.dtype, ref.shape, ref.dtype)
    if not torch.equal(out, ref):
        diff = (out.float() - ref.float()).abs()
        raise AssertionError(f"kernel output is not byte-exact: {int((diff != 0).sum())} of {diff.numel()} elements differ, max |diff| = {diff.max().item()}")
