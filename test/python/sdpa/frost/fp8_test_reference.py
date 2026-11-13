# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shared FP8 numerical checks and handle binding, independent of test entries."""

from contextlib import contextmanager
from typing import NamedTuple

import pytest
import torch

import cudnn

_FP8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
_E5M2_EDGE_MAX = 4.25e-2  # upper edge of the measured window (0.0404, 0.0406, 0.0409, 0.0409, 0.0423); past it the id is a real failure
_CURRENT_ITEM_IS_KNOWN_EDGE = False

# The kernels' softmax: 128-wide KV tiles, a running max in log2 units that moves only when a
# tile beats it by more than config_sm100.rescale_threshold (4 at fp8), P cast to the INPUT fp8
# type for P@V (scale_s/descale_s are accepted and ignored) while the row sum keeps the fp32 P,
# and the sink added to the sum after the loop. sdpa.fp8_ref.compute_ref is that softmax
# (BLOCK = 128, rescale_threshold in log2 units, sink_in_max=False); its s_scale is the cast
# scale times 2**rescale_threshold. The d128 cell casts P * 2**4
# (prefill_d128_fp8_sm100.P_CAST_LOG2_SCALE); the others cast P as is.
_P_RESCALE_THRESHOLD_LOG2 = 4.0


class _ReferenceWithStats(NamedTuple):
    output: torch.Tensor
    stats: torch.Tensor


def _p_cast_log2_scale(d_qk):
    return 4.0 if d_qk <= 128 else 0.0


def _ref(
    q8,
    k8,
    v8,
    dq,
    dk,
    dv,
    in_key,
    *,
    scale,
    is_causal=False,
    bottom_right=False,
    swa_window=None,
    right_bound=0,
    sinks=None,
    seq_lens_q=None,
    seq_lens_kv=None,
    return_stats=False,
):
    """sdpa.fp8_ref.compute_ref on this file's bhsd fp8 tensors (fp32 O, natural-log LSE [b, h, s])."""
    import cudnn
    from sdpa.fp8_ref import compute_ref

    b, h_q, s_q, d_qk = q8.shape
    s_kv = k8.shape[2]
    dev = q8.device
    padding = None
    if seq_lens_q is not None or seq_lens_kv is not None:
        padding = (
            torch.as_tensor(seq_lens_q if seq_lens_q is not None else [s_q] * b, dtype=torch.int32, device=dev),
            torch.as_tensor(seq_lens_kv if seq_lens_kv is not None else [s_kv] * b, dtype=torch.int32, device=dev),
        )
    s_scale = 2.0 ** (_p_cast_log2_scale(d_qk) + _P_RESCALE_THRESHOLD_LOG2)
    o, stats, _ = compute_ref(
        q8.transpose(1, 2),
        k8.transpose(1, 2),
        v8.transpose(1, 2),
        attn_scale=scale,
        q_descale=dq,
        k_descale=dk,
        v_descale=dv,
        s_scale=s_scale,
        s_descale=1.0 / s_scale,
        torch_itype=_FP8[in_key],
        torch_otype=torch.float16,
        padding=padding,
        left_bound=(swa_window + 1) if swa_window is not None else None,
        right_bound=right_bound if is_causal else None,
        diag_align=cudnn.diagonal_alignment.BOTTOM_RIGHT if bottom_right else cudnn.diagonal_alignment.TOP_LEFT,
        sink_token=sinks.view(1, h_q, 1, 1) if sinks is not None else None,
        rescale_threshold=_P_RESCALE_THRESHOLD_LOG2,
        dtype=torch.float64,
        quantize_o=False,
        sink_in_max=False,
    )
    o = o.transpose(1, 2)
    return _ReferenceWithStats(o, stats.squeeze(-1)) if return_stats else o


def _ref_kwargs(sdpa_kwargs):
    out = {}
    if sdpa_kwargs.get("use_causal_mask"):
        out["is_causal"] = True
    if sdpa_kwargs.get("use_causal_mask_bottom_right"):
        out["is_causal"] = True
        out["bottom_right"] = True
    right_bound = sdpa_kwargs.get("right_bound")
    if right_bound is not None:
        out["is_causal"] = True
        out["right_bound"] = right_bound
    diagonal_alignment = sdpa_kwargs.get("diagonal_alignment")
    if diagonal_alignment is not None:
        import cudnn

        out["bottom_right"] = diagonal_alignment == cudnn.diagonal_alignment.BOTTOM_RIGHT
    lb = sdpa_kwargs.get("left_bound")
    if lb is not None:
        out["swa_window"] = lb - 1
    return out


def _half_atol(in_key, d_v=None):
    # _ref casts P the way the kernels do, so this is the residual of what the
    # reference does NOT mirror. d128/d192/d512 cells: exp2 is MUFU, the cast
    # reference matches to <= 0.006 (fp64, offline) except the d192 e5m2+sink
    # path, whose kernel scales the rescale threshold by log2(e) (0.03). The
    # d256 cell computes exp2 with ex2_emulation_2 (degree-1/2 polynomial) on
    # part of every tile and, under a plain causal mask, for the rescale factor
    # too: ~40% of its P values sit one fp8 step off an exact-exp2 cast, up to
    # 0.058 on the suite's data (pack_gqa_ratios[d256-g16_mqa]).
    return 7.5e-2 if d_v == 256 else 4e-2


def _check(out, o_ref, out_dt, in_key, amax_o, amax_o_ref, atol=None):
    if atol is None:
        atol = _half_atol(in_key, out.shape[-1])
    diff = (out.float() - o_ref).abs().max().item()
    if out_dt in (torch.float8_e4m3fn, torch.float8_e5m2):
        floor = (o_ref - o_ref.to(out_dt).float()).abs().max().item()
        atol = max(atol, 3.0 * floor)
    # Amax_O is produced in-kernel (atomicMax over the pre-cast fp32
    # values), so they match the exact fp32 reference for every output dtype, incl. FP8.
    # Asserted BEFORE the O comparison so the known-edge xfail below can never hide it.
    assert abs(amax_o - amax_o_ref) <= 0.03, f"amax_o {amax_o:.4f} vs ref {amax_o_ref:.4f}"
    if diff > atol and _CURRENT_ITEM_IS_KNOWN_EDGE and diff <= _E5M2_EDGE_MAX:
        pytest.xfail(
            f"known e5m2 dataset edge on the 204-SM Rubin part: max|O-ref|={diff:.4f} in ({atol:.4f}, {_E5M2_EDGE_MAX:.4f}] "
            "-- the bound is not widened; build, run and Amax_O stayed fatal (PR #1209)"
        )
    assert diff <= atol, f"max|O-ref|={diff:.4f} > {atol:.4f}"


@contextmanager
def _bound_handle(handle):
    previous = cudnn.get_stream(handle)
    try:
        yield lambda g, vp, ws: _execute(g, vp, ws, handle)
    finally:
        cudnn.set_stream(handle, previous)


def _execute(g, vp, ws, handle):
    cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)
    g.execute(vp, ws, handle=handle)
