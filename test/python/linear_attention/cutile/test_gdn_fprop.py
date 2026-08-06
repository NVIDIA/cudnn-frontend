# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Forward (fprop) tests for the GDN cuTile kernel, validated against the fp64
recurrent reference in ``reference_gdn``.

Covers: fp16/bf16, MHA and GVA head configs, full-chunk and partial-chunk
sequence lengths, ragged varlen batches (cu_seqlens), zero-length sequences,
explicit/auto scale, alpha (decay gate) and beta (write strength) on/off,
chunked prefill with state carry-over, initial state, and head-dim variants
including K != V and K = 256.
"""

from __future__ import annotations

import math
import random

import pytest
import torch

from ..common import (
    FWD_TOL,
    GDN_MARKS,
    HEAD_CONFIGS,
    HEAD_CONFIGS_SMALL,
    STATE_TOL,
    run_gdn,
)
from ..conftest import gen_gates, gen_qkv
from ..reference_gdn import gdn_reference, rms_ratio

pytestmark = GDN_MARKS

_SEED = 42


def _seed_all(seed=_SEED):
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _check_fwd(o, o_ref, dtype, what="o"):
    assert torch.isfinite(o).all(), f"non-finite values in {what}"
    r = rms_ratio(o, o_ref)
    assert r < FWD_TOL[dtype], f"{what} rms ratio {r:.4g} >= {FWD_TOL[dtype]}"


def _check_state(fs, fs_ref, dtype):
    assert torch.isfinite(fs).all(), "non-finite values in final_state"
    r = rms_ratio(fs, fs_ref)
    assert r < STATE_TOL[dtype], f"final_state rms ratio {r:.4g} >= {STATE_TOL[dtype]}"


def _run_fprop_case(
    dtype,
    H,
    HV,
    B,
    T,
    K=128,
    V=128,
    scale=None,
    alpha=True,
    beta=True,
    initial_state=False,
    cu_seqlens=None,
    total_T=None,
):
    """Build inputs, run the kernel, and compare o + final_state against the
    fp64 recurrent reference. ``cu_seqlens`` implies a packed batch with
    B == 1 and T == total_T."""
    _seed_all()
    q, k, v = gen_qkv(B, total_T or T, H, HV, K, V, dtype)
    g, b = gen_gates(B, total_T or T, HV, dtype, alpha=alpha, beta=beta)

    S0 = None
    if initial_state:
        N = B if cu_seqlens is None else cu_seqlens.numel() - 1
        S0 = torch.randn(N, HV, K, V, device="cuda", dtype=torch.float32) * 0.05

    o, fs = run_gdn(q, k, v, g, b, scale=scale, initial_state=S0, output_final_state=True, cu_seqlens=cu_seqlens)
    torch.cuda.synchronize()

    with torch.no_grad():
        o_ref, fs_ref = gdn_reference(q, k, v, g, b, scale=scale, initial_state=S0, cu_seqlens=cu_seqlens)

    _check_fwd(o, o_ref, dtype)
    assert fs is not None
    _check_state(fs, fs_ref, dtype)


@pytest.mark.parametrize("beta", [False, True])
@pytest.mark.parametrize("alpha", [False, True])
@pytest.mark.parametrize("scale", [1.0, "auto"])
@pytest.mark.parametrize("H,HV", HEAD_CONFIGS)
@pytest.mark.parametrize("B,T", [(1, 64), (1, 128), (1, 256), (2, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fprop_basic(dtype, B, T, H, HV, scale, alpha, beta):
    if not alpha and not beta:
        pytest.skip("output amplitude grows unbounded along the token dimension")
    scale = 1.0 / math.sqrt(128) if scale == "auto" else scale
    _run_fprop_case(dtype, H, HV, B, T, scale=scale, alpha=alpha, beta=beta)


@pytest.mark.parametrize("H,HV", [(1, 1), (2, 4), (16, 64)])
@pytest.mark.parametrize("T", [31, 61, 91, 121, 251])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fprop_nonfull(dtype, T, H, HV):
    """Sequence lengths that are not a multiple of the 64-token chunk."""
    _run_fprop_case(dtype, H, HV, 1, T)


@pytest.mark.parametrize("H,HV", [(1, 1), (2, 4), (16, 64)])
@pytest.mark.parametrize(
    "seq_lens",
    [[256, 256], [511, 501], [64, 128, 512], [31, 63, 93, 123, 150, 500]],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fprop_varlen_ragged(dtype, seq_lens, H, HV):
    """Ragged multi-sequence batches through the packed cu_seqlens path."""
    bounds = [0]
    for sl in seq_lens:
        bounds.append(bounds[-1] + sl)
    cu = torch.tensor(bounds, dtype=torch.int32, device="cuda")
    _run_fprop_case(dtype, H, HV, 1, None, cu_seqlens=cu, total_T=bounds[-1])


@pytest.mark.parametrize("H,HV", [(1, 1), (16, 64)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fprop_zero_length_sequence(dtype, H, HV, T=64):
    """A zero-length sequence in the varlen batch must not perturb the others."""
    _seed_all()
    q, k, v = gen_qkv(1, T, H, HV, 128, 128, dtype)
    g, b = gen_gates(1, T, HV, dtype)
    cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    cu_with_empty = torch.tensor([0, T, T], dtype=torch.int32, device="cuda")

    o_ref, fs_ref = run_gdn(q, k, v, g, b, output_final_state=True, cu_seqlens=cu)
    o, fs = run_gdn(q, k, v, g, b, output_final_state=True, cu_seqlens=cu_with_empty)
    torch.cuda.synchronize()

    torch.testing.assert_close(o, o_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(fs[0], fs_ref[0], atol=1e-3, rtol=1e-3)
    assert (fs[1] == 0).all(), "state of a zero-length sequence must stay zero"


@pytest.mark.parametrize("H,HV", HEAD_CONFIGS_SMALL)
@pytest.mark.parametrize("T1,T2", [(128, 128), (64, 192), (192, 121)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fprop_chunked_prefill(dtype, T1, T2, H, HV, B=2, K=128, V=128):
    """Two-phase prefill: the final state of part 1 feeds part 2 as its
    initial state; the concatenated output must match a single-shot run."""
    _seed_all()
    T = T1 + T2
    q, k, v = gen_qkv(B, T, H, HV, K, V, dtype)
    g, b = gen_gates(B, T, HV, dtype)

    def part(t0, t1, S0):
        return run_gdn(
            q[:, t0:t1].contiguous(),
            k[:, t0:t1].contiguous(),
            v[:, t0:t1].contiguous(),
            g[:, t0:t1].contiguous(),
            b[:, t0:t1].contiguous(),
            initial_state=S0,
            output_final_state=True,
        )

    o1, fs1 = part(0, T1, None)
    o2, fs2 = part(T1, T, fs1)
    o = torch.cat([o1, o2], dim=1)
    torch.cuda.synchronize()

    with torch.no_grad():
        o_ref, fs_ref = gdn_reference(q, k, v, g, b)

    # The state round-trips through the op's output dtype between the two
    # calls, so allow slightly more than the single-shot forward tolerance.
    tol = 1.5 * FWD_TOL[dtype]
    r_o = rms_ratio(o, o_ref)
    r_s = rms_ratio(fs2, fs_ref)
    assert r_o < tol, f"chunked-prefill o rms ratio {r_o:.4g} >= {tol}"
    assert r_s < tol, f"chunked-prefill final_state rms ratio {r_s:.4g} >= {tol}"


@pytest.mark.parametrize("H,HV", HEAD_CONFIGS_SMALL)
@pytest.mark.parametrize("T", [128, 251])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fprop_initial_state(dtype, T, H, HV):
    """Random (non-zero) initial recurrent state."""
    _run_fprop_case(dtype, H, HV, 1, T, initial_state=True)


@pytest.mark.parametrize("K,V", [(64, 64), (64, 128), (128, 128), (256, 128)])
@pytest.mark.parametrize("T", [128, 251])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fprop_head_dims(dtype, T, K, V, H=2, HV=2):
    """Head-dim variants: K != V and the K = 256 upper bound."""
    _run_fprop_case(dtype, H, HV, 1, T, K=K, V=V)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def _dummy_inputs(H=2, HV=2, T=64, K=128, V=128):
    q, k, v = gen_qkv(1, T, H, HV, K, V, torch.bfloat16)
    g, b = gen_gates(1, T, HV, torch.bfloat16)
    return q[0], k[0], v[0], g[0], b[0]


def _cu(*bounds):
    return torch.tensor(bounds, dtype=torch.int32, device="cuda")


def test_invalid_qk_head_mismatch():
    q, k, v, g, b = _dummy_inputs()
    with pytest.raises(Exception, match="head|No valid engine"):
        run_gdn(q.unsqueeze(0), k[:, :1].unsqueeze(0), v.unsqueeze(0), g.unsqueeze(0), b.unsqueeze(0), cu_seqlens=_cu(0, 64))


def test_invalid_gva_head_ratio():
    q, k, v, g, b = _dummy_inputs(H=2, HV=2)
    with pytest.raises(Exception, match="divisible|multiple|head|No valid engine"):
        run_gdn(q.unsqueeze(0), k.unsqueeze(0), v.repeat(1, 3, 1)[:, :3].unsqueeze(0), g.unsqueeze(0), b.unsqueeze(0), cu_seqlens=_cu(0, 64))


def test_invalid_initial_state_count():
    q, k, v, g, b = _dummy_inputs(T=128)
    S0 = torch.zeros(1, 2, 128, 128, device="cuda", dtype=torch.float32)
    with pytest.raises(Exception, match="initial"):
        run_gdn(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), g.unsqueeze(0), b.unsqueeze(0), cu_seqlens=_cu(0, 64, 128), initial_state=S0)
