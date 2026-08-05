# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Backward (bprop) tests for the GDN cuTile kernel: gradients of q, k, v, g,
beta (and the initial state) are compared against autograd through the fp64
recurrent reference in ``reference_gdn`` using a shared random upstream
gradient.

Covers: fp16/bf16, MHA and GVA head configs, full-chunk and partial-chunk
sequence lengths, ragged varlen batches (cu_seqlens), zero-length sequences,
explicit/auto scale, alpha (decay gate) and beta (write strength) on/off,
chunked prefill with state carry-over, initial-state (dh0) and final-state
(dht) gradient paths, and head-dim variants including K != V and K = 256.
"""

from __future__ import annotations

import math
import random

import pytest
import torch

from ..common import (
    BWD_TOL,
    FWD_TOL,
    GDN_MARKS,
    HEAD_CONFIGS,
    HEAD_CONFIGS_SMALL,
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


def _run_bprop_case(
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
    state_grad=False,
    cu_seqlens=None,
    total_T=None,
):
    """Run kernel fwd+bwd and reference fwd+bwd with a shared upstream
    gradient; assert forward parity and per-input gradient parity."""
    _seed_all()
    Teff = total_T or T
    q0, k0, v0 = gen_qkv(B, Teff, H, HV, K, V, dtype)
    g0, b0 = gen_gates(B, Teff, HV, dtype, alpha=alpha, beta=beta)

    S0_data = None
    if initial_state:
        N = B if cu_seqlens is None else cu_seqlens.numel() - 1
        S0_data = torch.randn(N, HV, K, V, device="cuda", dtype=torch.float32) * 0.05

    w = torch.randn(B, Teff, HV, V, device="cuda", dtype=torch.float32)
    if state_grad:
        N = B if cu_seqlens is None else cu_seqlens.numel() - 1
        wf = torch.randn(N, HV, K, V, device="cuda", dtype=torch.float32)

    # --- kernel ---
    leaves = {
        "q": q0.clone().requires_grad_(),
        "k": k0.clone().requires_grad_(),
        "v": v0.clone().requires_grad_(),
        "g": g0.clone().requires_grad_(),
        "beta": b0.clone().requires_grad_(),
    }
    S0 = S0_data.clone().requires_grad_() if initial_state else None
    o, fs = run_gdn(
        leaves["q"],
        leaves["k"],
        leaves["v"],
        leaves["g"],
        leaves["beta"],
        scale=scale,
        initial_state=S0,
        output_final_state=state_grad,
        cu_seqlens=cu_seqlens,
    )
    loss = (o.float() * w).sum()
    if state_grad:
        loss = loss + (fs.float() * wf).sum()
    loss.backward()
    torch.cuda.synchronize()

    # --- fp64 reference ---
    ref_leaves = {n: t.detach().double().requires_grad_() for n, t in leaves.items()}
    S0_ref = S0_data.detach().double().requires_grad_() if initial_state else None
    o_ref, fs_ref = gdn_reference(
        ref_leaves["q"],
        ref_leaves["k"],
        ref_leaves["v"],
        ref_leaves["g"],
        ref_leaves["beta"],
        scale=scale,
        initial_state=S0_ref,
        cu_seqlens=cu_seqlens,
    )
    loss_ref = (o_ref * w.double()).sum()
    if state_grad:
        loss_ref = loss_ref + (fs_ref * wf.double()).sum()
    loss_ref.backward()

    r_o = rms_ratio(o, o_ref)
    assert r_o < FWD_TOL[dtype], f"forward o rms ratio {r_o:.4g} >= {FWD_TOL[dtype]}"

    pairs = [(f"d{n}", leaves[n].grad, ref_leaves[n].grad) for n in leaves]
    if initial_state:
        pairs.append(("dh0", S0.grad, S0_ref.grad))
    for name, got, ref in pairs:
        assert got is not None, f"no gradient for {name}"
        assert torch.isfinite(got).all(), f"non-finite gradient for {name}"
        r = rms_ratio(got, ref)
        assert r < BWD_TOL[dtype], f"{name} rms ratio {r:.4g} >= {BWD_TOL[dtype]}"


@pytest.mark.parametrize("alpha,beta", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("scale", [1.0, "auto"])
@pytest.mark.parametrize("H,HV", HEAD_CONFIGS)
@pytest.mark.parametrize("B,T", [(1, 64), (1, 128), (1, 256), (2, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bprop_basic(dtype, B, T, H, HV, scale, alpha, beta):
    scale = 1.0 / math.sqrt(128) if scale == "auto" else scale
    _run_bprop_case(dtype, H, HV, B, T, scale=scale, alpha=alpha, beta=beta)


@pytest.mark.parametrize("H,HV", [(1, 1), (2, 4), (16, 64)])
@pytest.mark.parametrize("T", [31, 61, 91, 121, 251])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bprop_nonfull(dtype, T, H, HV):
    """Backward through sequence lengths that are not chunk multiples."""
    _run_bprop_case(dtype, H, HV, 1, T)


@pytest.mark.parametrize("H,HV", [(1, 1), (2, 4), (16, 64)])
@pytest.mark.parametrize(
    "seq_lens",
    [[256, 256], [511, 501], [64, 128, 512], [31, 63, 93, 123, 150, 500], [255, 257]],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bprop_varlen_ragged(dtype, seq_lens, H, HV):
    """Backward through ragged varlen batches (cu_seqlens path)."""
    bounds = [0]
    for sl in seq_lens:
        bounds.append(bounds[-1] + sl)
    cu = torch.tensor(bounds, dtype=torch.int32, device="cuda")
    _run_bprop_case(dtype, H, HV, 1, None, cu_seqlens=cu, total_T=bounds[-1])


@pytest.mark.parametrize("H,HV", [(1, 1), (16, 64)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bprop_zero_length_sequence(dtype, H, HV, T=64):
    """A zero-length sequence in the varlen batch must not perturb the
    gradients of the others."""
    _seed_all()
    q, k, v = gen_qkv(1, T, H, HV, 128, 128, dtype)
    g, b = gen_gates(1, T, HV, dtype)
    w = torch.randn(1, T, HV, 128, device="cuda", dtype=torch.float32)

    def run(cu):
        leaves = {
            "q": q.clone().requires_grad_(),
            "k": k.clone().requires_grad_(),
            "v": v.clone().requires_grad_(),
            "g": g.clone().requires_grad_(),
            "beta": b.clone().requires_grad_(),
        }
        o, _ = run_gdn(leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"], cu_seqlens=cu)
        (o.float() * w).sum().backward()
        torch.cuda.synchronize()
        return {n: t.grad for n, t in leaves.items()}

    ref = run(torch.tensor([0, T], dtype=torch.int32, device="cuda"))
    got = run(torch.tensor([0, T, T], dtype=torch.int32, device="cuda"))
    for name in ref:
        torch.testing.assert_close(got[name], ref[name], atol=1e-3, rtol=1e-3, msg=f"d{name} perturbed by the zero-length sequence")


@pytest.mark.parametrize("H,HV", HEAD_CONFIGS_SMALL)
@pytest.mark.parametrize("T1,T2", [(128, 128), (64, 192), (192, 121)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bprop_chunked_prefill(dtype, T1, T2, H, HV, B=2, K=128, V=128):
    """Backward through a two-phase prefill: the state carried from part 1
    into part 2 chains the gradient (dht -> dh0) across the boundary; the
    leaf gradients must match a single-shot fp64 reference."""
    _seed_all()
    T = T1 + T2
    q0, k0, v0 = gen_qkv(B, T, H, HV, K, V, dtype)
    g0, b0 = gen_gates(B, T, HV, dtype)
    w = torch.randn(B, T, HV, V, device="cuda", dtype=torch.float32)

    leaves = {
        "q": q0.clone().requires_grad_(),
        "k": k0.clone().requires_grad_(),
        "v": v0.clone().requires_grad_(),
        "g": g0.clone().requires_grad_(),
        "beta": b0.clone().requires_grad_(),
    }

    def part(t0, t1, S0, output_final_state):
        return run_gdn(
            leaves["q"][:, t0:t1],
            leaves["k"][:, t0:t1],
            leaves["v"][:, t0:t1],
            leaves["g"][:, t0:t1],
            leaves["beta"][:, t0:t1],
            initial_state=S0,
            output_final_state=output_final_state,
        )

    o1, fs1 = part(0, T1, None, True)
    o2, _ = part(T1, T, fs1, False)
    o = torch.cat([o1, o2], dim=1)
    (o.float() * w).sum().backward()
    torch.cuda.synchronize()

    ref_leaves = {n: t.detach().double().requires_grad_() for n, t in leaves.items()}
    o_ref, _ = gdn_reference(ref_leaves["q"], ref_leaves["k"], ref_leaves["v"], ref_leaves["g"], ref_leaves["beta"])
    (o_ref * w.double()).sum().backward()

    # The carried state round-trips through the op's output dtype, so allow
    # slightly more than the single-shot backward tolerance.
    tol = 1.5 * BWD_TOL[dtype]
    for name in leaves:
        r = rms_ratio(leaves[name].grad, ref_leaves[name].grad)
        assert r < tol, f"chunked-prefill d{name} rms ratio {r:.4g} >= {tol}"


@pytest.mark.parametrize("H,HV", HEAD_CONFIGS_SMALL)
@pytest.mark.parametrize("T", [128, 251])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bprop_initial_state(dtype, T, H, HV):
    """Gradient flow through a random (non-zero) initial recurrent state
    (dh0 path, no final-state gradient)."""
    _run_bprop_case(dtype, H, HV, 1, T, initial_state=True)


@pytest.mark.parametrize("H,HV", HEAD_CONFIGS_SMALL)
@pytest.mark.parametrize("T", [128, 192, 251])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bprop_state_grads(dtype, T, H, HV):
    """Initial-state gradient (dh0) and final-state gradient (dht) paths:
    the loss includes the final state and the initial state requires grad."""
    _run_bprop_case(dtype, H, HV, 1, T, initial_state=True, state_grad=True)


@pytest.mark.parametrize("K,V", [(64, 64), (64, 128), (128, 128), (256, 128)])
@pytest.mark.parametrize("T", [128, 251])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_bprop_head_dims(dtype, T, K, V, H=2, HV=2):
    """Backward through head-dim variants: K != V and the K = 256 bound."""
    _run_bprop_case(dtype, H, HV, 1, T, K=K, V=V)
