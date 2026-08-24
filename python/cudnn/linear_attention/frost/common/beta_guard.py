# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Erase-side beta safeguard for GDN-2 (per-token spectral non-expansion).

Per (token, head) row: with the l2-normalized key k, weights w_d = k_d^2,
n = sum w, a = sum beta*w, nu = sum beta^2*w, and the gate headroom
c^2 = exp(-2*max_d g_d), the erase operator I - k (beta.k)^T stays
non-expansive under the decay budget iff

    n*nu - a^2 <= (c^2 - 1) * (1 - (1 - a)^2 / c^2)

Violating rows are shrunk toward the key-weighted mean mu = a/n
(a-preserving, so one shot), rounded to the io dtype, re-tested against a
quantization tolerance, and flattened to mu on a re-test failure.

One device function shared by the prefill, recompute, and bprop row-major
beta blocks so the three kernels produce bitwise-identical beta_eff; the
straight-through backward recomputes it, never differentiates through it.
"""

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm

from cudnn.frost.tile_dsl.swizzle import swizzle_xor_128b

GUARD_MARGIN = 1.0 / 32
GUARD_QUANT_TOL_MULT = 4.0
MACHINE_EPSILON_FP16 = 2.0**-10
MACHINE_EPSILON_BF16 = 2.0**-7


@cute.jit
def beta_guard(
    cfg: cutlass.Constexpr,
    raw_beta_regs,
    raw_k_regs,
    k_inv_norm: cutlass.Float32,
    gate_prefix_ptr,
    decay_row: cutlass.Int32,
    lane_in_row_group: cutlass.Int32,
) -> None:
    """Rewrite the 16 per-lane beta registers to beta_eff in place."""
    zero = cutlass.Float32(0.0)
    one = cutlass.Float32(1.0)

    # ---- gate headroom ---------------------------------------------------------
    prev_row = decay_row - cutlass.Int32(1)
    if decay_row == 0:
        prev_row = cutlass.Int32(0)
    max_ratio = zero
    for dim_half in cutlass.range_constexpr(2):
        dim_base = dim_half * (cfg.d_k // 2) + lane_in_row_group * 8
        for f32_group in cutlass.range_constexpr(2):
            f32_dim_base = dim_base + f32_group * 4
            f32_segment = f32_dim_base // 32
            f32_segment_dim = f32_dim_base - f32_segment * 32
            row_idx = f32_segment * (cfg.b_t * 32) + decay_row * 32 + swizzle_xor_128b(decay_row, f32_segment_dim, elem_bytes=4)
            prev_idx = f32_segment * (cfg.b_t * 32) + prev_row * 32 + swizzle_xor_128b(prev_row, f32_segment_dim, elem_bytes=4)
            exp_g_frag = (gate_prefix_ptr + row_idx).load(count=4, alignment=16)
            exp_g_prev_frag = (gate_prefix_ptr + prev_idx).load(count=4, alignment=16)
            for elem in cutlass.range_constexpr(4):
                prev_val = exp_g_prev_frag[elem]
                if decay_row == 0:
                    prev_val = one
                ratio = exp_g_frag[elem] * cute.math.rcp(prev_val, approx=True, ftz=True)
                max_ratio = cute.math.max(max_ratio, ratio)
    max_ratio = cute.math.max(max_ratio, cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, max_ratio, 4, 31, kind=nvvm.Shfl.BFLY)))
    max_ratio = cute.math.max(max_ratio, cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, max_ratio, 2, 31, kind=nvvm.Shfl.BFLY)))
    max_ratio = cute.math.max(max_ratio, cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, max_ratio, 1, 31, kind=nvvm.Shfl.BFLY)))

    # ---- key-weighted stats ----------------------------------------------------
    n_val = zero
    a_val = zero
    nu_val = zero
    for reg_idx in cutlass.range_constexpr(2 * 8):
        k_norm = raw_k_regs[reg_idx] * k_inv_norm
        weight = k_norm * k_norm
        beta_val = raw_beta_regs[reg_idx]
        n_val = n_val + weight
        a_val = a_val + beta_val * weight
        nu_val = nu_val + (beta_val * beta_val) * weight
    n_val = n_val + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, n_val, 4, 31, kind=nvvm.Shfl.BFLY))
    n_val = n_val + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, n_val, 2, 31, kind=nvvm.Shfl.BFLY))
    n_val = n_val + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, n_val, 1, 31, kind=nvvm.Shfl.BFLY))
    a_val = a_val + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, a_val, 4, 31, kind=nvvm.Shfl.BFLY))
    a_val = a_val + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, a_val, 2, 31, kind=nvvm.Shfl.BFLY))
    a_val = a_val + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, a_val, 1, 31, kind=nvvm.Shfl.BFLY))
    nu_val = nu_val + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, nu_val, 4, 31, kind=nvvm.Shfl.BFLY))
    nu_val = nu_val + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, nu_val, 2, 31, kind=nvvm.Shfl.BFLY))
    nu_val = nu_val + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, nu_val, 1, 31, kind=nvvm.Shfl.BFLY))

    # ---- sensor ----------------------------------------------------------------
    inv_c2 = max_ratio * max_ratio
    c2 = cute.math.rcp(inv_c2, approx=True, ftz=True)
    r2 = cute.math.max(n_val * nu_val - a_val * a_val, zero)
    r2_crit = cute.math.max((c2 - one) * (one - ((one - a_val) * (one - a_val)) * inv_c2), zero)
    unsafe = cutlass.Boolean(False)
    if n_val > cutlass.Float32(1.0e-20):
        if r2 > r2_crit:
            unsafe = cutlass.Boolean(True)

    # ---- projection ------------------------------------------------------------
    inv_n = cute.math.rcp(cute.math.max(n_val, cutlass.Float32(1.0e-20)), approx=True, ftz=True)
    mu = a_val * inv_n
    eta = cute.math.sqrt(cutlass.Float32(1.0 - GUARD_MARGIN) * r2_crit * cute.math.rcp(cute.math.max(r2, cutlass.Float32(1.0e-30)), approx=True, ftz=True))
    eta = cute.math.min(cute.math.max(eta, zero), one)

    # ---- quantize + re-test ----------------------------------------------------
    a_q = zero
    nu_q = zero
    for reg_idx in cutlass.range_constexpr(2 * 8):
        candidate = raw_beta_regs[reg_idx]
        if unsafe:
            candidate = mu + eta * (candidate - mu)
        candidate_q = candidate.to(cfg.io_dtype).to(cutlass.Float32)
        raw_beta_regs[reg_idx] = candidate_q
        k_norm = raw_k_regs[reg_idx] * k_inv_norm
        weight = k_norm * k_norm
        a_q = a_q + candidate_q * weight
        nu_q = nu_q + (candidate_q * candidate_q) * weight
    a_q = a_q + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, a_q, 4, 31, kind=nvvm.Shfl.BFLY))
    a_q = a_q + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, a_q, 2, 31, kind=nvvm.Shfl.BFLY))
    a_q = a_q + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, a_q, 1, 31, kind=nvvm.Shfl.BFLY))
    nu_q = nu_q + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, nu_q, 4, 31, kind=nvvm.Shfl.BFLY))
    nu_q = nu_q + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, nu_q, 2, 31, kind=nvvm.Shfl.BFLY))
    nu_q = nu_q + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, nu_q, 1, 31, kind=nvvm.Shfl.BFLY))
    r2_q = cute.math.max(n_val * nu_q - a_q * a_q, zero)
    r2_crit_q = cute.math.max((c2 - one) * (one - ((one - a_q) * (one - a_q)) * inv_c2), zero)
    quant_eps = MACHINE_EPSILON_FP16 if cfg.io_dtype == cutlass.Float16 else MACHINE_EPSILON_BF16
    quant_tol = cutlass.Float32(GUARD_QUANT_TOL_MULT * quant_eps) * (n_val * nu_q + a_q * a_q)
    fallback = cutlass.Boolean(False)
    if unsafe:
        if r2_q > r2_crit_q + quant_tol:
            fallback = cutlass.Boolean(True)
    mu_q = (a_q * inv_n).to(cfg.io_dtype).to(cutlass.Float32)
    for reg_idx in cutlass.range_constexpr(2 * 8):
        final_val = raw_beta_regs[reg_idx]
        if fallback:
            final_val = mu_q
        raw_beta_regs[reg_idx] = final_val
