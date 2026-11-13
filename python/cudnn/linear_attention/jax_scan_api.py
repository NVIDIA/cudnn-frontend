# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public JAX APIs for packed linear-attention scans and their gradients."""

from ._jax_execution import _DeltaConfig, _MambaConfig, _Operation, _backward, _differentiable, _forward


def gated_delta_net_fwd(
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    *,
    initial_state=None,
    a_log=None,
    dt_bias=None,
    output_final_state=False,
    scale=None,
    use_qk_l2norm_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    batch_invariant=False,
    checkpoint_every_n_tokens=0,
    gate_domain="log",
):
    """Return ``(output, final_state_or_None, residual)``; options are static."""
    config = _DeltaConfig(
        operation=_Operation.GDN,
        output_final_state=output_final_state,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        use_beta_sigmoid=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        gate_domain=gate_domain,
        batch_invariant=batch_invariant,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
    )
    return _forward((q, k, v, g, beta, None, cu_seqlens, initial_state, a_log, dt_bias), config)


def gated_delta_net(
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    *,
    initial_state=None,
    a_log=None,
    dt_bias=None,
    output_final_state=False,
    scale=None,
    use_qk_l2norm_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    batch_invariant=False,
    checkpoint_every_n_tokens=0,
    gate_domain="log",
):
    """Return ``(output, final_state_or_None)`` with JIT and first-order VJP support."""
    config = _DeltaConfig(
        operation=_Operation.GDN,
        output_final_state=output_final_state,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        use_beta_sigmoid=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        gate_domain=gate_domain,
        batch_invariant=batch_invariant,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
    )
    return _differentiable((q, k, v, g, beta, None, cu_seqlens, initial_state, a_log, dt_bias), config)


def gated_delta_net_v2_fwd(
    q,
    k,
    v,
    g,
    beta,
    w,
    cu_seqlens,
    *,
    initial_state=None,
    a_log=None,
    dt_bias=None,
    output_final_state=False,
    scale=None,
    use_qk_l2norm_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    batch_invariant=False,
    checkpoint_every_n_tokens=0,
    gate_domain="log",
    beta_guard=False,
    gate_lower_bound=None,
):
    """Return ``(output, final_state_or_None, residual)``; options are static."""
    config = _DeltaConfig(
        operation=_Operation.GDN2,
        output_final_state=output_final_state,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        use_beta_sigmoid=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        gate_domain=gate_domain,
        batch_invariant=batch_invariant,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        beta_guard=beta_guard,
        gate_lower_bound=gate_lower_bound,
    )
    return _forward((q, k, v, g, beta, w, cu_seqlens, initial_state, a_log, dt_bias), config)


def gated_delta_net_v2(
    q,
    k,
    v,
    g,
    beta,
    w,
    cu_seqlens,
    *,
    initial_state=None,
    a_log=None,
    dt_bias=None,
    output_final_state=False,
    scale=None,
    use_qk_l2norm_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    batch_invariant=False,
    checkpoint_every_n_tokens=0,
    gate_domain="log",
    beta_guard=False,
    gate_lower_bound=None,
):
    """Return ``(output, final_state_or_None)`` with JIT and first-order VJP support."""
    config = _DeltaConfig(
        operation=_Operation.GDN2,
        output_final_state=output_final_state,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        use_beta_sigmoid=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        gate_domain=gate_domain,
        batch_invariant=batch_invariant,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        beta_guard=beta_guard,
        gate_lower_bound=gate_lower_bound,
    )
    return _differentiable((q, k, v, g, beta, w, cu_seqlens, initial_state, a_log, dt_bias), config)


def gated_delta_product_fwd(
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    *,
    num_householder,
    initial_state=None,
    a_log=None,
    dt_bias=None,
    output_final_state=False,
    scale=None,
    use_qk_l2norm_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    batch_invariant=False,
    checkpoint_every_n_tokens=0,
    gate_domain="log",
):
    """Return ``(output, final_state_or_None, residual)``; options are static."""
    config = _DeltaConfig(
        operation=_Operation.GDP,
        output_final_state=output_final_state,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        use_beta_sigmoid=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        gate_domain=gate_domain,
        batch_invariant=batch_invariant,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        num_householder=num_householder,
    )
    return _forward((q, k, v, g, beta, None, cu_seqlens, initial_state, a_log, dt_bias), config)


def gated_delta_product(
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    *,
    num_householder,
    initial_state=None,
    a_log=None,
    dt_bias=None,
    output_final_state=False,
    scale=None,
    use_qk_l2norm_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    batch_invariant=False,
    checkpoint_every_n_tokens=0,
    gate_domain="log",
):
    """Return ``(output, final_state_or_None)`` with JIT and first-order VJP support."""
    config = _DeltaConfig(
        operation=_Operation.GDP,
        output_final_state=output_final_state,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        use_beta_sigmoid=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        gate_domain=gate_domain,
        batch_invariant=batch_invariant,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        num_householder=num_householder,
    )
    return _differentiable((q, k, v, g, beta, None, cu_seqlens, initial_state, a_log, dt_bias), config)


def mamba2_fwd(
    x,
    dt,
    A,
    B,
    C,
    cu_seqlens,
    *,
    D=None,
    dt_bias=None,
    z=None,
    initial_state=None,
    output_final_state=False,
    checkpoint_every_n_tokens=0,
    dt_softplus=True,
    dt_min=0.0,
    dt_max=None,
    batch_invariant=False,
):
    """Return output, optional final state, and residual for Mamba2 SSD.

    X is [T,H,64], B/C are [T,G,128], dt is [T,H], and A is [H].
    States are [sequences,H,64,128]. No implicit scaling is applied to C.
    """
    config = _MambaConfig(
        output_final_state=output_final_state,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        dt_softplus=dt_softplus,
        dt_min=dt_min,
        dt_max=dt_max,
        batch_invariant=batch_invariant,
    )
    return _forward((x, dt, A, B, C, cu_seqlens, initial_state, D, dt_bias, z), config)


def mamba2(
    x,
    dt,
    A,
    B,
    C,
    cu_seqlens,
    *,
    D=None,
    dt_bias=None,
    z=None,
    initial_state=None,
    output_final_state=False,
    checkpoint_every_n_tokens=0,
    dt_softplus=True,
    dt_min=0.0,
    dt_max=None,
    batch_invariant=False,
):
    """Mamba2 SSD with model-facing x/dt/A/B/C inputs; JIT and VJP supported.

    X is [T,H,64], B/C are [T,G,128], dt is [T,H], and A is [H].
    States are [sequences,H,64,128]. No implicit scaling is applied to C.
    """
    config = _MambaConfig(
        output_final_state=output_final_state,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        dt_softplus=dt_softplus,
        dt_min=dt_min,
        dt_max=dt_max,
        batch_invariant=batch_invariant,
    )
    return _differentiable((x, dt, A, B, C, cu_seqlens, initial_state, D, dt_bias, z), config)


def gated_delta_net_bwd(residual, doutput, *, d_final_state=None):
    """Return DeltaGradients; absent inputs have None gradients."""
    return _backward(residual, doutput, d_final_state)


def gated_delta_net_v2_bwd(residual, doutput, *, d_final_state=None):
    """Return DeltaGradients; absent inputs have None gradients."""
    return _backward(residual, doutput, d_final_state)


def gated_delta_product_bwd(residual, doutput, *, d_final_state=None):
    """Return DeltaGradients; absent inputs have None gradients."""
    return _backward(residual, doutput, d_final_state)


def mamba2_bwd(residual, doutput, *, d_final_state=None):
    """Return Mamba2Gradients; absent inputs have None gradients."""
    return _backward(residual, doutput, d_final_state)
