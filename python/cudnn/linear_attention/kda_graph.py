# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Framework-neutral KDA graph construction shared by torch and JAX."""

import cudnn


def build_fprop_graph(
    total,
    N,
    H,
    HK,
    HV,
    K,
    V,
    io_dtype,
    g_dtype,
    beta_dtype,
    state_dtype,
    cu_dtype,
    scale,
    output_final_state,
    use_qk_l2norm,
    batch_invariant,
    use_beta_sigmoid,
    allow_neg_eigval,
    safe_gate,
    gate_lower_bound,
    checkpoint,
    a_log_dtype=None,
    dt_bias_dtype=None,
):
    graph = cudnn.pygraph()
    HO = max(H, HV)
    q_t = graph.tensor([total, H, K], data_type=io_dtype, name="q")
    k_t = graph.tensor([total, HK, K], data_type=io_dtype, name="k")
    v_t = graph.tensor([total, HV, V], data_type=io_dtype, name="v")
    g_t = graph.tensor([total, HO, K], data_type=g_dtype, name="g")
    beta_t = graph.tensor([total, HO], data_type=beta_dtype, name="beta")
    cu_t = graph.tensor([N + 1], data_type=cu_dtype, name="cu_seqlens")
    state0_t = None
    if state_dtype is not None:
        state0_t = graph.tensor([N, HO, V, K], data_type=state_dtype, name="initial_state")
    a_log_t = None
    dt_bias_t = None
    if a_log_dtype is not None:
        a_log_t = graph.tensor([HO], data_type=a_log_dtype, name="a_log")
    if dt_bias_dtype is not None:
        dt_bias_t = graph.tensor([HO, K], data_type=dt_bias_dtype, name="dt_bias")
    O_t, fs_t, state_checkpoints_t = graph.kda(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        cu_seqlens=cu_t,
        initial_state=state0_t,
        a_log=a_log_t,
        dt_bias=dt_bias_t,
        scale=scale,
        output_final_state=output_final_state,
        use_qk_l2norm=use_qk_l2norm,
        batch_invariant=batch_invariant,
        use_beta_sigmoid=use_beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        gate_lower_bound=gate_lower_bound,
        checkpoint_every_n_tokens=checkpoint,
        name="kda",
    )
    return graph, dict(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        cu=cu_t,
        state0=state0_t,
        a_log=a_log_t,
        dt_bias=dt_bias_t,
        O=O_t,
        fs=fs_t,
        state_checkpoints=state_checkpoints_t,
    )


def build_bprop_graph(
    total,
    N,
    H,
    HK,
    HV,
    K,
    V,
    io_dtype,
    g_dtype,
    beta_dtype,
    state_dtype,
    dstate_in_dtype,
    cu_dtype,
    checkpoint_rows,
    scale,
    use_qk_l2norm,
    batch_invariant,
    use_beta_sigmoid=False,
    allow_neg_eigval=False,
    safe_gate=False,
    gate_lower_bound=None,
    a_log_dtype=None,
    dt_bias_dtype=None,
    checkpoint_every_n_tokens=0,
):
    graph = cudnn.pygraph()
    HO = max(H, HV)
    q_t = graph.tensor([total, H, K], data_type=io_dtype, name="q")
    k_t = graph.tensor([total, HK, K], data_type=io_dtype, name="k")
    v_t = graph.tensor([total, HV, V], data_type=io_dtype, name="v")
    g_t = graph.tensor([total, HO, K], data_type=g_dtype, name="g")
    beta_t = graph.tensor([total, HO], data_type=beta_dtype, name="beta")
    cu_t = graph.tensor([N + 1], data_type=cu_dtype, name="cu_seqlens")
    dO_t = graph.tensor([total, HO, V], data_type=io_dtype, name="dO")
    state0_t = None
    if state_dtype is not None:
        state0_t = graph.tensor([N, HO, V, K], data_type=state_dtype, name="initial_state")
    dfs_t = None
    if dstate_in_dtype is not None:
        dfs_t = graph.tensor([N, HO, V, K], data_type=dstate_in_dtype, name="d_final_state")
    checkpoints_t = None
    if checkpoint_rows is not None:
        checkpoints_t = graph.tensor([checkpoint_rows, HO, V, K], data_type=io_dtype, name="state_checkpoints")
    a_log_t = None
    dt_bias_t = None
    if a_log_dtype is not None:
        a_log_t = graph.tensor([HO], data_type=a_log_dtype, name="a_log")
    if dt_bias_dtype is not None:
        dt_bias_t = graph.tensor([HO, K], data_type=dt_bias_dtype, name="dt_bias")
    dQ_t, dK_t, dV_t, dG_t, dBeta_t, dstate0_t, dA_t, dDt_t = graph.kda_bwd(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        cu_seqlens=cu_t,
        dO=dO_t,
        state_checkpoints=checkpoints_t,
        initial_state=state0_t,
        d_final_state=dfs_t,
        a_log=a_log_t,
        dt_bias=dt_bias_t,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens or None,
        batch_invariant=batch_invariant,
        use_beta_sigmoid=use_beta_sigmoid or None,
        allow_neg_eigval=allow_neg_eigval or None,
        safe_gate=safe_gate or None,
        gate_lower_bound=gate_lower_bound,
        name="kda_bwd",
    )
    return graph, dict(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        cu=cu_t,
        dO=dO_t,
        state0=state0_t,
        dfs=dfs_t,
        a_log=a_log_t,
        dt_bias=dt_bias_t,
        dQ=dQ_t,
        dK=dK_t,
        dV=dV_t,
        dG=dG_t,
        dBeta=dBeta_t,
        dstate0=dstate0_t,
        d_a_log=dA_t,
        d_dt_bias=dDt_t,
        checkpoints=checkpoints_t,
    )
