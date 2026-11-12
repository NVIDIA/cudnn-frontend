# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind XLA buffers to the existing Frost GDN-2 compiled hosts."""

import inspect

import cutlass as ct

from .common import gate_bwd, head_reduce, piece_chain
from .kda_jax import state_chain_view, workspace_views
from .kernel import gdn2_bprop_f16, gdn2_bprop_summary_f16, gdn2_prefill_f16, gdn2_recompute_f16, gdn2_summary_f16
from .kernel import gdn2_chain_backward_f16 as chain_bwd
from .kernel import gdn2_chain_forward_f16 as chain_fwd
from .kernel import gdn2_uncut_backward_f16 as uncut_bwd
from .kernel import gdn2_uncut_forward_f16 as uncut_fwd


def make_launcher(plan, names):
    node = plan.node
    is_bwd = node.node_type.name == "GDN2_BWD"
    q, k, v, g = (node.inputs[n] for n in ("q", "k", "v", "g"))
    total, hq, dk = q.dim
    hk, hv, dv, ho = k.dim[1], v.dim[1], v.dim[2], g.dim[1]
    batch = node.inputs["cu_seqlens"].dim[0] - 1
    types = dict(HALF=ct.Float16, BFLOAT16=ct.BFloat16, FLOAT=ct.Float32)
    io_dtype, gate_dtype = types[q.get_data_type().name], types[g.get_data_type().name]
    state = node.inputs.get("initial_state")
    state_tensor = state if state is not None else node.outputs.get("final_state")
    state_dtype = types[state_tensor.get_data_type().name] if state_tensor is not None else ct.Float32
    chain, saved = plan.chain, plan.has_state_checkpoints
    coarse = is_bwd and plan.coarse_checkpoints
    recompute = is_bwd and plan.needs_recompute
    checkpoint = plan.checkpoint_cadence if is_bwd else plan.checkpoint if saved else 0
    span = plan.recompute_span_tokens if coarse else 0
    flags = dict(
        l2norm=plan.use_qk_l2norm,
        safe_gate=plan.safe_gate,
        log_gate=plan.log_gate,
        gate_scale_log2=plan.gate_lower_bound * gdn2_prefill_f16.LOG2_E,
        beta_sigmoid=plan.use_beta_sigmoid,
        allow_neg_eigval=plan.allow_neg_eigval,
        beta_guard=plan.beta_guard,
        max_active_clusters=plan.num_sm,
        d_k=dk,
    )
    config = dict(b_t=plan.b_t, scale=plan.scale, io_dtype=io_dtype, checkpoint_every_n=plan.b_t if is_bwd else checkpoint)
    if chain:
        config.update(
            unit_chunks=plan.unit_chunks,
            length_rule=plan.length_rule,
            dim_v=dv,
            dim_k=dk,
            chain_rows=piece_chain.chain_rows_per_cta(dv, dk, batch, ho, plan.num_sm),
            has_seed=state is not None,
            pieces=plan.pieces,
            heads_out=ho,
            num_seqs=batch,
        )
    if is_bwd:
        config.update(
            recompute=recompute,
            recompute_orders=plan.order_in_recompute,
            coarse=coarse,
            bwd_orders=saved,
            seed_span_chunks=span // plan.b_t,
            seed_every_n=checkpoint if coarse else 0,
        )
        if chain:
            config.update(
                fused_h_m=plan.fused_h_m,
                series=recompute,
                series_span_chunks=span // plan.b_t,
                has_dseed="d_final_state" in node.inputs,
                summary_cfg=gdn2_summary_f16.build_cfg(io_dtype, gate_dtype, use_initial_state=False, d_v=dv, **flags) if plan.fused_h_m else None,
                transition_cfg=(
                    None
                    if plan.fused_h_m
                    else gdn2_recompute_f16.build_cfg(
                        io_dtype,
                        ct.Float32,
                        gate_dtype,
                        use_initial_state=False,
                        store_final_state=True,
                        enable_checkpoints=False,
                        seed_checkpoints=False,
                        seed_identity=True,
                        v_is_zero=True,
                        d_v=dk,
                        **flags,
                    )
                ),
                series_cfg=(
                    gdn2_recompute_f16.build_cfg(
                        io_dtype,
                        ct.Float32,
                        gate_dtype,
                        use_initial_state=not coarse,
                        store_final_state=False,
                        enable_checkpoints=True,
                        seed_checkpoints=coarse,
                        d_v=dv,
                        **flags,
                    )
                    if recompute
                    else None
                ),
                bwd_summary_cfg=gdn2_bprop_summary_f16.build_cfg(io_dtype, gate_dtype, use_dstate_in=False, use_dstate0=True, d_v=dv, **flags),
                bprop_cfg=gdn2_bprop_f16.build_cfg(
                    io_dtype, gate_dtype, use_dstate_in=True, use_dstate0="d_initial_state" in node.outputs, use_initial_state=True, d_v=dv, **flags
                ),
            )
            host = chain_bwd.chain_backward_host
        else:
            has_state = not coarse and state is not None
            config.update(
                recompute_cfg=(
                    gdn2_recompute_f16.build_cfg(
                        io_dtype,
                        state_dtype if has_state else ct.Float32,
                        gate_dtype,
                        use_initial_state=has_state,
                        state_table=has_state,
                        store_final_state=False,
                        enable_checkpoints=True,
                        seed_checkpoints=coarse,
                        d_v=dv,
                        **flags,
                    )
                    if recompute
                    else None
                ),
                bprop_cfg=gdn2_bprop_f16.build_cfg(
                    io_dtype,
                    gate_dtype,
                    use_dstate_in="d_final_state" in node.inputs,
                    dstate_table="d_final_state" in node.inputs,
                    use_dstate0="d_initial_state" in node.outputs,
                    use_initial_state=state is not None,
                    d_v=dv,
                    **flags,
                ),
            )
            host = uncut_bwd.uncut_backward_host
    else:
        checkpoint_dtype = types[node.outputs["state_checkpoints"].get_data_type().name] if saved else None
        forward_flags = dict(store_final_state=plan.has_final_state, enable_checkpoints=saved, checkpoint_dtype=checkpoint_dtype)
        if chain:
            config.update(
                summary_cfg=gdn2_summary_f16.build_cfg(io_dtype, gate_dtype, use_initial_state=False, d_v=dv, **flags),
                prefill_cfg=gdn2_prefill_f16.build_cfg(io_dtype, ct.Float32, gate_dtype, use_initial_state=True, d_v=dv, **flags, **forward_flags),
            )
            host = chain_fwd.chain_forward_host
        else:
            from .kernel import gdn2_prep_f16, gdn2_prep_prefill_f16

            prefill = gdn2_prep_prefill_f16 if plan.prep else gdn2_prefill_f16
            prep_flags = {key: value for key, value in flags.items() if key != "max_active_clusters"}
            config.update(
                num_sms=plan.num_sm,
                tiles_per_head=plan.tiles_per_head,
                prep=plan.prep,
                prep_cfg=gdn2_prep_f16.build_cfg(io_dtype, gate_dtype, num_sm=plan.num_sm, **prep_flags) if plan.prep else None,
                prefill_cfg=prefill.build_cfg(
                    io_dtype,
                    state_dtype,
                    gate_dtype,
                    use_initial_state=state is not None,
                    state_table=state is not None,
                    d_v=dv // plan.tiles_per_head,
                    tiles_per_head=plan.tiles_per_head,
                    **flags,
                    **forward_flags,
                ),
            )
            host = uncut_fwd.uncut_forward_host
    parameters = inspect.signature(host).parameters
    scalar_types = {name: p.annotation for name, p in parameters.items() if p.annotation in (ct.Int32, ct.Float32)}
    regions = plan.workspace_regions

    def launch(stream, *buffers):
        ports = dict(zip(names, buffers[:-1]))
        scratch = workspace_views(buffers[-1], regions)
        q, k, v, g, beta, w, cu = (ports[n] for n in ("q", "k", "v", "g", "beta", "w", "cu_seqlens"))
        state0 = ports.get("initial_state")
        a, dt = (ports.get("a_log"), ports.get("dt_bias")) if plan.safe_gate else (None, None)
        args = dict(
            config,
            q=q,
            k=k,
            v=v,
            gate=g,
            beta=beta,
            w=w,
            a_log=a,
            dt_bias=dt,
            cu_seqlens=cu,
            checkpoints=scratch["state_checkpoints"] if recompute else ports.get("state_checkpoints"),
            work_items=scratch["work_items"],
            composed=None,
            dcomposed=None,
            has_piece_transitions=False,
            has_piece_state_grads=False,
            stream=stream,
        )
        if is_bwd:
            dq = scratch["dq_ho"] if hq != ho else ports["dQ"]
            dkey = scratch["dk_ho"] if hk != ho else ports["dK"]
            dvalue = scratch["dv_ho"] if hv != ho else ports["dV"]
            args.update(
                do=ports["dO"],
                dq=dq,
                dk=dkey,
                dv=dvalue,
                dw=ports["dW"],
                dgate=ports["dG"],
                dbeta=ports["dBeta"],
                dstate0=ports.get("d_initial_state"),
                gate_main=g if not plan.log_gate and not plan.safe_gate else None,
                beta_recompute=beta if chain or recompute else None,
                beta_bprop=beta if plan.beta_guard and plan.use_beta_sigmoid else None,
                seed_checkpoints=ports.get("state_checkpoints") if coarse else None,
                dstate_scratch=scratch["dstate_scratch"],
            )
        else:
            args.update(o=ports["O"], beta_prefill=beta, seed_indices=None, final_indices=None)
        if chain:
            args.update(
                cu_pieces=scratch["cu_pieces"],
                cu_pieces_main=scratch["cu_pieces"],
                main_rows=scratch["main_rows"],
                summary_rows=scratch["summary_rows"],
                main_count=scratch["main_count"],
                summary_count=scratch["summary_count"],
                work_items_summary=scratch["work_items_summary"],
                scheduler_all=scratch["scheduler_all"],
                summary_words=scratch.get("fused_tensormaps"),
                state_h_summary=scratch.get("state_h"),
                state_h_chain=state_chain_view(scratch.get("state_h")),
                state_m_chain=state_chain_view(scratch["state_m"]),
                state_x_chain=state_chain_view(scratch.get("state_x")),
                seed=state_chain_view(state0),
            )
            if is_bwd:
                args.update(
                    scheduler_summary=scratch["scheduler_summary"],
                    scheduler_recompute=scratch["scheduler_recompute"],
                    scheduler_m=scratch["scheduler_m"],
                    scheduler_series=scratch["scheduler_series"],
                    scheduler_bwd=scratch["scheduler_bwd"],
                    series_items=scratch.get("work_items_recompute"),
                    series_count=scratch.get("work_count_recompute"),
                    recompute_items=(scratch["work_items_recompute"] if coarse else scratch["work_items"]) if recompute else None,
                    recompute_count=(scratch["work_count_recompute"] if coarse else scratch["main_count"]) if recompute else None,
                    recompute_m_words=scratch["recompute_tensormaps_m"],
                    series_words=scratch.get("recompute_tensormaps_series"),
                    bprop_summary_words=scratch["summary_tensormaps"],
                    bprop_words=scratch["bwd_tensormaps"],
                    state_m_main=scratch["state_m"],
                    state_x_series=scratch.get("state_x") if recompute and not coarse else None,
                    state_g=scratch["state_g"],
                    state_g_chain=state_chain_view(scratch["state_g"]),
                    state_dx_end=scratch["state_dx_end"],
                    state_dx_end_chain=state_chain_view(scratch["state_dx_end"]),
                    dseed=state_chain_view(ports.get("d_final_state")),
                )
            else:
                args.update(
                    scheduler_summary=scratch["scheduler_h"],
                    scheduler_prefill=scratch["scheduler_prefill"],
                    prefill_words=scratch["tensormaps"],
                    state_m_summary=scratch["state_m"],
                    state_x_prefill=scratch["state_x"],
                    final_state=ports.get("final_state"),
                )
        else:
            args.update(work_count=scratch["work_count"])
            if is_bwd:
                args.update(
                    state_in=state0 if recompute and not coarse else None,
                    dstate_in=ports.get("d_final_state"),
                    series_items=(scratch["work_items_recompute"] if coarse else scratch["work_items"]) if recompute else None,
                    series_count=(scratch["work_count_recompute"] if coarse else scratch["work_count"]) if recompute else None,
                    scheduler_all=scratch["scheduler_all"],
                    scheduler_all_recompute=scratch["scheduler_all"] if recompute else None,
                    scheduler_all_bprop=scratch["scheduler_all"] if saved else None,
                    scheduler_recompute=scratch["scheduler_recompute"],
                    scheduler_bwd=scratch["scheduler_bwd"],
                    recompute_words=scratch.get("recompute_tensormaps"),
                    bprop_words=scratch["bwd_tensormaps"],
                    dstate_snapshot=scratch.get("dstate_snapshot"),
                )
            else:
                args.update(
                    state_in=state0,
                    state_out=ports.get("final_state"),
                    scheduler=scratch["scheduler"],
                    workspace=scratch["tensormaps"],
                    prep_k_decay=scratch.get("prep_k_decay"),
                    prep_q_decay=scratch.get("prep_q_decay"),
                    prep_t=scratch.get("prep_t"),
                    prep_a=scratch.get("prep_a"),
                    prep_diag=scratch.get("prep_diag"),
                    prep_words=scratch.get("prep_tensormaps"),
                    prep_rows=scratch.get("prep_rows"),
                    prep_row_count=scratch.get("prep_row_count"),
                    state_snapshot=scratch.get("state_snapshot"),
                )
        for name, dtype in scalar_types.items():
            args[name] = dtype(args[name])
        host(**{name: args[name] for name in parameters})
        if is_bwd:
            if plan.safe_gate:
                gate_bwd.channel_gate_bwd_launch(
                    dk,
                    ports["dG"],
                    g,
                    a,
                    dt,
                    scratch.get("gate_part_a"),
                    scratch.get("gate_part_dt"),
                    ports.get("d_a_log"),
                    ports.get("d_dt_bias"),
                    ct.Int32(total),
                    ct.Int32(ho),
                    ct.Int32(-(-total // gate_bwd.GATE_BWD_BLOCKS)),
                    ct.Float32(plan.gate_lower_bound),
                    stream,
                )
            for src, dst, h, dim in ((dq, ports["dQ"], hq, dk), (dkey, ports["dK"], hk, dk), (dvalue, ports["dV"], hv, dv)):
                if h != ho:
                    words = total * h * (dim // 2)
                    head_reduce.launch(
                        src,
                        dst,
                        ct.Int64(words),
                        ct.Int64(dst.stride[0] // 2),
                        ct.Int64(dst.stride[1] // 2),
                        ct.Int32(-(-words // head_reduce.BLOCK)),
                        ct.Int32(h),
                        ct.Int32(ho // h),
                        dim // 2,
                        io_dtype,
                        stream,
                    )

    return launch
