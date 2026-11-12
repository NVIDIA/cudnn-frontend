# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind XLA buffers to the existing Frost GDN and GDP compiled hosts."""

import inspect

import cutlass as ct

from .common import expand, gate_bwd, head_reduce, l2norm, piece_chain, split_k
from .kda_jax import contiguous_view, state_chain_view, workspace_views
from .kernel import gdn_bprop_summary_f16, gdn_prefill_f16, gdn_recompute_f16, gdn_summary_f16, gdn_tinv_f16
from .kernel import gdn_chain_backward_f16 as chain_bwd
from .kernel import gdn_chain_forward_f16 as chain_fwd
from .kernel import gdn_warmup_backward_f16 as warmup_bwd
from .kernel import gdn_warmup_forward_f16 as warmup_fwd


def make_launcher(plan, names):
    node = plan.node
    is_bwd = node.node_type.name in ("GDN_BWD", "GDP_BWD")
    q, k, v, g = (node.inputs[n] for n in ("q", "k", "v", "g"))
    rows, hq, dk = q.dim
    hk, hv, dv, ho = k.dim[-2], v.dim[-2], v.dim[-1], g.dim[1]
    n = plan.expand_num
    total = rows * n
    batch = node.inputs["cu_seqlens"].dim[0] - 1
    types = dict(HALF=ct.Float16, BFLOAT16=ct.BFloat16, FLOAT=ct.Float32)
    io_dtype = types[q.get_data_type().name]
    state = node.inputs.get("initial_state")
    state_tensor = state if state is not None else node.outputs.get("final_state")
    state_dtype = types[state_tensor.get_data_type().name] if state_tensor is not None else ct.Float32
    chain, saved = plan.chain, plan.has_state_checkpoints
    coarse = is_bwd and plan.coarse_checkpoints
    recompute = is_bwd and plan.needs_recompute
    pack = is_bwd and plan.pack_bwd_needed
    pack_q = pack and plan.pack_expands_q
    compact = is_bwd and plan.compact_qdo
    checkpoint = plan.checkpoint_cadence if is_bwd else plan.checkpoint if saved else 0
    span = plan.recompute_span_tokens if coarse else 0
    flags = dict(max_active_clusters=plan.num_sm, log_gate=plan.log_gate, safe_gate=plan.safe_gate, d_k=dk, expand_num=n)
    beta_flags = dict(beta_sigmoid=plan.use_beta_sigmoid, allow_neg_eigval=plan.allow_neg_eigval)
    config = dict(b_t=plan.b_t, expand_num=n, k_slots=1, v_slots=1, scale=plan.scale, checkpoint_every_n=plan.b_t if is_bwd else checkpoint, io_dtype=io_dtype)
    config["tinv_cfg"] = (
        gdn_tinv_f16.build_cfg(io_dtype, num_sm=plan.num_sm, log_gate=plan.log_gate, safe_gate=plan.safe_gate, d_k=dk, expand_num=n, **beta_flags)
        if plan.tinv_pass
        else None
    )
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
    else:
        config.update(
            split=plan.split,
            log_gate=plan.log_gate,
            safe_gate=plan.safe_gate,
            overhead_chunks=max(1, split_k.OVERHEAD_TOKENS // plan.b_t),
            warmup_cap=split_k.warmup_cap_chunks(n),
            n_heads_out=ho,
            num_sms=plan.num_sm,
            n_tiles=plan.n_tiles,
            ideal_chunks=plan.ideal or 0,
            batch_size=batch,
            log2_thresh=split_k.DEFAULT_LOG2_THRESHOLD,
            n_scan_ctas=split_k.chunk_scratch_rows(total, batch, plan.b_t),
            n_walk_ctas=batch * ho,
        )
    if is_bwd:
        bprop = plan.bwd
        bprop_flags = dict(
            use_initial_state=chain or state is not None,
            use_dstate_in=chain or "d_final_state" in node.inputs,
            use_dstate0="d_initial_state" in node.outputs,
            fused_l2norm=plan.use_qk_l2norm,
            d_v=dv,
            **flags,
            **beta_flags,
        )
        if not compact:
            bprop_flags["tinv_source"] = "gmem" if plan.tinv_pass else "compute"
        config.update(
            bprop=bprop,
            compact_qdo=compact,
            recompute=recompute,
            recompute_orders=plan.recompute_orders,
            recompute_order_gen=not (plan.recompute_orders and plan.split),
            coarse=coarse,
            bwd_orders=plan.bwd_orders,
            bwd_order_gen=plan.bwd_orders and not plan.split,
            tinv_pass=plan.tinv_pass,
            seed_span_chunks=span // plan.b_t,
            seed_every_n=checkpoint if coarse else 0,
            bprop_cfg=bprop.build_cfg(io_dtype, **bprop_flags),
        )
        if chain:
            config.update(
                fused_h_m=plan.fused_h_m,
                series=recompute,
                series_span_chunks=span // plan.b_t,
                summary_q_step=plan.summary_q_step,
                summary_cfg=gdn_summary_f16.build_cfg(io_dtype, ct.Float32, use_initial_state=False, d_v=dv, **flags) if plan.fused_h_m else None,
                transition_cfg=(
                    None
                    if plan.fused_h_m
                    else gdn_recompute_f16.build_cfg(
                        io_dtype,
                        ct.Float32,
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
                    gdn_recompute_f16.build_cfg(
                        io_dtype,
                        ct.Float32,
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
            )
            bwd_summary_cfg = gdn_bprop_summary_f16.build_cfg(io_dtype, use_dstate_in=False, d_v=dv, **flags)
            hosts = (chain_bwd.chain_backward_head_host, chain_bwd.chain_backward_tail_host)
        else:
            has_state = recompute and not coarse and state is not None
            config["recompute_cfg"] = (
                gdn_recompute_f16.build_cfg(
                    io_dtype,
                    state_dtype if has_state else ct.Float32,
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
            )
            hosts = (warmup_bwd.warmup_backward_host, bprop.host)
    else:
        checkpoint_dtype = types[node.outputs["state_checkpoints"].get_data_type().name] if saved else None
        forward_flags = dict(store_final_state=plan.has_final_state, enable_checkpoints=saved, checkpoint_dtype=checkpoint_dtype)
        if chain:
            config.update(
                summary_cfg=gdn_summary_f16.build_cfg(io_dtype, ct.Float32, use_initial_state=False, d_v=dv, **flags),
                prefill_cfg=gdn_prefill_f16.build_cfg(io_dtype, ct.Float32, use_initial_state=True, tinv_source="gmem", d_v=dv, **flags, **forward_flags),
            )
            hosts = (chain_fwd.chain_forward_host,)
        else:
            config.update(
                order_gen=not plan.split,
                tiles_per_head=plan.tiles_per_head,
                prefill_cfg=gdn_prefill_f16.build_cfg(
                    io_dtype,
                    state_dtype,
                    use_initial_state=state is not None,
                    state_table=state is not None,
                    tinv_source="compute",
                    d_v=dv // plan.tiles_per_head,
                    tiles_per_head=plan.tiles_per_head,
                    **flags,
                    **beta_flags,
                    **forward_flags,
                ),
            )
            hosts = (warmup_fwd.warmup_forward_host,)
    stages = tuple((host, inspect.signature(host).parameters) for host in hosts)
    regions = plan.workspace_regions

    def launch(stream, *buffers):
        ports = dict(zip(names, buffers[:-1]))
        scratch = workspace_views(buffers[-1], regions)
        q, g, cu = (ports[name] for name in ("q", "g", "cu_seqlens"))
        # GDP token/sub-token axes share one contiguous timeline in XLA buffers.
        k = contiguous_view(ports["k"].iterator, (total, hk, dk))
        v = contiguous_view(ports["v"].iterator, (total, hv, dv))
        beta = contiguous_view(ports["beta"].iterator, (total, ho))
        state0 = ports.get("initial_state")
        a, dt = (ports.get("a_log"), ports.get("dt_bias")) if plan.safe_gate else (None, None)
        inv_q, inv_k = (scratch.get("inv_q"), scratch.get("inv_k"))
        if is_bwd:
            do = ports["dO"]
            q_compact, do_compact = q, do
            if pack:
                chunks = total * ho * dv // 8 + (total * hq * dk // 8 if pack_q else 0)
                expand.pack_bwd_launch(
                    pack_q,
                    n,
                    ct.Int32(hq if pack_q else 1),
                    dk // 2 if pack_q else 4,
                    ct.Int32(ho),
                    dv // 2,
                    q if pack_q else do,
                    scratch["q_x"] if pack_q else scratch["do_x"],
                    do,
                    scratch["do_x"],
                    ct.Int32(-(-chunks // expand.BLOCK)),
                    stream,
                )
                do = scratch["do_x"]
                if pack_q:
                    q = scratch["q_x"]
        if plan.use_qk_l2norm:
            normalize_expand = n if pack else 1
            q_normalize_rows = total * hq if pack else rows * hq
            normalize_rows = q_normalize_rows + total * hk
            rows_per_cta = (l2norm.THREADS_PER_CTA // l2norm.FWD_LANES) * l2norm.FWD_ROWS_PER_GROUP
            l2norm.l2norm_qk_launch(
                q,
                k,
                scratch["q_n"],
                scratch["k_n"],
                inv_q,
                inv_k,
                ct.Int32(q_normalize_rows),
                ct.Int32(normalize_rows),
                ct.Int32(hq),
                ct.Int32(hk),
                ct.Int32(-(-normalize_rows // rows_per_cta)),
                dk,
                normalize_expand,
                n - 1 if pack else 0,
                pack,
                1,
                1,
                stream,
            )
            q, k = scratch["q_n"], scratch["k_n"]
        args = dict(
            config,
            q=q,
            k=k,
            v=v,
            gate=g,
            beta=beta,
            a_log=a,
            dt_bias=dt,
            cu_seqlens=cu,
            checkpoints=scratch["state_checkpoints"] if recompute else ports.get("state_checkpoints"),
            work_items=scratch["work_items"],
            composed=None,
            dcomposed=None,
            has_chunk_factors=False,
            has_piece_transitions=False,
            has_piece_state_grads=False,
            tinv=scratch.get("tinv"),
            tinv_words=scratch.get("tinv_tensormaps"),
            tinv_rows=scratch.get("tinv_rows"),
            tinv_row_count=scratch.get("tinv_row_count"),
            stream=stream,
        )
        if is_bwd:
            dq_dst = scratch["dq_x"] if pack else ports["dQ"]
            dk_dst = contiguous_view(ports["dK"].iterator, (total, hk, dk))
            dv_dst = contiguous_view(ports["dV"].iterator, (total, hv, dv))
            dq = scratch["dq_ho"] if hq != ho else dq_dst
            dkey = scratch["dk_ho"] if hk != ho else dk_dst
            dvalue = scratch["dv_ho"] if hv != ho else dv_dst
            args.update(
                do=do,
                do_=do,
                dq=dq,
                dk=dkey,
                dv=dvalue,
                dgate=ports["dG"],
                dbeta=contiguous_view(ports["dBeta"].iterator, (total, ho)),
                dstate0=ports.get("d_initial_state"),
                dstate_in=ports.get("d_final_state"),
                inv_q=inv_q,
                inv_k=inv_k,
                seed_checkpoints=ports.get("state_checkpoints") if coarse else None,
            )
        else:
            args.update(o=ports["O"], seed_indices=None, final_indices=None)
        if chain:
            args.update(
                cu_pieces=scratch["cu_pieces"],
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
                    summary_q=q_compact if pack_q else q,
                    summary_do=do_compact,
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
                    bprop_words=scratch["tensormaps"],
                    state_m_main=scratch["state_m"],
                    state_x_series=scratch.get("state_x") if recompute and not coarse else None,
                    state_g=scratch["state_g"],
                    state_dx_end=scratch["state_dx_end"],
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
            args.update(
                gate_table=g if plan.split else None,
                a_log_table=a,
                dt_bias_table=dt,
                cu_seqlens_table=cu,
                work_count=scratch["work_count"],
                work_items_table=scratch["work_items"],
                work_count_table=scratch["work_count"],
                item_scratch=scratch.get("item_scratch"),
                chunk_scratch=scratch.get("chunk_scratch"),
            )
            if is_bwd:
                args.update(
                    state_in=state0 if recompute and not coarse else None,
                    cu_seqlens_bprop=cu,
                    series_items=(scratch["work_items_recompute"] if coarse else scratch["work_items"]) if recompute else None,
                    series_count=(scratch["work_count_recompute"] if coarse else scratch["work_count"]) if recompute else None,
                    staging_recompute=scratch.get("item_scratch") if recompute and plan.recompute_orders else None,
                    staging_bprop=scratch.get("item_scratch") if plan.bwd_orders else None,
                    scheduler_all=scratch["scheduler_all"],
                    scheduler_all_recompute=scratch["scheduler_all"] if recompute and (plan.recompute_orders or coarse) else None,
                    scheduler_all_bprop=scratch["scheduler_all"] if plan.bwd_orders else None,
                    scheduler_recompute=scratch["scheduler_recompute"],
                    scheduler_counter=scratch["scheduler_bwd"],
                    recompute_words=scratch.get("recompute_tensormaps"),
                    bprop_words=scratch["tensormaps"],
                    tensormap_workspace=scratch["tensormaps"],
                    cfg=config["bprop_cfg"],
                )
            else:
                args.update(
                    state_in=state0,
                    state_out=ports.get("final_state"),
                    staging=scratch.get("item_scratch"),
                    scheduler=scratch["scheduler"],
                    workspace=scratch["tensormaps"],
                )
        for index, (host, parameters) in enumerate(stages):
            if is_bwd and chain and index == 1:
                args.update(summary_cfg=bwd_summary_cfg, has_seed="d_final_state" in node.inputs, seed=state_chain_view(ports.get("d_final_state")))
            host_args = {name: args[name] for name in parameters}
            for name, parameter in parameters.items():
                if parameter.annotation in (ct.Int32, ct.Float32):
                    host_args[name] = parameter.annotation(host_args[name])
            host(**host_args)
        if is_bwd:
            if plan.safe_gate:
                blocks = gate_bwd.scalar_gate_blocks(rows)
                gate_bwd.scalar_gate_bwd_launch(
                    ports["dG"],
                    g,
                    a,
                    dt,
                    scratch.get("gate_part_a"),
                    scratch.get("gate_part_dt"),
                    ports.get("d_a_log"),
                    ports.get("d_dt_bias"),
                    ct.Int32(rows),
                    ct.Int32(ho),
                    ct.Int32(-(-rows // blocks)),
                    ct.Int32(blocks),
                    ct.Int32(-(-ho // gate_bwd.SCALAR_HEAD_TILE)),
                    stream,
                )
            dq_rows = rows if compact else total
            for src, dst, h, dim, count in ((dq, dq_dst, hq, dk, dq_rows), (dkey, dk_dst, hk, dk, total), (dvalue, dv_dst, hv, dv, total)):
                if h != ho:
                    words = count * h * (dim // 2)
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
            if pack:
                chunks = rows * hq * dk // 8
                expand.gather_dq_launch(n, ct.Int32(hq), dk // 2, dq_dst, ports["dQ"], ct.Int32(-(-chunks // expand.BLOCK)), stream)

    return launch
