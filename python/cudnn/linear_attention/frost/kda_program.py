# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frost KDA launch program shared by native execution and XLA export."""

import math

import cutlass as ct
import cutlass.cute as cute

from .common import split_k, head_reduce, gate_bwd, piece_chain
from .kernel.kda_chain_prologue_f16 import chain_prologue
from .kernel import kda_summary_f16 as summary
from .kernel import kda_bprop_summary_f16 as bwd_summary
from .kernel import kda_prefill_f16 as fwd
from .kernel import kda_bprop_f16 as bwd
from .kernel import kda_recompute_f16 as recompute


def workspace_views(workspace, regions):
    types = dict(
        int32=ct.Int32,
        int64=ct.Int64,
        float32=ct.Float32,
        float16=ct.Float16,
        bfloat16=ct.BFloat16,
    )
    result = {}
    for name, offset, dtype, shape in regions:
        result[name] = cute.make_tensor(
            cute.recast_ptr(workspace.iterator + offset, dtype=types[dtype]),
            cute.make_layout(
                shape,
                stride=tuple(math.prod(shape[i + 1 :]) for i in range(len(shape))),
            ),
        )
    return result


def make_program(plan, names):
    """Snapshot the engine's scheduling and workspace decisions into an exportable program."""
    node = plan.node
    is_bwd = node.node_type.name == "KDA_BWD"
    q, k, v, g = (node.inputs[n] for n in ("q", "k", "v", "g"))
    total, hq, dk = q.dim
    hk, hv, dv = k.dim[1], v.dim[1], v.dim[2]
    ho = g.dim[1]
    batch = node.inputs["cu_seqlens"].dim[0] - 1
    io_dtype = ct.Float16 if q.get_data_type().name == "HALF" else ct.BFloat16
    dtype_map = dict(HALF=ct.Float16, BFLOAT16=ct.BFloat16, FLOAT=ct.Float32)
    gate_dtype = dtype_map[g.get_data_type().name]
    state = node.inputs.get("initial_state")
    state_output = node.outputs.get("final_state")
    state_tensor = state if state is not None else state_output
    state_type = state_tensor.get_data_type() if state_tensor is not None else None
    state_dtype = dtype_map[state_type.name] if state_type is not None else ct.Float32
    regions = plan.workspace_regions
    chain, pieces, unit_chunks, length_rule = plan.chain, plan.pieces, plan.unit_chunks, plan.length_rule
    split, num_sm, ideal = plan.split, plan.num_sm, plan.ideal
    safe_gate, log_gate, lower_bound, scale = plan.safe_gate, plan.log_gate, plan.gate_lower_bound, plan.scale
    gate_scale = lower_bound * split_k.RCP_LN2
    checkpoint = plan.checkpoint_cadence if is_bwd else plan.checkpoint
    saved_checkpoints = plan.has_state_checkpoints
    coarse = is_bwd and plan.coarse_checkpoints
    recompute_span = plan.recompute_span_tokens if coarse else 0
    needs_recompute = is_bwd and plan.needs_recompute
    rows = piece_chain.chain_rows_per_cta(dv, dk, batch, ho, num_sm)
    common = dict(
        l2norm=plan.use_qk_l2norm,
        safe_gate=safe_gate,
        log_gate=log_gate,
        gate_scale_log2=gate_scale,
        beta_sigmoid=plan.use_beta_sigmoid,
        allow_neg_eigval=plan.allow_neg_eigval,
        k_ratio=ho // hk,
        v_ratio=ho // hv,
        n_heads_out=ho,
        max_active_clusters=num_sm,
        d_k=dk,
        d_v=dv,
    )
    if is_bwd:
        cfg = bwd.build_cfg(
            io_dtype,
            gate_dtype,
            use_dstate_in=chain or "d_final_state" in node.inputs,
            use_dstate0="d_initial_state" in node.outputs,
            use_initial_state=chain or state is not None,
            q_ratio=ho // hq,
            **common,
        )
        if needs_recompute:
            recompute_cfg = recompute.build_cfg(
                io_dtype,
                ct.Float32 if chain else state_dtype,
                gate_dtype,
                use_initial_state=not coarse and (chain or state is not None),
                store_final_state=False,
                enable_checkpoints=True,
                seed_checkpoints=coarse,
                **common,
            )
        if chain:
            summary_grad_cfg = bwd_summary.build_cfg(
                io_dtype,
                gate_dtype,
                use_dstate_in=False,
                q_ratio=ho // hq,
                **{key: value for key, value in common.items() if key != "v_ratio"},
            )
            if saved_checkpoints:
                transition_cfg = recompute.build_cfg(
                    io_dtype,
                    ct.Float32,
                    gate_dtype,
                    use_initial_state=False,
                    store_final_state=True,
                    enable_checkpoints=False,
                    seed_identity=True,
                    v_is_zero=True,
                    **dict(common, d_v=dk, v_ratio=ho // hk),
                )
    else:
        cfg = fwd.build_cfg(
            io_dtype,
            ct.Float32 if chain else state_dtype,
            gate_dtype,
            use_initial_state=chain or state is not None,
            store_final_state=plan.has_final_state,
            enable_checkpoints=saved_checkpoints,
            q_ratio=ho // hq,
            **common,
        )
    if chain and (not is_bwd or not saved_checkpoints):
        summary_cfg = summary.build_cfg(io_dtype, gate_dtype, use_initial_state=False, **common)
    scan_rows, scan_blocks, scan_ctas = split_k.scan_geometry(total, batch, 16, ho, dk, num_sm)

    def launch(stream, *buffers):
        ports = dict(zip(names, buffers[:-1]))
        w = workspace_views(buffers[-1], regions)
        q, k, v, g, beta, cu = (ports[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens"))
        state0 = ports.get("initial_state")
        a, dt = (ports.get("a_log"), ports.get("dt_bias")) if safe_gate else (None, None)
        items, count = w["work_items"], w["main_count" if chain else "work_count"]
        sched = w["scheduler_all"] if chain or is_bwd else w["scheduler"]
        staging = w.get("item_scratch")
        checkpoints = ports.get("state_checkpoints")
        if needs_recompute:
            checkpoints = w["state_checkpoints"]
        if is_bwd:
            dq = w["dq_ho"] if hq != ho else ports["dQ"]
            dkey = w["dk_ho"] if hk != ho else ports["dK"]
            dvalue = w["dv_ho"] if hv != ho else ports["dV"]
        if chain:
            chain_prologue(
                pieces,
                unit_chunks,
                16,
                length_rule,
                ho,
                ct.Int32(recompute_span // 16),
                ct.Int32(16 if is_bwd else checkpoint),
                cu,
                w["cu_pieces"],
                w["main_rows"],
                w["summary_rows"],
                count,
                w["summary_count"],
                items,
                w["work_items_summary"],
                sched,
                w.get("work_items_recompute"),
                w.get("work_count_recompute"),
                w.get("fused_tensormaps"),
                None,
                w.get("recompute_tensormaps_m"),
                w.get("recompute_tensormaps_series"),
                w.get("tensormaps"),
                w.get("summary_tensormaps"),
                w.get("bwd_tensormaps"),
                q,
                k,
                v,
                g,
                ports.get("O"),
                ports.get("dO"),
                checkpoints,
                dq if is_bwd else None,
                dkey if is_bwd else None,
                dvalue if is_bwd else None,
                ports.get("dG"),
                stream,
            )
            cu = w["cu_pieces"]
            if not is_bwd or not saved_checkpoints:
                summary.host(
                    summary_cfg,
                    k,
                    v,
                    g,
                    a,
                    dt,
                    beta,
                    cu,
                    None,
                    w["state_h"],
                    w["state_m"],
                    w["work_items_summary"],
                    w["summary_count"],
                    w["scheduler_recompute" if is_bwd else "scheduler_h"],
                    w["fused_tensormaps"],
                    stream,
                )
                piece_chain.launch_state_chain(
                    ho,
                    dv,
                    dk,
                    rows,
                    pieces,
                    False,
                    state0 is not None,
                    False,
                    False,
                    ct.Int32(batch),
                    w["state_h"],
                    w["state_m"],
                    w["state_x"],
                    state0,
                    None,
                    None,
                    w["main_rows"],
                    stream,
                )
                state0 = w["state_x"]
            else:
                recompute.host(
                    transition_cfg,
                    k,
                    k,
                    g,
                    a,
                    dt,
                    beta,
                    cu,
                    None,
                    w["state_m"],
                    None,
                    w["work_items_summary"],
                    w["summary_count"],
                    w["scheduler_m"],
                    w["recompute_tensormaps_m"],
                    ct.Int32(0),
                    ct.Int32(0),
                    stream,
                )
            if is_bwd:
                bwd_summary.host(
                    summary_grad_cfg,
                    a,
                    dt,
                    beta,
                    cu,
                    w["state_g"],
                    None,
                    w["work_items_summary"],
                    w["summary_count"],
                    w["scheduler_summary"],
                    w["summary_tensormaps"],
                    ct.Float32(scale),
                    stream,
                )
                piece_chain.launch_state_chain(
                    ho,
                    dv,
                    dk,
                    rows,
                    pieces,
                    True,
                    ports.get("d_final_state") is not None,
                    False,
                    False,
                    ct.Int32(batch),
                    w["state_g"],
                    w["state_m"],
                    w["state_dx_end"],
                    ports.get("d_final_state"),
                    None,
                    None,
                    w["main_rows"],
                    stream,
                )
        elif split:
            split_k.launch(
                split=True,
                b_t=16,
                scan_rows=scan_rows,
                log_gate=log_gate,
                safe_gate=safe_gate,
                gate_channels=dk,
                overhead_chunks=max(1, split_k.OVERHEAD_TOKENS // 16),
                expand_num=1,
                warmup_cap=split_k.warmup_cap_chunks(dk, 1),
                full_scan=False,
                n_heads_out=ho,
                num_sms=num_sm,
                n_tiles=ct.Int32(batch * ho),
                ideal_chunks=ct.Int32(ideal),
                batch_size=ct.Int32(batch),
                log2_thresh=ct.Float32(split_k.DEFAULT_LOG2_THRESHOLD),
                gate_scale_log2=ct.Float32(gate_scale if safe_gate else 0.0),
                mGate=g,
                mALog=a,
                mDtBias=dt,
                mCuSeqlens=cu,
                mChunkVals=w["chunk_scratch"],
                mStaging=staging,
                mWorkItems=items,
                mCount=count,
                mScheduler=sched,
                n_scan_ctas=ct.Int32(scan_ctas),
                n_scan_blocks=ct.Int32(scan_blocks),
                n_walk_ctas=ct.Int32(batch * ho),
                stream=stream,
            )
        if not is_bwd:
            if not chain:
                fwd.prologue(
                    io_dtype=io_dtype,
                    b_t=16,
                    num_ctas=num_sm,
                    order_gen=not split,
                    q=q,
                    k=k,
                    v=v,
                    gate=g,
                    o=ports["O"],
                    state_checkpoints=checkpoints,
                    cu_seqlens=cu,
                    work_item_staging=staging,
                    work_count=count,
                    work_items=items,
                    scheduler_counter=sched,
                    tensormap_workspace=w["tensormaps"],
                    checkpoint_every_n=ct.Int32(checkpoint),
                    stream=stream,
                )
            fwd.host(
                cfg,
                q,
                k,
                v,
                g,
                a,
                dt,
                beta,
                cu,
                state0,
                ports["O"],
                ports.get("final_state"),
                items,
                count,
                w["scheduler_prefill"] if chain else sched,
                w["tensormaps"],
                ct.Int32(checkpoint),
                ct.Float32(scale),
                stream,
            )
        else:
            if needs_recompute:
                series_items = w["work_items_recompute"] if coarse else items
                series_count = w["work_count_recompute"] if coarse else count
                series_maps = w["recompute_tensormaps_series" if chain else "recompute_tensormaps"]
                if not chain:
                    recompute.prologue(
                        io_dtype=io_dtype,
                        b_t=16,
                        run_order=not coarse,
                        order_gen=not split,
                        gen_intervals=coarse,
                        k=k,
                        v=v,
                        gate=g,
                        state_checkpoints=checkpoints,
                        cu_seqlens=cu,
                        work_item_staging=None if coarse else staging,
                        work_count=series_count,
                        work_items=series_items,
                        scheduler_all=sched,
                        tensormap_workspace=series_maps,
                        checkpoint_every_n=ct.Int32(16),
                        seed_span_chunks=ct.Int32(recompute_span // 16),
                        stream=stream,
                    )
                recompute.host(
                    recompute_cfg,
                    k,
                    v,
                    g,
                    a,
                    dt,
                    beta,
                    cu,
                    None if coarse else state0,
                    None,
                    ports["state_checkpoints"] if coarse else None,
                    series_items,
                    series_count,
                    w["scheduler_series" if chain else "scheduler_recompute"],
                    series_maps,
                    ct.Int32(16),
                    ct.Int32(checkpoint if coarse else 0),
                    stream,
                )
            if not chain:
                bwd.prologue(
                    io_dtype=io_dtype,
                    b_t=16,
                    run_order=saved_checkpoints,
                    order_gen=not split,
                    q=q,
                    k=k,
                    v=v,
                    gate=g,
                    do=ports["dO"],
                    dq=dq,
                    dk=dkey,
                    dv=dvalue,
                    dgate=ports["dG"],
                    state_checkpoints=checkpoints,
                    cu_seqlens=cu,
                    work_item_staging=staging if saved_checkpoints else None,
                    work_count=count,
                    work_items=items,
                    scheduler_all=sched if saved_checkpoints else None,
                    tensormap_workspace=w["bwd_tensormaps"],
                    stream=stream,
                )
            bwd.host(
                cfg,
                a,
                dt,
                beta,
                g if not log_gate and not safe_gate else None,
                checkpoints,
                ports["dG"],
                ports["dBeta"],
                cu,
                ports.get("d_initial_state"),
                w["state_dx_end"] if chain else ports.get("d_final_state"),
                items,
                count,
                w["scheduler_bwd"],
                w["bwd_tensormaps"],
                ct.Float32(scale),
                stream,
            )
            if safe_gate:
                gate_bwd.channel_gate_bwd_launch(
                    dk,
                    ports["dG"],
                    g,
                    a,
                    dt,
                    w.get("gate_part_a"),
                    w.get("gate_part_dt"),
                    ports.get("d_a_log"),
                    ports.get("d_dt_bias"),
                    ct.Int32(total),
                    ct.Int32(ho),
                    ct.Int32(-(-total // gate_bwd.GATE_BWD_BLOCKS)),
                    ct.Float32(lower_bound),
                    stream,
                )
            for src, dst, h, dim in ((dq, ports["dQ"], hq, dk), (dkey, ports["dK"], hk, dk), (dvalue, ports["dV"], hv, dv)):
                if h != ho:
                    words = total * h * (dim // 2)
                    head_reduce.launch(
                        src,
                        dst,
                        ct.Int64(words),
                        ct.Int64(h * dim // 2),
                        ct.Int64(dim // 2),
                        ct.Int32(-(-words // head_reduce.BLOCK)),
                        h,
                        ho // h,
                        dim // 2,
                        io_dtype,
                        stream,
                    )

    return launch
