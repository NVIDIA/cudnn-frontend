# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose existing Frost launchers into one exportable CuTe host program."""

import math

import cutlass as ct
import cutlass.cute as cute

from .common import split_k, head_reduce, gate_bwd
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


def make_launcher(plan, input_names, output_names):
    """Snapshot engine metadata; no graph, buffers or runtime executables retained."""
    node = plan.node
    is_bwd = node.node_type.name == "KDA_BWD"
    q, k, v, g = (node.inputs[n] for n in ("q", "k", "v", "g"))
    total, hq, dk = q.dim
    hv, dv = v.dim[1:]
    ho = g.dim[1]
    batch = node.inputs["cu_seqlens"].dim[0] - 1
    io_dtype = ct.Float16 if q.get_data_type().name == "HALF" else ct.BFloat16
    dtype_map = dict(HALF=ct.Float16, BFLOAT16=ct.BFloat16, FLOAT=ct.Float32)
    gate_dtype = dtype_map[g.get_data_type().name]
    state = node.inputs.get("initial_state")
    state_dtype = dtype_map[state.get_data_type().name] if state is not None else ct.Float32
    regions = plan.workspace_regions
    split, num_sm, ideal = plan.split, plan.num_sm, plan.ideal
    safe_gate, lower_bound, scale = plan.safe_gate, plan.gate_lower_bound, plan.scale
    gate_scale = lower_bound * split_k.RCP_LN2 if safe_gate else 0.0
    checkpoint = plan.checkpoint_cadence if is_bwd else plan.checkpoint
    saved_checkpoints = plan.has_state_checkpoints
    common = dict(
        l2norm=plan.use_qk_l2norm,
        safe_gate=safe_gate,
        gate_scale_log2=gate_scale,
        beta_sigmoid=plan.use_beta_sigmoid,
        allow_neg_eigval=plan.allow_neg_eigval,
        k_ratio=ho // k.dim[1],
        v_ratio=ho // hv,
        n_heads_out=ho,
        max_active_clusters=num_sm,
        dynamic_scheduling=True,
        d_k=dk,
        d_v=dv,
    )
    if is_bwd:
        cfg = bwd.build_cfg(
            io_dtype,
            gate_dtype,
            use_dstate_in="d_final_state" in node.inputs,
            use_dstate0="d_initial_state" in node.outputs,
            use_initial_state=state is not None,
            q_ratio=ho // hq,
            **common,
        )
        recompute_cfg = recompute.build_cfg(
            io_dtype,
            state_dtype,
            gate_dtype,
            use_initial_state=state is not None,
            store_final_state=False,
            enable_checkpoints=True,
            **common,
        )
    else:
        cfg = fwd.build_cfg(
            io_dtype,
            state_dtype,
            gate_dtype,
            use_initial_state=state is not None,
            store_final_state=plan.has_final_state,
            enable_checkpoints=saved_checkpoints,
            q_ratio=ho // hq,
            **common,
        )
    scan_rows, scan_blocks, scan_ctas = split_k.scan_geometry(total, batch, 16, ho, dk, num_sm)

    def launch(stream, *buffers):
        ports = dict(zip(input_names + output_names, buffers[:-1]))
        w = workspace_views(buffers[-1], regions)
        q, k, v, g, beta, cu = (ports[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens"))
        state0 = ports.get("initial_state")
        a, dt = ports.get("a_log"), ports.get("dt_bias")
        items, count, sched = w["work_items"], w["work_count"], w["scheduler_all"]
        staging = w.get("item_scratch")
        if split:
            split_k.launch(
                split=True,
                b_t=16,
                scan_rows=scan_rows,
                log_gate=True,
                safe_gate=safe_gate,
                gate_channels=dk,
                overhead_chunks=max(1, split_k.OVERHEAD_TOKENS // 16),
                expand_num=1,
                has_scheduler=True,
                n_heads_out=ho,
                num_sms=num_sm,
                n_tiles=ct.Int32(batch * ho),
                ideal_chunks=ct.Int32(ideal),
                batch_size=ct.Int32(batch),
                log2_thresh=ct.Float32(split_k.DEFAULT_LOG2_THRESHOLD),
                gate_scale_log2=ct.Float32(gate_scale),
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
            fwd.prologue(
                io_dtype=io_dtype,
                b_t=16,
                num_ctas=num_sm,
                order_gen=not split,
                has_scheduler=True,
                q=q,
                k=k,
                v=v,
                gate=g,
                o=ports["O"],
                state_checkpoints=ports.get("state_checkpoints"),
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
                sched,
                w["tensormaps"],
                ct.Int32(checkpoint),
                ct.Float32(scale),
                stream,
            )
        else:
            checkpoints = ports.get("state_checkpoints")
            if not saved_checkpoints:
                checkpoints = w["state_checkpoints"]
                recompute.prologue(
                    io_dtype=io_dtype,
                    b_t=16,
                    run_order=True,
                    order_gen=not split,
                    has_scheduler=True,
                    gen_intervals=False,
                    k=k,
                    v=v,
                    gate=g,
                    state_checkpoints=checkpoints,
                    cu_seqlens=cu,
                    work_item_staging=staging,
                    work_count=count,
                    work_items=items,
                    scheduler_all=sched,
                    tensormap_workspace=w["recompute_tensormaps"],
                    checkpoint_every_n=ct.Int32(16),
                    seed_span_chunks=ct.Int32(0),
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
                    state0,
                    None,
                    None,
                    items,
                    count,
                    w["scheduler_recompute"],
                    w["recompute_tensormaps"],
                    ct.Int32(16),
                    ct.Int32(0),
                    stream,
                )
            dq, dkey, dvalue = (w.get(n + "_ho", ports[p]) for n, p in (("dq", "dQ"), ("dk", "dK"), ("dv", "dV")))
            bwd.prologue(
                io_dtype=io_dtype,
                b_t=16,
                run_order=saved_checkpoints,
                order_gen=not split,
                has_scheduler=saved_checkpoints,
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
                checkpoints,
                ports["dG"],
                ports["dBeta"],
                cu,
                ports.get("d_initial_state"),
                ports.get("d_final_state"),
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
            for src, dst, h, dim in (
                (dq, ports["dQ"], hq, dk),
                (dkey, ports["dK"], k.shape[1], dk),
                (dvalue, ports["dV"], hv, dv),
            ):
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
