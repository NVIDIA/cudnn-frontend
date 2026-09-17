# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind XLA buffers to the existing Frost KDA compiled hosts."""

import inspect
import math
from types import SimpleNamespace

import cutlass as ct
import cutlass.cute as cute

from .common import gate_bwd, head_reduce, piece_chain, split_k
from .kernel import kda_bprop_f16 as bwd
from .kernel import kda_bprop_summary_f16 as bwd_summary
from .kernel import kda_prefill_f16 as fwd
from .kernel import kda_recompute_f16 as recompute
from .kernel import kda_summary_f16 as summary
from .kernel.kda_chain_backward_f16 import chain_backward_host
from .kernel.kda_chain_forward_f16 import chain_forward_host
from .kernel.kda_warmup_backward_f16 import warmup_backward_host
from .kernel.kda_warmup_forward_f16 import warmup_forward_host


def contiguous_view(pointer, shape):
    return cute.make_tensor(pointer, cute.make_layout(shape, stride=tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))))


def workspace_views(workspace, regions):
    types = dict(int32=ct.Int32, int64=ct.Int64, float32=ct.Float32, float16=ct.Float16, bfloat16=ct.BFloat16)
    return {name: contiguous_view(cute.recast_ptr(workspace.iterator + offset, dtype=types[dtype]), shape) for name, offset, dtype, shape in regions}


def state_chain_view(tensor):
    return None if tensor is None else contiguous_view(tensor.iterator, (1, *tensor.shape[1:]))


def make_launcher(plan, names):
    node = plan.node
    is_bwd = node.node_type.name == "KDA_BWD"
    q, k, v, g = (node.inputs[n] for n in ("q", "k", "v", "g"))
    total, hq, dk = q.dim
    hk, hv, dv = k.dim[1], v.dim[1], v.dim[2]
    ho = g.dim[1]
    batch = node.inputs["cu_seqlens"].dim[0] - 1
    types = dict(HALF=ct.Float16, BFLOAT16=ct.BFloat16, FLOAT=ct.Float32)
    io_dtype, gate_dtype = types[q.get_data_type().name], types[g.get_data_type().name]
    state = node.inputs.get("initial_state")
    state_tensor = state if state is not None else node.outputs.get("final_state")
    state_dtype = types[state_tensor.get_data_type().name] if state_tensor is not None else ct.Float32
    chain, split = plan.chain, plan.split
    checkpoint = plan.checkpoint_cadence if is_bwd else plan.checkpoint
    saved = plan.has_state_checkpoints
    coarse = is_bwd and plan.coarse_checkpoints
    needs_recompute = is_bwd and plan.needs_recompute
    span = plan.recompute_span_tokens if coarse else 0
    safe_gate, log_gate = plan.safe_gate, plan.log_gate
    flags = dict(
        l2norm=plan.use_qk_l2norm,
        safe_gate=safe_gate,
        log_gate=log_gate,
        gate_scale_log2=plan.gate_lower_bound * split_k.RCP_LN2,
        beta_sigmoid=plan.use_beta_sigmoid,
        allow_neg_eigval=plan.allow_neg_eigval,
        max_active_clusters=plan.num_sm,
        d_k=dk,
        d_v=dv,
    )
    config = dict(b_t=16, scale=plan.scale, io_dtype=io_dtype, checkpoint_every_n=16 if is_bwd else checkpoint)
    if is_bwd:
        config.update(
            bprop_cfg=bwd.build_cfg(
                io_dtype,
                gate_dtype,
                use_dstate_in=chain or "d_final_state" in node.inputs,
                use_dstate0="d_initial_state" in node.outputs,
                use_initial_state=chain or state is not None,
                **flags,
            ),
            recompute_cfg=None,
            recompute=needs_recompute,
            recompute_orders=not coarse,
            recompute_order_gen=not coarse and not split,
            coarse=coarse,
            bwd_orders=saved,
            bwd_order_gen=not split,
            seed_span_chunks=span // 16,
            seed_every_n=checkpoint if coarse else 0,
        )
        if needs_recompute:
            config["recompute_cfg"] = recompute.build_cfg(
                io_dtype,
                ct.Float32 if chain or coarse else state_dtype,
                gate_dtype,
                use_initial_state=not coarse and (chain or state is not None),
                store_final_state=False,
                enable_checkpoints=True,
                seed_checkpoints=coarse,
                **flags,
            )
    else:
        config.update(
            order_gen=not split,
            prefill_cfg=fwd.build_cfg(
                io_dtype,
                ct.Float32 if chain else state_dtype,
                gate_dtype,
                use_initial_state=chain or state is not None,
                store_final_state=plan.has_final_state,
                enable_checkpoints=saved,
                **flags,
            ),
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
            summary_cfg=summary.build_cfg(io_dtype, gate_dtype, use_initial_state=False, **flags) if not is_bwd or not saved else None,
        )
        if is_bwd:
            config.update(
                fused_h_m=not saved,
                series=needs_recompute,
                series_cfg=config["recompute_cfg"],
                series_span_chunks=span // 16,
                bwd_summary_cfg=bwd_summary.build_cfg(io_dtype, gate_dtype, use_dstate_in=False, **flags),
                has_dseed="d_final_state" in node.inputs,
                transition_cfg=(
                    recompute.build_cfg(
                        io_dtype,
                        ct.Float32,
                        gate_dtype,
                        use_initial_state=False,
                        store_final_state=True,
                        enable_checkpoints=False,
                        seed_identity=True,
                        v_is_zero=True,
                        **dict(flags, d_v=dk),
                    )
                    if saved
                    else None
                ),
            )
        host = chain_backward_host if is_bwd else chain_forward_host
    else:
        facts = split_k.split_table_facts(
            SimpleNamespace(shape=tuple(g.dim), dtype={ct.Float16: "float16", ct.BFloat16: "bfloat16", ct.Float32: "float32"}[gate_dtype]),
            SimpleNamespace(shape=tuple(node.inputs["cu_seqlens"].dim)),
            split=split,
            n_tiles=plan.n_tiles,
            ideal_chunks=plan.ideal,
            num_sms=plan.num_sm,
            b_t=16,
            log2_threshold=None,
            log_gate=log_gate,
            safe_gate=safe_gate,
            gate_lower_bound=plan.gate_lower_bound if safe_gate else None,
            expand_num=1,
        )
        config.update(facts._asdict())
        config["log2_thresh"] = facts.log2_threshold
        host = warmup_backward_host if is_bwd else warmup_forward_host
    parameters = inspect.signature(host).parameters
    scalar_types = {name: p.annotation for name, p in parameters.items() if p.annotation in (ct.Int32, ct.Float32)}
    regions = plan.workspace_regions

    def launch(stream, *buffers):
        ports = dict(zip(names, buffers[:-1]))
        w = workspace_views(buffers[-1], regions)
        q, k, v, g, beta, cu = (ports[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens"))
        state0 = ports.get("initial_state")
        a, dt = (ports.get("a_log"), ports.get("dt_bias")) if safe_gate else (None, None)
        checkpoints = w["state_checkpoints"] if needs_recompute else ports.get("state_checkpoints")
        args = dict(
            config, q=q, k=k, v=v, gate=g, beta=beta, a_log=a, dt_bias=dt, cu_seqlens=cu, checkpoints=checkpoints, work_items=w["work_items"], stream=stream
        )
        if is_bwd:
            dq = w["dq_ho"] if hq != ho else ports["dQ"]
            dkey = w["dk_ho"] if hk != ho else ports["dK"]
            dvalue = w["dv_ho"] if hv != ho else ports["dV"]
            args.update(
                do=ports["dO"],
                dq=dq,
                dk=dkey,
                dv=dvalue,
                dgate=ports["dG"],
                dbeta=ports["dBeta"],
                dstate0=ports.get("d_initial_state"),
                gate_main=g if not log_gate and not safe_gate else None,
                seed_checkpoints=ports.get("state_checkpoints") if coarse else None,
            )
        if chain:
            args.update(
                cu_pieces=w["cu_pieces"],
                cu_pieces_main=w["cu_pieces"],
                main_rows=w["main_rows"],
                summary_rows=w["summary_rows"],
                main_count=w["main_count"],
                summary_count=w["summary_count"],
                work_items_summary=w["work_items_summary"],
                scheduler_all=w["scheduler_all"],
                summary_words=w.get("fused_tensormaps"),
                state_h_summary=w.get("state_h"),
                state_h_chain=state_chain_view(w.get("state_h")),
                state_m_chain=state_chain_view(w["state_m"]),
                state_x_chain=state_chain_view(w.get("state_x")),
                seed=state_chain_view(state0),
            )
            if is_bwd:
                args.update(
                    scheduler_summary=w["scheduler_summary"],
                    scheduler_recompute=w["scheduler_recompute"],
                    scheduler_m=w["scheduler_m"],
                    scheduler_series=w["scheduler_series"],
                    scheduler_bwd=w["scheduler_bwd"],
                    series_items=w.get("work_items_recompute"),
                    series_count=w.get("work_count_recompute"),
                    recompute_items=(w["work_items_recompute"] if coarse else w["work_items"]) if needs_recompute else None,
                    recompute_count=(w["work_count_recompute"] if coarse else w["main_count"]) if needs_recompute else None,
                    recompute_m_words=w["recompute_tensormaps_m"],
                    series_words=w.get("recompute_tensormaps_series"),
                    bprop_summary_words=w["summary_tensormaps"],
                    bprop_words=w["bwd_tensormaps"],
                    state_m_main=w["state_m"],
                    state_x_series=w.get("state_x") if needs_recompute and not coarse else None,
                    state_g=w["state_g"],
                    state_g_chain=state_chain_view(w["state_g"]),
                    state_dx_end=w["state_dx_end"],
                    state_dx_end_chain=state_chain_view(w["state_dx_end"]),
                    dseed=state_chain_view(ports.get("d_final_state")),
                )
            else:
                args.update(
                    o=ports["O"],
                    scheduler_summary=w["scheduler_h"],
                    scheduler_prefill=w["scheduler_prefill"],
                    prefill_words=w["tensormaps"],
                    state_m_summary=w["state_m"],
                    state_x_prefill=w["state_x"],
                    final_state=ports.get("final_state"),
                    seed_indices=None,
                    final_indices=None,
                )
        else:
            staging = w.get("item_scratch")
            args.update(
                gate_table=g if split else None,
                a_log_table=a,
                dt_bias_table=dt,
                cu_seqlens_table=cu,
                work_items_table=w["work_items"],
                work_count=w["work_count"],
                work_count_table=w["work_count"],
                item_scratch=staging,
                chunk_scratch=w.get("chunk_scratch"),
            )
            if is_bwd:
                args.update(
                    state_in=state0 if needs_recompute and not coarse else None,
                    dstate_in=ports.get("d_final_state"),
                    series_items=(w["work_items_recompute"] if coarse else w["work_items"]) if needs_recompute else None,
                    series_count=(w["work_count_recompute"] if coarse else w["work_count"]) if needs_recompute else None,
                    staging_recompute=staging if needs_recompute and not coarse else None,
                    staging_bprop=staging if saved else None,
                    scheduler_all=w["scheduler_all"],
                    scheduler_all_recompute=w["scheduler_all"] if needs_recompute else None,
                    scheduler_all_bprop=w["scheduler_all"] if saved else None,
                    scheduler_recompute=w["scheduler_recompute"],
                    scheduler_bwd=w["scheduler_bwd"],
                    recompute_words=w.get("recompute_tensormaps"),
                    bprop_words=w["bwd_tensormaps"],
                )
            else:
                args.update(
                    o=ports["O"],
                    state_in=state0,
                    state_out=ports.get("final_state"),
                    seed_indices=None,
                    final_indices=None,
                    staging=staging,
                    scheduler=w["scheduler"],
                    workspace=w["tensormaps"],
                )
        for name, dtype in scalar_types.items():
            args[name] = dtype(args[name])
        host(**{name: args[name] for name in parameters})
        if is_bwd:
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
