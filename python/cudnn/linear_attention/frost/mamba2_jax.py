# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind XLA buffers to the existing Frost Mamba2 compiled hosts."""

import inspect
import math

import cutlass as ct
import cutlass.cute as cute

from .common import gate_bwd, head_reduce, split_k
from .kda_jax import contiguous_view, workspace_views
from .kernel import mamba2_bprop_f16, mamba2_prefill_f16, mamba2_recompute_f16
from .kernel import mamba2_chain_backward_f16 as chain_bwd
from .kernel import mamba2_chain_forward_f16 as chain_fwd
from .kernel import mamba2_warmup_backward_f16 as warmup_bwd
from .kernel import mamba2_warmup_forward_f16 as warmup_fwd


def make_launcher(plan, names):
    node = plan.node
    is_bwd = node.node_type.name == "MAMBA2_BWD"
    q, k, v, g = (node.inputs[name] for name in ("q", "k", "v", "g"))
    total, hq, dk = q.dim
    hk, ho, dv = k.dim[1], g.dim[1], v.dim[2]
    batch = node.inputs["cu_seqlens"].dim[0] - 1
    types = dict(HALF=ct.Float16, BFLOAT16=ct.BFloat16, FLOAT=ct.Float32)
    io_dtype = types[q.get_data_type().name]

    def dtype(tensor):
        return types[tensor.get_data_type().name] if tensor is not None else ct.Float32

    state = node.inputs.get("initial_state")
    state_output = node.outputs.get("final_state")
    state_dtype = dtype(state if state is not None else state_output)
    chain = plan.chain
    checkpoint = plan.checkpoint_cadence if is_bwd else plan.checkpoint
    coarse = is_bwd and plan.coarse_checkpoints
    regenerate = is_bwd and not plan.reuse_series
    has_d, has_z = "d" in node.inputs, "z" in node.inputs
    d_per_channel = has_d and len(node.inputs["d"].dim) == 2
    flags = dict(dt_softplus=plan.dt_softplus, max_active_clusters=plan.num_sm, d_k=dk, d_v=dv)
    features = dict(has_d=has_d, d_per_channel=d_per_channel)
    config = dict(b_t=plan.b_t, dt_min=plan.dt_min, dt_max=plan.dt_max)
    if chain:
        config.update(
            unit_chunks=plan.unit_chunks,
            length_rule=plan.length_rule,
            dim_v=dv,
            dim_k=dk,
            chain_rows=plan.chain_rows,
            has_seed=state is not None,
            pieces=plan.pieces,
            heads_out=ho,
            num_seqs=batch,
        )
        if is_bwd:
            config["recompute_summary_cfg"], config["recompute_series_cfg"], config["bprop_summary_cfg"], config["bprop_cfg"] = chain_bwd.build_configs(
                io_dtype,
                regenerate=regenerate,
                coarse=coarse,
                seed_dtype=dtype(node.inputs.get("state_checkpoints")),
                use_dstate0=plan.use_dstate0,
                skip_bwd_partials=plan.skip_bwd_in_kernel,
                **features,
                **flags,
            )
            config.update(regenerate=regenerate, coarse=coarse, has_dfs=plan.use_dstate_in, seed_every_n_tokens=checkpoint if coarse else 0)
            host = chain_bwd.chain_backward_host
        else:
            config["summary_cfg"], config["prefill_cfg"] = chain_fwd.build_configs(
                io_dtype,
                store_final_state=plan.has_final_state,
                enable_checkpoints=plan.has_state_checkpoints,
                checkpoint_dtype=dtype(node.outputs.get("state_checkpoints")),
                has_z=has_z,
                store_pre_gate=plan.has_pre_gate,
                **features,
                **flags,
            )
            config["checkpoint_every_n"] = checkpoint if plan.has_state_checkpoints else 0
            host = chain_fwd.chain_forward_host
    else:
        config.update(
            split=plan.split,
            overhead_chunks=max(1, split_k.OVERHEAD_TOKENS // plan.b_t),
            warmup_cap=split_k.warmup_cap_chunks(1),
            n_heads_out=ho,
            num_sms=plan.num_sm,
            io_dtype=io_dtype,
            order_gen=not plan.split,
            n_tiles=plan.n_tiles,
            ideal_chunks=plan.ideal or 0,
            batch_size=batch,
            log2_thresh=split_k.DEFAULT_LOG2_THRESHOLD,
            n_scan_ctas=split_k.chunk_scratch_rows(total, batch, plan.b_t),
            n_walk_ctas=batch * ho,
        )
        if is_bwd:
            dstate = node.inputs.get("d_final_state")
            dstate0 = node.outputs.get("d_initial_state")
            config["bprop_cfg"] = mamba2_bprop_f16.build_cfg(
                io_dtype,
                dtype(dstate if dstate is not None else dstate0),
                use_dstate_in=plan.use_dstate_in,
                use_dstate0=plan.use_dstate0,
                skip_bwd_partials=plan.skip_bwd_in_kernel,
                **features,
                **flags,
            )
            if regenerate:
                seed = node.inputs.get("state_checkpoints") if coarse else state
                config["recompute_cfg"] = mamba2_recompute_f16.build_cfg(
                    io_dtype,
                    dtype(seed),
                    use_initial_state=seed is not None,
                    store_final_state=False,
                    enable_checkpoints=True,
                    seed_checkpoints=coarse,
                    **flags,
                )
                config.update(
                    coarse=coarse,
                    seed_span_chunks=plan.recompute_span_tokens // plan.b_t if coarse else 0,
                    seed_every_n_tokens=checkpoint if coarse else 0,
                )
            host = warmup_bwd.regenerate_backward_host if regenerate else warmup_bwd.warmup_backward_host
        else:
            config["prefill_cfg"] = mamba2_prefill_f16.build_cfg(
                io_dtype,
                state_dtype,
                use_initial_state=state is not None,
                store_final_state=plan.has_final_state,
                enable_checkpoints=plan.has_state_checkpoints,
                checkpoint_dtype=dtype(node.outputs.get("state_checkpoints")),
                has_z=has_z,
                store_pre_gate=plan.has_pre_gate,
                **features,
                **flags,
            )
            config["checkpoint_every_n"] = checkpoint if plan.has_state_checkpoints else 0
            host = warmup_fwd.warmup_forward_host
    parameters = inspect.signature(host).parameters
    scalar_types = {name: p.annotation for name, p in parameters.items() if p.annotation in (ct.Int32, ct.Float32)}
    regions = plan.workspace_regions

    @cute.jit
    def launch(stream, *buffers):
        ports = dict(zip(names, buffers[:-1]))
        w = workspace_views(buffers[-1], regions)
        q, k, v, g, a, cu = (ports[name] for name in ("q", "k", "v", "g", "a", "cu_seqlens"))
        state0, dt_bias = ports.get("initial_state"), ports.get("dt_bias")
        checkpoints = w["state_checkpoints"] if regenerate else ports.get("state_checkpoints")
        args = dict(
            config,
            q=q,
            k=k,
            v=v,
            gate=g,
            a=a,
            dt_bias=dt_bias,
            d=ports.get("d"),
            cu_seqlens=cu,
            checkpoints=checkpoints,
            work_items=w["work_items"],
            composed=None,
            dcomposed=None,
            stream=stream,
        )
        if is_bwd:
            do = ports["dO"]
            if has_z:
                num_vectors = total * (dv // (16 // (io_dtype.width // 8)))
                n_blocks = max(1, min(-(-num_vectors // gate_bwd.MAMBA2_ZGATE_THREADS), max(1, 4096 // ho)))
                gate_bwd.mamba2_zgate_bwd_launch(
                    do, ports["z"], ports["O_pre"], w["do_pre"], ports["dZ"], ct.Int32(num_vectors), dv, io_dtype, ct.Int32(ho), ct.Int32(n_blocks), stream
                )
                do = w["do_pre"]
            dq = w["dq_ho"] if plan.fold_dq else ports["dQ"]
            dkey = w["dk_ho"] if plan.fold_dk else ports["dK"]
            args.update(
                do=do,
                dq=dq,
                dk=dkey,
                dv=ports["dV"],
                dstate0=ports.get("d_initial_state"),
                exit_dots=w["exit_dots"],
                dot_q=w["dot_q"],
                dot_k=w["dot_k"],
                skip_bwd_part=w.get("skip_bwd_part"),
                exit_slots=w["gate_slots"],
                series=checkpoints,
                seed_checkpoints=ports.get("state_checkpoints") if coarse else None,
            )
        else:
            args.update(
                o=ports["O"],
                o_pre=ports.get("O_pre"),
                z=ports.get("z"),
                seed_indices=ports.get("state_indices"),
                final_indices=ports.get("state_indices"),
            )
        if chain:
            args.update(
                cu_pieces=w["cu_pieces"],
                main_rows=w["main_rows"],
                summary_rows=w["summary_rows"],
                main_count=w["main_count"],
                summary_count=w["summary_count"],
                work_items_summary=w["work_items_summary"],
                scheduler_all=w["scheduler_all"],
                state_h=w.get("state_h"),
                state_x=w.get("state_x"),
                transition=w["transition"],
                seed=state0,
            )
            if is_bwd:
                args.update(
                    scheduler_summary_fwd=w["scheduler_summary_fwd"],
                    scheduler_summary_bwd=w["scheduler_summary_bwd"],
                    scheduler_bprop=w["scheduler_bwd"],
                    scheduler_regenerate=w["scheduler_regenerate"],
                    prefill_words=w.get("recompute_tensormaps"),
                    bprop_words=w["tensormaps"],
                    state_g=w["state_g"],
                    state_dx=w["state_dx"],
                    dfs=ports.get("d_final_state"),
                    exit_base=w["exit_base"],
                )
            else:
                args.update(
                    scheduler_summary=w["scheduler_summary"],
                    scheduler_prefill=w["scheduler_prefill"],
                    words=w["tensormaps"],
                    final_state=ports.get("final_state"),
                )
        else:
            args.update(
                gate_table=g if plan.split else None,
                a_table=a if plan.split else None,
                dt_bias_table=dt_bias,
                cu_seqlens_table=cu,
                state_in=state0 if not coarse else None,
                state_out=ports.get("final_state"),
                dstate_in=ports.get("d_final_state"),
                work_items_table=w["work_items"],
                work_count=w["work_count"],
                work_count_table=w["work_count"],
                staging=w.get("item_scratch"),
                item_scratch=w.get("item_scratch"),
                chunk_scratch=w.get("chunk_scratch"),
                scheduler=w["scheduler_bwd"] if is_bwd else w["scheduler"],
                transition=None,
                workspace=w["tensormaps"],
            )
            if regenerate:
                args.update(
                    scheduler_recompute=w["scheduler_recompute"],
                    scheduler_bprop=w["scheduler_bwd"],
                    recompute_workspace=w["recompute_tensormaps"],
                    bprop_workspace=w["tensormaps"],
                    series_items=w.get("work_items_recompute"),
                    series_count=w.get("work_count_recompute"),
                )
        for name, scalar_type in scalar_types.items():
            args[name] = scalar_type(args[name])
        host(**{name: args[name] for name in parameters})
        if is_bwd:
            for src, dst, heads in ((dq, ports["dQ"], hq), (dkey, ports["dK"], hk)):
                if heads != ho:
                    words = total * heads * (dk // 2)
                    head_reduce.launch(
                        src,
                        dst,
                        ct.Int64(words),
                        ct.Int64(dst.stride[0] // 2),
                        ct.Int64(dst.stride[1] // 2),
                        ct.Int32(-(-words // head_reduce.BLOCK)),
                        ct.Int32(heads),
                        ct.Int32(ho // heads),
                        dk // 2,
                        io_dtype,
                        stream,
                    )
            skip_part = w.get("skip_bwd_part")
            if skip_part is not None:
                skip_part = contiguous_view(skip_part.iterator, (math.prod(skip_part.shape),))
            gate_bwd.mamba2_gate_bwd_launch(
                w["dot_q"],
                w["dot_k"],
                w["exit_dots"],
                g,
                a,
                dt_bias,
                w["gate_slots"],
                ports["dG"],
                w["gate_part_a"],
                w.get("gate_part_dt"),
                skip_part,
                ports["dA"],
                ports.get("d_dt_bias"),
                ports.get("dD"),
                ct.Int32(ho),
                ct.Int32(plan.n_slots),
                ct.Float32(plan.dt_min),
                ct.Float32(plan.dt_max),
                plan.b_t,
                plan.dt_softplus,
                dv if d_per_channel else 1,
                stream,
            )

    return launch
