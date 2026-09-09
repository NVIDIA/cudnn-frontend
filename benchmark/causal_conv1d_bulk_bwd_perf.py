# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Same-process schedule and FLA comparison for dense conv backward."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path

import cudnn
import torch

SOURCE_CUDNN = Path(__file__).resolve().parents[1] / "python" / "cudnn"
if str(SOURCE_CUDNN) not in cudnn.__path__:
    cudnn.__path__.insert(0, str(SOURCE_CUDNN))

from cudnn.causal_conv1d_bulk_sm100.backward import (
    compile_causal_conv1d_bulk_bwd_prototype,
)
from cudnn._causal_conv1d_arch import F32X2_COMPUTE_CAPABILITIES
from fla.modules.conv.triton.ops import causal_conv1d_bwd as fla_bwd
from fla.ops.utils import prepare_chunk_indices


def capture(call, warmup):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = call()
    torch.cuda.synchronize()
    return graph, outputs


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--channels", type=int, default=2048)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument(
        "--vec4-ab",
        action="store_true",
        help="time t128, v4-stream, FLA, and v2-cpasync where supported",
    )
    parser.add_argument(
        "--packed",
        action="store_true",
        help="split B=1 into three uneven sequences using device cu_seqlens",
    )
    args = parser.parse_args()
    generator = torch.Generator(device="cuda").manual_seed(20260829)
    shape = (1, args.tokens, args.channels)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25
    weight = torch.randn((args.channels, 4), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25
    bias = torch.randn((args.channels,), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25 if args.bias else None
    dy = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25
    cu_seqlens = None
    chunk_indices = None
    if args.packed:
        first = args.tokens // 4
        second = first + args.tokens // 3
        offsets = (0, first, second, args.tokens)
        if first <= 0 or second >= args.tokens:
            parser.error("--packed requires at least four tokens")
        cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.int32)
        cu_seqlens_cpu = torch.tensor(offsets, dtype=torch.int32)
        chunk_indices = prepare_chunk_indices(
            cu_seqlens,
            64,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )

    arms = {}
    native_outputs = {}
    native_keepalive = {}
    native_plans = {}
    native_schedule_skips = {}
    vec4_supported = not args.packed and bias is None and args.channels % 512 == 0 and args.tokens >= 16
    capability = torch.cuda.get_device_capability()
    cpasync_supported = vec4_supported and args.tokens >= 64 and capability in F32X2_COMPUTE_CAPABILITIES
    if args.vec4_ab and not vec4_supported:
        parser.error("--vec4-ab requires dense no-bias input, T>=16, and D divisible by 512")
    schedules = ["t128", "v4-stream"] if args.vec4_ab else ["t32", "t64", "t128", "t64-partial"]
    if args.vec4_ab:
        if cpasync_supported:
            schedules.append("v2-cpasync")
        else:
            native_schedule_skips["v2-cpasync"] = f"requires T>=64 and packed-f32x2 support; got T={args.tokens}, capability={capability}"
    if vec4_supported and not args.vec4_ab:
        schedules.append("v4-stream")
    for schedule in schedules:
        backend = compile_causal_conv1d_bulk_bwd_prototype(
            x,
            weight,
            dy,
            cu_seqlens,
            schedule=schedule,
            bias=bias,
        )
        dx = torch.empty_like(x)
        dw = torch.empty_like(weight, dtype=torch.float32)
        db = torch.empty_like(bias, dtype=torch.float32) if bias is not None else None
        workspace = torch.empty(backend.dweight_workspace_numel, device="cuda", dtype=torch.float32) if backend.dweight_workspace_numel else None
        packed_tile_map = (
            torch.empty(
                backend.packed_tile_map_numel,
                device="cuda",
                dtype=torch.int32,
            )
            if backend.packed_tile_map_numel
            else None
        )
        graph, _ = capture(
            lambda backend=backend, dx=dx, dw=dw, db=db, workspace=workspace, packed_tile_map=packed_tile_map: (
                backend.execute(
                    x,
                    weight,
                    dy,
                    dx,
                    dw,
                    cu_seqlens=cu_seqlens,
                    packed_tile_map=packed_tile_map,
                    dweight_workspace=workspace,
                    bias=bias,
                    db_accum=db,
                )
            ),
            args.warmup,
        )
        arms[schedule] = graph
        native_outputs[schedule] = (dx, dw, db, backend.dweight_workspace_bytes)
        native_plans[schedule] = {
            "kernel_variant": backend.kernel_variant,
            "sm_count": backend.sm_count,
            "tokens_per_cta": backend.tokens_per_cta,
            "token_ctas": backend.num_dweight_partials,
        }
        # CUDA graphs retain raw addresses, not these Python owners. Keep every
        # caller-owned buffer alive after later schedules replace loop locals.
        native_keepalive[schedule] = (backend, workspace, packed_tile_map)

    fla_graph, fla_outputs = capture(
        lambda: fla_bwd(
            x=x,
            dy=dy,
            dht=None,
            weight=weight,
            bias=bias,
            activation="silu",
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        ),
        args.warmup,
    )
    arms["fla"] = fla_graph
    fla_dx, fla_dw, fla_db = fla_outputs[:3]

    # Capture records work but does not guarantee that tensors allocated only
    # inside the captured region contain a completed result.  Materialize one
    # result from every arm before reading correctness outputs.
    for graph in arms.values():
        graph.replay()
    torch.cuda.synchronize()

    correctness = {}
    for name, (dx, dw, db, workspace_bytes) in native_outputs.items():
        dx_diff = dx.float() - fla_dx.float()
        dw_diff = dw - fla_dw.float()
        correctness[name] = {
            "dx_max_abs": float(dx_diff.abs().max()),
            "dx_rel_l2": float(torch.linalg.vector_norm(dx_diff) / torch.linalg.vector_norm(fla_dx.float())),
            "dw_max_abs": float(dw_diff.abs().max()),
            "dw_rel_l2": float(torch.linalg.vector_norm(dw_diff) / torch.linalg.vector_norm(fla_dw.float())),
            "workspace_bytes": workspace_bytes,
        }
        if db is not None:
            db_diff = db - fla_db.float()
            correctness[name].update(
                db_max_abs=float(db_diff.abs().max()),
                db_rel_l2=float(torch.linalg.vector_norm(db_diff) / torch.linalg.vector_norm(fla_db.float())),
            )
            torch.testing.assert_close(db, fla_db.float(), atol=1e-1, rtol=5e-2)
        torch.testing.assert_close(dx, fla_dx, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(dw, fla_dw.float(), atol=1e-1, rtol=5e-2)

    names = tuple(arms)
    timings = {name: [] for name in names}
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(args.samples * len(names))]
    for start, end in events:
        start.record()
        end.record()
    torch.cuda.synchronize()
    event_index = 0
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for sample in range(args.samples):
            order = names[sample % len(names) :] + names[: sample % len(names)]
            pending = []
            for name in order:
                start, end = events[event_index]
                event_index += 1
                start.record()
                arms[name].replay()
                end.record()
                pending.append((name, start, end))
            torch.cuda.synchronize()
            for name, start, end in pending:
                timings[name].append(float(start.elapsed_time(end) * 1000.0))
    finally:
        if gc_was_enabled:
            gc.enable()

    summaries = {name: {"median_us": statistics.median(values), "samples_us": values} for name, values in timings.items()}
    fastest_native = min(native_outputs, key=lambda name: summaries[name]["median_us"])
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "shape": list(shape),
                "bias": bias is not None,
                "cu_seqlens": None if cu_seqlens is None else cu_seqlens.cpu().tolist(),
                "native_plans": native_plans,
                "native_schedule_skips": native_schedule_skips,
                "correctness_vs_fla": correctness,
                "timings": summaries,
                "fastest_native": fastest_native,
                "fla_over_fastest_native": summaries["fla"]["median_us"] / summaries[fastest_native]["median_us"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
