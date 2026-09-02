# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark MoE grouped block-scale matmul (FP4/FP8 + per-block SF) configs vs a
cuBLAS BF16 batched-GEMM reference. `--shape` is G,M,N,K (even split, S=G*M).

The reference is NOT a block-scaled GEMM — it's dense BF16 over the G groups — so
the "vs cuBLAS" ratio is FP4/FP8-MoE throughput relative to equivalent BF16 cuBLAS.
"""

from __future__ import annotations

import argparse
import sys
import time

import cudnn  # noqa: F401
import cudnn.gemm.frost  # noqa: F401
import torch

from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.gemm.frost.kernel_registry import candidates as _candidates

from benchmark_utils import (
    add_sweep_args,
    find_cublas_time,
    group_offsets,
    kernel_match_token,
    nsys_run_and_parse,
    rand_e8m0,
    report_pool,
    resolve_nbuf,
    rotating,
    select_configs,
    set_bytes,
    spec_for,
    time_ms_delayed,
    time_ms_events,
    to_blocked,
)


def _vp_moe_bs(handles, token, weight, sfa, sfb, fto, output):
    """MoE block-scale single-GEMM variant-pack dict keyed by the graph's tensors."""
    TOK, W, SFA, SFB, FTO, OUT = handles
    return {TOK: token, W: weight, SFA: sfa, SFB: sfb, FTO: fto, OUT: output}


def _build_plan(g, cfg, name):
    """JIT-compile the recorded graph with a forced tile config -> compiled kernel."""
    _, cta_group = spec_for(name, _SPEC_MAP)
    return jit_from_cudnn_graph(g, config=cfg, cta_group=cta_group)


# combo : (is_fp4, block_size, a_dtype, sf_dtype)
_COMBOS = {
    "nvfp4": (True, 16, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E4M3),
    "mxfp4": (True, 32, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E8M0),
    "mxfp8": (False, 32, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E8M0),
}


# Graph + data setup.


def _graph_moe_bs(S: int, N: int, K: int, E: int, combo: str):
    is_fp4, block_size, a_dt, sf_dt = _COMBOS[combo]
    sf_k = K // block_size
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=a_dt)
    w = g.tensor(name="weight", dim=[E, K, N], stride=[K * N, 1, K], data_type=a_dt)
    SFA = g.tensor(
        name="SFA",
        dim=[1, S, sf_k],
        stride=[S * sf_k, sf_k, 1],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    SFB = g.tensor(
        name="SFB",
        dim=[E, sf_k, N],
        stride=[sf_k * N, 1, sf_k],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    fto = g.tensor(
        name="first_token_offset",
        dim=[E, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.INT32,
    )
    tok_d = g.block_scale_dequantize(input=tok, descale=SFA, block_size=[1, block_size])
    w_d = g.block_scale_dequantize(input=w, descale=SFB, block_size=[block_size, 1])
    out = g.moe_grouped_matmul(
        tok_d,
        w_d,
        fto,
        mode=cudnn.moe_grouped_matmul_mode.NONE,
        compute_data_type=cudnn.data_type.FLOAT,
        name="moe",
    )
    out.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)
    return g, (tok, w, SFA, SFB, fto, out)


def _build_spec_map():
    """Label -> (geometry cfg, cta_group) for every MoE-block-scale
    strategy. Enumerated from an nvfp4 graph (template set is combo-independent)."""
    chain = analyze(_graph_moe_bs(512, 256, 512, 2, "nvfp4")[0])
    m = {}
    for t, cfg in _candidates(chain):
        label = f"{cfg.name}_{t.cta_group}ctamma"
        m[label] = (cfg, t.cta_group)
    return m


_SPEC_MAP = _build_spec_map()


def _mkdata(S: int, N: int, K: int, E: int, combo: str):
    """MoE-block-scale runtime tensors: (token, weight, sfa, sfb, output).

    token/weight are packed FP4 or FP8. SFA is reordered + padded to 128 rows PER
    GROUP then concatenated (even split); SFB is per-expert; output is BF16."""
    dev = "cuda"
    torch.manual_seed(0)
    is_fp4, block_size, _, _ = _COMBOS[combo]
    sf_k = K // block_size
    group_m = S // E

    if is_fp4:
        tok = torch.randint(0, 256, (1, S, K // 2), dtype=torch.uint8, device=dev).view(torch.float4_e2m1fn_x2)
        w = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev).view(torch.float4_e2m1fn_x2)
    else:
        tok = (torch.randn(1, S, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        w = (torch.randn(E, N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)

    if combo == "nvfp4":
        sfa_log = torch.randint(1, 4, (S, sf_k), device=dev).to(torch.float8_e4m3fn)
        sfb_log = torch.randint(1, 4, (E, N, sf_k), device=dev).to(torch.float8_e4m3fn)
    else:
        sfa_log = rand_e8m0((S, sf_k), dev)
        sfb_log = rand_e8m0((E, N, sf_k), dev)

    # SFA padded to 128 rows PER GROUP (group sizes need not be 128-aligned); SFB per-expert.
    sfa = torch.cat([to_blocked(sfa_log[g * group_m : (g + 1) * group_m]) for g in range(E)]).view(1, -1, 1)
    sfb = torch.cat([to_blocked(sfb_log[e]) for e in range(E)]).reshape(E, sf_k, N)
    out = torch.empty(1, S, N, dtype=torch.bfloat16, device=dev)
    return tok, w, sfa, sfb, out


def _mkdata_bf16(S: int, N: int, K: int, E: int):
    """BF16 reference operands (token, weight, output) for the batched-GEMM ref."""
    dev = "cuda"
    tok = torch.empty(1, S, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device=dev)
    w = torch.empty(E, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device=dev)
    out = torch.empty(1, S, N, dtype=torch.bfloat16, device=dev)
    return tok, w, out


# Buffer rotation: rotate timed launches across N independent tensor sets so a
# kernel never re-reads its inputs from a hot L2 (inflates small-shape TFLOPS).


def _mkdata_pool(S: int, N: int, K: int, E: int, combo: str, nbuf: int):
    base = _mkdata(S, N, K, E, combo)
    pool = [base]
    for _ in range(max(0, nbuf - 1)):
        pool.append(tuple(t.clone() for t in base))
    return pool


def _mkdata_bf16_pool(S: int, N: int, K: int, E: int, nbuf: int):
    base = _mkdata_bf16(S, N, K, E)
    pool = [base]
    for _ in range(max(0, nbuf - 1)):
        pool.append(tuple(t.clone() for t in base))
    return pool


# cuBLAS reference — BF16 batched GEMM over the E equal-sized groups.


def _cublas_launch(buf, S: int, N: int, K: int, E: int) -> None:
    tok, w, out = buf
    group_m = S // E
    tok_g = tok.view(E, group_m, K)
    out_g = out.view(E, group_m, N)
    torch.matmul(tok_g, w.transpose(-1, -2), out=out_g)


# Worker mode (re-exec'd under nsys).


def _nsys_worker(shape, combo, configs, warmup, iters, nbuf, no_baseline=False) -> None:
    G, M, N, K = (int(x) for x in shape.split(","))
    S, E = G * M, G
    offsets = group_offsets(S, E)
    print(f"[worker] shape G={G} M={M} N={N} K={K} (S={S}) combo={combo}, " f"configs={len(configs)}, warmup={warmup}, iters={iters}, rotate={nbuf}")

    # 1. BF16 batched-GEMM reference.
    if not no_baseline:
        wbf = _mkdata_bf16(S, N, K, E)
        bf_pool = _mkdata_bf16_pool(S, N, K, E, nbuf)
        for _ in range(warmup):
            _cublas_launch(wbf, S, N, K, E)
        for i in range(iters):
            _cublas_launch(bf_pool[i % nbuf], S, N, K, E)
        torch.cuda.synchronize()
        del wbf, bf_pool
        torch.cuda.empty_cache()
    wset = _mkdata(S, N, K, E, combo)
    pool = _mkdata_pool(S, N, K, E, combo, nbuf)

    # 2. each MoE-block-scale config.
    for name in configs or list(_SPEC_MAP):
        spec = spec_for(name, _SPEC_MAP)
        cfg = spec[0] if spec else None
        if cfg is None:
            continue
        try:
            g, h = _graph_moe_bs(S, N, K, E, combo)
            plan = _build_plan(g, cfg, name)
            for _ in range(warmup):
                plan(_vp_moe_bs(h, wset[0], wset[1], wset[2], wset[3], offsets, wset[4]))
            for i in range(iters):
                t = pool[i % nbuf]
                plan(_vp_moe_bs(h, t[0], t[1], t[2], t[3], offsets, t[4]))
            torch.cuda.synchronize()
            print(f"[worker] OK   {name}")
        except Exception as e:
            print(f"[worker] FAIL {name}: {type(e).__name__}: {e}")


# Main.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        default="8,512,4096,4096",
        help="G,M,N,K (groups × per-group-M × N × K; default " "8,512,4096,4096 → S=G*M=4096 tokens)",
    )
    parser.add_argument("--combo", choices=tuple(_COMBOS), default="nvfp4")
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="skip the cuBLAS BF16 reference; its BF16 tensor set is never " "allocated, which is the larger footprint on big shapes",
    )
    add_sweep_args(parser)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA, skipping.")
        return 1

    parts = [int(x) for x in args.shape.split(",")]
    if len(parts) != 4:
        sys.exit("--shape must be G,M,N,K (four values: groups × per-group-M × N × K)")
    G, M, N, K = parts
    S, E = G * M, G  # total tokens, num groups (== experts)
    combo = args.combo
    wset = _mkdata(S, N, K, E, combo)  # dedicated warmup buffer, and the pool's size unit
    per_set_bytes = set_bytes(wset)
    nbuf = resolve_nbuf(args.rotate_buffers, per_set_bytes, free_divisor=4)

    if args._nsys_worker:
        configs = select_configs(args.configs, _SPEC_MAP) if args.configs else []
        _nsys_worker(args.shape, combo, configs, args.warmup, args.iters, nbuf, args.no_baseline)
        return 0

    flops = 2 * S * N * K
    config_names = select_configs(args.configs, _SPEC_MAP)

    print(f"\n=== moe_block_scale_matmul G={G} M={M} N={N} K={K}  " f"(S={S} tokens, ~{flops / 1e9:.1f} GFLOP) — {combo} ===")

    report_pool(nbuf, per_set_bytes)

    rows: list[tuple[str, float, float, str]] = []
    t0 = time.time()

    def _fmt_row(name, tflops, ms, note, ref_tflops) -> str:
        if note:
            return f"  {name:50s} {'':8s}   {'':7s}   {note}"
        if ref_tflops <= 0:
            return f"  {name:50s} {tflops:8.2f}   {ms:7.3f}   {'':>10s}"
        return f"  {name:50s} {tflops:8.2f}   {ms:7.3f}   {tflops / ref_tflops:>9.2f}×"

    if args.timing == "nsys":
        print("  [timing: nsys median kernel duration]\n")
        inner = ["--shape", args.shape, "--combo", combo, "--warmup", str(args.warmup), "--iters", str(args.iters), "--rotate-buffers", str(nbuf)]
        if config_names:
            inner += ["--configs", ",".join(config_names)]
        if args.no_baseline:
            inner += ["--no-baseline"]
        kern_times = nsys_run_and_parse(__file__, inner, tag="bench_moebs")
        cublas_tflops, cublas_ms = 0.0, float("nan")
        cublas_hit = None if args.no_baseline else find_cublas_time(kern_times)
        if cublas_hit:
            cublas_name, cublas_ms = cublas_hit
            cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12
            print(f"  cuBLAS (BF16) kernel: {cublas_name}")
        elif not args.no_baseline:
            cublas_tflops, cublas_ms = float("nan"), float("nan")
            print("  cuBLAS kernel: not detected in nsys output")
        for name in config_names:
            spec = spec_for(name, _SPEC_MAP)
            cfg = spec[0] if spec else None
            if cfg is None:
                rows.append((name, 0.0, float("inf"), "UNKNOWN_CONFIG"))
                continue
            tok = kernel_match_token(cfg, spec[1])
            matches = [(k, v) for k, v in kern_times.items() if tok in k]
            if not matches:
                rows.append((name, 0.0, float("inf"), "NO_KERNEL_IN_NSYS"))
                continue
            _, ms = max(matches, key=lambda x: x[1])
            rows.append((name, flops / (ms * 1e-3) / 1e12, ms, ""))
    else:
        timer = time_ms_delayed if args.timing == "delayed" else time_ms_events
        if args.timing == "delayed":
            print("  [timing: events around delayed back-to-back launches]\n")
        else:
            print("  [timing: torch.cuda.Event wall-clock (incl ~50us/call host overhead)]\n")
        offsets = group_offsets(S, E)
        cublas_tflops, cublas_ms = 0.0, float("nan")
        if not args.no_baseline:
            wbf = _mkdata_bf16(S, N, K, E)
            bf_pool = _mkdata_bf16_pool(S, N, K, E, nbuf)
            if args.stream:
                print("  ▶ running cuBLAS (BF16) reference ...", flush=True)
            cublas_ms = timer(
                rotating(lambda t: _cublas_launch(t, S, N, K, E), bf_pool),
                lambda: _cublas_launch(wbf, S, N, K, E),
                warmup=args.warmup,
                iters=args.iters,
            )
            cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12
            if args.stream:
                print(
                    _fmt_row(
                        "cuBLAS BF16 (reference)",
                        cublas_tflops,
                        cublas_ms,
                        "",
                        cublas_tflops,
                    ),
                    flush=True,
                )
            # Freed before the block-scale pool is built: the reference is done
            # with, and on large shapes its BF16 set is the bigger allocation.
            del wbf, bf_pool
            torch.cuda.empty_cache()
        pool = _mkdata_pool(S, N, K, E, combo, nbuf)

        ctx_dead = False
        for name in config_names:
            spec = spec_for(name, _SPEC_MAP)
            cfg = spec[0] if spec else None
            if cfg is None:
                row = (name, 0.0, float("inf"), "UNKNOWN_CONFIG")
            elif ctx_dead:
                row = (name, 0.0, float("inf"), "skipped (CUDA context dead)")
            else:
                if args.stream:
                    print(f"  ▶ running {name} ...", flush=True)
                try:
                    g, h = _graph_moe_bs(S, N, K, E, combo)
                    plan = _build_plan(g, cfg, name)
                    ms = timer(
                        rotating(
                            lambda t, _plan=plan, _h=h: _plan(_vp_moe_bs(_h, t[0], t[1], t[2], t[3], offsets, t[4])),
                            pool,
                        ),
                        lambda _plan=plan, _h=h: _plan(_vp_moe_bs(_h, wset[0], wset[1], wset[2], wset[3], offsets, wset[4])),
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                    row = (name, flops / (ms * 1e-3) / 1e12, ms, "")
                except Exception as e:
                    msg = str(e).splitlines()[0][:50] if str(e) else type(e).__name__
                    row = (name, 0.0, float("inf"), f"ERR {msg}")
                    if any(
                        s in str(e)
                        for s in (
                            "illegal memory access",
                            "unspecified launch failure",
                            "CUDA_ERROR_LAUNCH_FAILED",
                        )
                    ):
                        ctx_dead = True
            rows.append(row)
            if args.stream:
                print(_fmt_row(*row, cublas_tflops), flush=True)

    rows.sort(key=lambda r: -r[1])
    print("=" * 88)
    print(f"  {'config':50s} {'TFLOPS':>8s}   {'ms':>7s}   {'vs BF16':>10s}")
    print("=" * 88)
    for name, tflops, ms, note in rows:
        print(_fmt_row(name, tflops, ms, note, cublas_tflops))
    print("=" * 88)
    if cublas_tflops > 0:
        print(f"  {'cuBLAS BF16 (batched, reference)':50s} {cublas_tflops:8.2f}   {cublas_ms:7.3f}   {'1.00×':>10s}")
    else:
        print("  cuBLAS reference: n/a")

    ok = [r for r in rows if not r[3]]
    if ok:
        best_name, best_tflops, best_ms, _ = ok[0]
        ratio = f" ({best_tflops / cublas_tflops:.2f}× BF16 cuBLAS)" if cublas_tflops > 0 else ""
        print(f"\nbest GEMM: {best_name} — {best_tflops:.2f} TFLOPS{ratio}")
    print(f"total: {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
