# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark every MoE-capable CATALOG config on one MoE grouped-matmul shape vs
a cuBLAS batched-GEMM baseline. Mirrors benchmark_matmul.py.

`--shape` is `G,M,N,K` (G groups, M tokens/group even split, S=G*M tokens). Even
split makes the baseline a single batched GEMM over (E, group_m, K) @ (E, K, N).
"""

from __future__ import annotations

import argparse
import sys
import time

import cudnn  # noqa: F401
import cudnn.gemm.frost  # noqa: F401
import torch

from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
from cudnn.gemm.frost.fusion_ir import (
    FusionChain as _FC,
    MatmulSpec as _MS,
    MoeSpec as _MoeS,
    OutputSpec as _OS,
)
from cudnn.gemm.frost.kernel_registry import candidates as _candidates

from benchmark_utils import (
    add_sweep_args,
    find_cublas_time,
    group_offsets,
    kernel_match_token,
    nsys_run_and_parse,
    report_pool,
    resolve_nbuf,
    rotating,
    select_configs,
    spec_for,
    time_ms_delayed,
    time_ms_events,
)


def _vp_moe(handles, token, weight, fto, output):
    """MoE single-GEMM variant-pack dict keyed by the graph's tensors."""
    TOK, W, FTO, OUT = handles
    return {TOK: token, W: weight, FTO: fto, OUT: output}


def _build_plan(g, cfg, name):
    """JIT-compile the recorded graph with a forced tile config."""
    _, cta_group = spec_for(name, _SPEC_MAP)
    return jit_from_cudnn_graph(g, config=cfg, cta_group=cta_group)


def _build_spec_map():
    """Legacy label -> (geometry cfg, cta_group) for sweepable MoE
    strategies, via the registry funnel."""
    chain = _FC(
        matmul=_MS(
            M=4096,
            N=4096,
            K=4096,
            a_major="k",
            b_major="k",
            a_dtype="bf16",
            b_dtype="bf16",
            accum_dtype="fp32",
        ),
        output_specs=[_OS(source_ref=-1, dtype="bf16")],
        moe=_MoeS(num_experts=8),
    )
    m = {}
    for t, cfg in _candidates(chain):
        label = f"{cfg.name}_{t.cta_group}ctamma"
        m[label] = (cfg, t.cta_group)
    return m


_SPEC_MAP = _build_spec_map()

# ---------------------------------------------------------------------------
# Graph + data setup
# ---------------------------------------------------------------------------


def _graph_moe(S: int, N: int, K: int, E: int):
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    tok = g.tensor(
        name="token",
        dim=[1, S, K],
        stride=[S * K, K, 1],
        data_type=cudnn.data_type.BFLOAT16,
    )
    w = g.tensor(
        name="weight",
        dim=[E, K, N],
        stride=[K * N, 1, K],
        data_type=cudnn.data_type.BFLOAT16,
    )
    fto = g.tensor(
        name="first_token_offset",
        dim=[E, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.INT32,
    )
    out = g.moe_grouped_matmul(
        tok,
        w,
        fto,
        mode=cudnn.moe_grouped_matmul_mode.NONE,
        compute_data_type=cudnn.data_type.FLOAT,
        name="moe",
    )
    out.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)
    return g, (tok, w, fto, out)


def _mkdata(S: int, N: int, K: int, E: int):
    torch.manual_seed(0)
    tok = torch.empty(1, S, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    w = torch.empty(E, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    out = torch.empty(1, S, N, dtype=torch.bfloat16, device="cuda")
    return tok, w, out


# Buffer rotation — defeat the hot-L2 artifact on small shapes: rotate timed
# launches across nbuf independent tensor sets so a kernel doesn't re-read the
# prior launch's inputs from a hot L2 (see benchmark_matmul).


def _mkdata_pool(S: int, N: int, K: int, E: int, nbuf: int):
    """`nbuf` independent (token, weight, output) triples at distinct GMEM
    addresses (offsets are tiny + read-only, shared separately)."""
    tok, w, out = _mkdata(S, N, K, E)
    pool = [(tok, w, out)]
    for _ in range(max(0, nbuf - 1)):
        pool.append((tok.clone(), w.clone(), out.clone()))
    return pool


def _per_set_bytes(S: int, N: int, K: int, E: int) -> int:
    # BF16 = 2 bytes/elem; token:(S,K) weight:(E,N,K) output:(S,N).
    return 2 * (S * K + E * N * K + S * N)


# ---------------------------------------------------------------------------
# cuBLAS baseline — batched GEMM over the E equal-sized groups
# ---------------------------------------------------------------------------


def _cublas_launch(buf, S: int, N: int, K: int, E: int) -> None:
    """One batched GEMM: (E, group_m, K) @ (E, K, N) -> (E, group_m, N);
    equivalent to the grouped matmul when groups are equal-sized."""
    tok, w, out = buf
    group_m = S // E
    tok_g = tok.view(E, group_m, K)
    out_g = out.view(E, group_m, N)
    torch.matmul(tok_g, w.transpose(-1, -2), out=out_g)


# ---------------------------------------------------------------------------
# Worker mode (re-exec'd under nsys)
# ---------------------------------------------------------------------------


def _nsys_worker(shape, configs, warmup, iters, nbuf) -> None:
    G, M, N, K = (int(x) for x in shape.split(","))
    S, E = G * M, G
    wtok, ww, wout = _mkdata(S, N, K, E)  # dedicated warmup buffer
    pool = _mkdata_pool(S, N, K, E, nbuf)  # rotation pool for timed iters
    offsets = group_offsets(S, E)
    print(f"[worker] shape G={G} M={M} N={N} K={K} (S={S}), configs={len(configs)}, " f"warmup={warmup}, iters={iters}, rotate_buffers={nbuf}")

    # 1. cuBLAS baseline — batched GEMM.
    for _ in range(warmup):
        _cublas_launch((wtok, ww, wout), S, N, K, E)
    for i in range(iters):
        _cublas_launch(pool[i % nbuf], S, N, K, E)
    torch.cuda.synchronize()

    # 2. each MoE config.
    config_names = configs or list(_SPEC_MAP)
    for name in config_names:
        spec = spec_for(name, _SPEC_MAP)
        cfg = spec[0] if spec else None
        if cfg is None:
            continue
        try:
            g, h = _graph_moe(S, N, K, E)
            plan = _build_plan(g, cfg, name)
            for _ in range(warmup):
                plan(_vp_moe(h, wtok, ww, offsets, wout))
            for i in range(iters):
                tok, w, out = pool[i % nbuf]
                plan(_vp_moe(h, tok, w, offsets, out))
            torch.cuda.synchronize()
            print(f"[worker] OK   {name}")
        except Exception as e:
            print(f"[worker] FAIL {name}: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        default="8,512,4096,4096",
        help="G,M,N,K (groups × per-group-M × N × K; default " "8,512,4096,4096 → S=G*M=4096 tokens)",
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
    per_set = _per_set_bytes(S, N, K, E)
    nbuf = resolve_nbuf(args.rotate_buffers, per_set)

    if args._nsys_worker:
        configs = select_configs(args.configs, _SPEC_MAP) if args.configs else []
        _nsys_worker(args.shape, configs, args.warmup, args.iters, nbuf)
        return 0

    flops = 2 * S * N * K
    config_names = select_configs(args.configs, _SPEC_MAP)

    print(f"\n=== moe_grouped_matmul G={G} M={M} N={N} K={K}  " f"(S={S} tokens, ~{flops / 1e9:.1f} GFLOP) — BF16 ===")

    report_pool(nbuf, per_set)

    rows: list[tuple[str, float, float, str]] = []
    t0 = time.time()

    def _fmt_row(name, tflops, ms, note, ref_tflops) -> str:
        if note:
            return f"  {name:50s} {'':8s}   {'':7s}   {note}"
        ratio = tflops / ref_tflops if ref_tflops > 0 else 0.0
        return f"  {name:50s} {tflops:8.2f}   {ms:7.3f}   {ratio:>9.2f}×"

    if args.timing == "nsys":
        print("  [timing: nsys median kernel duration]\n")
        inner_args = ["--shape", args.shape, "--warmup", str(args.warmup), "--iters", str(args.iters), "--rotate-buffers", str(nbuf)]
        if config_names:
            inner_args += ["--configs", ",".join(config_names)]
        kern_times = nsys_run_and_parse(__file__, inner_args, tag="bench_moe")
        cublas_hit = find_cublas_time(kern_times)
        if cublas_hit:
            cublas_name, cublas_ms = cublas_hit
            cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12
            print(f"  cuBLAS kernel: {cublas_name}")
        else:
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
        wtok, ww, wout = _mkdata(S, N, K, E)
        pool = _mkdata_pool(S, N, K, E, nbuf)
        offsets = group_offsets(S, E)
        if args.stream:
            print("  ▶ running cuBLAS reference ...", flush=True)
        cublas_ms = timer(
            rotating(lambda t: _cublas_launch(t, S, N, K, E), pool),
            lambda: _cublas_launch((wtok, ww, wout), S, N, K, E),
            warmup=args.warmup,
            iters=args.iters,
        )
        cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12
        if args.stream:
            print(
                _fmt_row("cuBLAS (reference)", cublas_tflops, cublas_ms, "", cublas_tflops),
                flush=True,
            )

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
                    g, h = _graph_moe(S, N, K, E)
                    plan = _build_plan(g, cfg, name)
                    ms = timer(
                        rotating(
                            lambda t, _plan=plan, _h=h: _plan(_vp_moe(_h, t[0], t[1], offsets, t[2])),
                            pool,
                        ),
                        lambda _plan=plan, _h=h: _plan(_vp_moe(_h, wtok, ww, offsets, wout)),
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
    print(f"  {'config':50s} {'TFLOPS':>8s}   {'ms':>7s}   {'vs cuBLAS':>10s}")
    print("=" * 88)
    for name, tflops, ms, note in rows:
        print(_fmt_row(name, tflops, ms, note, cublas_tflops))
    print("=" * 88)
    if cublas_tflops > 0:
        print(f"  {'cuBLAS (batched, reference)':50s} {cublas_tflops:8.2f}   {cublas_ms:7.3f}   {'1.00×':>10s}")
    else:
        print("  cuBLAS reference: n/a")

    ok = [r for r in rows if not r[3]]
    if ok and cublas_tflops > 0:
        best_name, best_tflops, best_ms, _ = ok[0]
        print(f"\nbest GEMM: {best_name} — {best_tflops:.2f} TFLOPS " f"({best_tflops / cublas_tflops:.2f}× cuBLAS)")
    print(f"total: {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
