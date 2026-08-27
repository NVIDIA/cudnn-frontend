# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark every CATALOG config on a single matmul shape vs cuBLAS.

`--shape` is `B,M,N,K` (B independent same-shape GEMMs; B=1 = plain matmul).
Timing modes: delayed (default) / nsys / events. `--rotate-buffers` defeats
hot-L2 inflation on small shapes.

    python benchmark/gemm/frost/benchmark_matmul.py --shape 1,8192,8192,8192
"""

from __future__ import annotations

import argparse
import sys
import time

import cudnn  # noqa: F401
import cudnn.gemm.frost  # noqa: F401
import torch

from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
from cudnn.gemm.frost.fusion_ir import FusionChain as _FC, MatmulSpec as _MS, OutputSpec as _OS
from cudnn.gemm.frost.kernel_registry import candidates as _candidates

from benchmark_utils import (
    add_sweep_args,
    find_cublas_time,
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


def _build_spec_map():
    """Legacy label -> (geometry cfg, cta_group) for every sweepable
    matmul strategy, via the registry funnel. Labels reconstruct the old
    CONFIG_..._Nctamma form so --configs still accepts them."""
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
    )
    m = {}
    for t, cfg in _candidates(chain):
        label = f"{cfg.name}_{t.cta_group}ctamma"
        m[label] = (cfg, t.cta_group)
    return m


_SPEC_MAP = _build_spec_map()


def _vp(handles, a, b, c):
    """Variant-pack dict {cuDNN tensor: buffer} keyed by the graph's tensors."""
    A, B, C = handles
    return {A: a, B: b, C: c}


def _build_plan(g, cfg, name):
    """JIT-compile the recorded graph with a forced tile config."""
    _, cta_group = spec_for(name, _SPEC_MAP)
    return jit_from_cudnn_graph(g, config=cfg, cta_group=cta_group)


# ---------------------------------------------------------------------------
# Graph + data setup
# ---------------------------------------------------------------------------


def _graph_matmul(batch: int, M: int, N: int, K: int):
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[batch, M, K], stride=[M * K, K, 1])
    Bt = g.tensor(name="B", dim=[batch, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=Bt, name="mm")
    C.set_output(True)
    return g, (A, Bt, C)


def _mkdata(batch: int, M: int, N: int, K: int):
    torch.manual_seed(0)
    a = torch.empty(batch, M, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    b = torch.empty(batch, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    c = torch.empty(batch, M, N, dtype=torch.bfloat16, device="cuda")
    return a, b, c


# ---------------------------------------------------------------------------
# Buffer rotation — defeat the hot-L2 artifact on small shapes
# ---------------------------------------------------------------------------


def _mkdata_pool(batch: int, M: int, N: int, K: int, nbuf: int):
    """`nbuf` independent (a, b, c) triples at distinct GMEM addresses (nbuf<=1
    returns the single base triple)."""
    a, b, c = _mkdata(batch, M, N, K)
    pool = [(a, b, c)]
    # Distinct allocations (clone → fresh GMEM); contents don't matter for timing.
    for _ in range(max(0, nbuf - 1)):
        pool.append((a.clone(), b.clone(), c.clone()))
    return pool


def _per_set_bytes(batch: int, M: int, N: int, K: int) -> int:
    # BF16 = 2 bytes/elem; a:(batch,M,K) b:(batch,N,K) c:(batch,M,N).
    return 2 * batch * (M * K + N * K + M * N)


# ---------------------------------------------------------------------------
# Worker mode: just run the kernels under nsys profile, no Python timing.
# ---------------------------------------------------------------------------


def _nsys_worker(
    shape: str,
    configs: list[str],
    warmup: int,
    iters: int,
    nbuf: int,
) -> None:
    """Inner mode re-exec'd under nsys: run each config (and cuBLAS) for
    warmup+iters launches, no timing — nsys captures it. Timed iters rotate
    across the pool; warmup uses a dedicated buffer."""
    B, M, N, K = (int(x) for x in shape.split(","))
    wa, wb, wc = _mkdata(B, M, N, K)  # dedicated warmup buffer
    pool = _mkdata_pool(B, M, N, K, nbuf)  # rotation pool for timed iters

    print(f"[worker] shape={B}x{M}x{N}x{K}, configs={len(configs)}, " f"warmup={warmup}, iters={iters}, rotate_buffers={nbuf}")

    # 1. cuBLAS.
    for _ in range(warmup):
        torch.matmul(wa, wb.transpose(-1, -2), out=wc)
    for i in range(iters):
        a, b, c = pool[i % nbuf]
        torch.matmul(a, b.transpose(-1, -2), out=c)
    torch.cuda.synchronize()

    # 2. each GEMM config.
    config_names = configs or list(_SPEC_MAP)
    for name in config_names:
        spec = spec_for(name, _SPEC_MAP)
        if spec is None:
            continue
        cfg = spec[0]
        try:
            g, h = _graph_matmul(B, M, N, K)
            plan = _build_plan(g, cfg, name)
            for _ in range(warmup):
                plan(_vp(h, wa, wb, wc))
            for i in range(iters):
                a, b, c = pool[i % nbuf]
                plan(_vp(h, a, b, c))
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
        default="1,4096,4096,4096",
        help="B,M,N,K (default 1,4096,4096,4096; B = batch / number of " "independent same-shape GEMMs)",
    )
    add_sweep_args(parser)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA, skipping.")
        return 1

    parts = [int(x) for x in args.shape.split(",")]
    if len(parts) != 4:
        sys.exit("--shape must be B,M,N,K (four values; use B=1 for a plain matmul)")
    B, M, N, K = parts
    nbuf = resolve_nbuf(args.rotate_buffers, _per_set_bytes(B, M, N, K))

    if args._nsys_worker:
        configs = select_configs(args.configs, _SPEC_MAP) if args.configs else []
        _nsys_worker(args.shape, configs, args.warmup, args.iters, nbuf)
        return 0

    flops = 2 * B * M * N * K
    config_names = select_configs(args.configs, _SPEC_MAP)

    print(f"\n=== matmul B={B} {M}x{N}x{K}  (~{flops / 1e9:.1f} GFLOP) — BF16 ===")

    report_pool(nbuf, _per_set_bytes(B, M, N, K))

    rows: list[tuple[str, float, float, str]] = []  # (name, tflops, ms, note)
    t0 = time.time()

    def _fmt_row(name: str, tflops: float, ms: float, note: str, ref_tflops: float) -> str:
        if note:
            return f"  {name:50s} {'':8s}   {'':7s}   {note}"
        ratio = tflops / ref_tflops if ref_tflops > 0 else 0.0
        return f"  {name:50s} {tflops:8.2f}   {ms:7.3f}   {ratio:>9.2f}×"

    if args.timing == "nsys":
        print("  [timing: nsys median kernel duration]\n")
        inner_args = ["--shape", args.shape, "--warmup", str(args.warmup), "--iters", str(args.iters), "--rotate-buffers", str(nbuf)]
        if config_names:
            inner_args += ["--configs", ",".join(config_names)]
        kern_times = nsys_run_and_parse(__file__, inner_args, tag="benchmark_matmul")

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
            # Multiple specializations share a config name → pick the heaviest.
            _, ms = max(matches, key=lambda x: x[1])
            rows.append((name, flops / (ms * 1e-3) / 1e12, ms, ""))
    else:
        timer = time_ms_delayed if args.timing == "delayed" else time_ms_events
        if args.timing == "delayed":
            print("  [timing: events bracketed around delayed back-to-back " "launches — host overhead hidden behind a CUDA _sleep]\n")
        else:
            print(
                "  [timing: torch.cuda.Event wall-clock around python loop — "
                "includes ~50us/call Python+TVM-FFI dispatch overhead; use "
                "--timing delayed or --timing nsys for kernel-only timing]\n"
            )
        wa, wb, wc = _mkdata(B, M, N, K)  # dedicated warmup buffer
        pool = _mkdata_pool(B, M, N, K, nbuf)  # rotation pool for timed iters
        if args.stream:
            print("  ▶ running cuBLAS reference ...", flush=True)
        cublas_ms = timer(
            rotating(lambda t: torch.matmul(t[0], t[1].transpose(-1, -2), out=t[2]), pool),
            lambda: torch.matmul(wa, wb.transpose(-1, -2), out=wc),
            warmup=args.warmup,
            iters=args.iters,
        )
        cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12
        if args.stream:
            print(
                _fmt_row("cuBLAS (reference)", cublas_tflops, cublas_ms, "", cublas_tflops),
                flush=True,
            )

        # An async device fault sticky-poisons the CUDA context for the rest of
        # the process (every later launch returns LAUNCH_FAILED). After the
        # first such error, short-circuit the remaining configs as CTX_DEAD.
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
                    g, h = _graph_matmul(B, M, N, K)
                    plan = _build_plan(g, cfg, name)
                    ms = timer(
                        rotating(
                            lambda t, _plan=plan, _h=h: _plan(_vp(_h, t[0], t[1], t[2])),
                            pool,
                        ),
                        lambda _plan=plan, _h=h: _plan(_vp(_h, wa, wb, wc)),
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                    row = (name, flops / (ms * 1e-3) / 1e12, ms, "")
                except Exception as e:
                    msg = str(e).splitlines()[0][:50] if str(e) else type(e).__name__
                    row = (name, 0.0, float("inf"), f"ERR {msg}")
                    # Context-poisoning errors are unrecoverable — stop trying.
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
        print(f"  {'cuBLAS (reference)':50s} {cublas_tflops:8.2f}   {cublas_ms:7.3f}   {'1.00×':>10s}")
    else:
        print("  cuBLAS reference: n/a")

    ok = [r for r in rows if not r[3]]
    if ok and cublas_tflops > 0:
        best_name, best_tflops, _best_ms, _ = ok[0]
        print(f"\nbest GEMM: {best_name}" f" — {best_tflops:.2f} TFLOPS" f" ({best_tflops / cublas_tflops:.2f}× cuBLAS)")
    print(f"total: {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
