# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark every CATALOG config on a single matmul shape vs cuBLAS.

`--shape` is `B,M,N,K` (B independent same-shape GEMMs; B=1 = plain matmul).
`--dtype` selects bf16 (default), fp16, fp8 (E4M3 inputs / BF16 output), or
fp32. FP32 currently runs only the cuBLAS reference: FROST has no FP32 MMA.
Timing modes: delayed (default) / nsys / events. `--rotate-buffers` defeats
hot-L2 inflation on small shapes. `--sweep-swap-ab` benchmarks both
``swap_ab=False`` and ``swap_ab=True`` for every selected geometry.
`--sweep-split-k N` benchmarks ``split_k_slices=1..N``. When both are
specified, the two dimensions form a Cartesian product.

    python benchmark/gemm/frost/benchmark_matmul.py --shape 1,8192,8192,8192
    python benchmark/gemm/frost/benchmark_matmul.py --dtype fp8 --shape 1,4096,4096,4096
    python benchmark/gemm/frost/benchmark_matmul.py --dtype fp16 --shape 1,4096,4096,4096
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
    with_workspace,
    add_sweep_args,
    expand_config_variants,
    find_cublas_time,
    kernel_match_token,
    nsys_run_and_parse,
    report_pool,
    resolve_nbuf,
    rotating,
    select_config_variants,
    spec_for,
    time_ms_delayed,
    time_ms_events,
    validate_config_variant_args,
)

_DTYPES = {
    "bf16": (cudnn.data_type.BFLOAT16, torch.bfloat16),
    "fp16": (cudnn.data_type.HALF, torch.float16),
    "fp8": (cudnn.data_type.FP8_E4M3, torch.float8_e4m3fn),
    "fp32": (cudnn.data_type.FLOAT, torch.float32),
}


def _output_dtype(dtype: str) -> str:
    return "bf16" if dtype == "fp8" else dtype


def _build_spec_map(g):
    """Legacy label -> (geometry cfg, cta_group) for every sweepable
    matmul strategy, via the registry funnel. Labels reconstruct the old
    CONFIG_..._Nctamma form so --configs still accepts them."""
    chain = analyze(g)
    m = {}
    for t, cfg in _candidates(chain):
        # A family without the CTA-pair axis (sm120) has no cta_group at all.
        m[cfg.name] = (cfg, getattr(cfg, "cta_group", 1))
    return m


def _vp(handles, a, b, c):
    """Variant-pack dict {cuDNN tensor: buffer} keyed by the graph's tensors."""
    A, B, C = handles
    return {A: a, B: b, C: c}


def _build_plan(g, cfg, _name):
    """JIT-compile the recorded graph with a forced tile config."""
    return with_workspace(jit_from_cudnn_graph(g, config=cfg))


# ---------------------------------------------------------------------------
# Graph + data setup
# ---------------------------------------------------------------------------


def _graph_matmul(batch: int, M: int, N: int, K: int, dtype: str = "bf16"):
    g = cudnn.pygraph(
        io_data_type=_DTYPES[dtype][0],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[batch, M, K], stride=[M * K, K, 1])
    Bt = g.tensor(name="B", dim=[batch, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=Bt, name="mm")
    C.set_output(True).set_data_type(_DTYPES[_output_dtype(dtype)][0])
    return g, (A, Bt, C)


def _mkdata(batch: int, M: int, N: int, K: int, dtype: str = "bf16"):
    torch.manual_seed(0)
    tin, tout = _DTYPES[dtype][1], _DTYPES[_output_dtype(dtype)][1]
    a = torch.empty(batch, M, K, dtype=torch.int32).random_(-2, 2).to(dtype=tin, device="cuda")
    b = torch.empty(batch, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=tin, device="cuda")
    c = torch.empty(batch, M, N, dtype=tout, device="cuda")
    return a, b, c


def _make_cublas_call(dtype: str):
    if dtype != "fp8":
        return lambda t: torch.matmul(t[0], t[1].transpose(-1, -2), out=t[2])

    # Unit per-tensor scales give an ordinary FP8 GEMM, without block scaling.
    scale = torch.ones((), dtype=torch.float32, device="cuda")

    def call(t):
        a, b, c = t
        # _scaled_mm is 2D-only. Reuse the output buffers for each batch slice.
        for aa, bb, cc in zip(a, b, c):
            torch.ops.aten._scaled_mm.out(aa, bb.t(), scale, scale, out_dtype=cc.dtype, use_fast_accum=False, out=cc)

    return call


# ---------------------------------------------------------------------------
# Buffer rotation — defeat the hot-L2 artifact on small shapes
# ---------------------------------------------------------------------------


def _mkdata_pool(batch: int, M: int, N: int, K: int, nbuf: int, dtype: str = "bf16"):
    """`nbuf` independent (a, b, c) triples at distinct GMEM addresses (nbuf<=1
    returns the single base triple)."""
    a, b, c = _mkdata(batch, M, N, K, dtype)
    pool = [(a, b, c)]
    # Distinct allocations (clone → fresh GMEM); contents don't matter for timing.
    for _ in range(max(0, nbuf - 1)):
        pool.append((a.clone(), b.clone(), c.clone()))
    return pool


def _per_set_bytes(batch: int, M: int, N: int, K: int, dtype: str = "bf16") -> int:
    tin, tout = _DTYPES[dtype][1], _DTYPES[_output_dtype(dtype)][1]
    return batch * (tin.itemsize * (M * K + N * K) + tout.itemsize * M * N)


# ---------------------------------------------------------------------------
# Worker mode: just run the kernels under nsys profile, no Python timing.
# ---------------------------------------------------------------------------


def _nsys_worker(
    shape: str,
    configs: list[str],
    warmup: int,
    iters: int,
    nbuf: int,
    spec_map: dict,
    dtype: str = "bf16",
) -> None:
    """Inner mode re-exec'd under nsys: run each config (and cuBLAS) for
    warmup+iters launches, no timing — nsys captures it. Timed iters rotate
    across the pool; warmup uses a dedicated buffer."""
    B, M, N, K = (int(x) for x in shape.split(","))
    wa, wb, wc = _mkdata(B, M, N, K, dtype)  # dedicated warmup buffer
    pool = _mkdata_pool(B, M, N, K, nbuf, dtype)  # rotation pool for timed iters
    cublas_call = _make_cublas_call(dtype)

    print(f"[worker] shape={B}x{M}x{N}x{K}, dtype={dtype}, configs={len(configs)}, " f"warmup={warmup}, iters={iters}, rotate_buffers={nbuf}")

    # 1. cuBLAS.
    for _ in range(warmup):
        cublas_call((wa, wb, wc))
    for i in range(iters):
        cublas_call(pool[i % nbuf])
    torch.cuda.synchronize()

    # 2. each GEMM config.
    config_names = configs or list(spec_map)
    for name in config_names:
        spec = spec_for(name, spec_map)
        if spec is None:
            continue
        cfg = spec[0]
        try:
            g, h = _graph_matmul(B, M, N, K, dtype)
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
    parser.add_argument(
        "--dtype",
        choices=tuple(_DTYPES),
        default="bf16",
        help="input dtype (default bf16); fp8 = E4M3 with BF16 output; fp32 = FP32 cuBLAS reference only, TF32 disabled",
    )
    add_sweep_args(parser)
    args = parser.parse_args()
    validate_config_variant_args(parser, args)

    if not torch.cuda.is_available():
        print("No CUDA, skipping.")
        return 1

    parts = [int(x) for x in args.shape.split(",")]
    if len(parts) != 4:
        sys.exit("--shape must be B,M,N,K (four values; use B=1 for a plain matmul)")
    B, M, N, K = parts
    if args.dtype == "fp32":
        torch.set_float32_matmul_precision("highest")
    g, _ = _graph_matmul(B, M, N, K, args.dtype)
    spec_map = expand_config_variants(
        _build_spec_map(g),
        sweep_swap_ab=args.sweep_swap_ab,
        sweep_split_k=args.sweep_split_k,
    )
    per_set_bytes = _per_set_bytes(B, M, N, K, args.dtype)
    nbuf = resolve_nbuf(args.rotate_buffers, per_set_bytes)
    if args.dtype == "fp32" and not spec_map:
        print("  [FROST does not support FP32 inputs; running the FP32 cuBLAS reference only (TF32 disabled)]")
        if args.configs:
            print("  [--configs ignored: no FROST FP32 configurations]")
        args.configs = None

    if args._nsys_worker:
        configs = (
            select_config_variants(
                args.configs,
                spec_map,
                sweep_swap_ab=args.sweep_swap_ab,
                sweep_split_k=args.sweep_split_k,
            )
            if args.configs
            else []
        )
        _nsys_worker(args.shape, configs, args.warmup, args.iters, nbuf, spec_map, args.dtype)
        return 0

    flops = 2 * B * M * N * K
    config_names = select_config_variants(
        args.configs,
        spec_map,
        sweep_swap_ab=args.sweep_swap_ab,
        sweep_split_k=args.sweep_split_k,
    )

    dtype_label = "FP8 E4M3 in / BF16 out" if args.dtype == "fp8" else args.dtype.upper()
    print(f"\n=== matmul B={B} {M}x{N}x{K}  (~{flops / 1e9:.1f} GFLOP) — {dtype_label} ===")
    if args.dtype == "fp8" and B > 1:
        print(f"  [FP8 cuBLAS reference: {B} separate GEMM launches per batch]")

    report_pool(nbuf, per_set_bytes)

    rows: list[tuple[str, float, float, str]] = []  # (name, tflops, ms, note)
    t0 = time.time()

    def _fmt_row(name: str, tflops: float, ms: float, note: str, ref_tflops: float) -> str:
        if note:
            return f"  {name:50s} {'':8s}   {'':7s}   {note}"
        ratio = tflops / ref_tflops if ref_tflops > 0 else 0.0
        return f"  {name:50s} {tflops:8.2f}   {ms:7.3f}   {ratio:>9.2f}×"

    if args.timing == "nsys":
        print("  [timing: nsys median kernel duration]\n")
        inner_args = ["--shape", args.shape, "--dtype", args.dtype, "--warmup", str(args.warmup), "--iters", str(args.iters), "--rotate-buffers", str(nbuf)]
        if args.sweep_swap_ab:
            inner_args.append("--sweep-swap-ab")
        if args.sweep_split_k is not None:
            inner_args += ["--sweep-split-k", str(args.sweep_split_k)]
        if args.configs:
            inner_args += ["--configs", ",".join(config_names)]
        kern_times = nsys_run_and_parse(__file__, inner_args, tag="benchmark_matmul")

        cublas_hit = find_cublas_time(kern_times)
        if cublas_hit:
            cublas_name, cublas_ms = cublas_hit
            if args.dtype == "fp8":
                cublas_ms *= B  # nsys reports one 2D _scaled_mm launch.
            cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12
            print(f"  cuBLAS kernel: {cublas_name}")
        else:
            cublas_tflops, cublas_ms = float("nan"), float("nan")
            print("  cuBLAS kernel: not detected in nsys output")

        for name in config_names:
            spec = spec_for(name, spec_map)
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
        wa, wb, wc = _mkdata(B, M, N, K, args.dtype)  # dedicated warmup buffer
        pool = _mkdata_pool(B, M, N, K, nbuf, args.dtype)  # rotation pool for timed iters
        cublas_call = _make_cublas_call(args.dtype)
        if args.stream:
            print("  ▶ running cuBLAS reference ...", flush=True)
        cublas_ms = timer(
            rotating(cublas_call, pool),
            lambda: cublas_call((wa, wb, wc)),
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
            spec = spec_for(name, spec_map)
            cfg = spec[0] if spec else None
            if cfg is None:
                row = (name, 0.0, float("inf"), "UNKNOWN_CONFIG")
            elif ctx_dead:
                row = (name, 0.0, float("inf"), "skipped (CUDA context dead)")
            else:
                if args.stream:
                    print(f"  ▶ running {name} ...", flush=True)
                try:
                    g, h = _graph_matmul(B, M, N, K, args.dtype)
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
