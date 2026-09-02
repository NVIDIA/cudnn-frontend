# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the mixed-input-A mainloop matmul (narrow-loaded A cast to the wider
MMA dtype) vs the equivalent dense cuBLAS matmul on the widened operands.

Default: load=int8, tin=tout=bf16. Timing / buffer-rotation / CLI mirror
benchmark_matmul.py.

    python benchmark/gemm/frost/benchmark_matmul_mixed_input.py --load-dtype int8 --tin bf16 --tout bf16
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

# name -> (cudnn enum, torch dtype, element bytes, is_integer). load = narrow A
# storage; tin = MMA / B dtype; tout = output.
_DTYPES = {
    "int8": (cudnn.data_type.INT8, torch.int8, 1, True),
    "bf16": (cudnn.data_type.BFLOAT16, torch.bfloat16, 2, False),
    "fp16": (cudnn.data_type.HALF, torch.float16, 2, False),
    "fp32": (cudnn.data_type.FLOAT, torch.float32, 4, False),
    "e4m3": (cudnn.data_type.FP8_E4M3, torch.float8_e4m3fn, 1, False),
    "e5m2": (cudnn.data_type.FP8_E5M2, torch.float8_e5m2, 1, False),
}


def _dt(name: str):
    if name not in _DTYPES:
        sys.exit(f"unknown dtype {name!r}; choose from {sorted(_DTYPES)}")
    return _DTYPES[name]


# Enumerate config labels from a SUPPORTED bf16 mainloop chain — the real
# mixed-input graph is rejected at the mma-type funnel stage, returning no
# candidates. Labels reconstruct CONFIG_..._Nctamma so --configs is shared with
# benchmark_matmul.py.


def _enum_chain():
    M = N = K = 4096
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    Ai = g.identity(input=A, name="pw_in_mainloop0").set_data_type(cudnn.data_type.BFLOAT16)
    Bt = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=Ai, B=Bt, name="mm")
    C.set_output(True)
    return analyze(g)


def _build_spec_map():
    """Label -> (cfg, cta_group) for every mainloop strategy the funnel accepts."""
    chain = _enum_chain()
    m = {}
    for t, cfg in _candidates(chain):
        label = f"{cfg.name}_{t.cta_group}ctamma"
        m[label] = (cfg, t.cta_group)
    return m


_SPEC_MAP = _build_spec_map()


def _vp(handles, a, b, c):
    """Variant-pack dict {tensor: buffer}; `a` is the narrow (load-dtype) A root operand."""
    A, B, C = handles
    return {A: a, B: b, C: c}


def _build_plan(g, cfg, name):
    """JIT-compile the graph with a forced tile config -> callable kernel."""
    _, cta_group = spec_for(name, _SPEC_MAP)
    return jit_from_cudnn_graph(g, config=cfg, cta_group=cta_group)


# ---------------------------------------------------------------------------
# Graph + data setup
# ---------------------------------------------------------------------------


def _graph_mixed_input(batch: int, M: int, N: int, K: int, load_dt: str, tin_dt: str, tout_dt: str):
    """The mixed-input graph: identity(A_load) @ B_tin -> C_tout."""
    cu_load = _dt(load_dt)[0]
    cu_tin = _dt(tin_dt)[0]
    cu_tout = _dt(tout_dt)[0]
    g = cudnn.pygraph(
        io_data_type=cu_tin,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="aTensor", dim=[batch, M, K], stride=[M * K, K, 1], data_type=cu_load)
    Ai = g.identity(input=A, name="pw_in_mainloop0")
    Ai.set_data_type(cu_tin)
    Bt = g.tensor(name="bTensor", dim=[batch, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=Ai, B=Bt, name="mm")
    C.set_output(True).set_data_type(cu_tout)
    return g, (A, Bt, C)


def _mkdata(batch: int, M: int, N: int, K: int, load_dt: str, tin_dt: str, tout_dt: str):
    torch.manual_seed(0)
    _, t_load, _, load_int = _dt(load_dt)
    _, t_tin, _, _ = _dt(tin_dt)
    _, t_tout, _, _ = _dt(tout_dt)
    if load_int:
        a = torch.empty(batch, M, K, dtype=torch.int32).random_(-3, 3).to(dtype=t_load, device="cuda")
    else:
        a = torch.empty(batch, M, K, dtype=torch.int32).random_(-3, 3).to(dtype=t_tin, device="cuda").to(t_load)
    b = torch.empty(batch, N, K, dtype=torch.int32).random_(-3, 3).to(dtype=t_tin, device="cuda")
    c = torch.empty(batch, M, N, dtype=t_tout, device="cuda")
    return a, b, c


# cuBLAS reference: dense matmul on the widened operands (A cast load->tin).
def _cublas_ref(a, b, c, tin_dt: str):
    t_tin = _dt(tin_dt)[1]
    torch.matmul(a.to(t_tin), b.to(t_tin).transpose(-1, -2), out=c)


def _mkdata_pool(batch, M, N, K, load_dt, tin_dt, tout_dt, nbuf):
    a, b, c = _mkdata(batch, M, N, K, load_dt, tin_dt, tout_dt)
    pool = [(a, b, c)]
    for _ in range(max(0, nbuf - 1)):
        pool.append((a.clone(), b.clone(), c.clone()))
    return pool


def _per_set_bytes(batch, M, N, K, load_dt, tin_dt, tout_dt) -> int:
    bl = _dt(load_dt)[2]
    bin_ = _dt(tin_dt)[2]
    bout = _dt(tout_dt)[2]
    return batch * (bl * M * K + bin_ * N * K + bout * M * N)


# ---------------------------------------------------------------------------
# Worker mode
# ---------------------------------------------------------------------------


def _nsys_worker(shape, configs, warmup, iters, nbuf, load_dt, tin_dt, tout_dt) -> None:
    B, M, N, K = (int(x) for x in shape.split(","))
    wa, wb, wc = _mkdata(B, M, N, K, load_dt, tin_dt, tout_dt)
    pool = _mkdata_pool(B, M, N, K, load_dt, tin_dt, tout_dt, nbuf)

    print(
        f"[worker] shape={B}x{M}x{N}x{K}, load={load_dt} tin={tin_dt} tout={tout_dt}, "
        f"configs={len(configs)}, warmup={warmup}, iters={iters}, rotate_buffers={nbuf}"
    )

    # 1. cuBLAS reference — dense matmul on the widened operands.
    for _ in range(warmup):
        _cublas_ref(wa, wb, wc, tin_dt)
    for i in range(iters):
        a, b, c = pool[i % nbuf]
        _cublas_ref(a, b, c, tin_dt)
    torch.cuda.synchronize()

    # 2. each GEMM config.
    config_names = configs or list(_SPEC_MAP)
    for name in config_names:
        spec = spec_for(name, _SPEC_MAP)
        cfg = spec[0] if spec else None
        if cfg is None:
            continue
        try:
            g, h = _graph_mixed_input(B, M, N, K, load_dt, tin_dt, tout_dt)
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
    parser.add_argument("--shape", default="1,4096,4096,4096", help="B,M,N,K")
    parser.add_argument("--load-dtype", default="int8", help="A storage dtype (default int8)")
    parser.add_argument("--tin", default="bf16", help="compute / B dtype (default bf16)")
    parser.add_argument("--tout", default="bf16", help="output dtype (default bf16)")
    add_sweep_args(parser)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA, skipping.")
        return 1

    parts = [int(x) for x in args.shape.split(",")]
    if len(parts) != 4:
        sys.exit("--shape must be B,M,N,K (four values; use B=1 for a plain matmul)")
    B, M, N, K = parts
    load_dt, tin_dt, tout_dt = args.load_dtype, args.tin, args.tout
    per_set = _per_set_bytes(B, M, N, K, load_dt, tin_dt, tout_dt)
    nbuf = resolve_nbuf(args.rotate_buffers, per_set)

    if args._nsys_worker:
        configs = select_configs(args.configs, _SPEC_MAP) if args.configs else []
        _nsys_worker(args.shape, configs, args.warmup, args.iters, nbuf, load_dt, tin_dt, tout_dt)
        return 0

    flops = 2 * B * M * N * K
    config_names = select_configs(args.configs, _SPEC_MAP)

    print(f"\n=== mixed-input matmul B={B} {M}x{N}x{K}  (~{flops / 1e9:.1f} GFLOP) " f"— A={load_dt} -> {tin_dt} @ {tin_dt} -> {tout_dt} ===")

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
        inner_args = [
            "--shape",
            args.shape,
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
            "--rotate-buffers",
            str(nbuf),
            "--load-dtype",
            load_dt,
            "--tin",
            tin_dt,
            "--tout",
            tout_dt,
        ]
        if config_names:
            inner_args += ["--configs", ",".join(config_names)]
        kern_times = nsys_run_and_parse(__file__, inner_args, tag="benchmark_mixed")

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
            if spec is None:
                rows.append((name, 0.0, float("inf"), "UNKNOWN_CONFIG"))
                continue
            tok = kernel_match_token(spec[0], spec[1])
            matches = [(k, v) for k, v in kern_times.items() if tok in k]
            if not matches:
                rows.append((name, 0.0, float("inf"), "NO_KERNEL_IN_NSYS"))
                continue
            _, ms = max(matches, key=lambda x: x[1])
            rows.append((name, flops / (ms * 1e-3) / 1e12, ms, ""))
    else:
        timer = time_ms_delayed if args.timing == "delayed" else time_ms_events
        if args.timing == "delayed":
            print("  [timing: events bracketed around delayed back-to-back launches]\n")
        else:
            print("  [timing: torch.cuda.Event wall-clock around python loop " "(~50us/call overhead)]\n")
        wa, wb, wc = _mkdata(B, M, N, K, load_dt, tin_dt, tout_dt)
        pool = _mkdata_pool(B, M, N, K, load_dt, tin_dt, tout_dt, nbuf)
        if args.stream:
            print("  ▶ running cuBLAS reference ...", flush=True)
        cublas_ms = timer(
            rotating(lambda t: _cublas_ref(t[0], t[1], t[2], tin_dt), pool),
            lambda: _cublas_ref(wa, wb, wc, tin_dt),
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
                    g, h = _graph_mixed_input(B, M, N, K, load_dt, tin_dt, tout_dt)
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
        best_name, best_tflops, best_ms, _ = ok[0]
        print(f"\nbest GEMM: {best_name} — {best_tflops:.2f} TFLOPS " f"({best_tflops / cublas_tflops:.2f}× cuBLAS)")
    else:
        print("\nno GEMM config ran (mixed-input int8->bf16 mainloop may not be " "supported yet — see module docstring).")
    print(f"total: {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
