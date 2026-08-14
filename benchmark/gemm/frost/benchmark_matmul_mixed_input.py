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
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Callable

import cudnn  # noqa: F401
import cudnn.gemm.frost  # noqa: F401
import torch

from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
from cudnn.gemm.frost.tile_config import by_name as _by_name
from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.gemm.frost.kernel_registry import candidates as _candidates

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
    """Label -> (cfg, cta_group, scheduler) for every mainloop strategy the funnel accepts."""
    chain = _enum_chain()
    m = {}
    for t, cfg in _candidates(chain):
        label = f"{cfg.name}_{t.cta_group}ctamma" + ("_static" if t.static_sched else "")
        m[label] = (cfg, t.cta_group, t.scheduler)
    return m


_SPEC_MAP = _build_spec_map()

_LABEL_RE = re.compile(r"^(CONFIG_sm\d+_\d+x\d+x\d+_\d+x\d+x\d+_cluster\d+x\d+)_([12])ctamma(_static)?$")


def _spec_for(name):
    """(geometry cfg, cta_group, scheduler) for a --configs label, or None.

    The sweep set comes from the registry funnel over CATALOG; a label naming a
    geometry outside it (e.g. a num_mma_m > 1 tile, which `by_name` synthesizes) is
    still runnable, so parse it rather than reporting UNKNOWN_CONFIG."""
    spec = _SPEC_MAP.get(name)
    if spec is not None:
        return spec
    m = _LABEL_RE.match(name)
    if m is None:
        return None
    try:
        cfg = _by_name(m.group(1))
    except (KeyError, NotImplementedError):
        return None
    return cfg, int(m.group(2)), "static" if m.group(3) else "clc"


def _vp(handles, a, b, c):
    """Variant-pack dict {tensor: buffer}; `a` is the narrow (load-dtype) A root operand."""
    A, B, C = handles
    return {A: a, B: b, C: c}


def _build_plan(g, cfg, name):
    """JIT-compile the graph with a forced tile config -> callable kernel."""
    return jit_from_cudnn_graph(g, config=cfg, cta_group=_spec_for(name)[1], scheduler=_spec_for(name)[2])


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


# Buffer rotation — rotate timed launches across independent tensor copies so a
# kernel doesn't re-read the prior launch's data from a hot L2 (inflates
# small-shape TFLOPS). See benchmark_matmul.py.
_L2_BYTES_B200 = 126 * 1024 * 1024


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


_AUTO_POOL_BUDGET_BYTES = 4 * 1024 * 1024 * 1024
_AUTO_NBUF_CAP = 1024


def _auto_nbuf(batch, M, N, K, load_dt, tin_dt, tout_dt) -> int:
    per_set = _per_set_bytes(batch, M, N, K, load_dt, tin_dt, tout_dt)
    target = int(1.5 * _L2_BYTES_B200)
    nbuf = max(2, -(-target // per_set))
    budget = _AUTO_POOL_BUDGET_BYTES
    if torch.cuda.is_available():
        free, _total = torch.cuda.mem_get_info()
        budget = min(budget, free // 2)
    max_by_budget = max(1, budget // per_set)
    return max(1, min(nbuf, max_by_budget, _AUTO_NBUF_CAP))


def _resolve_nbuf(spec, batch, M, N, K, load_dt, tin_dt, tout_dt) -> int:
    if spec.strip().lower() == "auto":
        return _auto_nbuf(batch, M, N, K, load_dt, tin_dt, tout_dt)
    return max(1, int(spec))


def _rotating(fn_of_buf: Callable, pool: list) -> Callable:
    n = len(pool)
    return lambda i: fn_of_buf(pool[i % n])


# Timing (events / delayed) — identical to benchmark_matmul.py.


def _time_ms_events(timed_fn, warmup_fn, *, warmup, iters) -> float:
    for _ in range(warmup):
        warmup_fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(iters):
        timed_fn(i)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _time_ms_delayed(timed_fn, warmup_fn, *, warmup, iters) -> float:
    for _ in range(warmup):
        warmup_fn()
    torch.cuda.synchronize()
    delay_cycles = max(int(1e8), int((iters * 0.05 + 20.0) * 1.7e6))
    torch.cuda._sleep(delay_cycles)
    post_warmup = max(5, warmup)
    for _ in range(post_warmup):
        warmup_fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(iters):
        timed_fn(i)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


# ---------------------------------------------------------------------------
# nsys mode
# ---------------------------------------------------------------------------


def _nsys_run_and_parse(shape, configs, warmup, iters, nbuf, load_dt, tin_dt, tout_dt) -> dict[str, float]:
    nsys = "/usr/local/bin/nsys" if os.path.exists("/usr/local/bin/nsys") else shutil.which("nsys")
    if nsys is None:
        sys.exit("nsys not found — install nsight-systems or use the default events mode.")

    workdir = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"benchmark_mixed_nsys_{os.getpid()}")
    os.makedirs(workdir, exist_ok=True)
    report_prefix = os.path.join(workdir, "report")

    nsys_env = os.environ.copy()
    nsys_env.setdefault("TMPDIR", os.environ.get("TMPDIR", tempfile.gettempdir()))

    inner = [
        sys.executable,
        "-u",
        os.path.abspath(__file__),
        "--_nsys-worker",
        "--shape",
        shape,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--rotate-buffers",
        str(nbuf),
        "--load-dtype",
        load_dt,
        "--tin",
        tin_dt,
        "--tout",
        tout_dt,
    ]
    if configs:
        inner += ["--configs", ",".join(configs)]

    profile_cmd = [
        nsys,
        "profile",
        "-o",
        report_prefix,
        "--force-overwrite=true",
        "--cuda-um-cpu-page-faults=false",
        "--cuda-um-gpu-page-faults=false",
        "--trace=cuda",
    ] + inner

    print(f"  + {' '.join(profile_cmd)}\n")
    proc = subprocess.run(profile_cmd, capture_output=True, text=True, env=nsys_env)
    if proc.returncode != 0:
        print("nsys stdout:\n" + proc.stdout)
        print("nsys stderr:\n" + proc.stderr, file=sys.stderr)
        sys.exit(f"nsys profile exited {proc.returncode}")

    stats_cmd = [
        nsys,
        "stats",
        "--report",
        "cuda_gpu_kern_sum",
        "--force-export=true",
        report_prefix + ".nsys-rep",
    ]
    proc = subprocess.run(stats_cmd, capture_output=True, text=True, env=nsys_env)
    if proc.returncode != 0:
        print("nsys stats stdout:\n" + proc.stdout)
        print("nsys stats stderr:\n" + proc.stderr, file=sys.stderr)
        sys.exit(f"nsys stats exited {proc.returncode}")

    return _parse_nsys_stats(proc.stdout)


def _parse_nsys_stats(text: str) -> dict[str, float]:
    lines = text.splitlines()
    header_i = None
    for i, ln in enumerate(lines):
        if "Med (" in ln and "Name" in ln and ("ns)" in ln or "us)" in ln or "ms)" in ln):
            header_i = i
            break
    if header_i is None:
        sys.exit("could not find kernel-summary header in nsys stats output:\n  " + "\n  ".join(lines[:60]))

    m_unit = re.search(r"Med \((\w+)\)", lines[header_i])
    unit = m_unit.group(1) if m_unit else "ns"
    unit_div = {"ns": 1e6, "us": 1e3, "ms": 1.0, "s": 1e-3}.get(unit, 1e6)

    NUM_NUMERIC_COLS = 8
    MED_COL = 4

    result: dict[str, float] = {}
    in_data = False
    for j in range(header_i + 1, len(lines)):
        row = lines[j]
        stripped = row.strip()
        if not stripped:
            if in_data:
                break
            continue
        if set(stripped) <= set("- "):
            in_data = True
            continue
        if not in_data:
            continue
        if stripped.startswith("**") or stripped.startswith("##"):
            break
        toks = stripped.split()
        if len(toks) <= NUM_NUMERIC_COLS:
            continue
        try:
            med = float(toks[MED_COL].replace(",", ""))
        except ValueError:
            continue
        name = " ".join(toks[NUM_NUMERIC_COLS:]).rstrip()
        if not name:
            continue
        result[name] = med / unit_div
    return result


def _match_kernel_name(kern_name: str, config_name: str) -> bool:
    return config_name in kern_name


def _find_cublas_time(kern_times: dict[str, float]):
    cands = [(k, v) for k, v in kern_times.items() if k.startswith("nvjet_")]
    if not cands:
        return None
    return max(cands, key=lambda x: x[1])


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
        spec = _spec_for(name)
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
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--load-dtype", default="int8", help="A storage dtype (default int8)")
    parser.add_argument("--tin", default="bf16", help="compute / B dtype (default bf16)")
    parser.add_argument("--tout", default="bf16", help="output dtype (default bf16)")
    parser.add_argument(
        "--configs",
        default=None,
        help="comma-separated config names (default: every mainloop CATALOG entry)",
    )
    parser.add_argument("--timing", choices=("delayed", "events", "nsys"), default="delayed")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="print each config's result as it finishes (events/delayed only)",
    )
    parser.add_argument(
        "--rotate-buffers",
        default="auto",
        metavar="N",
        help="independent tensor copies to rotate timed launches across " "(default 'auto'; 1 disables).",
    )
    parser.add_argument("--_nsys-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA, skipping.")
        return 1

    parts = [int(x) for x in args.shape.split(",")]
    if len(parts) != 4:
        sys.exit("--shape must be B,M,N,K (four values; use B=1 for a plain matmul)")
    B, M, N, K = parts
    load_dt, tin_dt, tout_dt = args.load_dtype, args.tin, args.tout
    nbuf = _resolve_nbuf(args.rotate_buffers, B, M, N, K, load_dt, tin_dt, tout_dt)

    if getattr(args, "_nsys_worker"):
        configs = [c.strip() for c in args.configs.split(",")] if args.configs else []
        _nsys_worker(args.shape, configs, args.warmup, args.iters, nbuf, load_dt, tin_dt, tout_dt)
        return 0

    flops = 2 * B * M * N * K
    config_names = [c.strip() for c in args.configs.split(",")] if args.configs else list(_SPEC_MAP)

    print(f"\n=== mixed-input matmul B={B} {M}x{N}x{K}  (~{flops / 1e9:.1f} GFLOP) " f"— A={load_dt} -> {tin_dt} @ {tin_dt} -> {tout_dt} ===")

    if nbuf > 1:
        footprint = _per_set_bytes(B, M, N, K, load_dt, tin_dt, tout_dt) * nbuf
        print(f"  [rotate-buffers: {nbuf} copies/tensor, " f"{footprint / 1024 / 1024:.0f} MB pool]")
        if footprint < _L2_BYTES_B200:
            print(f"  [WARNING: pool ({footprint / 1024 / 1024:.0f} MB) < B200 L2 " f"(~{_L2_BYTES_B200 / 1024 / 1024:.0f} MB) — bump --rotate-buffers.]")
    else:
        print("  [rotate-buffers: disabled (1) — small-shape TFLOPS may be hot-L2-inflated]")

    rows: list[tuple[str, float, float, str]] = []
    t0 = time.time()

    def _fmt_row(name, tflops, ms, note, ref_tflops) -> str:
        if note:
            return f"  {name:50s} {'':8s}   {'':7s}   {note}"
        ratio = tflops / ref_tflops if ref_tflops > 0 else 0.0
        return f"  {name:50s} {tflops:8.2f}   {ms:7.3f}   {ratio:>9.2f}×"

    if args.timing == "nsys":
        print("  [timing: nsys median kernel duration]\n")
        kern_times = _nsys_run_and_parse(
            args.shape,
            config_names,
            args.warmup,
            args.iters,
            nbuf,
            load_dt,
            tin_dt,
            tout_dt,
        )

        cublas_hit = _find_cublas_time(kern_times)
        if cublas_hit:
            cublas_name, cublas_ms = cublas_hit
            cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12
            print(f"  cuBLAS kernel: {cublas_name}")
        else:
            cublas_tflops, cublas_ms = float("nan"), float("nan")
            print("  cuBLAS kernel: not detected in nsys output")

        for name in config_names:
            spec = _spec_for(name)
            cfg = spec[0] if spec else None
            if cfg is None:
                rows.append((name, 0.0, float("inf"), "UNKNOWN_CONFIG"))
                continue
            matches = [(k, v) for k, v in kern_times.items() if _match_kernel_name(k, name)]
            if not matches:
                rows.append((name, 0.0, float("inf"), "NO_KERNEL_IN_NSYS"))
                continue
            _, ms = max(matches, key=lambda x: x[1])
            rows.append((name, flops / (ms * 1e-3) / 1e12, ms, ""))
    else:
        timer = _time_ms_delayed if args.timing == "delayed" else _time_ms_events
        if args.timing == "delayed":
            print("  [timing: events bracketed around delayed back-to-back launches]\n")
        else:
            print("  [timing: torch.cuda.Event wall-clock around python loop " "(~50us/call overhead)]\n")
        wa, wb, wc = _mkdata(B, M, N, K, load_dt, tin_dt, tout_dt)
        pool = _mkdata_pool(B, M, N, K, load_dt, tin_dt, tout_dt, nbuf)
        if args.stream:
            print("  ▶ running cuBLAS reference ...", flush=True)
        cublas_ms = timer(
            _rotating(lambda t: _cublas_ref(t[0], t[1], t[2], tin_dt), pool),
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
            spec = _spec_for(name)
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
                        _rotating(
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
