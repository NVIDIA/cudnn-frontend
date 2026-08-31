# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Best sm120 config vs cuBLAS over a fixed M/N/K grid (bf16, batch 1).

For every shape in the 3x3x3 grid M, N, K in {4096, 8192, 16384} (27 shapes,
M outermost / K innermost), the sweep times EVERY feasible sm120 config plus
cuBLAS, keeps the fastest config per shape, and prints one summary table:
shape -> best config, its TFLOPS/ms, cuBLAS TFLOPS/ms, and the ratio. Needs a
CC 12.x GPU.

Timing modes: delayed (default) / events. `--rotate-buffers` defeats hot-L2
inflation. The full sweep is a few hundred configs per shape; narrow it with a
glob while iterating:

    python benchmark/gemm/frost/benchmark_matmul_sm120.py
    python benchmark/gemm/frost/benchmark_matmul_sm120.py --configs 'CONFIG_sm120_128x*' --stream
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
    resolve_nbuf,
    rotating,
    select_configs,
    spec_for,
    time_ms_delayed,
    time_ms_events,
)

_SHAPE_AXIS = (4096, 8192, 16384)
_SHAPES = tuple((m, n, k) for m in _SHAPE_AXIS for n in _SHAPE_AXIS for k in _SHAPE_AXIS)


def _sm120_spec_map(M: int, N: int, K: int) -> dict:
    """Label -> (geometry cfg, 1) for every sm120 config the funnel accepts for
    THIS problem shape (bf16 TN, the sweep's dtype/layout), via the same
    ``candidates`` gates the auto path applies."""
    chain = _FC(
        matmul=_MS(
            M=M,
            N=N,
            K=K,
            a_major="k",
            b_major="k",
            a_dtype="bf16",
            b_dtype="bf16",
            accum_dtype="fp32",
        ),
        output_specs=[_OS(source_ref=-1, dtype="bf16")],
    )
    return {cfg.name: (cfg, 1) for t, cfg in _candidates(chain) if cfg.pipeline == "sm120"}


def _cublas_provenance() -> str:
    """The libcublas file actually SERVING the calls. torch's wheel pins its own
    copy via RPATH (LD_LIBRARY_PATH cannot override it); an LD_PRELOAD'd build
    wins symbol binding even though both files stay mapped, so resolve a cuBLAS
    symbol's address and report the mapping it lands in."""
    import ctypes

    paths = sorted({ln.split()[-1] for ln in open("/proc/self/maps") if "libcublas.so" in ln})
    try:
        addr = ctypes.cast(ctypes.CDLL(None).cublasCreate_v2, ctypes.c_void_p).value
        for ln in open("/proc/self/maps"):
            if "libcublas.so" in ln:
                lo, hi = (int(x, 16) for x in ln.split()[0].split("-"))
                if lo <= addr < hi:
                    return ln.split()[-1]
    except Exception:
        pass  # no globally visible cuBLAS symbol: the single mapped copy serves
    return paths[0] if len(paths) == 1 else " / ".join(paths)


def _vp(handles, a, b, c):
    """Variant-pack dict {cuDNN tensor: buffer} keyed by the graph's tensors."""
    A, B, C = handles
    return {A: a, B: b, C: c}


def _graph_matmul(M: int, N: int, K: int):
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    Bt = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=Bt, name="mm")
    C.set_output(True)
    return g, (A, Bt, C)


def _mkdata(M: int, N: int, K: int):
    torch.manual_seed(0)
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    b = torch.empty(1, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    return a, b, c


def _mkdata_pool(M: int, N: int, K: int, nbuf: int):
    """`nbuf` independent (a, b, c) triples at distinct GMEM addresses."""
    a, b, c = _mkdata(M, N, K)
    pool = [(a, b, c)]
    for _ in range(max(0, nbuf - 1)):
        pool.append((a.clone(), b.clone(), c.clone()))
    return pool


def _per_set_bytes(M: int, N: int, K: int) -> int:
    # BF16 = 2 bytes/elem; a:(M,K) b:(N,K) c:(M,N).
    return 2 * (M * K + N * K + M * N)


def _short(name: str) -> str:
    """Table label: the CONFIG_sm120_ prefix is redundant in an sm120-only table."""
    return name.removeprefix("CONFIG_sm120_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_sweep_args(parser, nsys=False)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA, skipping.")
        return 1
    major, minor = torch.cuda.get_device_capability()
    if major != 12:
        print(f"This sweep is the sm120 (CC 12.x) family; the active GPU is sm_{major}{minor}. " "Use benchmark_matmul.py for the tcgen05 families.")
        return 1

    timer = time_ms_delayed if args.timing == "delayed" else time_ms_events
    print(f"\n=== sm120 best-config vs cuBLAS — {len(_SHAPES)} shapes, " f"M/N/K in {_SHAPE_AXIS}, BF16, batch 1 ===")
    print(f"  cuBLAS library: {_cublas_provenance()}")
    print(f"  [timing: {args.timing}; warmup={args.warmup} iters={args.iters}; " "per shape: every feasible sm120 config, best kept]\n")

    results = []  # (M, N, K, best_name, best_tf, best_ms, cublas_tf, cublas_ms, n_swept, n_err)
    t0 = time.time()

    for idx, (M, N, K) in enumerate(_SHAPES, 1):
        flops = 2 * M * N * K
        spec_map = _sm120_spec_map(M, N, K)
        config_names = select_configs(args.configs, spec_map)
        nbuf = resolve_nbuf(args.rotate_buffers, _per_set_bytes(M, N, K))
        wa, wb, wc = _mkdata(M, N, K)
        pool = _mkdata_pool(M, N, K, nbuf)

        cublas_ms = timer(
            rotating(lambda t: torch.matmul(t[0], t[1].transpose(-1, -2), out=t[2]), pool),
            lambda: torch.matmul(wa, wb.transpose(-1, -2), out=wc),
            warmup=args.warmup,
            iters=args.iters,
        )
        cublas_tf = flops / (cublas_ms * 1e-3) / 1e12

        best_name, best_ms = None, float("inf")
        n_err = 0
        for name in config_names:
            spec = spec_for(name, spec_map)
            if spec is None or spec[0].pipeline != "sm120":
                n_err += 1
                continue
            try:
                g, h = _graph_matmul(M, N, K)
                plan = jit_from_cudnn_graph(g, config=spec[0])
                ms = timer(
                    rotating(lambda t, _plan=plan, _h=h: _plan(_vp(_h, t[0], t[1], t[2])), pool),
                    lambda _plan=plan, _h=h: _plan(_vp(_h, wa, wb, wc)),
                    warmup=args.warmup,
                    iters=args.iters,
                )
            except Exception as e:
                n_err += 1
                msg = str(e).splitlines()[0][:60] if str(e) else type(e).__name__
                if args.stream:
                    print(f"    ERR {name}: {msg}", flush=True)
                # An async device fault sticky-poisons the CUDA context for the
                # rest of the process — nothing later can be trusted, so stop.
                if any(s in str(e) for s in ("illegal memory access", "unspecified launch failure", "CUDA_ERROR_LAUNCH_FAILED")):
                    print(f"CUDA context poisoned at {name} ({M}x{N}x{K}); aborting the grid.")
                    return 1
                continue
            if args.stream:
                print(f"    {name:62s} {flops / (ms * 1e-3) / 1e12:8.2f} TF  {ms:8.3f} ms", flush=True)
            if ms < best_ms:
                best_name, best_ms = name, ms

        best_tf = flops / (best_ms * 1e-3) / 1e12 if best_name else 0.0
        results.append((M, N, K, best_name, best_tf, best_ms, cublas_tf, cublas_ms, len(config_names), n_err))
        ratio = best_tf / cublas_tf if cublas_tf > 0 and best_name else 0.0
        print(
            f"[{idx:2d}/{len(_SHAPES)}] {M:>6d}x{N:>6d}x{K:>6d}  "
            f"best {_short(best_name) if best_name else 'n/a':44s} "
            f"{best_tf:7.2f} TF  vs cuBLAS {cublas_tf:7.2f} TF  ({ratio:4.2f}×)"
            f"{f'  [{n_err} err]' if n_err else ''}",
            flush=True,
        )

        # Big shapes hold multi-GB pools: release before the next allocation.
        del pool, wa, wb, wc
        torch.cuda.empty_cache()

    print("\n" + "=" * 132)
    print(f"  {'M':>6s} {'N':>6s} {'K':>6s}   {'best sm120 config':44s} {'TFLOPS':>8s} {'ms':>9s}   {'cuBLAS TF':>9s} {'ms':>9s}   {'sm120/cuBLAS':>12s}")
    print("=" * 132)
    ratios = []
    for M, N, K, best_name, best_tf, best_ms, cublas_tf, cublas_ms, _n, _e in results:
        if best_name is None:
            print(f"  {M:>6d} {N:>6d} {K:>6d}   {'(no config ran)':44s}")
            continue
        ratio = best_tf / cublas_tf if cublas_tf > 0 else 0.0
        ratios.append(ratio)
        print(f"  {M:>6d} {N:>6d} {K:>6d}   {_short(best_name):44s} {best_tf:8.2f} {best_ms:9.3f}   " f"{cublas_tf:9.2f} {cublas_ms:9.3f}   {ratio:11.2f}×")
    print("=" * 132)
    if ratios:
        geomean = 1.0
        for r in ratios:
            geomean *= r
        geomean **= 1.0 / len(ratios)
        print(f"  geomean sm120-best / cuBLAS over {len(ratios)} shapes: {geomean:.3f}×")
    print(f"total: {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
