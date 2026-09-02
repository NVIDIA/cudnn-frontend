# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark block-scale-compatible CATALOG configs on one shape (default nvfp4
in / BF16 out) vs a reference. Perf only, no CPU verify.

    python benchmark/gemm/frost/benchmark_block_scale_matmul.py
"""

from __future__ import annotations

import argparse
import sys
import time

import cudnn  # noqa: F401
import cudnn.gemm.frost  # noqa: F401
import torch

from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
from cudnn.gemm.frost.tile_config import CATALOG as _CATALOG

from benchmark_utils import (
    add_sweep_args,
    ceil_div,
    find_cublas_time,
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


def _build_spec_map():
    """Legacy label -> (geometry cfg, cta_group) for block-scale
    strategies (geometry must satisfy the SF 128x4 swizzle; K-tile bytes are
    arch-keyed: 128 on sm100, 384 on sm103)."""
    m = {}
    for cfg in _CATALOG:
        kb_want = 384 if cfg.pipeline == "sm103" else 128
        if cfg.mma_inst_m % 128 or cfg.mma_inst_n % 128 or cfg.cta_tile_k_bytes != kb_want:
            continue
        for cg in (1, 2):
            if cg == 2 and (cfg.cgrp_size_m % 2 or cfg.cta_tile_m == 64):
                continue
            m[f"{cfg.name}_{cg}ctamma"] = (cfg, cg)
    return m


_SPEC_MAP = _build_spec_map()


def _vp_bs(handles, a, b, c, sfa, sfb):
    """Block-scale single-GEMM variant-pack dict keyed by the graph's tensors."""
    A, B, C, SFA, SFB = handles
    return {A: a, B: b, SFA: sfa, SFB: sfb, C: c}


def _build_plan(g, cfg, name):
    """JIT-compile the recorded graph with a forced tile config."""
    _, cta_group = spec_for(name, _SPEC_MAP)
    return jit_from_cudnn_graph(g, config=cfg, cta_group=cta_group)


# Combo table (input dtype family + scale dtype + block size)

_COMBOS = {
    # combo : (is_fp4, block_size, a_dtype, sf_dtype)
    "nvfp4": (True, 16, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E4M3),
    "mxfp4": (True, 32, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E8M0),
    "mxfp8": (False, 32, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E8M0),
}


def _graph_block_scale(batch: int, M: int, N: int, K: int, combo: str):
    is_fp4, block_size, a_dt, sf_dt = _COMBOS[combo]
    sf_k = K // block_size
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[batch, M, K], stride=[M * K, K, 1], data_type=a_dt)
    Bt = g.tensor(name="B", dim=[batch, K, N], stride=[K * N, 1, K], data_type=a_dt)
    # SFs must be declared F8_128x4-reordered or the block-scale gate rejects it.
    SFA = g.tensor(
        name="SFA",
        dim=[batch, M, sf_k],
        stride=[M * sf_k, sf_k, 1],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    SFB = g.tensor(
        name="SFB",
        dim=[batch, sf_k, N],
        stride=[sf_k * N, 1, sf_k],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    Ad = g.block_scale_dequantize(input=A, descale=SFA, block_size=[1, block_size])
    Bd = g.block_scale_dequantize(input=Bt, descale=SFB, block_size=[block_size, 1])
    C = g.matmul(A=Ad, B=Bd, name="mm")
    C.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    return g, (A, Bt, C, SFA, SFB)


def _mkdata(batch: int, M: int, N: int, K: int, combo: str):
    """Block-scale runtime tensors: (a, b, c, sfa_blocked, sfb_blocked). a/b are
    packed FP4 or FP8; c is BF16; SFA/SFB are F8_128x4-reordered per batch."""
    dev = "cuda"
    torch.manual_seed(0)
    is_fp4, block_size, _, _ = _COMBOS[combo]
    sf_k = K // block_size

    if is_fp4:
        a_u8 = torch.randint(0, 256, (batch, M, K // 2), dtype=torch.uint8, device=dev)
        b_u8 = torch.randint(0, 256, (batch, N, K // 2), dtype=torch.uint8, device=dev)
        a = a_u8.view(torch.float4_e2m1fn_x2)
        b = b_u8.view(torch.float4_e2m1fn_x2)
    else:
        a = (torch.randn(batch, M, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        b = (torch.randn(batch, N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)

    if combo == "nvfp4":
        sfa_log = torch.randint(1, 4, (batch, M, sf_k), device=dev).to(torch.float8_e4m3fn)
        sfb_log = torch.randint(1, 4, (batch, N, sf_k), device=dev).to(torch.float8_e4m3fn)
    else:
        sfa_log = rand_e8m0((batch, M, sf_k), dev)
        sfb_log = rand_e8m0((batch, N, sf_k), dev)

    c = torch.empty(batch, M, N, dtype=torch.bfloat16, device=dev)
    # The blocked blob carries the PADDED dims (whole 128-row x 4-SF-K atoms) —
    # that is what the kernel reads. The graph keeps the logical [batch, M, sf_k].
    sf_k_pad = ceil_div(sf_k, 4) * 4
    sfa = torch.cat([to_blocked(sfa_log[i]) for i in range(batch)]).view(batch, ceil_div(M, 128) * 128, sf_k_pad)
    sfb = torch.cat([to_blocked(sfb_log[i]) for i in range(batch)]).view(batch, ceil_div(N, 128) * 128, sf_k_pad)
    return a, b, c, sfa, sfb


def _mkdata_pool(batch: int, M: int, N: int, K: int, combo: str, nbuf: int):
    """`nbuf` independent (a, b, c, sfa, sfb) sets at distinct GMEM addresses."""
    base = _mkdata(batch, M, N, K, combo)
    pool = [base]
    for _ in range(max(0, nbuf - 1)):
        pool.append(tuple(t.clone() for t in base))
    return pool


def _scaled_mm_ref(batch: int, M: int, N: int, K: int, combo: str, verbose: bool = True):
    """(label, call) for cuBLAS's OWN block-scaled GEMM of this combo, via
    torch.nn.functional.scaled_mm (cuBLASLt) — the apples-to-apples reference.
    Returns None if the env can't run it (some cuBLASLt builds return no
    heuristic algorithm → scaled_mm raises CUBLAS_STATUS_NOT_INITIALIZED).

    Scale layout: cuBLASLt block scaling on sm100 wants per-block scales in the
    padded SWIZZLE_32_4_4 buffer (numel round_up(M,128)·round_up(ceil(K/bs),4))
    plus, for nvfp4, a per-tensor fp32 global scale; mat_a row-major, mat_b
    column-major. Values random — a perf yardstick, not a correctness check."""
    import torch.nn.functional as F
    from torch.nn.functional import ScalingType, SwizzleType

    dev = "cuda"
    is_fp4, bs, _, _ = _COMBOS[combo]
    ru = lambda x, m: ((x + m - 1) // m) * m
    cd = lambda a, b: (a + b - 1) // b

    if is_fp4:
        a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=dev).view(torch.float4_e2m1fn_x2)
        b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev).view(torch.float4_e2m1fn_x2).t()
    else:
        a = torch.randn(M, K, device=dev).to(torch.float8_e4m3fn)
        b = torch.randn(N, K, device=dev).to(torch.float8_e4m3fn).t()

    na = ru(M, 128) * ru(cd(K, bs), 4)
    nb = ru(N, 128) * ru(cd(K, bs), 4)
    if combo == "nvfp4":
        sa = torch.randn(na, device=dev).to(torch.float8_e4m3fn)
        sb = torch.randn(nb, device=dev).to(torch.float8_e4m3fn)
        ga = torch.ones(1, device=dev)
        gb = torch.ones(1, device=dev)
        recipe = [ScalingType.BlockWise1x16, ScalingType.TensorWise]
        scA, scB = [sa, ga], [sb, gb]
        # scaled_mm wants one swizzle PER scale in the recipe.
        sw = [SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE]
    else:  # mxfp4 / mxfp8 — no global scale
        sa = torch.randn(na, device=dev).to(torch.float8_e8m0fnu)
        sb = torch.randn(nb, device=dev).to(torch.float8_e8m0fnu)
        recipe = [ScalingType.BlockWise1x32]
        scA, scB = [sa], [sb]
        sw = [SwizzleType.SWIZZLE_32_4_4]

    # scaled_mm has no batched form, so a batched ref is B back-to-back single-GEMM
    # calls (same operands); FLOPS counts all B so throughput stays comparable.
    def call():
        out = None
        for _ in range(batch):
            out = F.scaled_mm(a, b, scA, recipe, scB, recipe, sw, sw, None, torch.bfloat16)
        return out

    try:
        call()
        torch.cuda.synchronize()
    except Exception as e:
        # Print it — an unrunnable env and an API-shape bug look identical otherwise.
        if verbose:
            print(f"  [scaled_mm '{combo}' reference unavailable: {type(e).__name__}: {e}]")
        return None
    suffix = f" ×{batch}" if batch > 1 else ""
    return f"cuBLAS {combo} scaled_mm{suffix}", call


def _make_reference(batch: int, M: int, N: int, K: int, combo: str, ref_mode: str, verbose: bool = True):
    """(label, call) for the throughput reference row. Each call allocates its OWN
    tensors (repeated calls build the rotation pool); verbose=False suppresses the
    fallback note. ref_mode: scaled_mm (cuBLAS block-scaled, errors if unrunnable)
    / bf16 (dense BF16) / auto (scaled_mm, falling back to BF16)."""
    dev = "cuda"
    if ref_mode in ("auto", "scaled_mm"):
        ref = _scaled_mm_ref(batch, M, N, K, combo, verbose=verbose)
        if ref is not None:
            return ref
        if ref_mode == "scaled_mm":
            sys.exit(
                f"--ref scaled_mm: cuBLAS block-scaled GEMM for '{combo}' is not "
                f"runnable here — torch.nn.functional.scaled_mm raised (see the "
                f"message above). Use --ref bf16, or --ref auto to fall back "
                f"automatically."
            )
        if verbose:
            print("  [note: scaled_mm reference unavailable in this env — " "falling back to dense BF16 cuBLAS]")
    # BF16 fallback: one batched matmul covers all B (torch.matmul is batched).
    a = torch.randn(batch, M, K, dtype=torch.bfloat16, device=dev)
    b = torch.randn(batch, N, K, dtype=torch.bfloat16, device=dev)
    c = torch.empty(batch, M, N, dtype=torch.bfloat16, device=dev)
    label = "BF16 cuBLAS (fallback)" if ref_mode == "auto" else "BF16 cuBLAS"
    return label, (lambda: torch.matmul(a, b.transpose(-1, -2), out=c))


def _make_reference_pool(batch: int, M: int, N: int, K: int, combo: str, ref_mode: str, nbuf: int):
    """(label, warmup_call, [timed_call × nbuf]). Warmup uses a dedicated buffer;
    timed calls each allocate independent tensors so the loop can rotate."""
    label, warmup_call = _make_reference(batch, M, N, K, combo, ref_mode, verbose=True)
    timed_calls = [_make_reference(batch, M, N, K, combo, ref_mode, verbose=False)[1] for _ in range(nbuf)]
    return label, warmup_call, timed_calls


# Worker mode: run the kernels under nsys profile, no Python timing.


def _nsys_worker(shape, combo, configs, warmup, iters, ref_mode, nbuf) -> None:
    B, M, N, K = (int(x) for x in shape.split(","))
    wset = _mkdata(B, M, N, K, combo)  # dedicated warmup buffer
    pool = _mkdata_pool(B, M, N, K, combo, nbuf)  # rotation pool
    ref_label, ref_warmup, ref_timed = _make_reference_pool(B, M, N, K, combo, ref_mode, nbuf)

    print(f"[worker] shape={B}x{M}x{N}x{K}, combo={combo}, ref={ref_label}, " f"configs={len(configs)}, warmup={warmup}, iters={iters}, rotate_buffers={nbuf}")

    # 1. reference kernel (cuBLAS block-scaled, or BF16 fallback).
    for _ in range(warmup):
        ref_warmup()
    for i in range(iters):
        ref_timed[i % nbuf]()
    torch.cuda.synchronize()

    # 2. each block-scale config.
    config_names = configs or list(_SPEC_MAP)
    for name in config_names:
        spec = spec_for(name, _SPEC_MAP)
        cfg = spec[0] if spec else None
        if cfg is None:
            continue
        try:
            g, h = _graph_block_scale(B, M, N, K, combo)
            plan = _build_plan(g, cfg, name)
            wa, wb, wc, wsfa, wsfb = wset
            for _ in range(warmup):
                plan(_vp_bs(h, wa, wb, wc, wsfa, wsfb))
            for i in range(iters):
                s = pool[i % nbuf]
                plan(_vp_bs(h, s[0], s[1], s[2], s[3], s[4]))
            torch.cuda.synchronize()
            print(f"[worker] OK   {name}")
        except Exception as e:
            print(f"[worker] FAIL {name}: {type(e).__name__}: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        default="1,4096,4096,4096",
        help="B,M,N,K (default 1,4096,4096,4096; B = batch / number " "of independent same-shape block-scale GEMMs)",
    )
    parser.add_argument(
        "--combo",
        choices=tuple(_COMBOS),
        default="nvfp4",
        help="block-scale dtype family (default nvfp4: FP4 + E4M3 SF, block16)",
    )
    parser.add_argument(
        "--ref",
        choices=("auto", "scaled_mm", "bf16"),
        default="auto",
        help="reference kernel: scaled_mm = cuBLAS's OWN block-scaled "
        "FP4/FP8 GEMM (apples-to-apples, via F.scaled_mm); bf16 = "
        "dense BF16 cuBLAS yardstick; auto (default) = scaled_mm, "
        "falling back to bf16 if this env's cuBLASLt can't run it.",
    )
    add_sweep_args(parser)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA, skipping.")
        return 1

    combo = args.combo
    parts = [int(x) for x in args.shape.split(",")]
    if len(parts) != 4:
        sys.exit("--shape must be B,M,N,K (four values; use B=1 for a single GEMM)")
    B, M, N, K = parts
    wset = _mkdata(B, M, N, K, combo)  # dedicated warmup buffer, and the pool's size unit
    per_set = set_bytes(wset)
    nbuf = resolve_nbuf(args.rotate_buffers, per_set)

    if args._nsys_worker:
        configs = select_configs(args.configs, _SPEC_MAP) if args.configs else []
        _nsys_worker(args.shape, args.combo, configs, args.warmup, args.iters, args.ref, nbuf)
        return 0

    flops = 2 * B * M * N * K
    config_names = select_configs(args.configs, _SPEC_MAP)

    print(f"\n=== block-scale matmul B={B} {M}x{N}x{K}  (~{flops / 1e9:.1f} GFLOP) — " f"{combo} in / BF16 out ===")

    report_pool(nbuf, per_set)

    rows: list[tuple[str, float, float, str]] = []  # (name, tflops, ms, note)
    t0 = time.time()

    def _fmt_row(name: str, tflops: float, ms: float, note: str, ref_tflops: float) -> str:
        if note:
            return f"  {name:50s} {'':8s}   {'':7s}   {note}"
        ratio = tflops / ref_tflops if ref_tflops > 0 else 0.0
        return f"  {name:50s} {tflops:8.2f}   {ms:7.3f}   {ratio:>9.2f}×"

    ref_label = "reference"
    if args.timing == "nsys":
        print(f"  [timing: nsys median kernel duration; ref={args.ref}]\n")
        inner_args = [
            "--shape",
            args.shape,
            "--combo",
            combo,
            "--ref",
            args.ref,
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
            "--rotate-buffers",
            str(nbuf),
        ]
        if config_names:
            inner_args += ["--configs", ",".join(config_names)]
        kern_times = nsys_run_and_parse(__file__, inner_args, tag="bench_bs")

        cublas_hit = find_cublas_time(kern_times)
        if cublas_hit:
            cublas_name, cublas_ms = cublas_hit
            cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12
            ref_label = cublas_name
            print(f"  reference kernel: {cublas_name}")
        else:
            cublas_tflops, cublas_ms = float("nan"), float("nan")
            print("  reference kernel: not detected in nsys output")

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
            print("  [timing: events around delayed back-to-back launches — " "host overhead hidden behind a CUDA _sleep]\n")
        else:
            print("  [timing: torch.cuda.Event wall-clock around python loop — " "includes ~50us/call dispatch overhead]\n")

        pool = _mkdata_pool(B, M, N, K, combo, nbuf)  # rotation pool
        ref_label, ref_warmup, ref_timed = _make_reference_pool(B, M, N, K, combo, args.ref, nbuf)
        if args.stream:
            print(f"  ▶ running {ref_label} reference ...", flush=True)
        cublas_ms = timer(
            rotating(lambda call: call(), ref_timed),
            ref_warmup,
            warmup=args.warmup,
            iters=args.iters,
        )
        cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12
        if args.stream:
            print(
                _fmt_row(
                    f"{ref_label} (reference)",
                    cublas_tflops,
                    cublas_ms,
                    "",
                    cublas_tflops,
                ),
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
                    g, h = _graph_block_scale(B, M, N, K, combo)
                    plan = _build_plan(g, cfg, name)
                    wa, wb, wc, wsfa, wsfb = wset
                    ms = timer(
                        rotating(
                            lambda s, _plan=plan, _h=h: _plan(_vp_bs(_h, s[0], s[1], s[2], s[3], s[4])),
                            pool,
                        ),
                        lambda _plan=plan, _h=h: _plan(_vp_bs(_h, wa, wb, wc, wsfa, wsfb)),
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

    is_bf16_ref = "BF16" in ref_label
    if is_bf16_ref:
        print("  [reference is dense BF16 cuBLAS — fp4/fp8 peak is ~2× BF16, so >1× " "is expected and NOT a like-for-like win]")
    vs_col = "vs BF16" if is_bf16_ref else "vs cuBLAS"

    rows.sort(key=lambda r: -r[1])
    print("=" * 88)
    print(f"  {'config':50s} {'TFLOPS':>8s}   {'ms':>7s}   {vs_col:>10s}")
    print("=" * 88)
    for name, tflops, ms, note in rows:
        print(_fmt_row(name, tflops, ms, note, cublas_tflops))
    print("=" * 88)
    if cublas_tflops > 0:
        print(f"  {ref_label + ' (reference)':50s} {cublas_tflops:8.2f}   " f"{cublas_ms:7.3f}   {'1.00×':>10s}")
    else:
        print(f"  {ref_label} reference: n/a")

    ok = [r for r in rows if not r[3]]
    if ok and cublas_tflops > 0:
        best_name, best_tflops, best_ms, _ = ok[0]
        print(f"\nbest GEMM ({combo}): {best_name}" f" — {best_tflops:.2f} TFLOPS" f" ({best_tflops / cublas_tflops:.2f}× {ref_label})")
    print(f"total: {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
