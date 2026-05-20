"""
Benchmark MoE Grouped Matmul with LLM-inspired shapes.

Usage:
    python benchmarks/bench_moe.py
    python benchmarks/bench_moe.py --bench swiglu --mode e2e
    python benchmarks/bench_moe.py --bench swiglu --autotune 10
"""

import argparse
import statistics
from typing import Callable, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

_BENCH_MODE = "default"  # "default" (delay-kernel) or "e2e" (per-iter sync)

# ~2ms delay at ~2GHz GPU clock, same as cudnnTest's launch_delay_kernel
_DELAY_CYCLES = 4_000_000


def benchmark_fn(fn: Callable, *args, warmup: int = 10, repeat: int = 100, **kw) -> Tuple[float, float]:
    """Benchmark *fn*, returning (median_us, min_us).

    Modes (controlled by global ``_BENCH_MODE``):
      default : delay-kernel technique (same as cudnnTest) — a sleep kernel
                hides CPU launch overhead so the measured time ≈ pure kernel time.
      e2e     : per-iter CUDA-event timing with sync — measures end-to-end
                latency including launch overhead (useful for host-bound analysis).
    """
    for _ in range(warmup):
        fn(*args, **kw)
    torch.cuda.synchronize()

    if _BENCH_MODE == "e2e":
        times: List[float] = []
        for _ in range(repeat):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn(*args, **kw)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e) * 1000.0)
        return statistics.median(times), min(times)

    # default: delay-kernel technique (near pure kernel time)
    times: List[float] = []
    for _ in range(repeat):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(_DELAY_CYCLES)
        s.record()
        fn(*args, **kw)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)
    return statistics.median(times), min(times)


# ---------------------------------------------------------------------------
# cuDNN graph autotune helper (reusable)
# ---------------------------------------------------------------------------


def cudnn_autotune(
    graph,
    variant_pack: Dict[int, torch.Tensor],
    handle,
    max_plans: Optional[int] = None,
    warmup_iters: int = 3,
    bench_iters: int = 10,
    verbose: bool = True,
) -> None:
    """Autotune a cuDNN ``pygraph`` by timing candidate execution plans.

    After this call the *graph* has a single best plan selected and built,
    ready for ``graph.execute()``.

    Args:
        graph: A cuDNN ``pygraph`` that has passed ``check_support()``.
        variant_pack: UID → tensor mapping for execution.
        handle: cuDNN handle (with stream already set).
        max_plans: Try at most this many plans (``None`` = all).
        warmup_iters: Warm-up iterations per plan before timing.
        bench_iters: Timed iterations per plan.
        verbose: Print one-line summary when done.
    """
    import cudnn

    plan_count = graph.get_execution_plan_count()
    if plan_count <= 1:
        graph.build_plans()
        return

    graph.build_plans(policy=cudnn.build_plan_policy.ALL)

    n_try = min(plan_count, max_plans) if max_plans is not None else plan_count

    max_ws = 0
    for pi in range(n_try):
        max_ws = max(max_ws, graph.get_workspace_size_plan_at_index(pi))
    ws = torch.empty(max(max_ws, 1), dtype=torch.uint8, device="cuda")

    best_idx, best_time = 0, float("inf")
    for pi in range(n_try):
        try:
            graph.execute_plan_at_index(variant_pack, ws, pi, handle=handle)
            torch.cuda.synchronize()
        except Exception:
            continue

        for _ in range(warmup_iters):
            graph.execute_plan_at_index(variant_pack, ws, pi, handle=handle)
        torch.cuda.synchronize()

        s_ev = torch.cuda.Event(enable_timing=True)
        e_ev = torch.cuda.Event(enable_timing=True)
        s_ev.record()
        for _ in range(bench_iters):
            graph.execute_plan_at_index(variant_pack, ws, pi, handle=handle)
        e_ev.record()
        torch.cuda.synchronize()
        t = s_ev.elapsed_time(e_ev) / bench_iters
        if t < best_time:
            best_time, best_idx = t, pi

    graph.build_plan_at_index(best_idx)
    if verbose:
        name = graph.get_plan_name_at_index(best_idx)
        print(f"    autotuned {n_try}/{plan_count} plans, best: {name} ({best_time:.3f} ms)")

    del ws


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

MOE_MODELS = {
    "mixtral-8x7b": {
        "hidden": 4096,
        "intermediate": 14336,
        "num_experts": 8,
        "top_k": 2,
    },
    "deepseek-v3": {
        "hidden": 7168,
        "intermediate": 18432,
        "num_experts": 64,
        "top_k": 8,
    },
}

TOKEN_COUNTS = [1024, 4096, 16384]

# ---------------------------------------------------------------------------
# Benchmark: MoE grouped matmul (original)
# ---------------------------------------------------------------------------


def bench_moe(warmup: int, repeat: int):
    print()
    print("=" * 100)
    print("  MoE Grouped Matmul  (FFN projections in MoE models)")
    print("=" * 100)

    try:
        from cudnn.experimental.ops import moe_grouped_matmul
    except Exception as e:
        print(f"  [SKIP] {e}")
        return

    W = [18, 10, 10, 8, 14, 14, 10]
    print(
        "  ".join(
            c.ljust(w)
            for c, w in zip(
                ["model/proj", "tokens", "experts", "topk", "cuDNN (us)", "Naive (us)", "speedup"],
                W,
            )
        )
    )
    print("-" * 95)

    for name, cfg in MOE_MODELS.items():
        hidden = cfg["hidden"]
        intermediate = cfg["intermediate"]
        num_experts = cfg["num_experts"]
        top_k = cfg["top_k"]

        for proj, K_dim, N_dim in [
            ("up", hidden, intermediate),
            ("down", intermediate, hidden),
        ]:
            for n_tokens in TOKEN_COUNTS:
                total = n_tokens * top_k
                tpe = max(1, total // num_experts)
                total = tpe * num_experts
                try:
                    token = torch.randn(1, total, K_dim, dtype=torch.float16, device="cuda")
                    w_raw = torch.randn(num_experts, N_dim, K_dim, dtype=torch.float16, device="cuda")
                    weight = w_raw.transpose(1, 2)
                    fto = (torch.arange(num_experts, dtype=torch.int32, device="cuda") * tpe).reshape(-1, 1, 1)

                    cm, _ = benchmark_fn(
                        moe_grouped_matmul,
                        token,
                        weight,
                        fto,
                        mode="none",
                        top_k=top_k,
                        warmup=warmup,
                        repeat=repeat,
                    )

                    def naive(tok, wr, ne, t):
                        out = torch.empty(1, tok.shape[1], wr.shape[1], dtype=tok.dtype, device=tok.device)
                        for e in range(ne):
                            s = e * t
                            out[0, s : s + t] = tok[0, s : s + t] @ wr[e].T
                        return out

                    nm, _ = benchmark_fn(
                        naive,
                        token,
                        w_raw,
                        num_experts,
                        tpe,
                        warmup=warmup,
                        repeat=repeat,
                    )
                    sp = f"{nm/cm:.2f}x"
                    print(
                        "  ".join(
                            v.ljust(w)
                            for v, w in zip(
                                [f"{name}/{proj}", str(n_tokens), str(num_experts), str(top_k), f"{cm:.0f}", f"{nm:.0f}", sp],
                                W,
                            )
                        )
                    )
                except Exception as e:
                    print(f"  {name}/{proj} {n_tokens}: ERR {e}")


# ---------------------------------------------------------------------------
# Benchmark: CuteDSL MoE kernel vs cuDNN MoE (Grouped GEMM + SwiGLU)
# ---------------------------------------------------------------------------

_AUTOTUNE_PLANS: Optional[int] = None  # None = no autotune, int = max plans


def _ceil_div(a, b):
    return (a + b - 1) // b


def bench_grouped_gemm_glu_vs_cudnn(warmup: int, repeat: int):
    """Compare CuteDSL GroupedGemmGluSm100 vs cuDNN MoE for grouped GEMM + SwiGLU.

    CuteDSL: BlockScaledMoEGroupedGemmGluBiasKernel (FP4 in, BF16 out, fused SwiGLU+AMAX)
    cuDNN:   fused graph (2x moe_grouped_matmul + swish + mul, BF16)
    """
    import sys, os

    _REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(_REPO, "cudnn_frontend/test/python/fe_api"))
    sys.path.insert(0, os.path.join(_REPO, "cudnn_frontend/test/python"))

    print()
    print("=" * 130)
    print("  CuteDSL MoE Kernel  vs  cuDNN MoE  —  Grouped GEMM + SwiGLU benchmark")
    print()
    print("  CuteDSL : GroupedGemmGluSm100 (MXFP8 block-scaled grouped GEMM + SwiGLU + AMAX)")
    print("  cuDNN   : fused cuDNN graph (block_scale_dequantize + 2x moe_grouped_matmul + swish + mul, MXFP8)")
    if _AUTOTUNE_PLANS is not None:
        print(f"  cuDNN autotune : top {_AUTOTUNE_PLANS} plans")
    print("=" * 130)

    try:
        from cudnn import GroupedGemmGluSm100
        from cuda.bindings import driver as cuda_drv
        from test_fe_api_utils import create_and_permute_tensor, create_scale_factor_tensor
    except ImportError as e:
        print(f"  [SKIP] CuteDSL kernel unavailable: {e}")
        return

    try:
        import cudnn
    except ImportError as e:
        print(f"  [SKIP] cuDNN unavailable: {e}")
        return

    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        print(f"  [SKIP] Requires SM100+, found SM{major}{minor}")
        return

    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    cudnn_handle = cudnn.create_handle()
    cudnn.set_stream(handle=cudnn_handle, stream=torch.cuda.current_stream().cuda_stream)

    AB_DTYPE = torch.float8_e4m3fn
    SF_DTYPE = torch.float8_e8m0fnu
    SF_VEC = 32
    C_DTYPE = torch.bfloat16
    D_DTYPE = torch.bfloat16
    M_ALIGNED = 256
    MMA_TILER = (256, 256)
    CLUSTER_SHAPE = (2, 2)

    W = [18, 8, 6, 16, 16, 14, 14, 10]
    headers = [
        "model",
        "tokens",
        "tpe",
        "cuDNN MoE (us)",
        "CuteDSL (us)",
        "cuDNN TFLOPS",
        "CuteDSL TFLOPS",
        "CuteDSL/cuDNN",
    ]
    print("  ".join(h.ljust(w) for h, w in zip(headers, W)))
    print("-" * 120)

    compiled_cache: dict = {}

    for model_name, cfg in MOE_MODELS.items():
        hidden = cfg["hidden"]
        intermediate = cfg["intermediate"]
        num_experts = cfg["num_experts"]
        top_k = cfg["top_k"]

        K = hidden
        N = intermediate * 2
        L = num_experts
        N_OUT = N // 2

        for n_tokens in TOKEN_COUNTS:
            total_routed = n_tokens * top_k
            group_m = max(1, total_routed // num_experts)
            aligned_m = _ceil_div(group_m, M_ALIGNED) * M_ALIGNED
            valid_m = aligned_m * L
            tpe = group_m
            gemm_flops = 2.0 * valid_m * N * K

            cutedsl_us: float | None = None
            cudnn_us: float | None = None

            # ── CuteDSL: GroupedGemmGluSm100 (MXFP8 block-scaled) ──────
            try:
                padded_offsets = torch.tensor(
                    [aligned_m * (i + 1) for i in range(L)],
                    dtype=torch.int32,
                    device="cuda",
                )
                _, a = create_and_permute_tensor(1, valid_m, K, False, AB_DTYPE)
                _, b = create_and_permute_tensor(L, N, K, False, AB_DTYPE)
                _, sfa = create_scale_factor_tensor(1, valid_m, K, SF_VEC, SF_DTYPE)
                _, sfb = create_scale_factor_tensor(L, N, K, SF_VEC, SF_DTYPE)
                alpha = torch.ones(L, dtype=torch.float32, device="cuda")
                c = torch.empty_strided(
                    (valid_m, N, 1),
                    (N, 1, valid_m * N),
                    dtype=C_DTYPE,
                    device="cuda",
                )
                d = torch.empty_strided(
                    (valid_m, N_OUT, 1),
                    (N_OUT, 1, valid_m * N_OUT),
                    dtype=D_DTYPE,
                    device="cuda",
                )
                d_col = torch.empty_strided(
                    (valid_m, N_OUT, 1),
                    (N_OUT, 1, valid_m * N_OUT),
                    dtype=D_DTYPE,
                    device="cuda",
                )
                amax = torch.full((L, 1), float("-inf"), dtype=torch.float32, device="cuda")
                prob = torch.ones(valid_m, 1, 1, dtype=torch.float32, device="cuda")

                cache_key = (valid_m, N, K, L)
                if cache_key not in compiled_cache:
                    api = GroupedGemmGluSm100(
                        sample_a=a,
                        sample_b=b,
                        sample_c=c,
                        sample_d=d,
                        sample_sfa=sfa,
                        sample_sfb=sfb,
                        sample_padded_offsets=padded_offsets,
                        sample_alpha=alpha,
                        sample_d_col=d_col,
                        sample_amax=amax,
                        sample_prob=prob,
                        acc_dtype=torch.float32,
                        mma_tiler_mn=MMA_TILER,
                        cluster_shape_mn=CLUSTER_SHAPE,
                        sf_vec_size=SF_VEC,
                        vector_f32=False,
                        m_aligned=M_ALIGNED,
                        act_func="swiglu",
                    )
                    assert api.check_support(), "CuteDSL config not supported"
                    import time as _time

                    t0 = _time.time()
                    api.compile()
                    print(f"  [{model_name} tpe={tpe}] CuteDSL compiled in {_time.time() - t0:.1f}s")
                    compiled_cache[cache_key] = api
                else:
                    api = compiled_cache[cache_key]

                _api, _stream = api, stream
                _exec_kw = dict(
                    a_tensor=a,
                    b_tensor=b,
                    c_tensor=c,
                    d_tensor=d,
                    sfa_tensor=sfa,
                    sfb_tensor=sfb,
                    padded_offsets=padded_offsets,
                    alpha_tensor=alpha,
                    d_col_tensor=d_col,
                    amax_tensor=amax,
                    prob_tensor=prob,
                    current_stream=_stream,
                )

                def _run_cutedsl():
                    _api.execute(**_exec_kw)

                cutedsl_us, _ = benchmark_fn(_run_cutedsl, warmup=warmup, repeat=repeat)
            except Exception as e:
                import traceback

                print(f"  [{model_name} tpe={tpe}] CuteDSL ERR: {e}")
                traceback.print_exc()

            torch.cuda.empty_cache()

            # ── cuDNN MoE + SwiGLU (fused graph, MXFP8 dequant + GEMM + SwiGLU) ──
            try:
                total = group_m * L
                sf_block = SF_VEC
                sf_k = _ceil_div(K, sf_block)

                token_fp8 = torch.randn(1, total, K, device="cuda").to(torch.float8_e4m3fn)
                token_sf_dim = [1, total, sf_k]
                token_sf_stride = [total * sf_k, sf_k, 1]
                token_sf = torch.ones(token_sf_dim, dtype=torch.float8_e8m0fnu, device="cuda")

                w_gate_fp8 = torch.randn(L, N_OUT, K, device="cuda").to(torch.float8_e4m3fn).transpose(1, 2)
                w_gate_sf_dim = [L, sf_k, N_OUT]
                w_gate_sf_stride = [sf_k * N_OUT, 1, sf_k]
                w_gate_sf = torch.ones(w_gate_sf_dim, dtype=torch.float8_e8m0fnu, device="cuda")

                w_up_fp8 = torch.randn(L, N_OUT, K, device="cuda").to(torch.float8_e4m3fn).transpose(1, 2)
                w_up_sf_dim = [L, sf_k, N_OUT]
                w_up_sf_stride = [sf_k * N_OUT, 1, sf_k]
                w_up_sf = torch.ones(w_up_sf_dim, dtype=torch.float8_e8m0fnu, device="cuda")

                fto = (torch.arange(L, dtype=torch.int32, device="cuda") * group_m).reshape(-1, 1, 1)

                M_out = total
                compute_dtype = cudnn.data_type.FLOAT
                fp8_dtype = cudnn.data_type.FP8_E4M3
                sf_dtype_cudnn = cudnn.data_type.FP8_E8M0
                bf16 = cudnn.data_type.BFLOAT16
                _moe_mode = cudnn.moe_grouped_matmul_mode.NONE

                graph = cudnn.pygraph(
                    intermediate_data_type=cudnn.data_type.FLOAT,
                    compute_data_type=compute_dtype,
                    handle=cudnn_handle,
                )

                token_fp8_t = graph.tensor(
                    name="token_fp8",
                    dim=list(token_fp8.shape),
                    stride=list(token_fp8.stride()),
                    data_type=fp8_dtype,
                    uid=1,
                )
                token_sf_t = graph.tensor(
                    name="token_sf",
                    dim=token_sf_dim,
                    stride=token_sf_stride,
                    data_type=sf_dtype_cudnn,
                    uid=2,
                    reordering_type=cudnn.tensor_reordering.F8_128x4,
                )
                token_deq = graph.block_scale_dequantize(
                    token_fp8_t,
                    token_sf_t,
                    block_size=[1, sf_block],
                    name="dequant_token",
                )

                w_gate_fp8_t = graph.tensor(
                    name="w_gate_fp8",
                    dim=list(w_gate_fp8.shape),
                    stride=list(w_gate_fp8.stride()),
                    data_type=fp8_dtype,
                    uid=3,
                )
                w_gate_sf_t = graph.tensor(
                    name="w_gate_sf",
                    dim=w_gate_sf_dim,
                    stride=w_gate_sf_stride,
                    data_type=sf_dtype_cudnn,
                    uid=4,
                    reordering_type=cudnn.tensor_reordering.F8_128x4,
                )
                w_gate_deq = graph.block_scale_dequantize(
                    w_gate_fp8_t,
                    w_gate_sf_t,
                    block_size=[sf_block, 1],
                    name="dequant_w_gate",
                )

                w_up_fp8_t = graph.tensor(
                    name="w_up_fp8",
                    dim=list(w_up_fp8.shape),
                    stride=list(w_up_fp8.stride()),
                    data_type=fp8_dtype,
                    uid=5,
                )
                w_up_sf_t = graph.tensor(
                    name="w_up_sf",
                    dim=w_up_sf_dim,
                    stride=w_up_sf_stride,
                    data_type=sf_dtype_cudnn,
                    uid=6,
                    reordering_type=cudnn.tensor_reordering.F8_128x4,
                )
                w_up_deq = graph.block_scale_dequantize(
                    w_up_fp8_t,
                    w_up_sf_t,
                    block_size=[sf_block, 1],
                    name="dequant_w_up",
                )

                fto_t = graph.tensor(
                    name="fto",
                    dim=list(fto.shape),
                    stride=list(fto.stride()),
                    data_type=cudnn.data_type.INT32,
                    uid=7,
                )

                gate_out = graph.moe_grouped_matmul(
                    token=token_deq,
                    weight=w_gate_deq,
                    first_token_offset=fto_t,
                    mode=_moe_mode,
                    compute_data_type=compute_dtype,
                    top_k=top_k,
                    name="moe_gate",
                )
                up_out = graph.moe_grouped_matmul(
                    token=token_deq,
                    weight=w_up_deq,
                    first_token_offset=fto_t,
                    mode=_moe_mode,
                    compute_data_type=compute_dtype,
                    top_k=top_k,
                    name="moe_up",
                )

                gate_silu = graph.swish(gate_out, name="silu_gate")
                result_t = graph.mul(up_out, gate_silu, name="up_x_silu_gate")

                result_t.set_output(True).set_data_type(bf16).set_uid(100)

                graph.validate()
                graph.build_operation_graph()
                graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
                graph.check_support()

                result_gpu = torch.empty(1, M_out, N_OUT, dtype=torch.bfloat16, device="cuda")
                variant_pack = {
                    1: token_fp8,
                    2: token_sf,
                    3: w_gate_fp8,
                    4: w_gate_sf,
                    5: w_up_fp8,
                    6: w_up_sf,
                    7: fto,
                    100: result_gpu,
                }

                if _AUTOTUNE_PLANS is not None:
                    cudnn_autotune(graph, variant_pack, cudnn_handle, max_plans=_AUTOTUNE_PLANS)
                else:
                    graph.build_plans()

                ws_size = graph.get_workspace_size()
                workspace = torch.empty(max(ws_size, 1), dtype=torch.uint8, device="cuda")

                _graph, _vp, _ws, _hdl = graph, variant_pack, workspace, cudnn_handle

                def _run_cudnn():
                    _graph.execute(_vp, _ws, handle=_hdl)

                cudnn_us, _ = benchmark_fn(_run_cudnn, warmup=warmup, repeat=repeat)
                del token_fp8, token_sf, w_gate_fp8, w_gate_sf, w_up_fp8, w_up_sf
                del fto, result_gpu, workspace
            except Exception as e:
                import traceback

                print(f"  [{model_name} tpe={tpe}] cuDNN MoE ERR: {e}")
                traceback.print_exc()

            torch.cuda.empty_cache()

            # ── Report ─────────────────────────────────────────────────
            def _fmt(val, fmt_str):
                return fmt_str.format(val) if val is not None else "N/A"

            cutedsl_tf = gemm_flops / (cutedsl_us * 1e-6) / 1e12 if cutedsl_us else None
            cudnn_tf = gemm_flops / (cudnn_us * 1e-6) / 1e12 if cudnn_us else None
            sp = f"{cudnn_us / cutedsl_us:.2f}x" if cutedsl_us and cudnn_us else "N/A"
            vals = [
                model_name,
                str(n_tokens),
                str(tpe),
                _fmt(cudnn_us, "{:.0f}"),
                _fmt(cutedsl_us, "{:.0f}"),
                _fmt(cudnn_tf, "{:.1f}"),
                _fmt(cutedsl_tf, "{:.1f}"),
                sp,
            ]
            print("  ".join(v.ljust(w) for v, w in zip(vals, W)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    global _BENCH_MODE, _AUTOTUNE_PLANS
    parser = argparse.ArgumentParser(description="Benchmark MoE Grouped Matmul.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument(
        "--bench",
        choices=["moe", "swiglu", "all"],
        default="all",
        help="Which benchmark to run (default: all)",
    )
    parser.add_argument(
        "--mode",
        choices=["default", "e2e"],
        default="default",
        help="Timing mode: " "default = delay-kernel technique, near pure kernel time; " "e2e = per-iter CUDA events with sync, includes launch overhead",
    )
    parser.add_argument(
        "--autotune",
        type=int,
        default=None,
        metavar="N",
        help="Enable cuDNN autotune over top N execution plans (default: off, use heuristic)",
    )
    args = parser.parse_args()
    _BENCH_MODE = args.mode
    _AUTOTUNE_PLANS = args.autotune
    print(f"PyTorch: {torch.__version__}, Device: {torch.cuda.get_device_name()}")
    print(f"Timing mode: {_BENCH_MODE}" + (f", autotune top {_AUTOTUNE_PLANS} plans" if _AUTOTUNE_PLANS else ""))
    if args.bench in ("moe", "all"):
        bench_moe(args.warmup, args.repeat)
    if args.bench in ("swiglu", "all"):
        bench_grouped_gemm_glu_vs_cudnn(args.warmup, args.repeat)
    print("\nDone.")


if __name__ == "__main__":
    main()
