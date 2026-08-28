# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-library GEMM comparison for the frost sm120 matmul.

Benchmarks C = A @ B^T (TN, BF16 in / FP32 accumulate / BF16 out) across a
fixed 27-shape (M, N, K) sweep and prints a per-shape TFLOPS table comparing:

    frost      the sm120_matmul_1ctamma template (this repo)
    cuBLAS     torch.matmul (cuBLASLt)
    CUTLASS    classic CUTLASS python op (if installed)
    TensorRT   a single-MatMul engine (if installed)
    b12x       local-inference-lab/b12x (if importable)
    FlashInfer flashinfer (if installed and it exposes a BF16 dense GEMM)

Libraries that are unavailable on the machine are reported as SKIP with the
reason; the table renders whatever columns actually ran.

    CUDA_VISIBLE_DEVICES=2 python benchmark/gemm/frost/benchmark_matmul_sm120.py
    ... --shapes 4096x4096x4096,8192x8192x8192   # subset
    ... --libs frost,cublas                       # subset of libraries
    ... --check                                   # bit-exactness vs cuBLAS
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_utils import time_ms_delayed, time_ms_events  # noqa: E402

# The 27-shape sweep (M, N, K).
SHAPES: list[tuple[int, int, int]] = [(m, n, k) for m in (4096, 8192, 16384) for n in (4096, 8192, 16384) for k in (4096, 8192, 16384)]

LIBS = ["frost", "cublas", "cutlass", "tensorrt", "b12x", "flashinfer"]


class Skip(Exception):
    """Raised by an adapter when its library can't run on this machine."""


# ---------------------------------------------------------------------------
# Adapters. Each setup_<lib>(M, N, K, data) returns a zero-arg callable that
# launches one GEMM on the (a, b, c) buffers, or raises Skip(reason).
# Layout contract: a[1,M,K] and b[1,N,K] row-major (both K-major), c[1,M,N].
# Stream contract: the harness times torch's CURRENT (non-default) stream; a
# library that does not follow it must be handed that stream explicitly, or
# the timer measures an empty stream (torch/flashinfer follow it by default).
# ---------------------------------------------------------------------------


def setup_frost(M: int, N: int, K: int, data, args):
    import cudnn
    import cudnn.gemm.frost  # noqa: F401  — installs the pygraph recorder hook
    from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
    from cudnn.gemm.frost.tile_config import by_name

    major, minor = torch.cuda.get_device_capability()
    if not (120 <= major * 10 + minor < 130):
        raise Skip(f"needs a consumer-Blackwell GPU (sm120..12x), have sm_{major}{minor}")

    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[N * K, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True)
    compiled = jit_from_cudnn_graph(g, by_name(args.frost_config))
    a, b, c = data
    bd = compiled.binding
    pack = {bd.a_operands[0]: a, bd.b_operands[0]: b, bd.outputs[0]: c}
    # A direct CompiledFusedGemm call launches on the DEFAULT stream when no
    # stream is passed (the cuDNN-handle stream only arrives via engine
    # dispatch), so hand it the harness stream.
    stream = torch.cuda.current_stream().cuda_stream
    return lambda: compiled(pack, stream=stream)


def setup_cublas(M: int, N: int, K: int, data, args):
    a, b, c = data
    bt = b.transpose(-1, -2)
    return lambda: torch.matmul(a, bt, out=c)


def setup_cutlass(M: int, N: int, K: int, data, args):
    # The classic CUTLASS python interface (package `nvidia-cutlass`, module
    # cutlass.op). The nvidia-cutlass-dsl wheel installed alongside frost does
    # NOT ship a prebuilt dense-GEMM op for sm120 (cutlass.utils.gemm only has
    # sm100 tcgen05 helpers), so this probes for the classic API.
    try:
        import cutlass  # noqa: F401
        from cutlass.op import Gemm  # type: ignore[attr-defined]
    except Exception:
        raise Skip("no CUTLASS python GEMM op (nvidia-cutlass-dsl has no sm120 dense-GEMM entry; pip install nvidia-cutlass for cutlass.op.Gemm)")
    a, b, c = data
    a2, b2, c2 = a[0], b[0].transpose(0, 1), c[0]
    plan = Gemm(
        A=a2,
        B=b2,
        C=c2,
        D=c2,
        alpha=1.0,
        beta=0.0,
        element_accumulator=torch.float32,
    )
    return lambda: plan.run(a2, b2, c2, c2, alpha=1.0, beta=0.0, sync=False)


def setup_tensorrt(M: int, N: int, K: int, data, args):
    try:
        import tensorrt as trt
    except Exception:
        raise Skip("tensorrt not installed (pip install tensorrt)")
    a, b, c = data
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # Strongly typed: BF16 flows from the input dtypes to the output. The
    # weakly-typed route (BuilderFlag.BF16 + the ITensor.dtype setter) was
    # deprecated in TRT 10 and removed in TRT 11.
    network = builder.create_network(int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    a_in = network.add_input("A", trt.DataType.BF16, (M, K))
    b_in = network.add_input("B", trt.DataType.BF16, (N, K))
    mm = network.add_matrix_multiply(a_in, trt.MatrixOperation.NONE, b_in, trt.MatrixOperation.TRANSPOSE)
    out = mm.get_output(0)
    out.name = "C"
    network.mark_output(out)
    config = builder.create_builder_config()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise Skip("TensorRT engine build failed")
    engine = trt.Runtime(logger).deserialize_cuda_engine(blob)
    ctx = engine.create_execution_context()
    ctx.set_tensor_address("A", a.data_ptr())
    ctx.set_tensor_address("B", b.data_ptr())
    ctx.set_tensor_address("C", c.data_ptr())
    stream = torch.cuda.current_stream().cuda_stream
    # Keep the engine/context alive via the closure.
    return lambda _refs=(engine, ctx): ctx.execute_async_v3(stream)


def setup_b12x(M: int, N: int, K: int, data, args):
    # local-inference-lab / b12x. Not on PyPI: point PYTHONPATH (or --b12x-path)
    # at the checkout, then adapt the entry-point probe below to its real API.
    if args.b12x_path:
        sys.path.insert(0, args.b12x_path)
    mod = None
    for name in ("b12x", "local_inference_lab.b12x", "local_inference_lab"):
        try:
            import importlib

            mod = importlib.import_module(name)
            break
        except ImportError:
            continue
    if mod is None:
        raise Skip("b12x not importable (clone local-inference-lab and pass --b12x-path)")
    a, b, c = data
    for entry in ("matmul", "gemm", "mm"):
        fn = getattr(mod, entry, None)
        if callable(fn):
            return lambda _fn=fn: _fn(a[0], b[0], out=c[0])
    raise Skip(f"b12x imported ({mod.__name__}) but exposes no known GEMM entry — adapt setup_b12x()")


def setup_flashinfer(M: int, N: int, K: int, data, args):
    try:
        from flashinfer.gemm import mm_bf16
    except Exception:
        raise Skip("flashinfer not installed or too old for gemm.mm_bf16 (pip install flashinfer-python)")
    a, b, c = data
    # mm_bf16 wants A (m, k) row-major and B (k, n) column-major — exactly the
    # (N, K)-row-major buffer transposed.
    a2, bt, c2 = a[0], b[0].transpose(0, 1), c[0]
    # Its default backend='cudnn' can be broken independently of the others
    # (e.g. a cuDNN sub-library mismatch), and 'auto' dies with it rather than
    # falling back — so probe explicitly, first working backend wins.
    backends = [args.flashinfer_backend] if args.flashinfer_backend else ["cublaslt", "cudnn", "cutlass", "tgv", "tinygemm"]
    errs = []
    for bk in backends:

        def run(_bk=bk):
            return mm_bf16(a2, bt, out=c2, backend=_bk)

        try:
            run()
            torch.cuda.synchronize()
        except Exception as e:
            errs.append(f"{bk}: {str(e).splitlines()[0][:60]}")
            continue
        if not getattr(setup_flashinfer, "_noted", False):
            print(f"  [flashinfer: backend '{bk}']", flush=True)
            setup_flashinfer._noted = True
        return run
    raise Skip("no working mm_bf16 backend — " + "; ".join(errs))


SETUP = {
    "frost": setup_frost,
    "cublas": setup_cublas,
    "cutlass": setup_cutlass,
    "tensorrt": setup_tensorrt,
    "b12x": setup_b12x,
    "flashinfer": setup_flashinfer,
}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _mkdata(M: int, N: int, K: int, check: bool):
    torch.manual_seed(0)
    if check:
        # Small integers => exact FP32 reduction => bit-comparable outputs.
        a = torch.empty(1, M, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
        b = torch.empty(1, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    else:
        a = torch.randn(1, M, K, device="cuda").to(torch.bfloat16)
        b = torch.randn(1, N, K, device="cuda").to(torch.bfloat16)
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    return a, b, c


def _iters_for(flops: int) -> int:
    # ~3e13 timed FLOP per measurement, clamped to [6, 50] iterations.
    return max(6, min(50, int(3e13 // flops)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--libs", default=",".join(LIBS), help=f"comma list from {LIBS}")
    parser.add_argument("--shapes", default="", help="comma list of MxNxK to run (default: the full 27-shape sweep)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=0, help="timed iterations (0 = auto-scale by FLOPs)")
    parser.add_argument("--timing", choices=("delayed", "events"), default="delayed", help="delayed = kernel-only (host gaps hidden behind a CUDA sleep)")
    parser.add_argument("--frost-config", default="CONFIG_sm120_128x128x128_128x128x32_cluster1x1")
    parser.add_argument(
        "--flashinfer-backend", default="", help="pin the mm_bf16 backend (cublaslt/cudnn/cutlass/tgv/tinygemm/cutile); default: probe in order"
    )
    parser.add_argument("--b12x-path", default=os.environ.get("B12X_PATH", ""), help="path to the local-inference-lab checkout for b12x")
    parser.add_argument("--check", action="store_true", help="small-int inputs + compare every library bit-wise against cuBLAS")
    parser.add_argument("--csv", default="", help="also write results to this CSV file")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA device, exiting.")
        return 1
    # TRT's enqueueV3 inserts extra cudaStreamSynchronize calls on the DEFAULT
    # stream (it warns about exactly this); a non-default stream keeps the
    # delayed timer's back-to-back pipelining honest for every library.
    torch.cuda.set_stream(torch.cuda.Stream())
    libs = [x.strip() for x in args.libs.split(",") if x.strip()]
    unknown = [x for x in libs if x not in SETUP]
    if unknown:
        sys.exit(f"unknown libs {unknown}; choose from {LIBS}")
    if args.shapes:
        shapes = []
        for tok in args.shapes.split(","):
            m, n, k = (int(x) for x in tok.lower().split("x"))
            shapes.append((m, n, k))
    else:
        shapes = SHAPES

    dev = torch.cuda.get_device_name()
    cap = torch.cuda.get_device_capability()
    timer = time_ms_delayed if args.timing == "delayed" else time_ms_events
    print(f"GPU: {dev} (sm_{cap[0]}{cap[1]})  dtype: BF16 in / FP32 accum / BF16 out  layout: TN (C = A @ B^T)")
    print(f"timing: {args.timing}, warmup={args.warmup}, iters={'auto' if args.iters == 0 else args.iters}")

    skip_reasons: dict[str, str] = {}
    results: dict[tuple[int, int, int], dict[str, float]] = {}
    t0 = time.time()

    for M, N, K in shapes:
        flops = 2 * M * N * K
        iters = args.iters or _iters_for(flops)
        data = _mkdata(M, N, K, args.check)
        row: dict[str, float] = {}
        ref = None
        if args.check:
            a, b, _ = data
            ref = torch.matmul(a.to(torch.float32), b.transpose(-1, -2).to(torch.float32)).to(torch.bfloat16)
        print(f"\n--- {M}x{N}x{K}  ({flops / 1e12:.1f} TFLOP, iters={iters}) ---", flush=True)
        for lib in libs:
            if lib in skip_reasons:
                continue
            try:
                run = SETUP[lib](M, N, K, data, args)
                run()
                torch.cuda.synchronize()
                if ref is not None:
                    data[2].zero_()
                    run()
                    torch.cuda.synchronize()
                    bad = (data[2] != ref).sum().item()
                    if bad:
                        print(f"  {lib:10s} CHECK FAILED: {bad} mismatches vs fp32 reference", flush=True)
                ms = timer(lambda i, _r=run: _r(), lambda _r=run: _r(), warmup=args.warmup, iters=iters)
                row[lib] = flops / (ms * 1e-3) / 1e12
                print(f"  {lib:10s} {row[lib]:8.2f} TFLOPS   ({ms:.3f} ms)", flush=True)
            except Skip as e:
                skip_reasons[lib] = str(e)
                print(f"  {lib:10s} SKIP: {e}", flush=True)
            except Exception as e:
                msg = str(e).splitlines()[0][:70] if str(e) else type(e).__name__
                row[lib] = float("nan")
                print(f"  {lib:10s} ERROR: {msg}", flush=True)
        results[(M, N, K)] = row
        del data
        torch.cuda.empty_cache()

    active = [lib for lib in libs if lib not in skip_reasons]
    width = 11
    print("\n" + "=" * (22 + width * len(active)))
    print(f"  {'M x N x K':20s}" + "".join(f"{lib:>{width}s}" for lib in active))
    print("=" * (22 + width * len(active)))
    for (M, N, K), row in results.items():
        cells = "".join(f"{row.get(lib, float('nan')):>{width}.2f}" for lib in active)
        print(f"  {f'{M}x{N}x{K}':20s}" + cells)
    print("=" * (22 + width * len(active)))
    print("  (TFLOPS; higher is better)")
    for lib, why in skip_reasons.items():
        print(f"  SKIP {lib}: {why}")

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("M,N,K," + ",".join(active) + "\n")
            for (M, N, K), row in results.items():
                f.write(f"{M},{N},{K}," + ",".join(f"{row.get(lib, float('nan')):.2f}" for lib in active) + "\n")
        print(f"  CSV written to {args.csv}")
    print(f"total: {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
