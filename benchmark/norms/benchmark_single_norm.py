"""
Normalization benchmark (LayerNorm / RMSNorm).

This script benchmarks a single norm compute instance.
The backend can be chosen (cudnn, pytorch, torch_compile).
Performance is measured using torch profiler.

Can be used as CLI or imported as a module:

    # CLI usage
    python benchmark_single_norm.py --norm_type rms_norm --N 16384 --C 4096 ...

    # Module usage
    from benchmark_single_norm import run_benchmark
    result = run_benchmark(norm_type="rms_norm", N=16384, C=4096, ...)
"""

import argparse
import torch
import os
import numpy as np
import time
from typing import Optional, Dict, Any

from torch.profiler import profile, record_function, ProfilerActivity


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--norm_type", default="rms_norm", type=str, choices=["rms_norm", "layer_norm"], help="Normalization type")
    parser.add_argument("--N", default=16384, type=int, help="Number of rows (batch_size * seq_len)")
    parser.add_argument("--C", default=4096, type=int, help="Embedding dimension (norm dimension)")
    parser.add_argument("--epsilon", default=1e-5, type=float, help="Epsilon for numerical stability")
    parser.add_argument("--has_bias", action="store_true", help="Whether the norm has a bias term")
    parser.add_argument("--data_type", default="bfloat16", type=str, choices=["bfloat16", "float16"], help="Data type")
    parser.add_argument("--backend", default="cudnn", type=str, choices=["cudnn", "pytorch", "torch_compile", "quack"], help="Backend to use")
    parser.add_argument("--profile_pass", default="both", type=str, choices=["fwd", "bwd", "both"], help="Which pass to profile")
    parser.add_argument("--num_iterations", default=20, type=int, help="Number of iterations for performance measurement")
    parser.add_argument("--num_warmup_iterations", default=5, type=int, help="Number of warmup iterations before measuring")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--format_output", action="store_true", help="Format output as CSV for parsing")
    return parser.parse_args()


def compute_bandwidth_bytes(norm_type, N, C, has_bias, dtype_bytes, mode="fwd"):
    """
    Compute total bytes read and written for bandwidth calculation.

    Args:
        norm_type: "rms_norm" or "layer_norm"
        N: Number of rows
        C: Embedding dimension
        has_bias: Whether bias is used
        dtype_bytes: Bytes per element for the data type
        mode: "fwd" or "bwd"

    Returns:
        Total bytes (read + written)
    """
    if mode == "fwd":
        if norm_type == "rms_norm":
            # Read: X(N*C) + Scale(C); Write: Y(N*C) + InvVar(N)
            read_bytes = (N * C + C) * dtype_bytes
            write_bytes = N * C * dtype_bytes + N * 4  # InvVar is always float32
        else:  # layer_norm
            # Read: X(N*C) + Scale(C) + Bias(C); Write: Y(N*C) + Mean(N) + InvVar(N)
            read_bytes = (N * C + C + (C if has_bias else 0)) * dtype_bytes
            write_bytes = N * C * dtype_bytes + N * 4 + N * 4  # Mean, InvVar are float32
    else:  # bwd
        if norm_type == "rms_norm":
            # Read: DY(N*C) + X(N*C) + InvVar(N) + Scale(C); Write: DX(N*C) + DScale(C)
            read_bytes = (N * C + N * C + C) * dtype_bytes + N * 4
            write_bytes = (N * C + C) * dtype_bytes
            if has_bias:
                write_bytes += C * dtype_bytes  # DBias
        else:  # layer_norm
            # Read: DY(N*C) + X(N*C) + Mean(N) + InvVar(N) + Scale(C); Write: DX(N*C) + DScale(C) + DBias(C)
            read_bytes = (N * C + N * C + C) * dtype_bytes + N * 4 + N * 4
            write_bytes = (N * C + C + C) * dtype_bytes

    return read_bytes + write_bytes


def bandwidth_gbps(total_bytes, time_ms):
    """Convert total bytes and time to GB/s."""
    if time_ms <= 0:
        return 0.0
    return total_bytes / (time_ms * 1e-3) / 1e9


def run_benchmark(
    norm_type: str = "rms_norm",
    N: int = 16384,
    C: int = 4096,
    epsilon: float = 1e-5,
    has_bias: bool = False,
    data_type: str = "bfloat16",
    backend: str = "cudnn",
    profile_pass: str = "both",
    num_iterations: int = 20,
    num_warmup_iterations: int = 5,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run a single norm benchmark.

    This function spawns a subprocess to run the benchmark script.

    Returns:
        Dict with keys: fwd_time_ms, bwd_time_ms, fwd_gbps, bwd_gbps,
                        gpu_name, cudnn_version, cudnn_backend_version
    """
    import subprocess
    import sys

    script_path = os.path.abspath(__file__)
    cmd = [
        sys.executable,
        script_path,
        "--norm_type",
        norm_type,
        "--N",
        str(N),
        "--C",
        str(C),
        "--epsilon",
        str(epsilon),
        "--data_type",
        data_type,
        "--backend",
        backend,
        "--profile_pass",
        profile_pass,
        "--num_iterations",
        str(num_iterations),
        "--num_warmup_iterations",
        str(num_warmup_iterations),
        "--format_output",
    ]

    if has_bias:
        cmd.append("--has_bias")
    if verbose:
        cmd.append("--verbose")

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"Benchmark failed with return code {result.returncode}.\n" f"stderr: {result.stderr}\nstdout: {result.stdout}")

    # Parse CSV output
    # Format: backend,norm_type,N,C,epsilon,has_bias,data_type,fwd_time,bwd_time,fwd_gbps,bwd_gbps,num_iters
    output_line = result.stdout.strip().split("\n")[-1]
    parts = output_line.split(",")

    if len(parts) < 11:
        raise RuntimeError(f"Unexpected output format: {output_line}")

    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "Unknown"

    cudnn_version = None
    cudnn_backend_version = None
    try:
        import cudnn

        cudnn_version = cudnn.__version__
        cudnn_backend_version = cudnn.backend_version()
    except ImportError:
        pass

    return {
        "fwd_time_ms": float(parts[7]),
        "bwd_time_ms": float(parts[8]),
        "fwd_gbps": float(parts[9]),
        "bwd_gbps": float(parts[10]),
        "gpu_name": gpu_name,
        "cudnn_version": cudnn_version,
        "cudnn_backend_version": cudnn_backend_version,
    }


# ============================================================================
# Main benchmark implementation (runs when script is executed directly)
# ============================================================================

if __name__ != "__main__":
    pass
else:
    args = parse_args()

    if args.data_type == "bfloat16":
        target_dtype = torch.bfloat16
        dtype_bytes = 2
    elif args.data_type == "float16":
        target_dtype = torch.float16
        dtype_bytes = 2
    else:
        raise ValueError(f"Invalid data type: {args.data_type}")

    N = args.N
    C = args.C
    norm_type = args.norm_type
    has_bias = args.has_bias
    epsilon_value = args.epsilon
    num_iters = args.num_iterations
    dry_run_iters = args.num_warmup_iterations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "Requires CUDA device"

    run_fwd = args.profile_pass in ("fwd", "both")
    run_bwd = args.profile_pass in ("bwd", "both")

    l2_flush_size_mb = 256
    l2_flush_size = l2_flush_size_mb * 1024 * 1024
    l2_flush_buffer = torch.empty(l2_flush_size, device=device, dtype=torch.int8)

    # ============================================================
    # Backend setup
    # ============================================================

    if args.backend == "cudnn":
        try:
            import cudnn
        except ImportError:
            cudnn = None
        assert cudnn is not None, "cuDNN frontend not available"

        if args.verbose:
            print(f"[INFO] cuDNN Backend Version: {cudnn.backend_version()}")
            print(f"[INFO] cuDNN Frontend Version: {cudnn.__version__}")

        # Create initial tensors for graph setup
        x_gpu = torch.randn(N, C, 1, 1, dtype=target_dtype, device=device)
        scale_gpu = torch.randn(1, C, 1, 1, dtype=target_dtype, device=device)
        bias_gpu = torch.randn(1, C, 1, 1, dtype=target_dtype, device=device) if has_bias else None
        epsilon_cpu = torch.full((1, 1, 1, 1), epsilon_value, dtype=torch.float32, device="cpu")

        # Build forward graph
        graph_fwd = cudnn.pygraph(
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )

        X_fwd = graph_fwd.tensor_like(x_gpu.detach())
        scale_fwd = graph_fwd.tensor_like(scale_gpu.detach())
        bias_fwd = graph_fwd.tensor_like(bias_gpu.detach()) if has_bias else None
        epsilon_fwd = graph_fwd.tensor_like(epsilon_cpu)

        if norm_type == "rms_norm":
            Y_fwd, inv_var_fwd = graph_fwd.rmsnorm(
                name="RMS",
                norm_forward_phase=cudnn.norm_forward_phase.TRAINING,
                input=X_fwd,
                scale=scale_fwd,
                bias=bias_fwd,
                epsilon=epsilon_fwd,
            )
            mean_fwd = None
        else:  # layer_norm
            Y_fwd, mean_fwd, inv_var_fwd = graph_fwd.layernorm(
                name="LN",
                norm_forward_phase=cudnn.norm_forward_phase.TRAINING,
                input=X_fwd,
                scale=scale_fwd,
                bias=bias_fwd,
                epsilon=epsilon_fwd,
            )

        Y_fwd.set_output(True).set_data_type(target_dtype)
        inv_var_fwd.set_output(True).set_data_type(torch.float32)
        if mean_fwd is not None:
            mean_fwd.set_output(True).set_data_type(torch.float32)

        graph_fwd.validate()
        graph_fwd.build_operation_graph()
        graph_fwd.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph_fwd.check_support()
        graph_fwd.build_plans()

        # Build backward graph if needed
        if run_bwd:
            dy_gpu = torch.randn(N, C, 1, 1, dtype=target_dtype, device=device)
            inv_var_gpu = torch.randn(N, 1, 1, 1, dtype=torch.float32, device=device)
            mean_gpu = torch.randn(N, 1, 1, 1, dtype=torch.float32, device=device) if norm_type == "layer_norm" else None

            graph_bwd = cudnn.pygraph(
                intermediate_data_type=cudnn.data_type.FLOAT,
                compute_data_type=cudnn.data_type.FLOAT,
            )

            DY_bwd = graph_bwd.tensor_like(dy_gpu.detach())
            X_bwd = graph_bwd.tensor_like(x_gpu.detach())
            scale_bwd = graph_bwd.tensor_like(scale_gpu.detach())
            inv_var_bwd = graph_bwd.tensor_like(inv_var_gpu)

            if norm_type == "rms_norm":
                DX_bwd, Dscale_bwd, Dbias_bwd = graph_bwd.rmsnorm_backward(
                    name="DRMS",
                    grad=DY_bwd,
                    input=X_bwd,
                    scale=scale_bwd,
                    inv_variance=inv_var_bwd,
                    has_dbias=has_bias,
                )
            else:  # layer_norm
                mean_bwd = graph_bwd.tensor_like(mean_gpu)
                DX_bwd, Dscale_bwd, Dbias_bwd = graph_bwd.layernorm_backward(
                    name="DLN",
                    grad=DY_bwd,
                    input=X_bwd,
                    scale=scale_bwd,
                    mean=mean_bwd,
                    inv_variance=inv_var_bwd,
                )

            DX_bwd.set_output(True).set_data_type(target_dtype)
            Dscale_bwd.set_output(True).set_data_type(target_dtype)
            if Dbias_bwd is not None:
                Dbias_bwd.set_output(True).set_data_type(target_dtype)

            graph_bwd.validate()
            graph_bwd.build_operation_graph()
            graph_bwd.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
            graph_bwd.check_support()
            graph_bwd.build_plans()

    elif args.backend == "quack":
        try:
            from quack.rmsnorm import rmsnorm_fwd, rmsnorm_bwd
        except ImportError:
            raise ImportError("quack-kernels is not installed. Install with: pip install quack-kernels")

        assert norm_type == "rms_norm", "Quack backend only supports rms_norm"

    elif args.backend in ("pytorch", "torch_compile"):
        # PyTorch / torch.compile backend
        if norm_type == "rms_norm":

            def pytorch_rms_norm_fwd(x, scale, bias, epsilon):
                norm_x = torch.mean(x * x, dim=1, keepdim=True)
                inv_var = torch.rsqrt(norm_x.float() + epsilon)
                x_normed = x * inv_var.to(x.dtype)
                y = scale * x_normed
                if bias is not None:
                    y = y + bias
                return y

            if hasattr(torch.nn.functional, "rms_norm"):

                def pytorch_norm_fn(x, scale, bias, epsilon):
                    return torch.nn.functional.rms_norm(x, (C, 1, 1), weight=scale.squeeze(0), eps=epsilon)

            else:

                def pytorch_norm_fn(x, scale, bias, epsilon):
                    return pytorch_rms_norm_fwd(x, scale, bias, epsilon)

        else:  # layer_norm

            def pytorch_norm_fn(x, scale, bias, epsilon):
                return torch.nn.functional.layer_norm(
                    x,
                    [C, 1, 1],
                    weight=scale.squeeze(0),
                    bias=bias.squeeze(0) if bias is not None else None,
                    eps=epsilon,
                )

        if args.backend == "torch_compile":
            pytorch_norm_fn = torch.compile(pytorch_norm_fn)

    # ============================================================
    # Benchmark loop
    # ============================================================

    if args.verbose:
        print(f"[INFO] {torch.__version__ = }")
        print(f"[INFO] {torch.version.cuda = }")
        print(f"[INFO] {torch.cuda.get_device_name(torch.cuda.current_device())}")

    forward_times = []
    backward_times = []
    total_iters = num_iters + dry_run_iters

    for i in range(total_iters):
        # Fresh tensors each iteration
        x_gpu = torch.randn(N, C, 1, 1, dtype=target_dtype, device=device, requires_grad=(args.backend != "cudnn" and run_bwd))
        scale_gpu = torch.randn(1, C, 1, 1, dtype=target_dtype, device=device, requires_grad=(args.backend != "cudnn" and run_bwd))
        bias_gpu = torch.randn(1, C, 1, 1, dtype=target_dtype, device=device, requires_grad=(args.backend != "cudnn" and run_bwd)) if has_bias else None
        epsilon_cpu = torch.full((1, 1, 1, 1), epsilon_value, dtype=torch.float32, device="cpu")

        l2_flush_buffer.zero_()

        fwd_time = 0.0

        if args.backend == "cudnn":
            # cuDNN forward
            y_gpu = torch.empty(N, C, 1, 1, dtype=target_dtype, device=device)
            inv_var_actual = torch.empty(N, 1, 1, 1, dtype=torch.float32, device=device)
            mean_actual = torch.empty(N, 1, 1, 1, dtype=torch.float32, device=device) if norm_type == "layer_norm" else None

            variant_pack_fwd = {
                X_fwd: x_gpu.detach(),
                scale_fwd: scale_gpu.detach(),
                epsilon_fwd: epsilon_cpu,
                Y_fwd: y_gpu,
                inv_var_fwd: inv_var_actual,
            }
            if has_bias:
                variant_pack_fwd[bias_fwd] = bias_gpu.detach()
            if mean_fwd is not None:
                variant_pack_fwd[mean_fwd] = mean_actual

            workspace_size = graph_fwd.get_workspace_size()
            if run_bwd:
                workspace_size = max(workspace_size, graph_bwd.get_workspace_size())
            workspace = torch.empty(workspace_size, device="cuda", dtype=torch.uint8)

            if run_fwd:
                with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
                    with record_function("norm.forward"):
                        graph_fwd.execute(variant_pack_fwd, workspace)
                    torch.cuda.synchronize()

                matched_kernels = [item for item in prof.key_averages() if item.key.startswith("cudnn") or "kernel" in item.key.lower()]
                if matched_kernels:
                    fwd_time = sum(item.device_time for item in matched_kernels) / 1000
                    if i >= dry_run_iters:
                        forward_times.append(fwd_time)
            else:
                graph_fwd.execute(variant_pack_fwd, workspace)
                torch.cuda.synchronize()

            if run_bwd:
                l2_flush_buffer.zero_()
                dy_gpu = torch.randn(N, C, 1, 1, dtype=target_dtype, device=device)
                dx_gpu = torch.empty_like(x_gpu)
                dscale_gpu = torch.empty_like(scale_gpu)
                dbias_gpu = torch.empty_like(bias_gpu) if has_bias else None

                variant_pack_bwd = {
                    DY_bwd: dy_gpu.detach(),
                    X_bwd: x_gpu.detach(),
                    scale_bwd: scale_gpu.detach(),
                    inv_var_bwd: inv_var_actual,
                    DX_bwd: dx_gpu,
                    Dscale_bwd: dscale_gpu,
                }
                if Dbias_bwd is not None and dbias_gpu is not None:
                    variant_pack_bwd[Dbias_bwd] = dbias_gpu
                if norm_type == "layer_norm":
                    variant_pack_bwd[mean_bwd] = mean_actual

                with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
                    with record_function("norm.backward"):
                        graph_bwd.execute(variant_pack_bwd, workspace)
                    torch.cuda.synchronize()

                matched_kernels = [item for item in prof.key_averages() if item.key.startswith("cudnn") or "kernel" in item.key.lower()]
                if matched_kernels:
                    bwd_time = sum(item.device_time for item in matched_kernels) / 1000
                    if i >= dry_run_iters:
                        backward_times.append(bwd_time)

        elif args.backend == "quack":
            # Quack uses 2D tensors (N, C) and weight shape (C,)
            x_2d = x_gpu.squeeze(-1).squeeze(-1)  # (N, C)
            w_2d = scale_gpu.squeeze(0).squeeze(-1).squeeze(-1)  # (C,)
            b_2d = bias_gpu.squeeze(0).squeeze(-1).squeeze(-1) if has_bias else None  # (C,) or None

            if run_fwd:
                with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
                    with record_function("norm.forward"):
                        out, _, rstd = rmsnorm_fwd(x_2d, w_2d, bias=b_2d, eps=epsilon_value, store_rstd=True)
                    torch.cuda.synchronize()

                cuda_events = [item for item in prof.key_averages() if item.key == "norm.forward"]
                if cuda_events:
                    fwd_time = cuda_events[0].device_time / 1000
                else:
                    fwd_time = sum(item.device_time for item in prof.key_averages() if item.device_time > 0) / 1000
                if i >= dry_run_iters:
                    forward_times.append(fwd_time)
            else:
                out, _, rstd = rmsnorm_fwd(x_2d, w_2d, bias=b_2d, eps=epsilon_value, store_rstd=True)
                torch.cuda.synchronize()

            if run_bwd:
                l2_flush_buffer.zero_()
                dy_2d = torch.randn(N, C, dtype=target_dtype, device=device)

                with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
                    with record_function("norm.backward"):
                        rmsnorm_bwd(x_2d, w_2d, dy_2d, rstd, has_bias=has_bias)
                    torch.cuda.synchronize()

                cuda_events = [item for item in prof.key_averages() if item.key == "norm.backward"]
                if cuda_events:
                    bwd_time = cuda_events[0].device_time / 1000
                else:
                    bwd_time = sum(item.device_time for item in prof.key_averages() if item.device_time > 0) / 1000
                if i >= dry_run_iters:
                    backward_times.append(bwd_time)

        else:
            # PyTorch / torch.compile forward
            if run_fwd:
                with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
                    with record_function("norm.forward"):
                        output = pytorch_norm_fn(x_gpu, scale_gpu, bias_gpu, epsilon_value)
                    torch.cuda.synchronize()

                matched_kernels = [
                    item
                    for item in prof.key_averages()
                    if item.device_time > 0
                    and not item.key.startswith("aten::")
                    and not item.key.startswith("ProfilerStep")
                    and "norm" not in item.key.lower()
                    and item.key != "norm.forward"
                ]
                # Fallback: sum all CUDA kernel time
                if not matched_kernels:
                    matched_kernels = [item for item in prof.key_averages() if item.device_time > 0 and item.is_legacy]
                # Use total CUDA time from the record_function
                cuda_events = [item for item in prof.key_averages() if item.key == "norm.forward"]
                if cuda_events:
                    fwd_time = cuda_events[0].device_time / 1000
                else:
                    fwd_time = sum(item.device_time for item in prof.key_averages() if item.device_time > 0) / 1000
                if i >= dry_run_iters:
                    forward_times.append(fwd_time)
            else:
                output = pytorch_norm_fn(x_gpu, scale_gpu, bias_gpu, epsilon_value)
                torch.cuda.synchronize()

            if run_bwd:
                l2_flush_buffer.zero_()
                dy_gpu = torch.randn_like(output)

                with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
                    with record_function("norm.backward"):
                        output.backward(dy_gpu)
                    torch.cuda.synchronize()

                cuda_events = [item for item in prof.key_averages() if item.key == "norm.backward"]
                if cuda_events:
                    bwd_time = cuda_events[0].device_time / 1000
                else:
                    bwd_time = sum(item.device_time for item in prof.key_averages() if item.device_time > 0) / 1000
                if i >= dry_run_iters:
                    backward_times.append(bwd_time)

        sleep_time = min(fwd_time / 100, 1.0) if fwd_time > 0 else 0.0
        time.sleep(sleep_time)

    # Compute results
    fwd_median_time = np.median(np.array(forward_times[5:])) if len(forward_times) > 5 else (np.median(np.array(forward_times)) if forward_times else 0.0)
    bwd_median_time = np.median(np.array(backward_times[5:])) if len(backward_times) > 5 else (np.median(np.array(backward_times)) if backward_times else 0.0)

    fwd_bytes = compute_bandwidth_bytes(norm_type, N, C, has_bias, dtype_bytes, "fwd")
    bwd_bytes = compute_bandwidth_bytes(norm_type, N, C, has_bias, dtype_bytes, "bwd")
    fwd_bw = bandwidth_gbps(fwd_bytes, fwd_median_time)
    bwd_bw = bandwidth_gbps(bwd_bytes, bwd_median_time)

    if args.format_output:
        # CSV: backend,norm_type,N,C,epsilon,has_bias,data_type,fwd_time,bwd_time,fwd_gbps,bwd_gbps,num_iters
        print(
            f"{args.backend},{norm_type},{N},{C},{epsilon_value},{int(has_bias)},{args.data_type},"
            f"{fwd_median_time:.3f},{bwd_median_time:.3f},{fwd_bw:.1f},{bwd_bw:.1f},{num_iters}"
        )
    else:
        if run_fwd and run_bwd:
            print(f"{args.backend}:: Median (fwd, bwd): " f"{fwd_median_time:.3f} ms ({fwd_bw:.1f} GB/s), " f"{bwd_median_time:.3f} ms ({bwd_bw:.1f} GB/s)")
        elif run_fwd:
            print(f"{args.backend}:: Median (fwd): {fwd_median_time:.3f} ms ({fwd_bw:.1f} GB/s)")
        elif run_bwd:
            print(f"{args.backend}:: Median (bwd): {bwd_median_time:.3f} ms ({bwd_bw:.1f} GB/s)")
