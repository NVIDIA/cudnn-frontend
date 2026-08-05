# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Linear Attention (GDN / KDA / GDN-2) benchmark

This script benchmarks a single linear attention compute instance.
The linear attention backend can be chosen. Performance is measured using torch profiler.

Can be used as CLI or imported as a module:

    # CLI usage
    python benchmark_single_linear_attention.py --batch_size 1 --seqlen 8192 ...

    # Module usage
    from benchmark_single_linear_attention import run_benchmark
    result = run_benchmark(batch_size=1, seqlen=8192, ...)
"""

import argparse
import torch
import os
import numpy as np
import math
import threading
import time
from typing import Optional, Dict, Any

from torch.profiler import profile, record_function, ProfilerActivity

# Dense MMA throughput (FLOPs / clock / SM) for Blackwell datacenter SKUs.
# BF16/FP16 dense = 8192.
# Keys match the strings accepted by the --data_type CLI flag.
_BLACKWELL_DC_FLOPS_PER_CLOCK_PER_SM = {
    "bfloat16": 8192,
    "float16": 8192,
}

# Chunk size of the chunked linear attention algorithms (both backends tile
# the sequence into 64-token chunks; the FLOPs model below depends on it).
_CHUNK_SIZE = 64


def _peak_flops_per_clock_per_sm(dtype_str):
    """Return per-SM per-clock dense FLOPs for the current GPU + dtype.
    Returns None on unsupported arch (anything other than Blackwell DC for now)."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if props.major != 10:  # only Blackwell DC is in scope
        return None
    return _BLACKWELL_DC_FLOPS_PER_CLOCK_PER_SM.get(dtype_str)


class _SmClockSampler:
    """Background thread that polls SM clock via NVML at ~1 kHz.

    Used to capture the actual boost clock during the benchmark window.
    `nvmlDeviceGetMaxClockInfo` is unreliable on some Blackwell datacenter
    SKUs: it can report a value below the boost the kernel actually runs
    at, producing nonsensical (>100%) SOL numbers downstream.
    """

    def __init__(self):
        self._samples = []
        self._stop = threading.Event()
        self._thread = None
        self._handle = None
        self._pynvml = None

    def start(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        except Exception:
            self._pynvml = None
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        pynvml = self._pynvml
        while not self._stop.is_set():
            try:
                self._samples.append(pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_SM))
            except Exception:
                pass
            # Sample at ~1 kHz; kernels run much longer than this in aggregate
            # across warmup + measurement iterations.
            time.sleep(0.001)

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        try:
            if self._pynvml is not None:
                self._pynvml.nvmlShutdown()
        except Exception:
            pass

    def peak_mhz(self):
        """Return max sampled SM clock (MHz), or None if no samples."""
        return max(self._samples) if self._samples else None


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size to input to the layer")
    parser.add_argument("--seqlen", default=8192, type=int, help="Sequence length to input to the layer")
    parser.add_argument(
        "--num_q_heads",
        default=16,
        type=int,
        help="Number of query/key heads to input to the layer",
    )
    parser.add_argument(
        "--num_kv_heads",
        default=8,
        type=int,
        help="Number of value/gate heads to input to the layer (the recurrent state lives at these heads)",
    )
    parser.add_argument("--head_dim", default=128, type=int, help="Head dimension to input to the layer")
    parser.add_argument(
        "--head_dim_qk",
        default=None,
        type=int,
        help="Optional: head dimension for Q/K. If set, must also set --head_dim_vo",
    )
    parser.add_argument(
        "--head_dim_vo",
        default=None,
        type=int,
        help="Optional: head dimension for V/O. If set, must also set --head_dim_qk",
    )
    parser.add_argument(
        "--data_type",
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Data type. Can be bfloat16 or float16",
    )
    parser.add_argument(
        "--num_iterations",
        default=20,
        type=int,
        help="Number of iterations to run the layer for performance measurement",
    )
    parser.add_argument(
        "--num_warmup_iterations",
        default=0,
        type=int,
        help="Number of warmup iterations to run before measuring performance",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--fwd_bwd",
        action="store_true",
        help="Run both forward and backward pass (fwd only by default)",
    )
    parser.add_argument(
        "--profile_pass",
        default=None,
        type=str,
        choices=["fwd", "bwd", "both"],
        help="Which pass to profile (default: fwd unless --fwd_bwd is set).",
    )
    parser.add_argument(
        "--variant",
        default="gdn",
        type=str,
        help="Linear attention variant to use. Can be 'gdn' (scalar decay), 'kda' (per-key-channel decay), or 'gdn2' (channel-wise decay/erase/write gates, forward only).",
        choices=["gdn", "kda", "gdn2"],
    )
    parser.add_argument(
        "--store_on",
        action="store_true",
        help="Request the per-sequence final recurrent state from the forward pass (and feed its gradient in the backward pass)",
    )
    parser.add_argument(
        "--initial_state",
        action="store_true",
        help="Provide a per-sequence initial recurrent state (its gradient is produced in the backward pass)",
    )
    parser.add_argument(
        "--la_backend",
        default="cudnn",
        type=str,
        help="Linear attention backend to use",
        choices=[
            "fla",
            "cudnn",
        ],
    )
    parser.add_argument("--format_output", action="store_true", help="Format output to be used in benchmark")
    parser.add_argument(
        "--case_tag",
        default="",
        type=str,
        help="Tag to identify the case. Not used in calculations. Only for formatted output",
    )
    parser.add_argument(
        "--skip_ref",
        action="store_true",
        help="Skip reference linear attention implementation",
    )
    return parser.parse_args()


def run_benchmark(
    batch_size: int,
    seqlen: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int = 128,
    head_dim_qk: Optional[int] = None,
    head_dim_vo: Optional[int] = None,
    data_type: str = "bfloat16",
    backend: str = "cudnn",
    variant: str = "gdn",
    profile_pass: str = "fwd",
    num_iterations: int = 10,
    num_warmup_iterations: int = 0,
    skip_ref: bool = True,
    store_on: bool = False,
    initial_state: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run a single linear attention benchmark.

    This function can be called directly when using the module as a library.
    Internally uses subprocess to call this script with the appropriate arguments.

    Args:
        batch_size: Batch size
        seqlen: Sequence length
        num_q_heads: Number of query/key heads
        num_kv_heads: Number of value/gate heads
        head_dim: Head dimension (used if head_dim_qk/vo not specified)
        head_dim_qk: Head dimension for Q/K (optional, for asymmetric)
        head_dim_vo: Head dimension for V/O (optional, for asymmetric)
        data_type: Data type ("bfloat16", "float16")
        backend: Backend name ("cudnn", "fla")
        variant: Linear attention variant ("gdn", "kda", "gdn2")
        profile_pass: Which pass to profile ("fwd", "bwd", "both")
        num_iterations: Number of benchmark iterations
        num_warmup_iterations: Warmup iterations before measurement
        skip_ref: Skip reference validation
        store_on: Request the final recurrent state from the forward pass
        initial_state: Provide an initial recurrent state
        verbose: Print verbose output

    Returns:
        Dict with keys:
            - time_ms: Median time of the requested pass in milliseconds
            - tflops: TFLOPS for the requested pass
            - max_diff: Maximum difference vs reference
            - gpu_name: GPU name string
            - cudnn_version: cuDNN version (if available)

    Raises:
        RuntimeError: If the benchmark subprocess fails or profile_pass=="both"
            (callers must invoke once per pass so each has independent success)
    """
    if profile_pass == "both":
        raise RuntimeError("run_benchmark no longer accepts profile_pass='both'. Call once " "per pass ('fwd' or 'bwd') so failures remain independent.")
    import subprocess
    import sys

    # Build command
    script_path = os.path.abspath(__file__)
    cmd = [
        sys.executable,
        script_path,
        "--batch_size",
        str(batch_size),
        "--seqlen",
        str(seqlen),
        "--num_q_heads",
        str(num_q_heads),
        "--num_kv_heads",
        str(num_kv_heads),
        "--data_type",
        data_type,
        "--la_backend",
        backend,
        "--variant",
        variant,
        "--num_iterations",
        str(num_iterations),
        "--num_warmup_iterations",
        str(num_warmup_iterations),
        "--format_output",  # Get CSV-formatted output for parsing
    ]

    # Handle head dimensions
    if head_dim_qk is not None and head_dim_vo is not None:
        cmd.extend(["--head_dim_qk", str(head_dim_qk)])
        cmd.extend(["--head_dim_vo", str(head_dim_vo)])
    else:
        cmd.extend(["--head_dim", str(head_dim)])

    # Handle profile pass (single pass only)
    cmd.extend(["--profile_pass", profile_pass])

    # Handle flags
    if skip_ref:
        cmd.append("--skip_ref")
    if store_on:
        cmd.append("--store_on")
    if initial_state:
        cmd.append("--initial_state")
    if verbose:
        cmd.append("--verbose")

    # Run benchmark
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Benchmark failed with return code {result.returncode}.\n" f"stderr: {result.stderr}\n" f"stdout: {result.stdout}")

    # Parse CSV output
    # Format: case_tag,backend,variant,batch_size,seqlen,num_q_heads,num_kv_heads,head_dim,fwd_time,bwd_time,fwd_tflops,bwd_tflops,max_diff,num_iters
    output_line = result.stdout.strip().split("\n")[-1]
    parts = output_line.split(",")

    if len(parts) < 12:
        raise RuntimeError(f"Unexpected output format: {output_line}")

    # Get GPU name from torch
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "Unknown"

    # Try to get cudnn version
    cudnn_version = None
    cudnn_backend_version = None
    try:
        import cudnn

        cudnn_version = cudnn.__version__
        cudnn_backend_version = cudnn.backend_version()
    except ImportError:
        pass

    # Subprocess CSV layout keeps both fwd and bwd columns; pick the one
    # corresponding to the requested pass. The unused pass is 0 when not run.
    if profile_pass == "fwd":
        time_ms = float(parts[8])
        tflops = float(parts[10])
    else:  # "bwd"
        time_ms = float(parts[9])
        tflops = float(parts[11])

    return {
        "time_ms": time_ms,
        "tflops": tflops,
        "max_diff": float(parts[12]) if len(parts) > 12 else 0.0,
        "gpu_name": gpu_name,
        "cudnn_version": cudnn_version,
        "cudnn_backend_version": cudnn_backend_version,
    }


# ============================================================================
# Main benchmark implementation (runs when script is executed directly)
# ============================================================================

# Note: All code below this point is only executed when running as a script.
# When imported as a module, use the run_benchmark() function above.

if __name__ != "__main__":
    # Stop here when imported as module
    pass
else:
    # Parse command line arguments
    args = parse_args()

    if args.data_type == "bfloat16":
        target_dtype = torch.bfloat16
    elif args.data_type == "float16":
        target_dtype = torch.float16
    else:
        raise ValueError(f"Invalid data type: {args.data_type}")

    # Parse input arguments
    num_iters = args.num_iterations
    dry_run_iters = args.num_warmup_iterations
    batch_size = args.batch_size
    seqlen = args.seqlen
    num_q_heads = args.num_q_heads
    num_kv_heads = args.num_kv_heads
    if args.head_dim_qk is None and args.head_dim_vo is None:
        head_dim_qk = args.head_dim
        head_dim_vo = args.head_dim
    elif args.head_dim_qk is not None and args.head_dim_vo is not None:
        head_dim_qk = args.head_dim_qk
        head_dim_vo = args.head_dim_vo
    else:
        raise ValueError("Both --head_dim_qk and --head_dim_vo must be provided together when using asymmetric head dims.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "Requires CUDA device"
    if args.profile_pass is not None:
        run_fwd = args.profile_pass in ("fwd", "both")
        run_bwd = args.profile_pass in ("bwd", "both")
    elif args.fwd_bwd:
        run_fwd = True
        run_bwd = True
    else:
        run_fwd = True
        run_bwd = False
    # Grouped-value attention: the recurrent state lives at the value/gate
    # heads; several q/k heads may share one state (num_kv_heads groups over
    # num_q_heads).
    enable_gva = num_q_heads != num_kv_heads
    if enable_gva and max(num_q_heads, num_kv_heads) % min(num_q_heads, num_kv_heads) != 0:
        raise ValueError("num_q_heads and num_kv_heads must be equal or one a multiple of the other (grouped heads)")
    # The gates, output, and recurrent state live at HO = max(q, v) heads:
    # GVA groups v-heads over q-heads, GQA (gdn only) the reverse.
    num_o_heads = max(num_q_heads, num_kv_heads)
    if num_q_heads > num_kv_heads and args.variant != "gdn":
        raise ValueError("GQA (num_q_heads > num_kv_heads) is only supported with the 'gdn' variant")
    if args.variant == "gdn2" and run_bwd:
        raise ValueError("gdn2 is forward only (the backward kernel is a stub); use --profile_pass fwd")
    if args.variant == "gdn2" and args.la_backend == "fla":
        raise ValueError("gdn2 is only supported with the 'cudnn' backend")

    l2_flush_size_mb = 256
    l2_flush_size = l2_flush_size_mb * 1024 * 1024
    l2_flush_buffer = torch.empty(l2_flush_size, device=device, dtype=torch.int8)

    #############################################################
    ##### Set up linear attention function for each backend #####

    ## If using cuDNN FE, the torch custom ops route through the pygraph
    ## engines (FROST on SM100-class devices, cuTile elsewhere); autograd,
    ## graph caching, and workspace management live inside the op.
    if args.la_backend == "cudnn":
        attn_scale = head_dim_qk ** (-0.5)

        try:
            import cudnn
            from cudnn.linear_attention.ops import gated_delta_net, kimi_delta_attention, gated_delta_net_v2
        except ImportError:
            cudnn = None
        assert cudnn is not None

        if args.verbose:
            print(f"[INFO] cuDNN Backend Version: {cudnn.backend_version() = }")
            print(f"[INFO] cuDNN Frontend Version: {cudnn.__version__ = }")

        ## The graph-API linear attention ops are THD-only: token-packed
        ## [total_tokens, heads, dim] tensors plus cu_seqlens sequence
        ## boundaries. A dense batch is cu_seqlens = [0, T, 2T, ...].
        cu_seqlens = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * seqlen

        def cudnn_linear_attention(query, key, value, gate, beta, write_gate, s0):
            if args.variant == "gdn":
                return gated_delta_net(
                    query,
                    key,
                    value,
                    gate,
                    beta,
                    cu_seqlens,
                    scale=attn_scale,
                    initial_state=s0,
                    output_final_state=args.store_on,
                )
            elif args.variant == "kda":
                return kimi_delta_attention(
                    query,
                    key,
                    value,
                    gate,
                    beta,
                    cu_seqlens,
                    scale=attn_scale,
                    initial_state=s0,
                    output_final_state=args.store_on,
                    use_qk_l2norm_in_kernel=False,
                )
            else:  # gdn2
                return gated_delta_net_v2(
                    query,
                    key,
                    value,
                    gate,
                    beta,
                    write_gate,
                    cu_seqlens,
                    scale=attn_scale,
                    initial_state=s0,
                    output_final_state=args.store_on,
                    use_qk_l2norm_in_kernel=False,
                )

    if args.la_backend == "fla" or (not args.skip_ref):
        attn_scale = head_dim_qk ** (-0.5)

        if args.variant == "gdn":
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        elif args.variant == "kda":
            try:
                from fla.ops.kda import chunk_kda
            except ImportError as e:
                raise RuntimeError(f"The installed fla does not provide KDA (fla.ops.kda): {e}")
        else:  # gdn2
            raise ValueError("gdn2 is only supported with the 'cudnn' backend (no fla implementation, so no reference either); use --skip_ref")

        ## FLA takes dense (B, T, H, D) tensors; g is the log-space decay.
        def fla_linear_attention(query, key, value, gate, beta, write_gate, s0):
            if args.variant == "gdn":
                return chunk_gated_delta_rule(
                    query,
                    key,
                    value,
                    gate,
                    beta,
                    scale=attn_scale,
                    initial_state=s0,
                    output_final_state=args.store_on,
                )
            else:  # kda
                return chunk_kda(
                    query,
                    key,
                    value,
                    gate,
                    beta,
                    scale=attn_scale,
                    initial_state=s0,
                    output_final_state=args.store_on,
                    use_qk_l2norm_in_kernel=False,
                )

    def get_linear_attention_function(backend):
        if backend == "fla":
            return fla_linear_attention
        elif backend == "cudnn":
            return cudnn_linear_attention
        else:
            raise ValueError(f"Invalid backend: {backend}")

    # Util function for addressing different qkv formats for each backend
    # (cudnn is THD [B*T, H, D]; fla is dense [B, T, H, D])
    def preprocess_qkv(query, key, value, backend):
        if backend == "cudnn":
            return (
                query.reshape(batch_size * seqlen, *query.shape[2:]),
                key.reshape(batch_size * seqlen, *key.shape[2:]),
                value.reshape(batch_size * seqlen, *value.shape[2:]),
            )
        elif backend == "fla":
            return query, key, value
        else:
            raise ValueError(f"Invalid backend: {backend}")

    def preprocess_gates(gate, beta, write_gate, backend):
        if backend == "cudnn":
            return (
                gate.reshape(batch_size * seqlen, *gate.shape[2:]),
                beta.reshape(batch_size * seqlen, *beta.shape[2:]),
                write_gate.reshape(batch_size * seqlen, *write_gate.shape[2:]) if write_gate is not None else None,
            )
        elif backend == "fla":
            return gate, beta, write_gate
        else:
            raise ValueError(f"Invalid backend: {backend}")

    # Util function addressing different output formats for each backend
    def postprocess_o(output, backend):
        if backend == "cudnn":
            return output.reshape(batch_size, seqlen, num_o_heads, head_dim_vo)
        elif backend == "fla":
            return output
        else:
            raise ValueError(f"Invalid backend: {backend}")

    # Util functions for calculating flops and tflops/s achieved
    def flops(
        batch_size,
        seqlen,
        head_dim_qk,
        head_dim_vo,
        num_kv_heads,
        mode="fwd",
    ):
        assert mode in ["fwd", "bwd", "fwd_bwd"]

        # Chunked linear attention BMM FLOPs per 64-token chunk per (batch,
        # state head), chunk size C, dims K (qk) and V (vo):
        # Forward: 5 BMM classes =>
        #   intra scores + WY prep (2 x C*C*K), WY apply (C*C*K + C*C*V),
        #   intra output (C*C*V), inter output + state update (2 x C*K*V)
        # Backward: recompute + gradient chains, ~3x forward.
        C = _CHUNK_SIZE
        num_chunks = ceil_div(seqlen, C)
        per_chunk = 2 * (3 * C * C * head_dim_qk + 2 * C * C * head_dim_vo + 2 * C * head_dim_qk * head_dim_vo)
        base = batch_size * num_kv_heads * num_chunks * per_chunk
        if mode == "fwd":
            result = base
        elif mode == "bwd":
            result = base * 3
        else:  # fwd_bwd
            result = base * 4
        return result

    def tflops_per_sec(
        batch_size,
        seqlen,
        head_dim_qk,
        head_dim_vo,
        num_kv_heads,
        time,
        mode="fwd",
    ):
        assert mode in ["fwd", "bwd", "fwd_bwd"]
        f = flops(
            batch_size,
            seqlen,
            head_dim_qk,
            head_dim_vo,
            num_kv_heads,
            mode,
        )
        return f / time / 1e9 if not math.isnan(time) else 0.0  # Assume time is in msec

    ## Gate generators per variant. Decays are LOG-space (alpha = exp(g)),
    ## drawn from ranges the kernels' io-dtype arithmetic is conditioned for.
    def generate_gates(io_dtype):
        if args.variant == "gdn":
            # scalar decay [B, T, HO] fp32 + scalar write strength
            gate = torch.empty(batch_size, seqlen, num_o_heads, device=device).uniform_(0.1, 1.0).log()
            beta = torch.rand(batch_size, seqlen, num_o_heads, device=device)
            write_gate = None
        elif args.variant == "kda":
            # per-key-channel decay [B, T, HO, K] fp32 + post-sigmoid scalar beta
            gate = torch.empty(batch_size, seqlen, num_o_heads, head_dim_qk, device=device).uniform_(0.5, 1.0).log()
            beta = torch.rand(batch_size, seqlen, num_o_heads, device=device).sigmoid()
            write_gate = None
        else:  # gdn2
            # per-key decay/erase [B, T, HO, K] + per-value write gate [B, T, HO, V]
            gate = torch.empty(batch_size, seqlen, num_o_heads, head_dim_qk, device=device).uniform_(0.5, 1.0).log()
            beta = (torch.rand(batch_size, seqlen, num_o_heads, head_dim_qk, device=device).sigmoid() * 2.0).to(io_dtype)
            write_gate = torch.rand(batch_size, seqlen, num_o_heads, head_dim_vo, device=device).sigmoid().to(io_dtype)
        return gate, beta, write_gate

    #### Done setting up linear attention function per backend ##
    #############################################################

    ###### Linear Attention Benchmark -- Run ######
    ## Print System Info
    if args.verbose:
        print(f"[INFO] {torch.__version__ = }")
        print(f"[INFO] {torch.version.cuda = }")
        print(f"[INFO] {torch.cuda.is_available() = }")
        print(f"[INFO] {torch.cuda.device_count() = }")
        print(f"[INFO] {torch.cuda.current_device() = }")
        print(f"[INFO] {torch.cuda.get_device_name(torch.cuda.current_device()) = }")

    forward_times = []
    backward_times = []
    forward_diffs = []

    total_iters = num_iters + dry_run_iters

    first_error = True  # For suppressing error message beyond first error
    la_function = get_linear_attention_function(args.la_backend)

    # Sample SM clock throughout the benchmark window so SOL% uses the actual
    # boost clock the kernel ran at rather than nvml's (often-stale) max.
    _clock_sampler = _SmClockSampler()
    _clock_sampler.start()
    for i in range(total_iters):
        query = torch.randn(batch_size, seqlen, num_q_heads, head_dim_qk, dtype=target_dtype, device=device)
        key = torch.nn.functional.normalize(torch.randn(batch_size, seqlen, num_q_heads, head_dim_qk, dtype=torch.float32, device=device), dim=-1).to(
            target_dtype
        )
        value = torch.randn(batch_size, seqlen, num_kv_heads, head_dim_vo, dtype=target_dtype, device=device)
        gate, beta, write_gate = generate_gates(target_dtype)

        query, key, value = preprocess_qkv(query, key, value, args.la_backend)
        gate, beta, write_gate = preprocess_gates(gate, beta, write_gate, args.la_backend)
        if run_bwd:
            query.requires_grad_(True)
            key.requires_grad_(True)
            value.requires_grad_(True)
            gate.requires_grad_(True)
            beta.requires_grad_(True)

        # Per-sequence recurrent state ports (once-per-kernel I/O): the
        # initial state seeds the recurrence; the final state is requested
        # with --store_on and its gradient feeds the backward pass.
        s0 = None
        if args.initial_state:
            s0 = torch.randn(batch_size, num_o_heads, head_dim_qk, head_dim_vo, dtype=torch.float32, device=device) * 0.05
            if run_bwd:
                s0.requires_grad_(True)
        if args.la_backend == "cudnn":
            dOutput = torch.randn(batch_size * seqlen, num_o_heads, head_dim_vo, dtype=target_dtype, device=device)
        else:
            dOutput = torch.randn(batch_size, seqlen, num_o_heads, head_dim_vo, dtype=target_dtype, device=device)
        dFinal = None
        if args.store_on and run_bwd:
            dFinal = torch.randn(batch_size, num_o_heads, head_dim_qk, head_dim_vo, dtype=torch.float32, device=device) * 0.05

        l2_flush_buffer.zero_()

        # Run kernel with profiler for forward if requested, else run unprofiled to prep for backward
        if run_fwd:
            with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function("linear_attention.forward"):  # Custom marker
                    output, final_state = la_function(query, key, value, gate, beta, write_gate, s0)
                torch.cuda.synchronize()  # Ensure all kernels finish

            # Filter profiler results by kernel name prefix
            matched_kernels = [
                item
                for item in prof.key_averages()
                if item.key.startswith("cudnn")
                or item.key.startswith("kernel_cutlass")
                or item.key.startswith("triton_")
                or "chunk_" in item.key
                or "l2norm" in item.key
                or "cutile" in item.key
                or "_kernel" in item.key
                or "at::native::" in item.key
                or "(anonymous namespace)::" in item.key
            ]
            if len(matched_kernels) >= 1:
                fwd_time = sum(item.device_time for item in matched_kernels) / 1000
                if i >= dry_run_iters:
                    forward_times.append(fwd_time)
        else:
            output, final_state = la_function(query, key, value, gate, beta, write_gate, s0)
            torch.cuda.synchronize()

        if run_bwd:
            # Run backward pass

            l2_flush_buffer.zero_()

            # With --store_on the loss carries a final-state term, so the
            # backward also exercises the d_final_state path.
            grad_outputs = (output,)
            grads = (dOutput,)
            if args.store_on:
                grad_outputs = (output, final_state)
                grads = (dOutput, dFinal.to(final_state.dtype))

            with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function("linear_attention.backward"):  # Custom marker
                    torch.autograd.backward(grad_outputs, grads)

                    dQuery = query.grad
                    dKey = key.grad
                    dValue = value.grad

                    query.grad = None
                    key.grad = None
                    value.grad = None
                    gate.grad = None
                    beta.grad = None
                torch.cuda.synchronize()

            matched_kernels = [
                item
                for item in prof.key_averages()
                if "cudnn" in item.key
                or item.key.startswith("kernel_cutlass")
                or item.key.startswith("triton_")
                or "chunk_" in item.key
                or "l2norm" in item.key
                or "cutile" in item.key
                or "_kernel" in item.key
                or "at::native::" in item.key
                or "(anonymous namespace)::" in item.key
            ]
            if len(matched_kernels) >= 1:
                bwd_time = sum(item.device_time for item in matched_kernels) / 1000
                if i >= dry_run_iters:
                    backward_times.append(bwd_time)

        output = postprocess_o(output, args.la_backend)
        if not args.skip_ref and run_fwd and args.la_backend != "fla":
            try:
                query_ref = query.detach().reshape(batch_size, seqlen, num_q_heads, head_dim_qk)
                key_ref = key.detach().reshape(batch_size, seqlen, num_q_heads, head_dim_qk)
                value_ref = value.detach().reshape(batch_size, seqlen, num_kv_heads, head_dim_vo)
                gate_ref = gate.detach().reshape(batch_size, seqlen, *gate.shape[1:])
                beta_ref = beta.detach().reshape(batch_size, seqlen, *beta.shape[1:])
                s0_ref = s0.detach() if s0 is not None else None
                output_ref, _ = fla_linear_attention(query_ref, key_ref, value_ref, gate_ref, beta_ref, None, s0_ref)

                torch.testing.assert_close(output.detach(), output_ref, rtol=1e-2, atol=1e-2)
                forward_diffs.append(torch.max(torch.abs(output.detach() - output_ref.detach())).item())
            except Exception as e:
                if first_error:
                    print(
                        f"[WARN] Failed reference check. Target backend has been run, but output has not been validated. Failure may be due to incorrect output or reference function failure."
                    )
                    print(f"[WARN] See error message: {e}")
                    first_error = False
                forward_diffs.append(0.0)
        else:
            forward_diffs.append(0.0)

        del query, key, value, gate, beta, write_gate, output, final_state, s0, dOutput, dFinal

    _clock_sampler.stop()

    ## print results
    fwd_median_time = (
        np.median(np.array(forward_times[5:])) if len(forward_times) > 5 else (np.median(np.array(forward_times)) if len(forward_times) > 0 else 0.0)
    )
    fwd_tflops = 0.0
    if run_fwd and fwd_median_time > 0:
        fwd_tflops = tflops_per_sec(
            args.batch_size,
            args.seqlen,
            head_dim_qk,
            head_dim_vo,
            num_o_heads,
            fwd_median_time,
            "fwd",
        )

    bwd_median_time = (
        np.median(np.array(backward_times[5:])) if len(backward_times) > 5 else (np.median(np.array(backward_times)) if len(backward_times) > 0 else 0.0)
    )
    bwd_tflops = 0.0
    if run_bwd and bwd_median_time > 0:
        bwd_tflops = tflops_per_sec(
            args.batch_size,
            args.seqlen,
            head_dim_qk,
            head_dim_vo,
            num_o_heads,
            bwd_median_time,
            "bwd",
        )

    # Compute MMA SOL% using the per-arch FLOPs/clk/SM table and the actual
    # sampled boost clock observed during the benchmark window.
    _peak_mma_tflops = None
    try:
        _flops_per_clk_per_sm = _peak_flops_per_clock_per_sm(args.data_type)
        _num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        _sampled_mhz = _clock_sampler.peak_mhz()
        if _flops_per_clk_per_sm is not None and _sampled_mhz is not None:
            _peak_mma_tflops = _flops_per_clk_per_sm * _num_sms * _sampled_mhz / 1e6
    except Exception:
        pass

    fwd_sol_str = f", {fwd_tflops / _peak_mma_tflops * 100:.1f}% SOL" if _peak_mma_tflops and fwd_tflops > 0 else ""
    bwd_sol_str = f", {bwd_tflops / _peak_mma_tflops * 100:.1f}% SOL" if _peak_mma_tflops and bwd_tflops > 0 else ""

    if args.format_output:
        print(
            f"{args.case_tag},{args.la_backend},{args.variant},{args.batch_size},{args.seqlen},{args.num_q_heads},{args.num_kv_heads},{head_dim_qk},{fwd_median_time:.3f},{bwd_median_time:.3f},{fwd_tflops:.0f},{bwd_tflops:.0f},{(np.max(np.array(forward_diffs[5:])) if len(forward_diffs) > 5 else (np.max(np.array(forward_diffs)) if len(forward_diffs) > 0 else 0.0)):.6f},{num_iters}"
        )
    else:
        if run_fwd and run_bwd:
            print(
                f"{args.la_backend}/{args.variant}:: Median (fwd, bwd) Execution Times: {fwd_median_time:.3f} ms ({fwd_tflops:.0f} TFLOPS{fwd_sol_str}), {bwd_median_time:.3f} ms ({bwd_tflops:.0f} TFLOPS{bwd_sol_str})"
            )
        elif run_fwd:
            print(f"{args.la_backend}/{args.variant}:: Median (fwd) Execution Time: {fwd_median_time:.3f} ms ({fwd_tflops:.0f} TFLOPS{fwd_sol_str})")
        elif run_bwd:
            print(f"{args.la_backend}/{args.variant}:: Median (bwd) Execution Time: {bwd_median_time:.3f} ms ({bwd_tflops:.0f} TFLOPS{bwd_sol_str})")
