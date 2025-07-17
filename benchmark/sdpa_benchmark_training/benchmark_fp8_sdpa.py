"""
Scaled Dot Product Attention (SDPA) benchmark

This script benchmarks several SDPA backends including cuDNN using CUDA events.
Output csv and png files are saved in the artifacts directory.

"""

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import math
import time
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

###### SDPA Benchmark -- Setup ######
## Define constants for benchmarking
verbose = True
num_iters = 10  # Number of iterations to run for each config; take median time
dry_run_iters = 5  # Number of iterations to dry run for warmup
is_causal = True
enable_gqa = True
precisions = ["fp8", "bf16"]

total_iters = num_iters + dry_run_iters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Define SDPA configs
# Add or remove configs to benchmark; results will be included in output csv.
# Note: Altering the configs may result in incorrectly generated plots.

# (batch_size, q_seqlen, kv_seqlen, num_q_heads, num_kv_heads, head_dim)
sdpa_configs = [
    (1, 512, 512, 128, 8, 128),
    (1, 1024, 1024, 128, 8, 128),
    (1, 2048, 2048, 128, 8, 128),
    (1, 4096, 4096, 128, 8, 128),
    (1, 8192, 8192, 128, 8, 128),
    (1, 16384, 16384, 128, 8, 128),
    (1, 32768, 32768, 128, 8, 128),
    (1, 65536, 65536, 128, 8, 128),
    (1, 131072, 131072, 128, 8, 128),
]

## Define various SDPA functions for each backend


# Util functions for calculating flops and tflops/s achieved
def flops(
    batch_size,
    q_seqlen,
    kv_seqlen,
    head_dim,
    num_q_heads,
    num_kv_heads,
    causal,
    mode="fwd",
):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = (
        4
        * batch_size
        * q_seqlen
        * kv_seqlen
        * num_q_heads
        * head_dim
        // (2 if causal else 1)
    )
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


def tflops_per_sec(
    batch_size,
    q_seqlen,
    kv_seqlen,
    head_dim,
    num_q_heads,
    num_kv_heads,
    causal,
    time,
    mode="fwd",
):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = flops(
        batch_size,
        q_seqlen,
        kv_seqlen,
        head_dim,
        num_q_heads,
        num_kv_heads,
        causal,
        mode,
    )
    return f / time / 1e9 if not math.isnan(time) else 0.0  # Assume time is in msec


## Set up cuDNN Graph
try:
    import cudnn
except ImportError:
    cudnn = None
assert cudnn is not None


print(f"[INFO] cuDNN Backend Version: {cudnn.backend_version() = }")
print(f"[INFO] cuDNN Frontend Version: {cudnn.__version__ = }")

###### SDPA Benchmark -- Run ######
## Print System Info
print(f"[INFO] {torch.__version__ = }")
print(f"[INFO] {torch.version.cuda = }")
print(f"[INFO] {torch.cuda.is_available() = }")
print(f"[INFO] {torch.cuda.device_count() = }")
print(f"[INFO] {torch.cuda.current_device() = }")
print(f"[INFO] {torch.cuda.get_device_name(torch.cuda.current_device()) = }")

## Begin Benchmark
# Use torch's CUDA event to record time
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Define dataframe to store results
data_df = pd.DataFrame(
    columns=[
        "batch_size",
        "q_seqlen",
        "kv_seqlen",
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "is_causal",
        "precision",
        "forward_time",
        "backward_time",
    ]
)

if verbose:
    print(
        f"[INFO] Begin benchmark for layers (batch_size,q_seqlen,kv_seqlen,num_q_heads,num_kv_heads,head_dim)"
    )
    print(f"[INFO] {sdpa_configs = }")

# Iterate over each SDPA config
for sdpa_config in sdpa_configs:
    batch_size, q_seqlen, kv_seqlen, num_q_heads, num_kv_heads, head_dim = sdpa_config
    if verbose:
        print(f"[INFO] Running layer {sdpa_config}")

    # Iterate over each backend
    for cur_precision in precisions:
        print(f"[INFO]   Benchmarking data type {cur_precision}")

        fwd_times = []
        bwd_times = []

        is_dropout = False  # Hard coded
        dropout_prob = dropout_p if is_dropout else 0.0  # Hard coded to 0
        is_infer = False  # Hard coded
        attn_scale = head_dim ** (-0.5)

        ## Will define tensors to set up cuDNN graph once.
        if cur_precision == "fp8":
            query = torch.randint(
                256,
                (batch_size, num_q_heads, q_seqlen, head_dim),
                dtype=torch.uint8,
                device=device,
            )
            key = torch.randint(
                256,
                (batch_size, num_kv_heads, kv_seqlen, head_dim),
                dtype=torch.uint8,
                device=device,
            )
            value = torch.randint(
                256,
                (batch_size, num_kv_heads, kv_seqlen, head_dim),
                dtype=torch.uint8,
                device=device,
            )
            output = torch.empty(
                batch_size,
                num_q_heads,
                q_seqlen,
                head_dim,
                dtype=torch.uint8,
                device=device,
            )

            descale_q_gpu = torch.ones(1, 1, 1, 1, dtype=torch.float, device=device)
            descale_k_gpu = torch.ones(1, 1, 1, 1, dtype=torch.float, device=device)
            descale_v_gpu = torch.ones(1, 1, 1, 1, dtype=torch.float, device=device)
            descale_s_gpu = torch.ones(1, 1, 1, 1, dtype=torch.float, device=device)
            descale_o_gpu = torch.ones(1, 1, 1, 1, dtype=torch.float, device=device)
            descale_dO_gpu = torch.ones(1, 1, 1, 1, dtype=torch.float, device=device)
            descale_dP_gpu = torch.ones(1, 1, 1, 1, dtype=torch.float, device=device)
            scale_s_gpu = torch.ones(1, 1, 1, 1, dtype=torch.float, device=device)
            scale_o_gpu = torch.ones(1, 1, 1, 1, dtype=torch.float, device=device)
            scale_dQ_gpu = torch.ones(1, 1, 1, 1, dtype=torch.float, device=device)
            scale_dK_gpu = torch.ones(1, 1, 1, 1, dtype=torch.float, device=device)
            scale_dV_gpu = torch.ones(1, 1, 1, 1, dtype=torch.float, device=device)
            scale_dP_gpu = torch.ones(1, 1, 1, 1, dtype=torch.float, device=device)
            amax_s_gpu = torch.zeros(1, 1, 1, 1, dtype=torch.float, device=device)
            amax_o_gpu = torch.zeros(1, 1, 1, 1, dtype=torch.float, device=device)
            amax_dQ_gpu = torch.zeros(1, 1, 1, 1, dtype=torch.float, device=device)
            amax_dK_gpu = torch.zeros(1, 1, 1, 1, dtype=torch.float, device=device)
            amax_dV_gpu = torch.zeros(1, 1, 1, 1, dtype=torch.float, device=device)
            amax_dP_gpu = torch.zeros(1, 1, 1, 1, dtype=torch.float, device=device)
        elif cur_precision == "bf16":
            query = torch.randn(
                batch_size,
                num_q_heads,
                q_seqlen,
                head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            key = torch.randn(
                batch_size,
                num_kv_heads,
                kv_seqlen,
                head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            value = torch.randn(
                batch_size,
                num_kv_heads,
                kv_seqlen,
                head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            output = torch.empty(
                batch_size,
                num_q_heads,
                q_seqlen,
                head_dim,
                dtype=torch.bfloat16,
                device=device,
            )

        dQuery = torch.empty_like(query)
        dKey = torch.empty_like(key)
        dValue = torch.empty_like(value)
        dOutput = torch.empty_like(output)
        stats = torch.empty(
            batch_size, num_q_heads, q_seqlen, 1, dtype=torch.float32, device=device
        )
        if is_dropout:
            dropout_seed = torch.full(
                (1, 1, 1, 1), 123456, dtype=torch.int64, device="cuda"
            )
            dropout_offset = torch.full(
                (1, 1, 1, 1), 789, dtype=torch.int64, device="cuda"
            )

        # cuDNN graph forward
        graph_fwd = cudnn.pygraph(
            io_data_type=(
                cudnn.data_type.FP8_E4M3
                if cur_precision == "fp8"
                else cudnn.data_type.BFLOAT16
            ),
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )

        if is_dropout:
            seed_fwd = graph_fwd.tensor_like(dropout_seed)
            offset_fwd = graph_fwd.tensor_like(dropout_offset)
            dropout_tuple = (dropout_prob, seed_fwd, offset_fwd)

        if cur_precision == "fp8":
            q_fwd = graph_fwd.tensor_like(query).set_data_type(cudnn.data_type.FP8_E4M3)
            k_fwd = graph_fwd.tensor_like(key).set_data_type(cudnn.data_type.FP8_E4M3)
            v_fwd = graph_fwd.tensor_like(value).set_data_type(cudnn.data_type.FP8_E4M3)

            descale_q_fwd = graph_fwd.tensor_like(descale_q_gpu)
            descale_k_fwd = graph_fwd.tensor_like(descale_k_gpu)
            descale_v_fwd = graph_fwd.tensor_like(descale_v_gpu)
            descale_s_fwd = graph_fwd.tensor_like(descale_s_gpu)
            scale_s_fwd = graph_fwd.tensor_like(scale_s_gpu)
            scale_o_fwd = graph_fwd.tensor_like(scale_o_gpu)

            o_fwd, stats_fwd, amax_s_fwd, amax_o_fwd = graph_fwd.sdpa_fp8(
                q=q_fwd,
                k=k_fwd,
                v=v_fwd,
                descale_q=descale_q_fwd,
                descale_k=descale_k_fwd,
                descale_v=descale_v_fwd,
                descale_s=descale_s_fwd,
                scale_s=scale_s_fwd,
                scale_o=scale_o_fwd,
                # generate_stats=not is_infer,
                is_inference=is_infer,
                attn_scale=attn_scale,
                use_causal_mask=is_causal,
                use_padding_mask=False,
                # dropout=dropout_tuple if is_dropout else None,
            )
        elif cur_precision == "bf16":
            q_fwd = graph_fwd.tensor_like(query)
            k_fwd = graph_fwd.tensor_like(key)
            v_fwd = graph_fwd.tensor_like(value)
            o_fwd, stats_fwd = graph_fwd.sdpa(
                q=q_fwd,
                k=k_fwd,
                v=v_fwd,
                # generate_stats=not is_infer,
                is_inference=is_infer,
                attn_scale=attn_scale,
                use_causal_mask=is_causal,
                use_causal_mask_bottom_right=False,
                dropout=dropout_tuple if is_dropout else None,
            )

        if cur_precision == "fp8":
            o_fwd.set_output(True).set_dim(output.size()).set_stride(
                output.stride()
            ).set_data_type(cudnn.data_type.FP8_E4M3)
            (
                stats_fwd.set_output(True)
                .set_dim(stats.size())
                .set_stride(stats.stride())
                .set_data_type(cudnn.data_type.FLOAT)
                if not is_infer
                else None
            )
        elif cur_precision == "bf16":
            o_fwd.set_output(True).set_dim(output.size()).set_stride(output.stride())
            (
                stats_fwd.set_output(True)
                .set_dim(stats.size())
                .set_stride(stats.stride())
                .set_data_type(cudnn.data_type.FLOAT)
                if not is_infer
                else None
            )

        if cur_precision == "fp8":
            amax_s_fwd.set_output(True).set_dim(amax_s_gpu.size()).set_stride(
                amax_s_gpu.stride()
            ).set_data_type(cudnn.data_type.FLOAT)
            amax_o_fwd.set_output(True).set_dim(amax_o_gpu.size()).set_stride(
                amax_o_gpu.stride()
            ).set_data_type(cudnn.data_type.FLOAT)

        graph_fwd.validate()
        graph_fwd.build_operation_graph()
        graph_fwd.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph_fwd.check_support()
        graph_fwd.build_plans()

        # Now BWD
        graph_bwd = cudnn.pygraph(
            io_data_type=(
                cudnn.data_type.FP8_E4M3
                if cur_precision == "fp8"
                else cudnn.data_type.HALF
            ),
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )

        stats_bwd = graph_bwd.tensor_like(stats)
        if is_dropout:
            seed_bwd = graph_bwd.tensor_like(dropout_seed)
            offset_bwd = graph_bwd.tensor_like(dropout_offset)
            dropout_tuple = (dropout_prob, seed_bwd, offset_bwd)

        if cur_precision == "fp8":
            q_bwd = graph_bwd.tensor_like(query).set_data_type(cudnn.data_type.FP8_E4M3)
            k_bwd = graph_bwd.tensor_like(key).set_data_type(cudnn.data_type.FP8_E4M3)
            v_bwd = graph_bwd.tensor_like(value).set_data_type(cudnn.data_type.FP8_E4M3)
            o_bwd = graph_bwd.tensor_like(output).set_data_type(
                cudnn.data_type.FP8_E4M3
            )
            dO_bwd = graph_bwd.tensor_like(dOutput).set_data_type(
                cudnn.data_type.FP8_E4M3
            )

            descale_q_bwd = graph_bwd.tensor_like(descale_q_gpu)
            descale_k_bwd = graph_bwd.tensor_like(descale_k_gpu)
            descale_v_bwd = graph_bwd.tensor_like(descale_v_gpu)
            descale_o_bwd = graph_bwd.tensor_like(descale_o_gpu)
            descale_dO_bwd = graph_bwd.tensor_like(descale_dO_gpu)
            descale_s_bwd = graph_bwd.tensor_like(descale_s_gpu)
            descale_dP_bwd = graph_bwd.tensor_like(descale_dP_gpu)
            scale_s_bwd = graph_bwd.tensor_like(scale_s_gpu)
            scale_dQ_bwd = graph_bwd.tensor_like(scale_dQ_gpu)
            scale_dK_bwd = graph_bwd.tensor_like(scale_dK_gpu)
            scale_dV_bwd = graph_bwd.tensor_like(scale_dV_gpu)
            scale_dP_bwd = graph_bwd.tensor_like(scale_dP_gpu)

            (
                dQ_bwd,
                dK_bwd,
                dV_bwd,
                amax_dQ_bwd,
                amax_dK_bwd,
                amax_dV_bwd,
                amax_dP_bwd,
            ) = graph_bwd.sdpa_fp8_backward(
                q=q_bwd,
                k=k_bwd,
                v=v_bwd,
                o=o_bwd,
                dO=dO_bwd,
                stats=stats_bwd,
                descale_q=descale_q_bwd,
                descale_k=descale_k_bwd,
                descale_v=descale_v_bwd,
                descale_o=descale_o_bwd,
                descale_dO=descale_dO_bwd,
                descale_s=descale_s_bwd,
                descale_dP=descale_dP_bwd,
                scale_s=scale_s_bwd,
                scale_dQ=scale_dQ_bwd,
                scale_dK=scale_dK_bwd,
                scale_dV=scale_dV_bwd,
                scale_dP=scale_dP_bwd,
                attn_scale=attn_scale,
                use_causal_mask=is_causal,
                use_causal_mask_bottom_right=False,
                dropout=dropout_tuple if is_dropout else None,
            )
        elif cur_precision == "bf16":
            q_bwd = graph_bwd.tensor_like(query)
            k_bwd = graph_bwd.tensor_like(key)
            v_bwd = graph_bwd.tensor_like(value)
            o_bwd = graph_bwd.tensor_like(output)
            dO_bwd = graph_bwd.tensor_like(dOutput)

            dQ_bwd, dK_bwd, dV_bwd = graph_bwd.sdpa_backward(
                q=q_bwd,
                k=k_bwd,
                v=v_bwd,
                o=o_bwd,
                dO=dO_bwd,
                stats=stats_bwd,
                attn_scale=attn_scale,
                use_causal_mask=is_causal,
                use_causal_mask_bottom_right=False,
                dropout=dropout_tuple if is_dropout else None,
            )

        if cur_precision == "fp8":
            dQ_bwd.set_output(True).set_dim(dQuery.size()).set_stride(
                dQuery.stride()
            ).set_data_type(cudnn.data_type.FP8_E4M3)
            dK_bwd.set_output(True).set_dim(dKey.size()).set_stride(
                dKey.stride()
            ).set_data_type(cudnn.data_type.FP8_E4M3)
            dV_bwd.set_output(True).set_dim(dValue.size()).set_stride(
                dValue.stride()
            ).set_data_type(cudnn.data_type.FP8_E4M3)
            amax_dQ_bwd.set_output(True).set_dim(amax_dQ_gpu.size()).set_stride(
                amax_dQ_gpu.stride()
            ).set_data_type(cudnn.data_type.FLOAT)
            amax_dK_bwd.set_output(True).set_dim(amax_dK_gpu.size()).set_stride(
                amax_dK_gpu.stride()
            ).set_data_type(cudnn.data_type.FLOAT)
            amax_dV_bwd.set_output(True).set_dim(amax_dV_gpu.size()).set_stride(
                amax_dV_gpu.stride()
            ).set_data_type(cudnn.data_type.FLOAT)
            amax_dP_bwd.set_output(True).set_dim(amax_dP_gpu.size()).set_stride(
                amax_dP_gpu.stride()
            ).set_data_type(cudnn.data_type.FLOAT)
        elif cur_precision == "bf16":
            dQ_bwd.set_output(True).set_dim(dQuery.size()).set_stride(dQuery.stride())
            dK_bwd.set_output(True).set_dim(dKey.size()).set_stride(dKey.stride())
            dV_bwd.set_output(True).set_dim(dValue.size()).set_stride(dValue.stride())

        graph_bwd.validate()
        graph_bwd.build_operation_graph()
        graph_bwd.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph_bwd.check_support()
        graph_bwd.build_plans()

        if cur_precision == "fp8":
            variant_pack_fwd = {
                q_fwd: query,
                k_fwd: key,
                v_fwd: value,
                o_fwd: output,
                stats_fwd: stats,
                descale_q_fwd: descale_q_gpu,
                descale_k_fwd: descale_k_gpu,
                descale_v_fwd: descale_v_gpu,
                descale_s_fwd: descale_s_gpu,
                scale_s_fwd: scale_s_gpu,
                scale_o_fwd: scale_o_gpu,
                amax_s_fwd: amax_s_gpu,
                amax_o_fwd: amax_o_gpu,
            }

            variant_pack_bwd = {
                q_fwd: query,
                k_fwd: key,
                v_fwd: value,
                o_fwd: output,
                dQ_bwd: dQuery,
                dK_bwd: dKey,
                dV_bwd: dValue,
                dO_bwd: dOutput,
                stats_bwd: stats,
                descale_q_bwd: descale_q_gpu,
                descale_k_bwd: descale_k_gpu,
                descale_v_bwd: descale_v_gpu,
                descale_o_bwd: descale_o_gpu,
                descale_s_bwd: descale_s_gpu,
                descale_dP_bwd: descale_dP_gpu,
                descale_dO_bwd: descale_dO_gpu,
                scale_s_bwd: scale_s_gpu,
                scale_dQ_bwd: scale_dQ_gpu,
                scale_dK_bwd: scale_dK_gpu,
                scale_dV_bwd: scale_dV_gpu,
                scale_dP_bwd: scale_dP_gpu,
                amax_dQ_bwd: amax_dQ_gpu,
                amax_dK_bwd: amax_dK_gpu,
                amax_dV_bwd: amax_dV_gpu,
                amax_dP_bwd: amax_dP_gpu,
            }

            workspace = torch.empty(
                max(graph_fwd.get_workspace_size(), graph_bwd.get_workspace_size()),
                device="cuda",
                dtype=torch.uint8,
            )
        elif cur_precision == "bf16":
            variant_pack_fwd = {
                q_fwd: query,
                k_fwd: key,
                v_fwd: value,
                o_fwd: output,
                stats_fwd: stats,
            }
            variant_pack_bwd = {
                q_bwd: query,
                k_bwd: key,
                v_bwd: value,
                o_bwd: output,
                dO_bwd: dOutput,
                stats_bwd: stats,
                dQ_bwd: dQuery,
                dK_bwd: dKey,
                dV_bwd: dValue,
            }
            workspace = torch.empty(
                max(graph_fwd.get_workspace_size(), graph_bwd.get_workspace_size()),
                device="cuda",
                dtype=torch.uint8,
            )

        if is_dropout:
            variant_pack_fwd[seed_fwd] = dropout_seed
            variant_pack_fwd[offset_fwd] = dropout_offset
            variant_pack_bwd[seed_bwd] = dropout_seed
            variant_pack_bwd[offset_bwd] = dropout_offset

        if cur_precision == "fp8":
            del query, key, value, output, dQuery, dKey, dValue, dOutput, stats
        elif cur_precision == "bf16":
            del query, key, value, output, dQuery, dKey, dValue, dOutput, stats

        # Repeat measurement for warmup and actual measurement
        for i in range(total_iters):
            fwd_time = np.inf
            bwd_time = np.inf

            if True:
                if cur_precision == "fp8":
                    query = torch.randint(
                        256,
                        (batch_size, num_q_heads, q_seqlen, head_dim),
                        dtype=torch.uint8,
                        device=device,
                    )
                    key = torch.randint(
                        256,
                        (batch_size, num_kv_heads, kv_seqlen, head_dim),
                        dtype=torch.uint8,
                        device=device,
                    )
                    value = torch.randint(
                        256,
                        (batch_size, num_kv_heads, kv_seqlen, head_dim),
                        dtype=torch.uint8,
                        device=device,
                    )
                    descale_q_gpu = torch.ones(
                        1, 1, 1, 1, dtype=torch.float, device=device
                    )
                    descale_k_gpu = torch.ones(
                        1, 1, 1, 1, dtype=torch.float, device=device
                    )
                    descale_v_gpu = torch.ones(
                        1, 1, 1, 1, dtype=torch.float, device=device
                    )
                    descale_s_gpu = torch.ones(
                        1, 1, 1, 1, dtype=torch.float, device=device
                    )
                    descale_o_gpu = torch.ones(
                        1, 1, 1, 1, dtype=torch.float, device=device
                    )
                    descale_dO_gpu = torch.ones(
                        1, 1, 1, 1, dtype=torch.float, device=device
                    )
                    descale_dP_gpu = torch.ones(
                        1, 1, 1, 1, dtype=torch.float, device=device
                    )
                    scale_s_gpu = torch.ones(
                        1, 1, 1, 1, dtype=torch.float, device=device
                    )
                    scale_o_gpu = torch.ones(
                        1, 1, 1, 1, dtype=torch.float, device=device
                    )
                    scale_dQ_gpu = torch.ones(
                        1, 1, 1, 1, dtype=torch.float, device=device
                    )
                    scale_dK_gpu = torch.ones(
                        1, 1, 1, 1, dtype=torch.float, device=device
                    )
                    scale_dV_gpu = torch.ones(
                        1, 1, 1, 1, dtype=torch.float, device=device
                    )
                    scale_dP_gpu = torch.ones(
                        1, 1, 1, 1, dtype=torch.float, device=device
                    )
                    amax_s_gpu = torch.zeros(
                        1, 1, 1, 1, dtype=torch.float, device=device
                    )
                    amax_o_gpu = torch.zeros(
                        1, 1, 1, 1, dtype=torch.float, device=device
                    )
                    amax_dQ_gpu = torch.zeros(
                        1, 1, 1, 1, dtype=torch.float, device=device
                    )
                    amax_dK_gpu = torch.zeros(
                        1, 1, 1, 1, dtype=torch.float, device=device
                    )
                    amax_dV_gpu = torch.zeros(
                        1, 1, 1, 1, dtype=torch.float, device=device
                    )
                    amax_dP_gpu = torch.zeros(
                        1, 1, 1, 1, dtype=torch.float, device=device
                    )
                elif cur_precision == "bf16":
                    query = torch.randn(
                        batch_size,
                        num_q_heads,
                        q_seqlen,
                        head_dim,
                        dtype=torch.bfloat16,
                        device=device,
                        requires_grad=True,
                    )
                    key = torch.randn(
                        batch_size,
                        num_kv_heads,
                        kv_seqlen,
                        head_dim,
                        dtype=torch.bfloat16,
                        device=device,
                        requires_grad=True,
                    )
                    value = torch.randn(
                        batch_size,
                        num_kv_heads,
                        kv_seqlen,
                        head_dim,
                        dtype=torch.bfloat16,
                        device=device,
                        requires_grad=True,
                    )

                output = torch.empty(
                    batch_size,
                    num_q_heads,
                    q_seqlen,
                    head_dim,
                    dtype=torch.uint8 if cur_precision == "fp8" else torch.bfloat16,
                    device=device,
                )
                dQuery = torch.empty_like(query)
                dKey = torch.empty_like(key)
                dValue = torch.empty_like(value)
                dOutput = torch.empty_like(output)
                stats = torch.empty(
                    batch_size,
                    num_q_heads,
                    q_seqlen,
                    1,
                    dtype=torch.float32,
                    device=device,
                )
                if is_dropout:
                    dropout_seed = torch.full(
                        (1, 1, 1, 1), 123456, dtype=torch.int64, device="cuda"
                    )
                    dropout_offset = torch.full(
                        (1, 1, 1, 1), 789, dtype=torch.int64, device="cuda"
                    )

                if cur_precision == "fp8":
                    variant_pack_fwd = {
                        q_fwd: query,
                        k_fwd: key,
                        v_fwd: value,
                        o_fwd: output,
                        stats_fwd: stats,
                        descale_q_fwd: descale_q_gpu,
                        descale_k_fwd: descale_k_gpu,
                        descale_v_fwd: descale_v_gpu,
                        descale_s_fwd: descale_s_gpu,
                        scale_s_fwd: scale_s_gpu,
                        scale_o_fwd: scale_o_gpu,
                        amax_s_fwd: amax_s_gpu,
                        amax_o_fwd: amax_o_gpu,
                    }
                    variant_pack_bwd = {
                        q_bwd: query,
                        k_bwd: key,
                        v_bwd: value,
                        o_bwd: output,
                        dQ_bwd: dQuery,
                        dK_bwd: dKey,
                        dV_bwd: dValue,
                        dO_bwd: dOutput,
                        stats_bwd: stats,
                        descale_q_bwd: descale_q_gpu,
                        descale_k_bwd: descale_k_gpu,
                        descale_v_bwd: descale_v_gpu,
                        descale_o_bwd: descale_o_gpu,
                        descale_s_bwd: descale_s_gpu,
                        descale_dP_bwd: descale_dP_gpu,
                        descale_dO_bwd: descale_dO_gpu,
                        scale_s_bwd: scale_s_gpu,
                        scale_dQ_bwd: scale_dQ_gpu,
                        scale_dK_bwd: scale_dK_gpu,
                        scale_dV_bwd: scale_dV_gpu,
                        scale_dP_bwd: scale_dP_gpu,
                        amax_dQ_bwd: amax_dQ_gpu,
                        amax_dK_bwd: amax_dK_gpu,
                        amax_dV_bwd: amax_dV_gpu,
                        amax_dP_bwd: amax_dP_gpu,
                    }
                elif cur_precision == "bf16":
                    variant_pack_fwd = {
                        q_fwd: query,
                        k_fwd: key,
                        v_fwd: value,
                        o_fwd: output,
                        stats_fwd: stats,
                    }
                    variant_pack_bwd = {
                        q_bwd: query,
                        k_bwd: key,
                        v_bwd: value,
                        o_bwd: output,
                        dO_bwd: dOutput,
                        stats_bwd: stats,
                        dQ_bwd: dQuery,
                        dK_bwd: dKey,
                        dV_bwd: dValue,
                    }
                workspace = torch.empty(
                    max(graph_fwd.get_workspace_size(), graph_bwd.get_workspace_size()),
                    device="cuda",
                    dtype=torch.uint8,
                )

                if is_dropout:
                    variant_pack_fwd[seed_fwd] = dropout_seed
                    variant_pack_fwd[offset_fwd] = dropout_offset
                    variant_pack_bwd[seed_bwd] = dropout_seed
                    variant_pack_bwd[offset_bwd] = dropout_offset

                # Run forward pass
                with torch.autograd.profiler.emit_nvtx():
                    torch.cuda.nvtx.range_push(f"sdpa.forward.{cur_precision}")
                    start_event.record()
                    graph_fwd.execute(variant_pack_fwd, workspace)
                    end_event.record()
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_pop()
                # Record forward pass time
                fwd_time = start_event.elapsed_time(end_event)

                # For stable measurements, sleep for some time proportional to fwd_time
                if fwd_time is not np.inf:
                    sleep_time = np.min([fwd_time / 100, 1.0])
                else:
                    sleep_time = 0.01
                time.sleep(sleep_time)

                # Run backward pass
                if cur_precision == "fp8":
                    grad_output = torch.empty_like(output)
                elif cur_precision == "bf16":
                    grad_output = torch.randn_like(output)

                with torch.autograd.profiler.emit_nvtx():
                    torch.cuda.nvtx.range_push(f"sdpa.backward.{cur_precision}")
                    start_event.record()
                    graph_bwd.execute(variant_pack_bwd, workspace)
                    end_event.record()
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_pop()
                # Record backward pass time
                bwd_time = start_event.elapsed_time(end_event)

                # Postprocess output for ref check and run refcheck
                # Refcheck only checks forward pass

                # For stable measurements, sleep for some time proportional to fwd_time
                if bwd_time is not np.inf:
                    sleep_time = np.min([bwd_time / 100, 1.0])
                else:
                    sleep_time = 0.01
                time.sleep(sleep_time)

                # Clear tensors
                del query, key, value, output
            # except Exception as e:
            #     print(e)
            #     pass

            # Only record time after warmup
            if i >= dry_run_iters:
                fwd_times.append(fwd_time)
                bwd_times.append(bwd_time)

        # Append data to table
        data_df.loc[len(data_df)] = {
            "batch_size": batch_size,
            "q_seqlen": q_seqlen,
            "kv_seqlen": kv_seqlen,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "is_causal": is_causal,
            "precision": cur_precision,
            "forward_time": np.median(np.array(fwd_times)),  # Median fwd pass time
            "backward_time": np.median(np.array(bwd_times)),  # Median bwd pass time
        }

# Compute TFLOPs/sec achieved for each row in data_df
data_df["fwd_tflops_per_sec"] = data_df.apply(
    lambda row: tflops_per_sec(
        batch_size=row["batch_size"],
        q_seqlen=row["q_seqlen"],
        kv_seqlen=row["kv_seqlen"],
        head_dim=row["head_dim"],
        num_q_heads=row["num_q_heads"],
        num_kv_heads=row["num_kv_heads"],
        causal=row["is_causal"],
        time=row["forward_time"],
        mode="fwd",
    ),
    axis=1,
)

data_df["bwd_tflops_per_sec"] = data_df.apply(
    lambda row: tflops_per_sec(
        batch_size=row["batch_size"],
        q_seqlen=row["q_seqlen"],
        kv_seqlen=row["kv_seqlen"],
        head_dim=row["head_dim"],
        num_q_heads=row["num_q_heads"],
        num_kv_heads=row["num_kv_heads"],
        causal=row["is_causal"],
        time=row["backward_time"],
        mode="bwd",
    ),
    axis=1,
)

## Save results to a csv file
gpu_name = torch.cuda.get_device_name(torch.cuda.current_device()).replace(" ", "_")
output_file_name = f"./artifacts/sdpa_fp8_benchmark_results_{gpu_name}.csv"
if verbose:
    print(f"[INFO] Saving results to {output_file_name}")
try:
    data_df.to_csv(output_file_name, float_format="%.3f", index=False)
except Exception as e:
    print(f"[ERROR] Failed to save results to {output_file_name}: {e}")
    print(f"[INFO] Printing results to console instead")
    print(data_df.to_csv(float_format="%.3f", index=False))
    print(f"[INFO] Printing results to console done")

###### SDPA Benchmark -- Plot ######
## Generate plots for (num_q_heads=128, num_kv_heads=8, head_dim=128, is_causal=True)
baseline_df = data_df[
    (data_df["precision"] == "bf16") & (data_df["q_seqlen"] >= 4000)
].copy()
baseline_df.drop(
    columns=[
        "precision",
        "fwd_tflops_per_sec",
        "bwd_tflops_per_sec",
    ],
    inplace=True,
)
baseline_df.rename(
    columns={
        "forward_time": "baseline_forward_time",
        "backward_time": "baseline_backward_time",
    },
    inplace=True,
)

merged_df = baseline_df.merge(
    data_df,
    on=[
        "batch_size",
        "q_seqlen",
        "kv_seqlen",
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "is_causal",
    ],
)
merged_df["fwd_speedup"] = (
    merged_df["baseline_forward_time"] / merged_df["forward_time"]
)
merged_df["bwd_speedup"] = (
    merged_df["baseline_backward_time"] / merged_df["backward_time"]
)


# Configurations for bar plots
precision_ordering = {"bf16": 0, "fp8": 1}
precision_name = {"bf16": "BFloat16", "fp8": "FP8"}
precision_barplot_color = {
    precision_name["bf16"]: "#76b900",
    precision_name["fp8"]: "darkgreen",
}
LABEL_FONT_SIZE = 8
LEGEND_FONT_SIZE = 6
TITLE_FONT_SIZE = 9
# Select desired cases
plot_df = merged_df[
    (merged_df["is_causal"] == True)
    & (merged_df["num_q_heads"] == 128)
    & (merged_df["num_kv_heads"] == 8)
    & (merged_df["q_seqlen"] == merged_df["kv_seqlen"])
    & (merged_df["head_dim"] == 128)
].copy()

plot_df["precision_rank"] = plot_df["precision"].map(precision_ordering)
plot_df["precision_name"] = plot_df["precision"].map(precision_name)
plot_df.sort_values(["q_seqlen", "precision_rank"], inplace=True)

# Generate plots: forward on left subplot and backward on right subplot
YLIM_MAX = np.max([plot_df["fwd_speedup"].max(), plot_df["bwd_speedup"].max()]) * 1.1

plt.figure(figsize=(10, 4), dpi=200)
plt.subplot(1, 2, 1)
cur_plot_df = plot_df[plot_df.fwd_tflops_per_sec > 0]
ax = sns.barplot(
    data=cur_plot_df,
    x="q_seqlen",
    y="fwd_speedup",
    hue="precision_name",
    edgecolor="black",
    linewidth=0.5,
    palette=precision_barplot_color,
    width=0.6,
)
ax.legend_.set_title(None)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2fx", fontsize=6)
plt.xticks(rotation=45)
plt.xlabel("Sequence Length", fontsize=LABEL_FONT_SIZE)
plt.ylabel("Speedup", fontsize=LABEL_FONT_SIZE)
plt.title("SDPA Forward", fontsize=TITLE_FONT_SIZE)
plt.tick_params(axis="y", which="major", labelsize=LABEL_FONT_SIZE)
plt.tick_params(axis="x", which="major", labelsize=LABEL_FONT_SIZE)
plt.ylim(0.5, YLIM_MAX)
plt.legend(fontsize=LEGEND_FONT_SIZE, loc="upper left")

plt.subplot(1, 2, 2)
cur_plot_df = plot_df[plot_df.bwd_tflops_per_sec > 0]
ax = sns.barplot(
    data=cur_plot_df,
    x="q_seqlen",
    y="bwd_speedup",
    hue="precision_name",
    edgecolor="black",
    linewidth=0.5,
    palette=precision_barplot_color,
    width=0.6,
)
ax.legend_.set_title(None)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2fx", fontsize=6)
plt.xticks(rotation=45)
plt.xlabel("Sequence Length", fontsize=LABEL_FONT_SIZE)
plt.ylabel("Speedup", fontsize=LABEL_FONT_SIZE)
plt.title("SDPA Backward", fontsize=TITLE_FONT_SIZE)
plt.tick_params(axis="y", which="major", labelsize=LABEL_FONT_SIZE)
plt.tick_params(axis="x", which="major", labelsize=LABEL_FONT_SIZE)
plt.ylim(0.5, YLIM_MAX)
plt.legend(fontsize=LEGEND_FONT_SIZE, loc="upper left")

# Save plot
plt.tight_layout()
png_file_name = f"./artifacts/sdpa_fp8_benchmark_results_{gpu_name}.png"
if verbose:
    print(f"[INFO] Saving plot to {png_file_name}")
try:
    plt.savefig(png_file_name)
except Exception as e:
    print(f"[ERROR] Failed to save plot to {png_file_name}: {e}")
