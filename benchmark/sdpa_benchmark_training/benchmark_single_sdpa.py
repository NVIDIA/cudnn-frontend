"""
Scaled Dot Product Attention (SDPA) benchmark

This script benchmarks a single SDPA compute instance.
The SDPA backend can be chosen. Performance is measured using CUDA events.

"""

import argparse
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import os
import numpy as np
import functools
import time
import math

###### SDPA Benchmark -- Parse input arguments ######
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--batch_size", default=1, type=int, help="Batch size to input to the layer"
)
parser.add_argument(
    "--q_seqlen", default=8192, type=int, help="Sequence length to input to the layer"
)
parser.add_argument(
    "--kv_seqlen", default=8192, type=int, help="Sequence length to input to the layer"
)
parser.add_argument(
    "--num_q_heads",
    default=16,
    type=int,
    help="Number of query heads to input to the layer",
)
parser.add_argument(
    "--num_kv_heads",
    default=8,
    type=int,
    help="Number of key/value heads to input to the layer",
)
parser.add_argument(
    "--head_dim", default=128, type=int, help="Head dimension to input to the layer"
)
parser.add_argument(
    "--data_type",
    default="bfloat16",
    type=str,
    help="Data type to input to the layer. Can be bfloat16, float16, or fp8",
)
parser.add_argument(
    "--num_iterations",
    default=20,
    type=int,
    help="Number of iterations to run the layer",
)
parser.add_argument("--is_causal", action="store_true", help="Is causal masking on")
parser.add_argument("--verbose", action="store_true", help="Verbose output")
parser.add_argument(
    "--fwd_bwd",
    action="store_true",
    help="Run both forward and backward pass (fwd only by default)",
)
parser.add_argument(
    "--attn_mask",
    default="no_mask",
    type=str,
    help="Attn mask to use. Can be 'padding_causal' or 'no_mask'. If padding_causal, is_causal must be set to false. Only works for cuDNN FE or PyTorch backends.",
    choices=["padding_causal", "no_mask"],
)
parser.add_argument(
    "--sdpa_backend",
    default="pyt_cudnn",
    type=str,
    help="SDPA backend to use",
    choices=[
        "pyt_native",
        "pyt_math",
        "pyt_cudnn",
        "pyt_efficient_attention",
        "pyt_flash_attention",
        "flash_attention",
        "flash_attention_3",
        "cudnn_fe",
    ],
)
parser.add_argument(
    "--format_output", action="store_true", help="Format output to be used in benchmark"
)
parser.add_argument(
    "--case_tag",
    default="",
    type=str,
    help="Tag to identify the case. Not used in calculations. Only for formatted output",
)
args = parser.parse_args()

if args.data_type == "bfloat16":
    target_dtype = torch.bfloat16
elif args.data_type == "float16":
    target_dtype = torch.float16
elif args.data_type == "float":
    target_dtype = torch.float
elif args.data_type == "fp8":
    target_dtype = None
else:
    raise ValueError(f"Invalid data type: {args.data_type}")

if args.data_type == "fp8":
    if args.sdpa_backend not in ["cudnn_fe", "flash_attention_3"]:
        raise ValueError(
            f"FP8 is only supported for cudnn_fe and flash_attention_3 backends"
        )

if args.attn_mask == "padding_causal":
    assert not args.is_causal, "Padding causal attn mask requires is_causal to be false"
    assert (
        args.q_seqlen <= args.kv_seqlen
    ), "Padding causal attn mask requires q_seqlen <= kv_seqlen"

# Parse input arguments
num_iters = args.num_iterations
batch_size = args.batch_size
q_seqlen = args.q_seqlen
kv_seqlen = args.kv_seqlen
num_q_heads = args.num_q_heads
num_kv_heads = args.num_kv_heads
head_dim = args.head_dim
is_causal = args.is_causal
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda", "Requires CUDA device"

enable_gqa = num_q_heads != num_kv_heads

#############################################################
########### Set up SDPA function for each backend ###########
## Define various SDPA functions for each backend
if args.attn_mask == "padding_causal":
    # Mask construction: rectangular tensor + triangular tensor
    rect_mask = torch.ones(
        q_seqlen, (kv_seqlen - q_seqlen), dtype=torch.bool, device=device
    )
    tri_mask = torch.tril(
        torch.ones(q_seqlen, q_seqlen, dtype=torch.bool, device=device)
    )

    attn_mask = torch.cat(
        [rect_mask, tri_mask], dim=1
    )  # .unsqueeze(0).repeat(batch_size, 1, 1)
    padding_fraction = attn_mask.sum() / attn_mask.numel()
else:
    padding_fraction = 0.0

## If using cuDNN FE, set up cuDNN graph.
if args.sdpa_backend == "cudnn_fe":
    is_dropout = False  # Hard coded
    dropout_prob = dropout_p if is_dropout else 0.0  # Hard coded to 0
    is_infer = False  # Hard coded
    attn_scale = args.head_dim ** (-0.5)

    try:
        import cudnn
    except ImportError:
        cudnn = None
    assert cudnn is not None

    if args.verbose:
        print(f"[INFO] cuDNN Backend Version: {cudnn.backend_version() = }")
        print(f"[INFO] cuDNN Frontend Version: {cudnn.__version__ = }")

    # Helper function: Convert torch type to cuDNN type
    def convert_to_cudnn_type(torch_type):
        if torch_type == torch.float16:
            return cudnn.data_type.HALF
        elif torch_type == torch.bfloat16:
            return cudnn.data_type.BFLOAT16
        elif torch_type == torch.float32:
            return cudnn.data_type.FLOAT
        elif torch_type == torch.int32:
            return cudnn.data_type.INT32
        elif torch_type == torch.int64:
            return cudnn.data_type.INT64
        else:
            raise ValueError("Unsupported tensor data type.")

    ## Will define tensors to set up cuDNN graph once.
    if args.data_type == "fp8":
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
    else:
        query = torch.randn(
            batch_size,
            num_q_heads,
            q_seqlen,
            head_dim,
            dtype=target_dtype,
            device=device,
        )
        key = torch.randn(
            batch_size,
            num_kv_heads,
            kv_seqlen,
            head_dim,
            dtype=target_dtype,
            device=device,
        )
        value = torch.randn(
            batch_size,
            num_kv_heads,
            kv_seqlen,
            head_dim,
            dtype=target_dtype,
            device=device,
        )
        output = torch.empty(
            batch_size,
            num_q_heads,
            q_seqlen,
            head_dim,
            dtype=target_dtype,
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
        dropout_offset = torch.full((1, 1, 1, 1), 789, dtype=torch.int64, device="cuda")

    # cuDNN graph forward
    graph_fwd = cudnn.pygraph(
        io_data_type=(
            cudnn.data_type.FP8_E4M3
            if args.data_type == "fp8"
            else convert_to_cudnn_type(target_dtype)
        ),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    if is_dropout:
        seed_fwd = graph_fwd.tensor_like(dropout_seed)
        offset_fwd = graph_fwd.tensor_like(dropout_offset)
        dropout_tuple = (dropout_prob, seed_fwd, offset_fwd)

    if args.data_type == "fp8":
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
    else:
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
            use_causal_mask_bottom_right=args.attn_mask == "padding_causal",
            dropout=dropout_tuple if is_dropout else None,
        )

    if args.fwd_bwd:
        if args.data_type == "fp8":
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
        else:
            o_fwd.set_output(True).set_dim(output.size()).set_stride(output.stride())
            (
                stats_fwd.set_output(True)
                .set_dim(stats.size())
                .set_stride(stats.stride())
                .set_data_type(cudnn.data_type.FLOAT)
                if not is_infer
                else None
            )
    else:
        if args.data_type == "fp8":
            o_fwd.set_output(True).set_dim(output.size()).set_stride(
                output.stride()
            ).set_data_type(cudnn.data_type.FP8_E4M3)
        else:
            o_fwd.set_output(True).set_dim(output.size()).set_stride(output.stride())

    if args.data_type == "fp8":
        amax_s_fwd.set_output(True).set_dim(amax_s_gpu.size()).set_stride(
            amax_s_gpu.stride()
        ).set_data_type(cudnn.data_type.FLOAT)
        amax_o_fwd.set_output(True).set_dim(amax_o_gpu.size()).set_stride(
            amax_o_gpu.stride()
        ).set_data_type(cudnn.data_type.FLOAT)
    graph_fwd.validate()
    graph_fwd.build_operation_graph()
    graph_fwd.create_execution_plans([cudnn.heur_mode.A])
    graph_fwd.check_support()
    graph_fwd.build_plans()

    # If backward is requested, set up backward graph.
    if args.fwd_bwd:
        graph_bwd = cudnn.pygraph(
            io_data_type=(
                cudnn.data_type.FP8_E4M3
                if args.data_type == "fp8"
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

        if args.data_type == "fp8":
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
                use_causal_mask_bottom_right=args.attn_mask == "padding_causal",
                dropout=dropout_tuple if is_dropout else None,
            )
        else:
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
                use_causal_mask_bottom_right=args.attn_mask == "padding_causal",
                dropout=dropout_tuple if is_dropout else None,
            )

        if args.data_type == "fp8":
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
        else:
            dQ_bwd.set_output(True).set_dim(dQuery.size()).set_stride(dQuery.stride())
            dK_bwd.set_output(True).set_dim(dKey.size()).set_stride(dKey.stride())
            dV_bwd.set_output(True).set_dim(dValue.size()).set_stride(dValue.stride())

        graph_bwd.validate()
        graph_bwd.build_operation_graph()
        graph_bwd.create_execution_plans([cudnn.heur_mode.A])
        graph_bwd.check_support()
        graph_bwd.build_plans()

        if args.data_type == "fp8":
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
        else:
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
    else:
        if args.data_type == "fp8":
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
            workspace = torch.empty(
                graph_fwd.get_workspace_size(), device="cuda", dtype=torch.uint8
            )
        else:
            variant_pack_fwd = {
                q_fwd: query,
                k_fwd: key,
                v_fwd: value,
                o_fwd: output,
            }
            workspace = torch.empty(
                graph_fwd.get_workspace_size(), device="cuda", dtype=torch.uint8
            )
    if is_dropout:
        variant_pack_fwd[seed_fwd] = dropout_seed
        variant_pack_fwd[offset_fwd] = dropout_offset
        variant_pack_bwd[seed_bwd] = dropout_seed
        variant_pack_bwd[offset_bwd] = dropout_offset
## Done setting up cuDNN graph.


# Reference implementation for output check
def pyt_reference_sdpa(query, key, value):
    if args.attn_mask == "padding_causal":
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            enable_gqa=enable_gqa,
            is_causal=is_causal,
            attn_mask=attn_mask,
        )
    else:
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, enable_gqa=enable_gqa, is_causal=is_causal
        )


# For backends MATH, EFFICIENT_ATTENTION, CUDNN_ATTENTION, FLASH_ATTENTION
def pyt_backend_sdpa(query, key, value, backend):
    if args.attn_mask == "padding_causal":
        with sdpa_kernel(backends=[backend]):
            return torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                enable_gqa=enable_gqa,
                is_causal=is_causal,
                attn_mask=attn_mask,
            )
    else:
        with sdpa_kernel(backends=[backend]):
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, enable_gqa=enable_gqa, is_causal=is_causal
            )


if args.sdpa_backend == "flash_attention":
    import flash_attn
    from flash_attn import flash_attn_func

    # Flash Attention Native
    def flash_attention_sdpa(query, key, value):
        return flash_attn_func(query, key, value, causal=is_causal)


if args.sdpa_backend == "flash_attention_3":
    import flash_attn_interface

    def flash_attention_3_sdpa(query, key, value):
        output, _ = flash_attn_interface.flash_attn_func(
            query, key, value, causal=is_causal
        )
        return output


def get_sdpa_function(backend):
    if backend == "pyt_math":
        return functools.partial(pyt_backend_sdpa, backend=SDPBackend.MATH)
    elif backend == "pyt_efficient_attention":
        return functools.partial(
            pyt_backend_sdpa, backend=SDPBackend.EFFICIENT_ATTENTION
        )
    elif backend == "pyt_flash_attention":
        return functools.partial(pyt_backend_sdpa, backend=SDPBackend.FLASH_ATTENTION)
    elif backend == "pyt_cudnn":
        return functools.partial(pyt_backend_sdpa, backend=SDPBackend.CUDNN_ATTENTION)
    elif backend == "flash_attention":
        return flash_attention_sdpa
    elif backend == "flash_attention_3":
        return flash_attention_3_sdpa
    elif backend == "cudnn_fe":
        return None  # Will be set up separately
    else:
        raise ValueError(f"Invalid backend: {backend}")


# Util function for addressing different qkv formats for each backend
def preprocess_qkv(query, key, value, backend):
    if backend.startswith("pyt_") or backend == "cudnn_fe":
        return query, key, value
    elif backend.startswith("flash_attention"):
        query = torch.swapaxes(query, 1, 2)
        key = torch.swapaxes(key, 1, 2)
        value = torch.swapaxes(value, 1, 2)
        return query, key, value
    else:
        raise ValueError(f"Invalid backend: {backend}")


# Util function addressing different qkvo formats for each backend
def postprocess_qkvo(query, key, value, output, backend):
    if backend.startswith("pyt_") or backend == "cudnn_fe":
        return query, key, value, output
    elif backend.startswith("flash_attention"):
        output = torch.swapaxes(output, 1, 2)
        query = torch.swapaxes(query, 1, 2)
        key = torch.swapaxes(key, 1, 2)
        value = torch.swapaxes(value, 1, 2)
        return query, key, value, output
    else:
        raise ValueError(f"Invalid backend: {backend}")


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
    padding_fraction=0.0,
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
    if padding_fraction > 0.0:
        f = f * padding_fraction
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
    padding_fraction=0.0,
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
        padding_fraction,
    )
    return f / time / 1e9 if not math.isnan(time) else 0.0  # Assume time is in msec


###### Done setting up SDPA function for each backend #######
#############################################################

###### SDPA Benchmark -- Run ######
## Print System Info
if args.verbose:
    print(f"[INFO] {torch.__version__ = }")
    print(f"[INFO] {torch.version.cuda = }")
    print(f"[INFO] {torch.cuda.is_available() = }")
    print(f"[INFO] {torch.cuda.device_count() = }")
    print(f"[INFO] {torch.cuda.current_device() = }")
    print(f"[INFO] {torch.cuda.get_device_name(torch.cuda.current_device()) = }")
    if args.sdpa_backend == "pyt_cudnn":
        print(f"[INFO] {torch.backends.cudnn.version() = }")
        print(f"[INFO] {torch.backends.cudnn.enabled = }")
    elif args.sdpa_backend == "flash_attention":
        print(f"[INFO] {flash_attn.__version__ = }")

# Use torch's CUDA event to record time
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

forward_times = []
backward_times = []
forward_diffs = []

first_error = True  # For suppressing error message beyond first error
sdpa_function = get_sdpa_function(args.sdpa_backend)
for i in range(num_iters):
    if args.data_type == "fp8" and args.sdpa_backend == "cudnn_fe":
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
    elif args.data_type == "fp8" and args.sdpa_backend == "flash_attention_3":
        query = torch.randn(
            batch_size,
            num_q_heads,
            q_seqlen,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
            requires_grad=True,
        ).to(torch.float8_e4m3fn)
        key = torch.randn(
            batch_size,
            num_kv_heads,
            kv_seqlen,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
            requires_grad=True,
        ).to(torch.float8_e4m3fn)
        value = torch.randn(
            batch_size,
            num_kv_heads,
            kv_seqlen,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
            requires_grad=True,
        ).to(torch.float8_e4m3fn)
    else:
        query = torch.randn(
            batch_size,
            num_q_heads,
            q_seqlen,
            head_dim,
            dtype=target_dtype,
            device=device,
            requires_grad=True,
        )
        key = torch.randn(
            batch_size,
            num_kv_heads,
            kv_seqlen,
            head_dim,
            dtype=target_dtype,
            device=device,
            requires_grad=True,
        )
        value = torch.randn(
            batch_size,
            num_kv_heads,
            kv_seqlen,
            head_dim,
            dtype=target_dtype,
            device=device,
            requires_grad=True,
        )

    query, key, value = preprocess_qkv(query, key, value, args.sdpa_backend)

    if args.sdpa_backend == "cudnn_fe":
        output = torch.empty(
            batch_size,
            num_q_heads,
            q_seqlen,
            head_dim,
            dtype=torch.uint8 if args.data_type == "fp8" else target_dtype,
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

        # Only variant pack and workspace need to be updated for each iteration.
        if args.fwd_bwd:
            if args.data_type == "fp8":
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
            else:
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
        else:
            if args.data_type == "fp8":
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
            else:
                variant_pack_fwd = {
                    q_fwd: query,
                    k_fwd: key,
                    v_fwd: value,
                    o_fwd: output,
                }
            workspace = torch.empty(
                graph_fwd.get_workspace_size(), device="cuda", dtype=torch.uint8
            )

        if is_dropout:
            variant_pack_fwd[seed_fwd] = dropout_seed
            variant_pack_fwd[offset_fwd] = dropout_offset
            variant_pack_bwd[seed_bwd] = dropout_seed
            variant_pack_bwd[offset_bwd] = dropout_offset

    ## Run target kernel and measure time
    with torch.autograd.profiler.emit_nvtx():
        torch.cuda.nvtx.range_push("sdpa.forward")
        start_event.record()
        if args.sdpa_backend == "cudnn_fe":
            graph_fwd.execute(variant_pack_fwd, workspace)
        else:
            output = sdpa_function(query, key, value)
        end_event.record()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

    fwd_time = start_event.elapsed_time(end_event)
    forward_times.append(fwd_time)

    # Sleep for some time proportional to fwd_time for stable measurements
    sleep_time = np.min([fwd_time / 100, 1.0])
    time.sleep(sleep_time)

    if args.fwd_bwd:
        # Run backward pass
        if args.data_type == "fp8":
            grad_output = torch.empty_like(output)
        else:
            grad_output = torch.randn_like(output)
        with torch.autograd.profiler.emit_nvtx():
            torch.cuda.nvtx.range_push(f"sdpa.backward")
            start_event.record()
            if args.sdpa_backend == "cudnn_fe":
                graph_bwd.execute(variant_pack_bwd, workspace)
            else:
                output.backward(grad_output)
            end_event.record()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()

        bwd_time = start_event.elapsed_time(end_event)
        backward_times.append(bwd_time)

        sleep_time = np.min([bwd_time / 100, 1.0])
        time.sleep(sleep_time)

    query, key, value, output = postprocess_qkvo(
        query, key, value, output, args.sdpa_backend
    )
    if args.data_type != "fp8":
        try:
            output_ref = pyt_reference_sdpa(query, key, value)
            torch.testing.assert_close(output, output_ref, rtol=1e-2, atol=1e-2)
            forward_diffs.append(
                torch.max(torch.abs(output.detach() - output_ref.detach())).item()
            )
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

    time.sleep(sleep_time)

    if args.sdpa_backend == "cudnn_fe":
        del query, key, value, output, dQuery, dKey, dValue, dOutput, stats
    else:
        del query, key, value, output

## print results
fwd_median_time = np.median(np.array(forward_times[5:]))
fwd_tflops = tflops_per_sec(
    args.batch_size,
    args.q_seqlen,
    args.kv_seqlen,
    args.head_dim,
    args.num_q_heads,
    args.num_kv_heads,
    args.is_causal,
    fwd_median_time,
    "fwd",
    padding_fraction,
)
if args.fwd_bwd:
    bwd_median_time = np.median(np.array(backward_times[5:]))
    bwd_tflops = tflops_per_sec(
        args.batch_size,
        args.q_seqlen,
        args.kv_seqlen,
        args.head_dim,
        args.num_q_heads,
        args.num_kv_heads,
        args.is_causal,
        bwd_median_time,
        "bwd",
        padding_fraction,
    )
    if args.format_output:
        print(
            f"{args.case_tag},{args.sdpa_backend},{args.batch_size},{args.q_seqlen},{args.kv_seqlen},{args.num_q_heads},{args.num_kv_heads},{args.head_dim},{fwd_median_time:.3f},{bwd_median_time:.3f},{fwd_tflops:.0f},{bwd_tflops:.0f},{np.max(np.array(forward_diffs[5:])):.6f},{num_iters}"
        )
    else:
        print(
            f"{args.sdpa_backend}:: Median (fwd, bwd) Execution Times: {fwd_median_time:.3f} ms ({fwd_tflops:.0f} TFLOPS), {bwd_median_time:.3f} ms ({bwd_tflops:.0f} TFLOPS) (max difference vs. pyt_reference: {np.max(np.array(forward_diffs[5:])):.6f} from {num_iters} iterations)"
        )
else:
    if args.format_output:
        print(
            f"{args.case_tag},{args.sdpa_backend},{args.batch_size},{args.q_seqlen},{args.kv_seqlen},{args.num_q_heads},{args.num_kv_heads},{args.head_dim},{fwd_median_time:.3f},0,{fwd_tflops:.0f},0,{np.max(np.array(forward_diffs[5:])):.6f},{num_iters}"
        )
    else:
        print(
            f"{args.sdpa_backend}:: Median (fwd) Execution Times: {fwd_median_time:.3f} ms ({fwd_tflops:.0f} TFLOPS) (max difference vs. pyt_reference: {np.max(np.array(forward_diffs[5:])):.6f} from {num_iters} iterations)"
        )
