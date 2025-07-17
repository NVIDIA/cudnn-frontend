"""
Scaled Dot Product Attention (SDPA) benchmark

This script benchmarks several SDPA backends including cuDNN using CUDA events.
Output csv and png files are saved in the artifacts directory.

"""

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import flash_attn
from flash_attn import flash_attn_func
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
dtype = torch.bfloat16
num_iters = 10  # Number of iterations to run for each config; take median time
dry_run_iters = 5  # Number of iterations to dry run for warmup
is_causal = True
enable_gqa = True
backends = [
    "pyt_math",
    "pyt_cudnn",
    # 'pyt_efficient_attention', # Disabled for GQA
    "pyt_flash_attention",
    "flash_attention",
]

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
# Referece implementation for output check
def pyt_reference_sdpa(query, key, value):
    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, enable_gqa=enable_gqa, is_causal=is_causal
    )


# For backends MATH, EFFICIENT_ATTENTION, CUDNN_ATTENTION, FLASH_ATTENTION
def pyt_backend_sdpa(query, key, value, backend):
    with sdpa_kernel(backends=[backend]):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, enable_gqa=enable_gqa, is_causal=is_causal
        )


# Flash Attention Native
def flash_attention_sdpa(query, key, value):
    return flash_attn_func(query, key, value, causal=is_causal)


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
    else:
        raise ValueError(f"Invalid backend: {backend}")


# Util function for addressing different qkv formats for each backend
def preprocess_qkv(query, key, value, backend):
    if backend.startswith("pyt_"):
        return query, key, value
    elif backend == "flash_attention":
        query = torch.swapaxes(query, 1, 2)
        key = torch.swapaxes(key, 1, 2)
        value = torch.swapaxes(value, 1, 2)
        return query, key, value
    else:
        raise ValueError(f"Invalid backend: {backend}")


# Util function addressing different qkvo formats for each backend
def postprocess_qkvo(query, key, value, output, backend):
    if backend.startswith("pyt_"):
        return query, key, value, output
    elif backend == "flash_attention":
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


###### SDPA Benchmark -- Run ######
## Print System Info
print(f"[INFO] {torch.__version__ = }")
print(f"[INFO] {torch.version.cuda = }")
print(f"[INFO] {torch.cuda.is_available() = }")
print(f"[INFO] {torch.cuda.device_count() = }")
print(f"[INFO] {torch.cuda.current_device() = }")
print(f"[INFO] {torch.cuda.get_device_name(torch.cuda.current_device()) = }")
print(f"[INFO] {torch.backends.cudnn.version() = }")
print(f"[INFO] {torch.backends.cudnn.enabled = }")
print(f"[INFO] {flash_attn.__version__ = }")

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
        "backend",
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
    for cur_backend in backends:
        print(f"[INFO]   Benchmarking backend {cur_backend}")

        fwd_times = []
        bwd_times = []

        # Repeat measurement for warmup and actual measurement
        for i in range(total_iters):
            fwd_time = np.inf
            bwd_time = np.inf
            try:
                sdpa_function = get_sdpa_function(cur_backend)

                # Prepare input tensors
                query = torch.randn(
                    batch_size,
                    num_q_heads,
                    q_seqlen,
                    head_dim,
                    dtype=dtype,
                    device=device,
                    requires_grad=True,
                )
                key = torch.randn(
                    batch_size,
                    num_kv_heads,
                    kv_seqlen,
                    head_dim,
                    dtype=dtype,
                    device=device,
                    requires_grad=True,
                )
                value = torch.randn(
                    batch_size,
                    num_kv_heads,
                    kv_seqlen,
                    head_dim,
                    dtype=dtype,
                    device=device,
                    requires_grad=True,
                )
                query, key, value = preprocess_qkv(query, key, value, cur_backend)

                # Run forward pass
                with torch.autograd.profiler.emit_nvtx():
                    torch.cuda.nvtx.range_push(f"sdpa.forward.{cur_backend}")
                    start_event.record()
                    output = sdpa_function(query, key, value)
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
                grad_output = torch.randn_like(output)
                with torch.autograd.profiler.emit_nvtx():
                    torch.cuda.nvtx.range_push(f"sdpa.backward.{cur_backend}")
                    start_event.record()
                    output.backward(grad_output)
                    end_event.record()
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_pop()
                # Record backward pass time
                bwd_time = start_event.elapsed_time(end_event)

                # Postprocess output for ref check and run refcheck
                # Refcheck only checks forward pass
                query, key, value, output = postprocess_qkvo(
                    query, key, value, output, cur_backend
                )
                output_ref = pyt_reference_sdpa(query, key, value)
                torch.testing.assert_close(output, output_ref, rtol=1e-2, atol=1e-2)

                # For stable measurements, sleep for some time proportional to fwd_time
                if bwd_time is not np.inf:
                    sleep_time = np.min([bwd_time / 100, 1.0])
                else:
                    sleep_time = 0.01
                time.sleep(sleep_time)

                # Clear tensors
                del query, key, value, output, output_ref
            except Exception as e:
                pass

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
            "backend": cur_backend,
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
output_file_name = f"./artifacts/sdpa_bf16_benchmark_results_{gpu_name}.csv"
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

# Configurations for bar plots
backend_ordering = {
    "pyt_math": 0,
    "pyt_efficient_attention": 1,
    "pyt_flash_attention": 2,
    "flash_attention": 3,
    "pyt_cudnn": 4,
}
backend_name = {
    "pyt_math": "Standard Attention",
    "pyt_efficient_attention": "xFormers (PyTorch)",
    "pyt_flash_attention": "FAv2 (PyTorch)",
    "flash_attention": "FAv2 (Native)",
    "pyt_cudnn": "cuDNN (PyTorch)",
}
backend_barplot_color = {
    backend_name["pyt_math"]: "darkorange",
    backend_name["pyt_efficient_attention"]: "magenta",
    backend_name["pyt_flash_attention"]: "royalblue",
    backend_name["flash_attention"]: "lightcoral",
    backend_name["pyt_cudnn"]: "#76b900",
}
LABEL_FONT_SIZE = 8
LEGEND_FONT_SIZE = 6
TITLE_FONT_SIZE = 9
# Select desired cases
plot_df = data_df[
    (data_df["is_causal"] == True)
    & (data_df["num_q_heads"] == 128)
    & (data_df["num_kv_heads"] == 8)
    & (data_df["q_seqlen"] == data_df["kv_seqlen"])
    & (data_df["head_dim"] == 128)
].copy()

plot_df["backend_rank"] = plot_df["backend"].map(backend_ordering)
plot_df["backend_name"] = plot_df["backend"].map(backend_name)
plot_df.sort_values(["q_seqlen", "backend_rank"], inplace=True)

# Generate plots: forward on left subplot and backward on right subplot
YLIM_MAX = (
    np.max([plot_df["fwd_tflops_per_sec"].max(), plot_df["bwd_tflops_per_sec"].max()])
    * 1.1
)

plt.figure(figsize=(10, 4), dpi=200)
plt.subplot(1, 2, 1)
cur_plot_df = plot_df[plot_df.fwd_tflops_per_sec > 0]
ax = sns.barplot(
    data=cur_plot_df,
    x="q_seqlen",
    y="fwd_tflops_per_sec",
    hue="backend_name",
    edgecolor="black",
    linewidth=0.5,
    palette=backend_barplot_color,
)
ax.legend_.set_title(None)
for container in ax.containers:
    ax.bar_label(container, fmt="%.f", fontsize=6)
plt.xticks(rotation=45)
plt.xlabel("Sequence Length", fontsize=LABEL_FONT_SIZE)
plt.ylabel("Speed (TFLOPs/sec)", fontsize=LABEL_FONT_SIZE)
plt.title("SDPA Forward", fontsize=TITLE_FONT_SIZE)
plt.tick_params(axis="y", which="major", labelsize=LABEL_FONT_SIZE)
plt.tick_params(axis="x", which="major", labelsize=LABEL_FONT_SIZE)
plt.ylim(0, YLIM_MAX)
plt.legend(fontsize=LEGEND_FONT_SIZE, loc="upper left")

plt.subplot(1, 2, 2)
cur_plot_df = plot_df[plot_df.bwd_tflops_per_sec > 0]
ax = sns.barplot(
    data=cur_plot_df,
    x="q_seqlen",
    y="bwd_tflops_per_sec",
    hue="backend_name",
    edgecolor="black",
    linewidth=0.5,
    palette=backend_barplot_color,
)
ax.legend_.set_title(None)
for container in ax.containers:
    ax.bar_label(container, fmt="%.f", fontsize=6)
plt.xticks(rotation=45)
plt.xlabel("SequenceLength", fontsize=LABEL_FONT_SIZE)
plt.ylabel("Speed (TFLOPs/sec)", fontsize=LABEL_FONT_SIZE)
plt.title("SDPA Backward", fontsize=TITLE_FONT_SIZE)
plt.tick_params(axis="y", which="major", labelsize=LABEL_FONT_SIZE)
plt.tick_params(axis="x", which="major", labelsize=LABEL_FONT_SIZE)
plt.ylim(0, YLIM_MAX)
plt.legend(fontsize=LEGEND_FONT_SIZE, loc="upper left")

# Save plot
plt.tight_layout()
png_file_name = f"./artifacts/sdpa_bf16_benchmark_results_{gpu_name}.png"
if verbose:
    print(f"[INFO] Saving plot to {png_file_name}")
try:
    plt.savefig(png_file_name)
except Exception as e:
    print(f"[ERROR] Failed to save plot to {png_file_name}: {e}")
