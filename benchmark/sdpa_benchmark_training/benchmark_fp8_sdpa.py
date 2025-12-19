"""
Scaled Dot Product Attention (SDPA) benchmark

This script benchmarks several SDPA backends including cuDNN using torch profiler.
Output csv and png files are saved in the artifacts directory.

"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import os
import sys

###### SDPA Benchmark -- Setup ######
## Define constants for benchmarking
verbose = True
num_iters = 10  # Number of iterations to run for each config; take median time
dry_run_iters = 5  # Number of iterations to dry run for warmup
attn_mask = "top_left"  # Causal mask type (top_left is equivalent to is_causal=True)
backend = "cudnn_fe"  # Backend to use for benchmarking
precisions = ["fp8", "bf16"]

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

## Helper function to run benchmark_single_sdpa.py and parse its output
def run_single_benchmark(config, precision):
    """
    Run benchmark_single_sdpa.py for a single configuration and precision.
    
    Args:
        config: Tuple of (batch_size, q_seqlen, kv_seqlen, num_q_heads, num_kv_heads, head_dim)
        precision: Data type (e.g., "fp8", "bf16")
    
    Returns:
        Dictionary with benchmark results or None if failed
    """
    batch_size, q_seqlen, kv_seqlen, num_q_heads, num_kv_heads, head_dim = config
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark_script = os.path.join(script_dir, "benchmark_single_sdpa.py")
    
    # Map precision names
    data_type = "bfloat16" if precision == "bf16" else precision
    
    # Build command
    cmd = [
        sys.executable,  # Use the same Python interpreter
        benchmark_script,
        "--batch_size", str(batch_size),
        "--q_seqlen", str(q_seqlen),
        "--kv_seqlen", str(kv_seqlen),
        "--num_q_heads", str(num_q_heads),
        "--num_kv_heads", str(num_kv_heads),
        "--head_dim", str(head_dim),
        "--data_type", data_type,
        "--num_iterations", str(num_iters),
        "--num_warmup_iterations", str(dry_run_iters),
        "--sdpa_backend", backend,
        "--attn_mask", attn_mask,
        "--fwd_bwd",  # Run both forward and backward
        "--format_output",  # Get CSV-formatted output
        "--skip_ref",  # Skip reference check for speed
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,  # Don't raise exception on non-zero exit
        )
        
        if result.returncode != 0:
            if verbose:
                print(f"  [WARNING] Benchmark failed with return code {result.returncode}")
                print(f"  stderr: {result.stderr}")
            # Return entry with infinite times and 0 TFLOPs for failed benchmarks
            return {
                'batch_size': batch_size,
                'q_seqlen': q_seqlen,
                'kv_seqlen': kv_seqlen,
                'num_q_heads': num_q_heads,
                'num_kv_heads': num_kv_heads,
                'head_dim': head_dim,
                'is_causal': attn_mask == "top_left",
                'precision': precision,
                'forward_time': np.inf,
                'backward_time': np.inf,
                'fwd_tflops_per_sec': 0.0,
                'bwd_tflops_per_sec': 0.0,
            }
        
        # Parse output - format is:
        # case_tag,backend,batch_size,q_seqlen,kv_seqlen,num_q_heads,num_kv_heads,head_dim,fwd_time,bwd_time,fwd_tflops,bwd_tflops,max_diff,num_iters
        output_line = result.stdout.strip().split('\n')[-1]  # Get last line
        parts = output_line.split(',')
        
        if len(parts) < 12:
            if verbose:
                print(f"  [WARNING] Unexpected output format: {output_line}")
            # Return entry with infinite times and 0 TFLOPs for parse failures
            return {
                'batch_size': batch_size,
                'q_seqlen': q_seqlen,
                'kv_seqlen': kv_seqlen,
                'num_q_heads': num_q_heads,
                'num_kv_heads': num_kv_heads,
                'head_dim': head_dim,
                'is_causal': attn_mask == "top_left",
                'precision': precision,
                'forward_time': np.inf,
                'backward_time': np.inf,
                'fwd_tflops_per_sec': 0.0,
                'bwd_tflops_per_sec': 0.0,
            }
        
        return {
            'batch_size': int(parts[2]),
            'q_seqlen': int(parts[3]),
            'kv_seqlen': int(parts[4]),
            'num_q_heads': int(parts[5]),
            'num_kv_heads': int(parts[6]),
            'head_dim': int(parts[7]),
            'is_causal': attn_mask == "top_left",
            'precision': precision,
            'forward_time': float(parts[8]),
            'backward_time': float(parts[9]),
            'fwd_tflops_per_sec': float(parts[10]),
            'bwd_tflops_per_sec': float(parts[11]),
        }
    except Exception as e:
        if verbose:
            print(f"  [ERROR] Failed to run benchmark: {e}")
        # Return entry with infinite times and 0 TFLOPs for exceptions
        return {
            'batch_size': batch_size,
            'q_seqlen': q_seqlen,
            'kv_seqlen': kv_seqlen,
            'num_q_heads': num_q_heads,
            'num_kv_heads': num_kv_heads,
            'head_dim': head_dim,
            'is_causal': attn_mask == "top_left",
            'precision': precision,
            'forward_time': np.inf,
            'backward_time': np.inf,
            'fwd_tflops_per_sec': 0.0,
            'bwd_tflops_per_sec': 0.0,
        }

###### SDPA Benchmark -- Run ######
## Print System Info
try:
    import cudnn
    print(f"[INFO] cuDNN Backend Version: {cudnn.backend_version() = }")
    print(f"[INFO] cuDNN Frontend Version: {cudnn.__version__ = }")
except ImportError:
    pass

print(f"[INFO] {torch.__version__ = }")
print(f"[INFO] {torch.version.cuda = }")
print(f"[INFO] {torch.cuda.is_available() = }")
print(f"[INFO] {torch.cuda.device_count() = }")
print(f"[INFO] {torch.cuda.current_device() = }")
print(f"[INFO] {torch.cuda.get_device_name(torch.cuda.current_device()) = }")

## Begin Benchmark
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
        "fwd_tflops_per_sec",
        "bwd_tflops_per_sec",
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

    # Iterate over each precision
    for cur_precision in precisions:
        print(f"[INFO]   Benchmarking data type {cur_precision}")
        
        # Run benchmark via subprocess
        result = run_single_benchmark(sdpa_config, cur_precision)
        
        # Append data to table (result is always a dict, never None)
        data_df = pd.concat([data_df, pd.DataFrame([result])], ignore_index=True)

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
