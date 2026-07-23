"""Benchmark the experimental cuDNN PyTorch normalization ops."""

import argparse
import statistics

import torch

from cudnn.experimental.ops import layer_norm, rms_norm


MODELS = {
    "llama3-8b": 4096,
    "llama3-70b": 8192,
    "dit-xl": 1152,
}
TOKEN_COUNTS = {
    "prefill": [1024, 2048, 4096, 8192],
    "decode": [32, 64, 128, 256],
    "train": [4096, 8192, 16384],
}


def _benchmark(function, warmup, repeat):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(times)


def _run_case(op_name, tokens, hidden_size, warmup, repeat):
    input = torch.randn(tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(hidden_size, dtype=torch.bfloat16, device="cuda")
    if op_name == "rmsnorm":
        cudnn_fn = lambda: rms_norm(input, weight)
        torch_fn = lambda: torch.nn.functional.rms_norm(input, (hidden_size,), weight)
    else:
        bias = torch.randn(hidden_size, dtype=torch.bfloat16, device="cuda")
        cudnn_fn = lambda: layer_norm(input, (hidden_size,), weight, bias)
        torch_fn = lambda: torch.nn.functional.layer_norm(input, (hidden_size,), weight, bias)

    cudnn_us = _benchmark(cudnn_fn, warmup, repeat)
    torch_us = _benchmark(torch_fn, warmup, repeat)
    return cudnn_us, torch_us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=["rmsnorm", "layernorm", "all"], default="all")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()

    print("op,model,scenario,tokens,cudnn_us,pytorch_us,speedup")
    operations = ["rmsnorm", "layernorm"] if args.op == "all" else [args.op]
    for op_name in operations:
        for model, hidden_size in MODELS.items():
            for scenario, token_counts in TOKEN_COUNTS.items():
                for tokens in token_counts:
                    cudnn_us, torch_us = _run_case(op_name, tokens, hidden_size, args.warmup, args.repeat)
                    print(f"{op_name},{model},{scenario},{tokens},{cudnn_us:.1f},{torch_us:.1f},{torch_us / cudnn_us:.2f}")


if __name__ == "__main__":
    main()
