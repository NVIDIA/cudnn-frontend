# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure PyTorch eager or JAX JIT forward/training after warmup."""

import argparse
from contextlib import redirect_stdout
import csv
import importlib.metadata
import sys
import time

import torch

from cudnn.ops import SimplifiedSparseAttentionMetadata, simplified_sparse_attention, simplified_sparse_attention_reserve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-length", type=int, default=1024)
    parser.add_argument("--key-length", type=int, default=4096)
    parser.add_argument("--heads", type=int, choices=(16, 32, 96), default=16)
    parser.add_argument("--topk", type=int, choices=(512, 1024, 2048), default=512)
    parser.add_argument("--mode", choices=("forward", "training"), default="training")
    parser.add_argument("--framework", choices=("torch", "jax"), default="torch")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reserve", action="store_true")
    args = parser.parse_args()
    if min(args.query_length, args.key_length, args.iterations, args.warmup) <= 0:
        parser.error("Lengths, iterations, and warmup must be positive")
    if args.query_length > args.key_length:
        parser.error("This causal benchmark requires query-length <= key-length")
    if args.framework == "jax" and args.reserve:
        parser.error("JAX owns its scratch; --reserve applies only to PyTorch")
    torch.manual_seed(17)
    training = args.mode == "training"
    tq, tk, h = args.query_length, args.key_length, args.heads
    shapes = ((tq, h, 256), (tk, 1, 256), (tk, 1, 256), (tq, 128), (tk, 128))
    tensors = [torch.randn(shape, dtype=torch.bfloat16, device="cuda", requires_grad=training) for shape in shapes]
    metadata = SimplifiedSparseAttentionMetadata(
        torch.zeros(tq, dtype=torch.int32, device="cuda"),
        torch.arange(tk - tq + 1, tk + 1, dtype=torch.int32, device="cuda"),
        torch.ones(tq, dtype=torch.bool, device="cuda"),
        torch.tensor([float(tq)], device="cuda"),
        False,
        False,
        True,
    )
    upstream = torch.randn_like(tensors[0])
    loss_grad = torch.ones((), device="cuda")
    if args.reserve:
        simplified_sparse_attention_reserve(tq, tk, args.topk)

    def run():
        output, loss = simplified_sparse_attention(*tensors, metadata, args.topk)
        if training:
            return torch.autograd.grad((output, loss), tensors, (upstream, loss_grad))
        return output, loss

    jax_version = ""
    if args.framework == "jax":
        import jax
        import jax.numpy as jnp
        from cudnn.ops import simplified_sparse_attention_jax

        jax_version = jax.__version__
        jtensors = tuple(jax.dlpack.from_dlpack(t.detach()) for t in tensors)
        bounds = tuple(jax.dlpack.from_dlpack(t) for t in (metadata.key_starts, metadata.key_ends, metadata.query_valid, metadata.loss_denominator))
        jupstream = jax.dlpack.from_dlpack(upstream)

        @jax.jit
        def jrun(*values):
            def operation(*inputs):
                return simplified_sparse_attention_jax(*inputs, *bounds, topk=args.topk)

            if training:
                (_, loss), pullback = jax.vjp(operation, *values)
                return pullback((jupstream, jnp.ones_like(loss)))
            return operation(*values)

        def run():
            return jax.block_until_ready(jrun(*jtensors))

    with redirect_stdout(sys.stderr):
        for _ in range(args.warmup):
            result = run()
    for value in result:
        if not torch.isfinite(torch.from_dlpack(value) if args.framework == "jax" else value).all():
            raise RuntimeError("Non-finite warmup result")
    torch.cuda.synchronize()
    if args.framework == "jax":
        start = time.perf_counter()
        for _ in range(args.iterations):
            run()
        milliseconds = (time.perf_counter() - start) * 1000 / args.iterations
        peak_memory = ""
    else:
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            run()
        end.record()
        end.synchronize()
        milliseconds = start.elapsed_time(end) / args.iterations
        peak_memory = torch.cuda.max_memory_allocated() / 2**20
    from cutlass.base_dsl.version_info import CUDA_VERSION

    row = dict(
        gpu=torch.cuda.get_device_name(),
        sm="".join(map(str, torch.cuda.get_device_capability())),
        torch=torch.__version__,
        jax=jax_version,
        framework=args.framework,
        timing="wall_sync" if args.framework == "jax" else "cuda_events",
        dsl=importlib.metadata.version("nvidia-cutlass-dsl"),
        compiler_cuda=f"{CUDA_VERSION.major}.{CUDA_VERSION.minor}",
        query_length=tq,
        key_length=tk,
        heads=h,
        topk=args.topk,
        mode=args.mode,
        cuda_graph=False,
        milliseconds=milliseconds,
        peak_memory_mib=peak_memory,
    )
    writer = csv.DictWriter(sys.stdout, fieldnames=row)
    writer.writeheader()
    writer.writerow(row)


if __name__ == "__main__":
    main()
