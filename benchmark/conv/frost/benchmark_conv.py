# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sweep FROST convolution tile configs against native cuDNN with nsys.

The shape is ``N,C,D,H,W,K,T,R,S``. Inputs, filters, and outputs use compact
channels-last storage while retaining cuDNN's logical NCDHW/KCTRS shapes.
Only kernel execution is captured: graph construction, JIT compilation,
workspace allocation, and warmup happen before ``cudaProfilerStart``.
By default every dense-compatible entry in ``cudnn.conv.frost.CATALOG`` is
measured. Use ``--configs default`` for only the heuristic choice for this
shape, or pass comma-separated config names/globs to select a subset. Pass
``-o PREFIX`` to retain the cuDNN and per-config FROST reports; otherwise
reports are temporary.

Example:

    python benchmark/conv/frost/benchmark_conv.py \
        --shape 1,64,6,10,10,256,3,3,3

    python benchmark/conv/frost/benchmark_conv.py \
        --shape 1,64,6,10,10,256,3,3,3 \
        --configs 'CONFIG_sm100_128x*'
"""

from __future__ import annotations

import os
import sys

# FROST convolution is opt-in while it is experimental. Respect an explicit
# user choice, but make the benchmark work without a separate environment knob.
os.environ.setdefault("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")

import cudnn
import cudnn.conv.frost  # noqa: F401 -- make the optional implementation available
import torch

from cudnn.conv.frost.tile_config import CATALOG, _get_config, by_name

from benchmark_utils import (
    ConvShape,
    build_cudnn_plan,
    make_parser,
    output_spatial as _output_spatial,
    resolve_nbuf as _resolve_nbuf,
    run_sweep,
    select_tile_configs as _select_tile_configs,
    validate_args,
)


def _make_data(shape: ConvShape, output_spatial: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z, p, q = output_spatial
    image = torch.empty(
        (shape.n, shape.c, shape.d, shape.h, shape.w),
        dtype=torch.bfloat16,
        device="cuda",
        memory_format=torch.channels_last_3d,
    )
    weight = torch.empty(
        (shape.k, shape.c, shape.t, shape.r, shape.s),
        dtype=torch.bfloat16,
        device="cuda",
        memory_format=torch.channels_last_3d,
    )
    output = torch.empty(
        (shape.n, shape.k, z, p, q),
        dtype=torch.bfloat16,
        device="cuda",
        memory_format=torch.channels_last_3d,
    )
    return image, weight, output


def _per_set_bytes(shape: ConvShape, output_spatial: tuple[int, int, int]) -> int:
    z, p, q = output_spatial
    elements = shape.n * shape.c * shape.d * shape.h * shape.w + shape.k * shape.c * shape.t * shape.r * shape.s + shape.n * shape.k * z * p * q
    return 2 * elements  # BF16


def _build_cudnn_graph(
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    pre_padding: tuple[int, int, int],
    post_padding: tuple[int, int, int],
    stride: tuple[int, int, int],
    dilation: tuple[int, int, int],
):
    image_gpu, weight_gpu, _output_gpu = data
    graph = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, compute_data_type=cudnn.data_type.FLOAT)
    image = graph.tensor_like(image_gpu)
    weight = graph.tensor_like(weight_gpu)
    output = graph.conv_fprop(
        image,
        weight,
        name="conv",
        pre_padding=pre_padding,
        post_padding=post_padding,
        stride=stride,
        dilation=dilation,
    )
    output.set_output(True)
    workspace, name = build_cudnn_plan(graph)
    return graph, (image, weight, output), workspace, name


def _build_frost_kernel(
    tile_config_name: str,
    shape: ConvShape,
    pre_padding: tuple[int, int, int],
    post_padding: tuple[int, int, int],
    stride: tuple[int, int, int],
    dilation: tuple[int, int, int],
):
    """Compile one explicitly selected convolution tile configuration."""
    from cudnn.conv.frost.templates.sm100_conv import compile

    return compile(
        ncdhw=(shape.n, shape.c, shape.d, shape.h, shape.w),
        ktrs=(shape.k, shape.t, shape.r, shape.s),
        upper_padding_dhw=post_padding,
        lower_padding_dhw=pre_padding,
        stride_dhw=stride,
        dilation_dhw=dilation,
        tile_config=by_name(tile_config_name),
    )


def _execute(graph, tensors, workspace, data) -> None:
    image, weight, output = tensors
    image_gpu, weight_gpu, output_gpu = data
    graph.execute({image: image_gpu, weight: weight_gpu, output: output_gpu}, workspace)


def _execute_frost(compiled, data) -> None:
    from cuda.bindings import driver as cuda

    image_gpu, weight_gpu, output_gpu = data
    compiled(
        image_gpu.permute(0, 2, 3, 4, 1),
        weight_gpu.permute(0, 2, 3, 4, 1),
        output_gpu.permute(0, 2, 3, 4, 1),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def _worker(args, shape: ConvShape, output_spatial: tuple[int, int, int], nbuf: int) -> int:
    warmup_data = _make_data(shape, output_spatial)
    pool = [_make_data(shape, output_spatial) for _ in range(nbuf)]

    if args._implementation == "frost":
        compiled = _build_frost_kernel(
            args._tile_config,
            shape,
            args.pre_padding,
            args.post_padding,
            args.stride,
            args.dilation,
        )
        execute = lambda data: _execute_frost(compiled, data)
        plan_name = args._tile_config
    else:
        graph, tensors, workspace, plan_name = _build_cudnn_graph(
            warmup_data,
            args.pre_padding,
            args.post_padding,
            args.stride,
            args.dilation,
        )
        execute = lambda data: _execute(graph, tensors, workspace, data)

    for _ in range(args.warmup):
        execute(warmup_data)
    torch.cuda.synchronize()
    print(f"[worker] {args._implementation}: selected {plan_name!r}; capturing {args.iters} iterations", flush=True)

    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    try:
        for iteration in range(args.iters):
            execute(pool[iteration % nbuf])
        torch.cuda.synchronize()
    finally:
        cudart.cudaProfilerStop()
    return 0


def _parser():
    return make_parser(
        __doc__,
        "1,128,128,128,128,128,3,3,3",
        "comma-separated tile config names or globs; 'all' sweeps dense-compatible catalog entries "
        "and 'default' selects the heuristic choice (default: all)",
    )


def main() -> int:
    args = _parser().parse_args()
    if not torch.cuda.is_available():
        print("No CUDA device is available.", file=sys.stderr)
        return 1
    validate_args(args)

    shape = args.shape
    try:
        spatial = _output_spatial(shape, args.pre_padding, args.post_padding, args.stride, args.dilation)
        nbuf = _resolve_nbuf(args.rotate_buffers, _per_set_bytes(shape, spatial))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args._nsys_worker:
        return _worker(args, shape, spatial, nbuf)

    from cudnn.conv.frost.templates.sm100_conv import _dense_config_violation, _is_dense_selection_candidate

    z, p, q = spatial
    implicit_m = shape.n * z * p * q
    implicit_k = shape.t * shape.r * shape.s * shape.c
    channel_bytes = shape.c * 2
    heuristic_config = _get_config(
        implicit_m,
        shape.k,
        implicit_k,
        predicate=lambda config: _is_dense_selection_candidate(config, implicit_m, shape.k, channel_bytes),
    ).name
    dense_configs = tuple(config for config in CATALOG if _dense_config_violation(config, channel_bytes) is None)
    try:
        config_names = _select_tile_configs(args.configs, heuristic_config, dense_configs)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    flops = 2 * shape.n * shape.k * z * p * q * shape.c * shape.t * shape.r * shape.s
    per_set = _per_set_bytes(shape, spatial)
    print(
        f"\n=== conv fprop NCDHW={shape.n}x{shape.c}x{shape.d}x{shape.h}x{shape.w} "
        f"KCTRS={shape.k}x{shape.c}x{shape.t}x{shape.r}x{shape.s} "
        f"-> {shape.n}x{shape.k}x{z}x{p}x{q} — BF16 ==="
    )
    print("  [timing: nsys CUDA capture range; graph-build/JIT/warmup excluded]")
    print(f"  [rotate-buffers: {nbuf} tensor sets, {nbuf * per_set / 1024 / 1024:.0f} MB]")
    print(f"  [tile sweep: {len(config_names)} configuration(s)]")
    return run_sweep(__file__, args, shape, nbuf, config_names, flops)


if __name__ == "__main__":
    sys.exit(main())
