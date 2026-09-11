# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sweep NVFP4 block-scale FROST convolution configs with nsys.

The shape is ``N,C,D,H,W,K,T,R,S``. A and B are packed NVFP4 with E4M3
block-16 scales; D is BF16. All tensors use compact channels-last storage while
retaining cuDNN's logical NCDHW/KCTRS shapes. Only kernel execution is captured
with nsys: graph construction, JIT compilation, allocation, and warmup happen
before ``cudaProfilerStart``. Pass ``--check-correctness`` to compare each
FROST config with native cuDNN BF16 convolution of exactly dequantized inputs
before capture (rtol=atol=0.015625). Checks are disabled by default; failing
configs are excluded from timing results. This reference does not require a
native NVFP4 convolution plan.

By default every catalog entry compatible with block-scale convolution and the
requested C is measured. Use ``--configs default`` for only the heuristic
choice, or pass comma-separated config names/globs to select a subset.
When the backend exposes a native cuDNN plan, it is included as a baseline;
the sweep still runs on builds where Frost is the only available plan.

Example:

    python benchmark/conv/frost/benchmark_block_scale_conv.py \
        --shape 1,128,6,10,10,256,3,3,3

    python benchmark/conv/frost/benchmark_block_scale_conv.py \
        --shape 1,192,6,10,10,256,3,3,3 \
        --configs 'CONFIG_sm100_128x*96*'
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")

import cudnn
import cudnn.conv.frost  # noqa: F401 -- make the optional implementation available
import torch

from cudnn.conv.frost.tile_config import CATALOG, _get_config, by_name

from benchmark_utils import (
    ConvShape,
    build_cudnn_plan,
    channels_last_stride,
    make_parser,
    output_spatial,
    resolve_nbuf,
    run_sweep,
    select_tile_configs,
    validate_args,
)

_BLOCK_SIZE = 16


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _make_data(shape: ConvShape, spatial: tuple[int, int, int]) -> tuple[torch.Tensor, ...]:
    if not hasattr(torch, "float4_e2m1fn_x2"):
        raise RuntimeError("block-scale convolution requires torch.float4_e2m1fn_x2")

    z, p, q = spatial
    image_bytes = torch.empty(
        (shape.n, shape.d, shape.h, shape.w, shape.c // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    weight_bytes = torch.empty(
        (shape.k, shape.t, shape.r, shape.s, shape.c // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    image = image_bytes.view(torch.float4_e2m1fn_x2).permute(0, 4, 1, 2, 3)
    weight = weight_bytes.view(torch.float4_e2m1fn_x2).permute(0, 4, 1, 2, 3)

    sf_c = shape.c // _BLOCK_SIZE
    sfa_columns = _ceil_div(sf_c, 4) * 4
    sfa = torch.empty(shape.n * shape.d * shape.h * shape.w, sfa_columns, 1, dtype=torch.float8_e4m3fn, device="cuda")

    sfb_columns = _ceil_div(shape.c * shape.t * shape.r * shape.s, _BLOCK_SIZE)
    sfb_bytes = 512 * _ceil_div(shape.k, 128) * _ceil_div(sfb_columns, 4)
    sfb = torch.empty(sfb_bytes, dtype=torch.float8_e4m3fn, device="cuda")

    output = torch.empty(
        (shape.n, shape.k, z, p, q),
        dtype=torch.bfloat16,
        device="cuda",
        memory_format=torch.channels_last_3d,
    )
    return image, weight, sfa, sfb, output


def _per_set_bytes(shape: ConvShape, spatial: tuple[int, int, int]) -> int:
    z, p, q = spatial
    image_bytes = shape.n * shape.d * shape.h * shape.w * shape.c // 2
    weight_bytes = shape.k * shape.t * shape.r * shape.s * shape.c // 2
    sfa_bytes = shape.n * shape.d * shape.h * shape.w * _ceil_div(shape.c // _BLOCK_SIZE, 4) * 4
    sfb_columns = _ceil_div(shape.c * shape.t * shape.r * shape.s, _BLOCK_SIZE)
    sfb_bytes = 512 * _ceil_div(shape.k, 128) * _ceil_div(sfb_columns, 4)
    output_bytes = 2 * shape.n * shape.k * z * p * q
    return image_bytes + weight_bytes + sfa_bytes + sfb_bytes + output_bytes


def _build_cudnn_graph(
    shape: ConvShape,
    spatial: tuple[int, int, int],
    pre_padding: tuple[int, int, int],
    post_padding: tuple[int, int, int],
    stride: tuple[int, int, int],
    dilation: tuple[int, int, int],
):
    ncdhw = (shape.n, shape.c, shape.d, shape.h, shape.w)
    kctrs = (shape.k, shape.c, shape.t, shape.r, shape.s)
    output_shape = (shape.n, shape.k, *spatial)
    sf_c = shape.c // _BLOCK_SIZE

    graph = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    image = graph.tensor(name="A", dim=ncdhw, stride=channels_last_stride(ncdhw), data_type=cudnn.data_type.FP4_E2M1)
    weight = graph.tensor(name="B", dim=kctrs, stride=channels_last_stride(kctrs), data_type=cudnn.data_type.FP4_E2M1)
    sfa_shape = (shape.n, sf_c, shape.d, shape.h, shape.w)
    sfb_shape = (shape.k, sf_c, shape.t, shape.r, shape.s)
    sfa = graph.tensor(name="SFA", dim=sfa_shape, stride=channels_last_stride(sfa_shape), data_type=cudnn.data_type.FP8_E4M3)
    sfb = graph.tensor(
        name="SFB",
        dim=sfb_shape,
        stride=channels_last_stride(sfb_shape),
        data_type=cudnn.data_type.FP8_E4M3,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    dequant_image = graph.block_scale_dequantize(
        input=image,
        descale=sfa,
        block_size=(1, _BLOCK_SIZE, 1, 1, 1),
        compute_data_type=cudnn.data_type.FLOAT,
        name="dequantize_A",
    )
    dequant_weight = graph.block_scale_dequantize(
        input=weight,
        descale=sfb,
        block_size=(1, _BLOCK_SIZE, 1, 1, 1),
        compute_data_type=cudnn.data_type.FLOAT,
        name="dequantize_B",
    )
    output = graph.conv_fprop(
        dequant_image,
        dequant_weight,
        name="nvfp4_conv",
        pre_padding=pre_padding,
        post_padding=post_padding,
        stride=stride,
        dilation=dilation,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    output.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_dim(output_shape).set_stride(channels_last_stride(output_shape))
    workspace, name = build_cudnn_plan(graph)
    return graph, (image, weight, sfa, sfb, output), workspace, name


def _build_frost_kernel(
    tile_config_name: str,
    shape: ConvShape,
    pre_padding: tuple[int, int, int],
    post_padding: tuple[int, int, int],
    stride: tuple[int, int, int],
    dilation: tuple[int, int, int],
):
    import cutlass

    from cudnn.conv.frost.templates.sm100_block_scale_conv import compile

    return compile(
        ncdhw=(shape.n, shape.c, shape.d, shape.h, shape.w),
        ktrs=(shape.k, shape.t, shape.r, shape.s),
        upper_padding_dhw=post_padding,
        lower_padding_dhw=pre_padding,
        stride_dhw=stride,
        dilation_dhw=dilation,
        d_dtype=cutlass.BFloat16,
        tile_config=by_name(tile_config_name),
    )


def _execute_cudnn(graph, tensors, workspace, data) -> None:
    graph.execute(dict(zip(tensors, data)), workspace)


def _execute_frost(compiled, data, args) -> None:
    import cutlass
    from cuda.bindings import driver as cuda

    image, weight, sfa, sfb, output = data
    compiled(
        image.permute(0, 2, 3, 4, 1),
        weight.permute(0, 2, 3, 4, 1),
        output.permute(0, 2, 3, 4, 1),
        sfa,
        sfb,
        cutlass.Float32(1.0),
        None,
        cutlass.Float32(1.0),
        None,
        None,
        *(cutlass.Int32(value) for value in (*args.post_padding, *args.pre_padding, *args.stride, *args.dilation)),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def _initialize_check_data(data, shape: ConvShape) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize packed FP4 operands and scales, returning exact BF16 values."""
    image, weight, sfa, sfb, _output = data
    generator = torch.Generator(device=image.device).manual_seed(0)
    # All E2M1 nibble patterns are finite, including signed zero.
    lut = torch.tensor(
        (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6),
        dtype=torch.bfloat16,
        device=image.device,
    )
    sf_c = shape.c // _BLOCK_SIZE
    dequantized = []
    for operand, is_image in ((image, True), (weight, False)):
        packed = operand.permute(0, 2, 3, 4, 1).view(torch.uint8)
        packed.random_(0, 256, generator=generator)
        # Powers of two make dequantization exact in BF16 and keep values small.
        scales = (torch.randint(1, 3, (*packed.shape[:-1], sf_c), device=image.device, generator=generator) / 32).to(torch.float8_e4m3fn)
        if is_image:
            sfa.zero_()
            sfa[:, :sf_c, 0] = scales.reshape(-1, sf_c)
        else:
            # Pack logical [K, TRSC/16] scales into the F8_128x4 layout.
            columns = shape.t * shape.r * shape.s * sf_c
            row_blocks, column_blocks = _ceil_div(shape.k, 128), _ceil_div(columns, 4)
            padded = torch.zeros(row_blocks * 128, column_blocks * 4, dtype=scales.dtype, device=image.device)
            padded[: shape.k, :columns] = scales.reshape(shape.k, columns)
            blocks = padded.view(row_blocks, 128, column_blocks, 4).permute(0, 2, 1, 3)
            sfb.copy_(blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1))
        unpacked = torch.stack((lut[(packed & 0xF).long()], lut[(packed >> 4).long()]), dim=-1).flatten(-2)
        values = unpacked * scales.to(torch.bfloat16).repeat_interleave(_BLOCK_SIZE, dim=-1)
        dequantized.append(values.permute(0, 4, 1, 2, 3))
    return tuple(dequantized)


def _check_correctness(compiled, data, args, shape: ConvShape) -> None:
    """Compare with a native BF16 cuDNN plan, excluding all OSS delegates."""
    image, weight = _initialize_check_data(data, shape)
    output = data[-1]
    reference = torch.full_like(output, float("nan"))
    graph = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, compute_data_type=cudnn.data_type.FLOAT)
    image_desc, weight_desc = graph.tensor_like(image), graph.tensor_like(weight)
    output_desc = graph.conv_fprop(
        image_desc,
        weight_desc,
        pre_padding=args.pre_padding,
        post_padding=args.post_padding,
        stride=args.stride,
        dilation=args.dilation,
    )
    output_desc.set_output(True).set_dim(reference.shape).set_stride(reference.stride())
    workspace, plan_name = build_cudnn_plan(graph)
    _execute_cudnn(graph, (image_desc, weight_desc, output_desc), workspace, (image, weight, reference))
    output.fill_(float("nan"))  # Detect missing stores as well as wrong values.
    _execute_frost(compiled, data, args)
    torch.cuda.synchronize()
    tolerance = 2 * torch.finfo(torch.bfloat16).eps
    try:
        torch.testing.assert_close(output, reference, rtol=tolerance, atol=tolerance)
    except AssertionError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        raise RuntimeError(f"correctness FAILED: {args._tile_config} vs cuDNN BF16 backend (rtol=atol={tolerance})") from exc
    print(f"[correctness] {args._tile_config}: PASS vs cuDNN BF16 backend {plan_name!r} (rtol=atol={tolerance})", flush=True)


def _worker(args, shape: ConvShape, spatial: tuple[int, int, int], nbuf: int) -> int:
    warmup_data = _make_data(shape, spatial)

    if args._implementation == "frost":
        compiled = _build_frost_kernel(
            args._tile_config,
            shape,
            args.pre_padding,
            args.post_padding,
            args.stride,
            args.dilation,
        )
        execute = lambda data: _execute_frost(compiled, data, args)
        plan_name = args._tile_config
        if args.check_correctness:
            _check_correctness(compiled, warmup_data, args, shape)
    else:
        graph, tensors, workspace, plan_name = _build_cudnn_graph(
            shape,
            spatial,
            args.pre_padding,
            args.post_padding,
            args.stride,
            args.dilation,
        )
        execute = lambda data: _execute_cudnn(graph, tensors, workspace, data)

    pool = [_make_data(shape, spatial) for _ in range(nbuf)]
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
    parser = make_parser(
        __doc__,
        "1,128,6,10,10,256,3,3,3",
        "comma-separated tile config names or globs; 'all' sweeps block-scale-compatible catalog entries "
        "and 'default' selects the heuristic choice (default: all)",
    )
    parser.add_argument(
        "--check-correctness",
        action="store_true",
        help="check each FROST config against cuDNN BF16 backend on dequantized inputs before timing (default: disabled)",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not torch.cuda.is_available():
        print("No CUDA device is available.", file=sys.stderr)
        return 1
    validate_args(args)

    shape = args.shape
    try:
        spatial = output_spatial(shape, args.pre_padding, args.post_padding, args.stride, args.dilation)
        per_set = _per_set_bytes(shape, spatial)
        nbuf = resolve_nbuf(args.rotate_buffers, per_set)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args._nsys_worker:
        return _worker(args, shape, spatial, nbuf)

    from cudnn.conv.frost.templates.sm100_block_scale_conv import (
        _block_scale_config_violation,
        _block_scale_k_bytes,
        _is_block_scale_selection_candidate,
    )

    try:
        _block_scale_k_bytes(shape.c)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    z, p, q = spatial
    implicit_m = shape.n * z * p * q
    implicit_k = shape.t * shape.r * shape.s * shape.c
    heuristic_config = _get_config(
        implicit_m,
        shape.k,
        implicit_k,
        predicate=lambda config: _is_block_scale_selection_candidate(config, implicit_m, shape.k, shape.c),
    ).name
    compatible_configs = tuple(config for config in CATALOG if _block_scale_config_violation(config, shape.c) is None)
    try:
        config_names = select_tile_configs(args.configs, heuristic_config, compatible_configs)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    flops = 2 * shape.n * shape.k * z * p * q * shape.c * shape.t * shape.r * shape.s
    print(
        f"\n=== block-scale conv fprop NCDHW={shape.n}x{shape.c}x{shape.d}x{shape.h}x{shape.w} "
        f"KCTRS={shape.k}x{shape.c}x{shape.t}x{shape.r}x{shape.s} "
        f"-> {shape.n}x{shape.k}x{z}x{p}x{q} — NVFP4 x NVFP4 -> BF16 ==="
    )
    print("  [timing: nsys CUDA capture range; graph-build/JIT/warmup excluded]")
    print(f"  [rotate-buffers: {nbuf} tensor sets, {nbuf * per_set / 1024 / 1024:.0f} MB]")
    if args.check_correctness:
        print("  [correctness: each FROST config checked against cuDNN BF16 backend before capture; rtol=atol=0.015625]")
    print(f"  [tile sweep: {len(config_names)} configuration(s)]")
    return run_sweep(__file__, args, shape, nbuf, config_names, flops, allow_missing_cudnn=True)


if __name__ == "__main__":
    sys.exit(main())
