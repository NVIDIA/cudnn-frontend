# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark SM100 Conv3D post-operations against ``torch.compile``.

The shape is ``N,T,H,W,Ci,Co`` for the valid convolution output. Inputs add
two elements to T, H, and W. Seeded synthetic BF16 tensors are used; no model
checkpoint is required. Compilation, weight packing, correctness checks, and
CUDA Graph capture are outside the timed region.
For conv3d_rmsnorm_silu_pad, both paths write current frames and zero padding
but exclude caller-owned history copies. The e2e benchmark includes those copies.

--production filters the WAN shapes to relevant operation variants, retaining
conv3d_rmsnorm_silu as a generic comparison. Residual/history options are not swept.
Standalone preparation has no output-channel dimension (Co is shown as '-').

Examples:

    python benchmark/conv/cutedsl/benchmark_conv3d_postops.py
    python benchmark/conv/cutedsl/benchmark_conv3d_postops.py \
        --shape 32,4,320,240,160,160 --variant all --residual
    python benchmark/conv/cutedsl/benchmark_conv3d_postops.py --production
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from cudnn import (
    Conv3dBiasResidualPadSm100,
    Conv3dRawSm100,
    Conv3dRmsNormSiluPadSm100,
    Conv3dRmsNormSiluSm100,
    RmsNormSiluPadSm100,
    pack_conv3d_weight_sm100,
)

VARIANTS = ("conv3d_raw", "conv3d_rmsnorm_silu", "conv3d_rmsnorm_silu_pad", "conv3d_bias_residual_pad", "rmsnorm_silu_pad")
PRODUCTION_CASES = (
    ("32,1,320,240,160,160", ("conv3d_rmsnorm_silu", "conv3d_rmsnorm_silu_pad", "conv3d_bias_residual_pad", "rmsnorm_silu_pad")),
    ("32,4,320,240,160,160", ("conv3d_rmsnorm_silu", "conv3d_rmsnorm_silu_pad", "conv3d_bias_residual_pad", "rmsnorm_silu_pad")),
    ("32,1,160,120,160,320", ("conv3d_rmsnorm_silu", "conv3d_rmsnorm_silu_pad", "rmsnorm_silu_pad")),
    ("32,4,160,120,160,320", ("conv3d_rmsnorm_silu", "conv3d_rmsnorm_silu_pad", "rmsnorm_silu_pad")),
    ("32,1,160,120,320,320", ("conv3d_rmsnorm_silu", "conv3d_rmsnorm_silu_pad", "conv3d_bias_residual_pad")),
    ("32,4,160,120,320,320", ("conv3d_rmsnorm_silu", "conv3d_rmsnorm_silu_pad", "conv3d_bias_residual_pad")),
    ("32,1,80,60,320,640", ("conv3d_raw", "rmsnorm_silu_pad")),
    ("32,2,80,60,320,640", ("conv3d_raw", "rmsnorm_silu_pad")),
    ("32,1,80,60,640,640", ("conv3d_raw", "conv3d_bias_residual_pad", "rmsnorm_silu_pad")),
    ("32,2,80,60,640,640", ("conv3d_raw", "conv3d_bias_residual_pad", "rmsnorm_silu_pad")),
)
COMPILE_OPTIONS = {"emulate_precision_casts": True, "triton.cudagraphs": False}


@dataclass(frozen=True)
class Shape:
    """Describe a convolution output as N,T,H,W,Ci,Co."""

    n: int
    t: int
    h: int
    w: int
    ci: int
    co: int

    @classmethod
    def parse(cls, text: str) -> Shape:
        """Parse six positive, comma-separated convolution dimensions."""
        try:
            values = tuple(int(value) for value in text.split(","))
        except ValueError as error:
            raise argparse.ArgumentTypeError("shape values must be integers") from error
        if len(values) != 6 or min(values) <= 0:
            raise argparse.ArgumentTypeError("shape must be positive N,T,H,W,Ci,Co")
        return cls(*values)

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Return the NTHWC input shape including the convolution halo."""
        return self.n, self.t + 2, self.h + 2, self.w + 2, self.ci

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Return the contiguous NTHWC convolution output shape."""
        return self.n, self.t, self.h, self.w, self.co

    def __str__(self) -> str:
        """Format the dimensions as a comma-separated benchmark label."""
        values = (self.n, self.t, self.h, self.w, self.ci, self.co)
        return ",".join(str(value) for value in values)


@dataclass(frozen=True)
class Result:
    """Store the paired custom and compiled-reference timings."""

    variant: str
    shape: Shape
    custom_ms: float
    compiled_ms: float

    @property
    def shape_label(self) -> str:
        """Format the shape, omitting Co for standalone normalization."""
        s = self.shape
        return f"{s.n},{s.t},{s.h},{s.w},{s.ci},-" if self.variant == "rmsnorm_silu_pad" else str(s)

    @property
    def speedup(self) -> float:
        """Return compiled-reference latency divided by custom latency."""
        return self.compiled_ms / self.custom_ms


def _conv(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute valid Torch Conv3D with NTHWC input and output."""
    return F.conv3d(input.permute(0, 4, 1, 2, 3), weight).permute(0, 2, 3, 4, 1)


def _norm_silu(
    conv: torch.Tensor,
    bias: torch.Tensor,
    gamma: torch.Tensor,
    residual: torch.Tensor | None,
    residual_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply bias, optional residual, RMSNorm, and SiLU with explicit BF16 rounding."""
    value = (conv.float() + bias.float()).to(torch.bfloat16)
    if residual is not None:
        skip = residual
        if residual_bias is not None:
            skip = (skip.float() + residual_bias.float()).to(torch.bfloat16)
        value = (value.float() + skip.float()).to(torch.bfloat16)
    normalized = F.normalize(value.float(), dim=-1).to(torch.bfloat16)
    scaled = (normalized.float() * math.sqrt(value.shape[-1])).to(torch.bfloat16)
    affine = (scaled.float() * gamma.float()).to(torch.bfloat16)
    activated = F.silu(affine.float()).to(torch.bfloat16)
    return activated, value if residual is not None else None


def _compiled_reference(
    variant: str,
    *,
    prepared_outputs: tuple[torch.Tensor, torch.Tensor] | None = None,
    history: int = 0,
) -> Callable:
    """Build a compiled reference matching the selected variant's output contract."""
    if variant == "conv3d_raw":

        def reference(input, weight, bias, gamma, previous, residual, residual_bias):
            """Compute raw convolution without post-operations."""
            del bias, gamma, previous, residual, residual_bias
            return (_conv(input, weight),)

    elif variant == "conv3d_rmsnorm_silu":

        def reference(input, weight, bias, gamma, previous, residual, residual_bias):
            """Compute convolution and normalized activation with an optional saved residual sum."""
            del previous
            return _norm_silu(_conv(input, weight), bias, gamma, residual, residual_bias)

    elif variant == "conv3d_rmsnorm_silu_pad":
        assert prepared_outputs is not None
        padded, cache = prepared_outputs
        current_output = padded[:, 2:, 1:-1, 1:-1, :]
        current_frames = min(2, current_output.shape[1])
        current_cache = cache[:, -current_frames:]
        zero_regions = [
            padded[:, 2 - history :, 0, :, :],
            padded[:, 2 - history :, -1, :, :],
            padded[:, 2 - history :, 1:-1, 0, :],
            padded[:, 2 - history :, 1:-1, -1, :],
        ]
        if history < 2:
            zero_regions.append(padded[:, : 2 - history])

        def reference(input, weight, bias, gamma, residual, residual_bias, output, cache_output):
            """Write current activations and cache frames without touching existing history."""
            activated, residual_output = _norm_silu(_conv(input, weight), bias, gamma, residual, residual_bias)
            output.copy_(activated)
            cache_output.copy_(activated[:, -current_frames:])
            return residual_output

        def zero(destination):
            """Clear one padding region in place."""
            destination.zero_()

        compiled = torch.compile(reference, fullgraph=True, dynamic=False, options=COMPILE_OPTIONS)
        compiled_zero = torch.compile(zero, fullgraph=True, dynamic=False, options=COMPILE_OPTIONS)

        def prepared_reference(input, weight, bias, gamma, previous, residual, residual_bias):
            """Run compiled padding and activation writes while preserving caller-owned history."""
            # Separate destination views prevent functionalization from copying
            # untouched history. CUDA graphs exclude the extra Python launches.
            for region in zero_regions:
                compiled_zero(region)
            residual_output = compiled(input, weight, bias, gamma, residual, residual_bias, current_output, current_cache)
            return padded, cache, residual_output

        return prepared_reference

    elif variant == "rmsnorm_silu_pad":

        def reference(input, weight, bias, gamma, previous, residual, residual_bias):
            """Normalize activations, prepend history, and return padded output and cache."""
            del weight
            activated, residual_output = _norm_silu(input, bias, gamma, residual, residual_bias)
            joined = torch.cat((previous, activated), dim=1) if previous is not None else activated
            history = previous.shape[1] if previous is not None else 0
            padded = F.pad(joined, (0, 0, 1, 1, 1, 1, 2 - history, 0))
            return padded, joined[:, -2:].contiguous(), residual_output

    else:

        def reference(input, weight, bias, gamma, previous, residual, residual_bias):
            """Add convolution bias and residual, then pad the bottom and right edges."""
            del gamma, previous
            conv = (_conv(input, weight).float() + bias.float()).to(torch.bfloat16)
            skip = residual
            if residual_bias is not None:
                skip = (skip.float() + residual_bias.float()).to(torch.bfloat16)
            summed = (conv.float() + skip.float()).to(torch.bfloat16)
            return (F.pad(summed, (0, 0, 0, 1, 0, 1)),)

    return torch.compile(
        reference,
        fullgraph=True,
        dynamic=False,
        options=COMPILE_OPTIONS,
    )


def _capture(fn: Callable) -> tuple[torch.cuda.CUDAGraph, object]:
    """Warm up and capture a callable while retaining its output buffers."""
    for _ in range(3):
        output = fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


def _time_pair(reference: Callable, custom: Callable, repeats: int) -> tuple[float, float]:
    """Measure median graph-replay latency with alternating reference/custom order."""
    graphs = []
    captured_outputs = []
    for fn in (reference, custom):
        graph, output = _capture(fn)
        graphs.append(graph)
        # CUDA Graph outputs must remain alive while their graphs are replayed.
        captured_outputs.append(output)

    samples = [[], []]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for sample in range(7):
        for index in (0, 1) if sample % 2 == 0 else (1, 0):
            start.record()
            for _ in range(repeats):
                graphs[index].replay()
            end.record()
            end.synchronize()
            samples[index].append(start.elapsed_time(end) / repeats)
    return tuple(statistics.median(values) for values in samples)


def _assert_outputs_close(actual, expected) -> None:
    """Compare corresponding output tensors and optional-output presence."""
    assert len(actual) == len(expected)
    for actual_tensor, expected_tensor in zip(actual, expected):
        if actual_tensor is None or expected_tensor is None:
            assert actual_tensor is expected_tensor
            continue
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=0.02, rtol=0.02)


@torch.inference_mode()
def run_case(
    shape: Shape,
    variant: str,
    *,
    history: int,
    use_residual: bool,
    repeats: int,
) -> Result | None:
    """Validate and time one variant, or warn and skip a rejected configuration."""
    if variant == "conv3d_bias_residual_pad":
        use_residual = True
    elif variant == "conv3d_raw":
        use_residual = False
    has_norm = variant in ("conv3d_rmsnorm_silu", "conv3d_rmsnorm_silu_pad", "rmsnorm_silu_pad")
    generator = torch.Generator(device="cuda").manual_seed(shape.ci * 1009 + shape.co * 917 + shape.t)
    input_shape = (shape.n, shape.t, shape.h, shape.w, shape.ci) if variant == "rmsnorm_silu_pad" else shape.input_shape
    input = (
        torch.randn(
            input_shape,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    weight = packed_weight = None
    if variant != "rmsnorm_silu_pad":
        weight = (
            torch.randn(
                (shape.co, shape.ci, 3, 3, 3),
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * (shape.ci * 27) ** -0.5
        )
        packed_weight = pack_conv3d_weight_sm100(weight)
    post_channels = shape.ci if variant == "rmsnorm_silu_pad" else shape.co
    post_shape = (shape.n, shape.t, shape.h, shape.w, post_channels)
    bias = torch.randn(post_channels, generator=generator, device="cuda", dtype=torch.bfloat16) * 0.1 if variant != "conv3d_raw" else None
    gamma = torch.randn(post_channels, generator=generator, device="cuda", dtype=torch.bfloat16) if has_norm else None
    previous = (
        torch.randn(
            (shape.n, history, shape.h, shape.w, post_channels),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        if variant == "rmsnorm_silu_pad" and history
        else None
    )
    residual = (
        torch.randn(
            post_shape,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        if use_residual
        else None
    )
    residual_bias = bias.clone() if use_residual else None
    residual_output = torch.empty_like(residual) if has_norm and residual is not None else None
    if variant in ("conv3d_rmsnorm_silu_pad", "rmsnorm_silu_pad"):
        padded = torch.empty(
            (shape.n, shape.t + 2, shape.h + 2, shape.w + 2, post_channels),
            device="cuda",
            dtype=torch.bfloat16,
        )
        cache = torch.empty(
            (shape.n, min(2, shape.t + history), shape.h, shape.w, post_channels),
            device="cuda",
            dtype=torch.bfloat16,
        )

    # Reference closures share code objects; isolate each case's shape specialization.
    torch.compiler.reset()
    if variant == "conv3d_raw":
        output = torch.empty(shape.output_shape, device="cuda", dtype=torch.bfloat16)
        plan = Conv3dRawSm100(input, packed_weight, output)
        custom_call = lambda: (plan.execute(input, packed_weight, output),)
    elif variant == "conv3d_rmsnorm_silu":
        output = torch.empty(shape.output_shape, device="cuda", dtype=torch.bfloat16)
        plan = Conv3dRmsNormSiluSm100(
            input,
            packed_weight,
            bias,
            gamma,
            output,
            residual,
            residual_bias,
            residual_output,
        )
        custom_call = lambda: plan.execute(
            input,
            packed_weight,
            bias,
            gamma,
            output,
            residual,
            residual_bias,
            residual_output,
        )
    elif variant == "conv3d_rmsnorm_silu_pad":
        plan = Conv3dRmsNormSiluPadSm100(
            input,
            packed_weight,
            bias,
            gamma,
            padded,
            cache,
            history,
            residual,
            residual_bias,
            residual_output,
        )
        custom_call = lambda: plan.execute(
            input,
            packed_weight,
            bias,
            gamma,
            padded,
            cache,
            residual,
            residual_bias,
            residual_output,
        )
    elif variant == "rmsnorm_silu_pad":
        plan = RmsNormSiluPadSm100(
            input,
            gamma,
            padded,
            cache,
            bias,
            previous,
            residual,
            residual_bias,
            residual_output,
        )
        custom_call = lambda: plan.execute(
            input,
            gamma,
            padded,
            cache,
            bias,
            previous,
            residual,
            residual_bias,
            residual_output,
        )
    else:
        padded = torch.empty(
            (shape.n, shape.t, shape.h + 1, shape.w + 1, shape.co),
            device="cuda",
            dtype=torch.bfloat16,
        )
        plan = Conv3dBiasResidualPadSm100(input, packed_weight, bias, residual, padded, residual_bias)
        custom_call = lambda: (plan.execute(input, packed_weight, bias, residual, padded, residual_bias),)

    try:
        plan.check_support()
    except (ValueError, NotImplementedError) as error:
        print(f"WARNING: skipping {variant} shape={shape}: {error}", flush=True)
        return None
    plan.compile()
    prepared_outputs = None
    if variant == "conv3d_rmsnorm_silu_pad":
        # History is caller-owned; initialize it once, outside kernel timing.
        padded.zero_()
        cache.zero_()
        prepared_outputs = (torch.zeros_like(padded), torch.zeros_like(cache))
    compiled = _compiled_reference(
        variant,
        prepared_outputs=prepared_outputs,
        history=history,
    )
    compiled_call = lambda: compiled(input, weight, bias, gamma, previous, residual, residual_bias)
    expected = compiled_call()
    actual = custom_call()
    _assert_outputs_close(actual, expected)
    compiled_ms, custom_ms = _time_pair(compiled_call, custom_call, repeats)
    return Result(variant, shape, custom_ms, compiled_ms)


def _print_results(results: list[Result]) -> None:
    """Print paired latencies and speedups for the completed cases."""
    headers = (
        "Variant",
        "N,T,H,W,Ci,Co",
        "Custom (ms)",
        "torch.compile (ms)",
        "Speedup",
    )
    rows = [
        (
            result.variant,
            result.shape_label,
            f"{result.custom_ms:.4f}",
            f"{result.compiled_ms:.4f}",
            f"{result.speedup:.3f}x",
        )
        for result in results
    ]
    widths = [max(len(row[index]) for row in (headers, *rows)) for index in range(len(headers))]
    for row_index, row in enumerate((headers, *rows)):
        print(
            " | ".join(value.ljust(width) if index < 2 else value.rjust(width) for index, (value, width) in enumerate(zip(row, widths))),
            flush=True,
        )
        if row_index == 0:
            print("-+-".join("-" * width for width in widths), flush=True)


def main() -> None:
    """Parse benchmark options and run the selected shape/variant cases."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=Shape.parse,
        default=argparse.SUPPRESS,
        help=("output shape N,T,H,W,Ci,Co; may be repeated (default: 1,2,4,5,320,320)"),
    )
    parser.add_argument("--production", action="store_true", help="run relevant WAN shape/variant pairs, including generic Conv3D + norm/SiLU")
    parser.add_argument(
        "--variant",
        choices=("all", *VARIANTS),
        default="all",
        help="operation variant to benchmark",
    )
    parser.add_argument(
        "--history",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="number of prior activated frames for prepared output",
    )
    parser.add_argument(
        "--residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include residual and residual-bias inputs when supported",
    )
    parser.add_argument(
        "--replays",
        type=int,
        default=10,
        help="CUDA Graph replays per timing sample",
    )
    args = parser.parse_args()
    shapes_arg = getattr(args, "shape", None)
    if args.replays < 1:
        parser.error("--replays must be positive")
    if args.production and shapes_arg:
        parser.error("--production and --shape are mutually exclusive")
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    if args.production:
        cases = [(Shape.parse(shape), variant) for shape, shape_variants in PRODUCTION_CASES for variant in shape_variants if variant in variants]
    else:
        cases = [(shape, variant) for shape in shapes_arg or [Shape.parse("1,2,4,5,320,320")] for variant in variants]
    print(
        f"device={torch.cuda.get_device_name()} torch={torch.__version__} " f"compile_options={COMPILE_OPTIONS} synthetic_weights=true",
        flush=True,
    )
    results = []
    for shape, variant in cases:
        print(f"running {variant} shape={shape}", flush=True)
        result = run_case(shape, variant, history=args.history, use_residual=args.residual, repeats=args.replays)
        if result is not None:
            results.append(result)
    _print_results(results)


if __name__ == "__main__":
    main()
