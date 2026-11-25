# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Wan 2.1/2.2 VAE encoding or decoding with public cuDNN APIs.

Both timed paths compile the selected component as a full graph and retain
Diffusers temporal orchestration. Eager Diffusers under the same arithmetic
permission is the accuracy reference. History copies and allocation are timed;
compilation, validation, and warmup are excluded.

Use --model for a local checkpoint, or --wan-version for seeded synthetic
weights. See README.md for supported policies,
validation criteria, and example commands.
"""

from __future__ import annotations

import argparse
import copy
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from wan_encoder import CudnnWanEncoder
from wan_decoder import CudnnWanDecoder

WAN22_CONFIG = {
    "base_dim": 160,
    "decoder_base_dim": 256,
    "z_dim": 48,
    "dim_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_scales": [],
    "temperal_downsample": [False, True, True],
    "dropout": 0.0,
    "latents_mean": [0.0] * 48,
    "latents_std": [1.0] * 48,
    "is_residual": True,
    "in_channels": 12,
    "out_channels": 12,
    "patch_size": 2,
    "scale_factor_temporal": 4,
    "scale_factor_spatial": 16,
}
WAN21_CONFIG = {
    **WAN22_CONFIG,
    "base_dim": 96,
    "decoder_base_dim": 96,
    "z_dim": 16,
    "latents_mean": [0.0] * 16,
    "latents_std": [1.0] * 16,
    "is_residual": False,
    "in_channels": 3,
    "out_channels": 3,
    "patch_size": None,
    "scale_factor_spatial": 8,
}
# Inductor's intermediate padding can miscompile Wan's repeat/reshape shortcut
# with channels-last inputs. Preserve its declared strides in both timed paths.
COMPILE_OPTIONS = {"emulate_precision_casts": True, "comprehensive_padding": False}
RECOMPILE_LIMIT = 64


def load_vae(model: Path | None, seed: int, wan_version: str, dtype: torch.dtype) -> AutoencoderKLWan:
    """Load a local checkpoint or seeded WAN architecture on CUDA."""
    if model is None:
        torch.manual_seed(seed)
        vae = AutoencoderKLWan(**(WAN21_CONFIG if wan_version == "2.1" else WAN22_CONFIG))
    else:
        model = model.expanduser().resolve()
        if not model.is_dir():
            raise ValueError(f"--model must be a local directory, got {model}")
        subfolder = "vae" if (model / "vae").is_dir() else None
        vae = AutoencoderKLWan.from_pretrained(
            model,
            subfolder=subfolder,
            torch_dtype=dtype,
            local_files_only=True,
        )
    torch.nn.utils.convert_conv3d_weight_memory_format(vae, torch.channels_last_3d)
    torch.nn.utils.convert_conv2d_weight_memory_format(vae, torch.channels_last)
    return vae.to(device="cuda", dtype=dtype).eval()


def make_video(args: argparse.Namespace, vae) -> torch.Tensor:
    """Create reproducible video or latent input on the GPU."""
    generator = torch.Generator(device="cuda").manual_seed(args.seed + 1)
    return torch.randn(
        (
            (args.batch_size, 3, args.frames, args.height, args.width)
            if args.operation == "encode"
            else (
                args.batch_size,
                vae.config.z_dim,
                (args.frames - 1) // 4 + 1,
                args.height // vae.config.scale_factor_spatial,
                args.width // vae.config.scale_factor_spatial,
            )
        ),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32,
    )


def encode_parameters(vae: AutoencoderKLWan, video: torch.Tensor) -> torch.Tensor:
    """Run the Diffusers encode API and return its posterior parameters."""
    return vae.encode(video, return_dict=False)[0].parameters


def check_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    operation: str,
) -> None:
    """Validate outputs against eager Diffusers under the matching precision policy."""
    diff = (actual.float() - expected.float()).abs()
    reference_norm = torch.linalg.vector_norm(expected.float())
    relative_l2 = torch.linalg.vector_norm(diff) / reference_norm.clamp_min(torch.finfo(torch.float32).tiny)
    print(
        f"{name}: max_abs={diff.max().item():.6e} mean_abs={diff.mean().item():.6e} relative_l2={relative_l2.item():.6e}",
        flush=True,
    )
    if not torch.isfinite(actual).all():
        raise AssertionError(f"{name}: nonfinite output")
    if actual.dtype == torch.float32 and not torch.backends.cudnn.allow_tf32:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    elif operation == "encode":
        limit = 0.02 if actual.dtype == torch.bfloat16 else 0.01
        for label, value, reference in zip(("mean", "logvar"), actual.chunk(2, 1), expected.chunk(2, 1)):
            error = (value.float() - reference.float()).norm() / reference.float().norm().clamp_min(torch.finfo(torch.float32).tiny)
            print(f"  {label}: NRMSE={error.item():.6e}, limit={limit:g}", flush=True)
            assert error.item() <= limit, f"{name}: {label} NRMSE exceeds {limit}"
    else:
        psnr = 10 * torch.log10(4 / diff.square().mean().clamp_min(torch.finfo(torch.float32).tiny))
        print(f"  PSNR={psnr.item():.3f} dB, limit=50 dB", flush=True)
        assert psnr.item() >= 50, f"{name}: decoder PSNR below 50 dB"


def measure(
    paths: dict[str, Callable[[], torch.Tensor]],
    repeats: int,
    warmup: int = 2,
) -> dict[str, float]:
    """Warm each path before consecutive samples to settle its power/clock state."""
    samples = {name: [] for name in paths}
    for name, path in paths.items():
        print(f"warming {name}", flush=True)
        for _ in range(warmup):
            path()
        torch.cuda.synchronize()
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = path()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end))
            del output
    return {name: statistics.median(values) for name, values in samples.items()}


def parse_args() -> argparse.Namespace:
    """Parse the model, operation, arithmetic policy, and workload."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="local Diffusers WAN 2.1/2.2 directory or its vae subdirectory; architecture comes from the checkpoint",
    )
    parser.add_argument("--wan-version", choices=("2.1", "2.2"), default="2.2", help="architecture for synthetic weights; ignored with --model")
    parser.add_argument("--batch-size", type=int, default=1, help="input video batch size")
    parser.add_argument("--frames", type=int, default=17, help="input frames; must be 1 or 4n+1")
    parser.add_argument("--height", type=int, default=480, help="input video height")
    parser.add_argument("--width", type=int, default=832, help="input video width")
    parser.add_argument("--seed", type=int, default=42, help="model and input random seed")
    parser.add_argument("--warmup", type=int, default=2, help="full-model warmups per path")
    parser.add_argument("--repeats", type=int, default=3, help="timing samples per path")
    parser.add_argument("--atol", type=float, default=1e-4, help="strict FP32 absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-4, help="strict FP32 relative tolerance")
    parser.add_argument(
        "--profile-path",
        choices=("torch-compile", "cudnn-fused"),
        help="capture one warmed full-model call inside a cudaProfilerApi range",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate one execution of each path without timing",
    )
    parser.add_argument("--operation", choices=("encode", "decode"), default="encode")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--allow-tf32", action=argparse.BooleanOptionalAction, default=False, help="allow TF32 convolution; FP32 matrix multiplies remain strict"
    )
    args = parser.parse_args()
    for name in ("batch_size", "frames", "height", "width", "warmup", "repeats"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.frames != 1 and (args.frames - 1) % 4:
        parser.error("--frames must be 1 or 4n+1")
    return args


@torch.inference_mode()
def main() -> None:
    """Validate both implementations against eager Diffusers, then time or profile them."""
    args = parse_args()
    torch.compiler.config.recompile_limit = RECOMPILE_LIMIT
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    reference_vae = load_vae(args.model, args.seed, args.wan_version, dtype)
    if reference_vae.use_tiling or reference_vae.use_slicing:
        raise ValueError("the benchmark requires VAE tiling and slicing to be disabled")
    spatial_factor = reference_vae.config.scale_factor_spatial
    if args.height % spatial_factor or args.width % spatial_factor:
        raise ValueError(f"--height and --width must be divisible by {spatial_factor}")
    video = make_video(args, reference_vae)

    custom_vae = copy.deepcopy(reference_vae)
    component = "encoder" if args.operation == "encode" else "decoder"
    constructor = CudnnWanEncoder if args.operation == "encode" else CudnnWanDecoder
    setattr(custom_vae, component, constructor(getattr(custom_vae, component), is_residual=custom_vae.config.is_residual))

    def run(vae, value):
        if args.operation == "encode":
            return encode_parameters(vae, value)
        return vae.decode(value, return_dict=False)[0]

    custom_vae.eval()

    source = "synthetic random weights" if args.model is None else str(args.model)
    print(
        f"Wan {'2.2' if reference_vae.config.is_residual else '2.1'} VAE {component} | dtype={args.dtype} conv_tf32={args.allow_tf32} matmul_tf32=False | device={torch.cuda.get_device_name()} source={source} shape={tuple(video.shape)} recompile_limit={RECOMPILE_LIMIT}",
        flush=True,
    )

    # Compute the eager Diffusers output for validation.
    eager = run(reference_vae, video)

    # Both candidates retain the same public temporal orchestration.
    for label, vae in (("torch.compile", reference_vae), ("cuDNN fused", custom_vae)):
        setattr(vae, component, torch.compile(getattr(vae, component), fullgraph=True, dynamic=False, options=COMPILE_OPTIONS))
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        output = run(vae, video)
        torch.cuda.synchronize()
        print(f"{label}: first_call={time.perf_counter()-start:.3f} s peak_additional={(torch.cuda.max_memory_allocated()-base)/2**30:.3f} GiB", flush=True)
        check_close(f"{label} vs eager", output, eager, atol=args.atol, rtol=args.rtol, operation=args.operation)
        del output

    if args.check_only:
        return
    paths = {
        "torch.compile": lambda: run(reference_vae, video),
        "cuDNN fused": lambda: run(custom_vae, video),
    }
    if args.profile_path is not None:
        profile_name = args.profile_path
        profile_path = paths["torch.compile" if profile_name == "torch-compile" else "cuDNN fused"]
        print(f"warming {profile_name}", flush=True)
        for _ in range(args.warmup):
            profile_path()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push(profile_name)
        try:
            output = profile_path()
            del output
        finally:
            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        print(f"captured {profile_name}", flush=True)
        return

    timings = measure(paths, args.repeats, args.warmup)
    print(
        f"torch.compile={timings['torch.compile']:.3f} ms "
        f"cuDNN_fused={timings['cuDNN fused']:.3f} ms "
        f"speedup={timings['torch.compile'] / timings['cuDNN fused']:.3f}x",
        flush=True,
    )


if __name__ == "__main__":
    main()
