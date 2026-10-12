# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark WAN 2.2 VAE encoding with eager, compiled, and cuDNN paths.

The eager ``AutoencoderKLWan`` path is the functional reference. The timed
baseline runs the same ``AutoencoderKLWan.encode`` path after compiling its
encoder. The cuDNN path uses a purpose-built encoder module that reuses the
unchanged Diffusers layers while using cuDNN Conv3D post-operation kernels for
the input convolution, all four residual stages, the middle-block convolutions,
and output-head preparation. Attention and the output convolution remain in Torch.
The custom encoder copies history separately from fused Conv3D preparation;
these copies are included in the end-to-end timing.

By default the exact WAN 2.2 VAE architecture is instantiated with seeded
random weights. Pass ``--model`` to load a local Diffusers checkpoint. No model
is downloaded automatically. Random-weight runs test numerical agreement and
performance, not reconstruction quality.

Examples:

    python benchmark/e2e/Wan2.2-VAE/run_model.py
    python benchmark/e2e/Wan2.2-VAE/run_model.py \
        --model /path/to/Wan-AI/Wan2.2-TI2V-5B-Diffusers --frames 17 --height 480 --width 832
"""

from __future__ import annotations

import argparse
import copy
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from wan_encoder import CudnnWanEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _perfshare import profile_and_report

SYNTHETIC_CONFIG = {
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
COMPILE_OPTIONS = {"emulate_precision_casts": True}
RECOMPILE_LIMIT = 64


def load_vae(model: Path | None, seed: int) -> AutoencoderKLWan:
    """Load a local checkpoint or seeded WAN architecture as a BF16 CUDA model."""
    if model is None:
        torch.manual_seed(seed)
        vae = AutoencoderKLWan(**SYNTHETIC_CONFIG)
    else:
        model = model.expanduser().resolve()
        if not model.is_dir():
            raise ValueError(f"--model must be a local directory, got {model}")
        subfolder = "vae" if (model / "vae").is_dir() else None
        vae = AutoencoderKLWan.from_pretrained(
            model,
            subfolder=subfolder,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
    torch.nn.utils.convert_conv3d_weight_memory_format(vae, torch.channels_last_3d)
    torch.nn.utils.convert_conv2d_weight_memory_format(vae, torch.channels_last)
    return vae.to(device="cuda", dtype=torch.bfloat16).eval()


def make_video(args: argparse.Namespace) -> torch.Tensor:
    """Create a reproducible BF16 video batch on the GPU."""
    generator = torch.Generator(device="cuda").manual_seed(args.seed + 1)
    return torch.randn(
        (args.batch_size, 3, args.frames, args.height, args.width),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
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
) -> None:
    """Report numerical errors and assert elementwise agreement with eager output."""
    diff = (actual.float() - expected.float()).abs()
    reference_norm = torch.linalg.vector_norm(expected.float())
    relative_l2 = torch.linalg.vector_norm(diff) / reference_norm.clamp_min(torch.finfo(torch.float32).tiny)
    print(
        f"{name}: max_abs={diff.max().item():.6e} mean_abs={diff.mean().item():.6e} relative_l2={relative_l2.item():.6e}",
        flush=True,
    )
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def measure(
    paths: dict[str, Callable[[], torch.Tensor]],
    repeats: int,
) -> dict[str, float]:
    """Measure median CUDA-event latency while alternating encoder execution order."""
    samples = {name: [] for name in paths}
    for repeat in range(repeats):
        order = list(paths) if repeat % 2 == 0 else list(reversed(paths))
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = paths[name]()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end))
            del output
    return {name: statistics.median(values) for name, values in samples.items()}


def parse_args() -> argparse.Namespace:
    """Parse and validate the encoder benchmark workload and timing options."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="local Diffusers WAN 2.2 directory or its vae subdirectory",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="input video batch size")
    parser.add_argument("--frames", type=int, default=17, help="input frames; must be 1 or 4n+1")
    parser.add_argument("--height", type=int, default=480, help="input video height")
    parser.add_argument("--width", type=int, default=832, help="input video width")
    parser.add_argument("--seed", type=int, default=42, help="model and input random seed")
    parser.add_argument("--warmup", type=int, default=2, help="full encoder warmups per path")
    parser.add_argument("--repeats", type=int, default=5, help="timing samples per path")
    parser.add_argument("--atol", type=float, default=0.25, help="absolute validation tolerance")
    parser.add_argument("--rtol", type=float, default=1e-3, help="relative validation tolerance")
    parser.add_argument(
        "--profile-path",
        choices=("torch-compile", "cudnn-fused"),
        help="capture one warmed full encode inside a cudaProfilerApi range",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate one execution of each path without timing",
    )
    args = parser.parse_args()
    for name in ("batch_size", "frames", "height", "width", "warmup", "repeats"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.frames != 1 and (args.frames - 1) % 4:
        parser.error("--frames must be 1 or 4n+1")
    if args.height % 16 or args.width % 16:
        parser.error("--height and --width must be divisible by 16")
    return args


@torch.inference_mode()
def main() -> None:
    """Validate both encoders against eager Diffusers, then time or profile them."""
    args = parse_args()
    torch.compiler.config.recompile_limit = RECOMPILE_LIMIT
    reference_vae = load_vae(args.model, args.seed)
    if reference_vae.use_tiling or reference_vae.use_slicing:
        raise ValueError("the benchmark requires VAE tiling and slicing to be disabled")
    video = make_video(args)

    custom_vae = copy.deepcopy(reference_vae)
    custom_vae.encoder = CudnnWanEncoder(custom_vae.encoder)
    custom_vae.eval()

    source = "synthetic random weights" if args.model is None else str(args.model)
    print(
        f"device={torch.cuda.get_device_name()} source={source} shape={tuple(video.shape)} recompile_limit={RECOMPILE_LIMIT}",
        flush=True,
    )

    # Path 1: unmodified Diffusers is the functional reference, not a timed path.
    eager = encode_parameters(reference_vae, video)

    # Paths 2 and 3 retain AutoencoderKLWan's encode orchestration. Only their
    # encoder modules differ: compiled Diffusers versus the custom encoder.
    reference_vae.encoder = torch.compile(
        reference_vae.encoder,
        fullgraph=True,
        dynamic=False,
        options=COMPILE_OPTIONS,
    )
    custom_vae.encoder = torch.compile(
        custom_vae.encoder,
        fullgraph=False,
        dynamic=False,
        options=COMPILE_OPTIONS,
    )

    compiled = encode_parameters(reference_vae, video)
    custom = encode_parameters(custom_vae, video)
    torch.cuda.synchronize()
    check_close("torch.compile vs eager", compiled, eager, atol=args.atol, rtol=args.rtol)
    check_close("cuDNN fused vs eager", custom, eager, atol=args.atol, rtol=args.rtol)

    if args.check_only:
        return
    paths = {
        "torch.compile": lambda: encode_parameters(reference_vae, video),
        "cuDNN fused": lambda: encode_parameters(custom_vae, video),
    }
    for name, path in paths.items():
        print(f"warming {name}", flush=True)
        for _ in range(args.warmup):
            output = path()
            del output
    torch.cuda.synchronize()

    if args.profile_path is not None:
        profile_name = args.profile_path
        profile_path = paths["torch.compile" if profile_name == "torch-compile" else "cuDNN fused"]
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

    timings = measure(paths, args.repeats)
    print(
        f"torch.compile={timings['torch.compile']:.3f} ms "
        f"cuDNN_fused={timings['cuDNN fused']:.3f} ms "
        f"speedup={timings['torch.compile'] / timings['cuDNN fused']:.3f}x",
        flush=True,
    )

    for name, model in (("torch.compile", reference_vae), ("cuDNN fused", custom_vae)):
        print(
            f"\n{name}: supplemental forward-only perfshare report; median A/B speedup is reported above.",
            flush=True,
        )
        profile_and_report(model, video, step=encode_parameters, warmup=0, iters=args.repeats)


if __name__ == "__main__":
    main()
