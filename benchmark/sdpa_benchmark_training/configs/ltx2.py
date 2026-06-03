"""
LTX-2 SDPA Benchmark Configuration

Benchmarks the self-attention in LTX-2's video DiT (LTX-2 19B, released Oct
2025). Self-attention is bidirectional over patchified video-latent tokens,
so only ``no_mask`` is benchmarked.

Architecture (from ``LTX-2/transformer/config.json``):
    - num_attention_heads = 32   (MHA, no GQA)
    - attention_head_dim  = 128
    - num_layers          = 48
    - patch_size          = 1 x 1 x 1 (no patchify; one VAE voxel = one token)
    - vae_scale_factors   = [8, 32, 32]  (T x H x W compression)

Sequence lengths correspond to realistic LTX-2 generations. Tokens =
T_lat * (H/32) * (W/32), where T_lat = (frames - 1) / 8 + 1 from the causal
3D VAE (temporal 8x) and the spatial factors come from VAE 32x with no
further patchify.

    ( 6144, 121 frames,  512 x 768)
    (13376, 121 frames,  704 x 1216)
    (17556, 161 frames,  704 x 1216)
    (30240, 161 frames,  960 x 1536)
    (37632, 161 frames, 1024 x 1792)

Usage:
    python -m benchmark.sdpa_benchmark_training.runner --config ltx2
    python -m benchmark.sdpa_benchmark_training.runner --config ltx2 --dry-run
"""

from ..config_types import ModelPreset, BenchmarkConfig

LTX2 = ModelPreset(
    name="ltx2",
    num_q_heads=32,
    num_kv_heads=32,
    head_dim=128,
)

CONFIG = BenchmarkConfig(
    name="ltx2",
    models=[LTX2],
    seqlens=[
        (6144, 6144),  # 121 frames, 512x768
        (13376, 13376),  # 121 frames, 704x1216
        (17556, 17556),  # 161 frames, 704x1216
        (30240, 30240),  # 161 frames, 960x1536
        (37632, 37632),  # 161 frames, 1024x1792
    ],
    backends=["cudnn", "flash_attention_4"],
    data_types=["bfloat16"],  # Video DiTs are typically bf16-trained
    attn_masks=["no_mask"],  # Bidirectional diffusion DiT
    profile_pass="both",  # Forward and backward for training
    deterministic_bwd=[False, True],
    batch_size=1,
    num_iterations=10,
    output_dir="results",
)
