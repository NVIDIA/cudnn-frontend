"""
Wan 2.2 A14B SDPA Benchmark Configuration

Benchmarks the self-attention in Wan 2.2's A14B video DiT (used by both
T2V-A14B and I2V-A14B; the two share the same transformer topology).
Self-attention is bidirectional over the patchified video-latent tokens,
so only ``no_mask`` is benchmarked (no causal/sliding window).

Architecture (from the HF diffusers config):
    - num_attention_heads = 40   (MHA, no GQA)
    - attention_head_dim  = 128  (symmetric Q=K=V head dim, no decoupled rope)
    - num_layers          = 40
    - text cross-attn KV comes from UMT5 and is handled separately; this
      config only covers the self-attention GEMMs.

Sequence lengths correspond to real video-latent token counts for typical
Wan 2.2 generations. Tokens = T_lat * (H/16) * (W/16), where
T_lat = (frames - 1) / 4 + 1 from the causal 3D VAE and the (H/16, W/16)
factors come from VAE x8 spatial + patchify x2.

    ( 7800, 17 frames, 480p  832x480)
    (17160, 41 frames, 480p  832x480)
    (32760, 81 frames, 480p  832x480)   <- A14B default training size
    (48360, 121 frames, 480p 832x480)
    (75600, 81 frames, 720p 1280x720)

Usage:
    python -m benchmark.sdpa_benchmark_training.runner --config wan22
    python -m benchmark.sdpa_benchmark_training.runner --config wan22 --dry-run
"""

from ..config_types import ModelPreset, BenchmarkConfig

WAN22_A14B = ModelPreset(
    name="wan22_a14b",
    num_q_heads=40,
    num_kv_heads=40,
    head_dim=128,
)

CONFIG = BenchmarkConfig(
    name="wan22",
    models=[WAN22_A14B],
    seqlens=[
        (7800, 7800),  # 480p, 17 frames
        (17160, 17160),  # 480p, 41 frames
        (32760, 32760),  # 480p, 81 frames (default)
        (48360, 48360),  # 480p, 121 frames
        (75600, 75600),  # 720p, 81 frames
    ],
    backends=["cudnn", "flash_attention_4"],
    data_types=["bfloat16"],  # Wan 2.2 is released/trained in bf16; no official FP8 checkpoint
    attn_masks=["no_mask"],  # bidirectional diffusion DiT, no causal mask
    profile_pass="both",  # forward + backward for training
    deterministic_bwd=[False, True],
    batch_size=1,
    num_iterations=10,
    output_dir="results",
)
