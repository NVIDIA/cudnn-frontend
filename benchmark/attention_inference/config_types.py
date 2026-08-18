# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data types for attention inference benchmark configuration.

An inference benchmark has two phases, mirroring fwd/bwd in the training suite:

- ``context``:    prefill — s_q == s_kv, causal (or model-specific mask),
                  contiguous Q/K/V, compute-bound. Reported in TFLOPS.
- ``generation``: decode — small s_q (1 for pure decode, >1 for MTP/EAGLE draft
                  widths or chunked autoregression), long cached KV,
                  bandwidth-bound. Reported in ms and GB/s (+ % of memory SOL).

Model presets describe the attention *as served*:

- ``kind="gqa"``:          per-head K and V tensors (covers MHA/GQA/MQA via
                           num_kv_heads).
- ``kind="mla_absorbed"``: decode-time MLA — every query head attends to a
                           single shared latent record; K reads all
                           ``head_dim_qk`` elements, V the first
                           ``head_dim_vo`` of the *same* record, so KV bytes
                           are counted once (``kv_shared``).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ModelPreset:
    """Attention configuration for one model."""

    name: str
    num_q_heads: int
    num_kv_heads: int
    head_dim: int = 128
    head_dim_qk: Optional[int] = None  # defaults to head_dim
    head_dim_vo: Optional[int] = None  # defaults to head_dim
    kind: str = "gqa"  # "gqa" | "mla_absorbed"
    sliding_window_size: Optional[int] = None
    # Per-head attention sinks (learnable sink logits, e.g. gpt-oss).
    has_sink: bool = False
    # Softmax scale override (e.g. Kimi K3 absorbed MLA uses 1/sqrt(192), the
    # raw QK width, not 1/sqrt(576)). None -> 1/sqrt(head_dim_qk).
    sm_scale: Optional[float] = None

    def __post_init__(self):
        if self.head_dim_qk is None:
            self.head_dim_qk = self.head_dim
        if self.head_dim_vo is None:
            self.head_dim_vo = self.head_dim

    @property
    def kv_shared(self) -> bool:
        """True when K and V are views of one record (MLA absorbed / shared-KV MQA)."""
        return self.kind == "mla_absorbed"


def with_tp_shards(preset: "ModelPreset", degrees: List[int]) -> List["ModelPreset"]:
    """Expand a whole-model preset into per-GPU tensor-parallel shards.

    Follows what serving frameworks actually do: q heads must divide evenly
    across ranks, and kv heads either divide evenly or are replicated when
    there are fewer kv heads than ranks (which requires tp % num_kv_heads ==
    0, each rank then holding one kv head). Anything else is rejected — a
    model that doesn't shard this way isn't head-parallelized in practice.
    TP=1 keeps the original preset; shards are named "<name>-tp<n>".
    """
    from dataclasses import replace

    out = []
    for n in degrees:
        if n == 1:
            out.append(preset)
            continue
        if preset.num_q_heads % n != 0:
            raise ValueError(f"{preset.name}: {preset.num_q_heads} q heads do not shard across tp={n}")
        if preset.num_kv_heads % n != 0 and n % preset.num_kv_heads != 0:
            raise ValueError(f"{preset.name}: {preset.num_kv_heads} kv heads neither divide nor replicate across tp={n}")
        out.append(
            replace(
                preset,
                name=f"{preset.name}-tp{n}",
                num_q_heads=preset.num_q_heads // n,
                num_kv_heads=max(1, preset.num_kv_heads // n),
            )
        )
    return out


@dataclass
class InferenceBenchmarkConfig:
    """One benchmark suite: a set of models swept over context and generation shapes."""

    name: str
    models: List[ModelPreset]
    # context phase, kind 1: full prefill lengths (s_q == s_kv)
    context_seqlens: List[int] = field(default_factory=list)
    # context phase, kind 2: chunked prefill (s_q, s_kv) with a small incoming
    # chunk (e.g. 512/1024) attending to a long cache (e.g. 64k/128k); the
    # chunk sits at the end of the sequence (bottom-right causal alignment).
    context_chunked_shapes: List[Tuple[int, int]] = field(default_factory=list)
    # generation phase: list of (q_tokens, kv_len). q_tokens = 1 + MTP:
    # q_tokens=1 is pure decode (MTP=0); 2/3/4 are MTP=1/2/3 draft widths.
    generation_shapes: List[Tuple[int, int]] = field(default_factory=list)
    backends: List[str] = field(default_factory=lambda: ["cudnn"])
    data_types: List[str] = field(default_factory=lambda: ["bfloat16"])
    # KV-cache dtype axis for the generation phase ("bfloat16", "fp8_e4m3").
    # Named for the serving configuration it corresponds to (fp8 KV cache);
    # NB the cudnn paths realize "fp8_e4m3" as the full fp8 attention graph —
    # q/k/v/o all e4m3 with unit descales — not bf16 queries against an fp8
    # cache. Context always uses data_types.
    kv_cache_dtypes: List[str] = field(default_factory=lambda: ["bfloat16"])
    context_batch_size: int = 1
    generation_batch_sizes: List[int] = field(default_factory=lambda: [1, 32])
    page_size: int = 64  # for paged-KV backends (flashinfer / flash_mla / b12x)
    context_causal: bool = True  # video DiT models prefill bidirectionally
    num_iterations: int = 20
    num_warmup_iterations: int = 5
    output_dir: str = "results"


@dataclass
class BenchmarkResult:
    """Result of a single benchmark case (one row in the CSV)."""

    config_name: str
    model_name: str
    phase: str  # "context" | "generation"
    backend: str
    data_type: str
    kv_cache_dtype: str
    batch_size: int
    q_tokens: int
    kv_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_vo: int
    kind: str
    sliding_window_size: Optional[int]
    page_size: Optional[int]
    time_ms: float
    tflops: float
    gbps: float
    sol_pct: Optional[float]  # % of memory-bandwidth SOL — None if peak BW unknown
    num_iterations: int
    success: bool
    error_message: Optional[str] = None
    gpu_name: Optional[str] = None
    backend_detail: Optional[str] = None  # e.g. flashinfer kernel backend / num_splits used
    cudnn_backend_version: Optional[str] = None
