# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data types for the DSA (DeepSeek Sparse Attention) benchmark configuration.

Mirrors ``../attention_training/config_types.py``: a :class:`ModelPreset`
describes the sparse attention geometry as served, a
:class:`DsaBenchmarkConfig` sweeps presets over sequence lengths and passes,
and every executed case is one :class:`BenchmarkResult` row in the CSV.

DSA is flat MQA over a shared K=V record: each query head gathers the same
``topk`` KV rows selected by the lightning indexer (plus any sliding-window
tokens the model folds into the index list). Work per query is therefore
``topk`` rows regardless of sequence length, and TFLOPS are counted on the
gathered rows only.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ModelPreset:
    """Sparse attention configuration for one model.

    Attributes:
        name: Identifier used in CSV rows and chart legends.
        num_q_heads: Query heads (there is a single shared KV head).
        head_dim_qk: Width of the shared K record (512, or 576 = 512 + 64 RoPE).
        head_dim_vo: Width read as V / written as O (512).
        topk: Logical K — KV rows gathered per query.
        indexer_topk: Leading prefix of ``topk`` that the indexer selected; the
            forward kernel emits a separate LSE over it (0 = off). Forward only.
        has_sink: Per-head attention-sink logit in the softmax denominator.
    """

    name: str
    num_q_heads: int
    head_dim_qk: int = 512
    head_dim_vo: int = 512
    topk: int = 2048
    indexer_topk: int = 0
    has_sink: bool = True


@dataclass
class DsaBenchmarkConfig:
    """One benchmark suite: presets swept over seqlens x backends x dtypes x passes.

    Attributes:
        seqlens: ``(q_seqlen, kv_seqlen)`` pairs; the KV pool the indices
            address has ``kv_seqlen`` rows.
        backends: ``cudnn`` = the cuDNN Frontend DSA CuTe-DSL kernels.
        profile_pass: ``fwd``, ``bwd`` or ``both`` (expanded to two cases).
        deterministic_bwd: Deterministic modes to run for the backward pass.
        use_topk_length: Pass a per-query ``topk_length`` tensor (every row at
            the full top-k). Kernels with and without it are different
            compiled variants; production passes it.
    """

    name: str
    models: List[ModelPreset]
    seqlens: List[Tuple[int, int]]
    backends: List[str] = field(default_factory=lambda: ["cudnn"])
    data_types: List[str] = field(default_factory=lambda: ["bfloat16"])
    profile_pass: str = "both"
    deterministic_bwd: List[bool] = field(default_factory=lambda: [False])
    use_topk_length: bool = True
    num_iterations: int = 20
    num_warmup_iterations: int = 5
    output_dir: str = "results"


@dataclass
class BenchmarkResult:
    """Result of a single benchmark case (one CSV row)."""

    config_name: str
    model_name: str
    backend: str
    data_type: str

    q_seqlen: int
    kv_seqlen: int
    num_q_heads: int
    head_dim_qk: int
    head_dim_vo: int
    topk: int
    indexer_topk: int
    has_sink: bool
    use_topk_length: bool

    profile_pass: str
    deterministic_bwd: bool

    time_ms: float
    tflops: float
    num_iterations: int

    success: bool = True
    # True when the kernel's own support check rejected this (arch, shape,
    # pass) combination — an honest gap, not a crash. Always success=False.
    skipped: bool = False
    error_message: Optional[str] = None
    gpu_name: Optional[str] = None
    cudnn_version: Optional[str] = None
    # Dense-MMA peak TFLOPS for this GPU + dtype at the SM clock sampled
    # during the measurement — the SOL denominator. None if unavailable.
    peak_mma_tflops: Optional[float] = None
    backend_detail: Optional[str] = None
