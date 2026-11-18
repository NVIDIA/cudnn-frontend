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
        num_q_heads: Query heads (DSA: over a single shared KV head).
        head_dim_qk: Width of the shared K record (512, or 576 = 512 + 64 RoPE);
            256 for QSA.
        head_dim_vo: Width read as V / written as O (512; 256 for QSA).
        topk: Logical K — KV rows gathered per query (DSA), or the per-query
            block budget (QSA: up to ``topk`` complete causal blocks, plus the
            partial tail block).
        indexer_topk: Leading prefix of ``topk`` that the indexer selected; the
            forward kernel emits a separate LSE over it (0 = off). Forward only.
        has_sink: Per-head attention-sink logit in the softmax denominator.
        kind: ``dsa`` = flat MQA over one shared K=V record (``cudnn.DSA``);
            ``qsa`` = packed GQA over per-head K/V with a four-token block
            selection bitmap (``cudnn.ops.qsa_forward`` / ``qsa_backward``).
        num_kv_heads: KV heads (1 for DSA's shared record; QSA keeps its own).
        block_size: Tokens per selected block (QSA = 4; DSA gathers single rows).
    """

    name: str
    num_q_heads: int
    head_dim_qk: int = 512
    head_dim_vo: int = 512
    topk: int = 2048
    indexer_topk: int = 0
    has_sink: bool = True
    kind: str = "dsa"
    num_kv_heads: int = 1
    block_size: int = 1


def with_tp_shards(preset: ModelPreset, degrees: List[int]) -> List[ModelPreset]:
    """Expand a whole-model preset into per-GPU tensor-parallel shards.

    Same rules as ``../attention_inference/config_types.with_tp_shards`` and
    what training and serving frameworks do: q heads must divide evenly across
    ranks, and kv heads either divide evenly or are replicated when there are
    fewer kv heads than ranks (``tp % num_kv_heads == 0``, each rank then
    holding one kv head). Anything else is rejected. Tokens do not shard, so
    every rank runs the full sequence with its slice of heads. TP=1 keeps the
    original preset; shards are named ``<name>-tp<n>``. A shard the kernels do
    not serve is still a case: its support check lands it as a skipped row.
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

    # Preset kind and the geometry fields DSA rows leave at their defaults
    # (one shared KV record, single-row gather).
    kind: str = "dsa"
    num_kv_heads: int = 1
    block_size: int = 1

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
