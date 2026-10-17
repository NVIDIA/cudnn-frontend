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
        num_kv_heads: KV heads of the core record (1 for the shared latent;
            2 for the QSA-shaped GQA preset). Appended, defaulted.
        kv_layout: ``"shared"`` when K and V are one latent record (the DSA
            kernels' flat MQA contract) or ``"separate"`` for per-head K and
            V tensors (QSA). Only ``"shared"`` has a servable core kernel in
            this suite. Appended, defaulted.
        index_granularity: KV rows per index entry: 1 (token top-k) or 4
            (four-token blocks: QSA, GLM-5.3-Flash). ``topk`` stays the raw
            row count; the indexer selects ``topk // index_granularity``
            entries. Appended, defaulted.
        indexer_heads: Query heads of the lightning indexer; 0 means the model
            has no indexer rows in this suite (32 for DSv4.1 / GLM-5.3, 4 for
            QSA). Appended, defaulted.
        indexer_pool_ratio: Raw tokens pooled into one index key (1, or 4 for
            QSA / GLM-5.3-Flash); the indexer kernels' ``ratio`` argument.
            Appended, defaulted.
        indexer_modes: Indexer scoring modes to enumerate for indexer rows:
            ``"dense"`` (fused scoring + top-k over the causal prefix) and
            ``"hierarchical"`` (candidate-restricted scoring over a 16384-id
            list followed by the top-k, DSv4.1 after its 20th layer).
            Appended, defaulted.
        mtp_rows: Query rows per request in generation including MTP drafts
            (2 for DSv4.1 / GLM-5.3); the generation grid runs ``q_tokens``
            in ``sorted({1, mtp_rows})``. Appended, defaulted.
    """

    name: str
    num_q_heads: int
    head_dim_qk: int = 512
    head_dim_vo: int = 512
    topk: int = 2048
    indexer_topk: int = 0
    has_sink: bool = True
    num_kv_heads: int = 1
    kv_layout: str = "shared"
    index_granularity: int = 1
    indexer_heads: int = 0
    indexer_pool_ratio: int = 1
    indexer_modes: Tuple[str, ...] = ("dense",)
    mtp_rows: int = 1


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
        components: ``"core"`` (sparse core attention over the gathered rows)
            and/or ``"indexer"`` (lightning-indexer scoring + top-k). The
            legacy default enumerates exactly today's cases. Appended,
            defaulted.
        phases: ``"context"`` (prefill, one request), ``"generation"``
            (decode / MTP rows over a long cache, batched) and ``"bprop"``.
            ``None`` derives the list from ``profile_pass`` (``both`` ->
            context + bprop, ``fwd`` -> context, ``bwd`` -> bprop), so legacy
            configs are unchanged. Appended, defaulted.
        kv_dtypes: Storage format of the KV record / index-key pool:
            ``"bfloat16"``, ``"mxfp8"`` (e4m3 + e8m0 per 32) or ``"nvfp4"``
            (e2m1 + e4m3 per 16; MXFP4 for index keys). Formats no kernel
            consumes are enumerated as skipped coverage rows with the reason
            in ``error_message``. For indexer rows ``mxfp8`` quantizes Q and
            K (the kernel has no bf16-Q x mxfp8-K form). Appended, defaulted.
        paged: Flat pool (``False``) and/or a paged pool addressed through a
            block table (``True``) with ``page_size`` rows per page. Appended,
            defaulted.
        page_size: Rows per page for paged rows; appears in
            ``backend_detail`` as ``page=<n>``. Appended, defaulted.
        generation_batch_sizes: Requests per generation case. Appended,
            defaulted.
        generation_kv_lens: Cached KV length per request for generation
            cases; ``q_tokens`` per request is ``sorted({1, mtp_rows})`` of
            the preset. Appended, defaulted.
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
    components: List[str] = field(default_factory=lambda: ["core"])
    phases: Optional[List[str]] = None
    kv_dtypes: List[str] = field(default_factory=lambda: ["bfloat16"])
    paged: List[bool] = field(default_factory=lambda: [False])
    page_size: int = 64
    generation_batch_sizes: List[int] = field(default_factory=lambda: [8, 128])
    generation_kv_lens: List[int] = field(default_factory=lambda: [8192, 131072])


@dataclass
class BenchmarkResult:
    """Result of a single benchmark case (one CSV row).

    The first 25 columns are the legacy schema and stay in this order; the
    columns after ``backend_detail`` are appended and defaulted so a legacy
    row reads the same. ``q_seqlen`` / ``kv_seqlen`` are PER REQUEST in every
    phase: a generation row has ``q_seqlen = q_tokens``, ``kv_seqlen`` = the
    cached length and ``batch_size`` requests; the worker derives the flat row
    and pool counts from them.
    """

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
    # Appended, defaulted (design: indexer + low-precision + paged + generation axes).
    component: str = "core"  # "core" | "indexer"
    phase: str = "context"  # "context" | "generation" | "bprop"; profile_pass stays fwd/bwd
    kv_dtype: str = "bfloat16"  # storage format of the KV record / index-key pool
    paged: bool = False
    batch_size: int = 1  # requests per case (1 for context / bprop)
    indexer_mode: Optional[str] = None  # "dense" | "hierarchical" on indexer rows
    # Static per-token footprint of what the row gathers or scans: the KV
    # record (core) or the amortized index key (indexer), see kv_pool.
    bytes_per_token: Optional[int] = None
    # bytes_gathered / time for success rows whose detail reports it.
    gather_gbps: Optional[float] = None
    # Sub-step timings parsed out of backend_detail (`time_step1_ms=` / `time_step2_ms=`).
    time_step1_ms: Optional[float] = None
    time_step2_ms: Optional[float] = None
