"""
Core types for the SDPA benchmark configuration system.

This module defines the dataclasses used to configure and collect results
from SDPA benchmarks.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class ModelPreset:
    """
    Represents a named model configuration preset.

    Defines the attention head configuration for a specific model architecture.
    Can use either symmetric head dimensions (head_dim) or asymmetric
    (head_dim_qk, head_dim_vo) for models like DeepSeek V3.

    Attributes:
        name: Identifier for this preset (e.g., "llama3.1", "dsv3")
        num_q_heads: Number of query heads
        num_kv_heads: Number of key/value heads (differs from num_q_heads for GQA)
        head_dim: Head dimension (used if head_dim_qk/vo not specified)
        head_dim_qk: Head dimension for Q/K tensors (optional, for asymmetric)
        head_dim_vo: Head dimension for V/O tensors (optional, for asymmetric)

    Example:
        # Symmetric head dimensions (Llama 3.1)
        LLAMA3_1 = ModelPreset(
            name="llama3.1",
            num_q_heads=64,
            num_kv_heads=8,
            head_dim=128,
        )

        # Asymmetric head dimensions (DeepSeek V3)
        DSV3 = ModelPreset(
            name="dsv3",
            num_q_heads=128,
            num_kv_heads=128,
            head_dim_qk=192,
            head_dim_vo=128,
        )
    """

    name: str
    num_q_heads: int
    num_kv_heads: int
    head_dim: int = 128
    head_dim_qk: Optional[int] = None
    head_dim_vo: Optional[int] = None

    def __post_init__(self):
        """Resolve head dimensions after initialization."""
        if self.head_dim_qk is None:
            self.head_dim_qk = self.head_dim
        if self.head_dim_vo is None:
            self.head_dim_vo = self.head_dim


@dataclass
class BenchmarkConfig:
    """
    Configuration for a benchmark suite.

    Defines a set of benchmarks to run. The runner will expand this into
    individual benchmark cases via cartesian product of:
        models x seqlens x backends x data_types x attn_masks x deterministic_bwd

    Attributes:
        name: Identifier for this config (used in output filenames)
        models: List of ModelPreset to benchmark
        seqlens: List of (q_seqlen, kv_seqlen) tuples
        backends: List of backend names (e.g., ["cudnn", "flash_attention_4"])
        data_types: List of data types (e.g., ["bfloat16", "fp8"])
        attn_masks: List of attention masks (e.g., ["top_left", "no_mask"])
        profile_pass: Which pass to profile ("fwd", "bwd", or "both")
        batch_size: Batch size for all benchmarks
        num_iterations: Number of iterations per benchmark
        num_warmup_iterations: Warmup iterations before measurement
        skip_ref: Skip reference validation
        deterministic_bwd: List of deterministic modes to test for backward pass
        output_dir: Directory for output files

    Example:
        CONFIG = BenchmarkConfig(
            name="my_benchmark",
            models=[LLAMA3_1, DSV3],
            seqlens=[(4096, 4096), (8192, 8192)],
            backends=["cudnn", "flash_attention_4"],
            data_types=["bfloat16", "fp8"],
            attn_masks=["top_left", "no_mask"],
            profile_pass="fwd",
        )
    """

    name: str
    models: List[ModelPreset]
    seqlens: List[Tuple[int, int]]
    backends: List[str] = field(default_factory=lambda: ["cudnn"])
    data_types: List[str] = field(default_factory=lambda: ["bfloat16"])
    attn_masks: List[str] = field(default_factory=lambda: ["top_left"])
    profile_pass: str = "fwd"
    batch_size: int = 1
    num_iterations: int = 10
    num_warmup_iterations: int = 0
    skip_ref: bool = True
    deterministic_bwd: List[bool] = field(default_factory=lambda: [False])
    output_dir: str = "../results"


@dataclass
class BenchmarkResult:
    """
    Result from a single benchmark execution.

    Contains both the configuration that was run and the measured results.

    Attributes:
        config_name: Name of the BenchmarkConfig this result belongs to
        model_name: Name of the ModelPreset used
        backend: Backend that was used
        data_type: Data type that was used
        attn_mask: Attention mask that was used
        batch_size: Batch size
        q_seqlen: Query sequence length
        kv_seqlen: Key/value sequence length
        num_q_heads: Number of query heads
        num_kv_heads: Number of key/value heads
        head_dim_qk: Head dimension for Q/K
        head_dim_vo: Head dimension for V/O
        profile_pass: Which pass was profiled
        deterministic_bwd: Whether deterministic backward was used
        fwd_time_ms: Forward pass time in milliseconds
        bwd_time_ms: Backward pass time in milliseconds (0 if not run)
        fwd_tflops: Forward pass throughput in TFLOPS
        bwd_tflops: Backward pass throughput in TFLOPS
        max_diff: Maximum difference vs reference (if validated)
        num_iterations: Number of iterations run
        success: Whether the benchmark completed successfully
        error_message: Error message if benchmark failed
        gpu_name: Name of the GPU used
        cudnn_version: cuDNN version string
    """

    # Config identification
    config_name: str
    model_name: str
    backend: str
    data_type: str
    attn_mask: str

    # Dimensions
    batch_size: int
    q_seqlen: int
    kv_seqlen: int
    num_q_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_vo: int

    # Execution options
    profile_pass: str
    deterministic_bwd: bool

    # Results
    fwd_time_ms: float
    bwd_time_ms: float
    fwd_tflops: float
    bwd_tflops: float
    max_diff: float
    num_iterations: int

    # Status
    success: bool = True
    error_message: Optional[str] = None

    # Metadata
    gpu_name: Optional[str] = None
    cudnn_version: Optional[str] = None
    cudnn_backend_version: Optional[int] = None
