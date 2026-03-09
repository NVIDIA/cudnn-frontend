"""
Core types for the norm benchmark configuration system.

This module defines the dataclasses used to configure and collect results
from norm benchmarks.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class NormPreset:
    """
    Represents a named normalization configuration preset.

    Defines the norm parameters for a specific model's normalization layer.

    Attributes:
        name: Identifier for this preset (e.g., "llama3-8b")
        norm_type: Type of normalization ("rms_norm" or "layer_norm")
        N: Number of rows (batch_size * seq_len)
        C: Embedding dimension (normalization dimension)
        epsilon: Epsilon for numerical stability
        has_bias: Whether the norm has a bias term
    """

    name: str
    norm_type: str
    N: int
    C: int
    epsilon: float = 1e-5
    has_bias: bool = False


@dataclass
class NormBenchmarkConfig:
    """
    Configuration for a norm benchmark suite.

    The runner will expand this into individual benchmark cases via
    cartesian product of: norms x backends x data_types

    Attributes:
        name: Identifier for this config (used in output filenames)
        norms: List of NormPreset to benchmark
        backends: List of backend names ("cudnn", "pytorch", "torch_compile")
        data_types: List of data types (e.g., ["bfloat16"])
        profile_pass: Which pass to profile ("fwd", "bwd", or "both")
        num_iterations: Number of iterations per benchmark
        num_warmup_iterations: Warmup iterations before measurement
        output_dir: Directory for output files
    """

    name: str
    norms: List[NormPreset]
    backends: List[str] = field(default_factory=lambda: ["cudnn", "pytorch", "torch_compile"])
    data_types: List[str] = field(default_factory=lambda: ["bfloat16"])
    profile_pass: str = "both"
    num_iterations: int = 20
    num_warmup_iterations: int = 5
    output_dir: str = "../results"


@dataclass
class NormBenchmarkResult:
    """
    Result from a single norm benchmark execution.

    Contains both the configuration that was run and the measured results.
    """

    # Config identification
    config_name: str
    norm_name: str
    norm_type: str
    backend: str
    data_type: str

    # Dimensions
    N: int
    C: int
    epsilon: float
    has_bias: bool

    # Execution options
    profile_pass: str

    # Results
    fwd_time_ms: float
    bwd_time_ms: float
    fwd_gbps: float
    bwd_gbps: float
    num_iterations: int

    # Optional fields
    success: bool = True
    error_message: Optional[str] = None
    gpu_name: Optional[str] = None
    cudnn_version: Optional[str] = None
    cudnn_backend_version: Optional[int] = None
