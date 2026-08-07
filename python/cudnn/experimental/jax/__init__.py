"""Experimental JAX bindings for cuDNN-FE CuteDSL kernels.

MVP / draft. Exposes CuteDSL kernels as real JAX primitives (compose with
@jax.jit) via an internal jax-tvm-ffi transport, behind a stable FE-owned API.
See README.md for the architecture and the open GOTCHAs.
"""

from .gemm_amax import gemm_amax

__all__ = ["gemm_amax"]
