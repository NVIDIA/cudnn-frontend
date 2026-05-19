"""Shared cute.compile option helpers.

The cute DSL compiler accepts ``--gpu-arch <sm_XXX>`` to lock SASS to a
specific architecture. Without it, the compiler falls back to the device
arch reported by ``torch.cuda.get_device_capability()`` via the cute DSL's
internal map (see ``cutlass/base_dsl/runtime/cuda.py``). That map currently
hardcodes ``(10, 0) → "sm_100a"`` (B200) but treats unknown caps as
``"sm_<major><minor>"`` *without* the architecture-specific ``a`` suffix —
which silently drops sm_X-a-only features (TMA bulk, tcgen05, etc.) on
B300 and beyond.

So we always pass an explicit ``--gpu-arch`` chosen at runtime from the
device capability. ``compile_options(extra)`` is the single entry point;
DSA ``cute.compile`` call sites should route through it.
"""

from __future__ import annotations

from functools import lru_cache

import torch

# (compute_capability) → cute DSL --gpu-arch flag value.
# Both H100 and B200/B300 require the architecture-specific ``a`` variant
# because the indexer kernels use TMA / tcgen05 instructions that are only
# guaranteed to lower correctly under the ``a`` SASS gencode.
_ARCH_MAP = {
    (9, 0): "sm_90a",  # Hopper H100
    (10, 0): "sm_100a",  # Blackwell B200
    (10, 3): "sm_103a",  # Blackwell Ultra B300
}


@lru_cache(maxsize=None)
def gpu_arch_flag() -> str:
    """Return the ``sm_XXX`` value for the current CUDA device.

    Cached because torch.cuda.get_device_capability() is cheap but the
    function gets called inside every cute.compile site.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("cute.compile requires CUDA; no GPU available")
    cap = torch.cuda.get_device_capability()
    arch = _ARCH_MAP.get(cap)
    if arch is None:
        raise RuntimeError(
            f"Unsupported GPU compute capability {cap} for DSA CuTe kernels. " f"Add it to deepseek_sparse_attention/utils/compiler.py::_ARCH_MAP."
        )
    return arch


def compile_options(extra: str = "") -> str:
    """Build the ``options=`` string for ``cute.compile``.

    Always emits ``--enable-tvm-ffi`` and a runtime-chosen ``--gpu-arch``;
    pass any kernel-specific knobs (``--opt-level 3`` etc.) via ``extra``.

    Example:
        cute.compile(..., options=compile_options("--opt-level 3"))
    """
    parts = ["--enable-tvm-ffi", f"--gpu-arch {gpu_arch_flag()}"]
    if extra:
        parts.append(extra)
    return " ".join(parts)
