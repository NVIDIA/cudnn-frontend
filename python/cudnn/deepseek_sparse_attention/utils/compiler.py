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

from ..._cute_compiler import compile_options_for_target, gpu_arch_flag_for_compute_capability


@lru_cache(maxsize=None)
def gpu_arch_flag() -> str:
    """Return the ``sm_XXX`` value for the current CUDA device.

    Cached because torch.cuda.get_device_capability() is cheap but the
    function gets called inside every cute.compile site.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("cute.compile requires CUDA; no GPU available")
    major, minor = torch.cuda.get_device_capability()
    return gpu_arch_flag_for_compute_capability(major * 10 + minor)


def compile_options(extra: str = "") -> str:
    """Build the ``options=`` string for ``cute.compile``.

    Always emits ``--enable-tvm-ffi`` and a runtime-chosen ``--gpu-arch``;
    pass any kernel-specific knobs (``--opt-level 3`` etc.) via ``extra``.

    Example:
        cute.compile(..., options=compile_options("--opt-level 3"))
    """
    arch = gpu_arch_flag()
    parts = ["--enable-tvm-ffi", f"--gpu-arch {arch}"]
    if extra:
        parts.append(extra)
    return " ".join(parts)
