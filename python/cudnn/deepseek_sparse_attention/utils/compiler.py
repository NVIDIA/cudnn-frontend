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

# (compute_capability) → cute DSL --gpu-arch flag value.
# H100, B200/B300, and sm_100f require architecture-specific variants because
# the kernels use TMA / tcgen05 instructions that are only guaranteed to lower
# correctly under the matching SASS gencode.
_ARCH_MAP = {
    90: "sm_90a",  # Hopper H100
    100: "sm_100a",  # Blackwell B200
    103: "sm_103a",  # Blackwell Ultra B300
    107: "sm_100f",
}


def gpu_arch_flag_for_compute_capability(compute_capability: int) -> str:
    """Return the CuTe architecture flag for an explicit compilation target."""

    if isinstance(compute_capability, bool) or not isinstance(compute_capability, int):
        raise TypeError(f"compute_capability must be an int, got {type(compute_capability).__name__}")
    try:
        return _ARCH_MAP[compute_capability]
    except KeyError as error:
        raise RuntimeError(
            f"Unsupported GPU compute capability SM{compute_capability} for DSA CuTe kernels. "
            "Add it to deepseek_sparse_attention/utils/compiler.py::_ARCH_MAP."
        ) from error


def compile_options_for_target(compute_capability: int, extra: str = "") -> str:
    """Build CuTe compile options for a framework-supplied GPU target."""

    parts = ["--enable-tvm-ffi", f"--gpu-arch {gpu_arch_flag_for_compute_capability(compute_capability)}"]
    if extra:
        parts.append(extra)
    return " ".join(parts)


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
