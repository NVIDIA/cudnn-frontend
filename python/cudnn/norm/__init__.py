"""cudnn.norm: normalization operations backed by CUTLASS CuTe-DSL kernels.

The ``frost`` backend implements five norm variants (LayerNorm, RMSNorm,
GroupNorm, BatchNorm, InstanceNorm) with both fprop and bprop, for bf16 / fp16 /
fp32 I/O. See :mod:`cudnn.norm.frost` for the public API.
"""

from typing import Any

_LAZY_EXPORTS = {
    "NormVariant": ("cudnn.norm.frost", "NormVariant"),
    "norm_forward": ("cudnn.norm.frost", "norm_forward"),
    "norm_backward": ("cudnn.norm.frost", "norm_backward"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = list(_LAZY_EXPORTS)
