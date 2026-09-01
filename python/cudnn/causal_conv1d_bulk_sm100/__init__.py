# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private native backends for :func:`cudnn.ops.causal_conv1d`.

The lifecycle symbols remain available as explicit implementation and test
seams, but loading one backend must not import the others. They are deliberately
absent from the package's public export list; model code should use
``cudnn.ops``.
"""

from importlib import import_module

_LAZY_BACKEND_ATTRIBUTES = {
    "CausalConv1dBulkFwdSm100": ".api",
    "causal_conv1d_bulk_fwd_wrapper_sm100": ".api",
    "CausalConv1dBulkAutogradPrototype": ".autograd",
    "CausalConv1dBulkBwdPrototype": ".backward",
    "compile_causal_conv1d_bulk_bwd_prototype": ".backward",
}


def __getattr__(name: str):
    module_name = _LAZY_BACKEND_ATTRIBUTES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__: list[str] = []
