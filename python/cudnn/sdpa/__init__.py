# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lazy exports for SDPA operation packages."""

from importlib import import_module

_SYMBOLS = {
    "SdpafwdSm100D256": (".fwd", "SdpafwdSm100D256"),
    "sdpa_fwd_wrapper_sm100_d256": (".fwd", "sdpa_fwd_wrapper_sm100_d256"),
    "SdpabwdSm100D256": (".bwd", "SdpabwdSm100D256"),
    "sdpa_bwd_wrapper_sm100_d256": (".bwd", "sdpa_bwd_wrapper_sm100_d256"),
}


def __getattr__(name):
    if name in _SYMBOLS:
        module_name, symbol_name = _SYMBOLS[name]
        value = getattr(import_module(module_name, __name__), symbol_name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value


__all__ = list(_SYMBOLS)
