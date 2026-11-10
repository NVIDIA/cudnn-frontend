# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.linear_attention.frost: the FROST linear-attention engines (GDN, KDA, GDN-2, GDP, Mamba-2) on the SM100 / SM103 / SM107 chunked kernels
built on Cutlass primitives; forward and backward on SM100 / SM103 / SM107, ranked ahead of the cuTile fallbacks."""

# The fused GDN decode / verify step entries for SM100- and SM120-class parts (``GdnDecodeFused`` / ``GdnVerifyFused``) are exported here too:
# direct-call kernels with the FlashInfer Qwen 3.5 GDN block's argument contract, not graph engines, taking the rank's head counts and
# hidden width as a ``GdnGeometry`` (``kernel.gdn_geometry.Geometry``).  Their registration as that block's ``frost_fe`` backend is
# ``flashinfer_gdn_backend``, imported explicitly (the import registers; it is never reached from ``import cudnn``).

# Lazy: importing one family's engine must not drag its neighbours in.
import importlib
from typing import Any

_LAZY_EXPORTS = {
    "GdnFrostEngine": (".gdn_engine", "GdnFrostEngine"),
    "Gdn2FrostEngine": (".gdn2_engine", "Gdn2FrostEngine"),
    "KdaFrostEngine": (".kda_engine", "KdaFrostEngine"),
    "GdpFrostEngine": (".gdp_engine", "GdpFrostEngine"),
    "GdnSummaryFrostEngine": (".gdn_engine", "GdnSummaryFrostEngine"),
    "Gdn2SummaryFrostEngine": (".gdn2_engine", "Gdn2SummaryFrostEngine"),
    "KdaSummaryFrostEngine": (".kda_engine", "KdaSummaryFrostEngine"),
    "GdpSummaryFrostEngine": (".gdp_engine", "GdpSummaryFrostEngine"),
    "Mamba2FrostEngine": (".mamba2_engine", "Mamba2FrostEngine"),
    "Mamba2SummaryFrostEngine": (".mamba2_engine", "Mamba2SummaryFrostEngine"),
    "GdnDecodeFused": (".gdn_decode_fused", "GdnDecodeFused"),
    "GdnVerifyFused": (".gdn_verify_fused", "GdnVerifyFused"),
    "GdnGeometry": (".kernel.gdn_geometry", "Geometry"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attr = target
    value = getattr(importlib.import_module(module, __name__), attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


# the lazy names, resolved by __getattr__ on first access (the pattern of cudnn.ops and cudnn.linear_attention)
__all__ = list(_LAZY_EXPORTS)
