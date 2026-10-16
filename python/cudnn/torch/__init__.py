# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch integration for the cuDNN frontend Python API.

On PyTorch versions with ``torch.backends.cuda.enable_cudnn_sdp_python``,
that switch serves PyTorch's cuDNN SDPA ops from this package:

    torch.backends.cuda.enable_cudnn_sdp_python(True)
    # or TORCH_CUDNN_SDPA_USE_PYTHON=1 for an unmodified script

It calls ``install_sdpa_overrides()`` and keeps the returned handle.

On PyTorch 2.13 versions without that switch, importing this package
registers the ``"CUDNN"`` provider with ``torch.nn.attention``'s
flash-attention implementation registry instead (the mechanism FA3/FA4 use):

    import cudnn.torch
    torch.nn.attention.activate_flash_attention_impl("CUDNN")

Either way, ``F.scaled_dot_product_attention`` under
``sdpa_kernel([SDPBackend.CUDNN_ATTENTION])`` and
``torch.nn.attention.varlen.varlen_attn`` then run on the cuDNN *Python* API
(pygraph + engine Router: FROST OSS kernels or cuDNN-backend engines), with
hybrid fallback to the existing implementations for configurations the
python path does not serve yet.
"""

from cudnn.torch.sdpa_provider import SdpaOverrides, calls, install, install_sdpa_overrides, served_plan_names  # noqa: F401

__all__ = ["SdpaOverrides", "calls", "install", "install_sdpa_overrides", "served_plan_names"]
