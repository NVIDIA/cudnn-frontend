# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch integration for the cuDNN frontend Python API.

Importing this package registers the ``"CUDNN"`` provider with
``torch.nn.attention``'s flash-attention implementation registry
(PyTorch 2.13+, the same mechanism FA3/FA4 use). Registration is passive —
activation stays explicit:

    import cudnn.torch
    torch.nn.attention.activate_flash_attention_impl("CUDNN")

After activation, ``F.scaled_dot_product_attention`` under
``sdpa_kernel([SDPBackend.CUDNN_ATTENTION])`` and
``torch.nn.attention.varlen.varlen_attn`` run on the cuDNN *Python* API
(pygraph + engine Router: FROST OSS kernels or cuDNN-backend engines), with
hybrid fallback to the existing implementations for configurations the
python path does not serve yet. ``restore_flash_attention_impl()`` reverts.

On torch < 2.13 (no registry), ``cudnn.torch.install()`` applies the
``F.scaled_dot_product_attention`` overrides directly.
"""

from cudnn.torch.sdpa_provider import calls, install, served_plan_names  # noqa: F401
