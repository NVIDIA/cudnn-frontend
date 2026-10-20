# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The small CuTe-DSL kernels of the triangle-attention API layer (the attention core itself is the FROST SDPA engine row).

``prefix_lengths`` -- the one-launch derivation of a prefix mask's row lengths (``mask_is_prefix=True`` / ``prefix_mask_to_kv_lengths``).
Imported lazily by ``api.py`` on first use: ``import cudnn.triangle_attention`` pulls neither torch nor the DSL.
"""
