# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM80 D256 K+V-prefetch half identity; no runtime imports.

Serves the qwen flavor envelope (eight-aligned head dims up to 256 that the
K-prefetch envelopes do not cover); ``sm80/half_flavor_contract.py`` owns the
shared domain.
"""

NAME = "sdpa_fwd_sm80_d256_kv_prefetch"
TEMPLATE_NAME = "prefill_d256_f16"
KERNEL_FILE = "sm80/d256_kv_prefetch/kernel.py"
LEGACY_ROW = "sdpa_fwd_prefill_sm80"
FLAVORS = ("qwen",)
