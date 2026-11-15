# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM80 K-prefetch online half identity; no runtime imports.

Serves the gptoss / llama / dsv3 flavor envelopes (eight-aligned D_QK <= 192,
D_V <= 128); ``sm80/half_flavor_contract.py`` owns the shared domain.
"""

NAME = "sdpa_fwd_sm80_f16_k_prefetch"
TEMPLATE_NAME = "prefill_f16"
KERNEL_FILE = "sm80/f16_k_prefetch/kernel.py"
LEGACY_ROW = "sdpa_fwd_prefill_sm80"
FLAVORS = ("gptoss", "llama", "dsv3")
