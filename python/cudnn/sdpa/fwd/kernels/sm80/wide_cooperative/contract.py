# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM80 wide-head cooperative half identity; no runtime imports.

Serves the wide512 / mla576 flavor envelopes (eight-aligned head dims past
256, D_QK up to 576 and D_V up to 512); ``sm80/half_flavor_contract.py`` owns
the shared domain.
"""

NAME = "sdpa_fwd_sm80_wide_cooperative"
TEMPLATE_NAME = "prefill_wide_f16"
KERNEL_FILE = "sm80/wide_cooperative/kernel.py"
LEGACY_ROW = "sdpa_fwd_prefill_sm80"
FLAVORS = ("wide512", "mla576")
