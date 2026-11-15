# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One device body, with fixed SM100 and SM107 graph contracts."""

SM100 = "sdpa_bwd_sm100_d256"
SM107_D192 = "sdpa_bwd_sm107_d192"
SM107_D256 = "sdpa_bwd_sm107_d256_f16_2x2"
KERNEL_FILE = "sm100/d256_f16_2x2/kernel.py"
SM100_TEMPLATE_TAG = "sdpa_bwd_sm100_d256_main"
SM107_TEMPLATE_TAG = "sdpa_bwd_sm107_main_2x2"
