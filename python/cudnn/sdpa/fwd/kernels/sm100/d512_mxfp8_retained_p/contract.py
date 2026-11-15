# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The retained-P D512 MXFP8 implementation's identity: the second V/O slice reuses the first slice's P and alpha."""

NAME = "sdpa_fwd_sm100_d512_mxfp8_retained_p"
TEMPLATE_NAME = "prefill_d512_mxfp8_retained_p"
KERNEL_FILE = "sm100/d512_mxfp8_retained_p/kernel.py"
