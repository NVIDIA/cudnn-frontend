# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The output-slice recompute D512 MXFP8 implementation's identity: each 256-column V/O slice reruns QK-softmax-PV."""

NAME = "sdpa_fwd_sm100_d512_mxfp8_slice_recompute"
TEMPLATE_NAME = "prefill_d512_mxfp8"
KERNEL_FILE = "sm100/d512_mxfp8_slice_recompute/kernel.py"
