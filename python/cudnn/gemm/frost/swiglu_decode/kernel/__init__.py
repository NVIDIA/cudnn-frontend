# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kernel body and config of the block-scaled FP4 gate / up product with SwiGLU for decode rows.  ``swiglu_fp4_decode_config`` imports
without the CuTe DSL; the ``*_nvfp4`` body imports it (the host entry imports the body after its version gate)."""
