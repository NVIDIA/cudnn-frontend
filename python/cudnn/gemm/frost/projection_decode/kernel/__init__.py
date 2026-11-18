# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kernel body and config of the FP8 projection for decode rows.  ``fp8_projection_decode_config`` imports without the CuTe DSL; the
``*_e4m3`` body imports it (the host entry imports the body after its version gate)."""
