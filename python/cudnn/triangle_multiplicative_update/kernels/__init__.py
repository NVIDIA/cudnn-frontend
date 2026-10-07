# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernels of the triangle multiplicative update: ``contraction`` (K2, the batched GEMM contract on the FROST GEMM engine);
``trimul_proj`` (K1) and ``trimul_tail`` (K3) follow."""
