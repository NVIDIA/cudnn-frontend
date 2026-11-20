# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""MXFP8 forward geometry: the tile constants and the host geometry check; CPU-only.

The peer count is a runtime value from the workspace; shapes and routing are not static.
"""

CLUSTER_SIZE = 2
THREADS_PER_CTA = 256
MLP_TILE_ROWS = 256
MLP_TILE_COLUMNS = 256
SWIGLU_TILE_ROWS = 128
SWIGLU_TILE_COLUMNS = 128

# Forward task kinds; the megakernels pass ROUTED_SWIGLU for the backward's replayed routed SwiGLU.
SHARED_GATE, SHARED_UP, SHARED_SWIGLU, SHARED_DOWN = range(4)
ROUTED_GATE, ROUTED_UP, ROUTED_SWIGLU, ROUTED_DOWN = range(4, 8)


def validate_geometry(
    *, hidden_size, intermediate_size, num_local_experts, topk, macrobatch_size, minibatch_size, num_comm_sms, intermediate_alignment=256, hidden_alignment=256
):
    """Raise ``ValueError`` for a geometry the kernels do not admit.  ``intermediate_alignment`` /
    ``hidden_alignment``: 256 by default; the MXFP8 forward passes 128, a 128-wide intermediate and hidden
    tail."""
    if type(hidden_size) is not int or hidden_size < 256 or hidden_size % hidden_alignment:
        raise ValueError(f"hidden_size must be positive and divisible by {hidden_alignment}" + (" (and at least 256)" if hidden_alignment < 256 else ""))
    if type(minibatch_size) is not int or minibatch_size <= 0 or minibatch_size % 256:
        raise ValueError("minibatch_size must be positive and divisible by 256")
    if type(intermediate_size) is not int or intermediate_size < 256 or intermediate_size % intermediate_alignment:
        raise ValueError(
            f"intermediate_size must be positive and divisible by {intermediate_alignment}" + (" (and at least 256)" if intermediate_alignment < 256 else "")
        )
    if type(num_local_experts) is not int or num_local_experts <= 0:
        raise ValueError("num_local_experts must be positive")
    if type(topk) is not int or not 0 < topk <= 255:
        raise ValueError("topk must be an integer in [1, 255]")
    if type(macrobatch_size) is not int or macrobatch_size <= 0 or macrobatch_size % minibatch_size:
        raise ValueError("macrobatch_size must be a positive multiple of minibatch_size")
    if type(num_comm_sms) is not int or num_comm_sms <= 0 or num_comm_sms % 2:
        raise ValueError("num_comm_sms must be a positive even integer")
