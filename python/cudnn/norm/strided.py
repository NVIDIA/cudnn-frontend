# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile-time descriptors for strided normalization vectors."""


def fake_strided_vector(dtype, *, rows):
    """A 1D FFI input with an independent runtime stride, including broadcast."""
    import cutlass.cute as cute

    return cute.runtime.make_fake_tensor(dtype, (rows,), (cute.sym_int(64),))
