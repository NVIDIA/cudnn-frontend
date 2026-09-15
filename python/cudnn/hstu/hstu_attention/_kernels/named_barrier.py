# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named-barrier IDs used by the HSTU forward pipeline."""

# ID 0 remains available for block synchronization. The epilogue uses this
# base plus its stage index, reserving IDs 1 and 2.
EPILOGUE_BARRIER_BASE = 1
TMEM_POINTER_BARRIER = 3
TMEM_RELEASE_BARRIER = 4
