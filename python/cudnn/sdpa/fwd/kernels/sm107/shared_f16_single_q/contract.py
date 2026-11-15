# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fixed architecture contract; kernel/config remain owned by SM100."""

from ..shared_half_contract import SINGLE_Q as NAME, explicit_geometry, mismatch as _mismatch


def mismatch(facts, knobs):
    return _mismatch(NAME, facts, knobs)
