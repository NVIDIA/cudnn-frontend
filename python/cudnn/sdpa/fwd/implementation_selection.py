# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Pure body-selection policy shared by recommendation and placement.

Keep this layer independent of the candidate generator and registry. Explicit
implementation hosts consume their fixed contracts, not these selectors.
"""


def select_sm107_d512_geometry(facts, knobs):
    """Resolve the native wide-half body before compile, preserving the old default."""
    from .kernels.sm107.d512_half_contract import ROLE_SPLIT, TWO_BY_TWO, explicit_geometry

    pack = bool(knobs.pack_gqa)
    split = knobs.split_kv or 1
    group = facts.h_q // facts.h_kv if pack else 1
    two_by_two = facts.d_qk == 576 or (split == 1 and not facts.has_paged_kv and 64 % group == 0)
    # Native dense packed, paged, split and mixed-KV plans already require 2x2.
    two_by_two = two_by_two or not facts.uniform_dtype or (not facts.thd and (facts.has_paged_kv or pack or split > 1))
    return explicit_geometry(TWO_BY_TWO if two_by_two else ROLE_SPLIT, facts)
