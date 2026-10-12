# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Bulk metadata transport preserves storage observations independently of geometry."""

import pytest

from cudnn.sdpa.fwd import prepared as prep

pytestmark = [pytest.mark.L0]


@pytest.mark.parametrize("dtype,width", [("bfloat16", 2), ("float32", 4), ("float8_e4m3fn", 1), ("uint8", 1), ("int64", 8), ("", 1)])
@pytest.mark.parametrize("span", [-1, 0, 2**32 + 7])
def test_bulk_facts_preserve_wide_and_unknown_storage(dtype, width, span):
    role = "input"
    fact = prep.BufferFacts(2**40, dtype, (2, 3), span, (2, 3), (2**32, 1))
    facts = {role: fact, "absent": None}
    pack = prep._native_pack_from_facts(facts, (role, "absent", "missing"))
    assert pack.pointer(0) == fact.ptr
    assert pack.observed_bytes(0) == (-1 if span < 0 else span * width)
    assert pack._facts_as((0,), prep.BufferFacts, prep._DTYPE_BY_CODE) == [fact]
    assert not pack.is_filled(1) and not pack.is_filled(2)
    facts[role] = fact._replace(ptr=2**41, device=(2, 1))
    rebound = prep._native_pack_from_facts(facts, (role,))
    assert rebound.pointer(0) == 2**41
    assert pack.pointer(0) == 2**40
    assert pack._facts_as((0,), prep.BufferFacts, prep._DTYPE_BY_CODE)[0].device == (2, 3)


def test_bulk_facts_reject_observed_byte_overflow():
    fact = prep.BufferFacts(4096, "float32", (2, 0), 2**62, (1,), (1,))
    with pytest.raises(ValueError, match="fit in int64"):
        prep._native_pack_from_facts({"q": fact}, ("q",))
