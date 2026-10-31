# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Large-head dense layout admission matches the native address contract."""

from dataclasses import replace
from itertools import permutations, product
import math
from types import SimpleNamespace

import pytest
import torch
from cudnn.api_base import TensorDesc
from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm100
from cudnn.sdpa.bwd.engines import _sm100_dense_layout_mismatch
from cudnn.sdpa.bwd.prepared_sm100 import native_io_layout, native_io_strides

pytestmark = pytest.mark.L0
ROLES = ("q", "k", "v", "o", "do", "dq", "dk", "dv")


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("unused", [0, 3, -7, 2**33 + 1])
@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_singleton_tma_addresses(axis, unused, order):
    shape = [2, 3, 5, 264]
    shape[axis] = 1
    stride, span = [0, 0, 0, 1], 272
    for i in reversed(order):
        stride[i] = span
        span = shape[i] * span + 8
    stride[axis] = unused
    desc = SimpleNamespace(shape=tuple(shape), stride=tuple(stride))
    assert native_io_layout(desc)
    normalized = native_io_strides(desc)
    assert all(st > 0 and st % 8 == 0 for st in normalized[:3])
    before = {sum(i * st for i, st in zip(index, stride)) for index in product(*(range(n) for n in shape))}
    after = {sum(i * st for i, st in zip(index, normalized)) for index in product(*(range(n) for n in shape))}
    assert before == after and len(after) == math.prod(shape)


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("unaligned", [False, True])
def test_each_dense_port_has_one_admission_rule(role, unaligned, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 0))
    shape = (2, 4, 32, 264)
    desc = TensorDesc(
        dtype=torch.bfloat16, shape=shape, stride=(32 * 4 * 272, 272, 4 * 272, 1), stride_order=(3, 1, 2, 0), device=torch.device("cuda:0"), name="data"
    )
    descs = {name: replace(desc, name=name) for name in ROLES}
    if unaligned:
        descs[role] = replace(descs[role], stride=(32 * 4 * 273, 273, 4 * 273, 1))
    descs["stats"] = replace(desc, dtype=torch.float32, shape=(2, 4, 32), stride=(128, 32, 1), stride_order=(2, 1, 0), name="stats")
    # Graph facts spell the gradient ports with a capital O/Q/K/V.
    facts = SimpleNamespace(thd=False, port_layouts=tuple((n if len(n) == 1 else n[0] + n[1:].upper(), descs[n].shape, descs[n].stride) for n in ROLES))
    reason = _sm100_dense_layout_mismatch(facts, None)
    api = SdpaBwdDslSm100(**{"sample_" + n: t for n, t in descs.items()})
    if unaligned:
        assert reason and role in reason and "16-byte" in reason
        from cudnn.sdpa.bwd import api_dsl

        monkeypatch.setattr(api_dsl, "load_template", lambda *a, **kw: pytest.fail("unsupported pitches reached compilation"))
        with pytest.raises(ValueError, match=role + ".*16-byte"):
            api.compile()
    else:
        assert reason is None
        assert api.check_support()
