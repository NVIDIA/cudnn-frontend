# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-execute binding work of the FROST norm graph engines: the variant-pack tables are built once per lowered plan,
and an execution stream that is already torch's current one is not entered again.

    pytest test/python/norm/graph/test_frost_norm_binding_cost.py
"""

from contextlib import nullcontext

import pytest
import torch

pytest.importorskip("cudnn.norm", reason="frost norm engines require a built cudnn frontend")

from cudnn.norm import graph_analyzer as ga  # noqa: E402
from cudnn.norm.utils import on_stream  # noqa: E402

pytestmark = pytest.mark.L0


class _T:
    def __init__(self, uid, name):
        self.uid, self.name, self.queries = uid, name, 0

    def get_uid(self):
        self.queries += 1
        return self.uid

    def get_name(self):
        self.queries += 1
        return self.name


def test_variant_pack_tables_are_built_once():
    x, y, dup_a, dup_b = _T(1, "x"), _T(2, "y"), _T(3, "same"), _T(4, "same")
    binding = ga.NormBinding({"x": x, "y": y, "a": dup_a, "b": dup_b, "unused": None})
    first = ga.resolve_variant_pack({x: "bx", 2: "by", "same": "ambiguous", 3: "ba"}, binding)
    assert first == {id(x): "bx", id(y): "by", id(dup_a): "ba"}  # a shared name identifies neither tensor
    queried = [t.queries for t in (x, y, dup_a, dup_b)]
    second = ga.resolve_variant_pack({"x": "bx2", 4: "bb"}, binding)
    assert second == {id(x): "bx2", id(dup_b): "bb"}
    assert [t.queries for t in (x, y, dup_a, dup_b)] == queried, "the tables were rebuilt on a later execute"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_current_stream_is_not_entered_again():
    current = torch.cuda.current_stream()
    assert isinstance(on_stream(current.cuda_stream), nullcontext)
    side = torch.cuda.Stream()
    with on_stream(side.cuda_stream):
        assert torch.cuda.current_stream().cuda_stream == side.cuda_stream
    with torch.cuda.stream(side):
        with on_stream(0):  # an explicit request for the default stream still switches to it
            assert torch.cuda.current_stream() == torch.cuda.default_stream()
