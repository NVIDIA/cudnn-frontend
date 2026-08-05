# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The public ``cudnn.pygraph`` surface must be POSITIONALLY identical to the
classic API (callers pass name/handle/stride/... by position — pycudnnTest
does exactly that for the constructor). New parameters must be keyword-only so
they can never shift the classic order.

The classic order is read from the artifacts themselves (the pybind
constructor docstring and the classic patched ``tensor`` wrapper), not
hard-coded, so a future binding change fails here rather than in an
integration suite.
"""

import inspect

import pytest

import cudnn
from cudnn._pygraph import pygraph

pytestmark = pytest.mark.L0


def _skip_if_patched():
    """These tests introspect the pristine class. Repos that layer engines by
    monkey-patching cudnn.pygraph (e.g. an internal cudnn.TBD import replaces
    __init__/tensor/lifecycle methods process-wide) make signature
    introspection meaningless — skip loudly instead of failing on the
    wrapper's (*args, **kwargs) signature."""
    for name in ("__init__", "tensor", "create_execution_plans"):
        fn = getattr(pygraph, name)
        if "pygraph" not in getattr(fn, "__qualname__", ""):
            pytest.skip(f"cudnn.pygraph.{name} is monkey-patched ({getattr(fn, '__qualname__', '?')}); parity introspection requires the pristine class")


def _pybind_positional_params(doc: str):
    """Parse parameter names, in order, from a pybind11 signature docstring."""
    sig_line = next(line for line in doc.splitlines() if "(" in line)
    inner = sig_line[sig_line.index("(") + 1 : sig_line.rindex(")")]
    parts, depth, cur = [], 0, ""
    for ch in inner:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    names = []
    for p in parts:
        p = p.strip()
        if not p or p.startswith("self") or p.startswith("*"):
            continue
        names.append(p.split(":")[0].split("=")[0].strip())
    return names


def test_constructor_positional_parity():
    _skip_if_patched()
    classic = _pybind_positional_params(cudnn._pybind_module.backend_graph.__init__.__doc__)
    params = list(inspect.signature(pygraph.__init__).parameters.values())[1:]  # drop self
    positional = [p.name for p in params if p.kind == p.POSITIONAL_OR_KEYWORD]
    assert positional[: len(classic)] == classic, f"classic constructor order not preserved:\n classic={classic}\n ours    ={positional}"
    # everything new is keyword-only — it can never shift the classic order
    keyword_only = {p.name for p in params if p.kind == p.KEYWORD_ONLY}
    assert {"backends", "router"} <= keyword_only


def test_tensor_positional_parity():
    _skip_if_patched()
    # the classic public tensor() is the python wrapper patched onto the
    # pybind class — introspectable directly
    classic_fn = cudnn._pybind_module.backend_graph.tensor
    classic = [p.name for p in inspect.signature(classic_fn).parameters.values()][1:]  # drop self
    params = list(inspect.signature(pygraph.tensor).parameters.values())[1:]
    ours = [p.name for p in params if p.kind == p.POSITIONAL_OR_KEYWORD]
    assert ours[: len(classic)] == classic, f"classic tensor() order not preserved:\n classic={classic}\n ours    ={ours}"


def test_constructor_accepts_classic_positional_call():
    _skip_if_patched()
    """The exact pycudnnTest call shape: name positionally, rest by keyword."""
    g = pygraph("my_graph", io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    assert g._cpp_graph_kwargs["name"] == "my_graph"
    assert g.context.io_data_type == cudnn.data_type.HALF


def test_tensor_accepts_classic_positional_call():
    _skip_if_patched()
    """Full classic positional form: (dim, stride, data_type, is_virtual,
    is_pass_by_value, ragged_offset, reordering_type, name, uid, multiplier)."""
    g = pygraph()
    ro = g.tensor([2, 2], [2, 1], cudnn.data_type.INT32, False, False, None, None, "ragged", 7, 1)
    t = g.tensor([4, 4], [4, 1], cudnn.data_type.HALF, False, True, ro, cudnn.tensor_reordering.NONE, "classic", 9, 2)
    assert t.name == "classic" and t.uid == 9 and t.is_pass_by_value
    assert t.ragged_offset is ro and t.ragged_offset_multiplier == 2
    assert t.reordering_type is None  # classic NONE sentinel normalizes to unset
    # classic unset sentinels
    u = g.tensor([2, 2], None, cudnn.data_type.NOT_SET, False, False, None, None, "unset", -1)
    assert u.uid > 0 and not u.uid_assigned  # -1 == auto
    assert u.data_type is None or getattr(u.data_type, "name", "") != "NOT_SET"
