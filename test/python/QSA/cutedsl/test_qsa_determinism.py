# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bitwise repeatability of the QSA forward and backward on the public API and the indexed API.

Every route is run several times on the same inputs and every output and gradient must match the first run
exactly. The two-pass causal top-k backward has its own repeat test in test_qsa_two_pass_backward.py.
"""

import pytest
import torch

from cudnn.ops import qsa_backward, qsa_forward, qsa_indexed_backward, qsa_indexed_forward
from QSA.cutedsl.test_qsa import make_inputs, supported_gpu  # noqa: F401
from QSA.cutedsl.test_qsa_indexed_training import indices_for

REPEATS = 6

ROUTES = [
    pytest.param((3, 5, 9), id="packed-small"),
    pytest.param((61, 68), id="packed"),
    pytest.param((513,), id="gated-dense"),
    pytest.param((2056,), id="gathered"),
]


def assert_repeats(name_values):
    """``name_values()`` returns ``{name: tensor}``; every repeat must equal the first bitwise."""
    first = {name: value.clone() for name, value in name_values().items()}
    for repeat in range(1, REPEATS):
        for name, value in name_values().items():
            assert torch.equal(first[name], value), f"{name} differs bitwise on repeat {repeat}: {int((first[name] != value).sum())} elements"


@pytest.mark.L0
@pytest.mark.parametrize("heads", [(12, 1), (24, 2), (6, 1)])
@pytest.mark.parametrize("lengths", ROUTES)
def test_forward_backward_repeat_bitwise(supported_gpu, heads, lengths):
    inputs = make_inputs(lengths, *heads, empty=True)
    grad_out = inputs["q"].flip(0).contiguous()

    def run():
        out = qsa_forward(**inputs)
        grads = qsa_backward(grad_out, **inputs, out_fp32=out["out_fp32"], lse=out["lse"])
        return {**out, **grads}

    assert_repeats(run)


@pytest.mark.L0
@pytest.mark.parametrize("heads", [(12, 1), (6, 1)])
@pytest.mark.parametrize("lengths", [pytest.param((3, 5, 7), id="packed-small"), pytest.param((129, 133), id="packed"), pytest.param((2049,), id="long")])
def test_indexed_forward_backward_repeat_bitwise(supported_gpu, heads, lengths):
    inputs = make_inputs(lengths, *heads, empty=True, all_selected=True)
    indices = indices_for(inputs)
    grad = inputs["q"].clone()

    def run():
        out = qsa_indexed_forward(inputs["q"], inputs["k"], inputs["v"], indices, inputs["scale"])
        grads = qsa_indexed_backward(grad, inputs["q"], inputs["k"], inputs["v"], out["out_fp32"], out["lse"], indices, inputs["scale"])
        return {**out, **grads}

    assert_repeats(run)
