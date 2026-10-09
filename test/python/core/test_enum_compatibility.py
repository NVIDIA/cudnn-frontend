# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integer persistence is part of the Python binding's enum contract."""

import json

import pytest

import cudnn

pytestmark = pytest.mark.L0


@pytest.mark.parametrize(
    "name",
    [
        "data_type",
        "tensor_reordering",
        "scalar_type",
        "reshape_mode",
        "knob_type",
        "norm_forward_phase",
        "heur_mode",
        "convolution_mode",
        "reduction_mode",
        "build_plan_policy",
        "numerical_note",
        "behavior_note",
        "diagonal_alignment",
        "attention_implementation",
        "moe_grouped_matmul_mode",
    ],
)
def test_enum_integer_persistence(name):
    enum_type = getattr(cudnn._pybind_module, name)
    members = list(enum_type.__members__.values())
    encoded = json.dumps([int(member) for member in members])
    restored = json.loads(encoded)

    assert restored == [member.value for member in members]
    for member, value in zip(members, restored):
        assert enum_type(value) is member
        # Explicit integer conversion must not turn enums into integer subclasses.
        assert not isinstance(member, int)
        assert {member: member.name}[enum_type(value)] == member.name
