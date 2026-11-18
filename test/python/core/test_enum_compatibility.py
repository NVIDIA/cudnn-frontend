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


def test_behavior_note_frontend_band_preserves_existing_values():
    old = ("RUNTIME_COMPILATION", "REQUIRES_FILTER_INT8x32_REORDER", "REQUIRES_BIAS_INT8x32_REORDER", "SUPPORTS_CUDA_GRAPH_NATIVE_API", "CUBLASLT_DEPENDENCY")
    assert [int(getattr(cudnn.behavior_note, name)) for name in old] == list(range(1, 6))
    assert int(cudnn.behavior_note.SUPPORTS_SDPA_STATS_STRIDE_OVERRIDE) == 1000
