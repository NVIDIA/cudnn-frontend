# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Admission boundaries for the measured persistent schedule."""

import importlib.util
from pathlib import Path
import pytest

pytestmark = pytest.mark.L0


def _schedule(*args):
    path = Path(__file__).resolve().parents[4] / "python/cudnn/causal_conv1d_se/schedule.py"
    spec = importlib.util.spec_from_file_location("se_schedule_unit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.backward_schedule(*args)


@pytest.mark.parametrize(
    "tokens,channels,expected",
    [
        (0, 2048, ("tiled", 1)),
        (1, 2048, ("tiled", 1)),
        (8191, 2048, ("tiled", 64)),
        (8192, 2048, ("persistent", 37)),
        (8192, 768, ("persistent", 98)),
        (8192, 1032, ("tiled", 64)),
        (4095, 256, ("tiled", 64)),
        (4096, 256, ("tiled", 64)),
        (5155, 256, ("tiled", 81)),
        (8191, 256, ("tiled", 128)),
        (8192, 256, ("tiled", 64)),
        (64, 256, ("tiled", 1)),
        (2062, 1032, ("tiled", 33)),
        (2**31, 2048, ("tiled", 64)),
    ],
)
@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (10, 7)])
def test_measured_dispatch_boundaries(tokens, channels, expected, capability):
    assert _schedule(tokens, channels, 4, "torch.bfloat16", True, True, capability, 148) == expected


@pytest.mark.parametrize(
    "width,dtype,silu,native,capability",
    [
        (3, "torch.bfloat16", True, True, (10, 3)),
        (4, "torch.float32", True, True, (10, 3)),
        (4, "torch.float16", True, True, (10, 3)),
        (4, "torch.bfloat16", False, True, (10, 3)),
        (4, "torch.bfloat16", True, False, (10, 3)),
        (4, "torch.bfloat16", True, True, (9, 0)),
    ],
)
def test_preserve_unqualified_fallbacks(width, dtype, silu, native, capability):
    assert _schedule(131072, 2048, width, dtype, silu, native, capability, 148)[0] == "tiled"
