# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The 397B-geometry cell of the UNFUSED block backward -- L2 (``pytest -m L2``): ``d_model 4096``, 32 / 2 heads
(g = 16), B=1, S=4096, causal, norm, the main bound of ``test_block_backward.py`` (its helpers, its tolerances).

Its own module because a level is a MODULE mark here: under ``test_block_backward.py``'s module-level L0 the cell was
selected by ``-m L0`` too and gated on an environment variable instead -- the L-levels of ``pytest.ini`` are the repo's
selector, so the L2 selection runs it and the L0 smoke selection never sees it.  ~4 GB of block workspace plus the adapter's
dS chunk (one 16-head chunk of 4K x 4K bf16 = 512 MiB at S=4K); the fp64 oracle is the slow part.
"""

import os
import sys

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_block_backward import _GEOM_397B, _backward, _check_all_grads, requires_rubin  # noqa: E402


@requires_rubin
def test_397b_geometry():
    """The 397B geometry at B=1, S=4096, causal, norm: every gradient against fp64 autograd at the module's bounds
    (magnitudes printed)."""
    res = _backward({**_GEOM_397B}, batch=1, seq_len=4096, memo=False)
    worst = _check_all_grads(res)
    assert res.grads["dw_q_norm"].dtype == torch.float32
    print(f"397B worst cells: {worst}")
