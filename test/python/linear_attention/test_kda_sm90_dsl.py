# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rule 8 behaviour of the ``kda_hopper`` sm90 CuTe DSL engine.

The numerics are covered by the ``hopper`` backend in ``test_la.py``; the shared
Rule 8 detectors (allocation, workspace sizing, capture) run over both sm90
engines in ``test_kda_sm90_cuda.py``. Here: the seed contract the DSL engine
used to get wrong -- a ``torch.zeros`` allocated once and never re-zeroed -- and
the kernel's standalone entry, which is the caller side of the workspace
contract and allocates its own scratch.
"""

import pytest
import torch

import cudnn  # noqa: F401 -- import-order requirement, see test/python/conftest.py
from cudnn.linear_attention import kimi_delta_attention

from .test_kda_sm90_cuda import fwd_graph, make_case, requires_hopper, workspace_for

pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]

ENGINE = "kda_hopper"


def _kernel():
    return pytest.importorskip("cudnn.linear_attention.hopper.kernel.kda_prefill_sm90", reason="the sm90 CuTe DSL kernel needs the cutedsl extra")


@requires_hopper
def test_absent_initial_state_is_zeroed_on_the_execution_stream():
    """Omitting initial_state must equal passing an explicit zero state, on a
    side stream, with the workspace the seed is carved from filled with 0xFF
    (fp32 NaN) between executes.

    Previously the DSL plan owned a ``torch.zeros`` seed allocated once on the
    ambient stream and never re-zeroed. Now the seed is a workspace carve that
    is memset on the launch stream every execute (R4). The NaN fill also covers
    the kernel's own scratch regions, none of which may be read before written.
    """
    _kernel()
    case = make_case(T=512, H=4, N=2)
    q, k, v, g, beta, cu, s0 = case

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        seeded, seeded_pack, seeded_o, seeded_fs = fwd_graph(ENGINE, (q, k, v, g, beta, cu, torch.zeros_like(s0)))
        seeded.execute(seeded_pack, workspace_for(seeded))
        bare, bare_pack, bare_o, bare_fs = fwd_graph(ENGINE, case, seed=False)
        ws = workspace_for(bare)
        for _ in range(2):
            ws.fill_(255)
            bare.execute(bare_pack, ws)
        real, real_pack, real_o, _ = fwd_graph(ENGINE, case)
        real.execute(real_pack, workspace_for(real))
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    torch.testing.assert_close(bare_o.float(), seeded_o.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(bare_fs, seeded_fs, atol=2e-2, rtol=2e-2)
    # The seed is visible in the first tokens' outputs (it has decayed out of
    # final_state at the production gate), so this is the row that proves the
    # comparison above can fail.
    assert not torch.allclose(real_o[:16].float(), seeded_o[:16].float()), "a non-zero seed must change the answer, or the test proves nothing"


@requires_hopper
def test_standalone_run_allocates_its_own_scratch_and_matches_the_graph():
    """``kda_prefill_sm90.run`` is the CALLER side of the workspace contract:
    it sizes the scratch with ``workspace_layout`` and allocates it per call."""
    kernel = _kernel()
    q, k, v, g, beta, cu, s0 = make_case(T=512, H=4, N=2)
    ref_o, ref_fs = kimi_delta_attention(q, k, v, g, beta, cu, initial_state=s0, output_final_state=True, plan_name=ENGINE)[:2]

    layout = kernel.workspace_layout(512, 4, 2)
    assert layout.size > 0 and layout.size % 128 == 0
    assert [name for name, *_ in layout.regions] == list(kernel._SCRATCH_ORDER)

    o = torch.empty_like(v)
    fs = torch.empty_like(s0)
    kernel.run(q, k, v, g, beta, cu, s0, o, fs)
    torch.cuda.synchronize()
    torch.testing.assert_close(o.float(), ref_o.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(fs, ref_fs, atol=2e-2, rtol=2e-2)
