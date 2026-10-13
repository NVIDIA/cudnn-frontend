# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The softmax warpgroups' top-of-tile ``mb_o_empty`` wait must have a back-edge from EVERY waiter (GitHub #1532, #1525).

``mb_o_empty[qs]`` is a single-phase barrier TMA-STG arrives once per tile per O sub-tile.  Both softmax warpgroups used to wait
slot 0 at the top of every tile, but slot 0's producer chain -- MMA ``bmm2_done[0]`` -> correction qs=0 epilogue (SM0's stats) ->
``o_full[0]`` -> TMA-STG -> ``o_empty[0]`` -- never passes through warpgroup 1.  Any delay of warpgroup 1 across a tile boundary longer
than that chain (shortest on an EMPTY (q-tile, split) unit, supplied by a spinning sibling or GPU time-slicing) let TMA-STG complete a
second phase before warpgroup 1's parity wait; the wait aliased, warpgroup 1 never published that tile's stats, and the correction
(``stat_full[1]``) and TMA-STG (``o_full[1]``) deadlocked -- a hang with correct numerics on every tile that did complete.  Observed on
cc 10.3 at ~80 % of launches on square bottom-right-causal split-KV graphs (#1532) and intermittently on the cc 10.0 paged-THD
capture/replay CI case (#1525).  Fix: each warpgroup waits ITS OWN slot, whose chain runs through its own stats publish.

Two pins.  (1) Source, no GPU: none of the eleven two-warpgroup forward kernels waits ``mb_o_empty[0]`` at the top of the tile any more.  (2) Device,
in a SUBPROCESS under a wall budget so a regression can never wedge the suite: the smallest shape that hung deterministically on a
cc 10.3 part (B=1, H=32, S_q = S_kv = 2048, D=128, bf16, bottom-right causal, TILE_CGA_M=2, SPLIT_KV=16, PACK_GQA=0) runs 20
back-to-back synchronised executes and matches an fp32 reference.
"""

import os
import re
import subprocess
import sys
import textwrap

import pytest

from frost_test_utils import requires_blackwell, requires_dsl

pytestmark = [pytest.mark.L0]

_KERNELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "python", "cudnn", "sdpa", "fwd", "kernels")
# Every forward kernel with TWO softmax warpgroups on the classic pipeline (sub_tile_id 0 / 1).  ``sm100/decode_d128_f16.py`` is
# deliberately absent: SOFTMAX_WARPGROUPS=1 there, so its single warpgroup's own publish IS slot 0's back-edge.
_TWO_WARPGROUP_KERNELS = (
    "sm100/prefill_d128_f16.py",
    "sm100/prefill_d128_fp8.py",
    "sm100/prefill_d128_mxfp8.py",
    "sm100/prefill_d192_d128_fp8.py",
    "sm100/prefill_d192_d128_mxfp8.py",
    "sm107/prefill_d128_f16.py",
    "sm107/prefill_d128_fp8.py",
    "sm107/prefill_d128_mxfp8.py",
    "sm107/prefill_d192_d128_f16.py",
    "sm107/prefill_d192_d128_fp8.py",
    "sm107/prefill_d192_d128_mxfp8.py",
)


@pytest.mark.parametrize("kernel", _TWO_WARPGROUP_KERNELS)
def test_softmax_top_of_tile_wait_is_on_the_warpgroups_own_o_slot(kernel):
    src = open(os.path.join(_KERNELS, kernel)).read()
    code = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("#"))
    # Spelling-agnostic: the wait is `bars.mb_o_empty[<slot>].wait(epilogue_state...)` in most kernels and
    # `_wait_mbarrier(bars.mb_o_empty[<slot>], epilogue_state)` in the sm100 d192x128 fp8 one; the SLOT is what matters.
    assert not re.search(
        r"mb_o_empty\[0\].*epilogue_state", code
    ), f"{kernel}: a softmax warpgroup waits slot 0 of mb_o_empty at the top of the tile (no back-edge from warpgroup 1)"
    assert len(re.findall(r"mb_o_empty\[sub_tile_id\].*epilogue_state", code)) == 1, f"{kernel}: expected exactly one per-warpgroup top-of-tile mb_o_empty wait"


_CHILD = textwrap.dedent(r"""
    import math, os, sys, time
    os.environ["CUDNN_FRONTEND_ENABLE_FROST_ENGINES"] = "1"
    import torch
    import cudnn
    from cudnn.engines.engine_ids import is_python_engine
    b, hq, s, d, split, cga, n_exec = 1, 32, 2048, 128, 16, 2, 20
    K = cudnn.knob_type
    torch.manual_seed(0)
    mk = lambda: (torch.randn(b, s, hq, d, device="cuda") * 0.5).bfloat16().transpose(1, 2)
    q, k, v = mk(), mk(), mk()
    o = torch.empty(b, s, hq, d, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    h = cudnn.create_handle()
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, handle=h)
    qt, kt, vt = (g.tensor_like(x) for x in (q, k, v))
    ot, _ = g.sdpa(q=qt, k=kt, v=vt, attn_scale=1 / math.sqrt(d), generate_stats=False, use_causal_mask_bottom_right=True)
    ot.set_output(True).set_dim(list(o.shape)).set_stride(list(o.stride())).set_data_type(cudnn.data_type.BFLOAT16)
    g.validate(); g.build_operation_graph(); g.create_execution_plans([cudnn.heur_mode.A])
    first = next(i for i in range(len(g.plans)) if is_python_engine(g.plans[i].engine_id))
    engine, knobs = g.get_engine_and_knobs_at_index(first)
    pub = dict(knobs); pub.update({K.SPLIT_KV: split, K.TILE_CGA_M: cga, K.PACK_GQA: 0, K.SCHED_POLICY: 0})
    g.create_execution_plan(engine, pub)
    idx = g.get_execution_plan_count() - 1
    g.select_plan(idx); g.check_support(); g.build_plans()
    print("plan", g.get_plan_name_at_index(idx), flush=True)
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    for i in range(n_exec):
        g.execute({qt: q, kt: k, vt: v, ot: o}, ws, handle=h)
        torch.cuda.synchronize()
        print("execute", i, "ok", flush=True)
    sc = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) / math.sqrt(d)
    sc.masked_fill_(torch.arange(s, device="cuda")[None, :] > torch.arange(s, device="cuda")[:, None], float("-inf"))
    ref = torch.einsum("bhqk,bhkd->bhqd", sc.softmax(-1), v.float())
    err = (o.float() - ref).abs().max().item()
    print("max_abs_err", err, flush=True)
    assert err < 2e-2, err
    print("OK", flush=True)
    """)


@requires_blackwell
@requires_dsl
def test_square_causal_split_kv_cga2_completes_every_launch(tmp_path):
    """The #1532 reproducer, 20 synchronised executes in a child process with a wall budget: a hang is a budget miss, never a wedge."""
    script = tmp_path / "split_kv_cga2_child.py"
    script.write_text(_CHILD)
    budget_s = 900.0  # one cold compile (~2-3 min) + 20 executes; the hang parks the child in its first synchronize for ever
    try:
        proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=budget_s, cwd=str(tmp_path))
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        pytest.fail(f"HANG: the split-KV cga2 bottom-right-causal child did not finish within {budget_s:.0f} s (the #1532 signature).\n{out[-3000:]}")
    assert proc.returncode == 0, f"child failed rc={proc.returncode}\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    assert proc.stdout.count(" ok") == 20 and "OK" in proc.stdout, proc.stdout[-2000:]
