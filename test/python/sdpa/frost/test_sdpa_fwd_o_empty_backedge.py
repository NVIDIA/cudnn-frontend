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
A separate D512 MXFP8 probe delays softmax after both Dv slices have their P input: its once-per-tile gate must observe
a once-per-tile completion, rather than alias two O-empty phases. It checks dense and THD output, Stats and Amax after replay.
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
    # One own-slot wait per tile body: the sm107 d128 kernel carries a second body (the resident-bias pair-step tile) whose
    # top-of-step wait follows the same slot discipline; every other kernel has one body.
    waits = len(re.findall(r"mb_o_empty\[sub_tile_id\].*epilogue_state", code))
    assert waits == (
        2 if kernel == "sm107/prefill_d128_f16.py" else 1
    ), f"{kernel}: expected one per-warpgroup top-of-tile mb_o_empty wait per tile body, found {waits}"


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


_D512_SWA_CHILD = textwrap.dedent(r"""
    import hashlib, json, os, sys
    from pathlib import Path
    test_root, expected_fe, layout, work = sys.argv[1:]
    sys.path.insert(0, test_root)
    import conftest  # Allocator and Transformer Engine import ordering.
    sys.path.insert(0, str(Path(test_root) / "sdpa/frost"))
    import torch
    import cudnn
    from cudnn.sdpa.fwd import api_dsl
    from frost_test_utils import process_watchdog
    import test_sdpa_fwd_mxfp8_sm100 as ref

    assert Path(cudnn.__file__).resolve() == Path(expected_fe).resolve()
    original_load = api_dsl.load_template
    served = []

    def delayed(path, params, tag="template", **kwargs):
        p = Path(path)
        if p.parent.name == "sm100" and p.name == "prefill_d512_mxfp8.py":
            source = p.read_text()
            begin = source.index("def _softmax_warp_group(")
            end = source.index("\n@cute.jit", begin)
            section = source[begin:end]
            anchor = "        wait(mb_decoded.subview(sched_state.idx), sched_state.phase)\n"
            assert section.count(anchor) == 1
            # P is already published for BOTH Dv slices. Delay the softmax role
            # here so their two O-empty phases finish before its next tile wait.
            delay = (
                "        if cutlass.const_expr(_SWA_REUSE_P):\n"
                "            delayed_start = cute.arch.globaltimer()\n"
                "            while cute.arch.globaltimer() - delayed_start < cutlass.Int64(10_000_000):\n"
                "                pass\n"
            )
            target = Path(work) / "template_d512_delayed_softmax.py"
            target.write_text(source[:begin] + section.replace(anchor, delay + anchor) + source[end:])
            module = original_load(str(target), params, tag, **kwargs)
            assert module._SWA_REUSE_P, "the two-output-slice path was not exercised"
            if params.thd_varlen:
                # This kernel's THD claim loop normally gets the full envelope
                # grid. Enable the existing test launch cap on this private
                # module so FROST_THD_CLUSTERS=1 really reuses the same CTA.
                module.THD_PERSISTENT = True
            served.append(dict(sha256=hashlib.sha256(source.encode()).hexdigest(), params=repr(params)))
            return module
        return original_load(path, params, tag, **kwargs)

    api_dsl.load_template = delayed
    original_execute = cudnn.pygraph.execute

    def replayed(self, *args, **kwargs):
        # Compilation happens before execute; the watchdog covers device work.
        with process_watchdog(45.0, "D512 delayed softmax output-drain observer"):
            original_execute(self, *args, **kwargs)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                original_execute(self, *args, **kwargs)
            for _ in range(4):
                graph.replay()
                torch.cuda.synchronize()

    cudnn.pygraph.execute = replayed
    torch.manual_seed(1541)
    if layout == "dense":
        result = ref._run(1, 64, 1, 512, "e4m3", torch.float16, scale=512**-0.5,
                          sdpa_kwargs=dict(use_causal_mask=True, diagonal_band_left_bound=128),
                          d_qk=512, d_v=512, return_lse=True)
        output, expected, amax, lse, expected_lse = result
    else:
        os.environ["FROST_THD_CLUSTERS"] = "1"
        # Include keyless work as well as repeated live work on the same CTA.
        output, expected, amax, lse, expected_lse = ref._run_thd(
            [512, 256], [512, 0], 4, 1, "e4m3", torch.float16,
            scale=512**-0.5, causal=True, swa_window=127,
            stats=True, d_qk=512, d_v=512, with_lse_ref=True)
    assert served, "the delayed D512 template did not serve the graph"
    assert torch.isfinite(output).all()
    ref._check(output, expected, torch.float16, "e4m3", d_qk=512)
    # MXFP8 P quantization also perturbs the denominator; use the existing
    # MXFP8 Stats contract for both layouts, independent of observer timing.
    torch.testing.assert_close(lse.float(), expected_lse.float(), atol=ref._THD_STATS_ATOL, rtol=ref._THD_STATS_RTOL)
    assert abs(amax.item() - expected.abs().max().item()) <= 0.03
    print("D512_DELAYED_SOFTMAX_PASS", json.dumps(served), flush=True)
    """)


@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("layout", ["dense", "thd"])
@requires_blackwell
@requires_dsl
def test_d512_swa_waits_for_both_output_slices(tmp_path, layout):
    """A delayed once-per-tile observer must not alias the twice-per-tile O-empty phase."""
    from pathlib import Path

    import torch
    import cudnn

    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("the two-slice SWA=128 reuse path belongs to SM100 D512 MXFP8")
    script = tmp_path / "d512_delayed_softmax.py"
    script.write_text(_D512_SWA_CHILD)
    env = dict(os.environ, CUDNN_FRONTEND_ENABLE_FROST_ENGINES="1", CUDNN_FRONTEND_COMPILED_CACHE=str(tmp_path / "compiled"))
    proc = subprocess.run(
        [sys.executable, str(script), str(Path(__file__).resolve().parents[2]), cudnn.__file__, layout, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    (tmp_path / "delayed_softmax.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    assert proc.returncode == 0 and "D512_DELAYED_SOFTMAX_PASS" in proc.stdout, f"rc={proc.returncode}\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"


def test_scheduler_and_credit_target_select_the_same_layouts():
    """A local scheduler cannot wait for credits routed only to the CLC leader.

    Inspect the layout guards, independently of warp-role dispatch and source
    formatting. This is a protocol invariant, not a heuristic or timing pin.
    """
    import ast
    from pathlib import Path

    def layouts(node, parents):
        allowed = {False, True}
        while node in parents:
            parent = parents[node]
            if isinstance(parent, ast.If):
                test = parent.test
                if isinstance(test, ast.Call) and ast.unparse(test.func) == "cutlass.const_expr":
                    test = test.args[0]
                negate = isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)
                value = test.operand if negate else test
                if ast.unparse(value) in ("CFG.THD_VARLEN", "_THD"):
                    allowed &= {(node in parent.body) != negate}
            node = parent
        return allowed

    scanned = []
    for path in (Path(_KERNELS).parents[1]).rglob("*.py"):
        tree = ast.parse(path.read_text())
        aliases = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and any(a.name == "read_tile_id_arrive_on_leader" for a in n.names)]
        clc = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "scheduler_warp_loop"]
        if not (aliases or clc):
            continue
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        if not aliases:
            # A literal single-CTA caller has no peers: both helpers elect
            # one local arrival. It may also use the persistent THD scheduler.
            credits = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "read_tile_id_arrive"]
            assert credits and all(len(n.args) >= 2 and isinstance(n.args[1], ast.Constant) and n.args[1].value == 1 for n in credits), str(path)
            assert all(len(n.args) >= 4 and isinstance(n.args[3], ast.Constant) and n.args[3].value == 1 for n in clc), str(path)
            scanned.append(path)
            continue
        assert len(aliases) == len(clc) == 1, str(path)
        assert layouts(aliases[0], parents) == layouts(clc[0], parents), f"{path}: scheduler/credit layout mismatch"
        # The default import must still supply broadcasts on the other arm.
        if layouts(aliases[0], parents) != {False, True}:
            broadcasts = [
                n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and any(a.name == "read_tile_id_arrive" and a.asname is None for a in n.names)
            ]
            assert any(layouts(n, parents) == {False, True} for n in broadcasts), f"{path}: missing default broadcast credits"
        scanned.append(path)
    assert scanned, "no common CLC callers were checked"


@pytest.mark.parametrize("cga_size", [1, 2, 4])
@pytest.mark.parametrize("predicated", [False, True])
def test_leader_credit_honors_the_lowering_choice(cga_size, predicated):
    """Exercise every lane of the helper, including its constexpr branch choice.

    Plain-Python evaluation checks routing and the chosen DSL primitive; the
    existing whole-kernel SASS tests separately cover compiler lowering.
    """
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    path = Path(_KERNELS).parents[2] / "frost/tile_dsl/scheduler.py"
    functions = [
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name in ("_read_tile_id_lane_geometry", "read_tile_id_arrive_on_leader")
    ]
    for fn in functions:
        fn.decorator_list = []
        for arg in fn.args.args:
            arg.annotation = None
    arrived = []
    for lane in range(32):
        nvvm = SimpleNamespace(
            elect_sync=lambda: lane == 0,
            mapa=lambda mb, peer: peer,
            MemScope=SimpleNamespace(CTA="cta"),
            mbarrier_arrive=lambda mb, **kw: arrived.append(("branch", mb)),
            inline_ptx=lambda text, read_only_args, predicate: arrived.append(("predicated", read_only_args[0])) if predicate else None,
        )
        namespace = dict(
            cutlass=SimpleNamespace(Int32=int, const_expr=bool), nvvm=nvvm, cute=SimpleNamespace(arch=SimpleNamespace(thread_idx=lambda: (lane, 0, 0)))
        )
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
        namespace["read_tile_id_arrive_on_leader"](0, cga_size, predicated)
    assert arrived == [("predicated" if predicated and cga_size > 1 else "branch", 0)]
