# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Both SM90 templates, the D512 body and the dedicated D128, must use the prepared pointer launch everywhere."""

import pytest
import torch

from frost_test_utils import _SM, requires_dsl
from test_utils import torch_fork_set_rng
from test_sdpa_fwd_dsl_sm90 import sm100  # noqa: F401 -- strict SM90 graph-route fixture
from test_sdpa_fwd_dsl_sm100 import _bhsd, _ref_sdpa_full

pytestmark = [pytest.mark.L0, pytest.mark.skipif(_SM != 90, reason="requires SM90"), requires_dsl]


@pytest.mark.parametrize("thd, d", [pytest.param(False, 512, id="dense"), pytest.param(True, 512, id="thd"), pytest.param(False, 128, id="dense-d128")])
def test_graph_never_enters_tensor_adapter(sm100, monkeypatch, thd, d):
    """Prove both graph layouts use the prepared executor instead of the tensor adapter, on D512 and on the D128
    template the row proposes first at exact (128, 128) (dense only: THD there keeps the 64 x 64 tile, the D512
    body). Each case loads its own template, and only that one."""
    from cudnn.sdpa.fwd import api_dsl
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    def forbidden(*args, **kwargs):
        raise AssertionError("graph execution returned to the tensor adapter")

    monkeypatch.setattr(SdpaFwdDslSm90, "execute", forbidden)
    loaded, load = [], api_dsl._load_kernel_template
    monkeypatch.setattr(api_dsl, "_load_kernel_template", lambda filename, *a, **kw: loaded.append(filename) or load(filename, *a, **kw))
    if thd:
        sm100._run_thd_stats_case(seq_lens_q=[130, 0, 64, 7], seq_lens_kv=[130, 5, 200, 7], d=d, H_q=4, H_kv=2, mask="causal_br", cu_lens=True)
    else:
        sm100.test_sdpa_fwd_dsl_sm100_graph_api(torch.float16, True, d)
    assert loaded and set(loaded) == {f"sm90/prefill_d{d}_f16.py"}, loaded


# THD at D=128 runs the D512 template (config_sm90.NOT_SERVED), so only the dense arm adds D128.
_PREPARED_CASES = [pytest.param(False, 512, id="dense"), pytest.param(True, 512, id="thd"), pytest.param(False, 128, id="dense-d128")]


@pytest.mark.parametrize("thd, d", _PREPARED_CASES)
@torch_fork_set_rng(seed=0)
def test_prepared_rebind_capture_and_no_tensor_launch(monkeypatch, thd, d):
    """Reuse one artifact for fresh buffers and changed-input replay without tensor conversion."""
    import cutlass.cute as cute
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    b, h, sq, skv = 2, 2, 70, 100
    q, k, v, o = (_bhsd(b, h, s, d, torch.float16) for s in (sq, skv, skv, sq))
    api = SdpaFwdDslSm90(q, k, v, o, thd=thd)
    api.compile()
    assert api.flavor == (d, d), api.flavor
    spec = api._thd_spec if thd else api._dense_spec
    assert spec is not None
    assert all("tensor" not in slot for slot in spec.order)
    q_lens = torch.tensor([sq, sq], dtype=torch.int32, device="cuda")
    kv_lens = torch.tensor([skv, skv], dtype=torch.int32, device="cuda")
    workspace = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda") if thd else None
    kwargs = dict(seq_q_lens=q_lens, seq_kv_lens=kv_lens, workspace=workspace) if thd else {}
    run = lambda q, k, v, o: api.execute(q, k, v, o, **kwargs)
    run(q, k, v, o)
    fresh = tuple(torch.randn_like(t) for t in (q, k, v))
    fresh_o = torch.full_like(o, float("nan"))
    ref = _ref_sdpa_full(*fresh, scale=d**-0.5)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()

    def forbidden(*args, **kwargs):
        raise AssertionError("warm prepared execution called the tensor/JIT path")

    try:
        with monkeypatch.context() as guard:
            guard.setattr(cute, "compile", forbidden)
            guard.setattr(cute.runtime, "from_dlpack", forbidden)
            guard.setattr(api, "_compiled_kernel", forbidden)
            for name in ("view", "as_strided", "transpose", "contiguous"):
                guard.setattr(torch.Tensor, name, forbidden)
            torch.cuda.set_sync_debug_mode("error")
            try:
                allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
                for _ in range(3):
                    run(*fresh, fresh_o)
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            finally:
                torch.cuda.set_sync_debug_mode("default")
            with torch.cuda.graph(graph, stream=stream):
                run(*fresh, fresh_o)
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(fresh_o, ref, atol=5e-2, rtol=3e-2)
        # Replay must read the current bytes at the captured pointers.
        fresh[2].mul_(0.5)
        fresh_o.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(fresh_o, ref * 0.5, atol=5e-2, rtol=3e-2)
    finally:
        graph.reset()


# Causal persistent D128 records the route keeps on an SM90 GPU (more items than SMs), each on the NATURAL order
# check_support gives a record of the walk's rule: two NATURAL strides below CAUSAL_WALK_MIN_TILES Q tiles (b4 h16/4 S
# 512, T 4, 256 items; bottom-right b17 h32/4 S_q 384 S_kv 2048, T 3, GQA ratio 8) and a paired walk (b2 h16/4 S 1152,
# T 9, 288 items: the walk at every SM count from 32 to 132).
_PREPARED_CAUSAL_CASES = [
    pytest.param((4, 16, 4, 512, 512), False, False, id="causal-natural-stride"),
    pytest.param((17, 32, 4, 384, 2048), True, False, id="causal-br-stride"),
    pytest.param((2, 16, 4, 1152, 1152), False, True, id="causal-walk"),
]


@pytest.mark.parametrize("shape, bottom_right, walks", _PREPARED_CAUSAL_CASES)
@torch_fork_set_rng(seed=0)
def test_prepared_d128_causal_persistent_rebind_capture_no_sync_no_alloc(monkeypatch, shape, bottom_right, walks):
    """The causal persistent D128 records on the prepared launch: warm executes with fresh buffers never re-enter the
    tensor/JIT path, never synchronize and never allocate (Rules 1, 3 and 8 of python/cudnn/AGENTS.md), and a CUDA-graph
    replay reads the current bytes at the captured pointers, as test_prepared_rebind_capture_and_no_tensor_launch checks
    for the dense record. Each record walks the NATURAL order, the strides and the walk as _PREPARED_CAUSAL_CASES
    says."""
    import cutlass.cute as cute
    from cudnn.frost.tile_dsl.constants import SCHED_NATURAL
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    b, h_q, h_kv, sq, skv = shape
    q, o = (_bhsd(b, h_q, sq, 128, torch.float16) for _ in range(2))
    k, v = (_bhsd(b, h_kv, skv, 128, torch.float16) for _ in range(2))
    api = SdpaFwdDslSm90(q, k, v, o, is_causal=True, causal_bottom_right=bottom_right)
    api.compile()
    assert api.flavor == (128, 128), api.flavor
    assert (api._k_mod.PERSISTENT, api._k_mod.CAUSAL_WALK) == (True, 1), (shape, api._k_mod.PERSISTENT, api._k_mod.CAUSAL_WALK)
    p = api.params
    assert p.sched_policy == SCHED_NATURAL, p
    rule = api._k_mod.causal_walk_plan(
        shape, p.causal if p.bottom_right is None else p.bottom_right, p.sched_policy, p.qh_per_kh, api._persistent_ctas(q.device), 2
    )
    assert (rule is not None) is walks, (shape, rule)
    assert api._dense_spec is not None and all("tensor" not in slot for slot in api._dense_spec.order)
    run = lambda q, k, v, o: api.execute(q, k, v, o)
    run(q, k, v, o)
    fresh = tuple(torch.randn_like(t) for t in (q, k, v))
    fresh_o = torch.full_like(o, float("nan"))
    ref = _ref_sdpa_full(*fresh, scale=128**-0.5, is_causal=True, bottom_right=bottom_right)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()

    def forbidden(*args, **kwargs):
        raise AssertionError("warm prepared execution called the tensor/JIT path")

    try:
        with monkeypatch.context() as guard:
            guard.setattr(cute, "compile", forbidden)
            guard.setattr(cute.runtime, "from_dlpack", forbidden)
            guard.setattr(api, "_compiled_kernel", forbidden)
            for name in ("view", "as_strided", "transpose", "contiguous"):
                guard.setattr(torch.Tensor, name, forbidden)
            torch.cuda.set_sync_debug_mode("error")
            try:
                allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
                for _ in range(3):
                    run(*fresh, fresh_o)
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            finally:
                torch.cuda.set_sync_debug_mode("default")
            with torch.cuda.graph(graph, stream=stream):
                run(*fresh, fresh_o)
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(fresh_o, ref, atol=5e-2, rtol=3e-2)
        # Replay must read the current bytes at the captured pointers.
        fresh[2].mul_(0.5)
        fresh_o.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(fresh_o, ref * 0.5, atol=5e-2, rtol=3e-2)
    finally:
        graph.reset()


@torch_fork_set_rng(seed=0)
def test_prepared_d128_claim_counter_rebind_capture_no_sync_no_alloc(monkeypatch):
    """A claim-counter record (api_dsl.D128_CAUSAL_CLAIM 1 past a gate patched to 0) on the prepared launch: warm executes
    with fresh buffers and a caller workspace never re-enter the tensor/JIT path, never synchronize (the native binder's
    counter reset is an async memset on the launch stream) and never allocate (Rules 1, 3 and 8 of python/cudnn/AGENTS.md),
    and a CUDA-graph replay reads the current bytes at the captured pointers and zeroes the counter again each time, after
    the workspace is refilled with 0x7F (b4 h16/4 S 1152 causal: 576 items, L 5, the walk's record the route keeps)."""
    import cutlass.cute as cute
    from cudnn.sdpa.fwd import api_dsl
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    monkeypatch.setattr(api_dsl, "D128_CAUSAL_CLAIM", 1)
    monkeypatch.setattr(api_dsl, "D128_CLAIM_MIN_ROUNDS", 0)
    b, h_q, h_kv, sq, skv = 4, 16, 4, 1152, 1152
    q, o = (_bhsd(b, h_q, sq, 128, torch.float16) for _ in range(2))
    k, v = (_bhsd(b, h_kv, skv, 128, torch.float16) for _ in range(2))
    api = SdpaFwdDslSm90(q, k, v, o, is_causal=True)
    api.compile()
    assert (api._k_mod.PERSISTENT, api._k_mod.CAUSAL_WALK, api._k_mod.CAUSAL_CLAIM) == (True, 0, 1), api.params
    assert api.scratch_workspace_bytes() == 128 and api._dense_spec.claim_counter
    assert api._dense_spec is not None and all("tensor" not in slot for slot in api._dense_spec.order)
    workspace = torch.full((256,), 0x7F, dtype=torch.uint8, device="cuda")
    items = 9 * h_q * b
    run = lambda q, k, v, o: api.execute(q, k, v, o, workspace=workspace)
    run(q, k, v, o)
    fresh = tuple(torch.randn_like(t) for t in (q, k, v))
    fresh_o = torch.full_like(o, float("nan"))
    ref = _ref_sdpa_full(*fresh, scale=128**-0.5, is_causal=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()

    def forbidden(*args, **kwargs):
        raise AssertionError("warm prepared execution called the tensor/JIT path")

    try:
        with monkeypatch.context() as guard:
            guard.setattr(cute, "compile", forbidden)
            guard.setattr(cute.runtime, "from_dlpack", forbidden)
            guard.setattr(api, "_compiled_kernel", forbidden)
            for name in ("view", "as_strided", "transpose", "contiguous"):
                guard.setattr(torch.Tensor, name, forbidden)
            torch.cuda.set_sync_debug_mode("error")
            try:
                allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
                for _ in range(3):
                    run(*fresh, fresh_o)
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            finally:
                torch.cuda.set_sync_debug_mode("default")
            with torch.cuda.graph(graph, stream=stream):
                run(*fresh, fresh_o)
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(fresh_o, ref, atol=5e-2, rtol=3e-2)
        assert int(workspace[:4].view(torch.int32).item()) == items
        # Replay must read the current bytes at the captured pointers and zero the counter again.
        fresh[2].mul_(0.5)
        fresh_o.fill_(float("nan"))
        workspace.fill_(0x7F)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(fresh_o, ref * 0.5, atol=5e-2, rtol=3e-2)
        assert int(workspace[:4].view(torch.int32).item()) == items
    finally:
        graph.reset()


def test_prepared_thd_rejects_smaller_batch_and_misaligned_workspace():
    """Reject runtime metadata that violates the fixed Hopper scheduler or tensor-map ABI."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    q, k, v, o = (_bhsd(2, 2, 64, 512, torch.float16) for _ in range(4))
    api = SdpaFwdDslSm90(q, k, v, o, thd=True)
    api.compile()
    lens = torch.full((2,), 64, dtype=torch.int32, device="cuda")
    size = api.scratch_workspace_bytes()
    workspace = torch.empty(size + 128, dtype=torch.uint8, device="cuda")
    # SM90 embeds the batch in scheduler coordinates and THD map offsets.
    with pytest.raises(ValueError, match="exactly 2 sequences"):
        api.execute(q, k, v, o, seq_q_lens=lens[:1], seq_kv_lens=lens[:1], workspace=workspace)
    # 16-byte pointer alignment alone is insufficient for Hopper tensor maps.
    with pytest.raises(ValueError, match="128-byte aligned"):
        api.execute(q, k, v, o, seq_q_lens=lens, seq_kv_lens=lens, workspace=workspace[16 : 16 + size])


@pytest.mark.parametrize("thd", [False, True], ids=["dense", "thd"])
def test_prepared_artifact_reloads_in_fresh_process(thd, tmp_path, splits=1, pack_gqa=False, d=512):
    """Verify exported artifacts launch and replay after a new interpreter forbids JIT."""
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import cudnn

    child = r"""
import hashlib, json, os, sys
from pathlib import Path
package, thd, reload, splits, pack_gqa, d = sys.argv[1:]
sys.meta_path[:] = [f for f in sys.meta_path if "__editable__" not in getattr(f, "__module__", "")]
sys.path.insert(0, str(Path(package).resolve().parent.parent))
if os.environ.get("FE_TEST_CUDA_BINDINGS"):
    sys.path.insert(0, os.environ["FE_TEST_CUDA_BINDINGS"])
import torch, cudnn
import cutlass.cute as cute
from cudnn.frost import compiled_cache
from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90
assert Path(cudnn.__file__).resolve() == Path(package).resolve(), cudnn.__file__
if reload == "1":
    def forbidden(*args, **kwargs):
        raise AssertionError("fresh-process SM90 artifact invoked JIT")
    cute.compile = forbidden
torch.manual_seed(0)
q, k, v, o = (torch.randn(2, 64, 2, int(d), device="cuda", dtype=torch.float16).transpose(1, 2) for _ in range(4))
api = SdpaFwdDslSm90(q, k, v, o, thd=thd == "1", split_kv=int(splits), pack_gqa=pack_gqa == "1")
api.compile()
assert api.flavor == (int(d), int(d)), api.flavor
if reload == "1":
    assert hasattr(api._compiled_kernel, "_compiled_cache_raw")
lens = torch.full((2,), 64, device="cuda", dtype=torch.int32)
ws = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
kw = dict(seq_q_lens=lens, seq_kv_lens=lens, workspace=ws) if thd == "1" else dict(workspace=ws)
api.execute(q, k, v, o, **kw)
reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)
torch.testing.assert_close(o, reference, atol=5e-2, rtol=3e-2)
graph = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(graph):
        api.execute(q, k, v, o, **kw)
    o.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(o, reference, atol=5e-2, rtol=3e-2)
    digest = hashlib.sha256(o.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
finally:
    graph.reset()
print(json.dumps(dict(digest=digest, stats=compiled_cache.stats())))
"""
    env = dict(os.environ, CUDNN_FRONTEND_COMPILED_CACHE=str(tmp_path))
    env.pop("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", None)
    results = []
    for reload in (0, 1):
        result = subprocess.run(
            [sys.executable, "-c", child, cudnn.__file__, str(int(thd)), str(reload), str(splits), str(int(pack_gqa)), str(d)],
            env=env,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-5000:]
        results.append(json.loads(result.stdout.strip().splitlines()[-1]))
    first, second = results
    assert first["stats"]["misses"] > 0 and first["stats"]["hits"] == 0, first
    assert second["stats"]["misses"] == 0 and second["stats"]["hits"] > 0, second
    assert first["digest"] == second["digest"]


def test_prepared_d128_artifact_reloads_in_fresh_process(tmp_path):
    """The D128 template's artifact reloads and replays without JIT, as D512's does."""
    test_prepared_artifact_reloads_in_fresh_process(False, tmp_path, d=128)


@pytest.mark.parametrize("thd", [False, True], ids=["dense", "thd"])
@torch_fork_set_rng(seed=0)
def test_prepared_output_stride_above_int32(thd):
    """Write two distant live rows; a narrowed stride writes the poisoned low island."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    q = _bhsd(1, 1, 2, 512, torch.float16)
    k, v = (_bhsd(1, 1, 64, 512, torch.float16) for _ in range(2))
    stride = 2**31 + 1024
    if torch.cuda.mem_get_info()[0] < (stride + 512) * 2 + 1024**3:
        pytest.skip("wide-stride probe needs 5 GiB free")
    storage = torch.empty(stride + 512, device="cuda", dtype=torch.float16)
    o = storage.as_strided((1, 1, 2, 512), (512, 512, stride, 1))
    storage[:1536].fill_(float("nan"))
    o.fill_(float("nan"))
    api = SdpaFwdDslSm90(q, k, v, o, thd=thd)
    api.compile()
    kwargs = {}
    if thd:
        kwargs = dict(
            seq_q_lens=torch.tensor([2], dtype=torch.int32, device="cuda"),
            seq_kv_lens=torch.tensor([64], dtype=torch.int32, device="cuda"),
            workspace=torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda"),
        )
    api.execute(q, k, v, o, **kwargs)
    torch.testing.assert_close(o, _ref_sdpa_full(q, k, v, scale=512**-0.5), atol=5e-2, rtol=3e-2)
    assert torch.isnan(storage[512:1536]).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("layout", ["bhs", "packed_th1", "packed_th", "flat"])
@pytest.mark.parametrize("sq", [1, 70])
@torch_fork_set_rng(seed=0)
def test_standalone_thd_token_major_stats(monkeypatch, dtype, layout, sq):
    """The declared BHS Stats and packed storage bind the same bytes without tensor conversions."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    b, h, skv, d = 2, 2, 100, 512
    q, k, v, o = (_bhsd(b, h, s, d, dtype) for s in (sq, skv, skv, sq))
    stats_storage = torch.full((b, sq, h), float("nan"), dtype=torch.float32, device="cuda")
    stats = stats_storage.transpose(1, 2)
    api = SdpaFwdDslSm90(q, k, v, o, sample_lse=stats, thd=True)
    api.compile()
    lse = {
        "bhs": stats,
        "packed_th1": stats_storage.view(-1, h, 1),
        "packed_th": stats_storage.view(-1, h),
        "flat": stats_storage.view(-1),
    }[layout]
    q_lens = torch.full((b,), sq, dtype=torch.int32, device="cuda")
    kv_lens = torch.full((b,), skv, dtype=torch.int32, device="cuda")
    workspace = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")
    ref_o, ref_lse = _ref_sdpa_full(q, k, v, scale=d**-0.5, return_stats=True)

    def forbidden(*args, **kwargs):
        """Reject execute-time tensor conversion even when it aliases the same storage."""
        raise AssertionError("Stats binding must normalize metadata only")

    with monkeypatch.context() as guard:
        for name in ("view", "as_strided", "unsqueeze", "transpose", "contiguous"):
            guard.setattr(torch.Tensor, name, forbidden)
        api.execute(q, k, v, o, lse_tensor=lse, seq_q_lens=q_lens, seq_kv_lens=kv_lens, workspace=workspace)
    torch.testing.assert_close(o, ref_o, atol=5e-2, rtol=3e-2)
    torch.testing.assert_close(stats, ref_lse, atol=2e-2, rtol=2e-2)
