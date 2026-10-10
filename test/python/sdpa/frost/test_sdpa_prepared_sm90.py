# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM90's late-added D512 row must use the prepared pointer launch everywhere."""

import pytest
import torch

from frost_test_utils import _SM, requires_dsl
from test_utils import torch_fork_set_rng
from test_sdpa_fwd_dsl_sm90 import sm100  # noqa: F401 -- strict SM90 graph-route fixture
from test_sdpa_fwd_dsl_sm100 import _bhsd, _ref_sdpa_full

pytestmark = [pytest.mark.L0, pytest.mark.skipif(_SM != 90, reason="requires SM90"), requires_dsl]


@pytest.mark.parametrize("thd", [False, True], ids=["dense", "thd"])
def test_graph_never_enters_tensor_adapter(sm100, monkeypatch, thd):
    """Prove both graph layouts use the prepared executor instead of the tensor adapter."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    def forbidden(*args, **kwargs):
        raise AssertionError("graph execution returned to the tensor adapter")

    monkeypatch.setattr(SdpaFwdDslSm90, "execute", forbidden)
    if thd:
        sm100._run_thd_stats_case(seq_lens_q=[130, 0, 64, 7], seq_lens_kv=[130, 5, 200, 7], d=512, H_q=4, H_kv=2, mask="causal_br", cu_lens=True)
    else:
        sm100.test_sdpa_fwd_dsl_sm100_graph_api(torch.float16, True, 512)


@pytest.mark.parametrize("thd", [False, True], ids=["dense", "thd"])
@torch_fork_set_rng(seed=0)
def test_prepared_rebind_capture_and_no_tensor_launch(monkeypatch, thd):
    """Reuse one artifact for fresh buffers and changed-input replay without tensor conversion."""
    import cutlass.cute as cute
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    b, h, sq, skv, d = 2, 2, 70, 100, 512
    q, k, v, o = (_bhsd(b, h, s, d, torch.float16) for s in (sq, skv, skv, sq))
    api = SdpaFwdDslSm90(q, k, v, o, thd=thd)
    api.compile()
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
def test_prepared_artifact_reloads_in_fresh_process(thd, tmp_path):
    """Verify exported artifacts launch and replay after a new interpreter forbids JIT."""
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import cudnn

    child = r"""
import hashlib, json, sys
from pathlib import Path
import torch, cudnn
import cutlass.cute as cute
from cudnn.frost import compiled_cache
from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90
package, thd, reload = sys.argv[1:]
assert Path(cudnn.__file__).resolve() == Path(package).resolve(), cudnn.__file__
if reload == "1":
    def forbidden(*args, **kwargs):
        raise AssertionError("fresh-process SM90 artifact invoked JIT")
    cute.compile = forbidden
torch.manual_seed(0)
q, k, v, o = (torch.randn(2, 64, 2, 512, device="cuda", dtype=torch.float16).transpose(1, 2) for _ in range(4))
api = SdpaFwdDslSm90(q, k, v, o, thd=thd == "1")
api.compile()
if reload == "1":
    assert hasattr(api._compiled_kernel, "_compiled_cache_raw")
lens = torch.full((2,), 64, device="cuda", dtype=torch.int32)
ws = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
kw = dict(seq_q_lens=lens, seq_kv_lens=lens, workspace=ws) if thd == "1" else {}
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
            [sys.executable, "-c", child, cudnn.__file__, str(int(thd)), str(reload)],
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
