# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM80 packed metadata binds native integer/float storage and wide strides."""

import pytest
import torch

from frost_test_utils import _SM, requires_dsl
from sdpa.frost.test_sdpa_sm80_thd_forward_prepared import _check, _inputs, _prefix, _run

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 80, reason="requires native SM80")]


@pytest.mark.parametrize("d,dv", [(64, 64), (128, 128), (224, 224), (256, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("prefix_dtype", [torch.int32, torch.int64])
def test_packed_metadata_native_types_strides_and_replay(d, dv, dtype, prefix_dtype, monkeypatch):
    import cutlass.cute as cute

    inputs = _inputs(d, dv, dtype, capq=96, capkv=128)
    lq, lk = (17, 0, 23), (19, 0, 29)
    cq = _prefix(lq).to(prefix_dtype)
    ck = _prefix(lk).to(torch.int64 if prefix_dtype == torch.int32 else torch.int32)
    sink = torch.tensor([-0.7, 0.4, 1.1, -0.2], dtype=dtype, device="cuda")
    features = dict(causal=True, bottom=True, window=16, sink=sink)
    _check(inputs, _run(inputs, cq, ck, maxq=32, **features), lq, lk, **features)
    # Geometry is runtime metadata: different prefix/sink strides must reuse
    # the same immutable artifact, including after a zero-stride broadcast.
    for pitch in (2, 5):
        cq = _prefix(lq).to(prefix_dtype).repeat_interleave(pitch)[::pitch]
        ck = _prefix(lk).to(ck.dtype).repeat_interleave(pitch + 1)[:: pitch + 1]
        sink = sink.repeat_interleave(pitch)[::pitch]
        features["sink"] = sink
        original_to, original_contiguous = torch.Tensor.to, torch.Tensor.contiguous
        operands = (cq, ck, sink)

        def to(tensor, *args, **kwargs):
            assert all(tensor is not value for value in operands), "packed wrapper converted metadata"
            return original_to(tensor, *args, **kwargs)

        def contiguous(tensor, *args, **kwargs):
            assert all(tensor is not value for value in operands), "packed wrapper copied metadata"
            return original_contiguous(tensor, *args, **kwargs)

        with monkeypatch.context() as guard:
            guard.setattr(torch.Tensor, "to", to)
            guard.setattr(torch.Tensor, "contiguous", contiguous)
            guard.setattr(cute, "compile", lambda *a, **k: pytest.fail("metadata stride change recompiled"))
            result = _run(inputs, cq, ck, maxq=32, **features)
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    captured = _run(inputs, cq, ck, maxq=32, **features)
            except BaseException:
                graph.reset()
                raise
        _check(inputs, result, lq, lk, **features)
        try:
            lq, lk = (13, 4, 23), (17, 2, 29)
            cq.copy_(_prefix(lq))
            ck.copy_(_prefix(lk))
            sink.add_(0.5)
            inputs[0].mul_(0.75)
            for tensor in captured.values():
                tensor.fill_(float("nan"))
            graph.replay()
            _check(inputs, captured, lq, lk, **features)
        finally:
            graph.reset()

    features["sink"] = torch.tensor([0.5], dtype=dtype, device="cuda").expand(4)
    with monkeypatch.context() as guard:
        guard.setattr(cute, "compile", lambda *a, **k: pytest.fail("broadcast metadata recompiled"))
        result = _run(inputs, cq, ck, maxq=32, **features)
        empty = _run(inputs, cq.new_zeros(1).expand(4), ck.new_zeros(1).expand(4), maxq=32, **features)
    _check(inputs, result, lq, lk, **features)
    _check(inputs, empty, (0, 0, 0), (0, 0, 0), **features)


@pytest.mark.parametrize("d", [64, 128, 224, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_packed_backward_reuses_token_major_stats(d, dtype, monkeypatch):
    from cudnn.sdpa.bwd import api_dsl
    from sdpa.frost.test_sdpa_bwd_thd_sm80 import _thd_case
    from sdpa.frost.test_sdpa_bwd_prepared_thd_sm80 import _check as check_backward
    from sdpa.frost.test_sdpa_sm80_thd_wrapper_prepared import _wrapper
    import cutlass.cute as cute

    def case(lengths_q, lengths_kv, seed):
        result = _thd_case(lengths_q, lengths_kv, 4, d, dtype, hkv=2, cap_q=512, cap_kv=512, poison=True, causal=True, seed=seed)
        result.lse = result.lse.transpose(1, 2).contiguous().transpose(1, 2)
        return result

    first, fresh = case((96, 160), (128, 96), 29), case((128, 64), (64, 128), 43)
    cq, ck = _prefix(first.lens_q), _prefix(first.lens_kv)
    kwargs = dict(deterministic=True, max_s_q=160, max_s_kv=128)
    plans, original_plan = [], api_dsl._sm80_thd_plan

    def plan(*args, **kw):
        result = original_plan(*args, **kw)
        plans.append(result)
        return result

    monkeypatch.setattr(api_dsl, "_sm80_thd_plan", plan)
    original_to, original_contiguous = torch.Tensor.to, torch.Tensor.contiguous

    def to(tensor, *args, **kw):
        assert tensor is not first.lse, "packed backward converted native Stats"
        return original_to(tensor, *args, **kw)

    def contiguous(tensor, *args, **kw):
        assert tensor is not first.lse, "packed backward copied token-major Stats"
        return original_contiguous(tensor, *args, **kw)

    graph = torch.cuda.CUDAGraph()
    try:
        with monkeypatch.context() as guard:
            guard.setattr(torch.Tensor, "to", to)
            guard.setattr(torch.Tensor, "contiguous", contiguous)
            initial = _wrapper(first, cq, ck, **kwargs)
            guard.setattr(cute, "compile", lambda *a, **kw: pytest.fail("Stats replay recompiled"))
            with torch.cuda.graph(graph):
                captured = _wrapper(first, cq, ck, **kwargs)
        assert plans and all(p.thd_stats_token_major for p in plans)
        check_backward(first, *(initial[n] for n in ("dq_tensor", "dk_tensor", "dv_tensor")))
        for name in ("q", "k", "v", "o", "do", "lse"):
            getattr(first, name).copy_(getattr(fresh, name))
        cq.copy_(_prefix(fresh.lens_q))
        ck.copy_(_prefix(fresh.lens_kv))
        for tensor in captured.values():
            tensor.fill_(float("nan"))
        graph.replay()
        check_backward(fresh, *(captured[n] for n in ("dq_tensor", "dk_tensor", "dv_tensor")))
        for key, live in (("dq_tensor", fresh.t_q), ("dk_tensor", fresh.t_kv), ("dv_tensor", fresh.t_kv)):
            assert torch.count_nonzero(captured[key][:, live:]) == 0
    finally:
        graph.reset()


@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("d", [64, 256])
@pytest.mark.parametrize("role", ["q_prefix", "kv_prefix", "sink"])
@pytest.mark.parametrize("product", [False, True])
def test_packed_metadata_physical_wide_addresses(d, role, product):
    import gc

    lengths = (1, 1, 1, 1) if product else (2,)
    inputs = _inputs(d, d, torch.float16, capq=8, capkv=8)
    cq, ck = _prefix(lengths).to(torch.int64), _prefix(lengths)
    # Keep the softmax contribution large enough for a wrong prefix to fail
    # the numerical tolerances; large sinks would hide a missing KV sequence.
    sink_values = [7.0, 8.0, 9.0, 10.0] if role == "sink" else [-0.7, 0.4, 1.1, -0.2]
    compact_sink = torch.tensor(sink_values, dtype=torch.float16, device="cuda")
    source = {"q_prefix": cq, "kv_prefix": ck, "sink": compact_sink}[role]
    # Both the stride itself and a product of individually Int32-sized factors
    # must cross 2**32. Wrapped offsets remain allocated, with wrong safe data.
    stride = (2**31 - 16 if role == "sink" else 2**30 + 8) if product else 2**32 + 16
    elements = (source.numel() - 1) * stride + 64
    # Previous wide tests can leave reusable allocations in Torch's cache;
    # mem_get_info counts that memory as busy until the cache is released.
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info()[0] < elements * source.element_size() + 2**30:
        pytest.skip("physical wide metadata stride needs more free GPU memory")
    try:
        owner = torch.empty(elements, dtype=source.dtype, device=source.device)
    except torch.OutOfMemoryError:
        pytest.skip("insufficient memory for physical wide metadata stride")
    for index in range(source.numel()):
        owner[(index * stride) % 2**32] = 0
    owner[-16:].fill_(43)
    view = owner.as_strided(source.shape, (stride,))
    view.copy_(source)
    sink = compact_sink
    if role == "q_prefix":
        cq = view
    elif role == "kv_prefix":
        ck = view
    else:
        sink = view
    result = _run(inputs, cq, ck, maxq=2, sink=sink)
    _check(inputs, result, lengths, lengths, sink=compact_sink)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = _run(inputs, cq, ck, maxq=2, sink=sink)
        inputs[0].mul_(0.5)
        if role == "sink":
            sink.add_(0.25)
            compact_sink.add_(0.25)
        for tensor in captured.values():
            tensor.fill_(float("nan"))
        graph.replay()
        _check(inputs, captured, lengths, lengths, sink=compact_sink)
    finally:
        graph.reset()
    assert (owner[-16:] == 43).all()


def test_packed_metadata_fresh_process_artifact_reload(tmp_path):
    if _SM != 80:
        pytest.skip("fresh-process packed metadata reload requires native SM80")
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import cudnn

    child = r"""
import json, sys
from pathlib import Path
import torch, cudnn, pytest
import cutlass.cute as cute
from cudnn.frost import compiled_cache
folder, package, reload = sys.argv[1:]
assert Path(cudnn.__file__).resolve() == Path(package).resolve()
sys.path[:0] = [folder, str(Path(folder) / "sdpa/frost")]
from sdpa.frost.test_sdpa_sm80_packed_metadata import (
    test_packed_metadata_native_types_strides_and_replay,
    test_packed_backward_reuses_token_major_stats,
)
with pytest.MonkeyPatch.context() as patch:
    if reload == "1":
        patch.setattr(cute, "compile", lambda *a, **k: pytest.fail("artifact reload invoked JIT"))
    for d in (64, 256):
        test_packed_metadata_native_types_strides_and_replay(d, d, torch.float16, torch.int64, patch)
        test_packed_backward_reuses_token_major_stats(d, torch.float16, patch)
print(json.dumps(compiled_cache.stats()))
"""
    env = dict(os.environ, CUDNN_FRONTEND_COMPILED_CACHE=str(tmp_path))
    env.pop("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", None)
    results = []
    for reload in (0, 1):
        result = subprocess.run(
            [sys.executable, "-c", child, str(Path(__file__).parents[2]), cudnn.__file__, str(reload)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-5000:]
        results.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert results[0]["misses"] > 0 and results[0]["hits"] == 0
    assert results[1]["misses"] == 0 and results[1]["hits"] > 0
