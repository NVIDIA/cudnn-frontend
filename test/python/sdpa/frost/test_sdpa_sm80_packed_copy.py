# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared copies for existing standalone packed SM80 normalization."""

import pytest
import torch

from frost_test_utils import _SM, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]
requires_native_sm80 = pytest.mark.skipif(_SM != 80, reason="requires native SM80")


@requires_native_sm80
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("backward", [False, True])
def test_packed_wrapper_uses_prepared_half_copies(dtype, backward, monkeypatch):
    if backward:
        from cudnn.sdpa.bwd import api_dsl
        from test_sdpa_sm80_thd_wrapper_prepared import test_wrapper_rebind_capture_and_capacity_tails as check

        name = "_sm80_thd_backward"
    else:
        from cudnn.sdpa.fwd import api_dsl
        from test_sdpa_sm80_thd_forward_prepared import test_thd_wrapper_rebind_and_capture as check

        name = "_sm80_thd_forward"
    original = getattr(api_dsl, name)
    contiguous, cat = torch.Tensor.contiguous, torch.cat

    def run(*args, **kwargs):
        def forbid_contiguous(tensor, *a, **kw):
            if tensor.ndim == 4 and tensor.dtype in (torch.float16, torch.bfloat16):
                pytest.fail("packed wrapper rebuilt a half data copy")
            return contiguous(tensor, *a, **kw)

        def forbid_cat(tensors, *a, **kw):
            if any(t.ndim == 4 and t.dtype in (torch.float16, torch.bfloat16) for t in tensors):
                pytest.fail("packed wrapper rebuilt torch padding")
            return cat(tensors, *a, **kw)

        with monkeypatch.context() as guard:
            guard.setattr(torch.Tensor, "contiguous", forbid_contiguous)
            guard.setattr(torch, "cat", forbid_cat)
            return original(*args, **kwargs)

    monkeypatch.setattr(api_dsl, name, run)
    if backward:
        check(96, 96, dtype, monkeypatch)
    else:
        check(96, 96, dtype, True, monkeypatch)


@requires_native_sm80
@pytest.mark.parametrize("backward", [False, True])
def test_packed_copy_artifacts_reload_in_fresh_process(backward, tmp_path):
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
from cudnn.sdpa import packed_copy_sm80
folder, package, backward, reload = sys.argv[1:]
assert Path(cudnn.__file__).resolve() == Path(package).resolve()
sys.path[:0] = [folder, str(Path(folder).parents[1])]
from test_sdpa_sm80_packed_copy import test_packed_wrapper_uses_prepared_half_copies
original = packed_copy_sm80._plan
artifacts = []
def record(shapes):
    result = original(shapes)
    artifacts.append(result[0])
    return result
with pytest.MonkeyPatch.context() as patch:
    patch.setattr(packed_copy_sm80, "_plan", record)
    if reload == "1":
        patch.setattr(cute, "compile", lambda *a, **k: pytest.fail("packed copy fresh process invoked JIT"))
    test_packed_wrapper_uses_prepared_half_copies(torch.bfloat16, backward == "1", patch)
assert artifacts
if reload == "1":
    assert all(hasattr(a, "_compiled_cache_raw") for a in artifacts)
print(json.dumps(compiled_cache.stats()))
"""
    env = dict(os.environ, CUDNN_FRONTEND_COMPILED_CACHE=str(tmp_path))
    env.pop("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", None)
    results = []
    for reload in (0, 1):
        result = subprocess.run(
            [sys.executable, "-c", child, str(Path(__file__).parent), cudnn.__file__, str(int(backward)), str(reload)],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-5000:]
        results.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert results[0]["misses"] > 0 and results[0]["hits"] == 0
    assert results[1]["misses"] == 0 and results[1]["hits"] > 0


@pytest.mark.parametrize("device_index", [0, 1])
def test_packed_copy_restores_caller_device(device_index):
    from cudnn.sdpa.packed_copy_sm80 import copy_packed_half

    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    if torch.cuda.get_device_capability(device_index) != (8, 0):
        pytest.skip("requires SM80 operands; the caller's current device may use another architecture")
    original = torch.cuda.current_device()
    try:
        torch.cuda.set_device(1 - device_index)
        device = torch.device("cuda", device_index)
        torch.manual_seed(292)
        tensors = tuple(torch.randn(1, t, h, 97, device=device, dtype=torch.bfloat16)[..., :96] for t, h in ((17, 4), (33, 2)))
        result = copy_packed_half(tensors, (128, 128), (True, False), compact=True)
        assert torch.cuda.current_device() == 1 - device_index
        for got, want in zip(result, tensors):
            torch.testing.assert_close(got[..., :96], want, atol=0, rtol=0)
            assert torch.count_nonzero(got[..., 96:]) == 0
    finally:
        torch.cuda.set_device(original)


@requires_native_sm80
def test_packed_copy_rebinds_strides_and_keeps_captured_kernel_alive():
    import gc
    from cudnn.sdpa import packed_copy_sm80
    from cudnn.sdpa.fwd.kernels.sm80.staged_copy import compile_gather

    torch.manual_seed(752)
    # The allocation recipe is reused across these different runtime strides.
    # Clearing both owner memos after capture also exercises graph lifetime.
    for step, extra in ((1, 8), (2, 16), (1, 8)):
        tensors = tuple(torch.randn(1, t, h, 96 * step + extra, device="cuda", dtype=torch.bfloat16)[..., : 96 * step : step] for t, h in ((17, 4), (33, 2)))

        def run():
            return packed_copy_sm80.copy_packed_half(tensors, (128, 128), (True, False), compact=True)

        run()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                outputs = run()
            packed_copy_sm80._plan.cache_clear()
            compile_gather.cache_clear()
            gc.collect()
            for tensor in tensors:
                tensor.mul_(0.5)
            for tensor in outputs:
                tensor.fill_(float("nan"))
            graph.replay()
            for got, want in zip(outputs, tensors):
                torch.testing.assert_close(got[..., :96], want, atol=0, rtol=0)
                assert torch.count_nonzero(got[..., 96:]) == 0
        finally:
            graph.reset()


@requires_native_sm80
@pytest.mark.parametrize("role", [0, 1])
@pytest.mark.gpu_exclusive
def test_packed_copy_physical_int64_stride_and_replay(role):
    from cudnn.sdpa.packed_copy_sm80 import copy_packed_half

    torch.manual_seed(2091)
    tensors = [torch.randn(1, 2, h, 96, device="cuda", dtype=torch.float16) for h in (4, 2)]
    old = tensors[role]
    strides = (0, 2**32 + 16, 96, 1)
    span = 1 + sum((n - 1) * st for n, st in zip(old.shape, strides))
    if torch.cuda.mem_get_info()[0] < span * old.element_size() + 2**30:
        pytest.skip("wide-stride copy control needs about 9 GiB free")
    try:
        owner = torch.empty(span, device="cuda", dtype=old.dtype)
    except torch.OutOfMemoryError:
        pytest.skip("insufficient memory for the wide-stride copy allocation")
    tensors[role] = owner.as_strided(old.shape, strides).copy_(old)

    def run():
        return copy_packed_half(tuple(tensors), (128, 128), (True, False))

    def check(outputs):
        for got, want in zip(outputs, tensors):
            torch.testing.assert_close(got[..., :96], want, atol=0, rtol=0)
            assert torch.count_nonzero(got[..., 96:]) == 0

    check(run())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            outputs = run()
        tensors[role].mul_(0.5)
        for tensor in outputs:
            tensor.fill_(float("nan"))
        graph.replay()
        check(outputs)
    finally:
        graph.reset()


@requires_native_sm80
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_packed_backward_compacts_native_width_operands(dtype, monkeypatch):
    from cudnn.sdpa import packed_copy_sm80
    from sdpa.frost.test_sdpa_bwd_prepared_thd_sm80 import _check
    from sdpa.frost.test_sdpa_bwd_thd_sm80 import _thd_case
    from test_sdpa_sm80_thd_wrapper_prepared import _prefix, _wrapper

    case = _thd_case((96, 160), (128, 96), 4, 128, dtype, hkv=2, d_v=128, cap_q=512, cap_kv=512, poison=True, causal=True)
    for name in ("q", "k", "v", "o", "do"):
        value = getattr(case, name)
        setattr(case, name, value.transpose(-1, -2).contiguous().transpose(-1, -2))
    original, calls = packed_copy_sm80.copy_packed_half, []

    def record(tensors, *args, **kwargs):
        calls.append(tuple(not tensor.is_contiguous() for tensor in tensors))
        return original(tensors, *args, **kwargs)

    monkeypatch.setattr(packed_copy_sm80, "copy_packed_half", record)
    output = _wrapper(case, _prefix(case.cu_q), _prefix(case.cu_k), deterministic=True, max_s_q=160, max_s_kv=128)
    _check(case, *(output[name] for name in ("dq_tensor", "dk_tensor", "dv_tensor")))
    assert calls == [(True,) * 5]
