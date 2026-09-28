# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared initialization preserves the packed wrappers' zeroed capacity."""

import pytest
import torch

from frost_test_utils import _SM, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]
requires_native_sm80 = pytest.mark.skipif(_SM != 80, reason="requires native SM80")


@requires_native_sm80
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("backward", [False, True])
def test_packed_wrapper_uses_prepared_initialization(dtype, backward, monkeypatch):
    if backward:
        from cudnn.sdpa.bwd import api_dsl
        from test_sdpa_sm80_thd_wrapper_prepared import test_wrapper_rebind_capture_and_capacity_tails as check

        name = "_sm80_thd_backward"
    else:
        from cudnn.sdpa.fwd import api_dsl
        from test_sdpa_sm80_thd_forward_prepared import test_thd_wrapper_rebind_and_capture as check

        name = "_sm80_thd_forward"
    original = getattr(api_dsl, name)

    def run(*args, **kwargs):
        with monkeypatch.context() as guard:
            for entry in ("zeros", "zeros_like"):
                guard.setattr(torch, entry, lambda *a, **k: pytest.fail("packed wrapper rebuilt Torch zero initialization"))
            return original(*args, **kwargs)

    monkeypatch.setattr(api_dsl, name, run)
    if backward:
        check(96, 96, dtype, monkeypatch)
    else:
        check(96, 96, dtype, True, monkeypatch)


def _compile_init(operands):
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver
    from cudnn.frost.compiled_cache import positional_entry
    from cudnn.sdpa.fwd.kernels.sm80.packed_init import _host

    pointers = tuple(cute.runtime.make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=4) for _ in range(operands))
    artifact = cute.compile(_host, pointers, (cutlass.Int64(0),) * operands, driver.CUstream(0), options="--enable-tvm-ffi")
    entry = positional_entry(artifact)
    assert entry is not None
    return artifact, entry


@pytest.mark.parametrize("operands", [2, 6])
def test_initializer_rebinds_counts_and_preserves_guards(operands):
    import gc

    artifact, entry = _compile_init(operands)
    for sizes in ((0, 3, 5, 1023, 1025, 2049), (2051, 0, 7, 5, 4099, 19)):
        counts = sizes[:operands]
        owners = tuple(torch.full((n + 8,), 19, dtype=torch.int32, device="cuda") for n in counts)
        outputs = tuple(t[4:-4] for t in owners)
        pointers = tuple(t.data_ptr() for t in outputs)

        # Zero-size tensors report a null pointer; that region is never stepped.
        def run():
            entry(pointers, counts, torch.cuda.current_stream().cuda_stream)

        def check():
            for owner, output in zip(owners, outputs):
                assert torch.count_nonzero(output) == 0
                assert (owner[:4] == 19).all() and (owner[-4:] == 19).all()

        run()
        check()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                run()
            for output in outputs:
                output.fill_(7)
            graph.replay()
            check()
        finally:
            graph.reset()
    # The graph owns the launched function after the Python artifact is dropped.
    with torch.cuda.graph(graph):
        run()
    del artifact, entry
    gc.collect()
    for output in outputs:
        output.fill_(11)
    try:
        graph.replay()
        check()
    finally:
        graph.reset()


def _compile_half_layout_init():
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver
    from cudnn.frost.compiled_cache import positional_entry
    from cudnn.sdpa.fwd.kernels.sm80.packed_init import zero_outputs

    @cute.jit
    def host(output: cute.Pointer, words: cutlass.Int64, stream: driver.CUstream):
        tensor = cute.make_tensor(output, cute.make_layout((words, 2), stride=(2, 1)))
        zero_outputs((tensor,), stream)

    pointer = cute.runtime.make_ptr(cutlass.Uint16, 16, cute.AddressSpace.gmem, assumed_align=4)
    artifact = cute.compile(host, pointer, cutlass.Int64(0), driver.CUstream(0), options="--enable-tvm-ffi")
    entry = positional_entry(artifact)
    assert entry is not None
    return artifact, lambda pointers, counts, stream: entry(pointers[0], counts[0], stream)


@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("half_layout", [False, True])
def test_initializer_physical_word_count_above_uint32(half_layout):
    words = 2**32 + 32
    if torch.cuda.mem_get_info()[0] < (words + 32) * 4 + 2**30:
        pytest.skip("physical wide-count control needs about 17 GiB free")
    try:
        owner = torch.empty(words + 32, dtype=torch.int32, device="cuda")
    except torch.OutOfMemoryError:
        pytest.skip("insufficient memory for the physical wide-count allocation")
    output = owner[16:-16]
    artifact, entry = _compile_half_layout_init() if half_layout else _compile_init(1)
    owner[:16].fill_(19)
    owner[-16:].fill_(23)

    def poison():
        output[:32].fill_(7)
        output[2**31 : 2**31 + 32].fill_(11)
        output[-32:].fill_(13)

    def check():
        for values in (output[:32], output[2**31 : 2**31 + 32], output[-32:]):
            assert torch.count_nonzero(values) == 0
        assert (owner[:16] == 19).all() and (owner[-16:] == 23).all()

    def run():
        entry((output.data_ptr(),), (words,), torch.cuda.current_stream().cuda_stream)

    poison()
    run()
    check()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            run()
        poison()
        graph.replay()
        check()
    finally:
        graph.reset()


@requires_native_sm80
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mha_wrapper_preserves_unwritten_kv_capacity(dtype, monkeypatch):
    from test_sdpa_sm80_thd_wrapper_prepared import _prefix, _wrapper
    from sdpa.frost.test_sdpa_bwd_thd_sm80 import _thd_case
    from sdpa.frost.test_sdpa_bwd_prepared_thd_sm80 import _check

    case = _thd_case((96, 160), (128, 96), 4, 128, dtype, hkv=4, cap_q=512, cap_kv=512, poison=True, causal=True)
    original = torch.empty_like

    def allocate(*args, **kwargs):
        return original(*args, **kwargs).fill_(7)

    with monkeypatch.context() as guard:
        guard.setattr(torch, "empty_like", allocate)
        result = _wrapper(case, _prefix(case.cu_q), _prefix(case.cu_k), max_s_q=160, max_s_kv=128)
    _check(case, *(result[key] for key in ("dq_tensor", "dk_tensor", "dv_tensor")))
    assert torch.count_nonzero(result["dq_tensor"][:, case.t_q :]) == 0
    for key in ("dk_tensor", "dv_tensor"):
        assert (result[key][:, case.t_kv :] == 7).all()


@requires_native_sm80
@pytest.mark.parametrize("empty_kv", [False, True])
def test_forward_empty_capacity_skips_initialization(empty_kv):
    from test_sdpa_sm80_thd_forward_prepared import _inputs, _prefix, _run

    tensors = _inputs(128, 128, torch.bfloat16, capq=0, capkv=0 if empty_kv else 64)
    cq, ck = _prefix((0,)), _prefix((0 if empty_kv else 64,))
    result = _run(tensors, cq, ck, maxq=1)
    assert result["o_tensor"].shape == (1, 0, 4, 128)
    assert result["lse_tensor"].shape == (1, 4, 0)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            result = _run(tensors, cq, ck, maxq=1)
        graph.replay()
        torch.cuda.synchronize()
    finally:
        graph.reset()


@requires_native_sm80
@pytest.mark.parametrize("backward", [False, True])
def test_packed_initialization_reloads_in_fresh_process(backward, tmp_path):
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
folder, package, backward, reload = sys.argv[1:]
assert Path(cudnn.__file__).resolve() == Path(package).resolve()
sys.path[:0] = [folder, str(Path(folder).parents[1])]
from test_sdpa_sm80_packed_init import test_packed_wrapper_uses_prepared_initialization
with pytest.MonkeyPatch.context() as patch:
    if reload == "1":
        patch.setattr(cute, "compile", lambda *a, **k: pytest.fail("initialization artifact reload invoked JIT"))
    test_packed_wrapper_uses_prepared_initialization(torch.bfloat16, backward == "1", patch)
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
@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_packed_initialization_uses_operand_device(device_index, backward, explicit):
    from test_sdpa_sm80_thd_wrapper_prepared import _prefix, _wrapper
    from test_sdpa_sm80_thd_forward_prepared import _run, _check as fcheck
    from sdpa.frost.test_sdpa_bwd_thd_sm80 import _thd_case
    from sdpa.frost.test_sdpa_bwd_prepared_thd_sm80 import _check as bcheck

    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    if torch.cuda.get_device_capability(device_index) != (8, 0):
        pytest.skip("requires SM80 operands; caller device may use another architecture")
    original = torch.cuda.current_device()
    try:
        with torch.cuda.device(device_index):
            case = _thd_case((96, 160), (128, 96), 4, 128, torch.bfloat16, hkv=2, cap_q=512, cap_kv=512, poison=True, causal=True)
            cq, ck = _prefix(case.cu_q), _prefix(case.cu_k)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
        torch.cuda.set_device(1 - device_index)
        raw = stream.cuda_stream if explicit else None
        if backward:
            result = _wrapper(case, cq, ck, max_s_q=160, max_s_kv=128, current_stream=raw)
        else:
            result = _run((case.q, case.k, case.v), cq, ck, causal=True, maxq=160, stream=raw)
        assert torch.cuda.current_device() == 1 - device_index
        with torch.cuda.device(device_index):
            torch.cuda.synchronize()
            if backward:
                bcheck(case, *(result[key] for key in ("dq_tensor", "dk_tensor", "dv_tensor")))
                for key, live in (("dq_tensor", case.t_q), ("dk_tensor", case.t_kv), ("dv_tensor", case.t_kv)):
                    assert torch.count_nonzero(result[key][:, live:]) == 0
            else:
                fcheck((case.q, case.k, case.v), result, (96, 160), (128, 96), causal=True)
    finally:
        torch.cuda.set_device(original)
