# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native packed wrapper copies, persistent artifacts and stride declines."""

import pytest
import torch

from frost_test_utils import _SM, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]
requires_native_sm80 = pytest.mark.skipif(_SM != 80, reason="requires native SM80")


@requires_native_sm80
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("backward", [False, True])
def test_packed_wrapper_uses_native_operands(dtype, backward, monkeypatch):
    if backward:
        from cudnn.sdpa.bwd import api_dsl
        from test_sdpa_sm80_thd_wrapper_prepared import test_wrapper_rebind_capture_and_capacity_tails as check

        name = "_sm80_thd_backward"
    else:
        from cudnn.sdpa.fwd import api_dsl
        from test_sdpa_sm80_thd_forward_prepared import test_thd_wrapper_rebind_and_capture as check

        name = "_sm80_thd_forward"
    from test_sdpa_sm80_thd_native_widths import _forbid_copy

    _forbid_copy(monkeypatch)
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
def test_packed_wrapper_artifacts_reload_in_fresh_process(backward, tmp_path):
    import json
    import os
    from pathlib import Path
    from prepared_bwd_cache_utils import run_artifact_child
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
from test_sdpa_sm80_native_wrappers import test_packed_wrapper_uses_native_operands
if backward == "1":
    from cudnn.sdpa.bwd import api_dsl as owner
    name = "_sm80_thd_plan"
else:
    from cudnn.sdpa.fwd.kernels.sm80 import prepared_host as owner
    name = "compile_thd_host"
original = getattr(owner, name)
artifacts = []
def record(*args, **kwargs):
    result = original(*args, **kwargs)
    artifacts.append(result._prepared.artifact if backward == "1" else result[0])
    return result
with pytest.MonkeyPatch.context() as patch:
    patch.setattr(owner, name, record)
    if reload == "1":
        patch.setattr(cute, "compile", lambda *a, **k: pytest.fail("native packed wrapper fresh process invoked JIT"))
    test_packed_wrapper_uses_native_operands(torch.bfloat16, backward == "1", patch)
assert artifacts
if reload == "1":
    assert all(hasattr(a, "_compiled_cache_raw") for a in artifacts)
print(json.dumps(compiled_cache.stats()))
"""
    env = dict(os.environ, CUDNN_FRONTEND_COMPILED_CACHE=str(tmp_path))
    env.pop("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", None)
    results = []
    for reload in (0, 1):
        result = run_artifact_child(
            child,
            [str(Path(__file__).parent), cudnn.__file__, str(int(backward)), str(reload)],
            env=env,
            cwd=Path(__file__).parents[2],
            timeout=300,
        )
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-5000:]
        results.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert results[0]["misses"] > 0 and results[0]["hits"] == 0
    assert results[1]["misses"] == 0 and results[1]["hits"] > 0


@requires_native_sm80
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("role", ["q", "k", "v", "o", "do"])
def test_packed_backward_declines_nonunit_dimension_stride(dtype, role, monkeypatch):
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm80
    from sdpa.frost.test_sdpa_bwd_thd_sm80 import _thd_case
    from test_sdpa_sm80_thd_wrapper_prepared import _prefix, _wrapper

    case = _thd_case((33, 17), (41, 23), 4, 128, dtype, hkv=2, d_v=128, cap_q=80, cap_kv=96, poison=True, causal=True)
    value = getattr(case, role)
    pitched = torch.empty((*value.shape[:-1], 2 * value.shape[-1]), dtype=dtype, device=value.device)[..., ::2]
    setattr(case, role, pitched.copy_(value))

    def forbidden(*args, **kwargs):
        pytest.fail("invalid operand reached attention execution")

    monkeypatch.setattr(SdpaBwdDslSm80, "execute", forbidden)
    with pytest.raises(NotImplementedError, match=rf"SM80.*{role}.*(stride|packed rows)"):
        _wrapper(case, _prefix(case.cu_q), _prefix(case.cu_k), deterministic=True, max_s_q=40, max_s_kv=48)
