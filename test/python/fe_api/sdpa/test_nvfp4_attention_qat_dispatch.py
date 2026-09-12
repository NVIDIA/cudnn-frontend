# SPDX-License-Identifier: Apache-2.0

"""NVFP4 QAT automatic selection, forced routes and preparation-time fallback."""

import pytest
import torch

from test_utils import torch_fork_set_rng
from fe_api.sdpa.test_nvfp4_attention_qat_backward import _environment_supported, _reference_case

pytestmark = [pytest.mark.L0, pytest.mark.skipif(not _environment_supported(), reason="Requires Blackwell, CuTe DSL, and Triton")]


def _require_frost():
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

    installed, version = cutedsl_state()
    if torch.cuda.get_device_capability() != (10, 0) or not installed or cutedsl_too_old(version):
        pytest.skip("FROST execution requires SM100 and CuTe DSL >= 4.7.0")


def _declaration(batch=1, heads=2, sequence=256, kv_sequence=None):
    kv_sequence = sequence if kv_sequence is None else kv_sequence
    q = torch.empty((batch, heads, sequence, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.empty((batch, heads, kv_sequence, 128), device="cuda", dtype=torch.bfloat16)
    lse = torch.empty((batch, heads, sequence), device="cuda", dtype=torch.float32)
    return q, k, k, q, q, lse


def test_auto_prefers_frost():
    _require_frost()
    from cudnn import Nvfp4AttentionQatBackward

    inputs = _declaration()
    op = Nvfp4AttentionQatBackward(*inputs)
    assert op.backend == "auto"
    assert op.selected_backend is None
    assert op.check_support()
    assert op.selected_backend == "frost"
    assert op.fallback_reason is None
    explicit = Nvfp4AttentionQatBackward(*inputs, backend="frost")
    assert op.scratch_workspace_bytes() == explicit.scratch_workspace_bytes()


def test_auto_selection_is_fixed_after_support_check(monkeypatch):
    _require_frost()
    from cudnn import Nvfp4AttentionQatBackward

    op = Nvfp4AttentionQatBackward(*_declaration())
    op.check_support()
    workspace_bytes = op.scratch_workspace_bytes()

    def unexpected_selection(*args):
        pytest.fail("a prepared plan must not redo backend selection")

    monkeypatch.setattr(op, "_frost_support_reason", unexpected_selection)
    assert op.check_support() and op.selected_backend == "frost"
    assert op.scratch_workspace_bytes() == workspace_bytes


@pytest.mark.parametrize("capability", [(10, 3), (12, 0), (12, 1)])
def test_auto_architecture_selection_metadata_only(monkeypatch, capability):
    """Probe routing, not execution of a foreign architecture on this GPU."""
    from cudnn import Nvfp4AttentionQatBackward

    inputs = _declaration()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: capability)
    op = Nvfp4AttentionQatBackward(*inputs)
    assert op.check_support() and op.selected_backend == "triton"
    assert "SM100" in op.fallback_reason
    with pytest.raises(NotImplementedError, match="SM100"):
        Nvfp4AttentionQatBackward(*inputs, backend="frost").check_support()


@pytest.mark.parametrize(
    "declaration,options,reason",
    [
        ({"batch": 2}, {}, "B=1"),
        ({}, {"is_causal": True}, "noncausal"),
        ({"sequence": 257}, {}, "divisible by 256"),
        ({"kv_sequence": 512}, {}, "equal sequence"),
    ],
)
def test_auto_unsupported_frost_declaration(declaration, options, reason):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Exercise SM100 shape rejection independently of the architecture gate")
    from cudnn import Nvfp4AttentionQatBackward

    inputs = _declaration(**declaration)
    op = Nvfp4AttentionQatBackward(*inputs, **options)
    assert op.check_support() and op.selected_backend == "triton"
    assert reason in op.fallback_reason
    with pytest.raises(NotImplementedError, match=reason):
        Nvfp4AttentionQatBackward(*inputs, backend="frost", **options).check_support()


@torch_fork_set_rng(seed=107)
def test_auto_old_dsl_falls_back_without_importing_frost(monkeypatch):
    import builtins
    from cudnn import Nvfp4AttentionQatBackward, nvfp4_attention_qat_backward
    from cudnn.frost import buffers
    from cudnn.sdpa.bwd.qat import api

    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Exercise the DSL rejection independently of the architecture gate")
    inputs, _ = _reference_case(256, 256, is_causal=False)
    q, k, v, o, do, lse, scale = inputs
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.6.2")))
    monkeypatch.setattr(api, "_OBJECT_CACHE", api.OrderedDict())
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name.rsplit(".", 1)[-1] not in {"_frost", "_frost_kernel"}, "fallback imported a version-specific FROST kernel"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    op = Nvfp4AttentionQatBackward(*inputs[:6])
    assert op.check_support() and op.selected_backend == "triton"
    assert "4.6.2" in op.fallback_reason
    with pytest.raises(NotImplementedError, match="found 4.6.2"):
        Nvfp4AttentionQatBackward(*inputs[:6], backend="frost").check_support()
    got = nvfp4_attention_qat_backward(do, q, k, v, o, lse, softmax_scale=scale)
    expected = nvfp4_attention_qat_backward(do, q, k, v, o, lse, softmax_scale=scale, backend="triton")
    for a, b in zip(got, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert {plan.selected_backend for plan in api._OBJECT_CACHE.values()} == {"triton"}


@pytest.mark.parametrize(
    "options,match", [({"head_chunk": 3}, "divide"), ({"workspace_limit_bytes": -1}, "nonnegative"), ({"workspace_limit_bytes": True}, "nonnegative")]
)
def test_auto_rejects_invalid_options(options, match):
    from cudnn import Nvfp4AttentionQatBackward

    with pytest.raises(ValueError, match=match):
        Nvfp4AttentionQatBackward(*_declaration(), **options).check_support()


def test_auto_workspace_limit_selects_chunk_or_triton():
    _require_frost()
    from cudnn import Nvfp4AttentionQatBackward

    inputs = _declaration(sequence=512)
    limit = Nvfp4AttentionQatBackward(*inputs, backend="frost").scratch_workspace_bytes()
    # FROST needs exactly the Triton scratch (fake Q/K/V + delta): both fit or neither does.
    assert limit == Nvfp4AttentionQatBackward(*inputs, backend="triton").scratch_workspace_bytes()
    fits = Nvfp4AttentionQatBackward(*inputs, workspace_limit_bytes=limit)
    assert fits.check_support() and fits.selected_backend == "frost"
    assert fits.scratch_workspace_bytes() == limit
    with pytest.raises(NotImplementedError, match="workspace_limit_bytes"):
        Nvfp4AttentionQatBackward(*inputs, workspace_limit_bytes=limit - 1).check_support()
    with pytest.raises(NotImplementedError, match="workspace_limit_bytes"):
        Nvfp4AttentionQatBackward(*inputs, backend="frost", workspace_limit_bytes=limit - 1).check_support()
    with pytest.raises(NotImplementedError, match="workspace_limit_bytes"):
        Nvfp4AttentionQatBackward(*inputs, workspace_limit_bytes=0).check_support()
    # head_chunk only changes launch granularity, never the workspace.
    chunked = Nvfp4AttentionQatBackward(*inputs, head_chunk=1, workspace_limit_bytes=limit)
    assert chunked.check_support() and chunked.selected_backend == "frost"
    assert chunked.selected_head_chunk == 1 and chunked.scratch_workspace_bytes() == limit


@torch_fork_set_rng(seed=113)
def test_auto_workspace_selected_chunk_executes():
    _require_frost()
    from cudnn import Nvfp4AttentionQatBackward

    inputs, _ = _reference_case(512, 512, is_causal=False)
    budget = Nvfp4AttentionQatBackward(*inputs[:6], backend="frost").scratch_workspace_bytes()
    plans = [Nvfp4AttentionQatBackward(*inputs[:6], head_chunk=1, workspace_limit_bytes=budget), Nvfp4AttentionQatBackward(*inputs[:6], backend="triton")]
    results = []
    for op in plans:
        op.check_support()
        op.compile()
        outputs = tuple(torch.empty_like(t) for t in inputs[:3])
        workspace = torch.empty(op.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
        op.execute(*inputs[:6], *outputs, workspace)
        results.append(outputs)
    assert plans[0].selected_backend == "frost" and plans[0].selected_head_chunk == 1
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, atol=0.005, rtol=0.005)


@torch_fork_set_rng(seed=127)
def test_auto_batch_fallback_executes():
    from cudnn import Nvfp4AttentionQatBackward, nvfp4_attention_qat_backward

    single, _ = _reference_case(256, 256, is_causal=False)
    q, k, v, o, do, lse = (t.repeat((2,) + (1,) * (t.ndim - 1)) for t in single[:6])
    op = Nvfp4AttentionQatBackward(q, k, v, o, do, lse)
    assert op.check_support() and op.selected_backend == "triton"
    got = nvfp4_attention_qat_backward(do, q, k, v, o, lse)
    expected = nvfp4_attention_qat_backward(do, q, k, v, o, lse, backend="triton")
    for a, b in zip(got, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@torch_fork_set_rng(seed=109)
def test_auto_wrapper_routes_and_cache_separation(monkeypatch):
    _require_frost()
    from cudnn import Nvfp4AttentionQatBackward, nvfp4_attention_qat_backward
    from cudnn.sdpa.bwd.qat import api

    inputs, _ = _reference_case(256, 256, is_causal=False)
    q, k, v, o, do, lse, scale = inputs
    monkeypatch.setattr(api, "_OBJECT_CACHE", api.OrderedDict())
    triton_bytes = Nvfp4AttentionQatBackward(*inputs[:6], backend="triton").scratch_workspace_bytes()
    routes = []
    original_execute = Nvfp4AttentionQatBackward.execute

    def observed_execute(self, *args, **kwargs):
        routes.append(self.selected_backend)
        return original_execute(self, *args, **kwargs)

    monkeypatch.setattr(Nvfp4AttentionQatBackward, "execute", observed_execute)
    results = []
    for options in ({}, {"backend": "frost"}, {"backend": "triton"}, {"workspace_limit_bytes": triton_bytes}, {}):
        results.append(nvfp4_attention_qat_backward(do, q, k, v, o, lse, softmax_scale=scale, **options))
    # FROST's scratch equals Triton's, so a Triton-sized budget still selects FROST.
    assert routes == ["frost", "frost", "triton", "frost", "frost"]
    assert len(api._OBJECT_CACHE) == 4
    for invalid in ({"head_chunk": False}, {"workspace_limit_bytes": float(triton_bytes)}):
        with pytest.raises(ValueError, match="nonnegative integer"):
            nvfp4_attention_qat_backward(do, q, k, v, o, lse, **invalid)
    for result in results:
        for a, b in zip(result, results[0]):
            torch.testing.assert_close(a, b, atol=0.005, rtol=0.005)


def test_auto_compile_and_execute_errors_do_not_fall_back(monkeypatch):
    _require_frost()
    from cudnn import Nvfp4AttentionQatBackward
    from cudnn.sdpa.bwd.qat import _frost

    inputs = _declaration()
    op = Nvfp4AttentionQatBackward(*inputs)
    op.check_support()

    def compiler_error(*args, **kwargs):
        raise RuntimeError("sentinel compiler defect")

    monkeypatch.setattr(_frost.PreparedBackward, "compile", compiler_error)
    with pytest.raises(RuntimeError, match="sentinel compiler defect"):
        op.compile()
    assert op.selected_backend == "frost" and op._compiled_kernel is None

    class BrokenLaunch:
        def execute(self, *args, **kwargs):
            raise RuntimeError("sentinel launch defect")

    op._compiled_kernel = BrokenLaunch()
    outputs = tuple(torch.empty_like(t) for t in inputs[:3])
    workspace = torch.empty(op.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="sentinel launch defect"):
        op.execute(*inputs, *outputs, workspace)
    assert op.selected_backend == "frost" and op.fallback_reason is None
