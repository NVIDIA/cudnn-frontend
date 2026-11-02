# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Hopper D64: host routing, cross-target code generation and Hopper numerics.

Compilation on a different GPU is useful evidence, never a numerical qualification.
"""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import cudnn
from cudnn.sdpa.fwd import engines
from cudnn.sdpa.fwd.placement import LEAD, TRAIL, place
from cudnn.sdpa.fwd.heuristics import recommend
from frost_test_utils import _SM, requires_dsl
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = [pytest.mark.L0, requires_dsl]


def _facts(**over):
    values = dict(b=2, h_q=4, h_kv=4, s_q=70, s_kv=100, d_qk=64, d_v=64, dtype=cudnn.data_type.HALF, device_cc=(9, 0))
    values.update(over)
    return SdpaGraphFacts(**values)


def _caps():
    return next(spec.capabilities for spec in engines.ENGINE_SPECS if spec.name == "sdpa_fwd_prefill_sm90")


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("h_kv", [1, 4])
def test_native_d64_admission_and_backend_first_placement(causal, h_kv):
    facts = _facts(d_qk=64, d_v=64, device_cc=(9, 0), causal=causal, h_kv=h_kv)
    caps = _caps()
    assert engines.mismatch(caps, facts, engines.SdpaFwdKnobs(pack_gqa=False)) is None
    assert engines._selected_d_shape(caps, facts) == (64, 64)
    spec = next(spec for spec in engines.ENGINE_SPECS if spec.name == "sdpa_fwd_prefill_sm90")
    assert place(spec, facts) == TRAIL


@pytest.mark.parametrize("head", [(8, 8), (56, 64), (64, 56), (128, 128), (256, 512)])
def test_d64_admission_does_not_remove_d512_envelope_floor(head):
    assert engines.mismatch(_caps(), _facts(d_qk=head[0], d_v=head[1], device_cc=(9, 0))) is not None


@pytest.mark.parametrize(
    "extra",
    [dict(thd=True), dict(padded=True), dict(seq_q_trim=True), dict(has_sink=True), dict(window_left=16), dict(right_band_widening=True)],
)
def test_native_d64_declines_unimplemented_graph_features(extra):
    assert engines.mismatch(_caps(), _facts(d_qk=64, d_v=64, device_cc=(9, 0), **extra)) is not None


def test_native_d64_declines_pack_gqa():
    assert engines.mismatch(_caps(), _facts(d_qk=64, d_v=64, device_cc=(9, 0)), engines.SdpaFwdKnobs(pack_gqa=True)) is not None


def test_native_d64_exports_the_prepared_pointer_contract():
    from inspect import signature
    from cudnn.sdpa.fwd.api_dsl import _load_kernel_template
    from cudnn.sdpa.fwd.config_sm90 import TemplateParams
    from cudnn.sdpa.fwd.prepared import _host_parameter_order

    params = TemplateParams()
    native = _load_kernel_template("sm90/prefill_d64_f16.py", params, tag="sm90_d64_host_contract")
    control = _load_kernel_template("sm90/prefill_d512_f16.py", params, tag="sm90_d512_host_contract")
    # These are the required exports read by adapter compilation and the
    # prepared dense binder; CFG/N_Q and THD-only entries are optional or unused.
    required = (
        "_host",
        "compile",
        "PARAMS",
        "PREPARED_FIXED_BATCH",
        "PREPARED_WORKSPACE_ALIGNMENT",
        "PREPARED_DENSE_FLEX",
        "PREPARED_FIXED_SHAPE",
        "PREPARED_KV_TAIL_NATIVE",
    )
    for name in required:
        assert hasattr(native, name), f"native D64 is missing the prepared consumer's {name} export"
        assert hasattr(control, name)
        if name.startswith("PREPARED_"):
            assert getattr(native, name) == getattr(control, name)
    order = _host_parameter_order(native._host)
    assert order == _host_parameter_order(control._host)
    assert order[0] == "q_ptr" and order[-1] == "stream"
    assert tuple(signature(native.compile).parameters) == tuple(signature(control.compile).parameters)


def test_native_d64_recommendations_stay_unpacked():
    plans = recommend("A", _facts(h_q=8, h_kv=1, causal=True), {"sdpa_fwd_prefill_sm90": 20517})
    assert plans
    for plan in plans:
        assert plan.knobs.pack_gqa is False
        assert engines.mismatch(_caps(), _facts(h_q=8, h_kv=1, causal=True), plan.knobs) is None


@pytest.mark.parametrize("head", [(512, 512), (264, 504)])
def test_native_d512_domain_and_placement_are_preserved(head):
    facts = _facts(d_qk=head[0], d_v=head[1])
    assert engines.mismatch(_caps(), facts) is None
    assert engines._selected_d_shape(_caps(), facts) == (512, 512)
    spec = next(spec for spec in engines.ENGINE_SPECS if spec.name == "sdpa_fwd_prefill_sm90")
    assert place(spec, facts) == (LEAD if head == (512, 512) else TRAIL)


@pytest.mark.parametrize("dtype", [0, 1], ids=["fp16", "bf16"])
@pytest.mark.parametrize("case", ["dense-no-stats", "dense-log2-stats", "causal-tl-stats", "causal-br-no-stats", "zero-scale-stats", "negative-scale-stats"])
def test_native_d64_cross_compiles_hopper_instructions(tmp_path, dtype, case):
    """Trace/assemble sm_90a without launching it on the worker's GPU."""
    _cross_compile(tmp_path, dtype, case, "d64")


@pytest.mark.parametrize("dtype", [0, 1], ids=["fp16", "bf16"])
def test_native_d512_cross_compile_control(tmp_path, dtype):
    """The established large-head body still emits its own instruction geometry."""
    _cross_compile(tmp_path, dtype, "causal-tl-stats", "d512")


def _cross_compile(tmp_path, dtype, case, flavor):
    probe = r"""
import glob, os, pathlib, re, sys
os.environ["CUTE_DSL_ARCH"] = "sm_90a"
os.environ["CUTE_DSL_DUMP_DIR"] = sys.argv[1]
os.environ["CUTE_DSL_KEEP"] = "ptx,cubin"
os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
from cudnn.sdpa.fwd.api_dsl import _load_kernel_template
from cudnn.sdpa.fwd.config_sm90 import TemplateParams
from cudnn.frost.tile_dsl.constants import DTYPE_FP16, DTYPE_BF16
case, flavor = sys.argv[3:5]
d = 64 if flavor == "d64" else 512
causal, stats = not case.startswith("dense"), "stats" in case and "no-stats" not in case
h_kv = 1 if causal else 4
params = TemplateParams(dtype_qkv=(DTYPE_FP16, DTYPE_BF16)[int(sys.argv[2])], has_lse=stats, causal=causal,
                        bottom_right=False if case == "causal-tl-stats" else None, qh_per_kh=4//h_kv,
                        sched_policy=0 if not causal else 1 if stats else 2, stats_log2=case == "dense-log2-stats",
                        scale_mode=0 if case.startswith("zero") else -1 if case.startswith("negative") else 1)
mod = _load_kernel_template(f"sm90/prefill_{flavor}_f16.py", params, tag=f"sm90_{flavor}_compile_test")
kernel = mod.SM90FusedMultiHeadAttentionForward((2, 4, h_kv, 70, 100), qh_per_kh=4//h_kv)
assert kernel.HEAD_TILE == d and kernel.o_fragment_slots == (32 if d == 64 else 128)
assert kernel.threads_per_cta == (256 if d == 64 else 384)
artifact = mod.compile(2, 4, h_kv, 70, 100, (70*4*d,d,4*d), (100*h_kv*d,d,h_kv*d), (100*h_kv*d,d,h_kv*d), (70*4*d,d,4*d), (280,70,1) if stats else None, target="sm_90a", d_qk=d, d_v=d)
from types import SimpleNamespace
from cudnn.sdpa.fwd.prepared import _positional_order
raw, owner, order = _positional_order(SimpleNamespace(_k_mod=mod, _compiled_kernel=artifact))
assert callable(raw) and owner is artifact
assert order[0] == "q_ptr" and order[-1] == "stream"
paths = list(pathlib.Path(sys.argv[1]).rglob("*.ptx"))
assert paths, "compiler emitted no PTX"
ptx = "\n".join(path.read_text() for path in paths)
assert ".target sm_90a" in ptx
assert "m64n64k16" in ptx
assert ("m64n256k16" in ptx) is (d == 512)
if d == 64:
    groups = [part.count("wgmma.mma_async") for part in re.split(r"wgmma\.commit_group\.sync\.aligned\s*;", ptx)]
    assert any(groups) and all(count in (0, 4) for count in groups), "each QK/PV group contracts exactly four k16 steps"
assert re.search(r"setmaxnreg\.dec\.sync\.aligned\.u32\s+24;", ptx)
assert re.search(r"setmaxnreg\.inc\.sync\.aligned\.u32\s+" + ("200" if d == 64 else "240") + ";", ptx)
print("NATIVE_HOPPER_COMPILE_OK", flavor, len(paths))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, str(tmp_path), str(dtype), case, flavor], capture_output=True, text=True, timeout=300, env=dict(os.environ)
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-6000:]
    assert f"NATIVE_HOPPER_COMPILE_OK {flavor}" in result.stdout


@pytest.mark.skipif(_SM != 90, reason="native D64 execution requires actual Hopper SM90")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("h_kv", [1, 4])
@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize(
    "sq,skv,causal,bottom_right",
    [(70, 100, False, False), (70, 100, True, False), (70, 100, True, True), (100, 70, False, False), (100, 70, True, False)],
)
def test_native_d64_forward_and_changed_input_replay(dtype, causal, bottom_right, h_kv, stats, sq, skv, monkeypatch, scale=None):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    b, h, d = 2, 4, 64
    torch.manual_seed(0)
    q = torch.randn(b, sq, h, d, device="cuda", dtype=dtype).transpose(1, 2)
    k, v = (torch.randn(b, skv, h_kv, d, device="cuda", dtype=dtype).transpose(1, 2) for _ in range(2))
    storage = torch.full((b, sq, h, d + 8), float("nan"), device="cuda", dtype=dtype)
    o = storage[..., :d].transpose(1, 2)
    lse = torch.empty(b, h, sq, device="cuda", dtype=torch.float32) if stats else None
    api = SdpaFwdDslSm90(q, k, v, o, sample_lse=lse, is_causal=causal, causal_bottom_right=bottom_right, scale_softmax=scale)
    api.compile()
    assert Path(api._k_mod.__file__).name == "prefill_d64_f16.py"

    def reference(q=q, k=k, v=v, o=o, lse=lse, storage=storage):
        scores = (q.double() @ k.double().repeat_interleave(h // h_kv, dim=1).transpose(-1, -2)) * (d**-0.5 if scale is None else scale)
        if causal:
            diagonal = skv - sq if bottom_right else 0
            visible = torch.arange(skv, device="cuda")[None, :] <= torch.arange(sq, device="cuda")[:, None] + diagonal
            scores = scores.masked_fill(~visible, -float("inf"))
        expected = scores.softmax(-1).nan_to_num() @ v.double().repeat_interleave(h // h_kv, dim=1)
        torch.testing.assert_close(o, expected.to(dtype), atol=5e-2, rtol=3e-2)
        if stats:
            torch.testing.assert_close(lse, scores.logsumexp(-1).float(), atol=2e-2, rtol=2e-2)
        assert torch.isnan(storage[..., d:]).all()

    api.execute(q, k, v, o, lse_tensor=lse)
    reference()
    fresh = tuple(torch.randn_like(t) for t in (q, k, v))
    fresh_storage = torch.full_like(storage, float("nan"))
    fresh_o = fresh_storage[..., :d].transpose(1, 2)
    fresh_lse = torch.empty_like(lse) if stats else None
    import cutlass.cute as cute

    def forbidden(*args, **kwargs):
        raise AssertionError("warm execution entered a JIT/tensor-conversion path")

    capture = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with monkeypatch.context() as guard:
            guard.setattr(cute, "compile", forbidden)
            guard.setattr(cute.runtime, "from_dlpack", forbidden)
            torch.cuda.set_sync_debug_mode("error")
            try:
                allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
                api.execute(*fresh, fresh_o, lse_tensor=fresh_lse)
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            finally:
                torch.cuda.set_sync_debug_mode("default")
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(capture, stream=stream):
                api.execute(*fresh, fresh_o, lse_tensor=fresh_lse)
            fresh[2].mul_(0.5)
            capture.replay()
        torch.cuda.synchronize()
        reference(*fresh, fresh_o, fresh_lse, fresh_storage)
    finally:
        capture.reset()


@pytest.mark.skipif(_SM != 90, reason="native D64 graph execution requires actual Hopper SM90")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("stats", [False, True])
def test_native_d64_graph_uses_prepared_native_body(monkeypatch, dtype, causal, stats):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90
    import test_sdpa_fwd_dsl_sm100 as shared

    monkeypatch.setattr(shared, "_ARCH", "sm90")
    compile_api = SdpaFwdDslSm90.compile
    compiled = []

    def checked_compile(api):
        compile_api(api)
        assert Path(api._k_mod.__file__).name == "prefill_d64_f16.py"
        assert api._dense_spec is not None
        compiled.append(True)

    def forbidden(*args, **kwargs):
        raise AssertionError("graph returned to the tensor adapter")

    monkeypatch.setattr(SdpaFwdDslSm90, "compile", checked_compile)
    monkeypatch.setattr(SdpaFwdDslSm90, "execute", forbidden)
    q = torch.randn(2, 70, 4, 64, device="cuda", dtype=dtype).transpose(1, 2)
    k, v = (torch.randn(2, 100, 1, 64, device="cuda", dtype=dtype).transpose(1, 2) for _ in range(2))
    result = shared._run_dsl_graph(q, k, v, scale=64**-0.5, dtype=dtype, sdpa_kwargs=dict(use_causal_mask=causal), pack_gqa=False, return_stats=stats)
    o, lse = result if stats else (result, None)
    scores = (q.double() @ k.double().repeat_interleave(4, dim=1).transpose(-1, -2)) * 64**-0.5
    if causal:
        visible = torch.arange(100, device="cuda")[None, :] <= torch.arange(70, device="cuda")[:, None]
        scores = scores.masked_fill(~visible, -float("inf"))
    expected = scores.softmax(-1) @ v.double().repeat_interleave(4, dim=1)
    torch.testing.assert_close(o, expected.to(dtype), atol=5e-2, rtol=3e-2)
    if stats:
        torch.testing.assert_close(lse.squeeze(-1), scores.logsumexp(-1).float(), atol=2e-2, rtol=2e-2)
    assert compiled


@pytest.mark.skipif(_SM != 90, reason="native D64 scale execution requires actual Hopper SM90")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [0.0, -0.25], ids=["zero", "negative"])
def test_native_d64_zero_and_negative_scale_reference_and_replay(monkeypatch, dtype, scale):
    test_native_d64_forward_and_changed_input_replay(dtype, True, True, 1, True, 70, 100, monkeypatch, scale=scale)


def test_native_d64_graph_declines_unpadded_bottom_right_oversize():
    why = engines.mismatch(_caps(), _facts(s_q=100, s_kv=70, causal=True, bottom_right=True))
    assert why is not None and "S_q > S_kv" in why


@pytest.mark.skipif(_SM != 90, reason="SM90 standalone support checks require actual Hopper")
def test_native_d64_standalone_declines_unpadded_bottom_right_oversize():
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    q = torch.empty(1, 100, 2, 64, device="cuda", dtype=torch.float16).transpose(1, 2)
    k, v = (torch.empty(1, 70, 1, 64, device="cuda", dtype=torch.float16).transpose(1, 2) for _ in range(2))
    o = torch.empty_like(q)
    with pytest.raises(NotImplementedError, match="unpadded bottom-right S_q > S_kv"):
        SdpaFwdDslSm90(q, k, v, o, is_causal=True, causal_bottom_right=True).check_support()
