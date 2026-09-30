# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Current-buffer, stream and metadata contracts of the MXFP8 backward host."""

from dataclasses import replace

import pytest
import torch

import cudnn
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
from test_sdpa_bwd_mxfp8_sm100 import _run

pytestmark = [requires_pre_rubin_blackwell, requires_dsl]
_NAMES = {"o": "o_f16", "do": "dO", "do_T": "dO_T", "do_f16": "dO_f16", "sf_do": "sf_dO", "sf_do_T": "sf_dO_T"}


def _tensors(case):
    roles = {graph_name: role for role, graph_name in _NAMES.items()}
    return {roles.get(name, name): case.pack[ref] for name, ref in case.refs.items()}


def _api(case):
    from cudnn.sdpa.bwd.api_dsl_mxfp8_sm100 import SdpaBwdDslSm100Mxfp8

    api = SdpaBwdDslSm100Mxfp8(**{"sample_" + name: t for name, t in _tensors(case).items()}, scale_softmax=case.scale)
    api.check_support()
    api.compile()
    return api


def _execute(api, case, **kwargs):
    args = {(name if name.startswith("sf_") else name + "_tensor"): t for name, t in _tensors(case).items()}
    api.execute(**args, workspace=case.workspace, **kwargs)


def _check(case):
    for role, expected in zip(("dq", "dk", "dv"), case.expected):
        torch.testing.assert_close(case.pack[case.refs[role]], expected, rtol=0, atol=0)


def _case(seed=7):
    return _run(hq=4, hkv=2, sq=128, skv=160, seed=seed)


@pytest.mark.L0
class TestPreparedMxfp8Bwd:
    @pytest.mark.parametrize("standalone", [False, True])
    def test_warm_execute_has_no_tensor_plumbing(self, monkeypatch, standalone):
        import cutlass.cute as cute
        import cutlass.cute.runtime as runtime

        case = _case()
        api = _api(case) if standalone else None
        launch = lambda: _execute(api, case) if standalone else case.graph.execute(case.pack, case.workspace)
        launch()
        torch.cuda.synchronize()
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]

        def forbidden(*args, **kwargs):
            raise AssertionError("prepared backward rebuilt tensor arguments or compiled on execute")

        with monkeypatch.context() as patch:
            patch.setattr(cute, "compile", forbidden)
            patch.setattr(runtime, "from_dlpack", forbidden)
            for name in ("view", "reshape", "permute", "contiguous", "as_strided"):
                patch.setattr(torch.Tensor, name, forbidden)
            torch.cuda.set_sync_debug_mode("error")
            try:
                for _ in range(3):
                    launch()
            finally:
                torch.cuda.set_sync_debug_mode("default")
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        _check(case)

    @pytest.mark.parametrize("standalone", [False, True])
    def test_rebind_and_explicit_stream_replay(self, standalone):
        case, changed = _case(7), _case(91)
        api = _api(case)
        before = api._prepared.fn
        old, new = _tensors(case), _tensors(changed)
        assert any(not torch.equal(old[name], new[name]) for name in old if name.startswith("sf_"))
        rebound = {ref: changed.pack[changed.refs[name]] for name, ref in case.refs.items()}
        stream, ambient = torch.cuda.Stream(), torch.cuda.Stream()
        handle = cudnn.create_handle()
        cudnn.set_stream(handle, stream.cuda_stream)
        capture = torch.cuda.CUDAGraph()
        launch = lambda: (
            _execute(api, changed, current_stream=stream.cuda_stream) if standalone else case.graph.execute(rebound, changed.workspace, handle=handle)
        )
        try:
            changed.workspace.fill_(0xBD)
            for role in ("dq", "dk", "dv"):
                new[role].fill_(float("nan"))
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(ambient):
                launch()
            torch.cuda.current_stream().wait_stream(stream)
            _check(changed)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(capture, stream=stream):
                launch()
            # Reuse captured addresses with another complete, independently
            # checked input set, including changed SF exponents.
            for name, tensor in new.items():
                if name not in ("dq", "dk", "dv"):
                    tensor.copy_(old[name])
                else:
                    tensor.fill_(float("nan"))
            changed.workspace.fill_(0xAD)
            capture.replay()
            changed.expected = case.expected
            _check(changed)
            assert api._prepared.fn is before
        finally:
            capture.reset()
            cudnn.destroy_handle(handle)

    @pytest.mark.parametrize("role", ["q", "do", "dq", "stats"])
    @pytest.mark.parametrize("ordered", [False, True])
    def test_graph_raw_storage_and_override_validation(self, role, ordered, monkeypatch):
        case = _case()
        ref = case.refs[_NAMES.get(role, role)]
        tensor = case.pack[ref]
        backing = torch.empty(tensor.numel() + 2, device="cuda", dtype=tensor.dtype)
        declared = backing.as_strided(tensor.shape, tensor.stride()).copy_(tensor)
        case.pack[ref] = backing[::2]
        pack, kwargs = case.pack, {}
        if ordered:
            items = list(reversed(list(pack.items())))
            pack = [buf for _, buf in items]
            kwargs["tensor_uids"] = [t.get_uid() for t, _ in items]
        case.graph.execute(pack, case.workspace, **kwargs)
        if role == "dq":
            torch.testing.assert_close(declared, case.expected[0], rtol=0, atol=0)
        else:
            _check(case)
        plan = case.graph._compiled_plans[case.graph._plan_index]
        calls = []
        monkeypatch.setattr(plan._prepared, "spec", replace(plan._prepared.spec, fn=lambda *args: calls.append(args)))
        shape = list(ref.get_dim())
        shape[2] //= 2
        with pytest.raises(ValueError, match="runtime geometry"):
            case.graph.execute(
                pack, case.workspace, override_uids=[ref.get_uid()], override_shapes=[shape], override_strides=[list(ref.get_stride())], **kwargs
            )
        assert not calls

    @pytest.mark.parametrize("failure", ["q_layout", "sf_gap", "sf_short", "workspace_alias", "cpu", "missing"])
    def test_standalone_rejects_before_any_stage(self, failure):
        case = _case()
        api = _api(case)
        calls = []
        api._prepared = replace(api._prepared, fn=lambda *args: calls.append(args))
        if failure == "q_layout":
            case.pack[case.refs["q"]] = case.pack[case.refs["q"]].contiguous()
        elif failure in ("sf_gap", "sf_short"):
            t = case.pack[case.refs["sf_q"]]
            case.pack[case.refs["sf_q"]] = torch.empty(t.numel() * 2, dtype=t.dtype, device="cuda")[::2] if failure == "sf_gap" else t.flatten()[:-1]
        elif failure == "workspace_alias":
            t = case.pack[case.refs["q"]]
            case.pack[case.refs["q"]] = case.workspace[: t.numel()].view(t.dtype).as_strided(t.shape, t.stride())
        elif failure == "cpu":
            case.pack[case.refs["sf_q"]] = case.pack[case.refs["sf_q"]].cpu()
        else:
            case.pack[case.refs["sf_q"]] = None
        with pytest.raises(ValueError):
            _execute(api, case)
        assert not calls

    def test_standalone_opaque_sf_storage_and_flat_stats(self):
        case = _case()
        api = _api(case)
        for name, ref in case.refs.items():
            if name.startswith("sf_"):
                case.pack[ref] = case.pack[ref].view(torch.int32).reshape(2, -1).t()
        case.pack[case.refs["stats"]] = case.pack[case.refs["stats"]].flatten()
        _execute(api, case)
        _check(case)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("layout", ["sfa", "sfb"])
def test_sf_repack_steps_past_int32_with_allocated_guards(layout):
    """A narrowed block index writes the guard and leaves the real tail poisoned."""
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver
    from cudnn.sdpa.bwd.kernels.sm100.bprop_sf_repack_mxfp8 import Mxfp8SfRepackSm100

    planes = 2**20 + 1 if layout == "sfa" else 2**21 + 1
    repack = Mxfp8SfRepackSm100(128, 4, planes, layout)
    guard = 2**31
    required = 2 * guard + repack.src_bytes + repack.dst_bytes
    if torch.cuda.mem_get_info()[0] < required + 1024**3:
        pytest.skip("physical SF index test needs its allocated overflow guards")
    try:
        source = torch.full((guard + repack.src_bytes,), 33, dtype=torch.int8, device="cuda")
        target = torch.full((guard + repack.dst_bytes,), -1, dtype=torch.int8, device="cuda")
    except torch.OutOfMemoryError:
        pytest.skip("concurrent allocations exhausted physical SF guard storage")
    source[guard:].fill_(127)

    @cute.jit
    def host(src: cute.Pointer, dst: cute.Pointer, kernel: cutlass.Constexpr, stream: driver.CUstream):
        kernel(cute.make_tensor(src, cute.make_layout((kernel.src_bytes,))), cute.make_tensor(dst, cute.make_layout((kernel.dst_bytes,))), stream)

    pointer = cute.runtime.make_ptr(cutlass.Int8, 16, cute.AddressSpace.gmem, assumed_align=16)
    fn = cute.compile(host, pointer, pointer, repack, cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False), options="--enable-tvm-ffi")
    from cudnn.frost.compiled_cache import positional_entry

    launch = positional_entry(fn)
    launch(source.data_ptr() + guard, target.data_ptr() + guard, torch.cuda.current_stream().cuda_stream)
    # Both layouts finish with the last plane's peer slot. In SFA the last
    # row tile is odd, so only m1==1 is sourced; in SFB only m1<2 is sourced.
    index = torch.arange(repack.dst_bytes - 512, repack.dst_bytes, dtype=torch.int64, device="cuda")
    m1 = index // 4 % 4
    expected = torch.where(m1 == 1 if layout == "sfa" else m1 < 2, 127, 0).to(torch.int8)
    torch.testing.assert_close(target[-512:], expected, rtol=0, atol=0)
    # A signed-Int32 control wraps the first overflowing block to the start
    # of this fully allocated prefix. No intentionally invalid pointer runs.
    torch.testing.assert_close(target[:512], torch.full_like(target[:512], -1), rtol=0, atol=0)
