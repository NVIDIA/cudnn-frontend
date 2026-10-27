# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""D256 half correction/P1 ownership: rescale, physical tails and retained replay."""

from dataclasses import replace

import pytest
import torch

from frost_test_utils import _SM, assert_no_new_spills, requires_dsl, run_sass_probe, sass_probe_source
from test_utils import torch_fork_set_rng


@pytest.mark.L0
def test_half_fused_correction_config_and_register_pool():
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams, _validate_cfg_d256, make_cfg_d256, make_cfg_d256_mxfp8

    base = TemplateParams(dtype_qkv=2, window_right=0)
    cfg = make_cfg_d256(base)[0]
    assert cfg.HALF_FUSED_CORR_SPLIT_P and not cfg.FUSED_CORR_SPLIT_P
    assert cfg.SOFTMAX_WARPGROUPS == 1 and cfg.CORRECTION_WARPS == 4
    assert (cfg.SOFTMAX_REGS, cfg.CORRECTION_REGS, cfg.OTHER_REGS) == (240, 216, 48)
    assert 128 * (cfg.SOFTMAX_REGS + cfg.CORRECTION_REGS + cfg.OTHER_REGS) <= 65536
    with pytest.raises(ValueError, match="register budget"):
        _validate_cfg_d256(replace(cfg, CORRECTION_REGS=240))
    for changes in (
        dict(dtype_qkv=3),
        dict(window_right=None),
        dict(window_right=3),
        dict(window_left=128),
        dict(bottom_right=True),
        dict(seq_kv_lens_present=True),
        dict(thd_varlen=True, seq_kv_lens_present=True),
        dict(split_kv=2),
        dict(paged_kv=True, page_size=16, seq_kv_lens_present=True),
        dict(epilogue_gate=True),
    ):
        assert not make_cfg_d256(replace(base, **changes))[0].HALF_FUSED_CORR_SPLIT_P, changes
    mx = make_cfg_d256_mxfp8(TemplateParams(dtype_qkv=0, cta_mma=1, window_right=0))[0]
    assert mx.FUSED_CORR_SPLIT_P and not mx.HALF_FUSED_CORR_SPLIT_P


def _reference(q, k, v, sink, causal):
    hq, hkv = q.shape[1], k.shape[1]
    out = torch.empty_like(q, dtype=torch.float64)
    lse = torch.empty(q.shape[:-1] + (1,), device=q.device, dtype=torch.float64)
    rows = torch.arange(q.shape[2], device=q.device)[:, None]
    cols = torch.arange(k.shape[2], device=k.device)[None, :]
    for h in range(hq):
        scores = (q[:, h].double() @ k[:, h // (hq // hkv)].double().transpose(-1, -2)) / 16
        if causal:
            scores.masked_fill_(cols > rows, float("-inf"))
        if sink is not None:
            scores = torch.cat((scores, sink[:, h].double().expand(q.shape[0], q.shape[2], 1)), -1)
        probs = scores.softmax(-1)[..., : k.shape[2]]
        out[:, h] = probs @ v[:, h // (hq // hkv)].double()
        lse[:, h] = scores.logsumexp(-1, keepdim=True)
    return out, lse


@pytest.mark.L0
@requires_dsl
@pytest.mark.skipif(_SM not in (100, 103), reason="SM100/SM103 half D256 pipeline")
@pytest.mark.parametrize(
    "b,sq,sk,hq,hkv,pack,sink_enabled,dtype,causal,sched,d",
    [
        (1, 257, 385, 16, 1, True, False, torch.bfloat16, True, 1, 256),
        (2, 769, 1025, 6, 1, True, True, torch.bfloat16, True, 0, 256),
        (1, 513, 384, 8, 2, False, False, torch.bfloat16, True, 2, 256),
        (1, 257, 385, 4, 4, False, True, torch.bfloat16, True, 1, 256),
        (1, 257, 385, 8, 1, True, False, torch.float16, True, 1, 256),
        (1, 257, 385, 8, 1, True, False, torch.bfloat16, False, 1, 256),
        (1, 257, 385, 8, 1, True, False, torch.bfloat16, True, 1, 224),
    ],
    ids=["packed-tail", "partial-pack-rescale-sink", "unpacked-rectangular", "mha-sink", "legacy-fp16", "legacy-unmasked", "envelope224"],
)
@torch_fork_set_rng(seed=987)
def test_half_fused_correction_rebind_and_capture(b, sq, sk, hq, hkv, pack, sink_enabled, dtype, causal, sched, d, monkeypatch):
    import cudnn
    from cudnn.engines.manifest import MANIFEST
    from cudnn.sdpa.fwd import api_dsl

    loaded = []
    original = api_dsl.load_template

    def load(*args, **kwargs):
        mod = original(*args, **kwargs)
        if str(mod.__file__).endswith("/sm100/prefill_d256_f16.py"):
            loaded.append(mod)
        return mod

    monkeypatch.setattr(api_dsl, "load_template", load)

    def tensors():
        def make(s, h):
            return (torch.randn((b, s, h, d), device="cuda") * 0.25).to(dtype).transpose(1, 2)

        q, k, v = make(sq, hq), make(sk, hkv), make(sk, hkv)
        # The maximum increases by ~10 natural-log units each key tile. This
        # forces correction to rescale O while the other WG wants to reuse S.
        q[..., 0] = 1
        k[..., 0] = (torch.arange(sk, device="cuda") // 128 * 160).to(dtype)
        o = torch.empty_like(q)
        stats = torch.empty((b, hq, sq, 1), device="cuda", dtype=torch.float32)
        sink = torch.linspace(-2, 3, hq, device="cuda", dtype=torch.float32).view(1, hq, 1, 1) if sink_enabled else None
        return q, k, v, o, stats, sink

    initial = tensors()
    handle = cudnn.create_handle()
    io = cudnn.data_type.BFLOAT16 if dtype == torch.bfloat16 else cudnn.data_type.HALF
    graph = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, handle=handle)
    qd, kd, vd = (graph.tensor_like(t) for t in initial[:3])
    sinkd = graph.tensor_like(initial[5]) if sink_enabled else None
    extra = dict(sink_token=sinkd) if sink_enabled else {}
    od, sd = graph.sdpa(q=qd, k=kd, v=vd, generate_stats=True, attn_scale=1 / 16, use_causal_mask=causal, **extra)
    od.set_output(True).set_dim(initial[3].shape).set_stride(initial[3].stride())
    sd.set_output(True).set_dim(initial[4].shape).set_stride(initial[4].stride()).set_data_type(cudnn.data_type.FLOAT)
    graph.validate()
    graph.build_operation_graph()
    engine = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd").offered_ids()["sdpa_fwd_prefill_sm100"]
    knobs = {getattr(cudnn.knob_type, k): v for k, v in dict(TILE_M=128, TILE_N=128, TILE_CGA_M=2, PACK_GQA=int(pack), SCHED_POLICY=sched, SPLIT_KV=1).items()}
    graph.create_execution_plan(engine, knobs)
    graph.select_plan(graph.get_execution_plan_count() - 1)
    graph.check_support()
    graph.build_plans()
    assert loaded, "the explicit plan did not compile the native half D256 body"
    assert bool(loaded[-1].CFG.HALF_FUSED_CORR_SPLIT_P) == (dtype == torch.bfloat16 and causal)
    workspace = torch.empty(max(1, graph.get_workspace_size()), device="cuda", dtype=torch.uint8)

    def execute(t):
        vp = dict(zip((qd, kd, vd, od, sd), t[:5]))
        if sink_enabled:
            vp[sinkd] = t[5]
        cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)
        graph.execute(vp, workspace, handle=handle)

    def verify(t):
        expected, stats = _reference(*t[:3], t[5], causal)
        torch.testing.assert_close(t[3].double(), expected, atol=3e-3, rtol=2e-2)
        torch.testing.assert_close(t[4].double(), stats, atol=6e-4, rtol=0)

    try:
        for t in (initial, tensors()):  # New allocations through the SAME plan.
            t[3].fill_(float("nan"))
            t[4].fill_(float("nan"))
            execute(t)
            verify(t)
            capture = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(capture):
                    execute(t)
                for i in range(8):
                    if i == 4:
                        t[0].mul_(0.75)
                        t[2].neg_()
                    t[3].fill_(float("nan"))
                    t[4].fill_(float("nan"))
                    capture.replay()
                    verify(t)
            finally:
                capture.reset()
    finally:
        torch.cuda.synchronize()
        cudnn.destroy_handle(handle)


@pytest.mark.L0
@requires_dsl
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
@pytest.mark.parametrize("sched", [0, 1], ids=["natural", "lpt"])
def test_half_fused_correction_register_codegen(tmp_path, arch, sched):
    src = sass_probe_source(
        """
        params = TemplateParams(dtype_qkv=2, window_right=0, pack_gqa=True, qh_per_kh=16, **params_kw)
        mod = _load_sm100_kernel_module((256, 256), params, fp8=False, pertensor=False, rubin=False)
        assert mod.CFG.HALF_FUSED_CORR_SPLIT_P
        mod.compile(d_qk=256, d_v=256, has_lse=True)
        """,
        counts={
            "CORR_ALLOC": ("USETMAXREG", "0xd8"),
            "SOFTMAX_ALLOC": ("USETMAXREG", "0xf0"),
            "OTHER_FREE": ("USETMAXREG", "0x30"),
            "LDL": (" LDL",),
            "STL": (" STL",),
        },
    )
    # Force a real compiler run even if a previous process populated DSL's
    # file cache; the codegen assertion requires this invocation's artifact.
    src = 'import os; os.environ["CUTE_DSL_DISABLE_FILE_CACHING"] = "1"; os.environ["CUTE_DSL_NO_CACHE"] = "1"\n' + src
    result = run_sass_probe(tmp_path, probe_src=src, arch=arch, params={"sched_policy": sched}, tag="half_fused_correction")
    assert result.stats["CORR_ALLOC"] > 0 and result.stats["SOFTMAX_ALLOC"] > 0 and result.stats["OTHER_FREE"] > 0, result
    bounds = {"LDL": 16, "STL": 12} if sched == 0 else {"LDL": 0, "STL": 0}
    assert_no_new_spills(result.stats, bounds, tag=f"{arch} sched={sched}: ")
