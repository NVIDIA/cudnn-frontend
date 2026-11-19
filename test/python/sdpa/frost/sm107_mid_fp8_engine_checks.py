# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public FP8 D192/D256 records fix the body across rebuild and retained capture."""

import json

import pytest
import torch

import cudnn
from frost_test_utils import requires_dsl, requires_rubin
import test_sdpa_prepared_fp8 as shared

pytestmark = [pytest.mark.L0, requires_dsl, requires_rubin]


@pytest.mark.parametrize("d,dv", [(16, 144), (176, 112), (224, 208), (384, 320)])
@pytest.mark.parametrize("block", [16, 32])
def test_wide_envelopes_keep_the_adapter_block_output_decline(d, dv, block):
    """Removing a broad fallback proposal does not remove a buildable plan."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    dtype_o = torch.float4_e2m1fn_x2 if block == 16 else torch.float8_e4m3fn
    q = torch.empty((1, 1, 17, d), device="cuda", dtype=torch.float8_e4m3fn)
    k = torch.empty((1, 1, 128, d), device="cuda", dtype=torch.float8_e4m3fn)
    v = torch.empty((1, 1, 128, dv), device="cuda", dtype=torch.float8_e4m3fn)
    o = torch.empty((1, 1, 17, dv // 2 if block == 16 else dv), device="cuda", dtype=dtype_o)
    sf = torch.empty((1, 1, 128, max(4, (dv + block - 1) // block)), device="cuda", dtype=torch.float8_e4m3fn if block == 16 else torch.uint8)
    api = SdpaFwdDslSm100(sample_q=q, sample_k=k, sample_v=v, sample_o=o, sample_sf_o=sf, pertensor_fp8=True)
    with pytest.raises(NotImplementedError, match="d128 flavor only"):
        api.check_support()


@pytest.mark.parametrize("d,dv,flavor", [(192, 128, 192), (256, 256, 256), (176, 112, 192), (224, 208, 256)])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("output", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("split", [1, 4])
def test_public_record_rebuild_rebind_and_retained_capture(monkeypatch, d, dv, flavor, dtype, output, split):
    from cudnn.sdpa.fwd import heuristics
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    owner = "d192_fp8_multi_q" if flavor == 192 else "d256_fp8_single_q"
    engine_id = 20561 if flavor == 192 else 20562
    cga = 2 if flavor == 192 else 1
    record = (engine_id, SdpaFwdKnobs(tile_m=128, tile_n=128, cga=cga, sched_policy=0, pack_gqa=True, split_kv=split).to_public())

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit public replay consulted candidate selection")

    monkeypatch.setattr(heuristics, "recommend", forbidden)

    def build(saved):
        result = shared._case(arch="sm107", d=d, dv=dv, sq=17, skv=256, dtype=dtype, output_dtype=output, split_kv=split, record=saved)
        graph = result[0]
        assert graph.get_engine_and_knobs_at_index(graph._plan_index) == saved
        compiled = graph._compiled_plans[graph._plan_index]._compiled
        assert compiled.kernel_template == ("prefill_d192_d128_fp8" if flavor == 192 else "prefill_d256_fp8")
        assert compiled.kernel_module.__file__.endswith(f"sm107/{owner}/kernel.py")
        assert type(compiled.kernel_module.CFG).__module__ == f"cudnn.sdpa.fwd.kernels.sm107.{owner}.config"
        metadata = compiled.implementation_config
        assert graph.selected_engine.name == f"sdpa_fwd_sm107_{owner}"
        assert (metadata["query_tile"], metadata["cluster_m"], metadata["cta_group"], metadata["split_kv"]) == (128, cga, cga, split)
        return result

    g, vp, ws, bufs, tensors = build(record)
    g.execute(vp, ws)
    shared._check(bufs, thd=False, sq=17, skv=256)

    with shared._cuda_graph() as capture:
        with torch.cuda.graph(capture):
            previous = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                g.execute(vp, ws)
            finally:
                torch.cuda.set_sync_debug_mode(previous)
        rebuilt, rebound, scratch, other, _ = build(g.get_engine_and_knobs_at_index(g._plan_index))
        rebuilt.execute(rebound, scratch)
        shared._check(other, thd=False, sq=17, skv=256)
        bufs["descale_q"].fill_(0.5)
        bufs["scale_o"].fill_(1.7)
        for name in ("o", "lse", "amax_o"):
            bufs[name].fill_(float("nan"))
        capture.replay()
        shared._check(bufs, thd=False, sq=17, skv=256)

    # Eager execution rebinds caller storage and scales independently of the
    # retained executable's capture-time pointers.
    for name in ("q", "k", "v", "o", "lse", "amax_o", "descale_q", "scale_o"):
        bufs[name] = bufs[name].clone()
        vp[tensors[name]] = bufs[name]
    g.execute(vp, torch.empty_like(ws))
    shared._check(bufs, thd=False, sq=17, skv=256)


@pytest.mark.parametrize("d,dv,engine_id", [(192, 128, 20561), (176, 112, 20561), (256, 256, 20562), (224, 208, 20562)])
def test_retired_fp8_recipe_is_rejected_without_silent_substitution(d, dv, engine_id):
    import cudnn._plan_serdes as ps

    graph, _, _, _, _ = shared._case(arch="sm107", d=d, dv=dv, sq=17, skv=256)
    blob = graph.serialize(as_bytes=True, include_kernels=False)
    start = len(ps.MAGIC) + ps._LEN.size
    (length,) = ps._LEN.unpack_from(blob, len(ps.MAGIC))
    header = json.loads(blob[start : start + length])
    header["plan"].update(engine_id=20514, engine_name="sdpa_fwd_prefill_sm107_fp8", knobs=[])
    encoded = json.dumps(header).encode()
    target = cudnn.pygraph()
    with pytest.raises((ValueError, RuntimeError), match="20514|owned by|not supported|No execution plans"):
        target.deserialize(ps.MAGIC + ps._LEN.pack(len(encoded)) + encoded)
    assert not target._nodes and not target._planning_done
    # The same empty destination remains reusable for an eligible current recipe.
    target.deserialize(blob)
    assert target.get_engine_and_knobs_at_index(target._plan_index)[0] == engine_id


@pytest.mark.parametrize("d", [192, 256])
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group("gpu_exclusive")
def test_mid_fp8_refilled_stage_warm_stress(d):
    from warm_stress import StressCase, assert_warm_stress
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    cga, engine_id, b = (2, 20561, 256) if d == 192 else (1, 20562, 512)
    record = (engine_id, SdpaFwdKnobs(tile_m=128, tile_n=128, cga=cga, sched_policy=0, pack_gqa=True, split_kv=1).to_public())
    graph, vp, ws, bufs, _ = shared._case(arch="sm107", d=d, dv=128 if d == 192 else 256, b=b, sq=17, skv=1024, record=record)
    graph.execute(vp, ws)
    shared._check(bufs, thd=False, b=b, sq=17, skv=1024)
    # 1024 physical CTAs and eight KV tiles exercise successive ring reuse.
    assert_warm_stress(
        StressCase(
            f"SM107 D{d} per-tensor FP8",
            lambda: graph.execute(vp, ws),
            {"o": bufs["o"].transpose(1, 2), "stats": bufs["lse"], "amax_o": bufs["amax_o"]},
            thread_blocks=b * 2 * cga,
            refills_per_launch=4,
        ),
        warm=256,
        poisoned=256,
    )


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_d192_cga1_causal_public_record(dtype):
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs
    from test_sdpa_fwd_dsl_sm107 import _fp8_d192_oracle
    from determinism_utils import assert_replays

    record = (20561, SdpaFwdKnobs(tile_m=128, tile_n=128, cga=1, sched_policy=0, pack_gqa=False, split_kv=1).to_public())
    graph, vp, ws, bufs, _ = shared._case(
        arch="sm107",
        d=192,
        dv=128,
        b=2,
        heads=(2, 2),
        sq=129,
        skv=1024,
        dtype=dtype,
        output_dtype=torch.float8_e4m3fn,
        record=record,
        sdpa_kwargs={"right_bound": 0},
    )
    assert graph.get_engine_and_knobs_at_index(graph._plan_index) == record
    compiled = graph._compiled_plans[graph._plan_index]._compiled
    assert compiled.kernel_module.__file__.endswith("sm107/d192_fp8_multi_q/kernel.py")
    assert compiled.implementation_config["cluster_m"] == 1

    def check():
        q, k, v = (bufs[n].double() * bufs["descale_" + n].double() for n in ("q", "k", "v"))
        ref, stats = _fp8_d192_oracle(q, k, v, attn_scale=192**-0.5, causal=True)
        am = ref.abs().amax()
        ref *= bufs["scale_o"].double()
        floor = (ref - ref.to(bufs["o"].dtype).double()).abs().max().item()
        torch.testing.assert_close(bufs["o"].double(), ref, atol=max(0.03, 3.0 * floor), rtol=0.03)
        torch.testing.assert_close(bufs["lse"].double(), stats, atol=0.01, rtol=0.01)
        torch.testing.assert_close(bufs["amax_o"][0].double(), am, atol=0.03, rtol=0.03)

    def run():
        graph.execute(vp, ws)
        return {n: bufs[n].clone() for n in ("o", "lse", "amax_o")}

    assert_replays(run, repeats=3)
    check()
    with shared._cuda_graph() as capture:
        with torch.cuda.graph(capture):
            graph.execute(vp, ws)
        bufs["descale_v"].fill_(0.3)
        bufs["scale_o"].fill_(1.7)
        for name in ("o", "lse", "amax_o"):
            bufs[name].fill_(float("nan"))
        capture.replay()
        check()


def _d256_pair_problem(*, b, hq, hkv, sq, skv, dtype=torch.float8_e4m3fn, seed=0, zero_qk=False):
    """BHSD per-tensor FP8 operands for the standalone adapter, with unit scales (the operating point of the e2e half test).
    ``zero_qk``: Q = K = 0, so every running max stays put and no step rescales O (the PV-completion ring wraps unobserved by a rescale)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    draw = lambda h, s: (torch.randn(b, h, s, 256, device="cuda", generator=gen) * 0.5).to(dtype)  # noqa: E731
    q8, k8, v8 = draw(hq, sq), draw(hkv, skv), draw(hkv, skv)
    if zero_qk:
        q8.zero_()
        k8.zero_()
    return q8, k8, v8


def _d256_pair_reference(q8, k8, v8, *, causal=False, bottom_right=False):
    """fp64 softmax(scale Q K^T) V and the natural-log LSE; a keyless row (bottom-right causal, S_q > S_kv) is O = 0 / LSE = -inf."""
    rep = q8.shape[1] // k8.shape[1]
    scores = (q8.double() @ k8.double().repeat_interleave(rep, 1).transpose(-1, -2)) * 256**-0.5
    if causal:
        sq, skv = scores.shape[-2:]
        scores = scores.masked_fill(torch.ones(sq, skv, device=scores.device, dtype=torch.bool).triu(1 + (skv - sq if bottom_right else 0)), float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    out = torch.softmax(scores, dim=-1) @ v8.double().repeat_interleave(rep, 1)
    out = torch.where(torch.isinf(lse).unsqueeze(-1), torch.zeros_like(out), out)
    return out.float(), lse.float()


def _d256_pair_adapter(q8, k8, v8, out, lse, **kwargs):
    """The two-warpgroup body: per-tensor FP8 D256 at cga 2 with HALF softmax through the standalone adapter."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    api = SdpaFwdDslSm100(
        sample_q=q8,
        sample_k=k8,
        sample_v=v8,
        sample_o=out,
        sample_lse=lse,
        scale_softmax=256**-0.5,
        pertensor_fp8=True,
        has_amax_o=False,
        cga=2,
        softmax_precision=cudnn.data_type.HALF,
        **kwargs,
    )
    assert api.check_support()
    api.compile()
    cfg = api._k_mod.CFG
    assert api._k_mod.__file__.endswith("sm107/d256_fp8_single_q/kernel.py")
    assert (cfg.SOFTMAX_WARPGROUPS, cfg.Q_IN_TMEM, cfg.CTA_MMA, cfg.KV_LOOKAHEAD) == (2, 1, 2, 4)
    workspace = torch.empty(max(api.scratch_workspace_bytes(), 1), device="cuda", dtype=torch.uint8)

    def launch():
        api.execute(q_tensor=q8, k_tensor=k8, v_tensor=v8, o_tensor=out, lse_tensor=lse, workspace=workspace)

    return api, launch


def _d256_pair_check(out, lse, q8, k8, v8, *, causal=False, bottom_right=False):
    ref, ref_lse = _d256_pair_reference(q8, k8, v8, causal=causal, bottom_right=bottom_right)
    err = (out.float() - ref).abs().max().item()
    assert err <= 0.1 * ref.abs().max().item(), f"max err {err} vs the fp64 reference"
    torch.testing.assert_close(lse, ref_lse, atol=0.02, rtol=0.02, equal_nan=False)


def test_d256_pair_admission():
    """Explicit cga=2 reaches the pair on the HALF-softmax dense leg only; the default width and the public record stay at cga 1."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100, supported_cgas_for

    assert supported_cgas_for((256, 256), fp8=True, device_cc=(10, 7), pertensor=True) == (1, 2)
    assert supported_cgas_for((256, 256), fp8=True, device_cc=(10, 0), pertensor=True) == (1,)
    q8, k8, v8 = _d256_pair_problem(b=1, hq=2, hkv=2, sq=64, skv=256)
    out = torch.empty_like(q8, dtype=torch.bfloat16)
    lse = torch.empty(1, 2, 64, device="cuda", dtype=torch.float32)
    common = dict(sample_q=q8, sample_k=k8, sample_v=v8, sample_o=out, sample_lse=lse, scale_softmax=256**-0.5, pertensor_fp8=True, has_amax_o=False)
    for declined in (
        dict(cga=2),
        dict(cga=2, softmax_precision=cudnn.data_type.HALF, split_kv=2),
        dict(cga=2, softmax_precision=cudnn.data_type.HALF, pack_gqa=True),
    ):
        with pytest.raises(NotImplementedError, match="per-tensor FP8 2-CTA pair"):
            SdpaFwdDslSm100(**common, **declined).check_support()
    default = SdpaFwdDslSm100(**common, softmax_precision=cudnn.data_type.HALF)
    assert default.check_support()
    default.compile()
    assert (default._k_mod.CFG.CTA_MMA, default._k_mod.CFG.SOFTMAX_WARPGROUPS) == (1, 1)


_PAIR_CASES = {
    "odd-steps": dict(sq=200, skv=896),
    "even-steps": dict(sq=200, skv=1024),
    "no-rescale-ring-wrap": dict(sq=200, skv=1024, zero_qk=True),
    "causal": dict(sq=1024, skv=1024, is_causal=True),
    # 256 cluster tiles, three of every four Q tiles keyless: the first wave's CTAs finish their empty tiles early and take live
    # ones through the cluster launch control, so one CTA pair runs empty-to-live and live-to-empty handoffs.
    "bottom-right-empty-tiles": dict(sq=1024, skv=256, is_causal=True, causal_bottom_right=True, b=8, hq=8, hkv=4),
}


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("case", list(_PAIR_CASES))
def test_d256_pair_body_reference_and_replay(dtype, case):
    """The two-warpgroup body against an fp64 reference, then byte-identical replay under concurrent GPU work.  The zero-Q/K case
    never rescales O (the PV-completion ring wraps without a rescale wait); the bottom-right case runs the first three Q tiles of
    every head keyless ahead of the live one, with more cluster tiles than the device holds at once."""
    from determinism_utils import assert_replays

    spec = dict(_PAIR_CASES[case])
    sq, skv = spec.pop("sq"), spec.pop("skv")
    zero_qk = spec.pop("zero_qk", False)
    causal, bottom_right = spec.get("is_causal", False), spec.get("causal_bottom_right", False)
    b, hq, hkv = spec.pop("b", 2), spec.pop("hq", 4), spec.pop("hkv", 2)
    q8, k8, v8 = _d256_pair_problem(b=b, hq=hq, hkv=hkv, sq=sq, skv=skv, dtype=dtype, zero_qk=zero_qk)
    out = torch.empty(b, hq, sq, 256, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(b, hq, sq, device="cuda", dtype=torch.float32)
    _, launch = _d256_pair_adapter(q8, k8, v8, out, lse, **spec)
    launch()
    torch.cuda.synchronize()
    _d256_pair_check(out, lse, q8, k8, v8, causal=causal, bottom_right=bottom_right)

    def run():
        out.fill_(float("nan"))
        lse.fill_(float("nan"))
        launch()
        return {"o": out, "lse": lse}

    assert_replays(run)


@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group("gpu_exclusive")
def test_d256_pair_refilled_stage_warm_stress():
    """The alpha, running-max and final-stat slots of the two-warpgroup body are refilled inside one launch."""
    from warm_stress import StressCase, assert_warm_stress

    b, hq, sq, skv = 64, 16, 200, 1024
    q8, k8, v8 = _d256_pair_problem(b=b, hq=hq, hkv=hq, sq=sq, skv=skv)
    out = torch.empty(b, hq, sq, 256, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(b, hq, sq, device="cuda", dtype=torch.float32)
    _, launch = _d256_pair_adapter(q8, k8, v8, out, lse)
    launch()
    torch.cuda.synchronize()
    _d256_pair_check(out, lse, q8, k8, v8)
    # 2048 CTAs (two per 256-row cluster tile); eight KV steps refill each parity's alpha and running-max slot three times per tile.
    assert_warm_stress(
        StressCase("SM107 D256 per-tensor FP8 two-warpgroup body", launch, {"o": out, "lse": lse}, thread_blocks=b * hq * 2, refills_per_launch=3),
        warm=256,
        poisoned=256,
    )


def test_d256_pair_release_dependence(tmp_path):
    """Every shared-memory slot credit of the two-warpgroup body (alpha, running max, final stats) is an arrive whose address
    depends on the loaded words through the release-zero argument (Rule 11), pinned on the PTX of a fresh compile."""
    import os
    from pathlib import Path
    import subprocess
    import sys
    import textwrap

    import warm_stress

    source = textwrap.dedent("""
        import faulthandler, sys
        from pathlib import Path
        faulthandler.dump_traceback_later(280, exit=True)
        sys.meta_path[:] = [f for f in sys.meta_path if "editable" not in str(f).lower()]
        sys.path[:0] = sys.argv[1:3]
        import cudnn, torch
        assert Path(cudnn.__file__).resolve().parent.parent == Path(sys.argv[1]).resolve()
        from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100
        q8 = torch.zeros(1, 1, 256, 256, device="cuda", dtype=torch.float8_e4m3fn)
        k8 = torch.zeros(1, 1, 256, 256, device="cuda", dtype=torch.float8_e4m3fn)
        out = torch.empty(1, 1, 256, 256, device="cuda", dtype=torch.bfloat16)
        lse = torch.empty(1, 1, 256, device="cuda", dtype=torch.float32)
        api = SdpaFwdDslSm100(
            sample_q=q8, sample_k=k8, sample_v=k8, sample_o=out, sample_lse=lse, scale_softmax=256**-0.5,
            pertensor_fp8=True, has_amax_o=False, cga=2, softmax_precision=cudnn.data_type.HALF,
        )
        assert api.check_support()
        api.compile()
        assert api._k_mod.CFG.SOFTMAX_WARPGROUPS == 2
        """)
    test_root = Path(__file__).resolve().parents[2]
    source_root = Path(cudnn.__file__).resolve().parent.parent
    env = dict(
        os.environ,
        CUTE_DSL_DUMP_DIR=str(tmp_path),
        CUTE_DSL_KEEP="ptx,cubin",
        CUDNN_FRONTEND_DISABLE_COMPILED_CACHE="1",
        CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO="0",
    )
    result = subprocess.run([sys.executable, "-c", source, str(source_root), str(test_root)], env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    bodies = [p.read_text() for p in tmp_path.glob("*.ptx") if "tcgen05.mma" in p.read_text()]
    assert len(bodies) == 1, f"expected one attention PTX, found {len(bodies)}"
    # Four dependent credits: the running-max exchange of each softmax warpgroup (one word), the alpha slot (one word) and the
    # final-stat slot (two words).
    sites = warm_stress.assert_release_dependence(bodies[0], sites=4, words=1)
    assert sorted(site.loaded_registers for site in sites if site.dependent) == [1, 1, 1, 2]
