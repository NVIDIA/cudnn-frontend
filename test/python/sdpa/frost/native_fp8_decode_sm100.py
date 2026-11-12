# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native FP8 paged decode graph coverage, exposed by the SM100/103 CI entry.

Forward-only: backward has separate kernels. Independent FP64 numerics and
raw-byte capture/restore are separate checks. No heuristic ranking is assumed.
"""

import hashlib
from dataclasses import replace

import pytest
import torch

from sdpa.frost import test_sdpa_fwd_paged_sm100 as paged


def digest(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _observe_loader(monkeypatch):
    from cudnn.sdpa.fwd import api_dsl

    original = api_dsl._load_sm100_kernel_module
    loaded = []

    def observe(*args, **kwargs):
        module = original(*args, **kwargs)
        loaded.append(module)
        return module

    monkeypatch.setattr(api_dsl, "_load_sm100_kernel_module", observe)
    return loaded


def _assert_decode(mod):
    assert mod.__file__.endswith("sm100/decode_d128_fp8.py")
    assert mod.FROST_SOURCE_DIGEST
    assert (mod.TILE_K, mod.TILE_O, mod.N_Q, mod.COLS, mod.STAGES) == (128, 128, 16, 8, 5)


@pytest.mark.parametrize(
    "in_key,out_dtype,sq,group,split,stats,hnd",
    [
        (key, dt, sq, group, split, stats, hnd)
        for key, fp8 in [("e4m3", torch.float8_e4m3fn), ("e5m2", torch.float8_e5m2)]
        for dt, sq, group, split, stats, hnd in [
            (fp8, 1, 8, 1, True, True),
            (fp8, 1, 8, 4, True, True),
            (torch.bfloat16, 2, 4, 1, True, False),
            (torch.bfloat16, 2, 4, 4, True, False),
            (torch.float16, 2, 4, 4, False, False),
        ]
    ],
)
def test_native_fp8_decode_graph(in_key, out_dtype, sq, group, split, stats, hnd, monkeypatch):
    loaded = _observe_loader(monkeypatch)
    b, kh, d, page, pages = 3, 2, 128, 16, 33
    lens = [513, 77, 0]
    torch.manual_seed(918)
    row = dict(input=in_key, output=str(out_dtype), sq=sq, gqa=group, split=split, stats=stats, hnd=hnd, checks=[])
    graph, vp, ws, buf, tensors = paged._run_graph_fp8(
        b,
        kh * group,
        kh,
        d,
        page,
        pages,
        lens,
        hnd,
        in_key=in_key,
        out_dt=out_dtype,
        s_q=sq,
        stats=stats,
        explicit_split=split,
        explicit_pack=True,
        explicit_cga=1,
        v_table_layout="strided",
        q_lens=[sq, max(0, sq - 1), sq],
        return_case=True,
        override=True,
    )
    prepared = graph._compiled_plans[graph._plan_index]._prepared
    assert prepared is not None
    mod = loaded[-1]
    _assert_decode(mod)
    row["geometry"] = dict(m=mod.TILE_N, n=mod.N_Q, threads=mod.CFG.THREADS_PER_CTA, tmem_cols=mod.TMEM_COLS)
    row["workspace_bytes"] = ws.numel()
    row["engine"], knobs = graph.get_engine_and_knobs_at_index(graph._plan_index)
    row["knobs"] = {str(k): int(v) for k, v in knobs.items()}
    owner = prepared.spec.owner

    def reference():
        q = buf["q"].cpu().double()
        k, v = (buf[n].cpu().double() for n in ("k", "v"))  # logical [page, head, row, D]
        kt, vt = (buf[n].cpu().reshape(b, -1).long() for n in ("k_table", "v_table"))
        ql, kl = (buf[n].cpu().reshape(b).tolist() for n in ("seq_q", "seq_kv"))
        dq, dk, dv, so = (buf[n].item() for n in ("descale_q", "descale_k", "descale_v", "scale_o"))
        output = torch.zeros_like(q)
        logs = torch.full((b, kh * group, sq), -torch.inf, dtype=torch.float64)
        for bi in range(b):
            if not ql[bi] or not kl[bi]:
                continue
            kk = k[kt[bi]].permute(1, 0, 2, 3).reshape(kh, -1, d)[:, : kl[bi]].repeat_interleave(group, dim=0)
            vv = v[vt[bi]].permute(1, 0, 2, 3).reshape(kh, -1, d)[:, : kl[bi]].repeat_interleave(group, dim=0)
            scores = q[bi, :, : ql[bi]] @ kk.transpose(-1, -2) * (d**-0.5 * dq * dk)
            output[bi, :, : ql[bi]] = scores.softmax(-1) @ vv * dv
            logs[bi, :, : ql[bi]] = scores.logsumexp(-1)
        return output * so, logs, output.abs().max().item()

    def check(label):
        torch.cuda.synchronize()
        ref, logs, amax = reference()
        got = buf["o"].cpu().double()
        floor = (ref - ref.to(out_dtype).double()).abs().max().item()
        tolerance = max(0.04, 3 * floor) if out_dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else 0.04
        record = dict(
            label=label,
            max_error=(got - ref).abs().max().item(),
            tolerance=tolerance,
            quantization_floor=floor,
            amax_error=abs(buf["amax_o"].item() - amax),
            output_hash=digest(buf["o"]),
            amax_hash=digest(buf["amax_o"]),
        )
        if stats:
            record["stats_hash"] = digest(buf["lse"])
        row["checks"].append(record)
        assert torch.isfinite(got).all(), record
        assert record["max_error"] <= tolerance, record
        assert record["amax_error"] <= 0.03, record
        if stats:
            torch.testing.assert_close(buf["lse"].cpu().double().reshape_as(logs), logs, atol=6e-4, rtol=0)
        return tuple(record[k] for k in ("output_hash", "amax_hash") + (("stats_hash",) if stats else ()))

    def execute():
        graph.execute(vp, ws)

    original = check("eager")
    capture = torch.cuda.CUDAGraph()
    try:
        # Warm the exact public binder, then use its current-stream resolution in capture.
        mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            execute()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
        finally:
            torch.cuda.set_sync_debug_mode(mode)
        # PyTorch capture setup may synchronize; audit the execute body itself.
        with torch.cuda.graph(capture):
            torch.cuda.set_sync_debug_mode("error")
            try:
                execute()
            finally:
                torch.cuda.set_sync_debug_mode(mode)

        def poison():
            buf["o"].view(torch.uint8).fill_(127)
            buf["amax_o"].fill_(999)
            if stats:
                buf["lse"].fill_(torch.nan)

        poison()
        capture.replay()
        assert check("retained") == original
        changed_names = ("descale_q", "descale_k", "descale_v", "scale_o", "v")
        saved = {n: buf[n].clone() for n in changed_names}
        buf["descale_q"].mul_(-0.7)
        buf["descale_k"].mul_(1.3)
        poison()
        capture.replay()
        changed = check("first_changed_qk_scales")
        assert changed[0] != original[0]
        buf["descale_v"].mul_(0.6)
        buf["scale_o"].fill_(1.9)
        poison()
        capture.replay()
        next_changed = check("first_changed_vo_scales")
        assert next_changed[0] != changed[0] and next_changed[1] != changed[1]
        v_before = digest(buf["v"])
        buf["v"].copy_((-buf["v"].float()).to(buf["v"].dtype))
        assert digest(buf["v"]) != v_before, "V negative control must change encoded inputs"
        poison()
        capture.replay()
        assert check("first_changed_v")[0] != next_changed[0]
        for n, v in saved.items():
            buf[n].copy_(v)
        poison()
        capture.replay()
        assert check("restored") == original
        assert prepared.spec.owner is owner
    finally:
        capture.reset()


@pytest.mark.parametrize("feature", ["g3_unpacked", "mha", "q8_mha", "page8", "page256", "head_padding", "d80v112", "causal_window", "sink"])
def test_native_fp8_decode_features(feature, monkeypatch):
    loaded = _observe_loader(monkeypatch)
    group = 3 if feature == "g3_unpacked" else (1 if feature in ("mha", "q8_mha") else 4)
    page = 8 if feature == "page8" else (256 if feature == "page256" else 16)
    kwargs = {"head_padding": True} if feature == "head_padding" else {}
    if feature == "causal_window":
        kwargs.update(causal="bottom_right", window_left=73)
    if feature == "sink":
        kwargs.update(sinks=torch.linspace(-120, 120, 2 * group, device="cuda").view(1, -1, 1, 1))
    sq = 8 if feature == "q8_mha" else 2
    d, dv = (80, 112) if feature == "d80v112" else (128, 128)
    paged._run_graph_fp8(
        3,
        2 * group,
        2,
        d,
        page,
        (513 + page - 1) // page,
        [513, 77, 0],
        False,
        in_key="e4m3",
        out_dt=torch.bfloat16,
        s_q=sq,
        d_v=dv,
        q_lens=[sq, 1, 0],
        explicit_pack=feature != "g3_unpacked",
        explicit_split=1,
        explicit_cga=1,
        **kwargs,
    )
    _assert_decode(loaded[-1])


@pytest.mark.parametrize("sq,group,cga", [(1, 16, 1), (9, 1, 1), (1, 8, 2)])
def test_native_fp8_decode_route_boundaries(sq, group, cga, monkeypatch):
    loaded = _observe_loader(monkeypatch)
    paged._run_graph_fp8(2, 2 * group, 2, 128, 16, 9, [129, 77], True, s_q=sq, explicit_pack=True, explicit_split=1, explicit_cga=cga)
    assert loaded[-1].__file__.endswith("sm100/prefill_d128_fp8.py")


def test_native_fp8_decode_config():
    from cudnn.sdpa.fwd.config_sm100 import TemplateParamsDecode, make_cfg_d128_fp8_decode

    params = TemplateParamsDecode(
        dtype_qkv=0, dtype_o=2, cta_mma=1, paged_kv=True, page_size=16, seq_kv_lens_present=True, decode_q_tile=16, decode_live_columns=8
    )
    cfg, tma = make_cfg_d128_fp8_decode(params)
    assert (cfg.BPE, cfg.BPE_O, cfg.TILE_K_HW, cfg.STAGES_KV, cfg.THD_VARLEN) == (1, 2, 32, 5, 0)
    assert tma.QK_GRANU_ELEMS == tma.VO_GRANU_ELEMS == 128
    for change in [
        dict(dtype_qkv=3),
        dict(dtype_o=5),
        dict(cta_mma=2),
        dict(paged_kv=False, page_size=0),
        dict(thd_varlen=True),
        dict(pv_bf16=True),
        dict(softmax_f16=True),
        dict(dual_kv_stream=True),
        dict(decode_q_tile=32),
        dict(decode_live_columns=16),
        dict(decode_head_dim=256),
        dict(pack_gqa=True, qh_per_kh=16),
        dict(page_size=24),
        dict(seq_kv_lens_present=False),
    ]:
        with pytest.raises(ValueError):
            make_cfg_d128_fp8_decode(replace(params, **change))


def test_native_fp8_decode_keeps_packed_g3_decline():
    # The FP8 public row still requires a group dividing TILE_M. The native
    # lowering must not widen graph eligibility as a side effect.
    with pytest.raises(NotImplementedError, match="PackGQA requires h_q/h_kv to divide tile_m"):
        paged._run_graph_fp8(3, 6, 2, 128, 16, 33, [513, 77, 0], False, s_q=2, explicit_pack=True, explicit_split=1, explicit_cga=1)


@pytest.mark.parametrize(
    "in_key,stats,amax,log2,split",
    [
        ("e4m3", False, False, False, 1),
        ("e5m2", True, False, True, 4),
        ("e4m3", True, True, True, 1),
        ("e5m2", False, False, False, 4),
    ],
)
def test_native_fp8_decode_optional_outputs(in_key, stats, amax, log2, split, monkeypatch):
    from determinism_utils import assert_bitwise, snapshot

    loaded = _observe_loader(monkeypatch)
    graph, vp, ws, buf, _ = paged._run_graph_fp8(
        3,
        8,
        1,
        128,
        16,
        33,
        [513, 77, 0],
        True,
        in_key=in_key,
        out_dt=torch.bfloat16,
        s_q=1,
        stats=stats,
        amax=amax,
        stats_log2=log2,
        explicit_pack=True,
        explicit_split=split,
        explicit_cga=1,
        return_case=True,
    )
    _assert_decode(loaded[-1])
    names = ("o",) + (("lse",) if stats else ()) + (("amax_o",) if amax else ())
    baseline = snapshot({name: buf[name] for name in names})
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            graph.execute(vp, ws)
        buf["o"].fill_(torch.nan)
        if stats:
            buf["lse"].fill_(torch.nan)
        buf["amax_o"].fill_(999)
        capture.replay()
        assert_bitwise({name: buf[name] for name in names}, baseline, "optional-output retained replay")
        if not amax:
            assert buf["amax_o"].item() == 999
    finally:
        try:
            torch.cuda.synchronize()
        finally:
            capture.reset()


@pytest.mark.parametrize("in_key", ["e4m3", "e5m2"])
def test_native_fp8_decode_direct_partial_log_base(in_key, monkeypatch):
    from cudnn.sdpa.fwd import api_dsl

    original = api_dsl._load_sm100_kernel_module
    observed = []

    def direct_params(flavor, params, **kwargs):
        # The normal adapter clears this flag for partials. Exercise a direct
        # template declaration that still requests log2, as allowed by its ABI.
        assert params.split_kv == 4
        mod = original(flavor, replace(params, stats_log2=True), **kwargs)
        assert mod.CFG.STATS_LOG2 and mod.SPLIT_KV == 4
        observed.append(mod)
        return mod

    monkeypatch.setattr(api_dsl, "_load_sm100_kernel_module", direct_params)
    paged._run_graph_fp8(
        3, 8, 1, 128, 16, 33, [513, 77, 0], True, in_key=in_key, out_dt=torch.bfloat16, stats_log2=True, explicit_pack=True, explicit_split=4, explicit_cga=1
    )
    assert observed


def check_architecture_route(monkeypatch):
    from cudnn.sdpa.fwd import api_dsl
    from frost_test_utils import _SM

    original = api_dsl.SdpaFwdDslSm100.template_params
    checked = []

    def inspect(api):
        actual = api._device_cc
        try:
            api._device_cc = (10, 0)
            sm100 = original(api)
            api._device_cc = (10, 3)
            sm103 = original(api)
        finally:
            api._device_cc = actual
        assert sm100.decode_q_tile == 16 and sm100.decode_live_columns == 8
        assert sm103.decode_q_tile == 0
        checked.append(True)
        return original(api)

    monkeypatch.setattr(api_dsl.SdpaFwdDslSm100, "template_params", inspect)
    loaded = _observe_loader(monkeypatch)
    paged._run_graph_fp8(2, 16, 2, 128, 16, 9, [129, 77], True, s_q=1, explicit_pack=True, explicit_split=2, explicit_cga=1)
    assert checked
    expected = "decode_d128_fp8.py" if _SM == 100 else "prefill_d128_fp8.py"
    assert loaded[-1].__file__.endswith("sm100/" + expected)


@pytest.mark.parametrize("dtype_qkv,dtype_o,split", [(0, 0, 1), (1, 2, 4), (0, 3, 4), (1, 0, 1)])
def test_native_fp8_decode_score_release(tmp_path, dtype_qkv, dtype_o, split):
    """The production score-credit address must depend on all eight TMEM words."""
    import os
    from pathlib import Path
    import subprocess
    import sys
    import textwrap

    import warm_stress

    source = textwrap.dedent("""
        import faulthandler, sys
        faulthandler.dump_traceback_later(150, exit=True)
        sys.meta_path[:] = [f for f in sys.meta_path if "editable" not in str(f).lower()]
        sys.path[:0] = sys.argv[1:3]
        from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module
        from cudnn.sdpa.fwd.config_sm100 import TemplateParamsDecode
        params = TemplateParamsDecode(
            dtype_qkv=int(sys.argv[3]), dtype_o=int(sys.argv[4]), split_kv=int(sys.argv[5]),
            cta_mma=1, paged_kv=True, page_size=16, seq_kv_lens_present=True,
            decode_q_tile=16, decode_live_columns=8,
        )
        mod = _load_sm100_kernel_module((128, 128), params, fp8=True, pertensor=True)
        assert mod.__file__.endswith("decode_d128_fp8.py")
        mod.compile_prepared()
    """)
    test_root = Path(__file__).resolve().parents[2]
    source_root = test_root.parents[1] / "python"
    env = dict(
        os.environ,
        CUTE_DSL_DUMP_DIR=str(tmp_path),
        CUTE_DSL_KEEP="ptx,cubin",
        CUTE_DSL_ARCH="sm_100a",
        CUDNN_FRONTEND_DISABLE_COMPILED_CACHE="1",
        CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO="0",
    )
    argv = [sys.executable, "-c", source, str(source_root), str(test_root), str(dtype_qkv), str(dtype_o), str(split)]
    try:
        result = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"score release compile timed out:\n{exc.stdout}\n{exc.stderr}")
    assert result.returncode == 0, result.stdout + result.stderr
    bodies = [p.read_text() for p in tmp_path.glob("*.ptx") if "tcgen05.mma" in p.read_text()]
    assert len(bodies) == 1, f"expected one attention PTX, found {len(bodies)}"
    sites = warm_stress.assert_release_dependence(bodies[0], sites=1, words=8)
    (site,) = [site for site in sites if site.dependent]
    assert site.loads == 1 and site.loaded_registers == 8
