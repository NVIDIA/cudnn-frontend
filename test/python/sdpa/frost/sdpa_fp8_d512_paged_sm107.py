# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""SM107 D512 FP8 page-pool checks, collected by the architecture entrypoint."""

import pytest
import torch


@pytest.mark.parametrize("hnd", [False, True])
@pytest.mark.parametrize("split", [1, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def paged_capture(hnd, split, dtype, monkeypatch):
    from test_sdpa_prepared_fp8_paged import test_prepared_fp8_paged_strides_and_capture
    from cudnn.sdpa.fwd import api_dsl

    original = api_dsl._load_kernel_template
    observed = []

    def checked(path, *args, **kwargs):
        if "prefill_d512" in path:
            assert path == "sm107/prefill_d512_fp8.py"
            observed.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(api_dsl, "_load_kernel_template", checked)
    test_prepared_fp8_paged_strides_and_capture(hnd, split, dtype, 512, monkeypatch)
    assert observed, "the plan must compile the native SM107 D512 body"


def native_paged_contract():
    from dataclasses import replace
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    from cudnn.sdpa.fwd.config_sm107 import make_cfg_d512, make_cfg_d512_mxfp8
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module

    params = TemplateParams(dtype_qkv=0, dtype_o=2, paged_kv=True, page_size=16, seq_kv_lens_present=True)
    mod = _load_sm100_kernel_module((512, 512), params, fp8=True, pertensor=True, rubin=True)
    assert "sm107" in mod.__name__
    assert mod.DESC_VERSION == 1
    assert (mod.CFG.PAGED_KV, mod.CFG.PAGE_SIZE) == (1, 16)
    assert (mod.K_BOX_ROWS, mod.V_BOX_ROWS) == (16, 16)
    assert (mod.CFG.STAGES_KV, mod.CFG.XFER_STAGES) == (3, 4)
    for page in (8, 16, 32, 64, 128, 256):
        assert make_cfg_d512(replace(params, page_size=page))[0].PAGE_SIZE == page
    for invalid in (4, 24, 192):
        with pytest.raises(ValueError, match="page_size"):
            make_cfg_d512(replace(params, page_size=invalid))
    with pytest.raises(ValueError, match="dense Q"):
        make_cfg_d512(replace(params, thd_varlen=True))
    with pytest.raises(ValueError, match="per-tensor FP8"):
        make_cfg_d512(replace(params, dtype_qkv=2))
    with pytest.raises(ValueError, match="paged_kv"):
        make_cfg_d512_mxfp8(params)

    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.fwd import engines
    import cudnn

    caps = next(spec.capabilities for spec in engines.ENGINE_SPECS if spec.name == engines.engine_name(arch="sm107", fp8=True))
    facts = ga.SdpaGraphFacts(
        b=2,
        h_q=4,
        h_kv=2,
        s_q=16,
        s_kv=256,
        d_qk=512,
        d_v=512,
        dtype=cudnn.data_type.FP8_E4M3,
        dtype_o=cudnn.data_type.BFLOAT16,
        is_fp8=True,
        device_cc=(10, 7),
        has_paged_kv=True,
        page_size=16,
        padded=True,
    )
    for dims in ((512, 512), (384, 320)):
        selected = replace(facts, d_qk=dims[0], d_v=dims[1])
        assert engines.mismatch(caps, selected) is None
    for dims in ((64, 64), (128, 128), (192, 128), (256, 256)):
        reason = engines.mismatch(caps, replace(facts, d_qk=dims[0], d_v=dims[1]))
        assert reason is not None and "paged KV is wired" in reason
    assert "THD" in engines.mismatch(caps, replace(facts, thd=True))


@pytest.mark.parametrize("split", [1, 4])
@pytest.mark.parametrize("in_key,output", [("e4m3", torch.bfloat16), ("e5m2", torch.float8_e5m2)])
def paged_half_softmax(split, in_key, output):
    import cudnn
    from test_sdpa_fwd_paged_sm100 import _run_graph_fp8

    _run_graph_fp8(
        3,
        8,
        2,
        512,
        64,
        5,
        [257, 0, 129],
        True,
        in_key=in_key,
        out_dt=output,
        s_q=2,
        explicit_split=split,
        explicit_pack=True,
        causal="bottom_right",
        window_left=128,
        softmax_precision=cudnn.data_type.HALF,
    )


@pytest.mark.parametrize("split,in_key,output", [(1, "e4m3", torch.bfloat16), (4, "e5m2", torch.float8_e5m2)])
def paged_half_golden(split, in_key, output):
    import cudnn
    from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
    from test_sdpa_fp8_d512_paged_determinism import _runner

    run, inputs = _runner(split=split, hnd=True, in_key=in_key, dtype=output, softmax_precision=cudnn.data_type.HALF)
    expected = assert_replays(lambda: run([0]), repeats=4)
    target = {name: expected[name] for name in ("o", "lse")}
    for order in ([1, 0, 2], [2, 1, 0], [0, 1, 0, 2, 3], [4, 0, 3], [1] * 64 + [0] + [2] * 64):
        actual = snapshot(run(order))
        for row, sequence in enumerate(order):
            if sequence == 0:
                assert_bitwise({name: actual[name][row : row + 1] for name in target}, target, f"HALF batch {order}, row {row}")
    actual = assert_replays(lambda: run([0, 1, 2]), repeats=3)
    assert_golden(f"sdpa-fp8-d512-paged-half-{in_key}-{output}-split{split}", inputs, actual)
