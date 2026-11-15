# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quantized records retain their fixed body, numerics and per-call buffer binding."""

import pytest

from frost_test_utils import requires_dsl, requires_rubin

pytestmark = [pytest.mark.L0, requires_dsl, requires_rubin]


@pytest.mark.parametrize("family,diagnostic", [("fp8", False), ("fp8", True), ("mxfp8", False), ("mxfp8", True)])
@pytest.mark.parametrize("hkv", [2, 4], ids=["gqa", "mha"])
def test_fixed_record_rebuild_rebind_and_changed_input_capture(monkeypatch, family, diagnostic, hkv):
    import torch

    import test_sdpa_bwd_fp8_sm107 as fp8
    import test_sdpa_bwd_mxfp8_sm107 as mx
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3
    from cudnn.sdpa.bwd import api_dsl_sm107 as shared
    from cudnn.sdpa.bwd import config_sm107 as cfg
    from determinism_utils import assert_bitwise, assert_golden, snapshot

    record_id = 20605 if family == "fp8" else 20606
    expected_path = f"sm107/d256_{family}/kernel.py"
    seen, original = [], shared.load_template

    def spy(path, params, tag="template"):
        if tag == f"sdpa_bwd_sm107_main_{family}":
            seen.append((path.rsplit("/kernels/", 1)[-1], params.datapath_2x2_profile))
        return original(path, params, tag)

    monkeypatch.setattr(shared, "load_template", spy)
    monkeypatch.setattr(shared, "FP8_DS_DTYPE", DTYPE_BF16 if diagnostic else DTYPE_E4M3)
    monkeypatch.setattr(shared, "MXFP8_DS_SF_POLICY", cfg.DS_SF_P_C if diagnostic else cfg.DS_SF_P_B)

    def run(record):
        options = dict(hq=4, hkv=hkv, sq=257, skv=257, causal=True, seed=34605, record=record)
        result = fp8._run_fp8(grad_dtype=torch.bfloat16, **options).check() if family == "fp8" else mx._run_mxfp8(**options)
        return result

    first = run((record_id, {}))
    record = first.graph.get_engine_and_knobs_at_index(0)
    assert record == (record_id, {})
    # Unrelated legacy half-body switches must not redirect either quantized record.
    monkeypatch.setattr(shared, "BWD_D256_2X2", True)
    monkeypatch.setattr(shared, "BWD_D256_2X2_PROFILE", 1)
    second = run(record)
    assert seen == [(expected_path, 0), (expected_path, 0)]
    assert_bitwise(second.outs[0], first.outs[0], "fresh record rebuild")
    if family == "fp8":
        assert_bitwise(second.amax[0], first.amax[0], "fresh record amax")
    assert {t.data_ptr() for t in first.pack.values()}.isdisjoint({t.data_ptr() for t in second.pack.values()})

    # Rebind every port on the same compiled graph, including all scalar/SF ports.
    pack = {port: value.clone(memory_format=torch.preserve_format) for port, value in second.pack.items()}
    expected = snapshot({str(port.get_uid()): value for port, value in pack.items()})
    workspace = torch.empty_like(second.workspace).fill_(0xBD)
    second.graph.execute(pack, workspace)
    assert_bitwise({str(port.get_uid()): value for port, value in pack.items()}, expected, "rebound inputs and outputs")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        second.graph.execute(pack, workspace)
    stream.synchronize()
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture, stream=stream):
        second.graph.execute(pack, workspace)

    # Zero all representations of the upstream gradient after capture. Every
    # gradient and requested amax must then be zero, despite poisoned outputs.
    for port, value in pack.items():
        name = port.get_name()
        if name in ("dO", "dO_T", "dO_f16"):
            value.zero_()
    output_uids = {port.get_uid() for port in second.graph.get_outputs()}
    output_ports = [port for port in pack if port.get_uid() in output_uids]
    assert len(output_ports) == (7 if family == "fp8" else 3)
    for _ in range(4):
        for port in output_ports:
            pack[port].fill_(float("nan"))
        workspace.fill_(0xBD)
        capture.replay()
        torch.cuda.synchronize()
        for port in output_ports:
            assert torch.count_nonzero(pack[port]) == 0, port.get_name()

    inputs = {port.get_name(): first.pack[port] for port in first.graph.get_inputs()}
    outputs = dict(first.outs[0])
    if family == "fp8":
        outputs.update({"amax_" + name: value for name, value in first.amax[0].items()})
    assert_golden(f"sdpa_bwd_sm107_fixed_{family}_diagnostic{int(diagnostic)}_hkv{hkv}", inputs, outputs)
