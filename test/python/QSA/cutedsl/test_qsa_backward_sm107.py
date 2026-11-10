# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SM107 kernels of the QSA backward engine (``cudnn.QSA.kernels.bprop_sm107``) against the reference and against
the SM100 kernels, which serve SM107 when the driver has no oversized shared memory carveout."""

import pytest
import torch

from QSA.cutedsl import qsa_index_list_reference as ref
from QSA.cutedsl.test_qsa_backward_sm100 import _bitmap_backward, _check, _inputs, _miles_metadata, _skip_cu12, supported_device  # noqa: F401

pytestmark = pytest.mark.L0


@pytest.fixture
def rubin():
    from cudnn.QSA.kernels import bprop_sm107

    if torch.cuda.get_device_capability() != (10, 7) or not bprop_sm107.supported(torch.cuda.current_device()):
        pytest.skip("the SM107 backward kernels need SM107 and the oversized shared memory carveout of a CUDA 13.4 driver")


@pytest.mark.parametrize("ds_precision", ["fp32", "bf16"])
@pytest.mark.parametrize("h_kv, group", [(2, 12), (1, 12), (1, 6)])
@pytest.mark.parametrize("lengths", [(4099,), (3, 2051, 22529)])
def test_sm107_kernels_match_the_sm100_kernels(rubin, monkeypatch, lengths, h_kv, group, ds_precision):
    # the dense pair kernel alone, then all three kernels with packed documents off the 4-token grid
    from cudnn.QSA import _backward as engine
    from cudnn.QSA.kernels import bprop_sm100, bprop_sm107

    _skip_cu12(h_kv)
    monkeypatch.setattr(ref, "GROUP", group)
    inputs = _inputs(lengths, h_kv, seed=sum(lengths) + group)
    meta = _miles_metadata(inputs["topk_idxs"], lengths)
    rubin_hosts = {bprop_sm107._sparse_query_host, bprop_sm107._sparse_key_host, bprop_sm107._dpair_host}
    compile_cached = engine.compile_cached
    hosts = []

    def recording_compile(host, *args, **kwargs):
        # a compiled cache hit never reaches cute.compile, so the hosts are recorded where the engine requests them
        hosts.append(host)
        return compile_cached(host, *args, **kwargs)

    monkeypatch.setattr(engine, "compile_cached", recording_compile)
    monkeypatch.setattr(engine, "_PLANS", {})
    sm107 = _bitmap_backward(inputs, meta, ds_precision=ds_precision)
    assert engine._PLANS and all(plan.rubin for plan in engine._PLANS.values())
    assert (rubin_hosts if sum(lengths) > 22527 else {bprop_sm107._dpair_host}) <= set(hosts)
    assert {bprop_sm100._sparse_query_host, bprop_sm100._sparse_key_host, bprop_sm100._dpair_host}.isdisjoint(hosts)
    hosts.clear()
    monkeypatch.setattr(engine, "_PLANS", {})
    monkeypatch.setattr(bprop_sm107, "supported", lambda device_index: False)
    sm100 = _bitmap_backward(inputs, meta, ds_precision=ds_precision)
    assert engine._PLANS and not any(plan.rubin for plan in engine._PLANS.values())
    assert bprop_sm100._dpair_host in hosts and rubin_hosts.isdisjoint(hosts)
    _check(inputs, sm107)
    _check(inputs, sm100)
    for name, actual, expected in zip(("dq", "dk", "dv"), sm107, sm100):
        ref.assert_grads_close(name, actual, expected)
