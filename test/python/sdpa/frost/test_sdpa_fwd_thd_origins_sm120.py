# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""SM120 half THD addresses physical ports independently from logical lengths."""

import pytest
import torch

from frost_test_utils import requires_blackwell_geforce, requires_dsl
from sdpa.frost.test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as _check_origins

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell_geforce]


@pytest.mark.parametrize("dims", [64, 128, (192, 128), 256, 512, (576, 512)])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("stats", ["token", "head"])
def test_independent_origins_rebind_replay(dims, dtype, stats):
    # Mixed INT32/INT64 element/token offsets, independent Q/K/V/O/Stats gaps,
    # poisoned inputs/scratch, changed pitches and retained live/empty replay.
    _check_origins(dims, dtype, stats, True, False, 1, engine_name="sdpa_fwd_prefill_sm120", extra_checks=_check_invariance)


def _check_invariance(graph, nodes, values, state, workspace, overrides, flat, stats_layout):
    ql, kl, origins = state
    saved = {name: tensor.clone() for name, tensor in values.items()}
    pack = {nodes[name]: tensor for name, tensor in values.items()}

    def bits(tensor):
        return tensor.contiguous().view(torch.uint8).clone()

    def live(name, batch):
        view = flat(values[name]) if name == "o" else (values[name].T if stats_layout == "head" else values[name])
        return view[origins[name][batch] : origins[name][batch] + ql[0]]

    expected = {name: bits(live(name, 0)) for name in ("o", "stats")}
    full = {name: bits(values[name]) for name in ("o", "stats")}

    def restore():
        for name, tensor in values.items():
            tensor.copy_(saved[name])

    def repack_scales(order):
        # MXFP8 data origins can move independently; scales remain packed in
        # logical sequence order. Copy existing tile bytes, never requantize.
        for name, lengths in (("q", ql), ("k", kl), ("v", kl)):
            key = "sf_" + name
            if key not in values:
                continue
            starts = [0]
            for length in lengths:
                starts.append(starts[-1] + (length + 127) // 128)
            values[key].zero_()
            dst = 0
            for index in order:
                count = starts[index + 1] - starts[index]
                values[key][:, dst : dst + count].copy_(saved[key][:, starts[index] : starts[index + 1]])
                dst += count

    def run():
        values["o"].fill_(-17)
        values["stats"].fill_(-29)
        workspace.fill_(255)
        graph.execute(pack, workspace, **overrides)
        for name in expected:
            assert torch.equal(bits(live(name, 0)), expected[name]), name

    try:
        # Exact bytes include output gaps, signed zeros and infinity sentinels.
        for _ in range(2):
            run()
            for name in full:
                assert torch.equal(bits(values[name]), full[name]), name
        # Only metadata changes: the same sequence moves to a different batch
        # slot, while each output retains its independent physical address.
        permutation = torch.tensor([2, 0, 3, 1], device="cuda")
        for name in ("ql", "kl"):
            values[name].copy_(saved[name][permutation])
        for name in ("q", "k", "v", "o", "stats"):
            values["ro_" + name][:-1].copy_(saved["ro_" + name][:-1][permutation])
        repack_scales((2, 0, 3, 1))
        run()
        for name in full:
            assert torch.equal(bits(values[name]), full[name]), name
        restore()
        # Duplicate Q/K/V read origins into a different sequence; writes remain
        # disjoint and the duplicate must compute the same bits.
        values["ql"][2] = ql[0]
        values["kl"][2] = kl[0]
        for name in ("q", "k", "v"):
            values["ro_" + name][2].copy_(values["ro_" + name][0])
        repack_scales((0, 1, 0, 3))
        run()
        for name in expected:
            assert torch.equal(bits(live(name, 2)), expected[name]), name
        restore()
        values["ql"][1:] = 0
        values["kl"][1:] = 0
        repack_scales((0,))
        run()
        restore()
        for name in ("q", "k", "v"):
            length = ql[2] if name == "q" else kl[2]
            live_input = flat(values[name])[origins[name][2] : origins[name][2] + length]
            live_input.copy_(live_input.float() * 0.5)
        run()
    finally:
        restore()


@pytest.mark.parametrize("dims", [64, 128, (192, 128), 256, 512, (576, 512)])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_origin_golden(dims, dtype):
    from determinism_utils import assert_golden

    record = {}

    def collect(graph, nodes, values, state, workspace, overrides, flat, stats_layout):
        _check_invariance(graph, nodes, values, state, workspace, overrides, flat, stats_layout)
        ql, kl, origins = state
        if origins["q"][0] == 0:
            record["inputs"] = {
                name: flat(values[name])[origins[name][0] : origins[name][0] + (ql[0] if name == "q" else kl[0])].clone() for name in ("q", "k", "v")
            }
            record["outputs"] = {
                "o": flat(values["o"])[origins["o"][0] : origins["o"][0] + ql[0]].clone(),
                "stats": values["stats"][origins["stats"][0] : origins["stats"][0] + ql[0]].clone(),
            }

    _check_origins(dims, dtype, "token", True, False, 1, engine_name="sdpa_fwd_prefill_sm120", extra_checks=collect)
    d, dv = dims if isinstance(dims, tuple) else (dims, dims)
    # The complete reference, replay and batch checks finish before recording.
    assert_golden(f"sdpa-sm120-thd-origins-d{d}v{dv}-{dtype}", record["inputs"], record["outputs"])
