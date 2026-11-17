# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-plan repeatability and batch invariance of SM80 dense CU readers."""

import pytest
import torch

from dense_cu_lengths import check_reference, execute, make_case, set_lengths
from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from frost_test_utils import requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(torch.cuda.get_device_capability() != (8, 0), reason="requires native SM80")]


@pytest.mark.parametrize("dims,splits,pack_gqa", [((64, 64), 1, False), ((256, 256), 1, False), ((576, 512), 1, False), ((128, 128), 3, True)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_dense_cumulative_batch_invariance(dims, splits, pack_gqa, dtype):
    single = make_case(dims, dtype, (True, True), splits=splits, pack_gqa=pack_gqa, batch=1)
    set_lengths(single, [17], [65])
    original = {n: single.bufs[n].clone() for n in ("q", "k", "v")}

    def fresh():
        for name in ("o", "stats"):
            single.bufs[name] = torch.empty_like(single.bufs[name])
            single.pack[single.refs[name]] = single.bufs[name]
        execute(single)
        return {"o": single.bufs["o"][0], "lse": single.bufs["stats"][0]}

    expected = assert_replays(fresh, repeats=4)
    check_reference(single, [17], [65])
    for name, value in original.items():
        assert torch.equal(value, single.bufs[name])
    group = make_case(dims, dtype, (True, True), splits=splits, pack_gqa=pack_gqa, batch=129)
    for order in ([0] + [-1] * 128, [1, 0, 0] + [-1] * 126, [1, 2] * 64 + [0]):
        for name in ("q", "k", "v"):
            group.bufs[name].fill_(float("nan"))
            for index, identity in enumerate(order):
                if identity >= 0:
                    group.bufs[name][index].copy_(original[name][0] * (identity + 1))
        qlens = [17 if identity >= 0 else 0 for identity in order]
        klens = [65 if identity >= 0 else 0 for identity in order]
        set_lengths(group, qlens, klens)
        execute(group)
        for index, identity in enumerate(order):
            if identity == 0:
                assert_bitwise(snapshot({"o": group.bufs["o"][index], "lse": group.bufs["stats"][index]}), expected, f"batch slot {index}")
        empty = [i for i, identity in enumerate(order) if identity < 0]
        if empty:
            assert torch.all(group.bufs["o"][empty] == 0)
            assert torch.all(group.bufs["stats"][empty] == -torch.inf)
    group.bufs["k"][0].neg_()
    group.bufs["v"][0].add_(0.75)
    execute(group)
    assert_bitwise(snapshot({"o": group.bufs["o"][-1], "lse": group.bufs["stats"][-1]}), expected, "changed neighbor")
    inputs = {n: t[0].cpu() for n, t in original.items()}
    inputs.update({n: single.bufs[n].cpu() for n in ("q_len", "kv_len")})
    if "sink" in single.bufs:
        inputs["sink"] = single.bufs["sink"].cpu()
    assert_golden(f"sdpa-sm80-dense-cu-d{dims[0]}v{dims[1]}-split{splits}-{dtype}", inputs, expected)
