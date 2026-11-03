# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native dense descriptor coordinates preserve each physical B/H/S order."""

from itertools import permutations, product
from types import SimpleNamespace

import pytest

from cudnn.sdpa.fwd.config_sm100 import dense_bind_strides
from cudnn.sdpa.fwd.prepared import _dense_role_layout

pytestmark = [pytest.mark.L0]


@pytest.mark.parametrize("order", list(permutations(range(3))))
@pytest.mark.parametrize("gap", [0, 8, 24])
def test_native_dense_stride_permutations_preserve_addresses(order, gap):
    """Check the native binder against the actual declared element-address set."""
    shape = (2, 3, 5, 16)
    strides = [0, 0, 0, 1]
    span = shape[-1] + gap
    for axis in reversed(order):
        strides[axis] = span
        span = shape[axis] * span + gap
    bound, batch, sequence, need = _dense_role_layout(shape, tuple(strides), 3, 16, 5, 2, 2, True, "q")
    bs, ss, hs = bound
    assert (batch, sequence) == (2, 5)
    declared = {sum(i * st for i, st in zip(index, strides, strict=True)) for index in product(*(range(n) for n in shape))}
    addressed = {b * bs + h * hs + s * ss + d for b, h, s, d in product(*(range(n) for n in shape))}
    assert addressed == declared
    assert len(addressed) == 2 * 3 * 5 * 16
    assert need == max(addressed) + 1


@pytest.mark.parametrize("unused", [0, 3, -7, 2**33 + 1])
def test_native_dense_singleton_batch_does_not_constrain_physical_order(unused):
    assert dense_bind_strides((1, 3, 5, 16), (unused, 80, 16, 1), 2) == (80, 16, 80)


@pytest.mark.parametrize("strides", [(256, 80, 17, 1), (256, 79, 16, 1), (255, 80, 16, 1), (256, 80, 16, 2), (256, 64, 16, 1)])
def test_native_dense_declines_unaligned_or_overlapping_axes(strides):
    assert dense_bind_strides((2, 3, 5, 16), strides, 2) is None


@pytest.mark.parametrize("cc", [(10, 0), (10, 7), (12, 0)])
@pytest.mark.parametrize("role", ["q", "k", "v", "o"])
def test_graph_and_adapter_decline_unaligned_dense_port(cc, role, monkeypatch):
    import torch
    import cudnn
    from cudnn.api_base import TensorDesc
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100, SdpaFwdDslSm120

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: cc)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device=None: SimpleNamespace(shared_memory_per_block_optin=101376, multi_processor_count=68)
    )
    descs = {}
    for name in ("q", "k", "v", "o"):
        # Each declaration covers its full storage; only the selected port's
        # token pitch is unaligned. A decline must name that operand.
        hs = 144
        ss = 4 * hs + 16 + (name == role)
        descs[name] = TensorDesc(
            dtype=torch.bfloat16,
            shape=(2, 4, 128, 128),
            stride=(128 * ss, hs, ss, 1),
            stride_order=(3, 1, 2, 0),
            device=torch.device("cuda:0"),
            name=name,
        )
    api = (SdpaFwdDslSm120 if cc[0] == 12 else SdpaFwdDslSm100)(**{"sample_" + n: desc for n, desc in descs.items()})
    with pytest.raises(NotImplementedError, match=rf"{role}.*stride"):
        api.check_support()

    def ref(desc):
        return SimpleNamespace(get_dim=lambda: desc.shape, get_stride=lambda: desc.stride, get_data_type=lambda: cudnn.data_type.BFLOAT16)

    facts = ga.SdpaGraphFacts(
        b=2,
        h_q=4,
        h_kv=4,
        s_q=128,
        s_kv=128,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.BFLOAT16,
        device_cc=cc,
        scale=0.125,
        **{n + "_t": ref(desc) for n, desc in descs.items()},
    )
    spec = {(10, 0): engines._sm100_spec, (10, 7): engines._sm107_spec, (12, 0): engines._sm120_spec}[cc]()
    reason = engines.mismatch(spec.capabilities, facts)
    assert reason is not None and role in reason.lower() and "stride" in reason.lower()


@pytest.mark.parametrize("heads,token_pitch", [(24, 13312), (32, 17408)])
def test_fused_gate_slab_column_at_batch_one_token_is_declarable(heads, token_pitch):
    """A gate sliced out of a fused projection slab at batch 1 x one query token keeps the slab's token pitch on both
    singleton axes; canonicalized, the layout is BSHD-compact, so the graph row (``gate_layout_ok``) and the standalone
    adapter (``epilogue_gate_layout_declarable``) must both admit it -- one rule, never a plan that dies at check_support."""
    from cudnn.sdpa.fwd.config_sm107 import epilogue_gate_layout_declarable
    from cudnn.sdpa.graph_analyzer import gate_layout_ok

    shape, strides = (1, heads, 1, 256), (token_pitch, 256, token_pitch, 1)
    assert dense_bind_strides(shape, strides, 2) == (heads * 256, heads * 256, 256)
    assert gate_layout_ok(shape, strides, 2) is True
    assert epilogue_gate_layout_declarable(shape, strides, 2) is True


@pytest.mark.parametrize(
    "strides",
    [(512 * 8 * 260, 260, 8 * 260, 1), (512 * 8 * 128, 128, 8 * 128, 1), (512 * 8 * 256, 256, 8 * 256, 2)],
    ids=["unaligned-head-pitch", "overlapping-head-pitch", "strided-d"],
)
def test_fused_gate_layout_rule_still_declines_what_tma_cannot_express(strides):
    """The gate has no copy fallback: an unaligned head pitch, an overlapping head pitch and a non-contiguous head dim stay
    refused by both halves of the rule."""
    from cudnn.sdpa.fwd.config_sm107 import epilogue_gate_layout_declarable
    from cudnn.sdpa.graph_analyzer import gate_layout_ok

    shape = (1, 8, 512, 256)
    assert dense_bind_strides(shape, strides, 2) is None
    assert gate_layout_ok(shape, strides, 2) is False
    assert epilogue_gate_layout_declarable(shape, strides, 2) is False
