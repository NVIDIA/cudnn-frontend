# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native bulk COMBINE coverage, independent dequantized references and atomic controls.

The aligned 2-CTA case retains PR #1942's memcheck reproducer geometry. Run
with compute-sanitizer --tool memcheck --error-exitcode 86 and pytest -n0.
"""

from dataclasses import replace
from types import SimpleNamespace

import cudnn
import pytest
import torch

from cudnn.gemm.frost import compiler as C
from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.gemm.frost.tile_config import by_name
from gemm_test_utils import E2M1, Plan, graph_binding, requires_sm100, requires_sm107, to_blocked, unpack_fp4, vp_bs, with_static_segmented_capacity
from moe_block_scale_test_utils import _build_graph

pytestmark = pytest.mark.L0

_NORMAL = "CONFIG_sm100_128x256x128_128x256x32_cluster1x1_1ctamma"
_PAIR = "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma"
_SWAPPED = "CONFIG_sm100_256x256x128_128x256x64_cluster2x1_2ctamma"
_DTYPES = [(cudnn.data_type.BFLOAT16, torch.bfloat16), (cudnn.data_type.HALF, torch.float16), (cudnn.data_type.FLOAT, torch.float32)]


@pytest.mark.parametrize("output_dt,_torch_dt", _DTYPES)
def test_normal_two_cta_combine_uses_atomics(monkeypatch, output_dt, _torch_dt):
    """The SM100 memcheck failure cannot be re-admitted by output dtype or arch."""
    chain = analyze(_build_graph(2, 1026, 512, 1024, 6, output_dt=output_dt, combine_top_k=2))
    for arch in (100, 103, 107):
        monkeypatch.setattr(C, "_current_arch", lambda: arch)
        assert C._use_bulk_combine(chain, by_name(_NORMAL))
        assert not C._use_bulk_combine(chain, by_name(_PAIR))
        assert C._combine_smem_bytes(chain, by_name(_PAIR)) == 0


def _make_inputs(rows, features, k, tokens, dtype, bounds, padding=0):
    """Build small exact FP4 values and independently addressable routed data."""

    def packed(shape):
        """Pack two nonnegative E2M1 values per byte."""
        return torch.randint(0, 4, shape, dtype=torch.uint8, device="cuda") * 2 | torch.randint(0, 4, shape, dtype=torch.uint8, device="cuda") * 32

    a8, b8 = packed((1, rows, k // 2)), packed((2, features, k // 2))
    lut = torch.tensor(E2M1, dtype=torch.float32, device="cuda")
    sa = (torch.randint(1, 3, (rows, k // 16), device="cuda").float() / 32).to(torch.float8_e4m3fn)
    sb = (torch.randint(1, 3, (2, features, k // 16), device="cuda").float() / 32).to(torch.float8_e4m3fn)
    live = torch.cat([to_blocked(sa[lo:hi]).view(torch.uint8) for lo, hi in zip(bounds, bounds[1:])]).view(sa.dtype)
    sfa = with_static_segmented_capacity(live, rows, len(bounds) - 1, k // 16)
    sfb = torch.stack([to_blocked(sf).view(-1) for sf in sb]).view(2, k // 16, (features + 127) // 128 * 128)
    dest = torch.randperm(rows, device="cuda", dtype=torch.int32)
    index, slots = (dest // 2).view(1, rows, 1), (dest % 2).view(1, rows, 1)
    # Both invalid index directions and invalid slots must be ignored.
    index[0, 3, 0], index[0, 7, 0] = -1, tokens
    slots[0, 5, 0], slots[0, 11, 0] = -1, 2
    scores = torch.randint(0, 5, (1, tokens, 2), device="cuda").float() / 4
    # A leading guard also shifts the pointer for the unaligned-output case.
    guard = 1 if padding else 8
    storage = torch.full((guard + tokens * (features + padding) + 8,), -123, dtype=dtype, device="cuda")
    output = storage[guard : guard + tokens * (features + padding)].view(1, tokens, features + padding)[..., :features]
    mask = torch.ones_like(storage, dtype=torch.bool)
    mask.as_strided(output.shape, output.stride(), storage_offset=guard).fill_(False)
    return SimpleNamespace(
        a=a8.view(torch.float4_e2m1fn_x2),
        b=b8.view(torch.float4_e2m1fn_x2),
        a_ref=unpack_fp4(a8, lut).view(rows, k) * sa.float().repeat_interleave(16, 1),
        b_ref=unpack_fp4(b8, lut).view(2, features, k) * sb.float().repeat_interleave(16, 2),
        sfa=sfa,
        logical_sfa=sa,
        sfb=sfb,
        index=index,
        slots=slots,
        scores=scores,
        bounds=bounds,
        offsets=torch.tensor(bounds, dtype=torch.int32, device="cuda"),
        output=output,
        storage=storage,
        padding_mask=mask,
    )


def _variant_pack(plan, data, output=None):
    """Bind the original graph order even when the kernel swaps its operands."""
    binding = graph_binding(plan)
    pack = vp_bs(plan, data.a, data.b, data.output if output is None else output, data.sfa, data.sfb, fto=data.offsets)
    pack.update({binding.token_index: data.index, binding.token_ks: data.slots, binding.top_k_scores: data.scores})
    return pack


def _check(data):
    """Compare every output with FP32-dequantized GEMMs and check storage guards."""
    output = data.output
    expected = torch.zeros_like(output[0], dtype=torch.float32)
    for group, (lo, hi) in enumerate(zip(data.bounds, data.bounds[1:])):
        ids, slots = data.index.flatten()[lo:hi].long(), data.slots.flatten()[lo:hi].long()
        valid = (ids >= 0) & (ids < output.shape[1]) & (slots >= 0) & (slots < 2)
        contribution = data.a_ref[lo:hi] @ data.b_ref[group % 2].T
        weighted = (contribution[valid] * data.scores[0, ids[valid], slots[valid], None]).to(output.dtype).float()
        expected.index_add_(0, ids[valid], weighted)
    torch.testing.assert_close(output[0].float(), expected.to(output.dtype).float(), atol=1e-5, rtol=2 * torch.finfo(output.dtype).eps)
    assert torch.count_nonzero(output[0][expected.abs().sum(1) == 0]) == 0
    assert torch.all(data.storage[data.padding_mask] == -123), "COMBINE modified output padding or guards"


def _exercise(config_name, output_dt, dtype, features, padding=0, swap=False, rows=258, k=256, scale_replay=False):
    """Check native admission, atomic control, changed-input replay and rebinding."""
    torch.manual_seed(1942)
    bounds = [0, 1, 129, 129, rows - 1, rows - 1, rows - 1]
    if rows == 1026:
        bounds = [0, 1, 257, 257, 769, 1026, 1026]
    graph = _build_graph(2, rows, features, k, 6, output_dt=output_dt, combine_top_k=2)
    cfg = replace(by_name(config_name), swap_ab=swap)
    plan = Plan(graph, config=cfg)
    source = plan.generated_path.read_text()
    regular_bulk = not swap and cfg.cta_group == 1
    assert f"moe_combine_bulk_enabled = {regular_bulk}" in source
    assert f"moe_combine_swap_bulk_enabled = {swap}" in source
    data = _make_inputs(rows, features, k, rows // 2 + 7, dtype, bounds, padding)
    pack = _variant_pack(plan, data)
    workspace = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda")

    def launch():
        """Enqueue into caller-owned workspace on the current capture stream."""
        plan._compiled(pack, workspace=workspace, stream=torch.cuda.current_stream().cuda_stream)

    launch()
    _check(data)
    atomic = Plan(graph, config=cfg, force_stg_epi=True)
    atomic_output = torch.empty_like(data.output)
    atomic(_variant_pack(atomic, data, atomic_output))
    torch.testing.assert_close(data.output, atomic_output, atol=1e-5, rtol=2 * torch.finfo(dtype).eps)
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            launch()
        data.scores.copy_(data.scores.flip(2) + 0.125)
        valid = (data.index >= 0) & (data.index < data.output.shape[1])
        data.index[valid] = data.output.shape[1] - 1 - data.index[valid]
        data.output.fill_(99)
        capture.replay()
        _check(data)
        if scale_replay:
            data.bounds = [0, 3, 131, 131, rows - 1, rows - 1, rows - 1]
            data.offsets.copy_(torch.tensor(data.bounds, dtype=torch.int32, device="cuda"))
            data.logical_sfa.copy_((data.logical_sfa.float() * 2).to(data.logical_sfa.dtype))
            data.a_ref.mul_(2)
            live = torch.cat([to_blocked(data.logical_sfa[lo:hi]).view(torch.uint8) for lo, hi in zip(data.bounds, data.bounds[1:])]).view(data.sfa.dtype)
            data.sfa.fill_(1)
            data.sfa.view(-1)[: live.numel()].copy_(live)
            capture.replay()
            _check(data)
        capture.replay()
        _check(data)
    finally:
        capture.reset()
    # Runtime output/token count is independent of routed capacity; reuse the
    # same compiled artifact with fresh storage and a smaller output extent.
    smaller = _make_inputs(rows, features, k, rows // 2 - 5, dtype, bounds, padding)
    pack = _variant_pack(plan, smaller)
    launch()
    _check(smaller)


@requires_sm100
@pytest.mark.parametrize("output_dt,dtype", _DTYPES)
@pytest.mark.parametrize("features,padding", [(256, 0), (520, 0), (264, 1)], ids=["aligned", "tail", "unaligned"])
def test_bulk_combine(output_dt, dtype, features, padding):
    """Cover all normal bulk output dtypes and both atomic fallback predicates."""
    _exercise(_NORMAL, output_dt, dtype, features, padding)


@requires_sm100
@pytest.mark.parametrize("output_dt,dtype", _DTYPES)
def test_two_cta_atomic_fallback(output_dt, dtype):
    """Retain the reported SM100 2-CTA geometry as a native sanitizer target."""
    _exercise(_PAIR, output_dt, dtype, 512, rows=1026, k=1024)


@requires_sm107
@pytest.mark.parametrize("features,padding", [(256, 0), (520, 0), (264, 1)], ids=["aligned", "tail", "unaligned"])
def test_swapped_bulk_combine(features, padding):
    """Exercise the separately qualified SM107 swapped BF16 bulk route."""
    _exercise(_SWAPPED, cudnn.data_type.BFLOAT16, torch.bfloat16, features, padding, swap=True)


@requires_sm107
@pytest.mark.parametrize("swap", [False, True])
def test_bulk_combine_segmented_scale_replay(swap):
    """Change group boundaries and scales in exactly the declared SF capacity."""
    _exercise(_SWAPPED if swap else _NORMAL, cudnn.data_type.BFLOAT16, torch.bfloat16, 256, swap=swap, scale_replay=True)
