# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V-tail zero stores must complete before every compute warp reads V."""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from frost_test_utils import requires_blackwell_geforce, requires_dsl
from sdpa.frost import test_sdpa_fwd_fp8_sm120 as fp8
from sdpa.frost import test_sdpa_fwd_mxfp8_sm120 as mxfp8

pytestmark = [pytest.mark.L0, requires_blackwell_geforce, requires_dsl]


def _logical_v_scales(scales):
    # The columnwise F8_128x4 atom columns span all B*H scale rows. Rebatch
    # logical bytes, then swizzle again; concatenating physical B slices is wrong.
    b, h, blocks, d = scales.shape
    columns = b * h * blocks
    return scales.reshape(d // 128, columns // 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(d, columns).T.reshape(b, h, blocks, d)


def _pack_v_scales(values):
    from sdpa.mxfp8_quant import swizzle_sf_columnwise

    logical = torch.cat([_logical_v_scales(value) for value in values], dim=0)
    b, h, blocks, d = logical.shape
    return swizzle_sf_columnwise(logical.reshape(b * h * blocks, d)).reshape(b, h, blocks, d)


def _exercise(mode, dim, in_key):
    torch.manual_seed(807)
    qkv = tuple(torch.randn(2, h, s, dim, device="cuda").clamp(-2, 2) * 0.125 for h, s in ((4, 33), (2, 80), (2, 80)))
    # Keep the per-tensor quantization scale fixed across batch changes. Each
    # sequence has the same exact amax anchor inside its live input region.
    for value in qkv:
        value[:, 0, 0, 0] = 1
    tiles = (64, 64) if dim == 512 else (64, 128)
    input_dtype = getattr(torch, "float8_" + ("e4m3fn" if in_key == "e4m3" else "e5m2"))
    changed = tuple(value[1:2].clone() * 0.5 for value in qkv)
    for value in changed:
        value[0, 0, 0, 0] = 1
    sequences = [tuple(value[i : i + 1] for value in qkv) for i in range(2)] + [changed]
    # Quantize every sequence once; only repack its saved payload and scale bytes.
    quantized = [
        (
            tuple(fp8._quant(value, input_dtype) for value in seq)
            if mode == "fp8"
            else tuple(mxfp8._quantize(value, input_dtype, columnwise=i == 2) for i, value in enumerate(seq))
        )
        for seq in sequences
    ]

    def cat_payload(values):
        return torch.cat([value.view(torch.uint8) for value in values], dim=0).view(values[0].dtype)

    def run(indices):
        state = {}
        packed = []
        for port in range(3):
            values = [quantized[i][port] for i in indices]
            if mode == "fp8":
                assert all(value[1] == values[0][1] for value in values)
                packed.append((cat_payload([value[0] for value in values]), values[0][1]))
            else:
                packed.append(
                    mxfp8._Quantized(
                        cat_payload([value.data for value in values]),
                        _pack_v_scales([value.sf for value in values]) if port == 2 else torch.cat([value.sf for value in values], dim=0),
                        torch.cat([value.dequant for value in values], dim=0),
                        (len(indices), *values[0].sf_dims[1:]),
                    )
                )
        options = dict(
            quantized_qkv=tuple(packed),
            seq_lens_kv=[17] * len(indices),
            sdpa_kwargs={},
            tiles=tiles,
            pack_gqa=False,
            split_kv=1,
            capture=state,
        )
        if mode == "fp8":
            result = fp8._run(
                len(indices),
                4,
                2,
                33,
                80,
                D=dim,
                scale=dim**-0.5,
                io_dtype=input_dtype,
                o_dtype=torch.bfloat16,
                poison_kv_pad=True,
                **options,
            )
            fp8._check(*result)
        else:
            result = mxfp8._run(
                len(indices),
                4,
                2,
                33,
                80,
                d_qk=dim,
                in_key=in_key,
                out_dt=torch.bfloat16,
                poison_pad=True,
                **options,
            )
            mxfp8._check(result, d_qk=dim)
        return state

    base = run([0])
    inputs = {name: value.clone() for name, value in base["inputs"].items()}

    def repeat():
        for value in base["outputs"].values():
            value.fill_(float("nan"))
        base["run"]()
        return base["outputs"]

    expected = assert_replays(repeat, repeats=3, allowed_inf={})
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            base["run"]()
        for _ in range(3):
            for value in base["outputs"].values():
                value.fill_(float("nan"))
            graph.replay()
            assert_bitwise(base["outputs"], expected, "retained graph replay")
    finally:
        graph.reset()
    assert_bitwise(base["inputs"], inputs, "input mutation")

    variants = (([0, 1], (0,)), ([1, 0], (1,)), ([0, 0], (0, 1)), ([0, 2], (0,)))
    for indices, positions in variants:
        state = run(indices)
        snapshot(state["outputs"], allowed_inf={})
        for position in positions:
            # Check the preserved quantized payload, including NaN pad bytes.
            batched = ("q", "k", "v", "sfq", "sfk", "sfv") if mode == "mxfp8" else ("q", "k", "v")
            got_inputs = {name: _logical_v_scales(value) if name == "sfv" else value for name, value in state["inputs"].items()}
            want_inputs = {name: _logical_v_scales(value) if name == "sfv" else value for name, value in inputs.items()}
            assert_bitwise(
                {name: value[position : position + 1] if name in batched else value for name, value in got_inputs.items()},
                want_inputs,
                "quantized sequence payload and scales",
            )
            assert_bitwise(
                {name: state["outputs"][name][position : position + 1] for name in ("o", "stats")},
                {name: expected[name] for name in ("o", "stats")},
                "batch invariance",
            )
    return inputs, expected


@pytest.mark.parametrize("mode", ["fp8", "mxfp8"])
@pytest.mark.parametrize("dim", [128, 512])
@pytest.mark.parametrize("in_key", ["e4m3", "e5m2"])
def test_v_tail_replay_and_batch(mode, dim, in_key):
    _exercise(mode, dim, in_key)


@pytest.mark.parametrize("mode", ["fp8", "mxfp8"])
@pytest.mark.parametrize("dim", [128, 512])
@pytest.mark.parametrize("in_key", ["e4m3", "e5m2"])
def test_v_tail_golden(mode, dim, in_key):
    inputs, outputs = _exercise(mode, dim, in_key)
    assert_golden(f"sdpa-sm120-{mode}-v-tail-d{dim}-{in_key}", inputs, outputs, allowed_inf={})


@pytest.mark.parametrize("dim", [128, 512])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_v_tail_scale_repack(dim, dtype):
    """CPU fixture control: preserve scale bytes while rebatching their swizzle."""
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    generator = torch.Generator().manual_seed(807)
    values = torch.randn(2, 2, 80, dim, generator=generator)

    def scales(tensor):
        result = quantize_to_mxfp8(tensor, len(tensor), 2, 80, dim, fp8_dtype=dtype)
        return result[5].view(torch.uint8).reshape(len(tensor), 2, 4, dim)

    parts = [scales(values[:1]), scales(values[1:])]
    expected = scales(values)
    assert torch.equal(_pack_v_scales(parts), expected)
    # One row atom can hide the bad physical concatenation; multiple D atoms
    # interleave B/H columns differently and must expose that negative control.
    assert torch.equal(torch.cat(parts, dim=0), expected) == (dim == 128)
