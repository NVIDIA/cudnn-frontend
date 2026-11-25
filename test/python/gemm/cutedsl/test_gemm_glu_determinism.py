# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay, batch-invariance and golden checks for cudnn.ops.gemm_glu_forward (forward-only fusion)."""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from gemm.cutedsl.test_gemm_glu_utils import allocate_input_tensors, check_ref_gemm_glu, check_ref_gemm_glu_quant

pytestmark = pytest.mark.L0

# name -> (ab_dtype, sf_dtype, sf_vec_size, ab12_dtype, alpha)
PATHS = {
    "bf16": (torch.bfloat16, None, None, torch.float32, 0.05),
    "fp8": (torch.float8_e4m3fn, None, None, torch.float32, 0.05),
    "mxfp8": (torch.float8_e4m3fn, torch.float8_e8m0fnu, 32, torch.bfloat16, 0.02),
    "nvfp4": (torch.float4_e2m1fn_x2, torch.float8_e4m3fn, 16, torch.float32, 0.02),
}
TILES = {"1cta": ((128, 128), (1, 1)), "2cta": ((256, 256), (2, 2))}
GEGLU = dict(linear_offset=0.25, geglu_alpha=1.0, glu_clamp_max=1.5, glu_clamp_min=-1.0)

act_params = pytest.mark.parametrize("act_func", ["swiglu", "geglu"])
path_params = pytest.mark.parametrize("path", list(PATHS))
tile_params = pytest.mark.parametrize("tile", list(TILES))


@pytest.fixture(autouse=True)
def require_tcgen05():
    from cudnn.tensor_adapter import has_tcgen05

    if not torch.cuda.is_available() or not has_tcgen05(torch.cuda.get_device_capability()[0]):
        pytest.skip("gemm_glu_forward requires SM100 or newer")


def make_sequences(count, path, m, n, k, seed):
    """One L == 1 problem per sequence: quantized once, payload and scales fixed across batch changes."""
    ab_dtype, sf_dtype, sf_vec_size, _, _ = PATHS[path]
    names = ("a", "a_ref", "b", "b_ref", "sfa", "sfa_ref", "sfb", "sfb_ref", "norm_const")
    with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
        torch.manual_seed(seed)
        return [
            dict(zip(names, allocate_input_tensors(m, n, k, 1, ab_dtype, "k", "k", sf_dtype is not None, sf_vec_size, sf_dtype, torch.bfloat16)))
            for _ in range(count)
        ]


def cat_batch(tensors, to_physical, to_logical):
    dtype = tensors[0].dtype
    raw = dtype.itemsize == 1 and dtype != torch.uint8
    parts = [(t.view(torch.uint8) if raw else t).permute(to_physical) for t in tensors]
    packed = torch.cat(parts, 0).contiguous().permute(to_logical)
    return packed.view(dtype) if raw else packed


def pack(sequences):
    packed = {}
    for name in ("a", "b", "a_ref", "b_ref", "sfa_ref", "sfb_ref"):
        if sequences[0][name] is not None:
            packed[name] = cat_batch([s[name] for s in sequences], (2, 0, 1), (1, 2, 0))
    for name in ("sfa", "sfb"):
        packed[name] = None if sequences[0][name] is None else cat_batch([s[name] for s in sequences], (5, 2, 4, 0, 1, 3), (3, 4, 1, 5, 2, 0))
    return packed


def invoke(packed, path, tile, act_func):
    from cudnn.ops import gemm_glu_forward

    _, _, sf_vec_size, ab12_dtype, alpha = PATHS[path]
    mma_tiler_mn, cluster_shape_mn = TILES[tile]
    quant = {} if packed["sfa"] is None else dict(sfa_tensor=packed["sfa"], sfb_tensor=packed["sfb"], sf_vec_size=sf_vec_size, vector_f32=True)
    result = gemm_glu_forward(
        packed["a"],
        packed["b"],
        alpha=alpha,
        ab12_dtype=ab12_dtype,
        c_dtype=torch.bfloat16,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        act_func=act_func,
        **(GEGLU if act_func == "geglu" else {}),
        **quant,
    )
    return {name: value for name, value in zip(("ab12", "c", "sfc", "amax"), result) if value is not None}


def reference(packed, outputs, path, act_func):
    alpha = PATHS[path][4]
    glu = dict(act_func=act_func, **(GEGLU if act_func == "geglu" else {}))
    if packed["sfa"] is None:
        check_ref_gemm_glu(packed["a"], packed["b"], outputs["ab12"], outputs["c"], alpha=alpha, **glu)
    else:
        check_ref_gemm_glu_quant(
            packed["a"],
            packed["a_ref"],
            packed["b"],
            packed["b_ref"],
            packed["sfa_ref"],
            packed["sfb_ref"],
            outputs["ab12"],
            outputs["c"],
            None,
            outputs.get("amax"),
            None,
            PATHS[path][2],
            alpha=alpha,
            **glu,
        )


def inputs_of(packed):
    # Packed FP4 has no torch isfinite; its bytes are the payload.
    return {
        name: packed[name].view(torch.uint8) if packed[name].dtype == torch.float4_e2m1fn_x2 else packed[name]
        for name in ("a", "b", "sfa", "sfb")
        if packed[name] is not None
    }


def batch_slice(outputs, index):
    # amax is one reduction over the whole launch, so it is not a per-sequence result.
    return {name: value[:, :, index : index + 1].contiguous() for name, value in outputs.items() if name != "amax"}


@act_params
@path_params
@tile_params
def test_gemm_glu_replay(act_func, path, tile):
    packed = pack(make_sequences(3, path, 512, 512, 512, seed=1))
    original = snapshot(inputs_of(packed))
    outputs = assert_replays(lambda: invoke(packed, path, tile, act_func))
    reference(packed, outputs, path, act_func)
    assert_bitwise(snapshot(inputs_of(packed)), original, "input bytes")


@act_params
@path_params
@tile_params
def test_gemm_glu_batch_invariance(act_func, path, tile):
    sequences = make_sequences(4, path, 512, 512, 512, seed=2)
    alone = []
    for sequence in sequences:
        packed = pack([sequence])
        outputs = snapshot(invoke(packed, path, tile, act_func))
        reference(packed, outputs, path, act_func)
        alone.append(batch_slice(outputs, 0))
    orders = {"packed": (0, 1, 2), "reordered": (2, 0, 1), "duplicated": (1, 1, 0), "changed-neighbor": (0, 3, 2)}
    for label, order in orders.items():
        outputs = snapshot(invoke(pack([sequences[i] for i in order]), path, tile, act_func))
        for position, index in enumerate(order):
            assert_bitwise(batch_slice(outputs, position), alone[index], f"{label} sequence {index} at {position}")


@act_params
@path_params
def test_gemm_glu_golden(act_func, path):
    packed = pack(make_sequences(2, path, 256, 256, 256, seed=3))
    outputs = assert_replays(lambda: invoke(packed, path, "1cta", act_func), repeats=4)
    reference(packed, outputs, path, act_func)
    assert_golden(f"gemm_glu_forward_{act_func}_{path}", inputs_of(packed), outputs)
