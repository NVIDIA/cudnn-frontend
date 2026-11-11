# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Half Q stays half when native D128 reads independently typed FP8 K/V."""

import itertools

import pytest
import torch

import mixed_kv_cases as _mixed_checks

from frost_test_utils import _SM, requires_dsl
from test_utils import torch_fork_set_rng


@pytest.mark.L0
@pytest.mark.parametrize("qdtype", ["BFLOAT16", "HALF"])
@pytest.mark.parametrize("kvdtype", ["FP8_E4M3", "FP8_E5M2"])
def test_mixed_graph_dtype_facts_and_admission(qdtype, kvdtype, monkeypatch):
    return _mixed_checks.check_graph_dtype_facts_and_admission(qdtype, kvdtype, monkeypatch)


_CASES = [
    (q, kv, cga, layout, 1, "graph")
    for q, kv, cga, layout in itertools.product((torch.bfloat16, torch.float16), (torch.float8_e4m3fn, torch.float8_e5m2), (1, 2), ("dense", "paged"))
]
_CASES += [(q, torch.float8_e4m3fn, cga, "paged", 3, "graph") for q, cga in itertools.product((torch.bfloat16, torch.float16), (1, 2))]
_CASES += [(q, kv, 1, "dense", 1, "standalone") for q, kv in itertools.product((torch.bfloat16, torch.float16), (torch.float8_e4m3fn, torch.float8_e5m2))]


@pytest.mark.L0
@requires_dsl
@pytest.mark.skipif(_SM not in (100, 103), reason="native SM100/SM103 mixed KV")
@pytest.mark.parametrize("qdtype,kvdtype,cga,layout,splits,entry", _CASES)
@torch_fork_set_rng(seed=49382)
def test_mixed_dense_paged_rebind_capture(qdtype, kvdtype, cga, layout, splits, entry, monkeypatch, cudnn_handle):
    return _mixed_checks.check_dense_paged_rebind_capture(qdtype, kvdtype, cga, layout, splits, entry, monkeypatch, cudnn_handle)


@pytest.mark.L0
@requires_dsl
@pytest.mark.skipif(_SM not in (100, 103), reason="native SM100/SM103 mixed KV")
@pytest.mark.parametrize(
    "qdtype,kvdtype,paged,splits",
    [(q, kv, paged, 1) for q, kv, paged in itertools.product(("bfloat16", "float16"), ("float8_e4m3fn", "float8_e5m2"), (False, True))]
    + [(q, kv, paged, split) for q, kv, paged, split in itertools.product(("bfloat16", "float16"), ("float8_e4m3fn", "float8_e5m2"), (False, True), (2, 5))],
)
def test_mixed_physical_thd_origins(qdtype, kvdtype, paged, splits, monkeypatch):
    from cudnn.sdpa.fwd import api_dsl
    from test_sdpa_fwd_paged_thd_origins import test_paged_thd_origins_rebind_and_capture as paged_case
    from test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as thd_case

    loaded = []
    original = api_dsl._load_sm100_kernel_module

    def load(*args, **kwargs):
        module = original(*args, **kwargs)
        loaded.append(module)
        return module

    monkeypatch.setattr(api_dsl, "_load_sm100_kernel_module", load)
    stats = "head" if kvdtype == "float8_e5m2" else "token"
    if paged:
        paged_case(
            128,
            qdtype,
            "hnd" if stats == "head" else "nhd",
            stats,
            1 if splits > 1 else 2,
            splits,
            True,
            page_size=8 if stats == "token" else 128,
            # Preserve the half family's existing SM100 THD split domain.
            # The unsplit prefill body supports sinks; the packed split host
            # does not admit them for this architecture and head dimension.
            sink=0.25 if splits == 1 else None,
            kv_dtype_name=kvdtype,
        )
    else:
        thd_case(128, qdtype, stats, True, splits == 1, splits, kv_dtype_name=kvdtype)
    assert loaded and all(m.MIXED_FP8_KV for m in loaded)


@pytest.mark.L0
@pytest.mark.parametrize("q,kv", itertools.product((2, 3), (0, 1)))
def test_mixed_decode_resource_domain(q, kv):
    from dataclasses import replace
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams, make_cfg_d128_decode, _d128_smem_bytes
    from cudnn.sdpa.fwd.config_sm107 import make_cfg_d192 as rubin_cfg

    params = TemplateParams(dtype_qkv=q, cta_mma=1)
    uniform, _ = make_cfg_d128_decode(params)
    mixed, _ = make_cfg_d128_decode(replace(params, dtype_kv=kv))
    assert uniform.STAGES_KV == 3 and _d128_smem_bytes(uniform) == 224 * 1024
    assert mixed.STAGES_KV == 2 and _d128_smem_bytes(mixed) == 176 * 1024
    assert mixed.DTYPE_KV == kv and mixed.DTYPE_QKV == mixed.DTYPE_O == q
    with pytest.raises(ValueError, match="independent FP8 KV"):
        make_cfg_d128_decode(replace(params, dtype_kv=kv, single_q_head_dim=192, thd_varlen=True, seq_kv_lens_present=True))
    with pytest.raises(ValueError, match="independent FP8 KV"):
        rubin_cfg(replace(params, dtype_kv=kv))


@pytest.mark.L0
@requires_dsl
@pytest.mark.skipif(_SM not in (100, 103), reason="native SM100/SM103 mixed KV")
@pytest.mark.parametrize(
    "qdtype,kvdtype",
    list(itertools.product((torch.bfloat16, torch.float16), (torch.float8_e4m3fn, torch.float8_e5m2)))
    + [(torch.bfloat16, torch.bfloat16), (torch.float16, torch.float16)],
)
@pytest.mark.parametrize("prefix_q", [False, True])
@pytest.mark.parametrize("entry", ["graph", "standalone"])
@torch_fork_set_rng(seed=49383)
def test_half_q_single_q_ragged_decode(qdtype, kvdtype, prefix_q, entry, cudnn_handle):
    return _mixed_checks.check_single_q_ragged_decode(qdtype, kvdtype, prefix_q, entry, cudnn_handle)


@pytest.mark.L0
@requires_dsl
@pytest.mark.skipif(_SM not in (100, 103), reason="native SM100/SM103 mixed KV")
@pytest.mark.parametrize("qdtype,kvdtype", itertools.product((torch.bfloat16, torch.float16), (torch.float8_e4m3fn, torch.float8_e5m2)))
@torch_fork_set_rng(seed=49384)
def test_mixed_compact_thd_split_standalone(qdtype, kvdtype):
    return _mixed_checks.check_compact_thd_split_standalone(qdtype, kvdtype)


@pytest.mark.L0
@requires_dsl
@pytest.mark.skipif(_SM not in (100, 103), reason="native SM100/SM103 mixed KV")
@pytest.mark.parametrize("qdtype,kvdtype", itertools.product((torch.bfloat16, torch.float16), (torch.float8_e4m3fn, torch.float8_e5m2)))
@pytest.mark.parametrize("swizzle", [64, 128])
def test_mixed_conversion_all_encodings(qdtype, kvdtype, swizzle):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cudnn.frost.tile_dsl.handles import SmemTile
    from cudnn.sdpa.fwd.kernels._mixed_kv import convert_fp8_smem_tile

    @cute.kernel
    def convert_kernel(inp: cute.Tensor, out: cute.Tensor, swz: cutlass.Constexpr):
        lane = cute.arch.thread_idx()[0]
        raw = cutlass.Array(inp.element_type, 16384, alignment=1024, space=cutlass.AddressSpace.smem)
        dest = cutlass.Array(out.element_type, 16384, alignment=1024, space=cutlass.AddressSpace.smem)
        for i in cutlass.range(lane, 16384, 32):
            raw[i] = inp[i]
        cute.arch.barrier()
        width = cutlass.const_expr(swz // 2)
        src = SmemTile(raw, 16384, 16, 1024, 0, tma_granu_elems=width, tma_subtile_stride_elems=width * 128)
        dst = SmemTile(dest, 16384, 16, 1024, 0, tma_granu_elems=width, tma_subtile_stride_elems=width * 128)
        convert_fp8_smem_tile(src, dst, swz, out.element_type)
        cute.arch.barrier()
        for i in cutlass.range(lane, 16384, 32):
            out[i] = dest[i]

    @cute.jit
    def convert_host(inp: cute.Tensor, out: cute.Tensor, swz: cutlass.Constexpr):
        convert_kernel(inp, out, swz).launch(grid=(1, 1, 1), block=(32, 1, 1))

    # Enumerate every input code with different row positions. The reference
    # addresses implement the byte-SMEM and half-SMEM layouts independently.
    width = swizzle // 2
    ix = torch.arange(16384)
    raw = ((ix * 37 + (ix // width) * 13) % 256).to(torch.uint8)
    values = raw.view(kvdtype)
    row, col, outer = (ix // width) % 128, ix % width, (ix // (width * 128)) * (width * 128)
    if swizzle == 128:
        src = outer + row * width + ((col // 16) ^ ((row >> 1) & 3)) * 16 + col % 16
        dst = outer + row * width + (col ^ ((row & 7) * 8))
    else:
        src = outer + row * width + ((col // 16) ^ ((row >> 2) & 1)) * 16 + col % 16
        dst = outer + row * width + ((col // 8) ^ ((row >> 1) & 3)) * 8 + col % 8
    x = values.to("cuda")
    y = torch.empty(16384, dtype=qdtype, device="cuda")
    tx, ty = from_dlpack(x), from_dlpack(y)
    fn = cute.compile(convert_host, tx, ty, swizzle)
    fn(tx, ty)
    actual = y.cpu()
    expected = torch.empty(16384, dtype=qdtype)
    expected[dst] = values.float()[src].to(qdtype)
    finite = torch.isfinite(expected.float())
    torch.testing.assert_close(actual.view(torch.int16)[finite], expected.view(torch.int16)[finite], rtol=0, atol=0)
    assert torch.equal(torch.isnan(actual.float()), torch.isnan(expected.float()))
    assert torch.equal(torch.isinf(actual.float()), torch.isinf(expected.float()))
    non_nan = ~torch.isnan(expected.float())
    assert torch.equal(torch.signbit(actual.float())[non_nan], torch.signbit(expected.float())[non_nan])
