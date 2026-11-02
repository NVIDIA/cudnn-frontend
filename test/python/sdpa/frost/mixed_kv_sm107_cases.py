# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 mixed K/V checks, collected by the existing native architecture entry."""

import itertools
from dataclasses import replace

import pytest
import torch

import mixed_kv_cases as checks
from frost_test_utils import _SM
from test_utils import torch_fork_set_rng

_native = pytest.mark.skipif(_SM != 107, reason="native SM107 mixed half-Q/FP8-KV")
_ENGINE = "sdpa_fwd_prefill_sm107"
_DTYPES = list(itertools.product((torch.bfloat16, torch.float16), (torch.float8_e4m3fn, torch.float8_e5m2)))


@pytest.mark.parametrize("qdtype,kvdtype", itertools.product(("BFLOAT16", "HALF"), ("FP8_E4M3", "FP8_E5M2")))
def graph_admission(qdtype, kvdtype, monkeypatch):
    checks.check_graph_dtype_facts_and_admission(qdtype, kvdtype, monkeypatch, cc=(10, 7))


@pytest.mark.parametrize("q,kv,cga", itertools.product((2, 3), (0, 1), (1, 2)))
def native_resources(q, kv, cga):
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    from cudnn.sdpa.fwd import config_sm107 as cfg
    from cudnn.sdpa.fwd.config_d128_family import sdpa_smem_bytes, validate_smem

    params = TemplateParams(dtype_qkv=q, dtype_kv=kv, cta_mma=cga)
    mixed, _ = cfg.make_cfg_d128(params)
    assert mixed.DTYPE_KV == kv and mixed.DTYPE_QKV == mixed.DTYPE_O == q
    assert mixed.STAGES_KV == 2 * cga
    raw = 16384 // cga
    base = sdpa_smem_bytes(128, 128, 128, 128, 2, 2 * cga, cga, 2, 2, qo_alias=False) + cfg.SM107_ARCH.smem_fixed_overhead
    limited = replace(cfg.SM107_ARCH, smem_usable_bytes=base + raw - 1)
    validate_smem(mixed, "uniform-control", arch=limited, qo_alias=False)
    with pytest.raises(ValueError, match="raw FP8 KV staging slab"):
        validate_smem(mixed, "mixed-control", arch=limited, qo_alias=False, kv_copy_bytes=raw)
    for factory in (cfg.make_cfg_d32, cfg.make_cfg_d64, cfg.make_cfg_d192, cfg.make_cfg_d256):
        with pytest.raises(ValueError, match="independent FP8 KV"):
            factory(params)


_DENSE_CASES = [(q, kv, cga, "dense", 1, entry, True) for (q, kv), cga, entry in itertools.product(_DTYPES, (1, 2), ("graph", "standalone"))]
_DENSE_CASES += [(q, kv, 2, "dense", 3, "graph", False) for q, kv in _DTYPES]


@_native
@pytest.mark.parametrize("qdtype,kvdtype,cga,layout,splits,entry,pack", _DENSE_CASES)
@torch_fork_set_rng(seed=49382)
def dense_paged(qdtype, kvdtype, cga, layout, splits, entry, pack, monkeypatch, cudnn_handle):
    checks.check_dense_paged_rebind_capture(qdtype, kvdtype, cga, layout, splits, entry, monkeypatch, cudnn_handle, engine_name=_ENGINE, pack=pack)


@_native
@pytest.mark.parametrize("qdtype,kvdtype,cga", itertools.product(("bfloat16", "float16"), ("float8_e4m3fn", "float8_e5m2"), (1, 2)))
def paged_origins(qdtype, kvdtype, cga):
    from test_sdpa_fwd_paged_thd_origins import test_paged_thd_origins_rebind_and_capture as run

    hnd = kvdtype == "float8_e5m2"
    run(128, qdtype, "hnd" if hnd else "nhd", "head" if hnd else "token", cga, 1, True, page_size=128 if hnd else 8, sink=0.25, kv_dtype_name=kvdtype)


@_native
@pytest.mark.parametrize("qdtype,kvdtype,splits", itertools.product(("bfloat16", "float16"), ("float8_e4m3fn", "float8_e5m2"), (1, 5)))
def thd_origins(qdtype, kvdtype, splits):
    from test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as run

    run(128, qdtype, "head" if kvdtype == "float8_e5m2" else "token", True, splits == 1, splits, kv_dtype_name=kvdtype, engine_name=_ENGINE)


@_native
@pytest.mark.parametrize("qdtype,kvdtype", _DTYPES + [(torch.bfloat16, torch.bfloat16), (torch.float16, torch.float16)])
@pytest.mark.parametrize("prefix_q,entry", itertools.product((False, True), ("graph", "standalone")))
@torch_fork_set_rng(seed=49383)
def single_q(qdtype, kvdtype, prefix_q, entry, cudnn_handle):
    checks.check_single_q_ragged_decode(qdtype, kvdtype, prefix_q, entry, cudnn_handle, engine_name=_ENGINE)


@_native
@pytest.mark.parametrize(
    "qdtype,kvdtype",
    [(torch.bfloat16, torch.float8_e4m3fn), (torch.float16, torch.float8_e5m2), (torch.bfloat16, torch.bfloat16), (torch.float16, torch.float16)],
)
@pytest.mark.parametrize("cga,prefix_q", itertools.product((1, 2), (False, True)))
@torch_fork_set_rng(seed=49385)
def single_q_unsplit(qdtype, kvdtype, cga, prefix_q, cudnn_handle):
    # Admitting split decode must preserve native unsplit prefill at Q=1.
    checks.check_single_q_ragged_decode(qdtype, kvdtype, prefix_q, "graph", cudnn_handle, engine_name=_ENGINE, cga=cga, splits=1)


@_native
@pytest.mark.parametrize("qdtype,kvdtype", [(torch.bfloat16, torch.float8_e4m3fn), (torch.float16, torch.float8_e5m2)])
@pytest.mark.parametrize("entry", ["graph", "standalone"])
@torch_fork_set_rng(seed=49386)
def single_q_sink(qdtype, kvdtype, entry, cudnn_handle):
    checks.check_single_q_ragged_decode(qdtype, kvdtype, entry == "graph", entry, cudnn_handle, engine_name=_ENGINE, has_sink=True)


@_native
@pytest.mark.parametrize("qdtype,kvdtype", _DTYPES)
@torch_fork_set_rng(seed=49384)
def compact_split(qdtype, kvdtype):
    checks.check_compact_thd_split_standalone(qdtype, kvdtype)


@_native
@pytest.mark.parametrize("qdtype,kvdtype", _DTYPES)
def dense_paged_decline(qdtype, kvdtype, monkeypatch, cudnn_handle):
    # The existing SM107 half row requires THD queries for D128 paged KV.
    with pytest.raises(NotImplementedError, match="Rubin paged KV requires THD queries"):
        checks.check_dense_paged_rebind_capture(qdtype, kvdtype, 2, "paged", 1, "graph", monkeypatch, cudnn_handle, engine_name=_ENGINE)


@_native
@pytest.mark.parametrize("dtype_name,geometry", itertools.product(("bfloat16", "float16"), ("partial6", "window")))
def uniform_control(dtype_name, geometry):
    from test_sdpa_native_pack_sm107 import _check_native_d128_packing_dense_replay

    _check_native_d128_packing_dense_replay(dtype_name, 0, True, geometry)


@_native
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
@pytest.mark.parametrize("pack_gqa", [False, True])
def thd_uniform_control(dtype_name, pack_gqa):
    from test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as run

    run(128, dtype_name, "token", True, pack_gqa, 1, kv_dtype_name=dtype_name, engine_name=_ENGINE)


@_native
@pytest.mark.parametrize(
    "qdtype,kvdtype",
    [("bfloat16", "float8_e4m3fn"), ("float16", "float8_e5m2"), ("bfloat16", "bfloat16"), ("float16", "float16")],
)
@pytest.mark.parametrize("pack_gqa", [False, True])
def thd_cga1_origins(qdtype, kvdtype, pack_gqa):
    """Unsplit CGA1 must retain all five physical origins for uniform and mixed KV."""
    from test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as run

    run(128, qdtype, "head" if qdtype == "float16" else "token", True, pack_gqa, 1, kv_dtype_name=kvdtype, engine_name=_ENGINE, cga=1)
