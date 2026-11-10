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
    small, _ = cfg.make_cfg_d64(params)
    assert small.TILE_K == small.TILE_O == 64 and small.DTYPE_KV == kv
    for factory in (cfg.make_cfg_d32, cfg.make_cfg_d192, cfg.make_cfg_d256):
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
    import test_sdpa_fwd_paged_thd_origins as paged_checks

    hnd = kvdtype == "float8_e5m2"
    paged_checks.test_paged_thd_origins_rebind_and_capture(
        128, qdtype, "hnd" if hnd else "nhd", "head" if hnd else "token", cga, 1, True, page_size=128 if hnd else 8, sink=0.25, kv_dtype_name=kvdtype
    )


@_native
@pytest.mark.parametrize("qdtype,kvdtype,splits", itertools.product(("bfloat16", "float16"), ("float8_e4m3fn", "float8_e5m2"), (1, 5)))
def thd_origins(qdtype, kvdtype, splits):
    import test_sdpa_fwd_thd_origins as thd_checks

    thd_checks.test_independent_thd_origins_rebind_and_capture(
        128, qdtype, "head" if kvdtype == "float8_e5m2" else "token", True, splits == 1, splits, kv_dtype_name=kvdtype, engine_name=_ENGINE
    )


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
@pytest.mark.parametrize("cga,splits", [(1, 1), (2, 1), (1, 3), (2, 3)])
def dense_paged_native(qdtype, kvdtype, cga, splits, monkeypatch, cudnn_handle):
    checks.check_dense_paged_rebind_capture(qdtype, kvdtype, cga, "paged", splits, "graph", monkeypatch, cudnn_handle, engine_name=_ENGINE, cpu_reference=True)


@_native
@pytest.mark.parametrize("dtype_name,geometry", itertools.product(("bfloat16", "float16"), ("partial6", "window")))
def uniform_control(dtype_name, geometry):
    from test_sdpa_native_pack_sm107 import _check_native_d128_packing_dense_replay

    _check_native_d128_packing_dense_replay(dtype_name, 0, True, geometry)


@_native
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
@pytest.mark.parametrize("pack_gqa", [False, True])
def thd_uniform_control(dtype_name, pack_gqa):
    import test_sdpa_fwd_thd_origins as thd_checks

    thd_checks.test_independent_thd_origins_rebind_and_capture(128, dtype_name, "token", True, pack_gqa, 1, kv_dtype_name=dtype_name, engine_name=_ENGINE)


@_native
@pytest.mark.parametrize(
    "qdtype,kvdtype",
    [("bfloat16", "float8_e4m3fn"), ("float16", "float8_e5m2"), ("bfloat16", "bfloat16"), ("float16", "float16")],
)
@pytest.mark.parametrize("pack_gqa", [False, True])
def thd_cga1_origins(qdtype, kvdtype, pack_gqa):
    """Unsplit CGA1 must retain all five physical origins for uniform and mixed KV."""
    import test_sdpa_fwd_thd_origins as thd_checks

    thd_checks.test_independent_thd_origins_rebind_and_capture(
        128, qdtype, "head" if qdtype == "float16" else "token", True, pack_gqa, 1, kv_dtype_name=kvdtype, engine_name=_ENGINE, cga=1
    )


@_native
@pytest.mark.parametrize("qdtype,kvdtype", _DTYPES)
@pytest.mark.parametrize("cga,layout,splits", itertools.product((1, 2), ("dense", "paged"), (1, 3)))
@torch_fork_set_rng(seed=40138)
def d64_dense_paged(qdtype, kvdtype, cga, layout, splits, monkeypatch, cudnn_handle):
    checks.check_dense_paged_rebind_capture(
        qdtype, kvdtype, cga, layout, splits, "graph", monkeypatch, cudnn_handle, engine_name=_ENGINE, pack=True, cpu_reference=True, head_dim=64
    )


@_native
@pytest.mark.parametrize("qdtype,kvdtype,cga", itertools.product(("bfloat16", "float16"), ("float8_e4m3fn", "float8_e5m2"), (1, 2)))
@pytest.mark.parametrize("paged", [False, True])
def d64_thd_origins(qdtype, kvdtype, cga, paged):
    import test_sdpa_fwd_paged_thd_origins as paged_checks
    import test_sdpa_fwd_thd_origins as thd_checks

    if paged:
        paged_checks.test_paged_thd_origins_rebind_and_capture(
            64,
            qdtype,
            "mixed",
            "head" if cga == 1 else "token",
            cga,
            1,
            cga == 2,
            page_size=8 if kvdtype == "float8_e4m3fn" else 128,
            window_left=32,
            sink=0.25,
            kv_dtype_name=kvdtype,
            cpu_reference=True,
        )
    else:
        thd_checks.test_independent_thd_origins_rebind_and_capture(
            64, qdtype, "head" if cga == 1 else "token", True, cga == 2, 1, kv_dtype_name=kvdtype, engine_name=_ENGINE, cga=cga
        )


@_native
@pytest.mark.parametrize("qdtype,kvdtype", _DTYPES)
@pytest.mark.parametrize("cga", [1, 2])
@torch_fork_set_rng(seed=40139)
def d64_standalone(qdtype, kvdtype, cga, monkeypatch, cudnn_handle):
    checks.check_dense_paged_rebind_capture(
        qdtype, kvdtype, cga, "dense", 3, "standalone", monkeypatch, cudnn_handle, engine_name=_ENGINE, pack=False, cpu_reference=True, head_dim=64
    )


@_native
@pytest.mark.parametrize(
    "qdtype,kvdtype,cga,split,tile_n",
    [
        ("bfloat16", "float8_e4m3fn", 1, 1, 64),
        ("float16", "float8_e5m2", 2, 1, 128),
        ("bfloat16", "float8_e5m2", 2, 4, 64),
        ("float16", "float8_e4m3fn", 1, 4, 128),
    ],
)
def d64_replay_batch_golden(qdtype, kvdtype, cga, split, tile_n):
    import native_d128_dense_paged_determinism as replay_checks

    replay_checks.test_replay_batch_and_golden(qdtype, kvdtype, cga, split, tile_n, head_dim=64)


@pytest.mark.parametrize("qdtype,kvdtype", [("HALF", "FP8_E4M3"), ("BFLOAT16", "FP8_E5M2")])
def d64_admission(qdtype, kvdtype, monkeypatch):
    from cudnn.sdpa.fwd import engines

    spec, facts = checks.check_graph_dtype_facts_and_admission(qdtype, kvdtype, monkeypatch, cc=(10, 7), head_dim=64)
    reason = engines.mismatch(spec.capabilities, replace(facts, thd=True, stats_t=None), engines.SdpaFwdKnobs(cga=1, split_kv=3))
    assert reason and "THD execution unsplit" in reason, reason
    for dim in (32, 56, 72, 256, 512):
        assert engines.mismatch(spec.capabilities, replace(facts, d_qk=dim, d_v=dim), engines.SdpaFwdKnobs(cga=1))
    old_spec = next(s for s in engines.ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm100")
    assert engines.mismatch(old_spec.capabilities, replace(facts, device_cc=(10, 0)), engines.SdpaFwdKnobs(cga=1))


@_native
@pytest.mark.parametrize(
    "qdtype,kvdtype,cga,pack,tile_n",
    [
        ("bfloat16", "float8_e4m3fn", 1, False, 64),
        ("float16", "float8_e5m2", 2, False, 128),
        ("bfloat16", "float8_e5m2", 2, True, 64),
        ("float16", "float8_e4m3fn", 1, True, 128),
    ],
)
def d64_thd_batch_golden(qdtype, kvdtype, cga, pack, tile_n):
    import native_d64_paged_thd_determinism as thd_replay

    thd_replay.test_thd_batch_and_golden(qdtype, cga, pack, tile_n, kv_name=kvdtype)


@_native
@pytest.mark.parametrize("qdtype,kvdtype", _DTYPES)
@pytest.mark.parametrize("layout", ["dense", "paged"])
@torch_fork_set_rng(seed=40140)
def d64_single_q(qdtype, kvdtype, layout, monkeypatch, cudnn_handle):
    checks.check_dense_paged_rebind_capture(
        qdtype, kvdtype, 1, layout, 3, "graph", monkeypatch, cudnn_handle, engine_name=_ENGINE, pack=True, cpu_reference=True, head_dim=64, q_tokens=1
    )
