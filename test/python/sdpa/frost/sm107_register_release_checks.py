# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The four service warps release registers at one warpgroup-collective site."""

import pytest

from frost_test_utils import run_sass_probe, sass_probe_source

_BODY = """
d = params_kw.pop("__d")
has_lse = params_kw.pop("__has_lse", True)
params = TemplateParams(dtype_o=params_kw.get("dtype_qkv", 2), **params_kw)
mod = _load_sm100_kernel_module((d, d), params, rubin=True)
cfg = mod.CFG
assert cfg.MMA_WARP_ID % 4 == 0
assert (cfg.TMALDG_WARP_ID, cfg.TMASTG_WARP_ID, cfg.SCHED_WARP_ID) == tuple(cfg.MMA_WARP_ID + i for i in (1, 2, 3))
assert cfg.SCHED_WARP_ID + 1 == cfg.THREADS_PER_CTA // 32
mod.compile(d_qk=d, d_v=d, has_lse=has_lse, lse_kind="token" if cfg.THD_VARLEN else "dense", paged_hnd=bool(cfg.PAGED_KV), has_bias=bool(cfg.BIAS_MODE))
"""
_COUNTS = {
    "RELEASE": ("USETMAXREG.DEALLOC",),
    "ACQUIRE": ("USETMAXREG.TRY_ALLOC",),
}
_PAGED = dict(
    __d=128,
    cta_mma=2,
    dtype_qkv=2,
    qh_per_kh=4,
    pack_gqa=True,
    thd_varlen=True,
    seq_kv_lens_present=True,
    paged_kv=True,
    page_size=16,
    bottom_right=True,
    window_right=0,
)


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(dict(__d=32, cta_mma=1, dtype_qkv=2), id="dense-d32"),
        pytest.param(dict(__d=64, cta_mma=2, dtype_qkv=3, window_right=0), id="causal-d64"),
        pytest.param(dict(__d=128, cta_mma=2, dtype_qkv=2, window_right=0), id="dense-d128"),
        pytest.param(_PAGED, id="packed-paged-stats"),
        pytest.param(dict(_PAGED, __has_lse=False), id="packed-paged-no-stats"),
        pytest.param(dict(__d=32, cta_mma=1, dtype_qkv=2, bias_mode=1, n_chunk=6, tile_n=64, softmax_wgs=3), id="three-softmax-d32"),
        pytest.param(dict(__d=64, cta_mma=1, dtype_qkv=2, bias_mode=1, n_chunk=6, tile_n=64, softmax_wgs=3), id="three-softmax-d64"),
    ],
)
def test_service_warpgroup_has_one_register_release(tmp_path, params):
    """Count real instruction sites: correction and service groups each release once.

    The old four role-local service sites made five releases in the cubin.
    Equal register counts do not make distinct instruction sites collective.
    """
    probe = run_sass_probe(tmp_path, probe_src=sass_probe_source(_BODY, _COUNTS), arch="sm_107a", params=params, tag="register_release")
    assert probe.stats["RELEASE"] == 2, probe.stats
    assert probe.stats["ACQUIRE"] == 2, probe.stats
