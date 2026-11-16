# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fixed native work items preserve local configurations and never swap bodies."""

from dataclasses import asdict, replace
import importlib

import pytest

from cudnn.sdpa.fwd import config_sm107 as legacy
from cudnn.sdpa.fwd.engines import SdpaFwdKnobs
from cudnn.sdpa.fwd.kernels.sm107.small_half_contract import BATCH_PAIR, MULTI_Q, Geometry, explicit_geometry, mismatch
from sdpa.frost.test_sdpa_fwd_d512_implementation_contract import facts as _facts

pytestmark = pytest.mark.L0


def facts(d=64, **kw):
    return _facts(device_cc=(10, 7), device_sm_count=208, d_qk=d, d_v=128 if d == 192 else d, **kw)


def biased(d=64, **kw):
    values = dict(b=4, s_q=128, s_kv=512, has_bias=True, bias_dim=(1, 32, 128, 512), has_kv_mask=True)
    values.update(kw)
    return facts(d, **values)


@pytest.mark.parametrize("name", [MULTI_Q, BATCH_PAIR])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("tile_n", [64, 128])
@pytest.mark.parametrize("with_mask", [False, True])
def test_fixed_body_preserves_complete_local_configuration(name, d, tile_n, with_mask):
    f = biased(d, has_kv_mask=with_mask)
    # D64's ordinary streamed-bias item requires the two-CTA geometry.
    cga = 2 if name == MULTI_Q and d == 64 else 1
    k = SdpaFwdKnobs(tile_n=tile_n, cga=cga)
    g = explicit_geometry(name, f, k)
    k = g.canonical_knobs(k)
    assert mismatch(name, f, k) is None
    assert SdpaFwdKnobs.from_public(k.to_public()) == k
    mode = legacy.sm107_bias_mode_pick(d, cga, f.s_kv, pair=name == BATCH_PAIR, tiles_q=g.softmax_wgs, global_lengths=True, natural=True)
    p = legacy.TemplateParams(
        d_flavor=d, cta_mma=cga, tile_n=tile_n, bias_mode=mode, kv_mask_form=1 if with_mask else 0, n_chunk=g.n_chunk, softmax_wgs=g.softmax_wgs
    )
    assert g.configure_params(p) == p
    folder = "small_f16_batch_pair" if name == BATCH_PAIR else "d128_f16_multi_q"
    module = importlib.import_module("cudnn.sdpa.fwd.kernels.sm107." + folder + ".config")
    old, old_tma = getattr(legacy, f"make_cfg_d{d}")(p)
    new, new_tma = getattr(module, f"make_cfg_d{d}")(p)
    assert asdict(new) == asdict(old)
    assert asdict(new_tma) == asdict(old_tma)
    assert (new.N_CHUNK > 0) == (name == BATCH_PAIR)
    assert new.SOFTMAX_WARPGROUPS == g.softmax_wgs


@pytest.mark.parametrize("name,folder", [(MULTI_Q, "d128_f16_multi_q"), (BATCH_PAIR, "small_f16_batch_pair")])
@pytest.mark.parametrize("sq", [128, 256, 384, 512])
def test_explicit_host_keeps_work_item_across_default_selection_boundaries(monkeypatch, name, folder, sq):
    def forbidden(*a, **kw):
        raise AssertionError("explicit replay consulted the work-item selector")

    monkeypatch.setattr(legacy, "sm107_n_chunk_pick", forbidden)
    calls = []
    host = importlib.import_module("cudnn.sdpa.fwd.kernels.sm107." + folder + ".host")
    f = biased(32, s_q=sq, has_kv_mask=False)
    host.lower(object(), f, SdpaFwdKnobs(tile_n=128), lower_impl=lambda *a, **kw: calls.append((a, kw)), knob_type=SdpaFwdKnobs)
    args, kw = calls[0]
    g = kw["selected_kernel"]
    assert g.implementation == name
    assert g.kernel_file == f"sm107/{folder}/kernel.py"
    assert (g.n_chunk > 0) == (name == BATCH_PAIR)
    assert g.configure_params(legacy.TemplateParams(n_chunk=0 if name == BATCH_PAIR else 4)).n_chunk == g.n_chunk


@pytest.mark.parametrize(
    "changes", [dict(b=1), dict(has_bias=False), dict(bias_dim=(4, 32, 128, 512)), dict(thd=True), dict(has_paged_kv=True), dict(has_epilogue_gate=True)]
)
def test_pair_declines_incompatible_declarations(changes):
    assert mismatch(BATCH_PAIR, replace(biased(), **changes), None)


@pytest.mark.parametrize("changes", [dict(cga=2), dict(pack_gqa=True), dict(split_kv=3), dict(warp_spec_cfg=1), dict(tile_m=64)])
def test_pair_declines_conflicting_public_knobs(changes):
    assert mismatch(BATCH_PAIR, biased(), SdpaFwdKnobs(**changes))


def test_paged_d192_and_shared_single_q_boundaries():
    assert mismatch(MULTI_Q, facts(192), SdpaFwdKnobs(cga=2))
    f = facts(192, has_paged_kv=True, padded=True)
    g = explicit_geometry(MULTI_Q, f, SdpaFwdKnobs())
    k = g.canonical_knobs(SdpaFwdKnobs())
    assert mismatch(MULTI_Q, f, k) is None
    p = g.configure_params(legacy.TemplateParams(paged_kv=True, page_size=128, seq_kv_lens_present=True))
    from cudnn.sdpa.fwd.kernels.sm107.d128_f16_multi_q.config import make_cfg_d192_paged

    assert [asdict(x) for x in make_cfg_d192_paged(p)] == [asdict(x) for x in legacy.make_cfg_d192_paged(p)]
    assert mismatch(MULTI_Q, facts(128), SdpaFwdKnobs(cga=1))
    assert mismatch(MULTI_Q, facts(64, thd=True), SdpaFwdKnobs(cga=1, split_kv=3))


def test_three_group_chunk_and_padding_are_preserved():
    f = biased(32, s_kv=4096)
    k = SdpaFwdKnobs(tile_n=64)
    g = explicit_geometry(BATCH_PAIR, f, k)
    assert g.softmax_wgs == 3
    assert g.n_chunk > 0 and g.n_chunk % 3 == 0
    padded = explicit_geometry(BATCH_PAIR, replace(f, padded=True), k)
    tail = explicit_geometry(BATCH_PAIR, replace(f, s_kv=4097), k)
    assert padded.softmax_wgs == tail.softmax_wgs == 2
    assert padded.n_chunk % 2 == tail.n_chunk % 2 == 0


def test_geometry_rejects_opposite_work_item():
    with pytest.raises(ValueError, match="other work-item"):
        Geometry(MULTI_Q, 64, 1, 128, n_chunk=2)
    with pytest.raises(ValueError, match="other work-item"):
        Geometry(BATCH_PAIR, 64, 1, 128, n_chunk=0)


@pytest.mark.parametrize("name", [MULTI_Q, BATCH_PAIR])
def test_common_lowering_fixes_work_item_before_adapter_support(monkeypatch, name):
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa import graph_analyzer

    seen = {}

    class ConstructorReached(Exception):
        pass

    class Adapter:
        def __init__(self, *, n_chunk=None, softmax_wgs=None, **kwargs):
            seen.update(n_chunk=n_chunk, softmax_wgs=softmax_wgs)
            raise ConstructorReached

        def execute(self):
            pass

    monkeypatch.setattr(engines, "_adapter", lambda _: Adapter)
    monkeypatch.setattr(graph_analyzer, "tensor_desc_from_ir", lambda t, name: t)
    f = biased(32, s_kv=4096)
    g = explicit_geometry(name, f, SdpaFwdKnobs(tile_n=64))
    k = g.canonical_knobs(SdpaFwdKnobs())
    spec = next(s for s in engines.ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm107")
    with pytest.raises(ConstructorReached):
        engines.lower_dsl_prefill(spec, f, k, selected_kernel=g)
    assert seen == dict(n_chunk=g.n_chunk, softmax_wgs=g.softmax_wgs)
