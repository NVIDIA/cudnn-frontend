# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration ownership for the isolated half multi-Q implementation."""

from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import cudnn
from cudnn.sdpa.fwd import config_sm100 as legacy
from cudnn.sdpa.fwd.kernels.sm100.d128_f16_multi_q import config as own
from cudnn.sdpa.fwd.kernels.sm100.d64_f16_quadrant import config as quadrant
from cudnn.sdpa.fwd import engines, heuristics
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("dim", [32, 64, 128])
def test_multi_q_configuration_defaults_are_independent(dim, monkeypatch):
    record = getattr(own, f"CfgD{dim}")
    original = getattr(legacy, f"CfgD{dim}")
    expected = asdict(original())
    assert record.__bases__ == (object,)
    assert asdict(record()) == expected
    monkeypatch.setattr(original, "STAGES_KV", 999)
    assert record.STAGES_KV == expected["STAGES_KV"]


@pytest.mark.parametrize("dim", [32, 64, 128])
def test_multi_q_factory_does_not_call_legacy_implementation(dim, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("isolated configuration called a legacy implementation")

    for name in ("make_cfg_d32", "make_cfg_d64", "make_cfg_d128", "_make_cfg_small_d_sm100", "_validate_cfg_d64", "_validate_cfg_d128"):
        monkeypatch.setattr(legacy, name, forbidden)
    cfg, _ = getattr(own, f"make_cfg_d{dim}")(own.TemplateParams(d_flavor=dim, cta_mma=1))
    assert cfg.TILE_K == dim
    assert cfg.TILES_Q == 2


@pytest.mark.parametrize("dim", [32, 64, 128])
@pytest.mark.parametrize("override", [{"decode_tile": True}, {"decode_quad": True}, {"decode_q_tile": 16}, {"dual_kv_stream": True}, {"dtype_qkv": 0}])
def test_multi_q_factory_rejects_other_main_implementations(dim, override):
    with pytest.raises(ValueError, match="different half main implementation|FP16/BF16"):
        getattr(own, f"make_cfg_d{dim}")(own.TemplateParams(d_flavor=dim, cta_mma=1, **override))


def test_cold_multi_q_configuration_does_not_import_registry_or_legacy_factory():
    package_root = str(Path(cudnn.__file__).resolve().parent.parent)
    code = "\n".join(
        (
            "import sys",
            "sys.meta_path[:] = [finder for finder in sys.meta_path if 'editable' not in str(finder).lower()]",
            f"sys.path.insert(0, {package_root!r})",
            "import cudnn.sdpa.fwd.kernels.sm100.d128_f16_multi_q.config",
            "assert 'cudnn.sdpa.fwd.engines' not in sys.modules",
            "assert 'cudnn.sdpa.fwd.config_sm100' not in sys.modules",
            "assert 'cudnn.sdpa.fwd.kernels.sm100.d128_f16_multi_q.kernel' not in sys.modules",
        )
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_quadrant_configuration_is_independent_of_single_q_and_legacy(monkeypatch):
    from cudnn.sdpa.fwd.kernels.sm100.d128_f16_single_q import config as single

    expected = asdict(legacy.CfgD64Decode())
    assert quadrant.CfgD64Decode.__bases__ == (object,)
    assert asdict(quadrant.CfgD64Decode()) == expected

    def forbidden(*args, **kwargs):
        raise AssertionError("quadrant configuration called another implementation")

    for module in (legacy, single):
        monkeypatch.setattr(module, "make_cfg_d64_decode", forbidden)
        monkeypatch.setattr(module, "_validate_cfg_d64_decode", forbidden)
        monkeypatch.setattr(module.CfgD64Decode, "STAGES_KV", 999)
    cfg, _ = quadrant.make_cfg_d64_decode(quadrant.TemplateParams(d_flavor=64, cta_mma=1, decode_tile=True, decode_quad=True, early_q=True))
    assert cfg.TILE_K == cfg.TILE_O == 64
    assert cfg.TILES_Q == 1 and cfg.STAGES_KV == expected["STAGES_KV"]


@pytest.mark.parametrize("changes", [{"decode_quad": False}, {"d_flavor": 128}, {"early_q": False}, {"decode_tile": False}])
def test_quadrant_rejects_missing_cooperation_protocol(changes):
    args = dict(d_flavor=64, cta_mma=1, decode_tile=True, decode_quad=True, early_q=True)
    args.update(changes)
    with pytest.raises(ValueError, match="quadrant|decode_quad"):
        quadrant.make_cfg_d64_decode(quadrant.TemplateParams(**args))


def test_cold_quadrant_configuration_does_not_import_other_implementations():
    package_root = str(Path(cudnn.__file__).resolve().parent.parent)
    code = "\n".join(
        (
            "import sys",
            "sys.meta_path[:] = [finder for finder in sys.meta_path if 'editable' not in str(finder).lower()]",
            f"sys.path.insert(0, {package_root!r})",
            "import cudnn.sdpa.fwd.kernels.sm100.d64_f16_quadrant.config",
            "assert 'cudnn.sdpa.fwd.engines' not in sys.modules",
            "assert 'cudnn.sdpa.fwd.config_sm100' not in sys.modules",
            "assert 'cudnn.sdpa.fwd.kernels.sm100.d128_f16_single_q.config' not in sys.modules",
            "assert 'cudnn.sdpa.fwd.kernels.sm100.d64_f16_quadrant.kernel' not in sys.modules",
        )
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def half_facts(**changes):
    values = dict(b=2, h_q=8, h_kv=2, s_q=1, s_kv=4097, d_qk=64, d_v=64, dtype=cudnn.data_type.BFLOAT16, dtype_o=cudnn.data_type.BFLOAT16, device_cc=(10, 0))
    values.update(changes)
    return SdpaGraphFacts(**values)


@pytest.mark.parametrize("dim", [32, 64, 128])
@pytest.mark.parametrize("cc", [(10, 0), (10, 3)])
def test_retired_small_half_recipe_requires_replanning(dim, cc):
    spec = next(s for s in engines.ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm100")
    reason = engines.mismatch(spec.capabilities, half_facts(d_qk=dim, d_v=dim, device_cc=cc), engines.SdpaFwdKnobs(cga=1))
    assert "owned by the explicit implementation engines" in reason


@pytest.mark.parametrize(
    "dim,sq,cga,paged,owner,engine_id",
    [
        (32, 129, 1, False, "sdpa_fwd_sm100_d128_multi_q", 20535),
        (64, 129, 1, False, "sdpa_fwd_sm100_d128_multi_q", 20535),
        (64, 1, 1, False, "sdpa_fwd_sm100_d64_quadrant", 20536),
        (64, 33, 1, False, "sdpa_fwd_sm100_d128_single_q", 20532),
        (128, 33, 1, False, "sdpa_fwd_sm100_d128_single_q", 20532),
        (128, 1, 2, False, "sdpa_fwd_sm100_d128_multi_q", 20535),
        (128, 8, 1, True, "sdpa_fwd_sm100_d256_swap_ab", 20520),
    ],
)
def test_registered_small_half_records_name_the_selected_body(dim, sq, cga, paged, owner, engine_id):
    from cudnn.engines.manifest import MANIFEST

    f = half_facts(d_qk=dim, d_v=dim, s_q=sq, has_paged_kv=paged, padded=paged, page_size=16 if paged else 0)
    k = engines.SdpaFwdKnobs(cga=cga, pack_gqa=False)
    geometry = heuristics.select_sm100_small_half_geometry(f, k)
    assert geometry.implementation == owner
    canonical = geometry.canonical_knobs(k)
    spec = next(s for s in engines.ENGINE_SPECS if s.name == owner)
    assert engines.mismatch(spec.capabilities, f, canonical) is None
    assert engines.SdpaFwdKnobs.from_public(canonical.to_public()) == canonical
    family = next(family for family in MANIFEST if family.name == "frost_sdpa_fwd")
    assert family.offered_ids()[owner] == engine_id


@pytest.mark.parametrize("owner", ["sdpa_fwd_sm100_d64_quadrant", "sdpa_fwd_sm100_d128_single_q"])
def test_decode_owners_decline_the_legacy_ignored_n64_request(owner):
    spec = next(s for s in engines.ENGINE_SPECS if s.name == owner)
    k = engines.SdpaFwdKnobs(cga=1, tile_m=128, tile_n=64, pack_gqa=False)
    # The legacy factory accepted this request but built TILE_N=128. Public
    # records for a fixed implementation must not silently ignore the width.
    reason = engines.mismatch(spec.capabilities, half_facts(), k)
    assert "128x128" in reason


def test_registered_d32_paged_thd_keeps_the_d64_single_q_envelope():
    f = half_facts(d_qk=32, d_v=32, thd=True, has_paged_kv=True, padded=True, page_size=16)
    k = engines.SdpaFwdKnobs(cga=1, pack_gqa=False, tile_m=128, tile_n=128, split_kv=1)
    geometry = heuristics.select_sm100_small_half_geometry(f, k)
    spec = next(s for s in engines.ENGINE_SPECS if s.name == geometry.implementation)
    assert geometry.implementation == "sdpa_fwd_sm100_d128_single_q"
    assert geometry.d_flavor == 64
    assert engines.mismatch(spec.capabilities, f, k) is None


@pytest.mark.parametrize("dim", [32, 64, 128])
def test_fixed_multi_q_host_does_not_select_another_main_body(dim, monkeypatch):
    from cudnn.sdpa.fwd.kernels.sm100.d128_f16_multi_q import host

    def forbidden(*args, **kwargs):
        raise AssertionError("a selected implementation re-entered performance policy")

    monkeypatch.setattr(heuristics, "select_sm100_small_half_geometry", forbidden)
    spec = SimpleNamespace(capabilities=object())
    facts = half_facts(d_qk=dim, d_v=dim)
    knobs = engines.SdpaFwdKnobs(cga=1, tile_m=128)
    result = host.lower(
        spec,
        facts,
        knobs,
        lower_impl=lambda s, f, k, selected_kernel: (s, f, k, selected_kernel),
        knob_type=engines.SdpaFwdKnobs,
        selected_shape=lambda caps, f, k: (dim, dim),
        resolve_geometry=heuristics.resolve_sm100_half_multi_q_geometry,
    )
    selected = result[-1]
    assert selected.implementation == "sdpa_fwd_sm100_d128_multi_q"
    assert selected.d_flavor == dim
    params = selected.configure_params(own.TemplateParams())
    cfg, _ = getattr(own, f"make_cfg_d{dim}")(params)
    assert cfg.TILES_Q == 2 and not params.decode_tile and not params.decode_quad


@pytest.mark.parametrize("ratio,rows,accepted", [(1, 32, True), (1, 33, False), (4, 8, True), (4, 9, False), (12, 1, False)])
def test_quadrant_live_rows_are_a_safety_bound(ratio, rows, accepted):
    from cudnn.sdpa.fwd.kernels.sm100.d64_f16_quadrant import contract

    facts = half_facts(h_q=2 * ratio, s_q=rows)
    knobs = engines.SdpaFwdKnobs(cga=1, tile_m=128, pack_gqa=True)
    assert (contract.mismatch(facts, knobs) is None) is accepted


def test_fixed_quadrant_host_cannot_become_single_q(monkeypatch):
    from cudnn.sdpa.fwd.kernels.sm100.d64_f16_quadrant import host

    def forbidden(*args, **kwargs):
        raise AssertionError("quadrant host re-entered performance policy")

    monkeypatch.setattr(heuristics, "select_sm100_small_half_geometry", forbidden)
    selected = host.lower(None, half_facts(), None, lower_impl=lambda s, f, k, selected_kernel: selected_kernel, knob_type=engines.SdpaFwdKnobs)
    params = selected.configure_params(own.TemplateParams())
    assert selected.kernel_file.endswith("d64_f16_quadrant/kernel.py")
    assert params.decode_quad and params.decode_tile and params.early_q
    quadrant.make_cfg_d64_decode(params)


@pytest.mark.parametrize(
    "cc,dim,thd,split,early", [((10, 0), 128, True, 4, True), ((10, 3), 128, True, 4, False), ((10, 0), 64, True, 4, False), ((10, 3), 64, False, 1, True)]
)
def test_single_q_local_variant_preserves_architecture_gate(cc, dim, thd, split, early):
    from cudnn.sdpa.fwd.kernels.sm100.d128_f16_single_q.contract import small_geometry

    facts = half_facts(device_cc=cc, d_qk=dim, d_v=dim, thd=thd)
    selected = small_geometry(facts, engines.SdpaFwdKnobs(cga=1, split_kv=split))
    params = selected.configure_params(own.TemplateParams())
    assert params.early_q is early
    assert params.decode_tile is (dim == 64)
    assert not params.decode_quad and params.single_q_head_dim == 128


def test_paged_half_swap_ab_keeps_physical_and_live_columns_separate():
    facts = half_facts(d_qk=128, d_v=128, has_paged_kv=True, page_size=16, h_q=16)
    selected = heuristics.select_sm100_small_half_geometry(facts, engines.SdpaFwdKnobs(cga=1, pack_gqa=True, split_kv=4))
    assert selected.implementation == "sdpa_fwd_sm100_d256_swap_ab"
    assert selected.query_tile == 16
    params = selected.configure_params(own.TemplateParams())
    assert params.decode_head_dim == 128 and params.decode_live_columns == 8
    assert params.decode_q_tile == 16


@pytest.mark.parametrize(
    "shape,cga,sq,paged,expected",
    [
        (64, 1, 32, False, "sdpa_fwd_sm100_d64_quadrant"),
        (64, 1, 33, False, "sdpa_fwd_sm100_d128_single_q"),
        (64, 1, 129, False, "sdpa_fwd_sm100_d128_multi_q"),
        (64, 2, 1, False, "sdpa_fwd_sm100_d128_multi_q"),
        (128, 1, 8, True, "sdpa_fwd_sm100_d256_swap_ab"),
        (128, 1, 9, True, "sdpa_fwd_sm100_d128_single_q"),
    ],
)
def test_central_selection_retains_live_row_boundaries(shape, cga, sq, paged, expected):
    facts = half_facts(d_qk=shape, d_v=shape, s_q=sq, has_paged_kv=paged)
    selected = heuristics.select_sm100_small_half_geometry(facts, engines.SdpaFwdKnobs(cga=cga, pack_gqa=False))
    assert selected.implementation == expected


def test_mixed_fp8_kv_does_not_take_the_uniform_half_swap_ab_route():
    facts = half_facts(d_qk=128, d_v=128, has_paged_kv=True, uniform_dtype=False, kv_dtype=cudnn.data_type.FP8_E4M3)
    selected = heuristics.select_sm100_small_half_geometry(facts, engines.SdpaFwdKnobs(cga=1, pack_gqa=False))
    assert selected.implementation == "sdpa_fwd_sm100_d128_single_q"


def test_bias_keeps_the_multi_q_body_at_short_queries():
    selected = heuristics.select_sm100_small_half_geometry(half_facts(has_bias=True), engines.SdpaFwdKnobs(cga=2, pack_gqa=False))
    assert selected.implementation == "sdpa_fwd_sm100_d128_multi_q"


@pytest.mark.parametrize("sk,tile_n,cga,resolved_n", [(128, 128, 2, 128), (256, None, 1, 64)])
def test_default_bias_width_uses_the_existing_resident_budget(sk, tile_n, cga, resolved_n):
    knobs = engines.SdpaFwdKnobs(tile_n=tile_n, pack_gqa=False)
    selected = heuristics.select_sm100_small_half_geometry(half_facts(has_bias=True, s_kv=sk), knobs)
    assert selected.implementation == "sdpa_fwd_sm100_d128_multi_q"
    assert selected.cga == cga
    assert selected.canonical_knobs(knobs).tile_n == resolved_n


@pytest.mark.parametrize("cc", [(10, 0), (10, 3)])
@pytest.mark.parametrize("dtype", [cudnn.data_type.HALF, cudnn.data_type.BFLOAT16])
def test_default_paged_thd_width_keeps_the_prefill_body(cc, dtype):
    offset = SimpleNamespace(get_data_type=lambda: cudnn.data_type.INT32)
    tensor = SimpleNamespace(ragged_offset=offset, get_stride=lambda: (1024, 128, 1024, 1), get_ragged_offset_multiplier=lambda: 128)
    facts = half_facts(device_cc=cc, dtype=dtype, dtype_o=dtype, d_qk=128, d_v=128, thd=True, has_paged_kv=True, q_t=tensor, o_t=tensor)
    # This graph can use ragged decode when explicitly requested; an omitted
    # width must retain the adapter's unsplit CGA2 prefill default.
    assert engines._thd_decode_leg(engines.sm100_legacy_policy_spec().capabilities, facts)
    knobs = engines.SdpaFwdKnobs(pack_gqa=False)
    selected = heuristics.select_sm100_small_half_geometry(facts, knobs)
    assert selected.implementation == "sdpa_fwd_sm100_d128_multi_q"
    assert selected.canonical_knobs(knobs).cga == 2
    split = engines.SdpaFwdKnobs(cga=1, pack_gqa=False, split_kv=4)
    assert heuristics.select_sm100_small_half_geometry(facts, split).implementation == "sdpa_fwd_sm100_d128_single_q"


@pytest.mark.parametrize("sq", [1, 33])
@pytest.mark.parametrize("outputs", [(True, False), (False, True), (True, True)])
def test_default_score_outputs_keep_a_body_with_score_stores(sq, outputs):
    facts = half_facts(d_qk=128, d_v=128, s_q=sq, has_score_max=outputs[0], has_score_sum_exp=outputs[1])
    knobs = engines.SdpaFwdKnobs(pack_gqa=False)
    selected = heuristics.select_sm100_small_half_geometry(facts, knobs)
    assert selected.implementation == "sdpa_fwd_sm100_d128_multi_q"
    assert selected.canonical_knobs(knobs).cga == 2


@pytest.mark.parametrize(
    "dim,sk,padded,kv_mask,expected_n,expected_cga",
    [
        (32, 256, True, False, 64, 1),
        (32, 256, False, False, 128, 1),
        (64, 128, False, False, 128, 2),
        (64, 256, False, False, 64, 1),
        (64, 128, False, True, 64, 1),
    ],
)
def test_fixed_multi_q_host_fills_legal_bias_defaults(dim, sk, padded, kv_mask, expected_n, expected_cga, monkeypatch):
    from cudnn.sdpa.fwd.kernels.sm100.d128_f16_multi_q import host

    def forbidden(*args, **kwargs):
        raise AssertionError("a fixed multi-Q body consulted implementation selection")

    monkeypatch.setattr(heuristics, "select_sm100_small_half_geometry", forbidden)
    f = half_facts(d_qk=dim, d_v=dim, s_kv=sk, has_bias=True, padded=padded, has_kv_mask=kv_mask, kv_mask_dtype=cudnn.data_type.FLOAT if kv_mask else None)
    k, selected = host.lower(
        SimpleNamespace(capabilities=object()),
        f,
        None,
        lower_impl=lambda s, f, k, selected_kernel: (k, selected_kernel),
        knob_type=engines.SdpaFwdKnobs,
        selected_shape=lambda caps, facts, knobs: (dim, dim),
        resolve_geometry=heuristics.resolve_sm100_half_multi_q_geometry,
    )
    assert (k.tile_n, k.cga) == (expected_n, expected_cga)
    assert selected.implementation == "sdpa_fwd_sm100_d128_multi_q"


@pytest.mark.parametrize("folder", ["d128_f16_multi_q", "d64_f16_quadrant"])
def test_cold_host_import_does_not_load_registry_or_dsl(folder):
    package_root = str(Path(cudnn.__file__).resolve().parent.parent)
    code = "\n".join(
        (
            "import sys",
            "sys.meta_path[:] = [finder for finder in sys.meta_path if 'editable' not in str(finder).lower()]",
            f"sys.path.insert(0, {package_root!r})",
            f"import cudnn.sdpa.fwd.kernels.sm100.{folder}.host",
            "assert 'cudnn.sdpa.fwd.engines' not in sys.modules",
            "assert 'cudnn.sdpa.fwd.heuristics' not in sys.modules",
            "assert 'cutlass' not in sys.modules",
        )
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("dim", [32, 64, 128])
def test_multi_q_declines_thd_split_before_compilation(dim):
    spec = next(s for s in engines.ENGINE_SPECS if s.name == "sdpa_fwd_sm100_d128_multi_q")
    facts = half_facts(d_qk=dim, d_v=dim, thd=True, padded=True)
    knobs = engines.SdpaFwdKnobs(cga=1, split_kv=2, pack_gqa=False)
    assert "THD split-KV requires the single-Q implementation" in engines.mismatch(spec.capabilities, facts, knobs)
    with pytest.raises(ValueError, match="THD split-KV requires the single-Q implementation"):
        getattr(own, f"make_cfg_d{dim}")(own.TemplateParams(d_flavor=dim, cta_mma=1, thd_varlen=True, split_kv=2))
