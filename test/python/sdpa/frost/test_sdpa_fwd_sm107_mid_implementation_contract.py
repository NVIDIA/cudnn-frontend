# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Native mid-width records keep their body and the shared-body boundaries."""

from dataclasses import replace
import importlib

import pytest
import cudnn
from cudnn.sdpa.fwd.engines import SdpaFwdKnobs
from cudnn.sdpa.fwd.config_common import TemplateParams
from cudnn.sdpa.fwd.kernels.sm107.mid_half_contract import D192, LOOKAHEAD, SWAP_AB, explicit_geometry, mismatch
from sdpa.frost.test_sdpa_fwd_d512_implementation_contract import facts as _facts

pytestmark = pytest.mark.L0


def facts(d=256, **kwargs):
    return _facts(device_cc=(10, 7), device_sm_count=208, d_qk=d, d_v=128 if d == 192 else d, **kwargs)


@pytest.mark.parametrize("name,d,tile,cga", [(D192, 192, 128, 2), (LOOKAHEAD, 256, 128, 2), (LOOKAHEAD, 128, 128, 1), (SWAP_AB, 256, 16, 1)])
def test_fixed_geometry_preserves_parameters_and_public_record(name, d, tile, cga):
    f = facts(d, thd=d == 128)
    g = explicit_geometry(name, f, SdpaFwdKnobs())
    k = g.canonical_knobs(SdpaFwdKnobs())
    assert mismatch(name, f, k) is None
    assert (g.query_tile, g.cluster_m, g.cta_group) == (tile, cga, cga)
    assert SdpaFwdKnobs.from_public(k.to_public()) == k
    assert k.warp_spec_cfg == int(d == 128)
    p = TemplateParams(
        cta_mma=cga,
        d_flavor=128,
        softmax_wgs=1 if d == 128 else 2,
        s_slots=2 if d == 128 else 1,
        thd_varlen=d == 128,
        decode_q_tile=tile if name == SWAP_AB else 0,
    )
    assert g.configure_params(p) == p
    assert mismatch(name, f, replace(k, cga=3 - cga)) is not None


@pytest.mark.parametrize("name,d", [(D192, 192), (LOOKAHEAD, 256), (SWAP_AB, 256)])
@pytest.mark.parametrize("cc", [(10, 0), (10, 3), (12, 0)])
def test_native_owners_decline_other_architectures(name, d, cc):
    assert mismatch(name, replace(facts(d), device_cc=cc), None)


def test_shared_and_native_body_boundaries():
    # Paging changes D192's body, even though its head dimensions stay the same.
    assert mismatch(D192, facts(192, has_paged_kv=True), None)
    assert mismatch(D192, facts(192, thd=True), SdpaFwdKnobs(cga=1))
    assert mismatch(D192, facts(192, thd=True), SdpaFwdKnobs(cga=2)) is None
    # Same-dtype paging and nonpaged THD splitting use the SM100 lookahead.
    assert mismatch(LOOKAHEAD, facts(has_paged_kv=True), None)
    assert mismatch(LOOKAHEAD, facts(thd=True), SdpaFwdKnobs(split_kv=3))
    assert mismatch(LOOKAHEAD, facts(thd=True), SdpaFwdKnobs(split_kv=1)) is None
    assert mismatch(LOOKAHEAD, facts(has_paged_kv=True, uniform_dtype=False, kv_dtype=cudnn.data_type.FP8_E4M3), None) is None


@pytest.mark.parametrize("field,value", [("pack_gqa", True), ("split_kv", 3), ("warp_spec_cfg", 0), ("tile_n", 64)])
def test_d128_one_q_rejects_conflicting_topology(field, value):
    f = facts(128, thd=True)
    k = explicit_geometry(LOOKAHEAD, f, SdpaFwdKnobs()).canonical_knobs(SdpaFwdKnobs())
    assert mismatch(LOOKAHEAD, f, replace(k, **{field: value}))


@pytest.mark.parametrize("tile", [16, 32, 64])
def test_swap_ab_fixed_tile_and_split_gate(tile):
    f = facts(s_q=1, has_epilogue_gate=True)
    g = explicit_geometry(SWAP_AB, f, SdpaFwdKnobs(tile_m=tile))
    k = g.canonical_knobs(SdpaFwdKnobs(split_kv=3, pack_gqa=tile == 64))
    assert (g.query_tile, g.decode_q_tile, k.tile_m) == (tile, tile, tile)
    assert mismatch(SWAP_AB, f, k) is None
    assert mismatch(SWAP_AB, f, replace(k, split_kv=1))
    assert mismatch(SWAP_AB, replace(f, thd=True), k)
    assert mismatch(SWAP_AB, replace(f, attn_scale_prefolded=True), k)


def test_swap_ab_64_column_resource_boundary():
    f = facts(s_q=1)
    k = SdpaFwdKnobs(tile_m=64, pack_gqa=True)
    assert mismatch(SWAP_AB, f, k) is None
    assert mismatch(SWAP_AB, f, replace(k, pack_gqa=False))
    assert mismatch(SWAP_AB, replace(f, has_paged_kv=True), k)
    assert mismatch(SWAP_AB, replace(f, has_paged_kv=True, causal=True), k) is None


@pytest.mark.parametrize(
    "folder,name,d",
    [
        ("d192_f16_multi_q", D192, 192),
        ("d256_f16_lookahead", LOOKAHEAD, 256),
        ("d256_f16_lookahead", LOOKAHEAD, 128),
        ("d256_f16_swap_ab", SWAP_AB, 256),
    ],
)
@pytest.mark.parametrize("sq", [1, 129, 8192])
def test_fixed_host_passes_the_named_body_to_common_lowering(folder, name, d, sq):
    host = importlib.import_module("cudnn.sdpa.fwd.kernels.sm107." + folder + ".host")
    calls = []
    f = facts(d, s_q=sq, thd=d == 128)
    sentinel_spec = object()
    host.lower(sentinel_spec, f, None, lower_impl=lambda *a, **k: calls.append((a, k)), knob_type=SdpaFwdKnobs)
    args, kw = calls[0]
    assert args[:2] == (sentinel_spec, f)
    assert kw["selected_kernel"].implementation == name
    assert kw["selected_kernel"].kernel_file == f"sm107/{folder}/kernel.py"
