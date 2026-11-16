# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed SM90 dedicated D128 identity, with lightweight legality and geometry."""

from dataclasses import dataclass, replace
from cudnn import data_type

NAME = "sdpa_fwd_sm90_d128_multi_q"
TEMPLATE_NAME = "prefill_d128_f16"
KERNEL_FILE = "sm90/d128_f16_multi_q/kernel.py"
LEGACY_ROW = "sdpa_fwd_prefill_sm90"


def in_domain(facts):
    return (
        facts.device_cc == (9, 0)
        and (facts.d_qk, facts.d_v) == (128, 128)
        and not (facts.invalid or facts.is_backward or facts.is_fp8 or facts.is_mxfp8)
        and facts.dtype in (data_type.HALF, data_type.BFLOAT16)
    )


@dataclass(frozen=True)
class Geometry:
    implementation: str = NAME
    template: str = TEMPLATE_NAME
    kernel_file: str = KERNEL_FILE
    lowering_row_name: str = LEGACY_ROW
    query_tile: int = 128
    kv_tile: int = 128
    cga: int = 1
    flavor: tuple = (128, 128)
    native_d64: bool = False

    def canonical_knobs(self, knobs):
        return replace(
            knobs,
            tile_m=128,
            tile_n=128,
            cga=1,
            pack_gqa=False if knobs.pack_gqa is None else knobs.pack_gqa,
            split_kv=1 if knobs.split_kv is None else knobs.split_kv,
        )

    def compiled_mismatch(self, template, params, knobs):
        config = dict(
            template=template,
            query_tile=self.query_tile,
            kv_tile=self.kv_tile,
            scheduler=params.sched_policy,
            pack_gqa=bool(params.pack_gqa),
            split_kv=params.split_kv,
        )
        expected = dict(
            template=self.template,
            query_tile=knobs.tile_m,
            kv_tile=knobs.tile_n,
            # None is an omitted public tuning value. The shared adapter resolves
            # it using the same persistent/causal-walk rule as standalone.
            scheduler=params.sched_policy if knobs.sched_policy is None else knobs.sched_policy,
            pack_gqa=bool(knobs.pack_gqa),
            split_kv=knobs.split_kv,
        )
        from .config import TemplateParams

        if type(params) is not TemplateParams:
            return config, f"compiled parameters belong to {type(params).__module__}, expected {TemplateParams.__module__}"
        return config, None if config == expected else f"expected {expected}"


def mismatch(facts, knobs):
    if not in_domain(facts):
        return "requires exact D128/V128 FP16/BF16 on SM90"
    if knobs is not None:
        if knobs.tile_m not in (None, 128) or knobs.tile_n not in (None, 128):
            return "SM90 sdpa_fwd_sm90_d128_multi_q requires TILE_M=TILE_N=128"
        if knobs.cga not in (None, 1):
            return "SM90 sdpa_fwd_sm90_d128_multi_q requires one-CTA clusters"
    return None


def declaration_mismatch(facts):
    # Common band and storage diagnostics run before graph-only body limits.
    # Pack/split are already refused by this owner's declared knob domains.
    from cudnn.sdpa.fwd.config_sm90 import d128_graph_decline

    return d128_graph_decline(facts)
