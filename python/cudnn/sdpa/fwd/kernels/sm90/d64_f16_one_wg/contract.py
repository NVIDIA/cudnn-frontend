# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed SM90 native D64 identity, with lightweight legality and geometry."""

from dataclasses import dataclass, replace
from cudnn import data_type

NAME = "sdpa_fwd_sm90_d64_one_wg"
TEMPLATE_NAME = "prefill_d64_f16"
KERNEL_FILE = "sm90/d64_f16_one_wg/kernel.py"
LEGACY_ROW = "sdpa_fwd_prefill_sm90"


def in_domain(facts):
    # Exact D64: the three-WG body owns its paged leg;
    # the native two-WG body owns only the already-admitted nonpaged subset.
    return (
        facts.device_cc == (9, 0)
        and (facts.d_qk, facts.d_v) == (64, 64)
        and not facts.has_paged_kv
        and not (facts.invalid or facts.is_backward or facts.is_fp8 or facts.is_mxfp8)
        and facts.dtype in (data_type.HALF, data_type.BFLOAT16)
    )


@dataclass(frozen=True)
class Geometry:
    implementation: str = NAME
    template: str = TEMPLATE_NAME
    kernel_file: str = KERNEL_FILE
    lowering_row_name: str = LEGACY_ROW
    query_tile: int = 64
    kv_tile: int = 64
    cga: int = 1
    flavor: tuple = (512, 512)
    native_d64: bool = True

    def canonical_knobs(self, knobs):
        return replace(
            knobs,
            tile_m=64,
            tile_n=64,
            cga=1,
            sched_policy=0 if knobs.sched_policy is None else knobs.sched_policy,
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
            scheduler=knobs.sched_policy,
            pack_gqa=bool(knobs.pack_gqa),
            split_kv=knobs.split_kv,
        )
        from .config import TemplateParams

        if type(params) is not TemplateParams:
            return config, f"compiled parameters belong to {type(params).__module__}, expected {TemplateParams.__module__}"
        return config, None if config == expected else f"expected {expected}"


def mismatch(facts, knobs):
    if not in_domain(facts):
        return "requires exact D64/V64 FP16/BF16 on SM90 with nonpaged KV"
    if knobs is not None:
        if knobs.tile_m not in (None, 64) or knobs.tile_n not in (None, 64):
            return "SM90 sdpa_fwd_sm90_d64_one_wg requires TILE_M=TILE_N=64"
        if knobs.cga not in (None, 1):
            return "SM90 sdpa_fwd_sm90_d64_one_wg requires one-CTA clusters"
    return None


def declaration_mismatch(facts):
    """Shape-dependent native D64 limits, after the common band/layout checks."""
    from cudnn.sdpa.fwd.config_sm90 import native_d64_mismatch

    return native_d64_mismatch(
        thd=facts.thd,
        sink=facts.has_sink,
        pack_gqa=False,
        q_lengths=facts.seq_q_trim or facts.seq_q_t is not None or facts.cu_seq_q_t is not None,
        kv_lengths=facts.padded or facts.seq_kv_t is not None or facts.cu_seq_kv_t is not None,
        window_left=facts.window_left,
        window_right=(facts.right_bound or 1) if facts.right_band_widening else None,
        bottom_right=facts.causal and facts.bottom_right,
        s_q=facts.s_q,
        s_kv=facts.s_kv,
    )
