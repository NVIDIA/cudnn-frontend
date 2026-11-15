# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM80 K-prefetch online half identity and domain; no runtime imports.

The body serves the gptoss / llama / dsv3 flavor envelopes: every eight-aligned
head-dim pair with D_QK <= 192 and D_V <= 128 (config_sm80.pick_flavor). The
adapter keeps choosing its tile from the shape, so the head dims alone name
this body.
"""

from dataclasses import dataclass

from cudnn import data_type

NAME = "sdpa_fwd_sm80_f16_k_prefetch"
TEMPLATE_NAME = "prefill_f16"
KERNEL_FILE = "sm80/f16_k_prefetch/kernel.py"
LEGACY_ROW = "sdpa_fwd_prefill_sm80"
DEVICE_CCS = ((8, 0),)
# config_sm80.FLAVOR_CFGS gptoss / llama / dsv3; a test pins the equality.
FLAVORS = ("gptoss", "llama", "dsv3")
D_QK_MAX, D_V_MAX = 192, 128
SHAPES = frozenset((q, v) for q in range(8, D_QK_MAX + 1, 8) for v in range(8, D_V_MAX + 1, 8))


def in_domain(facts):
    return (
        facts.device_cc in DEVICE_CCS
        and (facts.d_qk, facts.d_v) in SHAPES
        and not (facts.invalid or facts.is_backward)
        and not (facts.is_fp8 or facts.is_mxfp8)
        and facts.dtype in (data_type.HALF, data_type.BFLOAT16)
    )


@dataclass(frozen=True)
class Geometry:
    implementation: str = NAME
    lowering_row_name: str = LEGACY_ROW
    template: str = TEMPLATE_NAME
    kernel_file: str = KERNEL_FILE

    def canonical_knobs(self, knobs):
        # Tiles stay unset unless requested: the adapter's shape rule chooses them, as on the legacy row.
        return knobs

    def compiled_mismatch(self, template, params, knobs):
        """``(config, reason)``: the compiled body's identity and why it differs from this selection."""
        config = dict(
            template=template,
            d_qk=params.d_qk,
            d_v=params.d_v,
            query_tile=params.tile_m,
            kv_tile=params.tile_n,
            scheduler=params.sched_policy,
            split_kv=params.split_kv,
        )
        expected = dict(config, template=self.template)
        for key, value in (("query_tile", knobs.tile_m), ("kv_tile", knobs.tile_n), ("scheduler", knobs.sched_policy), ("split_kv", knobs.split_kv)):
            if value is not None:
                expected[key] = value
        return config, (None if config == expected else f"expected {expected}")


def explicit_geometry(knobs=None):
    """The fixed body; no crossover."""
    return Geometry()


def select_geometry(facts, knobs):
    return Geometry()


def mismatch(facts, knobs):
    if facts.device_cc not in DEVICE_CCS:
        return f"requires SM80 (A100); current device is {facts.device_cc}"
    if not in_domain(facts):
        return f"{NAME} serves eight-aligned FP16/BF16 head dims with D_QK <= {D_QK_MAX} and D_V <= {D_V_MAX}"
    return None
