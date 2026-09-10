# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Master list of the sdpa/suites test framework.

Random-suite SuiteSpecs (and their knob factories) live next to their test
shims in ``context/``, ``generation/`` and ``bprop/`` — each of those modules
exports a ``SUITES`` list. This module aggregates them, generates the model
suites from ``models/catalog.py``, and exposes ``REGISTRY``: the single
lookup used by ``gen_coverage.py`` (which renders ``COVERAGE.md``) and the
model-suite parametrization. The aggregation also enforces globally unique
suite names.

16-bit is one family: f16 suites draw fp16 or bf16 per config (data_type
fuzz), exactly like the fp8 suites draw e4m3/e5m2 — no per-dtype duplicates.
"""

import zlib
from functools import partial

from sdpa.suites import knobs
from sdpa.suites.common import (
    SuiteSpec,
    combine,
    post_mxfp8,
    post_mxfp8_bwd_flags,
    post_train,
)
from sdpa.suites.models.catalog import CATALOG

from sdpa.suites.context import test_context_f16, test_context_fp8, test_context_mxfp8
from sdpa.suites.generation import test_generation_f16, test_generation_fp8
from sdpa.suites.bprop import test_bprop_f16, test_bprop_fp8, test_bprop_mxfp8

_SPECS = []
for _mod in (
    test_context_f16,
    test_context_fp8,
    test_context_mxfp8,
    test_generation_f16,
    test_generation_fp8,
    test_bprop_f16,
    test_bprop_fp8,
    test_bprop_mxfp8,
):
    _SPECS += _mod.SUITES


def _model_post(phase):
    if phase == "generation":

        def _post(cfg, rng, request):
            # Paged + ragged (packed THD Q/O against a paged KV cache) is a
            # valid serving combo but deferred — harness support only existed
            # for f16; tracked as a suite-wide extension (f16 + fp8) in the
            # issue tracker. Until then only non-THD draws get paged.
            cfg.is_paged = rng.random() < 0.5 and not cfg.is_ragged

        return _post
    if phase == "bprop":
        return post_train
    return None


for _preset in CATALOG:
    for _phase in ("context", "generation", "bprop"):
        # fp8 flavor only within the fp8 head-dim envelope (d<=192, the cap the
        # generic context.fp8 suite draws). qwen3.5's d=256 fp8 forward returns
        # garbage instead of declining (backend, tracked in an issue), so it
        # gets no fp8 flavor here.
        if _preset.head_dim_qk <= 192 and _preset.head_dim_vo <= 192:
            _SPECS.append(
                SuiteSpec(
                    name=f"models.{_preset.name}.{_phase}.fp8",
                    phase=_phase,
                    dtype="fp8",
                    level="L0",
                    num_tests=8,
                    rng_seed=zlib.crc32(f"{_preset.name}.{_phase}.fp8".encode()) % 100000,
                    knobs=partial(knobs.model_knobs_fp8, _preset, _phase),
                    exec_kind="fp8",
                    post=_model_post(_phase),
                    fuzzed=("batch", "seq lens", "layout", "mask flavor", "data", "e4m3/e5m2 in", "out fp8/fp16")
                    + (("paged 50%",) if _phase == "generation" else ()),
                    pinned=(
                        f"h_q={_preset.num_q_heads}",
                        f"h_kv={_preset.num_kv_heads}",
                        f"d_qk={_preset.head_dim_qk}",
                        f"d_v={_preset.head_dim_vo}",
                    ),
                    notes=f"{_preset.name} full/global attention layers, fp8-trained flavor",
                )
            )
        _SPECS.append(
            SuiteSpec(
                name=f"models.{_preset.name}.{_phase}",
                phase=_phase,
                dtype="f16",
                level="L0",
                num_tests=8,
                rng_seed=zlib.crc32(f"{_preset.name}.{_phase}".encode()) % 100000,
                knobs=partial(knobs.model_knobs, _preset, _phase),
                post=_model_post(_phase),
                fuzzed=("batch", "seq lens", "layout", "mask flavor", "data") + (("paged 50%",) if _phase == "generation" else ()),
                pinned=(
                    f"h_q={_preset.num_q_heads}",
                    f"h_kv={_preset.num_kv_heads}",
                    f"d_qk={_preset.head_dim_qk}",
                    f"d_v={_preset.head_dim_vo}",
                    f"sink={'fuzzed' if (_preset.with_sink and _phase != 'generation') else 'off'}",
                ),
                notes=f"{_preset.name} full/global attention layers",
            )
        )

# mxfp8 model flavor: context and bprop only — mxfp8 is a prefill/training
# format with no decode-shaped engine, so a generation flavor would be a
# 100%-waived shell. Out-of-envelope head dims (qwen35 d=256) waive at build.
for _preset in CATALOG:
    for _phase in ("context", "bprop"):
        _SPECS.append(
            SuiteSpec(
                name=f"models.{_preset.name}.{_phase}.mxfp8",
                phase=_phase,
                dtype="mxfp8",
                level="L0",
                num_tests=8,
                rng_seed=zlib.crc32(f"{_preset.name}.{_phase}.mxfp8".encode()) % 100000,
                knobs=partial(knobs.model_knobs_mxfp8, _preset, _phase),
                exec_kind="mxfp8",
                min_sm=(10, 0),
                post=combine(post_mxfp8, post_mxfp8_bwd_flags, post_train) if _phase == "bprop" else post_mxfp8,
                fuzzed=("batch", "seq lens", "mask flavor", "data", "e4m3/e5m2 in", "out fp16/bf16"),
                pinned=(
                    f"h_q={_preset.num_q_heads}",
                    f"h_kv={_preset.num_kv_heads}",
                    f"d_qk={_preset.head_dim_qk}",
                    f"d_v={_preset.head_dim_vo}",
                    "layout full",
                    "SM100+",
                ),
                notes=f"{_preset.name} mxfp8 flavor; no generation (no decode-shaped mxfp8 engine)",
            )
        )

REGISTRY = {spec.name: spec for spec in _SPECS}
assert len(REGISTRY) == len(_SPECS), "duplicate suite names in registry"
