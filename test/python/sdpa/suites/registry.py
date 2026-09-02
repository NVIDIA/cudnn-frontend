# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Master list of the sdpa/suites test framework.

Every suite is one SuiteSpec: phase (context/generation/bprop), dtype,
level, sweep size + seed, the knob set (what is fuzzed), the post hook
(what is forced), and any platform/version gates. ``COVERAGE.md`` is
rendered from this file by ``gen_coverage.py`` — edit here, regenerate there.

Same-geometry pairs: a fp16 suite and its bf16 sibling share rng_seed and
knob set, so they sweep identical geometry in both dtypes.
"""

from functools import partial

import torch

from sdpa.suites import knobs
from sdpa.suites.common import (
    SuiteSpec,
    combine,
    post_paged,
    post_train,
)
from sdpa.suites.models.catalog import CATALOG


def post_mxfp8(cfg, rng, request):
    cfg.is_mxfp8 = True


def post_mxfp8_bwd_flags(cfg, rng, request):
    cfg.use_causal_mask = cfg.left_bound is None and cfg.right_bound == 0


_COMMON_FUZZ = (
    "batch",
    "s_q/s_kv",
    "d_qk/d_v",
    "heads (MHA/GQA/MQA)",
    "strides+gaps",
    "data",
)
_MASK_FUZZ = ("mask: causal/left/right/band/none", "diag TL/BR")
_THD_FUZZ = (
    "stats token/head-major",
    "total_q/kv slack",
    "declare totals on graph",
    "ragged token gaps",
)

_SPECS = [
    # ---- context (prefill forward) ----
    SuiteSpec(
        name="context.fp16.dense",
        phase="context",
        dtype="fp16",
        level="L0",
        num_tests=128,
        rng_seed=888,
        knobs=partial(knobs.dense_fwd, torch.float16),
        fuzzed=_COMMON_FUZZ
        + _MASK_FUZZ
        + ("layout padded/cu_padded/full", "sink", "bias(1:5)", "unfuse_fma"),
        pinned=("infer",),
    ),
    SuiteSpec(
        name="context.bf16.dense",
        phase="context",
        dtype="bf16",
        level="L0",
        num_tests=128,
        rng_seed=888,
        knobs=partial(knobs.dense_fwd, torch.bfloat16),
        fuzzed=_COMMON_FUZZ
        + _MASK_FUZZ
        + ("layout padded/cu_padded/full", "sink", "bias(1:5)", "unfuse_fma"),
        pinned=("infer",),
    ),
    SuiteSpec(
        name="context.fp16.thd",
        phase="context",
        dtype="fp16",
        level="L0",
        num_tests=256,
        rng_seed=890,
        knobs=partial(knobs.thd_fwd, torch.float16),
        fuzzed=_COMMON_FUZZ + _MASK_FUZZ + _THD_FUZZ + ("sink",),
        pinned=("infer", "layout THD (ragged/cu_ragged)"),
    ),
    SuiteSpec(
        name="context.bf16.thd",
        phase="context",
        dtype="bf16",
        level="L0",
        num_tests=256,
        rng_seed=890,
        knobs=partial(knobs.thd_fwd, torch.bfloat16),
        fuzzed=_COMMON_FUZZ + _MASK_FUZZ + _THD_FUZZ + ("sink",),
        pinned=("infer", "layout THD (ragged/cu_ragged)"),
    ),
    SuiteSpec(
        name="context.fp16.thd_offset_mult",
        phase="context",
        dtype="fp16",
        level="L1",
        num_tests=128,
        rng_seed=892,
        knobs=partial(knobs.thd_offset_mult, torch.float16),
        fuzzed=_COMMON_FUZZ + _THD_FUZZ + ("layout ragged_mult/cu_ragged_mult", "sink"),
        pinned=("infer", "no mask", "diag TL"),
        notes="ragged offset multiplier; engines without the attribute waive at build",
    ),
    SuiteSpec(
        name="context.fp8.dense",
        phase="context",
        dtype="fp8",
        level="L0",
        num_tests=384,
        rng_seed=999,
        knobs=knobs.fp8_fwd,
        exec_kind="fp8",
        fuzzed=_COMMON_FUZZ
        + _MASK_FUZZ
        + (
            "e4m3/e5m2 in",
            "out fp8/fp16",
            "layout padded/full",
            "sink",
        ),
        pinned=("infer",),
    ),
    SuiteSpec(
        name="context.fp8.thd",
        phase="context",
        dtype="fp8",
        level="L0",
        num_tests=384,
        rng_seed=996,
        knobs=knobs.fp8_thd_fwd,
        exec_kind="fp8",
        fuzzed=_COMMON_FUZZ
        + (
            "e4m3/e5m2 in",
            "out fp8/fp16",
            "layout ragged/cu_ragged/cu_ragged_mult",
            "total_q/kv slack",
            "declare totals on graph",
        ),
        pinned=("infer", "no mask", "diag TL"),
    ),
    SuiteSpec(
        name="context.mxfp8.dense",
        phase="context",
        dtype="mxfp8",
        level="L0",
        num_tests=384,
        rng_seed=1001,
        knobs=knobs.mxfp8_fwd,
        exec_kind="mxfp8",
        min_sm=(10, 0),
        post=post_mxfp8,
        fuzzed=_COMMON_FUZZ
        + _MASK_FUZZ
        + (
            "e4m3/e5m2 in",
            "out fp16/bf16",
            "sink",
            "unfuse_fma",
        ),
        pinned=(
            "infer",
            "SM100+",
            "layout full (mxfp8 API has no seq-len args, #646)",
        ),
    ),
    SuiteSpec(
        name="context.mxfp8.thd",
        phase="context",
        dtype="mxfp8",
        level="L0",
        num_tests=128,
        rng_seed=1003,
        knobs=knobs.mxfp8_thd_fwd,
        exec_kind="mxfp8",
        min_sm=(10, 0),
        post=post_mxfp8,
        fuzzed=_COMMON_FUZZ
        + _MASK_FUZZ
        + (
            "e4m3/e5m2 in",
            "out fp16/bf16",
            "layout ragged/cu_ragged",
            "sink",
            "total_q/kv slack",
            "declare totals on graph",
        ),
        pinned=(
            "infer",
            "stats token-major TH1",
            "d=128/128 (frost THD leg)",
            "SM100+",
        ),
        notes="fwd only (no THD mxfp8 bwd engine); needs opt-in FROST engine "
        "(CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1) — skips otherwise: the native "
        "backend check_support-accepts THD mxfp8 but cannot execute it",
    ),
    # ---- generation (decode / small-s_q forward) ----
    SuiteSpec(
        name="generation.fp16.decode",
        phase="generation",
        dtype="fp16",
        level="L0",
        num_tests=64,
        rng_seed=111,
        knobs=partial(knobs.decode, torch.float16),
        fuzzed=_COMMON_FUZZ + ("diag TL/BR",),
        pinned=("infer", "s_q=1", "no mask", "layout full"),
    ),
    SuiteSpec(
        name="generation.bf16.decode",
        phase="generation",
        dtype="bf16",
        level="L0",
        num_tests=64,
        rng_seed=111,
        knobs=partial(knobs.decode, torch.bfloat16),
        fuzzed=_COMMON_FUZZ + ("diag TL/BR",),
        pinned=("infer", "s_q=1", "no mask", "layout full"),
    ),
    SuiteSpec(
        name="generation.fp16.lean",
        phase="generation",
        dtype="fp16",
        level="L0",
        num_tests=64,
        rng_seed=222,
        knobs=partial(knobs.lean_attn, torch.float16),
        fuzzed=_COMMON_FUZZ + ("diag TL/BR", "layout padded/full"),
        pinned=("infer", "s_q=1", "s_kv 513..4096", "no mask"),
    ),
    SuiteSpec(
        name="generation.bf16.lean",
        phase="generation",
        dtype="bf16",
        level="L0",
        num_tests=64,
        rng_seed=222,
        knobs=partial(knobs.lean_attn, torch.bfloat16),
        fuzzed=_COMMON_FUZZ + ("diag TL/BR", "layout padded/full"),
        pinned=("infer", "s_q=1", "s_kv 513..4096", "no mask"),
    ),
    SuiteSpec(
        name="generation.fp16.paged",
        phase="generation",
        dtype="fp16",
        level="L0",
        num_tests=128,
        rng_seed=887,
        knobs=partial(knobs.paged, torch.float16),
        post=post_paged,
        fuzzed=_COMMON_FUZZ + _MASK_FUZZ + ("block size 1..1024", "sink"),
        pinned=("infer", "s_q<=64", "layout padded", "paged KV"),
    ),
    SuiteSpec(
        name="generation.bf16.paged",
        phase="generation",
        dtype="bf16",
        level="L0",
        num_tests=128,
        rng_seed=887,
        knobs=partial(knobs.paged, torch.bfloat16),
        post=post_paged,
        fuzzed=_COMMON_FUZZ + _MASK_FUZZ + ("block size 1..1024", "sink"),
        pinned=("infer", "s_q<=64", "layout padded", "paged KV"),
    ),
    SuiteSpec(
        name="generation.fp16.thd_chunked",
        phase="generation",
        dtype="fp16",
        level="L0",
        num_tests=96,
        rng_seed=445,
        knobs=partial(knobs.thd_chunked, torch.float16),
        fuzzed=_COMMON_FUZZ + _MASK_FUZZ + _THD_FUZZ,
        pinned=("infer", "s_q<=64", "layout THD (ragged)"),
        notes="varlen chunked generation: packed THD chunks against long KV",
    ),
    SuiteSpec(
        name="generation.bf16.thd_chunked",
        phase="generation",
        dtype="bf16",
        level="L0",
        num_tests=96,
        rng_seed=445,
        knobs=partial(knobs.thd_chunked, torch.bfloat16),
        fuzzed=_COMMON_FUZZ + _MASK_FUZZ + _THD_FUZZ,
        pinned=("infer", "s_q<=64", "layout THD (ragged)"),
        notes="varlen chunked generation: packed THD chunks against long KV",
    ),
    SuiteSpec(
        name="generation.fp8.decode",
        phase="generation",
        dtype="fp8",
        level="L0",
        num_tests=128,
        rng_seed=993,
        knobs=knobs.fp8_decode,
        exec_kind="fp8",
        fuzzed=_COMMON_FUZZ
        + (
            "e4m3/e5m2 in",
            "out fp8/fp16",
            "diag TL/BR",
        ),
        pinned=("infer", "s_q=1", "no mask", "layout full"),
    ),
    SuiteSpec(
        name="generation.fp8.paged",
        phase="generation",
        dtype="fp8",
        level="L0",
        num_tests=96,
        rng_seed=997,
        knobs=knobs.fp8_paged,
        exec_kind="fp8",
        post=post_paged,
        fuzzed=_COMMON_FUZZ
        + (
            "e4m3/e5m2 in",
            "out fp8/fp16",
            "block size 16..128",
        ),
        pinned=("infer", "no mask", "diag TL", "layout padded", "paged KV"),
    ),
    # ---- bprop (training: forward + backward) ----
    SuiteSpec(
        name="bprop.fp16.dense",
        phase="bprop",
        dtype="fp16",
        level="L0",
        num_tests=192,
        rng_seed=844,
        knobs=partial(knobs.dense_bwd, torch.float16),
        post=post_train,
        fuzzed=_COMMON_FUZZ
        + _MASK_FUZZ
        + ("layout padded/full", "deterministic", "sink", "bias(1:7)"),
        pinned=("train"),
    ),
    SuiteSpec(
        name="bprop.bf16.dense",
        phase="bprop",
        dtype="bf16",
        level="L0",
        num_tests=192,
        rng_seed=844,
        knobs=partial(knobs.dense_bwd, torch.bfloat16),
        post=post_train,
        fuzzed=_COMMON_FUZZ
        + _MASK_FUZZ
        + ("layout padded/full", "deterministic", "sink", "bias(1:7)"),
        pinned=("train"),
    ),
    SuiteSpec(
        name="bprop.fp16.thd",
        phase="bprop",
        dtype="fp16",
        level="L0",
        num_tests=256,
        rng_seed=845,
        knobs=partial(knobs.thd_bwd, torch.float16),
        post=post_train,
        fuzzed=_COMMON_FUZZ + _MASK_FUZZ + _THD_FUZZ + ("deterministic", "sink"),
        pinned=("train", "layout THD (ragged)"),
    ),
    SuiteSpec(
        name="bprop.bf16.thd",
        phase="bprop",
        dtype="bf16",
        level="L0",
        num_tests=256,
        rng_seed=845,
        knobs=partial(knobs.thd_bwd, torch.bfloat16),
        post=post_train,
        fuzzed=_COMMON_FUZZ + _MASK_FUZZ + _THD_FUZZ + ("deterministic", "sink"),
        pinned=("train", "layout THD (ragged)"),
    ),
    SuiteSpec(
        name="bprop.fp8.dense",
        phase="bprop",
        dtype="fp8",
        level="L0",
        num_tests=256,
        rng_seed=998,
        knobs=knobs.fp8_bwd,
        exec_kind="fp8",
        post=post_train,
        fuzzed=_COMMON_FUZZ + _MASK_FUZZ + ("out fp8/fp16", "deterministic", "sink"),
        pinned=("train", "e4m3 in", "layout full"),
    ),
    SuiteSpec(
        name="bprop.fp8.thd",
        phase="bprop",
        dtype="fp8",
        level="L0",
        num_tests=256,
        rng_seed=995,
        knobs=knobs.fp8_thd_bwd,
        exec_kind="fp8",
        post=post_train,
        fuzzed=_COMMON_FUZZ
        + ("out fp8/fp16", "deterministic", "sink", "total_q/kv slack"),
        pinned=(
            "train",
            "e4m3 in",
            "no mask",
            "diag TL",
            "layout THD (ragged)",
        ),
        notes="ragged FP8 backward requires cuDNN > 9.21.0",
    ),
    SuiteSpec(
        name="bprop.mxfp8.dense",
        phase="bprop",
        dtype="mxfp8",
        level="L0",
        num_tests=256,
        rng_seed=1002,
        knobs=knobs.mxfp8_bwd,
        exec_kind="mxfp8",
        min_sm=(10, 0),
        post=combine(post_mxfp8, post_mxfp8_bwd_flags, post_train),
        fuzzed=_COMMON_FUZZ + _MASK_FUZZ + ("out fp16/bf16", "sink"),
        pinned=(
            "train",
            "e4m3 in",
            "deterministic",
            "layout full",
            "SM100+",
        ),
    ),
]


def _model_post(phase):
    if phase == "generation":

        def _post(cfg, rng, request):
            cfg.is_paged = rng.random() < 0.5

        return _post
    if phase == "bprop":
        return post_train
    return None


_MODEL_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

for _preset in CATALOG:
    for _phase in ("context", "generation", "bprop"):
        for _dt_name, _dt in _MODEL_DTYPES.items():
            _SPECS.append(
                SuiteSpec(
                    name=f"models.{_preset.name}.{_phase}.{_dt_name}",
                    phase=_phase,
                    dtype=_dt_name,
                    level="L0",
                    num_tests=4,
                    rng_seed=__import__("zlib").crc32(
                        f"{_preset.name}.{_phase}".encode()
                    )
                    % 100000,
                    knobs=partial(knobs.model_knobs, _preset, _phase, _dt),
                    post=_model_post(_phase),
                    fuzzed=("batch", "seq lens", "layout", "mask flavor", "data")
                    + (("paged 50%",) if _phase == "generation" else ()),
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

REGISTRY = {spec.name: spec for spec in _SPECS}
assert len(REGISTRY) == len(_SPECS), "duplicate suite names in registry"
