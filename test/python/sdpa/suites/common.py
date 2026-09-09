# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared machinery for the sdpa/suites test framework.

The framework is a registry-driven re-organization of the test_mhas_v2.py
fuzz coverage:

  - Test files under ``context/``, ``generation/`` and ``bprop/`` declare
    their suites in place: knob factory + SuiteSpec + shim, exported via a
    per-module ``SUITES`` list.
  - ``registry.py`` aggregates those lists, generates the model suites from
    ``models/catalog.py``, and exposes ``REGISTRY`` — the master lookup.
  - ``knobs.py`` keeps only shared ingredients (mask/diag tables, the 16-bit
    family draw, model knob factories).
  - ``COVERAGE.md`` is rendered from the registry by ``gen_coverage.py``.

Seeds are fully deterministic (no environment-variable overrides): a suite's
``(num_tests, rng_seed)`` pair defines its whole sweep, and every config
prints a self-contained repro command (``test_repro_suite.py::test_repro``).
"""

import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import pytest
import torch

from sdpa.random_config import ExecConfig, RandomizationContext
from sdpa.fp16 import exec_sdpa
from sdpa.fp8 import exec_sdpa_fp8
from sdpa.mxfp8 import exec_sdpa_mxfp8
from sdpa.helpers import print_section_begin

REPRO_FILE = os.path.join(os.path.dirname(__file__), "test_repro_suite.py")


def make_seeds(*, num_tests, rng_seed):
    """Deterministic seed sweep: (test_index, num_tests, data_seed) tuples."""
    rng = random.Random(rng_seed)
    return [(i + 1, num_tests, rng.randint(65536, 2147483647)) for i in range(num_tests)]


class Fixed:
    """Knob that always yields the same value (pins a dimension in a fuzz suite)."""

    def __init__(self, value):
        self.value = value

    def __call__(self, rng):
        return self.value


@dataclass(frozen=True)
class SuiteSpec:
    """One row of the master list. ``fuzzed``/``pinned`` are the human-readable
    contract of the suite and feed COVERAGE.md; ``knobs`` is the executable
    version of the same contract."""

    name: str  # e.g. "context.fp16.dense"
    phase: str  # context | generation | bprop
    dtype: str  # fp16 | bf16 | fp8 | mxfp8
    level: str  # pytest marker: L0 | L1
    num_tests: int
    rng_seed: int
    knobs: Callable[[], dict]  # -> kwargs for RandomizationContext
    exec_kind: str = "fp16"  # fp16 | fp8 | mxfp8 (which exec harness)
    post: Optional[Callable] = None  # post(cfg, rng, request) tweaks after randomization
    min_sm: Optional[tuple] = None  # torch.cuda.get_device_capability() gate
    fuzzed: tuple = ()
    pinned: tuple = ()
    notes: str = ""

    def seeds(self):
        return make_seeds(num_tests=self.num_tests, rng_seed=self.rng_seed)


def suite_seeds(name_or_spec):
    """Parametrize helper: seeds for a suite (SuiteSpec or registered name).

    Passing the SuiteSpec object (the norm for the random suites, whose specs
    live in the same test module) avoids importing the registry at collection
    time — required, since the registry itself imports the test modules."""
    if isinstance(name_or_spec, SuiteSpec):
        return name_or_spec.seeds()
    from sdpa.suites.registry import REGISTRY

    return REGISTRY[name_or_spec].seeds()


def model_params(phase):
    """Parametrize helper for the model suites of a phase: (suite_name, test_no)
    pairs with readable ids like ``llama31-fp16-test1``."""
    from sdpa.suites.registry import REGISTRY

    params = []
    for name, spec in REGISTRY.items():
        if not name.startswith("models.") or spec.phase != phase:
            continue
        model = name.split(".")[1]
        params += [pytest.param(name, t, id=f"{model}-{spec.dtype}-test{t[0]}") for t in spec.seeds()]
    return params


def _show_config(spec, cfg, test_no, env_info, request):
    is_dryrun = request.config.option.dryrun
    print()
    print_section_begin("DRY-RUN" if is_dryrun else "")
    print(
        f"#### Suite {spec.name}: test #{test_no[0]} of {test_no[1]} at",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "\n",
    )
    print(f"test_name        = {request.node.name}")
    print(f"platform_info    = {env_info['gpu_arch']} ({env_info['gpu_info']}), cudnn_ver={env_info['cudnn_ver']}")
    print()
    print(cfg.to_repro_cmd(REPRO_FILE))
    print(flush=True)


_EXEC = {
    "fp16": exec_sdpa,
    "fp8": exec_sdpa_fp8,
    "mxfp8": exec_sdpa_mxfp8,
}


def build_config(spec, test_no):
    """Draw one config for a suite: the same two-seed scheme as test_mhas_v2
    (geometry from the parametrized tuple's hash, data from its payload)."""
    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]
    rng = random.Random(geom_seed)
    with RandomizationContext(**spec.knobs()) as ctx:
        cfg = ctx(rng, data_seed, geom_seed)
    return cfg, rng


def run_suite(spec, env_info, test_no, request, cudnn_handle):
    if not isinstance(spec, SuiteSpec):  # registered name (model suites)
        from sdpa.suites.registry import REGISTRY

        spec = REGISTRY[spec]

    if spec.min_sm is not None and torch.cuda.get_device_capability() < spec.min_sm:
        pytest.skip(f"{spec.name} requires SM >= {spec.min_sm}")

    cfg, rng = build_config(spec, test_no)

    if spec.post is not None:
        spec.post(cfg, rng, request)

    _show_config(spec, cfg, test_no, env_info, request)

    _EXEC[spec.exec_kind](cfg, request, cudnn_handle)


# ---- shared fuzz-column vocabulary (SuiteSpec.fuzzed building blocks) -------

COMMON_FUZZ = (
    "batch",
    "s_q/s_kv",
    "d_qk/d_v",
    "heads (MHA/GQA/MQA)",
    "strides+gaps",
    "data",
)
MASK_FUZZ = ("mask: causal/left/right/band/none", "diag TL/BR")
THD_FUZZ = (
    "stats token/head-major",
    "total_q/kv slack",
    "declare totals on graph",
    "ragged token gaps",
)


# ---- shared post() helpers -------------------------------------------------


def post_mxfp8(cfg, rng, request):
    cfg.is_mxfp8 = True


def post_mxfp8_bwd_flags(cfg, rng, request):
    cfg.use_causal_mask = cfg.left_bound is None and cfg.right_bound == 0


def post_train(cfg, rng, request):
    cfg.is_infer = False


def post_paged(cfg, rng, request):
    cfg.is_paged = True


def combine(*posts):
    def _post(cfg, rng, request):
        for p in posts:
            p(cfg, rng, request)

    return _post
