# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CPU-only contracts of the DSA benchmark suite.

Nothing here touches a GPU: the runner is driven with an explicit compute
capability, ``subprocess.run`` is patched where a worker would be spawned,
and the torch generators run on the CPU device. Run from the repository root::

    python -m unittest discover -s benchmark/dsa/tests -v

The same classes are imported and collected under pytest L0 by
``test/python/deepseek_sparse_attention/cutedsl/test_DSA_benchmark_suite.py``,
which adds the GPU smoke and applies its own L0 marker (an imported class does
not inherit this module's ``pytestmark``; it is here for a direct pytest run).

What is pinned: the legacy ``deepseek_v4`` enumeration (20 cases, labels and
order byte-identical to the runner before the appended axes), the case
counts of the extended configs, every rule of :func:`support.decide_support`
with a positive and a negative probe on compute capabilities (9, 0), (10, 0),
(10, 3) and (10, 7), the byte-footprint constants, the FLOP model against a
brute-force causal count, :func:`kv_pool.make_topk_ids` validity, the
runner's RESULT / UNSUPPORTED parsing and static-skip path, the CSV column
order, the shipped presets against the kernels' variant table, the worker's
argv round trip and the chart generation on synthetic frames.
"""

import contextlib
import dataclasses
import importlib.util
import io
import os
import re
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from benchmark.dsa import benchmark_single_dsa as worker
from benchmark.dsa import kv_pool, support
from benchmark.dsa.config_types import BenchmarkResult, DsaBenchmarkConfig, ModelPreset
from benchmark.dsa.configs import list_configs, load_config
from benchmark.dsa.runner import UNSUPPORTED_EXIT_CODE, DsaBenchmarkRunner, parse_detail_tokens

pytestmark = pytest.mark.L0

CCS = ((9, 0), (10, 0), (10, 3), (10, 7))
DSL_PROBES = (None, "installed CuTe DSL has no target for this device")

# The 20 deepseek_v4 labels, in order, as the runner printed them before the appended axes
# existed (captured from `python -m benchmark.dsa.runner --config deepseek_v4 --dry-run`).
LEGACY_DSV4_LABELS = [
    "dsv4_flash | seq=2048x2048 | H64 d=512/512 K=640 | cudnn | bfloat16 | fwd",
    "dsv4_flash | seq=2048x2048 | H64 d=512/512 K=640 | cudnn | bfloat16 | bwd",
    "dsv4_flash | seq=4096x4096 | H64 d=512/512 K=640 | cudnn | bfloat16 | fwd",
    "dsv4_flash | seq=4096x4096 | H64 d=512/512 K=640 | cudnn | bfloat16 | bwd",
    "dsv4_flash | seq=8192x8192 | H64 d=512/512 K=640 | cudnn | bfloat16 | fwd",
    "dsv4_flash | seq=8192x8192 | H64 d=512/512 K=640 | cudnn | bfloat16 | bwd",
    "dsv4_flash | seq=16384x16384 | H64 d=512/512 K=640 | cudnn | bfloat16 | fwd",
    "dsv4_flash | seq=16384x16384 | H64 d=512/512 K=640 | cudnn | bfloat16 | bwd",
    "dsv4_flash | seq=32768x32768 | H64 d=512/512 K=640 | cudnn | bfloat16 | fwd",
    "dsv4_flash | seq=32768x32768 | H64 d=512/512 K=640 | cudnn | bfloat16 | bwd",
    "dsv4_pro | seq=2048x2048 | H128 d=512/512 K=1152 | cudnn | bfloat16 | fwd",
    "dsv4_pro | seq=2048x2048 | H128 d=512/512 K=1152 | cudnn | bfloat16 | bwd",
    "dsv4_pro | seq=4096x4096 | H128 d=512/512 K=1152 | cudnn | bfloat16 | fwd",
    "dsv4_pro | seq=4096x4096 | H128 d=512/512 K=1152 | cudnn | bfloat16 | bwd",
    "dsv4_pro | seq=8192x8192 | H128 d=512/512 K=1152 | cudnn | bfloat16 | fwd",
    "dsv4_pro | seq=8192x8192 | H128 d=512/512 K=1152 | cudnn | bfloat16 | bwd",
    "dsv4_pro | seq=16384x16384 | H128 d=512/512 K=1152 | cudnn | bfloat16 | fwd",
    "dsv4_pro | seq=16384x16384 | H128 d=512/512 K=1152 | cudnn | bfloat16 | bwd",
    "dsv4_pro | seq=32768x32768 | H128 d=512/512 K=1152 | cudnn | bfloat16 | fwd",
    "dsv4_pro | seq=32768x32768 | H128 d=512/512 K=1152 | cudnn | bfloat16 | bwd",
]

# The legacy CSV schema, in order, followed by the appended columns.
LEGACY_COLUMNS = [
    "config_name",
    "model_name",
    "backend",
    "data_type",
    "q_seqlen",
    "kv_seqlen",
    "num_q_heads",
    "head_dim_qk",
    "head_dim_vo",
    "topk",
    "indexer_topk",
    "has_sink",
    "use_topk_length",
    "profile_pass",
    "deterministic_bwd",
    "time_ms",
    "tflops",
    "num_iterations",
    "success",
    "skipped",
    "error_message",
    "gpu_name",
    "cudnn_version",
    "peak_mma_tflops",
    "backend_detail",
]
APPENDED_COLUMNS = ["component", "phase", "kv_dtype", "paged", "batch_size", "indexer_mode", "bytes_per_token", "gather_gbps", "time_step1_ms", "time_step2_ms"]

# Kernel messages spell the architecture as SM<digits>; the suite's reasons never do.
_KERNEL_ARCH_SPELLING = re.compile(r"\bSM\d")


def _runner(cc=(10, 0), dsl_target_error=None) -> DsaBenchmarkRunner:
    return DsaBenchmarkRunner(verbose=False, cc=cc, dsl_target_error=dsl_target_error)


def _cases(config_name: str, **filters):
    runner = _runner()
    return runner.filter_cases(list(runner.expand_config(load_config(config_name))), **filters)


def _flat_case(**overrides) -> dict:
    """A DSv4.1-Flash core context case as a flat field dict; keyword overrides set the probe."""
    case = dict(
        component="core",
        phase="context",
        kv_dtype="bfloat16",
        paged=False,
        indexer_mode=None,
        data_type="bfloat16",
        q_seqlen=4096,
        kv_seqlen=4096,
        batch_size=1,
        num_q_heads=64,
        head_dim_qk=512,
        head_dim_vo=512,
        topk=640,
        indexer_topk=512,
        num_kv_heads=1,
        kv_layout="shared",
        index_granularity=1,
        indexer_heads=32,
        indexer_pool_ratio=1,
        mtp_rows=2,
    )
    case.update(overrides)
    if case["component"] == "indexer" and case["indexer_mode"] is None:
        case["indexer_mode"] = "dense"
    return case


def _servable(config_name: str, cc, dsl_target_error=None) -> int:
    return sum(1 for case in _cases(config_name) if support.decide_support(case, cc, dsl_target_error) is None)


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


class LegacyEnumerationTest(unittest.TestCase):
    """deepseek_v4 must expand exactly as before the appended axes existed."""

    def test_deepseek_v4_expands_to_the_pinned_20_labels_in_order(self):
        runner = _runner()
        cases = list(runner.expand_config(load_config("deepseek_v4")))
        self.assertEqual([runner._case_label(c) for c in cases], LEGACY_DSV4_LABELS)

    def test_legacy_cases_carry_the_default_axes(self):
        for case in _cases("deepseek_v4"):
            self.assertEqual(case["component"], "core")
            self.assertEqual(case["kv_dtype"], "bfloat16")
            self.assertFalse(case["paged"])
            self.assertIsNone(case["indexer_mode"])
            self.assertEqual(case["batch_size"], 1)
            self.assertEqual(case["phase"], {"fwd": "context", "bwd": "bprop"}[case["profile_pass"]])
            self.assertEqual(
                support.static_columns(case),
                dict(component="core", phase=case["phase"], kv_dtype="bfloat16", paged=False, batch_size=1, indexer_mode=None, bytes_per_token=1024),
            )

    def test_phases_follow_profile_pass_when_unset(self):
        config = load_config("deepseek_v4")
        self.assertEqual(support.phases_for_config(config), ["context", "bprop"])
        self.assertEqual(support.phases_for_config(replace(config, profile_pass="fwd")), ["context"])
        self.assertEqual(support.phases_for_config(replace(config, profile_pass="bwd")), ["bprop"])
        self.assertEqual(support.phases_for_config(replace(config, phases=["generation"])), ["generation"])
        with self.assertRaises(ValueError):
            support.phases_for_config(replace(config, profile_pass="sideways"))
        self.assertEqual(len(list(_runner().expand_config(replace(config, profile_pass="fwd")))), 10)
        self.assertEqual([c["profile_pass"] for c in _runner().expand_config(replace(config, profile_pass="bwd"))], ["bwd"] * 10)

    def test_legacy_worker_argv_prefix_is_unchanged(self):
        case = _cases("deepseek_v4")[1]  # dsv4_flash 2048x2048 bwd
        cmd = DsaBenchmarkRunner.worker_command(case)
        self.assertTrue(cmd[1].endswith(os.path.join("benchmark", "dsa", "benchmark_single_dsa.py")))
        legacy = [
            "--profile_pass",
            "bwd",
            "--backend",
            "cudnn",
            "--q_seqlen",
            "2048",
            "--kv_seqlen",
            "2048",
            "--num_q_heads",
            "64",
            "--head_dim_qk",
            "512",
            "--head_dim_vo",
            "512",
            "--topk",
            "640",
            "--indexer_topk",
            "512",
            "--data_type",
            "bfloat16",
            "--num_iterations",
            "20",
            "--num_warmup_iterations",
            "5",
            "--has_sink",
            "--use_topk_length",
        ]
        self.assertEqual(cmd[2 : 2 + len(legacy)], legacy)
        appended = cmd[2 + len(legacy) :]
        self.assertEqual(appended[:4], ["--component", "core", "--phase", "bprop"])
        self.assertNotIn("--paged", appended)
        self.assertNotIn("--indexer_mode", appended)
        self.assertNotIn("--deterministic", cmd)
        self.assertEqual(appended[appended.index("--kv_dtype") + 1], "bfloat16")
        self.assertEqual(appended[appended.index("--batch_size") + 1], "1")


class ExtendedEnumerationTest(unittest.TestCase):
    """Case counts, order and labels of the extended configs."""

    def test_case_counts(self):
        self.assertEqual(len(_cases("deepseek_v41")), 324)
        self.assertEqual(len(_cases("glm53")), 384)
        self.assertEqual(len(_cases("qwen38_qsa")), 168)
        self.assertEqual(len(_cases("deepseek_v41", filter_component="indexer", filter_phase="generation")), 96)

    def test_count_decomposition_dsv41(self):
        cases = _cases("deepseek_v41")
        by_phase = {p: sum(1 for c in cases if c["phase"] == p) for p in support.PHASES}
        # 5 shapes x (core 3 kv_dtypes x 2 paged + indexer 3 x 2 x 2 modes) per shape phase; the generation grid is
        # 2 kv_lens x 2 batch sizes x 2 q_tokens with the same inner loops.
        self.assertEqual(by_phase, {"context": 90, "bprop": 90, "generation": 144})
        self.assertEqual(sum(1 for c in cases if c["component"] == "indexer" and c["indexer_mode"] == "hierarchical"), 108)

    def test_shape_sweep_precedes_generation_and_keeps_the_legacy_loop_order(self):
        cases = _cases("deepseek_v41")
        phases = [c["phase"] for c in cases]
        first_generation = phases.index("generation")
        self.assertTrue(all(p != "generation" for p in phases[:first_generation]))
        self.assertTrue(all(p == "generation" for p in phases[first_generation:]))
        # Within one shape: context rows then bprop rows (today's fwd/bwd order), each fanning out the appended axes.
        shape_0 = [c for c in cases[:first_generation] if (c["q_seqlen"], c["kv_seqlen"]) == (2048, 2048)]
        self.assertEqual([c["phase"] for c in shape_0], ["context"] * 18 + ["bprop"] * 18)
        self.assertEqual([c["component"] for c in shape_0[:18]], ["core"] * 6 + ["indexer"] * 12)
        self.assertEqual([c["kv_dtype"] for c in shape_0[:6]], ["bfloat16", "bfloat16", "mxfp8", "mxfp8", "nvfp4", "nvfp4"])
        self.assertEqual([c["paged"] for c in shape_0[:6]], [False, True] * 3)
        self.assertEqual([c["indexer_mode"] for c in shape_0[6:10]], ["dense", "hierarchical", "dense", "hierarchical"])

    def test_generation_grid_is_per_request(self):
        gen = _cases("deepseek_v41", filter_phase="generation")
        self.assertEqual(sorted({c["kv_seqlen"] for c in gen}), [8192, 131072])
        self.assertEqual(sorted({c["batch_size"] for c in gen}), [8, 128])
        self.assertEqual(sorted({c["q_seqlen"] for c in gen}), [1, 2])  # sorted({1, mtp_rows = 2})
        self.assertTrue(all(c["profile_pass"] == "fwd" for c in gen))
        self.assertTrue(all(not c["deterministic_bwd"] for c in gen))
        self.assertEqual(
            [(c["kv_seqlen"], c["batch_size"], c["q_seqlen"]) for c in gen[::18]],
            [(8192, 8, 1), (8192, 8, 2), (8192, 128, 1), (8192, 128, 2), (131072, 8, 1), (131072, 8, 2), (131072, 128, 1), (131072, 128, 2)],
        )
        # A preset without MTP drafts runs q_tokens = 1 only.
        flash = _cases("glm53", filter_model="glm53_flash", filter_phase="generation")
        self.assertEqual({c["q_seqlen"] for c in flash}, {1})
        self.assertEqual(len(flash), 48)

    def test_labels_tag_only_off_default_axes(self):
        runner = _runner()
        labels = [runner._case_label(c) for c in runner.expand_config(load_config("deepseek_v41"))]
        legacy_prefix = "dsv41_flash | seq=2048x2048 | H64 d=512/512 K=640 | cudnn | bfloat16 | fwd"
        self.assertEqual(labels[0], legacy_prefix)
        # Tag order: component, generation, kv_dtype, paged, mode; each only when off its default.
        self.assertEqual(labels[1], legacy_prefix + " | paged")
        self.assertEqual(labels[13], legacy_prefix + " | indexer | mxfp8 | paged | hierarchical")
        gen = [l for l in labels if "| gen " in l]
        self.assertEqual(gen[0], "dsv41_flash | seq=1x8192 | H64 d=512/512 K=640 | cudnn | bfloat16 | fwd | gen B=8 q=1")
        self.assertEqual(
            gen[-1], "dsv41_flash | seq=2x131072 | H64 d=512/512 K=640 | cudnn | bfloat16 | fwd | indexer | gen B=128 q=2 | nvfp4 | paged | hierarchical"
        )

    def test_filters(self):
        cases = _cases("glm53")
        n = len(cases)
        self.assertEqual(len(DsaBenchmarkRunner.filter_cases(cases, filter_paged=True)), n // 2)
        self.assertEqual(len(DsaBenchmarkRunner.filter_cases(cases, filter_paged=False)), n // 2)
        self.assertEqual(len(DsaBenchmarkRunner.filter_cases(cases, filter_kv_dtype="mxfp8")), n // 3)
        self.assertEqual(len(DsaBenchmarkRunner.filter_cases(cases, filter_model="flash")), sum(1 for c in cases if c["model"].name == "glm53_flash"))
        self.assertEqual(
            DsaBenchmarkRunner.filter_cases(cases, filter_pass="bwd"),
            DsaBenchmarkRunner.filter_cases(cases, filter_phase="bprop"),
        )
        # One core bf16 flat row per generation shape (model, kv_len, batch, q_tokens).
        narrow = DsaBenchmarkRunner.filter_cases(cases, filter_component="core", filter_phase="generation", filter_kv_dtype="bfloat16", filter_paged=False)
        shapes = {(c["model"].name, c["kv_seqlen"], c["batch_size"], c["q_seqlen"]) for c in cases if c["phase"] == "generation"}
        self.assertEqual(len(narrow), len(shapes))

    def test_list_configs_includes_the_opt_in_qsa_config(self):
        self.assertEqual(set(list_configs()) >= {"deepseek_v4", "deepseek_v41", "glm53", "qwen38_qsa"}, True)


class SupportRulesTest(unittest.TestCase):
    """Every rule of decide_support, positive and negative, on cc (9,0), (10,0), (10,3), (10,7)."""

    def decide(self, case, cc=(10, 0), dsl=None):
        reason = support.decide_support(case, cc, dsl)
        if reason is not None:
            self.assertIsNone(_KERNEL_ARCH_SPELLING.search(reason), reason)
        return reason

    def assertServed(self, case, cc=(10, 0), dsl=None):
        self.assertIsNone(self.decide(case, cc, dsl))

    def assertReason(self, case, fragment, cc=(10, 0), dsl=None):
        reason = self.decide(case, cc, dsl)
        self.assertIsNotNone(reason, f"expected a reason containing {fragment!r}")
        self.assertIn(fragment, reason)

    def test_rule_1_nvfp4_is_a_coverage_row_everywhere(self):
        for cc in CCS:
            for dsl in DSL_PROBES:
                for component in ("core", "indexer"):
                    for phase in support.PHASES:
                        self.assertReason(_flat_case(component=component, phase=phase, kv_dtype="nvfp4"), "NVFP4", cc, dsl)
        self.assertServed(_flat_case(kv_dtype="bfloat16"))

    def test_rule_2_core_mxfp8(self):
        for phase in support.PHASES:
            self.assertReason(_flat_case(phase=phase, kv_dtype="mxfp8"), "no MXFP8 KV core kernel")
        self.assertServed(_flat_case(component="indexer", phase="context", kv_dtype="mxfp8"))
        self.assertServed(_flat_case(component="indexer", phase="generation", kv_dtype="mxfp8", q_seqlen=2, kv_seqlen=8192, batch_size=8))

    def test_rule_3_low_precision_bprop(self):
        self.assertReason(_flat_case(component="indexer", phase="bprop", kv_dtype="mxfp8"), "no low-precision backward kernel")
        self.assertServed(_flat_case(component="indexer", phase="bprop", kv_dtype="bfloat16"))
        self.assertServed(_flat_case(phase="bprop", kv_dtype="bfloat16"))

    def test_rule_4_separate_kv_layout_wins_over_the_pair_rule(self):
        qsa = _flat_case(
            num_q_heads=24,
            head_dim_qk=256,
            head_dim_vo=256,
            topk=2048,
            indexer_topk=0,
            num_kv_heads=2,
            kv_layout="separate",
            index_granularity=4,
            indexer_heads=4,
            indexer_pool_ratio=4,
        )
        for cc in CCS:
            for phase in support.PHASES:
                self.assertReason(dict(qsa, phase=phase), "cc 10.7 sparse GQA kernel", cc)
        self.assertServed(_flat_case(kv_layout="shared"))

    def test_rule_5_core_pair_and_indexer_topk_prefix(self):
        self.assertReason(_flat_case(num_q_heads=32), "(32, 512)")
        self.assertReason(_flat_case(num_q_heads=128, head_dim_qk=576), "(128, 576)")
        self.assertReason(_flat_case(indexer_topk=256), "indexer_topk=256")
        self.assertReason(_flat_case(num_q_heads=128, indexer_topk=2048, topk=2176), "indexer_topk=2048")
        self.assertServed(_flat_case(num_q_heads=128, indexer_topk=1024, topk=1152))
        self.assertServed(_flat_case(num_q_heads=64, head_dim_qk=576, topk=2048, indexer_topk=0))
        # The pair rule applies to core rows in every phase; indexer rows ignore the core geometry.
        self.assertReason(_flat_case(phase="bprop", num_q_heads=32), "(32, 512)")
        self.assertServed(_flat_case(component="indexer", num_q_heads=32))

    def test_rule_6_core_forward_needs_a_10x_part(self):
        for phase, shape in (("context", {}), ("generation", dict(q_seqlen=1, kv_seqlen=8192, batch_size=8))):
            self.assertReason(_flat_case(phase=phase, **shape), "sparse core forward requires compute capability 10.x", (9, 0))
            self.assertReason(_flat_case(phase=phase, **shape), "sparse core forward requires compute capability 10.x", None)
            for cc in ((10, 0), (10, 3), (10, 7)):
                for dsl in DSL_PROBES:
                    self.assertServed(_flat_case(phase=phase, **shape), cc, dsl)  # the core forward is exempt from the DSL probe

    def test_rule_7_core_bprop_on_9x_or_10x_paged_allowed(self):
        for cc in CCS:
            self.assertServed(_flat_case(phase="bprop"), cc)
            self.assertServed(_flat_case(phase="bprop", paged=True), cc)
        self.assertReason(_flat_case(phase="bprop"), "sparse core backward requires compute capability 9.x or 10.x", None)
        self.assertReason(_flat_case(phase="bprop"), "9.x or 10.x", (8, 0))

    def test_rule_8_and_9_indexer_heads(self):
        self.assertReason(_flat_case(component="indexer", indexer_heads=0), "model has no indexer rows")
        self.assertReason(_flat_case(component="indexer", indexer_heads=4), "32 or 64 query heads per KV head (model has 4)")
        for heads in (32, 64):
            self.assertServed(_flat_case(component="indexer", indexer_heads=heads))

    def test_rule_10_paged_indexer(self):
        for phase in support.PHASES:
            self.assertReason(_flat_case(component="indexer", phase=phase, paged=True), "no paged index-key input")
        self.assertServed(_flat_case(component="indexer", paged=False))

    def test_rule_11_indexer_architecture(self):
        for phase in ("context", "generation"):
            shape = dict(q_seqlen=1, kv_seqlen=8192, batch_size=8) if phase == "generation" else {}
            self.assertReason(
                _flat_case(component="indexer", phase=phase, **shape), "indexer forward/top-k fused kernel requires compute capability 10.x", (9, 0)
            )
            self.assertReason(_flat_case(component="indexer", phase=phase, **shape), "10.x", None)
            for cc in ((10, 0), (10, 3)):
                self.assertServed(_flat_case(component="indexer", phase=phase, **shape), cc)
        # Backward: 9.x needs 64 heads; 10.x takes 32 or 64; anything else is out.
        self.assertReason(_flat_case(component="indexer", phase="bprop", indexer_heads=32), "requires at least 64 indexer heads (model has 32)", (9, 0))
        self.assertServed(_flat_case(component="indexer", phase="bprop", indexer_heads=64), (9, 0))
        self.assertServed(_flat_case(component="indexer", phase="bprop", indexer_heads=32), (10, 0))
        self.assertReason(_flat_case(component="indexer", phase="bprop", indexer_heads=64), "indexer backward requires compute capability 9.x or 10.x", None)

    def test_rule_12_and_13_hierarchical(self):
        self.assertReason(_flat_case(component="indexer", indexer_mode="hierarchical", kv_dtype="mxfp8"), "candidate-restricted indexer scoring is bf16-only")
        self.assertReason(_flat_case(component="indexer", indexer_mode="hierarchical", phase="bprop"), "hierarchical rows are forward-only")
        self.assertServed(_flat_case(component="indexer", indexer_mode="hierarchical"))
        self.assertServed(_flat_case(component="indexer", indexer_mode="hierarchical", phase="generation", q_seqlen=2, kv_seqlen=131072, batch_size=128))

    def test_rule_14_topk_against_the_per_request_pool(self):
        self.assertReason(_flat_case(phase="generation", q_seqlen=1, kv_seqlen=512, batch_size=8), "per-request kv_seqlen (512) is shorter than topk (640)")
        self.assertReason(_flat_case(q_seqlen=512, kv_seqlen=512), "topk (640) exceeds kv_seqlen (512)")
        self.assertReason(_flat_case(phase="bprop", q_seqlen=512, kv_seqlen=512), "topk (640) exceeds kv_seqlen (512)")
        self.assertServed(_flat_case(q_seqlen=640, kv_seqlen=640))
        # Indexer rows never read the core's topk: short pools are served with -1 padding (the GPU smoke runs S=512 / S=256).
        self.assertServed(_flat_case(component="indexer", q_seqlen=512, kv_seqlen=512))
        self.assertServed(_flat_case(component="indexer", phase="bprop", q_seqlen=256, kv_seqlen=256))

    def test_rule_15_shared_compiler_rows_read_the_dsl_probe_on_cc_10_7_only(self):
        message = "installed CuTe DSL lacks the target for compute capability 10.7"
        gated = [
            _flat_case(phase="bprop"),
            _flat_case(component="indexer"),
            _flat_case(component="indexer", phase="generation", q_seqlen=1, kv_seqlen=8192, batch_size=8),
            _flat_case(component="indexer", phase="bprop"),
        ]
        for case in gated:
            self.assertTrue(support.uses_shared_compiler(case))
            self.assertReason(case, message, (10, 7), DSL_PROBES[1])
            self.assertServed(case, (10, 7), None)
            for cc in ((10, 0), (10, 3)):
                self.assertServed(case, cc, DSL_PROBES[1])
        exempt = [_flat_case(), _flat_case(phase="generation", q_seqlen=2, kv_seqlen=8192, batch_size=8)]
        for case in exempt:
            self.assertFalse(support.uses_shared_compiler(case))
            self.assertServed(case, (10, 7), DSL_PROBES[1])
        # The DSL probe is the LAST rule: a rule above it still names its own reason.
        self.assertReason(_flat_case(component="indexer", paged=True), "no paged index-key input", (10, 7), DSL_PROBES[1])

    def test_servable_cell_counts_mirror_the_readme_skip_matrix(self):
        # The README quotes these two numbers; the per-rule tests above imply the rest.
        self.assertEqual(_servable("deepseek_v41", (10, 0)), 80)
        self.assertEqual(_servable("deepseek_v41", (10, 7), DSL_PROBES[1]), 26)

    def test_every_reason_is_the_suites_own_wording(self):
        for name in ("deepseek_v4", "deepseek_v41", "glm53", "qwen38_qsa"):
            for case in _cases(name):
                for cc in CCS + (None,):
                    for dsl in DSL_PROBES:
                        self.decide(case, cc, dsl)

    def test_case_fields_reads_geometry_from_the_attached_preset(self):
        case = _cases("deepseek_v41", filter_component="indexer", filter_phase="generation")[0]
        fields = support.case_fields(case)
        self.assertEqual((fields["num_q_heads"], fields["head_dim_qk"], fields["topk"], fields["indexer_topk"]), (64, 512, 640, 512))
        self.assertEqual((fields["indexer_heads"], fields["mtp_rows"], fields["phase"], fields["batch_size"]), (32, 2, "generation", 8))
        legacy = dict(model=load_config("deepseek_v4").models[1], profile_pass="bwd", q_seqlen=2048, kv_seqlen=2048)
        fields = support.case_fields(legacy)
        self.assertEqual(
            (fields["phase"], fields["component"], fields["kv_dtype"], fields["paged"], fields["indexer_mode"], fields["batch_size"]),
            ("bprop", "core", "bfloat16", False, None, 1),
        )
        self.assertEqual(fields["num_q_heads"], 128)


class ByteFootprintTest(unittest.TestCase):
    def test_record_bytes_constants(self):
        self.assertEqual([support.record_bytes(512, d) for d in ("bfloat16", "mxfp8", "nvfp4")], [1024, 528, 288])
        self.assertEqual([support.record_bytes(576, d) for d in ("bfloat16", "mxfp8", "nvfp4")], [1152, 594, 324])
        self.assertEqual([support.record_bytes(256, d, "separate", 2) for d in ("bfloat16", "mxfp8", "nvfp4")], [2048, 1056, 576])
        self.assertEqual(support.record_bytes(512, "float16"), 1024)
        with self.assertRaises(ValueError):
            support.record_bytes(512, "int8")
        with self.assertRaises(ValueError):
            support.record_bytes(512, "bfloat16", "interleaved")

    def test_index_key_bytes_constants(self):
        self.assertEqual([support.index_key_bytes(d) for d in ("bfloat16", "mxfp8", "nvfp4")], [256, 132, 68])
        self.assertEqual([support.index_key_bytes(d, 128, 4) for d in ("bfloat16", "mxfp8", "nvfp4")], [64, 33, 17])
        with self.assertRaises(ValueError):
            support.index_key_bytes("fp6")

    def test_kv_pool_re_exports_the_same_functions(self):
        self.assertIs(kv_pool.record_bytes, support.record_bytes)
        self.assertIs(kv_pool.index_key_bytes, support.index_key_bytes)

    def test_bytes_per_token_and_static_columns_per_row(self):
        expected = {
            ("core", "bfloat16"): 1024,
            ("core", "mxfp8"): 528,
            ("core", "nvfp4"): 288,
            ("indexer", "bfloat16"): 256,
            ("indexer", "mxfp8"): 132,
            ("indexer", "nvfp4"): 68,
        }
        for case in _cases("deepseek_v41"):
            self.assertEqual(support.bytes_per_token(case), expected[(case["component"], case["kv_dtype"])])
        for case in _cases("glm53", filter_model="glm53_flash", filter_component="indexer"):
            self.assertEqual(support.bytes_per_token(case), {"bfloat16": 64, "mxfp8": 33, "nvfp4": 17}[case["kv_dtype"]])
        for case in _cases("glm53", filter_model="glm53_flash", filter_component="core", filter_kv_dtype="bfloat16"):
            self.assertEqual(support.bytes_per_token(case), 1024)
        for case in _cases("glm53", filter_component="core", filter_kv_dtype="mxfp8"):
            self.assertEqual(support.bytes_per_token(case), 594 if case["model"].name == "glm53" else 528)
        qsa = _cases("qwen38_qsa")
        self.assertEqual({support.bytes_per_token(c) for c in qsa if c["component"] == "core"}, {2048, 1056, 576})
        self.assertEqual({support.bytes_per_token(c) for c in qsa if c["component"] == "indexer"}, {64, 33, 17})
        columns = support.static_columns(_cases("deepseek_v41", filter_component="indexer", filter_phase="generation")[1])
        self.assertEqual(
            columns,
            dict(component="indexer", phase="generation", kv_dtype="bfloat16", paged=False, batch_size=8, indexer_mode="hierarchical", bytes_per_token=256),
        )
        self.assertEqual(set(columns), set(APPENDED_COLUMNS[:7]))


def _brute_force_visible(q_tokens: int, kv_len: int, phase: str, ratio: int):
    """Keys whose complete ``ratio``-token group lies at or before the row position: the FE kernels' rule."""
    counts = []
    for j in range(q_tokens):
        pos = j if phase == "context" else kv_len - q_tokens + j
        counts.append(sum(1 for k in range(kv_len // ratio) if ratio * (k + 1) - 1 <= pos))
    return counts


class FlopModelTest(unittest.TestCase):
    def test_causal_visible_counts_match_a_brute_force_mask(self):
        for ratio in (1, 4):
            for q_tokens, kv_len, phase in ((37, 37, "context"), (64, 64, "context"), (1, 300, "generation"), (2, 301, "generation"), (5, 64, "generation")):
                with self.subTest(ratio=ratio, q_tokens=q_tokens, kv_len=kv_len, phase=phase):
                    counts = kv_pool.causal_visible_counts(q_tokens, kv_len, phase, ratio)
                    self.assertEqual(counts.dtype, torch.int64)
                    self.assertEqual(counts.device.type, "cpu")
                    self.assertEqual(counts.tolist(), _brute_force_visible(q_tokens, kv_len, phase, ratio))
        self.assertEqual(kv_pool.row_positions(2, 1024, "generation").tolist(), [1022, 1023])
        self.assertEqual(kv_pool.row_positions(3, 1024, "context").tolist(), [0, 1, 2])
        with self.assertRaises(ValueError):
            kv_pool.causal_visible_counts(1, 8, "bprop")
        with self.assertRaises(ValueError):
            kv_pool.causal_visible_counts(1, 8, "context", 0)

    def test_core_flop_formulas(self):
        self.assertEqual(worker.flops_fwd(4096, 64, 640, 512, 512), 2.0 * 4096 * 64 * 640 * 1024)
        self.assertEqual(worker.flops_bwd(4096, 64, 640, 576, 512), 2.0 * 4096 * 64 * 640 * (3 * 576 + 2 * 512))
        self.assertEqual(worker.flops_indexer_bwd(2048, 32, 512), 6.0 * 2048 * 32 * 128 * 512)

    def test_core_flops_count_the_gathered_ids(self):
        # Causal lists at s_q == topk hold ~half the nominal ids, and the TFLOPS must say so.
        _ids, length = kv_pool.make_topk_ids(1, 640, 640, 640, phase="context", device="cpu")
        gathered = worker.gathered_ids(640, 640, length)
        self.assertEqual(gathered, 640 * 641 // 2)
        self.assertEqual(worker.flops_fwd(640, 64, 640, 512, 512, gathered=gathered), 2.0 * 64 * gathered * 1024)
        self.assertEqual(worker.flops_bwd(640, 64, 640, 512, 512, gathered=gathered), 2.0 * 64 * gathered * (3 * 512 + 2 * 512))
        # The worker's context / bprop rows draw full lists: the gathered count is the legacy nominal count.
        _ids, length = kv_pool.make_topk_ids(1, 640, 640, 640, phase="context", causal=False, device="cpu")
        self.assertEqual(worker.gathered_ids(640, 640, length), 640 * 640)
        # Without the compact contract (no topk_length) every slot is gathered: the nominal count.
        self.assertEqual(worker.gathered_ids(640, 640, None), 640 * 640)
        # Generation rows at kv_len >> topk are all full.
        _ids, length = kv_pool.make_topk_ids(2, 2, 4096, 640, phase="generation", device="cpu")
        self.assertEqual(worker.gathered_ids(4, 640, length), 4 * 640)

    def test_indexer_dense_flops_count_visible_keys_only(self):
        # Context, ratio 1: every row j scores j + 1 keys -> S (S + 1) / 2 products, half the dense count.
        s = 256
        visible = int(kv_pool.causal_visible_counts(s, s, "context", 1).sum())
        self.assertEqual(visible, s * (s + 1) // 2)
        self.assertEqual(worker.flops_indexer_logits(32, visible), 2.0 * 32 * 128 * visible)
        # Ratio 4 pools: the first three rows see no complete pool.
        visible4 = kv_pool.causal_visible_counts(8, 8, "context", 4).tolist()
        self.assertEqual(visible4, [0, 0, 0, 1, 1, 1, 1, 2])
        # Generation: all rows see (almost) the whole pool.
        self.assertEqual(kv_pool.causal_visible_counts(2, 8192, "generation", 1).tolist(), [8191, 8192])
        self.assertEqual(kv_pool.causal_visible_counts(1, 8192, "generation", 4).tolist(), [2048])

    def test_hierarchical_candidates_are_causal_valid_with_the_newest_block_pinned(self):
        count, block = 64, 8
        ids, n_valid = worker.make_candidate_ids(2, 4, 100, "generation", count, block, "cpu", chunk_rows=3)
        self.assertEqual(ids.shape, (2, 4, count))
        self.assertEqual(ids.dtype, torch.int32)
        self.assertEqual(n_valid.tolist(), [[count] * 4] * 2)
        for req in range(2):
            for j in range(4):
                pos = 100 - 4 + j
                row = ids[req, j].tolist()
                local = [i - req * 100 for i in row]
                self.assertEqual(len(set(local)), count)
                self.assertTrue(all(0 <= t <= pos for t in local))
                self.assertTrue(set(range(pos - pos % block, pos + 1)) <= set(local))
        ids, n_valid = worker.make_candidate_ids(1, 20, 20, "context", count, block, "cpu")
        self.assertEqual(n_valid.view(-1).tolist(), list(range(1, 21)))
        for j in range(20):
            row = ids[0, j].tolist()
            self.assertEqual(sorted(row[: j + 1]), list(range(j + 1)))
            self.assertEqual(row[j + 1 :], [-1] * (count - j - 1))
        self.assertEqual(support.CANDIDATE_COUNT, 16384)
        self.assertEqual(support.CANDIDATE_BLOCK, 8)


class TopkIdsTest(unittest.TestCase):
    """make_topk_ids validity on the CPU device."""

    def check_rows(self, ids, topk_length, num_requests, q_tokens, kv_len, topk, phase, g=1, flat_index=None):
        self.assertEqual(ids.shape, (num_requests * q_tokens, topk))
        self.assertEqual(ids.dtype, torch.int32)
        self.assertEqual(topk_length.shape, (num_requests * q_tokens,))
        self.assertEqual(topk_length.dtype, torch.int32)
        positions = kv_pool.row_positions(q_tokens, kv_len, phase).tolist()
        for row in range(num_requests * q_tokens):
            req, j = divmod(row, q_tokens)
            n = int(topk_length[row])
            values = ids[row].tolist()
            self.assertEqual(n % g, 0)
            self.assertEqual(n, min(((positions[j] + 1) // g) * g, topk), f"row {row}")
            self.assertEqual(values[n:], [-1] * (topk - n), f"row {row}: -1 suffix")
            valid = values[:n]
            self.assertTrue(all(v >= 0 for v in valid), f"row {row}: -1 inside the valid prefix")
            self.assertEqual(len(set(valid)), n, f"row {row}: duplicate ids")
            if flat_index is None:
                tokens = [v - req * kv_len for v in valid]
            else:
                tokens = [flat_index[req][v] for v in valid]  # inverse map: pool row -> request-local token
            self.assertTrue(all(0 <= t <= positions[j] for t in tokens), f"row {row}: token outside the causal range")
            if g > 1:
                blocks = sorted(t // g for t in tokens)
                self.assertEqual(len(set(blocks)), n // g)
                for b in set(blocks):
                    self.assertTrue(set(range(b * g, b * g + g)) <= set(tokens), f"row {row}: block {b} incomplete")

    def test_context_rows_are_causal_unique_and_padded(self):
        gen = torch.Generator().manual_seed(1)
        ids, length = kv_pool.make_topk_ids(1, 200, 200, 64, phase="context", device="cpu", generator=gen, chunk_rows=37)
        self.check_rows(ids, length, 1, 200, 200, 64, "context")
        self.assertEqual(int(length[0]), 1)
        self.assertEqual(int(length[62]), 63)
        self.assertEqual(int(length[63]), 64)
        self.assertEqual(int(length[199]), 64)

    def test_full_lists_draw_from_the_whole_request(self):
        gen = torch.Generator().manual_seed(6)
        ids, length = kv_pool.make_topk_ids(2, 100, 100, 64, phase="context", causal=False, device="cpu", generator=gen)
        self.assertEqual(length.tolist(), [64] * 200)
        self.assertTrue((ids >= 0).all())
        for row in range(200):
            req = row // 100
            self.assertTrue(((ids[row] >= req * 100) & (ids[row] < (req + 1) * 100)).all())
            self.assertEqual(ids[row].unique().numel(), 64)
        # Row 0 is no longer limited to token 0: the lists are the per-token upper bound.
        self.assertGreater(int(ids[0].max()), 0)
        # Complete blocks with the full pool: lengths are topk from the first row.
        ids, length = kv_pool.make_topk_ids(1, 8, 40, 16, phase="context", causal=False, index_granularity=4, device="cpu", generator=gen)
        self.assertEqual(length.tolist(), [16] * 8)
        for row in range(8):
            blocks = sorted(set(int(t) // 4 for t in ids[row]))
            self.assertEqual(len(blocks), 4)
            self.assertEqual(sorted(ids[row].tolist()), sorted(b * 4 + i for b in blocks for i in range(4)))

    def test_generation_rows_stay_inside_their_own_request(self):
        gen = torch.Generator().manual_seed(2)
        ids, length = kv_pool.make_topk_ids(3, 2, 512, 128, phase="generation", device="cpu", generator=gen)
        self.check_rows(ids, length, 3, 2, 512, 128, "generation")
        self.assertEqual(length.tolist(), [128] * 6)
        for row in range(6):
            req = row // 2
            self.assertTrue(((ids[row] >= req * 512) & (ids[row] < (req + 1) * 512)).all())

    def test_index_granularity_picks_complete_blocks(self):
        gen = torch.Generator().manual_seed(3)
        ids, length = kv_pool.make_topk_ids(1, 40, 40, 16, phase="context", index_granularity=4, device="cpu", generator=gen)
        self.check_rows(ids, length, 1, 40, 40, 16, "context", g=4)
        self.assertEqual(length[:8].tolist(), [0, 0, 0, 4, 4, 4, 4, 8])
        self.assertTrue((ids[:3] == -1).all())
        ids, length = kv_pool.make_topk_ids(2, 1, 64, 16, phase="generation", index_granularity=4, device="cpu", generator=gen)
        self.check_rows(ids, length, 2, 1, 64, 16, "generation", g=4)

    def test_paged_mapping_is_a_bijection_and_equals_the_flat_ids_through_the_table(self):
        num_requests, kv_len, page, topk = 2, 200, 64, 48
        pool_gen = torch.Generator().manual_seed(4)
        paged = kv_pool.make_paged_pool(num_requests, kv_len, page, 16, torch.float32, "cpu", generator=pool_gen)
        max_pages = -(-kv_len // page)
        self.assertEqual(paged.block_table.shape, (num_requests, max_pages))
        self.assertEqual(paged.block_table.dtype, torch.int32)
        self.assertEqual(paged.pool.shape, ((num_requests * max_pages + 4) * page, 16))
        slots = paged.block_table.view(-1).tolist()
        self.assertEqual(len(set(slots)), len(slots))
        self.assertTrue(all(0 <= s < num_requests * max_pages + 4 for s in slots))
        # Every (request, token) maps to a distinct pool row.
        req = torch.arange(num_requests).view(-1, 1).expand(num_requests, kv_len)
        tok = torch.arange(kv_len).view(1, -1).expand(num_requests, kv_len)
        rows = paged.flat_index(req, tok)
        self.assertEqual(rows.dtype, torch.int64)
        flat_rows = rows.view(-1).tolist()
        self.assertEqual(len(set(flat_rows)), num_requests * kv_len)
        self.assertTrue(all(0 <= r < paged.pool.shape[0] for r in flat_rows))
        inverse = [dict(zip(rows[r].tolist(), range(kv_len))) for r in range(num_requests)]
        # The same generator state draws the same lists, so the paged ids are the flat ids through the table.
        flat_ids, flat_len = kv_pool.make_topk_ids(num_requests, 3, kv_len, topk, phase="generation", device="cpu", generator=torch.Generator().manual_seed(5))
        paged_ids, paged_len = kv_pool.make_topk_ids(
            num_requests,
            3,
            kv_len,
            topk,
            phase="generation",
            block_table=paged.block_table,
            page_size=page,
            device="cpu",
            generator=torch.Generator().manual_seed(5),
        )
        self.assertTrue(torch.equal(flat_len, paged_len))
        self.check_rows(paged_ids, paged_len, num_requests, 3, kv_len, topk, "generation", flat_index=inverse)
        request = torch.arange(num_requests * 3).view(-1, 1) // 3
        expected = torch.where(flat_ids >= 0, paged.flat_index(request, (flat_ids.long() - request * kv_len).clamp(min=0)), -1).to(torch.int32)
        self.assertTrue(torch.equal(paged_ids, expected))
        contiguous = kv_pool.make_paged_pool(1, 128, 64, 8, torch.float32, "cpu", slot_order="contiguous")
        self.assertEqual(contiguous.flat_index(0, torch.arange(128)).tolist(), list(range(128)))

    def test_argument_errors(self):
        with self.assertRaises(ValueError):
            kv_pool.make_topk_ids(1, 8, 8, 6, phase="context", index_granularity=4, device="cpu")
        with self.assertRaises(ValueError):
            kv_pool.make_topk_ids(1, 8, 8, 4, phase="context", page_size=64, device="cpu")
        with self.assertRaises(ValueError):
            kv_pool.make_topk_ids(1, 16, 8, 4, phase="context", device="cpu")
        with self.assertRaises(ValueError):
            kv_pool.make_paged_pool(1, 8, 4, 8, torch.float32, "cpu", slot_order="sorted")
        with self.assertRaises(ValueError):
            kv_pool.make_paged_pool(0, 8, 4, 8, torch.float32, "cpu")


class RunSingleTest(unittest.TestCase):
    """run_single with a patched subprocess: RESULT parsing, UNSUPPORTED rows and the static-skip path."""

    RESULT = "RESULT,1.500000,12.345,900.000,NVIDIA Test GPU,1.31.0,indexer_forward_top_k_wrapper precision=bf16 top_k=512 ratio=1 heads=32;fused;rows=16;bytes_gathered=4194304;time_step1_ms=1.500000\n"

    def setUp(self):
        self.runner = _runner(cc=(10, 0), dsl_target_error=None)

    def case(self, **filters):
        cases = _cases("deepseek_v41", **filters)
        self.assertTrue(cases)
        return cases[0]

    def test_rc0_result_line_lands_in_every_column(self):
        case = self.case(filter_component="indexer", filter_phase="generation", filter_kv_dtype="bfloat16", filter_paged=False)
        self.assertEqual(case["indexer_mode"], "dense")
        with patch("benchmark.dsa.runner.subprocess.run", return_value=_completed(0, stdout="noise\n" + self.RESULT)) as run:
            result = self.runner.run_single(case)
        self.assertEqual(run.call_count, 1)
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--component") + 1], "indexer")
        self.assertEqual(cmd[cmd.index("--phase") + 1], "generation")
        self.assertEqual(cmd[cmd.index("--batch_size") + 1], "8")
        self.assertEqual(cmd[cmd.index("--indexer_mode") + 1], "dense")
        self.assertEqual(cmd[cmd.index("--indexer_heads") + 1], "32")
        self.assertEqual(cmd[cmd.index("--mtp_rows") + 1], "2")
        self.assertEqual(cmd[cmd.index("--page_size") + 1], "64")
        self.assertNotIn("--paged", cmd)
        self.assertTrue(result.success)
        self.assertFalse(result.skipped)
        self.assertEqual(result.time_ms, 1.5)
        self.assertEqual(result.tflops, 12.345)
        self.assertEqual(result.peak_mma_tflops, 900.0)
        self.assertEqual(result.gpu_name, "NVIDIA Test GPU")
        self.assertEqual(result.cudnn_version, "1.31.0")
        self.assertEqual(result.backend_detail, self.RESULT.strip().split(",", 6)[6])
        self.assertEqual(result.time_step1_ms, 1.5)
        self.assertIsNone(result.time_step2_ms)
        self.assertAlmostEqual(result.gather_gbps, 4194304 / (1.5 * 1e6))
        self.assertEqual(
            (result.component, result.phase, result.kv_dtype, result.paged, result.batch_size, result.indexer_mode),
            ("indexer", "generation", "bfloat16", False, 8, "dense"),
        )
        self.assertEqual(result.bytes_per_token, 256)
        self.assertEqual((result.q_seqlen, result.kv_seqlen), (1, 8192))

    def test_two_step_tokens_and_no_gather_bytes(self):
        case = self.case(filter_component="indexer", filter_phase="context", filter_kv_dtype="bfloat16", filter_paged=False)
        line = "RESULT,2.000000,3.000,,GPU,1.31.0,SparseIndexerScoreRecompute+IndexerTopK candidates=16384 block=8 top_k=512 heads=32;consumer;time_step1_ms=1.250000;time_step2_ms=0.750000\n"
        with patch("benchmark.dsa.runner.subprocess.run", return_value=_completed(0, stdout=line)):
            result = self.runner.run_single(case)
        self.assertTrue(result.success)
        self.assertIsNone(result.peak_mma_tflops)
        self.assertEqual((result.time_step1_ms, result.time_step2_ms), (1.25, 0.75))
        self.assertIsNone(result.gather_gbps)

    def test_legacy_result_line_without_tokens(self):
        case = _cases("deepseek_v4")[0]
        line = "RESULT,0.500000,700.000,2000.000,GPU,1.31.0,SparseAttentionForward.execute padded_K=640 indexer_topk=512\n"
        with patch("benchmark.dsa.runner.subprocess.run", return_value=_completed(0, stdout=line)):
            result = self.runner.run_single(case)
        self.assertTrue(result.success)
        self.assertEqual(result.backend_detail, "SparseAttentionForward.execute padded_K=640 indexer_topk=512")
        self.assertIsNone(result.time_step1_ms)
        self.assertIsNone(result.gather_gbps)
        self.assertEqual(
            (result.component, result.phase, result.paged, result.batch_size, result.indexer_mode, result.bytes_per_token),
            ("core", "context", False, 1, None, 1024),
        )

    def test_rc3_is_a_skipped_row_with_the_workers_reason(self):
        case = self.case(filter_component="core", filter_phase="context", filter_kv_dtype="bfloat16", filter_paged=True)
        with patch(
            "benchmark.dsa.runner.subprocess.run", return_value=_completed(UNSUPPORTED_EXIT_CODE, stdout="UNSUPPORTED,kernel declined this shape; see detail\n")
        ) as run:
            result = self.runner.run_single(case)
        self.assertEqual(run.call_count, 1)
        self.assertIn("--paged", run.call_args.args[0])
        self.assertTrue(result.skipped)
        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "kernel declined this shape; see detail")
        self.assertEqual(result.time_ms, float("inf"))
        self.assertEqual(result.tflops, 0.0)
        self.assertTrue(result.paged)
        self.assertEqual(result.bytes_per_token, 1024)

    def test_static_skip_spawns_nothing(self):
        for case in (
            self.case(filter_kv_dtype="nvfp4"),
            self.case(filter_component="indexer", filter_paged=True),
            self.case(filter_component="core", filter_kv_dtype="mxfp8"),
        ):
            with patch("benchmark.dsa.runner.subprocess.run") as run:
                result = self.runner.run_single(case)
            run.assert_not_called()
            self.assertTrue(result.skipped)
            self.assertFalse(result.success)
            self.assertEqual(result.error_message, support.decide_support(case, (10, 0), None))
            self.assertEqual(result.time_ms, float("inf"))
            self.assertEqual(result.bytes_per_token, support.bytes_per_token(case))
            self.assertEqual(result.kv_dtype, case["kv_dtype"])
        with patch("benchmark.dsa.runner.subprocess.run") as run:
            for case in _cases("qwen38_qsa"):
                self.assertTrue(self.runner.run_single(case).skipped)
        run.assert_not_called()
        # On a 9.x part the core forward rows skip statically too.
        with patch("benchmark.dsa.runner.subprocess.run") as run:
            result = _runner(cc=(9, 0)).run_single(_cases("deepseek_v4")[0])
        run.assert_not_called()
        self.assertEqual(result.error_message, "sparse core forward requires compute capability 10.x")

    def test_failures_are_not_skips(self):
        case = _cases("deepseek_v4")[0]
        with patch("benchmark.dsa.runner.subprocess.run", return_value=_completed(1, stdout="", stderr="Traceback\nRuntimeError: launch failed\n")):
            result = self.runner.run_single(case)
        self.assertFalse(result.success)
        self.assertFalse(result.skipped)
        self.assertIn("RuntimeError: launch failed", result.error_message)
        with patch("benchmark.dsa.runner.subprocess.run", return_value=_completed(0, stdout="no result line\n")):
            result = self.runner.run_single(case)
        self.assertFalse(result.success)
        self.assertFalse(result.skipped)
        with patch("benchmark.dsa.runner.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1800)):
            result = self.runner.run_single(case)
        self.assertEqual(result.error_message, "timeout")
        self.assertFalse(result.skipped)

    def test_parse_detail_tokens(self):
        tokens = parse_detail_tokens("a;time_step1_ms=1.5;xtime_step2_ms=9;time_step2_ms=2.5e-1;bytes_gathered=1024;bytes_gathered=2048")
        self.assertEqual(tokens, {"time_step1_ms": 1.5, "time_step2_ms": 0.25, "bytes_gathered": 1024.0})
        self.assertEqual(parse_detail_tokens(None), {})
        self.assertEqual(parse_detail_tokens("SparseAttentionForward.execute padded_K=640 indexer_topk=512"), {})

    def test_zero_time_yields_no_gather_rate(self):
        case = self.case(filter_component="core", filter_phase="generation", filter_kv_dtype="bfloat16", filter_paged=False)
        line = "RESULT,0.000000,0.000,,GPU,1.31.0,x;bytes_gathered=4096\n"
        with patch("benchmark.dsa.runner.subprocess.run", return_value=_completed(0, stdout=line)):
            result = self.runner.run_single(case)
        self.assertIsNone(result.gather_gbps)


class SchemaTest(unittest.TestCase):
    def test_result_columns_are_the_legacy_25_then_the_appended_10(self):
        self.assertEqual([f.name for f in dataclasses.fields(BenchmarkResult)], LEGACY_COLUMNS + APPENDED_COLUMNS)
        row = BenchmarkResult(*(["x"] * 4), 1, 1, 64, 512, 512, 640, 512, True, True, "fwd", False, 1.0, 1.0, 20)
        self.assertEqual((row.component, row.phase, row.kv_dtype, row.paged, row.batch_size, row.indexer_mode), ("core", "context", "bfloat16", False, 1, None))
        self.assertEqual((row.bytes_per_token, row.gather_gbps, row.time_step1_ms, row.time_step2_ms), (None, None, None, None))

    def test_preset_and_config_fields_are_appended_and_defaulted(self):
        preset_fields = [f.name for f in dataclasses.fields(ModelPreset)]
        self.assertEqual(
            preset_fields[preset_fields.index("has_sink") + 1 :],
            ["num_kv_heads", "kv_layout", "index_granularity", "indexer_heads", "indexer_pool_ratio", "indexer_modes", "mtp_rows"],
        )
        preset = ModelPreset("p", 64)
        self.assertEqual(
            (
                preset.num_kv_heads,
                preset.kv_layout,
                preset.index_granularity,
                preset.indexer_heads,
                preset.indexer_pool_ratio,
                preset.indexer_modes,
                preset.mtp_rows,
            ),
            (1, "shared", 1, 0, 1, ("dense",), 1),
        )
        config_fields = [f.name for f in dataclasses.fields(DsaBenchmarkConfig)]
        self.assertEqual(
            config_fields[config_fields.index("output_dir") + 1 :],
            ["components", "phases", "kv_dtypes", "paged", "page_size", "generation_batch_sizes", "generation_kv_lens"],
        )
        config = DsaBenchmarkConfig("c", models=[preset], seqlens=[(1024, 1024)])
        self.assertEqual(
            (config.components, config.phases, config.kv_dtypes, config.paged, config.page_size, config.generation_batch_sizes, config.generation_kv_lens),
            (["core"], None, ["bfloat16"], [False], 64, [8, 128], [8192, 131072]),
        )

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas not installed")
    def test_csv_header_order(self):
        runner = _runner()
        case = _cases("deepseek_v4")[0]
        with patch("benchmark.dsa.runner.subprocess.run", return_value=_completed(0, stdout="RESULT,0.5,700,2000,GPU,1.31.0,detail\n")):
            results = [runner.run_single(case), runner.run_single(_cases("deepseek_v41", filter_kv_dtype="nvfp4")[0])]
        with tempfile.TemporaryDirectory() as tmp:
            path = runner.save_csv(results, load_config("deepseek_v4"), Path(tmp) / "out.csv")
            header = path.read_text().splitlines()[0].split(",")
        self.assertEqual(header, LEGACY_COLUMNS + APPENDED_COLUMNS)


class PresetContractTest(unittest.TestCase):
    def test_shipped_presets_fit_the_kernels(self):
        for name in list_configs():
            for model in load_config(name).models:
                with self.subTest(config=name, model=model.name):
                    self.assertEqual(model.topk % 64, 0)
                    self.assertEqual(model.topk % model.index_granularity, 0)
                    self.assertIn(model.kv_layout, ("shared", "separate"))
                    pair = (model.num_q_heads, model.head_dim_qk)
                    if model.kv_layout == "shared":
                        self.assertIn(pair, support.CORE_VARIANTS)
                        self.assertIn(model.indexer_topk, support.CORE_VARIANTS[pair])
                    else:
                        self.assertNotIn(pair, support.CORE_VARIANTS)  # the QSA follow-up kernel, not a flat-row variant
                    if model.indexer_heads:
                        self.assertEqual(support.indexer_topk_select(model) % 128, 0)  # IndexerBackward takes 128-multiples
                        self.assertIn(model.indexer_pool_ratio, (1, 4))

    def test_indexer_topk_select(self):
        selected = {m.name: support.indexer_topk_select(m) for name in list_configs() for m in load_config(name).models}
        self.assertEqual(selected, {"dsv4_flash": 512, "dsv4_pro": 1024, "dsv41_flash": 512, "glm53": 2048, "glm53_flash": 512, "qsa_qwen38": 512})
        self.assertEqual(support.indexer_topk_select(dict(topk=2048, indexer_topk=0, index_granularity=4)), 512)
        self.assertEqual(support.indexer_topk_select(dict(topk=640, indexer_topk=512)), 512)

    def test_core_variants_table(self):
        self.assertEqual(support.CORE_VARIANTS, {(64, 512): (0, 512, 1024, 2048), (64, 576): (0, 512, 1024, 2048), (128, 512): (0, 512, 1024)})
        self.assertEqual(support.SHARED_COMPILER_ROWS, {("core", "bprop"), ("indexer", "context"), ("indexer", "generation"), ("indexer", "bprop")})
        self.assertEqual(support.PROFILE_PASS_FOR_PHASE, {"context": "fwd", "generation": "fwd", "bprop": "bwd"})


class WorkerArgvTest(unittest.TestCase):
    """The worker's argparse surface and its agreement with the runner (no CUDA is touched)."""

    LEGACY = ["--profile_pass", "fwd", "--q_seqlen", "2048", "--kv_seqlen", "2048", "--num_q_heads", "64", "--topk", "640", "--indexer_topk", "512"]

    def test_defaults_reproduce_the_legacy_rows(self):
        args = worker.parse_args(self.LEGACY)
        self.assertEqual(
            (args.component, args.phase, args.kv_dtype, args.paged, args.page_size, args.batch_size), ("core", "context", "bfloat16", False, 64, 1)
        )
        self.assertEqual(
            (args.num_kv_heads, args.kv_layout, args.index_granularity, args.indexer_heads, args.indexer_pool_ratio, args.indexer_mode, args.mtp_rows),
            (1, "shared", 1, 0, 1, "dense", 1),
        )
        self.assertFalse(args.profile)
        self.assertEqual(worker.parse_args(["--profile_pass", "bwd"] + self.LEGACY[2:]).phase, "bprop")
        case = worker.case_from_args(args)
        self.assertIsNone(case["indexer_mode"])
        self.assertIsNone(support.decide_support(case, (10, 0), None))
        self.assertEqual(support.static_columns(case)["bytes_per_token"], 1024)

    def test_phase_must_match_profile_pass(self):
        with contextlib.redirect_stderr(io.StringIO()):  # argparse prints its usage text before exiting
            with self.assertRaises(SystemExit) as ctx:
                worker.parse_args(self.LEGACY + ["--phase", "bprop"])
            self.assertEqual(ctx.exception.code, 2)
            with self.assertRaises(SystemExit):
                worker.parse_args(["--profile_pass", "bwd"] + self.LEGACY[2:] + ["--phase", "generation"])
            with self.assertRaises(SystemExit):
                worker.parse_args(self.LEGACY + ["--batch_size", "0"])
        self.assertEqual(worker.parse_args(self.LEGACY + ["--phase", "generation", "--batch_size", "8"]).phase, "generation")

    def test_runner_argv_round_trips_to_the_same_support_decision(self):
        for name in ("deepseek_v4", "deepseek_v41", "glm53", "qwen38_qsa"):
            for case in _cases(name):
                args = worker.parse_args(DsaBenchmarkRunner.worker_command(case)[2:])
                flat = worker.case_from_args(args)
                self.assertEqual(support.static_columns(flat), support.static_columns(case), case)
                for cc in CCS + (None,):
                    for dsl in DSL_PROBES:
                        self.assertEqual(support.decide_support(flat, cc, dsl), support.decide_support(case, cc, dsl), case)

    def test_setups_cover_every_component_phase_cell(self):
        self.assertEqual(set(worker.SETUPS), {(c, p) for c in ("core", "indexer") for p in support.PHASES})
        self.assertEqual(worker.UNSUPPORTED_EXIT_CODE, UNSUPPORTED_EXIT_CODE)
        self.assertEqual(worker._FLOPS_PER_CLOCK_PER_SM[10]["mxfp8"], 16384)
        self.assertNotIn("nvfp4", worker._FLOPS_PER_CLOCK_PER_SM[10])


def _plotting_available() -> bool:
    return all(importlib.util.find_spec(m) for m in ("pandas", "matplotlib", "seaborn"))


@unittest.skipUnless(_plotting_available(), "pandas, matplotlib and seaborn are required")
class ChartTest(unittest.TestCase):
    def setUp(self):
        import matplotlib

        matplotlib.use("Agg")
        self.config = load_config("deepseek_v41")

    def rows(self, cases, success_filter):
        from dataclasses import asdict

        rows = []
        for i, case in enumerate(cases):
            base = dict(
                config_name=self.config.name,
                model_name=case["model"].name,
                backend=case["backend"],
                data_type=case["data_type"],
                q_seqlen=case["q_seqlen"],
                kv_seqlen=case["kv_seqlen"],
                num_q_heads=case["model"].num_q_heads,
                head_dim_qk=case["model"].head_dim_qk,
                head_dim_vo=case["model"].head_dim_vo,
                topk=case["model"].topk,
                indexer_topk=case["model"].indexer_topk,
                has_sink=case["model"].has_sink,
                use_topk_length=True,
                profile_pass=case["profile_pass"],
                deterministic_bwd=case["deterministic_bwd"],
                num_iterations=20,
            )
            base.update(support.static_columns(case))
            if success_filter(case):
                row = BenchmarkResult(
                    **base,
                    time_ms=1.0 + i % 7,
                    tflops=100.0 + i,
                    peak_mma_tflops=2000.0,
                    backend_detail="d",
                    time_step1_ms=0.5 if case["component"] == "indexer" else None,
                )
            else:
                row = BenchmarkResult(**base, time_ms=float("inf"), tflops=0.0, success=False, skipped=True, error_message="coverage row")
            rows.append(asdict(row))
        return rows

    def test_legacy_and_extended_figures(self):
        import pandas as pd

        from benchmark.dsa.charts import generate_charts

        cases = _cases("deepseek_v41")
        df = pd.DataFrame(self.rows(cases, lambda c: support.decide_support(c, (10, 0), None) is None))
        with tempfile.TemporaryDirectory() as tmp:
            paths = generate_charts(df, self.config, Path(tmp))
            names = sorted(p.name for p in paths)
            self.assertEqual(names, ["deepseek_v41.webp", "deepseek_v41_extended.webp"])
            self.assertTrue(all(p.stat().st_size > 0 for p in paths))

    def test_all_skipped_frame_draws_the_coverage_grid_only(self):
        import pandas as pd

        from benchmark.dsa.charts import generate_charts

        df = pd.DataFrame(self.rows(_cases("deepseek_v41"), lambda c: False))
        with tempfile.TemporaryDirectory() as tmp:
            paths = generate_charts(df, self.config, Path(tmp))
            self.assertEqual([p.name for p in paths], ["deepseek_v41_extended.webp"])
            self.assertEqual(generate_charts(pd.DataFrame(), self.config, Path(tmp)), [])

    def test_frame_re_read_from_a_saved_csv(self):
        # pandas hands back `paged` / `success` as strings or bools and `indexer_mode` as NaN on core rows;
        # the chart code normalizes them (charts._with_axis_defaults) before drawing.
        import pandas as pd

        from benchmark.dsa.charts import generate_charts

        cases = _cases("deepseek_v41")
        rows = self.rows(cases, lambda c: support.decide_support(c, (10, 0), None) is None)
        results = [BenchmarkResult(**row) for row in rows]
        with tempfile.TemporaryDirectory() as tmp:
            path = _runner().save_csv(results, self.config, Path(tmp) / "saved.csv")
            df = pd.read_csv(path)
            self.assertEqual(list(df.columns), LEGACY_COLUMNS + APPENDED_COLUMNS)
            self.assertTrue(df["indexer_mode"].isna().any())
            paths = generate_charts(df, self.config, Path(tmp))
            self.assertEqual(sorted(p.name for p in paths), ["deepseek_v41.webp", "deepseek_v41_extended.webp"])

    def test_legacy_frame_without_the_appended_columns(self):
        import pandas as pd

        from benchmark.dsa.charts import generate_charts

        config = load_config("deepseek_v4")
        rows = self.rows(_cases("deepseek_v4"), lambda c: True)
        for row in rows:
            row["config_name"] = config.name
            for column in APPENDED_COLUMNS:
                row.pop(column)
        df = pd.DataFrame(rows)
        self.assertEqual(list(df.columns), LEGACY_COLUMNS)
        with tempfile.TemporaryDirectory() as tmp:
            paths = generate_charts(df, config, Path(tmp))
            self.assertEqual([p.name for p in paths], ["deepseek_v4.webp"])


if __name__ == "__main__":
    unittest.main()
