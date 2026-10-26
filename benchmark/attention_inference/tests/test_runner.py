# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only case and result contracts for the inference benchmark runner."""

from dataclasses import replace
import subprocess
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from benchmark.attention_inference.config_types import InferenceBenchmarkConfig, ModelPreset
from benchmark.attention_inference.configs.qwen35 import CONFIG as QWEN35
from benchmark.attention_inference.runner import InferenceBenchmarkRunner

pytestmark = pytest.mark.L0


class PageSizeTest(unittest.TestCase):
    def setUp(self):
        self.runner = InferenceBenchmarkRunner()
        self.model = ModelPreset("gqa", num_q_heads=8, num_kv_heads=2)
        self.config = InferenceBenchmarkConfig(
            "pages",
            models=[self.model],
            generation_shapes=[(1, 256)],
            generation_batch_sizes=[1],
            backends=["b12x"],
            page_size=16,
            page_sizes=[16, 32, 64],
        )

    def pages(self, **changes):
        return [case["page_size"] for case in self.runner.expand_config(replace(self.config, **changes))]

    def test_b12x_paged_cases_use_unique_effective_sizes_in_order(self):
        self.assertEqual(self.pages(), [64])
        self.assertEqual(self.pages(page_sizes=[128, 32, 64, 16, 128]), [128, 64])
        self.assertEqual(self.pages(page_sizes=None), [64])
        self.assertEqual(self.pages(page_sizes=[]), [64])

    def test_b12x_absorbed_mla_routes_keep_their_distinct_page_contracts(self):
        mla = replace(self.model, kind="mla_absorbed", head_dim_qk=576, head_dim_vo=512)
        self.assertEqual(self.pages(models=[mla]), [16, 32, 64])
        self.assertEqual(self.pages(models=[mla], page_sizes=None), [16])
        shared_kv = replace(mla, head_dim_qk=512)
        self.assertEqual(self.pages(models=[shared_kv]), [64])

    def test_b12x_dense_fallback_and_context_do_not_sweep_pages(self):
        sliding = replace(self.model, sliding_window_size=128)
        self.assertEqual(self.pages(models=[sliding]), [16])
        self.assertEqual(self.pages(context_seqlens=[128], generation_shapes=[]), [16])

    def test_other_backends_keep_their_page_axes(self):
        for backend in ("flashinfer", "flash_mla"):
            with self.subTest(backend=backend):
                self.assertEqual(self.pages(backends=[backend], page_sizes=[32, 16, 32, 64]), [32, 16, 64])
        for backend in ("cudnn", "cudnn_oss", "flash_attention_4"):
            with self.subTest(backend=backend):
                self.assertEqual(self.pages(backends=[backend]), [16])

    def test_effective_size_reaches_subprocess_and_result_metadata(self):
        cases = list(self.runner.expand_config(self.config))
        self.assertEqual(len(cases), 1)
        with patch(
            "benchmark.attention_inference.runner.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, stdout="RESULT,1,2,3,,test-device,b12x paged (page=64)\n"),
        ) as run:
            result = self.runner.run_single(cases[0], cc=(12, 0))
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--page_size") + 1], "64")
        self.assertTrue(result.success)
        self.assertEqual(result.page_size, 64)

    def test_qwen35_counts_distinct_executed_page_sizes(self):
        self.assertEqual(len(list(self.runner.expand_config(QWEN35))), 276)  # 92 cases x cudnn / cudnn_oss / cudnn_auto
        for backend, count, pages in (("flashinfer", 192, {16, 32, 64}), ("b12x", 64, {64})):
            with self.subTest(backend=backend):
                config = replace(QWEN35, backends=[backend], context_seqlens=[], context_chunked_shapes=[])
                cases = list(self.runner.expand_config(config))
                self.assertEqual(len(cases), count)
                self.assertEqual({case["page_size"] for case in cases}, pages)

    def test_kv_layout_and_fill_axes(self):
        """cudnn legs sweep kv_layouts; mixed cache lengths ride only the paged cudnn legs and flashinfer, batch > 1."""
        base = replace(QWEN35, context_seqlens=[], context_chunked_shapes=[], page_sizes=None, kv_layouts=["dense", "paged"], kv_min_fills=[1.0, 0.25])
        for backend, layouts, fills in (
            ("cudnn_auto", {"dense", "paged"}, {("paged", 0.25)}),
            ("flashinfer", {"paged"}, {("paged", 0.25)}),
            ("flash_attention_4", {"dense"}, set()),
        ):
            with self.subTest(backend=backend):
                cases = list(self.runner.expand_config(replace(base, backends=[backend])))
                self.assertEqual({c["kv_layout"] for c in cases}, layouts)
                self.assertEqual({(c["kv_layout"], c["kv_min_fill"]) for c in cases if c["kv_min_fill"] < 1}, fills)
                self.assertTrue(all(c["batch_size"] > 1 for c in cases if c["kv_min_fill"] < 1))
        case = next(c for c in self.runner.expand_config(replace(base, backends=["cudnn"])) if c["kv_min_fill"] < 1)
        with patch(
            "benchmark.attention_inference.runner.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, stdout="RESULT,1,2,3,,test-device,cudnn plan=x\n"),
        ) as run:
            result = self.runner.run_single(case, cc=(10, 0))
        cmd = run.call_args.args[0]
        self.assertEqual((cmd[cmd.index("--kv_layout") + 1], cmd[cmd.index("--kv_min_fill") + 1]), ("paged", "0.25"))
        self.assertEqual((result.kv_layout, result.kv_min_fill), ("paged", 0.25))

    def test_b12x_sliding_window_generation_is_dense(self):
        swa = replace(QWEN35.models[0], sliding_window_size=128)
        cases = list(self.runner.expand_config(replace(QWEN35, models=[swa], backends=["b12x"], context_seqlens=[], context_chunked_shapes=[])))
        self.assertEqual({c["kv_layout"] for c in cases}, {"dense"})


@unittest.skipUnless(importlib.util.find_spec("torch"), "request_lens builds torch tensors")
class RequestLensTest(unittest.TestCase):
    def test_request_lens_is_deterministic_and_bounded(self):
        from benchmark.attention_inference.benchmark_single_attention import request_lens

        args = SimpleNamespace(batch_size=64, q_tokens=4, kv_len=8192, phase="generation", kv_min_fill=0.25)
        q, kv = request_lens(args)
        q2, kv2 = request_lens(args)
        self.assertTrue(bool((kv == kv2).all()) and bool((q == q2).all()))  # setup and the FLOP count see the same batch
        self.assertTrue(bool(((kv >= 2048) & (kv <= 8192)).all()) and bool((q == 4).all()))
        self.assertGreater(len(set(kv.tolist())), 1)
        _, kv_full = request_lens(SimpleNamespace(**{**vars(args), "kv_min_fill": 1.0}))
        self.assertTrue(bool((kv_full == 8192).all()))
        q_ctx, kv_ctx = request_lens(SimpleNamespace(batch_size=2, q_tokens=4096, kv_len=4096, phase="context", kv_min_fill=0.25))
        self.assertTrue(bool((q_ctx == 4096).all()) and bool((kv_ctx == 4096).all()))  # fill is a generation axis


if __name__ == "__main__":
    unittest.main()
