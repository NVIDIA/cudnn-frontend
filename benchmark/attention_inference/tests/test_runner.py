# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only case and result contracts for the inference benchmark runner."""

from dataclasses import replace
import subprocess
import unittest
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
        self.assertEqual(len(list(self.runner.expand_config(QWEN35))), 184)
        for backend, count, pages in (("flashinfer", 192, {16, 32, 64}), ("b12x", 64, {64})):
            with self.subTest(backend=backend):
                config = replace(QWEN35, backends=[backend], context_seqlens=[], context_chunked_shapes=[])
                cases = list(self.runner.expand_config(config))
                self.assertEqual(len(cases), count)
                self.assertEqual({case["page_size"] for case in cases}, pages)


if __name__ == "__main__":
    unittest.main()
