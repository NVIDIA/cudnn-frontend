# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only case and result contracts for the inference benchmark runner."""

from dataclasses import replace
import subprocess
import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from benchmark.attention_inference.config_types import InferenceBenchmarkConfig, ModelPreset, attn_mask
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
        # per cudnn leg: 92 dense + 92 paged + 32 mixed-length paged decodes (batch > 1)
        self.assertEqual(len(list(self.runner.expand_config(QWEN35))), 648)
        for backend, count, pages in (("flashinfer", 288, {16, 32, 64}), ("b12x", 64, {64})):
            with self.subTest(backend=backend):
                config = replace(QWEN35, backends=[backend], context_seqlens=[], context_chunked_shapes=[])
                cases = list(self.runner.expand_config(config))
                self.assertEqual(len(cases), count)
                self.assertEqual({case["page_size"] for case in cases}, pages)

    def test_kv_layout_and_fill_axes(self):
        """cudnn legs sweep kv_layouts; mixed cache lengths ride only the paged cudnn legs and flashinfer, batch > 1."""
        base = replace(QWEN35, context_seqlens=[], context_chunked_shapes=[], page_sizes=None)
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
        from benchmark.attention_inference.runner import kv_layouts_for

        both = SimpleNamespace(kv_layouts=["dense", "paged"])
        for leg in ("flashinfer_fa2", "flashinfer_cudnn"):
            # pinned FlashInfer generation always reads a paged cache; only its prefill has a ragged (dense) form
            self.assertEqual(kv_layouts_for(both, leg, "generation"), ["paged"])
            self.assertEqual(kv_layouts_for(both, leg, "context"), ["dense", "paged"])

    def test_serving_prefill_cases(self):
        """context_cases expand per layout with their fills and pages; the cudnn paged prefill legs record the total-Q
        declaration and pass it, with the Q fill, to the subprocess; pinned FlashInfer legs sweep kv_layouts."""
        from benchmark.attention_inference.configs.serving_prefill import CONFIG as SP

        cases = list(self.runner.expand_config(SP))
        self.assertEqual(len(cases), len(SP.models) * len(SP.backends) * len(SP.context_cases) * len(SP.kv_layouts))
        self.assertTrue(all(c["phase"] == "context" and c["kv_layout"] == "paged" for c in cases))
        self.assertIn(64, {c["page_size"] for c in cases})
        case = next(c for c in cases if c["backend"] == "cudnn_auto" and c["q_min_fill"] < 1)
        with patch(
            "benchmark.attention_inference.runner.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, stdout="RESULT,1,2,3,,test-device,cudnn plan=x\n"),
        ) as run:
            result = self.runner.run_single(case, cc=(10, 0))
        cmd = run.call_args.args[0]
        self.assertEqual((cmd[cmd.index("--q_min_fill") + 1], cmd[cmd.index("--cudnn_total_q") + 1]), ("0.5", "none"))
        self.assertEqual((result.q_min_fill, result.cudnn_total_q), (0.5, "none"))
        fi = next(c for c in cases if c["backend"] == "flashinfer_cudnn")
        with patch(
            "benchmark.attention_inference.runner.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, stdout="RESULT,1,2,3,,test-device,flashinfer\n"),
        ):
            self.assertIsNone(self.runner.run_single(fi, cc=(10, 0)).cudnn_total_q)

    def test_b12x_sliding_window_generation_is_dense(self):
        swa = replace(QWEN35.models[0], sliding_window_size=128)
        cases = list(self.runner.expand_config(replace(QWEN35, models=[swa], backends=["b12x"], context_seqlens=[], context_chunked_shapes=[])))
        self.assertEqual({c["kv_layout"] for c in cases}, {"dense"})


class CheckpointTest(unittest.TestCase):
    def test_interrupted_sweep_resumes_only_unfinished_cases_in_order(self):
        runner = InferenceBenchmarkRunner()
        config = InferenceBenchmarkConfig(
            "resume",
            models=[ModelPreset("gqa", num_q_heads=8, num_kv_heads=2)],
            context_seqlens=[128, 128, 256, 512],
        )
        ok = subprocess.CompletedProcess([], 0, stdout="RESULT,1,2,3,,test-device,cudnn plan=x\n")
        failed = subprocess.CompletedProcess([], 1, stdout="", stderr="failed to initialize")
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sweep.sqlite"
            with (
                patch("benchmark.attention_inference.runner._benchmark_environment", return_value={"device": "first", "capability": [8, 9]}),
                patch("benchmark.attention_inference.runner.subprocess.run", side_effect=[ok, failed, KeyboardInterrupt]) as run,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    runner.run_config(config, checkpoint=checkpoint)
                self.assertEqual([call.args[0][call.args[0].index("--q_tokens") + 1] for call in run.call_args_list], ["128", "256", "512"])
                run.side_effect = [ok, failed]
                run.reset_mock()
                results = runner.run_config(config, checkpoint=checkpoint, resume=True)
                self.assertEqual([call.args[0][call.args[0].index("--q_tokens") + 1] for call in run.call_args_list], ["256", "512"])
                self.assertEqual([(r.q_tokens, r.success) for r in results], [(128, True), (256, True), (512, False)])
                self.assertEqual(results[-1].error_message, "failed to initialize")

                run.side_effect = [ok, ok, ok]
                run.reset_mock()
                runner.run_config(replace(config, num_warmup_iterations=7), checkpoint=checkpoint, resume=True)
                self.assertEqual(run.call_count, 3)

            with (
                patch("benchmark.attention_inference.runner._benchmark_environment", return_value={"device": "changed", "capability": [8, 9]}),
                patch("benchmark.attention_inference.runner.subprocess.run") as run,
            ):
                with self.assertRaisesRegex(ValueError, "environment differs"):
                    runner.run_config(config, checkpoint=checkpoint, resume=True)
                run.assert_not_called()

    def test_checkpoint_requires_explicit_resume_and_preserves_existing_files(self):
        runner = InferenceBenchmarkRunner()
        config = InferenceBenchmarkConfig("resume", [ModelPreset("gqa", 8, 2)], context_seqlens=[128])
        with TemporaryDirectory() as directory, patch("benchmark.attention_inference.runner._benchmark_environment", return_value={"capability": [8, 9]}):
            checkpoint = Path(directory) / "sweep.sqlite"
            with self.assertRaisesRegex(ValueError, "Checkpoint does not exist"):
                runner.run_config(config, checkpoint=checkpoint, resume=True)
            self.assertFalse(checkpoint.exists())

            checkpoint.write_text("existing data")
            with self.assertRaisesRegex(ValueError, "already exists"):
                runner.run_config(config, checkpoint=checkpoint)
            self.assertEqual(checkpoint.read_text(), "existing data")


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
        q_ctx, kv_ctx = request_lens(SimpleNamespace(batch_size=8, q_tokens=4096, kv_len=4096, phase="context", kv_min_fill=0.25))
        # a serving prefill of mixed length: each full prefill is a square of its own length
        self.assertTrue(bool((q_ctx == kv_ctx).all()) and bool(((kv_ctx >= 1024) & (kv_ctx <= 4096)).all()))
        q_p, kv_p = request_lens(SimpleNamespace(batch_size=16, q_tokens=128, kv_len=8192, phase="context", kv_min_fill=0.5, q_min_fill=0.5))
        self.assertTrue(bool(((q_p >= 64) & (q_p <= 128)).all()) and bool((kv_p >= q_p).all()) and bool((kv_p <= 8192).all()))
        self.assertGreater(len(set(q_p.tolist())), 1)
        q_eq, kv_eq = request_lens(SimpleNamespace(batch_size=2, q_tokens=4096, kv_len=4096, phase="context", kv_min_fill=1.0))
        self.assertTrue(bool((q_eq == 4096).all()) and bool((kv_eq == 4096).all()))

    def test_mixed_lengths_declare_the_row_maximum(self):
        """The direct cuDNN graph declares the row's KV maximum, as FlashInfer's cudnn path does, not the drawn lengths' maximum."""
        from benchmark.attention_inference.benchmark_single_attention import declared_max_kv, request_lens

        args = SimpleNamespace(batch_size=4, q_tokens=2048, kv_len=2048, phase="context", kv_min_fill=0.25)
        _, kv = request_lens(args)
        self.assertLess(int(kv.max()), 2048)
        self.assertEqual(declared_max_kv(args, kv), 2048)
        # the direct graph's block table covers the declared maximum, or cuDNN rejects the page-table dims
        from benchmark.attention_inference.benchmark_single_attention import block_table_pages

        self.assertEqual(block_table_pages(SimpleNamespace(**vars(args), page_size=16), kv), 2048 // 16)


if __name__ == "__main__":
    unittest.main()


def test_attn_mask_names_the_alignment_the_graph_is_built_with():
    """The result row records the diagonal alignment benchmark_single_attention gives the cuDNN graph."""
    assert attn_mask("context", True, 2048, 2048, None) == "top_left"
    assert attn_mask("context", True, 512, 65536, None) == "bottom_right"
    assert attn_mask("context", False, 2048, 2048, None) == "no_mask"
    assert attn_mask("generation", True, 1, 8192, None) == "no_mask"
    assert attn_mask("generation", True, 1, 8192, 128) == "bottom_right"
    assert attn_mask("generation", True, 4, 8192, None) == "bottom_right"


@pytest.mark.parametrize(
    "name,clock_khz,width_bits,expected",
    [
        ("NVIDIA GH200 480GB", 2619000, 6144, 4022.784),
        ("NVIDIA GH200", 3200000, 6144, 4915.2),
        ("NVIDIA GH200", None, None, None),
        ("NVIDIA H200", None, None, 4800.0),
    ],
)
def test_peak_bandwidth_for_gh200(monkeypatch, name, clock_khz, width_bits, expected):
    torch = pytest.importorskip("torch")
    from benchmark.attention_inference.benchmark_single_attention import get_peak_bw_gbps

    monkeypatch.delenv("CUDNN_BENCH_PEAK_BW_GBPS", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: name)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    props = SimpleNamespace() if clock_khz is None else SimpleNamespace(memory_clock_rate=clock_khz, memory_bus_width=width_bits)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: props)
    assert get_peak_bw_gbps() == expected

    monkeypatch.setenv("CUDNN_BENCH_PEAK_BW_GBPS", "3750")
    assert get_peak_bw_gbps() == 3750
